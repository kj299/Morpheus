# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Measures how fast a principal would have had to travel between consecutive successful authentications."""

import ipaddress
import logging
import typing

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.determinism import DEFAULT_FLOAT_DECIMALS
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.geo_velocity import NS_PER_SECOND
from morpheus.utils.geo_velocity import GeoVelocityTracker
from morpheus.utils.geo_velocity import validate_coordinate

logger = logging.getLogger(__name__)

MEASURED = "measured"
FIRST = "first_for_principal"
NO_PRINCIPAL = "no_principal"
NO_TIME = "no_event_time"
NOT_SUCCESSFUL = "not_successful"
TOKEN_REFRESH = "token_refresh"
VPN_EGRESS = "vpn_egress"
NO_COORDINATE = "no_coordinate"
OUT_OF_ORDER = "out_of_order"

DEFAULT_SUCCESS_VALUES = ("success", "succeeded", "allow", "allowed", "0")
"""Values of the result column that count as a successful authentication."""

DEFAULT_REFRESH_VALUES = ("refresh", "refresh_token", "renew")
"""Values of the token type column that mark a record as a refresh rather than a fresh authentication."""


@register_stage("tc5-travel")
class TC5TravelStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write the implied travel speed between a principal's consecutive successful authentications.

    This is the input R-D-L5-003 reads, and it is one of the few layer 5 rules that needs no model at all: a
    great-circle distance, an elapsed time, and a speed no aircraft reaches. The arithmetic lives in
    {py:mod}`~morpheus.utils.geo_velocity`; what this stage contributes is deciding which records are eligible to
    be measured, which is where the rule's own exclusions live.

    **An ineligible record is not measured and does not become the anchor either.** That second half matters more
    than it looks. The rule excludes token refreshes because they carry the location the original authentication
    had rather than where the principal is now; a refresh that updated the anchor would erase a real journey, and
    one that was merely skipped from measurement but still recorded would do the same thing more quietly.

    **Every row says why it was or was not measured**, in `travel_status`. An empty result is this rule's
    characteristic failure -- a collector without coordinates, a result vocabulary that does not match, an
    exclusion list that swallowed the estate -- and a rule that fires on nothing looks exactly like a rule with
    nothing to fire on. The reasons are checked in a fixed order, so a record that is both a refresh and from a
    VPN range reports the same one every run:

    | Order | `travel_status` | Meaning |
    | --- | --- | --- |
    | 1 | `no_principal` | Nothing to attribute the journey to |
    | 2 | `no_event_time` | Nothing to measure elapsed time against |
    | 3 | `not_successful` | The rule is about successful authentications; a failure proves no presence |
    | 4 | `token_refresh` | Carries the original location, so it is evidence of nothing about now |
    | 5 | `vpn_egress` | The apparent location is the concentrator's, not the principal's |
    | 6 | `no_coordinate` | Absent or off-globe; a nonsense distance would alert on a geolocation database |
    | 7 | `out_of_order` | Arrived after a later record; measuring backwards would time a journey in reverse |
    | 8 | `first_for_principal` | Eligible, and the first one, so there is nothing to measure from |
    | 9 | `measured` | The three measurement columns are populated |

    The stage is stateful across messages and must run single-engine, or sharded by principal -- determinism
    control 4.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    principal_column : str, default = "user_principal"
        Column holding the authenticated principal, which journeys are measured per.
    latitude_column : str, default = "source_latitude"
        Column holding the source latitude in degrees.
    longitude_column : str, default = "source_longitude"
        Column holding the source longitude in degrees.
    time_column : str, default = "event_time"
        Column holding the record's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    result_column : str, default = "auth_result"
        Column holding the authentication result. A record whose result is not in `success_values` is not
        measured: a failed authentication is evidence that somebody tried, not that anybody was there.
    success_values : tuple, default = see `DEFAULT_SUCCESS_VALUES`
        Result values that count as success. Compared case-insensitively.
    refresh_column : str, default = "token_type"
        Column marking a record as a token refresh. Absent from the frame means no record is excluded on this
        ground, which over-measures rather than hiding journeys.
    refresh_values : tuple, default = see `DEFAULT_REFRESH_VALUES`
        Values of `refresh_column` that mark a refresh. Compared case-insensitively.
    source_ip_column : str, default = "source_ip"
        Column holding the source address, matched against `excluded_source_networks`.
    excluded_source_networks : tuple, default = ()
        Addresses and CIDR ranges whose apparent location belongs to a concentrator rather than to the principal.
        Both forms are accepted; a bare address is one host. Ships empty, and the rule fires on VPN users until an
        estate supplies its own egress ranges -- the same shape as R-D-L2-003's exclusion list, and for the same
        reason: this repository cannot know them.
    min_elapsed_seconds : int, default = 1
        Floor on the elapsed time the speed is computed against. See {py:mod}`~morpheus.utils.geo_velocity`:
        without it, two authentications bearing one timestamp divide by zero and the most impossible journey there
        is reads as no journey at all.
    max_entities : int, default = 100000
        Principals holding a previous location before the least recently seen is forgotten.
    decimals : int, default = 4
        Decimal places the distance and speed are rounded to, under determinism control 9.
    """

    def __init__(self,
                 c: Config,
                 principal_column: str = "user_principal",
                 latitude_column: str = "source_latitude",
                 longitude_column: str = "source_longitude",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 result_column: str = "auth_result",
                 success_values: typing.Sequence[str] = DEFAULT_SUCCESS_VALUES,
                 refresh_column: str = "token_type",
                 refresh_values: typing.Sequence[str] = DEFAULT_REFRESH_VALUES,
                 source_ip_column: str = "source_ip",
                 excluded_source_networks: typing.Sequence[str] = (),
                 min_elapsed_seconds: int = 1,
                 max_entities: int = 100_000,
                 decimals: int = DEFAULT_FLOAT_DECIMALS):
        super().__init__(c)

        if (min_elapsed_seconds <= 0):
            raise ValueError(f"min_elapsed_seconds must be positive, received {min_elapsed_seconds}")

        self._principal_column = principal_column
        self._latitude_column = latitude_column
        self._longitude_column = longitude_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._result_column = result_column
        self._success_values = {value.lower() for value in success_values}
        self._refresh_column = refresh_column
        self._refresh_values = {value.lower() for value in refresh_values}
        self._source_ip_column = source_ip_column

        # Parsed once, at build time. A malformed range is a configuration error, and finding it when the pipeline
        # is assembled is better than finding it as a per-row exception that quietly excludes nothing.
        self._excluded_networks = []

        for entry in excluded_source_networks:
            try:
                self._excluded_networks.append(ipaddress.ip_network(entry, strict=False))
            except ValueError as error:
                raise ValueError(f"excluded_source_networks entry {entry!r} is not an address or CIDR range: "
                                 f"{error}") from error

        self._tracker = GeoVelocityTracker(min_elapsed_ns=min_elapsed_seconds * NS_PER_SECOND,
                                           max_entities=max_entities,
                                           decimals=decimals)

        self._needed_columns["travel_status"] = TypeId.STRING
        self._needed_columns["travel_distance_km"] = TypeId.FLOAT64
        self._needed_columns["travel_elapsed_ns"] = TypeId.INT64
        self._needed_columns["travel_kmh"] = TypeId.FLOAT64
        self._needed_columns["travel_elapsed_floored"] = TypeId.BOOL8

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc5-travel"

    def accepted_types(self) -> tuple:
        """
        Accepted input types for this stage.

        Returns
        -------
        tuple
            Accepted input types.
        """
        return (ControlMessage, MessageMeta)

    def supports_cpp_node(self) -> bool:
        """Whether this stage supports a C++ node."""
        return False

    @property
    def tracked_principals(self) -> int:
        """Principals currently holding a previous location."""
        return self._tracker.tracked_entities

    def _is_excluded_address(self, value: typing.Any) -> typing.Optional[bool]:
        """
        Whether a source address falls in an excluded range. `None` where it cannot be parsed.

        An unparseable address is not treated as excluded. Excluding it would let a collector emitting a hostname
        where an address belongs silence the rule for the whole estate, which is the failure mode worth avoiding.
        """
        text = normalize_text(value)

        if (text is None):
            return None

        try:
            address = ipaddress.ip_address(text)
        except ValueError:
            return None

        return any(address in network for network in self._excluded_networks)

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the travel status and, where a journey was measured, its distance, elapsed time and speed.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming authentication records.

        Returns
        -------
        The input message, with the travel columns populated.

        Raises
        ------
        KeyError
            If the principal or time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._principal_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC5TravelStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            principals = to_host_list(df, self._principal_column)
            raw_times = to_host_list(df, self._time_column)
            row_count = len(principals)

            def optional(column: str) -> list:
                return to_host_list(df, column) if column in df.columns else [None] * row_count

            latitudes = optional(self._latitude_column)
            longitudes = optional(self._longitude_column)
            results = optional(self._result_column)
            refreshes = optional(self._refresh_column)
            addresses = optional(self._source_ip_column)

            has_result_column = self._result_column in df.columns

            status: list = []
            distance: list = []
            elapsed: list = []
            speed: list = []
            floored: list = []
            unparseable_addresses = 0
            bad_coordinates = 0

            for (position, raw_principal) in enumerate(principals):
                principal = normalize_text(raw_principal)

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                result = normalize_text(results[position])
                refresh = normalize_text(refreshes[position])
                excluded = self._is_excluded_address(addresses[position])

                unparseable_addresses += int(excluded is None and addresses[position] is not None)

                latitude = latitudes[position]
                longitude = longitudes[position]
                usable_coordinate = validate_coordinate(latitude, longitude)

                bad_coordinates += int(not usable_coordinate and latitude is not None and longitude is not None)

                if (principal is None):
                    reason = NO_PRINCIPAL
                elif (event_time_ns is None):
                    reason = NO_TIME
                elif (has_result_column and (result is None or result.lower() not in self._success_values)):
                    # A failed authentication is evidence that somebody tried, not that anybody was there. Where
                    # the column is absent entirely the source does not report results and every record is taken
                    # at face value, which over-measures rather than measuring nothing.
                    reason = NOT_SUCCESSFUL
                elif (refresh is not None and refresh.lower() in self._refresh_values):
                    reason = TOKEN_REFRESH
                elif (excluded is True):
                    reason = VPN_EGRESS
                elif (not usable_coordinate):
                    reason = NO_COORDINATE
                else:
                    reason = None

                if (reason is not None):
                    # Not measured, and deliberately not recorded either: an ineligible record that updated the
                    # anchor would erase the journey the next eligible one is supposed to reveal.
                    status.append(reason)
                    distance.append(None)
                    elapsed.append(None)
                    speed.append(None)
                    floored.append(False)
                    continue

                measurement = self._tracker.observe(principal, event_time_ns, latitude, longitude)

                if (measurement.out_of_order):
                    status.append(OUT_OF_ORDER)
                elif (measurement.first_for_entity):
                    status.append(FIRST)
                else:
                    status.append(MEASURED)

                distance.append(measurement.distance_km)
                elapsed.append(measurement.elapsed_ns)
                speed.append(measurement.implied_kmh)
                floored.append(measurement.elapsed_floored)

            assign_str_column(df, "travel_status", status)
            assign_nullable_float_column(df, "travel_distance_km", distance)
            assign_nullable_int_column(df, "travel_elapsed_ns", elapsed)
            assign_nullable_float_column(df, "travel_kmh", speed)
            df["travel_elapsed_floored"] = floored

        if (unparseable_addresses > 0):
            logger.warning(
                "TC5TravelStage could not parse %d of %d values in %s as addresses, so they were not matched "
                "against the excluded ranges and were measured. Treating them as excluded would let a collector "
                "emitting a hostname silence this rule for the whole estate.",
                unparseable_addresses,
                len(principals),
                self._source_ip_column)

        if (bad_coordinates > 0):
            logger.warning(
                "TC5TravelStage refused %d of %d coordinates that were present but not inside the globe; those "
                "records carry no measurement. A nonsense distance would read as impossible travel, which is an "
                "alert about a geolocation database rather than about a principal.",
                bad_coordinates,
                len(principals))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
