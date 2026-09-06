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
"""Counts a principal's authentication volume and the places, applications and devices it has reached from."""

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
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.distinct_window import NS_PER_SECOND
from morpheus.utils.distinct_window import DistinctWindowTracker
from morpheus.utils.entity_key import compose_key
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.value_novelty import ValueNoveltyTracker

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 24 * 3600
"""Trailing window `logcount` covers.

A day rather than the hour the layer 2 counts use. `logcount` is the digital fingerprinting feature the guide
names, and what it is for is the shape of a principal's working day; an hour of it is mostly noise about when
somebody happened to open a laptop.
"""

DEFAULT_LOCATION_COLUMNS = ("source_country", "source_region", "source_city")
"""Columns concatenated into the location `locincrement` counts distinct values of."""

LOCATION = "location"
APP = "app"
DEVICE = "device"

CUMULATIVE_FIELDS = (LOCATION, APP, DEVICE)
"""Fields counted cumulatively, never decaying. These are the `*increment` features."""

INCREMENT_COLUMNS = {LOCATION: "locincrement", APP: "appincrement", DEVICE: "deviceincrement"}
"""Output column per cumulative field. `locincrement` and `appincrement` keep the digital fingerprinting names."""

ACTIVITY_VALUE = "authentication"
"""The constant `logcount`'s window is fed.

`logcount` is a count, not a cardinality: nothing is being distinguished, so the value never varies and the
distinct figure the tracker also computes is meaningless here and is not emitted. Counting through the same
audited window as everything else is worth more than a fourth window implementation that would have to be
audited separately.
"""


@register_stage("tc5-novelty")
class TC5NoveltyStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write the TC-5 volume and novelty features for each authenticating principal.

    These are the features `morpheus_dfp` has already proven on the Duo and Azure schemas, which the guide names as
    the template for this layer: `logcount`, `locincrement` and `appincrement`, extended with a device increment
    and a windowed count of distinct source ASNs.

    Two mechanisms are at work and the difference is the point:

    - **The increments are cumulative and never decay.** A principal who has authenticated from three countries
      has an `locincrement` of three forever, and a fourth country raises it permanently. The guide is explicit
      that this is intended: the rule reading it (R-B-L5-002) is scored against a per-user loss scaler, which
      handles a legitimate relocation correctly, and a decaying count would keep re-firing on the same move.
    - **`logcount` and `asns_in_window` are windowed.** Volume and concurrent network origin are questions about
      now, and a cumulative answer to either would rise monotonically for every account in the estate.

    **A field the collector did not report is an omission, not a value the principal used.** Each cumulative field
    is therefore tracked on its own, and a row missing one simply does not observe it -- its increment is null for
    that row and its running count is untouched. Passing the omission through as a value would make `appincrement`
    rise the first time a collector dropped the field and stay raised, so a gap in collection would read as an
    application the principal had never used before. `morpheus.utils.value_novelty` treats `None` as a value on
    purpose, because a layer 1 port with no transceiver installed genuinely reports nothing and the transition is
    an event; at layer 5 the same convention would be a defect, which is why the fields are separated here rather
    than handed over together.

    `*_first_seen` is null on a principal's first sample rather than true. The first sample establishes what normal
    looks like for an entity and is not itself an event, and the increment column beside it already reads one,
    which carries the same fact without answering a question the history cannot yet answer.

    The stage is stateful across messages and must run single-engine, or sharded by principal -- determinism
    control 4, on the key this telemetry class is already organized around.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    principal_column : str, default = "user_principal"
        Column holding the authenticated principal, which every count is grouped by.
    location_columns : tuple, default = ("source_country", "source_region", "source_city")
        Columns concatenated into the location. A column the frame does not carry at all is skipped, so a source
        reporting only a country still counts locations, at the resolution it has. A column that is present and
        null on a row is not skipped: that row's location is null and no location is counted for it, which is the
        same rule every composed key in this fork follows. The alternative -- dropping the null part and composing
        what is left -- would turn one place into two whenever a collector's city lookup failed, and each of those
        would read as a country the principal had never authenticated from.
    app_column : str, default = "app"
        Column holding the application or service being authenticated to.
    device_column : str, default = "device_id"
        Column holding the device the authentication came from.
    asn_column : str, default = "source_asn"
        Column holding the source autonomous system number.
    time_column : str, default = "event_time"
        Column holding the record's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    window_seconds : int, default = 86400
        Trailing window `logcount` and `asns_in_window` cover.
    max_samples : int, default = 4096
        Observations retained per principal regardless of the window. When this binds the windowed figures are
        lower bounds and the row is marked saturated.
    max_values : int, default = 256
        Distinct values recalled per cumulative field per principal. Higher than the layer 1 default because a
        travelling principal legitimately accumulates locations and applications over years, where a port that has
        held more than a handful of optics is already the anomaly.
    max_entities : int, default = 100000
        Principals tracked before the least recently seen is forgotten. Sized for the tens of thousands of users
        the guide puts this telemetry class at.
    """

    def __init__(self,
                 c: Config,
                 principal_column: str = "user_principal",
                 location_columns: typing.Sequence[str] = DEFAULT_LOCATION_COLUMNS,
                 app_column: str = "app",
                 device_column: str = "device_id",
                 asn_column: str = "source_asn",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 max_samples: int = 4096,
                 max_values: int = 256,
                 max_entities: int = 100_000):
        super().__init__(c)

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        if (len(location_columns) == 0):
            raise ValueError("At least one location column is required")

        self._principal_column = principal_column
        self._location_columns = tuple(location_columns)
        self._app_column = app_column
        self._device_column = device_column
        self._asn_column = asn_column
        self._time_column = time_column
        self._time_unit = time_unit

        window_ns = window_seconds * NS_PER_SECOND

        self._activity = DistinctWindowTracker(window_ns=window_ns, max_samples=max_samples, max_entities=max_entities)
        self._asns = DistinctWindowTracker(window_ns=window_ns, max_samples=max_samples, max_entities=max_entities)
        self._novelty = {
            field: ValueNoveltyTracker([field], max_values=max_values, max_entities=max_entities)
            for field in CUMULATIVE_FIELDS
        }

        self._needed_columns["user_location"] = TypeId.STRING
        self._needed_columns["logcount"] = TypeId.INT64
        self._needed_columns["logcount_saturated"] = TypeId.BOOL8
        self._needed_columns["asns_in_window"] = TypeId.INT64
        self._needed_columns["asns_in_window_saturated"] = TypeId.BOOL8

        for field in CUMULATIVE_FIELDS:
            self._needed_columns[INCREMENT_COLUMNS[field]] = TypeId.INT64
            self._needed_columns[f"{field}_first_seen"] = TypeId.BOOL8

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc5-novelty"

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
        """Principals currently holding an activity window."""
        return self._activity.tracked_entities

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the volume count, the cumulative increments, and the windowed ASN count.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming authentication records.

        Returns
        -------
        The input message, with the TC-5 novelty columns populated.

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
                raise KeyError(f"TC5NoveltyStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            principals = to_host_list(df, self._principal_column)
            raw_times = to_host_list(df, self._time_column)
            row_count = len(principals)

            present_locations = [name for name in self._location_columns if name in df.columns]
            location_parts = [to_host_list(df, name) for name in present_locations]

            def optional(column: str) -> list:
                return to_host_list(df, column) if column in df.columns else [None] * row_count

            apps = optional(self._app_column)
            devices = optional(self._device_column)
            asns = optional(self._asn_column)

            locations: list = []
            logcount: list = []
            logcount_saturated: list = []
            asn_counts: list = []
            asn_saturated: list = []
            increments: dict[str, list] = {field: [] for field in CUMULATIVE_FIELDS}
            first_seen: dict[str, list] = {field: [] for field in CUMULATIVE_FIELDS}
            keyless = 0
            unordered = 0

            for (position, raw_principal) in enumerate(principals):
                principal = normalize_text(raw_principal)
                location = compose_key([part[position] for part in location_parts]) if location_parts else None
                locations.append(location)

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                if (principal is None or event_time_ns is None):
                    # Pooling records under a fabricated principal would make every unattributed authentication in
                    # the estate look like one extraordinarily busy account.
                    logcount.append(None)
                    logcount_saturated.append(False)
                    asn_counts.append(None)
                    asn_saturated.append(False)

                    for field in CUMULATIVE_FIELDS:
                        increments[field].append(None)
                        first_seen[field].append(None)

                    keyless += 1
                    continue

                activity = self._activity.observe(principal, event_time_ns, ACTIVITY_VALUE)
                logcount.append(activity.total)
                logcount_saturated.append(activity.saturated)
                unordered += int(activity.out_of_order)

                asn = normalize_text(asns[position])

                if (asn is None):
                    asn_counts.append(None)
                    asn_saturated.append(False)
                else:
                    seen_asn = self._asns.observe(principal, event_time_ns, asn)
                    asn_counts.append(seen_asn.distinct)
                    asn_saturated.append(seen_asn.saturated)

                values = {
                    LOCATION: location,
                    APP: normalize_text(apps[position]),
                    DEVICE: normalize_text(devices[position]),
                }

                for field in CUMULATIVE_FIELDS:
                    value = values[field]

                    if (value is None):
                        # An omission, not a value the principal used. Observing it would raise the increment the
                        # first time a collector dropped the field and leave it raised.
                        increments[field].append(None)
                        first_seen[field].append(None)
                        continue

                    result = self._novelty[field].observe(principal, event_time_ns, {field: value})

                    if (result.out_of_order):
                        increments[field].append(None)
                        first_seen[field].append(None)
                        continue

                    increments[field].append(result.distinct_counts[field])
                    first_seen[field].append(result.first_seen[field])

            assign_str_column(df, "user_location", locations)
            assign_nullable_int_column(df, "logcount", logcount)
            df["logcount_saturated"] = logcount_saturated
            assign_nullable_int_column(df, "asns_in_window", asn_counts)
            df["asns_in_window_saturated"] = asn_saturated

            for field in CUMULATIVE_FIELDS:
                assign_nullable_int_column(df, INCREMENT_COLUMNS[field], increments[field])
                assign_nullable_bool_column(df, f"{field}_first_seen", first_seen[field])

        if (keyless > 0):
            logger.warning(
                "TC5NoveltyStage saw %d of %d records with no principal or no usable event time; they carry no "
                "counts. Pooling them under a fabricated principal would make every unattributed authentication in "
                "the estate look like one extraordinarily busy account.",
                keyless,
                len(principals))

        if (unordered > 0):
            logger.warning(
                "TC5NoveltyStage saw %d of %d records out of order; they did not enter the activity window. "
                "Preserve per-principal ordering upstream with TotalOrderStage.",
                unordered,
                len(principals))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
