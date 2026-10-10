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
"""
Measures a host's fan-out, fan-in and byte asymmetry against the most it has shown in any hour of its own history.

R-B-L3-001 was written against the source's own fourteen-day history and shipped against a literal fifty, because
nothing kept a history. A literal is wrong in both directions at once: an infrastructure server whose ordinary hour
reaches seventy clients fires on every hour of its life, and a workstation that has reached three addresses an hour
for a fortnight and now reaches forty-nine never does. What the rule means is "more than this host has ever
reached", and this stage is what makes that a column.

It keeps three histories, each the pattern `TC2BaselineStage` applies to ports:

- **fan-out**, `dsts_per_src` keyed on the source, which R-B-L3-001 reads;
- **fan-in**, `srcs_per_dst` keyed on the *destination*, because the count on a row is about the row's destination
  and a workstation suddenly reached by many is the case worth seeing;
- **byte asymmetry**, `byte_asymmetry` keyed on the source, so "more lopsided than anything this host has sent" is
  a comparison with the host's own envelope rather than with a ratio somebody typed.

Each writes four columns prefixed by the value's own name: `<value>_baseline_max`, the highest hourly peak in the
window before this row's hour; `<value>_baseline_buckets`, how many hours that is over; `<value>_baseline_mature`,
whether there are enough of them to mean anything; and `<value>_step`, this row's value minus the baseline, positive
only when the host has never gone this far. The baseline is the maximum of the hourly peaks, which is the 100th
percentile of the guide's 99.5th: over a fortnight of hours the two differ by at most the single busiest hour, and
the maximum is the one a tracker can keep in constant memory per period.

**The key can be scoped by a group column.** An address is not an identity across routing domains: two tenants or
two VRFs can both own `10.0.0.5`, and a history keyed on the address alone would pool them into one host that
behaves like neither. `group_column` names the column that scopes them -- a tenant, a site, a VRF -- and a row
whose group is null gets no baseline rather than a guessed one.

The stage is stateful across messages and must run single-engine. Sharding by source keeps the fan-out and asymmetry
histories whole and scatters the fan-in one, exactly as for `TC3CardinalityStage`, which this stage follows and
which supplies two of its three values.
"""

import logging
import math
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
from morpheus.utils.bucket_peak import DEFAULT_MAX_BUCKETS
from morpheus.utils.bucket_peak import DEFAULT_MIN_BUCKETS
from morpheus.utils.bucket_peak import NS_PER_SECOND
from morpheus.utils.bucket_peak import BucketPeakTracker
from morpheus.utils.bucket_peak import measure_rows
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import compose_key
from morpheus.utils.entity_key import normalize_text

logger = logging.getLogger(__name__)

DEFAULT_BUCKET_SECONDS = 3600
"""The period a peak is kept per. The hour the cardinality counts are taken over."""

DEFAULT_WINDOW_SECONDS = 14 * 24 * 3600
"""How far back the periods reach. The fourteen days R-B-L3-001 names."""

BASELINE_SUFFIX = "_baseline_max"
BUCKETS_SUFFIX = "_baseline_buckets"
MATURE_SUFFIX = "_baseline_mature"
STEP_SUFFIX = "_step"


def _number(value: typing.Any, integral: bool) -> typing.Optional[typing.Union[int, float]]:
    """A host value as a count or a ratio, or `None` where the row carries none."""
    if (value is None or isinstance(value, bool)):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    # A null in an integer column widened to float arrives as NaN, which is not a value.
    if (math.isnan(number) or math.isinf(number)):
        return None

    return int(number) if integral else number


@register_stage("tc3-baseline")
class TC3BaselineStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Measure each host's fan-out, fan-in and byte asymmetry against the most it has shown in any hour of its history.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    src_column : str, default = "src_ip"
        Column holding the source address, which keys the fan-out and asymmetry histories.
    dst_column : str, default = "dst_ip"
        Column holding the destination address, which keys the fan-in history.
    fan_out_column : str, default = "dsts_per_src"
        Column holding the source's distinct-destination count. Set to `None` to keep no fan-out history.
    fan_in_column : str, default = "srcs_per_dst"
        Column holding the destination's distinct-source count. Set to `None` to keep no fan-in history.
    asymmetry_column : str, default = "byte_asymmetry"
        Column holding the flow's byte asymmetry. Set to `None` to keep no asymmetry history.
    group_column : str, optional
        Column scoping every key, for estates where one address names different hosts in different routing
        domains. A row whose group is null carries no baseline.
    time_column : str, default = "event_time"
        Column holding the flow's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    bucket_seconds : int, default = 3600
        The period a peak is kept per. Match it to the window the counts are taken over.
    window_seconds : int, default = 1209600
        How far back the committed periods reach. Fourteen days.
    min_buckets : int, default = 24
        Committed periods required before a baseline is published. A day of active hours.
    max_buckets : int, default = 1024
        Periods retained per host whatever the window implies.
    """

    def __init__(self,
                 c: Config,
                 src_column: str = "src_ip",
                 dst_column: str = "dst_ip",
                 fan_out_column: str = "dsts_per_src",
                 fan_in_column: str = "srcs_per_dst",
                 asymmetry_column: str = "byte_asymmetry",
                 group_column: str = None,
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 min_buckets: int = DEFAULT_MIN_BUCKETS,
                 max_buckets: int = DEFAULT_MAX_BUCKETS):
        super().__init__(c)

        if (bucket_seconds <= 0):
            raise ValueError(f"bucket_seconds must be positive, received {bucket_seconds}")

        # (value column, entity column, integral) for each history kept.
        self._histories = [(column, entity, integral)
                           for (column, entity, integral) in ((fan_out_column, src_column, True), (fan_in_column,
                                                                                                   dst_column, True),
                                                              (asymmetry_column, src_column, False)) if column]

        if (len(self._histories) == 0):
            raise ValueError("at least one of fan_out_column, fan_in_column and asymmetry_column is required")

        if (len({column for (column, _, _) in self._histories}) != len(self._histories)):
            raise ValueError("fan_out_column, fan_in_column and asymmetry_column must name different columns, "
                             "since each prefixes the columns its history writes")

        self._group_column = group_column
        self._time_column = time_column
        self._time_unit = time_unit

        self._trackers = {
            column:
                BucketPeakTracker(bucket_ns=bucket_seconds * NS_PER_SECOND,
                                  window_ns=window_seconds * NS_PER_SECOND,
                                  min_buckets=min_buckets,
                                  max_buckets=max_buckets)
            for (column, _, _) in self._histories
        }

        for (column, _, integral) in self._histories:
            self._needed_columns[f"{column}{BASELINE_SUFFIX}"] = TypeId.INT64 if integral else TypeId.FLOAT64
            self._needed_columns[f"{column}{BUCKETS_SUFFIX}"] = TypeId.INT64
            self._needed_columns[f"{column}{MATURE_SUFFIX}"] = TypeId.BOOL8
            self._needed_columns[f"{column}{STEP_SUFFIX}"] = TypeId.INT64 if integral else TypeId.FLOAT64

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc3-baseline"

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
    def tracked_entities(self) -> dict:
        """Hosts currently holding a history, per value column."""
        return {column: tracker.tracked_entities for (column, tracker) in self._trackers.items()}

    def _event_time_ns(self, raw: typing.Any) -> typing.Optional[int]:
        try:
            return to_epoch_ns(raw, time_unit=self._time_unit)
        except ValueError:
            return None

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write each history's baseline, depth, maturity and this row's step above it.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming message.

        Returns
        -------
        The input message, with the baseline columns populated.

        Raises
        ------
        KeyError
            If an entity, value, group or time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._time_column
                        ] + [column for (value, entity, _) in self._histories for column in (value, entity)]

            if (self._group_column):
                required.append(self._group_column)

            missing = sorted({column for column in required if column not in df.columns})

            if (len(missing) > 0):
                raise KeyError(f"TC3BaselineStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            raw_times = to_host_list(df, self._time_column)
            row_count = len(raw_times)
            times = [self._event_time_ns(raw) for raw in raw_times]
            groups = to_host_list(df, self._group_column) if self._group_column else None
            keyless = 0
            unordered = 0

            for (value_column, entity_column, integral) in self._histories:
                entities = to_host_list(df, entity_column)

                if (groups is None):
                    keys = [normalize_text(entity) for entity in entities]
                else:
                    keys = [compose_key((group, entity)) for (group, entity) in zip(groups, entities)]

                values = [_number(value, integral) for value in to_host_list(df, value_column)]
                measured = measure_rows(self._trackers[value_column], keys, values, times)
                keyless = max(keyless, measured.keyless)
                unordered = max(unordered, measured.unordered)

                assign = assign_nullable_int_column if integral else assign_nullable_float_column
                assign(df, f"{value_column}{BASELINE_SUFFIX}", measured.references)
                assign_nullable_int_column(df, f"{value_column}{BUCKETS_SUFFIX}", measured.buckets)
                assign_nullable_bool_column(df, f"{value_column}{MATURE_SUFFIX}", measured.mature)
                assign(df, f"{value_column}{STEP_SUFFIX}", measured.steps)

        if (keyless > 0):
            logger.warning(
                "TC3BaselineStage saw up to %d of %d rows with a null host, group or value; they carry no baseline "
                "and entered no history.",
                keyless,
                row_count)

        if (unordered > 0):
            logger.warning(
                "TC3BaselineStage saw up to %d of %d rows out of order or without a usable event time; they carry "
                "no baseline and did not advance any history. Preserve per-host ordering upstream.",
                unordered,
                row_count)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
