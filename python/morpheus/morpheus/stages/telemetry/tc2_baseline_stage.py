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
"""Measures a port's distinct-MAC count against the most it has carried in any period of its own history."""

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
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import normalize_text

logger = logging.getLogger(__name__)

DEFAULT_ENTITY_COLUMN = "port_key"
"""The port, as `TC2CardinalityStage` composes it: `site_id:switch_id:port_id`."""

DEFAULT_VALUE_COLUMN = "macs_per_port"
"""The count the baseline is of. Also the prefix of every column written."""

DEFAULT_BUCKET_SECONDS = 3600
"""The period a peak is kept per. The hour the cardinality count is taken over."""

DEFAULT_WINDOW_SECONDS = 30 * 24 * 3600
"""How far back the periods reach. The thirty days R-B-L2-002 names."""

BASELINE_SUFFIX = "_baseline_max"
BUCKETS_SUFFIX = "_baseline_buckets"
MATURE_SUFFIX = "_baseline_mature"
STEP_SUFFIX = "_step"


@register_stage("tc2-baseline")
class TC2BaselineStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Measure each port's distinct-MAC count against the most it has carried in any hour of its own history.

    This is what R-B-L2-002 reads. `morpheus.stages.telemetry.tc2_cardinality_stage.TC2CardinalityStage` says how
    many distinct MACs a port has carried in the last hour; this stage says whether that is more than the port has
    carried in any hour of the last thirty days, which is the step the rule fires on. It catches the condition
    R-D-L2-001 catches -- a second device, a hub, a switch behind an access port -- without the designation list
    that rule depends on, at the cost of precision: a port that carried a hub last week has a baseline that admits
    one this week.

    Four columns, prefixed by the value column's name:

    - `<value>_baseline_max`, the highest peak the count reached in any committed period inside the window;
    - `<value>_baseline_buckets`, how many committed periods that is over;
    - `<value>_baseline_mature`, whether there are enough of them for the baseline to mean anything;
    - `<value>_step`, this row's count minus the baseline, which is positive only when the port has never carried
      this many. Null until the baseline is mature.

    The baseline is the history of periods before this row's own, never including it, so every row of a period
    sees the same reference whichever batch carried it. A port the estate has only just met has no baseline and
    the rule stays quiet on it, which is R-D-L2-001's case to answer rather than this one's.

    The stage is stateful across messages and must run single-engine, or sharded by the entity column, which is
    determinism control 4. Place it after the cardinality stage, which supplies both the entity and the count.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    entity_column : str, default = "port_key"
        Column holding the entity the history is kept per.
    value_column : str, default = "macs_per_port"
        Column holding the count. Also the prefix of every column written.
    time_column : str, default = "event_time"
        Column holding the observation's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    bucket_seconds : int, default = 3600
        The period a peak is kept per. Match it to the window the count is taken over.
    window_seconds : int, default = 2592000
        How far back the committed periods reach. Thirty days.
    min_buckets : int, default = 24
        Committed periods required before a baseline is published. A day of hours.
    max_buckets : int, default = 1024
        Periods retained per entity whatever the window implies.
    """

    def __init__(self,
                 c: Config,
                 entity_column: str = DEFAULT_ENTITY_COLUMN,
                 value_column: str = DEFAULT_VALUE_COLUMN,
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 min_buckets: int = DEFAULT_MIN_BUCKETS,
                 max_buckets: int = DEFAULT_MAX_BUCKETS):
        super().__init__(c)

        if (not value_column):
            raise ValueError(
                "value_column is required; it names the count to baseline and prefixes every column written")

        if (bucket_seconds <= 0):
            raise ValueError(f"bucket_seconds must be positive, received {bucket_seconds}")

        self._entity_column = entity_column
        self._value_column = value_column
        self._time_column = time_column
        self._time_unit = time_unit

        self._tracker = BucketPeakTracker(bucket_ns=bucket_seconds * NS_PER_SECOND,
                                          window_ns=window_seconds * NS_PER_SECOND,
                                          min_buckets=min_buckets,
                                          max_buckets=max_buckets)

        self._needed_columns[f"{value_column}{BASELINE_SUFFIX}"] = TypeId.INT64
        self._needed_columns[f"{value_column}{BUCKETS_SUFFIX}"] = TypeId.INT64
        self._needed_columns[f"{value_column}{MATURE_SUFFIX}"] = TypeId.BOOL8
        self._needed_columns[f"{value_column}{STEP_SUFFIX}"] = TypeId.INT64

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc2-baseline"

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
    def tracked_entities(self) -> int:
        """Entities currently holding a history."""
        return self._tracker.tracked_entities

    @staticmethod
    def _count(value: typing.Any) -> typing.Optional[int]:
        """Return a host value as an integer count, or `None` where the row carries none."""
        if (value is None):
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        # A null in an integer column widened to float arrives as NaN, which is not a count.
        return None if math.isnan(number) else int(number)

    def _event_time_ns(self, raw: typing.Any) -> typing.Optional[int]:
        try:
            return to_epoch_ns(raw, time_unit=self._time_unit)
        except ValueError:
            return None

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the baseline, its depth, its maturity and this row's step above it.

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
            If the entity, value or time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._entity_column, self._value_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC2BaselineStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            entities = to_host_list(df, self._entity_column)
            values = to_host_list(df, self._value_column)
            raw_times = to_host_list(df, self._time_column)
            row_count = len(entities)

            keys = [normalize_text(entity) for entity in entities]
            counts = [self._count(value) for value in values]
            times = [None if key is None else self._event_time_ns(raw) for (key, raw) in zip(keys, raw_times)]

            measured = measure_rows(self._tracker, keys, counts, times)
            keyless = measured.keyless
            unordered = measured.unordered

            prefix = self._value_column
            assign_nullable_int_column(df, f"{prefix}{BASELINE_SUFFIX}", measured.references)
            assign_nullable_int_column(df, f"{prefix}{BUCKETS_SUFFIX}", measured.buckets)
            assign_nullable_bool_column(df, f"{prefix}{MATURE_SUFFIX}", measured.mature)
            assign_nullable_int_column(df, f"{prefix}{STEP_SUFFIX}", measured.steps)

        if (keyless > 0):
            logger.warning(
                "TC2BaselineStage saw %d of %d rows with a null entity or count; they carry no baseline and entered "
                "no history.",
                keyless,
                row_count)

        if (unordered > 0):
            logger.warning(
                "TC2BaselineStage saw %d of %d rows out of order or without a usable event time; they carry no "
                "baseline and did not advance any history. Preserve per-entity ordering upstream.",
                unordered,
                row_count)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
