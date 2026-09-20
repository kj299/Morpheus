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
How regularly each directed address pair talks, which is what R-B-L3-002 thresholds.

{py:mod}`~morpheus.utils.arrival_regularity` holds the arithmetic and the decisions inside it. This stage is the
part that has to be got right at the pipeline level, and there are two things.

**The entity is the directed pair, not the source.** A workstation that beacons to one address and browses to
five hundred others has a perfectly regular conversation buried in a wholly irregular stream, and a coefficient
over the source would average the beacon away. `src_ip:dst_ip` is composed through
{py:mod}`~morpheus.utils.entity_key` so it is the same string every other stage would build, and directed
because a beacon is a thing a host does *to* a server rather than a property of the pair.

**The window is a day and the rule wants twelve intervals**, so this stage holds far more state per entity than
the hourly ones beside it. A busy estate's pair count is the product of its hosts and everything they talk to,
which is why `max_entities` is a parameter here rather than a constant: the memory bound is the thing an operator
tunes, and the tracker drops the least recently seen pair rather than growing.
"""

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
from morpheus.utils.arrival_regularity import DEFAULT_MAX_ENTITIES
from morpheus.utils.arrival_regularity import DEFAULT_MIN_INTERVALS
from morpheus.utils.arrival_regularity import NS_PER_SECOND
from morpheus.utils.arrival_regularity import ArrivalRegularityTracker
from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import compose_key
from morpheus.utils.entity_key import normalize_text

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 24 * 3600
"""Trailing window arrivals are retained for. A day, because twelve intervals of an hourly beacon need half of one."""

PAIR_KEY = "flow_pair_key"
INTERVALS = "flow_intervals"
MEAN_INTERVAL = "flow_mean_interval_ns"
INTERVAL_CV = "flow_interval_cv"
SIZE_CV = "flow_size_cv"
REGULARITY_MATURE = "flow_regularity_mature"
REGULARITY_SATURATED = "flow_regularity_saturated"


@register_stage("tc3-beacon")
class TC3BeaconStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Measure the inter-arrival and size regularity of each directed address pair.

    The stage is stateful across messages and must run single-engine, or sharded by the pair key, which is the
    one sharding that keeps a conversation whole.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    src_column : str, default = "src_ip"
        Column holding the source address.
    dst_column : str, default = "dst_ip"
        Column holding the destination address.
    size_column : str, default = "bytes_out"
        Column whose regularity is measured alongside the timing. The byte count the source sent, because a
        beacon's payload is a fixed-length check-in and the reply is whatever the operator had queued.
    time_column : str, default = "event_time"
        Column holding the flow's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    window_seconds : int, default = 86400
        Trailing window arrivals are retained for.
    min_intervals : int, default = 12
        Intervals required before the coefficients are published, which is the figure R-B-L3-002 names.
    max_samples : int, default = 4096
        Arrivals retained per pair regardless of the window.
    max_entities : int, default = 500000
        Pairs retained before the least recently seen is dropped.
    """

    def __init__(self,
                 c: Config,
                 src_column: str = "src_ip",
                 dst_column: str = "dst_ip",
                 size_column: str = "bytes_out",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 min_intervals: int = DEFAULT_MIN_INTERVALS,
                 max_samples: int = 4096,
                 max_entities: int = DEFAULT_MAX_ENTITIES):
        super().__init__(c)

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        self._src_column = src_column
        self._dst_column = dst_column
        self._size_column = size_column
        self._time_column = time_column
        self._time_unit = time_unit

        self._tracker = ArrivalRegularityTracker(window_ns=window_seconds * NS_PER_SECOND,
                                                 min_intervals=min_intervals,
                                                 max_samples=max_samples,
                                                 max_entities=max_entities)

        self._needed_columns[PAIR_KEY] = TypeId.STRING
        self._needed_columns[INTERVALS] = TypeId.INT64
        self._needed_columns[MEAN_INTERVAL] = TypeId.INT64
        self._needed_columns[INTERVAL_CV] = TypeId.FLOAT64
        self._needed_columns[SIZE_CV] = TypeId.FLOAT64
        self._needed_columns[REGULARITY_MATURE] = TypeId.BOOL8
        self._needed_columns[REGULARITY_SATURATED] = TypeId.BOOL8

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc3-beacon"

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

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the pair key and the regularity columns.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming flow records.

        Returns
        -------
        The input message, with the regularity columns populated.

        Raises
        ------
        KeyError
            If the source, destination or time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._src_column, self._dst_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC3BeaconStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            sources = to_host_list(df, self._src_column)
            destinations = to_host_list(df, self._dst_column)
            raw_times = to_host_list(df, self._time_column)

            rows = len(sources)
            sizes = to_host_list(df, self._size_column) if self._size_column in df.columns else [None] * rows

            keys: list = []
            intervals: list = []
            means: list = []
            interval_cvs: list = []
            size_cvs: list = []
            mature: list = []
            saturated: list = []
            unordered = 0
            keyless = 0

            for position in range(rows):
                key = compose_key([normalize_text(sources[position]), normalize_text(destinations[position])])

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                keys.append(key)

                if (key is None or event_time_ns is None):
                    keyless += 1
                    intervals.append(None)
                    means.append(None)
                    interval_cvs.append(None)
                    size_cvs.append(None)
                    mature.append(False)
                    saturated.append(False)
                    continue

                result = self._tracker.observe(key, event_time_ns, self._size_of(sizes[position]))

                intervals.append(result.intervals)
                means.append(result.mean_interval_ns)
                interval_cvs.append(result.interval_cv)
                size_cvs.append(result.size_cv)
                mature.append(result.mature)
                saturated.append(result.saturated)
                unordered += int(result.out_of_order)

            assign_str_column(df, PAIR_KEY, keys)
            assign_nullable_int_column(df, INTERVALS, intervals)
            assign_nullable_int_column(df, MEAN_INTERVAL, means)
            df[INTERVAL_CV] = interval_cvs
            df[SIZE_CV] = size_cvs
            df[REGULARITY_MATURE] = mature
            df[REGULARITY_SATURATED] = saturated

            if (keyless > 0):
                logger.warning(
                    "TC3BeaconStage left %d of %d flows unmeasured for want of an address pair or a usable "
                    "event time.",
                    keyless,
                    rows)

            if (unordered > 0):
                logger.warning(
                    "TC3BeaconStage saw %d of %d flows out of order; they did not join any pair's rhythm. A "
                    "negative interval in the mean would make a shuffled stream look more regular than the "
                    "ordered one it came from.",
                    unordered,
                    rows)

        return message

    @staticmethod
    def _size_of(value: typing.Any) -> typing.Optional[float]:
        """The byte count as a number, or `None` where the collector did not supply one."""
        try:
            # `value != value` is the null check for a float NaN, which `is None` does not catch.
            if (value is None or value != value):  # pylint: disable=comparison-with-itself
                return None

            return float(value)
        except (TypeError, ValueError):
            return None

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
