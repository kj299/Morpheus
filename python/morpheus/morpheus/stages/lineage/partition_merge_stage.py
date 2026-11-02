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
Merges a multi-partition Kafka topic into one event-time order that does not depend on how the partitions interleaved.

Kafka orders a partition and nothing else. A consumer reading two partitions receives them interleaved however the
brokers, the fetch sizes and the network happened to deliver them, and the interleaving changes from one run to the
next. `TotalOrderStage` sorts within a batch, which settles the order of rows that share a batch, but which rows
share a batch is itself the interleaving: replay the same topic twice and the batches hold different rows, so every
stage that keeps a stream-wide clock -- the event clock, the window sealer's watermark -- advances differently, and
the output differs (guide 13, plan item 4).

This stage removes the interleaving. It holds every row until each partition has delivered a row at least as late,
then releases what no partition can still precede, in `(event time, topic, partition, offset)` order. The low
watermark is the earliest of the partitions' latest event times, and a row strictly before it is final: a partition
whose own rows arrive in event-time order cannot deliver anything earlier. The rows released over a run are therefore
the same rows in the same order whatever the interleaving, and only where the releases fall between batches varies,
which the stateful stages are built to be indifferent to (control 13's batch-split sweep).

**A redelivered message is dropped.** A consumer that restarts before committing reads again from its last commit,
so at-least-once delivery repeats the messages after it. An offset at or below the highest already seen on its
partition is a repeat, and passing it on would count a flow twice; it is dropped and counted.

**What it assumes, and what it does when that fails.** Each partition must be in event-time order, which it is when
the producer keys messages by entity or by collector and the collector emits in time order, as the envelope requires.
A row earlier than the watermark already released is late: it is released at once rather than held, so it is not
lost, and it is counted, because from that row on the output again depends on arrival. A partition that never
delivers holds the watermark back, and everything waits for the end of the stream; `expected_partitions` names how
many partitions the subscription covers, so an idle one is visible as a stalled stage rather than as a silent
reordering.

A row without an event time is released at once, unordered: it is a record the envelope check will quarantine, and
holding it would hold nothing else back.
"""

import logging
import typing

import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.checkpoint_mixin import CheckpointMixin
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.column_assign import to_host_frame
from morpheus.utils.event_clock import event_times_ns

logger = logging.getLogger(__name__)

_TIME = "__merge_time_ns"
_SEPARATOR = "\x1f"


def _integer(value: typing.Any) -> typing.Optional[int]:
    if (value is None or isinstance(value, bool)):
        return None

    try:
        if (pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@register_stage("partition-merge")
class PartitionMergeStage(CheckpointMixin, GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Release a multi-partition topic's rows in one event-time order, whatever order the partitions arrived in.

    Place it directly behind a Kafka source reading more than one partition with provenance on, and ahead of
    `LineageStampStage` and `TotalOrderStage`.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    expected_partitions : int, default = 1
        How many partitions, across every subscribed topic, deliver to this stage. No row is released on the
        watermark until this many have each delivered one, so a partition that is late to start cannot be
        overtaken. Must be positive.
    topic_column : str, default = "kafka_topic"
        Column holding the topic a row was read from. A frame without it is treated as one topic.
    partition_column : str, default = "kafka_partition"
        Column holding the partition a row was read from.
    offset_column : str, default = "kafka_offset"
        Column holding the row's offset in its partition, which is how a redelivered message is recognised.
    time_column : str, default = "event_time"
        Column holding the row's event time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    """

    _checkpoint_attributes = ("_pending", "_latest", "_offsets", "_released_through", "_emit_control_messages")

    def __init__(self,
                 c: Config,
                 expected_partitions: int = 1,
                 topic_column: str = "kafka_topic",
                 partition_column: str = "kafka_partition",
                 offset_column: str = "kafka_offset",
                 time_column: str = "event_time",
                 time_unit: str = "ns"):
        super().__init__(c)

        if (expected_partitions <= 0):
            raise ValueError(f"expected_partitions must be positive, received {expected_partitions}")

        self._expected_partitions = expected_partitions
        self._topic_column = topic_column
        self._partition_column = partition_column
        self._offset_column = offset_column
        self._time_column = time_column
        self._time_unit = time_unit

        self._pending: typing.Optional[pd.DataFrame] = None
        self._latest: dict = {}
        """Partition to the latest event time it has delivered."""
        self._offsets: dict = {}
        """Partition to the highest offset it has delivered."""
        self._released_through: typing.Optional[int] = None
        """The watermark everything before which has been released."""
        self._emit_control_messages = False

    @property
    def name(self) -> str:
        """Stage name."""
        return "partition-merge"

    @property
    def pending_rows(self) -> int:
        """Rows held for a partition that has not yet caught up."""
        return 0 if self._pending is None else len(self._pending)

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

    def _to_message(self, df: pd.DataFrame) -> typing.Union[ControlMessage, MessageMeta]:
        # Imported here so that this module remains importable in CPU-only environments where cuDF is absent.
        from morpheus.utils.type_utils import get_df_class

        meta = MessageMeta(get_df_class(self._config.execution_mode)(df.reset_index(drop=True)))

        if (self._emit_control_messages):
            message = ControlMessage()
            message.payload(meta)

            return message

        return meta

    def _sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows in release order: event time, then topic, partition and offset, with timeless rows first."""
        order = df.assign(__has_time=df[_TIME].notna(), __topic=df["__key"].str.split(_SEPARATOR).str[0])
        order = order.sort_values(["__has_time", _TIME, "__topic", "__partition", "__offset"],
                                  kind="mergesort",
                                  na_position="first")

        return order.drop(columns=["__has_time", "__topic"])

    def _release(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._sorted(df).drop(columns=[_TIME, "__key", "__partition", "__offset"])

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]) -> list:
        """
        Hold the message's rows and release every row the watermark has passed.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Rows from a Kafka source, with their topic, partition and offset.

        Returns
        -------
        list
            Zero or one message of released rows.

        Raises
        ------
        KeyError
            If the partition, offset or time column is absent.
        """
        self._emit_control_messages = isinstance(message, ControlMessage)
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return []

        df = to_host_frame(meta.copy_dataframe()).reset_index(drop=True)
        missing = [
            column for column in (self._partition_column, self._offset_column, self._time_column)
            if column not in df.columns
        ]

        if (len(missing) > 0):
            raise KeyError(f"PartitionMergeStage requires columns {missing} which are not present in the DataFrame. "
                           f"Available columns: {sorted(df.columns)}")

        topics = (df[self._topic_column].tolist() if self._topic_column in df.columns else [""] * len(df))
        partitions = [_integer(value) for value in df[self._partition_column].tolist()]
        offsets = [_integer(value) for value in df[self._offset_column].tolist()]
        times = event_times_ns(df, self._time_column, self._time_unit)

        keep = []
        keys = []
        repeated = 0
        late = 0

        for (position, (topic, partition, offset, time)) in enumerate(zip(topics, partitions, offsets, times)):
            key = f"{'' if topic is None or pd.isna(topic) else topic}{_SEPARATOR}{partition}"
            keys.append(key)

            if (offset is not None):
                if (key in self._offsets and offset <= self._offsets[key]):
                    repeated += 1
                    continue

                self._offsets[key] = offset

            if (time is not None):
                self._latest[key] = max(self._latest.get(key, time), time)

                if (self._released_through is not None and time < self._released_through):
                    late += 1

            keep.append(position)

        arrived = df.iloc[keep].copy()
        arrived[_TIME] = pd.array([times[position] for position in keep], dtype="Int64")
        arrived["__key"] = [keys[position] for position in keep]
        arrived["__partition"] = pd.array([partitions[position] for position in keep], dtype="Int64")
        arrived["__offset"] = pd.array([offsets[position] for position in keep], dtype="Int64")

        pending = arrived if self._pending is None else pd.concat([self._pending, arrived], ignore_index=True)

        if (repeated > 0):
            logger.warning(
                "PartitionMergeStage dropped %d of %d rows whose offset its partition had already delivered; a "
                "consumer restarted before committing reads them again, and passing them on would count them twice.",
                repeated,
                len(df))

        if (late > 0):
            logger.warning(
                "PartitionMergeStage received %d rows earlier than the watermark it had already released; they are "
                "released at once and the output from them on depends on arrival order. Each partition must be in "
                "event-time order.",
                late)

        if (len(self._latest) >= self._expected_partitions):
            watermark = min(self._latest.values())
            self._released_through = (watermark if self._released_through is None else max(
                self._released_through, watermark))

        ready = pending[_TIME].isna()

        if (self._released_through is not None):
            ready = ready | (pending[_TIME] < self._released_through).fillna(False)

        released = pending[ready.to_numpy(dtype=bool)]
        held = pending[~ready.to_numpy(dtype=bool)]
        self._pending = held.reset_index(drop=True) if len(held) > 0 else None

        if (len(released) == 0):
            return []

        return [self._to_message(self._release(released))]

    def on_completed(self) -> typing.Optional[list]:
        """Release every held row when the stream ends, unless the end is a stop for a checkpoint."""
        if (self.end_of_stream_suspended or self._pending is None):
            return None

        released = self._release(self._pending)
        self._pending = None

        return [self._to_message(released)]

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name,
                                 ops.map(self.on_data),
                                 ops.filter(lambda x: len(x) > 0),
                                 ops.on_completed(self.on_completed),
                                 ops.flatten())
        builder.make_edge(input_node, node)

        return node
