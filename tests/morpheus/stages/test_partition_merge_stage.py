#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""`PartitionMergeStage`: one event-time order out of partitions that arrive interleaved (plan section 4.4)."""

import itertools

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.stages.lineage.partition_merge_stage import PartitionMergeStage


def rows(*records) -> MessageMeta:
    """`(partition, offset, event_time, label)` tuples as a Kafka source's frame."""
    return MessageMeta(
        pd.DataFrame({
            "kafka_topic": ["t"] * len(records),
            "kafka_partition": [record[0] for record in records],
            "kafka_offset": [record[1] for record in records],
            "event_time": [record[2] for record in records],
            "label": [record[3] for record in records],
        }))


def labels(emitted: list) -> list:
    return [label for message in emitted for label in message.copy_dataframe()["label"].tolist()]


def run(stage: PartitionMergeStage, *messages) -> list:
    out = []

    for message in messages:
        out.extend(stage.on_data(message))

    out.extend(stage.on_completed() or [])

    return labels(out)


@pytest.mark.cpu_mode
def test_nothing_is_released_until_every_partition_has_spoken(config: Config):
    stage = PartitionMergeStage(config, expected_partitions=2)

    assert stage.on_data(rows((0, 0, 10, "a"), (0, 1, 20, "b"))) == []
    assert stage.pending_rows == 2

    released = stage.on_data(rows((1, 0, 15, "c")))

    # Partition 1 is at 15, so nothing before 15 can still arrive; 10 goes, 15 and 20 wait.
    assert labels(released) == ["a"]
    assert stage.pending_rows == 2


@pytest.mark.cpu_mode
def test_every_interleaving_releases_the_same_sequence(config: Config):
    zero = [(0, 0, 10, "a"), (0, 1, 30, "b"), (0, 2, 30, "c"), (0, 3, 50, "d")]
    one = [(1, 0, 20, "e"), (1, 1, 30, "f"), (1, 2, 40, "g")]
    seen = set()

    # Every way of interleaving the two partitions that keeps each in its own order.
    for picks in itertools.combinations(range(len(zero) + len(one)), len(zero)):
        (left, right) = (iter(zero), iter(one))
        sequence = [next(left) if position in picks else next(right) for position in range(len(zero) + len(one))]
        stage = PartitionMergeStage(config, expected_partitions=2)
        seen.add(tuple(run(stage, *[rows(record) for record in sequence])))

    # Ties at 30 are ordered by partition and then offset, whatever arrived first.
    assert seen == {("a", "e", "b", "c", "f", "g", "d")}


@pytest.mark.cpu_mode
def test_a_redelivered_message_is_dropped(config: Config):
    stage = PartitionMergeStage(config, expected_partitions=1)

    # A consumer restarted before its commit reads offsets 1 and 2 again.
    assert run(stage,
               rows((0, 0, 10, "a"), (0, 1, 20, "b"), (0, 2, 30, "c")),
               rows((0, 1, 20, "b"), (0, 2, 30, "c"), (0, 3, 40, "d"))) == ["a", "b", "c", "d"]


@pytest.mark.cpu_mode
def test_a_late_row_is_released_rather_than_lost(config: Config):
    stage = PartitionMergeStage(config, expected_partitions=1)

    assert labels(stage.on_data(rows((0, 0, 10, "a"), (0, 1, 30, "b")))) == ["a"]
    # Out of event-time order within its partition: earlier than what was already released. It is let through.
    assert labels(stage.on_data(rows((0, 2, 5, "late")))) == ["late"]


@pytest.mark.cpu_mode
def test_a_row_without_a_time_is_released_at_once(config: Config):
    stage = PartitionMergeStage(config, expected_partitions=2)

    assert labels(stage.on_data(rows((0, 0, None, "unparseable"), (0, 1, 10, "a")))) == ["unparseable"]


@pytest.mark.cpu_mode
def test_a_stop_for_a_checkpoint_keeps_the_held_rows(config: Config):
    stage = PartitionMergeStage(config, expected_partitions=2)
    stage.on_data(rows((0, 0, 10, "a"), (0, 1, 30, "c")))
    stage.suspend_end_of_stream()

    assert stage.on_completed() is None
    assert stage.pending_rows == 2

    resumed = PartitionMergeStage(config, expected_partitions=2)
    resumed.restore_state(stage.checkpoint_state())

    # Partition 1 arrives at 20 in the new process: everything before 20 is final, including what the old one held.
    assert labels(resumed.on_data(rows((1, 0, 20, "b")))) == ["a"]
    # And a message the old process had already read is still recognised as a repeat.
    assert resumed.on_data(rows((0, 1, 30, "c"))) == []
    assert labels(resumed.on_completed()) == ["b", "c"]


@pytest.mark.cpu_mode
def test_the_provenance_columns_are_kept_and_the_working_columns_are_not(config: Config):
    stage = PartitionMergeStage(config, expected_partitions=1)
    (message, ) = stage.on_data(rows((0, 0, 10, "a"))) + stage.on_completed()

    assert list(
        message.copy_dataframe().columns) == ["kafka_topic", "kafka_partition", "kafka_offset", "event_time", "label"]


@pytest.mark.cpu_mode
def test_a_frame_without_provenance_is_refused(config: Config):
    stage = PartitionMergeStage(config)

    with pytest.raises(KeyError, match="kafka_partition"):
        stage.on_data(MessageMeta(pd.DataFrame({"event_time": [1]})))


@pytest.mark.cpu_mode
def test_no_partitions_is_refused(config: Config):
    with pytest.raises(ValueError, match="expected_partitions"):
        PartitionMergeStage(config, expected_partitions=0)
