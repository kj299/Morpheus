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
"""
A recorded Kafka topic, served to `KafkaSourceStage` without a broker, in whatever interleaving a test chooses.

A broker decides how a consumer's partitions interleave, and no test can ask it for a particular one. This stands in
for the consumer instead: each partition is a list of messages in offset order, and `interleave` says which partition
the next message comes from, so a test can replay the same topic in two orders and compare what comes out. Every
partition reports its end once drained, as a broker does with `enable.partition.eof`, and every commit is recorded.
"""

import itertools
import json
import zlib

import confluent_kafka as ck
import pandas as pd

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.kafka_source_stage import KafkaSourceStage
from morpheus.stages.lineage.partition_merge_stage import PartitionMergeStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage

TOPIC = "collector.tc4"


class _Error:

    def __init__(self, code: int):
        self._code = code

    def code(self) -> int:
        return self._code


class Message:
    """What `confluent_kafka.Consumer.poll` returns: a record, or a partition's end."""

    def __init__(self, topic: str, partition: int, offset: int, value: bytes = None, end: bool = False):
        self._topic = topic
        self._partition = partition
        self._offset = offset
        self._value = value
        self._end = end

    def topic(self) -> str:
        return self._topic

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset

    def value(self) -> bytes:
        return self._value

    def error(self):
        return _Error(ck.KafkaError._PARTITION_EOF) if self._end else None


class _Assigned:

    def __init__(self, topic: str, partition: int):
        self.topic = topic
        self.partition = partition


class RecordedConsumer:
    """
    A consumer of a recorded topic.

    Parameters
    ----------
    partitions : dict
        Partition number to its message values, in offset order.
    interleave : list of int
        The partition each successive message is taken from, until that partition is drained. Partitions left with
        messages once it is exhausted are drained in partition order.
    topic : str
        The topic name every message carries.
    """

    def __init__(self, partitions: dict, interleave: list, topic: str = TOPIC):
        self.topic = topic
        self.partitions = sorted(partitions)
        self.queues = {number: list(enumerate(values)) for (number, values) in partitions.items()}
        self.order = list(interleave)
        self.ended: set = set()
        self.committed: list = []
        self.subscribed = None
        self.closed = False

    def subscribe(self, topics: list):
        self.subscribed = list(topics)

    def assignment(self) -> list:
        return [_Assigned(self.topic, number) for number in self.partitions]

    def poll(self, timeout: float = None):
        del timeout

        while self.order:
            number = self.order.pop(0)

            if (self.queues[number]):
                (offset, value) = self.queues[number].pop(0)

                return Message(self.topic, number, offset, value)

        for number in self.partitions:
            if (self.queues[number]):
                (offset, value) = self.queues[number].pop(0)

                return Message(self.topic, number, offset, value)

        for number in self.partitions:
            if (number not in self.ended):
                self.ended.add(number)

                return Message(self.topic, number, -1, end=True)

        return None

    def commit(self, message=None, asynchronous: bool = True):
        del asynchronous
        self.committed.append((message.partition(), message.offset()))

    def close(self):
        self.closed = True


def partition_lines(lines: list[str], key: str, partitions: int = 2, time_key: str = "event_time") -> dict:
    """
    A producer's partitioning of collector records: keyed on one field, each partition in event-time order.

    The key is hashed with CRC-32, which is fixed, rather than with Python's string hash, which is not.
    """
    records = [json.loads(line) for line in lines]
    split: dict = {number: [] for number in range(partitions)}

    for (record, line) in sorted(zip(records, lines), key=lambda pair: pair[0][time_key]):
        number = zlib.crc32(str(record[key]).encode("utf-8")) % partitions
        split[number].append(line.encode("utf-8"))

    return split


def round_robin(partitions: dict) -> list:
    """Alternate between the partitions, one message each."""
    turns = itertools.zip_longest(*[[number] * len(partitions[number]) for number in sorted(partitions)])

    return [number for turn in turns for number in turn if number is not None]


def one_after_another(partitions: dict) -> list:
    """Every message of the last partition, then every message of the one before, and so on."""
    return [number for number in sorted(partitions, reverse=True) for _ in partitions[number]]


def replay(config: Config, consumer: RecordedConsumer, parser_kwargs: dict = None) -> list[pd.DataFrame]:
    """
    Read a recorded topic through `KafkaSourceStage` and `PartitionMergeStage`; return the frames released.

    The consumer is installed for the duration of the run and removed after it.
    """
    previous = KafkaSourceStage.__dict__.get("_consumer_factory")
    KafkaSourceStage._consumer_factory = staticmethod(lambda params: consumer)  # pylint: disable=protected-access

    try:
        pipe = LinearPipeline(config)
        pipe.set_source(
            KafkaSourceStage(config,
                             bootstrap_servers="recorded:9092",
                             input_topic=[consumer.topic],
                             poll_interval="1millis",
                             provenance=True,
                             parser_kwargs=parser_kwargs,
                             stop_at_end=True))
        pipe.add_stage(PartitionMergeStage(config, expected_partitions=len(consumer.partitions)))
        sink = pipe.add_stage(InMemorySinkStage(config))
        pipe.run()
    finally:
        KafkaSourceStage._consumer_factory = previous  # pylint: disable=protected-access

    frames = [message.copy_dataframe() for message in sink.get_messages()]

    return [frame.to_pandas() if hasattr(frame, "to_pandas") else frame for frame in frames]
