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
"""
`KafkaSourceStage` with provenance, against a recorded topic rather than a broker (plan section 4.4).

The upstream tests of this stage need a broker and run in no tier. These read a recorded topic through the stage's
consumer factory, so what the fork changed -- provenance, parse before commit, a malformed line kept, a replay that
ends -- runs on every commit.
"""

import os
import sys

import pytest

from morpheus.config import Config
from morpheus.io.utils import get_json_reader
from morpheus.stages.input.kafka_source_stage import KafkaSourceStage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "determinism"))

# pylint: disable=wrong-import-position
import kafka_replay  # noqa: E402


def read(config: Config, partitions: dict, interleave: list, **kwargs):
    consumer = kafka_replay.RecordedConsumer(partitions, interleave)
    previous = KafkaSourceStage.__dict__["_consumer_factory"]
    KafkaSourceStage._consumer_factory = staticmethod(lambda params: consumer)  # pylint: disable=protected-access

    try:
        stage = KafkaSourceStage(config,
                                 bootstrap_servers="recorded:9092",
                                 input_topic=[kafka_replay.TOPIC],
                                 poll_interval="1millis",
                                 stop_at_end=True,
                                 **kwargs)
        stage._json_reader = get_json_reader(config.execution_mode)  # pylint: disable=protected-access
        frames = []

        class _Subscription:

            def is_subscribed(self) -> bool:
                return True

        for message in stage._source_generator(_Subscription()):  # pylint: disable=protected-access
            frames.append(message.copy_dataframe())
            # What had been committed when the batch was handed on.
            frames[-1].attrs["committed_before"] = list(consumer.committed)
    finally:
        KafkaSourceStage._consumer_factory = previous  # pylint: disable=protected-access

    return (frames, consumer)


@pytest.mark.cpu_mode
def test_each_row_says_where_it_was_read_from(config: Config):
    partitions = {0: [b'{"a": 1}', b'{"a": 2}'], 1: [b'{"a": 3}']}
    (frames, _) = read(config, partitions, [0, 1, 0], provenance=True)
    rows = [row for frame in frames for row in frame.to_dict(orient="records")]

    assert [(row["kafka_partition"], row["kafka_offset"], row["a"]) for row in rows] == [(0, 0, 1), (1, 0, 3),
                                                                                         (0, 1, 2)]
    assert {row["kafka_topic"] for row in rows} == {kafka_replay.TOPIC}


@pytest.mark.cpu_mode
def test_a_batch_is_committed_only_after_it_is_handed_on(config: Config):
    (frames, consumer) = read(config, {0: [b'{"a": 1}', b'{"a": 2}']}, [0, 0], provenance=True)

    assert frames[0].attrs["committed_before"] == []
    assert consumer.committed == [(0, 0), (0, 1)]


@pytest.mark.cpu_mode
def test_a_malformed_line_is_kept_in_its_place_rather_than_dropped(config: Config):
    # The upstream reader failed the whole batch on one bad line, logged it, and committed the batch anyway.
    (frames, consumer) = read(config, {0: [b'{"a": 1}', b'not json', b'[1, 2]', b'{"a": 4}']}, [0, 0, 0, 0],
                              provenance=True)
    rows = [row for frame in frames for row in frame.to_dict(orient="records")]

    assert [row["kafka_offset"] for row in rows] == [0, 1, 2, 3]
    assert [row["raw_payload"] for row in rows][1:3] == ["not json", "[1, 2]"]
    assert len(consumer.committed) == 4


@pytest.mark.cpu_mode
def test_a_tombstone_carries_no_row(config: Config):
    (frames, _) = read(config, {0: [b'{"a": 1}', None]}, [0, 0], provenance=True)

    assert sum(len(frame) for frame in frames) == 1


@pytest.mark.cpu_mode
def test_the_parser_arguments_reach_the_reader(config: Config):
    record = b'{"asn": "64512", "event_time": "2026-01-01T00:00:00"}'
    (guessed, _) = read(config, {0: [record]}, [0])
    (plain, _) = read(config, {0: [record]}, [0], parser_kwargs={"convert_dates": False, "dtype": False})

    # pandas' defaults turn an AS number sent as a string into a number; the fork's arguments keep what was sent.
    assert guessed[0]["asn"].tolist() == [64512]
    assert plain[0]["asn"].tolist() == ["64512"]


@pytest.mark.cpu_mode
def test_topics_are_subscribed_in_a_fixed_order(config: Config):
    stage = KafkaSourceStage(config, bootstrap_servers="recorded:9092", input_topic=["b", "a", "c", "a"])

    assert stage._topics == ["a", "b", "c"]  # pylint: disable=protected-access


@pytest.mark.cpu_mode
def test_the_python_reader_is_used_whenever_the_fork_s_options_are(config: Config):
    assert KafkaSourceStage(config, bootstrap_servers="b").supports_cpp_node()

    for options in ({"provenance": True}, {"parser_kwargs": {"dtype": False}}, {"stop_at_end": True}):
        assert not KafkaSourceStage(config, bootstrap_servers="b", **options).supports_cpp_node()
