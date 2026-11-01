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

import json

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.stages.output.siem_wire_sink_stage import SiemWireSinkStage
from morpheus.utils.type_utils import get_df_class


def records(config: Config) -> MessageMeta:
    return MessageMeta(
        get_df_class(config.execution_mode)(pd.DataFrame({
            "event_uid": ["a", "b"],
            "macs_per_port_step": [2.0, None],
            "port_key": ["hq:sw1:Gi1/0/1", "hq:sw1:Gi1/0/2"],
        })))


@pytest.mark.gpu_and_cpu_mode
def test_a_null_is_left_out_rather_than_sent(config: Config, tmp_path):
    # The defect the first search-head run found: R-B-L2-002 let four null steps through `macs_per_port_step>0`
    # because the field arrived present and null.
    path = tmp_path / "out.jsonlines"
    stage = SiemWireSinkStage(config, filename=str(path))
    stage.on_data(records(config))

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert lines[0]["macs_per_port_step"] == 2.0
    assert "macs_per_port_step" not in lines[1]
    assert stage.written == 2


@pytest.mark.cpu_mode
def test_append_continues_a_file_and_the_default_replaces_it(config: Config, tmp_path):
    path = tmp_path / "out.jsonlines"
    SiemWireSinkStage(config, filename=str(path)).on_data(records(config))
    SiemWireSinkStage(config, filename=str(path), append=True).on_data(records(config))

    assert len(path.read_text(encoding="utf-8").splitlines()) == 4

    SiemWireSinkStage(config, filename=str(path), overwrite=True)

    assert path.read_text(encoding="utf-8") == ""


@pytest.mark.cpu_mode
def test_an_existing_file_is_neither_replaced_nor_continued_unless_asked(config: Config, tmp_path):
    path = tmp_path / "out.jsonlines"
    path.write_text("kept\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        SiemWireSinkStage(config, filename=str(path))

    assert path.read_text(encoding="utf-8") == "kept\n"


@pytest.mark.cpu_mode
@pytest.mark.parametrize("kwargs",
                         [{}, {
                             "filename": "x", "kafka_topic": "t"
                         }, {
                             "filename": "x", "overwrite": True, "append": True
                         }])
def test_an_ambiguous_destination_is_refused(config: Config, kwargs: dict):
    with pytest.raises(ValueError):
        SiemWireSinkStage(config, **kwargs)


class _Recorder:
    """A producer whose broker acknowledges everything but the first `failures` produces to `failing_topic` (any)."""

    def __init__(self, servers: str, failures: int = 0, failing_topic: str = None):
        self.servers = servers
        self.produced: list = []
        self.keys: list = []
        self.flushed = 0
        self.failures = failures
        self.failing_topic = failing_topic
        self._reports: list = []

    def produce(self, topic: str, value: bytes, key: bytes = None, on_delivery=None):
        failed = self.failures > 0 and (self.failing_topic is None or topic == self.failing_topic)

        if (failed):
            self.failures -= 1
        else:
            self.produced.append((topic, value))
            self.keys.append(key)

        self._reports.append((on_delivery, "broker unavailable" if failed else None))

    def poll(self, timeout: float = None):
        del timeout

    def flush(self, timeout: float = None):
        del timeout
        self.flushed += 1

        for (report, error) in self._reports:
            report(error, None)

        self._reports.clear()

        return 0


def _install(monkeypatch, **recorder) -> list:
    built = []
    monkeypatch.setattr(SiemWireSinkStage,
                        "_producer_factory",
                        staticmethod(lambda servers: built.append(_Recorder(servers, **recorder)) or built[-1]))

    return built


@pytest.mark.cpu_mode
def test_kafka_receives_the_same_wire_lines(config: Config, monkeypatch):
    built = []
    monkeypatch.setattr(SiemWireSinkStage,
                        "_producer_factory",
                        staticmethod(lambda servers: built.append(_Recorder(servers)) or built[-1]))

    stage = SiemWireSinkStage(config, kafka_topic="morpheus.score.l2", kafka_bootstrap_servers="broker:9092")
    stage.on_data(records(config))
    stage.on_data(records(config))

    assert len(built) == 1
    producer = built[0]

    assert producer.servers == "broker:9092"
    assert [topic for (topic, _) in producer.produced] == ["morpheus.score.l2"] * 4
    assert "macs_per_port_step" not in json.loads(producer.produced[1][1])
    assert producer.flushed == 2


@pytest.mark.cpu_mode
def test_each_message_is_keyed_on_its_entity(config: Config, monkeypatch):
    # Keyed, one entity's records share a partition and keep their order for whatever reads the topic next.
    built = _install(monkeypatch)
    stage = SiemWireSinkStage(config, kafka_topic="morpheus.score.l2", kafka_bootstrap_servers="broker:9092")
    stage.on_data(records(config))

    lines = [json.loads(value) for (_, value) in built[0].produced]

    assert built[0].keys == [
        None if line.get("entity_key") is None else line["entity_key"].encode("utf-8") for line in lines
    ]


@pytest.mark.cpu_mode
def test_a_message_the_broker_refused_is_produced_again(config: Config, monkeypatch):
    built = _install(monkeypatch, failures=1, failing_topic="morpheus.score.l2")
    stage = SiemWireSinkStage(config, kafka_topic="morpheus.score.l2", kafka_bootstrap_servers="broker:9092")
    stage.on_data(records(config))

    assert len(built[0].produced) == 2
    assert stage.quarantined == 0


@pytest.mark.cpu_mode
def test_a_message_the_broker_keeps_refusing_goes_to_quarantine(config: Config, monkeypatch):
    built = _install(monkeypatch, failures=100, failing_topic="morpheus.score.l2")
    stage = SiemWireSinkStage(config,
                              kafka_topic="morpheus.score.l2",
                              kafka_bootstrap_servers="broker:9092",
                              max_retries=2,
                              quarantine_topic="morpheus.quarantine")
    stage.on_data(records(config))

    quarantined = [json.loads(value) for (topic, value) in built[0].produced if topic == "morpheus.quarantine"]

    assert stage.quarantined == 2
    assert len(quarantined) == 2
    assert {record["quarantine_reason"] for record in quarantined} == {"produce_failed"}
    assert {record["intended_topic"] for record in quarantined} == {"morpheus.score.l2"}
    assert all("broker unavailable" in record["error"] for record in quarantined)
    assert all(json.loads(record["payload"]) for record in quarantined)


@pytest.mark.cpu_mode
def test_a_message_with_nowhere_to_go_stops_the_pipeline(config: Config, monkeypatch):
    # The upstream sink logged a failed produce and carried on; a record lost that way leaves no trace downstream.
    _install(monkeypatch, failures=100, failing_topic="morpheus.score.l2")
    stage = SiemWireSinkStage(config, kafka_topic="morpheus.score.l2", kafka_bootstrap_servers="b:9092", max_retries=1)

    with pytest.raises(RuntimeError, match="Stopping rather than losing them"):
        stage.on_data(records(config))


@pytest.mark.cpu_mode
def test_a_quarantine_topic_that_also_refuses_stops_the_pipeline(config: Config, monkeypatch):
    _install(monkeypatch, failures=100, failing_topic=None)
    stage = SiemWireSinkStage(config,
                              kafka_topic="morpheus.score.l2",
                              kafka_bootstrap_servers="b:9092",
                              max_retries=0,
                              quarantine_topic="morpheus.quarantine")

    with pytest.raises(RuntimeError, match="or to the quarantine topic"):
        stage.on_data(records(config))
