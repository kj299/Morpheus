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

import math

import pandas as pd
import pytest

from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc2_baseline_stage import TC2BaselineStage
from morpheus.stages.telemetry.tc2_cardinality_stage import TC2CardinalityStage
from morpheus.utils.bucket_peak import NS_PER_SECOND
from morpheus.utils.type_utils import get_df_class

MINUTE_NS = 60 * NS_PER_SECOND
PERIOD_NS = 5 * MINUTE_NS
PORT = "hq:sw1:Gi1/0/3"

# One device for eight five-minute snapshots, then a hub: the count the cardinality stage would have written.
STEADY = [1] * 8
HUB = [1, 2, 3, 4, 5]


def frame(counts: list, entity: str = PORT, times: list = None) -> dict:
    count = len(counts)

    return {
        "port_key": [entity] * count,
        "event_time": [index * PERIOD_NS for index in range(count)] if times is None else times,
        "macs_per_port": counts,
        "ports_per_mac": [1] * count,
    }


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [
        None if (value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))) else value
        for value in series.tolist()
    ]


def run(config: Config, payload: dict, **kwargs) -> MessageMeta:
    defaults = {"bucket_seconds": 300, "min_buckets": 4}
    defaults.update(kwargs)
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC2BaselineStage(config, **defaults).on_data(meta)

    return meta


def test_execution_modes(config: Config):
    assert issubclass(TC2BaselineStage, GpuAndCpuMixin)

    assert set(TC2BaselineStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    needed = TC2BaselineStage(config).get_needed_columns()

    assert needed["macs_per_port_baseline_max"] == TypeId.INT64
    assert needed["macs_per_port_baseline_buckets"] == TypeId.INT64
    assert needed["macs_per_port_baseline_mature"] == TypeId.BOOL8
    assert needed["macs_per_port_step"] == TypeId.INT64


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(TC2BaselineStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.gpu_and_cpu_mode
def test_a_hub_is_a_step_above_the_ports_own_history(config: Config):
    # Eight snapshots of one device, then a snapshot carrying five: every row of the hub's snapshot is measured
    # against the eight periods before it, and the largest step is the hub's size.
    payload = frame(STEADY + HUB, times=[index * PERIOD_NS for index in range(8)] + [8 * PERIOD_NS] * 5)
    meta = run(config, payload)

    steps = _as_list(meta, "macs_per_port_step")
    assert steps[0:4] == [None, None, None, None], "no baseline until four periods are committed"
    assert steps[4:8] == [0, 0, 0, 0]
    assert steps[8:] == [0, 1, 2, 3, 4]
    assert _as_list(meta, "macs_per_port_baseline_max")[-1] == 1
    assert _as_list(meta, "macs_per_port_baseline_buckets")[-1] == 8
    assert _as_list(meta, "macs_per_port_baseline_mature")[-1] is True


@pytest.mark.gpu_and_cpu_mode
def test_the_next_period_has_absorbed_the_hub(config: Config):
    payload = frame(STEADY + HUB + [5],
                    times=[index * PERIOD_NS for index in range(8)] + [8 * PERIOD_NS] * 5 + [9 * PERIOD_NS])
    meta = run(config, payload)

    assert _as_list(meta, "macs_per_port_baseline_max")[-1] == 5
    assert _as_list(meta, "macs_per_port_step")[-1] == 0


@pytest.mark.gpu_and_cpu_mode
def test_a_port_the_estate_has_only_just_met_has_no_baseline(config: Config):
    meta = run(config, frame([1, 7, 7]))

    assert _as_list(meta, "macs_per_port_baseline_mature") == [False, False, False]
    assert _as_list(meta, "macs_per_port_step") == [None, None, None]


@pytest.mark.gpu_and_cpu_mode
def test_ports_do_not_share_a_history(config: Config):
    payload = {
        "port_key": ["hq:sw1:Gi1/0/1"] * 5 + ["hq:sw1:Gi1/0/2"] * 5,
        "event_time": [index * PERIOD_NS for index in range(5)] * 2,
        "macs_per_port": [6, 6, 6, 6, 6] + [1, 1, 1, 1, 2],
        "ports_per_mac": [1] * 10,
    }
    meta = run(config, payload)

    assert _as_list(meta, "macs_per_port_step")[4] == 0
    assert _as_list(meta, "macs_per_port_step")[9] == 1


@pytest.mark.gpu_and_cpu_mode
def test_another_count_can_be_baselined_and_names_its_own_columns(config: Config):
    meta = run(config, frame(STEADY), value_column="ports_per_mac")

    assert _as_list(meta, "ports_per_mac_step")[-1] == 0
    assert "macs_per_port_step" not in meta.get_column_names()


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    stage = TC2BaselineStage(config, bucket_seconds=300, min_buckets=4)
    df_class = get_df_class(config.execution_mode)

    stage.on_data(MessageMeta(df_class(frame(STEADY[0:5]))))
    second = MessageMeta(df_class(frame([1, 1, 1, 4], times=[index * PERIOD_NS for index in range(5, 9)])))
    stage.on_data(second)

    assert _as_list(second, "macs_per_port_baseline_buckets") == [5, 6, 7, 8]
    assert _as_list(second, "macs_per_port_step") == [0, 0, 0, 3]


@pytest.mark.gpu_and_cpu_mode
def test_the_reference_does_not_depend_on_where_the_batches_fall(config: Config):
    # Every row of a period is measured against the periods before it, so cutting the stream mid-period changes
    # nothing. This is determinism control 5 for the one stage whose reference is a function of the period.
    rows = frame(STEADY + HUB, times=[index * PERIOD_NS for index in range(8)] + [8 * PERIOD_NS] * 5)
    df_class = get_df_class(config.execution_mode)

    whole = MessageMeta(df_class(rows))
    TC2BaselineStage(config, bucket_seconds=300, min_buckets=4).on_data(whole)

    stage = TC2BaselineStage(config, bucket_seconds=300, min_buckets=4)
    pieces = []

    for (start, stop) in ((0, 10), (10, 13)):
        piece = MessageMeta(df_class({name: values[start:stop] for (name, values) in rows.items()}))
        stage.on_data(piece)
        pieces.extend(_as_list(piece, "macs_per_port_step"))

    assert pieces == _as_list(whole, "macs_per_port_step")


@pytest.mark.gpu_and_cpu_mode
def test_composes_with_the_cardinality_stage(config: Config):
    # The columns this stage reads are the cardinality stage's, so the two are run together here the way the
    # telemetry pipeline runs them: a MAC table with one address on a port for eight snapshots, then four more.
    macs = []
    times = []

    for snapshot in range(9):
        addresses = ["aa:00:00:00:00:01"] + ([f"de:ad:be:ef:00:0{index}"
                                              for index in range(1, 5)] if snapshot == 8 else [])

        for address in addresses:
            macs.append(address)
            times.append(snapshot * PERIOD_NS)

    payload = {
        "mac_address": macs,
        "event_time": times,
        "site_id": ["hq"] * len(macs),
        "switch_id": ["sw1"] * len(macs),
        "port_id": ["Gi1/0/3"] * len(macs),
        "vlan_id": [10] * len(macs),
    }
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC2CardinalityStage(config).on_data(meta)
    TC2BaselineStage(config, bucket_seconds=300, min_buckets=4).on_data(meta)

    steps = _as_list(meta, "macs_per_port_step")
    first = _as_list(meta, "macs_per_port_first_in_window")

    # The hub's four addresses each raise the count above everything the port has carried, and each is new to the
    # window: the rows R-B-L2-002 fires on, and the address that was always there is not among them.
    assert steps[-5:] == [0, 1, 2, 3, 4]
    assert first[-4:] == [True, True, True, True]
    assert _as_list(meta, "port_key")[-1] == PORT


@pytest.mark.gpu_and_cpu_mode
def test_out_of_order_row_is_flagged_not_scored(config: Config):
    payload = frame(STEADY[0:5] + [9], times=[0, 1, 2, 3, 4, 2])
    payload["event_time"] = [value * PERIOD_NS for value in payload["event_time"]]

    meta = run(config, payload)

    assert _as_list(meta, "macs_per_port_step")[4] == 0
    assert _as_list(meta, "macs_per_port_step")[5] is None


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    meta = MessageMeta(get_df_class(config.execution_mode)(frame(STEADY)))
    message = ControlMessage()
    message.payload(meta)

    returned = TC2BaselineStage(config, bucket_seconds=300, min_buckets=4).on_data(message)

    assert returned is message
    assert _as_list(message.payload(), "macs_per_port_step")[-1] == 0


@pytest.mark.gpu_and_cpu_mode
def test_tc2_baseline_stage_pipe(config: Config):
    source_df = get_df_class(config.execution_mode)(frame(STEADY))

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC2BaselineStage(config, bucket_seconds=300, min_buckets=4))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "macs_per_port_baseline_buckets")[-1] == 7


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = frame(STEADY)
    del payload["macs_per_port"]

    with pytest.raises(KeyError, match="macs_per_port"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError):
        TC2BaselineStage(config, value_column="")

    with pytest.raises(ValueError):
        TC2BaselineStage(config, bucket_seconds=0)

    with pytest.raises(ValueError):
        TC2BaselineStage(config, min_buckets=0)

    with pytest.raises(ValueError):
        TC2BaselineStage(config, bucket_seconds=3600, window_seconds=60)


@pytest.mark.cpu_mode
def test_null_entities_and_counts_are_not_pooled(config: Config):
    payload = frame(STEADY[0:5])
    payload["port_key"][2] = None
    payload["macs_per_port"][3] = None

    meta = run(config, payload)

    # The keyless row and the countless row carry nulls; the real port's history is three periods deep after them,
    # not five.
    assert _as_list(meta, "macs_per_port_step")[2] is None
    assert _as_list(meta, "macs_per_port_step")[3] is None
    assert _as_list(meta, "macs_per_port_baseline_buckets")[4] == 2
