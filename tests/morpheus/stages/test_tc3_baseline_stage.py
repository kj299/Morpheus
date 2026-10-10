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
from morpheus.stages.telemetry.tc3_baseline_stage import TC3BaselineStage
from morpheus.stages.telemetry.tc3_cardinality_stage import TC3CardinalityStage
from morpheus.utils.bucket_peak import NS_PER_SECOND
from morpheus.utils.type_utils import get_df_class

HOUR_NS = 3600 * NS_PER_SECOND
HOST = "10.0.0.50"
SERVER = "10.0.1.20"


def frame(fan_out: list, fan_in: list = None, asymmetry: list = None, times: list = None) -> dict:
    count = len(fan_out)

    return {
        "src_ip": [HOST] * count,
        "dst_ip": [SERVER] * count,
        "event_time": [index * HOUR_NS for index in range(count)] if times is None else times,
        "dsts_per_src": fan_out,
        "srcs_per_dst": [1] * count if fan_in is None else fan_in,
        "byte_asymmetry": [0.25] * count if asymmetry is None else asymmetry,
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
    defaults = {"min_buckets": 4}
    defaults.update(kwargs)
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC3BaselineStage(config, **defaults).on_data(meta)

    return meta


def test_execution_modes(config: Config):
    assert issubclass(TC3BaselineStage, GpuAndCpuMixin)

    assert set(TC3BaselineStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    needed = TC3BaselineStage(config).get_needed_columns()

    for name in ("dsts_per_src", "srcs_per_dst"):
        assert needed[f"{name}_baseline_max"] == TypeId.INT64
        assert needed[f"{name}_baseline_buckets"] == TypeId.INT64
        assert needed[f"{name}_baseline_mature"] == TypeId.BOOL8
        assert needed[f"{name}_step"] == TypeId.INT64

    # A ratio, not a count, so its baseline and step are not rounded to one.
    assert needed["byte_asymmetry_baseline_max"] == TypeId.FLOAT64
    assert needed["byte_asymmetry_step"] == TypeId.FLOAT64


def test_a_history_can_be_left_out(config: Config):
    needed = TC3BaselineStage(config, asymmetry_column=None).get_needed_columns()

    assert "dsts_per_src_step" in needed
    assert not any(name.startswith("byte_asymmetry") for name in needed)


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(TC3BaselineStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.gpu_and_cpu_mode
def test_a_scan_is_a_step_above_the_hosts_own_history(config: Config):
    # Four hours reaching three addresses each, then an hour reaching forty: measured against the four hours
    # before it, which never reached more than three.
    meta = run(config, frame([3, 3, 3, 3, 40]))

    steps = _as_list(meta, "dsts_per_src_step")
    assert steps[0:4] == [None, None, None, None], "no baseline until four hours are committed"
    assert steps[4] == 37
    assert _as_list(meta, "dsts_per_src_baseline_max")[4] == 3
    assert _as_list(meta, "dsts_per_src_baseline_mature")[4] is True


@pytest.mark.gpu_and_cpu_mode
def test_a_busy_server_is_measured_against_its_own_busy_hours(config: Config):
    # The infrastructure case a literal threshold cannot tell from a scan: seventy every hour is no step.
    meta = run(config, frame([70, 64, 70, 66, 69]))

    assert _as_list(meta, "dsts_per_src_step")[4] == -1
    assert _as_list(meta, "dsts_per_src_baseline_max")[4] == 70


@pytest.mark.gpu_and_cpu_mode
def test_fan_in_is_kept_per_destination(config: Config):
    # Two sources reaching one destination: the fan-in history is the destination's, whichever source the row is.
    payload = {
        "src_ip": ["10.0.0.1", "10.0.0.2"] * 3,
        "dst_ip": [SERVER] * 6,
        "event_time": [0, 0, HOUR_NS, HOUR_NS, 2 * HOUR_NS, 2 * HOUR_NS],
        "dsts_per_src": [1] * 6,
        "srcs_per_dst": [1, 2, 1, 2, 1, 9],
        "byte_asymmetry": [0.1] * 6,
    }
    meta = run(config, payload, min_buckets=2)

    assert _as_list(meta, "srcs_per_dst_baseline_max")[4:] == [2, 2]
    assert _as_list(meta, "srcs_per_dst_step")[4:] == [-1, 7]
    # The sources each have their own fan-out history, two hours deep, regardless.
    assert _as_list(meta, "dsts_per_src_baseline_buckets")[4:] == [2, 2]


@pytest.mark.gpu_and_cpu_mode
def test_asymmetry_is_measured_against_the_hosts_own_envelope(config: Config):
    meta = run(config, frame([1] * 5, asymmetry=[0.2, 0.5, 0.3, 0.4, 12.0]))

    assert _as_list(meta, "byte_asymmetry_baseline_max")[4] == pytest.approx(0.5)
    assert _as_list(meta, "byte_asymmetry_step")[4] == pytest.approx(11.5)


@pytest.mark.gpu_and_cpu_mode
def test_a_group_scopes_the_history(config: Config):
    # The same address in two tenants is two hosts. Without the group the second tenant's busy hours would be the
    # first tenant's baseline.
    payload = frame([60, 60, 60, 60, 3, 3, 3, 3, 40], times=[index * HOUR_NS for index in range(4)] * 2 + [5 * HOUR_NS])
    payload["tenant"] = ["a"] * 4 + ["b"] * 5

    pooled = run(config, payload)
    scoped = run(config, payload, group_column="tenant")

    # Pooled, the second tenant's rows run backwards in time and are refused, and its scan is measured against the
    # first tenant's busy hours, which hide it.
    assert _as_list(pooled, "dsts_per_src_step")[-1] == -20
    assert _as_list(scoped, "dsts_per_src_step")[-1] == 37


@pytest.mark.gpu_and_cpu_mode
def test_a_null_group_gets_no_baseline(config: Config):
    payload = frame([3, 3, 3, 3, 40])
    payload["tenant"] = ["a", "a", "a", "a", None]

    meta = run(config, payload, group_column="tenant")

    assert _as_list(meta, "dsts_per_src_step")[-1] is None


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    stage = TC3BaselineStage(config, min_buckets=4)
    df_class = get_df_class(config.execution_mode)

    stage.on_data(MessageMeta(df_class(frame([3, 3, 3, 3]))))
    second = MessageMeta(df_class(frame([5], times=[4 * HOUR_NS])))
    stage.on_data(second)

    assert _as_list(second, "dsts_per_src_step") == [2]


@pytest.mark.gpu_and_cpu_mode
def test_composes_with_the_cardinality_stage(config: Config):
    # A host reaching one server an hour for four hours and then six servers in the fifth.
    sources = []
    destinations = []
    times = []

    for hour in range(5):
        for index in range(6 if hour == 4 else 1):
            sources.append(HOST)
            destinations.append(f"10.0.9.{index + 1}")
            # Each hour's first flow more than an hour after the last, so no window spans two of them.
            times.append(hour * (HOUR_NS + 60 * NS_PER_SECOND) + (1800 if hour == 4 else 0) * NS_PER_SECOND +
                         index * NS_PER_SECOND)

    payload = {
        "src_ip": sources,
        "dst_ip": destinations,
        "dst_port": [445] * len(sources),
        "event_time": times,
        "byte_asymmetry": [0.5] * len(sources),
    }
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC3CardinalityStage(config).on_data(meta)
    TC3BaselineStage(config, min_buckets=4).on_data(meta)

    assert _as_list(meta, "dsts_per_src_step")[-6:] == [0, 1, 2, 3, 4, 5]


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    meta = MessageMeta(get_df_class(config.execution_mode)(frame([3, 3, 3, 3, 4])))
    message = ControlMessage()
    message.payload(meta)

    returned = TC3BaselineStage(config, min_buckets=4).on_data(message)

    assert returned is message
    assert _as_list(message.payload(), "dsts_per_src_step")[-1] == 1


@pytest.mark.gpu_and_cpu_mode
def test_tc3_baseline_stage_pipe(config: Config):
    source_df = get_df_class(config.execution_mode)(frame([3, 3, 3, 3, 4]))

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC3BaselineStage(config, min_buckets=4))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "dsts_per_src_baseline_buckets")[-1] == 4


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = frame([3, 3])
    del payload["srcs_per_dst"]

    with pytest.raises(KeyError, match="srcs_per_dst"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError):
        TC3BaselineStage(config, fan_out_column=None, fan_in_column=None, asymmetry_column=None)

    with pytest.raises(ValueError):
        TC3BaselineStage(config, fan_in_column="dsts_per_src")

    with pytest.raises(ValueError):
        TC3BaselineStage(config, bucket_seconds=0)

    with pytest.raises(ValueError):
        TC3BaselineStage(config, min_buckets=0)

    with pytest.raises(ValueError):
        TC3BaselineStage(config, bucket_seconds=3600, window_seconds=60)


@pytest.mark.cpu_mode
def test_null_hosts_and_values_are_not_pooled(config: Config):
    payload = frame([3, 3, 3, 3, 3, 3])
    payload["src_ip"][2] = None
    payload["dsts_per_src"][3] = None

    meta = run(config, payload)

    assert _as_list(meta, "dsts_per_src_step")[2] is None
    assert _as_list(meta, "dsts_per_src_step")[3] is None
    # The real host's history is three hours deep at the last row, not five.
    assert _as_list(meta, "dsts_per_src_baseline_buckets")[5] == 3
