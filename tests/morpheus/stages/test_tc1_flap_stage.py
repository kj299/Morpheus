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
from morpheus.stages.telemetry.tc1_flap_stage import TC1FlapStage
from morpheus.utils.link_flap import NS_PER_SECOND
from morpheus.utils.type_utils import get_df_class

MINUTE_NS = 60 * NS_PER_SECOND


def frame(statuses: list, last_changes: list = None, entity: str = "hq:sw1:Gi1/0/1") -> dict:
    count = len(statuses)
    payload = {
        "entity_key": [entity] * count,
        "event_time": [index * MINUTE_NS for index in range(count)],
        "oper_status": statuses,
    }

    if (last_changes is not None):
        payload["last_change_time"] = last_changes

    return payload


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [None if value is None or value is pd.NA or value != value else value for value in series.tolist()]


def run(config: Config, payload: dict, **kwargs) -> MessageMeta:
    defaults = {"last_change_column": "last_change_time"}
    defaults.update(kwargs)
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC1FlapStage(config, **defaults).on_data(meta)

    return meta


def test_execution_modes(config: Config):
    assert issubclass(TC1FlapStage, GpuAndCpuMixin)

    assert set(TC1FlapStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    needed = TC1FlapStage(config).get_needed_columns()

    assert needed["link_flaps"] == TypeId.INT64
    assert needed["link_flaps_in_window"] == TypeId.INT64
    assert needed["link_flap_device_reset"] == TypeId.BOOL8


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(TC1FlapStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.gpu_and_cpu_mode
def test_a_stable_port_never_counts(config: Config):
    meta = run(config, frame(["up"] * 4, last_changes=[0] * 4))

    assert _as_list(meta, "link_flaps") == [None, 0, 0, 0]
    assert _as_list(meta, "link_flaps_in_window") == [None, 0, 0, 0]


@pytest.mark.gpu_and_cpu_mode
def test_a_flap_inside_the_polling_gap_is_counted(config: Config):
    # Both polls saw "up", so a status comparison would call this port stable. It dropped and recovered between
    # them, which is the flapping port a naive diff misses.
    meta = run(config, frame(["up", "up"], last_changes=[0, 30 * NS_PER_SECOND]))

    assert _as_list(meta, "link_flaps") == [None, 2]
    assert _as_list(meta, "link_flap_unpolled") == [False, True]


@pytest.mark.gpu_and_cpu_mode
def test_a_visible_transition_counts_once(config: Config):
    meta = run(config, frame(["up", "down"], last_changes=[0, 30 * NS_PER_SECOND]))

    assert _as_list(meta, "link_flaps") == [None, 1]
    assert _as_list(meta, "link_flap_unpolled") == [False, False]


@pytest.mark.gpu_and_cpu_mode
def test_without_last_change_only_visible_transitions_are_seen(config: Config):
    # The honest degradation on a device that does not report the field.
    payload = frame(["up", "up", "down"])

    meta = run(config, payload, last_change_column=None)

    assert _as_list(meta, "link_flaps") == [None, 0, 1]


@pytest.mark.gpu_and_cpu_mode
def test_a_device_reboot_is_counted_and_labelled(config: Config):
    meta = run(config, frame(["up", "up"], last_changes=[10**12, 5 * NS_PER_SECOND]))

    # Two transitions, because the link went down with the device and came back, but labelled so a rule can
    # exclude a planned reboot rather than having it silently inflate flap counts.
    assert _as_list(meta, "link_flaps") == [None, 2]
    assert _as_list(meta, "link_flap_device_reset") == [False, True]


@pytest.mark.gpu_and_cpu_mode
def test_a_self_contradicting_device_is_flagged(config: Config):
    meta = run(config, frame(["up", "down"], last_changes=[1000, 1000]))

    assert _as_list(meta, "link_flaps") == [None, 1]
    assert _as_list(meta, "link_flap_last_change_inconsistent") == [False, True]


@pytest.mark.gpu_and_cpu_mode
def test_the_window_accumulates(config: Config):
    meta = run(config, frame(["up", "down", "up", "down"], last_changes=[0, MINUTE_NS, 2 * MINUTE_NS, 3 * MINUTE_NS]))

    assert _as_list(meta, "link_flaps_in_window") == [None, 1, 2, 3]


@pytest.mark.gpu_and_cpu_mode
def test_the_window_expires(config: Config):
    payload = {
        "entity_key": ["hq:sw1:Gi1/0/1"] * 3,
        "event_time": [0, MINUTE_NS, 40 * MINUTE_NS],
        "oper_status": ["up", "down", "down"],
        "last_change_time": [0, MINUTE_NS, MINUTE_NS],
    }

    meta = run(config, payload, window_seconds=10 * 60)

    # The transition aged out of the ten-minute window, so the port reads as quiet again rather than carrying its
    # history forever.
    assert _as_list(meta, "link_flaps_in_window") == [None, 1, 0]


@pytest.mark.gpu_and_cpu_mode
def test_ports_are_counted_separately(config: Config):
    payload = {
        "entity_key": ["hq:sw1:Gi1/0/1", "hq:sw1:Gi1/0/2"] * 2,
        "event_time": [0, 0, MINUTE_NS, MINUTE_NS],
        "oper_status": ["up", "up", "down", "up"],
        "last_change_time": [0, 0, MINUTE_NS, 0],
    }

    meta = run(config, payload)

    assert _as_list(meta, "link_flaps") == [None, None, 1, 0]


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    df_class = get_df_class(config.execution_mode)
    stage = TC1FlapStage(config, last_change_column="last_change_time")

    def message(event_time: int, status: str, last_change: int):
        return MessageMeta(
            df_class({
                "entity_key": ["hq:sw1:Gi1/0/1"],
                "event_time": [event_time],
                "oper_status": [status],
                "last_change_time": [last_change],
            }))

    stage.on_data(message(0, "up", 0))
    second = message(MINUTE_NS, "down", MINUTE_NS)
    stage.on_data(second)

    # The comparison spans the message boundary, which is the whole reason this stage is stateful.
    assert _as_list(second, "link_flaps") == [1]


@pytest.mark.gpu_and_cpu_mode
def test_warm_up_gaps_are_nulls_rather_than_nan(config: Config):
    meta = run(config, frame(["up", "up"], last_changes=[0, 0]))
    flaps = meta.get_data("link_flaps")

    # The first row has no predecessor and must read as a null, not as a zero and not as NaN: a zero would claim
    # the port was verified stable, and NaN is not valid JSON.
    assert int(flaps.isna().sum()) == 1

    if (config.execution_mode == ExecutionMode.CPU):
        assert pd.api.types.is_extension_array_dtype(flaps.dtype)


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    message = ControlMessage()
    payload = frame(["up", "down"], last_changes=[0, MINUTE_NS])
    message.payload(MessageMeta(get_df_class(config.execution_mode)(payload)))

    TC1FlapStage(config, last_change_column="last_change_time").on_data(message)

    assert _as_list(message.payload(), "link_flaps") == [None, 1]


@pytest.mark.gpu_and_cpu_mode
def test_tc1_flap_stage_pipe(config: Config):
    payload = frame(["up", "up", "down"], last_changes=[0, 30 * NS_PER_SECOND, 90 * NS_PER_SECOND])
    source_df = get_df_class(config.execution_mode)(payload)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC1FlapStage(config, last_change_column="last_change_time"))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "link_flaps_in_window") == [None, 2, 3]


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = frame(["up"], last_changes=[0])
    del payload["oper_status"]

    with pytest.raises(KeyError, match="oper_status"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError):
        TC1FlapStage(config, window_seconds=0)
