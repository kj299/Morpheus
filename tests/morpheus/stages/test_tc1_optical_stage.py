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
from morpheus.stages.telemetry.tc1_optical_stage import TC1OpticalStage
from morpheus.utils.optical_baseline import NS_PER_SECOND
from morpheus.utils.type_utils import get_df_class

MINUTE_NS = 60 * NS_PER_SECOND


def frame(rx_levels: list, entity: str = "hq:sw1:Gi1/0/1", tx_levels: list = None) -> dict:
    count = len(rx_levels)

    return {
        "entity_key": [entity] * count,
        "event_time": [index * MINUTE_NS for index in range(count)],
        "optical_tx_dbm": [-2.0] * count if tx_levels is None else tx_levels,
        "optical_rx_dbm": rx_levels,
    }


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [None if value is None or value is pd.NA or value != value else value for value in series.tolist()]


def run(config: Config, payload: dict, **kwargs) -> MessageMeta:
    defaults = {"min_samples": 3}
    defaults.update(kwargs)
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC1OpticalStage(config, **defaults).on_data(meta)

    return meta


def test_execution_modes(config: Config):
    assert issubclass(TC1OpticalStage, GpuAndCpuMixin)

    assert set(TC1OpticalStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    needed = TC1OpticalStage(config).get_needed_columns()

    assert needed["optical_rx_dbm_baseline"] == TypeId.FLOAT64
    assert needed["optical_rx_dbm_deviation"] == TypeId.FLOAT64
    assert needed["optical_rx_dbm_baseline_samples"] == TypeId.INT64


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(TC1OpticalStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.gpu_and_cpu_mode
def test_a_tap_shows_as_a_negative_step(config: Config):
    # Four steady readings, then a 2 dB drop: a passive splitter diverting light to a tap.
    meta = run(config, frame([-7.0, -7.0, -7.0, -7.0, -9.0]))

    assert _as_list(meta, "optical_rx_dbm_deviation") == [None, None, None, 0.0, -2.0]
    assert _as_list(meta, "optical_rx_dbm_baseline") == [None, None, None, -7.0, -7.0]


@pytest.mark.gpu_and_cpu_mode
def test_a_healthy_but_dark_link_does_not_alarm(config: Config):
    # A long span legitimately sits far below a short one. Only the deviation is comparable across ports, which is
    # the whole reason this is a baseline rather than a threshold.
    meta = run(config, frame([-24.0] * 5))

    assert _as_list(meta, "optical_rx_dbm_deviation")[-1] == 0.0


@pytest.mark.gpu_and_cpu_mode
def test_ports_do_not_share_a_baseline(config: Config):
    payload = {
        "entity_key": ["hq:sw1:Gi1/0/1", "hq:sw1:Gi1/0/2"] * 4,
        "event_time": [index // 2 * MINUTE_NS for index in range(8)],
        "optical_tx_dbm": [-2.0] * 8,
        "optical_rx_dbm": [-7.0, -24.0] * 4,
    }

    meta = run(config, payload, min_samples=2)

    # Each port is scored against itself, so neither the bright nor the dark link registers a deviation.
    assert _as_list(meta, "optical_rx_dbm_deviation")[-2:] == [0.0, 0.0]


@pytest.mark.gpu_and_cpu_mode
def test_transmit_and_receive_are_scored_separately(config: Config):
    payload = frame([-7.0] * 5, tx_levels=[-2.0, -2.0, -2.0, -2.0, -5.0])

    meta = run(config, payload)

    # The local laser fading is a different fault from losing light in the fibre, so the two must not be pooled.
    assert _as_list(meta, "optical_tx_dbm_deviation")[-1] == -3.0
    assert _as_list(meta, "optical_rx_dbm_deviation")[-1] == 0.0


@pytest.mark.gpu_and_cpu_mode
def test_a_copper_port_carries_nulls_rather_than_a_baseline(config: Config):
    # A port with no optic reports no level. The stage has to survive a mixed estate without inventing a reference.
    meta = run(config, frame([None] * 5))

    assert _as_list(meta, "optical_rx_dbm_baseline") == [None] * 5
    assert _as_list(meta, "optical_rx_dbm_deviation") == [None] * 5
    assert _as_list(meta, "optical_rx_dbm_baseline_samples") == [0] * 5


@pytest.mark.gpu_and_cpu_mode
def test_warm_up_publishes_nothing(config: Config):
    meta = run(config, frame([-7.0, -7.0]), min_samples=5)

    assert _as_list(meta, "optical_rx_dbm_deviation") == [None, None]
    assert _as_list(meta, "optical_rx_dbm_baseline_samples") == [0, 1]


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    df_class = get_df_class(config.execution_mode)
    stage = TC1OpticalStage(config, min_samples=3)

    def message(start: int, levels: list):
        return MessageMeta(
            df_class({
                "entity_key": ["hq:sw1:Gi1/0/1"] * len(levels),
                "event_time": [(start + index) * MINUTE_NS for index in range(len(levels))],
                "optical_tx_dbm": [-2.0] * len(levels),
                "optical_rx_dbm": levels,
            }))

    stage.on_data(message(0, [-7.0, -7.0, -7.0]))
    second = message(3, [-9.0])
    stage.on_data(second)

    # The baseline spans the message boundary, which is the whole reason this stage is stateful.
    assert _as_list(second, "optical_rx_dbm_deviation") == [-2.0]


@pytest.mark.gpu_and_cpu_mode
def test_warm_up_gaps_are_nulls_rather_than_nan(config: Config):
    meta = run(config, frame([-7.0, -7.0, -7.0, -7.0]))
    deviation = meta.get_data("optical_rx_dbm_deviation")

    # The three warm-up rows have no baseline yet, and those gaps have to be nulls. NaN is not valid JSON, so a
    # frame carrying NaN under one execution mode and null under the other puts a different document on the wire.
    assert int(deviation.isna().sum()) == 3

    if (config.execution_mode == ExecutionMode.CPU):
        # Under pandas that takes the nullable extension type; a plain float64 column stores NaN in the gaps.
        assert pd.api.types.is_extension_array_dtype(deviation.dtype)

    # The sample count is never null, so it is integral in both modes without help; assert it anyway, because a
    # single null appearing here later would silently widen the column to float.
    assert pd.api.types.is_integer_dtype(meta.get_data("optical_rx_dbm_baseline_samples").dtype)


@pytest.mark.gpu_and_cpu_mode
def test_out_of_order_sample_is_flagged_not_scored(config: Config):
    payload = {
        "entity_key": ["hq:sw1:Gi1/0/1"] * 5,
        "event_time": [0, MINUTE_NS, 2 * MINUTE_NS, 3 * MINUTE_NS, MINUTE_NS],
        "optical_tx_dbm": [-2.0] * 5,
        "optical_rx_dbm": [-7.0, -7.0, -7.0, -7.0, -30.0],
    }

    meta = run(config, payload)

    # The late reading is not scored, and must not have entered the history either.
    assert _as_list(meta, "optical_rx_dbm_deviation")[-1] is None


@pytest.mark.gpu_and_cpu_mode
def test_raw_levels_are_preserved(config: Config):
    meta = run(config, frame([-7.0, -7.0, -7.0, -9.0]))

    assert _as_list(meta, "optical_rx_dbm") == [-7.0, -7.0, -7.0, -9.0]
    assert meta.count == 4


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    message = ControlMessage()
    message.payload(MessageMeta(get_df_class(config.execution_mode)(frame([-7.0] * 5))))

    TC1OpticalStage(config, min_samples=3).on_data(message)

    assert _as_list(message.payload(), "optical_rx_dbm_deviation")[-1] == 0.0


@pytest.mark.gpu_and_cpu_mode
def test_tc1_optical_stage_pipe(config: Config):
    source_df = get_df_class(config.execution_mode)(frame([-7.0, -7.0, -7.0, -7.0, -9.0]))

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC1OpticalStage(config, min_samples=3))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "optical_rx_dbm_deviation")[-1] == -2.0


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = frame([-7.0])
    del payload["optical_rx_dbm"]

    with pytest.raises(KeyError, match="optical_rx_dbm"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError):
        TC1OpticalStage(config, channel_columns=[])

    with pytest.raises(ValueError):
        TC1OpticalStage(config, window_seconds=0)

    with pytest.raises(ValueError):
        TC1OpticalStage(config, min_samples=0)
