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
from morpheus.stages.telemetry.tc1_forecast_stage import TC1ForecastStage
from morpheus.utils.optical_forecast import NS_PER_SECOND
from morpheus.utils.optical_forecast import STATUS_BELOW_FLOOR
from morpheus.utils.optical_forecast import STATUS_IMMATURE
from morpheus.utils.optical_forecast import STATUS_NO_FLOOR
from morpheus.utils.optical_forecast import STATUS_NO_READING
from morpheus.utils.optical_forecast import STATUS_NONLINEAR
from morpheus.utils.optical_forecast import STATUS_NOT_DEGRADING
from morpheus.utils.optical_forecast import STATUS_PROJECTED
from morpheus.utils.type_utils import get_df_class

MINUTE_NS = 60 * NS_PER_SECOND
FLOORS = {"10GBASE-LR": -14.4, "1000BASE-LX": -19.0}

FAILING = [-7.0 - 0.1 * index for index in range(10)]
STEADY = [-7.0] * 10


def frame(levels: list,
          entity: str = "hq:sw1:Gi1/0/1",
          serials: list = None,
          types: list = None,
          times: list = None) -> dict:
    count = len(levels)

    return {
        "entity_key": [entity] * count,
        "event_time": [index * MINUTE_NS for index in range(count)] if times is None else times,
        "optical_rx_dbm": levels,
        "transceiver_serial": ["SN-A"] * count if serials is None else serials,
        "transceiver_type": ["10GBASE-LR"] * count if types is None else types,
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
    defaults = {"floors": FLOORS, "min_samples": 4}
    defaults.update(kwargs)
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC1ForecastStage(config, **defaults).on_data(meta)

    return meta


def test_execution_modes(config: Config):
    assert issubclass(TC1ForecastStage, GpuAndCpuMixin)

    assert set(TC1ForecastStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    needed = TC1ForecastStage(config).get_needed_columns()

    assert needed["optical_rx_dbm_trend_db_per_day"] == TypeId.FLOAT64
    assert needed["optical_rx_dbm_trend_residual_db"] == TypeId.FLOAT64
    assert needed["optical_rx_dbm_trend_significance"] == TypeId.FLOAT64
    assert needed["optical_rx_dbm_trend_samples"] == TypeId.INT64
    assert needed["optical_rx_dbm_floor_dbm"] == TypeId.FLOAT64
    assert needed["optical_rx_dbm_days_to_floor"] == TypeId.FLOAT64
    assert needed["optical_rx_dbm_forecast_status"] == TypeId.STRING


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(TC1ForecastStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.gpu_and_cpu_mode
def test_a_failing_optic_is_projected(config: Config):
    meta = run(config, frame(FAILING))

    statuses = _as_list(meta, "optical_rx_dbm_forecast_status")
    assert statuses[0:3] == [STATUS_IMMATURE] * 3
    assert statuses[3:] == [STATUS_PROJECTED] * 7

    # The line is at -7.9 now and falls 144 dB a day; the floor is 6.5 dB below it.
    assert _as_list(meta, "optical_rx_dbm_trend_db_per_day")[-1] == pytest.approx(-144.0)
    assert _as_list(meta, "optical_rx_dbm_floor_dbm")[-1] == -14.4
    assert _as_list(meta, "optical_rx_dbm_days_to_floor")[-1] == pytest.approx(6.5 / 144.0, abs=1e-4)
    assert _as_list(meta, "optical_rx_dbm_trend_samples")[-1] == 10


@pytest.mark.gpu_and_cpu_mode
def test_a_steady_optic_is_not_degrading(config: Config):
    meta = run(config, frame(STEADY))

    assert _as_list(meta, "optical_rx_dbm_forecast_status")[-1] == STATUS_NOT_DEGRADING
    assert _as_list(meta, "optical_rx_dbm_days_to_floor")[-1] is None


@pytest.mark.gpu_and_cpu_mode
def test_a_tap_is_a_step_and_not_a_forecast(config: Config):
    meta = run(config, frame(STEADY[0:8] + [-10.0] * 4))

    assert _as_list(meta, "optical_rx_dbm_forecast_status")[-1] == STATUS_NONLINEAR
    assert _as_list(meta, "optical_rx_dbm_days_to_floor")[-1] is None
    assert _as_list(meta, "optical_rx_dbm_trend_residual_db")[-1] > 0.5


@pytest.mark.gpu_and_cpu_mode
def test_the_floor_is_the_optics_own(config: Config):
    # Two ports losing light at the same rate from the same level, with different optics in them. The long-reach
    # optic works ten decibels further down, so it is given three times as long.
    payload = {
        "entity_key": ["hq:sw1:Gi1/0/1"] * 10 + ["hq:sw1:Gi1/0/2"] * 10,
        "event_time": [index * MINUTE_NS for index in range(10)] * 2,
        "optical_rx_dbm": FAILING * 2,
        "transceiver_serial": ["SN-A"] * 10 + ["SN-B"] * 10,
        "transceiver_type": ["10GBASE-LR"] * 10 + ["1000BASE-LX"] * 10,
    }
    meta = run(config, payload)

    floors = _as_list(meta, "optical_rx_dbm_floor_dbm")
    days = _as_list(meta, "optical_rx_dbm_days_to_floor")

    assert (floors[9], floors[19]) == (-14.4, -19.0)
    assert days[9] == pytest.approx(6.5 / 144.0, abs=1e-4)
    assert days[19] == pytest.approx(11.1 / 144.0, abs=1e-4)


@pytest.mark.gpu_and_cpu_mode
def test_the_type_is_matched_whatever_its_case_or_spacing(config: Config):
    meta = run(config, frame(FAILING, types=[" 10gbase-lr "] * 10))

    assert _as_list(meta, "optical_rx_dbm_floor_dbm")[-1] == -14.4
    assert _as_list(meta, "optical_rx_dbm_forecast_status")[-1] == STATUS_PROJECTED


@pytest.mark.gpu_and_cpu_mode
def test_an_unknown_type_has_no_floor_unless_a_default_is_given(config: Config):
    unknown = frame(FAILING, types=["40GBASE-SR4"] * 10)

    without = run(config, unknown)
    assert _as_list(without, "optical_rx_dbm_forecast_status")[-1] == STATUS_NO_FLOOR
    assert _as_list(without, "optical_rx_dbm_trend_db_per_day")[-1] == pytest.approx(-144.0)
    assert _as_list(without, "optical_rx_dbm_days_to_floor")[-1] is None

    with_default = run(config, unknown, default_floor_dbm=-10.0)
    assert _as_list(with_default, "optical_rx_dbm_forecast_status")[-1] == STATUS_PROJECTED
    assert _as_list(with_default, "optical_rx_dbm_floor_dbm")[-1] == -10.0
    assert _as_list(with_default, "optical_rx_dbm_days_to_floor")[-1] == pytest.approx(2.1 / 144.0, abs=1e-4)


@pytest.mark.gpu_and_cpu_mode
def test_a_frame_without_the_type_column_takes_the_default_floor(config: Config):
    payload = frame(FAILING)
    del payload["transceiver_type"]

    meta = run(config, payload, default_floor_dbm=-14.4)

    assert _as_list(meta, "optical_rx_dbm_forecast_status")[-1] == STATUS_PROJECTED


@pytest.mark.gpu_and_cpu_mode
def test_a_replaced_optic_starts_over(config: Config):
    # The failing optic is replaced after six polls; the new one is healthy. Its history begins at the swap, so the
    # trend it carries is its own and not the dead optic's.
    meta = run(config, frame(FAILING[0:6] + STEADY[0:6], serials=["SN-A"] * 6 + ["SN-B"] * 6))

    statuses = _as_list(meta, "optical_rx_dbm_forecast_status")
    assert statuses[5] == STATUS_PROJECTED
    assert statuses[6:9] == [STATUS_IMMATURE] * 3
    assert statuses[-1] == STATUS_NOT_DEGRADING
    assert _as_list(meta, "optical_rx_dbm_trend_samples")[6] == 1


@pytest.mark.gpu_and_cpu_mode
def test_fitting_across_a_replacement_is_a_choice_the_caller_makes(config: Config):
    meta = run(config, frame(FAILING[0:6] + STEADY[0:6], serials=["SN-A"] * 6 + ["SN-B"] * 6), optic_column=None)

    # Without an optic column the stage fits across the swap: the new optic's readings land on the old one's
    # history, and the recovered level reads as a line that stopped falling rather than as a fresh start.
    assert _as_list(meta, "optical_rx_dbm_trend_samples")[6] == 7
    assert _as_list(meta, "optical_rx_dbm_trend_samples")[-1] == 12
    assert _as_list(meta, "optical_rx_dbm_forecast_status")[-1] in (STATUS_NOT_DEGRADING, STATUS_NONLINEAR)
    assert STATUS_IMMATURE not in _as_list(meta, "optical_rx_dbm_forecast_status")[6:]


@pytest.mark.gpu_and_cpu_mode
def test_an_empty_cage_is_no_reading(config: Config):
    meta = run(config, frame(FAILING[0:5] + [None] + FAILING[5:9]))

    statuses = _as_list(meta, "optical_rx_dbm_forecast_status")
    assert statuses[5] == STATUS_NO_READING
    assert _as_list(meta, "optical_rx_dbm_trend_samples")[5] == 5
    assert statuses[6] == STATUS_PROJECTED


@pytest.mark.gpu_and_cpu_mode
def test_below_the_floor_is_reported_as_such(config: Config):
    meta = run(config, frame(FAILING), floors={"10GBASE-LR": -7.5})

    # The sixth reading is the floor itself.
    statuses = _as_list(meta, "optical_rx_dbm_forecast_status")
    assert statuses[4] == STATUS_PROJECTED
    assert statuses[5:] == [STATUS_BELOW_FLOOR] * 5
    assert _as_list(meta, "optical_rx_dbm_days_to_floor")[5] == 0.0


@pytest.mark.gpu_and_cpu_mode
def test_another_channel_can_be_fitted_and_names_its_own_columns(config: Config):
    payload = frame(STEADY)
    payload["optical_tx_dbm"] = FAILING

    meta = run(config, payload, channel_column="optical_tx_dbm")

    assert _as_list(meta, "optical_tx_dbm_forecast_status")[-1] == STATUS_PROJECTED
    assert "optical_rx_dbm_forecast_status" not in meta.get_column_names()


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    stage = TC1ForecastStage(config, floors=FLOORS, min_samples=4)
    df_class = get_df_class(config.execution_mode)

    first = MessageMeta(df_class(frame(FAILING[0:5])))
    stage.on_data(first)
    second = MessageMeta(df_class(frame(FAILING[5:10], times=[index * MINUTE_NS for index in range(5, 10)])))
    stage.on_data(second)

    assert _as_list(second, "optical_rx_dbm_trend_samples") == [6, 7, 8, 9, 10]
    assert _as_list(second, "optical_rx_dbm_forecast_status") == [STATUS_PROJECTED] * 5


@pytest.mark.gpu_and_cpu_mode
def test_out_of_order_sample_is_flagged_not_scored(config: Config):
    meta = run(config, frame(FAILING[0:5] + [-30.0], times=[0, 1, 2, 3, 4, 2]))

    statuses = _as_list(meta, "optical_rx_dbm_forecast_status")
    assert statuses[4] == STATUS_PROJECTED
    assert statuses[5] == STATUS_IMMATURE
    assert _as_list(meta, "optical_rx_dbm_trend_db_per_day")[5] is None


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    meta = MessageMeta(get_df_class(config.execution_mode)(frame(FAILING)))
    message = ControlMessage()
    message.payload(meta)

    returned = TC1ForecastStage(config, floors=FLOORS, min_samples=4).on_data(message)

    assert returned is message
    assert _as_list(message.payload(), "optical_rx_dbm_forecast_status")[-1] == STATUS_PROJECTED


@pytest.mark.gpu_and_cpu_mode
def test_tc1_forecast_stage_pipe(config: Config):
    source_df = get_df_class(config.execution_mode)(frame(FAILING))

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC1ForecastStage(config, floors=FLOORS, min_samples=4))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "optical_rx_dbm_forecast_status")[-1] == STATUS_PROJECTED


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = frame(FAILING)
    del payload["optical_rx_dbm"]

    with pytest.raises(KeyError, match="optical_rx_dbm"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError):
        TC1ForecastStage(config, channel_column="")

    with pytest.raises(ValueError):
        TC1ForecastStage(config, window_seconds=0)

    with pytest.raises(ValueError):
        TC1ForecastStage(config, min_samples=2)

    with pytest.raises(ValueError, match="blank"):
        TC1ForecastStage(config, floors={"   ": -14.4})


@pytest.mark.cpu_mode
def test_null_entity_keys_are_not_pooled_into_one_port(config: Config):
    payload = frame(FAILING)
    payload["entity_key"] = [None] * 10

    meta = run(config, payload)

    assert _as_list(meta, "optical_rx_dbm_forecast_status") == [None] * 10
    assert _as_list(meta, "optical_rx_dbm_trend_samples") == [None] * 10


@pytest.mark.cpu_mode
def test_a_null_key_does_not_disturb_a_real_ones_state(config: Config):
    payload = frame(FAILING[0:5])
    payload["entity_key"][2] = None

    meta = run(config, payload)

    # The real port saw four readings; the keyless row in the middle entered nobody's history.
    assert _as_list(meta, "optical_rx_dbm_trend_samples") == [1, 2, None, 3, 4]
    assert _as_list(meta, "optical_rx_dbm_forecast_status")[-1] == STATUS_PROJECTED
