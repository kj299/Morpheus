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
from morpheus.stages.telemetry.tc2_auth_stage import TC2AuthStage
from morpheus.utils.session_timer import NS_PER_SECOND
from morpheus.utils.type_utils import get_df_class


def frame(results: list, times: list = None, ports: list = None) -> dict:
    count = len(results)

    return {
        "site_id": ["hq"] * count,
        "switch_id": ["sw1"] * count,
        "port_id": ["Gi1/0/1"] * count if ports is None else ports,
        "dot1x_result": results,
        "event_time": [index * NS_PER_SECOND for index in range(count)] if times is None else times,
    }


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [None if value is None or value is pd.NA or value != value else value for value in series.tolist()]


def run(config: Config, payload: dict, **kwargs) -> MessageMeta:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC2AuthStage(config, **kwargs).on_data(meta)

    return meta


def test_execution_modes(config: Config):
    assert issubclass(TC2AuthStage, GpuAndCpuMixin)

    assert set(TC2AuthStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    needed = TC2AuthStage(config).get_needed_columns()

    assert needed["auth_elapsed_seconds"] == TypeId.FLOAT64
    assert needed["auth_attempts"] == TypeId.INT64
    assert needed["auth_unpaired"] == TypeId.BOOL8


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(TC2AuthStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.gpu_and_cpu_mode
def test_a_paired_exchange_is_timed(config: Config):
    meta = run(config, frame(["started", "success"], times=[0, 3 * NS_PER_SECOND]))

    assert _as_list(meta, "auth_elapsed_seconds") == [None, 3.0]
    assert _as_list(meta, "auth_attempts") == [None, 1]
    assert _as_list(meta, "auth_unpaired") == [None, False]


@pytest.mark.gpu_and_cpu_mode
def test_a_start_row_carries_nulls_not_zeros(config: Config):
    # A start establishes what the outcome will be measured against; it is not itself an authorization. A zero
    # here would match a rule looking for "authorized instantly".
    meta = run(config, frame(["started"]))

    assert _as_list(meta, "auth_elapsed_seconds") == [None]
    assert _as_list(meta, "auth_unpaired") == [None]


@pytest.mark.gpu_and_cpu_mode
def test_authorization_with_no_exchange_is_flagged(config: Config):
    # The bypass signal. MAC authentication bypass and a device bridged behind an authorized supplicant both look
    # like this from the switch.
    meta = run(config, frame(["success"]))

    assert _as_list(meta, "auth_unpaired") == [True]
    assert _as_list(meta, "auth_elapsed_seconds") == [None]


@pytest.mark.gpu_and_cpu_mode
def test_a_null_result_starts_the_clock(config: Config):
    # A source that emits the request with no result column populated still opens an exchange.
    meta = run(config, frame([None, "success"], times=[0, 2 * NS_PER_SECOND]))

    assert _as_list(meta, "auth_elapsed_seconds") == [None, 2.0]
    assert _as_list(meta, "auth_unpaired") == [None, False]


@pytest.mark.gpu_and_cpu_mode
def test_retries_are_counted(config: Config):
    meta = run(config, frame(["started", "started", "started", "success"], times=[0, 5, 8, 9]))

    # A success after three attempts is not a first-time one, and timing from the last attempt alone would hide
    # the two before it.
    assert _as_list(meta, "auth_attempts") == [None, None, None, 3]


@pytest.mark.gpu_and_cpu_mode
def test_a_failure_is_timed_like_any_other_outcome(config: Config):
    meta = run(config, frame(["started", "failure"], times=[0, 4 * NS_PER_SECOND]))

    assert _as_list(meta, "auth_elapsed_seconds") == [None, 4.0]
    assert _as_list(meta, "auth_unpaired") == [None, False]


@pytest.mark.gpu_and_cpu_mode
def test_ports_are_timed_separately(config: Config):
    payload = frame(["started", "started", "success", "success"],
                    times=[0, 0, 2 * NS_PER_SECOND, 9 * NS_PER_SECOND],
                    ports=["Gi1/0/1", "Gi1/0/2", "Gi1/0/1", "Gi1/0/2"])

    meta = run(config, payload)

    assert _as_list(meta, "auth_elapsed_seconds") == [None, None, 2.0, 9.0]
    assert _as_list(meta, "auth_port_key")[0] == "hq:sw1:Gi1/0/1"


@pytest.mark.gpu_and_cpu_mode
def test_pending_values_are_configurable(config: Config):
    meta = run(config, frame(["radius-request", "accept"], times=[0, NS_PER_SECOND]), pending_values=["radius-request"])

    assert _as_list(meta, "auth_elapsed_seconds") == [None, 1.0]


@pytest.mark.gpu_and_cpu_mode
def test_an_outcome_before_its_start_is_flagged_not_negated(config: Config):
    payload = frame(["started", "success"], times=[10 * NS_PER_SECOND, 2 * NS_PER_SECOND])

    meta = run(config, payload)

    # A naive subtraction would report an authorization that took minus eight seconds.
    assert _as_list(meta, "auth_elapsed_seconds") == [None, None]


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    df_class = get_df_class(config.execution_mode)
    stage = TC2AuthStage(config)

    stage.on_data(MessageMeta(df_class(frame(["started"], times=[0]))))
    second = MessageMeta(df_class(frame(["success"], times=[6 * NS_PER_SECOND])))
    stage.on_data(second)

    # The exchange spans the message boundary, which is the whole reason this stage is stateful.
    assert _as_list(second, "auth_elapsed_seconds") == [6.0]
    assert _as_list(second, "auth_unpaired") == [False]


@pytest.mark.gpu_and_cpu_mode
def test_an_open_exchange_is_still_pending(config: Config):
    stage = TC2AuthStage(config)
    stage.on_data(MessageMeta(get_df_class(config.execution_mode)(frame(["started"]))))

    assert stage.pending_count == 1


@pytest.mark.gpu_and_cpu_mode
def test_nulls_survive_as_nulls_not_false(config: Config):
    meta = run(config, frame(["started", "success"], times=[0, NS_PER_SECOND]))
    unpaired = meta.get_data("auth_unpaired")

    # The start row answers neither question. Coerced to False it would read as "an exchange we confirmed was
    # properly paired", which nobody established.
    assert int(unpaired.isna().sum()) == 1

    if (config.execution_mode == ExecutionMode.CPU):
        assert pd.api.types.is_extension_array_dtype(unpaired.dtype)


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    message = ControlMessage()
    payload = frame(["started", "success"], times=[0, 2 * NS_PER_SECOND])
    message.payload(MessageMeta(get_df_class(config.execution_mode)(payload)))

    TC2AuthStage(config).on_data(message)

    assert _as_list(message.payload(), "auth_elapsed_seconds") == [None, 2.0]


@pytest.mark.gpu_and_cpu_mode
def test_tc2_auth_stage_pipe(config: Config):
    payload = frame(["started", "success", "success"], times=[0, 2 * NS_PER_SECOND, 4 * NS_PER_SECOND])
    source_df = get_df_class(config.execution_mode)(payload)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC2AuthStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    # The second success has no exchange left in front of it, which is the bypass shape.
    assert _as_list(messages[0], "auth_unpaired") == [None, False, True]


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = frame(["started"])
    del payload["dot1x_result"]

    with pytest.raises(KeyError, match="dot1x_result"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError):
        TC2AuthStage(config, timeout_seconds=0)
