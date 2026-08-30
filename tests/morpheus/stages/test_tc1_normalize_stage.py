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
from morpheus.stages.telemetry.tc1_normalize_stage import TC1NormalizeStage
from morpheus.utils.counter_delta import NS_PER_SECOND
from morpheus.utils.type_utils import get_df_class

MINUTE_NS = 60 * NS_PER_SECOND
COUNTER32_CEILING = 1 << 32

# Three consecutive samples of one port: a baseline, a steady increase, then a reboot.
SAMPLES = {
    "site_id": ["hq", "hq", "hq"],
    "device_id": ["sw1", "sw1", "sw1"],
    "port_id": ["Gi1/0/1", "Gi1/0/1", "Gi1/0/1"],
    "event_time": [0, MINUTE_NS, 2 * MINUTE_NS],
    "uptime": [3600, 3660, 30],
    "crc_errors": [100, 142, 7],
    "symbol_errors": [0, 0, 0],
    "input_discards": [5, 5, 0],
    "output_discards": [1, 1, 0],
}


def make_stage(config: Config, **kwargs) -> TC1NormalizeStage:
    defaults = {"uptime_column": "uptime"}
    defaults.update(kwargs)

    return TC1NormalizeStage(config, **defaults)


def _as_list(meta: MessageMeta, column: str) -> list:
    """Read a column as host values, with every flavour of missing collapsed to `None`."""
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [None if value is None or value is pd.NA or value != value else value for value in series.tolist()]


@pytest.fixture(name="samples_df")
def samples_df_fixture(config: Config):
    yield get_df_class(config.execution_mode)(SAMPLES)


def test_execution_modes(config: Config):
    assert issubclass(TC1NormalizeStage, GpuAndCpuMixin)

    assert set(make_stage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    needed = make_stage(config).get_needed_columns()

    assert needed["entity_key"] == TypeId.STRING
    assert needed["interval_seconds"] == TypeId.FLOAT64
    assert needed["counter_reset"] == TypeId.BOOL8
    assert needed["crc_errors_delta"] == TypeId.INT64
    assert needed["output_discards_delta"] == TypeId.INT64


def test_cli_command_builds():
    # The lazy click build is where dict and Union annotations crash; this stage was written to avoid both.
    from click.testing import CliRunner

    registration = getattr(TC1NormalizeStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.gpu_and_cpu_mode
def test_tc1_normalize_stage_pipe(config: Config, samples_df):
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[samples_df]))
    pipe.add_stage(make_stage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1

    assert _as_list(messages[0], "entity_key") == ["hq:sw1:Gi1/0/1"] * 3
    # First sample has no predecessor, second is a steady increase, third follows a reboot.
    assert _as_list(messages[0], "crc_errors_delta") == [None, 42, 7]
    assert _as_list(messages[0], "counter_reset") == [False, False, True]


@pytest.mark.gpu_and_cpu_mode
def test_entity_key_composes_the_three_identifiers(config: Config):
    df_class = get_df_class(config.execution_mode)
    meta = MessageMeta(
        df_class({
            "site_id": ["hq", "dc2"],
            "device_id": ["sw1", "sw9"],
            "port_id": ["Gi1/0/1", "Te1/1/4"],
            "event_time": [0, 0],
            "crc_errors": [0, 0],
            "symbol_errors": [0, 0],
            "input_discards": [0, 0],
            "output_discards": [0, 0],
        }))

    make_stage(config).on_data(meta)

    assert _as_list(meta, "entity_key") == ["hq:sw1:Gi1/0/1", "dc2:sw9:Te1/1/4"]


@pytest.mark.gpu_and_cpu_mode
def test_interval_is_capped_at_uptime_after_a_reset(config: Config, samples_df):
    meta = MessageMeta(samples_df)

    make_stage(config).on_data(meta)

    # The third sample follows a reboot 30 seconds before it, so its counters cover 30 seconds, not the 60 second
    # polling gap. A rate computed against the gap would understate the error burst by half.
    assert _as_list(meta, "interval_seconds") == [None, 60.0, 30.0]


@pytest.mark.gpu_and_cpu_mode
def test_thirty_two_bit_wrap_is_corrected(config: Config):
    df_class = get_df_class(config.execution_mode)
    meta = MessageMeta(
        df_class({
            "site_id": ["hq", "hq"],
            "device_id": ["sw1", "sw1"],
            "port_id": ["Gi1/0/1", "Gi1/0/1"],
            "event_time": [0, MINUTE_NS],
            "uptime": [3600, 3660],
            "crc_errors": [COUNTER32_CEILING - 10, 5],
            "symbol_errors": [0, 0],
            "input_discards": [0, 0],
            "output_discards": [0, 0],
        }))

    make_stage(config, counter32_columns=["crc_errors"]).on_data(meta)

    assert _as_list(meta, "crc_errors_delta") == [None, 15]
    assert _as_list(meta, "counter_wrapped") == [False, True]
    assert _as_list(meta, "counter_reset") == [False, False]


@pytest.mark.gpu_and_cpu_mode
def test_ports_do_not_contaminate_each_other(config: Config):
    df_class = get_df_class(config.execution_mode)
    meta = MessageMeta(
        df_class({
            "site_id": ["hq"] * 4,
            "device_id": ["sw1"] * 4,
            "port_id": ["Gi1/0/1", "Gi1/0/2", "Gi1/0/1", "Gi1/0/2"],
            "event_time": [0, 0, MINUTE_NS, MINUTE_NS],
            "uptime": [3600, 3600, 3660, 3660],
            "crc_errors": [100, 900, 110, 901],
            "symbol_errors": [0, 0, 0, 0],
            "input_discards": [0, 0, 0, 0],
            "output_discards": [0, 0, 0, 0],
        }))

    make_stage(config).on_data(meta)

    assert _as_list(meta, "crc_errors_delta") == [None, None, 10, 1]


@pytest.mark.gpu_and_cpu_mode
def test_out_of_order_sample_is_flagged_not_negated(config: Config):
    df_class = get_df_class(config.execution_mode)
    meta = MessageMeta(
        df_class({
            "site_id": ["hq"] * 3,
            "device_id": ["sw1"] * 3,
            "port_id": ["Gi1/0/1"] * 3,
            "event_time": [0, 2 * MINUTE_NS, MINUTE_NS],
            "uptime": [3600, 3720, 3660],
            "crc_errors": [100, 200, 150],
            "symbol_errors": [0, 0, 0],
            "input_discards": [0, 0, 0],
            "output_discards": [0, 0, 0],
        }))

    make_stage(config).on_data(meta)

    # A naive subtraction would report minus fifty errors for the late sample.
    assert _as_list(meta, "sample_out_of_order") == [False, False, True]
    assert _as_list(meta, "crc_errors_delta") == [None, 100, None]


@pytest.mark.gpu_and_cpu_mode
def test_decrease_without_uptime_yields_no_delta(config: Config):
    df_class = get_df_class(config.execution_mode)
    meta = MessageMeta(
        df_class({
            "site_id": ["hq", "hq"],
            "device_id": ["sw1", "sw1"],
            "port_id": ["Gi1/0/1", "Gi1/0/1"],
            "event_time": [0, MINUTE_NS],
            "crc_errors": [4000, 5],
            "symbol_errors": [0, 0],
            "input_discards": [0, 0],
            "output_discards": [0, 0],
        }))

    make_stage(config, uptime_column=None).on_data(meta)

    assert _as_list(meta, "counter_reset") == [False, True]
    assert _as_list(meta, "crc_errors_delta") == [None, None]


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    df_class = get_df_class(config.execution_mode)
    stage = make_stage(config)

    def frame(event_time, crc):
        return df_class({
            "site_id": ["hq"],
            "device_id": ["sw1"],
            "port_id": ["Gi1/0/1"],
            "event_time": [event_time],
            "uptime": [3600 + event_time // NS_PER_SECOND],
            "crc_errors": [crc],
            "symbol_errors": [0],
            "input_discards": [0],
            "output_discards": [0],
        })

    stage.on_data(MessageMeta(frame(0, 100)))
    second = MessageMeta(frame(MINUTE_NS, 130))
    stage.on_data(second)

    # The delta spans the message boundary, which is the whole reason this stage is stateful.
    assert _as_list(second, "crc_errors_delta") == [30]


@pytest.mark.gpu_and_cpu_mode
def test_gapped_columns_keep_their_declared_type(config: Config, samples_df):
    meta = MessageMeta(samples_df)

    make_stage(config).on_data(meta)

    # The first row of every entity has no predecessor, so these columns always carry a null. Assigning a list with
    # a `None` in it lets pandas widen an integer count to float while cuDF keeps it integral, which would make the
    # emitted schema depend on execution mode. A count is an integer in both modes or the envelope is not a contract.
    assert pd.api.types.is_integer_dtype(meta.get_data("crc_errors_delta").dtype)
    assert pd.api.types.is_float_dtype(meta.get_data("interval_seconds").dtype)
    assert pd.api.types.is_bool_dtype(meta.get_data("counter_reset").dtype)


@pytest.mark.gpu_and_cpu_mode
def test_raw_counters_are_preserved(config: Config, samples_df):
    meta = MessageMeta(samples_df)

    make_stage(config).on_data(meta)

    assert _as_list(meta, "crc_errors") == [100, 142, 7]
    assert meta.count == 3


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config, samples_df):
    message = ControlMessage()
    message.payload(MessageMeta(samples_df))

    make_stage(config).on_data(message)

    assert _as_list(message.payload(), "entity_key")[0] == "hq:sw1:Gi1/0/1"


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    df_class = get_df_class(config.execution_mode)
    meta = MessageMeta(df_class({"site_id": ["hq"], "device_id": ["sw1"], "port_id": ["Gi1/0/1"]}))

    with pytest.raises(KeyError, match="event_time"):
        make_stage(config).on_data(meta)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError):
        make_stage(config, counter_columns=[])

    with pytest.raises(ValueError):
        make_stage(config, delta_suffix="")

    with pytest.raises(ValueError, match="not in counter_columns"):
        make_stage(config, counter32_columns=["nope"])
