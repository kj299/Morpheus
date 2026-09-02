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

import pytest

from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc1_normalize_stage import TC1NormalizeStage
from morpheus.utils.type_utils import get_df_class

# Two collectors, samples stamped in the same second, delivered scrambled.
SCRAMBLED = {
    "event_time": [30, 10, 10, 20],
    "collector_id": ["col-a", "col-b", "col-a", "col-a"],
    "collector_seq": [3, 1, 1, 2],
    "value": ["d", "b", "a", "c"],
}


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return series.tolist()


def test_execution_modes(config: Config):
    assert issubclass(TotalOrderStage, GpuAndCpuMixin)

    assert set(TotalOrderStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(TotalOrderStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.gpu_and_cpu_mode
def test_rows_come_out_in_total_order(config: Config):
    meta = TotalOrderStage(config).on_data(MessageMeta(get_df_class(config.execution_mode)(SCRAMBLED)))

    # event_time first, then collector_id, then collector_seq: the tie at t=10 is broken by the collector name.
    assert _as_list(meta, "value") == ["a", "b", "c", "d"]


@pytest.mark.gpu_and_cpu_mode
def test_the_index_is_fresh(config: Config):
    # The cumulative primitives downstream realign by index and need it unique and in sorted position.
    meta = TotalOrderStage(config).on_data(MessageMeta(get_df_class(config.execution_mode)(SCRAMBLED)))
    df = meta.copy_dataframe()
    index = df.index.to_pandas() if hasattr(df.index, "to_pandas") else df.index

    assert list(index) == [0, 1, 2, 3]


@pytest.mark.gpu_and_cpu_mode
def test_ties_are_refused_by_default(config: Config):
    payload = dict(SCRAMBLED)
    payload["collector_seq"] = [1, 1, 1, 1]
    payload["collector_id"] = ["col-a"] * 4

    with pytest.raises(ValueError, match="tied"):
        TotalOrderStage(config).on_data(MessageMeta(get_df_class(config.execution_mode)(payload)))


@pytest.mark.gpu_and_cpu_mode
def test_ties_can_be_accepted_explicitly(config: Config):
    payload = dict(SCRAMBLED)
    payload["collector_seq"] = [1, 1, 1, 1]
    payload["collector_id"] = ["col-a"] * 4

    meta = TotalOrderStage(config,
                           require_total_order=False).on_data(MessageMeta(get_df_class(config.execution_mode)(payload)))

    # Stable: the two t=10 rows keep their input order.
    assert _as_list(meta, "value") == ["b", "a", "c", "d"]


@pytest.mark.gpu_and_cpu_mode
def test_control_message_payload_is_replaced(config: Config):
    message = ControlMessage()
    message.payload(MessageMeta(get_df_class(config.execution_mode)(SCRAMBLED)))

    result = TotalOrderStage(config).on_data(message)

    assert result is message
    assert _as_list(message.payload(), "value") == ["a", "b", "c", "d"]


@pytest.mark.gpu_and_cpu_mode
def test_a_scrambled_batch_yields_the_same_deltas_as_an_ordered_one(config: Config):
    # The reason the stage exists: a counter delta is the difference from the previous sample, so the same rows in
    # a different order produce different features. Ordered first, the deltas are a function of the data alone.
    ordered = {
        "site_id": ["hq"] * 3,
        "device_id": ["sw1"] * 3,
        "port_id": ["Gi1/0/1"] * 3,
        "event_time": [0, 60 * 10**9, 120 * 10**9],
        "collector_id": ["col-a"] * 3,
        "collector_seq": [1, 2, 3],
        "crc_errors": [100, 110, 125],
        "symbol_errors": [0, 0, 0],
        "input_discards": [0, 0, 0],
        "output_discards": [0, 0, 0],
    }
    scrambled = {name: [values[2], values[0], values[1]] for (name, values) in ordered.items()}
    df_class = get_df_class(config.execution_mode)

    def deltas(payload: dict, sort: bool) -> list:
        meta = MessageMeta(df_class(payload))

        if (sort):
            meta = TotalOrderStage(config).on_data(meta)

        TC1NormalizeStage(config).on_data(meta)
        values = _as_list(meta, "crc_errors_delta")

        import pandas as pd
        return [None if pd.isna(value) else int(value) for value in values]

    assert deltas(ordered, sort=False) == [None, 10, 15]
    assert deltas(scrambled, sort=True) == [None, 10, 15]
    # And without the sort the scrambled batch answers a different question: the third sample arrives first and
    # the rest read as out of order.
    assert deltas(scrambled, sort=False) != [None, 10, 15]


@pytest.mark.gpu_and_cpu_mode
def test_total_order_stage_pipe(config: Config):
    source_df = get_df_class(config.execution_mode)(SCRAMBLED)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TotalOrderStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "value") == ["a", "b", "c", "d"]


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError):
        TotalOrderStage(config, order_columns=[])
