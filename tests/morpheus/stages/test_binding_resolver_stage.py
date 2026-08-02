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
from morpheus.stages.lineage.binding_resolver_stage import UNRESOLVED
from morpheus.stages.lineage.binding_resolver_stage import BindingResolverStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.binding_table import BindingTable
from morpheus.utils.type_utils import get_df_class

HOUR_NS = 3600 * NS_PER_SECOND

LEASES = pd.DataFrame({
    "ip": ["10.0.0.5", "10.0.0.5", "10.0.0.9"],
    "mac": ["aa:bb:cc:00:00:01", "aa:bb:cc:00:00:02", "aa:bb:cc:00:00:03"],
    "port_id": ["Gi1/0/1", "Gi1/0/2", "Gi1/0/9"],
    "bind_start": [0, 2 * HOUR_NS, 0],
    "bind_end": [HOUR_NS, 3 * HOUR_NS, 4 * HOUR_NS],
})

# Row 0 falls in the first lease, row 1 in the second, row 2 in the gap between them, row 3 on an unknown address.
OBSERVATIONS = {
    "src_ip": ["10.0.0.5", "10.0.0.5", "10.0.0.5", "192.0.2.1"],
    "event_time": [HOUR_NS // 2, int(2.5 * HOUR_NS), int(1.5 * HOUR_NS), HOUR_NS // 2],
}


@pytest.fixture(name="table")
def table_fixture() -> BindingTable:
    yield BindingTable.from_dataframe(LEASES,
                                      name="dhcp_lease",
                                      key_column="ip",
                                      value_columns=["mac", "port_id"],
                                      start_column="bind_start",
                                      end_column="bind_end")


@pytest.fixture(name="obs_df")
def obs_df_fixture(config: Config):
    yield get_df_class(config.execution_mode)(OBSERVATIONS)


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return series.tolist()


def test_execution_modes(config: Config, table: BindingTable):
    assert issubclass(BindingResolverStage, GpuAndCpuMixin)

    stage = BindingResolverStage(config, binding_table=table, key_column="src_ip")

    assert set(stage.supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config, table: BindingTable):
    stage = BindingResolverStage(config, binding_table=table, key_column="src_ip", uid_column="binding_uid")

    assert stage.get_needed_columns() == {
        "mac": TypeId.STRING,
        "port_id": TypeId.STRING,
        "resolution_method": TypeId.STRING,
        "binding_uid": TypeId.STRING,
    }


@pytest.mark.gpu_and_cpu_mode
def test_binding_resolver_stage_pipe(config: Config, table: BindingTable, obs_df):
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[obs_df]))
    pipe.add_stage(BindingResolverStage(config, binding_table=table, key_column="src_ip"))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "mac") == ["aa:bb:cc:00:00:01", "aa:bb:cc:00:00:02", None, None]
    assert _as_list(messages[0], "port_id") == ["Gi1/0/1", "Gi1/0/2", None, None]


@pytest.mark.gpu_and_cpu_mode
def test_method_column_marks_every_row(config: Config, table: BindingTable, obs_df):
    meta = MessageMeta(obs_df)

    BindingResolverStage(config, binding_table=table, key_column="src_ip").on_data(meta)

    # An unresolved row must be distinguishable from a resolved one, which is the whole point of the field.
    assert _as_list(meta, "resolution_method") == ["soft:dhcp_lease", "soft:dhcp_lease", UNRESOLVED, UNRESOLVED]


@pytest.mark.gpu_and_cpu_mode
def test_on_data_accepts_control_message(config: Config, table: BindingTable, obs_df):
    message = ControlMessage()
    message.payload(MessageMeta(obs_df))

    BindingResolverStage(config, binding_table=table, key_column="src_ip").on_data(message)

    assert _as_list(message.payload(), "mac")[0] == "aa:bb:cc:00:00:01"


@pytest.mark.gpu_and_cpu_mode
def test_input_columns_are_preserved(config: Config, table: BindingTable, obs_df):
    meta = MessageMeta(obs_df)

    BindingResolverStage(config, binding_table=table, key_column="src_ip").on_data(meta)

    assert set(OBSERVATIONS).issubset(set(meta.get_column_names()))
    assert meta.count == 4


@pytest.mark.gpu_and_cpu_mode
def test_resolution_is_repeatable(config: Config, table: BindingTable, obs_df):
    first = MessageMeta(obs_df.copy(deep=True))
    second = MessageMeta(obs_df.copy(deep=True))

    stage = BindingResolverStage(config, binding_table=table, key_column="src_ip", uid_column="binding_uid")
    stage.on_data(first)
    stage.on_data(second)

    assert _as_list(first, "binding_uid") == _as_list(second, "binding_uid")


@pytest.mark.gpu_and_cpu_mode
def test_uid_column_ties_back_to_the_binding(config: Config, table: BindingTable, obs_df):
    meta = MessageMeta(obs_df)

    BindingResolverStage(config, binding_table=table, key_column="src_ip", uid_column="binding_uid").on_data(meta)

    uids = _as_list(meta, "binding_uid")

    assert uids[0] == table.resolve("10.0.0.5", HOUR_NS // 2).uid
    assert uids[2] is None


@pytest.mark.gpu_and_cpu_mode
def test_output_columns_can_be_renamed_and_subset(config: Config, table: BindingTable, obs_df):
    meta = MessageMeta(obs_df)

    stage = BindingResolverStage(config,
                                 binding_table=table,
                                 key_column="src_ip",
                                 output_columns={"port_id": "switch_port"})
    stage.on_data(meta)

    assert _as_list(meta, "switch_port")[0] == "Gi1/0/1"
    assert "mac" not in meta.get_column_names()


@pytest.mark.gpu_and_cpu_mode
def test_second_resolution_event_times(config: Config, table: BindingTable):
    df_class = get_df_class(config.execution_mode)
    meta = MessageMeta(df_class({"src_ip": ["10.0.0.5"], "event_time": [1800]}))

    BindingResolverStage(config, binding_table=table, key_column="src_ip", time_unit="s").on_data(meta)

    assert _as_list(meta, "mac") == ["aa:bb:cc:00:00:01"]


@pytest.mark.gpu_and_cpu_mode
def test_datetime_event_times(config: Config, table: BindingTable):
    df_class = get_df_class(config.execution_mode)
    meta = MessageMeta(df_class({"src_ip": ["10.0.0.5"], "event_time": pd.to_datetime(["1970-01-01T00:30:00"])}))

    BindingResolverStage(config, binding_table=table, key_column="src_ip").on_data(meta)

    assert _as_list(meta, "mac") == ["aa:bb:cc:00:00:01"]


@pytest.mark.gpu_and_cpu_mode
def test_unresolved_can_be_made_fatal(config: Config, table: BindingTable, obs_df):
    meta = MessageMeta(obs_df)

    stage = BindingResolverStage(config, binding_table=table, key_column="src_ip", raise_on_unresolved=True)

    with pytest.raises(ValueError, match="did not resolve"):
        stage.on_data(meta)


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config, table: BindingTable):
    df_class = get_df_class(config.execution_mode)
    meta = MessageMeta(df_class({"src_ip": ["10.0.0.5"]}))

    with pytest.raises(KeyError, match="event_time"):
        BindingResolverStage(config, binding_table=table, key_column="src_ip").on_data(meta)


def test_constructor_validation(config: Config, table: BindingTable):
    with pytest.raises(ValueError):
        BindingResolverStage(config, binding_table=None, key_column="src_ip")

    with pytest.raises(ValueError):
        BindingResolverStage(config, binding_table=table, key_column="")

    with pytest.raises(ValueError):
        BindingResolverStage(config, binding_table=table, key_column="src_ip", method_column="")

    with pytest.raises(ValueError, match="does not provide"):
        BindingResolverStage(config, binding_table=table, key_column="src_ip", output_columns={"nope": "nope"})

    with pytest.raises(ValueError):
        BindingResolverStage(config, binding_table=table, key_column="src_ip", output_columns={})
