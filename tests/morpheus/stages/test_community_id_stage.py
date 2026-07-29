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

from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.community_id_stage import CommunityIdStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.community_id import PROTO_UDP
from morpheus.utils.community_id import community_id
from morpheus.utils.type_utils import get_df_class

# One flow expressed in both directions, plus an unrelated UDP flow.
FLOW_DATA = {
    "src_ip": ["128.232.110.120", "66.35.250.204", "192.168.1.52"],
    "dest_ip": ["66.35.250.204", "128.232.110.120", "8.8.8.8"],
    "protocol": ["tcp", "tcp", "udp"],
    "src_port": [34855, 80, 54585],
    "dest_port": [80, 34855, 53],
}

EXPECTED_TCP_ID = "1:LQU9qZlK+B5F3KDmev6m5PMibrg="


@pytest.fixture(name="flow_df")
def flow_df_fixture(config: Config):
    yield get_df_class(config.execution_mode)(FLOW_DATA)


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return series.tolist()


def test_execution_modes(config: Config):
    assert issubclass(CommunityIdStage, GpuAndCpuMixin)

    stage = CommunityIdStage(config)

    assert set(stage.supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    stage = CommunityIdStage(config, output_column="flow_key")

    assert stage.get_needed_columns() == {"flow_key": TypeId.STRING}


@pytest.mark.gpu_and_cpu_mode
def test_community_id_stage_pipe(config: Config, flow_df):
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[flow_df]))
    pipe.add_stage(CommunityIdStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1

    values = _as_list(messages[0], "community_id")

    assert len(values) == 3
    assert values[0] == EXPECTED_TCP_ID
    # Both directions of the same flow collapse onto one identifier; that is the whole point of the hash.
    assert values[0] == values[1]
    assert values[2] == community_id("192.168.1.52", "8.8.8.8", PROTO_UDP, 54585, 53)


@pytest.mark.gpu_and_cpu_mode
def test_on_data_accepts_message_meta(config: Config, flow_df):
    meta = MessageMeta(flow_df)

    CommunityIdStage(config).on_data(meta)

    assert _as_list(meta, "community_id")[0] == EXPECTED_TCP_ID


@pytest.mark.gpu_and_cpu_mode
def test_on_data_accepts_control_message(config: Config, flow_df):
    message = ControlMessage()
    message.payload(MessageMeta(flow_df))

    CommunityIdStage(config).on_data(message)

    assert _as_list(message.payload(), "community_id")[0] == EXPECTED_TCP_ID


@pytest.mark.gpu_and_cpu_mode
def test_input_columns_are_preserved(config: Config, flow_df):
    meta = MessageMeta(flow_df)

    CommunityIdStage(config).on_data(meta)

    assert set(FLOW_DATA).issubset(set(meta.get_column_names()))
    assert meta.count == 3


@pytest.mark.gpu_and_cpu_mode
def test_portless_protocol(config: Config):
    df_class = get_df_class(config.execution_mode)
    meta = MessageMeta(df_class({"src_ip": ["10.1.2.3"], "dest_ip": ["10.4.5.6"], "protocol": [47]}))

    CommunityIdStage(config, src_port_column=None, dst_port_column=None).on_data(meta)

    assert _as_list(meta, "community_id") == [community_id("10.1.2.3", "10.4.5.6", 47)]


@pytest.mark.gpu_and_cpu_mode
def test_custom_columns_seed_and_rendering(config: Config):
    df_class = get_df_class(config.execution_mode)
    meta = MessageMeta(df_class({"s": ["10.0.0.1"], "d": ["10.0.0.2"], "p": [6], "sp": [1], "dp": [2]}))

    stage = CommunityIdStage(config,
                             src_ip_column="s",
                             dst_ip_column="d",
                             protocol_column="p",
                             src_port_column="sp",
                             dst_port_column="dp",
                             seed=7,
                             use_base64=False,
                             output_column="cid")
    stage.on_data(meta)

    assert _as_list(meta, "cid") == [community_id("10.0.0.1", "10.0.0.2", 6, 1, 2, seed=7, use_base64=False)]


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    df_class = get_df_class(config.execution_mode)
    meta = MessageMeta(df_class({"src_ip": ["1.2.3.4"]}))

    with pytest.raises(KeyError, match="dest_ip"):
        CommunityIdStage(config).on_data(meta)


@pytest.mark.gpu_and_cpu_mode
def test_malformed_row_does_not_discard_the_batch(config: Config):
    df_class = get_df_class(config.execution_mode)
    meta = MessageMeta(
        df_class({
            "src_ip": ["bogus", "10.0.0.1"],
            "dest_ip": ["10.0.0.2", "10.0.0.2"],
            "protocol": [6, 6],
            "src_port": [1, 1],
            "dest_port": [2, 2]
        }))

    CommunityIdStage(config).on_data(meta)

    values = _as_list(meta, "community_id")

    assert values[0] is None
    assert values[1] == community_id("10.0.0.1", "10.0.0.2", 6, 1, 2)


@pytest.mark.gpu_and_cpu_mode
def test_malformed_row_raises_when_requested(config: Config):
    df_class = get_df_class(config.execution_mode)
    meta = MessageMeta(
        df_class({
            "src_ip": ["bogus"], "dest_ip": ["10.0.0.2"], "protocol": [6], "src_port": [1], "dest_port": [2]
        }))

    with pytest.raises(ValueError):
        CommunityIdStage(config, raise_on_failure=True).on_data(meta)


@pytest.mark.parametrize("kwargs", [{
    "src_port_column": None
}, {
    "dst_port_column": None
}, {
    "seed": -1
}, {
    "seed": 70000
}])
def test_constructor_validation(config: Config, kwargs: dict):
    with pytest.raises(ValueError):
        CommunityIdStage(config, **kwargs)
