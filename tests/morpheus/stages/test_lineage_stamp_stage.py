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
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.lineage import event_uid
from morpheus.utils.lineage import link_uid
from morpheus.utils.type_utils import get_df_class

ENVELOPE_DATA = {
    "collector_id": ["col-a", "col-a", "col-b"],
    "schema_version": ["TC-5/2.1.0", "TC-5/2.1.0", "TC-5/2.1.0"],
    "origin_hash": ["abc", "abc", "def"],
    "collector_seq": [1, 2, 1],
}

EXPECTED_UIDS = [
    event_uid("col-a", "TC-5/2.1.0", "abc", 1),
    event_uid("col-a", "TC-5/2.1.0", "abc", 2),
    event_uid("col-b", "TC-5/2.1.0", "def", 1),
]


@pytest.fixture(name="envelope_df")
def envelope_df_fixture(config: Config):
    yield get_df_class(config.execution_mode)(ENVELOPE_DATA)


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return series.tolist()


def test_execution_modes(config: Config):
    assert issubclass(LineageStampStage, GpuAndCpuMixin)

    stage = LineageStampStage(config)

    assert set(stage.supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns_without_edges(config: Config):
    stage = LineageStampStage(config)

    assert stage.get_needed_columns() == {"event_uid": TypeId.STRING}


def test_needed_columns_with_edges(config: Config):
    stage = LineageStampStage(config, parent_uid_column="parent_event_uid")

    assert stage.get_needed_columns() == {
        "event_uid": TypeId.STRING, "link_uid": TypeId.STRING, "join_method": TypeId.STRING
    }


@pytest.mark.gpu_and_cpu_mode
def test_lineage_stamp_stage_pipe(config: Config, envelope_df):
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[envelope_df]))
    pipe.add_stage(LineageStampStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "event_uid") == EXPECTED_UIDS


@pytest.mark.gpu_and_cpu_mode
def test_on_data_accepts_message_meta(config: Config, envelope_df):
    meta = MessageMeta(envelope_df)

    LineageStampStage(config).on_data(meta)

    assert _as_list(meta, "event_uid") == EXPECTED_UIDS


@pytest.mark.gpu_and_cpu_mode
def test_on_data_accepts_control_message(config: Config, envelope_df):
    message = ControlMessage()
    message.payload(MessageMeta(envelope_df))

    LineageStampStage(config).on_data(message)

    assert _as_list(message.payload(), "event_uid") == EXPECTED_UIDS


@pytest.mark.gpu_and_cpu_mode
def test_stamping_is_repeatable(config: Config, envelope_df):
    first = MessageMeta(envelope_df.copy(deep=True))
    second = MessageMeta(envelope_df.copy(deep=True))

    LineageStampStage(config).on_data(first)
    LineageStampStage(config).on_data(second)

    assert _as_list(first, "event_uid") == _as_list(second, "event_uid")


@pytest.mark.gpu_and_cpu_mode
def test_row_order_does_not_change_identity(config: Config, envelope_df):
    reordered = envelope_df.iloc[[2, 0, 1]].reset_index(drop=True)
    meta = MessageMeta(reordered)

    LineageStampStage(config).on_data(meta)

    assert _as_list(meta, "event_uid") == [EXPECTED_UIDS[2], EXPECTED_UIDS[0], EXPECTED_UIDS[1]]


@pytest.mark.gpu_and_cpu_mode
def test_edges_are_emitted(config: Config, envelope_df):
    envelope_df["parent_event_uid"] = ["p1", None, ""]
    meta = MessageMeta(envelope_df)

    stage = LineageStampStage(config,
                              parent_uid_column="parent_event_uid",
                              relation="carried_by",
                              join_method="hard:flow_id")
    stage.on_data(meta)

    links = _as_list(meta, "link_uid")

    assert links[0] == link_uid("p1", EXPECTED_UIDS[0], "carried_by", "hard:flow_id")
    # A record with no parent is a chain root, not an error.
    assert links[1] is None
    assert links[2] is None
    assert _as_list(meta, "join_method") == ["hard:flow_id"] * 3


@pytest.mark.gpu_and_cpu_mode
def test_custom_id_columns_and_digest_length(config: Config):
    df_class = get_df_class(config.execution_mode)
    meta = MessageMeta(df_class({"a": ["x"], "b": ["y"]}))

    LineageStampStage(config, id_columns=["a", "b"], digest_length=16, event_uid_column="uid").on_data(meta)

    assert _as_list(meta, "uid") == [event_uid("x", "y", digest_length=16)]


@pytest.mark.gpu_and_cpu_mode
def test_id_column_order_is_significant(config: Config):
    df_class = get_df_class(config.execution_mode)

    forward = MessageMeta(df_class({"a": ["x"], "b": ["y"]}))
    reverse = MessageMeta(df_class({"a": ["x"], "b": ["y"]}))

    LineageStampStage(config, id_columns=["a", "b"]).on_data(forward)
    LineageStampStage(config, id_columns=["b", "a"]).on_data(reverse)

    assert _as_list(forward, "event_uid") != _as_list(reverse, "event_uid")


@pytest.mark.gpu_and_cpu_mode
def test_missing_id_column_raises(config: Config):
    df_class = get_df_class(config.execution_mode)
    meta = MessageMeta(df_class({"collector_id": ["a"]}))

    with pytest.raises(KeyError, match="schema_version"):
        LineageStampStage(config).on_data(meta)


@pytest.mark.gpu_and_cpu_mode
def test_missing_parent_column_raises(config: Config, envelope_df):
    meta = MessageMeta(envelope_df)

    with pytest.raises(KeyError, match="nope"):
        LineageStampStage(config, parent_uid_column="nope").on_data(meta)


@pytest.mark.parametrize("kwargs", [{"id_columns": []}, {"digest_length": 0}, {"digest_length": 65}])
def test_constructor_validation(config: Config, kwargs: dict):
    with pytest.raises(ValueError):
        LineageStampStage(config, **kwargs)
