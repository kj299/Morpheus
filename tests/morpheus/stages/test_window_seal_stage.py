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
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.type_utils import get_df_class

SECOND_NS = 10**9

# period=100s, lateness=30s. The rows at 10s and 50s land in window 0 [0,100), the row at 120s in window 1
# [100,200), and the row at 400s in window 4 [400,500). The watermark reaching 400s seals windows 0 and 1
# (400 >= end + 30 for both), so the final row at 20s is late for window 0. Window 4 is still open at end of
# stream and seals on flush.
EVENT_TIMES_S = [10, 50, 120, 400, 20]

STREAM_DATA = {
    "event_time": [t * SECOND_NS for t in EVENT_TIMES_S],
    "entity": ["a", "b", "c", "d", "e"],
}


def make_stage(config: Config, **kwargs) -> WindowSealStage:
    defaults = {"period_seconds": 100, "lateness_seconds": 30}
    defaults.update(kwargs)

    return WindowSealStage(config, **defaults)


def _frames(messages: list) -> list[pd.DataFrame]:
    frames = []

    for message in messages:
        meta = message.payload() if isinstance(message, ControlMessage) else message
        df = meta.copy_dataframe()

        if (hasattr(df, "to_pandas")):
            df = df.to_pandas()

        frames.append(df)

    return frames


def test_execution_modes(config: Config):
    assert issubclass(WindowSealStage, GpuAndCpuMixin)

    stage = make_stage(config)

    assert set(stage.supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    stage = make_stage(config)

    assert stage.get_needed_columns() == {
        "window_id": TypeId.INT64,
        "window_start_ns": TypeId.INT64,
        "window_end_ns": TypeId.INT64,
        "revision": TypeId.INT64,
        "sealed_by": TypeId.STRING,
        "is_late": TypeId.BOOL8,
        "window_complete": TypeId.BOOL8,
    }


@pytest.mark.gpu_and_cpu_mode
def test_window_seal_stage_pipe(config: Config):
    df_class = get_df_class(config.execution_mode)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[df_class(STREAM_DATA)]))
    pipe.add_stage(make_stage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    frames = _frames(sink.get_messages())

    by_kind = {}
    for frame in frames:
        key = (int(frame["window_id"].iloc[0]), bool(frame["is_late"].iloc[0]))
        by_kind[key] = frame

    # Windows 0 and 1 sealed by the watermark; the late row for window 0 is its own message; window 4 sealed by
    # the flush.
    assert set(by_kind) == {(0, False), (1, False), (0, True), (4, False)}

    window_zero = by_kind[(0, False)]
    assert sorted(window_zero["entity"]) == ["a", "b"]
    assert (window_zero["sealed_by"] == "watermark").all()
    assert (window_zero["window_complete"]).all()
    assert (window_zero["revision"] == 0).all()
    assert (window_zero["window_start_ns"] == 0).all()
    assert (window_zero["window_end_ns"] == 100 * SECOND_NS).all()

    assert by_kind[(1, False)]["entity"].tolist() == ["c"]

    late = by_kind[(0, True)]
    assert late["entity"].tolist() == ["e"]
    assert (late["revision"] == 1).all()
    assert not late["window_complete"].any()

    assert by_kind[(4, False)]["entity"].tolist() == ["d"]
    assert (by_kind[(4, False)]["sealed_by"] == "flush").all()


@pytest.mark.gpu_and_cpu_mode
def test_replay_is_deterministic(config: Config):
    df_class = get_df_class(config.execution_mode)

    def run_once() -> list[pd.DataFrame]:
        pipe = LinearPipeline(config)
        pipe.set_source(InMemorySourceStage(config, dataframes=[df_class(STREAM_DATA)]))
        pipe.add_stage(make_stage(config))
        sink = pipe.add_stage(InMemorySinkStage(config))
        pipe.run()

        return _frames(sink.get_messages())

    first = run_once()
    second = run_once()

    assert len(first) == len(second)

    for (left, right) in zip(first, second):
        pd.testing.assert_frame_equal(left, right)


@pytest.mark.gpu_and_cpu_mode
def test_batching_does_not_change_membership(config: Config):
    df_class = get_df_class(config.execution_mode)
    whole = df_class(STREAM_DATA)
    split = [whole.iloc[[i]].reset_index(drop=True) for i in range(len(EVENT_TIMES_S))]

    def run(dataframes) -> dict:
        pipe = LinearPipeline(config)
        pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
        pipe.add_stage(make_stage(config))
        sink = pipe.add_stage(InMemorySinkStage(config))
        pipe.run()

        rows = {}
        for frame in _frames(sink.get_messages()):
            for row in frame.itertuples(index=False):
                rows[row.entity] = (row.window_id, row.is_late, row.window_complete)

        return rows

    assert run([whole]) == run(split)


@pytest.mark.gpu_and_cpu_mode
def test_on_data_emits_sealed_window(config: Config):
    df_class = get_df_class(config.execution_mode)
    stage = make_stage(config)

    assert len(stage.on_data(MessageMeta(df_class({"event_time": [10 * SECOND_NS], "entity": ["a"]})))) == 0

    emissions = stage.on_data(MessageMeta(df_class({"event_time": [400 * SECOND_NS], "entity": ["b"]})))
    frames = _frames(emissions)

    assert len(frames) == 1
    assert frames[0]["entity"].tolist() == ["a"]
    assert frames[0]["window_id"].tolist() == [0]


@pytest.mark.gpu_and_cpu_mode
def test_control_message_in_control_message_out(config: Config):
    df_class = get_df_class(config.execution_mode)
    stage = make_stage(config)

    message = ControlMessage()
    message.payload(MessageMeta(df_class({"event_time": [10 * SECOND_NS, 400 * SECOND_NS], "entity": ["a", "b"]})))

    emissions = stage.on_data(message)

    assert len(emissions) == 1
    assert isinstance(emissions[0], ControlMessage)


@pytest.mark.gpu_and_cpu_mode
def test_late_rows_never_join_a_published_window(config: Config):
    df_class = get_df_class(config.execution_mode)
    stage = make_stage(config)

    stage.on_data(MessageMeta(df_class({"event_time": [10 * SECOND_NS, 400 * SECOND_NS], "entity": ["a", "b"]})))
    late = _frames(stage.on_data(MessageMeta(df_class({"event_time": [20 * SECOND_NS], "entity": ["c"]}))))

    assert len(late) == 1
    assert late[0]["is_late"].all()
    assert not late[0]["window_complete"].any()
    assert late[0]["revision"].tolist() == [1]

    # A second late arrival for the same window gets the next revision, never a repeat.
    again = _frames(stage.on_data(MessageMeta(df_class({"event_time": [30 * SECOND_NS], "entity": ["d"]}))))
    assert again[0]["revision"].tolist() == [2]


@pytest.mark.gpu_and_cpu_mode
def test_order_columns_apply_a_stable_total_order(config: Config):
    df_class = get_df_class(config.execution_mode)
    stage = make_stage(config, order_columns=["event_time", "entity"])

    stage.on_data(
        MessageMeta(
            df_class({
                "event_time": [50 * SECOND_NS, 10 * SECOND_NS, 50 * SECOND_NS], "entity": ["b", "c", "a"]
            })))
    frames = _frames(stage.on_completed())

    assert frames[0]["entity"].tolist() == ["c", "a", "b"]


@pytest.mark.gpu_and_cpu_mode
def test_invalid_times_are_flagged_not_dropped(config: Config):
    df_class = get_df_class(config.execution_mode)
    stage = make_stage(config, time_unit="s", time_column="when")

    frames = _frames(stage.on_data(MessageMeta(df_class({"when": ["not-a-time"], "entity": ["a"]}))))

    assert len(frames) == 1
    assert frames[0]["window_id"].tolist() == [-1]
    assert frames[0]["sealed_by"].tolist() == ["invalid"]
    assert not frames[0]["window_complete"].any()


@pytest.mark.gpu_and_cpu_mode
def test_invalid_times_can_be_fatal(config: Config):
    df_class = get_df_class(config.execution_mode)
    stage = make_stage(config, time_unit="s", raise_on_invalid=True)

    with pytest.raises(ValueError, match="cannot be interpreted"):
        stage.on_data(MessageMeta(df_class({"event_time": ["nope"], "entity": ["a"]})))


@pytest.mark.gpu_and_cpu_mode
def test_flush_can_be_disabled(config: Config):
    df_class = get_df_class(config.execution_mode)
    stage = make_stage(config, seal_on_complete=False)

    stage.on_data(MessageMeta(df_class({"event_time": [10 * SECOND_NS], "entity": ["a"]})))

    assert stage.on_completed() is None


@pytest.mark.gpu_and_cpu_mode
def test_missing_time_column_raises(config: Config):
    df_class = get_df_class(config.execution_mode)

    with pytest.raises(KeyError, match="event_time"):
        make_stage(config).on_data(MessageMeta(df_class({"entity": ["a"]})))


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError):
        make_stage(config, period_seconds=0)

    with pytest.raises(ValueError):
        make_stage(config, lateness_seconds=-1)

    with pytest.raises(ValueError):
        make_stage(config, time_column="")
