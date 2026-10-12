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
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.stages.lineage.collector_sequence_stage import CollectorSequenceStage
from morpheus.utils.type_utils import get_df_class


def frame(sequences: list, collectors: list = None) -> dict:
    return {
        "collector_id": ["poller-a"] * len(sequences) if collectors is None else collectors,
        "collector_seq": sequences,
    }


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [None if value is None or value is pd.NA or value != value else value for value in series.tolist()]


def run(config: Config, payload: dict, stage: CollectorSequenceStage = None) -> MessageMeta:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    (stage or CollectorSequenceStage(config)).on_data(meta)

    return meta


def test_execution_modes(config: Config):
    assert issubclass(CollectorSequenceStage, GpuAndCpuMixin)

    assert set(CollectorSequenceStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    needed = CollectorSequenceStage(config).get_needed_columns()

    assert needed["collector_seq_gap"] == TypeId.INT64
    assert needed["collector_seq_regressed"] == TypeId.BOOL8


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(CollectorSequenceStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.gpu_and_cpu_mode
def test_a_contiguous_stream_has_no_gaps_and_the_first_record_answers_nothing(config: Config):
    meta = run(config, frame([10, 11, 12]))

    # Null on the first record: a stream nobody has seen before was not shown to be contiguous.
    assert _as_list(meta, "collector_seq_gap") == [None, 0, 0]
    assert _as_list(meta, "collector_seq_regressed") == [None, False, False]


@pytest.mark.gpu_and_cpu_mode
def test_a_gap_counts_the_records_the_collector_numbered_and_never_delivered(config: Config):
    meta = run(config, frame([1, 2, 6, 7]))

    assert _as_list(meta, "collector_seq_gap") == [None, 0, 3, 0]


@pytest.mark.gpu_and_cpu_mode
def test_a_repeat_or_an_earlier_number_is_a_regression_and_moves_nothing(config: Config):
    meta = run(config, frame([1, 2, 2, 0, 3]))

    assert _as_list(meta, "collector_seq_regressed") == [None, False, True, True, False]
    # The replayed records leave the position at two, so the next honest record is not a gap.
    assert _as_list(meta, "collector_seq_gap") == [None, 0, 0, 0, 0]


@pytest.mark.gpu_and_cpu_mode
def test_each_collector_keeps_its_own_position(config: Config):
    meta = run(config, frame([1, 1, 2, 2], collectors=["poller-a", "poller-b", "poller-a", "poller-b"]))

    assert _as_list(meta, "collector_seq_gap") == [None, None, 0, 0]
    assert _as_list(meta, "collector_seq_regressed") == [None, None, False, False]


@pytest.mark.gpu_and_cpu_mode
def test_the_position_survives_from_one_batch_to_the_next(config: Config):
    stage = CollectorSequenceStage(config)
    run(config, frame([1, 2]), stage)
    meta = run(config, frame([5]), stage)

    assert _as_list(meta, "collector_seq_gap") == [2]
    assert stage.tracked_collectors == 1


@pytest.mark.gpu_and_cpu_mode
def test_a_record_with_no_collector_or_no_number_is_placed_nowhere(config: Config):
    meta = run(config, frame([1, None, 2], collectors=["poller-a", "poller-a", None]))

    assert _as_list(meta, "collector_seq_gap") == [None, None, None]
    assert _as_list(meta, "collector_seq_regressed") == [None, None, None]


@pytest.mark.cpu_mode
def test_the_columns_it_reads_are_required(config: Config):
    with pytest.raises(KeyError, match="collector_seq"):
        CollectorSequenceStage(config).on_data(MessageMeta(pd.DataFrame({"collector_id": ["a"]})))
