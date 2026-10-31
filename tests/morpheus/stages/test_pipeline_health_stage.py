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

from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.pipeline_health_stage import HEALTH_COLUMNS
from morpheus.stages.lineage.pipeline_health_stage import PipelineHealthStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.type_utils import get_df_class

SECOND_NS = 10**9
PERIOD_NS = 300 * SECOND_NS


def sealed(windows: list, collectors: list = None, tenants: list = None, **columns) -> dict:
    """Rows as `WindowSealStage` leaves them: one per entry in `windows`, each in that window."""
    count = len(windows)

    return {
        "window_id": windows,
        "window_start_ns": [window * PERIOD_NS for window in windows],
        "window_end_ns": [(window + 1) * PERIOD_NS for window in windows],
        "collector_id": ["poller-a"] * count if collectors is None else collectors,
        "telemetry_class": ["tc1"] * count,
        "tenant_id": ["corp"] * count if tenants is None else tenants,
        **columns,
    }


def health(config: Config, payload: dict, stage: PipelineHealthStage = None, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("min_buckets", 2)
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    result = (stage or PipelineHealthStage(config, **kwargs)).on_data(meta).copy_dataframe()

    return result.to_pandas() if hasattr(result, "to_pandas") else result


def _values(frame: pd.DataFrame, column: str) -> list:
    return [None if value is None or value is pd.NA or value != value else value for value in frame[column]]


def test_execution_modes(config: Config):
    assert issubclass(PipelineHealthStage, GpuAndCpuMixin)

    assert set(PipelineHealthStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(PipelineHealthStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


def test_a_period_that_cannot_mean_anything_is_refused(config: Config):
    with pytest.raises(ValueError, match="period_seconds"):
        PipelineHealthStage(config, period_seconds=0)


@pytest.mark.gpu_and_cpu_mode
def test_one_record_per_window_collector_class_and_tenant(config: Config):
    result = health(config, sealed([0, 0, 0, 1], collectors=["poller-a", "poller-a", "poller-b", "poller-a"]))

    assert list(result.columns) == list(HEALTH_COLUMNS)
    assert list(zip(result["window_id"], result["collector_id"],
                    result["rows"])) == [(0, "poller-a", 2), (0, "poller-b", 1), (1, "poller-a", 1)]
    # The record is about the window, so it is dated at the window's end.
    assert list(result["event_time"]) == [PERIOD_NS, PERIOD_NS, 2 * PERIOD_NS]


@pytest.mark.gpu_and_cpu_mode
def test_the_stages_own_counts_are_gathered(config: Config):
    payload = sealed([0, 0, 0],
                     envelope_valid=[True, False, True],
                     clock_refused=[False, True, False],
                     entity_key=["hq:sw1:Gi1/0/1", None, "hq:sw1:Gi1/0/2"],
                     auth_unpaired=[None, True, False],
                     auth_abandoned=[0, 2, 1],
                     sample_out_of_order=[False, False, True],
                     is_late=[False, False, True],
                     resolution_method=["exact", "unresolved", "unresolved"],
                     macs_per_port_saturated=[False, True, False],
                     collector_seq_gap=[None, 0, 4],
                     collector_seq_regressed=[None, True, False])
    record = health(config, payload).iloc[0]

    assert record["envelope_invalid"] == 1
    assert record["clock_refused"] == 1
    assert record["keyless"] == 1
    assert record["unpaired"] == 1
    assert record["abandoned"] == 3
    assert record["out_of_order"] == 1
    assert record["late"] == 1
    assert record["unresolved"] == 2
    assert record["saturated"] == 1
    assert record["seq_gap"] == 4
    assert record["seq_regressed"] == 1


@pytest.mark.gpu_and_cpu_mode
def test_a_count_nothing_on_the_rows_can_answer_is_null_rather_than_zero(config: Config):
    record = health(config, sealed([0, 0])).iloc[0]

    # Nothing here can be unpaired, which is a different answer from nothing having been unpaired.
    assert pd.isna(record["unpaired"])
    assert pd.isna(record["seq_gap"])
    assert pd.isna(record["max_ingest_lag_seconds"])


@pytest.mark.gpu_and_cpu_mode
def test_a_window_no_model_scored_cannot_say_how_many_fell_back(config: Config):
    # Window 0 is the unscored week: the stamp leaves `model_fallback_used` null on every row. Window 1 was scored,
    # once by the fallback. The answer for window 0 is the same whether or not window 1 is in the batch, which is
    # what makes a run stopped before the scored week agree with one over the whole replay.
    together = health(config, sealed([0, 0, 1, 1], model_fallback_used=[None, None, True, False]))
    alone = health(config, sealed([0, 0], model_fallback_used=[None, None]))

    assert _values(together, "fallback") == [None, 1]
    assert _values(alone, "fallback") == [None]


@pytest.mark.gpu_and_cpu_mode
def test_the_lags_and_the_offsets_are_measured(config: Config):
    payload = sealed([0, 0, 0],
                     event_time=[0, 10 * SECOND_NS, 20 * SECOND_NS],
                     observed_time=[0, 10 * SECOND_NS, 50 * SECOND_NS],
                     ingest_time=[2 * SECOND_NS, 12 * SECOND_NS, 51 * SECOND_NS],
                     clock_offset_ms=[3, -5, 40])
    record = health(config, payload).iloc[0]

    assert record["max_ingest_lag_seconds"] == 2.0
    assert record["max_observe_lag_seconds"] == 30.0
    assert (record["clock_offset_ms_min"], record["clock_offset_ms_median"],
            record["clock_offset_ms_max"]) == (-5, 3, 40)


@pytest.mark.gpu_and_cpu_mode
def test_a_feed_that_doubles_steps_above_its_own_history_and_its_neighbour_does_not(config: Config):
    stage = PipelineHealthStage(config, min_buckets=2)
    windows = []
    tenants = []

    for window in range(4):
        windows += [window] * 3 + [window] * (6 if window == 3 else 3)
        tenants += ["steady"] * 3 + ["doubling"] * (6 if window == 3 else 3)

    result = health(config, sealed(windows, tenants=tenants), stage)
    last = result[result["window_id"] == 3].set_index("tenant_id")

    assert last.loc["doubling", "rows"] == 6
    assert last.loc["doubling", "rows_baseline_max"] == 3
    assert last.loc["doubling", "rows_step"] == 3
    assert last.loc["steady", "rows_step"] == 0
    assert bool(last.loc["doubling", "rows_baseline_mature"])


@pytest.mark.gpu_and_cpu_mode
def test_the_history_carries_from_one_batch_to_the_next(config: Config):
    stage = PipelineHealthStage(config, min_buckets=1)
    health(config, sealed([0, 0]), stage)
    result = health(config, sealed([1, 1, 1]), stage)

    assert _values(result, "rows_baseline_max") == [2]
    assert _values(result, "rows_step") == [1]


@pytest.mark.cpu_mode
def test_unsealed_rows_are_refused(config: Config):
    with pytest.raises(KeyError, match="window_id"):
        PipelineHealthStage(config).on_data(MessageMeta(pd.DataFrame({"collector_id": ["a"]})))


@pytest.mark.gpu_and_cpu_mode
def test_it_composes_behind_the_sealer(config: Config):
    df_class = get_df_class(config.execution_mode)
    raw = df_class({
        "event_time": [0, SECOND_NS, PERIOD_NS + SECOND_NS, 3 * PERIOD_NS],
        "event_uid": ["a", "b", "c", "d"],
        "collector_id": ["poller-a"] * 4,
        "collector_seq": [0, 1, 2, 3],
        "tenant_id": ["corp"] * 4,
    })

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[raw]))
    pipe.add_stage(WindowSealStage(config, period_seconds=300, lateness_seconds=0))
    pipe.add_stage(PipelineHealthStage(config, telemetry_class="tc1", min_buckets=1))
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    frames = [message.copy_dataframe() for message in sink.get_messages()]
    frames = [frame.to_pandas() if hasattr(frame, "to_pandas") else frame for frame in frames]
    result = pd.concat(frames, ignore_index=True)

    assert list(zip(result["window_id"], result["rows"])) == [(0, 2), (1, 1), (3, 1)]
    assert set(result["telemetry_class"]) == {"tc1"}
