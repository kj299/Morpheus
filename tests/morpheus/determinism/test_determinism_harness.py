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
"""
The six determinism checks from control 13 of the OSI behavioral analytics guide, run against the reference
lineage pipeline over the golden corpus.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.utils.determinism import diff_frames
from morpheus.utils.determinism import frame_digest
from morpheus.utils.determinism import permute_within_contiguous_groups
from morpheus.utils.lineage import window_id_from_timestamp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import lineage_pipeline  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_lineage_expected.csv")
DRIVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_lineage_pipeline.py")


@pytest.fixture(name="corpus", scope="module")
def corpus_fixture() -> pd.DataFrame:
    yield lineage_pipeline.build_corpus()


def _run(config: Config, dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    return lineage_pipeline.run_pipeline(config, dataframes)


def test_corpus_is_fixed(corpus: pd.DataFrame):
    # Check 1: the corpus itself must be reproducible, or every other check is meaningless.
    again = lineage_pipeline.build_corpus()

    pd.testing.assert_frame_equal(corpus, again)
    assert len(corpus) == 47


def test_corpus_covers_the_required_cases(corpus: pd.DataFrame):
    # Check 1 continued: control 13 names the cases the corpus must contain.
    windows = corpus["event_time"].map(
        lambda t: window_id_from_timestamp(int(t), lineage_pipeline.PERIOD_SECONDS * 10**9))

    # An empty window between active ones.
    assert 3 not in set(windows)
    assert {0, 1, 2, 4} <= set(windows)
    # A single-row entity.
    assert (corpus["src_ip"] == "10.0.9.99").sum() == 1
    # Clock-skewed records: event times that regress within the stream.
    assert (corpus["event_time"].diff().dropna() < 0).any()
    # A malformed record.
    assert (corpus["src_ip"] == "not-an-address").sum() == 1


@pytest.mark.cpu_mode
def test_double_run_diff(config: Config, corpus: pd.DataFrame):
    # Check 2: run the pipeline twice in the same process and diff the canonical outputs.
    first = _run(config, [corpus.copy()])
    second = _run(config, [corpus.copy()])

    assert diff_frames(first, second) is None
    assert frame_digest(first) == frame_digest(second)


@pytest.mark.slow
def test_cross_restart_diff(tmp_path):
    # Check 3: run in two fresh interpreters with different hash seeds. Catches PYTHONHASHSEED dependence and
    # state captured from the parent process, which the in-process double run cannot see.
    outputs = []

    for (label, hash_seed) in (("a", "0"), ("b", "4242")):
        out_path = tmp_path / f"restart_{label}.csv"
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hash_seed

        subprocess.run([sys.executable, DRIVER_PATH, str(out_path)], env=env, check=True, timeout=600)
        outputs.append(out_path.read_bytes())

    assert outputs[0] == outputs[1]


@pytest.mark.cpu_mode
def test_against_golden(config: Config, corpus: pd.DataFrame):
    # Check 4a: compare against the checked-in golden output, so drift introduced by a code or dependency change
    # is caught even when the change is internally consistent. A legitimate behavior change regenerates the golden
    # file via run_lineage_pipeline.py and reviews the diff.
    golden = pd.read_csv(GOLDEN_PATH)
    result = _run(config, [corpus.copy()])

    golden = golden.astype({column: result[column].dtype for column in result.columns})

    assert diff_frames(result, golden) is None, diff_frames(result, golden)


@pytest.mark.cpu_mode
def test_against_golden_via_compare_dataframe_stage(config: Config, corpus: pd.DataFrame):
    # Check 4b: the same assertion expressed with the stage the guide names, so the pattern is copyable into any
    # pipeline without a harness.
    from morpheus.pipeline import LinearPipeline
    from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
    from morpheus.stages.lineage.binding_resolver_stage import BindingResolverStage
    from morpheus.stages.lineage.community_id_stage import CommunityIdStage
    from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
    from morpheus.stages.lineage.window_seal_stage import WindowSealStage

    golden = pd.read_csv(GOLDEN_PATH)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[corpus.copy()]))
    pipe.add_stage(
        LineageStampStage(config, id_columns=["collector_id", "schema_version", "origin_hash", "collector_seq"]))
    pipe.add_stage(CommunityIdStage(config))
    pipe.add_stage(
        BindingResolverStage(config, binding_table=lineage_pipeline.build_binding_table(), key_column="src_ip"))
    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=lineage_pipeline.PERIOD_SECONDS,
                        lateness_seconds=lineage_pipeline.LATENESS_SECONDS,
                        order_columns=["event_time", "collector_id", "collector_seq"]))
    # window_seq is derived by the harness after collection, not by a stage, so it is excluded from the raw
    # in-pipeline comparison; the harness-side golden check covers it.
    comp_stage = pipe.add_stage(
        CompareDataFrameStage(config, compare_df=golden, index_col="event_uid", exclude=["window_seq"]))

    pipe.run()

    results = comp_stage.get_results(clear=False)

    assert results["diff_rows"] == 0, results
    assert results["total_rows"] == len(golden)
    assert len(results["missing_cols"]) == 0


@pytest.mark.cpu_mode
def test_batch_split_sweep(config: Config, corpus: pd.DataFrame):
    # Check 5: the same rows delivered as one frame, three frames, and one frame per row must produce identical
    # output. This is determinism control 5, batching must be irrelevant, verified end to end.
    whole = _run(config, [corpus.copy()])

    thirds = [
        corpus.iloc[0:16].reset_index(drop=True),
        corpus.iloc[16:32].reset_index(drop=True),
        corpus.iloc[32:].reset_index(drop=True),
    ]
    by_row = [corpus.iloc[[i]].reset_index(drop=True) for i in range(len(corpus))]

    assert diff_frames(whole, _run(config, thirds)) is None
    assert diff_frames(whole, _run(config, by_row)) is None


@pytest.mark.cpu_mode
def test_permutation_check_has_teeth(config: Config, corpus: pd.DataFrame):
    # The negative control for check 6: reintroduce the exact defect control 8 targets, a removed sort, and
    # assert the harness catches it. Canonicalization erases row order, so this only works because the pipeline
    # emits window_seq, a value derived from row order; without such a value the permutation check passes
    # unconditionally and proves nothing. A harness change that makes this test fail has disarmed check 6.
    windows = [window_id_from_timestamp(int(t), lineage_pipeline.PERIOD_SECONDS * 10**9) for t in corpus["event_time"]]

    unsorted_baseline = lineage_pipeline.run_pipeline(config, [corpus.copy()], order_columns=[])

    detected = False
    for seed in (1, 2, 3):
        shuffled = permute_within_contiguous_groups(corpus, windows, seed=seed)
        detected = detected or (diff_frames(
            unsorted_baseline, lineage_pipeline.run_pipeline(config, [shuffled], order_columns=[])) is not None)

    assert detected, ("Removing the window sort did not change any output under permutation; the permutation "
                      "check has lost its teeth.")


@pytest.mark.cpu_mode
def test_permutation_within_windows(config: Config, corpus: pd.DataFrame):
    # Check 6: shuffling row order within a window must not change the output. This is the direct test for
    # determinism control 8, and the one that catches an accidentally removed sort long after the fact.
    windows = [window_id_from_timestamp(int(t), lineage_pipeline.PERIOD_SECONDS * 10**9) for t in corpus["event_time"]]

    baseline = _run(config, [corpus.copy()])

    for seed in (1, 2, 3):
        shuffled = permute_within_contiguous_groups(corpus, windows, seed=seed)

        assert not shuffled["collector_seq"].equals(corpus["collector_seq"]), "permutation was a no-op"
        assert diff_frames(baseline, _run(config, [shuffled])) is None, f"seed {seed}: {'':s}" + str(
            diff_frames(baseline, _run(config, [shuffled])))
