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
Control 13's fourth check through `CompareDataFrameStage`, for every composed pipeline.

The guide names the stage as the way a pipeline is compared with its golden inside a pipeline, and until plan section
3.4 only the lineage pipeline did it (`test_determinism_harness.test_against_golden_via_compare_dataframe_stage`).
The other twelve reached their golden through a byte comparison of the rendering, which is the stronger statement
and stays, but it left the stage the guide offers as copyable unexercised on twelve of thirteen outputs, and that
was G88.

The twelve are not linear pipelines: each runs its classes as segments and joins them after collection, so there is
no single last stage to put a comparison behind. What every one of them has is the canonical frame its golden holds,
so the check here sends that frame, row-keyed by `telemetry_class` and `row_key`, through a pipeline of an
`InMemorySourceStage` and a `CompareDataFrameStage` holding the golden. Both sides are read as text, so a value the
two modes render differently -- `3.0` against `3`, `NaN` against an empty cell -- is a difference here too, and the
stage's float tolerance has nothing to forgive. The lineage pipeline is keyed by `event_uid` and `is_late` and is
compared the same way.

`test_the_stage_check_has_teeth` is the negative control: one cell of one golden is changed, and the stage must
count exactly one row that differs.
"""

import importlib
import os
import sys
from io import StringIO

import pandas as pd
import pytest

from morpheus.config import ExecutionMode
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import test_gpu_parity  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

CORPORA = test_gpu_parity.CORPORA

ROW_KEY = "_row"
"""The index column the comparison joins on: the module's key columns joined with a separator neither contains."""

AS_TEXT = {"dtype": str, "keep_default_na": False}


def _rendered(corpus_name: str, execution_mode: ExecutionMode) -> tuple[pd.DataFrame, list[str]]:
    """Run one composed pipeline in one mode and read its canonical rendering back as text, with its key columns."""
    module = importlib.import_module(f"{corpus_name}_pipeline")
    config = module.build_pipeline_config(execution_mode=execution_mode)

    if (corpus_name == "lineage"):
        result = module.run_pipeline(config, [module.build_corpus()])
        rendered = result.to_csv(index=False, lineterminator="\n")
    else:
        result = module.run_pipeline(config, module.build_corpus())
        rendered = module.render(result)

    return (pd.read_csv(StringIO(rendered), **AS_TEXT), list(module.KEY_COLUMNS))


def _keyed(frame: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    keyed = frame.copy()
    keyed[ROW_KEY] = keyed[key_columns].agg("\x1f".join, axis=1)

    return keyed


def _compare_in_pipeline(execution_mode: ExecutionMode, output: pd.DataFrame, golden: pd.DataFrame) -> dict:
    """Send `output` through a source and a `CompareDataFrameStage` holding `golden`, and return its results."""
    config = test_gpu_parity.lp.build_pipeline_config(execution_mode=execution_mode)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[output]))
    comparison = pipe.add_stage(CompareDataFrameStage(config, compare_df=golden, index_col=ROW_KEY))
    pipe.run()

    return comparison.get_results(clear=False)


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("corpus_name", CORPORA)
def test_the_output_matches_its_golden_through_the_stage(corpus_name: str, execution_mode: ExecutionMode):
    (output, key_columns) = _rendered(corpus_name, execution_mode)
    golden = pd.read_csv(os.path.join(HERE, f"golden_{corpus_name}_expected.csv"), **AS_TEXT)

    output = _keyed(output, key_columns)
    golden = _keyed(golden, key_columns)

    # The stage joins on the index, so a key two rows share would compare one of them twice and the other never.
    assert golden[ROW_KEY].is_unique

    results = _compare_in_pipeline(execution_mode, output, golden)

    assert results["diff_rows"] == 0, results
    assert results["matching_rows"] == len(golden)
    assert results["total_rows"] == len(golden)
    assert len(output) == len(golden)
    assert results["missing_cols"] == []
    assert results["extra_cols"] == []


@pytest.mark.cpu_mode
def test_the_stage_check_has_teeth():
    # The negative control: a golden one cell away from the output must cost exactly one row. A comparison that
    # forgave it -- a column the include list dropped, a key that collapsed rows, a text cell parsed as a number
    # inside the tolerance -- would pass every check above without looking.
    (output, key_columns) = _rendered("endpoint", ExecutionMode.CPU)
    golden = pd.read_csv(os.path.join(HERE, "golden_endpoint_expected.csv"), **AS_TEXT)

    column = next(name for name in golden.columns if name not in key_columns)
    golden.loc[len(golden) // 2, column] = golden.loc[len(golden) // 2, column] + "x"

    results = _compare_in_pipeline(ExecutionMode.CPU, _keyed(output, key_columns), _keyed(golden, key_columns))

    assert results["diff_rows"] == 1, results
