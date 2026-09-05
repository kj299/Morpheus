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
Whether the composed pipelines produce the same output in GPU mode as in CPU mode.

Every stage declares support for both execution modes, and 203 `gpu_mode` variants assert that per stage. None of
them composes a pipeline. Both determinism harnesses built their configuration in CPU mode and nothing else, so
control 13's six checks -- the golden, the double run, the cross-restart, the batch-split sweep, and both
permutation checks -- have only ever been asserted on one of the two modes the fork claims to support. A GPU run
of the per-stage variants cannot close that: it says each stage computes correctly on a device frame, not that
thirteen of them in a row reach the same answer the golden records.

The assertion here is the strongest available and needs no new artifact: run the same corpus through the same
pipeline in GPU mode and compare against the same checked-in golden the CPU tests compare against. Byte for byte
for the telemetry pipeline, since it already renders canonically for the cross-restart check.

Where the risk actually sits is the nullable columns. `assign_nullable_int_column` exists because pandas and cuDF
disagree about what a gap is -- pandas widens to float64 and calls it NaN, cuDF keeps int64 and marks it null --
and `bind_end` and `bind_gap_ns` are both nullable integers that are null on most rows. A mode disagreement there
would not crash; it would render `3.0` on one side and `3` on the other, or `NaN` against an empty cell, and only
a byte comparison of the whole frame would notice.

These tests carry the `gpu_mode` marker, so on a machine without a GPU they are deselected rather than passing
vacuously. The comparison itself is exercised in CPU mode by `test_the_parity_check_agrees_with_itself_on_cpu`,
so what is untested on a CPU-only machine is the GPU run, not this file's logic.
"""

import os
import sys

import pandas as pd
import pytest

from morpheus.config import ExecutionMode
from morpheus.utils.determinism import diff_frames

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import lineage_pipeline as lp  # noqa: E402
import telemetry_pipeline as tp  # noqa: E402

TELEMETRY_GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_telemetry_expected.csv")
LINEAGE_GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_lineage_expected.csv")


def _telemetry_matches_golden(execution_mode) -> None:
    """Run the telemetry pipeline in one mode and compare its canonical rendering with the golden, byte for byte."""
    config = tp.build_pipeline_config(execution_mode=execution_mode)
    rendered = tp.render(tp.run_pipeline(config, tp.build_corpus()))

    with open(TELEMETRY_GOLDEN, encoding="utf-8") as handle:
        golden_text = handle.read()

    if (rendered != golden_text):
        from io import StringIO
        as_text = {"dtype": str, "keep_default_na": False}
        difference = diff_frames(pd.read_csv(StringIO(rendered), **as_text),
                                 pd.read_csv(StringIO(golden_text), **as_text))

        pytest.fail(f"{execution_mode} output differs from the golden: {difference}. The golden is a CPU artifact, "
                    f"so a difference here is the two execution modes disagreeing, not drift.")


def _lineage_matches_golden(execution_mode) -> None:
    """The same comparison for the lineage pipeline, which has no canonical rendering of its own."""
    config = lp.build_pipeline_config(execution_mode=execution_mode)
    result = lp.run_pipeline(config, [lp.build_corpus()])
    golden = pd.read_csv(LINEAGE_GOLDEN)
    golden = golden.astype({column: result[column].dtype for column in result.columns})

    assert diff_frames(result, golden) is None, diff_frames(result, golden)


@pytest.mark.cpu_mode
def test_the_parity_check_agrees_with_itself_on_cpu():
    # Not a tautology: it is what keeps the two tests below from being written blind. Everything except the mode is
    # exercised here, so a machine with a GPU is testing the GPU run rather than this file.
    _telemetry_matches_golden(ExecutionMode.CPU)
    _lineage_matches_golden(ExecutionMode.CPU)


@pytest.mark.gpu_mode
def test_the_telemetry_pipeline_reaches_the_same_answer_on_a_gpu():
    # Fourteen stages composed, over the whole corpus, against the golden the CPU harness pins. The nullable
    # integer columns are what this is really asking about: `bind_end` and `bind_gap_ns` are null on most rows, and
    # the two modes have to render a null identically for the comparison to hold.
    _telemetry_matches_golden(ExecutionMode.GPU)


@pytest.mark.gpu_mode
def test_the_lineage_pipeline_reaches_the_same_answer_on_a_gpu():
    # The substrate the telemetry pipeline resolves through: event_uid and link_uid provenance, the Community ID
    # hash, binding resolution, and event-time window sealing.
    _lineage_matches_golden(ExecutionMode.GPU)


@pytest.mark.cpu_mode
def test_a_widened_count_is_restored_to_an_integer():
    # What the GPU parity run actually caught: cuDF cannot put a null inside an integer column, so it hands back
    # float64 with NaN and a count of 3 renders as `3.0` against a golden holding `3`. The repair is asserted here
    # rather than only on a GPU, because the arithmetic is ordinary pandas and needs no device.
    from host_frame import restore_integer_columns

    widened = pd.DataFrame({"arp_count_in_window": [3.0, float("nan"), 5.0]})
    restored = restore_integer_columns(widened.copy(), {"arp_count_in_window": True})

    native = pd.DataFrame({"arp_count_in_window": pd.array([3, None, 5], dtype="Int64")})

    assert restored.to_csv(index=False) == native.to_csv(index=False), \
        "the restored column must render exactly as the CPU path renders it natively"


@pytest.mark.cpu_mode
def test_a_column_that_was_never_an_integer_is_left_alone():
    # The negative control, and the reason the repair is keyed on the device frame's own dtypes rather than on
    # "looks like a whole number". An optical reading of 3.0 dBm is a float and must stay one.
    from host_frame import restore_integer_columns

    readings = pd.DataFrame({"optical_rx_dbm": [3.0, float("nan"), 5.0]})
    left = restore_integer_columns(readings.copy(), {"optical_rx_dbm": False})

    assert left["optical_rx_dbm"].dtype.kind == "f"
    assert left.to_csv(index=False) == readings.to_csv(index=False)
