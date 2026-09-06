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
The same values, carried in a different type, must reach the same answer.

Control 13 check 6 says row order must not decide the output. This is the same claim one level down: neither must
the dtype the values arrived in. A VLAN of 10 is the same VLAN whether the collector sent an integer, a float, or
a string, and whether the column happened to be widened by some unrelated row in the batch having none.

That last clause is what makes this more than tidiness, and it is how the defect actually arrives. pandas widens
an integer column to float the moment one row is missing, so which rows share a batch decides how every other row
in it is rendered. An entity named `10` in one batch and `10.0` in the next is two entities: its baseline restarts,
its counts start again, and a threshold it should have crossed is never reached. Nothing raises. The output is
simply a different answer to the same question, and control 13's batch-split sweep only catches it if the corpus
happens to contain the null that triggers the widening.

So the sweep here is deliberate rather than incidental: each key-bearing column is presented in every type that
can carry its values, including the widened-by-a-null shape, and across batch boundaries chosen to put the two
representations in different batches. One canonical output, or the fork does not mean what it says about entities.
"""

import os
import sys
import typing

import pandas as pd
import pytest

from morpheus.utils.determinism import diff_frames
from morpheus.utils.determinism import frame_digest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import telemetry_pipeline as tp  # noqa: E402


def _as(frame: pd.DataFrame, column: str, render) -> pd.DataFrame:
    frame = frame.copy()
    frame[column] = [render(value) for value in frame[column]]

    return frame


def _corpus_with(column: str, render) -> dict:
    """The corpus, with one column re-presented wherever it appears."""
    return {
        name: (_as(frame, column, render) if column in frame.columns else frame)
        for (name, frame) in tp.build_corpus().items()
    }


def _run(corpus: dict, batches: dict = None) -> pd.DataFrame:
    return tp.run_pipeline(tp.build_pipeline_config(), corpus, batches=batches)


def _same(result: pd.DataFrame, baseline: pd.DataFrame) -> typing.Optional[str]:
    """
    Whether two runs agree, and if not, the one line that says how.

    Compared by digest rather than by equality on the rendered text: these frames are megabytes, and letting the
    test framework diff them character by character turns a failure into a wait long enough to look like a hang.
    `diff_frames` names the first disagreement instead.
    """
    if (frame_digest(result) == frame_digest(baseline)):
        return None

    return diff_frames(result, baseline) or "the digests differ but no column does; the row sets differ"


def _split(corpus: dict, parts: int) -> dict:

    def cut(frame: pd.DataFrame) -> list:
        size = max(1, len(frame) // parts)

        return [frame.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(frame), size)]

    return {name: cut(frame) for (name, frame) in corpus.items()}


@pytest.fixture(name="baseline", scope="module")
def baseline_fixture() -> pd.DataFrame:
    yield _run(tp.build_corpus())


# `vlan_id` is the entity `ouis_per_vlan` counts by and an attribute every binding carries; `collector_seq` is
# hashed into `event_uid`, so a difference there renames every row in the corpus at once.
REPRESENTATIONS = {
    "integer": int,
    "float": float,
    "string": lambda value: str(int(value)),
}
"""
The types the same value can legitimately arrive in.

A string of the *float* -- `"10.0"` -- is deliberately not here. That is not a representation of the value a
collector could send; it is what a broken renderer produces downstream, and it is the symptom this file exists to
catch rather than an input the pipeline is obliged to parse back.
"""


@pytest.mark.cpu_mode
@pytest.mark.parametrize("column", ["vlan_id", "collector_seq"])
@pytest.mark.parametrize("representation", sorted(REPRESENTATIONS))
def test_a_key_bearing_column_reaches_the_same_answer_in_any_type(baseline: pd.DataFrame,
                                                                  column: str,
                                                                  representation: str):
    # `float_string` is the shape that actually shows up: a column widened to float by a missing sibling row, then
    # rendered by something that reached for `str` instead of the shared rule, so `10` arrives as `"10.0"`.
    # The column itself is excluded from the comparison, because a stage echoes its input rather than rewriting
    # it: a collector that sends a float gets a float back, and that is not a defect. The claim is about
    # everything *derived* from the value -- the identifiers, the counts, the keys, the resolved attributes --
    # none of which may depend on which type carried it.
    result = _run(_corpus_with(column, REPRESENTATIONS[representation]))
    difference = _same(result.drop(columns=[column]), baseline.drop(columns=[column]))

    assert difference is None, (f"{column} presented as {representation} produced a different answer: "
                                f"{difference}. The same values in a different type are the same values.")


@pytest.mark.cpu_mode
def test_a_column_widened_by_a_null_row_does_not_rename_its_entities(baseline: pd.DataFrame):
    """
    The mechanism, rather than a type chosen by hand: one row with no VLAN widens the column for every other row.

    The null row's own features are legitimately null, so this asserts the answer for every row that is not it.
    """
    corpus = tp.build_corpus()
    snapshots = corpus["tc2_mac"].copy()
    widened = snapshots.copy()
    widened.loc[widened.index[-1], "vlan_id"] = None
    corpus["tc2_mac"] = widened

    assert widened["vlan_id"].dtype.kind == "f", "the null did not widen the column; this test proves nothing"

    # Every row of the baseline except the one whose VLAN was removed must still be present, unchanged.
    removed = snapshots.iloc[-1]["mac_address"]
    kept = _run(corpus)
    kept = kept[kept["mac_address"] != removed].drop(columns=["vlan_id"]).reset_index(drop=True)
    expected = baseline[baseline["mac_address"] != removed].drop(columns=["vlan_id"]).reset_index(drop=True)
    difference = _same(kept, expected)

    assert difference is None, f"a null VLAN on one row changed the answer for rows that still had one: {difference}"


@pytest.mark.cpu_mode
@pytest.mark.parametrize("parts", [3, 7])
def test_a_representation_that_changes_at_a_batch_boundary_is_still_one_entity(baseline: pd.DataFrame, parts: int):
    """
    The two halves together: the same column arrives as an integer in one batch and a float in the next.

    This is control 13's batch-split sweep crossed with the representation sweep, and it is the shape the defect
    takes in a real feed -- not a collector that changes its mind, but one batch that happened to contain a row
    with nothing in the column.
    """
    integral = _split(tp.build_corpus(), parts)
    mixed = {}

    for (name, frames) in integral.items():
        mixed[name] = [
            _as(frame, "vlan_id", float) if (index % 2 and "vlan_id" in frame.columns) else frame
            for (index, frame) in enumerate(frames)
        ]

    result = _run(tp.build_corpus(), batches=mixed)
    difference = _same(result.drop(columns=["vlan_id"]), baseline.drop(columns=["vlan_id"]))

    assert difference is None, (f"a VLAN carried as an integer in one batch and a float in the next became two "
                                f"entities: {difference}")
