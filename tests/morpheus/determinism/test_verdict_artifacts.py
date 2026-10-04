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
The verdicts this repository cannot render itself, held to the artifacts they came from.

Two claims in the README and the guide rest on runs no CI here can make: the GPU conformance verdict, which needs a
card, and the layer 5 model run, which needs a card and Torch. Until 2026-10-04 both artifacts were ignored by git,
so the dates and counts the documents quoted were a memory of a run rather than a reference to one, and the
September figures stayed in both documents for two weeks after the tree they described had been outgrown.

The artifacts are now committed, dated, under `ci/artifacts/` and `examples/layer5_model/artifacts/`, exactly as the
runners wrote them. What this file asserts is that every number a document quotes about the standing runs is
computed from those files rather than restated: the phrases are built from the artifact's fields and searched for
in the documents, so a new run that changes a count fails here until the prose says so, and prose that drifts from
the artifact fails here without a run at all.

The documents are compared with their line wrapping and bold markers removed, because a phrase that wraps across a
line is the same claim, and a test that broke on reflowing a paragraph would be loosened the first time it did.
"""

import glob
import json
import os
import re

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

CONFORMANCE_DIR = os.path.join(REPO_ROOT, "ci", "artifacts")
MODEL_DIR = os.path.join(REPO_ROOT, "examples", "layer5_model", "artifacts")

README = os.path.join(REPO_ROOT, "README.md")
GUIDE = os.path.join(REPO_ROOT,
                     "docs",
                     "source",
                     "developer_guide",
                     "guides",
                     "11_predictive_behavioral_analytics_osi.md")
MODEL_README = os.path.join(REPO_ROOT, "examples", "layer5_model", "README.md")

WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _prose(path: str) -> str:
    """A document with bold markers removed and every run of whitespace collapsed to one space."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    return re.sub(r"\s+", " ", text.replace("**", ""))


def _conformance_runs() -> list[tuple[str, dict]]:
    """Every committed conformance artifact, oldest first; the file name carries the minute the run finished."""
    paths = sorted(glob.glob(os.path.join(CONFORMANCE_DIR, "gpu_conformance-*.json")))

    return [(os.path.basename(path), _load(path)) for path in paths]


def _model_runs() -> list[tuple[str, dict]]:
    paths = sorted(glob.glob(os.path.join(MODEL_DIR, "*", "layer5_model.json")))

    return [(os.path.relpath(path, MODEL_DIR), _load(path)) for path in paths]


def _when(artifact: dict) -> tuple[str, str]:
    """The date and the minute, in the form the documents write them: `2026-10-04` and `03:56`."""
    stamp = artifact["at"]

    return (stamp[:10], stamp[11:16])


def _thousands(value: int) -> str:
    return f"{value:,}"


@pytest.fixture(name="standing", scope="module")
def standing_fixture() -> dict:
    runs = _conformance_runs()
    assert len(runs) > 0, f"no conformance artifact under {CONFORMANCE_DIR}"

    return runs[-1][1]


@pytest.fixture(name="model", scope="module")
def model_fixture() -> dict:
    runs = _model_runs()
    assert len(runs) > 0, f"no model artifact under {MODEL_DIR}"

    return runs[-1][1]


def test_the_artifacts_are_where_we_think_they_are():
    for path in (CONFORMANCE_DIR, MODEL_DIR, README, GUIDE, MODEL_README):
        assert os.path.exists(path), path


def test_each_artifact_is_named_for_the_run_inside_it():
    # The name is what sorts them and what a reader goes by, so a file renamed or copied without its contents
    # changing would quietly make an older run the standing one.
    for (name, artifact) in _conformance_runs():
        (date, minute) = _when(artifact)
        assert name == f"gpu_conformance-{date}T{minute.replace(':', '')}Z.json", name

    for (name, artifact) in _model_runs():
        assert name.split(os.sep)[0] == _when(artifact)[0], name


def test_the_standing_conformance_run_passed_and_reconciles(standing: dict):
    # A newer failing run committed beside a passing one is a regression the documents would otherwise go on
    # describing as a pass.
    assert standing["verdict"] == "passed"

    for tier in ("gpu_mode", "unmarked_gpu_coverage"):
        summary = standing[tier]
        assert summary["outcome"] == "exited cleanly", tier
        assert summary["failures"] == [], tier
        assert summary["died_in"] is None, tier
        assert summary["unaccounted"] == 0, tier
        assert sum(summary["counts"].values()) == summary["collected"], tier


def test_the_readme_quotes_the_standing_conformance_run(standing: dict):
    (date, minute) = _when(standing)
    marked = standing["gpu_mode"]
    unmarked = standing["unmarked_gpu_coverage"]
    prose = _prose(README)

    for phrase in (f"On {date} every `gpu_mode` variant this fork had passed on a GPU",
                   f"At {minute} UTC",
                   f"driver {standing['device'].split(', ')[1]}",
                   f"{_thousands(marked['collected'])} collected, {_thousands(marked['counts']['passed'])} passed",
                   f"{_thousands(unmarked['collected'])} collected, {_thousands(unmarked['counts']['passed'])} passed, "
                   f"{unmarked['counts'].get('skipped', 0)} skipped"):
        assert phrase in prose, f"README.md does not say {phrase!r}, which is what the standing artifact records"


def test_the_guide_quotes_the_standing_conformance_run(standing: dict):
    (date, minute) = _when(standing)
    marked = standing["gpu_mode"]
    unmarked = standing["unmarked_gpu_coverage"]
    prose = _prose(GUIDE)

    for phrase in (f"The run that stands is {date} at {minute} UTC",
                   f"driver {standing['device'].split(', ')[1]}",
                   f"the marked tier {_thousands(marked['collected'])} collected and "
                   f"{_thousands(marked['counts']['passed'])} passed",
                   f"{_thousands(unmarked['collected'])} collected, {_thousands(unmarked['counts']['passed'])} "
                   f"passed and {unmarked['counts'].get('skipped', 0)} skipped"):
        assert phrase in prose, f"the guide does not say {phrase!r}, which is what the standing artifact records"


def test_the_failed_runs_the_documents_describe_are_the_committed_ones():
    # The README and the guide each recount what the runs before the standing one found. Those counts are as
    # checkable as the verdict's, and they are the ones a reader is likeliest to doubt.
    runs = [artifact for (_, artifact) in _conformance_runs() if artifact["verdict"] == "failed"]
    assert len(runs) == 2, "the documents describe two failed runs before the standing one"

    (first, second) = runs
    first_marked = first["gpu_mode"]
    (first_date, first_minute) = _when(first)
    readme = _prose(README)
    guide = _prose(GUIDE)

    assert (f"first, on {first_date} at {first_minute} UTC" in readme)
    assert (f"of {first_marked['collected']} tests {first_marked['counts']['failed']} failed and "
            f"{first_marked['counts']['error']} raised errors" in readme)
    assert (f"The first, on {first_date} at {first_minute} UTC, had {first_marked['counts']['failed']} of "
            f"{first_marked['collected']} tests fail and {first_marked['counts']['error']} raise errors" in guide)

    second_failed = second["gpu_mode"]["counts"]["failed"]
    assert second_failed == len(second["gpu_mode"]["failures"])
    assert all(name.endswith("::test_against_golden[gpu_mode]") for name in second["gpu_mode"]["failures"])

    for prose in (readme, guide):
        assert f"failed {WORDS[second_failed]} golden checks" in prose
        assert f"The second, on {_when(second)[0]}" in prose or f"The second run, on {_when(second)[0]}" in prose


def test_the_model_run_passed_every_check(model: dict):
    assert model["verdict"] == "passed"
    assert model["double_run_reproducible"] is True
    assert model["batch_invariant"] is True
    assert model["pipeline_double_run_reproducible"] is True
    assert model["pipeline_batch_invariant"] is True
    assert model["pipeline_differences"] == {}
    assert model["pipeline_principals_skipped"] == {}
    assert model["pipeline_principals_pinned"] == len(model["model_versions"]) == len(model["principals"])


def test_the_documents_quote_the_model_run(model: dict):
    (date, minute) = _when(model)
    rows = list(model["principals"].values())
    row_list = ", ".join(str(count) for count in rows[:-1]) + f" and {rows[-1]}"
    sizes = model["batch_sizes"]
    size_list = ", ".join(str(size) for size in sizes[:-1]) + f" and {sizes[-1]}"

    common = (f"{date} at {minute} UTC",
              f"torch=={model['torch']}",
              f"batch sizes {size_list}",
              f"same {model['pipeline_scored_rows']} scores",
              f"{model['pipeline_mean_abs_z_max']}")

    for path in (README, GUIDE, MODEL_README):
        prose = _prose(path)

        for phrase in common:
            assert phrase in prose, f"{os.path.basename(path)} does not say {phrase!r}"

    for path in (README, MODEL_README):
        assert f"{row_list}" in _prose(path), f"{os.path.basename(path)} does not give the rows as {row_list!r}"

    model_readme = _prose(MODEL_README)
    assert f"seed {model['seed']} and {model['epochs']} epochs" in model_readme

    for (principal, version) in model["model_versions"].items():
        assert f"| `{principal}` | `{version}` |" in model_readme, \
            f"the model README's digest table does not carry {version}"
