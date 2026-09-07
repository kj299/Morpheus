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
Whether the GPU conformance runner still selects every test file this fork has added.

The runner exists to stop a GPU run reporting success over a suite it never ran, and it names its own tiers as a
list. A list is exactly the thing that goes stale: five TC-5 stage files sat outside both tiers through two
merges, so the layer 5 work went entirely unmeasured while the runner reported a clean verdict on everything
else. The wider upstream tier would have caught them, and it skips itself whenever the checkout's Git LFS
fixtures are unfetched -- which is the state of every checkout this has ever been run on.

The fork's own test files are identified by their copyright header rather than by a second hand-maintained list,
which would have the same failure mode as the first. Every file this fork added carries the same line and no
upstream file does; at the time of writing that identifies fifty-six files, exactly matching what the repository
history says the fork added.

Which tier a file belongs in is a real distinction, not bookkeeping. A file with `gpu_mode` variants belongs in
`TARGETS`, where the marker selects it. A file with none belongs in `UNMARKED`, where it runs in the default
execution mode -- which on a machine with a GPU is the GPU. Putting an unmarked file in `TARGETS` is the quiet
failure: it is listed, it looks covered, and `-m gpu_mode` selects nothing from it.

A directory entry has that same failure and hides it better, which is why there are none left. The runner used
to name `tests/morpheus/determinism` whole; it reads as complete, and `-m gpu_mode` took 29 of its 349 tests,
leaving the liveness registry and five other files deselected on every GPU run. This test exempted the directory
from the marker check because a directory has no markers to check -- so the one entry that most needed checking
was the one entry excused from it. Files are named individually now and nothing is exempt.
"""

import os
import re

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
RUNNER = os.path.join(REPO_ROOT, "ci", "scripts", "gpu_conformance.sh")

FORK_HEADER = "Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved."
"""The line every test file this fork added carries, and no upstream file does."""

SEARCHED = (os.path.join("tests", "morpheus", "determinism"),
            os.path.join("tests", "morpheus", "stages"),
            os.path.join("tests", "morpheus", "utils"))
"""Where a fork test file can live. All three are searched: the runner names files, not directories."""


def _array(name: str) -> list:
    """The entries of one bash array in the runner, read from the script rather than restated here."""
    with open(RUNNER, encoding="utf-8") as handle:
        text = handle.read()

    match = re.search(rf"^{name}=\((.*?)^\)", text, re.MULTILINE | re.DOTALL)

    assert match is not None, f"{name} is not an array in the runner any more"

    return [line.strip() for line in match.group(1).splitlines() if line.strip() and not line.strip().startswith("#")]


def _fork_test_files() -> set:
    found = set()

    for relative in SEARCHED:
        directory = os.path.join(REPO_ROOT, relative)

        for name in sorted(os.listdir(directory)):
            if (not name.startswith("test_") or not name.endswith(".py")):
                continue

            path = os.path.join(directory, name)

            with open(path, encoding="utf-8") as handle:
                header = "".join(handle.readline() for _ in range(4))

            if (FORK_HEADER in header):
                found.add(os.path.join(relative, name).replace(os.sep, "/"))

    return found


def _has_gpu_mode_variants(relative_path: str) -> bool:
    """Whether a file carries any execution-mode marker, read from its source.

    Read rather than collected: collecting requires a Morpheus runtime and, for some of these files, a device.
    The markers are what the runner's own `-m gpu_mode` filter matches on, so the source is the authority.
    """
    with open(os.path.join(REPO_ROOT, relative_path), encoding="utf-8") as handle:
        text = handle.read()

    # Anchored to an actual decorator line rather than to the marker's name anywhere in the file. This very file
    # names both markers in its own assertions, and a substring search called it marked -- which is the same class
    # of mistake as the list it exists to check: something that looks covered and is not.
    return re.search(r"^\s*@pytest\.mark\.(gpu_mode|gpu_and_cpu_mode)\b", text, re.MULTILINE) is not None


@pytest.fixture(name="tiers", scope="module")
def tiers_fixture() -> dict:
    yield {"TARGETS": _array("TARGETS"), "UNMARKED": _array("UNMARKED")}


def test_the_runner_is_where_we_think_it_is(tiers: dict):
    # Without this the assertions below pass over an empty parse, which is the failure mode a guard must not have.
    assert os.path.exists(RUNNER)
    assert len(tiers["TARGETS"]) > 15
    assert len(tiers["UNMARKED"]) > 5

    for tier in ("TARGETS", "UNMARKED"):
        for entry in tiers[tier]:
            assert entry.endswith(".py"), (f"{tier} names the directory {entry}. A directory reads as complete "
                                           f"and is not checkable against the markers inside it, which is how "
                                           f"six determinism files were deselected on every GPU run.")


def test_this_fork_is_identifiable_by_its_header():
    files = _fork_test_files()

    # A floor rather than an exact count, so adding a test file is not a failure here -- being outside both tiers
    # is what fails, below. Too low a floor would let the identification silently stop working.
    assert len(files) >= 56, f"only {len(files)} fork test files identified; the header rule has stopped working"
    assert "tests/morpheus/determinism/test_stage_parameter_liveness.py" in files
    assert "tests/morpheus/stages/test_tc5_travel_stage.py" in files
    assert "tests/morpheus/utils/test_geo_velocity.py" in files


def test_every_fork_test_file_is_in_a_tier(tiers: dict):
    listed = set(tiers["TARGETS"]) | set(tiers["UNMARKED"])
    missing = sorted(_fork_test_files() - listed)

    assert missing == [], (f"the GPU conformance runner selects neither tier for {missing}. A file in no tier is "
                          f"not measured on a GPU at all, and the runner reports a clean verdict without it -- "
                          f"which is the failure this whole script exists to prevent. Add each to TARGETS if it "
                          f"carries execution-mode markers, or to UNMARKED if it does not.")


def test_nothing_listed_has_gone_away(tiers: dict):
    for tier in ("TARGETS", "UNMARKED"):
        for entry in tiers[tier]:
            assert os.path.exists(os.path.join(REPO_ROOT, entry)), f"{tier} names {entry}, which does not exist"


def test_each_file_is_in_the_tier_its_markers_put_it_in(tiers: dict):
    # The quiet failure this separates out: an unmarked file listed in TARGETS looks covered and contributes
    # nothing, because `-m gpu_mode` selects nothing from it.
    for entry in tiers["TARGETS"]:
        assert _has_gpu_mode_variants(entry), (f"TARGETS names {entry}, which carries no execution-mode marker, so "
                                               f"`-m gpu_mode` selects nothing from it. It belongs in UNMARKED.")

    for entry in tiers["UNMARKED"]:
        assert not _has_gpu_mode_variants(entry), (f"UNMARKED names {entry}, which carries execution-mode markers. "
                                                   f"It belongs in TARGETS, where the marker selects it.")


def test_the_selection_floor_is_computed_rather_than_pinned():
    # A constant floor goes stale the moment the tier grows, which is the same failure as a stale tier list one
    # level up: the tier that had lost five files still cleared a floor written when it held fourteen. What is
    # asserted is that the runner derives the floor from the tier, not the number it derives.
    with open(RUNNER, encoding="utf-8") as handle:
        text = handle.read()

    assert re.search(r"^MINIMUM_SELECTED=\$\(\(.*TARGETS\[@\].*\)\)", text, re.MULTILINE) is not None, \
        "MINIMUM_SELECTED is a constant again; it has to be computed from TARGETS or it goes stale with the tier"
