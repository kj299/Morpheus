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
The sentence that counts what this fork has built, in both of the documents that state it.

"N stages and M supporting modules, covered by K distinct tests" is the first concrete claim a reader meets, and
it is made twice: once in the top-level README and once in the guide's summary of what is verified versus
designed. All three numbers were maintained by hand, in both places, and all three went stale. The stage count
was right when it was written and then sat through two commits that added a stage each. The module count sat
through the one that added `dfencoder_scorer`. The test count sat through fourteen. A number a reader can check
and find wrong is worse than no number, because checking it is the cheapest test there is of whether the rest of
the document is maintained.

So the numbers are computed here from the repository and compared against each sentence. What each one counts:

- **Stages**: files named `*_stage.py` under `python/morpheus/morpheus/stages/` that this fork wrote.
- **Supporting modules**: modules under `python/morpheus/morpheus/utils/` that this fork wrote.
- **Distinct tests**: `def test_` definitions across the files the GPU conformance runner names, which is this
  fork's own test suite. Definitions rather than collected items, deliberately, and the word "distinct" in the
  sentence says so. A test parametrized over the execution modes is one test that runs in two places, and the
  old figure counted it twice -- which is most of the difference between the number this replaces and the
  smaller one that replaces it. Definitions are also the count that can be taken without a collection, and a
  document check that had to run pytest in a subprocess to answer would be the slowest test in the suite.

**Authorship is decided by the shape of the copyright line, not by the year in it.** Every file this fork added
carries a single year; every upstream file carries a range ending in the year upstream last touched it. A rule
written against the literal year would need somebody to remember to come back and widen it, which is the habit
this whole file exists because nobody has.
"""

import os
import re

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DOCUMENTS = {
    "README":
        os.path.join(REPO_ROOT, "README.md"),
    "guide":
        os.path.join(REPO_ROOT,
                     "docs",
                     "source",
                     "developer_guide",
                     "guides",
                     "11_predictive_behavioral_analytics_osi.md"),
}
"""Both places the claim is made. They are checked against the repository rather than against each other, so
two documents agreeing on a wrong number is still a failure."""
RUNNER = os.path.join(REPO_ROOT, "ci", "scripts", "gpu_conformance.sh")
STAGES = os.path.join(REPO_ROOT, "python", "morpheus", "morpheus", "stages")
UTILS = os.path.join(REPO_ROOT, "python", "morpheus", "morpheus", "utils")

CLAIM = re.compile(r"([\w-]+)\s+stages\s+and\s+([\w-]+)\s+supporting\s+modules,"
                   r"\s+covered\s+by\s+([\d,]+)\s+distinct\s+tests")
"""The claim, matched across a line break: the guide wraps its prose and the README does not, and a
pattern that only read one of them would leave the other unchecked while looking like it covered both."""

SINGLE_YEAR = re.compile(r"Copyright \(c\) \d{4},")
"""A copyright naming one year rather than a range, which is what every file this fork added carries."""

TEST_DEFINITION = re.compile(r"^\s*def test_", re.MULTILINE)

UNITS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def from_words(text: str) -> int:
    """A written number back to an integer, for the two counts the README spells out.

    Raises rather than returning a sentinel: a word this cannot read is a sentence that has been rewritten, and
    quietly comparing against zero would turn that into a passing test.
    """
    lowered = text.lower()

    if (lowered in UNITS):
        return UNITS.index(lowered)

    (tens, _, unit) = lowered.partition("-")

    if (tens in TENS):
        return TENS[tens] + (UNITS.index(unit) if unit else 0)

    raise ValueError(f"cannot read {text!r} as a number")


def is_forks(path: str) -> bool:
    """Whether this fork wrote the file, read from its copyright line."""
    with open(path, encoding="utf-8") as handle:
        header = "".join(handle.readline() for _ in range(4))

    return SINGLE_YEAR.search(header) is not None


def fork_stages() -> list:
    found = []

    for (directory, _, names) in os.walk(STAGES):
        for name in names:
            path = os.path.join(directory, name)

            if (name.endswith("_stage.py") and is_forks(path)):
                found.append(os.path.relpath(path, REPO_ROOT))

    return sorted(found)


def fork_modules() -> list:
    return sorted(
        os.path.relpath(os.path.join(UTILS, name), REPO_ROOT) for name in os.listdir(UTILS)
        if name.endswith(".py") and name != "__init__.py" and is_forks(os.path.join(UTILS, name)))


def fork_test_files() -> list:
    """The files the GPU conformance runner names, read from the runner rather than restated here.

    That list is itself checked against the fork's test headers by
    `tests/morpheus/utils/test_gpu_conformance_targets.py`, so a test file that exists and is in neither tier
    fails there rather than silently lowering the count here.
    """
    with open(RUNNER, encoding="utf-8") as handle:
        text = handle.read()

    entries = []

    for name in ("TARGETS", "UNMARKED"):
        match = re.search(rf"^{name}=\((.*?)^\)", text, re.MULTILINE | re.DOTALL)

        assert match is not None, f"{name} is not an array in the runner any more"

        entries.extend(line.strip() for line in match.group(1).splitlines()
                       if line.strip() and not line.strip().startswith("#"))

    return sorted(entries)


def count_tests() -> int:
    total = 0

    for relative in fork_test_files():
        with open(os.path.join(REPO_ROOT, relative), encoding="utf-8") as handle:
            total += len(TEST_DEFINITION.findall(handle.read()))

    return total


def claim_in(document: str) -> tuple:
    """The three numbers one document states, as integers."""
    with open(DOCUMENTS[document], encoding="utf-8") as handle:
        match = CLAIM.search(handle.read())

    assert match is not None, (f"the {document} no longer states what this fork has built in the form this "
                               f"checks. Restore the sentence or update the pattern -- do not delete the check, "
                               f"which is the only thing keeping those numbers true.")

    return (from_words(match.group(1)), from_words(match.group(2)), int(match.group(3).replace(",", "")))


def test_the_inputs_are_where_we_think_they_are():
    # Guards the three assertions below against passing over empty walks, which is how a count check becomes a
    # check that two zeroes are equal.
    assert len(fork_stages()) > 20
    assert len(fork_modules()) > 20
    assert len(fork_test_files()) > 60


def test_the_copyright_rule_separates_this_fork_from_upstream():
    # The whole inventory rests on this. Upstream's `type_utils` and this fork's `entity_key` sit in one package,
    # and the only thing between them is the shape of a line near the top.
    assert is_forks(os.path.join(UTILS, "entity_key.py"))
    assert is_forks(os.path.join(STAGES, "telemetry", "tc1_binding_stage.py"))
    assert not is_forks(os.path.join(UTILS, "type_utils.py"))
    assert not is_forks(os.path.join(STAGES, "input", "file_source_stage.py"))


@pytest.mark.parametrize("document", sorted(DOCUMENTS))
def test_the_stated_stage_count_is_the_one_that_is_there(document: str):
    stages = fork_stages()
    stated = claim_in(document)[0]

    assert stated == len(stages), (f"the {document} says {stated} stages; this fork has written {len(stages)}: "
                                   f"{stages}")


@pytest.mark.parametrize("document", sorted(DOCUMENTS))
def test_the_stated_module_count_is_the_one_that_is_there(document: str):
    modules = fork_modules()
    stated = claim_in(document)[1]

    assert stated == len(modules), (f"the {document} says {stated} supporting modules; this fork has written "
                                    f"{len(modules)}: {modules}")


@pytest.mark.parametrize("document", sorted(DOCUMENTS))
def test_the_stated_test_count_is_the_one_that_is_there(document: str):
    written = count_tests()
    stated = claim_in(document)[2]

    assert stated == written, (f"the {document} says {stated} distinct tests; the files the conformance runner "
                               f"names define {written}. Both documents carried the same stale figure for "
                               f"fourteen commits before this check existed.")
