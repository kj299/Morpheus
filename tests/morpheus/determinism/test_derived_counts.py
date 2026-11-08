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
Counts are read from what they count (guide 13, plan item 7; plan section 4.6b).

Stage counts, rule counts, golden file lists and number words in prose drifted every time something was added: a
README said fifty-seven stages when sixty-two shipped, a docstring said `props.conf` declared nineteen stanzas when it
declared twenty, and two tests named every composed corpus by hand, one of them checking its list against the disk
rather than reading the disk. Tests now read a count from its source of truth and prose says "every" where it can.
These hold the tests to it:

- no test spells out a whole collection a glob finds -- every corpus, every segment, every composition, every
  sourcetype, index, saved search or input class -- or all of one but one;
- no test keeps a table of number words: a count a document spells out is read with `_utils.number_words.from_words`.
"""

import ast
import glob
import os
import sys

import pytest

from _utils.number_words import from_words
from morpheus.utils import splunk_conf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
APP = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "TA-morpheus-lineage", "default")
SAMPLES = os.path.join(REPO_ROOT, "examples", "behavioral_analytics", "samples")
NUMBERS = "tests/_utils/number_words.py"
SELF = os.path.relpath(os.path.abspath(__file__), REPO_ROOT)
"""This file, whose negative controls are the tables it looks for."""

sys.path.insert(0, HERE)

# pylint: disable=wrong-import-position
import run_golden  # noqa: E402
from test_one_definition import fork_python_files  # noqa: E402


def collections() -> dict[str, set]:
    """Every collection a test might be tempted to list, each read from where it is defined."""
    return {
        "corpora": set(run_golden.CORPORA),
        "segments": {name
                     for name in os.listdir(SAMPLES) if os.path.isdir(os.path.join(SAMPLES, name))},
        "compositions": {
            os.path.basename(path)[:-len(".py")]
            for path in glob.glob(os.path.join(REPO_ROOT, "python", "morpheus", "morpheus", "compositions", "*.py"))
        } - {"__init__"},
        "input classes": {
            os.path.basename(path)[:-len(".jsonlines")]
            for path in glob.glob(os.path.join(SAMPLES, "*", "*.jsonlines"))
        },
        "sourcetypes": set(splunk_conf.stanzas(os.path.join(APP, "props.conf"))),
        "indexes": set(splunk_conf.stanzas(os.path.join(APP, "indexes.conf"))),
        "saved searches": set(splunk_conf.stanzas(os.path.join(APP, "savedsearches.conf"))),
    }


def spelled_out_collections(source: str, known: dict[str, set]) -> list[str]:
    """Each literal in `source` that lists a whole collection, or all of it but one, by line."""
    found = []

    for node in ast.walk(ast.parse(source)):
        if (not isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict))):
            continue

        elements = node.keys if isinstance(node, ast.Dict) else node.elts
        strings = {
            element.value
            for element in elements if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }

        if (len(strings) < 3 or len(strings) != len(elements)):
            continue

        found.extend(f"line {node.lineno}: {name}, {len(strings)} of {len(members)}"
                     for (name, members) in known.items() if strings <= members and len(strings) >= len(members) - 1)

    return found


def _is_number_word(value) -> bool:
    try:
        from_words(value)
    except (ValueError, AttributeError):
        return False

    return True


def number_word_tables(source: str) -> list[str]:
    """Each table of number words in `source`: a mapping of three or more of them, or the words in one string."""
    found = []

    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Dict)):
            words = [key.value for key in node.keys if isinstance(key, ast.Constant) and _is_number_word(key.value)]

            if (len(words) >= 3):
                found.append(f"line {node.lineno}: a table of {len(words)} number words")
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str) and "one two three" in node.value):
            found.append(f"line {node.lineno}: the number words in a string")

    return found


def _test_sources() -> dict[str, str]:
    found = {}

    for path in fork_python_files():
        if (path.startswith("tests/") and path not in (NUMBERS, SELF)):
            with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as handle:
                found[path] = handle.read()

    return found


def test_the_collections_are_found():
    sizes = {name: len(members) for (name, members) in collections().items()}

    assert all(size >= 3 for size in sizes.values()), sizes


def test_no_test_spells_out_a_collection_a_glob_finds():
    known = collections()
    found = {
        path: offences
        for (path, source) in _test_sources().items()
        for offences in [spelled_out_collections(source, known)] if offences
    }

    assert found == {}


@pytest.mark.parametrize("dropped", [0, 1])
def test_the_check_sees_a_collection_spelled_out(dropped: int):
    members = sorted(collections()["corpora"])
    source = f"CORPORA = {tuple(members[dropped:])!r}\n"

    assert spelled_out_collections(
        source, collections()) == [f"line 1: corpora, {len(members) - dropped} of "
                                   f"{len(members)}"]


def test_the_check_passes_a_selection():
    source = 'RESTARTABLE = ("endpoint", "saas", "session")\n'

    assert spelled_out_collections(source, collections()) == []


def test_no_test_keeps_a_table_of_number_words():
    found = {
        path: offences
        for (path, source) in _test_sources().items()
        for offences in [number_word_tables(source)] if offences
    }

    assert found == {}


def test_the_check_sees_a_table_of_number_words():
    source = 'WORDS = {"nineteen": 19, "twenty": 20, "twenty-one": 21}\nUNITS = "zero one two three".split()\n'

    assert len(number_word_tables(source)) == 2


@pytest.mark.parametrize("word,number",
                         [("none", 0), ("eleven", 11), ("Twenty", 20), ("fifty-eight", 58), ("Eighty-two", 82),
                          ("ninety-nine", 99)])
def test_a_spelled_count_is_read_as_its_number(word: str, number: int):
    assert from_words(word) == number


@pytest.mark.parametrize("word", ["twenty-eleven", "hundred", "eighty-", "", "two hundred"])
def test_a_word_that_is_not_a_count_is_refused(word: str):
    with pytest.raises(ValueError):
        from_words(word)
