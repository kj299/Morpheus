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
Nothing outside `tests/` imports from `tests/` (guide 13, plan item 5; plan section 4.5).

The compositions, the shard split stage, the resumable pipeline and the host-frame helper were imported by the
segment runners and the example tools from `tests/morpheus/determinism/`, by putting that directory on the path. A
deployment that runs a test module is one whose behaviour changes when a test is edited. The compositions now live in
`morpheus.compositions` and the corpora in `examples/behavioral_analytics/corpora/`; this holds the tree to that, by
reading every Python file outside `tests/` for the two ways back in: a path insertion that names the test tree, and an
import of a module that exists only there.
"""

import ast
import os
import re

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
TESTS = os.path.join(REPO_ROOT, "tests")
SEARCHED = ("python", "examples", "ci", "scripts", "docs")

TEST_TREE_PATH = re.compile(r"""["']tests["']\s*(?:,|/)|["']tests/|/\s*["']tests["']""")
"""A path built to reach the test tree: `"tests", "morpheus"` in a join, or `/ "tests" /` on a `Path`."""


def _test_only_modules() -> set:
    """Module names defined under `tests/` and nowhere else a Python import could find them."""
    names = set()

    for (directory, _, files) in os.walk(TESTS):
        for name in files:
            if (name.endswith(".py") and name != "__init__.py" and not name.startswith("test_")):
                names.add(name[:-3])

    elsewhere = set()

    for top in SEARCHED:
        for (directory, _, files) in os.walk(os.path.join(REPO_ROOT, top)):
            elsewhere.update(name[:-3] for name in files if name.endswith(".py"))

    return names - elsewhere - {"conftest"}


def _sources() -> list:
    found = []

    for top in SEARCHED:
        for (directory, _, files) in os.walk(os.path.join(REPO_ROOT, top)):
            found.extend(os.path.join(directory, name) for name in files if name.endswith(".py"))

    return sorted(found)


def _offences(path: str, test_only: set) -> list:
    with open(path, encoding="utf-8") as handle:
        source = handle.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    offences = []

    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "insert"
                and TEST_TREE_PATH.search(ast.get_source_segment(source, node) or "")):
            offences.append(f"line {node.lineno}: puts the test tree on the path")

        if (isinstance(node, ast.Import)):
            offences.extend(f"line {node.lineno}: imports {alias.name}" for alias in node.names
                            if alias.name.split(".")[0] in test_only)

        if (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                and node.module.split(".")[0] in test_only):
            offences.append(f"line {node.lineno}: imports from {node.module}")

    return offences


def test_nothing_outside_tests_imports_from_tests():
    test_only = _test_only_modules()

    # The shims the tests keep for the old names are test-only modules; the check must know them to mean anything.
    assert {"transport_pipeline", "kafka_replay"} <= test_only

    found = {
        os.path.relpath(path, REPO_ROOT): offences
        for path in _sources()
        for offences in [_offences(path, test_only)] if offences
    }

    assert found == {}


@pytest.mark.parametrize("line",
                         [
                             'sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "morpheus", "determinism"))',
                             'sys.path.insert(0, str(REPO_ROOT / "tests" / "morpheus"))',
                         ])
def test_the_check_sees_a_path_back_into_the_tests(tmp_path, line: str):
    path = tmp_path / "tool.py"
    path.write_text(f"import os\nimport sys\nREPO_ROOT = ''\n{line}\nimport transport_pipeline\n", encoding="utf-8")

    offences = _offences(str(path), _test_only_modules())

    assert any("test tree" in offence for offence in offences)
    assert any("transport_pipeline" in offence for offence in offences)


def test_the_check_passes_the_package_and_the_corpora(tmp_path):
    path = tmp_path / "tool.py"
    path.write_text("from morpheus.compositions import transport\nfrom corpora import merged\n", encoding="utf-8")

    assert _offences(str(path), _test_only_modules()) == []
