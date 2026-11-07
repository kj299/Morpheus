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
One copy of each helper (guide 13, plan item 6; plan section 4.6a).

The audit found three readers of `savedsearches.conf`, thirteen golden drivers that differed in one module name, and,
once it looked, forty function bodies written out more than once across the fork's own Python: the rendering every
composition used, the cross-restart check every harness ran, the list conversion every stage test carried. A copy
is a place a fix lands in one file and not the other. These hold the tree to one definition of each:

- no two top-level functions in the fork's Python files mean the same thing: the same body, reading the same
  names, each of which resolves to the same definition, followed through the module's own constants and functions
  to what it imports;
- only `morpheus.utils.splunk_conf` parses a Splunk `.conf` file;
- `run_golden.py` is the one golden driver.

Two bodies that read alike but bind a module's own constant are not copies: `at(day)` in a corpus counted from
another epoch is another function. The fork's files are the manifest's `fork_added` rows that are kept.
"""

import ast
import builtins
import csv
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
MANIFEST = os.path.join(REPO_ROOT, "ci", "artifacts", "repository_audit_manifest.csv")
CONF_READER = "python/morpheus/morpheus/utils/splunk_conf.py"
SELF = os.path.relpath(os.path.abspath(__file__), REPO_ROOT)
"""This file, which names what it looks for."""
BUILTINS = frozenset(dir(builtins))


def fork_python_files() -> list[str]:
    """Every Python file the fork wrote and keeps, read from the audit manifest."""
    with open(MANIFEST, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    return sorted(row["path"] for row in rows if row["class"] == "fork_added" and row["verdict"] != "remove"
                  and row["path"].endswith(".py") and os.path.exists(os.path.join(REPO_ROOT, row["path"])))


def _without_docstring(body: list) -> list:
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        return body[1:]

    return body


def _bound_names(node: ast.AST) -> set:
    """Every name a function binds for itself: its parameters, its assignments, its loop and comprehension targets."""
    bound = set()

    for child in ast.walk(node):
        if (isinstance(child, ast.arg)):
            bound.add(child.arg)
        elif (isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del))):
            bound.add(child.id)
        elif (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child is not node):
            bound.add(child.name)
        elif (isinstance(child, ast.alias)):
            bound.add((child.asname or child.name).split(".")[0])
        elif (isinstance(child, ast.ExceptHandler) and child.name):
            bound.add(child.name)

    return bound


def _read_names(node: ast.AST) -> set:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}


class Module:
    """One file's top-level definitions, and what each of its names finally means."""

    def __init__(self, source: str):
        self.tree = ast.parse(source)
        self.definitions: dict = {}
        self.imports: dict = {}

        for node in self.tree.body:
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))):
                self.definitions[node.name] = node
            elif (isinstance(node, (ast.Assign, ast.AnnAssign))):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]

                for target in targets:
                    for name in (child.id for child in ast.walk(target) if isinstance(child, ast.Name)):
                        self.definitions[name] = node
            elif (isinstance(node, ast.Import)):
                for alias in node.names:
                    self.imports[(alias.asname or alias.name).split(".")[0]] = alias.name.split(".")[0]
            elif (isinstance(node, ast.ImportFrom)):
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = f"{'.' * node.level}{node.module or ''}.{alias.name}"

        self._resolved: dict = {}

    def meaning(self, name: str) -> str:
        """What a name read by a definition finally is: an import, a builtin, or a definition and what it reads."""
        if (name in self._resolved):
            return self._resolved[name]

        if (name in self.imports):
            return f"import {self.imports[name]}"

        if (name not in self.definitions):
            return f"builtin {name}" if name in BUILTINS else f"unbound {name}"

        # A definition that reads itself, directly or through another, is named rather than followed.
        self._resolved[name] = f"recursive {name}"
        self._resolved[name] = self.fingerprint(self.definitions[name])

        return self._resolved[name]

    def fingerprint(self, node: ast.AST) -> str:
        """A definition's body and the meaning of every name it reads from outside itself."""
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
            shape = ast.dump(ast.Module(body=_without_docstring(node.body), type_ignores=[]))
            arguments = ast.dump(node.args)
            free = _read_names(node) - _bound_names(node)
        elif (isinstance(node, ast.ClassDef)):
            shape = ast.dump(ast.Module(body=_without_docstring(node.body), type_ignores=[]))
            arguments = ast.dump(ast.Tuple(elts=node.bases, ctx=ast.Load()))
            free = _read_names(node) - _bound_names(node)
        else:
            value = node.value if node.value is not None else ast.Constant(None)
            shape = ast.dump(value)
            arguments = ""
            free = _read_names(value)

        meanings = "; ".join(f"{name}={self.meaning(name)}" for name in sorted(free))

        return f"{shape} ({arguments}) [{meanings}]"


TRIVIAL_NODES = 10
"""A one-statement body smaller than this -- a filter, one call, a cast -- says all it does on its face, and two files
agreeing on it are not sharing a helper. `False if pd.isna(value) else bool(value)` is ten, and a rule."""

_NOISE = (ast.expr_context, ast.operator, ast.cmpop, ast.boolop, ast.unaryop)


def _trivial(node: ast.FunctionDef) -> bool:
    body = _without_docstring(node.body)

    return len(body) == 1 and sum(1 for child in ast.walk(body[0]) if not isinstance(child, _NOISE)) < TRIVIAL_NODES


def second_definitions(sources: dict[str, str]) -> list[list[str]]:
    """Groups of `path:function` that mean the same thing, largest first."""
    seen: dict = {}

    for (path, source) in sorted(sources.items()):
        try:
            module = Module(source)
        except SyntaxError:
            continue

        for node in module.tree.body:
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _trivial(node)):
                seen.setdefault(module.fingerprint(node), []).append(f"{path}:{node.name}")

    return sorted((group for group in seen.values() if len(group) > 1), key=lambda group: (-len(group), group))


def _sources() -> dict:
    found = {}

    for path in fork_python_files():
        with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as handle:
            found[path] = handle.read()

    return found


def test_the_fork_s_python_files_are_found():
    files = fork_python_files()

    assert len(files) > 300
    assert CONF_READER in files
    assert "tests/morpheus/determinism/golden.py" in files


def test_no_function_is_defined_twice():
    assert second_definitions(_sources()) == []


def test_a_copied_helper_is_seen():
    helper = "import pandas as pd\n\ndef cell(frame, column):\n    value = frame[column].iloc[0]\n    return value\n"

    assert second_definitions({"a.py": helper, "b.py": helper}) == [["a.py:cell", "b.py:cell"]]


def test_a_copy_reading_another_module_s_constant_is_another_function():
    first = "DAY = 86400\n\ndef at(day):\n    seconds = day * DAY\n    return seconds * 10**9\n"
    second = "DAY = 86400 * 7\n\ndef at(day):\n    seconds = day * DAY\n    return seconds * 10**9\n"

    assert second_definitions({"a.py": first, "b.py": second}) == []
    assert second_definitions({"a.py": first, "b.py": first}) == [["a.py:at", "b.py:at"]]


def test_a_copy_calling_its_own_module_s_function_follows_it():
    first = "def rows():\n    return [1]\n\ndef build():\n    found = rows()\n    return list(found)\n"
    second = "def rows():\n    return [2]\n\ndef build():\n    found = rows()\n    return list(found)\n"

    assert second_definitions({"a.py": first, "b.py": second}) == []


def test_only_splunk_conf_parses_a_conf_file():
    readers = [
        path for (path, source) in _sources().items()
        if path not in (CONF_READER, SELF) and ("configparser" in source or r'r"\\\s*\r?\n\s*"' in source)
    ]

    assert readers == []


@pytest.mark.parametrize("name", ["test_network_harness.py", "test_determinism_harness.py"])
def test_every_harness_runs_the_one_golden_driver(name: str):
    with open(os.path.join(REPO_ROOT, "tests", "morpheus", "determinism", name), encoding="utf-8") as handle:
        source = handle.read()

    assert "golden.cross_restart_outputs(" in source


def test_there_is_one_golden_driver():
    determinism = os.path.join(REPO_ROOT, "tests", "morpheus", "determinism")
    drivers = sorted(name for name in os.listdir(determinism) if name.startswith("run_") and name.endswith(".py"))

    assert drivers == ["run_golden.py"]
