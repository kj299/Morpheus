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
The repository audit's manifest is kept honest by the tree it describes.

`ci/artifacts/repository_audit_manifest.csv` gives every tracked file a verdict: keep, update, keep-dependency or
remove. A manifest that names files the tree no longer has, or misses files the tree gained, is the drift this test
exists to catch; so is a file marked for removal that a file marked to keep still imports, because carrying out the
removal would then break what stays. A file marked `update` may import one marked for removal only when its row says
what changes. The audit guide quotes the manifest's counts, and they are checked here too.
"""

import ast
import collections
import csv
import os
import re
import subprocess

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MANIFEST = os.path.join(REPO_ROOT, "ci", "artifacts", "repository_audit_manifest.csv")
REMOVALS = os.path.join(REPO_ROOT, "ci", "artifacts", "repository_audit_removals.txt")
GUIDE = os.path.join(REPO_ROOT, "docs", "source", "developer_guide", "guides", "13_repository_audit.md")
VERDICTS = ("keep", "update", "keep-dependency", "remove")
PACKAGE_ROOTS = ("python/morpheus/", "python/morpheus_dfp/")
CHECKED_PACKAGES = ("morpheus", "morpheus_dfp", "morpheus_llm")
"""Top-level packages whose imports must resolve to a file in the tree. `morpheus_llm` is listed although the tree no
longer has it, so that an import of it reads as what it is: an import of something removed."""


def _tracked() -> list:
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
    return sorted(line for line in out.splitlines() if line)


@pytest.fixture(name="manifest", scope="module")
def manifest_fixture() -> dict:
    with open(MANIFEST, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    return {row["path"]: row for row in rows}


def test_every_tracked_file_has_exactly_one_verdict(manifest: dict):
    tracked = _tracked()
    missing = sorted(set(tracked) - set(manifest))
    stale = sorted(set(manifest) - set(tracked))

    assert missing == [], f"tracked files without a verdict: {missing[:20]}"
    assert stale == [], f"verdicts for files the tree no longer has: {stale[:20]}"
    assert all(row["verdict"] in VERDICTS for row in manifest.values())
    assert all(row["purpose"].strip() and row["why_here"].strip() and row["reason"].strip()
               for row in manifest.values())


def test_the_removal_list_is_the_manifest_s_remove_rows(manifest: dict):
    with open(REMOVALS, encoding="utf-8") as handle:
        listed = sorted(line.strip() for line in handle if line.strip() and not line.startswith("#"))

    marked = sorted(path for (path, row) in manifest.items() if row["verdict"] == "remove")

    assert listed == marked


def _module_index(paths: list) -> dict:
    index = {}

    for path in paths:
        # A compiled module is in the tree as its type stub, which is enough to say it exists.
        if not path.endswith((".py", ".pyi")):
            continue

        for root in PACKAGE_ROOTS:
            if path.startswith(root):
                module = path[len(root):path.rindex(".")].replace("/", ".")
                index.setdefault(module[:-9] if module.endswith(".__init__") else module, path)

    return index


def _parse(path: str) -> ast.AST:
    try:
        with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as handle:
            return ast.parse(handle.read())
    except (SyntaxError, UnicodeDecodeError):
        return ast.Module(body=[], type_ignores=[])


def _imports(path: str) -> set:
    tree = _parse(path)

    names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)

    return names


def _imports_of_removed(manifest: dict) -> dict:
    index = _module_index(list(manifest))
    removed = {path for (path, row) in manifest.items() if row["verdict"] == "remove"}
    found = collections.defaultdict(set)

    for (path, row) in manifest.items():
        if row["verdict"] == "remove" or not path.endswith(".py"):
            continue

        for name in _imports(path):
            parts = name.split(".")

            for length in range(len(parts), 0, -1):
                target = index.get(".".join(parts[:length]))

                if target is not None:
                    if target in removed:
                        found[path].add(target)
                    break

    return found


def test_nothing_kept_imports_what_is_marked_for_removal(manifest: dict):
    # The removal is marked, not done; this is the check that doing it would leave the kept tree importable.
    offenders = {
        path: targets
        for (path, targets) in _imports_of_removed(manifest).items()
        if manifest[path]["verdict"] in ("keep", "keep-dependency")
    }

    assert not offenders, f"kept files importing files marked for removal: {offenders}"


def test_a_file_the_removal_breaks_says_how_it_changes(manifest: dict):
    # An `update` file may still import what is marked for removal, because its edit lands with the removal; the
    # edit has to be written down, or the removal change has nothing to follow.
    silent = sorted(path for path in _imports_of_removed(manifest) if not manifest[path]["action"].strip())

    assert silent == [], f"files importing removals with no recorded change: {silent}"


def _unresolved(tree: ast.AST, index: dict) -> set:
    """
    The modules a parsed file imports, from the checked packages, that no file in the tree provides.

    `import a.b` and `from a.b import c` both need `a.b` to exist. `c` may be a submodule or a name defined in `a.b`,
    and only the first can be told apart from the tree, so a `from` import is checked down to its module and no
    further.
    """
    wanted = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            wanted.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            wanted.add(node.module)

    return {name for name in wanted if name.split(".")[0] in CHECKED_PACKAGES and name not in index}


def test_nothing_kept_imports_a_module_the_tree_no_longer_has(manifest: dict):
    # The check above sees only removals still to be made, because a file already deleted has left both the tree and
    # the manifest. This one resolves imports against the tree itself, so a removal already carried out cannot leave
    # a kept file importing what it took away.
    index = _module_index(_tracked())
    offenders = {}

    for (path, row) in manifest.items():
        if row["verdict"] == "remove" or not path.endswith(".py"):
            continue

        missing = _unresolved(_parse(path), index)

        if missing:
            offenders[path] = sorted(missing)

    assert not offenders, f"kept files importing modules the tree does not have: {offenders}"


def test_an_import_of_a_deleted_module_is_seen():
    # The resolver itself, against a tree that has one module, so the check above cannot pass by seeing nothing.
    index = {"morpheus.utils.entity_key": "python/morpheus/morpheus/utils/entity_key.py"}
    source = ast.parse("import morpheus.modules.input.multi_file_source\n"
                       "from morpheus_llm.llm import LLMEngine\n"
                       "from morpheus.utils.entity_key import is_missing\n"
                       "import numpy\n")

    assert _unresolved(source, index) == {"morpheus.modules.input.multi_file_source", "morpheus_llm.llm"}


def test_the_guide_quotes_the_manifest_s_counts(manifest: dict):
    counts = collections.Counter(row["verdict"] for row in manifest.values())

    with open(GUIDE, encoding="utf-8") as handle:
        guide = handle.read()

    table = re.search(r"\| Verdict \| Files \|.*?\n\n", guide, re.S)

    assert table is not None, "the audit guide has no verdict table"

    stated = {
        verdict: int(count)
        for (verdict, count) in re.findall(r"^\| `([a-z-]+)` \| (\d+) \|", table.group(0), re.M)
    }

    assert stated == dict(counts), f"the guide says {stated}; the manifest holds {dict(counts)}"
    assert f"{len(manifest):,} tracked files" in guide
