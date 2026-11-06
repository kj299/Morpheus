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
The estate's context is an input to a composition, and the package holds none of it (plan section 4.5b).

Until 4.5b most compositions built their identity, asset and application stores from a synthetic estate written into
their own modules, so a runner fed a real estate's traffic enriched it with the fixture's. These hold the package to
the change: every composition that enriches takes an `EstateContext` with no default; no module of
`morpheus.compositions` writes a version, an address, a principal or a MAC down; and a corpus's context survives the
files a runner reads it from.
"""

import ast
import importlib
import inspect
import json
import os
import pathlib
import re
import sys

import pytest

from morpheus.compositions.estate_context import EXTRAS_FILE
from morpheus.compositions.estate_context import KINDS
from morpheus.compositions.estate_context import EstateContext
from morpheus.compositions.estate_context import MissingContext
from morpheus.compositions.estate_context import file_name
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = HERE.parents[2]
COMPOSITIONS = REPO_ROOT / "python" / "morpheus" / "morpheus" / "compositions"
EXAMPLE = REPO_ROOT / "examples" / "behavioral_analytics"

sys.path.insert(0, str(EXAMPLE))

ESTATE_LITERALS = {
    "an IPv4 address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "an email address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "a MAC address": re.compile(r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b"),
}
"""What an estate is written down as. A composition's string constants name columns, classes, sourcetypes and rules."""

VERSION_WRITERS = {"make_version", "Version", "snapshot_changes"}
"""What writes a context version down; a composition rebuilds stores from records it is given (`from_records`)."""


def _composition_modules() -> list[str]:
    return sorted(path.stem for path in COMPOSITIONS.glob("*.py") if not path.stem.startswith("_"))


def _string_constants(tree: ast.AST) -> list[ast.Constant]:
    docstrings = set()

    for node in ast.walk(tree):
        body = getattr(node, "body", None)

        if (isinstance(body, list)):
            # A docstring, or an attribute docstring after an assignment, is prose about the estate, not the estate.
            for statement in body:
                if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)):
                    docstrings.add(id(statement.value))

    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    ]


def _estate_in(source: str) -> list[str]:
    tree = ast.parse(source)
    found = []

    for node in _string_constants(tree):
        found.extend(f"line {node.lineno}: {what} {node.value!r}" for (what, pattern) in ESTATE_LITERALS.items()
                     if pattern.search(node.value))

    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)

            if (name in VERSION_WRITERS):
                found.append(f"line {node.lineno}: writes a context version with {name}")

        if (isinstance(node, (ast.Import, ast.ImportFrom))):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            found.extend(f"line {node.lineno}: imports {module}" for module in modules
                         if module.split(".")[0] == "corpora")

    return found


def test_no_composition_holds_an_estate():
    found = {
        module: offences
        for module in _composition_modules()
        for offences in [_estate_in((COMPOSITIONS / f"{module}.py").read_text(encoding="utf-8"))] if offences
    }

    assert found == {}


@pytest.mark.parametrize("source,expected",
                         [
                             ('HOSTS = {"10.0.1.11": "finance"}\n', "IPv4"),
                             ('ATTACKER = "dana@example.com"\n', "email"),
                             ('MAC_A = "aa:bb:cc:00:00:01"\n', "MAC"),
                             ('store = make_version("asset", "h", 0, None, 0)\n', "make_version"),
                             ("from corpora import network\n", "corpora"),
                         ])
def test_the_check_sees_an_estate_written_down(source: str, expected: str):
    assert any(expected in offence for offence in _estate_in(source))


def test_the_check_passes_prose_about_an_estate():
    source = '"""Joins 10.0.1.11 to its owner, alice@example.com."""\nPREFIX = "ctx_"\n"""As 10.0.0.1 is."""\n'

    assert _estate_in(source) == []


def _context_parameters() -> dict:
    found = {}

    for module_name in _composition_modules():
        module = importlib.import_module(f"morpheus.compositions.{module_name}")

        for (name, function) in inspect.getmembers(module, inspect.isfunction):
            if (function.__module__ == module.__name__ and "context" in inspect.signature(function).parameters):
                found[f"{module_name}.{name}"] = inspect.signature(function).parameters["context"]

    return found


def test_every_composition_that_enriches_takes_its_context_with_no_default():
    parameters = _context_parameters()

    assert {
        f"{name}.run_pipeline"
        for name in ("network", "transport", "session", "presentation", "application", "saas", "endpoint", "host_score",
                     "telemetry", "estate", "lineage", "campaign")
    } <= set(parameters)

    for (name, parameter) in parameters.items():
        assert parameter.kind == inspect.Parameter.KEYWORD_ONLY, name
        assert parameter.default is None, name


def _store(name: str) -> BitemporalStore:
    return BitemporalStore(
        name,
        [
            make_version(name, "ws-01", 0, None, 0, values={
                "role": "workstation", "owner": "finance"
            }),
            make_version(name, "ws-02", 0, 10, 0, values={
                "role": "server", "owner": None
            }),
            make_version(name, "ws-02", 10, None, 5, values={
                "role": "workstation", "owner": "ops"
            }),
        ])


def test_require_returns_the_stores_in_the_order_named():
    (asset, identity) = (_store("asset"), _store("identity"))
    context = EstateContext(identity=identity, asset=asset)

    assert context.require("network", "asset", "identity") == (asset, identity)


def test_require_names_what_is_missing_and_where_it_is_read_from():
    context = EstateContext(asset=_store("asset"))

    with pytest.raises(MissingContext, match="presentation enriches with the estate's application context") as caught:
        context.require("presentation", "asset", "application")

    assert file_name("application") in str(caught.value)
    assert file_name("asset") not in str(caught.value)


def test_require_extra_names_what_is_missing():
    context = EstateContext(extras={"vrrp_addresses": ["10.0.0.1"]})

    assert context.require_extra("telemetry", "vrrp_addresses") == (["10.0.0.1"], )

    with pytest.raises(MissingContext, match="bindings"):
        context.require_extra("lineage", "vrrp_addresses", "bindings")


def test_missing_context_is_a_value_error():
    assert issubclass(MissingContext, ValueError)


def test_a_directory_reads_back_to_the_same_versions(tmp_path):
    context = EstateContext(identity=_store("identity"),
                            asset=_store("asset"),
                            application=_store("application"),
                            extras={
                                "vrrp_addresses": ["10.0.0.1"], "site_locations": {
                                    "a": [51.5, -0.1]
                                }
                            })

    written = context.write_directory(str(tmp_path))
    read = EstateContext.from_directory(str(tmp_path))

    assert sorted(os.path.basename(path) for path in written) == sorted([file_name(kind)
                                                                         for kind in KINDS] + [EXTRAS_FILE])

    for kind in KINDS:
        assert getattr(read, kind).versions() == getattr(context, kind).versions(), kind

    assert read.extras == context.extras


def test_what_is_absent_from_a_directory_is_absent_from_its_context(tmp_path):
    EstateContext(asset=_store("asset")).write_directory(str(tmp_path))
    read = EstateContext.from_directory(str(tmp_path))

    assert read.asset is not None
    assert (read.identity, read.application, read.directory, dict(read.extras)) == (None, None, None, {})


def test_an_extra_json_cannot_carry_is_left_out_and_then_refused(tmp_path):
    context = EstateContext(extras={"handshake_asset": _store("asset"), "session_end_ns": 7})
    context.write_directory(str(tmp_path))

    assert json.loads((tmp_path / EXTRAS_FILE).read_text(encoding="utf-8")) == {"session_end_ns": 7}

    with pytest.raises(MissingContext, match="handshake_asset"):
        EstateContext.from_directory(str(tmp_path)).require_extra("campaign", "handshake_asset")


def test_render_is_what_write_directory_writes(tmp_path):
    context = EstateContext(asset=_store("asset"), extras={"vrrp_addresses": ["10.0.0.1"]})
    context.write_directory(str(tmp_path))

    assert {path.name: path.read_text(encoding="utf-8") for path in tmp_path.iterdir()} == context.render()


@pytest.mark.parametrize(
    "corpus",
    ["network", "transport", "session", "presentation", "application", "saas", "endpoint", "host_score", "telemetry"])
def test_a_segment_corpus_context_survives_its_files(corpus: str, tmp_path):
    from corpora import merged  # pylint: disable=import-outside-toplevel

    context = merged(corpus).context()
    context.write_directory(str(tmp_path))
    read = EstateContext.from_directory(str(tmp_path))

    for kind in KINDS:
        if (getattr(context, kind) is None):
            assert getattr(read, kind) is None, kind
        else:
            assert getattr(read, kind).versions() == getattr(context, kind).versions(), kind

    assert read.extras == json.loads(json.dumps(dict(context.extras)))
