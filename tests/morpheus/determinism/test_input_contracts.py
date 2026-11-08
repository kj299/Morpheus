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
Every input class has a contract, every corpus builder keeps to it, and the samples are what the builders build.

`examples/behavioral_analytics/input_contracts.json` names, per class, each column a collector sends with its type,
unit and source. This holds every corpus builder in the tree to it -- every corpus in `CORPORA`, not only the one
each class's sample is taken from -- so a builder that gains a column without the contract saying what it is fails
here, and checks that the committed samples and `collector_contract.md` are what `make_input_samples.py` renders
now.
"""

import importlib
import json
import os
import re
import sys

import pytest

from _utils.number_words import from_words

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(HERE, "..", "..", "..", "examples", "behavioral_analytics")

sys.path.insert(0, HERE)
sys.path.insert(0, EXAMPLE)

# pylint: disable=wrong-import-position
import make_input_samples as samples  # noqa: E402
import run_golden  # noqa: E402

SINGLE_FRAME = {"lineage": "one frame of its own reference shape, with no class"}
"""The corpora without per-class input frames, each with the reason."""

CORPORA = tuple(name for name in run_golden.CORPORA if name not in SINGLE_FRAME)
"""Every corpus with per-class input frames: every corpus with a golden file, less those above."""

JSON_TYPES = {"integer": (int, ), "float": (int, float), "string": (str, ), "boolean": (bool, )}

CONTRACTS = samples.load_contracts()


@pytest.fixture(name="builders", scope="module")
def builders_fixture() -> dict:
    return {name: importlib.import_module(f"{name}_pipeline").build_corpus() for name in CORPORA}


def _columns(class_name: str) -> tuple[set, set]:
    """Every column a class's contract names, and the ones it requires."""
    spec = CONTRACTS["classes"][class_name]
    envelope = {column["name"] for column in CONTRACTS["envelope"]} if spec["envelope"] else set()
    named = envelope | {column["name"] for column in spec["columns"]}
    required = envelope | {column["name"] for column in spec["columns"] if column["required"]}

    return (named, required)


def test_every_frame_a_builder_produces_has_a_contract(builders: dict):
    unknown = sorted({
        (corpus, name)
        for (corpus, frames) in builders.items()
        for name in frames
        if samples.contract_class(name) is not None and samples.contract_class(name) not in CONTRACTS["classes"]
    })

    assert unknown == [], f"classes no contract describes: {unknown}"


@pytest.mark.parametrize("corpus", CORPORA)
def test_every_builder_sends_what_its_contract_names_and_nothing_else(builders: dict, corpus: str):
    for (name, frame) in builders[corpus].items():
        class_name = samples.contract_class(name)

        if (class_name is None):
            continue

        (named, required) = _columns(class_name)
        columns = set(frame.columns)

        assert columns <= named, f"{corpus}/{name} sends {sorted(columns - named)}, which the contract does not name"
        assert required <= columns, f"{corpus}/{name} omits {sorted(required - columns)}, which the contract requires"


def test_every_column_a_contract_names_is_sent_by_some_builder(builders: dict):
    # A contract row nothing sends is a column a collector would be asked to build for nobody.
    sent: dict[str, set] = {}

    for frames in builders.values():
        for (name, frame) in frames.items():
            class_name = samples.contract_class(name)

            if (class_name is not None):
                sent.setdefault(class_name, set()).update(frame.columns)

    for class_name in CONTRACTS["classes"]:
        (named, _) = _columns(class_name)

        assert named <= sent.get(class_name, set()), f"{class_name}: {sorted(named - sent.get(class_name, set()))}"


def test_an_optional_column_is_one_some_builder_omits(builders: dict):
    # And a required one is sent by every builder, which the parametrized test above already holds; this is the
    # other direction, so 'optional' cannot be the lazy answer for a column every builder carries.
    carried: dict[str, list[set]] = {}

    for frames in builders.values():
        for (name, frame) in frames.items():
            class_name = samples.contract_class(name)

            if (class_name is not None and name == class_name):
                carried.setdefault(class_name, []).append(set(frame.columns))

    for (class_name, spec) in CONTRACTS["classes"].items():
        everywhere = set.intersection(*carried[class_name])

        for column in spec["columns"]:
            if (not column["required"]):
                assert column["name"] not in everywhere, f"{class_name}.{column['name']} is sent by every builder"


SCHEMA_VERSION = re.compile(r"(?P<class>\w+)\.v(?P<major>\d+)")
"""`<class>.v<major>`: the one form every class's `schema_version` takes (plan section 4.7a)."""

DIGEST = re.compile(r"[0-9a-f]{12,}")
"""A bare lowercase hex digest: an identifier the collector invented rather than the one its source gives."""


def envelope_offences(class_name: str, record: dict) -> list[str]:
    """What is wrong with a record's envelope against the one convention, or nothing."""
    offences = []
    version = SCHEMA_VERSION.fullmatch(str(record.get("schema_version")))

    if (version is None or version.group("class") != class_name):
        offences.append(f"schema_version {record.get('schema_version')!r} is not {class_name}.v<major>")

    origin = record.get("origin_hash")

    if (not isinstance(origin, str) or origin == "" or DIGEST.fullmatch(origin)):
        offences.append(f"origin_hash {origin!r} is not the source's own identifier")

    return offences


def test_every_sample_follows_the_one_envelope_convention():
    found = {}

    for path in sorted(samples.SAMPLES.glob("*/*.jsonlines")):
        class_name = samples.contract_class(path.stem)

        if (class_name is None or not CONTRACTS["classes"][class_name]["envelope"]):
            continue

        with open(path, encoding="utf-8") as handle:
            for (line_number, line) in enumerate(handle, start=1):
                for offence in envelope_offences(class_name, json.loads(line)):
                    found.setdefault(f"{path.parent.name}/{path.name}", []).append(f"line {line_number}: {offence}")

    assert {name: offences[:3] for (name, offences) in found.items()} == {}


@pytest.mark.parametrize("record,expected",
                         [({
                             "schema_version": "TC-1/1.0.0", "origin_hash": "port@1"
                         }, "schema_version"), ({
                             "schema_version": "tc3.v1", "origin_hash": "port@1"
                         }, "schema_version"),
                          ({
                              "schema_version": "tc1.v1", "origin_hash": "5a26e75489d75e6e"
                          }, "origin_hash"), ({
                              "schema_version": "tc1.v1", "origin_hash": None
                          }, "origin_hash")])
def test_the_convention_check_sees_each_departure(record: dict, expected: str):
    offences = envelope_offences("tc1", record)

    assert len(offences) == 1 and offences[0].startswith(expected)


def test_the_convention_admits_a_guid_and_an_observation():
    assert envelope_offences("tc5_auth", {
        "schema_version": "tc5_auth.v1", "origin_hash": "0b0f7a7e-5f43-5a7e-9a6c-2f4f8a1d3c55"
    }) == []
    assert envelope_offences("tc1", {"schema_version": "tc1.v1", "origin_hash": "sw1/Gi1/0/1@60000000000"}) == []


def test_every_sample_value_has_its_contract_type():
    for path in sorted(samples.SAMPLES.glob("*/*.jsonlines")):
        class_name = samples.contract_class(path.stem)

        if (class_name is None):
            continue

        spec = CONTRACTS["classes"][class_name]
        columns = {column["name"]: column for column in spec["columns"]}

        if (spec["envelope"]):
            columns.update({column["name"]: column for column in CONTRACTS["envelope"]})

        with open(path, encoding="utf-8") as handle:
            for (line_number, line) in enumerate(handle, start=1):
                for (name, value) in json.loads(line).items():
                    expected = JSON_TYPES[columns[name]["type"]]
                    # bool is an int in Python; an integer column holding true is still wrong.
                    wrong = value is not None and (not isinstance(value, expected) or
                                                   (isinstance(value, bool) and bool not in expected))

                    assert not wrong, f"{path.name}:{line_number} {name}={value!r} is not {columns[name]['type']}"


def test_every_class_has_a_sample_in_its_home_segment():
    for (class_name, spec) in CONTRACTS["classes"].items():
        assert spec["home"] in samples.SEGMENTS
        assert (samples.SAMPLES / spec["home"] / f"{class_name}.jsonlines").is_file(), class_name


def test_the_samples_are_what_the_builders_build():
    rendered = samples.render_samples()
    committed = set(samples.SAMPLES.glob("*/*.jsonlines")) | set(samples.SAMPLES.glob(f"*/{samples.CONTEXT_DIR}/*"))

    assert committed == set(rendered), "regenerate with make_input_samples.py"

    for (path, text) in rendered.items():
        assert path.read_text(encoding="utf-8") == text, f"{path} is stale; regenerate with make_input_samples.py"


def test_the_contract_document_is_rendered_from_the_contract():
    assert samples.CONTRACT_DOCUMENT.read_text(encoding="utf-8") == samples.render_contract(), (
        "collector_contract.md is stale; regenerate with make_input_samples.py")


def test_the_documents_count_the_classes_the_contract_holds():
    # Three documents say how many input classes there are, and the retrospective's own figure was already six
    # behind when this was written.

    root = os.path.join(HERE, "..", "..", "..")
    documents = [
        os.path.join(root, "docs", "source", "developer_guide", "guides", "11_predictive_behavioral_analytics_osi.md"),
        os.path.join(EXAMPLE, "README.md"),
    ]

    for path in documents:
        with open(path, encoding="utf-8") as handle:
            stated = re.findall(r"all ([a-z-]+) input classes", handle.read())

        assert stated, path
        assert {from_words(word) for word in stated} == {len(CONTRACTS["classes"])}, (path, stated)
