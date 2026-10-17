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
The transport configuration names what the pipeline produces and where the app reads it.

`examples/splunk_lineage_app/transport/` is rendered from the sourcetype inventory. These tests hold the committed
rendering to the renderer, and the renderer to the four things it has to agree with: the runners' topic and file
names, the index each search reads a sourcetype from, the stanzas `props.conf` parses, and the indexes
`indexes.conf` defines.
"""

import configparser
import importlib.util
import json
import os
import re
import sys

import pytest

from morpheus.utils.siem_sourcetypes import INDEX_BY_PREFIX
from morpheus.utils.siem_sourcetypes import PRODUCED
from morpheus.utils.siem_sourcetypes import index_for
from morpheus.utils.siem_sourcetypes import stanza_names
from morpheus.utils.siem_wire import filename_for
from morpheus.utils.siem_wire import topic_for

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
APP = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app")
TRANSPORT = os.path.join(APP, "transport")
DEFAULT = os.path.join(APP, "TA-morpheus-lineage", "default")

sys.path.insert(0, TRANSPORT)

# pylint: disable=wrong-import-position
import make_transport  # noqa: E402

RENDERED = {
    "kafka-connect-splunk.json": make_transport.render_connector,
    "inputs.conf": make_transport.render_inputs,
    "README.md": make_transport.render_readme,
}

SEARCH_PAIR = re.compile(r"index=(behavior_\w+)\s+sourcetype=([a-z0-9:]+)\b")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _conf(path: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read(path)

    return parser


def _connector() -> dict:
    return json.loads(_read(os.path.join(TRANSPORT, "kafka-connect-splunk.json")))["config"]


@pytest.mark.parametrize("name", sorted(RENDERED))
def test_the_committed_file_is_what_the_renderer_writes(name: str):
    assert _read(os.path.join(TRANSPORT, name)) == RENDERED[name](), f"run make_transport.py; {name} is stale"


def test_every_sourcetype_has_an_index_and_the_searches_agree():
    for name in stanza_names():
        index_for(name)  # raises for a sourcetype the table does not place

    pairs = set(SEARCH_PAIR.findall(_read(os.path.join(DEFAULT, "savedsearches.conf"))))

    assert len(pairs) > 10

    # `sourcetype=morpheus:score:l` is a prefix search over every layer, and the table places it the same way.
    assert all(index_for(sourcetype) == index for (index, sourcetype) in pairs), sorted(pairs)


def test_the_validation_script_places_sourcetypes_the_same_way():
    # The search-head runner is kept to the standard library so it runs on a host without Morpheus; its copy of the
    # table has to be this one.
    spec = importlib.util.spec_from_file_location("run_search_head",
                                                  os.path.join(APP, "validate", "run_search_head.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # The prefixes do not overlap, so the order of the pairs carries nothing.
    assert set(module.INDEX_BY_PREFIX) == set(INDEX_BY_PREFIX)


def test_the_connector_maps_every_produced_sourcetype_positionally():
    config = _connector()
    topics = config["topics"].split(",")
    sourcetypes = config["splunk.sourcetypes"].split(",")
    indexes = config["splunk.indexes"].split(",")

    assert sourcetypes == sorted(PRODUCED)
    assert topics == [topic_for(name) for name in sourcetypes]
    assert indexes == [index_for(name) for name in sourcetypes]
    assert len(topics) == len(set(topics))


def test_the_connector_posts_raw_records_separated_by_newlines_with_acknowledgement():
    config = _connector()

    assert config["splunk.hec.raw"] == "true"
    assert config["splunk.hec.raw.line.breaker"] == "\n"
    assert config["splunk.hec.ack.enabled"] == "true"
    assert config["splunk.hec.ssl.validate.certs"] == "true"
    # The token comes through a configuration provider; a literal here would be a secret in the repository.
    assert config["splunk.hec.token"].startswith("${file:")


def test_the_props_stanzas_break_on_newlines_and_do_not_merge_lines():
    # The connector's line breaker is a newline, and a stanza that merged lines or broke elsewhere would make one
    # event of a batch.
    props = _conf(os.path.join(DEFAULT, "props.conf"))

    for name in sorted(PRODUCED):
        assert props.has_section(name), name
        assert props.get(name, "SHOULD_LINEMERGE").lower() == "false", name
        assert not props.has_option(name, "LINE_BREAKER"), name


def test_the_forwarder_monitors_every_runner_file_onto_its_sourcetype_and_index():
    inputs = _conf(os.path.join(TRANSPORT, "inputs.conf"))
    monitors = {section for section in inputs.sections() if section.startswith("monitor://")}

    assert monitors == {f"monitor://{make_transport.SPOOL}/*/{filename_for(name)}" for name in PRODUCED}

    for name in sorted(PRODUCED):
        section = f"monitor://{make_transport.SPOOL}/*/{filename_for(name)}"

        assert inputs.get(section, "sourcetype") == name
        assert inputs.get(section, "index") == index_for(name)
        assert inputs.get(section, "crcSalt") == "<SOURCE>"


def test_the_token_allows_exactly_the_indexes_the_sourcetypes_use():
    inputs = _conf(os.path.join(TRANSPORT, "inputs.conf"))
    section = f"http://{make_transport.HEC_TOKEN_NAME}"

    assert inputs.get(section, "indexes").split(",") == sorted({index_for(name) for name in PRODUCED})
    assert inputs.get(section, "index") in inputs.get(section, "indexes").split(",")
    assert inputs.get(section, "useACK") == "1"
    assert inputs.get("http", "enableSSL") == "1"


def test_every_index_named_exists_in_the_app():
    indexes = _conf(os.path.join(DEFAULT, "indexes.conf"))
    named = {index_for(name) for name in PRODUCED} | set(_connector()["splunk.indexes"].split(","))

    assert all(indexes.has_section(index) for index in named), sorted(named - set(indexes.sections()))


def test_the_app_readme_points_at_the_transport_directory():
    readme = _read(os.path.join(APP, "README.md"))

    assert "transport/kafka-connect-splunk.json" in readme
    assert "transport/inputs.conf" in readme
