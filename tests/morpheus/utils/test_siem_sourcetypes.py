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
The other half of the parsing contract: every stanza the app parses is accounted for by a producer or a reason.

`props.conf` says what the SIEM will look for. Until now nothing said what this fork emits, so a stanza could
anchor `_time` on a field no producer writes and the only symptom would be events silently stamped at index time.
These tests read the app's actual configuration file rather than a copy of its values, so the two halves cannot
drift apart.
"""

import configparser
import os
import re

import pytest

from morpheus.utils.siem_sourcetypes import PRODUCED
from morpheus.utils.siem_sourcetypes import UNPRODUCED
from morpheus.utils.siem_sourcetypes import Sourcetype
from morpheus.utils.siem_sourcetypes import Unproduced
from morpheus.utils.siem_sourcetypes import describe
from morpheus.utils.siem_sourcetypes import sourcetype
from morpheus.utils.siem_sourcetypes import stanza_names

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PROPS_PATH = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "TA-morpheus-lineage", "default", "props.conf")

# `"event_time"\s*:\s*"` anchors on event_time. The field name is what the prefix opens with.
ANCHOR_PATTERN = re.compile(r'^"(?P<field>[A-Za-z0-9_]+)"')


def load_props() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(PROPS_PATH)

    return parser


def anchor_column(time_prefix: str) -> str:
    match = ANCHOR_PATTERN.match(time_prefix)

    assert match is not None, f"TIME_PREFIX does not open with a quoted field name: {time_prefix}"

    return match.group("field")


def test_the_configuration_file_is_where_we_think_it_is():
    # If this fails every other test here is asserting against an empty parse, which would pass vacuously.
    assert os.path.exists(PROPS_PATH)
    assert len(load_props().sections()) > 0


def test_every_stanza_is_accounted_for_exactly_once():
    stanzas = set(load_props().sections())

    assert stanzas == set(stanza_names()), (
        f"props.conf and siem_sourcetypes disagree. Only in props.conf: {sorted(stanzas - set(stanza_names()))}. "
        f"Only in siem_sourcetypes: {sorted(set(stanza_names()) - stanzas)}.")
    assert set(PRODUCED).isdisjoint(set(UNPRODUCED)), "a sourcetype cannot be both produced and unproduced"


@pytest.mark.parametrize("name", sorted(load_props().sections()))
def test_the_declared_time_column_is_the_one_the_stanza_anchors_on(name: str):
    # The assertion the whole module exists for. A producer rendering the wrong column would leave _time falling
    # back to index time, with nothing raised anywhere.
    settings = load_props()[name]

    assert describe(name).time_column == anchor_column(settings["TIME_PREFIX"])


@pytest.mark.parametrize("name", sorted(PRODUCED))
def test_a_produced_sourcetype_renders_its_own_anchor(name: str):
    entry = PRODUCED[name]

    assert entry.name == name
    assert entry.time_column in entry.time_columns, "the anchor column must be one of the rendered columns"
    assert entry.producer != ""
    # A column carrying exact nanoseconds is not a timestamp to round; see the module docstring.
    assert not any(column.endswith("_ns") for column in entry.time_columns)


@pytest.mark.parametrize("name", sorted(UNPRODUCED))
def test_an_unproduced_sourcetype_says_what_is_missing(name: str):
    assert UNPRODUCED[name].name == name
    assert len(UNPRODUCED[name].missing) > 40, "name what would have to be built, not just that it is absent"


def test_the_unproduced_count_is_pinned():
    # Seven of the fourteen stanzas are configuration for producers this fork has not built. Pinned rather than
    # merely recorded, so that landing a producer is a deliberate edit here and not a silent drift in what the
    # app appears to support -- which is exactly what this assertion caught when layer 5 gained one.
    assert len(UNPRODUCED) == 7
    assert len(PRODUCED) == 7


def test_asking_for_an_unproduced_sourcetype_says_what_would_have_to_exist():
    with pytest.raises(ValueError, match="TC-0 context store"):
        sourcetype("context:identity")

    assert isinstance(describe("context:identity"), Unproduced)


def test_asking_for_a_produced_sourcetype_returns_it():
    entry = sourcetype("morpheus:score:l2")

    assert isinstance(entry, Sourcetype)
    assert entry.time_column == "event_time"


def test_an_unknown_sourcetype_lists_the_known_ones():
    with pytest.raises(KeyError, match="morpheus:score:l2"):
        describe("morpheus:score:l9")
