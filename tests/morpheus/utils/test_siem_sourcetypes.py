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
    # Every stanza has a producer. Pinned rather than merely recorded, so that adding a stanza, with or without a
    # producer, is a deliberate edit here and not a silent drift in what the app appears to support.
    assert len(UNPRODUCED) == 0
    assert len(PRODUCED) == 19


NUMBER_WORDS = {
    # "none" because the sentence says none of the stanzas is without a producer; the rest reach the stanza total,
    # and a total past nineteen fails here with a KeyError naming the word to add.
    "none": 0,
    **{
        word: number
        for (number, word) in enumerate(("zero one two three four five six seven eight nine ten eleven twelve "
                                         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split())
    },
}

GUIDE = os.path.join(REPO_ROOT,
                     "docs",
                     "source",
                     "developer_guide",
                     "guides",
                     "11_predictive_behavioral_analytics_osi.md")


def test_the_guide_states_the_same_unproduced_count_the_module_holds():
    # The count above is pinned in code and was correct; the sentence in the guide saying the same thing was not,
    # and stayed wrong through the whole layer 5 increment. A number written in prose beside a number held in code
    # is a number that will disagree with it, and nothing but a reader was checking. This is that reader.
    with open(GUIDE, encoding="utf-8") as handle:
        text = re.sub(r"\s+", " ", handle.read())

    # "have no producer" while there were some; "is without a producer" once there were none, because "none of the
    # stanzas have no producer" says the opposite of what it means to anyone reading it quickly.
    match = re.search(r"(\w+) of the (\w+) stanzas (?:have no producer|is without a producer)", text, re.IGNORECASE)

    assert match is not None, "the guide no longer states the unproduced count; keep it or drop this test"

    stated_unproduced = NUMBER_WORDS[match.group(1).lower()]
    stated_total = NUMBER_WORDS[match.group(2).lower()]

    assert stated_unproduced == len(UNPRODUCED), (
        f"the guide says {match.group(1)} stanzas have no producer; the module holds {len(UNPRODUCED)}")
    assert stated_total == len(PRODUCED) + len(UNPRODUCED)


def test_the_guide_no_longer_breaks_down_an_unproduced_count():
    # The guide once split the unproduced count into score sourcetypes, the layer 1 inventory feed and the context
    # store. Nothing is unproduced now, and the count test above holds the guide to that; what must not survive is
    # the old breakdown beside it.
    with open(GUIDE, encoding="utf-8") as handle:
        text = re.sub(r"\s+", " ", handle.read())

    assert len(UNPRODUCED) == 0, "a stanza lost its producer; restore the breakdown check this replaced"
    breakdown = re.search(r"stanzas have no producer in this fork: \w+ (?:is|are) the score sourcetypes?", text)

    assert breakdown is None, "the guide still breaks down an unproduced count the module no longer holds"


def test_asking_for_an_unproduced_sourcetype_says_what_would_have_to_exist(monkeypatch):
    # Nothing is unproduced any longer, so the mechanism is exercised on a stanza that exists only for this test.
    placeholder = Unproduced("context:placeholder", "valid_from", "A producer this test pretends is not built yet.")
    monkeypatch.setitem(UNPRODUCED, placeholder.name, placeholder)

    with pytest.raises(ValueError, match="pretends is not built"):
        sourcetype(placeholder.name)

    assert isinstance(describe(placeholder.name), Unproduced)


def test_the_context_store_is_produced():
    entry = sourcetype("context:identity")

    assert entry.time_column == "valid_from"
    assert "recorded_at" in entry.time_columns
    assert sourcetype("context:asset").time_column == "valid_from"


def test_asking_for_a_produced_sourcetype_returns_it():
    entry = sourcetype("morpheus:score:l2")

    assert isinstance(entry, Sourcetype)
    assert entry.time_column == "event_time"


def test_an_unknown_sourcetype_lists_the_known_ones():
    with pytest.raises(KeyError, match="morpheus:score:l2"):
        describe("morpheus:score:l9")
