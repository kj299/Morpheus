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
The first two detections, applied in Python to the harness corpus, and checked against the Splunk app's own stanzas.

R-D-L2-004 and R-D-L2-005 are the first rules in the design guide that exist as code. They ship as saved searches,
and a saved search cannot run here. What can run here is the predicate each search encodes, over the same columns,
on the corpus with the anomalies planted in it. Each must fire exactly once, on the planted row, with the fields
an analyst needs to trace it. The stanzas are then read from the app itself, so the SPL and the Python cannot
drift apart silently.
"""

import os
import re
import sys

import pandas as pd
import pytest

from morpheus.utils.binding_closer import CONFLICT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import telemetry_pipeline as tp  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
APP_DEFAULT = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "TA-morpheus-lineage", "default")
SAVED_SEARCHES = os.path.join(APP_DEFAULT, "savedsearches.conf")
PROPS = os.path.join(APP_DEFAULT, "props.conf")

NS = tp.NS_PER_SECOND

RULES = {
    "R-D-L2-004": "R-D-L2-004 - MAC in two places at once",
    "R-D-L2-005": "R-D-L2-005 - Authorization without authentication",
}


def read_conf(path: str) -> dict[str, dict[str, str]]:
    """Parse a Splunk .conf file: `[stanza]` headers, `key = value` pairs, backslash line continuations."""
    with open(path, encoding="utf-8") as handle:
        raw = handle.read()

    joined = re.sub(r"\\\n", " ", raw)
    stanzas: dict[str, dict[str, str]] = {}
    current = None

    for line in joined.splitlines():
        stripped = line.strip()

        if (not stripped or stripped.startswith("#")):
            continue

        if (stripped.startswith("[") and stripped.endswith("]")):
            current = stripped[1:-1]
            stanzas[current] = {}
        elif (current is not None and "=" in stripped):
            (key, value) = stripped.split("=", 1)
            stanzas[current][key.strip()] = value.strip()

    return stanzas


@pytest.fixture(name="result", scope="module")
def result_fixture() -> pd.DataFrame:
    yield tp.run_pipeline(tp.build_pipeline_config(), tp.build_corpus())


@pytest.fixture(name="searches", scope="module")
def searches_fixture() -> dict[str, dict[str, str]]:
    yield read_conf(SAVED_SEARCHES)


# --- The predicates, in Python, over the planted corpus ---------------------------------------------------------


@pytest.mark.cpu_mode
def test_r_d_l2_004_fires_exactly_once_on_the_spoof(result: pd.DataFrame):
    bindings = result[result["telemetry_class"] == "tc2_binding"]
    detections = bindings[bindings["bind_end_reason"] == CONFLICT]

    assert len(detections) == 1
    hit = detections.iloc[0]
    assert hit["mac_address"] == tp.MAC_A
    assert hit["port_key"] == f"{tp.SITE}:{tp.SWITCH}:Gi1/0/1"
    # The detection is about the closing, and the closing is one tick past the conflicting sighting.
    assert hit["bind_end"] == tp.SPOOF_AT_SECONDS * NS + 1
    # Everything the search's `table` names is present, so an analyst can trace it without a second query.
    for column in ("mac_address",
                   "port_key",
                   "site_id",
                   "switch_id",
                   "port_id",
                   "vlan_id",
                   "bind_start",
                   "bind_end",
                   "bind_observations"):
        assert pd.notna(hit[column]), column


@pytest.mark.cpu_mode
def test_r_d_l2_005_fires_exactly_once_on_the_bypass(result: pd.DataFrame):
    auth = result[result["telemetry_class"] == "tc2_auth"]
    detections = auth[auth["auth_unpaired"] == True]  # noqa: E712  pylint: disable=singleton-comparison

    assert len(detections) == 1
    hit = detections.iloc[0]
    assert hit["event_time"] == tp.BYPASS_AT_SECONDS * NS
    assert hit["auth_port_key"] == f"{tp.SITE}:{tp.SWITCH}:{tp.BYPASS_PORT}"
    assert hit["dot1x_result"] == "success"

    for column in ("auth_port_key", "site_id", "switch_id", "port_id", "dot1x_result", "event_uid"):
        assert pd.notna(hit[column]), column

    # Every paired exchange in the corpus reads False, not null: the rule's negative is an answer, not an absence.
    outcomes = auth[auth["dot1x_result"] == "success"]
    assert (outcomes["auth_unpaired"] == False).sum() == len(outcomes) - 1  # noqa: E712  pylint: disable=singleton-comparison


# --- The stanzas, read from the app itself ------------------------------------------------------------------------


def test_both_detections_are_defined(searches: dict[str, dict[str, str]]):
    for stanza in RULES.values():
        assert stanza in searches, stanza


@pytest.mark.parametrize("rule_id", sorted(RULES))
def test_the_search_reads_the_column_the_stage_emits(rule_id: str, searches: dict[str, dict[str, str]]):
    spl = searches[RULES[rule_id]]["search"]

    expected = {
        "R-D-L2-004": ("sourcetype=binding:l2", f"bind_end_reason={CONFLICT}", "port_key"),
        "R-D-L2-005": ("sourcetype=morpheus:score:l2", "auth_unpaired=true", "auth_port_key", "event_uid"),
    }[rule_id]

    for fragment in expected:
        assert fragment in spl, f"{rule_id} does not read {fragment}"

    assert f'rule_id = "{rule_id}"' in spl


@pytest.mark.parametrize("rule_id", sorted(RULES))
def test_the_search_follows_the_scheduling_discipline(rule_id: str, searches: dict[str, dict[str, str]]):
    # Part 5's rules for a reproducible search: never end at now, snap to the minute, continuous scheduling.
    stanza = searches[RULES[rule_id]]

    assert stanza["realtime_schedule"] == "0"
    assert stanza["enableSched"] == "1"
    assert stanza["dispatch.latest_time"].endswith("@m")
    assert stanza["dispatch.earliest_time"].endswith("@m")

    trailing_minutes = int(re.fullmatch(r"-(\d+)m@m", stanza["dispatch.latest_time"]).group(1))
    assert trailing_minutes >= tp.LATENESS_SECONDS // 60, "the window must trail by at least the lateness horizon"

    # Window width equals the cadence, so consecutive runs are disjoint and a detection is emitted once.
    earliest_minutes = int(re.fullmatch(r"-(\d+)m@m", stanza["dispatch.earliest_time"]).group(1))
    cadence_minutes = int(re.fullmatch(r"\*/(\d+) \* \* \* \*", stanza["cron_schedule"]).group(1))
    assert earliest_minutes - trailing_minutes == cadence_minutes


def test_the_binding_sourcetype_is_timed_on_the_end():
    # A binding opened hours ago and closed by a conflict now must land in the detection window that covers now.
    props = read_conf(PROPS)

    assert "binding:l2" in props
    assert '"bind_end"' in props["binding:l2"]["TIME_PREFIX"]
    assert props["binding:l2"]["TIME_FORMAT"] == props["morpheus:score:l2"]["TIME_FORMAT"]
