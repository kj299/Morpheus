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
The thresholds the detections fire at, held in three places that must agree: the searches, the registry, and the
digest every risk record carries (gap G60).
"""

import collections
import configparser
import csv
import json
import os
import re
import sys

import pytest

from morpheus.utils.rule_thresholds import DIRECTIONS
from morpheus.utils.rule_thresholds import KINDS
from morpheus.utils.rule_thresholds import OPERATORS
from morpheus.utils.rule_thresholds import derive_threshold
from morpheus.utils.rule_thresholds import extract_thresholds
from morpheus.utils.rule_thresholds import thresholds_hash

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
APP = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app")
SAVEDSEARCHES = os.path.join(APP, "TA-morpheus-lineage", "default", "savedsearches.conf")
REGISTRY = os.path.join(APP, "TA-morpheus-lineage", "lookups", "rule_thresholds.csv")
RULE_METADATA = os.path.join(APP, "TA-morpheus-lineage", "lookups", "rule_metadata.csv")
VALIDATE = os.path.join(APP, "validate")
SAMPLE_EVENTS = os.path.join(VALIDATE, "sample_events")

LOOKUP = "| lookup rule_metadata rule_id OUTPUT thresholds_hash | collect index=behavior_risk"


def _detections() -> dict:
    with open(SAVEDSEARCHES, encoding="utf-8") as handle:
        text = re.sub(r"\\\s*\r?\n\s*", " ", handle.read())

    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    parser.read_string(text)

    return {name.split(" - ")[0]: parser[name]["search"] for name in parser.sections() if name.startswith("R-")}


def _registry() -> list[dict]:
    with open(REGISTRY, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _registered() -> dict:
    by_rule = collections.defaultdict(set)

    for row in _registry():
        by_rule[row["rule_id"]].add((row["threshold"], row["operator"], float(row["value"])))

    return by_rule


def _tool():
    sys.path.insert(0, VALIDATE)

    try:
        import derive_thresholds  # pylint: disable=import-outside-toplevel,import-error
    finally:
        sys.path.remove(VALIDATE)

    return derive_thresholds


# --- Reading a search -----------------------------------------------------------------------------------------


def test_a_named_threshold_is_read_by_its_name():
    search = "... | eval rising_threshold = 4, sigma_threshold = 1.5 | where drift_rising_windows >= rising_threshold"

    assert extract_thresholds(search) == [("rising_threshold", "=", 4.0), ("sigma_threshold", "=", 1.5)]


def test_a_comparison_is_read_with_its_operand_and_operator():
    search = ("index=x http_distinct_paths>200 | where abs(optical_tx_dbm_deviation) < 0.5 "
              "AND (syn / flags) >= 0.9 AND optical_rx_dbm_deviation <= -1.0")

    assert extract_thresholds(search) == [("http_distinct_paths", ">", 200.0),
                                          ("abs(optical_tx_dbm_deviation)", "<", 0.5), ("(syn / flags)", ">=", 0.9),
                                          ("optical_rx_dbm_deviation", "<=", -1.0)]


def test_zero_is_the_condition_and_not_a_threshold():
    # `step > 0` says the count rose and `flags > 0` guards a division; neither is a number a baseline could move.
    assert extract_thresholds("| where dsts_per_src_step > 0 AND flags > 0 AND dsts_per_src > 50") == [
        ("dsts_per_src", ">", 50.0)
    ]


def test_a_threshold_used_twice_is_registered_once():
    search = "| where t_login - t_fanout <= 1800 | where t_process - t_fanout <= 1800"

    assert extract_thresholds(search) == [("t_fanout", "<=", 1800.0)]


# --- Deriving a threshold -------------------------------------------------------------------------------------


def test_a_rule_firing_above_is_proposed_from_the_top_of_its_baseline():
    values = list(range(1, 101))

    assert derive_threshold(values, "above", 0.99) == 99
    assert derive_threshold(values, "below", 0.99) == 1


def test_the_proposal_is_always_a_value_the_baseline_held():
    values = [0.5, 2.25, 7.0]

    assert derive_threshold(values, "above", 0.5) in values
    assert derive_threshold(values, "below", 0.5) in values


def test_a_gap_in_the_baseline_is_not_a_measurement():
    # A null, a missing field rendered as text and a boolean all reach this function from exported JSON; none of
    # them is a value of the field, and a `true` counted as 1 would place a threshold at one.
    assert derive_threshold([None, "", "n/a", True, float("nan"), 3, 5], "above", 0.99) == 5
    assert derive_threshold([None, True], "above", 0.99) is None
    assert derive_threshold([], "below", 0.5) is None


@pytest.mark.parametrize(("fires", "quantile"), [("over", 0.9), ("above", 0.0), ("above", 1.0)])
def test_a_meaningless_request_is_refused(fires: str, quantile: float):
    with pytest.raises(ValueError):
        derive_threshold([1, 2, 3], fires, quantile)


# --- The digest -----------------------------------------------------------------------------------------------


def test_the_digest_is_the_numbers_and_not_how_they_were_written():
    one = [("rising_threshold", "=", 4.0), ("sigma_threshold", "=", 1.5)]

    assert thresholds_hash(one) == thresholds_hash(list(reversed(one)))
    assert thresholds_hash(one) == thresholds_hash([("rising_threshold", "=", 4), ("sigma_threshold", "=", 1.5)])


def test_a_retune_changes_the_digest():
    assert thresholds_hash([("kmh_threshold", "=", 900.0)]) != thresholds_hash([("kmh_threshold", "=", 800.0)])
    assert thresholds_hash([("x", ">", 2.0)]) != thresholds_hash([("x", ">=", 2.0)])


def test_a_rule_with_no_threshold_has_a_digest_of_its_own():
    assert len(thresholds_hash([])) == 16
    assert thresholds_hash([]) != thresholds_hash([("x", ">", 1.0)])


# --- The app --------------------------------------------------------------------------------------------------


def test_the_registry_is_every_threshold_the_searches_compare_against():
    # The registry is what a reviewer reads and what the digest is computed from, so a stanza retuned without its
    # row -- or a row left behind when a threshold is removed -- fails here.
    detections = _detections()
    registered = _registered()

    assert len(detections) == 62
    assert set(registered) <= set(detections)

    for (rule_id, search) in detections.items():
        assert set(extract_thresholds(search)) == registered.get(rule_id, set()), rule_id


def test_the_check_notices_a_retuned_stanza():
    # Negative control for the test above: the same comparison over a search with one number changed fails.
    search = _detections()["R-D-L5-003"].replace("kmh_threshold = 900", "kmh_threshold = 800")

    assert set(extract_thresholds(search)) != _registered()["R-D-L5-003"]


def test_every_registry_row_says_what_kind_of_number_it_is():
    rows = _registry()

    assert len(rows) == 72

    for row in rows:
        label = f"{row['rule_id']} {row['threshold']}"

        assert row["kind"] in KINDS, label
        assert row["operator"] in OPERATORS, label
        assert row["fires"] in DIRECTIONS, label

        if (row["field"]):
            # Only a placeholder is proposed from a baseline; every other kind is a decision, not a measurement.
            assert row["kind"] == "placeholder", label
            assert row["sourcetype"].startswith("morpheus:score:") and row["telemetry_class"], label
            assert 0.5 < float(row["quantile"]) < 1.0, label
        else:
            assert not (row["sourcetype"] or row["telemetry_class"] or row["quantile"]), label

    # The guide names its conventional values; nothing else may claim to be one.
    conventional = {(row["rule_id"], row["threshold"]) for row in rows if row["kind"] == "conventional"}

    assert conventional == {("R-D-L5-003", "kmh_threshold"), ("R-C-005", "site_travel_kmh"),
                            ("R-D-L7-005", "http_4xx_in_window")}


def test_every_field_a_row_derives_from_is_carried_by_its_events():
    # A row naming a field its sourcetype and class never carry would be reported "no measurement" for ever, which
    # reads as a quiet estate rather than a wrong row.
    carried = collections.defaultdict(set)

    for name in os.listdir(SAMPLE_EVENTS):
        if (name.startswith("morpheus_score_")):
            sourcetype = name[:-len(".jsonlines")].replace("morpheus_score_", "morpheus:score:")

            with open(os.path.join(SAMPLE_EVENTS, name), encoding="utf-8") as handle:
                for line in handle:
                    event = json.loads(line)
                    carried[(sourcetype, event.get("telemetry_class"))].update(key for (key, value) in event.items()
                                                                               if value is not None)

    for row in _registry():
        if (row["field"]):
            assert row["field"] in carried[(row["sourcetype"], row["telemetry_class"])], row


def test_every_rule_s_digest_is_its_registered_thresholds():
    registered = _registered()

    with open(RULE_METADATA, encoding="utf-8", newline="") as handle:
        metadata = {row["rule_id"]: row for row in csv.DictReader(handle)}

    assert set(metadata) == set(_detections())

    for (rule_id, row) in metadata.items():
        assert row["thresholds_hash"] == thresholds_hash(registered.get(rule_id, set())), rule_id


def test_every_risk_record_carries_its_rule_s_digest():
    # The digest is only useful on the record of a notable: two notables fired under different thresholds then
    # differ in a field, and an analyst comparing them can see the configuration changed rather than the behaviour.
    for (rule_id, search) in _detections().items():
        assert re.sub(r"\s+", " ", search).rstrip().endswith(LOOKUP), rule_id


# --- The tool -------------------------------------------------------------------------------------------------


def test_the_tool_proposes_every_placeholder_it_has_a_field_for(tmp_path):
    tool = _tool()
    out = tmp_path / "proposals.json"

    assert tool.main(["--events", SAMPLE_EVENTS, "--out", str(out)]) == 0

    with open(out, encoding="utf-8") as handle:
        report = json.load(handle)["thresholds"]

    rows = _registry()

    assert [(entry["rule_id"], entry["threshold"]) for entry in report] == [(row["rule_id"], row["threshold"])
                                                                            for row in rows]

    for (entry, row) in zip(report, rows):
        if (row["field"]):
            assert entry["status"] == "proposed" and entry["samples"] > 0, entry
        else:
            assert entry["status"].startswith("not proposed"), entry
            assert "proposed" not in entry, entry

        assert entry["shipped"] == float(row["value"])


def test_the_baseline_period_bounds_what_the_tool_reads():
    # The sample events fall in January and February 1970; a period after them holds no measurement, and the
    # tool says so rather than proposing from nothing.
    tool = _tool()
    report = tool.propose(
        tool.read_registry(),
        SAMPLE_EVENTS,
        start=tool._day("2026-01-01"),  # pylint: disable=protected-access
        end=tool._day("2026-02-01"))  # pylint: disable=protected-access

    assert all(entry["status"] != "proposed" for entry in report)
    assert any("holds no measurement" in entry["status"] for entry in report)
