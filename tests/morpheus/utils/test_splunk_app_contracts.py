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
The numeric contracts shared between the pipeline, the Splunk app, and the design guide.

Four values have to agree across artifacts that no single test suite covers: the bucket width, the binding
retention, the Community ID seed, and the `event_time` rendering. Each of them fails silently when it drifts --
a lookup that matches nothing, a lookup that outlives or predeceases its index, flow hashes that no other tool
in the estate agrees with. Prose in three documents is not enforcement, so these tests read the shipped
configuration directly and compare it against the values in code.

The fourth contract, the `event_time` rendering, lives in `test_siem_wire.py` beside the renderer it constrains.

A fifth, the thirteen months Part 2 of the guide asks every scored class to be kept, sits here beside the binding
retention it is ordered against: the scored indexes keep 395 days, the score sourcetypes accept a backfill that old,
and the bindings that attribute those events outlive them.
"""

import inspect
import os
import re

import pytest

from _utils.numbers import from_words
from morpheus.stages.lineage.community_id_stage import CommunityIdStage
from morpheus.utils import splunk_conf
from morpheus.utils.binding_table import DEFAULT_BUCKET_SECONDS
from morpheus.utils.binding_table import DEFAULT_L1_BUCKET_SECONDS
from morpheus.utils.community_id import community_id
from morpheus.utils.community_id import community_id_series
from morpheus.utils.siem_sourcetypes import PRODUCED

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
APP_ROOT = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "TA-morpheus-lineage")
SAVEDSEARCHES_PATH = os.path.join(APP_ROOT, "default", "savedsearches.conf")
INDEXES_PATH = os.path.join(APP_ROOT, "default", "indexes.conf")
PROPS_PATH = os.path.join(APP_ROOT, "default", "props.conf")
TRANSFORMS_PATH = os.path.join(APP_ROOT, "default", "transforms.conf")
GUIDE_PATH = os.path.join(REPO_ROOT,
                          "docs",
                          "source",
                          "developer_guide",
                          "guides",
                          "11_predictive_behavioral_analytics_osi.md")

# `| where bucket >= floor((now() - 34560000) / 300)` -- the expiry job, carrying both contracts at once.
EXPIRY_PATTERN = re.compile(r"floor\(\(now\(\)\s*-\s*(\d+)\)\s*/\s*(\d+)\)")

# `| eval bucket=floor(_time/300)` -- the query-side discretization.
DISCRETIZE_PATTERN = re.compile(r"floor\(_time\s*/\s*(\d+)\)")

BINDING_INDEXES = ["behavior_bindings"]

SCORED_RETENTION_DAYS = 395
"""Thirteen months, what Part 2 asks of TC-1, TC-5, TC-6 and TC-7, and what the app keeps every scored event for."""

SCORED_INDEXES = [
    "behavior_events",
    "behavior_summary",
    "behavior_risk",
    "behavior_health",
    "behavior_quarantine",
    "behavior_late",
]
"""The scored events' index and every index derived from them: rollups, risk records, the pipeline's account of the
windows they were scored in, and the whole events held back for a clock or a seal."""


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def expiry_pairs(path: str) -> list[tuple[int, int]]:
    """Every `(retention_seconds, bucket_seconds)` pair appearing in an expiry expression."""
    return [(int(retention), int(bucket)) for (retention, bucket) in EXPIRY_PATTERN.findall(read_text(path))]


def test_the_patterns_still_find_something():
    # Guards every other test here: a stanza rewrite that breaks these patterns must fail loudly rather than
    # silently reduce the assertions below to vacuous truths over empty lists.
    assert len(expiry_pairs(SAVEDSEARCHES_PATH)) > 0
    assert len(expiry_pairs(GUIDE_PATH)) > 0
    assert len(DISCRETIZE_PATTERN.findall(read_text(GUIDE_PATH))) > 0


@pytest.mark.parametrize("path", [SAVEDSEARCHES_PATH, GUIDE_PATH], ids=["app", "guide"])
def test_bucket_width_matches_the_pipeline(path: str):
    # Contract 1. The pipeline expands bindings across buckets of a declared width; the SIEM rediscretizes event
    # times with the same divisor. Disagreement means every lookup misses, with no error.
    #
    # There are two widths, because there are two kinds of interval. Both are declared in code, and a divisor
    # appearing here that is neither of them is a number somebody typed.
    divisors = {bucket for (_, bucket) in expiry_pairs(path)}
    divisors.update(int(value) for value in DISCRETIZE_PATTERN.findall(read_text(path)))

    assert divisors == {DEFAULT_BUCKET_SECONDS, DEFAULT_L1_BUCKET_SECONDS}, f"{path} discretizes on {divisors}"


def test_each_expiry_job_uses_the_width_of_the_lookup_it_trims():
    # The half of contract 1 that a set comparison cannot see. With two widths in play, an expiry job that
    # divides by the other lookup's width still passes every assertion above and quietly trims the wrong rows:
    # at 300 against a day-bucketed collection the cutoff lands 288 times too high and empties the lookup.
    expected = {
        "binding_l2_l3": DEFAULT_BUCKET_SECONDS,
        "binding_l2": DEFAULT_BUCKET_SECONDS,
        "binding_l5_l2": DEFAULT_BUCKET_SECONDS,
        "binding_l1_history": DEFAULT_L1_BUCKET_SECONDS,
    }

    # The trim may not reach past another `inputlookup`: the adjacency refresh reads its own lookup to merge and
    # trims nothing, and a lazy match from there would name it after the next expiry job's width.
    trimmed = dict(
        re.findall(
            r"inputlookup\s+(?:\w+=\S+\s+)*(\S+)(?:(?!inputlookup).)*?floor\(\(now\(\)\s*-\s*\d+\)\s*/\s*(\d+)\)",
            read_text(SAVEDSEARCHES_PATH),
            re.DOTALL))

    assert {name: int(bucket) for (name, bucket) in trimmed.items()} == expected


def test_binding_retention_matches_between_index_and_expiry_job():
    # Contract 2. outputlookup never removes rows, so a separate job expires them. If the job's cutoff outlives
    # the index the lookup keeps rows whose events are gone; if it predeceases the index, events become
    # unattributable while still queryable. The README warns that these must move together; this enforces it.
    parser = splunk_conf.read(INDEXES_PATH)

    for index_name in BINDING_INDEXES:
        index_retention = int(parser[index_name]["frozenTimePeriodInSecs"])
        job_cutoffs = {retention for (retention, _) in expiry_pairs(SAVEDSEARCHES_PATH)}

        assert job_cutoffs == {index_retention}, (f"{index_name} retains {index_retention}s but the expiry job "
                                                  f"cuts at {job_cutoffs}")


def test_the_guide_and_the_app_agree_on_both_numbers():
    # The guide quotes the app's stanzas; when the two disagree the app is normative, but disagreement is still a
    # documentation defect and it has happened before on this branch.
    assert set(expiry_pairs(GUIDE_PATH)) == set(expiry_pairs(SAVEDSEARCHES_PATH))


def _retention_by_index() -> dict:
    parser = splunk_conf.read(INDEXES_PATH)

    return {name: int(parser[name]["frozenTimePeriodInSecs"]) for name in parser.sections()}


def _max_days_ago_by_sourcetype() -> dict:
    props = splunk_conf.read(PROPS_PATH)

    return {
        name: int(props[name]["MAX_DAYS_AGO"])
        for name in props.sections() if props.has_option(name, "MAX_DAYS_AGO")
    }


def test_every_scored_index_keeps_thirteen_months():
    # Contract 5 (gap G74). Ninety days was the app's figure until plan section 3.5, against a requirement of
    # thirteen months that the guide stated and the app contradicted: an optic investigated in month eleven had no
    # events left to explain its score.
    retention = _retention_by_index()

    short = {name: retention[name] for name in SCORED_INDEXES if retention[name] != SCORED_RETENTION_DAYS * 86400}

    assert short == {}, f"scored indexes not kept {SCORED_RETENTION_DAYS} days: {short}"

    # Every index is either scored, a binding or context store, or the derivable edges; a new one has to say which.
    assert set(retention) == set(SCORED_INDEXES) | set(BINDING_INDEXES) | {"behavior_context", "behavior_lineage"}


def test_bindings_outlive_the_events_they_attribute():
    # The asymmetry the index file calls load-bearing, now that the events' side moved: an event kept longer than
    # the binding that names its host is an event nobody can attribute.
    retention = _retention_by_index()

    for name in BINDING_INDEXES + ["behavior_context"]:
        assert retention[name] > max(retention[scored] for scored in SCORED_INDEXES), name

    assert retention["behavior_lineage"] < retention["behavior_events"]


def test_every_score_sourcetype_accepts_a_backfill_as_old_as_its_index_keeps():
    # The other half of G74: a backfill older than `MAX_DAYS_AGO` is not rejected but quietly re-timed, so a
    # retention the sourcetype cannot be replayed into is a retention on paper.
    max_days_ago = _max_days_ago_by_sourcetype()
    scored = {name: days for (name, days) in max_days_ago.items() if name.startswith("morpheus:score:")}

    assert set(scored) == {name for name in PRODUCED if name.startswith("morpheus:score:")}
    assert scored == {name: SCORED_RETENTION_DAYS for name in scored}
    assert max_days_ago["morpheus:late"] == SCORED_RETENTION_DAYS


def test_the_guide_quotes_the_app_s_retention():
    quoted = {
        name: int(seconds)
        for (name, seconds) in re.findall(r"^\[(behavior_\w+)\]\n(?:[^\[\n]*\n)*?frozenTimePeriodInSecs = (\d+)",
                                          read_text(GUIDE_PATH), re.MULTILINE)
    }
    retention = _retention_by_index()

    assert "behavior_events" in quoted
    assert quoted == {name: retention[name] for name in quoted}


def test_the_app_readme_states_the_number_of_searches_it_ships():
    # The README's inventory of savedsearches.conf listed eleven of the thirteen searches by name and had done
    # since before the layer 5 detections landed. A list of contents maintained by hand beside the thing it lists
    # drifts the moment the thing grows, and a reader counting stanzas gets a different answer than the document.
    shipped = len(splunk_conf.stanzas(SAVEDSEARCHES_PATH))

    readme = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "README.md")

    with open(readme, encoding="utf-8") as handle:
        text = re.sub(r"\s+", " ", handle.read())

    # `[\w-]+` rather than `\w+`: the counts go into the twenties, and "twenty-one" carries a hyphen that a
    # word-character class does not match, which would read as the README having stopped stating a number at all.
    match = re.search(r"savedsearches\.conf` \| Search heads \| ([\w-]+) searches:", text)

    assert match is not None, "the app README no longer states how many searches it ships"

    stated = from_words(match.group(1))

    assert stated == shipped, f"the README says {match.group(1)} searches; savedsearches.conf holds {shipped}"


def test_every_detection_search_is_named_somewhere_in_the_app_readme():
    # The other half of the same drift: the README named the four layer 2 detections as the things not yet run
    # against a live instance, and never gained the two layer 5 ones. What has and has not met a search head is
    # the most consequential sentence in that document, so every rule it could be wrong about is checked.
    rules = [name for name in splunk_conf.stanzas(SAVEDSEARCHES_PATH) if name.startswith("R-")]

    assert len(rules) >= 7, f"expected the detection searches to still be there, found {rules}"

    readme = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "README.md")

    with open(readme, encoding="utf-8") as handle:
        text = handle.read()

    for name in rules:
        rule_id = name.split(" - ")[0]

        assert rule_id in text, f"{rule_id} ships in the app and is not mentioned in its README"


def test_community_id_seed_defaults_to_zero():
    # Contract 3. The seed is part of the hash input, so a non-default value produces flow identifiers that no
    # other tool in the estate agrees with, forfeiting the entire reason the field exists.
    for callable_under_test in (community_id, community_id_series, CommunityIdStage.__init__):
        default = inspect.signature(callable_under_test).parameters["seed"].default

        assert default == 0, f"{callable_under_test.__qualname__} defaults its seed to {default}"


def test_kvstore_lookups_expose_the_key_they_are_written_by():
    # A KV Store lookup can only address a specific record when `_key` is in its fields_list. Without it the refresh
    # jobs' `key_field=_key` matches nothing, every overlapping window appends duplicates instead of overwriting,
    # and determinism control 11's idempotent sink does not hold at the SIEM boundary.
    transforms = splunk_conf.read(TRANSFORMS_PATH)

    for lookup in ("binding_l2_l3", "binding_l2", "binding_l5_l2", "binding_l1", "binding_l1_history"):
        fields = [field.strip() for field in transforms[lookup]["fields_list"].split(",")]

        assert "_key" in fields, f"{lookup} cannot be written by key without _key in fields_list"

    # Every job that writes a KV Store lookup writes it by key, the refresh jobs and the expiry jobs alike. The jobs
    # are the app's own binding lookup stanzas, every refresh and every expiry, each writing its lookup once.
    searches = splunk_conf.searches(SAVEDSEARCHES_PATH)
    jobs = sorted(name for name in searches if name.startswith("Binding lookup - "))
    writes = {name: re.findall(r"outputlookup\s+binding_\S+(.*)", searches[name]) for name in jobs}

    assert jobs, "the app ships no binding lookup job"
    assert sorted(name for (name, search) in searches.items() if re.search(r"outputlookup\s+binding_", search)) == jobs

    for (name, tails) in writes.items():
        assert len(tails) == 1, f"{name} writes its lookup {len(tails)} times"
        assert "key_field=_key" in tails[0], f"{name} writes its lookup without a key: {tails[0]!r}"


RULE_METADATA_PATH = os.path.join(APP_ROOT, "lookups", "rule_metadata.csv")


def _detections() -> dict:
    parsed = splunk_conf.read(SAVEDSEARCHES_PATH, preserve_case=True)

    return {name: parsed[name] for name in parsed.sections() if name.startswith("R-")}


def _rule_metadata() -> dict:
    import csv  # pylint: disable=import-outside-toplevel

    with open(RULE_METADATA_PATH, encoding="utf-8", newline="") as handle:
        return {row["rule_id"]: row for row in csv.DictReader(handle)}


def _offset_seconds(spec: str) -> int:
    if (spec.startswith("@")):
        return 0

    (amount, unit) = re.match(r"-(\d+)([mhdw])", spec).groups()

    return int(amount) * {"m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]


def _period_seconds(period: str) -> int:
    (amount, unit) = re.fullmatch(r"(\d+)([smhd])", period).groups()

    return int(amount) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def test_every_detection_writes_its_rows_to_the_risk_index():
    # The output loop, closed: a detection's rows were discarded when its job ended, so the chain and summary
    # searches summed a risk_score nothing stored. Every detection now ends by collecting what it returned.
    detections = _detections()

    # A floor rather than the count: adding a detection must not fail here for bookkeeping, and the collect count
    # below already equals the number of detections, whatever it is.
    assert len(detections) >= 58
    assert read_text(SAVEDSEARCHES_PATH).count("| collect index=behavior_risk") == len(detections)

    for (name, stanza) in detections.items():
        assert stanza["search"].rstrip().endswith("| collect index=behavior_risk"), name
        assert re.search(r'rule_id\s*=\s*"' + re.escape(name.split(" - ")[0]) + '"', stanza["search"]), name

    assert "[behavior_risk]" in read_text(INDEXES_PATH)


def test_every_detection_is_suppressed_on_its_stated_key_for_its_stated_period():
    detections = _detections()
    metadata = _rule_metadata()

    assert set(metadata) == {name.split(" - ")[0] for name in detections}

    for (name, stanza) in detections.items():
        row = metadata[name.split(" - ")[0]]

        assert row["saved_search"] == name
        assert stanza["alert.suppress"] == "1", name
        assert stanza["alert.digest_mode"] == "0", f"{name}: field suppression needs one alert per result"
        assert stanza["alert.suppress.fields"] == row["suppress_fields"], name
        assert stanza["alert.suppress.period"] == row["suppress_period"], name


def test_the_suppression_key_is_on_every_row_the_detection_returns():
    # A suppression field the rows do not carry suppresses on a null, which is every notable the rule ever raises
    # after the first, for as long as the period runs.
    for (name, stanza) in _detections().items():
        projected = set(re.findall(r"\|\s*table\s+([^|]*)", stanza["search"])[-1].split())

        for field in stanza["alert.suppress.fields"].split(","):
            assert field in projected, f"{name} suppresses on {field}, which its table does not carry"


def test_the_suppression_period_covers_every_run_that_can_see_the_same_row():
    # Overlapping windows return a row on every run that covers it; the period is the window, at least an hour,
    # rounded up to the hour, so the notable is raised once.
    for (name, stanza) in _detections().items():
        span = _offset_seconds(stanza["dispatch.earliest_time"]) - _offset_seconds(stanza["dispatch.latest_time"])
        period = _period_seconds(stanza["alert.suppress.period"])

        assert period >= max(span, 3600), name
        assert period % 3600 == 0 and period - max(span, 3600) < 3600, name


def test_no_rule_claims_hysteresis_it_does_not_have():
    # Control 9's other half waits on a model scoring near a threshold; until then every row says so.
    assert {row["hysteresis"] for row in _rule_metadata().values()} == {"none"}
    assert "[rule_metadata]" in read_text(TRANSFORMS_PATH)
