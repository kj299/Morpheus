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
The search-head validation package has to still be true when someone reads it.

An expectations file goes stale silently. Someone changes a corpus, the numbers in the document stop matching what
a deployment would see, and the next person to run the validation cannot tell a real disagreement from a document
nobody updated -- which is worse than having no expectations at all, because it looks authoritative.

So the counts are checked against what the pipeline actually produces, and the checked-in sample events against a
fresh generation. What cannot be checked here is Splunk: this asserts the expectation is honest, not that a search
head agrees with it.
"""

import collections
import datetime
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
VALIDATE = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "validate")
EXPECTED = os.path.join(VALIDATE, "expected_results.json")
EVENTS = os.path.join(VALIDATE, "sample_events")
GENERATOR = os.path.join(VALIDATE, "make_sample_events.py")
SAVEDSEARCHES = os.path.join(REPO_ROOT,
                             "examples",
                             "splunk_lineage_app",
                             "TA-morpheus-lineage",
                             "default",
                             "savedsearches.conf")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import session_pipeline as sp  # noqa: E402
import telemetry_pipeline as tp  # noqa: E402

GAP_THRESHOLD_NS = 60 * 10**9
TRAVEL_KMH_THRESHOLD = 900
MFA_CHALLENGE_THRESHOLD = 5
MFA_DENIAL_THRESHOLD = 4
"""The thresholds the saved searches state, repeated here so the predicate this file evaluates is the
predicate the app ships rather than an approximation of it."""


@pytest.fixture(name="expected", scope="module")
def expected_fixture() -> dict:
    with open(EXPECTED, encoding="utf-8") as handle:
        yield json.load(handle)


@pytest.fixture(name="telemetry", scope="module")
def telemetry_fixture() -> pd.DataFrame:
    yield tp.run_pipeline(tp.build_pipeline_config(), tp.build_corpus())


@pytest.fixture(name="sessions", scope="module")
def sessions_fixture() -> pd.DataFrame:
    yield sp.run_pipeline(sp.build_pipeline_config(), sp.build_corpus())


def _searches() -> set:
    import configparser
    import re

    with open(SAVEDSEARCHES, encoding="utf-8") as handle:
        folded = re.sub(r"\\\s*\r?\n\s*", " ", handle.read())

    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read_string(folded)

    return {name for name in parser.sections() if parser.has_option(name, "search")}


def test_every_shipped_search_has_a_written_expectation(expected: dict):
    # A search with no entry is one nobody has said what to expect from, which on this app means nobody can tell
    # its correct empty result from a broken one.
    assert set(expected["searches"]) == _searches()


def test_the_detections_return_exactly_what_is_written(expected: dict, telemetry: pd.DataFrame):
    scored = telemetry[telemetry["telemetry_class"].isin(["tc1", "tc2_mac", "tc2_arp", "tc2_auth"])]
    bindings = telemetry[telemetry["telemetry_class"] == "tc2_binding"]

    contested = scored[(scored["macs_claiming_sender_ip"].fillna(0) > 1) & (scored["arp_sender_ip_excluded"] == False)]  # noqa: E712  pylint: disable=singleton-comparison
    spoofs = bindings[bindings["bind_end_reason"].isin(["conflict", "displaced"])
                      & (bindings["bind_gap_ns"] <= GAP_THRESHOLD_NS)]
    bypasses = scored[scored["auth_unpaired"] == True]  # noqa: E712  pylint: disable=singleton-comparison

    searches = expected["searches"]

    # R-D-L2-003 aggregates by sender address, so what an analyst sees is one notable per contested address.
    assert searches["R-D-L2-003 - ARP anomaly"]["contributing_rows"] == len(contested)
    assert searches["R-D-L2-003 - ARP anomaly"]["expected_rows"] == contested["arp_sender_ip"].nunique()
    assert searches["R-D-L2-003 - ARP anomaly"]["key_values"]["arp_sender_ip"] in set(contested["arp_sender_ip"])

    assert searches["R-D-L2-004 - MAC in two places at once"]["expected_rows"] == len(spoofs)
    written = {(row["mac_address"], row["port_key"], row["bind_end_reason"])
               for row in searches["R-D-L2-004 - MAC in two places at once"]["key_values"]}
    assert written == set(zip(spoofs["mac_address"], spoofs["port_key"], spoofs["bind_end_reason"]))

    assert searches["R-D-L2-005 - Authorization without authentication"]["expected_rows"] == len(bypasses)
    named = {row["mac_address"] for row in searches["R-D-L2-005 - Authorization without authentication"]["key_values"]}
    assert named == set(bypasses["mac_address"])

    # R-D-L2-001 is gated by a lookup that ships header-only, so it must expect nothing and say how many rows are
    # waiting behind the gate. A non-zero expectation here would be a claim the app cannot keep.
    first_in_window = int((scored["macs_per_port_first_in_window"] == True).sum())  # noqa: E712  pylint: disable=singleton-comparison
    assert searches["R-D-L2-001 - MAC address count exceeded on an access port"]["expected_rows"] == 0
    assert searches["R-D-L2-001 - MAC address count exceeded on an access port"][
        "candidate_rows_before_the_lookup"] == first_in_window


def test_the_layer_5_detections_return_exactly_what_is_written(expected: dict, sessions: pd.DataFrame):
    # The same discipline as the layer 2 rules above, and for the same reason: an expectations file that goes
    # stale is worse than none, because it looks authoritative. Each predicate is evaluated here exactly as the
    # saved search states it.
    searches = expected["searches"]

    travel = sessions[(sessions["travel_status"] == "measured") & (sessions["travel_kmh"] >= TRAVEL_KMH_THRESHOLD)]
    fatigue = sessions[(sessions["mfa_denied_then_approved"] == True)  # noqa: E712  pylint: disable=singleton-comparison
                       & (sessions["mfa_attempts_in_window"] > MFA_CHALLENGE_THRESHOLD)
                       & (sessions["mfa_denials_in_window"] >= MFA_DENIAL_THRESHOLD)]

    assert searches["R-D-L5-003 - Impossible travel"]["expected_rows"] == len(travel)
    written = {(row["user_principal"], row["user_location"])
               for row in searches["R-D-L5-003 - Impossible travel"]["key_values"]}
    assert written == set(zip(travel["user_principal"], travel["user_location"]))

    assert searches["R-D-L5-004 - Multi-factor fatigue"]["expected_rows"] == len(fatigue)
    named = searches["R-D-L5-004 - Multi-factor fatigue"]["key_values"][0]
    assert named["user_principal"] == fatigue["user_principal"].iloc[0]
    assert named["mfa_denials_in_window"] == int(fatigue["mfa_denials_in_window"].iloc[0])


def _scored_events() -> list:
    # What a search head would hold for `sourcetype=morpheus:score:l*`. The sourcetype is the filename with the
    # colons swapped, which is how the generator writes them, so the glob here is the search's glob.
    events = []

    for name in sorted(os.listdir(EVENTS)):
        if (not name.startswith("morpheus_score_l") or not name.endswith(".jsonlines")):
            continue

        with open(os.path.join(EVENTS, name), encoding="utf-8") as handle:
            events.extend(json.loads(line) for line in handle if line.strip())

    return events


def _five_minute_bin(event_time: str) -> int:
    stamp = datetime.datetime.strptime(event_time.replace("UTC", ""), "%Y-%m-%dT%H:%M:%S.%f")
    seconds = int(stamp.replace(tzinfo=datetime.timezone.utc).timestamp())

    return seconds - (seconds % 300)


def test_the_behavior_summary_groups_exactly_what_is_written(expected: dict):
    # `stats ... by` drops a row that is missing any grouping field, so a missing envelope field does not show up
    # as an error anywhere -- it shows up as a smaller number, or as zero. That is why this is evaluated rather
    # than asserted: the row count is the only place the omission would surface.
    events = _scored_events()
    entry = expected["searches"]["Behavior summary - per-layer scores"]

    keyless = [event for event in events if event.get("osi_layer") is None or not event.get("entity_key")]
    assert keyless == [], f"{len(keyless)} scored events would be dropped by the grouping"

    groups = {(_five_minute_bin(event["event_time"]), event["osi_layer"], event["entity_key"], event["lineage_id"])
              for event in events}

    assert entry["contributing_rows"] == len(events)
    assert entry["expected_rows"] == len(groups)
    assert entry["expected_empty"] is False


def test_the_chain_assembly_blocker_is_the_lineage_and_not_the_envelope(expected: dict):
    # An expected-empty search is only honest while its stated reason is the reason. This one was blocked by a
    # missing `osi_layer`; it is now blocked by single-layer lineage. Asserting the new reason means the entry
    # cannot quietly keep claiming the old one after the corpus starts linking layers.
    entry = expected["searches"]["Chain assembly - cross-layer risk"]

    events = _scored_events()

    with open(os.path.join(EVENTS, "morpheus_edge.jsonlines"), encoding="utf-8") as handle:
        edges = [json.loads(line) for line in handle if line.strip()]

    layers = collections.defaultdict(set)

    for event in events + edges:
        if (event.get("lineage_id") is not None):
            layers[event["lineage_id"]].add(event.get("osi_layer"))

    spans = [len(seen - {None}) for seen in layers.values()]

    assert entry["distinct_lineage_ids"] == len(layers)
    assert entry["maximum_layer_span"] == max(spans)
    assert max(spans) < 3, "a chain now spans three layers, so this search is no longer correctly empty"


def test_every_expected_empty_search_says_why(expected: dict):
    empty = {name: entry for (name, entry) in expected["searches"].items() if entry.get("expected_empty")}

    # Four of thirteen. That ratio is the honest state of this app, and stating it is the package's main job. It
    # improved by two when the layer 5 rules landed with events to fire on, by one more when `TC1BindingStage`
    # gave `binding:l1` a producer, and by one again when `EnvelopeStampStage` put `osi_layer` and `entity_key`
    # on every record and the behavior summary finally had a grouping that keeps its rows.
    assert len(empty) == 4

    for (name, entry) in empty.items():
        assert entry["expected_rows"] == 0, name
        assert len(entry["why"]) > 60, f"{name}: an expected-empty search needs a reason, not a shrug"


def _event_files() -> dict:
    files = {}

    for name in sorted(os.listdir(EVENTS)):
        with open(os.path.join(EVENTS, name), encoding="utf-8") as handle:
            files[name] = handle.read()

    return files


def test_the_checked_in_events_are_what_the_pipeline_produces():
    # Regenerating must be a no-op. If it is not, what a SIEM would receive has changed and the diff belongs in a
    # pull request rather than being discovered on a search head.
    before = _event_files()
    completed = subprocess.run([sys.executable, GENERATOR],
                               capture_output=True,
                               text=True,
                               check=False,
                               timeout=900,
                               cwd=REPO_ROOT)

    assert completed.returncode == 0, completed.stderr[-3000:]

    after = _event_files()

    assert set(before) == set(after)

    for (name, content) in before.items():
        assert content == after[name], f"{name} changed; regenerate and review the diff"


def test_the_conformance_runner_refuses_without_a_device(tmp_path):
    # The runner's own guard, asserted here because the failure it prevents is a green run that checked nothing.
    #
    # The device is hidden rather than assumed absent. This test used to call the runner bare, which was harmless
    # only while it was deselected on the machines that have a card: once it ran there, the runner did not
    # refuse -- it ran the entire suite, from inside a test the suite was running, until the timeout. Stubbing
    # nvidia-smi to report nothing exercises the guard identically on either machine and cannot recurse, because
    # the script stops at the device check.
    runner = os.path.join(REPO_ROOT, "ci", "scripts", "gpu_conformance.sh")

    assert os.access(runner, os.X_OK), "the runner must be executable or the one command is not one command"

    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    stub = stub_bin / "nvidia-smi"
    stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    stub.chmod(0o755)

    environment = dict(os.environ, PATH=f"{stub_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    artifact = str(tmp_path / "gpu_conformance_probe.json")

    completed = subprocess.run([runner, artifact],
                               capture_output=True,
                               text=True,
                               check=False,
                               timeout=300,
                               cwd=REPO_ROOT,
                               env=environment)

    assert completed.returncode != 0, "a machine with no visible GPU must not report a passing GPU verdict"
    assert "no CUDA device" in completed.stdout + completed.stderr

    # And the artifact, because a refusal that leaves no file behind reads afterwards as a run never started.
    with open(artifact, encoding="utf-8") as handle:
        report = json.load(handle)

    assert report["verdict"] == "failed"
    assert "no CUDA device" in report["reason"]
