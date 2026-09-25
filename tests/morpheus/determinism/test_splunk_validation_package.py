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
VALIDATION = os.path.join(VALIDATE, "VALIDATION.md")
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
DRIFT_RISING_THRESHOLD = 4
DRIFT_SIGMA_THRESHOLD = 1.5
DRIFT_MEAN_CEILING = 2.0
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


def test_the_drift_rule_returns_exactly_what_is_written(expected: dict, sessions: pd.DataFrame):
    # R-P-L5-006 as the search states it, then deduplicated the way the search deduplicates. The entry names
    # every principal-day, because a rule that fires on reference arithmetic is one whose every firing has to be
    # accounted for -- a new one appearing unexplained is exactly the drift in meaning this file exists to catch.
    auth = sessions[sessions["telemetry_class"] == "tc5_auth"]
    entry = expected["searches"]["R-P-L5-006 - Drift trajectory"]

    fires = auth[(auth["drift_mature"] == True)  # noqa: E712  pylint: disable=singleton-comparison
                 & (auth["drift_rising_windows"].astype(float) >= DRIFT_RISING_THRESHOLD)
                 & (auth["drift_rise_sigmas"].astype(float) > DRIFT_SIGMA_THRESHOLD)
                 & (auth["mean_abs_z"].astype(float) < DRIFT_MEAN_CEILING)]
    deduplicated = fires.drop_duplicates(["user_principal", "day_window_id"])

    assert entry["contributing_rows"] == len(fires)
    assert entry["expected_rows"] == len(deduplicated)

    written = {(row["user_principal"], row["day_window_id"]) for row in entry["key_values"]}
    assert written == set(zip(deduplicated["user_principal"], deduplicated["day_window_id"].astype(int)))


SCAN_SYN_RATIO = 0.9
SCAN_PORTS = 50
REFUSAL_RATIO = 0.5
REFUSAL_FAN_OUT = 10
"""The layer 4 searches' own thresholds, repeated here so the predicate this file evaluates is the predicate the
app ships rather than an approximation of it."""


def _layer_4_events() -> list:
    """What a search head would hold for `sourcetype=morpheus:score:l4`."""
    with open(os.path.join(EVENTS, "morpheus_score_l4.jsonlines"), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _layer_4_bins(events: list) -> dict:
    """One entry per flow per bin, aggregated the way the layer 4 searches aggregate.

    `max` over the counts, because they are running totals and rise monotonically through a bin. Never `max` over
    a ratio column: a running ratio is not monotone, and an ordinary handshake opens with a bare SYN, so
    `max(flow_syn_ratio)` over it is the scan's figure exactly. The searches divide the maxima and so does this.
    """
    bins: dict = {}

    for event in events:
        key = (event["flow_id"], event["rollup_time_ns"], event["src_ip"], event["dst_ip"], event["dst_port"])
        state = bins.setdefault(key, {"syn": 0, "ack": 0, "rst": 0, "all": 0})

        for (name, column) in (("syn", "flow_syn"), ("ack", "flow_ack"), ("rst", "flow_rst"), ("all", "flow_all")):
            state[name] = max(state[name], event[column] or 0)

    return bins


def test_the_layer_4_detections_return_exactly_what_is_written(expected: dict):
    # The three layer 4 rules evaluated over the events the app is actually fed, rather than over the frame the
    # pipeline produced. Everything between the two -- the wire rendering, the boolean spellings, the columns
    # SiemWireStage carries -- is where a number in this file could quietly stop being the number a search head
    # would return, and it is the half the harness cannot see.
    events = _layer_4_events()
    bins = _layer_4_bins(events)
    searches = expected["searches"]

    # R-D-L4-002: unanswered SYNs, then the breadth. Both halves, in the order the search applies them.
    unanswered = {
        key: state
        for (key, state) in bins.items()
        if state["all"] > 0 and state["ack"] == 0 and (state["syn"] / state["all"]) >= SCAN_SYN_RATIO
    }
    reach: dict = collections.defaultdict(set)
    hosts: dict = collections.defaultdict(set)

    for (_, rollup, src_ip, dst_ip, dst_port) in unanswered:
        reach[(src_ip, rollup)].add(dst_port)
        hosts[(src_ip, rollup)].add(dst_ip)

    scans = {key: ports for (key, ports) in reach.items() if len(ports) > SCAN_PORTS}
    entry = searches["R-D-L4-002 - SYN without completion"]

    assert entry["candidate_flows_before_the_breadth_filter"] == len(unanswered)
    assert entry["contributing_rows"] == len(next(iter(scans.values())))
    assert entry["expected_rows"] == len(scans)
    assert entry["key_values"]["src_ip"] == next(iter(scans))[0]
    assert entry["key_values"]["destination_ports"] == len(next(iter(scans.values())))
    assert entry["key_values"]["destination_hosts"] == len(hosts[next(iter(scans))])

    # R-D-L4-003: one predicate, and then which side of it fans out.
    refusing = [
        key for (key, state) in bins.items() if state["all"] > 0 and (state["rst"] / state["all"]) >= REFUSAL_RATIO
    ]
    refused_by: dict = collections.defaultdict(set)
    refusing_to: dict = collections.defaultdict(set)

    for (_, rollup, src_ip, dst_ip, _port) in refusing:
        refusing_to[(src_ip, rollup)].add(dst_ip)
        refused_by[(dst_ip, rollup)].add(src_ip)

    notables: dict = {}

    for (_, rollup, src_ip, dst_ip, _port) in refusing:
        clients = len(refusing_to[(src_ip, rollup)])
        servers = len(refused_by[(dst_ip, rollup)])

        if (clients >= REFUSAL_FAN_OUT and servers == 1):
            notables[(src_ip, "service_outage")] = clients
        elif (servers >= REFUSAL_FAN_OUT and clients == 1):
            notables[(dst_ip, "closed_port_enumeration")] = servers

    entry = searches["R-D-L4-003 - RST ratio"]

    assert entry["contributing_rows"] == len(refusing)
    assert entry["expected_rows"] == len(notables)
    assert {
        (row["entity_key"], row["refusal_direction"]): row["counterparties"]
        for row in entry["key_values"]
    } == notables

    # R-B-L4-005: either envelope, because a bulk transfer arriving as ordinary-sized packets moves one and not
    # the other, and that is the shape requiring both would miss.
    breached = [
        event for event in events
        if event.get("flow_data_len_envelope_breached") is True or event.get("flow_bpp_envelope_breached") is True
    ]
    entry = searches["R-B-L4-005 - Transfer envelope breach"]

    assert entry["contributing_rows"] == len(breached)
    assert entry["expected_rows"] == len({event["transfer_triple"] for event in breached})
    assert entry["key_values"]["transfer_triple"] == breached[0]["transfer_triple"]
    assert entry["key_values"]["transferred"] == breached[0]["flow_data_len"]
    assert entry["key_values"]["volume_envelope"] == breached[0]["flow_data_len_envelope"]
    assert entry["key_values"]["volume_ratio"] == breached[0]["flow_data_len_envelope_ratio"]


TUNNEL_ENTROPY = 4.0
TUNNEL_LABEL_LENGTH = 30
TUNNEL_SUBDOMAINS = 100
ENUMERATION_PATHS = 200
ENUMERATION_RATIO = 0.7
"""The layer 7 searches' own thresholds, repeated here so the predicates this file evaluates are the ones the
app ships."""


def _layer_7_events() -> list:
    """What a search head would hold for `sourcetype=morpheus:score:l7`."""
    with open(os.path.join(EVENTS, "morpheus_score_l7.jsonlines"), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_the_layer_7_detections_return_exactly_what_is_written(expected: dict):
    # Evaluated over the events the app is fed rather than the frame the pipeline produced. For R-B-L7-001 this is
    # not a formality: the search computes its own count, of distinct subdomains among queries clearing the other
    # two conditions, which is a different figure from the stage's count of every subdomain. Only the search's
    # version holds all three conditions to the same traffic, and this is the place it is computed as the search
    # computes it.
    events = _layer_7_events()
    searches = expected["searches"]

    qualifying = [
        event for event in events if (event.get("dns_subdomain_entropy") or 0) > TUNNEL_ENTROPY and (
            event.get("dns_mean_label_length") or 0) > TUNNEL_LABEL_LENGTH
    ]
    subdomains: dict = collections.defaultdict(set)
    clients: dict = collections.defaultdict(set)

    for event in qualifying:
        subdomains[event["dns_registered_domain"]].add(event["dns_subdomain"])
        clients[event["dns_registered_domain"]].add(event["src_ip"])

    tunnels = {domain: names for (domain, names) in subdomains.items() if len(names) > TUNNEL_SUBDOMAINS}
    entry = searches["R-B-L7-001 - DNS tunneling"]

    assert entry["candidate_rows_clearing_entropy_and_length"] == len(qualifying)
    assert entry["expected_rows"] == len(tunnels)
    assert entry["key_values"]["dns_registered_domain"] in tunnels

    domain = entry["key_values"]["dns_registered_domain"]

    assert entry["key_values"]["qualifying_subdomains"] == len(tunnels[domain])
    assert entry["contributing_rows"] == sum(1 for event in qualifying if event["dns_registered_domain"] == domain)
    assert entry["key_values"]["clients"] == len(clients[domain])

    # R-D-L7-005, read off the same row the way the search reads it, the ratio as a multiplication.
    enumerating = [
        event for event in events
        if (event.get("http_distinct_paths") or 0) > ENUMERATION_PATHS and event.get("http_4xx_in_window") is not None
        and event["http_4xx_in_window"] > ENUMERATION_RATIO * event["http_2xx_in_window"]
    ]
    by_client: dict = collections.defaultdict(list)

    for event in enumerating:
        by_client[event["src_ip"]].append(event)

    entry = searches["R-D-L7-005 - Enumeration"]

    assert entry["contributing_rows"] == len(enumerating)
    assert entry["expected_rows"] == len(by_client)

    written = {row["src_ip"]: (row["distinct_paths"], row["refusals"], row["successes"]) for row in entry["key_values"]}
    derived = {
        client: (max(event["http_distinct_paths"] for event in rows),
                 max(event["http_4xx_in_window"] for event in rows),
                 max(event["http_2xx_in_window"] for event in rows))
        for (client, rows) in by_client.items()
    }

    assert written == derived


def test_the_saas_detections_return_exactly_what_is_written(expected: dict):
    # Evaluated over the events the app is fed, the way the two searches read them. R-B-L7-002 reads one row and
    # takes its severity from the classification the enrichment attached; R-P-L7-006 reads every week of a
    # principal at once -- here the whole corpus, which gives the same answer as the four-week dispatch window
    # because every rise in it fits inside four weeks.
    events = _layer_7_events()
    searches = expected["searches"]
    severity = {"restricted": 75, "confidential": 60, "internal": 40, "public": 20}

    bulk = [
        event for event in events
        if event.get("saas_baseline_mature") is True and (event.get("saas_record_ratio") or 0) > 5
    ]
    derived = sorted(({
        "user_principal": event["user_principal"],
        "classification": event.get("ctx_object_data_classification") or "unclassified",
        "risk_score": severity.get(event.get("ctx_object_data_classification"), 40),
    } for event in bulk),
                     key=lambda row: row["user_principal"])
    entry = searches["R-B-L7-002 - Bulk data access"]

    assert entry["contributing_rows"] == len(bulk)
    assert entry["expected_rows"] == len(derived)
    assert entry["key_values"] == derived

    rising: dict = collections.defaultdict(int)
    roles: dict = collections.defaultdict(set)

    for event in events:
        if (event.get("saas_object_types_in_week") is None):
            continue

        principal = event["user_principal"]
        rising[principal] = max(rising[principal], event.get("drift_rising_windows") or 0)
        roles[principal].add(event.get("ctx_groups") or "none")

    watchlisted = sorted(({
        "user_principal": principal, "rising_weeks": rising[principal], "role": next(iter(roles[principal]))
    } for principal in rising if rising[principal] >= 4 and len(roles[principal]) == 1),
                         key=lambda row: row["user_principal"])
    entry = searches["R-P-L7-006 - Access breadth trajectory"]

    assert entry["expected_rows"] == len(watchlisted)
    assert entry["key_values"] == watchlisted


def test_the_endpoint_detection_returns_exactly_what_is_written(expected: dict):
    # Evaluated over the events the app is fed, the way the search reads them: one row at a time, the stage's
    # novelty flag the trigger and the integrity level the severity, with no level at the default.
    severity = {"system": 70, "high": 55, "medium": 40, "low": 25}
    novel = [event for event in _layer_7_events() if event.get("endpoint_pair_novel") is True]
    derived = sorted(({
        "hostname": event["hostname"],
        "integrity": event.get("endpoint_integrity") or "unknown",
        "endpoint_host_only": event["endpoint_host_only"],
        "risk_score": severity.get(event.get("endpoint_integrity"), 40),
    } for event in novel),
                     key=lambda row: row["hostname"])
    entry = expected["searches"]["R-B-L7-004 - Process ancestry novelty"]

    assert entry["contributing_rows"] == len(novel)
    assert entry["expected_rows"] == len(derived)
    assert entry["key_values"] == derived


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


def test_the_chain_assembly_blocker_is_the_risk_and_not_the_span(expected: dict):
    # An expected-empty search is only honest while its stated reason is the reason, and this one has now been
    # empty for three different reasons. It was a missing `osi_layer`, then single-layer lineage, and neither
    # holds any more: the estate pipeline carries a principal down to the port they sat at, so chains reach three
    # layers. What is left is the risk threshold on the next line, which no pipeline event can satisfy because
    # `risk_score` is written by the detection searches rather than by a stage. Both halves are asserted, so the
    # entry cannot go on claiming a blocker it has outgrown, and cannot claim this one after risk starts arriving.
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
    assert entry["two_layer_chains"] == spans.count(2)
    assert entry["three_layer_chains"] == sum(1 for span in spans if span >= 3)

    assert max(spans) >= 3, "the search's own threshold is dc(osi_layer) >= 3, and nothing reaches it"
    assert max(spans) < 4, "a span of four would fire through the peak_z branch, so empty would be wrong"

    scored_with_risk = [event for event in events if event.get("risk_score") is not None]

    assert scored_with_risk == [], "risk_score is now on pipeline events, so the stated blocker is gone"


def test_the_validation_document_says_the_same_thing_the_expectation_file_does(expected: dict):
    # VALIDATION.md is the page a deployment actually reads; `expected_results.json` is the page the tests read.
    # Two documents describing one run, maintained by hand, is the shape that has drifted in five places in this
    # repository -- and it drifted here: the summary table went a whole increment without the layer 3 searches
    # in it, still saying "four of the fourteen" over a row count from two increments earlier. Every number in
    # that table now comes from the same file these assertions do.
    import re  # pylint: disable=import-outside-toplevel

    with open(VALIDATION, encoding="utf-8") as handle:
        document = handle.read()

    rows = dict(re.findall(r"^\| ([^|]+?) \| \*\*(\d+)\*\* \|", document, re.MULTILINE))

    assert len(rows) == len(expected["searches"]), (f"the summary table lists {len(rows)} searches and the "
                                                    f"expectation file holds {len(expected['searches'])}")

    # The table's labels are shortened for a reader, so a row matches the entry it names outright where the two
    # agree and by rule identifier where the prose was cut down. A label matching nothing, or more than one, is
    # a row about a search that no longer exists under that name.
    def named(label: str) -> list:
        wanted = label.replace(" - ", ", ").lower()
        exact = [name for name in expected["searches"] if name.replace(" - ", ", ").lower() == wanted]

        return exact or [name for name in expected["searches"] if name.startswith(f"{label.split(',')[0]} -")]

    for (label, stated) in rows.items():
        entries = named(label)

        assert len(entries) == 1, f"{label!r} in VALIDATION.md matches {entries} in the expectation file"
        assert int(stated) == expected["searches"][entries[0]]["expected_rows"], (
            f"VALIDATION.md says {label} returns {stated}; the expectation file says "
            f"{expected['searches'][entries[0]]['expected_rows']}")

    # The notes quote figures of their own, and checking only the Rows column let two of them go stale through a
    # whole increment: the summary note said 2586 scored events and the chain note 658 chains, both from layer 4,
    # while the rows beside them had been kept current. The two aggregates are the ones whose notes carry a figure
    # the expectation file also holds, so those figures are checked too.
    summary = expected["searches"]["Behavior summary - per-layer scores"]
    chain = expected["searches"]["Chain assembly - cross-layer risk"]

    summary_note = re.search(r"^\| Behavior summary[^|]*\|[^|]*\| (.+?) \|$", document, re.MULTILINE).group(1)
    chain_note = re.search(r"^\| Chain assembly[^|]*\|[^|]*\| (.+?) \|$", document, re.MULTILINE).group(1)

    assert f"over the {summary['contributing_rows']} scored events" in summary_note, summary_note[:120]
    assert f"of the {chain['distinct_lineage_ids']} chains" in chain_note, chain_note[:160]

    empty = sum(1 for entry in expected["searches"].values() if entry.get("expected_empty"))
    stated_empty = re.search(r"\*\*(\w+) of the ([\w-]+) should return nothing\.\*\*", document)

    assert stated_empty is not None, "VALIDATION.md no longer states how many searches return nothing"
    assert NUMBER_WORDS[stated_empty.group(1).lower()] == empty
    assert NUMBER_WORDS[stated_empty.group(2).lower()] == len(expected["searches"])


NUMBER_WORDS = {
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twenty-one": 21,
    "twenty-two": 22,
    "twenty-three": 23,
    "twenty-four": 24,
    "twenty-five": 25,
    "twenty-six": 26,
    "twenty-seven": 27,
    "twenty-eight": 28,
    "twenty-nine": 29,
    "thirty": 30,
    "thirty-one": 31,
    "thirty-two": 32,
    "thirty-three": 33,
    "thirty-four": 34,
    "thirty-five": 35,
}
"""Only the range these two counts can plausibly take. A word outside it fails with a `KeyError` naming the word,
which is the right failure: the document said something nobody here anticipated."""


def test_every_expected_empty_search_says_why(expected: dict):
    empty = {name: entry for (name, entry) in expected["searches"].items() if entry.get("expected_empty")}

    # Seven of thirty-four. That ratio is the honest state of this app, and stating it is the package's main job.
    # The seventh is R-P-L3-005, which reads the behavior summary this package does not populate -- the same
    # deployment-step blocker the chain assembly search has, arriving with layer 3 rather than being discovered.
    # It
    # improved by two when the layer 5 rules landed with events to fire on, by one more when `TC1BindingStage`
    # gave `binding:l1` a producer, and by one again when `EnvelopeStampStage` put `osi_layer` and `entity_key`
    # on every record and the behavior summary finally had a grouping that keeps its rows.
    #
    # Then it went the other way twice, which is what this number is for. The layer 1 history expiry is empty for
    # the same benign reason its layer 2 twin is: nothing in a fresh corpus is old enough to expire. The L2/L3
    # refresh is empty because it always was and the document said otherwise -- it selects
    # `binding_table=dhcp_lease`, this corpus has no DHCP source, and the 80 bucketed rows it was credited with
    # are a MAC table under a different name. A count that only ever improves is a count nobody is checking.
    assert len(empty) == 7

    for (name, entry) in empty.items():
        assert entry["expected_rows"] == 0, name
        assert len(entry["why"]) > 60, f"{name}: an expected-empty search needs a reason, not a shrug"


def test_the_port_history_holds_the_optic_that_was_replaced_and_nothing_else(telemetry: pd.DataFrame):
    # The layer 1 lookup used to answer every question in the present tense: keyed on the port alone, a port whose
    # optic is replaced collapses to one row and an investigation into last Tuesday gets Wednesday's optic. The
    # history lookup is the other tense, and its value is as much in what it leaves out as in what it holds.
    ports = tp.build_port_binding_table(telemetry[telemetry["telemetry_class"] == "tc1_binding"])
    swapped = f"{tp.SITE}:{tp.SWITCH}:{tp.XCVR_SWAP_PORT}"

    superseded = ports.superseded()

    # One interval, on the one port anybody touched. The other three ports are described for all time by the row
    # the current-state lookup holds, and cost the history collection nothing.
    assert superseded.size == 1
    assert superseded.key_count == 1

    before = (tp.XCVR_SWAP_AT_MINUTE - 10) * 60 * tp.NS_PER_SECOND
    last = tp.CORPUS_SECONDS * tp.NS_PER_SECOND - 1
    optic = tp.PORT_INVENTORY_COLUMNS.index("transceiver_serial")

    # The two tenses disagree, which is the whole point. Resolving the port before the swap through the history
    # gives the optic that was in it; the current-state answer for the same port is the one in it now.
    assert superseded.resolve(swapped, before).values[optic] == f"XCVR-{tp.SWITCH}-{tp.XCVR_SWAP_PORT}"
    assert ports.resolve(swapped, last).values[optic] == f"XCVR-{tp.SWITCH}-{tp.XCVR_SWAP_PORT}-B"


def test_the_history_the_app_receives_is_one_day_bucket_on_one_port():
    # What the search head is actually fed, rather than what the pipeline could produce. The rows ride the shared
    # bucketed sourcetype and the refresh tells them apart by `binding_table`, so a rename upstream would leave
    # the L1 history refresh reading the layer 2 bindings, which carry neither a port nor an optic.
    with open(os.path.join(EVENTS, "binding_bucketed.jsonlines"), encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]

    history = [row for row in rows if row["binding_table"] == "port_inventory"]

    assert len(history) == 1
    assert history[0]["port_id"] == tp.XCVR_SWAP_PORT
    assert history[0]["switch_id"] == tp.SWITCH
    assert history[0]["transceiver_serial"] == f"XCVR-{tp.SWITCH}-{tp.XCVR_SWAP_PORT}"

    # The cost of the approximation, stated rather than discovered. The corpus is one hour and the bucket is a
    # day, so both intervals fall in bucket zero and every instant in it resolves to the earlier optic -- right
    # for the fifty minutes before the swap, wrong for the ten after. That error is bounded by the bucket width,
    # where the unbucketed lookup's error was bounded by nothing at all.
    assert history[0]["bucket"] == 0

    # The history is additive. The interval stream the current-state refresh reads still carries both of the
    # swapped port's intervals, so the two lookups are built from the same records rather than one taking rows
    # away from the other.
    with open(os.path.join(EVENTS, "binding_l1.jsonlines"), encoding="utf-8") as handle:
        intervals = [json.loads(line) for line in handle]

    on_swapped = [row["transceiver_serial"] for row in intervals if row["port_id"] == tp.XCVR_SWAP_PORT]

    assert sorted(on_swapped) == [f"XCVR-{tp.SWITCH}-{tp.XCVR_SWAP_PORT}", f"XCVR-{tp.SWITCH}-{tp.XCVR_SWAP_PORT}-B"]


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
