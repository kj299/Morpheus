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
The shipped detections, applied in Python to the harness corpus, and checked against the Splunk app's own stanzas.

A saved search cannot run here. What can run is the predicate each search encodes, over the same columns, on the
corpus with the anomalies planted in it. Each must fire exactly where it was planted and nowhere else, with the
fields an analyst needs to trace it. The stanzas are then read from the app itself, so the SPL and the Python
cannot drift apart silently.

The two layer 1 rules and the five layer 2 rules are asserted end to end here, corpus and stanza both. The two layer
5 rules are asserted against their own corpus in `test_session_harness.py` and against their expected row counts in
`test_splunk_validation_package.py`, because both of those already hold the layer 5 pipeline; what they are
checked for here is the half neither of those covers -- that the SPL names the columns the stages emit, and that
the scheduling follows Part 5's discipline.
"""

import os
import re
import sys

import pandas as pd
import pytest

from morpheus.utils.binding_closer import CONFLICT
from morpheus.utils.binding_closer import DISPLACED
from morpheus.utils.optical_forecast import DEFAULT_MIN_SAMPLES
from morpheus.utils.optical_forecast import STATUS_IMMATURE
from morpheus.utils.optical_forecast import STATUS_NONLINEAR
from morpheus.utils.optical_forecast import STATUS_NOT_DEGRADING
from morpheus.utils.optical_forecast import STATUS_PROJECTED

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import telemetry_pipeline as tp  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
APP_DEFAULT = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "TA-morpheus-lineage", "default")
SAVED_SEARCHES = os.path.join(APP_DEFAULT, "savedsearches.conf")
PROPS = os.path.join(APP_DEFAULT, "props.conf")

NS = tp.NS_PER_SECOND

RULES = {
    "R-D-L1-001": "R-D-L1-001 - Transceiver substitution",
    "R-P-L1-004": "R-P-L1-004 - Optical degradation forecast",
    "R-B-L2-002": "R-B-L2-002 - Port-to-MAC binding novelty",
    "R-D-L2-001": "R-D-L2-001 - MAC address count exceeded on an access port",
    "R-D-L2-003": "R-D-L2-003 - ARP anomaly",
    "R-D-L2-004": "R-D-L2-004 - MAC in two places at once",
    "R-D-L2-005": "R-D-L2-005 - Authorization without authentication",
    "R-D-L5-003": "R-D-L5-003 - Impossible travel",
    "R-D-L5-004": "R-D-L5-004 - Multi-factor fatigue",
    "R-D-L5-007": "R-D-L5-007 - Off-hours authentication",
    "R-D-L5-008": "R-D-L5-008 - New authentication location or device",
    "R-D-L5-009": "R-D-L5-009 - Failed authentication run ending in success",
    "R-B-L5-001": "R-B-L5-001 - Composite authentication anomaly",
    "R-B-L5-002": "R-B-L5-002 - Location novelty anomaly",
    "R-P-L5-006": "R-P-L5-006 - Drift trajectory",
    "R-B-L7-002": "R-B-L7-002 - Bulk data access",
    "R-P-L7-006": "R-P-L7-006 - Access breadth trajectory",
    "R-C-005": "R-C-005 - Credential replay across the stack",
}
"""The rules whose stanzas this file reads back: the layer 1 and 2 rules whose predicates it asserts over the
telemetry corpus, and the layer 5, 7 and chained rules whose predicates live in the session, SaaS and campaign
harnesses but whose SPL had no reader here until the retrospective's steps 2 and 4."""


def _gap_threshold_ns() -> int:
    """The threshold R-D-L2-004 ships with, read from the search itself.

    Hard-coding it here would let the rule be tuned in the app while the test went on asserting the old value, and
    the corpus is planted either side of it deliberately: one spoof inside, one legitimate move outside.
    """
    with open(SAVED_SEARCHES, encoding="utf-8") as handle:
        return int(re.search(r"gap_threshold\s*=\s*(\d+)", handle.read()).group(1))


GAP_THRESHOLD_NS = _gap_threshold_ns()

LOOKUP = os.path.join(REPO_ROOT,
                      "examples",
                      "splunk_lineage_app",
                      "TA-morpheus-lineage",
                      "lookups",
                      "port_designations.csv")


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
def test_r_d_l1_001_fires_on_the_swap_the_link_never_dropped_for_and_not_on_the_other(result: pd.DataFrame):
    # The rule's predicate: a serial that differs from the port's previous poll, on a poll the flap count says the
    # link never moved for. Two optics are replaced in this hour and the serial changes once on each port; what
    # separates them is whether the device recorded the link dropping in between.
    layer_1 = result[result["telemetry_class"] == "tc1"].sort_values(["entity_key", "event_time"])
    changed = layer_1[layer_1["transceiver_serial_changed"] == True]  # noqa: E712  pylint: disable=singleton-comparison

    swapped_port = f"{tp.SITE}:{tp.SWITCH}:{tp.XCVR_SWAP_PORT}"
    maintained_port = f"{tp.SITE}:{tp.SWITCH}:{tp.MAINTENANCE_PORT}"

    assert set(changed["entity_key"]) == {swapped_port, maintained_port}
    assert len(changed) == 2

    detections = changed[(changed["link_flaps"] == 0) & (changed["oper_status"] != "down")]

    assert len(detections) == 1, "the swap nothing polled a transition for, and not the maintenance swap"

    hit = detections.iloc[0]
    assert hit["entity_key"] == swapped_port
    assert hit["event_time"] == tp.XCVR_SWAP_AT_MINUTE * 60 * NS
    assert hit["transceiver_serial"] == f"XCVR-{tp.SWITCH}-{tp.XCVR_SWAP_PORT}-B"
    assert hit["transceiver_serial_first_seen"] == True  # noqa: E712  pylint: disable=singleton-comparison
    assert hit["transceiver_serial_distinct_count"] == 2

    # The search carries the serial from the port's preceding poll onto the notable, so it names both optics.
    # The same shift, in pandas: the previous row of the same port.
    previous = layer_1.groupby("entity_key")["transceiver_serial"].shift(1)
    assert previous.loc[hit.name] == f"XCVR-{tp.SWITCH}-{tp.XCVR_SWAP_PORT}"

    # The negative control. The serial changed just the same, but the device's own record of the link says it
    # dropped and came back between the two polls -- which is what replacing an optic does to a link.
    maintained = changed[changed["entity_key"] == maintained_port].iloc[0]
    assert maintained["event_time"] == tp.MAINTENANCE_SWAP_AT_MINUTE * 60 * NS
    assert maintained["link_flaps"] >= 2
    assert maintained["link_flap_unpolled"] == True  # noqa: E712  pylint: disable=singleton-comparison
    assert maintained["oper_status"] == "up", "both polls read up; only ifLastChange reveals the transition"

    # Everything the search's `table` names is present, so an analyst can trace it without a second query.
    for column in ("entity_key",
                   "site_id",
                   "device_id",
                   "port_id",
                   "transceiver_serial",
                   "transceiver_serial_first_seen",
                   "transceiver_serial_distinct_count",
                   "oper_status",
                   "link_flaps",
                   "event_uid",
                   "lineage_id"):
        assert pd.notna(hit[column]), column


@pytest.mark.cpu_mode
def test_r_p_l1_004_fires_on_the_failing_optic_and_not_on_the_tap(result: pd.DataFrame):
    # The rule's predicate: a poll whose fitted trend is projected to reach the optic's floor within fourteen
    # days. The search then takes one row per port, with the shortest time to the floor in the window.
    layer_1 = result[result["telemetry_class"] == "tc1"]
    projected = layer_1[(layer_1["optical_rx_dbm_forecast_status"] == STATUS_PROJECTED)
                        & (layer_1["optical_rx_dbm_days_to_floor"] <= 14)]

    failing = f"{tp.SITE}:{tp.SWITCH}:{tp.MAINTENANCE_PORT}"
    swap = tp.MAINTENANCE_SWAP_AT_MINUTE * 60 * NS

    # One port, and only while the failing optic was in it: from the first poll with enough history for a fit
    # until the poll before the swap.
    assert set(projected["entity_key"]) == {failing}
    assert projected["event_time"].max() < swap
    assert len(projected) == tp.MAINTENANCE_SWAP_AT_MINUTE - DEFAULT_MIN_SAMPLES + 1
    assert set(projected["transceiver_serial"]) == {f"XCVR-{tp.SWITCH}-{tp.MAINTENANCE_PORT}"}

    # The slope is the planted slide, in the rule's unit, and the floor is the optic's own.
    last = projected.sort_values("event_time").iloc[-1]
    assert last["optical_rx_dbm_trend_db_per_day"] == pytest.approx(-tp.DEGRADATION_DB_PER_MINUTE * 24 * 60, rel=0.05)
    assert last["optical_rx_dbm_floor_dbm"] == tp.OPTIC_FLOOR_DBM
    assert 0 < last["optical_rx_dbm_days_to_floor"] < 1

    # The tap on the hub port loses three decibels at once, which a line fitted through it would call a slope
    # steep enough to fire this rule within the hour. The stage reports it as a step instead, and never projects it.
    tapped = layer_1[layer_1["entity_key"] == f"{tp.SITE}:{tp.SWITCH}:{tp.HUB_PORT}"]
    after_tap = tapped[tapped["event_time"] > tp.TAP_AT_MINUTE * 60 * NS]

    assert STATUS_PROJECTED not in set(tapped["optical_rx_dbm_forecast_status"])
    assert set(after_tap["optical_rx_dbm_forecast_status"]) == {STATUS_NONLINEAR}

    # The replacement optic begins a history of its own, and the steady ports' jitter is not a trend.
    replaced = layer_1[(layer_1["entity_key"] == failing) & (layer_1["event_time"] >= swap)]
    assert set(replaced["optical_rx_dbm_forecast_status"]) <= {STATUS_IMMATURE, STATUS_NOT_DEGRADING}

    for column in ("entity_key",
                   "optical_rx_dbm_days_to_floor",
                   "optical_rx_dbm_trend_db_per_day",
                   "optical_rx_dbm",
                   "optical_rx_dbm_floor_dbm",
                   "optical_rx_dbm_trend_samples",
                   "transceiver_type",
                   "transceiver_serial",
                   "event_uid",
                   "lineage_id"):
        assert pd.notna(last[column]), column


@pytest.mark.cpu_mode
def test_r_d_l2_004_fires_on_both_spoofs_and_not_on_the_move(result: pd.DataFrame):
    # The rule's own predicate, applied to the frame the pipeline produced: a close that means the MAC turned up
    # somewhere else, with the two sightings closer together than the device could have moved.
    bindings = result[result["telemetry_class"] == "tc2_binding"]
    elsewhere = bindings[bindings["bind_end_reason"].isin([CONFLICT, DISPLACED])]
    detections = elsewhere[elsewhere["bind_gap_ns"] <= GAP_THRESHOLD_NS].sort_values("bind_gap_ns")

    assert len(detections) == 2, "the simultaneous spoof and the cross-switch one, and nothing else"

    # The original spoof: two ports on one switch at one instant, so the gap is zero.
    simultaneous = detections.iloc[0]
    assert simultaneous["mac_address"] == tp.MAC_A
    assert simultaneous["bind_end_reason"] == CONFLICT
    assert simultaneous["bind_gap_ns"] == 0
    assert simultaneous["port_key"] == f"{tp.SITE}:{tp.SWITCH}:Gi1/0/1"
    # The detection is about the closing, and the closing is one tick past the conflicting sighting.
    assert simultaneous["bind_end"] == tp.SPOOF_AT_SECONDS * NS + 1

    # The one the old rule could never have caught: a second switch claiming the MAC one sweep later.
    cross_switch = detections.iloc[1]
    assert cross_switch["mac_address"] == tp.MAC_B
    assert cross_switch["bind_end_reason"] == DISPLACED
    assert cross_switch["bind_gap_ns"] == tp.SWEEP_OFFSET_SECONDS * NS
    assert cross_switch["port_key"] == f"{tp.SITE}:{tp.SWITCH}:Gi1/0/2"

    # Everything the search's `table` names is present, so an analyst can trace it without a second query.
    for column in ("mac_address",
                   "port_key",
                   "site_id",
                   "switch_id",
                   "port_id",
                   "vlan_id",
                   "bind_start",
                   "bind_end",
                   "bind_gap_ns",
                   "bind_observations"):
        assert pd.notna(simultaneous[column]), column


@pytest.mark.cpu_mode
def test_a_legitimate_move_is_displaced_but_does_not_fire(result: pd.DataFrame):
    # The negative control, and the reason the rule is a gap rather than a reason. A device that changes port
    # between snapshots is displaced exactly as a spoofed MAC is; only the gap tells them apart. Without this the
    # rule would be a MAC-moved alarm, and every roaming laptop in the estate would be a notable.
    bindings = result[result["telemetry_class"] == "tc2_binding"]
    moved = bindings[(bindings["mac_address"] == tp.ROAM_MAC) & (bindings["bind_end_reason"] == DISPLACED)]

    assert len(moved) == 1
    assert moved.iloc[0]["bind_gap_ns"] == tp.PERIOD_SECONDS * NS
    assert moved.iloc[0]["bind_gap_ns"] > GAP_THRESHOLD_NS, "a whole cadence apart is a move, not a spoof"


@pytest.mark.cpu_mode
def test_r_d_l2_005_fires_on_both_bypasses_and_nothing_else(result: pd.DataFrame):
    auth = result[result["telemetry_class"] == "tc2_auth"]
    detections = auth[auth["auth_unpaired"] == True].sort_values("event_time")  # noqa: E712  pylint: disable=singleton-comparison

    # Two planted bypasses: one on a quiet port, one that arrived while a legitimate exchange on its own port was
    # still open. The second only shows up because exchanges are keyed by device; timed per port it would have
    # paired with the innocent device's request and read as an ordinary authorized session.
    assert len(detections) == 2
    assert list(detections["event_time"]) == [tp.BYPASS_AT_SECONDS * NS, (tp.CONCURRENT_BYPASS_AT_SECONDS + 10) * NS]
    assert list(detections["auth_port_key"]) == [
        f"{tp.SITE}:{tp.SWITCH}:{tp.BYPASS_PORT}",
        f"{tp.SITE}:{tp.SWITCH}:{tp.CONCURRENT_BYPASS_PORT}",
    ]
    assert set(detections["dot1x_result"]) == {"success"}

    # Every field the saved search puts on the alert, including the ones that say which device: on a shared port
    # the port key alone cannot distinguish the bypass from the legitimate session beside it.
    for (_, hit) in detections.iterrows():
        for column in ("auth_port_key", "site_id", "switch_id", "port_id", "dot1x_result", "event_uid", "mac_address"):
            assert pd.notna(hit[column]), column

    # Every paired exchange in the corpus reads False, not null: the rule's negative is an answer, not an absence.
    outcomes = auth[auth["dot1x_result"] == "success"]
    assert (outcomes["auth_unpaired"] == False).sum() == len(outcomes) - 2  # noqa: E712  pylint: disable=singleton-comparison


@pytest.mark.cpu_mode
def test_r_d_l2_001_fires_once_per_offending_mac_on_designated_ports(result: pd.DataFrame):
    # The search: first-in-window rows on ports the inventory designates single-host, where the count exceeds the
    # permitted number. The corpus's own designation list stands in for the lookup.
    rows = result[result["telemetry_class"] == "tc2_mac"]
    designated = rows[rows["port_key"].isin(tp.SINGLE_HOST_PORTS)]
    detections = designated[(designated["macs_per_port_first_in_window"] == True)  # noqa: E712  pylint: disable=singleton-comparison
                            & (designated["macs_per_port"] > 1)]

    hub_port = f"{tp.SITE}:{tp.SWITCH}:{tp.HUB_PORT}"
    spoofed_port = f"{tp.SITE}:{tp.SWITCH}:Gi1/0/2"

    # Four hub MACs, each once as it appears; and the spoofed MAC once, when it shows up on a second port.
    assert set(detections["port_key"]) == {hub_port, spoofed_port}
    assert sorted(detections[detections["port_key"] == hub_port]["mac_address"]) == sorted(tp.HUB_MACS)
    assert list(detections[detections["port_key"] == spoofed_port]["mac_address"]) == [tp.MAC_A]
    # Nothing saturated: the counts are exact, not lower bounds.
    assert not detections["macs_per_port_saturated"].any()


@pytest.mark.cpu_mode
def test_r_b_l2_002_finds_the_hub_and_the_spoof_without_a_designation_list(result: pd.DataFrame):
    # The search: first-in-window rows whose count the port has never reached in any period of its own history,
    # grouped by port. No lookup, no list.
    rows = result[result["telemetry_class"] == "tc2_mac"]
    detections = rows[(rows["macs_per_port_first_in_window"] == True)  # noqa: E712  pylint: disable=singleton-comparison
                      & (rows["macs_per_port_step"] > 0)]

    hub_port = f"{tp.SITE}:{tp.SWITCH}:{tp.HUB_PORT}"
    spoofed_port = f"{tp.SITE}:{tp.SWITCH}:Gi1/0/2"

    # The same two ports R-D-L2-001 names, found from the ports' own histories: the hub's four addresses in the
    # snapshot that carried them, four above the one address the port had in each of the six snapshots before,
    # and the spoofed address when it turned up on a second port, one above that port's record.
    assert set(detections["port_key"]) == {hub_port, spoofed_port}

    hub = detections[detections["port_key"] == hub_port]
    assert sorted(hub["mac_address"]) == sorted(tp.HUB_MACS)
    assert set(hub["event_time"]) == {tp.HUB_FROM_SECONDS * NS}
    assert sorted(hub["macs_per_port_step"]) == list(range(1, len(tp.HUB_MACS) + 1))
    assert set(hub["macs_per_port_baseline_max"]) == {1}
    assert set(hub["macs_per_port_baseline_buckets"]) == {tp.HUB_FROM_SECONDS // tp.PERIOD_SECONDS}

    spoof = detections[detections["port_key"] == spoofed_port]
    assert list(spoof["mac_address"]) == [tp.MAC_A]
    assert list(spoof["macs_per_port_step"]) == [1]
    assert list(spoof["event_time"]) == [tp.SPOOF_AT_SECONDS * NS]

    # The next snapshot's baseline has absorbed the hub, so the rule fires once per step rather than once per
    # snapshot for as long as the hub stays plugged in.
    later = rows[(rows["port_key"] == hub_port) & (rows["event_time"] > tp.HUB_FROM_SECONDS * NS)]
    assert (later["macs_per_port_step"] == 0).all()
    assert set(later["macs_per_port_baseline_max"]) == {1 + len(tp.HUB_MACS)}

    # The peer switch's ports were met inside this hour. The one the roaming device left has a baseline and
    # nothing above it; the ones it arrived on have no baseline at all, and neither is a step.
    peer = rows[rows["switch_id"] == tp.PEER_SWITCH]
    assert not (peer["macs_per_port_step"].fillna(0) > 0).any()

    for column in ("port_key",
                   "macs_per_port",
                   "macs_per_port_baseline_max",
                   "macs_per_port_step",
                   "mac_address",
                   "macs_per_port_saturated",
                   "event_uid",
                   "lineage_id"):
        assert pd.notna(hub.iloc[-1][column]), column


@pytest.mark.cpu_mode
def test_r_d_l2_003_fires_on_the_flooded_gateway_and_not_on_the_redundancy_pair(result: pd.DataFrame):
    arp = result[result["telemetry_class"] == "tc2_arp"]
    candidates = arp[(arp["macs_claiming_sender_ip"].fillna(0) > 1) & (arp["arp_sender_ip_excluded"] == False)]  # noqa: E712  pylint: disable=singleton-comparison

    # The search groups by address within the window; here, one contested address in the whole corpus.
    assert set(candidates["arp_sender_ip"]) == {tp.GATEWAY_IP}
    assert set(candidates["arp_sender_mac"]) == {tp.MAC_A, tp.ROUTER_MAC}
    assert candidates["macs_claiming_sender_ip"].max() == 2

    # The VRRP pair is contested by design, marked excluded, and therefore not a candidate. Without the exclusion
    # this would be a second detection, and a wrong one.
    vrrp = arp[arp["arp_sender_ip"] == tp.VRRP_IP]
    assert vrrp["macs_claiming_sender_ip"].max() == 2
    assert (vrrp["arp_sender_ip_excluded"] == True).all()  # noqa: E712  pylint: disable=singleton-comparison


# --- The stanzas, read from the app itself ------------------------------------------------------------------------


def test_every_shipped_detection_is_defined(searches: dict[str, dict[str, str]]):
    for stanza in RULES.values():
        assert stanza in searches, stanza


@pytest.mark.parametrize("rule_id", sorted(RULES))
def test_the_search_reads_the_column_the_stage_emits(rule_id: str, searches: dict[str, dict[str, str]]):
    spl = searches[RULES[rule_id]]["search"]

    expected = {
        "R-D-L1-001": ("sourcetype=morpheus:score:l1",
                       'transceiver_serial_changed="true"',
                       "link_flaps=0",
                       'oper_status!="down"',
                       "last(transceiver_serial) AS previous_serial BY entity_key"),
        "R-P-L1-004": ("sourcetype=morpheus:score:l1",
                       'optical_rx_dbm_forecast_status="projected"',
                       "optical_rx_dbm_days_to_floor<=14",
                       "min(optical_rx_dbm_days_to_floor) AS days_to_floor",
                       "BY entity_key"),
        "R-B-L2-002": ("sourcetype=morpheus:score:l2",
                       "macs_per_port_first_in_window=true",
                       "macs_per_port_step>0",
                       "max(macs_per_port_baseline_max) AS baseline_max",
                       "BY port_key"),
        "R-D-L2-001": ("sourcetype=morpheus:score:l2",
                       "macs_per_port_first_in_window=true",
                       "lookup port_designations port_key",
                       'designation="single-host"',
                       "macs_per_port_saturated"),
        "R-D-L2-003": ("sourcetype=morpheus:score:l2", "macs_claiming_sender_ip>1", "arp_sender_ip_excluded=false"),
        "R-D-L2-004": ("sourcetype=binding:l2",
                       f"bind_end_reason={CONFLICT}",
                       f"bind_end_reason={DISPLACED}",
                       "bind_gap_ns",
                       "port_key"),
        "R-D-L2-005": ("sourcetype=morpheus:score:l2", "auth_unpaired=true", "auth_port_key", "event_uid"),
        # The layer 5 pair. Their predicates are asserted against the corpus in test_session_harness.py and their
        # row counts in test_splunk_validation_package.py; what is checked here is that the SPL reads the columns
        # the stages actually emit, which is the drift these fragments exist to catch.
        "R-D-L5-003": ("sourcetype=morpheus:score:l5",
                       "travel_status=measured",
                       "travel_kmh",
                       "travel_elapsed_ns",
                       "user_principal"),
        "R-D-L5-004": ("sourcetype=morpheus:score:l5",
                       "mfa_denied_then_approved=true",
                       "mfa_attempts_in_window",
                       "mfa_denials_in_window",
                       "consecutive_mfa_denials"),
        "R-P-L5-006": ("sourcetype=morpheus:score:l5",
                       "drift_mature=true",
                       "drift_rising_windows >= rising_threshold",
                       "drift_rise_sigmas > sigma_threshold",
                       "mean_abs_z < mean_ceiling",
                       "abs(drift_acceleration) < acceleration_ceiling",
                       "dedup user_principal day_window_id",
                       "outputlookup principal_watchlist"),
        "R-D-L5-007": ("sourcetype=morpheus:score:l5",
                       "hour_unseen=true",
                       "cadence_mature=true",
                       "auth_result=success",
                       "hour_surprise_bits"),
        "R-D-L5-008": ("sourcetype=morpheus:score:l5",
                       "auth_result=success",
                       "cadence_mature=true",
                       'location_first_seen == "true"',
                       'device_first_seen == "true"'),
        "R-D-L5-009": ("sourcetype=morpheus:score:l5",
                       "auth_failed_then_succeeded=true",
                       "consecutive_auth_failures >= failure_threshold",
                       "auth_failures_in_window"),
        "R-B-L5-001": ("sourcetype=morpheus:score:l5",
                       "model_fallback_used=false",
                       "max_abs_z >= max_threshold",
                       "mean_abs_z >= mean_threshold"),
        "R-B-L5-002":
            ("sourcetype=morpheus:score:l5", "model_fallback_used=false", "locincrement_z_loss >= loss_threshold"),
        "R-B-L7-002": ("sourcetype=morpheus:score:l7",
                       "saas_baseline_mature=true",
                       "saas_record_ratio>5",
                       "ctx_object_data_classification",
                       "saas_record_baseline",
                       "lookup principal_watchlist user_principal"),
        "R-P-L7-006": ("sourcetype=morpheus:score:l7",
                       "saas_object_types_in_week=*",
                       "max(drift_rising_windows) AS rising_weeks",
                       "dc(role) AS role_versions",
                       "values(week_window_id) AS weeks",
                       "BY user_principal",
                       "outputlookup principal_watchlist"),
        "R-C-005": ("sourcetype=morpheus:score:l5",
                    'site_travel_status="measured"',
                    "last(login_port_key) AS previous_port",
                    "last(login_site_id) AS previous_site",
                    "site_travel_kmh >= 900",
                    "BY user_principal"),
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

    # Both boundaries snap to a unit, so a run never ends at "now" and two runs over the same interval agree.
    latest = re.fullmatch(r"(?:-(\d+)([mhdw]))?@([mhdw]\d*)", stanza["dispatch.latest_time"])
    earliest = re.fullmatch(r"-(\d+)([mhdw])@([mhdw]\d*)", stanza["dispatch.earliest_time"])

    assert latest is not None, stanza["dispatch.latest_time"]
    assert earliest is not None, stanza["dispatch.earliest_time"]

    unit_minutes = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    trailing_minutes = int(latest.group(1) or 0) * unit_minutes[latest.group(2) or "m"]
    earliest_minutes = int(earliest.group(1)) * unit_minutes[earliest.group(2)]

    cadence_minutes = _cadence_minutes(stanza["cron_schedule"])

    if (cadence_minutes < 1440):
        # A rule that runs more than daily trails by at least the lateness horizon, and its window width equals
        # its cadence, so consecutive runs are disjoint and a detection is emitted once. The hourly watchlists
        # run at a fixed minute over the hour before.
        assert trailing_minutes >= tp.LATENESS_SECONDS // 60, "the window must trail by at least the lateness horizon"

        if (earliest_minutes - trailing_minutes != cadence_minutes):
            # A chained rule reaches back for the steps before its last one, so its window is wider than its
            # cadence and a chain is re-emitted on every run inside that window until suppression lands
            # (retrospective gap G20, issue #57). A single-layer rule has no such excuse.
            assert rule_id.startswith("R-C-"), f"{rule_id}'s window is not its cadence"
            assert earliest_minutes - trailing_minutes > cadence_minutes
    else:
        # The daily and weekly watchlists snap to the day or week and look back over at least one cadence, and
        # they deduplicate on the window they read, so an overlapping run repeats nothing.
        assert latest.group(3)[0] in ("m", "d", "w"), stanza["dispatch.latest_time"]
        assert earliest_minutes >= cadence_minutes
        assert "dedup" in searches[RULES[rule_id]]["search"] or "| stats" in searches[RULES[rule_id]]["search"]


def _cadence_minutes(cron: str) -> int:
    """How often a cron expression fires, in minutes, for the shapes the app uses."""
    every = re.fullmatch(r"\*/(\d+) \* \* \* \*", cron)
    hourly = re.fullmatch(r"\d+ \* \* \* \*", cron)
    daily = re.fullmatch(r"\d+ \d+ \* \* \*", cron)
    weekly = re.fullmatch(r"\d+ \d+ \* \* \d", cron)

    if (every is not None):
        return int(every.group(1))

    if (hourly is not None):
        return 60

    if (daily is not None):
        return 1440

    assert weekly is not None, cron

    return 10080


def test_layer_3_seals_hourly_as_the_lateral_movement_chain_assumes(searches: dict[str, dict[str, str]]):
    # R-C-001 joins `window_id = window_id + 1` to mean "the hour before". That is true only because every layer 3
    # corpus seals at 3600 seconds, where WindowSealStage's default is 300; the stanza now says so, and this pins
    # the two corpora the rule is asserted over to the period the SPL encodes.
    import campaign_pipeline  # pylint: disable=import-outside-toplevel
    import network_pipeline  # pylint: disable=import-outside-toplevel

    assert network_pipeline.PERIOD_SECONDS == 3600
    assert campaign_pipeline.PERIOD_SECONDS == 3600

    stanza = searches["R-C-001 - Lateral movement chain"]

    assert "window_id = window_id + 1" in stanza["search"]
    assert "seals hourly" in stanza["description"]
    assert "period_seconds=3600" in stanza["description"]


def test_the_binding_sourcetype_is_timed_on_the_end():
    # A binding opened hours ago and closed by a conflict now must land in the detection window that covers now.
    props = read_conf(PROPS)

    assert "binding:l2" in props
    assert '"bind_end"' in props["binding:l2"]["TIME_PREFIX"]
    assert props["binding:l2"]["TIME_FORMAT"] == props["morpheus:score:l2"]["TIME_FORMAT"]


def test_the_designation_lookup_ships_with_its_schema_and_no_guesses():
    # A designation list is a fact about one estate. The file defines the columns the search reads and nothing else,
    # so the rule fires on nothing until the inventory populates it.
    with open(LOOKUP, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    assert lines == ["port_key,designation,max_macs"]

    transforms = read_conf(os.path.join(APP_DEFAULT, "transforms.conf"))
    assert transforms["port_designations"]["filename"] == os.path.basename(LOOKUP)


def test_the_open_binding_sourcetype_is_timed_on_the_start():
    # A provisional record has no end yet; its only time is when the binding opened.
    props = read_conf(PROPS)

    assert "binding:l2:open" in props
    assert '"bind_start"' in props["binding:l2:open"]["TIME_PREFIX"]
    assert props["binding:l2:open"]["TIME_FORMAT"] == props["binding:l2"]["TIME_FORMAT"]
