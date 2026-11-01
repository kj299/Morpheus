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
import importlib.util
import json
import os
import re
import subprocess
import sys
import typing

import pandas as pd
import pytest

from morpheus.stages.telemetry.tc7_endpoint_stage import normalize_image_path

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
import campaign_pipeline  # noqa: E402
import endpoint_pipeline  # noqa: E402
import session_pipeline as sp  # noqa: E402
import telemetry_pipeline as tp  # noqa: E402


def _stanza_search(name: str) -> str:
    """One saved search's SPL, with its continuation lines folded."""
    import configparser  # pylint: disable=import-outside-toplevel

    with open(SAVEDSEARCHES, encoding="utf-8") as handle:
        folded = re.sub(r"\\\s*\r?\n\s*", " ", handle.read())

    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read_string(folded)

    return parser[name]["search"]


def threshold(stanza: str, pattern: str) -> float:
    """
    A threshold as the shipped search states it, so the predicate this file evaluates is the app's and not a copy.

    Six layer 5 thresholds lived here as constants for weeks; a stanza edit would have left every test green while
    the package's expectations went stale. `pattern` names the eval variable or the comparison, with one group
    around the number.
    """

    match = re.search(pattern, _stanza_search(stanza))

    assert match is not None, f"{stanza} no longer states {pattern!r}"

    return float(match.group(1))


GAP_THRESHOLD_NS = int(threshold("R-D-L2-004 - MAC in two places at once", r"gap_threshold\s*=\s*(\d+)"))
TRAVEL_KMH_THRESHOLD = threshold("R-D-L5-003 - Impossible travel", r"kmh_threshold\s*=\s*([\d.]+)")
MFA_CHALLENGE_THRESHOLD = threshold("R-D-L5-004 - Multi-factor fatigue", r"challenge_threshold\s*=\s*([\d.]+)")
MFA_DENIAL_THRESHOLD = threshold("R-D-L5-004 - Multi-factor fatigue", r"denial_threshold\s*=\s*([\d.]+)")
DRIFT_RISING_THRESHOLD = threshold("R-P-L5-006 - Drift trajectory", r"rising_threshold\s*=\s*([\d.]+)")
DRIFT_SIGMA_THRESHOLD = threshold("R-P-L5-006 - Drift trajectory", r"sigma_threshold\s*=\s*([\d.]+)")
DRIFT_MEAN_CEILING = threshold("R-P-L5-006 - Drift trajectory", r"mean_ceiling\s*=\s*([\d.]+)")
DRIFT_ACCELERATION_CEILING = threshold("R-P-L5-006 - Drift trajectory", r"acceleration_ceiling\s*=\s*([\d.]+)")
HOST_DRIFT_RISING_THRESHOLD = threshold("R-P-L3-009 - Host drift trajectory", r"rising_threshold\s*=\s*([\d.]+)")
HOST_DRIFT_SIGMA_THRESHOLD = threshold("R-P-L3-009 - Host drift trajectory", r"sigma_threshold\s*=\s*([\d.]+)")
STACK_TRAJECTORY = "R-P-L6-006 - Client stack trajectory"
SUBDOMAIN_TRAJECTORY = "R-P-L7-009 - Subdomain count trajectory"
STACK_RISING_THRESHOLD = threshold(STACK_TRAJECTORY, r"rising_threshold\s*=\s*([\d.]+)")
STACK_SIGMA_THRESHOLD = threshold(STACK_TRAJECTORY, r"sigma_threshold\s*=\s*([\d.]+)")
SUBDOMAIN_RISING_THRESHOLD = threshold(SUBDOMAIN_TRAJECTORY, r"rising_threshold\s*=\s*([\d.]+)")
SUBDOMAIN_SIGMA_THRESHOLD = threshold(SUBDOMAIN_TRAJECTORY, r"sigma_threshold\s*=\s*([\d.]+)")
PATH_TRAJECTORY = "R-P-L7-010 - Path count trajectory"
BREADTH_TRAJECTORY = "R-P-L7-011 - Process breadth trajectory"
PATH_RISING_THRESHOLD = threshold(PATH_TRAJECTORY, r"rising_threshold\s*=\s*([\d.]+)")
PATH_SIGMA_THRESHOLD = threshold(PATH_TRAJECTORY, r"sigma_threshold\s*=\s*([\d.]+)")
BREADTH_RISING_THRESHOLD = threshold(BREADTH_TRAJECTORY, r"rising_threshold\s*=\s*([\d.]+)")
BREADTH_SIGMA_THRESHOLD = threshold(BREADTH_TRAJECTORY, r"sigma_threshold\s*=\s*([\d.]+)")
FAILURE_RUN_THRESHOLD = threshold("R-D-L5-009 - Failed authentication run ending in success",
                                  r"failure_threshold\s*=\s*([\d.]+)")
COMPOSITE_MAX_THRESHOLD = threshold("R-B-L5-001 - Composite authentication anomaly", r"max_threshold\s*=\s*([\d.]+)")
COMPOSITE_MEAN_THRESHOLD = threshold("R-B-L5-001 - Composite authentication anomaly", r"mean_threshold\s*=\s*([\d.]+)")
LOCATION_LOSS_THRESHOLD = threshold("R-B-L5-002 - Location novelty anomaly", r"loss_threshold\s*=\s*([\d.]+)")
DURATION_RATIO_THRESHOLD = threshold("R-B-L5-005 - Session duration anomaly", r"ratio_threshold\s*=\s*([\d.]+)")
FORECAST_DAYS_THRESHOLD = threshold("R-P-L1-004 - Optical degradation forecast",
                                    r"optical_rx_dbm_days_to_floor\s*<=\s*([\d.]+)")
TAP = "R-D-L1-002 - Optical tap step"
FLAP = "R-D-L1-003 - Link flap instability"
ERRORS = "R-B-L1-005 - Error rate step"
NEIGHBOR = "R-D-L1-006 - LLDP neighbor change"
VOLUME = "R-B-L1-007 - Interface volume departure"
CONFIGURATION = "R-D-L1-008 - Configuration change outside the change window"
TRAJECTORY = "R-P-L1-009 - Reboot-rate trajectory"
COMMISSIONING = "R-D-L1-010 - Optical level below commissioning"
PHYSICAL = "R-C-003 - Physical compromise chain"
VENDOR = "R-B-L2-006 - New vendor on a VLAN"
SLOW = "R-B-L2-007 - Slow 802.1X authorization"
INSTANT = "R-B-L2-008 - Instant 802.1X authorization"
TAP_DEVIATION = threshold(TAP, r"optical_rx_dbm_deviation\s*<=\s*(-?[\d.]+)")
TAP_SAMPLES = threshold(TAP, r"optical_rx_dbm_baseline_samples\s*>=\s*([\d.]+)")
TAP_TRANSMIT = threshold(TAP, r"abs\(optical_tx_dbm_deviation\)\s*<\s*([\d.]+)")
COMMISSIONING_MARGIN = threshold(COMMISSIONING, r"rx_below_commissioning_db\s*>=\s*([\d.]+)")
FLAP_TRANSITIONS = threshold(FLAP, r"link_flaps_in_window\s*>=\s*([\d.]+)")
ERROR_STEP = threshold(ERRORS, r"error_rate_step\s*>\s*([\d.]+)")
SURGE_FACTOR = threshold(VOLUME, r"surge_factor\s*=\s*([\d.]+)")
SILENCE_FACTOR = threshold(VOLUME, r"silence_factor\s*=\s*([\d.]+)")
RISING_PERIODS = threshold(TRAJECTORY, r"device_reboot_rising_buckets\s*>=\s*([\d.]+)")
PHYSICAL_TOLERANCE_SECONDS = int(threshold(PHYSICAL, r"t_vendor >= t_port - (\d+) AND"))
PHYSICAL_WINDOW_SECONDS = int(threshold(PHYSICAL, r"t_vendor - t_port <= (\d+)"))
SLOW_RATIO = threshold(SLOW, r"auth_elapsed_ratio\s*>=\s*([\d.]+)")
SLOW_ATTEMPTS = threshold(SLOW, r"auth_attempts\s*>=\s*([\d.]+)")
INSTANT_RATIO = threshold(INSTANT, r"auth_elapsed_ratio\s*<=\s*([\d.]+)")
TTL_SHIFT = "R-B-L3-004 - TTL fingerprint shift"
TTL_SHIFTED_FLOWS = threshold(TTL_SHIFT, r"shifted_flows\s*>=\s*([\d.]+)")
"""The thresholds the saved searches state, read from the stanzas rather than repeated here."""


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

    with open(SAVEDSEARCHES, encoding="utf-8") as handle:
        folded = re.sub(r"\\\s*\r?\n\s*", " ", handle.read())

    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read_string(folded)

    return {name for name in parser.sections() if parser.has_option(name, "search")}


def _search_text(name: str) -> str:
    import configparser

    with open(SAVEDSEARCHES, encoding="utf-8") as handle:
        folded = re.sub(r"\\\s*\r?\n\s*", " ", handle.read())

    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read_string(folded)

    return parser[name]["search"]


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

    # R-B-L2-002 needs no gate: the first-in-window rows whose count the port has never reached, one notable per
    # port with the largest step and the addresses that made it. The estate's ports are derived here from the
    # pipeline; the campaign's desk ports reach the same sourcetype from another corpus and are derived from the
    # indexed events in the layer 2 test below.
    stepped = scored[(scored["macs_per_port_first_in_window"] == True)  # noqa: E712  pylint: disable=singleton-comparison
                     & (scored["macs_per_port_step"] > 0)]
    novelty = searches["R-B-L2-002 - Port-to-MAC binding novelty"]
    estate_ports = set(scored["port_key"].dropna())

    written = {(row["port_key"], row["step"], row["new_mac_count"])
               for row in novelty["key_values"] if row["port_key"] in estate_ports}
    assert written == {(port, int(group["macs_per_port_step"].max()), group["mac_address"].nunique())
                       for (port, group) in stepped.groupby("port_key")}


def test_the_layer_1_detection_returns_exactly_what_is_written(expected: dict, telemetry: pd.DataFrame):
    # R-D-L1-001's predicate as the search states it: a serial that differs from the port's previous poll, on a
    # poll the flap count says the link never moved for. The previous serial is the search's own streamstats,
    # done here as the same shift within the port.
    layer_1 = telemetry[telemetry["telemetry_class"] == "tc1"].sort_values(["entity_key", "event_time"])
    previous = layer_1.groupby("entity_key")["transceiver_serial"].shift(1)
    changed = layer_1[layer_1["transceiver_serial_changed"] == True]  # noqa: E712  pylint: disable=singleton-comparison
    substitutions = changed[(changed["link_flaps"] == 0) & (changed["oper_status"] != "down")]

    entry = expected["searches"]["R-D-L1-001 - Transceiver substitution"]

    assert entry["expected_rows"] == len(substitutions)
    written = {(row["entity_key"], row["previous_serial"], row["transceiver_serial"]) for row in entry["key_values"]}
    assert written == {(row["entity_key"], previous.loc[index], row["transceiver_serial"])
                       for (index, row) in substitutions.iterrows()}

    # The other replaced optic is in the data and is quiet for the reason the expectation gives: the device saw
    # the link drop for it. An expectation of one row over a corpus with one serial change would prove nothing.
    quiet = changed.drop(substitutions.index)

    assert len(quiet) == 1
    assert (quiet["link_flaps"] >= 2).all()


def test_the_forecast_returns_exactly_what_is_written(expected: dict, telemetry: pd.DataFrame):
    # R-P-L1-004's predicate as the search states it, then its `stats ... BY entity_key`: one row per port whose
    # fitted trend reaches the floor within fourteen days, carrying the shortest time to it in the window.
    layer_1 = telemetry[telemetry["telemetry_class"] == "tc1"]
    projected = layer_1[(layer_1["optical_rx_dbm_forecast_status"] == "projected")
                        & (layer_1["optical_rx_dbm_days_to_floor"] <= FORECAST_DAYS_THRESHOLD)]

    entry = expected["searches"]["R-P-L1-004 - Optical degradation forecast"]

    assert entry["expected_rows"] == projected["entity_key"].nunique()
    assert entry["contributing_rows"] == len(projected)

    written = {(row["entity_key"], row["transceiver_type"], row["floor_dbm"]) for row in entry["key_values"]}
    assert written == set(
        zip(projected["entity_key"], projected["transceiver_type"], projected["optical_rx_dbm_floor_dbm"]))

    # The controls are in the data under the statuses the expectation names, so a stage that started projecting
    # the tap's step or the steady ports' jitter would be caught here as extra rows rather than read as a feature.
    assert set(layer_1["optical_rx_dbm_forecast_status"]) >= {"projected", "nonlinear", "not_degrading", "immature"}


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
                 & (auth["mean_abs_z"].astype(float) < DRIFT_MEAN_CEILING)
                 & (auth["drift_acceleration"].astype(float).abs() < DRIFT_ACCELERATION_CEILING)]
    deduplicated = fires.drop_duplicates(["user_principal", "day_window_id"])

    assert entry["contributing_rows"] == len(fires)
    assert entry["expected_rows"] == len(deduplicated)

    written = {(row["user_principal"], row["day_window_id"]) for row in entry["key_values"]}
    assert written == set(zip(deduplicated["user_principal"], deduplicated["day_window_id"].astype(int)))


def test_the_host_drift_rule_returns_exactly_what_is_written(expected: dict):
    # R-P-L3-009 as the search states it, over the host segment's events and deduplicated as the search
    # deduplicates. The entry names every host-day, as the principals' rule's does: a firing nobody can account for
    # is the drift in meaning this file exists to catch, and the rule has no ceiling to hand a loud day to another.
    events = _events_of("morpheus_score_host.jsonlines")
    entry = expected["searches"]["R-P-L3-009 - Host drift trajectory"]

    fires = [
        event for event in events if event.get("window_complete") is True and event.get("drift_mature") is True
        and float(event.get("drift_rising_windows") or 0) >= HOST_DRIFT_RISING_THRESHOLD
        and float(event.get("drift_rise_sigmas") or 0) > HOST_DRIFT_SIGMA_THRESHOLD
    ]
    host_days = sorted({(event["host_key"], int(event["day_window_id"])) for event in fires})

    assert entry["contributing_rows"] == len(fires)
    assert entry["expected_rows"] == len(host_days)
    assert [(row["host_key"], row["day_window_id"]) for row in entry["key_values"]] == host_days

    # One host, and its rising days from the first the drift stage calls mature: the fourth rising day meets both
    # conditions and is not yet mature, which the expectation's reason records rather than tunes away.
    assert {host for (host, _) in host_days} == {entry["key_values"][0]["host_key"]}
    assert [day for (_, day) in host_days] == list(range(host_days[0][1], host_days[0][1] + len(host_days)))

    immature = [
        event for event in events
        if event.get("drift_mature") is False and float(event.get("drift_rising_windows") or 0) >=
        HOST_DRIFT_RISING_THRESHOLD and float(event.get("drift_rise_sigmas") or 0) > HOST_DRIFT_SIGMA_THRESHOLD
    ]

    assert {int(event["day_window_id"]) for event in immature} == {host_days[0][1] - 1}

    # The summary reads these rows too, and peak_z is a number on every scored one.
    scored = [event for event in events if event.get("mean_abs_z") is not None]

    assert scored and all(event.get("max_abs_z") is not None and event.get("osi_layer") == 3 for event in scored)


def _trajectory_days(events: list, rising: float, sigmas: float, key: str = "src_ip") -> tuple:
    """A trajectory rule as its search states it: the rows that carry the trajectory, and the client-days they make."""
    fires = [
        event for event in events
        if event.get("window_complete") is True and event.get("drift_mature") is True and float(
            event.get("drift_rising_windows") or 0) >= rising and float(event.get("drift_rise_sigmas") or 0) > sigmas
    ]
    immature = [
        event for event in events
        if event.get("drift_mature") is False and float(event.get("drift_rising_windows") or 0) >= rising
        and float(event.get("drift_rise_sigmas") or 0) > sigmas
    ]

    return (fires, immature, sorted({(event[key], int(event["day_window_id"])) for event in fires}))


def _asserts_one_client_from_its_fifth_day(entry: dict, fires: list, immature: list, client_days: list) -> None:
    # The same shape as the principals' and the hosts' trajectories: one entity, its rising days from the first the
    # drift stage calls mature, and the fourth rising day meeting both conditions while not yet mature.
    assert entry["contributing_rows"] == len(fires)
    assert entry["expected_rows"] == len(client_days)
    assert [(row["src_ip"], row["day_window_id"]) for row in entry["key_values"]] == client_days
    assert {client for (client, _) in client_days} == {entry["key_values"][0]["src_ip"]}
    assert [day for (_, day) in client_days] == list(range(client_days[0][1], client_days[0][1] + len(client_days)))
    assert {(event["src_ip"], int(event["day_window_id"]))
            for event in immature} == {(client_days[0][0], client_days[0][1] - 1)}


def test_the_stack_trajectory_returns_exactly_what_is_written(expected: dict):
    # R-P-L6-006 over the layer 6 events, deduplicated as the search deduplicates. The creeping host is also
    # R-B-L6-001's six times over, which the fingerprint test counts; here the point is that the two rules read the
    # same handshakes and that the control which acquires two stacks and stops never reaches four rising days.
    events = _events_of("morpheus_score_l6.jsonlines")
    (fires, immature, host_days) = _trajectory_days(events, STACK_RISING_THRESHOLD, STACK_SIGMA_THRESHOLD)
    entry = expected["searches"][STACK_TRAJECTORY]

    _asserts_one_client_from_its_fifth_day(entry, fires, immature, host_days)

    creeper = host_days[0][0]
    by_day = collections.defaultdict(set)
    rising = collections.defaultdict(int)

    for event in events:
        if (event.get("day_window_id") is not None):
            by_day[(event["src_ip"], int(event["day_window_id"]))].add(event["ja4_client_distinct"])
            rising[event["src_ip"]] = max(rising[event["src_ip"]], int(event.get("drift_rising_windows") or 0))

    days = sorted(day for (host, day) in by_day if host == creeper)

    assert [max(by_day[(creeper, day)]) for day in days] == list(range(1, len(days) + 1)), "one new stack a day"
    assert rising[creeper] == len(days)

    # The control rises for three days and holds; nobody else rises at all.
    others = {host: count for (host, count) in rising.items() if host != creeper}

    assert max(others.values()) < STACK_RISING_THRESHOLD
    assert sorted(others.values())[-1] == 3 and sorted(others.values())[-2] == 1

    # Every firing is a row the fingerprint rule's host, so the watchlist entry is keyed on the host the lease names.
    assert {event.get("host_key") for event in fires} == {"ws-0-24"}


def test_the_subdomain_trajectory_returns_exactly_what_is_written(expected: dict):
    # R-P-L7-009 over the DNS rows of the layer 7 events, as the search reads them: `dns_registered_domain=*` is the
    # DNS variant's row, and the HTTP and service rows carry no drift. The slow tunnel reads the same column as
    # R-B-L7-001 and never reaches that rule's floor, which the layer 7 test asserts from the other side.
    events = [event for event in _layer_7_events() if event.get("dns_registered_domain") is not None]
    (fires, immature, client_days) = _trajectory_days(events, SUBDOMAIN_RISING_THRESHOLD, SUBDOMAIN_SIGMA_THRESHOLD)
    entry = expected["searches"][SUBDOMAIN_TRAJECTORY]

    _asserts_one_client_from_its_fifth_day(entry, fires, immature, client_days)

    tunneller = client_days[0][0]
    busiest = collections.defaultdict(int)
    rising = collections.defaultdict(int)

    for event in events:
        if (event.get("day_window_id") is not None):
            key = (event["src_ip"], int(event["day_window_id"]))
            busiest[key] = max(busiest[key], int(event.get("dns_subdomains_per_domain") or 0))
            rising[event["src_ip"]] = max(rising[event["src_ip"]], int(event.get("drift_rising_windows") or 0))

    days = sorted(day for (client, day) in busiest if client == tunneller)
    counts = [busiest[(tunneller, day)] for day in days]

    assert counts == sorted(counts) and len(set(counts)) == len(counts), "the tunnel climbs every day"
    assert max(counts) < TUNNEL_SUBDOMAINS, "and never reaches the literal R-B-L7-001 reads"
    assert rising[tunneller] == len(days)
    assert max(count for (client, count) in rising.items() if client != tunneller) < SUBDOMAIN_RISING_THRESHOLD

    # The HTTP rows carry a trajectory of their own, R-P-L7-010's, and the service rows none. The
    # `dns_registered_domain=*` term keeps the slow enumerator's rising days out of this rule, as it keeps the SaaS
    # corpus's weekly drift, which shares the sourcetype and is R-P-L7-006's.
    http = [event for event in _layer_7_events() if event.get("telemetry_class") == "tc7_http"]
    services = [event for event in _layer_7_events() if event.get("telemetry_class") == "tc7_service"]

    assert services and all(event.get("drift_rising_windows") is None for event in services)
    assert _trajectory_days(http, SUBDOMAIN_RISING_THRESHOLD, SUBDOMAIN_SIGMA_THRESHOLD)[0]
    assert not any(event.get("dns_registered_domain") for event in http)
    assert {event.get("host_key") for event in fires} == {"ws-0-46"}


def test_the_path_trajectory_returns_exactly_what_is_written(expected: dict):
    # R-P-L7-010 over the HTTP rows, as the search reads them: `http_distinct_paths=*` is the HTTP variant's row. The
    # slow enumerator reads the column R-D-L7-005 reads and never reaches that rule's floor, which the layer 7 tests
    # assert from the other side; its line is projected to reach it, and the row carries how many days that takes.
    events = [event for event in _layer_7_events() if event.get("http_distinct_paths") is not None]
    (fires, immature, client_days) = _trajectory_days(events, PATH_RISING_THRESHOLD, PATH_SIGMA_THRESHOLD)
    entry = expected["searches"][PATH_TRAJECTORY]

    _asserts_one_client_from_its_fifth_day(entry, fires, immature, client_days)

    enumerator = client_days[0][0]
    busiest = collections.defaultdict(int)
    to_ceiling = {}

    for event in events:
        if (event.get("day_window_id") is not None and event["src_ip"] == enumerator):
            day = int(event["day_window_id"])
            busiest[day] = max(busiest[day], int(event["http_distinct_paths"]))
            to_ceiling[day] = event.get("drift_windows_to_ceiling")

    counts = [busiest[day] for day in sorted(busiest)]
    floor = threshold("R-D-L7-005 - Enumeration", r"http_distinct_paths>(\d+)")

    assert counts == sorted(counts) and len(set(counts)) == len(counts), "the probing climbs every day"
    assert max(counts) < floor, "and never reaches the floor R-D-L7-005 reads"
    assert [to_ceiling[day] for (_, day) in client_days] == [row["windows_to_ceiling"] for row in entry["key_values"]]
    assert all(days > 0 for days in to_ceiling.values() if days is not None)
    assert len({event.get("host_key") or event["src_ip"] for event in fires}) == 1


def test_the_process_breadth_trajectory_returns_exactly_what_is_written(expected: dict):
    # R-P-L7-011 over the endpoint rows, keyed on the host as the search deduplicates. Every pair the creeping build
    # server takes up its peer already runs, so R-B-L7-004 is quiet on all of them; the breadth is what moves.
    events = [event for event in _layer_7_events() if event.get("endpoint_pair") is not None]
    (fires, _, host_days) = _trajectory_days(events, BREADTH_RISING_THRESHOLD, BREADTH_SIGMA_THRESHOLD, "endpoint_host")
    entry = expected["searches"][BREADTH_TRAJECTORY]

    assert entry["contributing_rows"] == len(fires)
    assert entry["expected_rows"] == len(host_days)
    assert [(row["endpoint_host"], row["day_window_id"]) for row in entry["key_values"]] == host_days
    assert len({host for (host, _) in host_days}) == 1
    tools = _peer_tool_pairs()

    assert not any(event.get("endpoint_pair_novel") is True for event in fires if event.get("endpoint_pair") in tools)
    assert {event.get("host_key") for event in fires} == {host_days[0][0]}


def _peer_tool_pairs() -> set:
    """The pairs the creeping build server takes up, as the endpoint stage spells them."""
    return {
        f"{normalize_image_path(parent)} -> {normalize_image_path(image)}"
        for (parent, image) in endpoint_pipeline.BUILD_TOOLS
    }


def _true(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("boolean").fillna(False)


def test_the_principal_baseline_rules_return_exactly_what_is_written(expected: dict, sessions: pd.DataFrame):
    # R-D-L5-007, R-D-L5-008 and R-D-L5-009, each evaluated as its search states it. The three read columns the
    # layer 5 stages had computed for weeks with no search reading any of them.
    auth = sessions[sessions["telemetry_class"] == "tc5_auth"]
    searches = expected["searches"]
    success = auth["auth_result"] == "success"

    off_hours = auth[_true(auth, "hour_unseen") & _true(auth, "cadence_mature") & success]
    novelty = auth[success & _true(auth, "cadence_mature")
                   & (_true(auth, "location_first_seen") | _true(auth, "device_first_seen"))]
    failures = auth[_true(auth, "auth_failed_then_succeeded")
                    & (auth["consecutive_auth_failures"].astype(float) >= FAILURE_RUN_THRESHOLD)]

    entry = searches["R-D-L5-007 - Off-hours authentication"]
    assert entry["expected_rows"] == len(off_hours)
    assert {(row["user_principal"], row["local_hour"])
            for row in entry["key_values"]} == set(zip(off_hours["user_principal"],
                                                       off_hours["local_hour"].astype(int)))

    entry = searches["R-D-L5-008 - New authentication location or device"]
    assert entry["expected_rows"] == len(novelty)
    assert {(row["user_principal"], row["user_location"], row["cadence_samples"])
            for row in entry["key_values"]
            } == set(zip(novelty["user_principal"], novelty["user_location"], novelty["cadence_samples"].astype(int)))

    entry = searches["R-D-L5-009 - Failed authentication run ending in success"]
    assert entry["expected_rows"] == len(failures)
    assert {(row["user_principal"], row["consecutive_auth_failures"])
            for row in entry["key_values"]
            } == set(zip(failures["user_principal"], failures["consecutive_auth_failures"].astype(int)))


def test_the_session_duration_rule_returns_exactly_what_is_written(expected: dict):
    # R-B-L5-005 over the events a search head would hold, with its exclusion and its severity as the search states
    # them, and its candidates without the exclusion recorded beside it.
    sessions = [
        event for event in _events_of("morpheus_score_l5.jsonlines") if event.get("session_duration_mature") is True
    ]
    candidates = [event for event in sessions if event["session_duration_ratio"] > DURATION_RATIO_THRESHOLD]
    fires = [event for event in candidates if (event.get("ctx_account_type") or "unknown") != "service"]
    severity = {"admin": 70, "elevated": 55}
    entry = expected["searches"]["R-B-L5-005 - Session duration anomaly"]

    assert entry["expected_rows"] == len(fires)
    assert entry["candidate_rows_without_the_account_type_exclusion"] == len(candidates)
    assert [{
        "user_principal": event["user_principal"],
        "session_key": event["session_key"],
        "session_duration_ratio": event["session_duration_ratio"],
        "risk_score": severity.get(event.get("ctx_privilege_level"), 40),
    } for event in fires] == entry["key_values"]

    search = _stanza_search("R-B-L5-005 - Session duration anomaly")

    for (level, score) in severity.items():
        assert f'ctx_privilege_level == "{level}", {score}' in search
    assert 'account_type != "service"' in search


def test_the_model_rules_return_exactly_what_is_written(expected: dict, sessions: pd.DataFrame):
    # R-B-L5-001 and R-B-L5-002 read only rows a principal's own model scored. Their expectations record what the
    # search returns, who it returns, and separately what it would return without the gate, so a population score
    # that one day crosses a threshold shows up as a changed count rather than as a gate quietly doing the work.
    auth = sessions[sessions["telemetry_class"] == "tc5_auth"]
    scored = auth[auth["mean_abs_z"].notna()]
    owned = scored[scored["model_fallback_used"].astype("boolean") == False]  # noqa: E712  pylint: disable=singleton-comparison

    composite = scored[(scored["max_abs_z"].astype(float) >= COMPOSITE_MAX_THRESHOLD)
                       & (scored["mean_abs_z"].astype(float) >= COMPOSITE_MEAN_THRESHOLD)]
    location = scored[scored["locincrement_z_loss"].astype(float) >= LOCATION_LOSS_THRESHOLD]

    for (name, ungated) in (("R-B-L5-001 - Composite authentication anomaly", composite),
                            ("R-B-L5-002 - Location novelty anomaly", location)):
        entry = expected["searches"][name]
        gated = ungated.loc[ungated.index.intersection(owned.index)]

        assert entry["expected_rows"] == len(gated) > 0, name
        assert entry["expected_empty"] is False, name
        assert entry["candidate_rows_without_the_gate"] == len(ungated), name
        assert entry["by_principal"] == dict(sorted(collections.Counter(gated["user_principal"]).items())), name
        assert "model_fallback_used=false" in _stanza_search(name), name


def test_the_predictive_rules_write_the_watchlist_the_bulk_rule_reads(expected: dict):
    # The watchlist is a KV Store lookup, so what this package can check is the contract around it: both
    # predictive searches write the lookup the app defines, keyed so a re-run overwrites, and the bulk-access
    # search reads that lookup by the field the writers key it on. The entries R-P-L5-006 writes are recomputed;
    # VALIDATION.md states the count a search head should hold, and it is checked against that here.
    import configparser  # pylint: disable=import-outside-toplevel

    transforms = configparser.ConfigParser(interpolation=None, strict=False)
    transforms.read(os.path.join(os.path.dirname(SAVEDSEARCHES), "transforms.conf"))
    kvstore = configparser.ConfigParser(interpolation=None, strict=False)
    kvstore.read(os.path.join(os.path.dirname(SAVEDSEARCHES), "collections.conf"))

    fields = [name.strip() for name in transforms["principal_watchlist"]["fields_list"].split(",")]
    collection = transforms["principal_watchlist"]["collection"]

    assert fields[0] == "_key"
    assert {f"field.{name}" for name in fields[1:]} <= set(kvstore[collection])

    for name in ("R-P-L5-006 - Drift trajectory", "R-P-L7-006 - Access breadth trajectory"):
        search = _stanza_search(name)

        assert "outputlookup principal_watchlist append=true key_field=_key" in search, name

        for field in fields[1:]:
            assert re.search(rf"\b{field}\b", search), f"{name} writes no {field}"

    assert "lookup principal_watchlist user_principal OUTPUT" in _stanza_search("R-B-L7-002 - Bulk data access")

    # The hosts' list is the same contract keyed on the host: the three host trajectory rules write it by key, and the
    # process-ancestry rule reads it by the host its rows name.
    host_fields = [name.strip() for name in transforms["host_watchlist"]["fields_list"].split(",")]

    assert host_fields[0] == "_key" and "host_key" in host_fields
    assert {f"field.{name}" for name in host_fields[1:]} <= set(kvstore[transforms["host_watchlist"]["collection"]])

    for name in ("R-P-L3-009 - Host drift trajectory", STACK_TRAJECTORY, SUBDOMAIN_TRAJECTORY):
        host_search = _stanza_search(name)

        assert "outputlookup host_watchlist append=true key_field=_key" in host_search, name

        for field in host_fields[1:]:
            assert re.search(rf"\b{field}\b", host_search), f"{name} writes no {field}"

    # The layer 6 and 7 rows name the host through the lease; a row the lease does not cover is keyed on its address.
    for name in (STACK_TRAJECTORY, SUBDOMAIN_TRAJECTORY, PATH_TRAJECTORY):
        assert "host_key = coalesce(host_key, src_ip)" in _stanza_search(name), name

    assert "host_key = coalesce(host_key, endpoint_host)" in _stanza_search(BREADTH_TRAJECTORY)

    assert "lookup host_watchlist host_key AS endpoint_host OUTPUT" in _stanza_search(
        "R-B-L7-004 - Process ancestry novelty")

    drift = expected["searches"]["R-P-L5-006 - Drift trajectory"]
    breadth = expected["searches"]["R-P-L7-006 - Access breadth trajectory"]
    host_drift = expected["searches"]["R-P-L3-009 - Host drift trajectory"]
    stack = expected["searches"][STACK_TRAJECTORY]
    subdomains = expected["searches"][SUBDOMAIN_TRAJECTORY]

    with open(VALIDATION, encoding="utf-8") as handle:
        document = handle.read()

    assert f"`R-P-L5-006` with `drift-trajectory` and {drift['expected_rows']} entries" in document
    assert f"`R-P-L7-006` with `access-breadth` and {breadth['expected_rows']}" in document
    assert f"`R-P-L3-009` with `host-drift-trajectory` and {host_drift['expected_rows']} entries" in document
    assert f"`R-P-L6-006` with `stack-trajectory` and {stack['expected_rows']} entries" in document
    assert f"`R-P-L7-009` with `subdomain-trajectory` and {subdomains['expected_rows']} entries" in document
    assert f"`R-P-L7-010` with `path-trajectory` and {expected['searches'][PATH_TRAJECTORY]['expected_rows']} entries" \
        in document
    assert (f"`R-P-L7-011` with `process-breadth-trajectory` and "
            f"{expected['searches'][BREADTH_TRAJECTORY]['expected_rows']} entries") in document

    # None of the principals the bulk rule reports is watched, which is why its severities stand as written.
    watched = {row["user_principal"] for row in drift["key_values"] + breadth["key_values"]}
    bulk = {row["user_principal"] for row in expected["searches"]["R-B-L7-002 - Bulk data access"]["key_values"]}

    assert watched and not watched & bulk

    # And one host the process-ancestry rule reports is on the hosts' list: the creeping build server R-P-L7-011
    # watches, whose curl is raised by the boost. The host corpus's machines share nothing with the endpoint corpus's.
    watched_hosts = {entry["host_key"] for entry in _host_watchlist()}
    ancestry = expected["searches"]["R-B-L7-004 - Process ancestry novelty"]
    reported_hosts = {row["endpoint_host"] for row in ancestry["key_values"]}

    assert len(watched_hosts) == 5 and watched_hosts & reported_hosts == {endpoint_pipeline.CREEPING}


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


TUNNEL = "R-B-L7-001 - DNS tunneling"
ENUMERATION = "R-D-L7-005 - Enumeration"
SERVICE = "R-B-L7-007 - Service reached by new clients"
GENERATION = "R-B-L7-008 - Domain generation"
STACK = "R-B-L4-004 - Stack fingerprint mismatch"
TUNNEL_ENTROPY = threshold(TUNNEL, r"dns_subdomain_entropy\s*>\s*([\d.]+)")
TUNNEL_LABEL_LENGTH = threshold(TUNNEL, r"dns_mean_label_length\s*>\s*([\d.]+)")
TUNNEL_SUBDOMAINS = threshold(TUNNEL, r"qualifying_subdomains\s*>\s*([\d.]+)")
TUNNEL_MULTIPLE = threshold(TUNNEL, r"dns_subdomains_per_domain_ratio\s*>=\s*([\d.]+)")
ENUMERATION_PATHS = threshold(ENUMERATION, r"http_distinct_paths\s*>\s*([\d.]+)")
ENUMERATION_RATIO = threshold(ENUMERATION, r"http_4xx_in_window\s*>\s*([\d.]+)\s*\*")
ENUMERATION_MULTIPLE = threshold(ENUMERATION, r"http_distinct_paths_ratio\s*>=\s*([\d.]+)")
SERVICE_CLIENTS = threshold(SERVICE, r"clients\s*>=\s*([\d.]+)")
SERVICE_MULTIPLE = threshold(SERVICE, r"times_busiest_day\s*>=\s*([\d.]+)")
GENERATION_DOMAINS = threshold(GENERATION, r"dns_domains_per_client\s*>\s*([\d.]+)")
GENERATION_NXDOMAIN = threshold(GENERATION, r"dns_nxdomain_ratio\s*>\s*([\d.]+)")
GENERATION_MULTIPLE = threshold(GENERATION, r"dns_domains_per_client_ratio\s*>=\s*([\d.]+)")
"""The layer 7 searches' own thresholds, read from the stanzas so the predicates this file evaluates are the ones
the app ships."""


def _layer_7_events() -> list:
    """What a search head would hold for `sourcetype=morpheus:score:l7`."""
    with open(os.path.join(EVENTS, "morpheus_score_l7.jsonlines"), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _beyond_baseline(event: dict, column: str, multiple: float) -> bool:
    """`(<column>_baseline_mature=false OR <column>_ratio>=multiple)`: a client with a week of history is held to a
    multiple of its own busiest day, and one without is held to the literal alone."""
    return event.get(f"{column}_baseline_mature") is False or (_number(event, f"{column}_ratio") or 0.0) >= multiple


def test_the_layer_7_detections_return_exactly_what_is_written(expected: dict):
    # Evaluated over the events the app is fed rather than the frame the pipeline produced. For R-B-L7-001 this is
    # not a formality: the search computes its own count, of distinct subdomains among queries clearing the other
    # conditions, which is a different figure from the stage's count of every subdomain. Only the search's
    # version holds all four conditions to the same traffic, and this is the place it is computed as the search
    # computes it -- per domain and hour, because a reputation agent's every day clears the literal over a week.
    events = _layer_7_events()
    searches = expected["searches"]

    candidates = [
        event for event in events if (event.get("dns_subdomain_entropy") or 0) > TUNNEL_ENTROPY and (
            event.get("dns_mean_label_length") or 0) > TUNNEL_LABEL_LENGTH
    ]
    qualifying = [
        event for event in candidates if _beyond_baseline(event, "dns_subdomains_per_domain", TUNNEL_MULTIPLE)
    ]
    hours: dict = collections.defaultdict(list)

    for event in qualifying:
        hours[(event["dns_registered_domain"], event["window_id"])].append(event)

    tunnels = {
        key: rows
        for (key, rows) in hours.items() if len({event["dns_subdomain"]
                                                 for event in rows}) > TUNNEL_SUBDOMAINS
    }
    entry = searches[TUNNEL]

    assert entry["candidate_rows_clearing_entropy_and_length"] == len(candidates)
    assert entry["candidate_rows_clearing_the_baseline"] == len(qualifying)
    assert entry["expected_rows"] == len(tunnels)

    key = (entry["key_values"]["dns_registered_domain"], entry["key_values"]["window_id"])
    rows = tunnels[key]

    assert entry["key_values"]["qualifying_subdomains"] == len({event["dns_subdomain"] for event in rows})
    assert entry["key_values"]["times_busiest_day"] == max(event["dns_subdomains_per_domain_ratio"] for event in rows)
    assert entry["contributing_rows"] == len(rows)
    assert entry["key_values"]["clients"] == len({event["src_ip"] for event in rows})

    # The reputation agent is the control the baseline exists for: it reaches the subdomain count on its event
    # day and is quiet only because that day is not twice its busiest.
    lookups = [event for event in candidates if event["dns_registered_domain"] == "filerep-av.net"]

    assert max(event["dns_subdomains_per_domain"] for event in lookups) > TUNNEL_SUBDOMAINS
    assert "filerep-av.net" not in {domain for (domain, _) in tunnels}

    # R-D-L7-005, read off the same row the way the search reads it, the ratio as a multiplication.
    refused = [
        event for event in events
        if (event.get("http_distinct_paths") or 0) > ENUMERATION_PATHS and event.get("http_4xx_in_window") is not None
        and event["http_4xx_in_window"] > ENUMERATION_RATIO * event["http_2xx_in_window"]
    ]
    enumerating = [event for event in refused if _beyond_baseline(event, "http_distinct_paths", ENUMERATION_MULTIPLE)]
    by_client: dict = collections.defaultdict(list)

    for event in enumerating:
        by_client[event["src_ip"]].append(event)

    entry = searches[ENUMERATION]

    assert entry["candidate_rows_before_the_baseline"] == len(refused)
    assert entry["contributing_rows"] == len(enumerating)
    assert entry["expected_rows"] == len(by_client)

    written = {
        row["src_ip"]: (row["distinct_paths"], row["refusals"], row["successes"], tuple(row["user_agent_families"]))
        for row in entry["key_values"]
    }
    derived = {
        client: (max(event["http_distinct_paths"] for event in rows),
                 max(event["http_4xx_in_window"] for event in rows),
                 max(event["http_2xx_in_window"] for event in rows),
                 tuple(sorted({event["http_ua_family"]
                               for event in rows})))
        for (client, rows) in by_client.items()
    }

    assert written == derived


def test_the_sensitive_data_detection_names_what_the_model_found_in_each_body(expected: dict):
    # A row predicate over the events the app is fed: `sid_labels=*` drops a request that carried no body, whose
    # field the serializer leaves out, and `len(sid_labels) > 0` drops a body the model read and found clean.
    found = [
        event for event in _layer_7_events()
        if event.get("window_complete") is True and event.get("sid_labels") not in (None, "")
    ]
    read_clean = [event for event in _layer_7_events() if event.get("sid_labels") == ""]
    entry = expected["searches"]["R-B-L7-003 - Sensitive data in transit"]

    assert entry["expected_rows"] == entry["contributing_rows"] == len(found)
    assert entry["key_values"] == sorted(({
        "src_ip": event["src_ip"], "sid_labels": event["sid_labels"]
    } for event in found),
                                         key=lambda row: row["sid_labels"])
    assert {event["src_ip"] for event in read_clean} == {"10.0.0.49"}
    assert all("request_body" not in row for row in entry["key_values"])


def test_the_service_and_generation_detections_return_exactly_what_is_written(expected: dict):
    # R-B-L7-007 aggregates per service and hour and then thresholds the aggregate, so the predicate is evaluated
    # on the groups; R-B-L7-008 is a row predicate and a `stats ... BY src_ip`.
    events = _layer_7_events()
    searches = expected["searches"]
    services: dict = collections.defaultdict(list)

    for event in events:
        if (event.get("service_key") and event.get("service_clients_baseline_mature") is True):
            services[(event["service_key"], event["window_id"])].append(event)

    def new_fingerprints(rows: list) -> list:
        return [event for event in rows if event.get("service_client_fingerprint_first_seen") is True]

    reached = {
        key: rows
        for (key, rows) in services.items()
        if max(event["service_clients_in_window"] for event in rows) >= SERVICE_CLIENTS and max(
            event["service_clients_ratio"] for event in rows) >= SERVICE_MULTIPLE and len(new_fingerprints(rows)) > 0
    }
    entry = searches[SERVICE]
    written = entry["key_values"]
    rows = reached[(written["service_key"], written["window_id"])]

    assert entry["expected_rows"] == len(reached)
    assert entry["contributing_rows"] == len(rows)
    assert written["clients"] == max(event["service_clients_in_window"] for event in rows)
    assert written["busiest_day"] == max(event["service_clients_baseline"] for event in rows)
    assert written["times_busiest_day"] == max(event["service_clients_ratio"] for event in rows)
    assert written["new_fingerprints"] == len(new_fingerprints(rows))
    assert written["new_client_software"] == sorted({event["ja4_client"] for event in new_fingerprints(rows)})
    assert {written["service_key_source"]} == {event["service_key_source"] for event in rows}
    assert {written["application"]} == {event["ctx_application_name"] for event in rows}
    assert {written["owning_team"]} == {event["ctx_application_owning_team"] for event in rows}

    generating = [
        event for event in events
        if (event.get("dns_domains_per_client") or 0) > GENERATION_DOMAINS and (event.get("dns_nxdomain_ratio") or 0) >
        GENERATION_NXDOMAIN and _beyond_baseline(event, "dns_domains_per_client", GENERATION_MULTIPLE)
    ]
    entry = searches[GENERATION]
    written = entry["key_values"]

    assert entry["contributing_rows"] == len(generating)
    assert entry["expected_rows"] == len({event["src_ip"] for event in generating})

    rows = [event for event in generating if event["src_ip"] == written["src_ip"]]

    assert written["registered_domains"] == max(event["dns_domains_per_client"] for event in rows)
    assert written["times_busiest_day"] == max(event["dns_domains_per_client_ratio"] for event in rows)
    assert written["nxdomain_share"] == max(event["dns_nxdomain_ratio"] for event in rows)
    assert written["nxdomain_answers"] == max(event["dns_nxdomain_in_window"] for event in rows)


STACK_FAMILIES = {
    "windows": "windows", "macos": "darwin", "ios": "darwin", "linux": "linux", "android": "linux", "chromeos": "linux"
}
"""R-B-L4-004's `case()`: the stack family each operating system a User-Agent can claim runs."""


def test_the_stack_mismatch_returns_exactly_what_is_written(expected: dict):
    # The search's two `stats ... BY` and its inner join on host_key, evaluated over the two sourcetypes it reads.
    stacks: dict = collections.defaultdict(list)
    claims: dict = collections.defaultdict(list)

    for event in _events_of("morpheus_score_l4.jsonlines"):
        if (event.get("host_key") and event.get("stack_os_family")):
            stacks[(event["host_key"], event["stack_os_family"])].append(event)

    for event in _layer_7_events():
        if (event.get("host_key") and event.get("http_ua_os")):
            claims[(event["host_key"], STACK_FAMILIES.get(event["http_ua_os"]))].append(event)

    mismatches = {
        (host, family, claimed): (rows, claims[(claimed_host, claimed)])
        for ((host, family), rows) in stacks.items()
        for (claimed_host, claimed) in claims if claimed_host == host and claimed is not None and claimed != family
    }
    entry = expected["searches"][STACK]
    written = entry["key_values"]

    assert entry["expected_rows"] == len(mismatches)

    (rows, requests) = mismatches[(written["host_key"], written["stack_os_family"], written["claimed_stack"])]

    assert written["stack_fingerprints"] == sorted({event["stack_fingerprint"] for event in rows})
    assert written["stack_new_to_host"] == max(int(event.get("stack_fingerprint_first_seen") is True) for event in rows)
    assert written["claimed_os"] == sorted({event["http_ua_os"] for event in requests})
    assert written["user_agent_families"] == sorted({event["http_ua_family"] for event in requests})


def test_the_issuer_anomaly_returns_exactly_what_is_written(expected: dict):
    # Grouped on the TLS destination, so an interception of one name on a shared address is its own notable.
    intercepted = [
        event for event in _events_of("morpheus_score_l6.jsonlines") if event.get("cert_issuer_differs") is True
        and event.get("cert_issuer_mature") is True and event.get("cert_issuer_distinct") == 1
    ]
    grouped: dict = collections.defaultdict(list)

    for event in intercepted:
        grouped[event["tls_destination_key"]].append(event)

    entry = expected["searches"]["R-D-L6-002 - Certificate issuer anomaly"]

    assert entry["contributing_rows"] == len(intercepted)
    assert entry["expected_rows"] == len(grouped)
    assert {(row["tls_destination_key"], row["presented_issuer"], row["established_issuer"])
            for row in entry["key_values"]} == {(key, rows[0]["certificate_issuer"], rows[0]["cert_issuer_established"])
                                                for (key, rows) in grouped.items()}


def test_the_self_signed_rule_returns_exactly_what_is_written(expected: dict):
    # R-D-L6-003 grouped on the destination, as the search groups it, with `count` and `min(cert_validity_days)` as
    # the search aggregates them. The self-signed certificates it leaves alone are the control: every one is on a
    # destination the classification calls not globally routable, which is the management interface nobody bought
    # a certificate for.
    handshakes = _events_of("morpheus_score_l6.jsonlines")
    external: dict = collections.defaultdict(list)

    for event in handshakes:
        if (event.get("cert_self_signed_external") is True):
            external[event["dst_ip"]].append(event)

    entry = expected["searches"]["R-D-L6-003 - Self-signed to external destination"]

    assert entry["contributing_rows"] == sum(len(rows) for rows in external.values())
    assert entry["expected_rows"] == len(external)
    assert [entry["key_values"]] == [{
        "dst_ip": dst_ip,
        "connections": len(rows),
        "shortest_validity_days": min(row["cert_validity_days"] for row in rows),
    } for (dst_ip, rows) in sorted(external.items())]

    internal = [
        event for event in handshakes
        if event.get("cert_self_signed") is True and event.get("cert_self_signed_external") is not True
    ]

    assert internal and all(event.get("cert_destination_is_global") is False for event in internal)
    assert len({event["dst_ip"] for event in internal}) == 1


def test_the_cipher_downgrade_returns_exactly_what_is_written(expected: dict):
    # R-B-L6-004 grouped on the pair, as the search groups it; `values()` of the tier and the floor are one value
    # each here, which is what the written entry states them as. The two controls the expectation names are in the
    # mature rows the rule leaves alone: a pair that moves between suites without ever falling below its floor, and
    # a pair sitting at a legacy floor permanently.
    handshakes = _events_of("morpheus_score_l6.jsonlines")
    downgraded: dict = collections.defaultdict(list)

    for event in handshakes:
        if (event.get("cipher_downgraded") is True and event.get("cipher_mature") is True):
            downgraded[(event["tls_pair_key"], event["src_ip"], event["dst_ip"])].append(event)

    entry = expected["searches"]["R-B-L6-004 - Cipher downgrade"]
    written = entry["key_values"]

    assert entry["contributing_rows"] == sum(len(rows) for rows in downgraded.values())
    assert entry["expected_rows"] == len(downgraded)

    ((pair, _, _), rows), = downgraded.items()

    assert pair == written["tls_pair_key"] == f"{rows[0]['src_ip']}:{rows[0]['dst_ip']}"
    assert {row["cipher_tier"] for row in rows} == {written["negotiated_tier"]}
    assert {row["cipher_floor_tier"] for row in rows} == {written["historical_floor"]}

    quiet = [
        event for event in handshakes if event.get("cipher_mature") is True and event.get("cipher_downgraded") is False
    ]
    suites: dict = collections.defaultdict(set)

    for event in quiet:
        suites[event["tls_pair_key"]].add(event["cipher_suite"])

    assert any(len(negotiated) > 1 for negotiated in suites.values()), "the varied host"
    assert any(event["cipher_tier"] == event["cipher_floor_tier"] == "legacy" for event in quiet), "the appliance"


def test_the_content_mismatch_returns_exactly_what_is_written(expected: dict):
    # R-D-L6-005 grouped on the source and destination, as the search groups it. Every transfer the comparison
    # reached is accounted for: the one that crossed a category, and the re-encodings, whose two types differ
    # inside one category and which a rule comparing types rather than categories would return as well.
    handshakes = _events_of("morpheus_score_l6.jsonlines")
    crossed: dict = collections.defaultdict(list)

    for event in handshakes:
        if (event.get("content_category_crossed") is True):
            crossed[(event["src_ip"], event["dst_ip"])].append(event)

    entry = expected["searches"]["R-D-L6-005 - Content type mismatch"]
    written = entry["key_values"]

    assert entry["contributing_rows"] == sum(len(rows) for rows in crossed.values())
    assert entry["expected_rows"] == len(crossed)

    ((src_ip, dst_ip), rows), = crossed.items()

    assert (src_ip, dst_ip) == (written["src_ip"], written["dst_ip"])

    for (column, name) in (("content_type_declared", "declared_type"), ("content_type_detected", "detected_type"),
                           ("content_category_declared", "declared_category"), ("content_category_detected",
                                                                                "detected_category")):
        assert {row[column] for row in rows} == {written[name]}, name

    compared = [
        event for event in handshakes if event.get("content_type_declared") and event.get("content_type_detected")
    ]
    recoded = [event for event in compared if event.get("content_category_crossed") is False]

    assert len(recoded) + len(rows) == len(compared)
    assert all(event["content_type_declared"] != event["content_type_detected"] for event in compared)
    assert recoded and all(event["content_category_declared"] == event["content_category_detected"]
                           for event in recoded)


def test_the_tls_table_coverage_counts_exactly_what_is_written(expected: dict):
    # One summary row over every complete layer 6 row, and the value is in the numbers. The search sums
    # `if(<column>="true", 1, 0)`, which over a column the wire dropped is zero -- the same answer as a corpus
    # entirely inside the two tables -- so the two columns are asserted present on every row, as booleans, before
    # their counts are. Written as 105 handshakes when the layer 6 corpus was 105 rows; the recorded run reports the
    # one row and not its numbers, and nothing here had derived them until now.
    handshakes = [event for event in _events_of("morpheus_score_l6.jsonlines") if event.get("window_complete") is True]
    entry = expected["searches"]["TLS table coverage - unrecognized values"]
    written = entry["key_values"]
    search = _stanza_search("TLS table coverage - unrecognized values")

    for column in ("cipher_unrecognized", "content_category_unclassified"):
        assert f'if({column}="true", 1, 0)' in search, column
        assert all(isinstance(event.get(column), bool) for event in handshakes), column

    assert entry["expected_rows"] == 1 and " BY " not in search.split("| stats", 1)[1].split("|", 1)[0]
    assert entry["contributing_rows"] == written["handshakes"] == len(handshakes)
    assert written["unrecognized_ciphers"] == sum(1 for event in handshakes if event["cipher_unrecognized"] is True)
    assert written["unclassified_content_types"] == sum(1 for event in handshakes
                                                        if event["content_category_unclassified"] is True)


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
    # novelty flag the trigger and the integrity level the severity, with no level at the default, and a host on the
    # hosts' list raised by the boost. The list is the predictive searches' entries, which the runner dispatches
    # first and expires last.
    severity = {"system": 70, "high": 55, "medium": 40, "low": 25}
    scale = {"high": 1.5, "low": 0.5}
    boost = threshold("R-B-L7-004 - Process ancestry novelty", r"watchlist_boost\s*=\s*(\d+)")
    watched = {entry["host_key"] for entry in _host_watchlist()}
    novel = [event for event in _layer_7_events() if event.get("endpoint_pair_novel") is True]
    derived = sorted(({
        "endpoint_host":
            event["endpoint_host"],
        "integrity":
            event.get("endpoint_integrity") or "unknown",
        "criticality":
            event.get("ctx_criticality") or "unknown",
        "endpoint_host_only":
            event["endpoint_host_only"],
        "risk_score":
            round(severity.get(event.get("endpoint_integrity"), 40) * scale.get(event.get("ctx_criticality"), 1.0)) +
            (boost if event["endpoint_host"] in watched else 0),
    } for event in novel),
                     key=lambda row: (row["endpoint_host"], row["risk_score"]))
    entry = expected["searches"]["R-B-L7-004 - Process ancestry novelty"]

    assert entry["contributing_rows"] == len(novel)
    assert entry["expected_rows"] == len(derived)
    assert entry["key_values"] == derived


def test_the_lateral_movement_chain_returns_exactly_what_is_written(expected: dict):
    # Evaluated over every flow, login and process the app is fed, from every corpus, the way the search reads
    # them -- not only over the campaign's -- with the same predicate the campaign harness asserts.
    events = [
        event for event in _scored_events()
        if event.get("telemetry_class") in (campaign_pipeline.FLOW_CLASS, campaign_pipeline.AUTH_CLASS,
                                            campaign_pipeline.PROCESS_CLASS)
    ]
    frame = pd.DataFrame(events)
    stamps = pd.to_datetime(frame["event_time"].str.replace("UTC", "", regex=False), utc=True)
    frame["event_time"] = (stamps - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(nanoseconds=1)

    chains = sorted(({
        "src_ip": source, "user_principal": principal, "target": target
    } for (source, principal, target) in campaign_pipeline.lateral_movement(frame)),
                    key=lambda row: (row["src_ip"], row["user_principal"]))
    entry = expected["searches"]["R-C-001 - Lateral movement chain"]

    assert entry["expected_rows"] == len(chains)
    assert entry["key_values"] == chains


def test_the_staged_exfiltration_chain_returns_exactly_what_is_written(expected: dict):
    # Evaluated over every export, session, transfer and handshake the app is fed, from every corpus, with the
    # predicate the campaign harness asserts.
    events = [
        event for event in _scored_events()
        if event.get("telemetry_class") in (campaign_pipeline.SAAS_CLASS, campaign_pipeline.SESSION_CLASS,
                                            campaign_pipeline.TRANSFER_CLASS, campaign_pipeline.HANDSHAKE_CLASS)
    ]
    frame = pd.DataFrame(events)
    stamps = pd.to_datetime(frame["event_time"].str.replace("UTC", "", regex=False), utc=True)
    frame["event_time"] = (stamps - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(nanoseconds=1)

    # The join method as the search reports it: a session's own key where every step carried one, the address
    # otherwise.
    chains = sorted(({
        "user_principal": principal, "src_ip": address, "join_method": method
    } for ((principal, address), (*_, method)) in campaign_pipeline.exfiltration_chains(frame).items()),
                    key=lambda row: (row["user_principal"], row["src_ip"]))
    entry = expected["searches"]["R-C-004 - Staged exfiltration"]

    assert entry["expected_rows"] == len(chains)
    assert entry["key_values"] == chains


def _events_of(sourcetype_file: str) -> list:
    with open(os.path.join(EVENTS, sourcetype_file), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_the_fingerprint_and_beacon_detections_count_exactly_what_is_written(expected: dict):
    # Evaluated over the events the app is fed, the way the two searches read them. Neither count was checked
    # against the events until the campaign corpus added hosts to both; a written expectation that nothing
    # re-derives goes stale the first time another corpus shares the sourcetype.
    searches = expected["searches"]
    fingerprints = [
        event for event in _events_of("morpheus_score_l6.jsonlines") if event.get("ja4_client_first_seen") is True and (
            event.get("ja4_client_observations") or 0) >= 20 and event.get("ctx_found") is True
    ]
    entry = searches["R-B-L6-001 - New TLS client fingerprint"]

    assert entry["contributing_rows"] == len(fingerprints)
    assert entry["expected_rows"] == len({event["src_ip"] for event in fingerprints})
    assert entry["key_values"]["src_ip"] in {event["src_ip"] for event in fingerprints}

    def regular(value) -> bool:
        return value is not None and value < 0.15

    beacons = [
        event for event in _events_of("morpheus_score_l3.jsonlines") if event.get("flow_regularity_mature") is True
        and regular(event.get("flow_interval_cv")) and regular(event.get("flow_size_cv"))
    ]
    entry = searches["R-B-L3-002 - Beaconing"]

    assert entry["contributing_rows"] == len(beacons)
    assert entry["expected_rows"] == len({event["flow_pair_key"] for event in beacons})
    assert entry["key_values"] == sorted({event["flow_pair_key"] for event in beacons})


def test_the_reserved_range_egress_returns_exactly_what_is_written(expected: dict):
    # R-D-L3-003 writes one risk record per flow, classed as the search's `if()` classes it: reserved first,
    # multicast otherwise. The flows it leaves alone are asserted classified rather than merely absent, because a
    # stage that stopped writing the two flags would leave the base clause matching nothing, which reads as a
    # clean corpus.
    flows = _events_of("morpheus_score_l3.jsonlines")
    egress = [event for event in flows if event.get("dst_is_reserved") is True or event.get("dst_is_multicast") is True]
    ordinary = [
        event for event in flows if event.get("dst_is_reserved") is False and event.get("dst_is_multicast") is False
    ]
    entry = expected["searches"]["R-D-L3-003 - Reserved-range egress"]

    assert entry["expected_rows"] == len(egress)
    assert entry["key_values"] == [{
        "src_ip": event["src_ip"],
        "dst_ip": event["dst_ip"],
        "destination_class": "reserved" if event["dst_is_reserved"] is True else "multicast",
    } for event in egress]
    assert len(ordinary) + len(egress) == len(flows), "a flow the classification did not decide"


def test_the_ttl_shift_returns_exactly_what_is_written(expected: dict):
    # R-B-L3-004's `stats ... BY src_ip` over the shifted, mature rows, then its `where` on the aggregate, with the
    # floor read from the stanza. The aggregate is what makes it a statement about a distribution: a source whose
    # TTL moved on fewer flows than the floor is in the data and is not reported.
    shifted: dict = collections.defaultdict(list)

    for event in _events_of("morpheus_score_l3.jsonlines"):
        if (event.get("ip_ttl_shifted") is True and event.get("ip_ttl_mature") is True):
            shifted[event["src_ip"]].append(event)

    sources = {src_ip: rows for (src_ip, rows) in shifted.items() if len(rows) >= TTL_SHIFTED_FLOWS}
    entry = expected["searches"][TTL_SHIFT]

    assert entry["contributing_rows"] == sum(len(rows) for rows in sources.values())
    assert entry["expected_rows"] == len(sources)
    assert entry["key_values"] == [{
        "src_ip": src_ip,
        "shifted_flows": len(rows),
        "shifts": sorted({row["ip_ttl_shift"]
                          for row in rows}),
        "established_ttl": max(row["ip_ttl_established"] for row in rows),
        "distinct_ttls": max(row["ip_ttl_distinct"] for row in rows),
    } for (src_ip, rows) in sorted(sources.items())]

    # The control the expectation names never moves: every source with a mature TTL that is not reported has no
    # shifted row at all, so the floor is not what keeps it quiet.
    assert not {src_ip for src_ip in shifted if src_ip not in sources}


def test_the_layer_3_host_baseline_detections_return_exactly_what_is_written(expected: dict):
    # Evaluated over every flow the app is fed, the way the four searches read them: the three that measure a host
    # against its own fortnight, and the port sweep. Flows from the campaign carry no baseline, so they reach none.
    flows = _events_of("morpheus_score_l3.jsonlines")

    def number(event, column) -> float:
        value = event.get(column)

        return float("nan") if value is None else float(value)

    fan_out = [
        event for event in flows
        if event.get("dsts_per_src_baseline_mature") is True and number(event, "dsts_per_src_step") > 0
        and number(event, "dsts_per_src") > 50 and number(event, "internal_dst_ratio") > 0.5
    ]
    fan_in = [
        event for event in flows
        if event.get("dst_ctx_device_role") == "workstation" and event.get("srcs_per_dst_baseline_mature") is True
        and number(event, "srcs_per_dst_step") > 0 and number(event, "srcs_per_dst") >= 10
    ]
    upload = [
        event for event in flows
        if event.get("dst_asn_first_seen") is True and event.get("byte_asymmetry_baseline_mature") is True
        and number(event, "byte_asymmetry_step") > 0 and number(event, "byte_asymmetry") >= 10
    ]
    sweep = [
        event for event in flows if number(event, "dst_ports_per_src") >= 25 and event.get("dst_is_private") is True
    ]

    for (name, rows, keys) in (("R-B-L3-001 - Fan-out expansion", fan_out,
                                ("src_ip", )), ("R-B-L3-006 - Fan-in onto a workstation", fan_in, ("dst_ip", )),
                               ("R-B-L3-007 - First-contact upload", upload,
                                ("src_ip", "bgp_as_dst")), ("R-D-L3-008 - Port sweep", sweep, ("src_ip", ))):
        entry = expected["searches"][name]
        grouped = sorted({tuple(event[key] for key in keys) for event in rows})

        assert entry["contributing_rows"] == len(rows), name
        assert entry["expected_rows"] == len(grouped), name
        assert [list(group) for group in grouped] == entry["key_values"], name


def test_the_connection_evidence_report_returns_exactly_what_is_written(expected: dict):
    # Every scored record on layers 3, 4 and 6 that carries a Community ID, grouped on it, kept where all three
    # layers saw the connection.
    layers: dict = collections.defaultdict(set)

    for sourcetype_file in ("morpheus_score_l3.jsonlines", "morpheus_score_l4.jsonlines",
                            "morpheus_score_l6.jsonlines"):
        for event in _events_of(sourcetype_file):
            if (event.get("community_id")):
                layers[event["community_id"]].add(event["osi_layer"])

    joined = sorted(community_id for (community_id, seen) in layers.items() if len(seen) == 3)
    entry = expected["searches"]["Connection evidence - layers 3, 4 and 6"]

    assert entry["expected_rows"] == len(joined)
    assert entry["key_values"] == joined
    assert entry["connections_seen_at_two_layers"] == sum(1 for seen in layers.values() if len(seen) == 2)


def test_the_host_evidence_report_returns_exactly_what_is_written(expected: dict):
    # Every scored record on layers 3, 4, 6 and 7 that resolved to a machine, grouped on it, kept where two layers
    # or more saw the machine. Written as seventeen through a step that added two machines at layers 4 and 7; the
    # search head counted nineteen, and nothing here had derived it.
    layers: dict = collections.defaultdict(set)

    for sourcetype_file in ("morpheus_score_l3.jsonlines",
                            "morpheus_score_l4.jsonlines",
                            "morpheus_score_l6.jsonlines",
                            "morpheus_score_l7.jsonlines"):
        for event in _events_of(sourcetype_file):
            if (event.get("host_key") and event.get("window_complete") is True):
                layers[event["host_key"]].add(event["osi_layer"])

    machines = sorted(host for (host, seen) in layers.items() if len(seen) >= 2)
    entry = expected["searches"]["Host evidence - one machine across layers"]

    assert entry["expected_rows"] == len(machines)
    assert entry["key_values"] == machines


def test_the_command_and_control_chain_returns_exactly_what_is_written(expected: dict):
    events = [
        event for event in _scored_events()
        if event.get("telemetry_class") in (campaign_pipeline.FLOW_CLASS, campaign_pipeline.HANDSHAKE_CLASS)
    ]
    frame = pd.DataFrame(events)
    stamps = pd.to_datetime(frame["event_time"].str.replace("UTC", "", regex=False), utc=True)
    frame["event_time"] = (stamps - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(nanoseconds=1)

    chains = sorted(({
        "src_ip": source, "dst_ip": destination, "join_method": method
    } for ((source, destination), (*_, method)) in campaign_pipeline.c2_chains(frame).items()),
                    key=lambda row: (row["src_ip"], row["dst_ip"]))
    entry = expected["searches"]["R-C-002 - TLS anomaly precedes beaconing"]

    assert entry["expected_rows"] == len(chains)
    assert entry["key_values"] == chains


def test_the_credential_replay_returns_exactly_what_is_written(expected: dict):
    # Read from the indexed sign-ins, so a column the search needs and the wire dropped shows up here as a miss.
    events = [event for event in _scored_events() if event.get("telemetry_class") == campaign_pipeline.AUTH_CLASS]
    frame = pd.DataFrame(events)
    stamps = pd.to_datetime(frame["event_time"].str.replace("UTC", "", regex=False), utc=True)
    frame["event_time"] = (stamps - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(nanoseconds=1)

    replays = sorted(({
        "user_principal": principal, "previous_port": previous_port, "login_port_key": port
    } for (principal, (previous_port, port, _)) in campaign_pipeline.credential_replays(frame).items()),
                     key=lambda row: row["user_principal"])
    entry = expected["searches"]["R-C-005 - Credential replay across the stack"]

    assert entry["expected_rows"] == len(replays)
    assert entry["key_values"] == replays


def test_the_physical_compromise_chain_returns_exactly_what_is_written(expected: dict):
    # Read from the indexed port polls, table walks and flows, so a column one of the three joins needs and the wire
    # dropped shows up here as a miss. The tolerance and the window are the stanza's.
    classes = (campaign_pipeline.PORT_CLASS, campaign_pipeline.MAC_CLASS, campaign_pipeline.FLOW_CLASS)
    events = [event for event in _scored_events() if event.get("telemetry_class") in classes]
    frame = pd.DataFrame(events)
    stamps = pd.to_datetime(frame["event_time"].str.replace("UTC", "", regex=False), utc=True)
    frame["event_time"] = (stamps - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(nanoseconds=1)

    fired = campaign_pipeline.physical_compromises(frame,
                                                   window_seconds=PHYSICAL_WINDOW_SECONDS,
                                                   tolerance_seconds=PHYSICAL_TOLERANCE_SECONDS)
    entry = expected["searches"][PHYSICAL]

    assert entry["expected_rows"] == len(fired)
    assert [(row["port_key"], row["mac_address"]) for row in entry["key_values"]] == sorted(fired)


def _scored_events() -> list:
    # What a search head would hold for `sourcetype=morpheus:score:*`: the seven layers' rows and the host
    # segment's. The sourcetype is the filename with the colons swapped, which is how the generator writes them, so
    # the glob here is the search's glob.
    events = []

    for name in sorted(os.listdir(EVENTS)):
        if (not name.startswith("morpheus_score_") or not name.endswith(".jsonlines")):
            continue

        with open(os.path.join(EVENTS, name), encoding="utf-8") as handle:
            events.extend(json.loads(line) for line in handle if line.strip())

    return events


def _events_read_by(name: str) -> list:
    """
    The scored events a search's base search reads, by its sourcetype glob: `morpheus:score:l*` is the seven layers'
    rows and `morpheus:score:*` those and the host segment's. Chain assembly reads the former, because the host
    frame is one layer with nothing to chain; the summary reads the latter, so a machine has a `peak_z`.
    """
    glob = re.search(r"sourcetype=morpheus:score:(\S+)", _search_text(name)).group(1)
    prefix = "morpheus_score_" + glob.rstrip("*")

    return [
        event for name in sorted(os.listdir(EVENTS)) if name.startswith(prefix) and name.endswith(".jsonlines")
        for event in _events_of(name)
    ]


def _five_minute_bin(event_time: str) -> int:
    stamp = datetime.datetime.strptime(event_time.replace("UTC", ""), "%Y-%m-%dT%H:%M:%S.%f")
    seconds = int(stamp.replace(tzinfo=datetime.timezone.utc).timestamp())

    return seconds - (seconds % 300)


def test_the_behavior_summary_groups_exactly_what_is_written(expected: dict):
    # `stats ... by` drops a row that is missing any grouping field, so a missing envelope field does not show up
    # as an error anywhere -- it shows up as a smaller number, or as zero. That is why this is evaluated rather
    # than asserted: the row count is the only place the omission would surface.
    events = _events_read_by("Behavior summary - per-layer scores")
    entry = expected["searches"]["Behavior summary - per-layer scores"]

    assert len(events) == len(_scored_events()), "the summary no longer reads every score sourcetype"

    keyless = [event for event in events if event.get("osi_layer") is None or not event.get("entity_key")]
    assert keyless == [], f"{len(keyless)} scored events would be dropped by the grouping"

    groups = {(_five_minute_bin(event["event_time"]), event["osi_layer"], event["entity_key"], event["lineage_id"])
              for event in events}

    assert entry["contributing_rows"] == len(events)
    assert entry["expected_rows"] == len(groups)
    assert entry["expected_empty"] is False


def _fan_out_trajectory(events: list) -> list:
    """
    R-P-L3-005 over the summary the Behavior summary search would collect from these events: the layer 3 rows,
    grouped as that search groups them, with `peak_destinations` the bin's largest `dsts_per_src`, then two passes
    of `streamstats current=f last(...)` per entity in `sort 0 entity_key _time lineage_id` order. `last` skips
    nulls, so `previous` is the last peak seen before a row and `earlier` the last `previous`.
    """
    peaks = {}

    for event in events:
        if (event.get("osi_layer") != 3):
            continue

        key = (event["entity_key"], _five_minute_bin(event["event_time"]), event["lineage_id"])
        value = event.get("dsts_per_src")
        current = peaks.setdefault(key, None)

        if (value is not None):
            peaks[key] = value if current is None else max(current, value)

    fired = []
    (entity, previous, earlier) = (None, None, None)

    for key in sorted(peaks):
        if (key[0] != entity):
            (entity, previous, earlier) = (key[0], None, None)

        peak = peaks[key]

        if (None not in (peak, previous, earlier) and peak > previous > earlier):
            fired.append((key, earlier, previous, peak))

        (previous, earlier) = (peak if peak is not None else previous, previous if previous is not None else earlier)

    return fired


def test_the_fan_out_trajectory_returns_exactly_what_is_written(expected: dict):
    # Empty on the first two search-head runs, and not only because the summary had not been collected: the rule
    # read the value two rows back as `last(previous_destinations)` inside the same streamstats that created
    # `previous_destinations`, which Splunk evaluates as null on every row. The search now takes two passes, and
    # the runner dispatches it after the summary is searchable.

    search = _search_text("R-P-L3-005 - Fan-out trajectory")
    passes = re.findall(r"streamstats current=f last\((\w+)\) AS (\w+) BY entity_key", search)

    assert passes == [("peak_destinations", "previous_destinations"), ("previous_destinations", "earlier_destinations")]

    fired = _fan_out_trajectory(_scored_events())
    entry = expected["searches"]["R-P-L3-005 - Fan-out trajectory"]

    assert entry["expected_rows"] == len(fired) > 0
    assert entry["expected_empty"] is False
    assert entry["by_source"] == dict(sorted(collections.Counter(key[0] for (key, _, _, _) in fired).items()))

    # The planted scanner's hours climb 5, 20, 45, 80; in five-minute bins that is the rising run the rule reads.
    scanner = [(earlier, previous, peak) for (key, earlier, previous, peak) in fired if key[0] == "10.0.0.50"]

    assert scanner and all(earlier < previous < peak for (earlier, previous, peak) in scanner)


def test_the_chain_assembly_returns_the_tapped_desk_and_not_the_flooding_host(expected: dict):
    # The search was empty for four reasons in turn: a missing `osi_layer`, single-layer lineage, a risk sum
    # nothing wrote, and a risk sum that was written and too small, because the two detections that accused a
    # three-layer chain accused different chains. The desk spoof puts the ARP anomaly's 55 on the tapped port's
    # chain beside R-D-L1-002's 50, so one chain reaches the 60 the search needs and the flooding host's, at 55,
    # stays the quiet case beside it. R-D-L1-010's 30, the tapped port read against its commissioning level, lands
    # on the same chain. Both are asserted from the events, as is the risk on every three-layer chain, and the
    # search-head run records the per-chain totals.

    entry = expected["searches"]["Chain assembly - cross-layer risk"]
    chain = _search_text("Chain assembly - cross-layer risk")
    events = _events_read_by("Chain assembly - cross-layer risk")

    with open(os.path.join(EVENTS, "morpheus_edge.jsonlines"), encoding="utf-8") as handle:
        edges = [json.loads(line) for line in handle if line.strip()]

    method_field = re.search(r"values\((\w+)\)\s+AS\s+methods", chain).group(1)
    edge_field = re.search(r"values\((\w+)\)\s+AS\s+edge_methods", chain).group(1)
    risk_threshold = float(re.search(r"total_risk\s*>=\s*([\d.]+)", chain).group(1))
    layers = collections.defaultdict(set)
    methods = collections.defaultdict(set)

    for event in events + edges:
        if (event.get("lineage_id") is not None):
            layers[event["lineage_id"]].add(event.get("osi_layer"))

            # `makemv delim=";"` before the stats: one value per hop the ladder took. An edge names its one join.
            if (event.get(method_field) is not None):
                methods[event["lineage_id"]].update(event[method_field].split(";"))

            if (event.get(edge_field) is not None):
                methods[event["lineage_id"]].add(event[edge_field])

    spans = {lineage: len(seen - {None}) for (lineage, seen) in layers.items()}
    three_layer = {lineage for (lineage, span) in spans.items() if span >= 3}

    assert entry["distinct_lineage_ids"] == len(layers)
    assert entry["maximum_layer_span"] == max(spans.values()) == 3
    assert entry["two_layer_chains"] == sum(1 for span in spans.values() if span == 2)
    assert entry["three_layer_chains"] == len(three_layer)
    assert entry["chains_with_a_method"] == len(methods)
    assert entry["three_layer_chain_methods"] == sorted(
        {method
         for lineage in three_layer
         for method in methods[lineage]})

    # Every layer 1 and 2 detection's risk records, as the stanzas state them: a record adds its score once to each
    # lineage it names, which for a rule that groups is every lineage its group's rows were on. These are the
    # detections whose records can land on a chain spanning three layers, since those chains are the estate's.
    risk = collections.Counter()

    for rule in LAYER_1_AND_2:
        score = _risk_score(rule)

        for rows in _notables(rule):
            for lineage in {row["lineage_id"] for row in rows if row.get("lineage_id") is not None}:
                risk[lineage] += score

    totals = collections.Counter(str(int(risk.get(lineage, 0))) for lineage in three_layer)
    accused = {lineage: risk[lineage] for lineage in three_layer if risk.get(lineage, 0) >= risk_threshold}

    assert entry["three_layer_chain_risk"] == dict(sorted(totals.items(), key=lambda item: int(item[0])))
    assert entry["expected_rows"] == len(accused) == 1 and entry["expected_empty"] is False

    # The one row is Carol's port in the window the tap and the desk spoof share, accused by all three rules, and the
    # flooding host's chain is the one at 55: the threshold, not the absence of a second accusation, keeps it out.
    (lineage, total) = next(iter(accused.items()))
    rules = sorted({
        rule
        for rule in LAYER_1_AND_2
        for rows in _notables(rule) if any(row.get("lineage_id") == lineage for row in rows)
    })
    row = entry["key_values"][0]

    assert (row["lineage_id"], row["total_risk"], row["layer_span"]) == (lineage, total, spans[lineage])
    assert row["rules"] == rules == ["R-D-L1-002", "R-D-L1-010", "R-D-L2-003"]
    assert row["methods"] == sorted(methods[lineage])
    assert sorted(risk[lineage] for lineage in three_layer if 0 < risk.get(lineage, 0) < risk_threshold) == [55]


def test_the_validation_document_says_the_same_thing_the_expectation_file_does(expected: dict):
    # VALIDATION.md is the page a deployment actually reads; `expected_results.json` is the page the tests read.
    # Two documents describing one run, maintained by hand, is the shape that has drifted in five places in this
    # repository -- and it drifted here: the summary table went a whole increment without the layer 3 searches
    # in it, still saying "four of the fourteen" over a row count from two increments earlier. Every number in
    # that table now comes from the same file these assertions do.

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
    "one": 1,
    "two": 2,
    "three": 3,
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
    "thirty-six": 36,
    "thirty-seven": 37,
    "thirty-eight": 38,
    "thirty-nine": 39,
    "forty": 40,
    "forty-one": 41,
    "forty-two": 42,
    "forty-three": 43,
    "forty-four": 44,
    "forty-five": 45,
    "forty-six": 46,
    "forty-seven": 47,
    "forty-eight": 48,
    "forty-nine": 49,
    "fifty": 50,
    "fifty-one": 51,
    "fifty-two": 52,
    "fifty-three": 53,
    "fifty-four": 54,
    "fifty-five": 55,
    "fifty-six": 56,
    "fifty-seven": 57,
    "fifty-eight": 58,
    "fifty-nine": 59,
    "sixty": 60,
    "sixty-one": 61,
    "sixty-two": 62,
    "sixty-six": 66,
    "seventy-two": 72,
    "seventy-five": 75,
    "seventy-eight": 78,
    "seventy-nine": 79,
    "eighty": 80,
    "eighty-one": 81,
    "eighty-two": 82,
    "eighty-three": 83,
    "eighty-four": 84,
    "eighty-five": 85,
    "eighty-six": 86,
}
"""Only the range these two counts can plausibly take. A word outside it fails with a `KeyError` naming the word,
which is the right failure: the document said something nobody here anticipated."""


def test_every_expected_empty_search_says_why(expected: dict):
    empty = {name: entry for (name, entry) in expected["searches"].items() if entry.get("expected_empty")}

    # Seven of thirty-four, when this was written. That ratio is the honest state of this app, and stating it is the
    # package's main job.
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
    #
    # Six of thirty-seven now: R-C-002 left the list when it was rewritten to read scored events rather than
    # notables and the campaign corpus gave it a chain to find.
    # Six of thirty-eight after R-C-005, which arrived with a chain to find, six of thirty-nine after
    # R-D-L1-001, whose optic swap the corpus already held, six of forty after R-P-L1-004, which arrived with a
    # failing optic planted for it, and six of forty-one after R-B-L2-002, whose hub the corpus already held.
    #
    # Nine of forty-seven after the layer 5 baseline searches, and every one of the three new empties is correct
    # for a reason it states: R-B-L5-001 and R-B-L5-002 read only rows a principal's own model scored, and none
    # here was; the watchlist expiry drops entries for events that are years old. Nine of forty-eight after
    # R-B-L5-005, which arrived with a long session planted for it.
    #
    # Eight of forty-eight after the first search-head run, and the change is a correction rather than a rule: the
    # watchlist expiry was empty only against 1970 timestamps, and the runner dates the events so R-P-L7-006's two
    # entries are still inside their thirty days when it runs.
    #
    # Seven after the risk write path, and the one that left was broken rather than starved: R-P-L3-005 read the
    # value two rows back from a field its own streamstats was still creating, and fires fifteen times now that
    # it takes two passes over a summary the run collects first.
    #
    # Five after the learned models, and the two that left were waiting for exactly that: R-B-L5-001 and R-B-L5-002
    # read only rows a principal's own model scored, and each principal now has one, trained on a fortnight the
    # scored week follows.
    #
    # Five of sixty-two after step 10, whose ten searches each arrived with something planted for it to find.
    #
    # Two after step 12. The L2/L3 refresh left: `TC2BindingStage` closes the campaign DHCP server's
    # acknowledgements into leases, and their buckets are what the refresh selects. The expiry jobs left with it,
    # and their reason had been wrong all along: an expiry rewrites the lookup with every row it keeps, and it had
    # kept none only because each refresh keyed its rows on a bucket counted from 1970. Keyed on the bucket the
    # row's own time falls in, which a validation run moves to the weeks before it, every row is kept.
    #
    # One after the desk spoof. Chain assembly left last, and not because a threshold moved: the two detections
    # that accused a three-layer chain had accused different chains, and the corpus now holds the actor the search
    # exists for, a tapped desk whose workstation claims the gateway. The one that remains, R-D-L2-001, waits for
    # the designation list only an estate can supply.
    assert len(empty) == 1

    for (name, entry) in empty.items():
        assert entry["expected_rows"] == 0, name
        assert len(entry["why"]) > 60, f"{name}: an expected-empty search needs a reason, not a shrug"


SUPERSEDED_PORTS = sorted([(tp.SWITCH, port) for port in list(tp.SWAPS) + [tp.REPATCH_PORT, tp.INLINE_PORT]] +
                          [(tp.REBOOTING_SWITCH, port) for port in tp.REBOOT_PORTS])
"""The ports whose bindings something superseded: two optics replaced, a re-patch, an insertion, and the restarted
switch's ports, whose neighbour was unknown for the first poll after it came back."""


def test_the_port_history_holds_what_was_superseded_and_nothing_else(telemetry: pd.DataFrame):
    # The layer 1 lookup used to answer every question in the present tense: keyed on the port alone, a port whose
    # optic is replaced collapses to one row and an investigation into last Tuesday gets Wednesday's optic. The
    # history lookup is the other tense, and its value is as much in what it leaves out as in what it holds.
    ports = tp.build_port_binding_table(telemetry[telemetry["telemetry_class"] == "tc1_binding"])

    superseded = ports.superseded()

    # One interval on each port somebody re-cabled or re-fitted, and two on each port of the restarted switch: its
    # old neighbour, then one poll of none. The ports nobody touched are described for all time by the row the
    # current-state lookup holds, and cost the history collection nothing.
    assert superseded.key_count == len(SUPERSEDED_PORTS)
    assert superseded.size == len(tp.SWAPS) + 2 + 2 * len(tp.REBOOT_PORTS)

    last = tp.CORPUS_SECONDS * tp.NS_PER_SECOND - 1
    optic = tp.PORT_INVENTORY_COLUMNS.index("transceiver_serial")

    for (port_id, minute) in tp.SWAPS.items():
        swapped = f"{tp.SITE}:{tp.SWITCH}:{port_id}"
        before = (minute - 10) * 60 * tp.NS_PER_SECOND

        # The two tenses disagree, which is the whole point. Resolving the port before the swap through the
        # history gives the optic that was in it; the current-state answer for the same port is the one in it now.
        assert superseded.resolve(swapped, before).values[optic] == f"XCVR-{tp.SWITCH}-{port_id}"
        assert ports.resolve(swapped, last).values[optic] == f"XCVR-{tp.SWITCH}-{port_id}-B"


def test_the_history_the_app_receives_is_one_day_bucket_per_superseded_port():
    # What the search head is actually fed, rather than what the pipeline could produce. The rows ride the shared
    # bucketed sourcetype and the refresh tells them apart by `binding_table`, so a rename upstream would leave
    # the L1 history refresh reading the layer 2 bindings, which carry neither a port nor an optic.
    with open(os.path.join(EVENTS, "binding_bucketed.jsonlines"), encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]

    history = sorted((row for row in rows if row["binding_table"] == "port_inventory"),
                     key=lambda row: (row["switch_id"], row["port_id"]))

    # One row per port and day, so the restarted switch's two short intervals per port share their port's row.
    assert [(row["switch_id"], row["port_id"]) for row in history] == SUPERSEDED_PORTS

    for row in history:
        # Every superseded interval held the port's first optic: only the two swaps changed one.
        assert row["transceiver_serial"] == f"XCVR-{row['switch_id']}-{row['port_id']}"

        # The cost of the approximation, stated rather than discovered. The corpus is one hour and the bucket is
        # a day, so both of a port's intervals fall in bucket zero and every instant in it resolves to the earlier
        # optic -- right for the minutes before the swap, wrong for the minutes after. That error is bounded by
        # the bucket width, where the unbucketed lookup's error was bounded by nothing at all.
        assert row["bucket"] == 0

    # The history is additive. The interval stream the current-state refresh reads still carries both of each
    # swapped port's intervals, so the two lookups are built from the same records rather than one taking rows
    # away from the other.
    with open(os.path.join(EVENTS, "binding_l1.jsonlines"), encoding="utf-8") as handle:
        intervals = [json.loads(line) for line in handle]

    for port_id in tp.SWAPS:
        on_swapped = [
            row["transceiver_serial"] for row in intervals
            if row["switch_id"] == tp.SWITCH and row["port_id"] == port_id
        ]

        assert sorted(on_swapped) == [f"XCVR-{tp.SWITCH}-{port_id}", f"XCVR-{tp.SWITCH}-{port_id}-B"]


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


def test_the_thresholds_are_read_from_the_stanzas_and_not_remembered():
    # The values the guide gives, so a stanza retuned by accident fails here rather than silently retuning the
    # package's expectations with it; a deliberate retune edits both.
    assert (TRAVEL_KMH_THRESHOLD, MFA_CHALLENGE_THRESHOLD, MFA_DENIAL_THRESHOLD) == (900, 5, 4)
    assert (DRIFT_RISING_THRESHOLD, DRIFT_SIGMA_THRESHOLD, DRIFT_MEAN_CEILING) == (4, 1.5, 2.0)
    assert (GAP_THRESHOLD_NS, FORECAST_DAYS_THRESHOLD) == (60 * 10**9, 14)
    assert TTL_SHIFTED_FLOWS == 3


def test_the_binding_health_rows_are_the_classes_that_resolve_bindings(expected: dict):
    # The search groups every scored event that carries a resolution outcome by sourcetype and class. It used to
    # read morpheus:score:l3 alone, where no producer writes resolution_method, and could only ever return nothing.
    groups: dict = {}

    for name in sorted(os.listdir(EVENTS)):
        if (not name.startswith("morpheus_score_l") or not name.endswith(".jsonlines")):
            continue

        sourcetype = name[:-len(".jsonlines")].replace("morpheus_score_", "morpheus:score:")

        with open(os.path.join(EVENTS, name), encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)

                if ("resolution_method" in event):
                    key = f"{sourcetype}/{event['telemetry_class']}"
                    group = groups.setdefault(key, {"total": 0, "unresolved": 0})
                    group["total"] += 1
                    group["unresolved"] += int(event["resolution_method"] == "unresolved")

    entry = expected["searches"]["Binding health - unresolved rate"]

    assert len(groups) > 0
    assert entry["expected_rows"] == len(groups)
    assert entry["expected_empty"] is False
    assert entry["groups"] == groups

    # One class, and that is the correction the first search-head run forced: four more carried the column with a
    # null on every row, and while nulls were sent the search counted them as healthy.
    assert list(groups) == ["morpheus:score:l2/tc2_arp"]

    # No class in the corpus is degraded, and the one with unresolved rows is the planted ARP rate.
    for (key, group) in groups.items():
        assert group["unresolved"] / group["total"] <= 0.2, key

    assert groups["morpheus:score:l2/tc2_arp"]["unresolved"] > 0


def test_every_scored_sample_event_carries_the_determinism_envelope():
    # What a SIEM receives is the test, not what the pipeline holds: the envelope has to survive rendering and the
    # all-null column drop the generator applies, on every score sourcetype.
    events = _scored_events()

    assert len(events) > 0

    for event in events:
        assert event["determinism_tier"] == "D1", event
        assert event["config_hash"], event
        assert event["pipeline_fingerprint"], event


def test_the_scored_layer_5_samples_name_the_model_and_the_fallback():
    # Every layer 5 rule tables `model_version` and `model_fallback_used`, so an analyst reading a notable sees
    # which principal's committed model scored it, or that the population fallback did. A scored event names a
    # pinned `dfencoder/<principal>:<digest>` of its own principal, or the fallback with the flag set; an event the
    # models were trained on, or one no scorer reads, names nothing.
    events = _events_of("morpheus_score_l5.jsonlines")
    scored = [event for event in events if event.get("mean_abs_z") is not None]

    assert len(scored) > 0

    own = [event for event in scored if event["model_fallback_used"] is False]
    fallback = [event for event in scored if event["model_fallback_used"] is True]

    assert len(own) + len(fallback) == len(scored)
    assert own and fallback

    for event in own:
        assert event["model_version"].startswith(f"dfencoder/{event['user_principal']}:"), event

    for event in fallback:
        assert event["model_version"] == "reference-arithmetic:0", event

    unscored = [event for event in events if event.get("mean_abs_z") is None]

    assert len(unscored) > 0
    assert all(event.get("model_version") is None for event in unscored)


def test_the_documents_quote_the_risk_records_the_detections_write(expected: dict):
    # Every detection's rows become risk records, so the count the documents give is the detections' row total.
    total = sum(entry["expected_rows"] for (name, entry) in expected["searches"].items() if name.startswith("R-"))
    guide = os.path.join(REPO_ROOT,
                         "docs",
                         "source",
                         "developer_guide",
                         "guides",
                         "11_predictive_behavioral_analytics_osi.md")

    with open(VALIDATION, encoding="utf-8") as handle:
        assert f"holds exactly those rows -- {total} --" in " ".join(handle.read().split())

    with open(guide, encoding="utf-8") as handle:
        assert f"holds {total} risk records for the {total} rows the detections returned" in " ".join(
            handle.read().split())


# --- Layers 1 and 2, as the search head holds them ---------------------------------------------------------------


def _number(event: dict, field: str):
    value = event.get(field)

    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _over(event: dict, field: str, bound: float) -> bool:
    value = _number(event, field)

    return value is not None and value > bound


def _at_least(event: dict, field: str, bound: float) -> bool:
    value = _number(event, field)

    return value is not None and value >= bound


def _at_most(event: dict, field: str, bound: float) -> bool:
    value = _number(event, field)

    return value is not None and value <= bound


def _departed(event: dict) -> typing.Optional[str]:
    """R-B-L1-007's `where` and its `direction`, or `None` where the poll did not depart."""
    (now, peak, trough) = (_number(event, name) for name in ("bits_per_second", "bits_per_second_baseline_max",
                                                             "bits_per_second_baseline_min"))

    if (event.get("bits_per_second_baseline_mature") is not True or now is None or peak is None):
        return None

    if (now > SURGE_FACTOR * peak):
        return "surge"

    if (trough is not None and trough > 0 and now < SILENCE_FACTOR * trough):
        return "silence"

    return None


def _below_commissioning(event: dict) -> typing.Optional[float]:
    """R-D-L1-010's `eval`: how far a port's light sits below its commissioning level, or `None` where the inventory
    gave the port no level. The level arrives as the text the enrichment writes, which `tonumber` reads."""
    (level, now) = (event.get("ctx_port_commissioning_rx_dbm"), _number(event, "optical_rx_dbm"))

    if (event.get("ctx_port_found") is not True or level is None or now is None):
        return None

    return round(float(level) - now, 3)


LAYER_1_AND_2 = {
    # rule id: (sourcetype file, the base search and `where` as a predicate, the `stats BY` field or None for a
    # rule that writes one risk record per row).
    "R-D-L1-001": ("morpheus_score_l1.jsonlines", lambda e: e.get("transceiver_serial_changed") is True and _number(
        e, "link_flaps") == 0 and e.get("oper_status") != "down",
                   None),
    "R-D-L1-002": ("morpheus_score_l1.jsonlines", lambda e: _at_most(e, "optical_rx_dbm_deviation", TAP_DEVIATION) and
                   _at_least(e, "optical_rx_dbm_baseline_samples", TAP_SAMPLES) and e.get("transceiver_serial_changed")
                   is not True and abs(_number(e, "optical_tx_dbm_deviation") or 0.0) < TAP_TRANSMIT,
                   "entity_key"),
    "R-D-L1-003":
        ("morpheus_score_l1.jsonlines",
         lambda e: _at_least(e, "link_flaps_in_window", FLAP_TRANSITIONS) and e.get("link_flap_device_reset") is False,
         "entity_key"),
    "R-P-L1-004": ("morpheus_score_l1.jsonlines", lambda e: e.get("optical_rx_dbm_forecast_status") == "projected" and
                   _at_most(e, "optical_rx_dbm_days_to_floor", FORECAST_DAYS_THRESHOLD),
                   "entity_key"),
    "R-B-L1-005": ("morpheus_score_l1.jsonlines",
                   lambda e: e.get("error_rate_baseline_mature") is True and _over(e, "error_rate_step", ERROR_STEP),
                   "entity_key"),
    "R-D-L1-006": ("morpheus_score_l1.jsonlines", lambda e: e.get("lldp_neighbor_chassis_id_changed") is True and e.get(
        "lldp_neighbor_chassis_id_first_seen") is True and e.get("lldp_neighbor_chassis_id") is not None,
                   None),
    "R-B-L1-007": ("morpheus_score_l1.jsonlines", lambda e: _departed(e) is not None, "entity_key"),
    "R-D-L1-008": ("morpheus_score_l1.jsonlines",
                   lambda e: e.get("config_changed") is True and e.get("config_change_in_window") is False,
                   None),
    "R-P-L1-009": ("morpheus_score_l1.jsonlines", lambda e: e.get("device_reboot_baseline_mature") is True and
                   _at_least(e, "device_reboot_rising_buckets", RISING_PERIODS),
                   None),
    "R-D-L1-010":
        ("morpheus_score_l1.jsonlines", lambda e:
         (_below_commissioning(e) or 0.0) >= COMMISSIONING_MARGIN and e.get("transceiver_serial_changed") is not True,
         "entity_key"),
    "R-B-L2-002": ("morpheus_score_l2.jsonlines",
                   lambda e: e.get("macs_per_port_first_in_window") is True and _over(e, "macs_per_port_step", 0),
                   "port_key"),
    "R-D-L2-003": ("morpheus_score_l2.jsonlines",
                   lambda e: _over(e, "macs_claiming_sender_ip", 1) and e.get("arp_sender_ip_excluded") is False,
                   "arp_sender_ip"),
    "R-D-L2-005": ("morpheus_score_l2.jsonlines", lambda e: e.get("auth_unpaired") is True, None),
    "R-B-L2-006": ("morpheus_score_l2.jsonlines", lambda e: e.get("ouis_per_vlan_first_in_window") is True and e.get(
        "ouis_per_vlan_baseline_mature") is True and _over(e, "ouis_per_vlan_step", 0),
                   "vlan_key"),
    "R-B-L2-007":
        ("morpheus_score_l2.jsonlines",
         lambda e: _at_least(e, "auth_elapsed_ratio", SLOW_RATIO) and _at_least(e, "auth_attempts", SLOW_ATTEMPTS),
         None),
    "R-B-L2-008": ("morpheus_score_l2.jsonlines", lambda e: _at_most(e, "auth_elapsed_ratio", INSTANT_RATIO), None),
}
"""Every layer 1 and 2 detection whose risk records name a lineage, as its stanza states it. R-D-L2-004 reads the
binding records, which carry none, and R-D-L2-001's lookup ships empty."""


def _notables(rule_id: str) -> list:
    """One entry per risk record the rule writes over the sample events: the rows behind it."""
    (source, predicate, group) = LAYER_1_AND_2[rule_id]
    rows = [event for event in _events_of(source) if predicate(event)]

    if (group is None):
        return [[row] for row in rows]

    grouped = collections.defaultdict(list)

    for row in rows:
        grouped[row[group]].append(row)

    return [grouped[key] for key in sorted(grouped)]


def _risk_score(rule_id: str) -> float:
    stanza = next(name for name in _searches() if name.startswith(rule_id + " "))

    return float(re.search(r'rule_id = "' + rule_id + r'", risk_score = (\d+)', _search_text(stanza)).group(1))


def test_the_new_layer_1_detections_return_exactly_what_is_written(expected: dict):
    searches = expected["searches"]

    for (rule, key) in (("R-D-L1-002", TAP), ("R-D-L1-003", FLAP), ("R-B-L1-005", ERRORS), ("R-B-L1-007", VOLUME),
                        ("R-D-L1-010", COMMISSIONING)):
        notables = _notables(rule)

        assert searches[key]["expected_rows"] == len(notables), key
        assert searches[key]["contributing_rows"] == sum(len(rows) for rows in notables), key
        assert [row["entity_key"] for row in searches[key]["key_values"]] == [rows[0]["entity_key"]
                                                                             for rows in notables], key

    directions = [row["direction"] for row in searches[VOLUME]["key_values"]]
    assert directions == [sorted({_departed(row) for row in rows})[0] for rows in _notables("R-B-L1-007")]

    neighbors = _notables("R-D-L1-006")
    assert searches[NEIGHBOR]["expected_rows"] == len(neighbors)
    assert sorted((row["entity_key"], row["lldp_neighbor_chassis_id"]) for row in searches[NEIGHBOR]["key_values"]) \
        == sorted((rows[0]["entity_key"], rows[0]["lldp_neighbor_chassis_id"]) for rows in neighbors)

    # The device's own rules, one risk record per poll: the change outside the window, and the third period running
    # with more restarts than the one before.
    for (rule, key) in (("R-D-L1-008", CONFIGURATION), ("R-P-L1-009", TRAJECTORY)):
        notables = _notables(rule)

        assert searches[key]["expected_rows"] == len(notables), key
        assert [row["entity_key"] for row in searches[key]["key_values"]] == [rows[0]["entity_key"]
                                                                             for rows in notables], key


def test_the_restart_report_is_one_row_per_device_and_window(expected: dict):
    # The device's own restart and its ports' flags, gathered under the device: the port rows carry no device key,
    # so the search composes one from the site and the device, as the device stage composes its own.
    reported = [
        event for event in _events_of("morpheus_score_l1.jsonlines") if event.get("device_rebooted") is True
        or event.get("counter_reset") is True or event.get("link_flap_device_reset") is True
    ]
    devices = collections.defaultdict(lambda: {"restarts": 0, "ports": set()})

    for event in reported:
        device = devices[(event.get("device_key") or f"{event['site_id']}:{event['device_id']}", event["window_id"])]
        device["restarts"] += int(event.get("device_rebooted") is True)

        if (event.get("port_id") is not None):
            device["ports"].add(event["port_id"])

    entry = expected["searches"]["Device restart"]

    assert entry["expected_rows"] == len(devices)
    assert [(row["device_key"], row["window_id"], row["device_restarts"], row["ports_reporting"])
            for row in entry["key_values"]] == [(key, window, device["restarts"], len(device["ports"]))
                                                for ((key, window), device) in sorted(devices.items())]


def test_the_adjacency_lookup_holds_one_row_per_link(expected: dict):
    links = {event["link_key"] for event in _events_of("morpheus_score_l1.jsonlines") if event.get("link_key")}

    assert expected["searches"]["Topology - LLDP adjacency"]["expected_rows"] == len(links)


def test_the_new_layer_2_detections_return_exactly_what_is_written(expected: dict):
    searches = expected["searches"]
    novelty = _notables("R-B-L2-002")
    entry = searches["R-B-L2-002 - Port-to-MAC binding novelty"]

    assert entry["expected_rows"] == len(novelty)
    assert entry["contributing_rows"] == sum(len(rows) for rows in novelty)
    assert sorted(row["port_key"] for row in entry["key_values"]) == [rows[0]["port_key"] for rows in novelty]

    vendors = _notables("R-B-L2-006")

    assert searches[VENDOR]["expected_rows"] == len(vendors)
    # `values(oui)` is every vendor new to the segment in the group, space-separated here as a multivalue reads.
    assert [(row["vlan_key"], row["new_ouis"]) for row in searches[VENDOR]["key_values"]
            ] == [(rows[0]["vlan_key"], " ".join(sorted({row["mac_address"][:8]
                                                         for row in rows}))) for rows in vendors]

    for (rule, key) in (("R-B-L2-007", SLOW), ("R-B-L2-008", INSTANT)):
        notables = _notables(rule)

        assert searches[key]["expected_rows"] == len(notables), key
        assert [(row["auth_port_key"], row["dot1x_identity"]) for row in searches[key]["key_values"]] == [
            (rows[0]["auth_port_key"], rows[0]["dot1x_identity"]) for rows in notables
        ], key


# --- The pipeline's own reports, read off the health, quarantine and late streams ------------------------------


def _optional_events(sourcetype_file: str) -> list:
    return _events_of(sourcetype_file) if os.path.exists(os.path.join(EVENTS, sourcetype_file)) else []


def test_no_scored_event_reaches_the_detections_from_an_incomplete_window():
    # The late row goes on its own sourcetype, so the detections' index holds only complete windows. Every
    # detection also filters on window_complete=true, which is the second guard; this is the first.
    for name in sorted(os.listdir(EVENTS)):
        if (name.startswith("morpheus_score_l") and name.endswith(".jsonlines")):
            assert all(event.get("window_complete") is True for event in _events_of(name)), name

    late = _events_of("morpheus_late.jsonlines")

    assert len(late) > 0
    assert all(event["is_late"] is True and event["window_complete"] is False for event in late)


def test_every_detection_reads_only_complete_windows():
    import configparser  # pylint: disable=import-outside-toplevel

    with open(SAVEDSEARCHES, encoding="utf-8") as handle:
        folded = re.sub(r"\\\s*\r?\n\s*", " ", handle.read())

    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read_string(folded)

    for name in parser.sections():
        query = parser[name].get("search", "")
        reads = re.findall(r"index=behavior_events sourcetype=morpheus:score:l[1-7*]( window_complete=true)?", query)

        assert all(reads), f"{name} reads behavior_events without window_complete=true"


def test_the_refusal_rate_is_what_the_quarantine_and_health_streams_say(expected: dict):
    received = collections.Counter()
    refused = collections.Counter()

    for event in _events_of("morpheus_quarantine.jsonlines"):
        key = (event["collector_id"], event["tenant_id"])
        received[key] += 1
        refused[key] += 1

    for event in _events_of("morpheus_health.jsonlines"):
        key = (event["collector_id"], event["tenant_id"])
        received[key] += event["rows"]
        refused[key] += event.get("clock_refused", 0)

    rows = [{
        "collector_id": collector,
        "tenant_id": tenant,
        "received": received[(collector, tenant)],
        "refusal_rate": round(refused[(collector, tenant)] / received[(collector, tenant)], 3)
    } for (collector, tenant) in sorted(received) if refused[(collector, tenant)] > 0]
    entry = expected["searches"]["Pipeline health - refusal rate"]

    assert entry["expected_rows"] == len(rows)
    assert [{
        key: row[key]
        for key in ("collector_id", "tenant_id", "received", "refusal_rate")
    } for row in entry["key_values"]] == rows


def test_collector_health_names_the_collectors_whose_streams_are_broken(expected: dict):
    gaps = collections.Counter()
    regressed = collections.Counter()
    worst_offset = collections.Counter()

    for event in _events_of("morpheus_health.jsonlines"):
        collector = event["collector_id"]
        gaps[collector] += event.get("seq_gap", 0)
        regressed[collector] += event.get("seq_regressed", 0)
        worst_offset[collector] = max(worst_offset[collector],
                                      abs(event.get("clock_offset_ms_min", 0)),
                                      abs(event.get("clock_offset_ms_max", 0)))

    flagged = sorted(collector for collector in gaps
                     if gaps[collector] > 0 or regressed[collector] > 0 or worst_offset[collector] > 1000)
    entry = expected["searches"]["Collector health"]

    assert entry["expected_rows"] == len(flagged)
    assert [(row["collector_id"], row["seq_gap"], row["seq_regressed"]) for row in entry["key_values"]
            ] == [(collector, gaps[collector], regressed[collector]) for collector in flagged]


STAGE_COUNTS = ("keyless", "unpaired", "abandoned", "unclassified", "out_of_order", "saturated", "fallback")
"""The health counts `Pipeline health - stage counts` sums, each a thing a stage could not do."""


def test_the_stage_counts_name_each_class_a_stage_could_not_fully_handle(expected: dict):
    # Splunk's sum skips a null, and a count no row of the class can answer is null in the record, so a class whose
    # stages never write a column sums to nothing rather than to zero and is kept only for a count it can answer.
    sums: dict = collections.defaultdict(dict)

    for event in _events_of("morpheus_health.jsonlines"):
        key = (event["collector_id"], event["telemetry_class"], event["tenant_id"])

        for name in STAGE_COUNTS:
            if (event.get(name) is not None):
                sums[key][name] = sums[key].get(name, 0) + event[name]

    flagged = [(key, sums[key]) for key in sorted(sums) if any(value > 0 for value in sums[key].values())]
    entry = expected["searches"]["Pipeline health - stage counts"]

    assert entry["expected_rows"] == len(flagged)
    assert entry["expected_empty"] is (len(flagged) == 0)
    assert [(row["collector_id"],
             row["telemetry_class"],
             row["tenant_id"], {
                 name: row[name]
                 for name in STAGE_COUNTS if name in row
             }) for row in entry["key_values"]] == [(*key, counts) for (key, counts) in flagged]


def test_late_arrivals_are_counted_by_class(expected: dict):
    counts = collections.Counter((event["telemetry_class"], event["collector_id"], event["tenant_id"])
                                 for event in _optional_events("morpheus_late.jsonlines"))
    entry = expected["searches"]["Late arrivals by class"]

    assert entry["expected_rows"] == len(counts)
    assert [(row["telemetry_class"], row["collector_id"], row["tenant_id"], row["late_rows"])
            for row in entry["key_values"]] == [(*key, count) for (key, count) in sorted(counts.items())]


def test_the_platform_step_is_the_doubling_tenant_alone(expected: dict):
    import envelope  # pylint: disable=import-outside-toplevel

    records = pd.DataFrame(_events_of("morpheus_health.jsonlines"))
    steps = envelope.platform_steps(records)
    entry = expected["searches"]["Platform baseline step"]

    assert entry["expected_rows"] == len(steps)
    assert [(row["collector_id"],
             row["telemetry_class"],
             row["tenant_id"],
             row["previous_baseline"],
             row["previous_rows"],
             row["rows"]) for row in entry["key_values"]] == [(row["collector_id"],
                                                               row["telemetry_class"],
                                                               row["tenant_id"],
                                                               int(row["previous_baseline"]),
                                                               int(row["previous_rows"]),
                                                               int(row["rows"])) for (_, row) in steps.iterrows()]


def test_the_edge_stream_is_edges():
    # `morpheus:edge` carried the reference corpus's flow records for its whole life: no parent, no child, no join
    # method and no chain, so chain assembly's read of `lineage_id` from it was hollow. Every record is now an edge
    # `EdgeEmitStage` wrote, and each names the two records it joins, how, and the chain the child was sealed into.
    with open(os.path.join(EVENTS, "morpheus_edge.jsonlines"), encoding="utf-8") as handle:
        edges = [json.loads(line) for line in handle if line.strip()]

    scored = {event["event_uid"]: event for event in _scored_events()}

    assert len(edges) > 0

    for edge in edges:
        for field in ("link_uid",
                      "parent_event_uid",
                      "child_event_uid",
                      "relation_type",
                      "join_method",
                      "lineage_id",
                      "child_osi_layer"):
            assert edge.get(field) is not None, (field, edge)

        child = scored[edge["child_event_uid"]]

        assert child["lineage_id"] == edge["lineage_id"]
        assert child["osi_layer"] == edge["child_osi_layer"]

    assert {edge["relation_type"] for edge in edges} == {"resolved_through", "carried_by"}
    assert {edge["join_method"] for edge in edges if edge["relation_type"] == "carried_by"} == {"hard:community_id"}


def _bucketed(table: str) -> list:
    with open(os.path.join(EVENTS, "binding_bucketed.jsonlines"), encoding="utf-8") as handle:
        return [row for row in (json.loads(line) for line in handle if line.strip()) if row["binding_table"] == table]


def _identity_walk() -> list:
    """
    'Identity walk - sign-in to switch port' over the sample events, with the three lookups built the way their
    refreshes build them: one row per key and bucket, keyed as the search keys them.
    """
    leases = {(row["ip"], row["bucket"]): row["mac"] for row in _bucketed("dhcp_lease")}
    macs = {(row["key"], row["bucket"]): row["port_key"] for row in _bucketed("mac_table")}
    identities = {(row["dot1x_identity"], row["bucket"]): row["auth_port_key"] for row in _bucketed("dot1x_supplicant")}
    walked = []

    for event in _scored_events():
        if (event.get("osi_layer") != 5 or event.get("window_complete") is not True):
            continue

        if (not event.get("desk_identity") and not event.get("login_port_key")):
            continue

        bucket = _five_minute_bin(event["event_time"]) // 300
        by_identity = identities.get((event.get("desk_identity"), bucket))
        mac = leases.get((event.get("source_ip"), bucket))
        by_address = macs.get((mac, bucket))
        port = by_identity if by_identity is not None else by_address

        if (port is not None):
            walked.append({
                "user_principal": event["user_principal"],
                "rung": "dot1x_identity" if by_identity is not None else "address",
                "walked_port_key": port,
                "resolved_port_key": event.get("desk_port_key") or event.get("login_port_key"),
            })

    return walked


def test_each_binding_refresh_fills_its_lookup(expected: dict):
    # The three rungs Part 4's ladder walks, each fed from the bucketed bindings its refresh selects. The L2/L3
    # refresh was empty for its whole life because nothing produced a lease; the other two did not exist.
    for (search, table, key) in (("Binding lookup - L2/L3 refresh", "dhcp_lease",
                                  "ip"), ("Binding lookup - L2 refresh", "mac_table", "key"),
                                 ("Binding lookup - L5/L2 refresh", "dot1x_supplicant", "dot1x_identity")):
        rows = _bucketed(table)
        entry = expected["searches"][search]

        assert len(rows) > 0, table
        assert len({(row[key], row["bucket"]) for row in rows}) == len(rows), f"{table}: one row per key and bucket"
        assert entry["expected_rows"] == len(rows) and entry["expected_empty"] is False, search

        stanza = _search_text(search)

        assert f"binding_table={table}" in stanza and "key_field=_key" in stanza, search


def test_the_identity_walk_reaches_the_port_the_pipeline_resolved(expected: dict):
    # Part 4's ladder walked twice: by the pipeline, which wrote the port on each sign-in, and by the SIEM through
    # the lookups. A desk sign-in walks by its 802.1X identity; a host login by its address to a MAC and the MAC
    # to a port. The two must land on the same port for every sign-in either reaches.
    walked = _identity_walk()
    entry = expected["searches"]["Identity walk - sign-in to switch port"]

    assert len(walked) > 0
    assert all(row["walked_port_key"] == row["resolved_port_key"] for row in walked)
    assert {row["rung"] for row in walked} == {"dot1x_identity", "address"}
    assert entry["expected_rows"] == len(walked)
    assert entry["walked_by_rung"] == dict(sorted(collections.Counter(row["rung"] for row in walked).items()))


# --- The expiry jobs, as far as a run without a KV Store can evaluate them ---------------------------------------


def _runner():
    """`run_search_head.py` as a module, for where it places the events before a run; nothing runs at import."""
    spec = importlib.util.spec_from_file_location("run_search_head", os.path.join(VALIDATE, "run_search_head.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def _seconds(event_time: str) -> float:
    stamp = datetime.datetime.strptime(event_time.replace("UTC", ""), "%Y-%m-%dT%H:%M:%S.%f")

    return stamp.replace(tzinfo=datetime.timezone.utc).timestamp()


def _newest_timestamp() -> float:
    """The newest timestamp in the sample events, found as the runner finds it: it is the one the runner lands."""
    lines = [
        line for (name, content) in _event_files().items() if name.endswith(".jsonlines")
        for line in content.splitlines()
    ]

    return _runner().newest_and_oldest(lines)[0]


def _kept_by_the_run(age_seconds: float, retention_seconds: float, bucket_seconds: float = 0.0) -> bool:
    """
    Whether an expiry measured against `now()` keeps a row this far behind the newest event, on the run the runner
    makes. The runner moves every timestamp by whole weeks so the newest lands between `LANDING_SECONDS` and a week
    more before the run, so where in that band the run falls is not known here. A row is kept whatever the day when
    its age is at most the retention less the band's far edge, and dropped whatever the day when its age is more
    than the retention less the near edge -- plus one bucket, for an expiry that compares buckets rather than
    seconds. A row between the two is kept or dropped by the day of the run, and an expectation for it cannot be
    stated, so this fails rather than guesses.
    """
    runner = _runner()
    kept_whatever_the_day = retention_seconds - (runner.LANDING_SECONDS + runner.WEEK_SECONDS)
    dropped_whatever_the_day = retention_seconds - runner.LANDING_SECONDS + bucket_seconds

    assert age_seconds <= kept_whatever_the_day or age_seconds > dropped_whatever_the_day, (
        f"a row {age_seconds / 86400:.1f} days behind the newest event is kept or dropped by the day of the run")

    return age_seconds <= kept_whatever_the_day


def _watchlist_terms(name: str) -> tuple:
    """A predictive rule's `reason` and how long its entries live, as its stanza states them."""
    search = _stanza_search(name)

    return (re.search(r'reason = "([\w-]+)"', search).group(1),
            float(re.search(r"expires = _time \+ (\d+)", search).group(1)))


def _entries(name: str, groups: dict, rule_id: str, key: str) -> list:
    """The watchlist rows one rule writes: one per group, stamped with the group's `_time` and the rule's terms."""
    (reason, retention) = _watchlist_terms(name)

    return [{
        key: entity,
        "rule_id": rule_id,
        "reason": reason,
        "day_window_id": day,
        "added": added,
        "expires": added + retention
    } for ((entity, day), added) in sorted(groups.items())]


def _latest_per_day(rows: list, key: str) -> dict:
    """`dedup <key> day_window_id` over a search that returns newest first keeps each group's latest row."""
    latest: dict = collections.defaultdict(float)

    for event in rows:
        group = (event[key], int(event["day_window_id"]))
        latest[group] = max(latest[group], _seconds(event["event_time"]))

    return latest


def _principal_watchlist() -> list:
    """
    The rows the two principal rules write, each stamped as its writer stamps it: R-P-L5-006 the latest row of each
    principal-day it fired, R-P-L7-006 the `latest(_time)` of each watched principal's SaaS rows, with no day.
    """
    drift = [
        event for event in _events_of("morpheus_score_l5.jsonlines")
        if event.get("window_complete") is True and event.get("drift_mature") is True
        and float(event.get("drift_rising_windows") or 0) >= DRIFT_RISING_THRESHOLD
        and float(event.get("drift_rise_sigmas") or 0) > DRIFT_SIGMA_THRESHOLD and float(event.get("mean_abs_z") or 0) <
        DRIFT_MEAN_CEILING and abs(float(event.get("drift_acceleration") or 0)) < DRIFT_ACCELERATION_CEILING
    ]
    entries = _entries("R-P-L5-006 - Drift trajectory",
                       _latest_per_day(drift, "user_principal"),
                       "R-P-L5-006",
                       "user_principal")

    rising: dict = collections.defaultdict(int)
    roles: dict = collections.defaultdict(set)
    latest: dict = collections.defaultdict(float)

    for event in _layer_7_events():
        if (event.get("saas_object_types_in_week") is None):
            continue

        principal = event["user_principal"]
        rising[principal] = max(rising[principal], event.get("drift_rising_windows") or 0)
        roles[principal].add(event.get("ctx_groups") or "none")
        latest[principal] = max(latest[principal], _seconds(event["event_time"]))

    weeks = threshold("R-P-L7-006 - Access breadth trajectory", r"rising_weeks\s*>=\s*([\d.]+)")
    breadth = {
        (principal, None): latest[principal]
        for principal in rising if rising[principal] >= weeks and len(roles[principal]) == 1
    }

    return entries + _entries("R-P-L7-006 - Access breadth trajectory", breadth, "R-P-L7-006", "user_principal")


def _host_watchlist() -> list:
    """The rows the five host rules write: the latest row of each host-day each fired, keyed as the stanzas key them,
    on the host the lease names or the address where no lease covers the row."""
    drift = [
        event for event in _events_of("morpheus_score_host.jsonlines")
        if event.get("window_complete") is True and event.get("drift_mature") is True
        and float(event.get("drift_rising_windows") or 0) >= HOST_DRIFT_RISING_THRESHOLD
        and float(event.get("drift_rise_sigmas") or 0) > HOST_DRIFT_SIGMA_THRESHOLD
    ]
    entries = _entries("R-P-L3-009 - Host drift trajectory",
                       _latest_per_day(drift, "host_key"),
                       "R-P-L3-009",
                       "host_key")
    queries = [event for event in _layer_7_events() if event.get("dns_registered_domain") is not None]

    requests = [event for event in _layer_7_events() if event.get("http_distinct_paths") is not None]
    processes = [event for event in _layer_7_events() if event.get("endpoint_pair") is not None]
    handshakes = _events_of("morpheus_score_l6.jsonlines")

    trajectories = [
        (STACK_TRAJECTORY, "R-P-L6-006", handshakes, STACK_RISING_THRESHOLD, STACK_SIGMA_THRESHOLD, "src_ip"),
        (SUBDOMAIN_TRAJECTORY, "R-P-L7-009", queries, SUBDOMAIN_RISING_THRESHOLD, SUBDOMAIN_SIGMA_THRESHOLD, "src_ip"),
        (PATH_TRAJECTORY, "R-P-L7-010", requests, PATH_RISING_THRESHOLD, PATH_SIGMA_THRESHOLD, "src_ip"),
        (BREADTH_TRAJECTORY,
         "R-P-L7-011",
         processes,
         BREADTH_RISING_THRESHOLD,
         BREADTH_SIGMA_THRESHOLD,
         "endpoint_host"),
    ]

    for (name, rule_id, events, rising, sigmas, subject) in trajectories:
        fires = [
            dict(event, host_key=event.get("host_key") or event[subject])
            for event in _trajectory_days(events, rising, sigmas, subject)[0]
        ]
        entries += _entries(name, _latest_per_day(fires, "host_key"), rule_id, "host_key")

    return entries


def _asserts_the_expiry_keeps_what_is_written(name: str, entries: list, expected: dict, key: str,
                                              writers: dict) -> None:
    """
    An expiry job as far as a run without a KV Store can evaluate it: the entries the writers put on the list are
    recomputed from the events, and each is kept or dropped as its `expires` compares with the run's clock where
    `run_search_head.py` places the events. What is not reproduced is the lookup itself. A search head's list holds
    what every earlier run appended too; the expectation is for a list the predictive searches have just written,
    which is the state the runner dispatches them in.
    """
    newest = _newest_timestamp()
    kept = [entry for entry in entries if _kept_by_the_run(newest - entry["added"], entry["expires"] - entry["added"])]
    dropped = [entry for entry in entries if entry not in kept]
    written = expected["searches"][name]

    assert "where expires >= now()" in _stanza_search(name)
    assert collections.Counter(entry["rule_id"] for entry in entries) == {
        rule_id: expected["searches"][writer]["expected_rows"] for (rule_id, writer) in writers.items()
    }, "the lists hold one entry per row the writers return"
    assert written["expected_rows"] == len(kept)
    assert written["key_values"] == [{
        field: entry[field]
        for field in (key, "rule_id", "reason", "day_window_id") if entry[field] is not None
    } for entry in kept]
    assert {entry["rule_id"] for entry in kept} and {entry["rule_id"] for entry in dropped}, "both outcomes occur"
    assert not {entry["rule_id"] for entry in kept} & {entry["rule_id"] for entry in dropped}, "and by rule"


def test_the_principal_watchlist_expiry_keeps_exactly_what_is_written(expected: dict):
    # The two principals R-P-L7-006 watched are stamped with their latest SaaS rows, days behind the newest event, so
    # their thirty days are ahead of any run; R-P-L5-006's six principal-days are more than five weeks behind it, so
    # they are gone whatever the day. Nothing is near the band where the run's date would decide.
    _asserts_the_expiry_keeps_what_is_written(
        "Principal watchlist - expiry",
        _principal_watchlist(),
        expected,
        "user_principal", {
            "R-P-L5-006": "R-P-L5-006 - Drift trajectory", "R-P-L7-006": "R-P-L7-006 - Access breadth trajectory"
        })


def test_the_host_watchlist_expiry_keeps_exactly_what_is_written(expected: dict):
    # The drifting host's three days are the newest events in the corpus; the stack and subdomain trajectories'
    # seven host-days are in their corpora's first week, more than seven weeks behind, so the hosts' list keeps
    # R-P-L3-009's entries and nothing else.
    _asserts_the_expiry_keeps_what_is_written(
        "Host watchlist - expiry",
        _host_watchlist(),
        expected,
        "host_key",
        {
            "R-P-L3-009": "R-P-L3-009 - Host drift trajectory",
            "R-P-L6-006": STACK_TRAJECTORY,
            "R-P-L7-009": SUBDOMAIN_TRAJECTORY,
            "R-P-L7-010": PATH_TRAJECTORY,
            "R-P-L7-011": BREADTH_TRAJECTORY
        })


def test_the_l5_l2_expiry_keeps_exactly_what_is_written(expected: dict):
    """
    'Binding lookup - L5/L2 expiry' as far as a run without a KV Store can evaluate it: the lookup's rows are the
    refresh's, one per identity and bucket from the supplicant bindings, and each is kept or dropped by the bucket
    its own time falls in against the retention the stanza states, where `run_search_head.py` places the events.
    The job returns every row it keeps, so its rows are the lookup's, named here by identity and port. What is not
    reproduced is the store: a search head's lookup also holds what earlier refreshes appended.
    """
    name = "Binding lookup - L5/L2 expiry"
    search = _stanza_search(name)
    retention = float(re.search(r"bucket >= floor\(\(now\(\) - (\d+)\)", search).group(1))
    bucket_seconds = float(re.search(r"now\(\) - \d+\) / (\d+)\)", search).group(1))
    rows = _bucketed("dot1x_supplicant")
    newest = _newest_timestamp()

    # The refresh computes the bucket the expiry compares from `_time`, which `props.conf` stamps from `bucket_start`,
    # so the stored ordinal is that bucket and the arithmetic here is the search's.
    assert rows and all(row["bucket"] == int(_seconds(row["bucket_start"]) // bucket_seconds) for row in rows)
    assert bucket_seconds == 300

    kept = [row for row in rows if _kept_by_the_run(newest - _seconds(row["bucket_start"]), retention, bucket_seconds)]
    entry = expected["searches"][name]

    # A corpus two months long against a retention of more than a year: every bucket is inside it, whatever the day.
    assert entry["expected_rows"] == len(kept) == len(
        rows) == expected["searches"]["Binding lookup - L5/L2 refresh"]["expected_rows"]
    assert entry["key_values"] == [{
        "dot1x_identity": identity, "port_key": port_key, "kept_buckets": count
    } for ((identity, port_key),
           count) in sorted(collections.Counter((row["dot1x_identity"], row["auth_port_key"]) for row in kept).items())]
