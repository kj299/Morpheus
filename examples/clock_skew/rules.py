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
Every shipped detection, reduced to what it accuses, over the output of the pipeline that feeds it.

`run_experiment.py` asks how much clock skew each rule tolerates; this module is what it asks with. Each function
takes the canonical output frame of one composed pipeline and returns `{rule_id: set of key tuples}`, one tuple
per entity the shipped search would accuse, keyed the way the search's final `stats ... BY` or notable dedup key
groups it -- and never on a time, because a decision keyed on when would differ under every non-zero offset and
measure the injection rather than the damage.

Every number a search states is read from `savedsearches.conf`, and so is every boolean term these mirror without
a number, through `_requires`: a function here that decided a copy of the rule rather than the rule would be the
failure this experiment exists to find. Splunk's own semantics are kept where they decide anything: `field=true`
never matches a null, a comparison with a null is false, and `stats ... BY` drops a row whose group field is null.

What is not mirrored is the dispatch window. Each search runs every few minutes over a trailing slice; these
evaluate the whole run at once, the same simplification `expected_results.json` makes. Where that could change a
decision it is said beside the rule.
"""

import configparser
import csv
import os
import re

import pandas as pd

SAVED_SEARCHES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..",
                              "splunk_lineage_app",
                              "TA-morpheus-lineage",
                              "default",
                              "savedsearches.conf")

LOOKUPS = os.path.join(os.path.dirname(SAVED_SEARCHES), "..", "lookups")

# --- Reading the shipped searches -----------------------------------------------------------------------------

_STANZAS: dict = {}


def stanza(name: str) -> dict:
    """One stanza of the shipped `savedsearches.conf`, its continuation lines folded."""
    if (not _STANZAS):
        with open(SAVED_SEARCHES, encoding="utf-8") as handle:
            folded = re.sub(r"\\\s*\r?\n\s*", " ", handle.read())

        parser = configparser.ConfigParser(interpolation=None, strict=False)
        parser.optionxform = str
        parser.read_string(folded)
        _STANZAS.update({section: dict(parser[section]) for section in parser.sections()})

    return _STANZAS[name]


def threshold(name: str, pattern: str, key: str = "search") -> float:
    """A number as the shipped search states it, so the experiment decides the app's rule and not a copy."""
    match = re.search(pattern, stanza(name)[key])

    if (match is None):
        raise ValueError(f"{name} no longer states {pattern!r} in {key}")

    return float(match.group(1))


def requires(name: str, literal: str) -> None:
    """A term the mirrored predicate depends on and states no number for; refuse to decide if it has changed."""
    if (re.search(literal, stanza(name)["search"]) is None):
        raise ValueError(f"{name} no longer states {literal!r}")


# --- Reading the pipeline frame as Splunk reads the events ---------------------------------------------------


def rows(result: pd.DataFrame, telemetry_class: str) -> pd.DataFrame:
    return result[result["telemetry_class"] == telemetry_class]


def keys(frame: pd.DataFrame, columns: list) -> set:
    """The accused entities, as tuples of strings; `stats ... BY` drops a row whose group field is null."""
    present = frame.dropna(subset=columns)

    return {tuple(str(row[column]) for column in columns) for (_, row) in present.iterrows()}


def true(frame: pd.DataFrame, column: str) -> pd.Series:
    """`column=true`: only a true flag matches; a null or absent one never does."""
    if (column not in frame.columns):
        return pd.Series(False, index=frame.index)

    return frame[column].astype("boolean").fillna(False).astype(bool)


def false(frame: pd.DataFrame, column: str) -> pd.Series:
    """`column=false`, with the same treatment of a null."""
    if (column not in frame.columns):
        return pd.Series(False, index=frame.index)

    return (~frame[column].astype("boolean")).fillna(False).astype(bool)


def number(frame: pd.DataFrame, column: str) -> pd.Series:
    """A numeric field as a float, a null kept as NaN so every comparison with it is false."""
    if (column not in frame.columns):
        return pd.Series(float("nan"), index=frame.index)

    return pd.to_numeric(frame[column], errors="coerce").astype(float)


# --- Layer 1, over the estate --------------------------------------------------------------------------------

SUBSTITUTION = "R-D-L1-001 - Transceiver substitution"
FORECAST = "R-P-L1-004 - Optical degradation forecast"
SUBSTITUTION_LINK_FLAPS = threshold(SUBSTITUTION, r"link_flaps\s*=\s*([\d.]+)")
FORECAST_DAYS = threshold(FORECAST, r"optical_rx_dbm_days_to_floor\s*<=\s*([\d.]+)")


def layer_1_decisions(result: pd.DataFrame) -> dict:
    """
    R-D-L1-001: `streamstats current=f window=1 last(transceiver_serial) AS previous_serial BY entity_key | where
    transceiver_serial_changed="true" AND link_flaps=0 AND oper_status!="down"`, one notable per poll, keyed on
    the port and the two serials -- the poll's time is not the accusation. The previous serial is taken over the
    whole hour, where the search's streamstats resets at each five-minute dispatch.

    R-P-L1-004: `optical_rx_dbm_forecast_status="projected" optical_rx_dbm_days_to_floor<=14 | stats ... BY
    entity_key`.
    """
    polls = rows(result, "tc1").sort_values(["entity_key", "event_time"], kind="stable")
    polls = polls.assign(previous_serial=polls.groupby("entity_key")["transceiver_serial"].shift(1))
    substituted = polls[true(polls, "transceiver_serial_changed")
                        & (number(polls, "link_flaps") == SUBSTITUTION_LINK_FLAPS)
                        & polls["oper_status"].notna() & (polls["oper_status"] != "down")]
    projected = polls[(polls["optical_rx_dbm_forecast_status"] == "projected")
                      & (number(polls, "optical_rx_dbm_days_to_floor") <= FORECAST_DAYS)]

    return {
        "R-D-L1-001": keys(substituted, ["entity_key", "previous_serial", "transceiver_serial"]),
        "R-P-L1-004": keys(projected, ["entity_key"]),
    }


# --- Layer 2, over the estate --------------------------------------------------------------------------------

NOVELTY = "R-B-L2-002 - Port-to-MAC binding novelty"
NOVELTY_STEP = threshold(NOVELTY, r"macs_per_port_step\s*>\s*([\d.]+)")
SPOOF_GAP_NS = int(threshold("R-D-L2-004 - MAC in two places at once", r"gap_threshold\s*=\s*(\d+)"))


def layer_2_decisions(result: pd.DataFrame, single_host_ports: set) -> dict:
    """
    The five layer 2 detections. R-D-L2-001 joins `port_designations`, which ships header-only, so the designated
    ports are the corpus's; R-B-L2-002 finds the same ports without a designation list.
    """
    from morpheus.utils.binding_closer import CONFLICT  # pylint: disable=import-outside-toplevel
    from morpheus.utils.binding_closer import DISPLACED  # pylint: disable=import-outside-toplevel

    macs = rows(result, "tc2_mac")
    designated = macs[macs["port_key"].isin(single_host_ports)]
    too_many = designated[true(designated, "macs_per_port_first_in_window") & (number(designated, "macs_per_port") > 1)]
    stepped = macs[true(macs, "macs_per_port_first_in_window") & (number(macs, "macs_per_port_step") > NOVELTY_STEP)]

    arp = rows(result, "tc2_arp")
    contested = arp[(number(arp, "macs_claiming_sender_ip") > 1) & false(arp, "arp_sender_ip_excluded")]

    bindings = rows(result, "tc2_binding")
    elsewhere = bindings[bindings["bind_end_reason"].isin([CONFLICT, DISPLACED])]
    spoofs = elsewhere[number(elsewhere, "bind_gap_ns") <= SPOOF_GAP_NS]

    auth = rows(result, "tc2_auth")
    unpaired = auth[true(auth, "auth_unpaired")]

    return {
        "R-D-L2-001": keys(too_many, ["port_key", "mac_address"]),
        "R-B-L2-002": keys(stepped, ["port_key"]),
        "R-D-L2-003": keys(contested, ["arp_sender_ip", "arp_sender_mac"]),
        "R-D-L2-004": keys(spoofs, ["mac_address", "port_key"]),
        "R-D-L2-005": keys(unpaired, ["auth_port_key", "mac_address"]),
    }


# --- Layer 3, over the network corpus --------------------------------------------------------------------------

FANOUT = "R-B-L3-001 - Fan-out expansion"
BEACON = "R-B-L3-002 - Beaconing"
TTL = "R-B-L3-004 - TTL fingerprint shift"
SUMMARY = "Behavior summary - per-layer scores"
FANOUT_DESTINATIONS = threshold(FANOUT, r"dsts_per_src\s*>\s*([\d.]+)")
FANOUT_INTERNAL_RATIO = threshold(FANOUT, r"internal_dst_ratio\s*>\s*([\d.]+)")
BEACON_INTERVAL_CV = threshold(BEACON, r"flow_interval_cv\s*<\s*([\d.]+)")
BEACON_SIZE_CV = threshold(BEACON, r"flow_size_cv\s*<\s*([\d.]+)")
TTL_SHIFTED_FLOWS = threshold(TTL, r"shifted_flows\s*>=\s*([\d.]+)")
SUMMARY_BIN_NS = int(threshold(SUMMARY, r"bin\s+_time\s+span\s*=\s*(\d+)m")) * 60 * 1_000_000_000


def _allowed_scanners() -> set:
    """`lookup scanner_allowlist src_ip OUTPUT allowed`, from the shipped lookup."""
    with open(os.path.join(LOOKUPS, "scanner_allowlist.csv"), encoding="utf-8", newline="") as handle:
        return {row["src_ip"] for row in csv.DictReader(handle) if (row.get("allowed") or "") == "true"}


def behavior_summary(result: pd.DataFrame) -> pd.DataFrame:
    """
    The summary search's collect, for layer 3: `bin _time span=5m | stats max(dsts_per_src) AS peak_destinations
    BY _time osi_layer entity_key lineage_id`, with `_time` the event time.
    """
    scored = result[(result["osi_layer"] == 3).fillna(False)]
    scored = scored[scored["entity_key"].notna() & (scored["entity_key"] != "") & scored["lineage_id"].notna()]
    binned = scored.assign(_time=(scored["event_time"].astype("int64") // SUMMARY_BIN_NS) * SUMMARY_BIN_NS,
                           peak=number(scored, "dsts_per_src"))

    return (binned.groupby(["_time", "osi_layer", "entity_key", "lineage_id"],
                           dropna=False)["peak"].max().rename("peak_destinations").reset_index())


def _fan_out_trajectory(flows: pd.DataFrame) -> set:
    """
    R-P-L3-005 over the summary rebuilt above: `streamstats current=f last(peak_destinations) AS
    previous_destinations BY entity_key | streamstats current=f last(previous_destinations) AS earlier_destinations
    BY entity_key | where isnotnull(earlier_destinations) AND peak_destinations > previous_destinations AND
    previous_destinations > earlier_destinations`, keyed on the source and the hourly chain the rise completed in.

    This modelled two passes before the search had them. The search used to compute both in one streamstats, and
    a search head answered the question this docstring left open: the second `last()` cannot see a field the same
    command is creating, so the search fired on nothing while this function fired on the rise. The search now
    takes two passes, and this is what it computes.
    """
    summary = behavior_summary(flows).sort_values(["entity_key", "_time", "lineage_id"], kind="mergesort")
    accused = set()

    for (entity, group) in summary.groupby("entity_key", sort=True):
        (previous, earlier) = (None, None)

        for (_, row) in group.iterrows():
            peak = None if pd.isna(row["peak_destinations"]) else float(row["peak_destinations"])

            if (None not in (earlier, previous, peak) and peak > previous > earlier):
                accused.add((str(entity), str(row["lineage_id"])))

            if (previous is not None):
                earlier = previous

            if (peak is not None):
                previous = peak

    return accused


def layer_3_decisions(result: pd.DataFrame) -> dict:
    """The five layer 3 rules."""
    flows = rows(result, "tc3")
    fanning = flows[(number(flows, "dsts_per_src") > FANOUT_DESTINATIONS)
                    & (number(flows, "internal_dst_ratio") > FANOUT_INTERNAL_RATIO)]
    fanning = fanning[~fanning["src_ip"].isin(_allowed_scanners())]
    beaconing = flows[true(flows, "flow_regularity_mature") & (number(flows, "flow_interval_cv") < BEACON_INTERVAL_CV)
                      & (number(flows, "flow_size_cv") < BEACON_SIZE_CV)]
    reserved = flows[true(flows, "dst_is_reserved") | true(flows, "dst_is_multicast")]
    shifted = flows[true(flows, "ip_ttl_shifted") & true(flows, "ip_ttl_mature")].groupby("src_ip").size()

    return {
        "R-B-L3-001": keys(fanning, ["src_ip"]),
        "R-B-L3-002": keys(beaconing, ["flow_pair_key"]),
        # One notable per flow: the search ends in `table`. event_uid comes from the collector sequence, not the clock.
        "R-D-L3-003": keys(reserved, ["src_ip", "dst_ip", "dst_port", "event_uid"]),
        "R-B-L3-004": {(str(source), )
                       for (source, count) in shifted.items() if count >= TTL_SHIFTED_FLOWS},
        "R-P-L3-005": _fan_out_trajectory(flows),
    }


# --- Layer 4, over the transport corpus ------------------------------------------------------------------------

SYN = "R-D-L4-002 - SYN without completion"
RST = "R-D-L4-003 - RST ratio"
SCAN_SYN_RATIO = threshold(SYN, r"\(syn\s*/\s*flags\)\s*>=\s*([\d.]+)")
SCAN_PORTS = threshold(SYN, r"where\s+destination_ports\s*>\s*([\d.]+)")
SCAN_ACK = threshold(SYN, r"\back\s*=\s*([\d.]+)")
SCAN_FLAGS_FLOOR = threshold(SYN, r"where\s+flags\s*>\s*([\d.]+)")
REFUSAL_RATIO = threshold(RST, r"\(rst\s*/\s*flags\)\s*>=\s*([\d.]+)")
REFUSAL_FLAGS_FLOOR = threshold(RST, r"where\s+flags\s*>\s*([\d.]+)")
OUTAGE_CLIENTS = threshold(RST, r"clients_refused\s*>=\s*([\d.]+)\s*AND\s*servers_refusing\s*=")
OUTAGE_SERVERS = threshold(RST, r"clients_refused\s*>=\s*[\d.]+\s*AND\s*servers_refusing\s*=\s*([\d.]+)")
ENUMERATION_SERVERS = threshold(RST, r"servers_refusing\s*>=\s*([\d.]+)\s*AND\s*clients_refused\s*=")
ENUMERATION_CLIENTS = threshold(RST, r"servers_refusing\s*>=\s*[\d.]+\s*AND\s*clients_refused\s*=\s*([\d.]+)")


def _flow_bins(tc4: pd.DataFrame) -> pd.DataFrame:
    """Both refusal searches' first `stats`: one row per flow per bin, the maxima of the running flag counts."""
    by = ["flow_id", "rollup_time_ns", "src_ip", "dst_ip", "dst_port"]
    frame = tc4.dropna(subset=by).copy()

    for column in ("flow_syn", "flow_ack", "flow_rst", "flow_all"):
        frame[column] = number(frame, column)

    return frame.groupby(by).agg(syn=("flow_syn", "max"),
                                 ack=("flow_ack", "max"),
                                 rst=("flow_rst", "max"),
                                 flags=("flow_all", "max")).reset_index()


def layer_4_decisions(result: pd.DataFrame) -> dict:
    """
    The three layer 4 rules. R-D-L4-002 and R-D-L4-003 group by `rollup_time_ns`, the bin, so a scan the skew
    splits across two bins is a different accusation -- which is what an analyst would see.
    """
    tc4 = rows(result, "tc4")
    bins = _flow_bins(tc4)

    unanswered = bins[(bins["flags"] > SCAN_FLAGS_FLOOR) & (bins["ack"] == SCAN_ACK)
                      & ((bins["syn"] / bins["flags"]) >= SCAN_SYN_RATIO)]
    reach = unanswered.groupby(["src_ip", "rollup_time_ns"])["dst_port"].nunique().reset_index()

    refusing = bins[(bins["flags"] > REFUSAL_FLAGS_FLOOR) & ((bins["rst"] / bins["flags"]) >= REFUSAL_RATIO)].copy()
    refusals = set()

    if (len(refusing) > 0):
        refusing["clients_refused"] = refusing.groupby(["src_ip", "rollup_time_ns"])["dst_ip"].transform("nunique")
        refusing["servers_refusing"] = refusing.groupby(["dst_ip", "rollup_time_ns"])["src_ip"].transform("nunique")
        outage = (refusing["clients_refused"] >= OUTAGE_CLIENTS) & (refusing["servers_refusing"] == OUTAGE_SERVERS)
        enumeration = (~outage & (refusing["servers_refusing"] >= ENUMERATION_SERVERS)
                       & (refusing["clients_refused"] == ENUMERATION_CLIENTS))
        refusing["refusal_direction"] = None
        refusing.loc[outage, "refusal_direction"] = "service_outage"
        refusing.loc[enumeration, "refusal_direction"] = "closed_port_enumeration"
        refusing = refusing[refusing["refusal_direction"].notna()]
        refusing = refusing.assign(
            entity_key=refusing["src_ip"].where(refusing["refusal_direction"] == "service_outage", refusing["dst_ip"]))
        refusals = keys(refusing, ["entity_key", "refusal_direction", "rollup_time_ns"])

    breached = tc4[true(tc4, "flow_data_len_envelope_breached") | true(tc4, "flow_bpp_envelope_breached")]

    return {
        "R-D-L4-002": keys(reach[reach["dst_port"] > SCAN_PORTS], ["src_ip", "rollup_time_ns"]),
        "R-D-L4-003": refusals,
        "R-B-L4-005": keys(breached, ["transfer_triple"]),
    }


# --- Layer 6, over the presentation corpus ---------------------------------------------------------------------

SETTLED_HANDSHAKES = threshold("R-B-L6-001 - New TLS client fingerprint", r"ja4_client_observations\s*>=\s*([\d.]+)")
SINGLE_ISSUER = threshold("R-D-L6-002 - Certificate issuer anomaly", r"cert_issuer_distinct\s*=\s*([\d.]+)")


def layer_6_decisions(result: pd.DataFrame) -> dict:
    """The five layer 6 rules, each a filter on the handshake row and a `stats ... BY`."""
    tc6 = rows(result, "tc6")
    new_stack = tc6[true(tc6, "ja4_client_first_seen") & (number(tc6, "ja4_client_observations") >= SETTLED_HANDSHAKES)]
    intercepted = tc6[true(tc6, "cert_issuer_differs") & true(tc6, "cert_issuer_mature")
                      & (number(tc6, "cert_issuer_distinct") == SINGLE_ISSUER)]

    return {
        "R-B-L6-001":
            keys(new_stack, ["src_ip"]),
        "R-D-L6-002":
            keys(intercepted, ["dst_ip"]),
        "R-D-L6-003":
            keys(tc6[true(tc6, "cert_self_signed_external")], ["dst_ip"]),
        "R-B-L6-004":
            keys(tc6[true(tc6, "cipher_downgraded") & true(tc6, "cipher_mature")],
                 ["tls_pair_key", "src_ip", "dst_ip"]),
        "R-D-L6-005":
            keys(tc6[true(tc6, "content_category_crossed")], ["src_ip", "dst_ip"]),
    }


# --- Layer 5, over the session corpus -------------------------------------------------------------------------

TRAVEL = "R-D-L5-003 - Impossible travel"
FATIGUE = "R-D-L5-004 - Multi-factor fatigue"
DRIFT = "R-P-L5-006 - Drift trajectory"
IMPOSSIBLE_KMH = threshold(TRAVEL, r"kmh_threshold\s*=\s*([\d.]+)")
FATIGUE_CHALLENGES = threshold(FATIGUE, r"challenge_threshold\s*=\s*([\d.]+)")
FATIGUE_DENIALS = threshold(FATIGUE, r"denial_threshold\s*=\s*([\d.]+)")
DRIFT_RISING_WINDOWS = threshold(DRIFT, r"rising_threshold\s*=\s*([\d.]+)")
DRIFT_RISE_SIGMAS = threshold(DRIFT, r"sigma_threshold\s*=\s*([\d.]+)")
DRIFT_MEAN_CEILING = threshold(DRIFT, r"mean_ceiling\s*=\s*([\d.]+)")
DRIFT_ACCELERATION_CEILING = threshold(DRIFT, r"acceleration_ceiling\s*=\s*([\d.]+)")
FAILURE_RUN = threshold("R-D-L5-009 - Failed authentication run ending in success", r"failure_threshold\s*=\s*([\d.]+)")
COMPOSITE_MAX = threshold("R-B-L5-001 - Composite authentication anomaly", r"max_threshold\s*=\s*([\d.]+)")
COMPOSITE_MEAN = threshold("R-B-L5-001 - Composite authentication anomaly", r"mean_threshold\s*=\s*([\d.]+)")
LOCATION_LOSS = threshold("R-B-L5-002 - Location novelty anomaly", r"loss_threshold\s*=\s*([\d.]+)")
DURATION_RATIO = threshold("R-B-L5-005 - Session duration anomaly", r"ratio_threshold\s*=\s*([\d.]+)")

requires(TRAVEL, r"travel_status=measured")


def layer_5_decisions(result: pd.DataFrame) -> dict:
    """
    The nine layer 5 rules. Keyed on the principal, the search's `entity_key`, and for R-P-L5-006 the day it fired;
    the notable dedup keys carry an hourly `window_id`, which is a time and would count an accusation that moved an
    hour as a different one. R-B-L5-001 and R-B-L5-002 are gated on `model_fallback_used=false`, so they read the
    principals' own committed models and never the joiner's population score.
    """
    auth = rows(result, "tc5_auth")
    sessions = rows(result, "tc5_session")
    success = auth["auth_result"] == "success"

    travelled = auth[(auth["travel_status"] == "measured") & (number(auth, "travel_kmh") >= IMPOSSIBLE_KMH)]
    fatigued = auth[(number(auth, "mfa_attempts_in_window") > FATIGUE_CHALLENGES)
                    & (number(auth, "mfa_denials_in_window") >= FATIGUE_DENIALS)
                    & true(auth, "mfa_denied_then_approved")]
    drifting = auth[true(auth, "drift_mature") & (number(auth, "drift_rising_windows") >= DRIFT_RISING_WINDOWS)
                    & (number(auth, "drift_rise_sigmas") > DRIFT_RISE_SIGMAS)
                    & (number(auth, "mean_abs_z") < DRIFT_MEAN_CEILING)
                    & (number(auth, "drift_acceleration").abs() < DRIFT_ACCELERATION_CEILING)]
    off_hours = auth[true(auth, "hour_unseen") & true(auth, "cadence_mature") & success]
    novel = auth[success & true(auth, "cadence_mature")
                 & (true(auth, "location_first_seen") | true(auth, "device_first_seen"))]
    failures = auth[true(auth, "auth_failed_then_succeeded")
                    & (number(auth, "consecutive_auth_failures") >= FAILURE_RUN)]
    composite = auth[false(auth, "model_fallback_used") & (number(auth, "max_abs_z") >= COMPOSITE_MAX)
                     & (number(auth, "mean_abs_z") >= COMPOSITE_MEAN)]
    location = auth[false(auth, "model_fallback_used") & (number(auth, "locincrement_z_loss") >= LOCATION_LOSS)]
    account_type = sessions["ctx_account_type"].fillna("unknown")
    long_sessions = sessions[true(sessions, "session_duration_mature") & (account_type != "service")
                             & (number(sessions, "session_duration_ratio") > DURATION_RATIO)]

    return {
        "R-D-L5-003": keys(travelled, ["user_principal"]),
        "R-D-L5-004": keys(fatigued, ["user_principal"]),
        "R-P-L5-006": keys(drifting, ["user_principal", "day_window_id"]),
        "R-D-L5-007": keys(off_hours, ["user_principal"]),
        "R-D-L5-008": keys(novel, ["user_principal"]),
        "R-D-L5-009": keys(failures, ["user_principal"]),
        "R-B-L5-001": keys(composite, ["user_principal"]),
        "R-B-L5-002": keys(location, ["user_principal"]),
        "R-B-L5-005": keys(long_sessions, ["user_principal", "session_key"]),
    }


# --- Layer 7, over the application, SaaS and endpoint corpora -------------------------------------------------

TUNNEL = "R-B-L7-001 - DNS tunneling"
ENUMERATION = "R-D-L7-005 - Enumeration"
BULK = "R-B-L7-002 - Bulk data access"
BREADTH = "R-P-L7-006 - Access breadth trajectory"
ANCESTRY = "R-B-L7-004 - Process ancestry novelty"
TUNNEL_ENTROPY = threshold(TUNNEL, r"dns_subdomain_entropy\s*>\s*([\d.]+)")
TUNNEL_LABEL_LENGTH = threshold(TUNNEL, r"dns_mean_label_length\s*>\s*([\d.]+)")
TUNNEL_SUBDOMAINS = threshold(TUNNEL, r"qualifying_subdomains\s*>\s*([\d.]+)")
ENUMERATION_PATHS = threshold(ENUMERATION, r"http_distinct_paths\s*>\s*([\d.]+)")
ENUMERATION_RATIO = threshold(ENUMERATION, r"http_4xx_in_window\s*>\s*([\d.]+)\s*\*\s*http_2xx_in_window")
BULK_RATIO = threshold(BULK, r"saas_record_ratio\s*>\s*([\d.]+)")
BREADTH_RISING_WEEKS = threshold(BREADTH, r"rising_weeks\s*>=\s*([\d.]+)")
BREADTH_ROLE_VERSIONS = threshold(BREADTH, r"role_versions\s*=\s*([\d.]+)")
BREADTH_SPAN_WEEKS = int(threshold(BREADTH, r"-(\d+)w@w1", key="dispatch.earliest_time"))

requires(BULK, r"saas_baseline_mature=true")
requires(ANCESTRY, r"endpoint_pair_novel=true")
requires(BREADTH, r"saas_object_types_in_week=\*")
requires(BREADTH, r'coalesce\(ctx_groups,\s*"none"\)')


def application_decisions(result: pd.DataFrame) -> dict:
    """R-B-L7-001 over DNS and R-D-L7-005 over HTTP."""
    dns = rows(result, "tc7_dns")
    http = rows(result, "tc7_http")

    qualifying = dns[(number(dns, "dns_subdomain_entropy") > TUNNEL_ENTROPY)
                     & (number(dns, "dns_mean_label_length") > TUNNEL_LABEL_LENGTH)]
    counts = qualifying.groupby("dns_registered_domain")["dns_subdomain"].nunique()
    enumerating = http[(number(http, "http_distinct_paths") > ENUMERATION_PATHS)
                       & (number(http, "http_4xx_in_window") > ENUMERATION_RATIO * number(http, "http_2xx_in_window"))]

    return {
        "R-B-L7-001": {(str(domain), )
                       for domain in counts[counts > TUNNEL_SUBDOMAINS].index},
        "R-D-L7-005": keys(enumerating, ["src_ip"]),
    }


def _creeping(saas: pd.DataFrame) -> set:
    """R-P-L7-006 run each Monday over its span: `where rising_weeks >= 4 AND role_versions = 1`, union of runs."""
    breadth = saas[saas["saas_object_types_in_week"].notna()]

    if (len(breadth) == 0):
        return set()

    breadth = breadth.assign(role=breadth["ctx_groups"].fillna("none"),
                             week=pd.to_numeric(breadth["week_window_id"]).astype(int),
                             rising=number(breadth, "drift_rising_windows"))
    fired = set()

    for week in sorted(breadth["week"].unique()):
        span = breadth[breadth["week"].between(week - BREADTH_SPAN_WEEKS + 1, week)]

        for (principal, group) in span.groupby("user_principal"):
            rising = group["rising"].max()

            if (pd.notna(rising) and rising >= BREADTH_RISING_WEEKS
                    and group["role"].nunique() == BREADTH_ROLE_VERSIONS):
                fired.add((str(principal), ))

    return fired


def saas_decisions(result: pd.DataFrame) -> dict:
    """
    R-B-L7-002 and R-P-L7-006. The watchlist R-P-L7-006 writes only raises R-B-L7-002's risk score, never whether
    it fires, so it is not part of what R-B-L7-002 accuses.
    """
    saas = rows(result, "tc7_saas")
    bulk = saas[true(saas, "saas_baseline_mature") & (number(saas, "saas_record_ratio") > BULK_RATIO)]

    return {
        "R-B-L7-002": keys(bulk, ["user_principal", "operation"]),
        "R-P-L7-006": _creeping(saas),
    }


def endpoint_decisions(result: pd.DataFrame) -> dict:
    """R-B-L7-004: `endpoint_pair_novel=true`, one notable per pair per host."""
    endpoint = rows(result, "tc7_endpoint")

    return {"R-B-L7-004": keys(endpoint[true(endpoint, "endpoint_pair_novel")], ["hostname", "endpoint_pair"])}


# --- The chains, over the campaign corpus ---------------------------------------------------------------------

LATERAL = "R-C-001 - Lateral movement chain"
C2 = "R-C-002 - TLS anomaly precedes beaconing"
EXFIL = "R-C-004 - Staged exfiltration"
REPLAY = "R-C-005 - Credential replay across the stack"
NS_PER_SECOND = 1_000_000_000

LATERAL_LOGIN_TOLERANCE_NS = int(threshold(LATERAL, r"t_login >= t_fanout - (\d+) AND")) * NS_PER_SECOND
LATERAL_LOGIN_WINDOW_NS = int(threshold(LATERAL, r"t_login - t_fanout <= (\d+)")) * NS_PER_SECOND
LATERAL_PROCESS_TOLERANCE_NS = int(threshold(LATERAL, r"t_process >= t_login - (\d+) AND")) * NS_PER_SECOND
LATERAL_PROCESS_WINDOW_NS = int(threshold(LATERAL, r"t_process - t_fanout <= (\d+)")) * NS_PER_SECOND
LATERAL_PREVIOUS_WINDOW = int(threshold(LATERAL, r"eval window_id = window_id \+ (\d+)"))
C2_TOLERANCE_NS = int(threshold(C2, r"t_beacon >= t_tls - (\d+) AND")) * NS_PER_SECOND
C2_WINDOW_NS = int(threshold(C2, r"t_beacon - t_tls <= (\d+)")) * NS_PER_SECOND
C2_MIN_OBSERVATIONS = threshold(C2, r"ja4_client_observations>=(\d+)")
C2_INTERVAL_CV = threshold(C2, r"flow_interval_cv<([\d.]+)")
C2_SIZE_CV = threshold(C2, r"flow_size_cv<([\d.]+)")
EXFIL_RECORD_RATIO = threshold(EXFIL, r"saas_record_ratio>([\d.]+)")
EXFIL_BREACH_TOLERANCE_NS = int(threshold(EXFIL, r"t_breach >= t_export - (\d+) AND")) * NS_PER_SECOND
EXFIL_BREACH_WINDOW_NS = int(threshold(EXFIL, r"t_breach - t_export <= (\d+)")) * NS_PER_SECOND
EXFIL_HANDSHAKE_TOLERANCE_NS = int(threshold(EXFIL, r"t_handshake >= t_breach - (\d+) AND")) * NS_PER_SECOND
EXFIL_HANDSHAKE_WINDOW_NS = int(threshold(EXFIL, r"t_handshake - t_export <= (\d+)")) * NS_PER_SECOND
REPLAY_KMH = threshold(REPLAY, r"site_travel_kmh >= ([\d.]+)")
JOIN_TOLERANCE_NS = {
    "R-C-001": LATERAL_LOGIN_TOLERANCE_NS, "R-C-002": C2_TOLERANCE_NS, "R-C-004": EXFIL_BREACH_TOLERANCE_NS
}
"""The tolerance each joined chain allows a later step to precede an earlier one by, as its search states it."""

requires(EXFIL, r"cert_issuer_new_to_estate=true")


def _lateral_movement(result: pd.DataFrame) -> set:
    """
    R-C-001: a source's first flow in an hour whose fan-out rises above its peak in the hour before, then a first
    login from that source to a host within the window, then a novel process on that host, each step allowed to
    precede the last by the join tolerance. Keyed `(src_ip, user_principal, target)`, the search's `stats ... BY`.
    """
    flows = rows(result, "tc3")
    flows = flows.loc[flows["dsts_per_src"].notna(), ["src_ip", "window_id", "dsts_per_src", "event_time"]]
    flows = flows.astype({"window_id": "int64", "dsts_per_src": "int64", "event_time": "int64"})
    peaks = flows.groupby(["src_ip", "window_id"])["dsts_per_src"].max().rename("previous_peak").reset_index()
    peaks["window_id"] = peaks["window_id"] + LATERAL_PREVIOUS_WINDOW
    rising = flows.merge(peaks, on=["src_ip", "window_id"], how="inner")
    rising = rising[rising["dsts_per_src"] > rising["previous_peak"]]
    starts = rising.groupby(["src_ip", "window_id"], as_index=False)["event_time"].min()

    logins = rows(result, "tc5_auth")
    logins = logins[(logins["auth_result"] == "success") & true(logins, "target_host_first_seen")]
    processes = rows(result, "tc7_endpoint")
    processes = processes[true(processes, "endpoint_pair_novel")]
    fired = set()

    for (_, start) in starts.iterrows():
        t_fanout = int(start["event_time"])

        for (_, login) in logins[logins["source_ip"] == start["src_ip"]].iterrows():
            t_login = int(login["event_time"])

            if not (t_fanout - LATERAL_LOGIN_TOLERANCE_NS <= t_login <= t_fanout + LATERAL_LOGIN_WINDOW_NS):
                continue

            target = str(login["target_host"]).lower()

            for (_, process) in processes[processes["hostname"].str.lower() == target].iterrows():
                t_process = int(process["event_time"])

                if (t_process >= t_login - LATERAL_PROCESS_TOLERANCE_NS
                        and t_process - t_fanout <= LATERAL_PROCESS_WINDOW_NS):
                    fired.add((str(start["src_ip"]), str(login["user_principal"]), target))

    return fired


def _tls_before_beaconing(result: pd.DataFrame) -> set:
    """R-C-002: a settled host's new client fingerprint to a destination, then that pair's first beacon within the
    window, the beacon allowed to precede it by the tolerance. Keyed `(src_ip, dst_ip)`."""
    handshakes = rows(result, "tc6")
    fingerprints = handshakes[true(handshakes, "ja4_client_first_seen")
                              & (number(handshakes, "ja4_client_observations") >= C2_MIN_OBSERVATIONS)]
    flows = rows(result, "tc3")
    flows = flows[true(flows, "flow_regularity_mature") & (number(flows, "flow_interval_cv") < C2_INTERVAL_CV)
                  & (number(flows, "flow_size_cv") < C2_SIZE_CV)]
    beacons = flows.groupby(["src_ip", "dst_ip"])["event_time"].min().astype("int64")
    fired = set()

    for (_, fingerprint) in fingerprints.iterrows():
        pair = (fingerprint["src_ip"], fingerprint["dst_ip"])

        if (pair in beacons.index):
            t_tls = int(fingerprint["event_time"])
            t_beacon = int(beacons[pair])

            if (t_tls - C2_TOLERANCE_NS <= t_beacon <= t_tls + C2_WINDOW_NS):
                fired.add((str(pair[0]), str(pair[1])))

    return fired


def _staged_exfiltration(result: pd.DataFrame) -> set:
    """R-C-004: a bulk export, then an envelope breach from an address the principal's open session holds, then a
    handshake from that address to an issuer new to the estate, all within the window. Keyed `(user_principal,
    src_ip)`. The session interval is `min` of its starts and `max` of its ends, as the search's stats has it."""
    exports = rows(result, "tc7_saas")
    exports = exports[true(exports, "saas_baseline_mature")
                      & (number(exports, "saas_record_ratio") > EXFIL_RECORD_RATIO)]
    transfers = rows(result, "tc4")
    breaches = transfers[true(transfers, "flow_data_len_envelope_breached")
                         | true(transfers, "flow_bpp_envelope_breached")]
    handshakes = rows(result, "tc6")
    handshakes = handshakes[true(handshakes, "cert_issuer_new_to_estate")]

    lifecycle = rows(result, "tc5_session").astype({"event_time": "int64"})
    by = ["session_key", "user_principal", "source_ip"]
    opened = lifecycle[lifecycle["session_lifecycle"] == "start"].groupby(by)["event_time"].min().rename("opened")
    closed = lifecycle[lifecycle["session_lifecycle"] == "end"].groupby(by)["event_time"].max().rename("closed")
    sessions = opened.reset_index().merge(closed.reset_index(), on=by, how="left")
    fired = set()

    for (_, export) in exports.iterrows():
        t_export = int(export["event_time"])
        own = sessions[sessions["user_principal"] == export["user_principal"]]

        for (_, breach) in breaches.iterrows():
            t_breach = int(breach["event_time"])
            address = breach["src_ip"]
            held = own[(own["source_ip"] == address) & (own["opened"] <= t_breach)
                       & (own["closed"].isna() | (own["closed"] >= t_breach))]

            if (len(held) == 0
                    or not t_export - EXFIL_BREACH_TOLERANCE_NS <= t_breach <= t_export + EXFIL_BREACH_WINDOW_NS):
                continue

            for (_, handshake) in handshakes[handshakes["src_ip"] == address].iterrows():
                t_handshake = int(handshake["event_time"])

                if (t_handshake >= t_breach - EXFIL_HANDSHAKE_TOLERANCE_NS
                        and t_handshake - t_export <= EXFIL_HANDSHAKE_WINDOW_NS):
                    fired.add((str(export["user_principal"]), str(address)))

    return fired


def _credential_replays(result: pd.DataFrame) -> set:
    """R-C-005: a principal's sign-in resolved to a site faster than the journey from the last one allows. No join
    tolerance; both sign-ins come through one identity provider, so its clock moves them together."""
    logins = rows(result, "tc5_auth")
    logins = logins[logins["site_travel_status"].isin(["measured", "first_for_principal"])]
    fired = set()

    for (principal, group) in logins.sort_values("event_time", kind="mergesort").groupby("user_principal"):
        later = group.iloc[1:]

        if (((later["site_travel_status"] == "measured") & (number(later, "site_travel_kmh") >= REPLAY_KMH)).any()):
            fired.add((str(principal), ))

    return fired


def chain_decisions(result: pd.DataFrame) -> dict:
    """The four chained detections over the campaign pipeline's output."""
    return {
        "R-C-001": _lateral_movement(result),
        "R-C-002": _tls_before_beaconing(result),
        "R-C-004": _staged_exfiltration(result),
        "R-C-005": _credential_replays(result),
    }
