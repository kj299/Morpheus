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
How much clock skew between collectors does each shipped detection tolerate, measured rather than argued.

Every join in this design is a join on time across sources that do not share a clock, and the guide's collection
section argues qualitatively that some features are far more sensitive than others. Nothing measured it. This
script does: it spreads the collectors' clocks across a window of a given width, re-runs the composed pipelines
over the same seeded corpora, and reports which columns move, by how much, and at what width each rule changes
its mind.

**The magnitude is the width of the spread, not a shift.** A shift moves every record together and a pipeline
keyed on event time barely notices; what breaks a join is two collectors disagreeing. So at magnitude `M` the
offsets are spread evenly across `[-M/2, +M/2]` by sorted collector name: the worst pair of clocks disagrees by
exactly `M`, every source but the middle one moves, and no source is privileged by the scheme. `M` is then
readable as the sentence an estate needs -- "this rule needs the fleet synchronized to better than `M`".

**A decision is keyed on what it accuses, never on when.** A detection identified by timestamp would differ
under every non-zero offset, which would measure the injection rather than the damage. Each rule here reduces to
the set of entities it flags -- a MAC and a port, an address, a principal -- so a change in that set means the
rule accused something different, which is the only change an analyst would ever see. Counts are reported
beside the sets, because the same entity flagged twice as often is also a change.

Row identity survives the perturbation, which is what makes a column-by-column answer possible at all:
`event_uid` is derived from the collector sequence rather than from the timestamp, so the same record keeps its
identifier however its clock is set, and baseline and skewed frames align row for row.

What this does not do is correct anything. `clock_source` and `clock_offset_ms` are in the envelope and nothing
produces or consumes them; this measures the damage a deployment takes today, not the damage after a correction
that has not been built.
"""

import argparse
import datetime
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HARNESS = os.path.join(REPO_ROOT, "tests", "morpheus", "determinism")

MILLISECOND_NS = 1_000_000
SECOND_NS = 1000 * MILLISECOND_NS

MAGNITUDES_NS = (
    MILLISECOND_NS,
    10 * MILLISECOND_NS,
    100 * MILLISECOND_NS,
    SECOND_NS,
    10 * SECOND_NS,
    30 * SECOND_NS,
    60 * SECOND_NS,
)
"""The spread widths swept, one millisecond to one minute.

A millisecond is what a well-run NTP fleet on a local network holds. A minute is what an estate with a device
nobody has looked at in a year actually has. The decades between them are where the answer is.
"""

HALF_WINDOW_NS = 1800 * SECOND_NS
"""Half of the layer 5 pipeline's hourly window.

Forty-five of the layer 5 corpus's hundred and five authentications sit exactly on an hour mark, because the
corpus builds its times from whole hours. An event on a boundary changes window under an offset of one
nanosecond, so a sweep of that corpus measures how round its numbers are as much as how fragile the rules are.
Moving every clock by half a window puts the same events in the middle of theirs and changes nothing else.
"""

IMPOSSIBLE_KMH = 900
"""R-D-L5-003's threshold, matching the saved search and the layer 5 harness."""

FATIGUE_CHALLENGES = 5
FATIGUE_DENIALS = 4
"""R-D-L5-004's two thresholds: more than five challenges in the window, at least four of them denied."""

DRIFT_RISING_WINDOWS = 4
DRIFT_RISE_SIGMAS = 1.5
DRIFT_MEAN_CEILING = 2.0
"""R-P-L5-006's thresholds, matching the saved search."""


def offsets_for(sources: list, magnitude_ns: int) -> dict:
    """
    Spread the sources' clocks evenly across a window `magnitude_ns` wide, by sorted name.

    Sorted so the assignment is a property of the estate rather than of dictionary order, and evenly spread so
    that the magnitude means one thing: the worst pair of clocks in the estate disagrees by exactly this much.
    """
    ordered = sorted(sources)

    if (len(ordered) == 0):
        return {}

    if (len(ordered) == 1):
        return {ordered[0]: 0}

    span = len(ordered) - 1

    return {source: -(magnitude_ns // 2) + (index * magnitude_ns) // span for (index, source) in enumerate(ordered)}


COLLECTOR_CLOCKS = "collector_id"
"""The clock that stamped the record on its way out of the collector."""

SWITCH_CLOCKS = {"tc1": "device_id", "tc2_mac": "switch_id", "tc2_auth": "switch_id"}
"""The clock of the device that made the observation, where the record says which device that was.

The guide's collection section makes a specific claim -- that the MAC-in-two-places interval absorbs each
switch's offset directly -- and that claim is about switches rather than about collectors. A MAC table is
usually one feed carrying every switch, so a collector-level offset moves both sightings of a displaced MAC
together and cancels out of the interval entirely. Only a per-device offset tests what the guide said. ARP has
no device column here, and a laptop is not a clock that stamps anything, so both stay put.
"""


def sources_in(corpus: dict, columns) -> list:
    found = set()

    for (name, frame) in corpus.items():
        column = columns if isinstance(columns, str) else columns.get(name)

        if (column is not None and column in frame.columns):
            found |= set(frame[column].dropna())

    return sorted(found)


def skew_corpus(corpus: dict, offsets: dict, columns) -> dict:
    from morpheus.utils.determinism import apply_clock_skew  # pylint: disable=import-outside-toplevel

    skewed = {}

    for (name, frame) in corpus.items():
        column = columns if isinstance(columns, str) else columns.get(name)

        if (column is None or column not in frame.columns):
            skewed[name] = frame.copy().reset_index(drop=True)
            continue

        present = {source: offset for (source, offset) in offsets.items() if source in set(frame[column].dropna())}
        skewed[name] = apply_clock_skew(frame, present, source_column=column)

    return skewed


# --- What each rule accuses -----------------------------------------------------------------------------------


def _rows(result, telemetry_class: str):
    return result[result["telemetry_class"] == telemetry_class]


def _keys(frame, columns: list) -> set:
    return {tuple(str(row[column]) for column in columns) for (_, row) in frame.iterrows()}


def layer_2_decisions(result, single_host_ports: set, spoof_gap_ns: int) -> dict:
    """The four layer 2 detections, each reduced to the set of entities it accuses."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    from morpheus.utils.binding_closer import CONFLICT  # pylint: disable=import-outside-toplevel
    from morpheus.utils.binding_closer import DISPLACED  # pylint: disable=import-outside-toplevel

    macs = _rows(result, "tc2_mac")
    designated = macs[macs["port_key"].isin(single_host_ports)]
    too_many = designated[(designated["macs_per_port_first_in_window"] == True)  # noqa: E712  pylint: disable=singleton-comparison
                          & (designated["macs_per_port"] > 1)]

    arp = _rows(result, "tc2_arp")
    contested = arp[(arp["macs_claiming_sender_ip"].fillna(0) > 1) & (arp["arp_sender_ip_excluded"] == False)]  # noqa: E712  pylint: disable=singleton-comparison

    bindings = _rows(result, "tc2_binding")
    elsewhere = bindings[bindings["bind_end_reason"].isin([CONFLICT, DISPLACED])]
    spoofs = elsewhere[elsewhere["bind_gap_ns"] <= spoof_gap_ns]

    auth = _rows(result, "tc2_auth")
    unpaired = auth[auth["auth_unpaired"] == True]  # noqa: E712  pylint: disable=singleton-comparison

    del pd

    return {
        "R-D-L2-001": _keys(too_many, ["port_key", "mac_address"]),
        "R-D-L2-003": _keys(contested, ["arp_sender_ip", "arp_sender_mac"]),
        "R-D-L2-004": _keys(spoofs, ["mac_address", "port_key"]),
        "R-D-L2-005": _keys(unpaired, ["auth_port_key", "mac_address"]),
    }


def layer_5_decisions(result) -> dict:
    """The three layer 5 rules, each reduced to the set of principals it accuses."""
    auth = _rows(result, "tc5_auth")

    travelled = auth[auth["travel_kmh"].fillna(0) > IMPOSSIBLE_KMH]
    fatigued = auth[(auth["mfa_attempts_in_window"].fillna(0) > FATIGUE_CHALLENGES)
                    & (auth["mfa_denials_in_window"].fillna(0) >= FATIGUE_DENIALS)
                    & (auth["mfa_denied_then_approved"] == True)]  # noqa: E712  pylint: disable=singleton-comparison

    drifting = auth
    if ("drift_mature" in auth.columns):
        drifting = auth[(auth["drift_mature"] == True)  # noqa: E712  pylint: disable=singleton-comparison
                        & (auth["drift_rising_windows"].fillna(0) >= DRIFT_RISING_WINDOWS)
                        & (auth["drift_rise_sigmas"].fillna(0) > DRIFT_RISE_SIGMAS)
                        & (auth["mean_abs_z"].fillna(99) < DRIFT_MEAN_CEILING)]

    return {
        "R-D-L5-003": _keys(travelled, ["user_principal"]),
        "R-D-L5-004": _keys(fatigued, ["user_principal"]),
        "R-P-L5-006": _keys(drifting, ["user_principal", "day_window_id"]),
    }


def chain_reach(result) -> dict:
    """
    How far the ladder still reaches, rung by rung.

    The span of a chain and the attribution inside it are different questions, and they do not move together.
    A chain can hold all three layers while an observation inside it has quietly changed which port it is
    attributed to, so both are counted: the spans, the sign-ins that still found the desk they were made at,
    and the ARP observations still rooted on a port rather than falling back to their own address.
    """
    chained = result[result["lineage_id"].notna() & (result["lineage_id"] != "")]
    spans = chained.groupby("lineage_id")["osi_layer"].nunique()

    auth = _rows(result, "tc5_auth")
    resolved = int(auth["desk_port_key"].notna().sum()) if ("desk_port_key" in auth.columns) else 0

    arp = _rows(result, "tc2_arp")
    rooted = 0

    if ("chain_anchor_source" in arp.columns):
        rooted = int((arp["chain_anchor_source"] == "resolved_port_key").sum())

    return {
        "chains": int(len(spans)),
        "three_layer_chains": int((spans >= 3).sum()),
        "two_layer_chains": int((spans == 2).sum()),
        "sign_ins_resolved_to_a_port": resolved,
        "arp_observations_rooted_on_a_port": rooted,
    }


# --- Comparing one run against the baseline -------------------------------------------------------------------


def compare_columns(baseline, skewed, key_columns: list, ignore: set) -> list:
    """Per column, how many rows moved and by how much, over rows aligned on their identifiers."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    left = baseline.set_index(key_columns).sort_index()
    right = skewed.set_index(key_columns).sort_index()
    shared = left.index.intersection(right.index)
    left = left.loc[shared]
    right = right.loc[shared]

    moved = []

    for column in left.columns:
        if (column in ignore or column not in right.columns):
            continue

        a = left[column]
        b = right[column]
        unequal = ~((a == b) | (a.isna() & b.isna()))
        rows = int(unequal.sum())

        if (rows == 0):
            continue

        entry = {"column": column, "rows_changed": rows}

        try:
            delta = (pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")).abs()
            largest = delta.max()

            if (pd.notna(largest)):
                entry["max_abs_delta"] = float(largest)
        except (TypeError, ValueError):
            pass

        moved.append(entry)

    return sorted(moved, key=lambda entry: (-entry["rows_changed"], entry["column"]))


def compare_decisions(baseline: dict, skewed: dict) -> dict:
    """Per rule, what it stopped accusing and what it started accusing."""
    report = {}

    for rule in sorted(baseline):
        before = baseline[rule]
        after = skewed[rule]
        report[rule] = {
            "baseline": len(before),
            "skewed": len(after),
            "no_longer_flagged": sorted(":".join(key) for key in (before - after)),
            "newly_flagged": sorted(":".join(key) for key in (after - before)),
            "changed": before != after,
        }

    return report


# --- The sweep ------------------------------------------------------------------------------------------------


def gap_threshold_ns() -> int:
    """R-D-L2-004's threshold, read from the shipped search rather than restated here."""
    import re  # pylint: disable=import-outside-toplevel

    path = os.path.join(REPO_ROOT,
                        "examples",
                        "splunk_lineage_app",
                        "TA-morpheus-lineage",
                        "default",
                        "savedsearches.conf")

    with open(path, encoding="utf-8") as handle:
        return int(re.search(r"gap_threshold\s*=\s*(\d+)", handle.read()).group(1))


KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS = {"event_time"}
"""`event_time` is the injection itself, so reporting it as a casualty would be counting the bullet as a wound."""


def shift_corpus(corpus: dict, columns, shift_ns: int) -> dict:
    """Move every clock by the same amount, which is a shift rather than a skew and changes no relationship."""
    if (shift_ns == 0):
        return corpus

    return skew_corpus(corpus, {source: shift_ns for source in sources_in(corpus, columns)}, columns)


def sweep(label: str, module, decisions, extra, columns=COLLECTOR_CLOCKS, shift_ns: int = 0) -> dict:
    """
    Run one pipeline at every magnitude and report what moved.

    `shift_ns` moves every clock together before the sweep begins. That is not a perturbation -- no two clocks
    disagree any more than they did -- but it decides where the corpus sits relative to the window boundaries,
    and an event exactly on a boundary crosses it under an offset of one nanosecond. Sweeping the same corpus
    on and off the boundary is what separates a rule that is genuinely sensitive to skew from a rule that
    merely inherited a corpus built on round numbers.
    """
    from morpheus.utils.determinism import diff_frames  # pylint: disable=import-outside-toplevel

    config = module.build_pipeline_config()
    corpus = shift_corpus(module.build_corpus(), columns, shift_ns)
    sources = sources_in(corpus, columns)
    baseline = module.run_pipeline(config, corpus)
    baseline_decisions = decisions(baseline)

    print(f"\n=== {label}: {len(sources)} clocks, {len(baseline)} rows ===")
    print(f"    clocks: {', '.join(sources)}")

    runs = []

    for magnitude in MAGNITUDES_NS:
        offsets = offsets_for(sources, magnitude)
        result = module.run_pipeline(config, skew_corpus(corpus, offsets, columns))

        difference = diff_frames(baseline, result)
        moved_columns = compare_columns(baseline, result, KEY_COLUMNS, IGNORE_COLUMNS)
        rules = compare_decisions(baseline_decisions, decisions(result))
        changed = sorted(rule for (rule, entry) in rules.items() if entry["changed"])

        runs.append({
            "spread_ns": magnitude,
            "spread": _readable(magnitude),
            "offsets_ns": offsets,
            "output_identical": difference is None,
            "first_difference": difference,
            "columns_moved": moved_columns,
            "rules": rules,
            "rules_changed": changed,
            "measures": extra(result),
        })

        names = ", ".join(entry["column"] for entry in moved_columns[:4]) or "nothing"
        print(f"    {_readable(magnitude):>6}: {len(moved_columns):>3} columns moved ({names}), "
              f"rules changed: {', '.join(changed) or 'none'}")

    return {
        "clocks": sources,
        "skewed_by": columns if isinstance(columns, str) else dict(sorted(columns.items())),
        "uniform_shift_ns": shift_ns,
        "baseline_rows": int(len(baseline)),
        "baseline_measures": extra(baseline),
        "baseline_decisions": {
            rule: len(keys)
            for (rule, keys) in baseline_decisions.items()
        },
        "runs": runs,
    }


def _readable(nanoseconds: int) -> str:
    if (nanoseconds < SECOND_NS):
        return f"{nanoseconds // MILLISECOND_NS}ms"

    return f"{nanoseconds // SECOND_NS}s"


def breaking_point(sweeps: dict) -> dict:
    """
    Per rule, the narrowest spread at which it changed its mind, in each sweep that carries it.

    Kept per sweep rather than collapsed to one number, because the sweeps disagree and the disagreement is the
    result: a rule can be untouched by a minute of disagreement between collectors and undone by the same
    disagreement between switches, and a rule can look fragile on a corpus built from whole hours and be
    unmoved once the same events sit in the middle of their windows. One minimum across all of them would hide
    exactly the finding.
    """
    smallest: dict = {}

    for (name, report) in sweeps.items():
        for rule in report["baseline_decisions"]:
            smallest.setdefault(rule, {})[name] = None

        for run in report["runs"]:
            for rule in run["rules_changed"]:
                current = smallest[rule][name]

                if (current is None or run["spread_ns"] < current["spread_ns"]):
                    smallest[rule][name] = {"spread_ns": run["spread_ns"], "spread": run["spread"]}

    return {rule: dict(sorted(entry.items())) for (rule, entry) in sorted(smallest.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", default=os.path.join(REPO_ROOT, "clock_skew.json"))
    arguments = parser.parse_args()

    sys.path.insert(0, HARNESS)

    import estate_pipeline  # pylint: disable=import-outside-toplevel
    import session_pipeline  # pylint: disable=import-outside-toplevel
    import telemetry_pipeline  # pylint: disable=import-outside-toplevel

    threshold = gap_threshold_ns()
    single_host = telemetry_pipeline.SINGLE_HOST_PORTS

    layer_2 = lambda result: layer_2_decisions(result, single_host, threshold)  # noqa: E731  pylint: disable=unnecessary-lambda-assignment

    sweeps = {
        "estate_collectors":
            sweep("estate, collector clocks: layers 1, 2 and 5 over one hour", estate_pipeline, layer_2, chain_reach),
        "estate_switches":
            sweep("estate, switch clocks: the guide's own claim, tested",
                  estate_pipeline,
                  layer_2,
                  chain_reach,
                  columns=SWITCH_CLOCKS),
        "session":
            sweep("session, collector clocks: layer 5 over a week", session_pipeline, layer_5_decisions, lambda _: {}),
        "session_off_boundary":
            sweep("session, same but moved off the hour marks first",
                  session_pipeline,
                  layer_5_decisions, lambda _: {},
                  shift_ns=HALF_WINDOW_NS),
    }

    report = {
        "at":
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "spread_widths": [_readable(magnitude) for magnitude in MAGNITUDES_NS],
        "gap_threshold_ns":
            threshold,
        "breaking_point":
            breaking_point(sweeps),
        "sweeps":
            sweeps,
        "measures": ("the spread between collectors' clocks that each shipped rule tolerates, on these corpora. "
                     "The magnitude is the width the clocks are spread across, so the worst pair disagrees by "
                     "exactly that much. A rule is counted as changed when the set of entities it accuses "
                     "changes, never when only a timestamp moves. Nothing here corrects for skew: clock_source "
                     "and clock_offset_ms are in the envelope and no stage reads them, so this is the damage a "
                     "deployment takes today."),
    }

    with open(arguments.artifact, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print("\n=== breaking point: the narrowest spread that changed the rule's mind ===")

    for (rule, per_sweep) in report["breaking_point"].items():
        where = ", ".join(f"{name} {entry['spread'] if entry else 'never'}" for (name, entry) in per_sweep.items())
        print(f"    {rule}: {where}")

    print(f"\nArtifact written to {arguments.artifact}.")

    return 0


if (__name__ == "__main__"):
    sys.exit(main())
