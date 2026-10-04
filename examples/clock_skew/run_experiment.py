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

CHAIN_MAGNITUDES_NS = MAGNITUDES_NS + tuple(seconds * SECOND_NS for seconds in (120, 240, 600, 1800, 3600, 7200, 10800))
"""The chained rules' ladder: the same widths, then on past their 120-second join tolerance to three hours.

A chain allows a later step to precede an earlier one by its tolerance, and its steps come from different
collectors, so the tolerance is a claim about how far apart two clocks may be. A ladder that stopped at a minute
could not test it. Three hours is past every chain's window.
"""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rules  # noqa: E402  pylint: disable=wrong-import-position


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

    auth = rules.rows(result, "tc5_auth")
    resolved = int(auth["desk_port_key"].notna().sum()) if ("desk_port_key" in auth.columns) else 0

    arp = rules.rows(result, "tc2_arp")
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
    return rules.SPOOF_GAP_NS


KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS = {"event_time"}
"""`event_time` is the injection itself, so reporting it as a casualty would be counting the bullet as a wound."""


def shift_corpus(corpus: dict, columns, shift_ns: int) -> dict:
    """Move every clock by the same amount, which is a shift rather than a skew and changes no relationship."""
    if (shift_ns == 0):
        return corpus

    return skew_corpus(corpus, {source: shift_ns for source in sources_in(corpus, columns)}, columns)


def sweep(label: str,
          module,
          decisions,
          extra=lambda _: {},
          columns=COLLECTOR_CLOCKS,
          shift_ns: int = 0,
          magnitudes=MAGNITUDES_NS) -> dict:
    """
    Run one pipeline at every magnitude and report what moved.

    `shift_ns` moves every clock together before the sweep begins. That is not a perturbation -- no two clocks
    disagree any more than they did -- but it decides where the corpus sits relative to the window boundaries,
    and an event exactly on a boundary crosses it under an offset of one nanosecond. Sweeping the same corpus
    on and off the boundary is what separates a rule that is genuinely sensitive to skew from a rule that
    merely inherited a corpus built on round numbers.

    A sweep over one clock measures nothing: `offsets_for` gives a lone clock no offset, because nothing can
    disagree with itself. Such a sweep still runs, so its report is complete, and `breaking_point` says it was not
    measured rather than that the rule never moved.
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

    for magnitude in magnitudes:
        offsets = offsets_for(sources, magnitude)
        result = module.run_pipeline(config, skew_corpus(corpus, offsets, columns))

        difference = diff_frames(baseline, result)
        moved_columns = compare_columns(baseline, result, KEY_COLUMNS, IGNORE_COLUMNS)
        verdicts = compare_decisions(baseline_decisions, decisions(result))
        changed = sorted(rule for (rule, entry) in verdicts.items() if entry["changed"])

        runs.append({
            "spread_ns": magnitude,
            "spread": _readable(magnitude),
            "offsets_ns": offsets,
            "output_identical": difference is None,
            "first_difference": difference,
            "columns_moved": moved_columns,
            "rules": verdicts,
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


NOT_MEASURED = "one clock"
"""What `breaking_point` reports for a sweep with a single clock, which can perturb nothing."""


def breaking_point(sweeps: dict) -> dict:
    """
    Per rule, the narrowest spread at which it changed its mind, in each sweep that carries it.

    Kept per sweep rather than collapsed to one number, because the sweeps disagree and the disagreement is the
    result: a rule can be untouched by a minute of disagreement between collectors and undone by the same
    disagreement between switches, and a rule can look fragile on a corpus built from whole hours and be
    unmoved once the same events sit in the middle of their windows. One minimum across all of them would hide
    exactly the finding.

    Three answers, kept distinct because they mean different things: a spread, where the rule changed; `None`,
    where it held across the whole ladder; and `NOT_MEASURED`, where the sweep had one clock and so perturbed
    nothing. Reporting the last as `None` would turn the absence of a measurement into a tolerance.
    """
    smallest: dict = {}

    for (name, report) in sweeps.items():
        single = len(report["clocks"]) < 2

        for rule in report["baseline_decisions"]:
            smallest.setdefault(rule, {})[name] = NOT_MEASURED if single else None

        if (single):
            continue

        for run in report["runs"]:
            for rule in run["rules_changed"]:
                current = smallest[rule][name]

                if (current is None or run["spread_ns"] < current["spread_ns"]):
                    smallest[rule][name] = {"spread_ns": run["spread_ns"], "spread": run["spread"]}

    return {rule: dict(sorted(entry.items())) for (rule, entry) in sorted(smallest.items())}


def refine(module, decisions, rule: str, low_ns: int, high_ns: int, columns=COLLECTOR_CLOCKS) -> dict:
    """
    Narrow a rule's breaking point to the second, between a spread where it held and one where it did not.

    Bisects on whether the rule's accusations differ from the baseline, which assumes one transition between the
    two; the result names both edges, so a caller can check them, and what was lost and gained at the upper one.
    """
    config = module.build_pipeline_config()
    corpus = module.build_corpus()
    sources = sources_in(corpus, columns)
    baseline = decisions(module.run_pipeline(config, corpus))[rule]

    def decided(magnitude: int) -> set:
        return decisions(module.run_pipeline(config, skew_corpus(corpus, offsets_for(sources, magnitude),
                                                                 columns)))[rule]

    while (high_ns - low_ns > SECOND_NS):
        middle = ((low_ns + high_ns) // 2 // SECOND_NS) * SECOND_NS

        if (middle in (low_ns, high_ns)):
            break

        if (decided(middle) != baseline):
            high_ns = middle
        else:
            low_ns = middle

    changed = decided(high_ns)

    return {
        "held_at": _readable(low_ns),
        "changed_at": _readable(high_ns),
        "changed_at_ns": high_ns,
        "no_longer_flagged": sorted(":".join(key) for key in baseline - changed),
        "newly_flagged": sorted(":".join(key) for key in changed - baseline),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", default=os.path.join(REPO_ROOT, "clock_skew.json"))
    arguments = parser.parse_args()

    sys.path.insert(0, HARNESS)

    # pylint: disable=import-outside-toplevel
    import application_pipeline
    import campaign_pipeline
    import endpoint_pipeline
    import estate_pipeline
    import network_pipeline
    import presentation_pipeline
    import saas_pipeline
    import session_pipeline
    import telemetry_pipeline
    import transport_pipeline

    # pylint: enable=import-outside-toplevel

    single_host = telemetry_pipeline.SINGLE_HOST_PORTS

    def estate(result) -> dict:
        return {**rules.layer_1_decisions(result), **rules.layer_2_decisions(result, single_host)}

    sweeps = {
        "estate_collectors":
            sweep("estate, collector clocks: layers 1, 2 and 5 over one hour", estate_pipeline, estate, chain_reach),
        "estate_switches":
            sweep("estate, switch clocks: the guide's own claim, tested",
                  estate_pipeline,
                  estate,
                  chain_reach,
                  columns=SWITCH_CLOCKS),
        "session":
            sweep("session, collector clocks: layer 5 over a week", session_pipeline, rules.layer_5_decisions),
        "session_off_boundary":
            sweep("session, same but moved off the hour marks first",
                  session_pipeline,
                  rules.layer_5_decisions,
                  shift_ns=HALF_WINDOW_NS),
        "network":
            sweep("network, collector clocks: layer 3", network_pipeline, rules.layer_3_decisions),
        "transport":
            sweep("transport, collector clocks: layer 4", transport_pipeline, rules.layer_4_decisions),
        "presentation":
            sweep("presentation, collector clocks: layer 6", presentation_pipeline, rules.layer_6_decisions),
        "application":
            sweep("application, collector clocks: DNS and HTTP", application_pipeline, rules.application_decisions),
        "saas":
            sweep("saas, collector clocks: the provider's audit log", saas_pipeline, rules.saas_decisions),
        "endpoint":
            sweep("endpoint, collector clocks", endpoint_pipeline, rules.endpoint_decisions),
        "endpoint_hosts":
            sweep("endpoint, host clocks: each agent stamps its own process starts",
                  endpoint_pipeline,
                  rules.endpoint_decisions,
                  columns="hostname"),
        "campaign":
            sweep("campaign, collector clocks: the four chains, to three hours",
                  campaign_pipeline,
                  rules.chain_decisions,
                  magnitudes=CHAIN_MAGNITUDES_NS),
    }

    points = breaking_point(sweeps)
    chain_edges = {}

    for (rule, per_sweep) in points.items():
        entry = per_sweep.get("campaign")

        if (isinstance(entry, dict)):
            ladder = [magnitude for magnitude in CHAIN_MAGNITUDES_NS if magnitude < entry["spread_ns"]]
            chain_edges[rule] = refine(campaign_pipeline, rules.chain_decisions, rule, ladder[-1], entry["spread_ns"])

    report = {
        "at":
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "spread_widths": [_readable(magnitude) for magnitude in MAGNITUDES_NS],
        "chain_spread_widths": [_readable(magnitude) for magnitude in CHAIN_MAGNITUDES_NS],
        "gap_threshold_ns":
            gap_threshold_ns(),
        "join_tolerance_ns":
            rules.JOIN_TOLERANCE_NS,
        "breaking_point":
            points,
        "chain_breaking_point_to_the_second":
            chain_edges,
        "sweeps":
            sweeps,
        "measures": ("the spread between clocks that each shipped rule tolerates, on these corpora. The magnitude is "
                     "the width the clocks are spread across, so the worst pair disagrees by exactly that much. A "
                     "rule is counted as changed when the set of entities it accuses changes, never when only a "
                     "timestamp moves. A sweep over a corpus with one clock perturbs nothing, and is reported as "
                     "not measured rather than as a tolerance. Nothing here corrects for skew: clock_source and "
                     "clock_offset_ms are in the envelope and no stage reads them, so this is the damage a "
                     "deployment takes today."),
    }

    with open(arguments.artifact, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print("\n=== breaking point: the narrowest spread that changed the rule's mind ===")

    for (rule, per_sweep) in report["breaking_point"].items():
        where = ", ".join(f"{name} {entry['spread'] if isinstance(entry, dict) else (entry or 'never')}"
                          for (name, entry) in per_sweep.items())
        print(f"    {rule}: {where}")

    for (rule, edge) in chain_edges.items():
        print(f"    {rule}: held at {edge['held_at']}, changed at {edge['changed_at']}; "
              f"lost {edge['no_longer_flagged'] or 'nothing'}, gained {edge['newly_flagged'] or 'nothing'}")

    print(f"\nArtifact written to {arguments.artifact}.")

    return 0


if (__name__ == "__main__"):
    sys.exit(main())
