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
The clock-skew experiment's machinery, and the findings it produced, asserted so neither can go stale.

The guide asked how much clock skew between nodes degrades behavioral integrity, and answered that nothing had
measured it. `examples/clock_skew/run_experiment.py` measures it. What is written into the guide from that run
is three claims, and a claim in a document decays the moment the code stops agreeing with it -- so each is
asserted here against the pipelines rather than quoted from the artifact.

The findings, and why each is worth pinning:

- **R-D-L2-004 survives a minute of disagreement between collectors and fails one between switches.** The
  guide predicted the failure but not the condition. Two sightings of a displaced MAC arrive through one MAC
  table feed, so a collector's offset moves both and cancels out of the interval entirely; only the switches'
  own clocks pull them apart. A test that swept one axis would have confirmed the guide and missed the point.
- **A millisecond moves R-P-L5-006's inputs and not its accusations.** A hundred and twenty-seven of the layer 5
  corpus's three hundred and eighty-five authentications sit exactly on an hour mark, and an event on a boundary
  changes hour under an offset of one nanosecond, so a millisecond moves their hours, their surprise and their
  scores. With the reference arithmetic that once changed which principal-days the rule accused; with the
  learned models it does not, and neither does a minute once the events are moved off the marks. The sensitivity
  is to boundary proximity, not to magnitude, and whether it reaches a decision depends on the scorer.
- **The ladder holds.** Three-layer chains and the sign-ins that resolved to a port are unchanged at every width
  swept. Which window a chain belongs to moves; whether the ladder reaches does not.
"""

import importlib
import importlib.util
import os
import re
import sys

import pytest

from morpheus.utils.binding_closer import CONFLICT
from morpheus.utils.binding_closer import DISPLACED

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import estate_pipeline as ep  # noqa: E402
import session_pipeline as sp  # noqa: E402
import telemetry_pipeline as tp  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUNNER = os.path.join(REPO_ROOT, "examples", "clock_skew", "run_experiment.py")

BOUNDARY_ALIGNED_AUTHENTICATIONS = 127
"""Layer 5 authentications sitting exactly on an hour mark, out of three hundred and eighty-five.

The whole explanation of the millisecond finding rests on this number, so it is asserted rather than asserted
about. If the corpus stops building its times from whole hours, this fails and the explanation is rewritten
rather than quietly left describing a corpus that no longer exists.
"""


@pytest.fixture(name="experiment", scope="module")
def experiment_fixture():
    """Import the runner as a module. Nothing runs at import; `main` is guarded."""
    spec = importlib.util.spec_from_file_location("clock_skew_run_experiment", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# --- The machinery ------------------------------------------------------------------------------------------


def test_the_spread_is_exactly_the_magnitude(experiment):
    # The magnitude has to mean one thing for the answer to be quotable: the worst pair of clocks in the estate
    # disagrees by exactly this much. A scheme where it meant the largest single offset would report half.
    for count in (2, 3, 5, 8):
        sources = [f"source-{index}" for index in range(count)]
        offsets = experiment.offsets_for(sources, experiment.SECOND_NS)

        assert set(offsets) == set(sources)
        assert max(offsets.values()) - min(offsets.values()) == experiment.SECOND_NS


def test_one_clock_cannot_disagree_with_itself(experiment):
    assert experiment.offsets_for(["only"], experiment.SECOND_NS) == {"only": 0}
    assert experiment.offsets_for([], experiment.SECOND_NS) == {}


def test_the_assignment_is_a_property_of_the_estate_and_not_of_dictionary_order(experiment):
    forwards = experiment.offsets_for(["a", "b", "c"], experiment.SECOND_NS)
    backwards = experiment.offsets_for(["c", "b", "a"], experiment.SECOND_NS)

    assert forwards == backwards


def test_the_corpus_the_millisecond_finding_is_explained_by_is_the_corpus_that_exists():
    # The explanation written into the guide is arithmetic about this corpus, not a general claim, so the
    # arithmetic is checked here.
    auth = sp.build_corpus()["tc5_auth"]
    hour = 3600 * sp.NS_PER_SECOND
    on_the_mark = int((auth["event_time"].astype("int64") % hour == 0).sum())

    assert len(auth) == 385
    assert on_the_mark == BOUNDARY_ALIGNED_AUTHENTICATIONS


# --- The findings -------------------------------------------------------------------------------------------


def _spoofs(result, threshold_ns: int) -> set:
    bindings = result[result["telemetry_class"] == "tc2_binding"]
    elsewhere = bindings[bindings["bind_end_reason"].isin([CONFLICT, DISPLACED])]
    caught = elsewhere[elsewhere["bind_gap_ns"] <= threshold_ns]

    return {(row["mac_address"], row["port_key"]) for (_, row) in caught.iterrows()}


def _drift(experiment, result) -> set:
    # The rule as the shipped search states it, through the experiment's own reduction. This file used to restate
    # the predicate, and the copy had lost the acceleration ceiling the search gained: it counted a seventh
    # principal-day the search does not return.
    return experiment.rules.layer_5_decisions(result)["R-P-L5-006"]


@pytest.mark.slow
@pytest.mark.cpu_mode
def test_the_spoof_rule_survives_the_collectors_disagreeing_by_a_minute(experiment):
    # Both sightings of the displaced MAC come through one MAC table feed, so that feed's offset moves them
    # together and the interval between them is untouched. The guide expected this rule to absorb an offset
    # directly; against a collector's clock it absorbs nothing, because there is nothing to absorb.
    config = ep.build_pipeline_config()
    corpus = ep.build_corpus()
    threshold = experiment.gap_threshold_ns()

    baseline = _spoofs(ep.run_pipeline(config, corpus), threshold)
    offsets = experiment.offsets_for(experiment.sources_in(corpus, experiment.COLLECTOR_CLOCKS),
                                     60 * experiment.SECOND_NS)
    skewed = _spoofs(ep.run_pipeline(config, experiment.skew_corpus(corpus, offsets, experiment.COLLECTOR_CLOCKS)),
                     threshold)

    assert len(baseline) == 2, "the simultaneous spoof and the cross-switch one"
    assert skewed == baseline


@pytest.mark.slow
@pytest.mark.cpu_mode
def test_the_spoof_rule_fails_when_the_switches_disagree_by_a_minute(experiment):
    # The same magnitude against the clocks the guide was actually talking about. The cross-switch sighting is
    # two seconds after its own, and the rule fires below a sixty-second gap, so spreading the switches across a
    # minute pushes the pair to sixty-two seconds and the spoof stops being a spoof. The simultaneous one is
    # within a single switch and keeps firing, which is what makes this a lost detection rather than a lost run.
    config = ep.build_pipeline_config()
    corpus = ep.build_corpus()
    threshold = experiment.gap_threshold_ns()

    baseline = _spoofs(ep.run_pipeline(config, corpus), threshold)
    offsets = experiment.offsets_for(experiment.sources_in(corpus, experiment.SWITCH_CLOCKS), 60 * experiment.SECOND_NS)
    skewed = _spoofs(ep.run_pipeline(config, experiment.skew_corpus(corpus, offsets, experiment.SWITCH_CLOCKS)),
                     threshold)

    lost = baseline - skewed

    assert len(lost) == 1, f"exactly the cross-switch spoof should be lost, lost {lost}"
    assert next(iter(lost))[0] == tp.MAC_B
    assert skewed - baseline == set(), "nothing new is accused, so this is a miss and not a false positive"


@pytest.mark.slow
@pytest.mark.cpu_mode
def test_a_millisecond_moves_the_drift_rules_inputs_and_not_what_it_accuses(experiment):
    # A millisecond of skew moves every boundary-aligned sign-in into the previous hour, and with it that hour's
    # surprise and the score; under the reference arithmetic that changed three of the six principal-days the
    # rule accused. Under the learned models the rise each principal-day shows still moves and the accusations do
    # not, and a minute moves nothing once the events are off the marks. Both halves are asserted, so a later
    # scorer that makes the rule fragile again is a failure here rather than a paragraph nobody rereads.
    config = sp.build_pipeline_config()
    corpus = sp.build_corpus()
    sources = experiment.sources_in(corpus, experiment.COLLECTOR_CLOCKS)

    millisecond = experiment.offsets_for(sources, experiment.MILLISECOND_NS)
    skewed = sp.run_pipeline(config, experiment.skew_corpus(corpus, millisecond, experiment.COLLECTOR_CLOCKS))
    reference = sp.run_pipeline(config, corpus)
    on_mark = _drift(experiment, skewed)
    baseline = _drift(experiment, reference)

    assert len(baseline) == 6
    assert on_mark == baseline

    before = reference[reference["telemetry_class"] == "tc5_auth"].set_index("event_uid")
    after = skewed[skewed["telemetry_class"] == "tc5_auth"].set_index("event_uid").loc[before.index]

    assert int((before["local_hour"] != after["local_hour"]).sum()) == BOUNDARY_ALIGNED_AUTHENTICATIONS
    rise = (before["drift_rise_sigmas"].astype(float).fillna(-1), after["drift_rise_sigmas"].astype(float).fillna(-1))
    assert (rise[0] != rise[1]).any()

    moved = experiment.shift_corpus(corpus, experiment.COLLECTOR_CLOCKS, experiment.HALF_WINDOW_NS)
    minute = experiment.offsets_for(sources, 60 * experiment.SECOND_NS)
    off_mark = _drift(experiment, sp.run_pipeline(config, moved))
    off_mark_skewed = _drift(
        experiment, sp.run_pipeline(config, experiment.skew_corpus(moved, minute, experiment.COLLECTOR_CLOCKS)))

    assert off_mark == baseline, "moving off the boundary must not change what the rule finds in the first place"
    assert off_mark_skewed == off_mark


@pytest.mark.slow
@pytest.mark.cpu_mode
def test_the_ladder_still_reaches_three_layers_under_a_minute_of_skew(experiment):
    # The chains are the thing most obviously built on time, and the span holds. What moves is which window a
    # chain belongs to, which is why the chain count wobbles while the span does not.
    config = ep.build_pipeline_config()
    corpus = ep.build_corpus()

    for columns in (experiment.COLLECTOR_CLOCKS, experiment.SWITCH_CLOCKS):
        offsets = experiment.offsets_for(experiment.sources_in(corpus, columns), 60 * experiment.SECOND_NS)
        result = ep.run_pipeline(config, experiment.skew_corpus(corpus, offsets, columns))
        reach = experiment.chain_reach(result)

        assert reach["three_layer_chains"] == 15, columns
        assert reach["sign_ins_resolved_to_a_port"] == 15, columns


@pytest.mark.slow
@pytest.mark.cpu_mode
def test_the_ladders_lower_rung_moves_where_its_upper_rung_does_not(experiment):
    # The finding the span alone would have hidden. A principal reaches a port through an 802.1X session that
    # lasts most of the hour, so a minute of skew never takes it outside; an ARP observation reaches a port
    # through a MAC binding that lasts one poll cadence, and a minute is a large fraction of that. Five of the
    # thousand and forty-five observations rooted on a port lose theirs net -- ten fall back to their own address,
    # five acquire a port they did not have -- while every chain keeps its span and every sign-in keeps its desk.
    # Attribution and reach are not the same property and do not fail together. The DMZ poller is a sixth clock
    # in the spread, so a minute's spread places the other five a little differently than it did.
    config = ep.build_pipeline_config()
    corpus = ep.build_corpus()
    offsets = experiment.offsets_for(experiment.sources_in(corpus, experiment.COLLECTOR_CLOCKS),
                                     60 * experiment.SECOND_NS)

    baseline = ep.run_pipeline(config, corpus)
    skewed = ep.run_pipeline(config, experiment.skew_corpus(corpus, offsets, experiment.COLLECTOR_CLOCKS))

    assert experiment.chain_reach(baseline)["arp_observations_rooted_on_a_port"] == 1045
    assert experiment.chain_reach(skewed)["arp_observations_rooted_on_a_port"] == 1040

    left = baseline.set_index(["telemetry_class", "row_key"]).sort_index()
    right = skewed.set_index(["telemetry_class", "row_key"]).sort_index()
    shared = left.index.intersection(right.index)
    left = left.loc[shared]
    right = right.loc[shared]

    moved = left["chain_anchor_source"] != right["chain_anchor_source"]
    became = left[moved].groupby([left[moved]["chain_anchor_source"], right[moved]["chain_anchor_source"]]).size()

    assert became.to_dict() == {
        ("resolved_port_key", "arp_sender_ip"): 10,
        ("arp_sender_ip", "resolved_port_key"): 5,
    }


# --- Every shipped detection, and what the sweeps can and cannot say about it ------------------------------------

DECIDERS = ("layer_1_decisions",
            "layer_2_decisions",
            "layer_3_decisions",
            "layer_4_decisions",
            "layer_5_decisions",
            "layer_6_decisions",
            "application_decisions",
            "saas_decisions",
            "endpoint_decisions",
            "host_decisions",
            "chain_decisions")

SINGLE_CLOCK_CORPORA = ("saas", "endpoint")
"""Corpora whose every record comes through one collector, so a collector-clock sweep perturbs nothing."""

HOST_RESOLVED_CORPORA = {"network": "tc3", "transport": "tc4", "presentation": "tc6"}
"""Corpora whose rows resolve to a host through a lease and an inventory report, and the class their rules read.

Step 12 gave each the DHCP server's leases and the inventory feed's reports, so each corpus has three collectors
where it had one, and a collector sweep moves the bindings against the rows they resolve. Every row a rule reads
still comes through the one collector, so for the rules the sweep is a uniform shift.
"""

CHAIN_EDGES_SECONDS = {"R-C-001": 800, "R-C-002": 1400, "R-C-004": 4000}
"""The narrowest collector spread that makes each chain accuse a control, measured to the second.

Each is arithmetic about this corpus, written down in the experiment's README. The campaign has eleven collectors,
spread by sorted name, so the clock at index `i` sits at `-M/2 + i*M/10`. R-C-001 is the login clock (`dc-01`, at
-M/2) and the process clock (`edr-01`, at -M/5) moving 3M/10 apart, which first admits a control whose process
came 360 seconds before its login (-360 + 3M/10 >= -120 at M = 800); R-C-002 is a beacon 4020 seconds after its
fingerprint, inside the hour once the flow clock (+M/10) and the inspection clock (+2M/5) close by 3M/10
(M = 1400); and R-C-004 is a session ending on `vpn-01` (+M/2) 1200 seconds before a breach on `pcap-01` (+M/5),
overtaken at 3M/10 (M = 4000).
"""

REPLAY_LOST_SECONDS = 7201
"""The narrowest collector spread at which R-C-005 changes: it stops accusing its attacker, and accuses nothing new.

The attacker's first sign-in, on `dc-01` at -M/2, comes an hour after the first time the HQ MAC table (`hq-mac`,
the middle of eleven clocks, at 0) saw that desk, so past M = 7200 it precedes that sighting, resolves to no port,
and the replay has one site rather than two. The control signed in an hour after the lease ended now moves against
the DHCP server's clock (`dhcp-01`, at -2M/5) rather than an unskewed table, so the two close by only M/10 and the
control cannot be admitted below M = 36000, far past the ladder's three hours.
"""


def _shipped_detections() -> set:
    path = os.path.join(REPO_ROOT,
                        "examples",
                        "splunk_lineage_app",
                        "TA-morpheus-lineage",
                        "default",
                        "savedsearches.conf")

    with open(path, encoding="utf-8") as handle:
        return set(re.findall(r'rule_id = "(R-[A-Z]-[\w-]+)"', handle.read()))


def test_every_shipped_detection_is_decided_once(experiment):
    # A detection with no decision function is a rule this experiment silently says nothing about, which is how
    # thirty-one of thirty-eight went unswept while the README's table read as the result.
    import inspect

    decided = []

    for name in DECIDERS:
        source = inspect.getsource(getattr(experiment.rules, name))
        decided.extend(re.findall(r'"(R-[A-Z]-[\w-]+)":', source))

    assert sorted(decided) == sorted(set(decided)), "a rule is decided twice"
    assert set(decided) == _shipped_detections()
    assert len(decided) == 61


def test_a_sweep_over_one_clock_is_not_reported_as_a_tolerance(experiment):
    # The distinction the report exists to keep: a rule that held across the ladder, and a sweep that moved nothing.
    sweeps = {
        "one": {
            "clocks": ["only"], "baseline_decisions": {
                "R-X-001": 1
            }, "runs": []
        },
        "many": {
            "clocks": ["a", "b"], "baseline_decisions": {
                "R-X-001": 1
            }, "runs": []
        },
    }

    assert experiment.breaking_point(sweeps) == {"R-X-001": {"many": None, "one": experiment.NOT_MEASURED}}


@pytest.mark.parametrize("corpus_name", SINGLE_CLOCK_CORPORA)
def test_the_corpora_reported_as_unmeasured_have_one_collector(experiment, corpus_name):
    module = importlib.import_module(f"{corpus_name}_pipeline")

    assert len(experiment.sources_in(module.build_corpus(), experiment.COLLECTOR_CLOCKS)) == 1


@pytest.mark.parametrize("corpus_name", sorted(HOST_RESOLVED_CORPORA))
def test_a_host_resolved_corpus_reads_its_rules_inputs_through_one_collector(experiment, corpus_name):
    corpus = importlib.import_module(f"{corpus_name}_pipeline").build_corpus()

    assert len(experiment.sources_in(corpus, experiment.COLLECTOR_CLOCKS)) == 3
    assert corpus[HOST_RESOLVED_CORPORA[corpus_name]]["collector_id"].nunique() == 1


def test_every_layer_5_sign_in_comes_through_one_collector(experiment):
    # Why the session sweep's "never" for the sign-in rules is a uniform shift rather than a disagreement: the
    # corpus has three collectors, and every authentication comes through one of them.
    corpus = sp.build_corpus()

    assert len(experiment.sources_in(corpus, experiment.COLLECTOR_CLOCKS)) == 3
    assert set(corpus["tc5_auth"]["collector_id"]) == {"idp"}


def test_each_application_class_has_a_collector_of_its_own(experiment):
    import application_pipeline  # pylint: disable=import-outside-toplevel

    corpus = application_pipeline.build_corpus()

    # Three collectors for the three classes the rules read -- the resolver, the proxy and the load balancer in
    # front of the services -- and the DHCP server and inventory feed the hosts resolve through.
    assert len(experiment.sources_in(corpus, experiment.COLLECTOR_CLOCKS)) == 5
    assert all(frame["collector_id"].nunique() == 1 for frame in corpus.values())


def test_the_host_frame_is_fed_by_three_collectors_and_resolved_through_two(experiment):
    # The host frame is the one row in which three collectors' records meet: the flow exporter's, the inspection
    # point's and the EDR's, each the only collector of its class, with the DHCP server and the inventory feed the
    # hosts resolve through. A collector sweep moves the three feeds apart, so R-P-L3-009 is measured against clocks
    # that disagree rather than under a uniform shift.
    import host_score_pipeline as hs_  # pylint: disable=import-outside-toplevel

    corpus = hs_.build_corpus()

    assert len(experiment.sources_in(corpus, experiment.COLLECTOR_CLOCKS)) == 5
    assert all(corpus[name]["collector_id"].nunique() == 1 for name in hs_.FEATURE_CLASSES)
    assert len({corpus[name]["collector_id"].iloc[0] for name in hs_.FEATURE_CLASSES}) == 3


@pytest.mark.slow
@pytest.mark.cpu_mode
def test_the_host_drift_rule_holds_to_a_minute_across_the_feeds(experiment):
    # R-P-L3-009 reads a day's mean of hourly frames, and a minute of spread between the feeds moves at most an
    # hour's edge of records into the neighbouring frame: the three host-days it accuses are the same at a second
    # and at a minute, and it accuses nothing new.
    import host_score_pipeline as hs_  # pylint: disable=import-outside-toplevel

    report = experiment.sweep("host, collector clocks",
                              hs_,
                              experiment.rules.host_decisions,
                              magnitudes=(experiment.SECOND_NS, 60 * experiment.SECOND_NS))
    (second, minute) = report["runs"]

    assert len(report["clocks"]) == 5
    assert report["baseline_decisions"]["R-P-L3-009"] == 3
    assert second["rules_changed"] == [] and minute["rules_changed"] == []


def _chains_at(experiment, corpus, seconds: float) -> dict:
    import campaign_pipeline  # pylint: disable=import-outside-toplevel

    offsets = experiment.offsets_for(experiment.sources_in(corpus, experiment.COLLECTOR_CLOCKS),
                                     int(seconds * experiment.SECOND_NS))
    skewed = experiment.skew_corpus(corpus, offsets, experiment.COLLECTOR_CLOCKS)

    return experiment.rules.chain_decisions(
        campaign_pipeline.run_pipeline(campaign_pipeline.build_pipeline_config(), skewed))


@pytest.mark.slow
@pytest.mark.cpu_mode
def test_every_chain_holds_at_twice_its_join_tolerance(experiment):
    # The tolerance is a statement about two clocks; at twice it, the worst pair of collectors disagrees by more than
    # any chain allows one step to precede another, and still nothing changes, because the corpus never puts two
    # steps that close. What this measures is the corpus's slack, and it is written down as that.
    import campaign_pipeline  # pylint: disable=import-outside-toplevel

    corpus = campaign_pipeline.build_corpus()
    baseline = experiment.rules.chain_decisions(
        campaign_pipeline.run_pipeline(campaign_pipeline.build_pipeline_config(), corpus))
    widest = 2 * max(experiment.rules.JOIN_TOLERANCE_NS.values()) / experiment.SECOND_NS

    assert widest == 240
    assert _chains_at(experiment, corpus, widest) == baseline
    assert all(len(keys) == 1 for keys in baseline.values()), "each chain accuses its attacker alone"


@pytest.mark.slow
@pytest.mark.cpu_mode
@pytest.mark.parametrize("rule", sorted(CHAIN_EDGES_SECONDS))
def test_each_chain_first_breaks_by_accusing_a_control(experiment, rule):
    # The way each chain fails is the finding: a second short of its edge nothing has changed, and at the edge the
    # chain adds a control the corpus planted for it to stay quiet on, while still accusing its attacker.
    import campaign_pipeline  # pylint: disable=import-outside-toplevel

    corpus = campaign_pipeline.build_corpus()
    baseline = experiment.rules.chain_decisions(
        campaign_pipeline.run_pipeline(campaign_pipeline.build_pipeline_config(), corpus))[rule]
    edge = CHAIN_EDGES_SECONDS[rule]

    assert _chains_at(experiment, corpus, edge - 1)[rule] == baseline
    broken = _chains_at(experiment, corpus, edge)[rule]

    assert baseline < broken and len(broken - baseline) == 1


@pytest.mark.slow
@pytest.mark.cpu_mode
def test_credential_replay_first_breaks_by_losing_its_attacker(experiment):
    # The one chain that fails the other way. When step 12 put the desk leases on the DHCP server's clock, the control
    # this rule once admitted stopped moving against a table no sweep touched, and the first thing a spread does to
    # the rule is take its attacker's first site away.
    import campaign_pipeline  # pylint: disable=import-outside-toplevel

    corpus = campaign_pipeline.build_corpus()
    baseline = experiment.rules.chain_decisions(
        campaign_pipeline.run_pipeline(campaign_pipeline.build_pipeline_config(), corpus))["R-C-005"]

    assert _chains_at(experiment, corpus, REPLAY_LOST_SECONDS - 1)["R-C-005"] == baseline
    assert _chains_at(experiment, corpus, REPLAY_LOST_SECONDS)["R-C-005"] == set()
    assert baseline == {(campaign_pipeline.QUIN, )}


@pytest.mark.slow
@pytest.mark.cpu_mode
def test_the_physical_chain_holds_at_the_widest_spread_swept(experiment):
    # R-C-003's port event and its new vendor come through one poller, the one at the middle of the campaign's
    # eleven clocks, so no collector spread pulls them apart; the flow that completes it has four hours to land and
    # the lease it resolves through moves the other way. Three hours of spread, the most the ladder reaches, leaves it
    # accusing its attacker and nothing else.
    import campaign_pipeline  # pylint: disable=import-outside-toplevel

    corpus = campaign_pipeline.build_corpus()
    widest = max(experiment.CHAIN_MAGNITUDES_NS) / experiment.SECOND_NS

    assert widest == 10800
    assert _chains_at(experiment, corpus, widest)["R-C-003"] == {("hq:hq-sw1:Gi1/0/41", "b8:27:eb:3c:00:41")}
    assert campaign_pipeline.PORT_POLLER == "hq-mac", "the port polls share the MAC table's poller and its clock"


# --- The corrected curve and the exporter axis --------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.cpu_mode
def test_quarantine_holds_nothing_inside_its_bound_and_only_takes_away_beyond_it(experiment):
    # With the collectors reporting the offsets they were given, a spread of a second leaves every clock within
    # half a second and EnvelopeValidateStage holds only the DMZ poller it always held. At ten seconds the outer
    # collectors are three and five seconds out and their records are held whole, so the rules that read them lose
    # what they accused -- and never accuse anything new, which is the trade quarantine makes against skew.
    def estate(result):
        return {
            **experiment.rules.layer_1_decisions(result),
            **experiment.rules.layer_2_decisions(result, tp.SINGLE_HOST_PORTS)
        }

    report = experiment.sweep("estate, quarantine on",
                              ep,
                              estate,
                              magnitudes=(experiment.SECOND_NS, 10 * experiment.SECOND_NS),
                              report_offsets=True)
    (inside, beyond) = report["runs"]

    assert report["baseline_quarantined"] == inside["quarantined"] == len(tp.QUARANTINED_MINUTES)
    assert inside["rules_changed"] == []

    assert beyond["quarantined"] > inside["quarantined"]
    assert len(beyond["rules_changed"]) > 0
    assert all(entry["newly_flagged"] == [] for entry in beyond["rules"].values())


@pytest.mark.slow
@pytest.mark.cpu_mode
def test_a_routers_clock_moves_only_the_fan_out_trajectory(experiment):
    # The exporter axis: each router stamps its own flows, and the collector's reported offset says nothing about
    # it. The eight layer 3 rules were unmeasured while the corpus named no exporter; on three router clocks seven
    # hold to a minute, and R-P-L3-005 -- a rise across three five-minute bins, the narrowest window any layer 3 rule
    # reads -- loses one of the scanner's twenty at thirty seconds and gains nothing.
    import network_pipeline as np_  # pylint: disable=import-outside-toplevel

    report = experiment.sweep("network, exporter clocks",
                              np_,
                              experiment.rules.layer_3_decisions,
                              columns="exporter_id",
                              magnitudes=(10 * experiment.SECOND_NS, 30 * experiment.SECOND_NS))
    (held, moved) = report["runs"]

    assert len(report["clocks"]) == 3
    assert held["rules_changed"] == []
    assert moved["rules_changed"] == ["R-P-L3-005"]
    assert moved["rules"]["R-P-L3-005"]["newly_flagged"] == []
    assert len(moved["rules"]["R-P-L3-005"]["no_longer_flagged"]) == 1
