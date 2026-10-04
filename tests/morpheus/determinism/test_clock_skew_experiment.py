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
- **R-P-L5-006 looks fragile at a millisecond and is not.** Forty-five of the layer 5 corpus's hundred and five
  authentications sit exactly on an hour mark, and an event on a boundary changes window under an offset of one
  nanosecond. Move the same events into the middle of their windows and a minute of skew changes nothing. The
  sensitivity is to boundary proximity, not to magnitude.
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

BOUNDARY_ALIGNED_AUTHENTICATIONS = 45
"""Layer 5 authentications sitting exactly on an hour mark, out of one hundred and five.

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

    assert len(auth) == 105
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
def test_the_drift_rule_changes_on_the_hour_marks_and_not_off_them(experiment):
    # A millisecond of skew changes what this rule accuses, and a minute of it does not, once the same events
    # are moved into the middle of their windows. Both halves are the finding: reporting only the first would
    # say the rule needs millisecond synchronization, which is not what was measured.
    config = sp.build_pipeline_config()
    corpus = sp.build_corpus()
    sources = experiment.sources_in(corpus, experiment.COLLECTOR_CLOCKS)

    millisecond = experiment.offsets_for(sources, experiment.MILLISECOND_NS)
    on_mark = _drift(experiment,
                     sp.run_pipeline(config, experiment.skew_corpus(corpus, millisecond, experiment.COLLECTOR_CLOCKS)))
    baseline = _drift(experiment, sp.run_pipeline(config, corpus))

    assert len(baseline) == 6
    assert len(baseline - on_mark) == 3 and on_mark < baseline, "three of the six principal-days are lost"

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
    # through a MAC binding that lasts one poll cadence, and a minute is a large fraction of that. Nine of the
    # thousand and forty observations rooted on a port lose or change theirs -- fifteen fall back to their own
    # address, six acquire a port they did not have -- while every chain keeps its span and every sign-in keeps
    # its desk. Attribution and reach are not the same property and do not fail together.
    config = ep.build_pipeline_config()
    corpus = ep.build_corpus()
    offsets = experiment.offsets_for(experiment.sources_in(corpus, experiment.COLLECTOR_CLOCKS),
                                     60 * experiment.SECOND_NS)

    baseline = ep.run_pipeline(config, corpus)
    skewed = ep.run_pipeline(config, experiment.skew_corpus(corpus, offsets, experiment.COLLECTOR_CLOCKS))

    assert experiment.chain_reach(baseline)["arp_observations_rooted_on_a_port"] == 1040
    assert experiment.chain_reach(skewed)["arp_observations_rooted_on_a_port"] == 1031

    left = baseline.set_index(["telemetry_class", "row_key"]).sort_index()
    right = skewed.set_index(["telemetry_class", "row_key"]).sort_index()
    shared = left.index.intersection(right.index)
    left = left.loc[shared]
    right = right.loc[shared]

    moved = left["chain_anchor_source"] != right["chain_anchor_source"]
    became = left[moved].groupby([left[moved]["chain_anchor_source"], right[moved]["chain_anchor_source"]]).size()

    assert became.to_dict() == {
        ("resolved_port_key", "arp_sender_ip"): 15,
        ("arp_sender_ip", "resolved_port_key"): 6,
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
            "chain_decisions")

SINGLE_CLOCK_CORPORA = ("network", "transport", "presentation", "saas", "endpoint")
"""Corpora whose every record comes through one collector, so a collector-clock sweep perturbs nothing."""

CHAIN_EDGES_SECONDS = {"R-C-001": 960, "R-C-002": 1120, "R-C-004": 3200, "R-C-005": 7201}
"""The narrowest collector spread that changes each chain's accusations, measured to the second.

Each is arithmetic about this corpus, written down in the experiment's README: R-C-001 is the login clock moving
M/2 toward the fan-out and the process clock M/4 away, which first admits a control whose process came 360 seconds
before its login (-360 + M/4 >= -120 at M = 960); R-C-002 is a beacon 4020 seconds after its fingerprint, inside
the hour once the two clocks close by 3M/8 (M = 1120); R-C-004 is a session ending 1200 seconds before a breach,
overtaken at 3M/8 (M = 3200); and R-C-005 is a sign-in an hour after its lease ended, back inside it once its clock
moves by more than M/2 against the unskewed lease table (M > 7200).
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
    assert len(decided) == 38


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


def test_every_layer_5_sign_in_comes_through_one_collector(experiment):
    # Why the session sweep's "never" for the sign-in rules is a uniform shift rather than a disagreement: the
    # corpus has three collectors, and every authentication comes through one of them.
    corpus = sp.build_corpus()

    assert len(experiment.sources_in(corpus, experiment.COLLECTOR_CLOCKS)) == 3
    assert set(corpus["tc5_auth"]["collector_id"]) == {"idp"}


def test_each_application_class_has_a_collector_of_its_own(experiment):
    import application_pipeline  # pylint: disable=import-outside-toplevel

    corpus = application_pipeline.build_corpus()

    assert len(experiment.sources_in(corpus, experiment.COLLECTOR_CLOCKS)) == 2
    assert all(frame["collector_id"].nunique() == 1 for frame in corpus.values())


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
