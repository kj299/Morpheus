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

import pytest

from morpheus.utils.established_value import NS_PER_SECOND
from morpheus.utils.established_value import ValueHistoryTracker
from morpheus.utils.established_value import minimum_of
from morpheus.utils.established_value import mode_of

ISSUER = "CN=Corp Issuing CA, O=Example"
OTHER = "CN=Interception Proxy, O=Unknown"


def feed(tracker: ValueHistoryTracker, entity: str, values: list, start: int = 0, step: int = NS_PER_SECOND):
    """Observe a series and return every result, so a test can look at the one it cares about."""
    return [tracker.observe(entity, start + index * step, value) for (index, value) in enumerate(values)]


# --- The two reductions ----------------------------------------------------------------------------------------


def test_the_mode_is_the_commonest_value():
    assert mode_of([ISSUER, ISSUER, OTHER]) == ISSUER


def test_a_tie_breaks_towards_the_larger_value():
    # Not a preference for the larger string. `Counter.most_common` orders ties by insertion, so a tie would
    # otherwise resolve to whichever arrived first -- a reference that depends on arrival order, which is the
    # property control 8 exists to remove.
    assert mode_of(["a", "b"]) == "b"
    assert mode_of(["b", "a"]) == "b"


def test_the_minimum_is_the_weakest_ever_accepted():
    assert minimum_of([5, 4, 5, 3, 5]) == 3


def test_an_empty_series_reduces_to_nothing():
    assert mode_of([]) is None
    assert minimum_of([]) is None


# --- The reference ---------------------------------------------------------------------------------------------


def test_a_destination_with_a_settled_issuer_has_a_reference():
    tracker = ValueHistoryTracker(min_samples=3)
    results = feed(tracker, "10.0.1.5", [ISSUER] * 5)

    assert results[-1].reference == ISSUER
    assert results[-1].mature
    assert results[-1].distinct == 1


def test_an_interception_differs_from_the_destination_that_never_presented_it():
    tracker = ValueHistoryTracker(min_samples=3)
    results = feed(tracker, "10.0.1.5", [ISSUER] * 5 + [OTHER])

    assert results[-1].value == OTHER
    assert results[-1].reference == ISSUER
    assert results[-1].value != results[-1].reference


def test_a_destination_already_presenting_several_issuers_says_so():
    # Why `distinct` is on the row. A destination behind four certificate authorities produces a difference that
    # means far less than the same difference from one that has only ever presented one, and a search that could
    # not tell them apart would tune itself into silence on both.
    tracker = ValueHistoryTracker(min_samples=3)
    results = feed(tracker, "10.0.1.6", ["a", "b", "c", "d", "a"])

    assert results[-1].distinct == 4


def test_an_observation_is_not_folded_into_its_own_reference():
    # The first sight of a new issuer must be compared against the history it is not yet part of. Fold it in and
    # a destination that has presented one issuer five times reports a reference of that issuer either way, and
    # the check passes while meaning nothing -- so the case that separates them is a history of exactly two.
    tracker = ValueHistoryTracker(min_samples=2)
    results = feed(tracker, "10.0.1.7", [OTHER, OTHER, OTHER])

    assert results[2].reference == OTHER
    assert results[2].samples == 2


def test_no_reference_is_published_before_the_history_supports_one():
    # `min_samples` counts prior observations, so the reference appears on the observation after the floor is
    # reached rather than on it.
    tracker = ValueHistoryTracker(min_samples=3)
    results = feed(tracker, "10.0.1.8", [ISSUER] * 4)

    assert [result.mature for result in results] == [False, False, False, True]
    assert results[3].samples == 3


def test_a_mode_over_two_observations_is_the_first_value_seen_twice():
    # Why the floor exists at all, demonstrated rather than asserted in prose. With a floor of one, a
    # destination's second observation is compared against its first, so a rotation to a new certificate
    # authority reads as interception on the day it happens.
    reckless = ValueHistoryTracker(min_samples=1)
    results = feed(reckless, "10.0.1.9", [ISSUER, OTHER])

    assert results[1].reference == ISSUER
    assert results[1].value != results[1].reference


# --- The downgrade reading -------------------------------------------------------------------------------------


def test_a_pair_that_has_always_negotiated_well_has_a_high_floor():
    tracker = ValueHistoryTracker(reduction=minimum_of, min_samples=3)
    results = feed(tracker, "10.0.0.5=10.0.1.5", [5, 4, 5, 5, 4])

    assert results[-1].reference == 4


def test_a_downgrade_lands_below_the_pair_s_own_floor():
    tracker = ValueHistoryTracker(reduction=minimum_of, min_samples=3)
    results = feed(tracker, "10.0.0.5=10.0.1.5", [4, 4, 4, 4, 1])

    assert results[-1].value < results[-1].reference


def test_a_mode_would_have_called_an_ordinary_negotiation_a_downgrade():
    # Why the downgrade rule reduces with a minimum and the issuer rule with a mode. A pair that settles on a
    # strong suite nine times in ten has a mode of that suite, and the tenth, weaker, entirely routine
    # negotiation sits below it. Only the pair's own floor distinguishes routine from attack.
    history = [5] * 9 + [4]

    assert mode_of(history) == 5
    assert minimum_of(history) == 4


# --- Windowing, ordering and bounds ----------------------------------------------------------------------------


def test_an_old_observation_falls_out_of_the_window():
    tracker = ValueHistoryTracker(window_ns=10 * NS_PER_SECOND, min_samples=1)
    feed(tracker, "10.0.1.5", [OTHER] * 3)
    late = tracker.observe("10.0.1.5", 1000 * NS_PER_SECOND, ISSUER)

    assert late.samples == 0
    assert late.reference is None


def test_a_rotation_stops_being_a_difference_once_it_is_what_the_destination_does():
    # A certificate authority migration is a genuine change and then a fact. The reference has to follow it, or
    # the rule reports the estate's own new normal every day until somebody silences it.
    tracker = ValueHistoryTracker(min_samples=3)
    feed(tracker, "10.0.1.5", [ISSUER] * 5)
    later = feed(tracker, "10.0.1.5", [OTHER] * 8, start=100 * NS_PER_SECOND)

    assert later[0].reference == ISSUER
    assert later[-1].reference == OTHER


def test_an_out_of_order_observation_leaves_the_history_alone():
    tracker = ValueHistoryTracker(min_samples=2)
    feed(tracker, "10.0.1.5", [ISSUER] * 4)

    stale = tracker.observe("10.0.1.5", 0, OTHER)
    after = tracker.observe("10.0.1.5", 100 * NS_PER_SECOND, ISSUER)

    assert stale.out_of_order
    assert after.samples == 4
    assert after.reference == ISSUER


def test_observations_at_one_instant_share_a_reference_whatever_their_order():

    def tied(order):
        tracker = ValueHistoryTracker(min_samples=2)
        feed(tracker, "10.0.1.5", [ISSUER] * 3)

        return {value: tracker.observe("10.0.1.5", 3 * NS_PER_SECOND, value) for value in order}

    forwards = tied([OTHER, OTHER, ISSUER])
    backwards = tied([ISSUER, OTHER, OTHER])

    for results in (forwards, backwards):
        for result in results.values():
            assert not result.out_of_order
            assert result.reference == ISSUER
            assert result.samples == 3


def test_an_instant_joins_the_history_once_it_has_passed():
    tracker = ValueHistoryTracker(min_samples=2)
    feed(tracker, "10.0.1.5", [ISSUER] * 2)

    for _ in range(3):
        tracker.observe("10.0.1.5", 5 * NS_PER_SECOND, OTHER)

    after = tracker.observe("10.0.1.5", 6 * NS_PER_SECOND, ISSUER)

    assert after.samples == 5
    assert after.reference == OTHER


def test_two_entities_keep_separate_references():
    tracker = ValueHistoryTracker(min_samples=2)
    feed(tracker, "10.0.1.5", [ISSUER] * 3)
    feed(tracker, "10.0.1.6", [OTHER] * 3)

    assert tracker.observe("10.0.1.5", 10**12, ISSUER).reference == ISSUER
    assert tracker.observe("10.0.1.6", 10**12, ISSUER).reference == OTHER


def test_the_sample_cap_narrows_the_claim_and_says_so():
    tracker = ValueHistoryTracker(min_samples=1, max_samples=3)
    results = feed(tracker, "10.0.1.5", [ISSUER] * 6)

    assert results[-1].saturated
    assert results[-1].samples == 3


def test_the_least_recently_seen_entity_is_dropped():
    tracker = ValueHistoryTracker(min_samples=1, max_entities=2)
    feed(tracker, "a", [ISSUER] * 2)
    feed(tracker, "b", [ISSUER] * 2, start=10 * NS_PER_SECOND)
    feed(tracker, "c", [ISSUER] * 2, start=20 * NS_PER_SECOND)

    assert tracker.tracked_entities == 2
    assert tracker.observe("a", 10**12, ISSUER).reference is None


def test_observing_an_entity_again_keeps_it_from_being_evicted():
    tracker = ValueHistoryTracker(min_samples=1, max_entities=2)
    feed(tracker, "a", [ISSUER] * 2)
    feed(tracker, "b", [ISSUER] * 2, start=10 * NS_PER_SECOND)
    tracker.observe("a", 20 * NS_PER_SECOND, ISSUER)
    feed(tracker, "c", [ISSUER] * 2, start=30 * NS_PER_SECOND)

    assert tracker.observe("a", 10**12, ISSUER).reference == ISSUER
    assert tracker.observe("b", 10**12, ISSUER).reference is None


@pytest.mark.parametrize(("kwargs", "match"),
                         [({
                             "window_ns": 0
                         }, "window_ns"), ({
                             "min_samples": 0
                         }, "min_samples"), ({
                             "min_samples": 5, "max_samples": 2
                         }, "max_samples"), ({
                             "max_entities": 0
                         }, "max_entities")])
def test_the_bounds_are_refused_rather_than_clamped(kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        ValueHistoryTracker(**kwargs)


def test_an_entity_key_is_required():
    with pytest.raises(ValueError, match="entity_key"):
        ValueHistoryTracker().observe("", 0, ISSUER)


def test_a_float_event_time_is_refused():
    with pytest.raises(ValueError, match="event_time_ns"):
        ValueHistoryTracker().observe("10.0.1.5", 1.5, ISSUER)


def test_a_boolean_is_not_an_event_time():
    with pytest.raises(ValueError, match="event_time_ns"):
        ValueHistoryTracker().observe("10.0.1.5", True, ISSUER)


def test_replaying_a_stream_reproduces_it():
    series = [ISSUER, ISSUER, OTHER, ISSUER, OTHER, OTHER, ISSUER]

    first = [result.reference for result in feed(ValueHistoryTracker(min_samples=2), "10.0.1.5", series)]
    second = [result.reference for result in feed(ValueHistoryTracker(min_samples=2), "10.0.1.5", series)]

    assert first == second
