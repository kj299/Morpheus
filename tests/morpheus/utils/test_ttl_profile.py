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

from morpheus.utils.ttl_profile import NS_PER_SECOND
from morpheus.utils.ttl_profile import TtlProfileTracker
from morpheus.utils.ttl_profile import established_ttl

SOURCE = "10.0.0.11"
SECOND = NS_PER_SECOND
START = 10**18


def run(tracker: TtlProfileTracker, ttls, key: str = SOURCE, start: int = START):
    """Feed a series of packets a second apart and return the last result."""
    result = None

    for (index, ttl) in enumerate(ttls):
        result = tracker.observe(key, start + index * SECOND, ttl)

    return result


def test_a_steady_source_has_a_reference_and_no_shift():
    result = run(TtlProfileTracker(), [64] * 10)

    assert result.established == 64
    assert result.shift == 0
    assert result.shifted is False
    assert result.mature is True
    assert result.distinct == 1


def test_an_interposed_device_shows_as_one_hop_lost():
    # The case the rule exists for. Something that forwards a packet decrements the field exactly once, so the
    # signal is a shift of one -- which is why `min_shift` is one and the comparison is inclusive.
    tracker = TtlProfileTracker()
    run(tracker, [64] * 10)

    tapped = tracker.observe(SOURCE, START + 10 * SECOND, 63)

    assert tapped.established == 64
    assert tapped.shift == -1
    assert tapped.shifted is True


def test_requiring_more_than_one_hop_would_miss_the_interposition():
    # The guide's wording for R-B-L3-004 says "more than one hop-equivalent", and read strictly that excludes
    # the single forwarding device it names as the cause. Asserted rather than argued: at min_shift=2 the tapped
    # packet above goes unflagged.
    tracker = TtlProfileTracker(min_shift=2)
    run(tracker, [64] * 10)

    tapped = tracker.observe(SOURCE, START + 10 * SECOND, 63)

    assert tapped.shift == -1
    assert tapped.shifted is False


def test_a_spoofed_initial_value_shows_as_a_large_shift():
    # The other reading of the same arithmetic: a forger has to guess what the real host sends, and guessing 128
    # where the host sends 64 is not subtle.
    tracker = TtlProfileTracker()
    run(tracker, [64] * 10)

    forged = tracker.observe(SOURCE, START + 10 * SECOND, 128)

    assert forged.shift == 64
    assert forged.shifted is True


def test_the_reference_is_the_mode_rather_than_the_mean():
    # A source that is really two hosts behind one address is bimodal, and a mean lands between the two and
    # describes neither. The mode describes the commonest path and `distinct` says there is more than one.
    result = run(TtlProfileTracker(), [64, 64, 64, 64, 64, 64, 128, 128, 64])

    assert result.established == 64
    assert result.distinct == 2
    assert result.shift == 0


def test_a_tie_breaks_towards_the_larger_value():
    # Where a device has been interposed for exactly half the window, 63 and 64 have equal support. Calling 63
    # the reference would report the device's absence as the anomaly, which is backwards.
    assert established_ttl([64, 63]) == 64
    assert established_ttl([63, 64, 63, 64]) == 64


def test_a_packet_is_not_folded_into_its_own_reference():
    # The step this exists to expose is damped by a sample that has already pulled the reference towards itself,
    # which is the rule `optical_baseline` follows for the same reason.
    tracker = TtlProfileTracker(min_samples=2)
    run(tracker, [64, 64])

    moved = tracker.observe(SOURCE, START + 2 * SECOND, 63)

    assert moved.established == 64
    assert moved.shift == -1


def test_no_reference_is_published_before_there_is_one():
    # `min_samples` counts packets *prior* to the one being scored, so the reference appears on the packet after
    # the fifth rather than on it. Pinned because the off-by-one is invisible until a corpus is one packet short
    # of ever producing a shift.
    tracker = TtlProfileTracker(min_samples=5)
    fifth = run(tracker, [64] * 5)

    assert fifth.samples == 5
    assert fifth.established is None
    assert fifth.shift is None
    assert fifth.mature is False
    assert fifth.shifted is False

    sixth = tracker.observe(SOURCE, START + 5 * SECOND, 63)

    assert sixth.mature is True
    assert sixth.established == 64
    assert sixth.shift == -1


def test_a_re_route_stops_being_an_anomaly_once_it_is_what_the_source_does():
    # The reason the window is trailing. A genuine path change is an anomaly for as long as it is new and a fact
    # afterwards, and a reference that never moved would alarm on it forever.
    tracker = TtlProfileTracker(min_samples=3)
    run(tracker, [64] * 5)

    first = tracker.observe(SOURCE, START + 5 * SECOND, 63)

    assert first.shifted is True

    settled = run(tracker, [63] * 12, start=START + 6 * SECOND)

    assert settled.established == 63
    assert settled.shift == 0
    assert settled.shifted is False


def test_an_old_packet_falls_out_of_the_window():
    tracker = TtlProfileTracker(window_ns=10 * SECOND, min_samples=1)
    run(tracker, [128] * 5)

    later = tracker.observe(SOURCE, START + 3600 * SECOND, 64)

    assert later.samples == 1
    assert later.established is None
    assert later.mature is False


def test_an_out_of_order_packet_leaves_the_reference_alone():
    tracker = TtlProfileTracker()
    ordered = run(tracker, [64] * 10)

    backwards = tracker.observe(SOURCE, START - SECOND, 32)

    assert backwards.out_of_order is True
    assert backwards.shift is None
    assert backwards.established == ordered.established
    assert backwards.samples == ordered.samples


def test_two_sources_keep_separate_references():
    tracker = TtlProfileTracker()
    run(tracker, [64] * 10, key="10.0.0.11")
    run(tracker, [128] * 10, key="10.0.0.12")

    assert tracker.observe("10.0.0.11", START + 20 * SECOND, 64).shift == 0
    assert tracker.observe("10.0.0.12", START + 20 * SECOND, 64).shift == -64
    assert tracker.tracked_entities == 2


def test_the_sample_cap_narrows_the_claim_and_says_so():
    tracker = TtlProfileTracker(max_samples=4)
    result = run(tracker, [64] * 20)

    assert result.saturated is True
    assert result.samples == 4


def test_the_least_recently_seen_source_is_dropped():
    tracker = TtlProfileTracker(max_entities=2)

    for (index, key) in enumerate(("10.0.0.1", "10.0.0.2", "10.0.0.3")):
        tracker.observe(key, START + index, 64)

    assert tracker.tracked_entities == 2


def test_an_empty_series_has_no_reference():
    assert established_ttl([]) is None


@pytest.mark.parametrize("ttl", [-1, 256, 64.0, True, None])
def test_a_value_outside_the_field_is_refused(ttl):
    # A TTL of 300 did not come off a wire, and profiling it would put a parsing fault in the reference every
    # other packet is then compared against.
    with pytest.raises(ValueError, match="ttl"):
        TtlProfileTracker().observe(SOURCE, START, ttl)


@pytest.mark.parametrize(("kwargs", "match"),
                         [({
                             "window_ns": 0
                         }, "window_ns"), ({
                             "min_samples": 0
                         }, "min_samples"), ({
                             "min_shift": 0
                         }, "min_shift"), ({
                             "max_samples": 0
                         }, "max_samples"), ({
                             "max_entities": 0
                         }, "max_entities")])
def test_the_bounds_are_refused_rather_than_clamped(kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        TtlProfileTracker(**kwargs)


def test_an_entity_key_is_required():
    with pytest.raises(ValueError, match="entity_key"):
        TtlProfileTracker().observe("", START, 64)


def test_a_float_event_time_is_refused():
    with pytest.raises(ValueError, match="event_time_ns"):
        TtlProfileTracker().observe(SOURCE, 1.0e18, 64)
