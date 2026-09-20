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

import random
import statistics

import pytest

from morpheus.utils.arrival_regularity import NS_PER_SECOND
from morpheus.utils.arrival_regularity import ArrivalRegularityTracker
from morpheus.utils.arrival_regularity import coefficient_of_variation

PAIR = "10.0.0.11:198.51.100.7"
BEACON_PERIOD_NS = 60 * NS_PER_SECOND
BEACON_THRESHOLD = 0.15
"""The figure R-B-L3-002 names, repeated here so the tests evaluate the rule's predicate rather than one near it."""


def run(tracker: ArrivalRegularityTracker, times, sizes=None, key: str = PAIR):
    """Feed a series of arrivals and return the last result."""
    result = None

    for (index, event_time_ns) in enumerate(times):
        result = tracker.observe(key, event_time_ns, None if sizes is None else sizes[index])

    return result


def beacon(count: int, jitter_ns: int = 0, seed: int = 7, size: int = 512):
    """A timer with optional jitter, and the fixed-size payload a beacon carries."""
    rng = random.Random(seed)
    times = []
    sizes = []
    now = 10**18

    for _ in range(count):
        times.append(now + (rng.randint(-jitter_ns, jitter_ns) if jitter_ns else 0))
        sizes.append(size)
        now += BEACON_PERIOD_NS

    return (times, sizes)


def test_a_metronome_is_perfectly_regular():
    (times, sizes) = beacon(13)
    result = run(ArrivalRegularityTracker(), times, sizes)

    assert result.intervals == 12
    assert result.mature is True
    assert result.interval_cv == pytest.approx(0.0)
    assert result.size_cv == pytest.approx(0.0)
    assert result.mean_interval_ns == BEACON_PERIOD_NS


def test_a_jittered_beacon_still_reads_as_one():
    # The reason the rule is a ratio rather than a deviation: a real implant jitters, deliberately, and the
    # threshold has to survive that. A second of jitter on a minute's period is well inside 0.15.
    (times, sizes) = beacon(30, jitter_ns=NS_PER_SECOND)
    result = run(ArrivalRegularityTracker(), times, sizes)

    assert result.mature is True
    assert result.interval_cv < BEACON_THRESHOLD


def test_a_person_does_not_read_as_a_beacon():
    # The negative control. Somebody opening files through the working day produces gaps that are ragged at the
    # scale of the gaps themselves, which is exactly what the coefficient measures.
    rng = random.Random(11)
    times = []
    now = 10**18

    for _ in range(30):
        now += rng.randint(5 * NS_PER_SECOND, 900 * NS_PER_SECOND)
        times.append(now)

    result = run(ArrivalRegularityTracker(), times)

    assert result.mature is True
    assert result.interval_cv > BEACON_THRESHOLD


def test_the_coefficient_is_scale_free():
    # A beacon every minute and one every hour are equally regular, and a detector that noticed the difference
    # would be thresholding the period rather than the regularity.
    fast = run(ArrivalRegularityTracker(), [10**18 + index * 60 * NS_PER_SECOND for index in range(13)])
    slow = run(ArrivalRegularityTracker(), [10**18 + index * 3600 * NS_PER_SECOND for index in range(13)])

    assert fast.interval_cv == pytest.approx(slow.interval_cv)
    assert fast.mean_interval_ns != slow.mean_interval_ns


def test_twelve_intervals_are_thirteen_arrivals():
    # The off-by-one the rule's own wording invites. Twelve arrivals are eleven intervals and must not publish.
    tracker = ArrivalRegularityTracker()
    (times, _) = beacon(12)
    twelfth = run(tracker, times)

    assert twelfth.arrivals == 12
    assert twelfth.intervals == 11
    assert twelfth.mature is False
    assert twelfth.interval_cv is None

    thirteenth = tracker.observe(PAIR, times[-1] + BEACON_PERIOD_NS)

    assert thirteenth.intervals == 12
    assert thirteenth.mature is True
    assert thirteenth.interval_cv is not None


def test_arrivals_at_one_instant_have_no_coefficient():
    # The undefined case, and the direction it must not fail in. A mean gap of zero is not a gap of zero
    # variation: returning 0.0 would make a collector that batches its output the most beacon-like thing in the
    # estate, which is how a detector ends up firing on its own plumbing.
    tracker = ArrivalRegularityTracker()
    result = None

    for index in range(20):
        # Strictly increasing by a nanosecond so the arrivals are accepted, then collapsed by the mean being
        # dominated by nothing -- the genuinely simultaneous case is refused as out of order, below.
        result = tracker.observe(PAIR, 10**18 + index)

    assert result.mature is True
    assert result.interval_cv == pytest.approx(0.0)

    assert coefficient_of_variation([0.0, 0.0, 0.0]) is None


def test_a_repeated_timestamp_is_refused_rather_than_averaged():
    tracker = ArrivalRegularityTracker()
    (times, _) = beacon(13)
    run(tracker, times)

    repeated = tracker.observe(PAIR, times[-1])

    assert repeated.out_of_order is True
    assert repeated.arrivals == 13


def test_an_out_of_order_arrival_leaves_the_window_alone():
    # A negative interval would pull the mean down and the deviation up in ways that depend on batching, which is
    # determinism control 8's whole concern.
    tracker = ArrivalRegularityTracker()
    (times, _) = beacon(13)
    ordered = run(tracker, times)

    backwards = tracker.observe(PAIR, times[0] - BEACON_PERIOD_NS)

    assert backwards.out_of_order is True
    assert backwards.interval_cv == pytest.approx(ordered.interval_cv)
    assert backwards.arrivals == ordered.arrivals


def test_a_pause_longer_than_the_window_leaves_less_to_measure():
    # Not one enormous interval, which would read as irregular and say the opposite of what happened, and not a
    # retained history either, where one ancient arrival would hold the mean up forever.
    tracker = ArrivalRegularityTracker(window_ns=3600 * NS_PER_SECOND)
    (times, _) = beacon(13)
    run(tracker, times)

    after = tracker.observe(PAIR, times[-1] + 10 * 3600 * NS_PER_SECOND)

    assert after.arrivals == 1
    assert after.intervals == 0
    assert after.mature is False
    assert after.mean_interval_ns is None


def test_two_pairs_do_not_share_a_rhythm():
    # Keying on the source alone would mix a beacon in with everything else that host does, which is the reason
    # R-B-L3-002 keys on the directed pair.
    tracker = ArrivalRegularityTracker()
    (times, _) = beacon(13)

    for event_time_ns in times:
        tracker.observe("10.0.0.11:198.51.100.7", event_time_ns)

    other = tracker.observe("10.0.0.11:203.0.113.9", times[-1])

    assert other.arrivals == 1
    assert tracker.tracked_entities == 2


def test_size_regularity_is_reported_beside_the_timing():
    # A pair regular in time and wild in size is a poll; regular in both is a poll carrying a fixed message. The
    # rule wants both halves and an operator needs to see which one fired.
    (times, _) = beacon(13)
    varied = [100 * (index + 1) for index in range(13)]

    steady = run(ArrivalRegularityTracker(), times, [512] * 13)
    noisy = run(ArrivalRegularityTracker(), times, varied)

    assert steady.interval_cv == pytest.approx(noisy.interval_cv)
    assert steady.size_cv == pytest.approx(0.0)
    assert noisy.size_cv > BEACON_THRESHOLD


def test_a_partly_supplied_size_series_publishes_no_size_coefficient():
    # A coefficient over the arrivals that happened to carry a byte count would describe which records the
    # collector filled in rather than what the pair transferred.
    (times, sizes) = beacon(13)
    sizes[4] = None
    result = run(ArrivalRegularityTracker(), times, sizes)

    assert result.mature is True
    assert result.interval_cv is not None
    assert result.size_cv is None


def test_the_sample_cap_narrows_the_claim_and_says_so():
    tracker = ArrivalRegularityTracker(max_samples=10)
    (times, _) = beacon(30)
    result = run(tracker, times)

    assert result.saturated is True
    assert result.arrivals == 10
    assert result.intervals == 9


def test_the_least_recently_seen_entity_is_dropped():
    tracker = ArrivalRegularityTracker(max_entities=2)

    for (index, key) in enumerate(("a:1", "b:2", "c:3")):
        tracker.observe(key, 10**18 + index)

    assert tracker.tracked_entities == 2


def test_the_population_deviation_is_the_one_used():
    # Recorded because the two differ by about four percent at twelve intervals, which is enough to move a value
    # across a threshold of 0.15 -- so which one this is cannot be left to whoever reads `statistics` next.
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert coefficient_of_variation(values) == pytest.approx(statistics.pstdev(values) / statistics.fmean(values))
    assert coefficient_of_variation(values) != pytest.approx(statistics.stdev(values) / statistics.fmean(values))


def test_a_series_too_short_to_vary_has_no_coefficient():
    assert coefficient_of_variation([]) is None
    assert coefficient_of_variation([4.0]) is None


@pytest.mark.parametrize(("kwargs", "match"),
                         [({
                             "window_ns": 0
                         }, "window_ns"), ({
                             "min_intervals": 1
                         }, "min_intervals"), ({
                             "max_samples": 1
                         }, "max_samples"), ({
                             "max_entities": 0
                         }, "max_entities")])
def test_the_bounds_are_refused_rather_than_clamped(kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        ArrivalRegularityTracker(**kwargs)


def test_an_entity_key_is_required():
    with pytest.raises(ValueError, match="entity_key"):
        ArrivalRegularityTracker().observe("", 10**18)


def test_a_float_event_time_is_refused():
    # Nanoseconds as a float lose precision above a microsecond of epoch time, which is every real timestamp.
    with pytest.raises(ValueError, match="event_time_ns"):
        ArrivalRegularityTracker().observe(PAIR, 1.0e18)
