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

import math

import pytest

from morpheus.utils.cyclic_histogram import CyclicHistogramTracker

HOUR_NS = 3600 * 10**9
USER = "alice@example.com"

OFFICE_HOURS = list(range(9, 18))
"""The buckets a nine-to-five account occupies, as hours of the day."""


def tracker(**kwargs) -> CyclicHistogramTracker:
    defaults = {"buckets": 24, "min_samples": 8}
    defaults.update(kwargs)

    return CyclicHistogramTracker(**defaults)


def feed(subject: CyclicHistogramTracker, buckets: list, entity: str = USER, start: int = 0, step: int = HOUR_NS):
    return [subject.observe(entity, start + index * step, bucket) for (index, bucket) in enumerate(buckets)]


def test_a_first_observation_is_neither_surprising_nor_reassuring():
    # With no history, the smoothed share is exactly the uniform one. Reporting anything else would be inventing a
    # verdict out of an empty histogram.
    result = feed(tracker(), [3])[0]

    assert result.samples == 0
    assert result.bucket_count == 0
    assert result.smoothed_share == pytest.approx(1 / 24, abs=1e-4)
    assert result.surprise_bits == pytest.approx(math.log2(24), abs=1e-4)
    assert result.mature is False


def test_the_sample_is_judged_against_history_that_excludes_it():
    # The convention that matters. Were the sample counted first, a user's very first 03:00 authentication would
    # hold a share of one in one -- the least surprising value the measure has, at the moment it should be the most.
    subject = tracker()
    feed(subject, OFFICE_HOURS * 4)

    night = subject.observe(USER, 100 * HOUR_NS, 3)

    assert night.bucket_count == 0
    assert night.bucket_unseen is True

    # And the count it reported is the history before it, not after.
    assert night.samples == len(OFFICE_HOURS) * 4


def test_the_same_hour_becomes_unremarkable_as_it_repeats():
    results = feed(tracker(), [3] * 12)
    bits = [result.surprise_bits for result in results]

    assert bits == sorted(bits, reverse=True)
    assert bits[0] > bits[-1]


def test_a_long_history_makes_a_new_bucket_more_surprising_than_a_short_one_does():
    # The ordering the smoothing exists to produce: the same 03:00 login is a bigger claim about a two-year-old
    # account than about one in its second week, and it falls out of the history length rather than a threshold.
    veteran = tracker()
    feed(veteran, OFFICE_HOURS * 40)
    veteran_night = veteran.observe(USER, 10_000 * HOUR_NS, 3)

    joiner = tracker()
    feed(joiner, OFFICE_HOURS)
    joiner_night = joiner.observe(USER, 10_000 * HOUR_NS, 3)

    assert veteran_night.surprise_bits > joiner_night.surprise_bits
    assert veteran_night.bucket_unseen is joiner_night.bucket_unseen is True


def test_an_unseen_bucket_is_finite_rather_than_infinite():
    # Add-one smoothing is the whole reason this is a number a threshold can be written against.
    subject = tracker()
    feed(subject, [9] * 500)

    result = subject.observe(USER, 1_000 * HOUR_NS, 3)

    assert math.isfinite(result.surprise_bits)
    assert result.smoothed_share > 0.0


def test_maturity_is_reported_rather_than_withheld():
    # A brand new account authenticating at 03:00 is not obviously the less interesting case, so the tracker says
    # how much history it has and leaves the choice to the caller.
    subject = tracker(min_samples=8)
    results = feed(subject, [9] * 10)

    assert [result.mature for result in results] == [False] * 8 + [True] * 2


def test_the_share_is_the_arithmetic_it_claims_to_be():
    subject = tracker()
    feed(subject, [9] * 3 + [10] * 2)

    result = subject.observe(USER, 99 * HOUR_NS, 9)

    # Three prior observations of hour 9, five prior samples, twenty-four buckets.
    assert result.smoothed_share == pytest.approx(4 / 29, abs=1e-4)
    assert result.surprise_bits == pytest.approx(-math.log2(4 / 29), abs=1e-4)


def test_entities_do_not_see_each_others_history():
    subject = tracker()
    feed(subject, [9] * 20, entity="alice@example.com")

    bob = subject.observe("bob@example.com", 99 * HOUR_NS, 9)

    assert bob.samples == 0
    assert bob.bucket_unseen is True


def test_an_out_of_order_sample_does_not_enter_the_histogram():
    subject = tracker()
    feed(subject, [9] * 4)

    late = subject.observe(USER, 0, 3)

    assert late.out_of_order is True
    assert late.samples == 4

    # And the history it declined to join is unchanged.
    after = subject.observe(USER, 100 * HOUR_NS, 3)

    assert after.samples == 4
    assert after.bucket_unseen is True


def test_a_bucket_outside_the_cycle_is_refused():
    # A sentinel would be counted as a real bucket, and would then make every genuine observation of it look less
    # unusual than it is.
    subject = tracker()

    with pytest.raises(ValueError, match=r"bucket must fall in \[0, 24\)"):
        subject.observe(USER, 0, 24)

    with pytest.raises(ValueError, match=r"bucket must fall in \[0, 24\)"):
        subject.observe(USER, 0, -1)


def test_a_seven_bucket_cycle_is_the_same_measure():
    subject = tracker(buckets=7)
    weekdays = subject.observe(USER, 0, 0)

    assert weekdays.smoothed_share == pytest.approx(1 / 7, abs=1e-4)
    assert subject.buckets == 7


def test_the_least_recently_seen_entity_is_forgotten():
    subject = tracker(max_entities=2)

    subject.observe("a", 0, 9)
    subject.observe("b", HOUR_NS, 9)
    subject.observe("c", 2 * HOUR_NS, 9)

    assert subject.tracked_entities == 2

    # "a" was evicted, so it reads as a new entity rather than as one carrying a stale share.
    revived = subject.observe("a", 3 * HOUR_NS, 9)

    assert revived.samples == 0


def test_replaying_a_stream_reproduces_it():
    buckets = OFFICE_HOURS * 3 + [3, 4] + OFFICE_HOURS

    first = [result.surprise_bits for result in feed(tracker(), buckets)]
    second = [result.surprise_bits for result in feed(tracker(), buckets)]

    assert first == second


def test_the_constructor_refuses_a_cycle_it_cannot_count_over():
    with pytest.raises(ValueError, match="buckets must be positive"):
        CyclicHistogramTracker(buckets=0)

    with pytest.raises(ValueError, match="min_samples must not be negative"):
        CyclicHistogramTracker(buckets=24, min_samples=-1)

    with pytest.raises(ValueError, match="max_entities must be positive"):
        CyclicHistogramTracker(buckets=24, max_entities=0)
