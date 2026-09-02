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

from morpheus.utils.ratio_window import NS_PER_SECOND
from morpheus.utils.ratio_window import RatioWindowTracker

SECOND_NS = NS_PER_SECOND
HOST = "00:11:22:33:44:55"


def tracker(**kwargs) -> RatioWindowTracker:
    defaults = {"window_ns": 300 * SECOND_NS, "min_denominator": 4}
    defaults.update(kwargs)

    return RatioWindowTracker(**defaults)


def feed(subject: RatioWindowTracker, flags: list, entity: str = HOST, start: int = 0, step: int = SECOND_NS):
    return [subject.observe(entity, start + index * step, flag) for (index, flag) in enumerate(flags)]


def test_no_ratio_below_the_denominator_floor():
    # One event out of one is 1.0, which reads as total saturation and means almost nothing.
    results = feed(tracker(), [True, True, True, True])

    assert [result.ratio for result in results[0:3]] == [None, None, None]
    assert results[3].ratio == pytest.approx(1.0)


def test_a_quiet_host_reads_zero():
    results = feed(tracker(), [False] * 6)

    assert results[-1].ratio == pytest.approx(0.0)
    assert results[-1].numerator == 0
    assert results[-1].denominator == 6


def test_a_flooding_host_reads_high():
    results = feed(tracker(), [False, False, True, True, True, True])

    assert results[-1].numerator == 4
    assert results[-1].denominator == 6
    assert results[-1].ratio == pytest.approx(4 / 6)


def test_the_current_event_is_counted():
    # The proportion has to move on the event that changes it, not on the one after.
    results = feed(tracker(min_denominator=1), [True])

    assert results[0].ratio == pytest.approx(1.0)


def test_a_busy_host_is_not_penalised_for_volume():
    # The whole reason this is a proportion. Both hosts sent four gratuitous events; only one is mostly gratuitous.
    subject = tracker()

    chatty = feed(subject, [True] * 4 + [False] * 16, entity="chatty")
    quiet = feed(subject, [True] * 4, entity="quiet")

    assert chatty[-1].numerator == quiet[-1].numerator == 4
    assert chatty[-1].ratio == pytest.approx(0.2)
    assert quiet[-1].ratio == pytest.approx(1.0)


def test_events_leave_the_window():
    subject = tracker(min_denominator=1, window_ns=5 * SECOND_NS)

    feed(subject, [True, True, True, True], start=0)
    late = subject.observe(HOST, 60 * SECOND_NS, False)

    # An hour of quiet later, the earlier burst is outside the window and the host reads clean again.
    assert late.numerator == 0
    assert late.denominator == 1
    assert late.ratio == pytest.approx(0.0)


def test_the_numerator_shrinks_with_the_window():
    # The running numerator has to fall as events expire, or the proportion climbs forever.
    subject = tracker(min_denominator=1, window_ns=3 * SECOND_NS)

    results = feed(subject, [True, True, True, False, False, False])

    assert results[-1].numerator == 0
    assert results[-1].ratio == pytest.approx(0.0)


def test_entities_are_measured_separately():
    subject = tracker()

    feed(subject, [True] * 4, entity="attacker")
    innocent = feed(subject, [False] * 4, entity="bystander", start=10 * SECOND_NS)

    assert innocent[-1].ratio == pytest.approx(0.0)


def test_saturation_is_reported_rather_than_hidden():
    subject = tracker(min_denominator=1, max_samples=3, window_ns=10**18)

    results = feed(subject, [True] * 6)

    assert [result.saturated for result in results] == [False, False, False, True, True, True]
    assert results[-1].denominator == 3


def test_out_of_order_event_is_flagged_and_ignored():
    subject = tracker(min_denominator=1)

    feed(subject, [False, False], start=0)
    late = subject.observe(HOST, 0, True)
    following = subject.observe(HOST, 10 * SECOND_NS, False)

    assert late.out_of_order is True
    assert late.ratio is None
    # The late event must not have entered the window, or it would inflate every later proportion.
    assert following.numerator == 0


def test_entities_are_lru_bounded():
    subject = tracker(max_entities=2)

    for index in range(4):
        subject.observe(f"host-{index}", index * SECOND_NS, True)

    assert subject.tracked_entities == 2


def test_constructor_validation():
    with pytest.raises(ValueError):
        tracker(window_ns=0)

    with pytest.raises(ValueError):
        tracker(min_denominator=0)

    with pytest.raises(ValueError, match="max_samples"):
        tracker(min_denominator=100, max_samples=10)

    with pytest.raises(ValueError):
        tracker(max_entities=0)


def test_a_flood_inside_one_tick_is_counted_in_full():
    # Many sources stamp at one-second resolution, so twenty announcements from one host land on one timestamp.
    # Rejecting equal timestamps kept the denominator at one, the ratio below its floor, and the flood invisible,
    # which is the one event this proportion exists to expose.
    results = feed(tracker(min_denominator=4), [True] * 20, step=0)

    assert results[-1].denominator == 20
    assert results[-1].ratio == pytest.approx(1.0)
    assert not any(result.out_of_order for result in results)
