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

from morpheus.utils.bucket_peak import NS_PER_SECOND
from morpheus.utils.bucket_peak import BucketPeakTracker

MINUTE_NS = 60 * NS_PER_SECOND
HOUR_NS = 60 * MINUTE_NS
PORT = "hq:sw1:Gi1/0/3"


def tracker(**kwargs) -> BucketPeakTracker:
    defaults = {"bucket_ns": HOUR_NS, "window_ns": 24 * HOUR_NS, "min_buckets": 3}
    defaults.update(kwargs)

    return BucketPeakTracker(**defaults)


def feed(subject: BucketPeakTracker, counts: list, entity: str = PORT, start: int = 0, step: int = HOUR_NS) -> list:
    """Feed one count per period, one period apart, returning every result."""
    return [subject.observe(entity, start + index * step, count) for (index, count) in enumerate(counts)]


def test_no_reference_until_min_buckets():
    results = feed(tracker(), [1, 1, 1, 1])

    # The first period has no history; the second and third have one and two committed periods, below the floor.
    # The fourth is measured against three.
    assert [result.mature for result in results] == [False, False, False, True]
    assert [result.reference for result in results] == [None, None, None, 1]
    assert [result.buckets for result in results] == [0, 1, 2, 3]
    assert results[3].step == 0


def test_a_count_the_port_has_never_reached_is_a_step():
    results = feed(tracker(), [1, 1, 1, 5])

    assert results[3].reference == 1
    assert results[3].step == 4


def test_the_reference_is_the_peak_not_the_latest():
    # The port carried three devices once, two days of hours ago; one device since. The baseline remembers the
    # three, so two devices now is not a step and four is.
    results = feed(tracker(), [3, 1, 1, 1, 2, 4])

    assert results[4].reference == 3
    assert results[4].step == -1
    assert results[5].step == 1


def test_the_open_period_does_not_raise_its_own_reference():
    subject = tracker()
    feed(subject, [1, 1, 1])

    # Four rows inside one period, the count climbing with each: a hub plugged in. Every row is measured against
    # the periods before this one, so each reads as a step, and the fourth is the largest.
    rows = [subject.observe(PORT, 3 * HOUR_NS + index * MINUTE_NS, count) for (index, count) in enumerate([2, 3, 4, 5])]

    assert [row.reference for row in rows] == [1, 1, 1, 1]
    assert [row.step for row in rows] == [1, 2, 3, 4]

    # The next period's reference has absorbed the hub, and five devices is now what this port carries.
    later = subject.observe(PORT, 4 * HOUR_NS, 5)

    assert later.reference == 5
    assert later.step == 0


def test_rows_at_one_instant_share_a_reference():
    subject = tracker()
    feed(subject, [1, 1, 1])

    # A MAC table snapshot stamps every address on a port at the same instant. Equal timestamps are not out of
    # order, and each row is measured against the same history.
    rows = [subject.observe(PORT, 3 * HOUR_NS, count) for count in [1, 2, 3]]

    assert [row.out_of_order for row in rows] == [False, False, False]
    assert [row.step for row in rows] == [0, 1, 2]


def test_a_quiet_period_adds_nothing():
    subject = tracker()
    feed(subject, [1, 1, 1])
    # Nothing for two hours, then a row: the history is still three periods deep, not five.
    later = subject.observe(PORT, 6 * HOUR_NS, 1)

    assert later.buckets == 3
    assert later.mature is True


def test_periods_outside_the_window_age_out():
    subject = tracker(window_ns=3 * HOUR_NS)
    results = feed(subject, [5, 1, 1, 1, 1, 1])

    # A period is inside the window while any of it is. The period that carried five ends at the first hour; at
    # the fourth observation the horizon is at that hour exactly and the period has just left, so the reference
    # falls to one over the three periods that remain. One observation earlier it was still five.
    assert results[3].reference == 5
    assert results[4].reference == 1
    assert results[4].buckets == 3


def test_periods_are_capped_and_the_cap_is_reported():
    results = feed(tracker(max_buckets=3), [1] * 8)

    assert results[-1].buckets == 3
    assert results[-1].saturated is True
    assert results[3].saturated is False


def test_an_evicted_peak_leaves_the_reference():
    results = feed(tracker(max_buckets=3), [5, 1, 1, 1, 1])

    # The period with five was evicted to keep three; the reference is over the ones that remain.
    assert results[-1].reference == 1


def test_out_of_order_observation_is_refused():
    subject = tracker()
    feed(subject, [1, 1, 1, 1])
    late = subject.observe(PORT, 1 * HOUR_NS, 9)
    following = subject.observe(PORT, 4 * HOUR_NS, 1)

    assert late.out_of_order is True
    assert late.step is None
    # The late nine did not enter any period's peak, or the reference here would be nine.
    assert following.reference == 1


def test_ports_do_not_share_a_history():
    subject = tracker()
    feed(subject, [5, 5, 5, 5], entity="hq:sw1:Gi1/0/1")
    other = feed(subject, [1, 1, 1, 2], entity="hq:sw1:Gi1/0/2")

    assert other[-1].reference == 1
    assert other[-1].step == 1
    assert subject.tracked_entities == 2


def test_entities_are_lru_bounded():
    subject = tracker(max_entities=2)

    for index in range(4):
        subject.observe(f"port-{index}", index * HOUR_NS, 1)

    assert subject.tracked_entities == 2


def test_dropped_entity_starts_over_rather_than_guessing():
    subject = tracker(max_entities=1)
    feed(subject, [1, 1, 1, 1], entity="port-a")
    subject.observe("port-b", 4 * HOUR_NS, 1)
    revived = subject.observe("port-a", 5 * HOUR_NS, 7)

    assert revived.mature is False
    assert revived.step is None


def test_a_null_count_takes_part_in_no_peak():
    subject = tracker()
    feed(subject, [1, 1, 1])
    unknown = subject.observe(PORT, 3 * HOUR_NS, None)

    assert unknown.reference == 1
    assert unknown.step is None

    later = subject.observe(PORT, 4 * HOUR_NS, 1)

    # The period with only a null in it committed no peak, and the reference is still the one it was.
    assert later.reference == 1
    assert later.buckets == 4


def test_replaying_the_stream_reproduces_every_result():
    counts = [1, 1, 3, 1, 1, 2, 5, 5, 1]

    assert feed(tracker(), counts) == feed(tracker(), counts)


def test_constructor_validation():
    with pytest.raises(ValueError, match="bucket_ns"):
        tracker(bucket_ns=0)

    with pytest.raises(ValueError, match="window_ns"):
        tracker(bucket_ns=HOUR_NS, window_ns=MINUTE_NS)

    with pytest.raises(ValueError, match="min_buckets"):
        tracker(min_buckets=0)

    with pytest.raises(ValueError, match="max_buckets"):
        tracker(min_buckets=10, max_buckets=3)

    with pytest.raises(ValueError, match="max_entities"):
        tracker(max_entities=0)
