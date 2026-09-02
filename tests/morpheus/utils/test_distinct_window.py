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

from morpheus.utils.distinct_window import NS_PER_SECOND
from morpheus.utils.distinct_window import DistinctWindowTracker

MINUTE_NS = 60 * NS_PER_SECOND
PORT = "hq:sw1:Gi1/0/1"


def tracker(**kwargs) -> DistinctWindowTracker:
    defaults = {"window_ns": 60 * MINUTE_NS}
    defaults.update(kwargs)

    return DistinctWindowTracker(**defaults)


def feed(subject: DistinctWindowTracker, values: list, entity: str = PORT, start: int = 0, step: int = MINUTE_NS):
    return [subject.observe(entity, start + index * step, value) for (index, value) in enumerate(values)]


def test_the_current_sample_is_inside_its_own_window():
    # "Distinct MACs in the last hour" includes the one that just arrived, so the alert fires on the sample that
    # pushes the count over a threshold rather than on the one after it.
    results = feed(tracker(), ["mac-a"])

    assert results[0].distinct == 1
    assert results[0].total == 1
    assert results[0].first_in_window is True


def test_a_single_device_port_stays_at_one():
    results = feed(tracker(), ["mac-a"] * 5)

    assert [result.distinct for result in results] == [1, 1, 1, 1, 1]
    assert results[-1].total == 5


def test_an_unauthorized_switch_raises_the_count():
    # Several MACs behind one access port is the classic signature.
    results = feed(tracker(), ["mac-a", "mac-b", "mac-c", "mac-a"])

    assert [result.distinct for result in results] == [1, 2, 3, 3]


def test_first_in_window_marks_only_the_first_appearance():
    results = feed(tracker(), ["mac-a", "mac-b", "mac-a"])

    assert [result.first_in_window for result in results] == [True, True, False]


def test_values_leave_the_window():
    subject = tracker(window_ns=5 * MINUTE_NS)

    feed(subject, ["mac-a", "mac-b"], start=0)
    # An hour later both earlier MACs are outside the window, so only the new one counts.
    late = subject.observe(PORT, 60 * MINUTE_NS, "mac-c")

    assert late.distinct == 1
    assert late.total == 1


def test_a_returning_value_is_new_to_the_window_again():
    # Window-scoped novelty is perishable by design, which is what distinguishes it from never having been seen.
    subject = tracker(window_ns=5 * MINUTE_NS)

    subject.observe(PORT, 0, "mac-a")
    returned = subject.observe(PORT, 60 * MINUTE_NS, "mac-a")

    assert returned.first_in_window is True


def test_the_boundary_is_half_open():
    # A value exactly one window old has left it; anything more recent has not.
    subject = tracker(window_ns=5 * MINUTE_NS)

    subject.observe(PORT, 0, "mac-a")
    subject.observe(PORT, MINUTE_NS, "mac-b")
    result = subject.observe(PORT, 5 * MINUTE_NS, "mac-c")

    assert result.distinct == 2


def test_entities_are_counted_separately():
    subject = tracker()

    feed(subject, ["mac-a", "mac-b"], entity="hq:sw1:Gi1/0/1")
    other = feed(subject, ["mac-c"], entity="hq:sw1:Gi1/0/2", start=10 * MINUTE_NS)

    # Two ports each carrying their own devices is the normal case; pooling them would alarm on every switch.
    assert other[0].distinct == 1


def test_the_inverse_keying_works_the_same_way():
    # Distinct ports per MAC, which catches spoofing or a device moving. The primitive does not care which way
    # round the entity and the value are.
    subject = tracker()

    results = feed(subject, ["Gi1/0/1", "Gi1/0/2", "Gi1/0/3"], entity="00:11:22:33:44:55")

    assert [result.distinct for result in results] == [1, 2, 3]


def test_saturation_is_reported_rather_than_hidden():
    # A MAC flood is what this feature exists to notice and also what would exhaust memory, so the cap turns the
    # count into a floor and says so instead of silently under-reporting.
    subject = tracker(max_samples=3, window_ns=10**18)

    results = feed(subject, [f"mac-{index}" for index in range(6)])

    assert [result.saturated for result in results] == [False, False, False, True, True, True]
    assert results[-1].distinct == 3
    assert results[-1].total == 3


def test_an_unsaturated_window_is_not_flagged():
    results = feed(tracker(max_samples=100), ["mac-a", "mac-b"])

    assert [result.saturated for result in results] == [False, False]


def test_out_of_order_sample_is_flagged_and_ignored():
    subject = tracker()

    feed(subject, ["mac-a", "mac-b"], start=0)
    late = subject.observe(PORT, 0, "mac-zzz")
    following = subject.observe(PORT, 5 * MINUTE_NS, "mac-a")

    assert late.out_of_order is True
    # The late value must not have entered the window, or a later count would include a sample that arrived after
    # the counts it should have preceded.
    assert following.distinct == 2


def test_none_is_counted_like_any_other_value():
    # A port reporting no OUI is a state, not missing data.
    results = feed(tracker(), [None, None, "aa:bb:cc"])

    assert [result.distinct for result in results] == [1, 1, 2]


def test_entities_are_lru_bounded():
    subject = tracker(max_entities=2)

    for index in range(4):
        subject.observe(f"port-{index}", index * MINUTE_NS, "mac-a")

    assert subject.tracked_entities == 2


def test_a_dropped_entity_starts_from_an_empty_window():
    subject = tracker(max_entities=1)

    subject.observe("port-a", 0, "mac-a")
    subject.observe("port-b", MINUTE_NS, "mac-b")
    revived = subject.observe("port-a", 2 * MINUTE_NS, "mac-c")

    assert revived.distinct == 1
    assert revived.first_in_window is True


def test_occurrences_do_not_leak_as_the_window_slides():
    # The occurrence index is what keeps distinct counting constant-time, and it has to shrink as well as grow: a
    # value left in it after leaving the window would inflate the count forever after. Distinct values are needed
    # to show that, since a repeated one reads as a count of 1 whether or not the index is being decremented.
    subject = tracker(window_ns=2 * MINUTE_NS)

    results = feed(subject, ["mac-a", "mac-b", "mac-c", "mac-d", "mac-e"])

    # Only the two values still inside the two-minute window are counted, not all five ever seen.
    assert results[-1].distinct == 2
    assert results[-1].total == 2


def test_constructor_validation():
    with pytest.raises(ValueError):
        tracker(window_ns=0)

    with pytest.raises(ValueError):
        tracker(max_samples=0)

    with pytest.raises(ValueError):
        tracker(max_entities=0)


def test_a_snapshot_reports_every_value_at_one_instant():
    # A MAC table snapshot lists every address on a port with the snapshot's own timestamp, and a source stamping
    # at one-second resolution puts a whole burst on one tick. Neither is a late sample. Rejecting equal timestamps
    # read a five-host hub as one host on every snapshot-shaped collector, which is the primary TC-2 cadence.
    results = feed(tracker(), ["mac-a", "mac-b", "mac-c", "mac-d", "mac-e"], step=0)

    assert [result.distinct for result in results] == [1, 2, 3, 4, 5]
    assert not any(result.out_of_order for result in results)


def test_equal_timestamps_still_expire_against_the_horizon():
    # Admitting equal timestamps must not let a stale value linger: the horizon is judged against the new sample
    # exactly as for a later one.
    subject = tracker(window_ns=10 * MINUTE_NS)

    subject.observe(PORT, 0, "old")
    subject.observe(PORT, 20 * MINUTE_NS, "new-a")
    result = subject.observe(PORT, 20 * MINUTE_NS, "new-b")

    assert result.distinct == 2
