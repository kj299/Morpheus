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

from morpheus.utils.value_novelty import ValueNoveltyTracker

SERIAL = "transceiver_serial"
DAY_NS = 24 * 3600 * 10**9
PORT = "hq:sw1:Gi1/0/1"


def tracker(**kwargs) -> ValueNoveltyTracker:
    defaults = {"field_names": [SERIAL]}
    defaults.update(kwargs)

    return ValueNoveltyTracker(**defaults)


def feed(subject: ValueNoveltyTracker, serials: list, entity: str = PORT, start: int = 0, step: int = DAY_NS):
    return [subject.observe(entity, start + index * step, {SERIAL: serial}) for (index, serial) in enumerate(serials)]


def test_the_first_sample_is_not_an_event():
    results = feed(tracker(), ["SN-AAA"])

    # It establishes what normal looks like for this port. Reporting a change would alert on every port the first
    # time it is ever polled.
    assert results[0].changed[SERIAL] is None
    assert results[0].first_seen[SERIAL] is None
    assert results[0].distinct_counts[SERIAL] == 1


def test_a_steady_port_never_changes():
    results = feed(tracker(), ["SN-AAA"] * 4)

    assert [result.changed[SERIAL] for result in results[1:]] == [False, False, False]
    assert results[-1].distinct_counts[SERIAL] == 1


def test_a_substitution_is_detected():
    results = feed(tracker(), ["SN-AAA", "SN-BBB"])

    assert results[1].changed[SERIAL] is True
    assert results[1].first_seen[SERIAL] is True
    assert results[1].distinct_counts[SERIAL] == 2


def test_a_change_across_a_month_boundary_is_detected():
    # This is the whole point of the primitive. Under a period-bucketed count these two samples fall in different
    # buckets, each holding one distinct value, and the substitution is invisible. Here it is simply a change.
    subject = tracker()

    subject.observe(PORT, 0, {SERIAL: "SN-AAA"})
    result = subject.observe(PORT, 40 * DAY_NS, {SERIAL: "SN-BBB"})

    assert result.changed[SERIAL] is True


def test_a_change_across_any_gap_is_detected():
    # No calendar is involved at all, so a year between polls is the same as a minute.
    subject = tracker()

    subject.observe(PORT, 0, {SERIAL: "SN-AAA"})
    result = subject.observe(PORT, 400 * DAY_NS, {SERIAL: "SN-BBB"})

    assert result.changed[SERIAL] is True


def test_returning_to_a_previous_value_is_a_change_but_not_new():
    results = feed(tracker(), ["SN-AAA", "SN-BBB", "SN-AAA"])

    # An optic swapped out and swapped back is two substitutions, and the second one is not unexplained by the
    # estate's own history. Both facts are reported, because they are separately actionable.
    assert results[2].changed[SERIAL] is True
    assert results[2].first_seen[SERIAL] is False
    assert results[2].distinct_counts[SERIAL] == 2


def test_an_empty_cage_is_a_value():
    results = feed(tracker(), [None, None, "SN-AAA", "SN-AAA", None])

    # Installing an optic and removing it are both changes; a null is a physical state, not missing data.
    assert [result.changed[SERIAL] for result in results[1:]] == [False, True, False, True]


def test_ports_do_not_contaminate_each_other():
    subject = tracker()

    feed(subject, ["SN-AAA", "SN-AAA"], entity="hq:sw1:Gi1/0/1")
    results = feed(subject, ["SN-BBB", "SN-BBB"], entity="hq:sw1:Gi1/0/2")

    # Two ports legitimately hold different optics; only a change within one port is a substitution.
    assert results[1].changed[SERIAL] is False


def test_fields_are_tracked_independently():
    subject = ValueNoveltyTracker(field_names=[SERIAL, "lldp_neighbor_chassis_id"])

    subject.observe(PORT, 0, {SERIAL: "SN-AAA", "lldp_neighbor_chassis_id": "chassis-a"})
    result = subject.observe(PORT, DAY_NS, {SERIAL: "SN-AAA", "lldp_neighbor_chassis_id": "chassis-b"})

    # Swapping the optic and re-cabling to a different neighbor are different events.
    assert result.changed[SERIAL] is False
    assert result.changed["lldp_neighbor_chassis_id"] is True


def test_out_of_order_sample_is_flagged_and_ignored():
    subject = tracker()

    feed(subject, ["SN-AAA", "SN-AAA"])
    late = subject.observe(PORT, 0, {SERIAL: "SN-ZZZ"})
    following = subject.observe(PORT, 5 * DAY_NS, {SERIAL: "SN-AAA"})

    assert late.out_of_order is True
    assert late.changed[SERIAL] is None
    # The late value must not have become the state the next sample is compared against.
    assert following.changed[SERIAL] is False


def test_distinct_count_keeps_rising_past_the_recall_bound():
    subject = tracker(max_values=2)

    results = feed(subject, [f"SN-{index}" for index in range(6)])

    # The recall set is bounded, but the count of values that were new when seen is not, so it does not silently
    # stop rising on exactly the port that is churning hardest.
    assert results[-1].distinct_counts[SERIAL] == 6


def test_an_evicted_value_reads_as_new_again():
    # The documented limit of `first_seen`, and it errs toward over-reporting novelty, which is the safe direction.
    subject = tracker(max_values=2)

    results = feed(subject, ["SN-AAA", "SN-BBB", "SN-CCC", "SN-AAA"])

    assert results[3].changed[SERIAL] is True
    assert results[3].first_seen[SERIAL] is True


def test_entities_are_lru_bounded():
    subject = tracker(max_entities=2)

    for index in range(4):
        subject.observe(f"port-{index}", index * DAY_NS, {SERIAL: "SN-AAA"})

    assert subject.tracked_entities == 2


def test_dropped_entity_starts_over_rather_than_claiming_a_change():
    subject = tracker(max_entities=1)

    subject.observe("port-a", 0, {SERIAL: "SN-AAA"})
    subject.observe("port-b", DAY_NS, {SERIAL: "SN-AAA"})
    revived = subject.observe("port-a", 2 * DAY_NS, {SERIAL: "SN-BBB"})

    # Having forgotten the port, the tracker cannot substantiate a change and says so rather than guessing.
    assert revived.changed[SERIAL] is None


def test_constructor_validation():
    with pytest.raises(ValueError):
        ValueNoveltyTracker(field_names=[])

    with pytest.raises(ValueError):
        tracker(max_values=0)

    with pytest.raises(ValueError):
        tracker(max_entities=0)
