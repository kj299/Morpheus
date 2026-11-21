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

import pytest

from morpheus.utils.distinct_window import NS_PER_SECOND
from morpheus.utils.sum_window import SumWindowResult
from morpheus.utils.sum_window import SumWindowTracker
from morpheus.utils.tracker_state import from_bytes
from morpheus.utils.tracker_state import to_bytes

MINUTE_NS = 60 * NS_PER_SECOND
SERVICE = "api.corp.example:443"


def tracker(**kwargs) -> SumWindowTracker:
    defaults = {"window_ns": 60 * MINUTE_NS}
    defaults.update(kwargs)

    return SumWindowTracker(**defaults)


def feed(subject: SumWindowTracker, amounts: list, entity: str = SERVICE, start: int = 0, step: int = MINUTE_NS):
    return [subject.observe(entity, start + index * step, amount) for (index, amount) in enumerate(amounts)]


def test_the_current_observation_is_inside_its_own_window():
    assert feed(tracker(), [10])[0] == SumWindowResult(count=1, total=10.0, saturated=False, out_of_order=False)


def test_counts_and_sums_accumulate_within_the_window():
    results = feed(tracker(), [10, 20, 30])

    assert [result.count for result in results] == [1, 2, 3]
    assert [result.total for result in results] == [10.0, 30.0, 60.0]
    assert not any(result.saturated for result in results)


def test_an_observation_of_nothing_is_counted_and_adds_nothing():
    results = feed(tracker(), [10, None, 5])

    assert [result.count for result in results] == [1, 2, 3]
    assert [result.total for result in results] == [10.0, 10.0, 15.0]


def test_an_observation_leaves_when_the_window_has_passed_it():
    subject = tracker()
    subject.observe(SERVICE, 0, 10)
    subject.observe(SERVICE, 30 * MINUTE_NS, 20)

    # Exactly a window later the first is out: the window is open at its far end, as the distinct window's is.
    result = subject.observe(SERVICE, 60 * MINUTE_NS, 30)

    assert (result.count, result.total) == (2, 50.0)
    assert result.saturated is False


def test_expiry_by_age_is_not_saturation():
    subject = tracker(max_samples=2)
    subject.observe(SERVICE, 0, 10)
    subject.observe(SERVICE, MINUTE_NS, 20)

    # Both earlier observations have aged out by now, so the cap displaces nothing the window would have kept.
    result = subject.observe(SERVICE, 3 * 60 * MINUTE_NS, 30)

    assert (result.count, result.total, result.saturated) == (1, 30.0, False)


def test_a_binding_cap_is_reported_and_the_figures_are_what_it_kept():
    results = feed(tracker(max_samples=3), [10, 20, 30, 40, 50])

    assert [result.count for result in results] == [1, 2, 3, 3, 3]
    assert [result.total for result in results] == [10.0, 30.0, 60.0, 90.0, 120.0]
    assert [result.saturated for result in results] == [False, False, False, True, True]


def test_an_earlier_observation_is_refused_and_changes_nothing():
    subject = tracker()
    subject.observe(SERVICE, 10 * MINUTE_NS, 10)

    late = subject.observe(SERVICE, 5 * MINUTE_NS, 99)

    assert late == SumWindowResult(count=1, total=10.0, saturated=False, out_of_order=True)
    assert subject.observe(SERVICE, 11 * MINUTE_NS, 5) == SumWindowResult(count=2,
                                                                          total=15.0,
                                                                          saturated=False,
                                                                          out_of_order=False)


def test_observations_at_one_instant_are_all_counted():
    subject = tracker()
    results = [subject.observe(SERVICE, MINUTE_NS, amount) for amount in (1, 2, 3)]

    assert [result.count for result in results] == [1, 2, 3]
    assert not any(result.out_of_order for result in results)


def test_entities_do_not_share_a_window():
    subject = tracker()
    subject.observe("a:443", 0, 10)

    assert subject.observe("b:443", MINUTE_NS, 5).count == 1
    assert subject.observe("a:443", 2 * MINUTE_NS, 1).total == 11.0
    assert subject.tracked_entities == 2


def test_the_least_recently_seen_entity_is_forgotten_first():
    subject = tracker(max_entities=2)
    subject.observe("a:443", 0, 10)
    subject.observe("b:443", MINUTE_NS, 10)
    subject.observe("a:443", 2 * MINUTE_NS, 10)
    subject.observe("c:443", 3 * MINUTE_NS, 10)

    assert subject.tracked_entities == 2
    # `b` went, so it starts from an empty window; `a` kept its two.
    assert subject.observe("a:443", 4 * MINUTE_NS, 10).count == 3
    assert subject.observe("b:443", 5 * MINUTE_NS, 10).count == 1


@pytest.mark.parametrize("name", ["window_ns", "max_samples", "max_entities"])
def test_constructor_rejects_a_bound_that_is_not_positive(name: str):
    with pytest.raises(ValueError, match=name):
        SumWindowTracker(**{name: 0})


def test_a_restored_tracker_continues_where_the_stopped_one_was():
    stopped = tracker(max_samples=3)
    feed(stopped, [10, 20, 30, 40])
    feed(stopped, [5], entity="b:443", start=4 * MINUTE_NS)

    resumed = SumWindowTracker.from_state(from_bytes(to_bytes(stopped.to_state())))

    assert to_bytes(resumed.to_state()) == to_bytes(stopped.to_state())

    (expected, restored) = [[subject.observe(SERVICE, 5 * MINUTE_NS, 50), subject.observe("b:443", 6 * MINUTE_NS, 1)]
                            for subject in (stopped, resumed)]

    assert restored == expected
    assert restored[0].saturated is True
