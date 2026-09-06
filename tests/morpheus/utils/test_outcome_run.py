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

from morpheus.utils.outcome_run import NS_PER_SECOND
from morpheus.utils.outcome_run import OutcomeRunTracker

MINUTE_NS = 60 * NS_PER_SECOND
ALICE = "alice@example.com"

DENY = False
ALLOW = True


def tracker(**kwargs) -> OutcomeRunTracker:
    defaults = {"window_ns": 10 * MINUTE_NS}
    defaults.update(kwargs)

    return OutcomeRunTracker(**defaults)


def feed(subject: OutcomeRunTracker,
         outcomes: list,
         entity: str = ALICE,
         start: int = 0,
         step: int = NS_PER_SECOND) -> list:
    return [subject.observe(entity, start + index * step, outcome) for (index, outcome) in enumerate(outcomes)]


def test_the_mfa_fatigue_shape_is_reported_on_the_approval():
    # R-D-L5-004: more than five challenges in ten minutes, at least four denials, and an approval at the end.
    # Every part of that has to be readable off the row carrying the approval, or the rule cannot be written.
    results = feed(tracker(), [DENY] * 4 + [ALLOW])
    approval = results[-1]

    assert approval.attempts == 5
    assert approval.failures == 4
    assert approval.consecutive_failures == 4
    assert approval.failure_then_success is True


def test_denials_need_not_be_contiguous_to_be_counted():
    # A fatigue attack interleaves with the victim's own traffic, so the denials are not a clean run. `failures`
    # counts them regardless; `consecutive_failures` reports only the unbroken tail.
    results = feed(tracker(), [DENY, DENY, ALLOW, DENY, DENY, ALLOW])
    final = results[-1]

    assert final.failures == 4
    assert final.consecutive_failures == 2
    assert final.failure_then_success is True


def test_a_success_with_nothing_before_it_is_not_the_pattern():
    results = feed(tracker(), [ALLOW])

    assert results[0].failure_then_success is False
    assert results[0].consecutive_failures == 0


def test_a_success_after_a_success_is_not_the_pattern():
    results = feed(tracker(), [DENY, ALLOW, ALLOW])

    assert results[1].failure_then_success is True
    assert results[2].failure_then_success is False
    assert results[2].consecutive_failures == 0


def test_a_failure_never_reports_the_pattern_however_long_the_run():
    # Failures alone are the most ordinary event in an estate. The trailing approval is what makes this
    # actionable, so a run with no approval at the end reports its length and nothing more.
    results = feed(tracker(), [DENY] * 6)

    assert [result.failure_then_success for result in results] == [False] * 6
    assert results[-1].consecutive_failures == 5


def test_the_run_is_read_before_the_attempt_joins_it():
    results = feed(tracker(), [DENY, DENY])

    # The second denial saw one before it, not two.
    assert results[1].consecutive_failures == 1
    assert results[1].failures == 2


def test_the_counts_decay_out_of_the_window():
    subject = tracker(window_ns=5 * MINUTE_NS)
    feed(subject, [DENY] * 4)

    late = subject.observe(ALICE, 30 * MINUTE_NS, ALLOW)

    assert late.attempts == 1
    assert late.failures == 0
    assert late.consecutive_failures == 0
    assert late.failure_then_success is False


def test_a_run_that_began_before_the_window_is_reported_short():
    # Under-reporting rather than inventing history the window has already forgotten.
    subject = tracker(window_ns=5 * MINUTE_NS)

    for minute in range(6):
        subject.observe(ALICE, minute * MINUTE_NS, DENY)

    result = subject.observe(ALICE, 6 * MINUTE_NS, ALLOW)

    assert result.consecutive_failures == result.attempts - 1
    assert result.consecutive_failures < 6


def test_several_attempts_on_one_tick_all_count():
    # Identity providers stamp to the second, so a fatigue burst puts several challenges on one tick. Rejecting
    # them as duplicates would read the attack this feature exists to catch as a single prompt.
    results = feed(tracker(), [DENY] * 4 + [ALLOW], step=0)

    assert results[-1].attempts == 5
    assert results[-1].failures == 4
    assert results[-1].failure_then_success is True
    assert [result.out_of_order for result in results] == [False] * 5


def test_entities_do_not_share_a_window():
    subject = tracker()
    feed(subject, [DENY] * 4, entity=ALICE)

    bob = subject.observe("bob@example.com", 10 * NS_PER_SECOND, ALLOW)

    assert bob.failures == 0
    assert bob.failure_then_success is False


def test_an_out_of_order_attempt_does_not_join_the_window():
    subject = tracker()
    feed(subject, [DENY, DENY], start=10 * MINUTE_NS)

    late = subject.observe(ALICE, 0, ALLOW)

    assert late.out_of_order is True
    assert late.failure_then_success is False

    # And the run it declined to end is still going.
    after = subject.observe(ALICE, 11 * MINUTE_NS, ALLOW)

    assert after.consecutive_failures == 2
    assert after.failure_then_success is True


def test_the_sample_cap_makes_the_counts_a_floor_and_says_so():
    subject = tracker(max_samples=3)
    results = feed(subject, [DENY] * 5)

    assert results[-1].attempts == 3
    assert results[-1].saturated is True
    assert results[0].saturated is False


def test_an_evicted_failure_stops_being_counted():
    subject = tracker(max_samples=2)
    results = feed(subject, [DENY, DENY, ALLOW])

    assert results[-1].attempts == 2
    assert results[-1].failures == 1


def test_the_least_recently_seen_entity_is_forgotten():
    subject = tracker(max_entities=2)

    subject.observe("a", 0, DENY)
    subject.observe("b", NS_PER_SECOND, DENY)
    subject.observe("c", 2 * NS_PER_SECOND, DENY)

    assert subject.tracked_entities == 2

    revived = subject.observe("a", 3 * NS_PER_SECOND, ALLOW)

    assert revived.failures == 0
    assert revived.failure_then_success is False


def test_replaying_a_stream_reproduces_it():
    outcomes = [DENY, DENY, ALLOW, DENY, DENY, DENY, ALLOW, ALLOW]

    def run():
        return [(result.failures, result.consecutive_failures, result.failure_then_success)
                for result in feed(tracker(), outcomes)]

    assert run() == run()


def test_the_constructor_refuses_values_it_cannot_use():
    with pytest.raises(ValueError, match="window_ns must be positive"):
        OutcomeRunTracker(window_ns=0)

    with pytest.raises(ValueError, match="max_samples must be positive"):
        OutcomeRunTracker(max_samples=0)

    with pytest.raises(ValueError, match="max_entities must be positive"):
        OutcomeRunTracker(max_entities=0)
