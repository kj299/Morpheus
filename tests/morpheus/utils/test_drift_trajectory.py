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

from morpheus.utils.drift_trajectory import DriftTrajectoryTracker

ALICE = "alice@example.com"

FLAT = [1.0, 1.1, 0.9, 1.0, 1.05, 0.95]
"""Six windows of ordinary variation, which is the baseline a rise is measured against."""


def tracker(**kwargs) -> DriftTrajectoryTracker:
    defaults = {"min_windows": 4}
    defaults.update(kwargs)

    return DriftTrajectoryTracker(**defaults)


def feed(subject: DriftTrajectoryTracker, values: list, entity: str = ALICE, start: int = 100) -> list:
    return [subject.observe(entity, start + index, value) for (index, value) in enumerate(values)]


def test_the_first_window_has_no_trajectory():
    result = feed(tracker(), [1.0])[0]

    assert result.velocity is None
    assert result.acceleration is None
    assert result.rising_windows == 1
    assert result.total_rise is None
    assert result.mature is False


def test_velocity_is_the_first_difference():
    results = feed(tracker(), [1.0, 1.5, 1.9])

    assert results[1].velocity == pytest.approx(0.5)
    assert results[2].velocity == pytest.approx(0.4)


def test_acceleration_is_the_second_difference():
    # The guide's second predictive mechanism: a large positive second difference is drift becoming a spike.
    results = feed(tracker(), [1.0, 1.1, 1.5])

    assert results[2].acceleration == pytest.approx(0.3)
    assert results[1].acceleration is None


def test_the_rule_shape_is_four_consecutive_rising_windows():
    # R-P-L5-006: mean_abs_z increasing monotonically across four consecutive windows. Four windows is three
    # increases, and the count is of windows rather than of increases because that is how the rule is written.
    results = feed(tracker(), FLAT + [1.2, 1.6, 2.1, 2.7])

    # The run begins at the last window of the flat stretch, which is the one the first increase rose from, so
    # four increases make a run of five windows. The rule's condition is the threshold, not the exact count.
    assert [result.rising_windows for result in results[-4:]] == [2, 3, 4, 5]
    assert results[-1].rising_windows >= 4


def test_a_run_starts_at_the_window_before_its_first_increase():
    # Worth asserting on its own, because it decides what `total_rise` is measured from and it is the thing a
    # reader is most likely to assume the other way round.
    results = feed(tracker(), [5.0, 1.0, 1.4, 1.9])

    assert results[1].rising_windows == 1
    assert results[3].rising_windows == 3
    assert results[3].total_rise == pytest.approx(1.9 - 1.0, abs=1e-6)


def test_the_rise_is_measured_in_the_entitys_own_spread():
    subject = tracker()
    feed(subject, FLAT)

    results = [subject.observe(ALICE, 106 + index, value) for (index, value) in enumerate([1.2, 1.6, 2.1, 2.7])]
    final = results[-1]

    # Measured from the window the run started at, which is the last of the flat stretch.
    assert final.total_rise == pytest.approx(2.7 - FLAT[-1], abs=1e-3)
    assert final.baseline_sigma is not None
    assert final.rise_sigmas == pytest.approx(final.total_rise / final.baseline_sigma, abs=1e-3)
    # A flat history makes an ordinary-looking rise a large number of sigmas, which is the point of the measure.
    assert final.rise_sigmas > 1.5


def test_the_same_rise_against_a_noisy_history_is_fewer_sigmas():
    # The comparison that makes this a statement about the principal rather than about the number.
    # Both histories end on a fall to the same level, so both runs start from the same value and the only thing
    # that differs is the spread the rise is expressed in. Getting that wrong is easy: an ending that rises puts
    # the run's start further back and changes the rise itself.
    quiet = tracker()
    feed(quiet, [1.0, 1.02, 0.98, 1.0, 1.01, 1.0])
    quiet_rise = [quiet.observe(ALICE, 106 + i, v) for (i, v) in enumerate([1.2, 1.6, 2.1, 2.7])][-1]

    noisy = tracker()
    feed(noisy, [0.2, 2.0, 0.3, 1.9, 2.4, 1.0])
    noisy_rise = [noisy.observe(ALICE, 106 + i, v) for (i, v) in enumerate([1.2, 1.6, 2.1, 2.7])][-1]

    assert noisy_rise.total_rise == pytest.approx(quiet_rise.total_rise, abs=1e-6)
    assert noisy_rise.rise_sigmas < quiet_rise.rise_sigmas


def test_a_history_that_never_varied_reports_no_sigmas_rather_than_infinity():
    # Dividing by a zero spread would make the flattest history the most alarming, which is backwards.
    subject = tracker()
    feed(subject, [1.0] * 6)

    result = subject.observe(ALICE, 106, 1.5)

    assert result.baseline_sigma == 0.0
    assert result.rise_sigmas is None


def test_the_spread_excludes_the_window_being_judged():
    # A rising sequence that inflated its own denominator would suppress the signal it exists to produce.
    subject = tracker()
    feed(subject, [1.0, 1.0])

    result = subject.observe(ALICE, 102, 9.0)

    assert result.baseline_sigma == 0.0, "the spike was counted in the spread it is measured against"


def test_a_fall_breaks_the_run():
    results = feed(tracker(), [1.0, 1.2, 1.4, 1.1, 1.3])

    assert [result.rising_windows for result in results] == [1, 2, 3, 1, 2]


def test_a_flat_window_breaks_the_run_too():
    # "Increasing monotonically" is strict. A window that held its level did not increase.
    results = feed(tracker(), [1.0, 1.2, 1.2, 1.4])

    assert [result.rising_windows for result in results] == [1, 2, 1, 2]


def test_a_gap_in_windows_restarts_the_run_rather_than_extending_it():
    # A principal who did not authenticate for two days has not been rising for four consecutive windows, and
    # reporting them as though they had would be drift where there was absence.
    subject = tracker()
    subject.observe(ALICE, 100, 1.0)
    subject.observe(ALICE, 101, 1.2)
    skipped = subject.observe(ALICE, 105, 1.5)

    assert skipped.run_restarted is True
    assert skipped.rising_windows == 1
    assert skipped.velocity is None


def test_a_run_that_continues_after_a_gap_starts_from_the_gap():
    subject = tracker()
    feed(subject, [1.0, 1.2])
    subject.observe(ALICE, 110, 1.5)
    after = subject.observe(ALICE, 111, 1.9)

    assert after.rising_windows == 2
    assert after.total_rise == pytest.approx(1.9 - 1.5, abs=1e-6)


def test_maturity_is_reported_rather_than_used_to_withhold():
    # A brand new principal whose score is climbing is not obviously the less interesting case.
    results = feed(tracker(min_windows=3), [1.0, 1.1, 1.2, 1.3, 1.4])

    assert [result.mature for result in results] == [False, False, False, True, True]


def test_entities_do_not_share_a_trajectory():
    subject = tracker()
    feed(subject, [1.0, 1.2, 1.4], entity=ALICE)

    bob = subject.observe("bob@example.com", 103, 5.0)

    assert bob.rising_windows == 1
    assert bob.baseline_sigma is None


def test_an_out_of_order_window_does_not_join_the_trajectory():
    subject = tracker()
    feed(subject, [1.0, 1.2, 1.4])

    late = subject.observe(ALICE, 100, 9.0)

    assert late.out_of_order is True

    # And the run it declined to join is intact.
    after = subject.observe(ALICE, 103, 1.6)

    assert after.rising_windows == 4


def test_only_the_retained_windows_form_the_baseline():
    # The spread should describe recent normal. A principal who changed roles a year ago is not still being
    # compared with who they were.
    subject = tracker(max_windows=4)
    feed(subject, [10.0, 10.0, 1.0, 1.0])

    result = subject.observe(ALICE, 104, 1.1)

    assert result.windows == 4
    assert result.baseline_sigma == pytest.approx(5.196, abs=0.01)


def test_the_least_recently_seen_entity_is_forgotten():
    subject = tracker(max_entities=2)

    subject.observe("a", 100, 1.0)
    subject.observe("b", 100, 1.0)
    subject.observe("c", 100, 1.0)

    assert subject.tracked_entities == 2
    assert subject.observe("a", 101, 1.0).windows == 1


def test_replaying_a_stream_reproduces_it():
    values = FLAT + [1.2, 1.6, 2.1, 2.7, 2.0]

    def run():
        return [(r.velocity, r.acceleration, r.rising_windows, r.rise_sigmas) for r in feed(tracker(), values)]

    assert run() == run()


def test_the_constructor_refuses_values_it_cannot_use():
    with pytest.raises(ValueError, match="max_windows must be at least 2"):
        DriftTrajectoryTracker(max_windows=1)

    with pytest.raises(ValueError, match="min_windows must not be negative"):
        DriftTrajectoryTracker(min_windows=-1)

    with pytest.raises(ValueError, match="max_entities must be positive"):
        DriftTrajectoryTracker(max_entities=0)
