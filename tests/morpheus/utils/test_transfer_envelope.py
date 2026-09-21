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

from morpheus.utils.transfer_envelope import NS_PER_SECOND
from morpheus.utils.transfer_envelope import TransferEnvelopeTracker
from morpheus.utils.transfer_envelope import nearest_rank

TRIPLE = "10.0.0.5:10.0.1.200:445"
HOUR = 3600 * NS_PER_SECOND
START = 10**18


def run(tracker: TransferEnvelopeTracker, values, key: str = TRIPLE, start: int = START, step: int = HOUR):
    """Feed a series of transfers an hour apart and return the last result."""
    result = None

    for (index, value) in enumerate(values):
        result = tracker.observe(key, start + index * step, value)

    return result


def test_the_baseline_is_a_value_the_entity_actually_transferred():
    # Nearest rank rather than interpolation. A reference no transfer ever had is harder to explain to whoever
    # is triaging the alert than one that points at a real transfer, and there is no float tie-break to get
    # wrong on replay.
    values = [float(index) for index in range(1, 101)]
    baseline = nearest_rank(values, 0.99)

    assert baseline in values
    assert baseline == 99.0


def test_a_quantile_over_too_few_samples_is_simply_the_maximum():
    # The property the minimum sample count exists because of. By nearest rank the 99th percentile of anything
    # up to a hundred samples *is* its largest value, so a sparse triple's "99th percentile" is its record and
    # three times a record is a much weaker test than the rule intends.
    for count in (2, 10, 50, 99):
        sample = [float(index) for index in range(1, count + 1)]

        assert nearest_rank(sample, 0.99) == max(sample), count

    hundred = [float(index) for index in range(1, 101)]

    assert nearest_rank(hundred, 0.99) < max(hundred)


def test_no_baseline_is_published_from_too_few_samples():
    tracker = TransferEnvelopeTracker(min_samples=100)
    early = run(tracker, [100.0] * 50)

    assert early.baseline is None
    assert early.ratio is None
    assert early.mature is False
    assert early.breached is False


def test_a_transfer_three_times_the_envelope_is_a_breach():
    # Two fresh trackers rather than two transfers into one, because the first transfer joins the baseline the
    # second is measured against -- which is the behaviour the test below is about and would hide this one.
    ordinary = run(TransferEnvelopeTracker(min_samples=10), [100.0] * 20 + [250.0])
    breach = run(TransferEnvelopeTracker(min_samples=10), [100.0] * 20 + [300.0])

    assert ordinary.baseline == 100.0
    assert ordinary.ratio == pytest.approx(2.5)
    assert ordinary.breached is False

    assert breach.baseline == 100.0
    assert breach.ratio == pytest.approx(3.0)
    assert breach.breached is True


def test_a_breach_raises_the_envelope_the_next_one_is_measured_against():
    # Not a defect: the baseline is the triple's own recent history, and a transfer that happened is part of it.
    # Worth pinning because it means a sustained exfiltration is loudest on its first flow and quieter after,
    # which is the opposite of what somebody tuning the threshold would assume.
    tracker = TransferEnvelopeTracker(min_samples=10)
    run(tracker, [100.0] * 20)

    first = tracker.observe(TRIPLE, START + 20 * HOUR, 1000.0)
    second = tracker.observe(TRIPLE, START + 21 * HOUR, 1000.0)

    assert first.breached is True
    assert second.baseline == 1000.0
    assert second.ratio == pytest.approx(1.0)
    assert second.breached is False


def test_a_transfer_is_not_folded_into_its_own_baseline():
    # One enormous transfer would otherwise partly excuse itself by raising the very reference it is compared
    # against, which is the rule `optical_baseline` follows for the same reason.
    tracker = TransferEnvelopeTracker(min_samples=5)
    run(tracker, [10.0] * 10)

    huge = tracker.observe(TRIPLE, START + 10 * HOUR, 1000.0)

    assert huge.baseline == 10.0
    assert huge.ratio == pytest.approx(100.0)

    # And the one after it sees the raised baseline, because by then the transfer is history.
    after = tracker.observe(TRIPLE, START + 11 * HOUR, 1000.0)

    assert after.baseline == 1000.0


def test_an_envelope_of_zero_yields_no_ratio():
    # A triple that has only ever sent empty payloads has no scale for "three times bigger" to mean anything
    # against, and an infinity here would make the quietest pair in the estate the loudest alert in it.
    tracker = TransferEnvelopeTracker(min_samples=3)
    run(tracker, [0.0] * 5)

    sudden = tracker.observe(TRIPLE, START + 5 * HOUR, 5000.0)

    assert sudden.baseline == 0.0
    assert sudden.mature is True
    assert sudden.ratio is None
    assert sudden.breached is False


def test_each_triple_keeps_its_own_normal():
    # The reason the guide says to baseline per triple. Forty megabytes to a backup server is a Tuesday and the
    # same forty megabytes to a printer is the most interesting thing in the estate.
    tracker = TransferEnvelopeTracker(min_samples=5)
    run(tracker, [1_000_000.0] * 10, key="10.0.0.5:10.0.1.200:445")
    run(tracker, [1_000.0] * 10, key="10.0.0.5:10.0.1.201:9100")

    to_backup = tracker.observe("10.0.0.5:10.0.1.200:445", START + 20 * HOUR, 2_000_000.0)
    to_printer = tracker.observe("10.0.0.5:10.0.1.201:9100", START + 20 * HOUR, 2_000_000.0)

    assert to_backup.breached is False
    assert to_printer.breached is True
    assert tracker.tracked_entities == 2


def test_an_old_transfer_falls_out_of_the_window():
    tracker = TransferEnvelopeTracker(window_ns=10 * HOUR, min_samples=3)
    run(tracker, [1000.0] * 5)

    later = tracker.observe(TRIPLE, START + 1000 * HOUR, 10.0)

    assert later.samples == 1
    assert later.baseline is None
    assert later.mature is False


def test_an_out_of_order_transfer_leaves_the_envelope_alone():
    tracker = TransferEnvelopeTracker(min_samples=3)
    ordered = run(tracker, [100.0] * 10)

    backwards = tracker.observe(TRIPLE, START - HOUR, 9999.0)

    assert backwards.out_of_order is True
    assert backwards.ratio is None
    assert backwards.samples == ordered.samples
    assert backwards.baseline == ordered.baseline


def test_the_sample_cap_narrows_the_claim_and_says_so():
    tracker = TransferEnvelopeTracker(min_samples=2, max_samples=5)
    result = run(tracker, [100.0] * 20)

    assert result.saturated is True
    assert result.samples == 5


def test_the_least_recently_seen_entity_is_dropped():
    tracker = TransferEnvelopeTracker(max_entities=2)

    for (index, key) in enumerate(("a", "b", "c")):
        tracker.observe(key, START + index, 1.0)

    assert tracker.tracked_entities == 2


def test_an_empty_sample_has_no_quantile():
    assert nearest_rank([], 0.99) is None


def test_the_quantile_is_bounded():
    with pytest.raises(ValueError, match="quantile"):
        nearest_rank([1.0], 1.5)


@pytest.mark.parametrize(("kwargs", "match"),
                         [({
                             "window_ns": 0
                         }, "window_ns"), ({
                             "quantile": 2.0
                         }, "quantile"), ({
                             "multiplier": 1.0
                         }, "multiplier"), ({
                             "min_samples": 0
                         }, "min_samples"), ({
                             "max_samples": 0
                         }, "max_samples"), ({
                             "max_entities": 0
                         }, "max_entities")])
def test_the_bounds_are_refused_rather_than_clamped(kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        TransferEnvelopeTracker(**kwargs)


def test_an_entity_key_is_required():
    with pytest.raises(ValueError, match="entity_key"):
        TransferEnvelopeTracker().observe("", START, 1.0)


def test_a_float_event_time_is_refused():
    with pytest.raises(ValueError, match="event_time_ns"):
        TransferEnvelopeTracker().observe(TRIPLE, 1.0e18, 1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_transfer_is_refused(value):
    # A baseline built from an infinity excuses every transfer after it, which is the silent direction.
    with pytest.raises(ValueError, match="finite"):
        TransferEnvelopeTracker().observe(TRIPLE, START, value)


def test_a_value_that_is_not_a_number_is_refused():
    with pytest.raises(ValueError, match="number"):
        TransferEnvelopeTracker().observe(TRIPLE, START, "large")
