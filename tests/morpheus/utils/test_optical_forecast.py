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

from morpheus.utils.optical_forecast import NS_PER_SECOND
from morpheus.utils.optical_forecast import STATUS_BELOW_FLOOR
from morpheus.utils.optical_forecast import STATUS_IMMATURE
from morpheus.utils.optical_forecast import STATUS_NO_FLOOR
from morpheus.utils.optical_forecast import STATUS_NO_READING
from morpheus.utils.optical_forecast import STATUS_NONLINEAR
from morpheus.utils.optical_forecast import STATUS_NOT_DEGRADING
from morpheus.utils.optical_forecast import STATUS_PROJECTED
from morpheus.utils.optical_forecast import STATUSES
from morpheus.utils.optical_forecast import OpticalForecastTracker

MINUTE_NS = 60 * NS_PER_SECOND
PORT = "hq:sw1:Gi1/0/1"
FLOOR = -14.4

# A tenth of a decibel a minute is 144 dB a day: a connector failing fast enough to watch, and arithmetic a reader
# can check by hand.
FAILING = [-7.0 - 0.1 * index for index in range(10)]
STEADY = [-7.0] * 10
# Readings jittering around one level the way optical diagnostics do, with no lean to them.
NOISY = [-7.00, -7.03, -6.98, -7.01, -6.97, -7.02, -6.99, -7.00, -7.02, -6.98]


def tracker(**kwargs) -> OpticalForecastTracker:
    defaults = {"min_samples": 4, "window_ns": 60 * MINUTE_NS}
    defaults.update(kwargs)

    return OpticalForecastTracker(**defaults)


def feed(subject: OpticalForecastTracker,
         levels: list,
         entity: str = PORT,
         floor: float = FLOOR,
         optic: str = "SN-A",
         start: int = 0,
         step: int = MINUTE_NS) -> list:
    """Feed a series of receive levels one step apart, returning every result."""
    return [
        subject.observe(entity, start + index * step, level, floor_dbm=floor, optic=optic)
        for (index, level) in enumerate(levels)
    ]


def test_no_fit_until_min_samples():
    results = feed(tracker(), FAILING)

    assert [result.status for result in results[0:3]] == [STATUS_IMMATURE] * 3
    assert [result.trend_db_per_day for result in results[0:3]] == [None] * 3
    assert results[3].status == STATUS_PROJECTED
    assert results[3].samples == 4


def test_a_steady_degradation_is_projected_to_the_floor():
    last = feed(tracker(), FAILING)[-1]

    # Exactly linear readings, so the figures are the ones a reader computes: the line is at -7.9 now, falling
    # 144 dB a day, and the floor is 6.5 dB below it.
    assert last.status == STATUS_PROJECTED
    assert last.trend_db_per_day == pytest.approx(-144.0)
    assert last.residual_db == pytest.approx(0.0)
    assert last.floor_dbm == FLOOR
    assert last.days_to_floor == pytest.approx(6.5 / 144.0, abs=1e-4)
    assert last.samples == len(FAILING)


def test_the_projection_shortens_as_the_optic_fails():
    days = [result.days_to_floor for result in feed(tracker(), FAILING) if result.status == STATUS_PROJECTED]

    assert len(days) == len(FAILING) - 3
    assert days == sorted(days, reverse=True)


def test_exactly_linear_readings_are_as_significant_as_a_slope_can_be():
    last = feed(tracker(), FAILING)[-1]

    # The residual of readings that sit on a line is zero up to the arithmetic's own rounding, so the slope is
    # either infinitely far from flat or as near that as a double gets. Either way it is nowhere near the bar.
    assert math.isinf(last.significance) or last.significance > 1e6


def test_a_step_is_not_a_line():
    # Eight steady readings and a three-decibel drop held for four more: an inline tap. A line through that is
    # steep, and a forecast read off it would give the optic hours. The residual says the readings are not on it.
    results = feed(tracker(), STEADY[0:8] + [-10.0] * 4)

    assert results[-1].status == STATUS_NONLINEAR
    assert results[-1].residual_db > 0.5
    assert results[-1].days_to_floor is None
    # The slope is still reported, so a reader can see what the fiction would have said.
    assert results[-1].trend_db_per_day < 0


def test_noise_is_not_a_trend():
    results = feed(tracker(), NOISY)

    # The readings lean one way or another by chance, and the slope is whatever that lean happens to be. Measured
    # against its own standard error it is nowhere near a degradation, and nothing is projected from it.
    assert results[-1].status == STATUS_NOT_DEGRADING
    assert results[-1].significance < 4.0
    assert results[-1].days_to_floor is None
    assert STATUS_PROJECTED not in {result.status for result in results}


def test_a_small_lean_can_be_forced_through_by_lowering_the_bar():
    # The same noise under a tracker that accepts any slope: the parameter is what separates the two answers.
    results = feed(tracker(min_significance=0.0), NOISY)
    slopes = [result.trend_db_per_day for result in results if result.trend_db_per_day is not None]

    assert any(slope < 0 for slope in slopes)
    assert STATUS_PROJECTED in {result.status for result in results}


def test_a_rising_level_is_not_degrading():
    results = feed(tracker(), [-9.0 + 0.1 * index for index in range(10)])

    assert results[-1].status == STATUS_NOT_DEGRADING
    assert results[-1].trend_db_per_day == pytest.approx(144.0)
    assert results[-1].days_to_floor is None


def test_at_or_below_the_floor_is_reported_now_not_forecast():
    results = feed(tracker(), FAILING, floor=-7.5)

    # The sixth reading is the floor itself. From there the status says the link is in trouble now, and the time to
    # the floor is nothing.
    assert results[4].status == STATUS_PROJECTED
    assert results[5].status == STATUS_BELOW_FLOOR
    assert results[5].days_to_floor == 0.0
    assert results[-1].status == STATUS_BELOW_FLOOR


def test_no_floor_reports_the_trend_and_projects_nothing():
    last = feed(tracker(), FAILING, floor=None)[-1]

    assert last.status == STATUS_NO_FLOOR
    assert last.trend_db_per_day == pytest.approx(-144.0)
    assert last.floor_dbm is None
    assert last.days_to_floor is None


def test_an_empty_cage_contributes_nothing():
    subject = tracker()
    feed(subject, FAILING[0:5])
    empty = subject.observe(PORT, 5 * MINUTE_NS, None, floor_dbm=FLOOR, optic="SN-A")
    # The reading the line predicts for the sixth minute, so the gap is the only thing that could disturb the fit.
    following = subject.observe(PORT, 6 * MINUTE_NS, -7.0 - 0.1 * 6, floor_dbm=FLOOR, optic="SN-A")

    assert empty.status == STATUS_NO_READING
    assert empty.samples == 5
    assert empty.trend_db_per_day is None
    # The gap did not enter the fit, and the readings either side of it still sit on one line.
    assert following.status == STATUS_PROJECTED
    assert following.samples == 6
    assert following.residual_db == pytest.approx(0.0)


def test_a_replaced_optic_starts_over():
    subject = tracker()
    feed(subject, FAILING)
    replaced = subject.observe(PORT, 10 * MINUTE_NS, -7.0, floor_dbm=FLOOR, optic="SN-B")

    # The failing optic's readings say nothing about the one that replaced it. The new optic begins with a history
    # of one, and its fit, once it has one, is its own.
    assert replaced.history_reset is True
    assert replaced.status == STATUS_IMMATURE
    assert replaced.samples == 1

    settled = feed(subject, STEADY[0:4], optic="SN-B", start=11 * MINUTE_NS)[-1]

    assert settled.history_reset is False
    assert settled.status == STATUS_NOT_DEGRADING
    assert settled.trend_db_per_day == pytest.approx(0.0)


def test_an_emptied_cage_is_a_change_of_optic_too():
    subject = tracker()
    feed(subject, FAILING[0:5])
    emptied = subject.observe(PORT, 5 * MINUTE_NS, None, floor_dbm=FLOOR, optic=None)

    assert emptied.history_reset is True
    assert emptied.samples == 0


def test_readings_outside_the_window_are_dropped():
    subject = tracker(window_ns=5 * MINUTE_NS)
    results = feed(subject, FAILING)

    # Five minutes of window at a minute a reading holds five readings: this one and the four before it.
    assert results[-1].samples == 5
    assert results[-1].status == STATUS_PROJECTED


def test_samples_are_capped_and_the_cap_is_reported():
    results = feed(tracker(max_samples=6), FAILING)

    assert results[-1].samples == 6
    assert results[-1].saturated is True
    assert results[4].saturated is False


def test_out_of_order_sample_is_refused_and_leaves_the_fit_alone():
    subject = tracker()
    feed(subject, FAILING[0:6])
    late = subject.observe(PORT, 2 * MINUTE_NS, -30.0, floor_dbm=FLOOR, optic="SN-A")
    following = subject.observe(PORT, 6 * MINUTE_NS, FAILING[6], floor_dbm=FLOOR, optic="SN-A")

    assert late.out_of_order is True
    assert late.trend_db_per_day is None
    assert late.days_to_floor is None
    # Had the late reading entered the history, the residual here would be enormous rather than nothing.
    assert following.residual_db == pytest.approx(0.0)
    assert following.samples == 7


def test_a_duplicate_poll_is_refused_too():
    subject = tracker()
    feed(subject, FAILING[0:4])
    duplicate = subject.observe(PORT, 3 * MINUTE_NS, FAILING[3], floor_dbm=FLOOR, optic="SN-A")

    assert duplicate.out_of_order is True


def test_ports_do_not_share_a_fit():
    subject = tracker()
    feed(subject, FAILING, entity="hq:sw1:Gi1/0/1")
    other = feed(subject, STEADY, entity="hq:sw1:Gi1/0/2")

    assert other[-1].status == STATUS_NOT_DEGRADING
    assert subject.tracked_entities == 2


def test_entities_are_lru_bounded():
    subject = tracker(max_entities=2)

    for index in range(4):
        subject.observe(f"port-{index}", index * MINUTE_NS, -7.0, floor_dbm=FLOOR)

    assert subject.tracked_entities == 2


def test_dropped_entity_starts_over_rather_than_guessing():
    subject = tracker(max_entities=1)
    feed(subject, FAILING[0:5], entity="port-a")
    subject.observe("port-b", 5 * MINUTE_NS, -7.0, floor_dbm=FLOOR)
    revived = subject.observe("port-a", 6 * MINUTE_NS, FAILING[6], floor_dbm=FLOOR)

    assert revived.status == STATUS_IMMATURE
    assert revived.samples == 1


def test_figures_are_quantized():
    last = feed(tracker(decimals=2), [-7.0 - 0.0137 * index for index in range(10)])[-1]

    for figure in (last.trend_db_per_day, last.residual_db, last.days_to_floor):
        assert figure == round(figure, 2)


def test_replaying_the_stream_reproduces_every_forecast():
    first = feed(tracker(), FAILING + [-10.0] * 3 + NOISY)
    second = feed(tracker(), FAILING + [-10.0] * 3 + NOISY)

    assert first == second


def test_every_status_the_tracker_can_report_is_in_the_domain():
    seen = set()
    subject = tracker()

    seen.update(result.status for result in feed(subject, FAILING))
    seen.add(subject.observe(PORT, 10 * MINUTE_NS, None, floor_dbm=FLOOR, optic="SN-A").status)
    seen.update(result.status for result in feed(subject, STEADY[0:8] + [-10.0] * 4, start=11 * MINUTE_NS))
    seen.update(result.status for result in feed(tracker(), NOISY))
    seen.update(result.status for result in feed(tracker(), FAILING, floor=None))
    seen.update(result.status for result in feed(tracker(), FAILING, floor=-7.5))

    assert seen == set(STATUSES)


def test_constructor_validation():
    with pytest.raises(ValueError):
        tracker(window_ns=0)

    with pytest.raises(ValueError, match="min_samples"):
        tracker(min_samples=2)

    with pytest.raises(ValueError, match="max_samples"):
        tracker(min_samples=10, max_samples=3)

    with pytest.raises(ValueError, match="max_residual_db"):
        tracker(max_residual_db=0)

    with pytest.raises(ValueError, match="min_significance"):
        tracker(min_significance=-1)

    with pytest.raises(ValueError, match="max_entities"):
        tracker(max_entities=0)
