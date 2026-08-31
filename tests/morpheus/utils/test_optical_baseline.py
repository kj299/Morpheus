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

from morpheus.utils.optical_baseline import NS_PER_SECOND
from morpheus.utils.optical_baseline import OpticalBaselineTracker

MINUTE_NS = 60 * NS_PER_SECOND
RX = "optical_rx_dbm"
TX = "optical_tx_dbm"


def tracker(**kwargs) -> OpticalBaselineTracker:
    defaults = {"channel_names": [TX, RX], "min_samples": 3, "window_ns": 60 * MINUTE_NS}
    defaults.update(kwargs)

    return OpticalBaselineTracker(**defaults)


def feed(subject: OpticalBaselineTracker,
         levels: list,
         entity: str = "hq:sw1:Gi1/0/1",
         start: int = 0,
         step: int = MINUTE_NS):
    """Feed a series of receive levels one step apart, returning every result."""
    return [
        subject.observe(entity, start + index * step, {
            RX: level, TX: -2.0
        }) for (index, level) in enumerate(levels)
    ]


def test_no_baseline_until_min_samples():
    results = feed(tracker(), [-7.0, -7.1, -6.9, -7.0])

    # The first three have too little history to describe anything, so they publish nothing rather than a median
    # of one or two readings dressed up as a reference.
    assert [result.baselines[RX] for result in results[0:3]] == [None, None, None]
    assert [result.deviations[RX] for result in results[0:3]] == [None, None, None]
    assert results[3].baselines[RX] == pytest.approx(-7.0)
    assert results[3].sample_counts[RX] == 3


def test_a_tap_shows_as_a_negative_step():
    # Four steady readings, then a 2 dB drop: a passive splitter diverting light to a tap.
    results = feed(tracker(), [-7.0, -7.0, -7.0, -7.0, -9.0])

    assert results[4].baselines[RX] == pytest.approx(-7.0)
    assert results[4].deviations[RX] == pytest.approx(-2.0)


def test_the_current_reading_is_excluded_from_its_own_baseline():
    # Included, the drop would be damped toward zero by its own weight in the median.
    results = feed(tracker(min_samples=2), [-7.0, -7.0, -9.0])

    assert results[2].baselines[RX] == pytest.approx(-7.0)
    assert results[2].deviations[RX] == pytest.approx(-2.0)


def test_the_median_resists_one_wild_reading():
    # Optical diagnostics occasionally report nonsense. A mean baseline would be dragged down by roughly 6 dB here
    # and would then under-report every later deviation.
    results = feed(tracker(), [-7.0, -30.0, -7.0, -7.0, -7.0])

    assert results[4].baselines[RX] == pytest.approx(-7.0)
    assert results[4].deviations[RX] == pytest.approx(0.0)


def test_readings_outside_the_window_are_dropped():
    subject = tracker(min_samples=1, window_ns=5 * MINUTE_NS)

    feed(subject, [-20.0, -20.0], start=0)
    # Ten minutes later, both earlier readings are outside the window and cannot anchor the baseline.
    late = subject.observe("hq:sw1:Gi1/0/1", 10 * MINUTE_NS, {RX: -7.0, TX: -2.0})

    assert late.baselines[RX] is None
    assert late.sample_counts[RX] == 0


def test_the_baseline_follows_the_link():
    # The honest limit. Once the window has rolled past every pre-step reading, the median describes the new level
    # and the deviation goes back to zero, so the step is a transient signal rather than a standing state.
    subject = tracker(min_samples=3, window_ns=5 * MINUTE_NS)

    stepped = feed(subject, [-7.0, -7.0, -7.0, -9.0], start=0)
    settled = feed(subject, [-9.0, -9.0, -9.0, -9.0], start=10 * MINUTE_NS)

    # Caught at the step, against a baseline still made of pre-step readings.
    assert stepped[-1].deviations[RX] == pytest.approx(-2.0)
    # Seven minutes later every pre-step reading has aged out and the same 2 dB loss reads as normal.
    assert settled[-1].deviations[RX] == pytest.approx(0.0)


def test_transmit_and_receive_are_independent():
    subject = tracker(min_samples=2)

    for index in range(3):
        subject.observe("hq:sw1:Gi1/0/1", index * MINUTE_NS, {TX: -2.0, RX: -7.0})

    # The local laser fades while the path is unchanged, which is a different fault from losing light in the fibre.
    result = subject.observe("hq:sw1:Gi1/0/1", 3 * MINUTE_NS, {TX: -5.0, RX: -7.0})

    assert result.deviations[TX] == pytest.approx(-3.0)
    assert result.deviations[RX] == pytest.approx(0.0)


def test_an_empty_cage_contributes_nothing():
    subject = tracker(min_samples=2)

    for index in range(3):
        subject.observe("hq:sw1:Gi1/0/1", index * MINUTE_NS, {TX: -2.0, RX: None})

    result = subject.observe("hq:sw1:Gi1/0/1", 3 * MINUTE_NS, {TX: -2.0, RX: None})

    # A port with no optic reports no level. Treating that as a reading would baseline the port against nothing.
    assert result.baselines[RX] is None
    assert result.deviations[RX] is None
    assert result.deviations[TX] == pytest.approx(0.0)


def test_ports_do_not_share_a_baseline():
    subject = tracker(min_samples=2)

    feed(subject, [-7.0, -7.0, -7.0], entity="hq:sw1:Gi1/0/1")
    feed(subject, [-20.0, -20.0, -20.0], entity="hq:sw1:Gi1/0/2")

    quiet = subject.observe("hq:sw1:Gi1/0/1", 9 * MINUTE_NS, {RX: -7.0, TX: -2.0})
    dark = subject.observe("hq:sw1:Gi1/0/2", 9 * MINUTE_NS, {RX: -20.0, TX: -2.0})

    # A long link legitimately sits 13 dB below a short one. Sharing a baseline would alarm on both forever.
    assert quiet.deviations[RX] == pytest.approx(0.0)
    assert dark.deviations[RX] == pytest.approx(0.0)


def test_out_of_order_sample_is_flagged_and_ignored():
    subject = tracker(min_samples=2)

    feed(subject, [-7.0, -7.0, -7.0])
    late = subject.observe("hq:sw1:Gi1/0/1", MINUTE_NS, {RX: -30.0, TX: -2.0})
    following = subject.observe("hq:sw1:Gi1/0/1", 5 * MINUTE_NS, {RX: -7.0, TX: -2.0})

    assert late.out_of_order is True
    assert late.deviations[RX] is None
    # The late reading must not have entered the history, or it would drag every later baseline.
    assert following.baselines[RX] == pytest.approx(-7.0)


def test_entities_are_lru_bounded():
    subject = tracker(min_samples=1, max_entities=2)

    for index in range(4):
        subject.observe(f"port-{index}", index * MINUTE_NS, {RX: -7.0, TX: -2.0})

    assert subject.tracked_entities == 2


def test_dropped_entity_starts_over_rather_than_guessing():
    subject = tracker(min_samples=2, max_entities=1)

    subject.observe("port-a", 0, {RX: -7.0, TX: -2.0})
    subject.observe("port-a", MINUTE_NS, {RX: -7.0, TX: -2.0})
    subject.observe("port-b", 2 * MINUTE_NS, {RX: -7.0, TX: -2.0})
    revived = subject.observe("port-a", 3 * MINUTE_NS, {RX: -7.0, TX: -2.0})

    assert revived.baselines[RX] is None


def test_samples_are_capped_per_channel():
    subject = tracker(min_samples=1, max_samples=3, window_ns=10**18)

    results = feed(subject, [-7.0] * 10)

    assert results[-1].sample_counts[RX] == 3


def test_constructor_validation():
    with pytest.raises(ValueError):
        OpticalBaselineTracker(channel_names=[])

    with pytest.raises(ValueError):
        tracker(window_ns=0)

    with pytest.raises(ValueError):
        tracker(min_samples=0)

    with pytest.raises(ValueError, match="max_samples"):
        tracker(min_samples=10, max_samples=3)

    with pytest.raises(ValueError):
        tracker(max_entities=0)
