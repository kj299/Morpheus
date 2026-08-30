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

import random

import pytest

from morpheus.utils.window_seal import BatchResult
from morpheus.utils.window_seal import WindowSealer

PERIOD = 100
LATENESS = 30


def build_sealer(**kwargs) -> WindowSealer:
    defaults = {"period_ns": PERIOD, "lateness_ns": LATENESS, "epoch_ns": 0}
    defaults.update(kwargs)

    return WindowSealer(**defaults)


def flatten(result: BatchResult) -> list[tuple]:
    return [(a.window_id, a.late) for a in result.assignments]


def test_window_bounds_are_half_open_and_epoch_anchored():
    sealer = build_sealer(epoch_ns=50)

    assert sealer.window_id(50) == 0
    assert sealer.window_id(149) == 0
    assert sealer.window_id(150) == 1
    assert sealer.window_bounds(0) == (50, 150)
    assert sealer.window_bounds(1) == (150, 250)


def test_windows_are_anchored_to_the_epoch_not_the_first_event():
    # The first observed event lands mid-window; the boundary must not move to meet it.
    sealer = build_sealer()
    result = sealer.observe([170])

    assert flatten(result) == [(1, False)]
    assert sealer.window_bounds(1) == (100, 200)


def test_seal_requires_watermark_past_end_plus_horizon():
    sealer = build_sealer()

    # Window 0 ends at 100 and seals when the watermark reaches 130.
    assert len(sealer.observe([10]).sealed) == 0
    assert len(sealer.observe([129]).sealed) == 0
    assert sealer.observe([130]).sealed == [0]


def test_row_after_seal_is_late():
    sealer = build_sealer()
    sealer.observe([10, 130])

    result = sealer.observe([20])

    assert flatten(result) == [(0, True)]
    assert len(result.sealed) == 0


def test_late_classification_is_against_the_watermark_before_the_row():
    # A single batch where the sealing row precedes the late row: the late row must be judged against the watermark
    # advanced by the earlier row in the same batch, which is what makes the outcome batch-independent.
    sealer = build_sealer()

    result = sealer.observe([10, 130, 20])

    assert flatten(result) == [(0, False), (1, False), (0, True)]
    assert result.sealed == [0]


def test_batching_is_irrelevant():
    times = [10, 50, 120, 130, 20, 260, 250, 400, 5, 410]

    one_batch = build_sealer()
    per_row = build_sealer()

    whole = one_batch.observe(list(times))

    assignments = []
    sealed = []
    for t in times:
        result = per_row.observe([t])
        assignments.extend(flatten(result))
        sealed.extend(result.sealed)

    assert flatten(whole) == assignments
    assert whole.sealed == sealed


def test_out_of_order_within_horizon_is_on_time():
    sealer = build_sealer()

    result = sealer.observe([10, 110, 90])

    # 90 arrives after 110 but window 0 has not sealed (needs watermark 130), so it is on time.
    assert flatten(result) == [(0, False), (1, False), (0, False)]


def test_multiple_windows_seal_in_ascending_order():
    sealer = build_sealer()
    sealer.observe([10, 110])

    # A jump far ahead seals both open windows at once, in window order.
    result = sealer.observe([1000])

    assert result.sealed == [0, 1]


def test_empty_windows_are_not_emitted():
    sealer = build_sealer()

    # Windows 1 through 8 never receive a row; only window 0 seals.
    result = sealer.observe([10, 900])

    assert result.sealed == [0]


def test_watermark_never_regresses():
    sealer = build_sealer()
    sealer.observe([500])

    assert sealer.watermark_ns == 500
    sealer.observe([100])
    assert sealer.watermark_ns == 500


def test_none_time_gets_no_window_and_does_not_advance_the_watermark():
    sealer = build_sealer()

    result = sealer.observe([10, None, 20])

    assert flatten(result) == [(0, False), (None, False), (0, False)]
    assert sealer.watermark_ns == 20


def test_zero_lateness_seals_at_the_boundary():
    sealer = build_sealer(lateness_ns=0)

    assert len(sealer.observe([10, 99]).sealed) == 0
    assert sealer.observe([100]).sealed == [0]


def test_negative_times_use_floor_division():
    sealer = build_sealer()

    result = sealer.observe([-1])

    assert flatten(result) == [(-1, False)]
    assert sealer.window_bounds(-1) == (-100, 0)


def test_flush_returns_open_windows_in_order_and_clears():
    sealer = build_sealer()
    sealer.observe([210, 10, 110])

    # The watermark reached 210 first, so the row at 10 is late and window 0 never opens; 1 and 2 remain open.
    assert sealer.flush() == [1, 2]
    assert sealer.flush() == []
    assert sealer.open_window_ids == []


def test_revisions_start_at_zero_and_increment():
    sealer = build_sealer()

    assert sealer.next_revision(7) == 0
    assert sealer.next_revision(7) == 1
    assert sealer.next_revision(8) == 0
    assert sealer.next_revision(7) == 2


def test_revision_memory_is_bounded():
    sealer = build_sealer(revision_memory=2)

    sealer.next_revision(1)
    sealer.next_revision(2)
    sealer.next_revision(3)

    # Window 1 was evicted, so its numbering restarts.
    assert sealer.next_revision(1) == 0
    # Window 3 was retained.
    assert sealer.next_revision(3) == 1


def test_replay_reproduces_the_same_outcome():
    rng = random.Random(1337)
    times = [rng.randrange(0, 2000) for _ in range(500)]

    first = build_sealer()
    second = build_sealer()

    first_out = [(flatten(r), r.sealed) for r in (first.observe(times[i:i + 37]) for i in range(0, len(times), 37))]
    second_out = [(flatten(r), r.sealed) for r in (second.observe(times[i:i + 37]) for i in range(0, len(times), 37))]

    assert first_out == second_out
    assert first.flush() == second.flush()


def test_constructor_validation():
    with pytest.raises(ValueError):
        WindowSealer(period_ns=0, lateness_ns=0)

    with pytest.raises(ValueError):
        WindowSealer(period_ns=100, lateness_ns=-1)

    with pytest.raises(ValueError):
        WindowSealer(period_ns=100, lateness_ns=0, revision_memory=0)
