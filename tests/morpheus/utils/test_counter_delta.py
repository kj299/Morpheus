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

from morpheus.utils.counter_delta import NS_PER_SECOND
from morpheus.utils.counter_delta import CounterTracker

MINUTE_NS = 60 * NS_PER_SECOND
COUNTER32_CEILING = 1 << 32

PORT = "hq:sw1:Gi1/0/1"


def build_tracker(**kwargs) -> CounterTracker:
    defaults = {"counter_names": ["crc_errors", "input_discards"]}
    defaults.update(kwargs)

    return CounterTracker(**defaults)


def test_first_sample_yields_no_deltas():
    # There is nothing to subtract from. Emitting zero here would look like a quiet interface.
    result = build_tracker().observe(PORT, 0, {"crc_errors": 100, "input_discards": 5})

    assert result.deltas == {"crc_errors": None, "input_discards": None}
    assert result.interval_ns is None
    assert not result.counter_reset
    assert not result.out_of_order


def test_steady_increase():
    tracker = build_tracker()
    tracker.observe(PORT, 0, {"crc_errors": 100, "input_discards": 5}, uptime_ns=10 * MINUTE_NS)
    result = tracker.observe(PORT, MINUTE_NS, {"crc_errors": 142, "input_discards": 5}, uptime_ns=11 * MINUTE_NS)

    assert result.deltas == {"crc_errors": 42, "input_discards": 0}
    assert result.interval_ns == MINUTE_NS
    assert not result.counter_reset
    assert not result.counter_wrapped


def test_entities_are_tracked_independently():
    tracker = build_tracker()
    other = "hq:sw1:Gi1/0/2"

    tracker.observe(PORT, 0, {"crc_errors": 100, "input_discards": 0})
    tracker.observe(other, 0, {"crc_errors": 900, "input_discards": 0})

    assert tracker.observe(PORT, MINUTE_NS, {"crc_errors": 110, "input_discards": 0}).deltas["crc_errors"] == 10
    assert tracker.observe(other, MINUTE_NS, {"crc_errors": 901, "input_discards": 0}).deltas["crc_errors"] == 1


def test_wrap_is_corrected_when_uptime_shows_no_reboot():
    # A 32-bit counter near its ceiling rolls over. Uptime kept advancing, so this is a wrap, not a reboot.
    tracker = build_tracker(counter_bits={"crc_errors": 32, "input_discards": 64})
    tracker.observe(PORT, 0, {"crc_errors": COUNTER32_CEILING - 10, "input_discards": 0}, uptime_ns=100 * MINUTE_NS)

    result = tracker.observe(PORT, MINUTE_NS, {"crc_errors": 5, "input_discards": 0}, uptime_ns=101 * MINUTE_NS)

    assert result.deltas["crc_errors"] == 15
    assert result.counter_wrapped
    assert not result.counter_reset


def test_reboot_is_a_reset_not_a_wrap():
    # The same apparent decrease, but uptime went backwards. Treating it as a wrap would invent four billion errors.
    tracker = build_tracker(counter_bits=32)
    tracker.observe(PORT, 0, {"crc_errors": COUNTER32_CEILING - 10, "input_discards": 0}, uptime_ns=100 * MINUTE_NS)

    result = tracker.observe(PORT, MINUTE_NS, {"crc_errors": 5, "input_discards": 0}, uptime_ns=MINUTE_NS // 2)

    assert result.counter_reset
    assert not result.counter_wrapped
    # Everything present now accumulated since the restart.
    assert result.deltas["crc_errors"] == 5


def test_reboot_within_the_interval_is_caught_even_when_uptime_advanced():
    # Uptime is larger than last time, so a plain comparison misses it, but it is smaller than the sampling gap,
    # which means the device came up during that gap.
    tracker = build_tracker()
    tracker.observe(PORT, 0, {"crc_errors": 100, "input_discards": 0}, uptime_ns=30 * NS_PER_SECOND)

    result = tracker.observe(PORT, 10 * MINUTE_NS, {"crc_errors": 7, "input_discards": 0}, uptime_ns=2 * MINUTE_NS)

    assert result.counter_reset
    assert result.deltas["crc_errors"] == 7


def test_reset_interval_is_capped_at_uptime():
    # The counter only accumulated since the device came up, so a rate must divide by that, not by the poll gap.
    tracker = build_tracker()
    tracker.observe(PORT, 0, {"crc_errors": 100, "input_discards": 0}, uptime_ns=50 * MINUTE_NS)

    result = tracker.observe(PORT, 10 * MINUTE_NS, {"crc_errors": 60, "input_discards": 0}, uptime_ns=2 * MINUTE_NS)

    assert result.interval_ns == 2 * MINUTE_NS


def test_decrease_without_uptime_is_flagged_and_never_guessed():
    # Wrap and reboot are indistinguishable here. Both candidate answers are fabrications, so neither is offered.
    tracker = build_tracker()
    tracker.observe(PORT, 0, {"crc_errors": 4000, "input_discards": 0})

    result = tracker.observe(PORT, MINUTE_NS, {"crc_errors": 5, "input_discards": 0})

    assert result.counter_reset
    assert result.deltas["crc_errors"] is None
    assert result.deltas["input_discards"] is None


def test_out_of_order_sample_yields_nothing_and_preserves_state():
    tracker = build_tracker()
    tracker.observe(PORT, 0, {"crc_errors": 100, "input_discards": 0})
    tracker.observe(PORT, 2 * MINUTE_NS, {"crc_errors": 200, "input_discards": 0})

    late = tracker.observe(PORT, MINUTE_NS, {"crc_errors": 150, "input_discards": 0})

    assert late.out_of_order
    assert late.deltas["crc_errors"] is None

    # State still reflects the later sample, so the next in-order delta spans from it rather than backwards.
    following = tracker.observe(PORT, 3 * MINUTE_NS, {"crc_errors": 260, "input_discards": 0})

    assert following.deltas["crc_errors"] == 60


def test_repeated_timestamp_is_out_of_order():
    tracker = build_tracker()
    tracker.observe(PORT, MINUTE_NS, {"crc_errors": 100, "input_discards": 0})

    assert tracker.observe(PORT, MINUTE_NS, {"crc_errors": 105, "input_discards": 0}).out_of_order


def test_missing_counter_affects_only_itself():
    tracker = build_tracker()
    tracker.observe(PORT, 0, {"crc_errors": 100, "input_discards": 5})

    result = tracker.observe(PORT, MINUTE_NS, {"crc_errors": 110})

    assert result.deltas["crc_errors"] == 10
    assert result.deltas["input_discards"] is None


def test_counter_returning_after_absence_yields_no_delta():
    # The gap means the counter may have moved arbitrarily; the next sample after it re-establishes a baseline.
    tracker = build_tracker()
    tracker.observe(PORT, 0, {"crc_errors": 100, "input_discards": 5})
    tracker.observe(PORT, MINUTE_NS, {"crc_errors": 110})

    assert tracker.observe(PORT, 2 * MINUTE_NS, {
        "crc_errors": 120, "input_discards": 9
    }).deltas["input_discards"] is None


def test_replay_reproduces_the_same_deltas():
    samples = [(0, 100, 30), (MINUTE_NS, 150, 31), (2 * MINUTE_NS, 40, 1), (3 * MINUTE_NS, 90, 2)]

    def run() -> list:
        tracker = build_tracker()

        return [
            tracker.observe(PORT, t, {
                "crc_errors": c, "input_discards": 0
            }, uptime_ns=u * MINUTE_NS) for (t, c, u) in samples
        ]

    assert run() == run()


def test_entity_state_is_bounded():
    tracker = build_tracker(max_entities=2)

    for port in ("a", "b", "c"):
        tracker.observe(port, 0, {"crc_errors": 100, "input_discards": 0})

    assert tracker.tracked_entities == 2
    # "a" was evicted, so its next sample is a first observation rather than a wrong delta.
    assert tracker.observe("a", MINUTE_NS, {"crc_errors": 500, "input_discards": 0}).deltas["crc_errors"] is None


def test_recently_seen_entities_survive_eviction():
    tracker = build_tracker(max_entities=2)
    tracker.observe("a", 0, {"crc_errors": 100, "input_discards": 0})
    tracker.observe("b", 0, {"crc_errors": 100, "input_discards": 0})
    tracker.observe("a", MINUTE_NS, {"crc_errors": 110, "input_discards": 0})
    tracker.observe("c", MINUTE_NS, {"crc_errors": 100, "input_discards": 0})

    # "b" was least recently seen and went; "a" was refreshed and stayed.
    assert tracker.observe("a", 2 * MINUTE_NS, {"crc_errors": 120, "input_discards": 0}).deltas["crc_errors"] == 10


def test_constructor_validation():
    with pytest.raises(ValueError):
        CounterTracker([])

    with pytest.raises(ValueError):
        CounterTracker(["a"], max_entities=0)

    with pytest.raises(ValueError):
        CounterTracker(["a"], counter_bits=0)
