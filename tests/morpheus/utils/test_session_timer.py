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

from morpheus.utils.session_timer import NS_PER_SECOND
from morpheus.utils.session_timer import SessionTimer

PORT = "hq:sw1:Gi1/0/1"


def test_a_paired_exchange_is_timed():
    subject = SessionTimer()

    subject.begin(PORT, 0)
    timing = subject.complete(PORT, 3 * NS_PER_SECOND, outcome="success")

    assert timing.elapsed_ns == 3 * NS_PER_SECOND
    assert timing.outcome == "success"
    assert timing.attempts == 1
    assert timing.unpaired is False


def test_an_unpaired_outcome_is_the_signal_not_a_gap():
    # Authorization with no authentication in front of it is what a bypass looks like from the switch. Dropping it
    # would make the feature silent on exactly the condition it exists to detect.
    subject = SessionTimer()

    timing = subject.complete(PORT, NS_PER_SECOND, outcome="success")

    assert timing.unpaired is True
    assert timing.elapsed_ns is None
    assert timing.attempts == 0


def test_retries_are_counted_and_the_clock_restarts():
    subject = SessionTimer()

    subject.begin(PORT, 0)
    subject.begin(PORT, 5 * NS_PER_SECOND)
    subject.begin(PORT, 8 * NS_PER_SECOND)
    timing = subject.complete(PORT, 9 * NS_PER_SECOND, outcome="success")

    # Timed from the last attempt, but the count carries: a success after three tries is not a first-time one.
    assert timing.elapsed_ns == NS_PER_SECOND
    assert timing.attempts == 3


def test_completing_twice_leaves_the_second_unpaired():
    subject = SessionTimer()

    subject.begin(PORT, 0)
    subject.complete(PORT, NS_PER_SECOND)
    second = subject.complete(PORT, 2 * NS_PER_SECOND)

    assert second.unpaired is True


def test_ports_are_paired_separately():
    subject = SessionTimer()

    subject.begin("hq:sw1:Gi1/0/1", 0)
    subject.begin("hq:sw1:Gi1/0/2", 5 * NS_PER_SECOND)

    assert subject.complete("hq:sw1:Gi1/0/1", 2 * NS_PER_SECOND).elapsed_ns == 2 * NS_PER_SECOND
    assert subject.complete("hq:sw1:Gi1/0/2", 6 * NS_PER_SECOND).elapsed_ns == NS_PER_SECOND


def test_an_outcome_before_its_begin_is_flagged_not_negated():
    subject = SessionTimer()

    subject.begin(PORT, 10 * NS_PER_SECOND)
    timing = subject.complete(PORT, 2 * NS_PER_SECOND)

    # A naive subtraction would report an authorization that took minus eight seconds.
    assert timing.out_of_order is True
    assert timing.elapsed_ns is None


def test_stale_exchanges_expire():
    subject = SessionTimer(timeout_ns=60 * NS_PER_SECOND)

    subject.begin(PORT, 0)

    assert subject.expire(30 * NS_PER_SECOND) == []
    assert subject.expire(120 * NS_PER_SECOND) == [PORT]
    assert subject.pending_count == 0


def test_an_expired_exchange_makes_its_outcome_unpaired():
    # The honest consequence: a result that arrives after the exchange was abandoned reports as authorization with
    # nothing in front of it, which over-reports the bypass signal rather than hiding it.
    subject = SessionTimer(timeout_ns=60 * NS_PER_SECOND)

    subject.begin(PORT, 0)
    subject.expire(120 * NS_PER_SECOND)

    assert subject.complete(PORT, 130 * NS_PER_SECOND).unpaired is True


def test_pending_exchanges_are_bounded():
    subject = SessionTimer(max_pending=2)

    for index in range(4):
        subject.begin(f"port-{index}", index * NS_PER_SECOND)

    assert subject.pending_count == 2


def test_constructor_validation():
    with pytest.raises(ValueError):
        SessionTimer(timeout_ns=0)

    with pytest.raises(ValueError):
        SessionTimer(max_pending=0)


def test_expiring_again_at_one_horizon_finds_nothing_and_expiring_later_still_works():
    # The same per-row scan the binding closer pays, for the same reason: `expire` runs once per row on that row's
    # own event time. Skipping a horizon that has not advanced is exact because every exchange begun since the last
    # call was stamped at or after it, so a second call has nothing left to abandon.
    timer = SessionTimer(timeout_ns=10 * NS_PER_SECOND)

    timer.begin("hq:sw1:Gi1/0/1", 0)
    now = 60 * NS_PER_SECOND

    assert timer.expire(now) == ["hq:sw1:Gi1/0/1"]
    assert timer.expire(now) == []

    timer.begin("hq:sw1:Gi1/0/2", now)

    assert timer.expire(now + 5 * NS_PER_SECOND) == []
    assert timer.expire(now + 60 * NS_PER_SECOND) == ["hq:sw1:Gi1/0/2"]
