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

from morpheus.utils.link_flap import NS_PER_SECOND
from morpheus.utils.link_flap import LinkFlapTracker

MINUTE_NS = 60 * NS_PER_SECOND
PORT = "hq:sw1:Gi1/0/1"


def test_first_sample_has_no_count():
    subject = LinkFlapTracker()

    result = subject.observe(PORT, 0, "up", last_change_ns=0)

    # Nothing to compare against, so no count rather than a zero that would read as "verified stable".
    assert result.flaps is None
    assert result.flaps_in_window is None


def test_a_stable_port_never_counts():
    subject = LinkFlapTracker()

    subject.observe(PORT, 0, "up", last_change_ns=0)
    result = subject.observe(PORT, MINUTE_NS, "up", last_change_ns=0)

    assert result.flaps == 0
    assert result.flaps_in_window == 0
    assert result.last_change_advanced is False


def test_a_visible_transition_counts_once():
    subject = LinkFlapTracker()

    subject.observe(PORT, 0, "up", last_change_ns=0)
    result = subject.observe(PORT, MINUTE_NS, "down", last_change_ns=30 * NS_PER_SECOND)

    assert result.flaps == 1
    assert result.last_change_advanced is True


def test_a_flap_inside_the_polling_gap_is_counted():
    # The case that makes this more than a status comparison. The port dropped and recovered between two polls, so
    # both polls saw "up" and a naive diff would report a perfectly stable port.
    subject = LinkFlapTracker()

    subject.observe(PORT, 0, "up", last_change_ns=0)
    result = subject.observe(PORT, MINUTE_NS, "up", last_change_ns=30 * NS_PER_SECOND)

    assert result.flaps == 2
    assert result.last_change_advanced is True


def test_without_last_change_only_visible_transitions_are_seen():
    # The honest degradation. A device that does not report the field hides every sub-poll flap, and the tracker
    # reports what it can see rather than inferring what it cannot.
    subject = LinkFlapTracker()

    subject.observe(PORT, 0, "up")
    hidden = subject.observe(PORT, MINUTE_NS, "up")
    visible = subject.observe(PORT, 2 * MINUTE_NS, "down")

    assert hidden.flaps == 0
    assert visible.flaps == 1
    assert visible.last_change_advanced is False


def test_a_device_reboot_counts_as_a_transition():
    subject = LinkFlapTracker()

    subject.observe(PORT, 0, "up", last_change_ns=10**12)
    # The device restarted, so its notion of when the interface last changed began again from a smaller value.
    result = subject.observe(PORT, MINUTE_NS, "up", last_change_ns=5 * NS_PER_SECOND)

    # The link went down with the device and came back with it, which is two transitions, and the cause is
    # labelled so a rule can exclude a planned reboot rather than having it silently folded into flap counts.
    assert result.flaps == 2
    assert result.device_reset is True


def test_a_device_that_contradicts_itself_is_flagged():
    subject = LinkFlapTracker()

    subject.observe(PORT, 0, "up", last_change_ns=1000)
    result = subject.observe(PORT, MINUTE_NS, "down", last_change_ns=1000)

    # The status moved while the device claimed nothing changed. The observed transition is still counted.
    assert result.flaps == 1
    assert result.last_change_inconsistent is True


def test_the_window_sums_and_then_expires():
    subject = LinkFlapTracker(window_ns=5 * MINUTE_NS)

    subject.observe(PORT, 0, "up", last_change_ns=0)
    subject.observe(PORT, MINUTE_NS, "down", last_change_ns=MINUTE_NS)
    accumulated = subject.observe(PORT, 2 * MINUTE_NS, "up", last_change_ns=2 * MINUTE_NS)

    assert accumulated.flaps_in_window == 2

    # Twenty minutes later both transitions are outside the window and the port reads as quiet again.
    expired = subject.observe(PORT, 22 * MINUTE_NS, "up", last_change_ns=2 * MINUTE_NS)

    assert expired.flaps == 0
    assert expired.flaps_in_window == 0


def test_ports_are_counted_separately():
    subject = LinkFlapTracker()

    for port in (PORT, "hq:sw1:Gi1/0/2"):
        subject.observe(port, 0, "up", last_change_ns=0)

    flapping = subject.observe(PORT, MINUTE_NS, "down", last_change_ns=MINUTE_NS)
    quiet = subject.observe("hq:sw1:Gi1/0/2", MINUTE_NS, "up", last_change_ns=0)

    assert flapping.flaps_in_window == 1
    assert quiet.flaps_in_window == 0


def test_out_of_order_sample_is_flagged_and_ignored():
    subject = LinkFlapTracker()

    subject.observe(PORT, 0, "up", last_change_ns=0)
    subject.observe(PORT, 2 * MINUTE_NS, "up", last_change_ns=0)
    late = subject.observe(PORT, MINUTE_NS, "down", last_change_ns=MINUTE_NS)
    following = subject.observe(PORT, 3 * MINUTE_NS, "up", last_change_ns=0)

    assert late.out_of_order is True
    assert late.flaps is None
    # The late sample must not have become the state the next one is compared against.
    assert following.flaps == 0


def test_a_missing_status_does_not_invent_a_transition():
    subject = LinkFlapTracker()

    subject.observe(PORT, 0, "up", last_change_ns=0)
    result = subject.observe(PORT, MINUTE_NS, None, last_change_ns=0)

    assert result.flaps == 0


def test_entities_are_lru_bounded():
    subject = LinkFlapTracker(max_entities=2)

    for index in range(4):
        subject.observe(f"port-{index}", 0, "up", last_change_ns=0)

    assert subject.tracked_entities == 2


def test_dropped_entity_starts_over_rather_than_guessing():
    subject = LinkFlapTracker(max_entities=1)

    subject.observe("port-a", 0, "up", last_change_ns=0)
    subject.observe("port-b", MINUTE_NS, "up", last_change_ns=0)
    revived = subject.observe("port-a", 2 * MINUTE_NS, "down", last_change_ns=2 * MINUTE_NS)

    assert revived.flaps is None


def test_events_are_capped():
    subject = LinkFlapTracker(window_ns=10**18, max_events=3)

    subject.observe(PORT, 0, "up", last_change_ns=0)

    for index in range(1, 11):
        subject.observe(PORT, index * MINUTE_NS, "up", last_change_ns=index * MINUTE_NS)

    result = subject.observe(PORT, 11 * MINUTE_NS, "up", last_change_ns=11 * MINUTE_NS)

    # Three retained records of two transitions each, rather than an unbounded deque on a hard-flapping port.
    assert result.flaps_in_window == 6


def test_constructor_validation():
    with pytest.raises(ValueError):
        LinkFlapTracker(window_ns=0)

    with pytest.raises(ValueError):
        LinkFlapTracker(max_events=0)

    with pytest.raises(ValueError):
        LinkFlapTracker(max_entities=0)
