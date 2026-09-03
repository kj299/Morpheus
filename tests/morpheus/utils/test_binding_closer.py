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

from morpheus.utils.binding_closer import CONFLICT
from morpheus.utils.binding_closer import DISPLACED
from morpheus.utils.binding_closer import EVICTED
from morpheus.utils.binding_closer import EXPLICIT
from morpheus.utils.binding_closer import IDLE_TIMEOUT
from morpheus.utils.binding_closer import INFERRED_REASONS
from morpheus.utils.binding_closer import MINIMUM_TICK_NS
from morpheus.utils.binding_closer import NS_PER_SECOND
from morpheus.utils.binding_closer import SNAPSHOT_ABSENT
from morpheus.utils.binding_closer import BindingCloser
from morpheus.utils.binding_table import Binding

MINUTE_NS = 60 * NS_PER_SECOND
MAC = "00:11:22:33:44:55"
PORT_A = {"switch_id": "sw1", "port_id": "Gi1/0/1", "vlan_id": "10"}
PORT_B = {"switch_id": "sw1", "port_id": "Gi1/0/2", "vlan_id": "10"}


def closer(**kwargs) -> BindingCloser:
    defaults = {"attribute_names": ["switch_id", "port_id", "vlan_id"]}
    defaults.update(kwargs)

    return BindingCloser(**defaults)


def as_binding(closed) -> Binding:
    """Render a closed binding the way `BindingTable` consumes it, so coverage can be checked directly."""
    return Binding(key=closed.key,
                   start_ns=closed.bind_start_ns,
                   end_ns=closed.bind_end_ns,
                   values=tuple(closed.attributes.values()),
                   uid="test")


def test_a_repeated_observation_extends_rather_than_reopening():
    subject = closer()

    for index in range(4):
        result = subject.observe(MAC, index * MINUTE_NS, PORT_A)
        assert result.closed == []

    assert subject.open_count == 1


def test_displacement_closes_the_old_binding():
    subject = closer()

    subject.observe(MAC, 0, PORT_A)
    subject.observe(MAC, MINUTE_NS, PORT_A)
    result = subject.observe(MAC, 3 * MINUTE_NS, PORT_B)

    assert len(result.closed) == 1
    closed = result.closed[0]

    assert closed.attributes == PORT_A
    assert closed.end_reason == DISPLACED
    assert closed.bind_start_ns == 0
    # Closed at the last observation on the old port, not at the new one: the move happened somewhere between them
    # and claiming the later time would have the binding cover a period the MAC may already have left.
    assert closed.bind_end_ns == MINUTE_NS + MINIMUM_TICK_NS


def test_the_gap_left_by_displacement_resolves_to_nothing():
    subject = closer()

    subject.observe(MAC, 0, PORT_A)
    subject.observe(MAC, MINUTE_NS, PORT_A)
    closed = subject.observe(MAC, 3 * MINUTE_NS, PORT_B).closed[0]
    binding = as_binding(closed)

    # Two minutes after the last sighting on port A, nobody knows where the MAC was. A gap answers "unknown"; a
    # stretched binding would answer "port A", confidently and possibly wrongly.
    assert binding.covers(MINUTE_NS) is True
    assert binding.covers(2 * MINUTE_NS) is False


def test_the_last_observation_falls_inside_its_own_binding():
    # The interval is half-open, so an end placed exactly at the last observation would exclude it.
    subject = closer()

    subject.observe(MAC, 0, PORT_A)
    subject.observe(MAC, MINUTE_NS, PORT_A)
    binding = as_binding(subject.observe(MAC, 2 * MINUTE_NS, PORT_B).closed[0])

    assert binding.covers(MINUTE_NS) is True


def test_a_single_observation_still_covers_its_own_instant():
    # Without the minimum tick this would be a zero-width interval, which resolves nothing at all.
    subject = closer()

    subject.observe(MAC, 5 * MINUTE_NS, PORT_A)
    binding = as_binding(subject.observe(MAC, 6 * MINUTE_NS, PORT_B).closed[0])

    assert binding.duration_ns == MINIMUM_TICK_NS
    assert binding.covers(5 * MINUTE_NS) is True


def test_an_explicit_stop_is_taken_at_its_word():
    subject = closer()

    subject.observe(MAC, 0, PORT_A)
    subject.observe(MAC, MINUTE_NS, PORT_A)
    closed = subject.close(MAC, 10 * MINUTE_NS)

    # The one end that is a fact rather than an inference, so the source's time is used directly.
    assert closed.end_reason == EXPLICIT
    assert closed.end_observed is True
    assert closed.bind_end_ns == 10 * MINUTE_NS


def test_an_explicit_stop_never_excludes_the_last_observation():
    subject = closer()

    subject.observe(MAC, 0, PORT_A)
    subject.observe(MAC, MINUTE_NS, PORT_A)
    # A stop record that arrives stamped at or before the last sighting still has to leave that sighting covered.
    closed = subject.close(MAC, MINUTE_NS)

    assert closed.bind_end_ns == MINUTE_NS + MINIMUM_TICK_NS


def test_an_unmatched_stop_is_not_an_error():
    subject = closer()

    assert subject.close("00:00:00:00:00:00", MINUTE_NS) is None


def test_a_snapshot_closes_what_it_no_longer_lists():
    subject = closer()

    subject.observe(MAC, 0, PORT_A)
    subject.observe("aa:bb:cc:dd:ee:ff", 0, PORT_B)

    closed = subject.reconcile(5 * MINUTE_NS, present_keys=[MAC])

    assert len(closed) == 1
    assert closed[0].key == "aa:bb:cc:dd:ee:ff"
    assert closed[0].end_reason == SNAPSHOT_ABSENT
    assert subject.open_count == 1


def test_a_scoped_snapshot_leaves_other_devices_alone():
    subject = closer()

    subject.observe(MAC, 0, PORT_A)
    subject.observe("aa:bb:cc:dd:ee:ff", 0, {"switch_id": "sw9", "port_id": "Gi1/0/1", "vlan_id": "10"})

    # A snapshot of one switch says nothing about MACs bound elsewhere. Without the scope this call would close
    # every binding in the estate that the single device happened not to list.
    closed = subject.reconcile(5 * MINUTE_NS, present_keys=[], scope={"switch_id": "sw1"})

    assert [record.key for record in closed] == [MAC]
    assert subject.open_count == 1


def test_a_stale_snapshot_does_not_close_a_live_binding():
    # Snapshots arrive late. One taken at t=1 that does not list the MAC says nothing about a sighting at t=5, and
    # acting on it would end a binding that is demonstrably still live.
    subject = closer()

    subject.observe(MAC, 5 * MINUTE_NS, PORT_A)
    closed = subject.reconcile(MINUTE_NS, present_keys=[])

    assert closed == []
    assert subject.open_count == 1


def test_idle_bindings_expire():
    subject = closer(idle_timeout_ns=5 * MINUTE_NS)

    subject.observe(MAC, 0, PORT_A)

    assert subject.expire(3 * MINUTE_NS) == []

    closed = subject.expire(10 * MINUTE_NS)

    assert len(closed) == 1
    assert closed[0].end_reason == IDLE_TIMEOUT
    assert closed[0].bind_end_ns == MINIMUM_TICK_NS


def test_eviction_emits_rather_than_drops():
    # A dropped open binding would be data loss, not merely a stale count: the interval would never reach the
    # table at all.
    subject = closer(max_open=2)

    subject.observe("mac-a", 0, PORT_A)
    subject.observe("mac-b", MINUTE_NS, PORT_A)
    result = subject.observe("mac-c", 2 * MINUTE_NS, PORT_A)

    assert [record.key for record in result.closed] == ["mac-a"]
    assert result.closed[0].end_reason == EVICTED
    assert subject.open_count == 2


def test_drain_closes_everything():
    subject = closer()

    subject.observe("mac-a", 0, PORT_A)
    subject.observe("mac-b", MINUTE_NS, PORT_B)

    drained = subject.drain()

    assert [record.key for record in drained] == ["mac-a", "mac-b"]
    assert subject.open_count == 0


def test_out_of_order_sample_is_flagged_and_ignored():
    subject = closer()

    subject.observe(MAC, 0, PORT_A)
    subject.observe(MAC, 5 * MINUTE_NS, PORT_A)
    late = subject.observe(MAC, MINUTE_NS, PORT_B)

    assert late.out_of_order is True
    assert late.closed == []

    # The late sample must not have displaced the binding, or the emitted interval would depend on delivery order.
    drained = subject.drain()
    assert drained[0].attributes == PORT_A
    assert drained[0].last_seen_ns == 5 * MINUTE_NS


def test_observations_are_counted():
    subject = closer()

    for index in range(3):
        subject.observe(MAC, index * MINUTE_NS, PORT_A)

    assert subject.drain()[0].observations == 3


def test_only_explicit_ends_are_observed():
    # A consumer that trusts only stated ends has one predicate to filter on.
    assert EXPLICIT not in INFERRED_REASONS
    assert {DISPLACED, SNAPSHOT_ABSENT, IDLE_TIMEOUT, EVICTED}.issubset(INFERRED_REASONS)


def test_a_vlan_change_on_the_same_port_is_a_displacement():
    # The binding target is the whole tuple, so moving VLAN without moving port still ends the old binding.
    subject = closer()

    subject.observe(MAC, 0, PORT_A)
    result = subject.observe(MAC, MINUTE_NS, {**PORT_A, "vlan_id": "20"})

    assert len(result.closed) == 1
    assert result.closed[0].attributes["vlan_id"] == "10"


def test_untracked_attributes_do_not_displace():
    # Only the declared target matters; a changing signal strength or counter must not split the binding.
    subject = closer(attribute_names=["switch_id", "port_id"])

    subject.observe(MAC, 0, {**PORT_A, "wireless_rssi": -50})
    result = subject.observe(MAC, MINUTE_NS, {**PORT_A, "wireless_rssi": -80})

    assert result.closed == []


def test_constructor_validation():
    with pytest.raises(ValueError):
        BindingCloser(attribute_names=[])

    with pytest.raises(ValueError):
        closer(idle_timeout_ns=0)

    with pytest.raises(ValueError):
        closer(max_open=0)


def test_a_key_in_two_places_at_one_instant_is_a_conflict_not_a_late_sample():
    # Two switches polled in the same second both report the MAC. That is the spoofing signal, and it used to be
    # discarded as out of order, which turned the strongest evidence layer 2 produces into a data quality warning.
    subject = closer()

    subject.observe(MAC, 0, PORT_A)
    subject.observe(MAC, 5 * MINUTE_NS, PORT_A)
    result = subject.observe(MAC, 5 * MINUTE_NS, PORT_B)

    assert result.out_of_order is False
    assert len(result.closed) == 1
    assert result.closed[0].end_reason == CONFLICT
    assert CONFLICT in INFERRED_REASONS

    # Neither sighting precedes the other, so the two intervals overlap by exactly the minimum tick: the honest
    # record of one key in two places at once, rather than a guess about which came first.
    reopened = subject.drain()[0]
    assert result.closed[0].bind_end_ns == 5 * MINUTE_NS + MINIMUM_TICK_NS
    assert reopened.bind_start_ns == 5 * MINUTE_NS
    assert reopened.attributes == PORT_B


def test_a_repeat_sighting_at_the_same_instant_extends_rather_than_conflicts():
    # The same port reporting the same MAC twice in one snapshot is a duplicate row, not a conflict.
    subject = closer()

    subject.observe(MAC, 0, PORT_A)
    result = subject.observe(MAC, 0, PORT_A)

    assert result.out_of_order is False
    assert result.closed == []
    assert subject.drain()[0].observations == 2


def test_opening_is_reported_once_per_binding_not_per_sample():
    # A consumer emitting provisional records needs to know when a binding opened, and only then.
    subject = closer()

    first = subject.observe(MAC, 0, PORT_A)
    extended = subject.observe(MAC, MINUTE_NS, PORT_A)
    moved = subject.observe(MAC, 2 * MINUTE_NS, PORT_B)

    assert first.opened is True
    assert extended.opened is False
    # A displacement closes one binding and opens another in the same observation.
    assert moved.opened is True and len(moved.closed) == 1


def test_an_open_binding_can_be_read_without_closing_it():
    subject = closer()

    subject.observe(MAC, 0, PORT_A)
    subject.observe(MAC, MINUTE_NS, PORT_A)
    record = subject.open_binding(MAC)

    assert record.attributes == PORT_A
    assert record.bind_start_ns == 0
    assert record.last_seen_ns == MINUTE_NS
    assert record.observations == 2
    # Reading did not end it.
    assert subject.open_count == 1
    assert subject.open_binding("00:00:00:00:00:00") is None
