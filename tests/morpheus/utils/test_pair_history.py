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

from morpheus.utils.pair_history import DAY_NS
from morpheus.utils.pair_history import PairHistoryTracker

HOST = "ws-01"
SHELL = ("explorer.exe", "cmd.exe")
BROWSER = ("explorer.exe", "chrome.exe")
START = 10**18


def tracker(**kwargs) -> PairHistoryTracker:
    defaults = {"window_ns": 30 * DAY_NS, "warmup_ns": 7 * DAY_NS}
    defaults.update(kwargs)

    return PairHistoryTracker(**defaults)


def test_a_pair_seen_before_is_seen():
    subject = tracker()
    first = subject.observe(HOST, START, SHELL)
    again = subject.observe(HOST, START + DAY_NS, SHELL)

    assert first.seen is False
    assert again.seen is True


def test_a_pair_last_seen_outside_the_window_is_not_seen():
    subject = tracker()
    subject.observe(HOST, START, SHELL)

    inside = subject.observe(HOST, START + 30 * DAY_NS, BROWSER)
    stale = subject.observe(HOST, START + 31 * DAY_NS, SHELL)

    assert inside.seen is False
    assert stale.seen is False
    # The window is inclusive at its edge: thirty days ago is still "in thirty days".
    edge = tracker()
    edge.observe(HOST, START, SHELL)
    assert edge.observe(HOST, START + 30 * DAY_NS, SHELL).seen is True


def test_no_entity_is_mature_before_the_warmup():
    subject = tracker()

    assert subject.observe(HOST, START, SHELL).mature is False
    assert subject.observe(HOST, START + 7 * DAY_NS - 1, SHELL).mature is False
    assert subject.observe(HOST, START + 7 * DAY_NS, BROWSER).mature is True


def test_pairs_at_one_instant_are_not_prior_to_each_other_whatever_their_order():
    # The same pair twice in one second: neither sighting is history for the other.
    for order in ((SHELL, SHELL, BROWSER), (BROWSER, SHELL, SHELL)):
        subject = tracker()
        results = [subject.observe(HOST, START, pair) for pair in order]

        assert all(result.seen is False for result in results)


def test_a_second_sighting_at_one_instant_keeps_the_history_before_it():
    subject = tracker()
    subject.observe(HOST, START, SHELL)

    first = subject.observe(HOST, START + DAY_NS, SHELL)
    second = subject.observe(HOST, START + DAY_NS, SHELL)

    assert first.seen is True
    assert second.seen is True


def test_an_out_of_order_sighting_is_refused_and_teaches_nothing():
    subject = tracker()
    subject.observe(HOST, START + DAY_NS, SHELL)

    late = subject.observe(HOST, START, BROWSER)
    after = subject.observe(HOST, START + 2 * DAY_NS, BROWSER)

    assert late.out_of_order is True
    assert late.seen is None
    assert after.seen is False


def test_entities_keep_separate_histories():
    subject = tracker()
    subject.observe("ws-01", START, SHELL)

    assert subject.observe("ws-02", START + DAY_NS, SHELL).seen is False


def test_the_pair_bound_evicts_the_least_recently_seen_and_says_so():
    subject = tracker(max_pairs=2)
    subject.observe(HOST, START, ("a", "1"))
    subject.observe(HOST, START + 1, ("a", "2"))
    subject.observe(HOST, START + 2, ("a", "1"))
    crowded = subject.observe(HOST, START + 3, ("a", "3"))

    assert crowded.saturated is True
    # The recently repeated pair survived; the one not seen since was forgotten, and returns as novel.
    assert subject.observe(HOST, START + 4, ("a", "1")).seen is True
    assert subject.observe(HOST, START + 5, ("a", "2")).seen is False


def test_the_entity_bound_forgets_the_least_recently_seen_entity():
    subject = tracker(max_entities=2)

    for (index, host) in enumerate(("a", "b", "c")):
        subject.observe(host, START + index, SHELL)

    assert subject.tracked_entities == 2
    assert subject.observe("a", START + 10, SHELL).seen is False


def test_constructor_and_arguments_are_validated():
    with pytest.raises(ValueError):
        PairHistoryTracker(window_ns=0)

    with pytest.raises(ValueError):
        PairHistoryTracker(warmup_ns=-1)

    with pytest.raises(ValueError):
        PairHistoryTracker(max_pairs=0)

    with pytest.raises(ValueError):
        PairHistoryTracker(max_entities=0)

    with pytest.raises(ValueError):
        tracker().observe("", START, SHELL)

    with pytest.raises(ValueError):
        tracker().observe(HOST, float(START), SHELL)
