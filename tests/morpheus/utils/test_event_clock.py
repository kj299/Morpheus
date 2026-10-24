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

import logging

import pandas as pd
import pytest

from morpheus.utils.event_clock import CLOCK_REFUSED_COLUMN
from morpheus.utils.event_clock import DEFAULT_MAX_SKEW_SECONDS
from morpheus.utils.event_clock import NS_PER_SECOND
from morpheus.utils.event_clock import EventClock
from morpheus.utils.event_clock import clock_for_skew_seconds
from morpheus.utils.event_clock import event_times_ns
from morpheus.utils.event_clock import refuse_rows

HOUR_NS = 3600 * NS_PER_SECOND
YEAR_NS = 365 * 24 * HOUR_NS


def clock(skew_seconds: int = 3600) -> EventClock:
    return EventClock(max_skew_ns=skew_seconds * NS_PER_SECOND)


def test_the_first_time_establishes_the_clock():
    subject = clock()

    assert subject.value_ns is None
    assert subject.accept(1000 * NS_PER_SECOND) is True
    assert subject.value_ns == 1000 * NS_PER_SECOND


def test_a_time_inside_the_skew_is_believed_and_advances_the_clock():
    subject = clock(skew_seconds=3600)
    subject.accept(1000 * NS_PER_SECOND)

    assert subject.accept(1000 * NS_PER_SECOND + HOUR_NS) is True
    assert subject.value_ns == 1000 * NS_PER_SECOND + HOUR_NS


def test_a_time_beyond_the_skew_is_refused_and_leaves_the_clock_alone():
    # The defect this exists for: a device whose clock is wrong by years must not drag the stream's clock with it,
    # because every stateful stage expires its own state against that clock.
    subject = clock(skew_seconds=3600)
    subject.accept(1000 * NS_PER_SECOND)

    assert subject.accept(1000 * NS_PER_SECOND + 10 * YEAR_NS) is False
    assert subject.value_ns == 1000 * NS_PER_SECOND


def test_refusing_a_time_does_not_wedge_the_clock():
    # The second half of the defect. The stages skip an expiry whose horizon has not advanced, so a clock left
    # stranded in the far future would silently disable expiry for the rest of the run.
    subject = clock(skew_seconds=3600)
    subject.accept(1000 * NS_PER_SECOND)
    subject.accept(1000 * NS_PER_SECOND + 10 * YEAR_NS)

    assert subject.accept(1000 * NS_PER_SECOND + 60 * NS_PER_SECOND) is True
    assert subject.value_ns == 1060 * NS_PER_SECOND


def test_an_earlier_time_is_believed_but_does_not_move_the_clock_backwards():
    # A late row is ordinary; it is the stages' own out-of-order guards that decide what to do with it. The clock
    # only ever tracks the furthest the stream has got.
    subject = clock()
    subject.accept(1000 * NS_PER_SECOND)

    assert subject.accept(500 * NS_PER_SECOND) is True
    assert subject.value_ns == 1000 * NS_PER_SECOND


def test_the_boundary_is_inclusive():
    subject = clock(skew_seconds=3600)
    subject.accept(0)

    assert subject.accept(HOUR_NS) is True, "exactly at the skew is still believable"

    subject = clock(skew_seconds=3600)
    subject.accept(0)

    assert subject.accept(HOUR_NS + 1) is False, "one nanosecond past it is not"


def test_the_clock_is_a_function_of_the_sequence_not_of_batching():
    # Determinism control 13 checks that dividing the stream differently does not change the output. The clock is
    # fed row by row in stream order, so the same rows must leave it in the same place however they were grouped.
    times = [t * NS_PER_SECOND for t in (0, 30, 10, 90, 60)]

    whole = clock()
    for value in times:
        whole.accept(value)

    split = clock()
    for value in times[:2]:
        split.accept(value)
    for value in times[2:]:
        split.accept(value)

    assert whole.value_ns == split.value_ns == 90 * NS_PER_SECOND


def test_the_default_admits_an_operational_gap_and_refuses_a_broken_clock():
    # The default is a week: long enough that a pipeline stopped over a weekend, or a replay of an archive with
    # holes in it, is ordinary; short enough that a year is not.
    subject = EventClock()
    subject.accept(0)

    assert DEFAULT_MAX_SKEW_SECONDS == 7 * 24 * 3600
    assert subject.accept(3 * 24 * HOUR_NS) is True, "a long weekend is not a broken clock"
    assert subject.accept(YEAR_NS) is False, "a year is"


@pytest.mark.parametrize("skew", [0, -1])
def test_a_non_positive_skew_is_refused(skew: int):
    with pytest.raises(ValueError):
        EventClock(max_skew_ns=skew)


# --- The stage-side helpers -------------------------------------------------------------------------------------


def test_the_stage_parameter_builds_the_clock_and_names_itself_when_refused():
    subject = clock_for_skew_seconds(2)
    subject.accept(0)

    assert subject.accept(2 * NS_PER_SECOND) is True
    assert subject.accept(4 * NS_PER_SECOND + 1) is False

    with pytest.raises(ValueError, match="max_clock_skew_seconds"):
        clock_for_skew_seconds(0)


def test_event_times_are_read_per_row_with_none_for_a_null_or_an_unreadable_value():
    df = pd.DataFrame({"when": ["1970-01-01T00:00:10Z", None, "not a time", float("nan"), "1970-01-01T00:00:20Z"]})

    assert event_times_ns(df, "when") == [10 * NS_PER_SECOND, None, None, None, 20 * NS_PER_SECOND]


def test_numeric_event_times_are_scaled_by_the_unit():
    df = pd.DataFrame({"when": [10, 20]})

    assert event_times_ns(df, "when", "s") == [10 * NS_PER_SECOND, 20 * NS_PER_SECOND]
    assert event_times_ns(df, "when", "ms") == [10 * 10**6, 20 * 10**6]


def test_refusing_rows_marks_every_frame_and_warns_only_when_a_row_was_refused(caplog):
    # The mark is unconditional: a frame with nothing refused carries False on every row, so the health record's
    # count is 0 however the stream was batched, where a missing column would read as unanswered.
    quiet = pd.DataFrame({"x": [1, 2]})

    with caplog.at_level(logging.WARNING):
        count = refuse_rows(quiet, [False, False], logger=logging.getLogger("t"), stage="S", what="rows", outcome="o.")

    assert count == 0
    assert quiet[CLOCK_REFUSED_COLUMN].tolist() == [False, False]
    assert caplog.records == []

    loud = pd.DataFrame({"x": [1, 2, 3]})

    with caplog.at_level(logging.WARNING):
        count = refuse_rows(loud, [False, True, True],
                            logger=logging.getLogger("t"),
                            stage="TC3TtlStage",
                            what="flows",
                            outcome="they joined no reference.")

    assert count == 2
    assert loud[CLOCK_REFUSED_COLUMN].tolist() == [False, True, True]
    assert [record.getMessage() for record in caplog.records] == [
        "TC3TtlStage refused 2 of 3 flows whose event time was further ahead of the stream than "
        "max_clock_skew_seconds allows; they joined no reference."
    ]


def test_refusing_rows_keeps_an_earlier_stage_mark():
    df = pd.DataFrame({"x": [1, 2, 3], CLOCK_REFUSED_COLUMN: [True, False, None]})

    refuse_rows(df, [False, True, False], logger=logging.getLogger("t"), stage="S", what="rows", outcome="o.")

    assert [bool(value) for value in df[CLOCK_REFUSED_COLUMN]] == [True, True, False]
