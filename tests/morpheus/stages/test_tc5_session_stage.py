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

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc5_session_stage import TC5SessionStage
from morpheus.utils.session_timer import NS_PER_SECOND
from morpheus.utils.type_utils import get_df_class

HOUR_NS = 3600 * NS_PER_SECOND
ALICE = "alice@example.com"


def frame(actions: list, sessions: list = None, times: list = None, principals: list = None) -> dict:
    count = len(actions)

    return {
        "session_id": ["s-1"] * count if sessions is None else sessions,
        "user_principal": [ALICE] * count if principals is None else principals,
        "session_action": actions,
        "event_time": [index * NS_PER_SECOND for index in range(count)] if times is None else times,
    }


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    # `pd.isna` rather than the `value != value` NaN trick, which pylint reads as a redundant self-comparison. It
    # covers None, pandas.NA and NaN together, and which of the three arrives depends on the column's dtype.
    return [None if pd.isna(value) else value for value in series.tolist()]


def run(config: Config, payload: dict, **kwargs) -> MessageMeta:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC5SessionStage(config, **kwargs).on_data(meta)

    return meta


@pytest.mark.gpu_and_cpu_mode
def test_a_start_and_its_stop_become_a_duration(config: Config):
    meta = run(config, frame(["start", "end"], times=[0, 90 * NS_PER_SECOND]))

    assert _as_list(meta, "session_duration_ns") == [None, 90 * NS_PER_SECOND]
    assert _as_list(meta, "session_duration_s") == [None, 90]
    assert _as_list(meta, "session_unpaired") == [None, False]


@pytest.mark.gpu_and_cpu_mode
def test_the_seconds_are_a_floor_and_the_nanoseconds_are_exact(config: Config):
    # Anything doing arithmetic reads the nanosecond column; the seconds exist so a threshold is legible.
    meta = run(config, frame(["start", "end"], times=[0, NS_PER_SECOND + 999_999_999]))

    assert _as_list(meta, "session_duration_ns") == [None, NS_PER_SECOND + 999_999_999]
    assert _as_list(meta, "session_duration_s") == [None, 1]


@pytest.mark.gpu_and_cpu_mode
def test_a_stop_with_no_start_is_reported_rather_than_dropped(config: Config):
    # Ordinary at the beginning of a stream, and a lost record everywhere else. A null duration with no explanation
    # reads as missing data instead of as an explained absence.
    meta = run(config, frame(["end"]))

    assert _as_list(meta, "session_unpaired") == [True]
    assert _as_list(meta, "session_duration_ns") == [None]


@pytest.mark.gpu_and_cpu_mode
def test_a_second_start_on_one_identifier_is_counted_not_treated_as_a_retry(config: Config):
    # A session identifier is meant to be unique to a session. Two starts on one is a collector duplicating records
    # or an identifier being reused, and either is worth telling apart from a clean pairing.
    meta = run(config, frame(["start", "start", "end"], times=[0, NS_PER_SECOND, 10 * NS_PER_SECOND]))

    assert _as_list(meta, "session_starts") == [None, None, 2]
    # The duration runs from the most recent start, which is what the timer holds.
    assert _as_list(meta, "session_duration_s") == [None, None, 9]


@pytest.mark.gpu_and_cpu_mode
def test_a_clean_pairing_reports_one_start(config: Config):
    meta = run(config, frame(["start", "end"], times=[0, NS_PER_SECOND]))

    assert _as_list(meta, "session_starts") == [None, 1]


@pytest.mark.gpu_and_cpu_mode
def test_two_principals_sessions_do_not_close_each_other(config: Config):
    payload = frame(["start", "start", "end", "end"],
                    sessions=["s-a", "s-b", "s-b", "s-a"],
                    principals=[ALICE, "bob@example.com", "bob@example.com", ALICE],
                    times=[0, NS_PER_SECOND, 5 * NS_PER_SECOND, 30 * NS_PER_SECOND])

    meta = run(config, payload)

    assert _as_list(meta, "session_duration_s") == [None, None, 4, 30]


@pytest.mark.gpu_and_cpu_mode
def test_a_stop_carrying_no_principal_still_pairs(config: Config):
    # The normal shape of a Windows logoff record. Keying the pairing on the principal as well as the identifier
    # would fail to pair the two, and every such session would report unpaired.
    payload = frame(["start", "end"], principals=[ALICE, None], times=[0, 60 * NS_PER_SECOND])

    meta = run(config, payload)

    assert _as_list(meta, "session_unpaired") == [None, False]
    assert _as_list(meta, "session_duration_s") == [None, 60]


@pytest.mark.gpu_and_cpu_mode
def test_the_session_key_composes_the_principal_with_the_identifier(config: Config):
    meta = run(config, frame(["start"]))

    assert _as_list(meta, "session_key") == [f"{ALICE}:s-1"]


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_part_yields_a_null_key_rather_than_the_string_none(config: Config):
    meta = run(config, frame(["start"], principals=[None]))

    assert _as_list(meta, "session_key") == [None]


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("absent", [None, pd.NA, "", "   "], ids=["none", "pandas_na", "blank", "whitespace"])
def test_a_record_naming_neither_end_of_the_lifecycle_pairs_with_nothing(config: Config, absent):
    # Unlike an 802.1X result, where a null genuinely means an exchange that has not resolved, an empty lifecycle
    # column says nothing about which end of a session this is. Guessing "start" would pair a later stop against a
    # record that was never a start.
    meta = run(config, frame(["start", absent, "end"], times=[0, NS_PER_SECOND, 10 * NS_PER_SECOND]))

    assert _as_list(meta, "session_duration_s") == [None, None, 10]
    assert _as_list(meta, "session_unpaired") == [None, None, False]


@pytest.mark.gpu_and_cpu_mode
def test_an_unrecognised_action_pairs_with_nothing(config: Config):
    meta = run(config, frame(["start", "renew", "end"], times=[0, NS_PER_SECOND, 10 * NS_PER_SECOND]))

    assert _as_list(meta, "session_duration_s") == [None, None, 10]


@pytest.mark.gpu_and_cpu_mode
def test_the_lifecycle_vocabulary_is_case_insensitive(config: Config):
    meta = run(config, frame(["Logon", "LOGOFF"], times=[0, 5 * NS_PER_SECOND]))

    assert _as_list(meta, "session_duration_s") == [None, 5]


@pytest.mark.gpu_and_cpu_mode
def test_a_collectors_own_vocabulary_can_be_supplied(config: Config):
    meta = run(config,
               frame(["Acct-Start", "Acct-Stop"], times=[0, 5 * NS_PER_SECOND]),
               start_actions=("acct-start", ),
               end_actions=("acct-stop", ))

    assert _as_list(meta, "session_duration_s") == [None, 5]


@pytest.mark.gpu_and_cpu_mode
def test_a_null_identifier_pairs_with_nothing(config: Config):
    meta = run(config, frame(["start", "end"], sessions=[None, None], times=[0, NS_PER_SECOND]))

    assert _as_list(meta, "session_unpaired") == [None, None]
    assert _as_list(meta, "session_duration_ns") == [None, None]


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    stage = TC5SessionStage(config)

    first = MessageMeta(get_df_class(config.execution_mode)(frame(["start"], times=[0])))
    stage.on_data(first)

    assert stage.open_sessions == 1

    second = MessageMeta(get_df_class(config.execution_mode)(frame(["end"], times=[120 * NS_PER_SECOND])))
    stage.on_data(second)

    assert _as_list(second, "session_duration_s") == [120]
    assert stage.open_sessions == 0


@pytest.mark.gpu_and_cpu_mode
def test_nulls_survive_as_nulls_not_false(config: Config):
    meta = run(config, frame(["start"]))

    assert _as_list(meta, "session_unpaired") == [None]


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    message = ControlMessage()
    message.payload(
        MessageMeta(get_df_class(config.execution_mode)(frame(["start", "end"], times=[0, 3 * NS_PER_SECOND]))))

    TC5SessionStage(config).on_data(message)

    assert _as_list(message.payload(), "session_duration_s") == [None, 3]


@pytest.mark.gpu_and_cpu_mode
def test_tc5_session_stage_pipe(config: Config):
    payload = frame(["start", "end", "end"], times=[0, 60 * NS_PER_SECOND, 90 * NS_PER_SECOND])
    source_df = get_df_class(config.execution_mode)(payload)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC5SessionStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "session_unpaired") == [None, False, True]


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = frame(["start"])
    del payload["session_action"]

    with pytest.raises(KeyError, match="session_action"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        TC5SessionStage(config, timeout_seconds=0)

    with pytest.raises(ValueError, match="max_clock_skew_seconds must be positive"):
        TC5SessionStage(config, max_clock_skew_seconds=0)


def test_a_lifecycle_value_cannot_open_and_close(config: Config):
    # Which one it did would depend on the order the two sets were tested in rather than on anything in the record.
    with pytest.raises(ValueError, match="must not overlap"):
        TC5SessionStage(config, start_actions=("start", "refresh"), end_actions=("end", "refresh"))


@pytest.mark.cpu_mode
def test_a_session_past_the_timeout_is_abandoned(config: Config):
    stage = TC5SessionStage(config, timeout_seconds=3600)

    first = MessageMeta(get_df_class(config.execution_mode)(frame(["start"], times=[0])))
    stage.on_data(first)

    # Two hours later, a stop for a different session drives the clock past the timeout.
    second = MessageMeta(
        get_df_class(config.execution_mode)(frame(["start", "end"],
                                                  sessions=["s-2", "s-2"],
                                                  times=[2 * HOUR_NS, 2 * HOUR_NS + NS_PER_SECOND])))
    stage.on_data(second)

    assert stage.open_sessions == 0

    # The lost session's stop, if it ever arrives, reads as unpaired rather than as a two-day session.
    third = MessageMeta(get_df_class(config.execution_mode)(frame(["end"], times=[3 * HOUR_NS])))
    stage.on_data(third)

    assert _as_list(third, "session_unpaired") == [True]


@pytest.mark.cpu_mode
def test_a_session_inside_the_timeout_still_pairs(config: Config):
    stage = TC5SessionStage(config, timeout_seconds=3600)

    first = MessageMeta(get_df_class(config.execution_mode)(frame(["start"], times=[0])))
    stage.on_data(first)

    second = MessageMeta(get_df_class(config.execution_mode)(frame(["end"], times=[1800 * NS_PER_SECOND])))
    stage.on_data(second)

    assert _as_list(second, "session_duration_s") == [1800]


@pytest.mark.cpu_mode
def test_expiry_falls_in_the_same_place_however_the_stream_is_batched(config: Config):
    payload = frame(["start", "end"], times=[0, 10 * HOUR_NS])

    whole = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC5SessionStage(config, timeout_seconds=3600).on_data(whole)

    split = TC5SessionStage(config, timeout_seconds=3600)
    rows = [
        MessageMeta(get_df_class(config.execution_mode)(frame([action], times=[time])))
        for (action, time) in zip(payload["session_action"], payload["event_time"])
    ]

    for row in rows:
        split.on_data(row)

    assert _as_list(whole, "session_unpaired")[1] is True
    assert _as_list(rows[1], "session_unpaired")[0] is True


@pytest.mark.cpu_mode
def test_one_row_from_a_broken_clock_does_not_abandon_every_open_session(config: Config):
    stage = TC5SessionStage(config, timeout_seconds=3600, max_clock_skew_seconds=86400)

    opened = MessageMeta(get_df_class(config.execution_mode)(frame(["start"], times=[0])))
    stage.on_data(opened)

    # A source whose clock is wrong by years. The row is refused rather than allowed to drive expiry.
    broken = MessageMeta(
        get_df_class(config.execution_mode)(frame(["start"], sessions=["s-9"], times=[10 * 365 * 24 * HOUR_NS])))
    stage.on_data(broken)

    assert _as_list(broken, "session_unpaired") == [None]
    assert stage.open_sessions == 1

    closed = MessageMeta(get_df_class(config.execution_mode)(frame(["end"], times=[600 * NS_PER_SECOND])))
    stage.on_data(closed)

    assert _as_list(closed, "session_duration_s") == [600]
