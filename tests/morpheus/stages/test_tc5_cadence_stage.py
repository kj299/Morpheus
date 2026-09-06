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

import datetime

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc5_cadence_stage import NS_PER_DAY
from morpheus.stages.telemetry.tc5_cadence_stage import NS_PER_HOUR
from morpheus.stages.telemetry.tc5_cadence_stage import TC5CadenceStage
from morpheus.utils.type_utils import get_df_class

ALICE = "alice@example.com"


def at(day: int, hour: int) -> int:
    """Epoch nanoseconds for an hour on a day counted from 1970-01-01."""
    return day * NS_PER_DAY + hour * NS_PER_HOUR


def frame(times: list, principals: list = None) -> dict:
    count = len(times)

    return {
        "user_principal": [ALICE] * count if principals is None else principals,
        "event_time": times,
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
    TC5CadenceStage(config, **kwargs).on_data(meta)

    return meta


def office_week(days: int = 20) -> list:
    """One 10:00 authentication on each of `days` consecutive days."""
    return [at(day, 10) for day in range(days)]


@pytest.mark.gpu_and_cpu_mode
def test_the_hour_and_weekday_match_the_calendar(config: Config):
    # Integer arithmetic on the epoch, checked against the standard library rather than against itself.
    times = [at(0, 0), at(1, 13), at(19_000, 7)]
    meta = run(config, frame(times))

    expected = [datetime.datetime.fromtimestamp(time / 10**9, tz=datetime.timezone.utc) for time in times]

    assert _as_list(meta, "local_hour") == [moment.hour for moment in expected]
    assert _as_list(meta, "local_weekday") == [moment.weekday() for moment in expected]


@pytest.mark.gpu_and_cpu_mode
def test_the_epoch_was_a_thursday(config: Config):
    meta = run(config, frame([at(0, 0)]))

    assert _as_list(meta, "local_weekday") == [3]


@pytest.mark.gpu_and_cpu_mode
def test_the_offset_moves_the_hour_and_can_move_the_day(config: Config):
    # 1970-01-01 23:00 UTC is 1970-01-02 08:00 in a nine-hour zone, which is a Friday rather than a Thursday.
    meta = run(config, frame([at(0, 23)]), utc_offset_minutes=9 * 60)

    assert _as_list(meta, "local_hour") == [8]
    assert _as_list(meta, "local_weekday") == [4]


@pytest.mark.gpu_and_cpu_mode
def test_a_negative_offset_can_move_the_day_backwards(config: Config):
    # 1970-01-02 02:00 UTC is 1970-01-01 21:00 five hours west.
    meta = run(config, frame([at(1, 2)]), utc_offset_minutes=-5 * 60)

    assert _as_list(meta, "local_hour") == [21]
    assert _as_list(meta, "local_weekday") == [3]


@pytest.mark.gpu_and_cpu_mode
def test_an_offset_off_the_hour_is_honoured(config: Config):
    # Several populated zones are not on an hour boundary, which is why the parameter is in minutes.
    meta = run(config, frame([at(0, 10)]), utc_offset_minutes=345)

    assert _as_list(meta, "local_hour") == [15]


@pytest.mark.gpu_and_cpu_mode
def test_a_night_login_is_unseen_against_a_history_of_office_hours(config: Config):
    times = office_week() + [at(20, 3)]
    meta = run(config, frame(times))

    assert _as_list(meta, "hour_unseen")[-1] is True
    assert _as_list(meta, "hour_share")[-1] < _as_list(meta, "hour_share")[-2]


@pytest.mark.gpu_and_cpu_mode
def test_a_repeated_hour_becomes_unremarkable(config: Config):
    meta = run(config, frame(office_week()))
    bits = _as_list(meta, "hour_surprise_bits")

    assert bits == sorted(bits, reverse=True)
    assert bits[0] > bits[-1]


@pytest.mark.gpu_and_cpu_mode
def test_the_sample_is_judged_against_history_that_excludes_it(config: Config):
    # Were the sample counted first, a principal's very first 03:00 authentication would hold the least surprising
    # share the measure has, at the moment it should hold the most.
    meta = run(config, frame(office_week(4) + [at(4, 3)]))

    assert _as_list(meta, "cadence_samples") == [0, 1, 2, 3, 4]
    assert _as_list(meta, "hour_unseen") == [True, False, False, False, True]


@pytest.mark.gpu_and_cpu_mode
def test_maturity_is_reported_rather_than_used_to_withhold_a_score(config: Config):
    # A brand new account authenticating at 03:00 is not obviously the less interesting case, so the score is
    # written either way and the rule decides.
    meta = run(config, frame(office_week(4)), min_samples=2)

    assert _as_list(meta, "cadence_mature") == [False, False, True, True]
    assert all(value is not None for value in _as_list(meta, "hour_share"))


@pytest.mark.gpu_and_cpu_mode
def test_a_weekend_login_is_scored_separately_from_the_hour(config: Config):
    # Twelve weekdays at 10:00, then a Saturday at 10:00. The hour is ordinary and the day is not.
    weekdays = [at(day, 10) for day in range(4, 16) if (day + 3) % 7 < 5]
    saturday = next(at(day, 10) for day in range(16, 30) if (day + 3) % 7 == 5)

    meta = run(config, frame(weekdays + [saturday]))

    assert _as_list(meta, "hour_unseen")[-1] is False
    assert _as_list(meta, "weekday_unseen")[-1] is True
    assert _as_list(meta, "weekday_surprise_bits")[-1] > _as_list(meta, "hour_surprise_bits")[-1]


@pytest.mark.gpu_and_cpu_mode
def test_principals_do_not_share_a_histogram(config: Config):
    times = office_week(6) + [at(6, 3)]
    principals = [ALICE] * 6 + ["bob@example.com"]

    meta = run(config, frame(times, principals=principals))

    assert _as_list(meta, "cadence_samples")[-1] == 0
    assert _as_list(meta, "hour_share")[-1] == pytest.approx(1 / 24, abs=1e-4)


@pytest.mark.gpu_and_cpu_mode
def test_a_record_with_no_principal_still_carries_its_hour(config: Config):
    # The hour is a fact about the record rather than about the principal, and a rule may want it even where no
    # history exists to score it against.
    meta = run(config, frame([at(0, 9), at(0, 10)], principals=[None, ALICE]))

    assert _as_list(meta, "local_hour") == [9, 10]
    assert _as_list(meta, "hour_share") == [None, pytest.approx(1 / 24, abs=1e-4)]
    assert _as_list(meta, "cadence_samples") == [None, 0]


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    stage = TC5CadenceStage(config)

    first = MessageMeta(get_df_class(config.execution_mode)(frame(office_week(5))))
    stage.on_data(first)

    second = MessageMeta(get_df_class(config.execution_mode)(frame([at(5, 10)])))
    stage.on_data(second)

    assert _as_list(second, "cadence_samples") == [5]
    assert _as_list(second, "hour_unseen") == [False]
    assert stage.tracked_principals == 1


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    message = ControlMessage()
    message.payload(MessageMeta(get_df_class(config.execution_mode)(frame(office_week(3)))))

    TC5CadenceStage(config).on_data(message)

    assert _as_list(message.payload(), "cadence_samples") == [0, 1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_tc5_cadence_stage_pipe(config: Config):
    source_df = get_df_class(config.execution_mode)(frame(office_week(4) + [at(4, 3)]))

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC5CadenceStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "hour_unseen") == [True, False, False, False, True]


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = frame([at(0, 10)])
    del payload["user_principal"]

    with pytest.raises(KeyError, match="user_principal"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError, match="utc_offset_minutes must be inside a day"):
        TC5CadenceStage(config, utc_offset_minutes=24 * 60)

    with pytest.raises(ValueError, match="utc_offset_minutes must be inside a day"):
        TC5CadenceStage(config, utc_offset_minutes=-24 * 60)


@pytest.mark.cpu_mode
def test_an_out_of_order_record_is_scored_but_does_not_join_the_histogram(config: Config):
    stage = TC5CadenceStage(config)

    first = MessageMeta(get_df_class(config.execution_mode)(frame(office_week(4))))
    stage.on_data(first)

    late = MessageMeta(get_df_class(config.execution_mode)(frame([at(0, 3)])))
    stage.on_data(late)

    assert _as_list(late, "cadence_samples") == [4]

    # And 03:00 is still unseen, because the late record did not join the history.
    after = MessageMeta(get_df_class(config.execution_mode)(frame([at(9, 3)])))
    stage.on_data(after)

    assert _as_list(after, "hour_unseen") == [True]
    assert _as_list(after, "cadence_samples") == [4]
