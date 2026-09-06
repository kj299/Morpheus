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
from morpheus.stages.telemetry.tc5_risk_stage import TC5RiskStage
from morpheus.utils.outcome_run import NS_PER_SECOND
from morpheus.utils.type_utils import get_df_class

MINUTE_NS = 60 * NS_PER_SECOND
ALICE = "alice@example.com"


def frame(results: list,
          mfa_results: list = None,
          mfa_used: list = None,
          principals: list = None,
          times: list = None) -> dict:
    count = len(results)

    def column(supplied, default):
        return [default] * count if supplied is None else supplied

    return {
        "user_principal": column(principals, ALICE),
        "auth_result": results,
        "mfa_used": column(mfa_used, False),
        "mfa_result": column(mfa_results, None),
        "event_time": [index * NS_PER_SECOND for index in range(count)] if times is None else times,
    }


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [None if pd.isna(value) else value for value in series.tolist()]


def run(config: Config, payload: dict, **kwargs) -> MessageMeta:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC5RiskStage(config, **kwargs).on_data(meta)

    return meta


def fatigue() -> dict:
    """R-D-L5-004's shape: five challenges inside ten minutes, four denials, then an approval."""
    return frame(["failure"] * 4 + ["success"],
                 mfa_results=["denied"] * 4 + ["approved"],
                 mfa_used=[True] * 5,
                 times=[index * MINUTE_NS for index in range(5)])


@pytest.mark.gpu_and_cpu_mode
def test_the_mfa_fatigue_shape_is_readable_off_the_approving_row(config: Config):
    # Every part of the rule has to be readable off one row, or the rule cannot be written as a search.
    meta = run(config, fatigue())

    assert _as_list(meta, "mfa_attempts_in_window")[-1] == 5
    assert _as_list(meta, "mfa_denials_in_window")[-1] == 4
    assert _as_list(meta, "consecutive_mfa_denials")[-1] == 4
    assert _as_list(meta, "mfa_denied_then_approved")[-1] is True


@pytest.mark.gpu_and_cpu_mode
def test_the_denials_alone_never_report_the_pattern(config: Config):
    # The approval is what makes this actionable rather than noisy. Without it the rule fires on every user who
    # fumbles a prompt.
    meta = run(config, fatigue())

    assert _as_list(meta, "mfa_denied_then_approved")[:4] == [False] * 4


@pytest.mark.gpu_and_cpu_mode
def test_denials_need_not_be_contiguous_to_be_counted(config: Config):
    # A fatigue attack interleaves with the victim's own traffic, so the denials are not a clean run.
    payload = frame(["failure", "failure", "success", "failure", "failure", "success"],
                    mfa_results=["denied", "denied", "approved", "denied", "denied", "approved"],
                    mfa_used=[True] * 6,
                    times=[index * MINUTE_NS for index in range(6)])

    meta = run(config, payload)

    assert _as_list(meta, "mfa_denials_in_window")[-1] == 4
    assert _as_list(meta, "consecutive_mfa_denials")[-1] == 2


@pytest.mark.gpu_and_cpu_mode
def test_the_counts_decay_out_of_the_ten_minute_window(config: Config):
    # The rule says ten minutes. A denial from an hour ago is not part of this burst.
    payload = frame(["failure"] * 4 + ["success"],
                    mfa_results=["denied"] * 4 + ["approved"],
                    mfa_used=[True] * 5,
                    times=[0, MINUTE_NS, 2 * MINUTE_NS, 3 * MINUTE_NS, 60 * MINUTE_NS])

    meta = run(config, payload)

    assert _as_list(meta, "mfa_denials_in_window")[-1] == 0
    assert _as_list(meta, "mfa_denied_then_approved")[-1] is False


@pytest.mark.gpu_and_cpu_mode
def test_ordinary_authentication_outcomes_are_counted_the_same_way(config: Config):
    meta = run(config, frame(["failure", "failure", "success"]))

    assert _as_list(meta, "auth_failures_in_window")[-1] == 2
    assert _as_list(meta, "consecutive_auth_failures")[-1] == 2
    assert _as_list(meta, "auth_failed_then_succeeded") == [False, False, True]


@pytest.mark.gpu_and_cpu_mode
def test_a_success_with_nothing_before_it_is_not_the_pattern(config: Config):
    meta = run(config, frame(["success"]))

    assert _as_list(meta, "auth_failed_then_succeeded") == [False]
    assert _as_list(meta, "consecutive_auth_failures") == [0]


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("absent", [None, pd.NA, "", "   "], ids=["none", "pandas_na", "blank", "whitespace"])
def test_an_unknown_outcome_is_not_guessed_at(config: Config, absent):
    # Counted as a failure it inflates a run; counted as a success it ends one that is still going. Neither is
    # better than declining, so the row carries no authentication counts and the run is left where it was.
    meta = run(config, frame(["failure", absent, "success"]))

    assert _as_list(meta, "auth_failures_in_window") == [1, None, 1]
    assert _as_list(meta, "consecutive_auth_failures")[-1] == 1
    assert _as_list(meta, "auth_failed_then_succeeded")[-1] is True


@pytest.mark.gpu_and_cpu_mode
def test_a_record_is_a_challenge_if_either_field_says_so(config: Config):
    # Sources differ on which they populate; requiring both would leave the feature silent on half of them.
    payload = frame(["success"] * 3, mfa_used=[True, False, False], mfa_results=[None, "approved", None])

    meta = run(config, payload)

    assert _as_list(meta, "mfa_challenge") == [True, True, False]


@pytest.mark.gpu_and_cpu_mode
def test_a_challenge_with_no_outcome_joins_no_run(config: Config):
    # It says a factor was involved, not how it resolved, so it can neither extend a run of denials nor end one.
    payload = frame(["failure", "success"], mfa_used=[True, True], mfa_results=[None, "approved"])

    meta = run(config, payload)

    assert _as_list(meta, "mfa_challenge") == [True, True]
    assert _as_list(meta, "mfa_attempts_in_window") == [None, 1]
    assert _as_list(meta, "mfa_denied_then_approved") == [None, False]


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("flag", [True, "true", "Yes", "1"], ids=["bool", "string", "mixed_case", "numeric_string"])
def test_a_textual_flag_is_read_as_true(config: Config, flag):
    meta = run(config, frame(["success"], mfa_used=[flag]))

    assert _as_list(meta, "mfa_challenge") == [True]


@pytest.mark.gpu_and_cpu_mode
def test_the_ratio_is_withheld_until_there_is_enough_to_measure(config: Config):
    # Below the minimum a proportion is noise rather than a measurement.
    meta = run(config, frame(["success"] * 4, mfa_used=[True, True, False, False]), min_denominator=4)

    assert _as_list(meta, "mfa_ratio")[:3] == [None, None, None]
    assert _as_list(meta, "mfa_ratio")[3] == pytest.approx(0.5)


@pytest.mark.gpu_and_cpu_mode
def test_the_ratio_uses_a_wider_window_than_the_runs(config: Config):
    # A proportion over ten minutes describes one sitting rather than how a principal normally authenticates, so
    # the two windows are separate parameters.
    payload = frame(["success"] * 4,
                    mfa_used=[True, True, False, False],
                    times=[0, MINUTE_NS, 2 * MINUTE_NS, 60 * MINUTE_NS])

    meta = run(config, payload, min_denominator=1, run_window_seconds=600, ratio_window_seconds=86400)

    # The last record is an hour later: outside the run window, inside the ratio window.
    assert _as_list(meta, "mfa_ratio")[-1] == pytest.approx(0.5)
    assert _as_list(meta, "auth_attempts_in_window")[-1] == 1


@pytest.mark.gpu_and_cpu_mode
def test_principals_do_not_share_a_window(config: Config):
    payload = frame(["failure", "failure", "success"], principals=[ALICE, ALICE, "bob@example.com"])

    meta = run(config, payload)

    assert _as_list(meta, "auth_failures_in_window")[-1] == 0
    assert _as_list(meta, "auth_failed_then_succeeded")[-1] is False


@pytest.mark.gpu_and_cpu_mode
def test_a_record_with_no_principal_carries_no_counts(config: Config):
    meta = run(config, frame(["failure", "success"], principals=[None, ALICE]))

    assert _as_list(meta, "auth_failures_in_window") == [None, 0]
    assert _as_list(meta, "mfa_ratio") == [None, None]


@pytest.mark.gpu_and_cpu_mode
def test_the_result_vocabulary_is_case_insensitive(config: Config):
    meta = run(config, frame(["FAILURE", "Success"]))

    assert _as_list(meta, "auth_failed_then_succeeded") == [False, True]


@pytest.mark.gpu_and_cpu_mode
def test_a_collectors_own_vocabulary_can_be_supplied(config: Config):
    meta = run(config,
               frame(["reject", "accept"], mfa_results=["reject", "accept"], mfa_used=[True, True]),
               success_values=("accept", ),
               mfa_success_values=("accept", ))

    assert _as_list(meta, "auth_failed_then_succeeded") == [False, True]
    assert _as_list(meta, "mfa_denied_then_approved") == [False, True]


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    stage = TC5RiskStage(config)

    first = MessageMeta(
        get_df_class(config.execution_mode)(frame(["failure"] * 4,
                                                  mfa_results=["denied"] * 4,
                                                  mfa_used=[True] * 4,
                                                  times=[index * MINUTE_NS for index in range(4)])))
    stage.on_data(first)

    second = MessageMeta(
        get_df_class(config.execution_mode)(frame(["success"],
                                                  mfa_results=["approved"],
                                                  mfa_used=[True],
                                                  times=[4 * MINUTE_NS])))
    stage.on_data(second)

    assert _as_list(second, "mfa_denials_in_window") == [4]
    assert _as_list(second, "mfa_denied_then_approved") == [True]
    assert stage.tracked_principals == 1


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    message = ControlMessage()
    message.payload(MessageMeta(get_df_class(config.execution_mode)(fatigue())))

    TC5RiskStage(config).on_data(message)

    assert _as_list(message.payload(), "mfa_denied_then_approved")[-1] is True


@pytest.mark.gpu_and_cpu_mode
def test_tc5_risk_stage_pipe(config: Config):
    source_df = get_df_class(config.execution_mode)(fatigue())

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC5RiskStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "consecutive_mfa_denials")[-1] == 4


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = frame(["success"])
    del payload["user_principal"]

    with pytest.raises(KeyError, match="user_principal"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError, match="run_window_seconds must be positive"):
        TC5RiskStage(config, run_window_seconds=0)

    with pytest.raises(ValueError, match="ratio_window_seconds must be positive"):
        TC5RiskStage(config, ratio_window_seconds=0)


@pytest.mark.cpu_mode
def test_the_sample_cap_marks_the_counts_as_a_floor(config: Config):
    # `min_denominator` travels down with it: the shared ratio window refuses a cap it could never publish a
    # proportion from, which is a build-time error rather than a silently null column.
    meta = run(config, frame(["failure"] * 5), max_samples=3, min_denominator=1)

    assert _as_list(meta, "risk_counts_saturated")[-1] is True
    assert _as_list(meta, "auth_attempts_in_window")[-1] == 3


@pytest.mark.cpu_mode
def test_an_out_of_order_record_does_not_join_a_run(config: Config):
    stage = TC5RiskStage(config)

    first = MessageMeta(
        get_df_class(config.execution_mode)(frame(["failure", "failure"], times=[5 * MINUTE_NS, 6 * MINUTE_NS])))
    stage.on_data(first)

    late = MessageMeta(get_df_class(config.execution_mode)(frame(["success"], times=[0])))
    stage.on_data(late)

    assert _as_list(late, "auth_failed_then_succeeded") == [False]

    # And the run it declined to end is still going.
    after = MessageMeta(get_df_class(config.execution_mode)(frame(["success"], times=[7 * MINUTE_NS])))
    stage.on_data(after)

    assert _as_list(after, "consecutive_auth_failures") == [2]
    assert _as_list(after, "auth_failed_then_succeeded") == [True]
