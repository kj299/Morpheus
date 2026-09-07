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
from morpheus.stages.telemetry.tc5_drift_stage import TC5DriftStage
from morpheus.utils.type_utils import get_df_class

ALICE = "alice@example.com"
BOB = "bob@example.com"

FLAT = [1.0, 1.02, 0.98, 1.0, 1.01, 1.0]
CLIMB = [1.2, 1.6, 2.1, 2.7]


def frame(scores: list, principals: list = None, windows: list = None) -> dict:
    count = len(scores)

    return {
        "user_principal": [ALICE] * count if principals is None else principals,
        "window_id": list(range(100, 100 + count)) if windows is None else windows,
        "mean_abs_z": scores,
    }


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [None if pd.isna(value) else value for value in series.tolist()]


def run(config: Config, payload: dict, **kwargs) -> MessageMeta:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC5DriftStage(config, **kwargs).on_data(meta)

    return meta


@pytest.mark.gpu_and_cpu_mode
def test_the_predictive_shape_is_readable_off_the_last_row(config: Config):
    # R-P-L5-006: four consecutive rising windows and a rise above 1.5 of the principal's own standard
    # deviations, with no single window crossing the alerting threshold. Every part has to be on one row.
    meta = run(config, frame(FLAT + CLIMB))

    assert _as_list(meta, "drift_rising_windows")[-1] >= 4
    assert _as_list(meta, "drift_rise_sigmas")[-1] > 1.5
    assert _as_list(meta, "drift_mature")[-1] is True
    # And the rule's other half: no single window is extreme on its own.
    assert max(FLAT + CLIMB) < 6.0


@pytest.mark.gpu_and_cpu_mode
def test_a_flat_principal_produces_no_run(config: Config):
    meta = run(config, frame(FLAT))

    assert max(value for value in _as_list(meta, "drift_rising_windows") if value is not None) < 4


@pytest.mark.gpu_and_cpu_mode
def test_velocity_and_acceleration_are_the_two_differences(config: Config):
    meta = run(config, frame([1.0, 1.1, 1.5]))

    assert _as_list(meta, "drift_velocity") == [None, pytest.approx(0.1), pytest.approx(0.4)]
    assert _as_list(meta, "drift_acceleration")[-1] == pytest.approx(0.3)


@pytest.mark.gpu_and_cpu_mode
def test_a_gap_in_windows_restarts_the_run(config: Config):
    # A principal who did not authenticate for two days has not been rising for four consecutive windows.
    meta = run(config, frame([1.0, 1.2, 1.5], windows=[100, 101, 110]))

    assert _as_list(meta, "drift_run_restarted") == [False, False, True]
    assert _as_list(meta, "drift_rising_windows") == [1, 2, 1]


@pytest.mark.gpu_and_cpu_mode
def test_principals_do_not_share_a_trajectory(config: Config):
    payload = frame([1.0, 1.2, 5.0], principals=[ALICE, ALICE, BOB], windows=[100, 101, 102])

    meta = run(config, payload)

    assert _as_list(meta, "drift_rising_windows") == [1, 2, 1]


@pytest.mark.gpu_and_cpu_mode
def test_a_pipeline_with_no_model_carries_nulls_rather_than_failing(config: Config):
    # What this fork actually ships as: no per-entity model, so nothing produces mean_abs_z. The column is
    # absent, every drift column is null, and R-P-L5-006 fires on nothing until a model exists.
    payload = frame([1.0, 1.2])
    del payload["mean_abs_z"]

    meta = run(config, payload)

    assert _as_list(meta, "drift_rising_windows") == [None, None]
    assert _as_list(meta, "drift_rise_sigmas") == [None, None]


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("absent", [None, float("nan")], ids=["none", "nan"])
def test_an_unscored_window_breaks_the_run_rather_than_being_bridged(config: Config, absent):
    # The stronger half of the same rule. The unscored window carries no trajectory, and the window after it
    # starts a new run rather than continuing across the gap: a rise measured over a window nothing scored
    # would be a claim about a window nothing scored.
    meta = run(config, frame([1.0, absent, 1.5]))

    assert _as_list(meta, "drift_rising_windows") == [1, None, 1]


@pytest.mark.gpu_and_cpu_mode
def test_a_principal_who_has_never_varied_reports_no_sigmas(config: Config):
    # Dividing by a zero spread would make the flattest history the most alarming.
    meta = run(config, frame([1.0] * 5 + [1.5]))

    assert _as_list(meta, "drift_baseline_sigma")[-1] == 0.0
    assert _as_list(meta, "drift_rise_sigmas")[-1] is None


@pytest.mark.gpu_and_cpu_mode
def test_a_nullable_score_column_carrying_pandas_na_does_not_raise(config: Config):
    # The shape this stage will actually be handed once TC5ScoreStage runs in front of it. That stage writes
    # `mean_abs_z` through `assign_nullable_float_column`, so the column is a pandas nullable Float64 and its
    # gaps are `pd.NA` rather than `float("nan")`.
    #
    # The distinction is not cosmetic. For `pd.NA`, `value is None` is False and `value != value` raises
    # "boolean value of NA is ambiguous" instead of returning True -- so the null check that worked on every
    # corpus tested until now killed the pipeline the first time a real scorer fed it. Only `pd.isna` handles
    # all three of None, NaN and NA.
    from morpheus.utils.column_assign import assign_nullable_float_column  # pylint: disable=import-outside-toplevel

    payload = frame([1.0, 1.0, 1.0])
    del payload["mean_abs_z"]
    df = get_df_class(config.execution_mode)(payload)
    assign_nullable_float_column(df, "mean_abs_z", [1.0, None, 2.0])
    meta = MessageMeta(df)

    TC5DriftStage(config).on_data(meta)

    assert _as_list(meta, "drift_rising_windows") == [1, None, 1]


@pytest.mark.gpu_and_cpu_mode
def test_a_differently_named_score_column_is_honoured(config: Config):
    payload = frame(FLAT + CLIMB)
    payload["max_abs_z"] = payload.pop("mean_abs_z")

    meta = run(config, payload, score_column="max_abs_z")

    assert _as_list(meta, "drift_rising_windows")[-1] >= 4


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    stage = TC5DriftStage(config)

    first = MessageMeta(get_df_class(config.execution_mode)(frame(FLAT)))
    stage.on_data(first)

    second = MessageMeta(get_df_class(config.execution_mode)(frame(CLIMB, windows=[106, 107, 108, 109])))
    stage.on_data(second)

    assert _as_list(second, "drift_rising_windows")[-1] >= 4
    assert stage.tracked_entities == 1


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    message = ControlMessage()
    message.payload(MessageMeta(get_df_class(config.execution_mode)(frame([1.0, 1.2]))))

    TC5DriftStage(config).on_data(message)

    assert _as_list(message.payload(), "drift_rising_windows") == [1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_tc5_drift_stage_pipe(config: Config):
    source_df = get_df_class(config.execution_mode)(frame(FLAT + CLIMB))

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC5DriftStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "drift_rising_windows")[-1] >= 4


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    # Matched against the stage's own message rather than just the column name: a KeyError raised deep inside a
    # column read would satisfy a looser assertion while telling an operator nothing about which stage wanted
    # what, and this is the wording that lists what the frame did carry.
    payload = frame([1.0])
    del payload["window_id"]

    with pytest.raises(KeyError, match="TC5DriftStage requires columns.*window_id"):
        run(config, payload)


def _at_warning(caplog: pytest.LogCaptureFixture, work) -> str:
    """Run `work` with warnings captured off the emitting module's logger, which does not propagate."""
    import logging

    from morpheus.stages.telemetry import tc5_drift_stage

    tc5_drift_stage.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING):
            work()
    finally:
        tc5_drift_stage.logger.removeHandler(caplog.handler)

    return caplog.text


@pytest.mark.gpu_and_cpu_mode
def test_several_rows_for_one_window_are_reported(config: Config, caplog: pytest.LogCaptureFixture):
    # The trajectory counts windows, so a frame carrying one row per authentication rather than one row per
    # window reads four authentications in an afternoon as a four-window run -- exactly the shape R-P-L5-006
    # fires on. The stage cannot aggregate on the operator's behalf without guessing which score to keep, so it
    # says what it saw. Silence here would make the rule fire on a batching choice.
    payload = frame([1.0, 1.4, 1.9, 2.5], windows=[100, 100, 100, 100])

    assert "repeating an entity's window" in _at_warning(caplog, lambda: run(config, payload))


@pytest.mark.gpu_and_cpu_mode
def test_one_row_per_window_is_not_reported(config: Config, caplog: pytest.LogCaptureFixture):
    # The negative control. A warning that fires on well-formed frames is a warning operators learn to ignore.
    assert "repeating an entity's window" not in _at_warning(caplog, lambda: run(config, frame(CLIMB)))


@pytest.mark.cpu_mode
def test_the_scoreless_warning_is_said_once_rather_than_per_batch(config: Config, caplog: pytest.LogCaptureFixture):
    # This fork ships with nothing producing mean_abs_z, so the scoreless branch is the branch every deployment
    # without a model takes on every batch forever. Saying it once is what keeps it a notice rather than the
    # log, and a notice drowned in its own repetition is not a notice.
    stage = TC5DriftStage(config)
    payload = frame([1.0, 1.2])
    del payload["mean_abs_z"]

    def two_batches():
        for _ in range(2):
            stage.on_data(MessageMeta(get_df_class(config.execution_mode)(dict(payload))))

    assert _at_warning(caplog, two_batches).count("found no mean_abs_z column") == 1


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError, match="max_windows must be at least 2"):
        TC5DriftStage(config, max_windows=1)


@pytest.mark.cpu_mode
def test_an_out_of_order_window_does_not_join_the_trajectory(config: Config):
    stage = TC5DriftStage(config)

    first = MessageMeta(get_df_class(config.execution_mode)(frame([1.0, 1.2], windows=[110, 111])))
    stage.on_data(first)

    late = MessageMeta(get_df_class(config.execution_mode)(frame([9.0], windows=[100])))
    stage.on_data(late)

    after = MessageMeta(get_df_class(config.execution_mode)(frame([1.5], windows=[112])))
    stage.on_data(after)

    assert _as_list(after, "drift_rising_windows") == [3]
