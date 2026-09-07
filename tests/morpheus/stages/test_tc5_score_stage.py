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
from morpheus.stages.telemetry.tc5_score_stage import TC5ScoreStage
from morpheus.utils.model_manifest import ModelManifest
from morpheus.utils.type_utils import get_df_class

ALICE = "alice@example.com"
BOB = "bob@example.com"
FEATURES = ["logcount", "mfa_ratio"]

MANIFEST = ModelManifest(window_id=7, models={ALICE: "dfp-alice:3", BOB: "dfp-bob:2"}, fallback="dfp-generic:1")


class RecordingScorer:
    """Returns the feature value itself as its own z-score, and records how it was called.

    Deliberately trivial arithmetic: this exercises the plumbing, and a stub that computed something clever
    would make a failing assertion ambiguous between the stage and the stub.
    """

    def __init__(self, multiplier: float = 1.0):
        self.calls: list = []
        self._multiplier = multiplier

    def score(self, model_version: str, features: list) -> list:
        self.calls.append((model_version, len(features)))

        return [{name: value * self._multiplier for (name, value) in row.items()} for row in features]


def frame(principals: list, logcounts: list, ratios: list, windows: list = None) -> dict:
    payload = {"user_principal": principals, "logcount": logcounts, "mfa_ratio": ratios}

    if (windows is not None):
        payload["window_id"] = windows

    return payload


def run(config: Config, payload: dict, scorer=None, manifest=None, **kwargs) -> pd.DataFrame:
    stage = TC5ScoreStage(config,
                          scorer=scorer if scorer is not None else RecordingScorer(),
                          manifest=manifest if manifest is not None else MANIFEST,
                          feature_columns=FEATURES,
                          **kwargs)
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    stage.on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_the_columns_four_rules_have_been_waiting_for_are_emitted(config: Config):
    # The whole point. Until now nothing produced these, so R-B-L5-001, R-B-L5-002, R-B-L5-005 and R-P-L5-006
    # read columns that did not exist.
    result = run(config, frame([ALICE, ALICE], [2.0, 4.0], [0.5, 0.25]))

    assert list(result["mean_abs_z"]) == [1.25, 2.125]
    assert list(result["max_abs_z"]) == [2.0, 4.0]
    assert list(result["logcount_z_loss"]) == [2.0, 4.0]
    assert list(result["mfa_ratio_z_loss"]) == [0.5, 0.25]


@pytest.mark.gpu_and_cpu_mode
def test_the_mean_and_the_maximum_are_derived_rather_than_asked_for(config: Config):
    # The scorer returns per-feature losses only. Deriving the summaries here is what stops a model reporting a
    # mean that disagrees with the losses printed beside it -- a discrepancy no rule could detect and every
    # analyst would trust.
    result = run(config, frame([ALICE], [6.0], [0.0]))

    assert result["mean_abs_z"].iloc[0] == 3.0
    assert result["max_abs_z"].iloc[0] == 6.0


@pytest.mark.gpu_and_cpu_mode
def test_a_negative_loss_is_scored_by_its_magnitude(config: Config):
    # They are absolute z-scores. A feature two deviations below its own mean is as far from ordinary as one two
    # above, and signing the summary would let the two cancel.
    result = run(config, frame([ALICE], [-4.0], [2.0]))

    assert result["max_abs_z"].iloc[0] == 4.0
    assert result["mean_abs_z"].iloc[0] == 3.0


@pytest.mark.gpu_and_cpu_mode
def test_each_entity_is_scored_against_its_own_pinned_model(config: Config):
    scorer = RecordingScorer()
    run(config, frame([ALICE, BOB, ALICE], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]), scorer=scorer)

    assert scorer.calls == [("dfp-alice:3", 2), ("dfp-bob:2", 1)]


@pytest.mark.gpu_and_cpu_mode
def test_an_entity_with_no_model_of_its_own_gets_the_declared_fallback(config: Config):
    scorer = RecordingScorer()
    run(config, frame(["carol@example.com"], [1.0], [1.0]), scorer=scorer)

    assert scorer.calls == [("dfp-generic:1", 1)]


@pytest.mark.gpu_and_cpu_mode
def test_entities_are_scored_in_a_fixed_order(config: Config):
    # A scorer holding shared state -- a Torch global generator being the obvious one -- would otherwise produce
    # output depending on which entity happened to come first, which depends on how the stream was batched.
    first = RecordingScorer()
    second = RecordingScorer()

    run(config, frame([ALICE, BOB], [1.0, 1.0], [1.0, 1.0]), scorer=first)
    run(config, frame([BOB, ALICE], [1.0, 1.0], [1.0, 1.0]), scorer=second)

    assert first.calls == second.calls


@pytest.mark.gpu_and_cpu_mode
def test_a_row_with_a_gap_in_a_feature_carries_no_score(config: Config):
    # Null rather than zero. A zero says the entity looked exactly average, which is a confident claim; a gap is
    # the absence of any claim, and the drift trajectory downstream treats the two very differently.
    result = run(config, frame([ALICE, ALICE], [2.0, None], [0.5, 0.5]))

    assert result["mean_abs_z"].iloc[0] == 1.25
    assert pd.isna(result["mean_abs_z"].iloc[1])
    assert pd.isna(result["logcount_z_loss"].iloc[1])


@pytest.mark.gpu_and_cpu_mode
def test_a_row_with_no_entity_carries_no_score(config: Config):
    result = run(config, frame([None], [2.0], [0.5]))

    assert pd.isna(result["mean_abs_z"].iloc[0])


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("payload, ids",
                         [
                             (([ALICE, None], [1.0, 1.0], [1.0, 1.0]), "no_entity"),
                             (([ALICE, ALICE], [1.0, None], [1.0, 1.0]), "gap_in_a_feature"),
                         ],
                         ids=["no_entity", "gap_in_a_feature"])
def test_an_unscorable_row_is_never_handed_to_the_model(config: Config, payload, ids):
    # Not merely that it comes back null -- a NaN feature would propagate to a null through the arithmetic on
    # its own, so the output cannot tell the two apart. What matters is that the model is never asked. A NaN
    # reaching a Torch forward pass is its own failure, and one that tends to surface as a whole batch of NaN
    # rather than as an error at the row that caused it.
    del ids
    scorer = RecordingScorer()
    run(config, frame(*payload), scorer=scorer)

    assert scorer.calls == [("dfp-alice:3", 1)]


@pytest.mark.gpu_and_cpu_mode
def test_scores_do_not_depend_on_how_the_rows_are_batched(config: Config):
    # Control 5, at the stage rather than at the model. Two batches must give what one batch gave, or every
    # threshold tuned against these scores is tuned against the batching.
    whole = run(config, frame([ALICE, BOB, ALICE], [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]))

    stage = TC5ScoreStage(config, scorer=RecordingScorer(), manifest=MANIFEST, feature_columns=FEATURES)
    pieces = []

    for payload in (frame([ALICE, BOB], [1.0, 2.0], [0.1, 0.2]), frame([ALICE], [3.0], [0.3])):
        meta = MessageMeta(get_df_class(config.execution_mode)(payload))
        stage.on_data(meta)
        df = meta.copy_dataframe()
        pieces.append(df.to_pandas() if hasattr(df, "to_pandas") else df)

    split = pd.concat(pieces, ignore_index=True)

    assert list(whole["mean_abs_z"]) == list(split["mean_abs_z"])


@pytest.mark.gpu_and_cpu_mode
def test_a_row_from_another_window_is_refused(config: Config):
    # Control 1's whole point: a window scored against models pinned for a different one produces scores that
    # look entirely ordinary and are not comparable to anything.
    with pytest.raises(ValueError, match="resolved for window 7"):
        run(config, frame([ALICE], [1.0], [0.5], windows=[8]))


@pytest.mark.gpu_and_cpu_mode
def test_a_scorer_that_returns_the_wrong_number_of_rows_is_refused(config: Config):
    # Silently misaligned scores are worse than none: every row would carry a number belonging to a different
    # row, and nothing downstream could tell.
    class Truncating(RecordingScorer):

        def score(self, model_version: str, features: list) -> list:
            return super().score(model_version, features)[:-1]

    with pytest.raises(ValueError, match="returned 1 rows"):
        run(config, frame([ALICE, ALICE], [1.0, 2.0], [0.5, 0.5]), scorer=Truncating())


@pytest.mark.gpu_and_cpu_mode
def test_the_scores_are_quantized(config: Config):
    result = run(config, frame([ALICE], [1.0 / 3.0], [1.0 / 3.0]))

    assert result["mean_abs_z"].iloc[0] == 0.3333


@pytest.mark.gpu_and_cpu_mode
def test_the_summary_is_the_summary_of_what_was_published(config: Config):
    # Quantization happens before the summary, not after. Averaging the raw losses and rounding afterwards is
    # marginally more accurate and leaves the row internally inconsistent: an analyst averaging the loss columns
    # printed on the row would get a number the row does not show. One ten-thousandth of a deviation is a smaller
    # price than a row that disagrees with itself.
    scorer = RecordingScorer(multiplier=1.0 / 3.0)
    result = run(config, frame([ALICE], [1.0], [2.0]), scorer=scorer)
    losses = [result["logcount_z_loss"].iloc[0], result["mfa_ratio_z_loss"].iloc[0]]

    assert result["mean_abs_z"].iloc[0] == pytest.approx(sum(losses) / len(losses), abs=1e-9)
    assert result["max_abs_z"].iloc[0] == max(losses)


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    stage = TC5ScoreStage(config, scorer=RecordingScorer(), manifest=MANIFEST, feature_columns=FEATURES)
    message = ControlMessage()
    message.payload(MessageMeta(get_df_class(config.execution_mode)(frame([ALICE], [1.0], [0.5]))))

    assert stage.on_data(message) is message


@pytest.mark.gpu_and_cpu_mode
def test_tc5_score_stage_pipe(config: Config):
    payload = get_df_class(config.execution_mode)(frame([ALICE, BOB], [1.0, 2.0], [0.5, 0.25]))
    pipeline = LinearPipeline(config)
    pipeline.set_source(InMemorySourceStage(config, [payload]))
    pipeline.add_stage(TC5ScoreStage(config, scorer=RecordingScorer(), manifest=MANIFEST, feature_columns=FEATURES))
    sink = pipeline.add_stage(InMemorySinkStage(config))
    pipeline.run()

    assert len(sink.get_messages()) == 1


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = frame([ALICE], [1.0], [0.5])
    del payload["mfa_ratio"]

    with pytest.raises(KeyError, match="TC5ScoreStage requires columns.*mfa_ratio"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError, match="scorer is required"):
        TC5ScoreStage(config, scorer=None, manifest=MANIFEST, feature_columns=FEATURES)

    with pytest.raises(ValueError, match="manifest is required"):
        TC5ScoreStage(config, scorer=RecordingScorer(), manifest=None, feature_columns=FEATURES)

    with pytest.raises(ValueError, match="feature_columns must name at least one"):
        TC5ScoreStage(config, scorer=RecordingScorer(), manifest=MANIFEST, feature_columns=[])
