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
"""
The adapter that puts the autoencoder behind `TC5ScoreStage`, tested where no autoencoder can run.

Everything asserted here is about the adapter: that it aligns rows, applies the absolute value, refuses a version
it does not hold, and refuses rows shaped differently from what the model was trained on. The model is a stub
that records what it was asked and answers arithmetically. The model's own numbers are measured by
`examples/layer5_model/run_model.py` on the machine that has the card.
"""

import sys

import pandas as pd
import pytest

from morpheus.utils.dfencoder_scorer import DfencoderScorer
from morpheus.utils.dfencoder_scorer import build_manifest
from morpheus.utils.dfencoder_scorer import train_dfencoder_models

FEATURES = ["logcount", "locincrement", "mfa_ratio"]
VERSION = "dfencoder/alice@example.com:0123456789abcdef"


class StubModel:
    """Answers with each feature times a signed factor, so alignment, order and the absolute value are visible."""

    def __init__(self, factor: float = -2.0, rows_returned=None, drop_column: str = None):
        self.factor = factor
        self.rows_returned = rows_returned
        self.drop_column = drop_column
        self.received = None

    def get_results(self, df: pd.DataFrame, return_abs: bool = False) -> pd.DataFrame:
        del return_abs
        self.received = df
        result = pd.DataFrame({f"{name}_z_loss": df[name] * self.factor for name in df.columns})

        if (self.drop_column is not None):
            result = result.drop(columns=[f"{self.drop_column}_z_loss"])

        if (self.rows_returned is not None):
            result = result.iloc[:self.rows_returned]

        return result


def rows() -> list:
    return [
        {
            "logcount": 3.0, "locincrement": 1.0, "mfa_ratio": 0.5
        },
        {
            "logcount": 5.0, "locincrement": 2.0, "mfa_ratio": 1.0
        },
    ]


def test_scores_are_the_absolute_losses_aligned_to_the_rows():
    model = StubModel(factor=-2.0)
    scorer = DfencoderScorer({VERSION: model}, FEATURES)

    scored = scorer.score(VERSION, rows())

    assert scored == [
        {
            "logcount": 6.0, "locincrement": 2.0, "mfa_ratio": 1.0
        },
        {
            "logcount": 10.0, "locincrement": 4.0, "mfa_ratio": 2.0
        },
    ]


def test_the_model_is_asked_in_training_column_order_whatever_the_row_order():
    # A row dict in a different key order is the same row. The frame the model sees is in the order it was
    # trained on, as float64, or the losses would land on the wrong features.
    model = StubModel()
    scorer = DfencoderScorer({VERSION: model}, FEATURES)

    scorer.score(VERSION, [{"mfa_ratio": 0.5, "logcount": 3.0, "locincrement": 1.0}])

    assert list(model.received.columns) == FEATURES
    assert str(model.received["logcount"].dtype) == "float64"
    assert model.received["logcount"].iloc[0] == 3.0


def test_a_version_the_scorer_does_not_hold_is_refused_by_name():
    scorer = DfencoderScorer({VERSION: StubModel()}, FEATURES)

    with pytest.raises(KeyError, match="dfencoder/bob@example.com:ffff"):
        scorer.score("dfencoder/bob@example.com:ffff", rows())


def test_a_row_with_the_wrong_features_is_refused():
    scorer = DfencoderScorer({VERSION: StubModel()}, FEATURES)

    with pytest.raises(ValueError, match="carries"):
        scorer.score(VERSION, [{"logcount": 3.0, "locincrement": 1.0}])

    with pytest.raises(ValueError, match="carries"):
        scorer.score(VERSION, [{"logcount": 3.0, "locincrement": 1.0, "mfa_ratio": 0.5, "extra": 1.0}])


def test_a_model_that_drops_rows_is_refused():
    scorer = DfencoderScorer({VERSION: StubModel(rows_returned=1)}, FEATURES)

    with pytest.raises(ValueError, match="returned 1 rows for 2"):
        scorer.score(VERSION, rows())


def test_a_model_without_a_loss_per_feature_is_refused():
    scorer = DfencoderScorer({VERSION: StubModel(drop_column="mfa_ratio")}, FEATURES)

    with pytest.raises(ValueError, match="mfa_ratio_z_loss"):
        scorer.score(VERSION, rows())


def test_no_rows_score_to_no_rows():
    assert DfencoderScorer({VERSION: StubModel()}, FEATURES).score(VERSION, []) == []


def test_the_scorer_refuses_to_be_built_wrong():
    with pytest.raises(ValueError, match="at least one fitted model"):
        DfencoderScorer({}, FEATURES)

    with pytest.raises(ValueError, match="not pinned"):
        DfencoderScorer({"dfencoder/alice": StubModel()}, FEATURES)

    with pytest.raises(ValueError, match="no get_results"):
        DfencoderScorer({VERSION: object()}, FEATURES)

    with pytest.raises(ValueError, match="feature_columns"):
        DfencoderScorer({VERSION: StubModel()}, [])


def test_versions_are_listed_sorted():
    other = "dfencoder/bob@example.com:0000000000000000"
    scorer = DfencoderScorer({VERSION: StubModel(), other: StubModel()}, FEATURES)

    assert scorer.versions == sorted([VERSION, other])


def test_training_refuses_without_torch_in_plain_words():
    # The adapter does not need Torch; training does. Forced rather than depended on: a `None` entry in
    # `sys.modules` makes the import raise here whether or not the machine has Torch, which is the same
    # discipline the runner's refusal tests follow.
    saved = sys.modules.get("torch", "absent")
    sys.modules["torch"] = None

    try:
        with pytest.raises(ImportError, match="needs Torch"):
            train_dfencoder_models({"alice@example.com": pd.DataFrame({name: [1.0] * 4
                                                                       for name in FEATURES})},
                                   FEATURES,
                                   seed=42,
                                   epochs=1,
                                   eval_batch_size=8)
    finally:
        if (saved == "absent"):
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = saved


def test_the_manifest_pins_every_principal_to_its_own_model_and_nobody_else():
    versions = {"alice@example.com": VERSION, "bob@example.com": "dfencoder/bob@example.com:0000000000000000"}
    manifest = build_manifest(versions, window_id=0)

    assert manifest.resolve("alice@example.com", 0).model_version == VERSION
    assert manifest.resolve("alice@example.com", 0).fallback_used is False

    with pytest.raises(ValueError, match="no fallback"):
        manifest.resolve("carol@example.com", 0)

    with pytest.raises(ValueError, match="pins nobody"):
        build_manifest({}, window_id=0)
