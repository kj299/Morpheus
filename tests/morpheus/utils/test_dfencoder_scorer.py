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

Most of what is asserted here is about the adapter: that it aligns rows, applies the absolute value, refuses a
version it does not hold, and refuses rows shaped differently from what the model was trained on. The model there
is a stub that records what it was asked and answers arithmetically.

The rest is about the committed model format: that `NumpyAutoEncoder` scores a fixed row to numbers worked out by
hand, that a model's version is a digest of every number it scores with, and that a file whose numbers no longer
match the version beside them is refused. Where Torch and `morpheus.models.dfencoder` import, the numpy forward
pass is also checked against the upstream class it reimplements.
"""

import copy
import json
import math
import sys

import numpy as np
import pandas as pd
import pytest

from morpheus.utils.dfencoder_scorer import MODEL_FORMAT
from morpheus.utils.dfencoder_scorer import DfencoderScorer
from morpheus.utils.dfencoder_scorer import NumpyAutoEncoder
from morpheus.utils.dfencoder_scorer import build_manifest
from morpheus.utils.dfencoder_scorer import export_model
from morpheus.utils.dfencoder_scorer import load_models
from morpheus.utils.dfencoder_scorer import load_torch_autoencoder
from morpheus.utils.dfencoder_scorer import model_digest
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
        self.call_sizes = []

    def get_results(self, df: pd.DataFrame, return_abs: bool = False) -> pd.DataFrame:
        del return_abs
        self.received = df
        self.call_sizes.append(len(df))
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
    # The guard is per model call, so the call is given both rows at once: at one row per call there is no
    # shorter answer than the question.
    scorer = DfencoderScorer({VERSION: StubModel(rows_returned=1)}, FEATURES, rows_per_call=2)

    with pytest.raises(ValueError, match="returned 1 rows for 2"):
        scorer.score(VERSION, rows())


def test_a_model_without_a_loss_per_feature_is_refused():
    scorer = DfencoderScorer({VERSION: StubModel(drop_column="mfa_ratio")}, FEATURES)

    with pytest.raises(ValueError, match="mfa_ratio_z_loss"):
        scorer.score(VERSION, rows())


def test_a_row_is_scored_on_its_own_however_many_came_with_it():
    # Control 5 at the adapter's boundary. TC5ScoreStage hands over the rows an entity has in the message it is
    # holding, so the group's size is a fact about the batching, not about the data. The model must not see it.
    model = StubModel()
    scorer = DfencoderScorer({VERSION: model}, FEATURES)

    alone = scorer.score(VERSION, rows()[:1])
    together = scorer.score(VERSION, rows())

    assert model.call_sizes == [1, 1, 1]
    assert together[0] == alone[0]


def test_a_larger_batch_is_asked_for_in_fixed_sized_calls():
    model = StubModel()
    scorer = DfencoderScorer({VERSION: model}, FEATURES, rows_per_call=2)

    scorer.score(VERSION, rows() + rows() + rows()[:1])

    assert model.call_sizes == [2, 2, 1]


def test_a_batch_size_below_one_is_refused():
    with pytest.raises(ValueError, match="positive integer"):
        DfencoderScorer({VERSION: StubModel()}, FEATURES, rows_per_call=0)

    with pytest.raises(ValueError, match="positive integer"):
        DfencoderScorer({VERSION: StubModel()}, FEATURES, rows_per_call=True)


def test_no_rows_score_to_no_rows():
    assert not DfencoderScorer({VERSION: StubModel()}, FEATURES).score(VERSION, [])


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


def hand_model() -> dict:
    """
    A two-feature model small enough to score by hand.

    One encoder unit that adds the two standardized features, through a relu, and an output layer that copies
    that sum back to both. Every weight is exactly representable in float32, so the hand arithmetic below is the
    arithmetic the forward pass does.
    """
    return {
        "format": MODEL_FORMAT,
        "features": ["logcount", "locincrement"],
        "layers": [{
            "stack": "encoder", "activation": "relu", "weight": [[1.0, 1.0]], "bias": [0.0]
        }],
        "output": {
            "weight": [[1.0], [1.0]], "bias": [0.0, 0.0]
        },
        "numeric_scaler": {
            "logcount": {
                "mean": 4.0, "std": 2.0
            }, "locincrement": {
                "mean": 1.0, "std": 1.0
            }
        },
        "loss_scaler": {
            "logcount": {
                "mean": 0.5, "std": 0.25
            }, "locincrement": {
                "mean": 0.25, "std": 0.5
            }
        },
    }


def test_a_fixed_row_scores_to_the_numbers_worked_out_by_hand():
    # logcount 8 standardizes to (8 - 4) / 2 = 2 and locincrement 3 to (3 - 1) / 1 = 2. The encoder sums them
    # to 4, which the relu passes, and the output copies 4 back to both. The squared errors are (4 - 2)^2 = 4 for
    # each, and against the training losses they are (4 - 0.5) / 0.25 = 14 and (4 - 0.25) / 0.5 = 7.5.
    model = NumpyAutoEncoder(hand_model())
    results = model.get_results(pd.DataFrame({"logcount": [8.0], "locincrement": [3.0]}), return_abs=True)

    assert results["logcount_loss"].iloc[0] == 4.0
    assert results["locincrement_loss"].iloc[0] == 4.0
    assert results["logcount_z_loss"].iloc[0] == 14.0
    assert results["locincrement_z_loss"].iloc[0] == 7.5
    assert results["max_abs_z"].iloc[0] == 14.0
    assert results["mean_abs_z"].iloc[0] == 10.75


def test_the_relu_clamps_and_the_sign_survives_unless_asked_for_the_absolute():
    # Both features at their training means standardize to zero; the sum is zero and the reconstruction is
    # zero, so each loss is zero and sits below its training mean -- a negative z that the absolute flips.
    model = NumpyAutoEncoder(hand_model())
    frame = pd.DataFrame({"logcount": [4.0], "locincrement": [1.0]})

    assert model.get_results(frame)["logcount_z_loss"].iloc[0] == -2.0
    assert model.get_results(frame, return_abs=True)["logcount_z_loss"].iloc[0] == 2.0

    # Both well below their means: the sum is negative, the relu returns zero, and the reconstruction is zero.
    below = model.get_results(pd.DataFrame({"logcount": [0.0], "locincrement": [0.0]}))
    assert below["logcount_loss"].iloc[0] == 4.0
    assert below["locincrement_loss"].iloc[0] == 1.0


def test_the_numpy_model_slots_behind_the_adapter():
    version = f"dfencoder/alice@example.com:{model_digest(hand_model())}"
    scorer = DfencoderScorer({version: NumpyAutoEncoder(hand_model())}, ["logcount", "locincrement"])

    assert scorer.score(version, [{"logcount": 8.0, "locincrement": 3.0}]) == [{"logcount": 14.0, "locincrement": 7.5}]


def test_a_gap_is_refused_rather_than_filled():
    model = NumpyAutoEncoder(hand_model())

    with pytest.raises(ValueError, match="null or non-finite"):
        model.get_results(pd.DataFrame({"logcount": [float("nan")], "locincrement": [1.0]}))


def test_the_digest_covers_every_number_the_score_depends_on():
    base = model_digest(hand_model())

    assert len(base) == 16
    assert model_digest(copy.deepcopy(hand_model())) == base

    # A weight, a training statistic of the inputs and one of the losses each change the score, so each must
    # change the version. The JSON round trip changes nothing, because the values are written exactly.
    for (path, value) in ((("layers", 0, "weight", 0, 0), 1.5), (("numeric_scaler", "logcount", "std"), 3.0),
                          (("loss_scaler", "locincrement", "mean"), 0.3), (("output", "bias", 1), 0.125)):
        changed = hand_model()
        target = changed

        for key in path[:-1]:
            target = target[key]

        target[path[-1]] = value
        assert model_digest(changed) != base, path

    assert model_digest(json.loads(json.dumps(hand_model()))) == base


def test_a_document_that_does_not_chain_or_is_another_format_is_refused():
    wrong = hand_model()
    wrong["format"] = "something-else/1"

    with pytest.raises(ValueError, match="this reads"):
        NumpyAutoEncoder(wrong)

    wrong = hand_model()
    wrong["output"]["weight"] = [[1.0, 1.0], [1.0, 1.0]]

    with pytest.raises(ValueError, match="output layer"):
        NumpyAutoEncoder(wrong)

    wrong = hand_model()
    wrong["layers"][0]["activation"] = "tanh"

    with pytest.raises(ValueError, match="not one this evaluates"):
        NumpyAutoEncoder(wrong)


def write_models(path, entries: dict) -> str:
    target = path / "models.json"
    target.write_text(json.dumps({"models": entries}), encoding="utf-8")

    return str(target)


def test_a_committed_file_loads_under_the_version_its_numbers_digest_to(tmp_path):
    version = f"dfencoder/alice@example.com:{model_digest(hand_model())}"
    entries = {"alice@example.com": {"model_version": version, "model": hand_model()}}
    loaded = load_models(write_models(tmp_path, entries))

    assert list(loaded) == ["alice@example.com"]
    assert loaded["alice@example.com"][0] == version


def test_a_file_edited_after_training_or_moved_between_principals_is_refused(tmp_path):
    version = f"dfencoder/alice@example.com:{model_digest(hand_model())}"
    edited = hand_model()
    edited["output"]["bias"] = [0.5, 0.0]

    with pytest.raises(ValueError, match="changed after it was trained"):
        load_models(write_models(tmp_path, {"alice@example.com": {"model_version": version, "model": edited}}))

    with pytest.raises(ValueError, match="changed after it was trained"):
        load_models(write_models(tmp_path, {"bob@example.com": {"model_version": version, "model": hand_model()}}))


def _dfencoder_or_skip():
    pytest.importorskip("torch")

    try:
        from morpheus.models.dfencoder import AutoEncoder  # pylint: disable=import-outside-toplevel
    except Exception as error:  # pylint: disable=broad-exception-caught
        # The package imports cuDF first, which raises on a machine with Torch but no CUDA driver.
        pytest.skip(f"morpheus.models.dfencoder does not import here: {error}")

    return AutoEncoder


def test_the_numpy_forward_pass_agrees_with_the_upstream_model():
    _dfencoder_or_skip()

    rng = np.random.default_rng(7)
    features = ["logcount", "locincrement", "mfa_ratio"]
    frame = pd.DataFrame(rng.normal(size=(24, 3)) * [2.0, 0.5, 0.1] + [4.0, 1.0, 0.9], columns=features)
    frame["locincrement"] = 1.0
    trained = train_dfencoder_models({"alice@example.com": frame},
                                     features,
                                     seed=42,
                                     epochs=3,
                                     eval_batch_size=8,
                                     device="cpu")
    version = trained.versions["alice@example.com"]
    model = trained.models[version]
    document = json.loads(json.dumps(export_model(model, features)))

    assert version.endswith(f":{model_digest(document)}")

    probe = frame.iloc[:6].copy()
    probe.loc[probe.index[0], "locincrement"] = 3.0
    expected = model.get_results(probe, return_abs=True)
    reloaded = load_torch_autoencoder(document).get_results(probe, return_abs=True)
    numpy_results = NumpyAutoEncoder(document).get_results(probe, return_abs=True)

    for name in features:
        column = f"{name}_z_loss"
        # The reloaded upstream model is the same model: the committed numbers lose nothing.
        assert reloaded[column].tolist() == expected[column].tolist()

        # The numpy pass is float64 against Torch's float32, so they agree to float32's precision.
        for (ours, theirs) in zip(numpy_results[column], expected[column]):
            assert math.isclose(ours, float(theirs), rel_tol=1e-5, abs_tol=1e-5), (name, ours, theirs)
