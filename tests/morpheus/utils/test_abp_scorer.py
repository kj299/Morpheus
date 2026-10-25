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
The CPU booster against the upstream file it was converted from, and against an evaluation that uses no XGBoost.

`conversion_check.json` holds 220 rows XGBoost 1.7.6 scored from the original legacy-format `abp-pcap-xgb.bst`:
the 20 feature rows upstream's own test recorded for its packet capture sample, and 200 rows drawn across the
features' ranges so both classes appear. The scorer must reproduce every margin and probability exactly. A tree walk
over the JSON's arrays checks the same numbers without XGBoost, so a defect in how XGBoost reads the file would not
pass by agreeing with itself.
"""

import json
import math
import os

import numpy as np
import pytest

from morpheus.utils.abp_scorer import AbpScorer
from morpheus.utils.model_inventory import REPOSITORY_INVENTORY
from morpheus.utils.model_inventory import load_entry

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INVENTORY = os.path.join(REPO_ROOT, REPOSITORY_INVENTORY)


@pytest.fixture(name="entry", scope="module")
def entry_fixture():
    return load_entry(INVENTORY, "abp-pcap-xgb")


@pytest.fixture(name="scorer", scope="module")
def scorer_fixture(entry) -> AbpScorer:
    return AbpScorer(entry)


@pytest.fixture(name="check", scope="module")
def check_fixture(entry) -> list:
    with open(entry.files["conversion_check"], encoding="utf-8") as handle:
        document = json.load(handle)

    return document["rows"] + document["probes"]["rows"]


def walk(model: dict, row: np.ndarray) -> float:
    """The margin of one row, summed from each tree's leaf in float32 as XGBoost does, without XGBoost."""
    learner = model["learner"]
    base = float(learner["learner_model_param"]["base_score"])
    margin = np.float32(math.log(base / (1.0 - base)))

    for tree in learner["gradient_booster"]["model"]["trees"]:
        node = 0

        while (tree["left_children"][node] != -1):
            value = row[tree["split_indices"][node]]

            if (np.isnan(value)):
                left = bool(tree["default_left"][node])
            else:
                left = value < np.float32(tree["split_conditions"][node])

            node = tree["left_children"][node] if left else tree["right_children"][node]

        margin = np.float32(margin + np.float32(tree["split_conditions"][node]))

    return float(margin)


def test_the_converted_booster_reproduces_the_upstream_file_exactly(scorer: AbpScorer, check: list):
    matrix = np.array([row["features"] for row in check], dtype=np.float32)

    assert len(check) == 220
    assert (scorer.margins(matrix) == np.array([row["margin"] for row in check], dtype=np.float32)).all()
    assert (scorer.probabilities(matrix) == np.array([row["probability"] for row in check], dtype=np.float32)).all()


def test_both_classes_are_in_the_check(scorer: AbpScorer, check: list):
    # Upstream's recorded sample is all one class; a check that never crosses the threshold cannot tell a scorer
    # that inverted it from one that did not.
    verdicts = {row["probability"] > scorer.threshold for row in check}

    assert verdicts == {True, False}


def test_a_tree_walk_without_xgboost_gives_the_same_margins(entry, scorer: AbpScorer, check: list):
    with open(entry.model_path, encoding="utf-8") as handle:
        model = json.load(handle)

    matrix = np.array([row["features"] for row in check], dtype=np.float32)
    walked = np.array([walk(model, row) for row in matrix], dtype=np.float32)

    np.testing.assert_allclose(walked, scorer.margins(matrix), rtol=0, atol=1e-5)


def test_a_missing_value_takes_each_split_s_default_branch(entry, scorer: AbpScorer, check: list):
    with open(entry.model_path, encoding="utf-8") as handle:
        model = json.load(handle)

    row = np.array(check[0]["features"], dtype=np.float32)
    row[5] = np.nan

    assert scorer.margins(row[np.newaxis, :])[0] == pytest.approx(walk(model, row), abs=1e-5)


def test_the_matrix_is_built_by_feature_name_with_none_as_missing(scorer: AbpScorer):
    columns = {name: [float(index), None] for (index, name) in enumerate(scorer.features)}
    matrix = scorer.matrix(columns)

    assert matrix.dtype == np.float32
    assert matrix.shape == (2, 13)
    assert list(matrix[0]) == [float(index) for index in range(13)]
    assert np.isnan(matrix[1]).all()


def test_a_feature_without_a_column_is_refused(scorer: AbpScorer):
    columns = {name: [1.0] for name in scorer.features[:-1]}

    with pytest.raises(KeyError, match="fin/all"):
        scorer.matrix(columns)


def test_a_matrix_of_the_wrong_width_is_refused(scorer: AbpScorer):
    with pytest.raises(ValueError, match="13 features"):
        scorer.probabilities(np.zeros((1, 12), dtype=np.float32))
