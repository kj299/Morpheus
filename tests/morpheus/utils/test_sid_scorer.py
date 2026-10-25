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
The CPU SID scorer against the validation set the model shipped with, and the properties the guide claims for it.

The model file and the validation set are stored in Git LFS. The workflow fetches both before running the tiers, so
in CI an unfetched file fails these tests; on a developer checkout without them they are skipped, with the command
that fetches them.
"""

import csv
import os

import numpy as np
import pytest

from morpheus.utils.model_inventory import REPOSITORY_INVENTORY
from morpheus.utils.model_inventory import ModelFileUnavailable
from morpheus.utils.model_inventory import load_entry
from morpheus.utils.sid_scorer import SidScorer
from morpheus.utils.sid_scorer import windows

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INVENTORY = os.path.join(REPO_ROOT, REPOSITORY_INVENTORY)
FILLER = " ".join(["the weather report said clouds again today"] * 50)
KEY = "aws_secret_access_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"


def sid_entry():  # pylint: disable=inconsistent-return-statements
    """The SID entry, or a skip naming the fetch command, or a failure in CI where the fetch is part of the job."""
    try:
        return load_entry(INVENTORY, "sid-minibert-20230424")
    except ModelFileUnavailable as error:
        if (os.environ.get("CI")):
            pytest.fail(f"the workflow fetches the inventory's LFS files before the tiers, and {error}")

        pytest.skip(str(error))


@pytest.fixture(name="entry", scope="module")
def entry_fixture():
    return sid_entry()


@pytest.fixture(name="scorer", scope="module")
def scorer_fixture(entry) -> SidScorer:
    return SidScorer(entry)


@pytest.fixture(name="validation", scope="module")
def validation_fixture(entry) -> tuple:
    with open(entry.files["validation"], encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    return (rows[0], rows[1:])


def test_a_sequence_that_fits_is_one_window():
    assert windows(list(range(10)), 256, 64, False) == [list(range(10))]


def test_a_long_sequence_is_read_in_overlapping_windows_to_its_end():
    ids = list(range(300))
    result = windows(ids, 256, 64, False)

    assert [window[0] for window in result] == [0, 64]
    assert result[-1][-1] == 299
    assert all(len(window) <= 256 for window in result)


def test_truncation_keeps_only_the_first_window():
    assert windows(list(range(300)), 256, 64, True) == [list(range(256))]


def test_the_labels_are_the_validation_set_s_columns_in_order(scorer: SidScorer, validation: tuple):
    (header, _) = validation

    assert [f"si_{label}" for label in scorer.labels] == header[2:12]


@pytest.mark.slow
def test_the_validation_set_is_decided_as_the_inventory_records(entry, scorer: SidScorer, validation: tuple):
    (_, rows) = validation
    recorded = entry.settings["validation"]
    probabilities = np.array(scorer.probabilities([row[1] for row in rows]))
    truth = np.array([[value == "True" for value in row[2:12]] for row in rows])
    wrong = (probabilities > scorer.threshold) != truth

    assert len(rows) == recorded["rows"]
    assert sum(len(scorer.windows(scorer.token_ids(row[1]))) > 1
               for row in rows) == recorded["rows_longer_than_max_length"]
    assert int(wrong.sum()) == recorded["labels_wrong"]
    assert int(wrong.any(axis=1).sum()) == recorded["rows_wrong"]


def test_a_window_scores_the_same_alone_and_in_any_batch(scorer: SidScorer, validation: tuple):
    (_, rows) = validation
    pieces = [scorer.windows(scorer.token_ids(row[1]))[0] for row in rows[:40]]
    batched = scorer.logits(pieces)
    alone = np.concatenate([scorer.logits([piece]) for piece in pieces])
    shuffled = scorer.logits(pieces[::-1])[::-1]

    assert (batched == alone).all()
    assert (batched == shuffled).all()


def test_a_key_past_the_first_window_is_found(scorer: SidScorer):
    ids = scorer.token_ids(f"{FILLER} {KEY}")
    secret_keys = scorer.labels.index("secret_keys")
    first_window_only = 1.0 / (1.0 + np.exp(-float(scorer.logits([ids[:256]])[0][secret_keys])))

    assert len(ids) > 256
    assert first_window_only < scorer.threshold
    assert scorer.probabilities([f"{FILLER} {KEY}"])[0][secret_keys] > scorer.threshold


def test_text_with_no_tokens_is_not_scored(scorer: SidScorer):
    result = scorer.probabilities(["", None, "   ", KEY])

    assert result[:3] == [None, None, None]
    assert result[3][scorer.labels.index("secret_keys")] > scorer.threshold
