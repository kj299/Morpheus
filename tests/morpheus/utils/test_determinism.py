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

from morpheus.utils.determinism import canonicalize
from morpheus.utils.determinism import diff_frames
from morpheus.utils.determinism import frame_digest
from morpheus.utils.determinism import permute_within_contiguous_groups
from morpheus.utils.determinism import quantize_value


def test_quantize_value_uses_half_even():
    # Half-even rounding is the whole point: .5 cases go to the even neighbor on both sides.
    assert quantize_value(0.00005, 4) == 0.0
    assert quantize_value(0.00015, 4) == 0.0002
    assert quantize_value(1.23456, 4) == 1.2346


def test_quantize_value_passes_nan_through():
    assert pd.isna(quantize_value(float("nan")))


def test_canonicalize_is_order_and_column_order_insensitive():
    left = pd.DataFrame({"uid": ["b", "a"], "value": [2, 1]})
    right = pd.DataFrame({"value": [1, 2], "uid": ["a", "b"]}, index=[7, 9])

    pd.testing.assert_frame_equal(canonicalize(left, ["uid"]), canonicalize(right, ["uid"]))


def test_canonicalize_quantizes_floats():
    # The values differ at four decimal places (0.1234 versus 0.1235) but agree at three (0.123).
    noisy = pd.DataFrame({"uid": ["a"], "score": [0.12341]})
    clean = pd.DataFrame({"uid": ["a"], "score": [0.12349]})

    assert diff_frames(canonicalize(noisy, ["uid"]), canonicalize(clean, ["uid"])) is not None
    assert diff_frames(canonicalize(noisy, ["uid"], float_decimals=3), canonicalize(clean, ["uid"],
                                                                                    float_decimals=3)) is None


def test_canonicalize_drops_ignored_columns():
    df = pd.DataFrame({"uid": ["a"], "processing_time": [123]})

    assert "processing_time" not in canonicalize(df, ["uid"], ignore_columns=["processing_time"]).columns


def test_canonicalize_rejects_missing_or_ambiguous_keys():
    df = pd.DataFrame({"uid": ["a", "a"], "value": [1, 2]})

    with pytest.raises(ValueError, match="not present"):
        canonicalize(df, ["nope"])

    with pytest.raises(ValueError, match="ambiguous"):
        canonicalize(df, ["uid"])

    with pytest.raises(ValueError, match="key column"):
        canonicalize(df, [])


def test_diff_frames_reports_the_first_disagreement():
    left = canonicalize(pd.DataFrame({"uid": ["a", "b"], "value": [1, 2]}), ["uid"])
    right = canonicalize(pd.DataFrame({"uid": ["a", "b"], "value": [1, 3]}), ["uid"])

    description = diff_frames(left, right)

    assert "value" in description
    assert "2" in description and "3" in description


def test_diff_frames_reports_shape_differences():
    left = canonicalize(pd.DataFrame({"uid": ["a"], "value": [1]}), ["uid"])

    assert "Column sets differ" in diff_frames(left, left.rename(columns={"value": "other"}))
    assert "Row counts differ" in diff_frames(left, left.iloc[0:0])


def test_diff_frames_treats_matching_nulls_as_equal():
    left = canonicalize(pd.DataFrame({"uid": ["a"], "value": [None]}), ["uid"])
    right = canonicalize(pd.DataFrame({"uid": ["a"], "value": [None]}), ["uid"])

    assert diff_frames(left, right) is None


def test_frame_digest_is_stable_and_sensitive():
    df = canonicalize(pd.DataFrame({"uid": ["a", "b"], "value": [1, 2]}), ["uid"])
    changed = canonicalize(pd.DataFrame({"uid": ["a", "b"], "value": [1, 9]}), ["uid"])

    assert frame_digest(df) == frame_digest(df.copy())
    assert frame_digest(df) != frame_digest(changed)


def test_permute_shuffles_only_within_contiguous_runs():
    df = pd.DataFrame({"row": list(range(6))})
    groups = [0, 0, 0, 1, 0, 0]

    result = permute_within_contiguous_groups(df, groups, seed=1)

    # The run boundaries are preserved: the first three rows stay first, row 3 stays put, the last two stay last.
    assert sorted(result["row"][0:3]) == [0, 1, 2]
    assert result["row"][3] == 3
    assert sorted(result["row"][4:6]) == [4, 5]


def test_permute_is_reproducible_and_seed_sensitive():
    df = pd.DataFrame({"row": list(range(20))})
    groups = [0] * 20

    first = permute_within_contiguous_groups(df, groups, seed=7)
    second = permute_within_contiguous_groups(df, groups, seed=7)
    different = permute_within_contiguous_groups(df, groups, seed=8)

    pd.testing.assert_frame_equal(first, second)
    assert not first["row"].equals(different["row"])


def test_permute_rejects_length_mismatch():
    with pytest.raises(ValueError):
        permute_within_contiguous_groups(pd.DataFrame({"row": [1]}), [0, 1], seed=1)
