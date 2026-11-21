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

import numpy as np
import pandas as pd
import pytest

from morpheus.utils.column_assign import as_nullable_integers
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import to_host_frame
from morpheus.utils.column_assign import truthy


def test_host_frame_leaves_a_pandas_frame_alone():
    frame = pd.DataFrame({"count": pd.array([1, None, 3], dtype="Int64"), "name": ["a", "b", "c"]})

    host = to_host_frame(frame)

    assert host["count"].dtype == "Int64"
    assert host["name"].tolist() == ["a", "b", "c"]


def test_host_frame_does_not_coerce_a_host_float():
    # A float column is a float column in both modes; the helper repairs the round trip, it does not retype data.
    frame = pd.DataFrame({"ratio": [0.5, None, 1.5]})

    assert to_host_frame(frame)["ratio"].dtype == "float64"


@pytest.mark.gpu_mode
def test_host_frame_keeps_a_device_integer_integral():
    import cudf

    frame = cudf.DataFrame({"name": ["a", "b", "c"]})
    assign_nullable_int_column(frame, "count", [1, None, 3])

    assert frame["count"].dtype == "int64", "precondition: cuDF holds the gap in an integer column"

    host = to_host_frame(frame)

    assert host["count"].dtype == "Int64"
    assert host["count"].tolist() == [1, pd.NA, 3]

    # The point of the type is that it survives the trip back, so the frame returns to the device as it left.
    assert cudf.DataFrame(host)["count"].dtype == "int64"


@pytest.mark.gpu_mode
def test_host_frame_without_the_helper_widens():
    # The negative control: this is the conversion the helper exists to replace.
    import cudf

    frame = cudf.DataFrame({"name": ["a", "b", "c"]})
    assign_nullable_int_column(frame, "count", [1, None, 3])

    assert frame.to_pandas()["count"].dtype == "float64"


def test_collecting_for_a_concatenation_casts_a_host_frame_too():
    # A stage that sends its frame back must not retype one that never left the host, so the default leaves a
    # numpy integer column alone. Code that concatenates what it collects asks for the cast in both modes.
    frame = pd.DataFrame({
        "count": np.array([1, 2, 3], dtype="int64"), "name": ["a", "b", "c"], "ratio": [0.5, 1.0, 1.5]
    })

    assert to_host_frame(frame)["count"].dtype == "int64"

    collected = to_host_frame(frame, nullable_ints=True)

    assert collected["count"].dtype == "Int64"
    assert collected["ratio"].dtype == "float64"
    assert collected["name"].tolist() == ["a", "b", "c"]
    assert frame["count"].dtype == "int64", "the frame passed in is not written to"


def test_a_collected_integer_column_survives_the_fill_of_a_concatenation():
    counts = pd.DataFrame({"count": np.array([3], dtype="int64")})
    other_class = pd.DataFrame({"event_time": [1, 2]})

    assert pd.concat([counts, other_class], ignore_index=True)["count"].dtype.kind == "f"

    carried = pd.concat([to_host_frame(counts, nullable_ints=True), other_class], ignore_index=True)

    assert carried["count"].dtype == "Int64"
    assert carried["count"].tolist() == [3, pd.NA, pd.NA]


def test_the_cast_alone_names_its_columns_and_returns_the_frame():
    frame = pd.DataFrame({"count": np.array([1, 2], dtype="int64"), "other": np.array([3, 4], dtype="int64")})

    assert as_nullable_integers(frame, ["count"]) is frame
    assert frame["count"].dtype == "Int64"
    assert frame["other"].dtype == "int64"


@pytest.mark.gpu_mode
def test_collecting_a_device_frame_gives_the_same_types_as_a_host_one():
    import cudf

    payload = {"count": [1, 2, 3], "name": ["a", "b", "c"]}
    on_host = to_host_frame(pd.DataFrame(payload), nullable_ints=True)
    from_device = to_host_frame(cudf.DataFrame(payload), nullable_ints=True)

    assert from_device["count"].dtype == on_host["count"].dtype == "Int64"
    assert from_device.to_csv(index=False) == on_host.to_csv(index=False)


@pytest.mark.parametrize("value,expected",
                         [(True, True), (False, False), (pd.NA, False), (None, False), (float("nan"), False), (1, True),
                          (0, False)])
def test_a_null_flag_reads_as_not_fired(value, expected: bool):
    assert truthy(value) is expected
