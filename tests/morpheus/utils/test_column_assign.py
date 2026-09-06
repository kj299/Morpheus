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

from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import to_host_frame


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
