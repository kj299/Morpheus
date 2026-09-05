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
The device-to-host conversion, which is where the two execution modes stopped agreeing.

A GPU run of the composed telemetry pipeline rendered `arp_count_in_window` as `3.0` where the CPU golden holds
`3`. Nothing raised. cuDF's `to_pandas` defaults to `nullable=False`, which cannot put a null inside an integer
column, so it widens the column to float64 and writes NaN -- and every windowed count in this fork is null on the
rows belonging to other telemetry classes.

The behaviour is only observable on a GPU, but the rule is not: what has to hold is that the conversion asks for
nullable dtypes. That is assertable on any machine against a stand-in, which is what the first test here does, so
a change that drops the argument fails in ordinary CI rather than waiting for someone to find a GPU.
"""

import pandas as pd

from morpheus.utils.column_assign import to_host
from morpheus.utils.column_assign import to_host_list


class _StandInDeviceFrame:
    """Records how `to_host` asks for its conversion. A real cuDF frame needs a GPU; the contract does not."""

    def __init__(self, converted="converted"):
        self.converted = converted
        self.calls: list[dict] = []

    def to_pandas(self, **kwargs):
        self.calls.append(kwargs)

        return self.converted


def test_a_device_frame_is_converted_asking_for_nullable_dtypes():
    # The whole defect in one assertion. Without `nullable=True` an Int64 column holding 3 comes back as 3.0, the
    # golden comparison fails on a cell nobody would think to look at, and every digest taken on a device frame
    # differs from the same digest taken on a host frame.
    frame = _StandInDeviceFrame()

    assert to_host(frame) == "converted"
    assert frame.calls == [{"nullable": True}]


def test_a_host_frame_is_returned_untouched():
    # Which is why this change cannot move the CPU path: on a pandas frame the conversion does not happen at all.
    frame = pd.DataFrame({"a": [1, 2, 3]})

    assert to_host(frame) is frame


def test_to_host_list_reads_through_the_same_conversion():
    # The stages read their input columns through this. On a device frame it used to hand them floats where the
    # CPU path handed them integers, which is the same defect one layer further in.
    frame = pd.DataFrame({"count": pd.array([3, None, 5], dtype="Int64")})
    values = to_host_list(frame, "count")

    assert values[0] == 3 and values[2] == 5
    assert pd.isna(values[1])
    assert not isinstance(values[0], float), "a count must not come back as a float"
