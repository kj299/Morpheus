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
Collecting a frame without letting the concatenation decide what an integer is.

Both harnesses collect their output by converting whatever each sink produced into a pandas frame and
concatenating the lot, then comparing that rendering against a checked-in golden. In GPU mode the rendering
differed from the CPU golden on `arp_count_in_window`: `3.0` against `3`.

The conversion is not where it happens, which is what made the first repair miss. A telemetry class's frame carries
only its own columns, so the concatenation has to fill that column with gaps for every other class's rows -- and
what pandas fills with depends on the dtype it starts from. `assign_nullable_int_column` gives the CPU path a
nullable `Int64`, which survives the fill. cuDF's `to_pandas` gives a plain numpy `int64` whenever that frame has
no gaps of its own, and a numpy integer column cannot hold one, so the concatenation widens it to float64 and every
count in it grows a decimal point.

So the fix is not a better conversion, it is refusing to let the concatenation choose. Integer columns are carried
as nullable integers in both modes, before anything is joined, which is what the CPU path already did by accident
of `assign_nullable_int_column` and what the GPU path now does deliberately. Columns of every other kind are left
exactly as they were.

One repair was tried before this and was worse: asking `to_pandas` for types that can hold a gap. It fixes integer
columns and changes how *every* gap returns to the host, so object columns yield `pandas.NA` where they yielded
`None` and stage code testing `value is None` stops recognising a missing value. Measured on a GPU it turned three
failures into nine. The conversion itself is therefore left alone.
"""

import pandas as pd

INTEGER_KINDS = ("i", "u")


def to_host_frame(df):
    """
    Copy a frame to the host and carry its integer columns as nullable integers.

    Parameters
    ----------
    df : `pandas.DataFrame` or `cudf.DataFrame`
        The frame to collect. A device frame is converted first; a host frame is used as it stands.

    Returns
    -------
    `pandas.DataFrame`
        The same data, with every integer column held as pandas `Int64` so that a later concatenation cannot
        widen it. Applied in both execution modes, because a rule that only one mode follows is the defect.
    """
    integer_columns = [name for (name, dtype) in df.dtypes.items() if dtype.kind in INTEGER_KINDS]
    host = df.to_pandas() if hasattr(df, "to_pandas") else df

    return as_nullable_integers(host, integer_columns)


def as_nullable_integers(host: pd.DataFrame, columns) -> pd.DataFrame:
    """The cast on its own, so the rule can be asserted without a device frame."""
    for name in columns:
        host[name] = host[name].astype("Int64")

    return host
