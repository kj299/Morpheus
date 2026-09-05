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
Bringing a device frame to the host without turning its counts into decimals.

The two harnesses collect their output by converting whatever the sink produced into a pandas frame, then compare
that rendering against a checked-in golden. In GPU mode the conversion loses something: cuDF's `to_pandas` cannot
put a null inside an integer column, so it widens the column to float64 and writes NaN. A count of 3 arrives as
`3.0`, and the byte comparison against a CPU-produced golden fails on a cell that is not wrong so much as
differently spelled. Every windowed count in this fork is null on the rows belonging to other telemetry classes,
so this reaches most of the integer columns rather than an unlucky few.

Asking `to_pandas` for nullable dtypes instead is the obvious repair and it is the wrong one. It changes how
*every* null comes back, not only those in integer columns: object columns start yielding `pandas.NA` where they
used to yield `None`, and stage code that tests `value is None` stops recognising a missing value. Measured on a
GPU, that turned three failures into nine, all of them null-handling tests across the ARP, auth, binding-resolver
and lineage-stamp stages. The conversion is left alone here for that reason.

What is repaired is narrower and local to the comparison: the device frame is asked which columns were integers
before the conversion, and any of those that came back as floats are cast to pandas `Int64`, which is exactly the
dtype the CPU path produces natively for the same column. Nothing else is touched, so every other column renders
as it did before.
"""

import pandas as pd

INTEGER_KINDS = ("i", "u")


def to_host_frame(df):
    """
    Copy a frame to the host, restoring integer columns the conversion widened.

    Parameters
    ----------
    df : `pandas.DataFrame` or `cudf.DataFrame`
        The frame to bring to the host. A host frame is returned unchanged, so this cannot move the CPU path.

    Returns
    -------
    `pandas.DataFrame`
    """
    if (not hasattr(df, "to_pandas")):
        return df

    was_integer = {name: dtype.kind in INTEGER_KINDS for (name, dtype) in df.dtypes.items()}
    host = df.to_pandas()

    for (name, integral) in was_integer.items():
        if (integral and host[name].dtype.kind == "f"):
            host[name] = host[name].astype("Int64")

    return host


def restore_integer_columns(host: pd.DataFrame, was_integer: dict) -> pd.DataFrame:
    """The cast on its own, so it can be asserted without a device frame."""
    for (name, integral) in was_integer.items():
        if (integral and host[name].dtype.kind == "f"):
            host[name] = host[name].astype("Int64")

    return host
