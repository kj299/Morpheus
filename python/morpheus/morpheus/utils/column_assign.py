# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Column helpers for stages that compute on the host regardless of execution mode.

Several stages here deliberately do their work on the host so that a value never depends on whether a GPU or CPU
pipeline produced it: an identifier that differs by execution mode defeats its purpose, and so does a column whose
dtype does. These helpers move values across that boundary and write them back with a type that is the same in
both modes.
"""

import math

import pandas as pd

from morpheus.utils.type_aliases import DataFrameType


def _as_integers(values: list) -> list:
    """
    Coerce a list bound for an integer column to integers, leaving every flavour of missing as `None`.

    The values reaching this module are often derived rather than read: a counter delta, a flap count, an attempt
    tally. Whatever they were computed from may have arrived as a float -- reading a column that holds a gap gives
    floats on the host, because a plain integer column cannot hold one -- and a whole number that happens to be
    carried in a float is still a whole number.

    pandas coerces such a list on its own when the target type is stated. cuDF does not, so the column that was
    asked to be an integer becomes a float, and the same count renders as `2` in one execution mode and `2.0` in
    the other. Coercing here rather than trusting either constructor is what makes the two agree, and it is what
    the function's own name promises.
    """
    coerced = []

    for value in values:
        if (value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA):
            coerced.append(None)
        else:
            coerced.append(int(value))

    return coerced


def to_host_list(df: DataFrameType, column: str) -> list:
    """
    Return a column's values as a host-side Python list, copying from device memory when necessary.

    Parameters
    ----------
    df : `pandas.DataFrame` or `cudf.DataFrame`
        Frame to read from.
    column : str
        Column name.

    Returns
    -------
    list
        The column's values as Python objects.
    """
    series = df[column]

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return series.tolist()


def assign_str_column(df: DataFrameType, column: str, values: list):
    """
    Write a host-side list of string values into `df`, matching the DataFrame's own type.

    Parameters
    ----------
    df : `pandas.DataFrame` or `cudf.DataFrame`
        Frame to write to.
    column : str
        Column name. Overwritten if it already exists.
    values : list
        One value per row. `None` entries become nulls.
    """
    # Imported here so that this module remains importable in CPU-only environments where cuDF is absent.
    from morpheus.utils.type_utils import is_cudf_type

    if (is_cudf_type(df)):
        import cudf
        df[column] = cudf.Series(values, index=df.index, dtype="str")
    else:
        df[column] = pd.Series(values, index=df.index, dtype="object")


def assign_nullable_int_column(df: DataFrameType, column: str, values: list):
    """
    Write a host-side list of integers that may contain nulls, as a nullable integer column in both modes.

    Assigning such a list directly would produce a different dtype per execution mode: pandas widens to float64 and
    represents the gaps as NaN, while cuDF keeps int64 with a null mask. A count that is a float in one mode and an
    integer in the other is the same class of defect as an identifier that differs by mode, and it surfaces far
    downstream, so the nullable integer type is selected explicitly here.

    Parameters
    ----------
    df : `pandas.DataFrame` or `cudf.DataFrame`
        Frame to write to.
    column : str
        Column name. Overwritten if it already exists.
    values : list
        One value per row. `None` entries become nulls.
    """
    # Imported here so that this module remains importable in CPU-only environments where cuDF is absent.
    from morpheus.utils.type_utils import is_cudf_type

    values = _as_integers(values)

    if (is_cudf_type(df)):
        import cudf
        df[column] = cudf.Series(values, index=df.index, dtype="int64")
    else:
        df[column] = pd.Series(pd.array(values, dtype="Int64"), index=df.index)


def assign_nullable_float_column(df: DataFrameType, column: str, values: list):
    """
    Write a host-side list of floats that may contain nulls, as a nullable float column in both modes.

    Both modes settle on float64 here, but they disagree on what a gap is: pandas stores `None` as NaN while cuDF
    stores a real null. That difference survives all the way to the wire, where NaN is not valid JSON and null is,
    so the gap is made an explicit null in both.

    Parameters
    ----------
    df : `pandas.DataFrame` or `cudf.DataFrame`
        Frame to write to.
    column : str
        Column name. Overwritten if it already exists.
    values : list
        One value per row. `None` entries become nulls.
    """
    # Imported here so that this module remains importable in CPU-only environments where cuDF is absent.
    from morpheus.utils.type_utils import is_cudf_type

    if (is_cudf_type(df)):
        import cudf
        df[column] = cudf.Series(values, index=df.index, dtype="float64")
    else:
        df[column] = pd.Series(pd.array(values, dtype="Float64"), index=df.index)


def assign_nullable_bool_column(df: DataFrameType, column: str, values: list):
    """
    Write a host-side list of booleans that may contain nulls, as a nullable boolean column in both modes.

    A null here means "not answerable", which is a different claim from `False`. Assigning such a list directly
    would let pandas widen to object and store `None`, so a rule reading the column as a boolean would coerce that
    `None` into `False` and count a question nobody could answer as a negative answer.

    Parameters
    ----------
    df : `pandas.DataFrame` or `cudf.DataFrame`
        Frame to write to.
    column : str
        Column name. Overwritten if it already exists.
    values : list
        One value per row. `None` entries become nulls.
    """
    # Imported here so that this module remains importable in CPU-only environments where cuDF is absent.
    from morpheus.utils.type_utils import is_cudf_type

    if (is_cudf_type(df)):
        import cudf
        df[column] = cudf.Series(values, index=df.index, dtype="bool")
    else:
        df[column] = pd.Series(pd.array(values, dtype="boolean"), index=df.index)


def to_host_frame(df: DataFrameType) -> pd.DataFrame:
    """
    Return a whole frame on the host, keeping integer columns integral across the copy.

    A stage that has to do its work in pandas has to bring the frame across the device boundary and then send the
    result back, and that round trip is not type-preserving in one direction. A device integer column that holds a
    null cannot be represented by a plain host integer column, so the copy widens it to a float and turns the null
    into a NaN; converting the result back to a device frame keeps the float. Nothing in the round trip announces
    the change, and what surfaces at the far end is a count that reads `2.0` on a GPU run and `2` on a CPU one.

    Naming the nullable host integer type on the way across keeps the column integral, so the frame that goes back
    to the device holds the type it left with. A frame that is already on the host is returned as a shallow copy,
    untouched: it never lost the type in the first place, and coercing it would move the CPU mode instead of
    meeting it.

    Parameters
    ----------
    df : `pandas.DataFrame` or `cudf.DataFrame`
        Frame to copy to the host.

    Returns
    -------
    `pandas.DataFrame`
        The frame on the host, with every integer column readable as an integer.
    """
    if (not hasattr(df, "to_pandas")):
        return df.copy(deep=False)

    integer_columns = [name for (name, dtype) in df.dtypes.items() if getattr(dtype, "kind", "O") in ("i", "u")]
    host = df.to_pandas()

    for name in integer_columns:
        host[name] = host[name].astype("Int64")

    return host
