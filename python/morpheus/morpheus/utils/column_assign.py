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

from morpheus.utils.type_aliases import DataFrameType


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
        import pandas as pd
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

    if (is_cudf_type(df)):
        import cudf
        df[column] = cudf.Series(values, index=df.index, dtype="int64")
    else:
        import pandas as pd
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
        import pandas as pd
        df[column] = pd.Series(pd.array(values, dtype="Float64"), index=df.index)
