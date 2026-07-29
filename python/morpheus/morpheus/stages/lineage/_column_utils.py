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
Column helpers shared by the lineage stages.

Both stages hash on the host regardless of execution mode, so that an identifier never depends on whether a GPU or CPU
pipeline produced it. These helpers move values across that boundary.
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
