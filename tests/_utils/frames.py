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
What the fork's stage tests do to a frame, written once (guide 13 plan item 6).

Each stage test drove its stage the same way -- a payload into a `MessageMeta` of the execution mode's frame type,
`on_data`, and the result back as pandas -- and read a column back as a list with its nulls as `None`. Twenty-five
files carried their own copy of one of those, in four spellings of the null rule. These are the one copy.
"""

import math

import pandas as pd

from morpheus.compositions.host_frame import truthy  # noqa: F401 pylint: disable=unused-import
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.utils.type_utils import get_df_class


def pandas_frame(meta: MessageMeta) -> pd.DataFrame:
    """A message's frame as pandas, whichever execution mode built it."""
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


def drive(config: Config, stage, payload: dict) -> pd.DataFrame:
    """`payload` through `stage.on_data` once, as the execution mode's frame type, and the result as pandas."""
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    stage.on_data(meta)

    return pandas_frame(meta)


def without_row(payload: dict, position: int) -> dict:
    """The same frame with one row left out, which is what a refused row must be indistinguishable from."""
    return {
        name: [value for (index, value) in enumerate(values) if index != position]
        for (name, values) in payload.items()
    }


def is_null(value) -> bool:
    """Whether a cell is a gap however the frame spelled it: `None`, `pd.NA`, `NaT` or a float NaN."""
    return value is None or value is pd.NA or value is pd.NaT or (isinstance(value, float) and math.isnan(value))


def cell(value):
    """A cell with its gap as `None`."""
    return None if is_null(value) else value


def column_values(meta: MessageMeta, column: str, nulls: bool = True) -> list:
    """
    One column of a message as a list. With `nulls`, every gap is `None`; without, the values are as the frame holds
    them, for a test that asserts what a gap is.
    """
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [cell(value) for value in series.tolist()] if nulls else series.tolist()
