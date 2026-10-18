# Copyright (c) 2023-2026, NVIDIA CORPORATION.
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

import csv
import io

import pandas as pd

from morpheus.io.utils import get_csv_reader
from morpheus.utils.type_aliases import DataFrameModule
from morpheus.utils.type_aliases import DataFrameType

TYPE_DICT = {
    "bool": "bool",
    "count": "int64",
    "int": "int64",
    "double": "float64",
    "time": "float64",
    "interval": "float64",
    "string": "str",
    "pattern": "str",
    "port": "int64",
    "addr": "str",
    "subnet": "str",
    "enum": "str",
    "function": "str",
    "event": "str",
    "hook": "str",
    "file": "str",
    "opaque": "str",
    "any": "str",
}


def parse(filepath: str, df_type: DataFrameModule = "cudf") -> DataFrameType:
    """
    Parse Zeek log file and return cuDF dataframe. Uses header comments to get column names/types
    and configure parser.

    Parameters
    ----------
    filepath : str
        File path of Zeek log file
    df_type : DataFrameTypeStr, default 'cudf'
        Type of dataframe to return. Either 'cudf' or 'pandas'

    Returns
    -------
    DataFrameType
        Parsed Zeek log dataframe
    """
    if (df_type == "pandas"):
        return _parse_pandas(filepath)

    csv_reader = get_csv_reader(df_type)
    header_gdf = csv_reader(filepath, names=["line"], nrows=8)
    lines_gdf = header_gdf["line"].str.split()

    column_names = lines_gdf.iloc[6][1:]
    column_types = lines_gdf.iloc[7][1:]
    column_dtypes = list(map(lambda x: TYPE_DICT.get(x, "str"), column_types))

    log_gdf = csv_reader(
        filepath,
        delimiter="\t",
        dtype=column_dtypes,
        names=column_names,
        skiprows=8,
        skipfooter=1,
    )
    return log_gdf


PANDAS_TYPE_DICT = {**TYPE_DICT, "count": "Int64", "int": "Int64", "port": "Int64", "bool": "boolean"}
"""Nullable types for pandas, since Zeek writes its unset marker in any column and int64 cannot hold it."""


def _parse_pandas(filepath: str) -> pd.DataFrame:
    """
    Parse a Zeek ASCII log with pandas, reading the header the way Zeek writes it.

    The cuDF path reads the header as one CSV column, which pandas refuses on the `#set_separator\t,` line Zeek
    writes into every log, and reads the body with fixed integer types and a dropped last line, which fail on the
    unset marker and lose a row of a log Zeek has not yet closed. Here the header is read line by line, the body
    is every line that is not a `#` directive, the `#unset_field` marker is a null, `T` and `F` are booleans, and
    the counts are nullable.
    """
    directives = {}
    body = []

    with open(filepath, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")

            if (line.lstrip().startswith("#")):
                parts = line.split(None, 1)
                directives[parts[0].lstrip()] = parts[1] if len(parts) > 1 else ""
            elif (line):
                body.append(line)

    names = directives["#fields"].split()
    types = directives["#types"].split()
    unset = directives.get("#unset_field", "-").strip()
    dtypes = {name: PANDAS_TYPE_DICT.get(kind, "str") for (name, kind) in zip(names, types)}

    frame = pd.read_csv(
        io.StringIO("\n".join(body) + "\n"),
        sep="\t",
        names=names,
        dtype={name: ("str" if kind in ("Int64", "boolean") else kind)
               for (name, kind) in dtypes.items()},
        na_values=[unset],
        keep_default_na=False,
        quoting=csv.QUOTE_NONE)

    for (name, kind) in dtypes.items():
        if (kind == "boolean"):
            frame[name] = frame[name].map({"T": True, "F": False, "True": True, "False": False}).astype("boolean")
        elif (kind == "Int64"):
            frame[name] = pd.to_numeric(frame[name]).astype("Int64")

    return frame
