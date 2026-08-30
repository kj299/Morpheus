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
GPU-native lineage identifier hashing.

The host-side hashing in `morpheus.utils.lineage` is the reference: identifiers must never depend on which execution
mode produced them. This module provides a cuDF implementation of the same digests via
`Series.hash_values(method="sha256")`, for GPU pipelines where the host hop dominates the lineage stamp's cost.

Because the whole value of the identifiers rests on cross-implementation agreement, this path is *gated*, not
trusted: `verify_digest_equivalence` hashes a set of probe vectors through both implementations and raises on any
disagreement, and `LineageStampStage` runs that gate before the first GPU-hashed batch. If a future cuDF changes its
byte semantics, the pipeline fails closed instead of silently minting identifiers nothing else can reproduce.

The equivalence argument only holds where the two implementations demonstrably render values identically, so the GPU
path accepts only string and integer identifier columns and refuses nulls. Booleans, floats, and timestamps render
differently between Python and cuDF (`str(True)` is `"True"`, cuDF renders `True`); rather than hash a divergence,
those columns are rejected with instructions to pre-render them as strings or use the host path.

This module imports without cuDF present; the cuDF import happens inside the functions that need it.
"""

import typing

from morpheus.utils.lineage import DEFAULT_DIGEST_LENGTH
from morpheus.utils.lineage import UNIT_SEPARATOR
from morpheus.utils.lineage import event_uid
from morpheus.utils.lineage import link_uid

_PROBE_ROWS = [
    # The unit-separator boundary case: these two must not collide, and must match the host digests exactly.
    ("ab", "c", 0),
    ("a", "bc", 0),
    # Empty strings, unicode outside ASCII, and a long value.
    ("", "x", 1),
    ("collector-π", "TC-5/2.1.0", 42),
    ("y" * 4096, "z", -7),
    # A realistic envelope row.
    ("collector-a", "TC-5/2.1.0", 9000000000),
]


def _coerce_id_column(series, column: str):
    """
    Render one identifier column as strings, byte-identically to the host path's `str(value)`.

    Raises
    ------
    ValueError
        If the column's dtype is not string or integer, or if it contains nulls. Both would make the GPU digest
        diverge from the host digest, which is precisely what this path must never do.
    """
    from cudf.api import types as cudf_types

    if (series.isnull().any()):
        raise ValueError(f"GPU hashing requires non-null values in id column {column!r}. Fill or drop nulls "
                         "upstream, or use the host hashing path.")

    dtype = series.dtype

    if (cudf_types.is_string_dtype(dtype)):
        return series

    if (cudf_types.is_integer_dtype(dtype)):
        return series.astype("str")

    raise ValueError(f"GPU hashing supports string and integer id columns; column {column!r} has dtype {dtype}. "
                     "Booleans, floats, and timestamps render differently between Python and cuDF, so hashing them "
                     "here would mint identifiers the host path cannot reproduce. Pre-render the column as a string "
                     "or use the host hashing path.")


def event_uid_series_cudf(df, id_columns: typing.Sequence[str], digest_length: int = DEFAULT_DIGEST_LENGTH):
    """
    Compute `event_uid` for every row of a cuDF DataFrame, on the device.

    Produces digests byte-identical to `morpheus.utils.lineage.event_uid_series` for string and integer columns;
    the equivalence is enforced by `verify_digest_equivalence`, which callers must run before trusting this path.

    Parameters
    ----------
    df : `cudf.DataFrame`
        Frame holding the identifier columns.
    id_columns : list of str
        Columns feeding the identifier, in a significant order.
    digest_length : int, default = 32
        Number of hexadecimal characters retained from each SHA-256 digest.

    Returns
    -------
    `cudf.Series`
        One truncated hex digest per row.
    """
    if (len(id_columns) == 0):
        raise ValueError("At least one id column is required")

    if (not 1 <= digest_length <= 64):
        raise ValueError(f"digest_length must be between 1 and 64, received {digest_length}")

    coerced = [_coerce_id_column(df[column], column) for column in id_columns]

    joined = coerced[0]
    if (len(coerced) > 1):
        joined = joined.str.cat(others=coerced[1:], sep=UNIT_SEPARATOR)

    return joined.hash_values(method="sha256").str.slice(0, digest_length)


def link_uid_series_cudf(parent_uids,
                         child_uids,
                         relation: str,
                         join_method: str,
                         digest_length: int = DEFAULT_DIGEST_LENGTH):
    """
    Compute `link_uid` for a batch of parent-child edges, on the device.

    Matches `morpheus.utils.lineage.link_uid_series`: a null or empty parent marks a chain root and yields a null
    edge rather than an error.

    Parameters
    ----------
    parent_uids : `cudf.Series`
        Parent identifiers; nulls and empty strings mark chain roots.
    child_uids : `cudf.Series`
        Child identifiers, as produced by `event_uid_series_cudf`.
    relation : str
        Semantic relationship recorded on every edge.
    join_method : str
        How the edge was established.
    digest_length : int, default = 32
        Number of hexadecimal characters retained from each SHA-256 digest.

    Returns
    -------
    `cudf.Series`
        One truncated hex digest per row, null where the row is a chain root.
    """
    if (not 1 <= digest_length <= 64):
        raise ValueError(f"digest_length must be between 1 and 64, received {digest_length}")

    if (len(parent_uids) != len(child_uids)):
        raise ValueError(f"parent_uids and child_uids must be the same length, received {len(parent_uids)} and "
                         f"{len(child_uids)}")

    parents = parent_uids.astype("str")
    root_mask = parents.isnull() | (parents.str.len() == 0).fillna(True)

    suffix = f"{UNIT_SEPARATOR}{relation}{UNIT_SEPARATOR}{join_method}"
    joined = parents.fillna("").str.cat(others=[child_uids], sep=UNIT_SEPARATOR) + suffix

    digests = joined.hash_values(method="sha256").str.slice(0, digest_length)

    return digests.where(~root_mask, None)


def verify_digest_equivalence(digest_length: int = DEFAULT_DIGEST_LENGTH) -> None:
    """
    Prove, on this machine and this cuDF, that the GPU digests match the host digests.

    Hashes the probe vectors through both implementations, covering the separator boundary case, empty strings,
    non-ASCII text, long values, and negative integers, for both `event_uid` and `link_uid` including the
    chain-root null. Cheap enough to run at pipeline startup, which is where `LineageStampStage` runs it.

    Raises
    ------
    RuntimeError
        On the first disagreement, naming both digests. A pipeline receiving this error must not fall back
        silently; identifiers that only one implementation can reproduce are worse than a crash.
    """
    import cudf

    df = cudf.DataFrame({
        "a": [row[0] for row in _PROBE_ROWS],
        "b": [row[1] for row in _PROBE_ROWS],
        "n": [row[2] for row in _PROBE_ROWS],
    })

    gpu_uids = event_uid_series_cudf(df, ["a", "b", "n"], digest_length=digest_length).to_pandas().tolist()

    for (position, row) in enumerate(_PROBE_ROWS):
        host_uid = event_uid(*row, digest_length=digest_length)

        if (gpu_uids[position] != host_uid):
            raise RuntimeError(f"GPU digest disagrees with host digest for probe row {row!r}: "
                               f"{gpu_uids[position]!r} versus {host_uid!r}. The GPU hashing path must not be used "
                               "on this cuDF version.")

    parents = cudf.Series([gpu_uids[0], None, ""])
    children = cudf.Series([gpu_uids[1], gpu_uids[2], gpu_uids[3]])
    gpu_links = link_uid_series_cudf(parents, children, "carried_by", "hard:flow_id",
                                     digest_length=digest_length).to_pandas().tolist()

    expected_link = link_uid(gpu_uids[0], gpu_uids[1], "carried_by", "hard:flow_id", digest_length=digest_length)

    if (gpu_links[0] != expected_link or gpu_links[1] is not None or gpu_links[2] is not None):
        raise RuntimeError(f"GPU link digest disagrees with host link digest: {gpu_links!r} versus "
                           f"[{expected_link!r}, None, None]. The GPU hashing path must not be used on this cuDF "
                           "version.")
