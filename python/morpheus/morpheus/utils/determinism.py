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
Helpers for proving pipeline determinism in CI.

Determinism claims decay silently, so control 13 of the OSI behavioral analytics guide requires them to be enforced by
a harness: run the pipeline over a fixed corpus, canonicalize the output, and compare, across repeat runs, process
restarts, batch splits, and input permutations. This module supplies the comparison half of that harness. The pipeline
half belongs to the caller, since only the caller knows how to build and run its pipeline.

Canonicalization exists because a deterministic pipeline is allowed to emit the same rows in a different physical
arrangement: the row set is the contract, the emission order of otherwise-identical messages is not. `canonicalize`
reduces a DataFrame to a normal form, sorted rows, sorted columns, plain index, optionally quantized floats, and
`diff_frames` explains the first disagreement in terms a build log can show.

`sort_for_cumulative_features` is the one piece here that runs in the pipeline rather than in the harness. Control 8
requires a total row order to be established before any cumulative feature is computed, because those features
accumulate down the frame; without it the harness above has nothing to prove.
"""

import decimal
import hashlib
import random
import typing

import pandas as pd

DEFAULT_FLOAT_DECIMALS = 4
"""Decimal places used when quantizing float columns, matching determinism control 9's score quantum."""

DEFAULT_ORDER_COLUMNS = ("event_time", "collector_id", "collector_seq")
"""Total order for cumulative features, per determinism control 8.

`event_time` alone is insufficient because ties are common at second or millisecond resolution, and
`collector_id` with `collector_seq` breaks every remaining tie: the envelope requires `collector_seq` to be strictly
monotonic per collector, so the pair is unique on its own.
"""


def sort_for_cumulative_features(df: pd.DataFrame,
                                 order_columns: typing.Sequence[str] = DEFAULT_ORDER_COLUMNS,
                                 require_total_order: bool = True) -> pd.DataFrame:
    """
    Impose determinism control 8's total row order, which every cumulative feature depends on.

    `IncrementColumn` and `DistinctIncrementColumn` accumulate down the frame, so the value they produce for a row is
    a function of which rows preceded it. Two runs that see the same rows in different orders therefore produce
    different features and different scores. Nothing about the result looks wrong, which is what makes this the most
    easily missed defect in the pipeline: an unsorted batch does not raise, it silently answers a different question.

    The sort is stable, so rows that compare equal on every order column keep their relative order rather than being
    permuted arbitrarily. That is a weaker guarantee than it appears, because it makes the output a function of the
    input arrangement, which is exactly what determinism is supposed to eliminate. `require_total_order` is therefore
    on by default: ties mean the order columns do not identify a row, and the caller is told so rather than being
    handed a frame whose features depend on how the batch happened to arrive.

    The index is reset, both because the sorted position is the only meaningful one afterwards and because the
    cumulative primitives realign their output by index and need it to be unique.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Frame to order. A cuDF frame is accepted and copied to the host, since the cumulative primitives run there.
    order_columns : list of str, default = `DEFAULT_ORDER_COLUMNS`
        Sort key, in significant order.
    require_total_order : bool, default = True
        Raise when the order columns leave ties, rather than falling back on the input's own arrangement.

    Returns
    -------
    `pandas.DataFrame`
        The rows in total order, with a fresh index.

    Raises
    ------
    KeyError
        If an order column is absent.
    ValueError
        If `order_columns` is empty, or if `require_total_order` is set and the columns leave ties.
    """
    if (hasattr(df, "to_pandas")):
        df = df.to_pandas()

    order_columns = list(order_columns)

    if (len(order_columns) == 0):
        raise ValueError("At least one order column is required")

    missing = [column for column in order_columns if column not in df.columns]

    if (len(missing) > 0):
        raise KeyError(f"Order columns {missing} are not present in the DataFrame. "
                       f"Available columns: {sorted(df.columns)}")

    if (require_total_order and len(df) > 0):
        tied = int(df.duplicated(subset=order_columns).sum())

        if (tied > 0):
            raise ValueError(
                f"Order columns {order_columns} leave {tied} tied row(s), so they do not define a total order and "
                "any cumulative feature computed after this point would depend on the order the batch arrived in. "
                "Add a tie-breaking column, or pass require_total_order=False to accept the input's own order for "
                "ties.")

    return df.sort_values(order_columns, kind="mergesort").reset_index(drop=True)


def quantize_value(value: float, decimals: int = DEFAULT_FLOAT_DECIMALS) -> float:
    """
    Round a float with explicit half-even rounding.

    `ROUND_HALF_EVEN` is specified rather than assumed because the default rounding mode differs between platforms
    and libraries, which is precisely the class of variation this module exists to remove.

    Parameters
    ----------
    value : float
        Value to round.
    decimals : int, default = 4
        Decimal places to keep.

    Returns
    -------
    float
        The rounded value. NaN passes through unchanged.
    """
    if (pd.isna(value)):
        return value

    quantum = decimal.Decimal(1).scaleb(-decimals)

    return float(decimal.Decimal(repr(value)).quantize(quantum, rounding=decimal.ROUND_HALF_EVEN))


def canonicalize(df: pd.DataFrame,
                 key_columns: typing.Sequence[str],
                 ignore_columns: typing.Sequence[str] = (),
                 float_decimals: int = DEFAULT_FLOAT_DECIMALS) -> pd.DataFrame:
    """
    Reduce a DataFrame to a normal form in which two deterministic runs compare equal.

    Rows are sorted by the key columns with a stable sort, columns are sorted by name, the index is dropped, ignored
    columns are removed, and float columns are quantized. Anything two runs are permitted to legitimately vary in,
    such as a wall-clock `processing_time`, belongs in `ignore_columns`; everything else is part of the contract.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Frame to canonicalize. A cuDF frame is accepted and copied to the host.
    key_columns : list of str
        Columns that uniquely identify a row, used as the sort key. An `event_uid` is the natural choice.
    ignore_columns : list of str, default = ()
        Columns excluded from the comparison.
    float_decimals : int, default = 4
        Decimal places for float quantization, per determinism control 9.

    Returns
    -------
    `pandas.DataFrame`

    Raises
    ------
    ValueError
        If a key column is absent, or if the key columns do not uniquely identify every row, which would make the
        sort order, and therefore the comparison, ambiguous.
    """
    if (hasattr(df, "to_pandas")):
        df = df.to_pandas()

    if (len(key_columns) == 0):
        raise ValueError("At least one key column is required")

    missing = [column for column in key_columns if column not in df.columns]

    if (len(missing) > 0):
        raise ValueError(f"Key columns {missing} are not present in the DataFrame. "
                         f"Available columns: {sorted(df.columns)}")

    result = df.drop(columns=[column for column in ignore_columns if column in df.columns])

    if (len(result) > 0 and result.duplicated(subset=list(key_columns)).any()):
        raise ValueError(f"Key columns {list(key_columns)} do not uniquely identify every row; the canonical order "
                         "would be ambiguous. Add a tie-breaking column.")

    for column in result.columns:
        if (pd.api.types.is_float_dtype(result[column])):
            result[column] = result[column].map(lambda value: quantize_value(value, float_decimals))

    result = result.sort_values(list(key_columns), kind="mergesort")

    return result[sorted(result.columns)].reset_index(drop=True)


def diff_frames(left: pd.DataFrame, right: pd.DataFrame) -> typing.Optional[str]:
    """
    Describe the first difference between two canonicalized frames, or return `None` when they are equal.

    Parameters
    ----------
    left : `pandas.DataFrame`
        First frame, already canonicalized.
    right : `pandas.DataFrame`
        Second frame, already canonicalized.

    Returns
    -------
    str or None
        A one-line human-readable description of the first disagreement, or `None` when the frames match exactly.
    """
    if (list(left.columns) != list(right.columns)):
        return f"Column sets differ: {list(left.columns)} versus {list(right.columns)}"

    if (len(left) != len(right)):
        return f"Row counts differ: {len(left)} versus {len(right)}"

    for column in left.columns:
        unequal = ~((left[column] == right[column]) | (left[column].isna() & right[column].isna()))

        if (unequal.any()):
            position = int(unequal.idxmax())

            return (f"Column {column!r} differs first at canonical row {position}: "
                    f"{left[column].iloc[position]!r} versus {right[column].iloc[position]!r}")

    return None


def frame_digest(df: pd.DataFrame) -> str:
    """
    A SHA-256 digest of a canonicalized frame's CSV rendering.

    Two runs are byte-identical at tier D0 exactly when their digests match. The digest is convenient for a build
    log; when it differs, `diff_frames` says why.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Frame to digest, already canonicalized.

    Returns
    -------
    str
        Hex digest.
    """
    rendered = df.to_csv(index=False, lineterminator="\n")

    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def permute_within_contiguous_groups(df: pd.DataFrame, group_values: typing.Sequence, seed: int) -> pd.DataFrame:
    """
    Shuffle rows within each contiguous run of equal group values, preserving the order of the runs.

    This is the input transformation for determinism control 13's permutation test: shuffling within a window must
    not change the output, but moving a row across a window boundary, or past the point in the stream where its
    window sealed, legitimately would. Restricting the shuffle to contiguous runs of the same group value permutes
    order exactly where the pipeline claims order must not matter.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Frame whose rows are shuffled.
    group_values : sequence
        One group value per row, typically the row's window ordinal.
    seed : int
        Seed for the shuffle, so the permutation itself is reproducible.

    Returns
    -------
    `pandas.DataFrame`
        A new frame with rows permuted within runs, index reset.

    Raises
    ------
    ValueError
        If `group_values` does not match the frame's length.
    """
    if (len(group_values) != len(df)):
        raise ValueError(f"group_values has {len(group_values)} entries for {len(df)} rows")

    rng = random.Random(seed)
    order: list[int] = []
    run: list[int] = []
    previous = object()

    for (position, value) in enumerate(group_values):
        if (value != previous and len(run) > 0):
            rng.shuffle(run)
            order.extend(run)
            run = []

        run.append(position)
        previous = value

    rng.shuffle(run)
    order.extend(run)

    return df.iloc[order].reset_index(drop=True)
