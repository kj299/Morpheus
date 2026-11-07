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
"""
What every composed corpus's harness asks of its golden file, written once (guide 13 plan item 6).

A harness holds its composition to `golden_<corpus>_expected.csv`: run twice in fresh interpreters under different
hash seeds, the bytes agree (`cross_restart_outputs`), and the in-process result renders to the file
(`assert_matches_golden`). Both go through `run_golden.py`, the one driver that regenerates a golden. The
batching and permutation checks cut and shuffle a corpus the same way everywhere (`split`, `permuted`).
"""

import os
import subprocess
import sys
from io import StringIO

import pandas as pd
import pytest

from morpheus.utils.determinism import diff_frames
from morpheus.utils.determinism import permute_within_contiguous_groups
from morpheus.utils.lineage import window_id_from_timestamp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
from run_golden import golden_path  # noqa: E402

NS_PER_SECOND = 10**9

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "run_golden.py")
"""The driver every harness's cross-restart check runs, and the one a golden is regenerated with."""

HASH_SEEDS = ("0", "4242")
"""Two `PYTHONHASHSEED` values; a composition whose output depends on set or dict order differs between them."""


def cross_restart_outputs(corpus: str, execution_mode, tmp_path, timeout: int = 900) -> list[bytes]:
    """The corpus's rendered output from one fresh interpreter per hash seed, as bytes."""
    mode = "gpu" if execution_mode.value == "GPU" else "cpu"
    outputs = []

    for (index, hash_seed) in enumerate(HASH_SEEDS):
        out_path = tmp_path / f"restart_{index}.csv"
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hash_seed

        subprocess.run([sys.executable, DRIVER, corpus, str(out_path), mode], env=env, check=True, timeout=timeout)
        outputs.append(out_path.read_bytes())

    return outputs


def assert_matches_golden(corpus: str, rendered: str):
    """Fail, naming the first difference, unless `rendered` is the corpus's golden file byte for byte."""
    with open(golden_path(corpus), encoding="utf-8") as handle:
        golden_text = handle.read()

    if (rendered != golden_text):
        as_text = {"dtype": str, "keep_default_na": False}
        difference = diff_frames(pd.read_csv(StringIO(rendered), **as_text),
                                 pd.read_csv(StringIO(golden_text), **as_text))

        pytest.fail(f"Output drifted from {os.path.basename(golden_path(corpus))}: {difference}. If the change is "
                    f"intended, regenerate the golden with `{os.path.basename(DRIVER)} {corpus}` and review the diff.")


def split(frame: pd.DataFrame, parts: int) -> list:
    """`frame` cut into about `parts` consecutive batches, as a source delivering it in pieces would."""
    size = max(1, len(frame) // parts)

    return [frame.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(frame), size)]


def event_windows(frame: pd.DataFrame, period_seconds: int) -> list:
    """
    Each row's sealing window, from its `event_time`. A frame with no event time (an inventory, which no window
    holds and whose store does not depend on its records' order) is one group.
    """
    if ("event_time" not in frame.columns):
        return [0] * len(frame)

    return [window_id_from_timestamp(int(stamp), period_seconds * NS_PER_SECOND) for stamp in frame["event_time"]]


def permuted(corpus: dict, seed: int, period_seconds: int) -> dict:
    """Every frame of `corpus` shuffled within its sealing windows, which a composition must not be able to see."""
    return {
        name: permute_within_contiguous_groups(frame, event_windows(frame, period_seconds), seed=seed)
        for (name, frame) in corpus.items()
    }


def composed(module) -> pd.DataFrame:
    """A corpus module's composition over its own corpus, in the module's default configuration."""
    return module.run_pipeline(module.build_pipeline_config(), module.build_corpus())


def flags(frame: pd.DataFrame, column: str) -> pd.Series:
    """A nullable flag column as plain booleans, with "not answerable" reading as "did not fire"."""
    return frame[column].astype("boolean").fillna(False).astype(bool)


def assert_every_row_carries_a_community_id(result: pd.DataFrame):
    """The equality join key across layers 3, 4 and 6: present on every row, and in its version-1 form."""
    assert result["community_id"].notna().all()
    assert result["community_id"].str.startswith("1:").all()
