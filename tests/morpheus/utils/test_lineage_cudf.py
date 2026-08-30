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

import random
import string

import pytest

from morpheus.utils.lineage import event_uid
from morpheus.utils.lineage import link_uid

# cuDF raises a runtime error rather than ImportError on machines without a usable GPU, so a plain importorskip is
# not enough to keep collection clean there.
try:
    import cudf
except Exception as import_error:  # pylint: disable=broad-except
    pytest.skip(f"cudf unavailable: {import_error}", allow_module_level=True)

# pylint: disable=wrong-import-position,ungrouped-imports
from morpheus.utils.lineage_cudf import event_uid_series_cudf  # noqa: E402
from morpheus.utils.lineage_cudf import link_uid_series_cudf  # noqa: E402
from morpheus.utils.lineage_cudf import verify_digest_equivalence  # noqa: E402


def test_equivalence_gate_passes():
    # The gate is the contract: if this fails on some future cuDF, the GPU path must not be used there.
    verify_digest_equivalence()
    verify_digest_equivalence(digest_length=16)


def test_event_uid_matches_host_on_fuzzed_rows():
    rng = random.Random(20260830)
    alphabet = string.ascii_letters + string.digits + "π🜚-_.:"

    collectors = ["".join(rng.choices(alphabet, k=rng.randrange(1, 40))) for _ in range(500)]
    versions = ["".join(rng.choices(alphabet, k=rng.randrange(1, 12))) for _ in range(500)]
    seqs = [rng.randrange(-2**62, 2**62) for _ in range(500)]

    df = cudf.DataFrame({"c": collectors, "v": versions, "s": seqs})
    gpu = event_uid_series_cudf(df, ["c", "v", "s"]).to_pandas().tolist()
    host = [event_uid(c, v, s) for (c, v, s) in zip(collectors, versions, seqs)]

    assert gpu == host


def test_event_uid_separator_boundary_matches_host():
    df = cudf.DataFrame({"a": ["ab", "a"], "b": ["c", "bc"]})
    gpu = event_uid_series_cudf(df, ["a", "b"]).to_pandas().tolist()

    assert gpu == [event_uid("ab", "c"), event_uid("a", "bc")]
    assert gpu[0] != gpu[1]


def test_single_column_and_digest_length():
    df = cudf.DataFrame({"a": ["only"]})

    assert event_uid_series_cudf(df, ["a"],
                                 digest_length=16).to_pandas().tolist() == [event_uid("only", digest_length=16)]


def test_link_uid_matches_host_and_marks_roots():
    parents = cudf.Series(["p1", None, ""])
    children = cudf.Series(["c1", "c2", "c3"])

    gpu = link_uid_series_cudf(parents, children, "carried_by", "hard:flow_id").to_pandas()

    assert gpu.iloc[0] == link_uid("p1", "c1", "carried_by", "hard:flow_id")
    assert gpu.iloc[1] is None or gpu.isna().iloc[1]
    assert gpu.iloc[2] is None or gpu.isna().iloc[2]


def test_rejects_unsupported_dtypes():
    with pytest.raises(ValueError, match="string and integer"):
        event_uid_series_cudf(cudf.DataFrame({"a": [True, False]}), ["a"])

    with pytest.raises(ValueError, match="string and integer"):
        event_uid_series_cudf(cudf.DataFrame({"a": [1.5]}), ["a"])


def test_rejects_nulls():
    with pytest.raises(ValueError, match="non-null"):
        event_uid_series_cudf(cudf.DataFrame({"a": ["x", None]}), ["a"])


def test_validation():
    df = cudf.DataFrame({"a": ["x"]})

    with pytest.raises(ValueError):
        event_uid_series_cudf(df, [])

    with pytest.raises(ValueError):
        event_uid_series_cudf(df, ["a"], digest_length=0)

    with pytest.raises(ValueError):
        link_uid_series_cudf(cudf.Series(["p"]), cudf.Series(["c", "d"]), "r", "m")


def test_gate_fails_closed_on_event_uid_disagreement(monkeypatch: pytest.MonkeyPatch):
    # The negative control for the gate itself: a check that has never been seen to fail proves nothing. Forcing
    # the host reference to return a wrong digest must make the gate raise, or the gate is decoration.
    import morpheus.utils.lineage_cudf as lineage_cudf_module

    monkeypatch.setattr(lineage_cudf_module, "event_uid", lambda *args, **kwargs: "0" * 32)

    with pytest.raises(RuntimeError, match="disagrees"):
        verify_digest_equivalence()


def test_gate_fails_closed_on_link_uid_disagreement(monkeypatch: pytest.MonkeyPatch):
    import morpheus.utils.lineage_cudf as lineage_cudf_module

    monkeypatch.setattr(lineage_cudf_module, "link_uid", lambda *args, **kwargs: "0" * 32)

    with pytest.raises(RuntimeError, match="disagrees"):
        verify_digest_equivalence()
