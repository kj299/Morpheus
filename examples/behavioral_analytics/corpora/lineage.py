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
The synthetic corpus `morpheus.compositions.lineage` is tested over, and the facts about it the harnesses assert.

Moved out of `tests/morpheus/determinism/lineage_pipeline.py` under plan section 4.5, which put the composition in the
package and left the data here, where the tests, the sample and fixture generators and the experiments read it. The
composition module's docstring describes what this corpus plants and why.
"""

import random

import pandas as pd

from morpheus.compositions import envelope
from morpheus.compositions.lineage import _ENTITIES
from morpheus.utils.binding_table import NS_PER_SECOND

CORPUS_SEED = 20260830


def build_corpus() -> pd.DataFrame:
    """
    Build the fixed input corpus, one row per network flow observation.

    Every value is derived from `CORPUS_SEED`, so the corpus is as fixed as a checked-in file while remaining
    reviewable as code.
    """
    rng = random.Random(CORPUS_SEED)
    rows = []

    def add(time_s: float, src_ip: str, collector: str, dst_ip: str = "192.0.2.50", protocol: str = "tcp"):
        rows.append({
            "event_time": int(time_s * NS_PER_SECOND),
            "src_ip": src_ip,
            "dest_ip": dst_ip,
            "protocol": protocol,
            "src_port": rng.randrange(1024, 65535),
            "dest_port": 443,
            "collector_id": collector,
            "schema_version": "TC-3/1.0.0",
            "origin_hash": f"{rng.getrandbits(64):016x}",
        })

    # Normal traffic: three entities active through windows 0, 1, and 2 ([0, 300), [300, 600), [600, 900) seconds).
    for window_start in (0, 300, 600):
        for (ip, _, _) in _ENTITIES:
            for _ in range(4):
                add(window_start + rng.uniform(1, 299), ip, "col-a")

    # Clock skew: records that arrive out of order but within the lateness horizon, so they are on time.
    add(610.0, _ENTITIES[0][0], "col-b")
    add(305.5, _ENTITIES[0][0], "col-b")
    add(12.25, _ENTITIES[1][0], "col-b")

    # A single-row entity, unknown to the binding table, so it also exercises the unresolved path.
    add(320.0, "10.0.9.99", "col-b")

    # An empty window: nothing between [900, 1200) seconds; activity resumes in window 4 and then jumps far
    # enough ahead to seal everything up to it.
    for (ip, _, _) in _ENTITIES:
        add(1210.0 + rng.uniform(0, 80), ip, "col-a")

    add(3600.0, _ENTITIES[2][0], "col-a")

    # Late arrivals: rows for window 0 and window 1 arriving after the watermark passed their horizons.
    add(15.0, _ENTITIES[0][0], "col-c")
    add(410.0, _ENTITIES[1][0], "col-c")

    # A malformed flow: the address cannot be hashed, so the Community ID is null but the row must survive.
    add(3620.0, "not-an-address", "col-c")

    df = pd.DataFrame(rows)
    df["collector_seq"] = range(1, len(df) + 1)
    df = envelope.attach(df)

    return df
