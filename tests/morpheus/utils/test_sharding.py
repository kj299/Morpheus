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

import json
import os
import subprocess
import sys

import pytest

from morpheus.utils.sharding import shard_name
from morpheus.utils.sharding import shard_names
from morpheus.utils.sharding import stable_shard

PRINCIPALS = [f"user-{index}@example.com" for index in range(2000)]


def test_the_same_key_lands_on_the_same_shard():
    assert stable_shard("alice@example.com", 8) == stable_shard("alice@example.com", 8)


def test_every_shard_is_inside_the_range():
    assert all(0 <= stable_shard(key, 8) < 8 for key in PRINCIPALS)


def test_sharding_survives_a_different_hash_seed():
    """
    The property the whole module exists for, asserted directly rather than left to control 13 to notice.

    Python's built-in `hash` is randomized per process by `PYTHONHASHSEED`, so a pipeline sharded with it puts
    an entity on a different branch on every restart -- and the branch is what holds that entity's state, so its
    history would be split differently each time. Each run is internally consistent and no two agree, which is
    the worst shape a defect can take.
    """
    script = ("import json, sys;"
              "from morpheus.utils.sharding import stable_shard;"
              "print(json.dumps([stable_shard(k, 8) for k in json.loads(sys.argv[1])]))")

    keys = PRINCIPALS[:64]
    here = [stable_shard(key, 8) for key in keys]
    outputs = []

    for seed in ("0", "12345"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run([sys.executable, "-c", script, json.dumps(keys)],
                                   capture_output=True,
                                   text=True,
                                   check=True,
                                   timeout=120,
                                   env=environment)
        outputs.append(json.loads(completed.stdout))

    assert outputs[0] == outputs[1], "sharding changed with PYTHONHASHSEED; it is using a randomized hash"
    assert outputs[0] == here


def test_the_built_in_hash_would_have_failed_that():
    # The negative control. Without it the test above passes on any implementation and proves nothing about the
    # choice of hash, only that the module is consistent with itself inside one process.
    script = "import sys; print(hash(sys.argv[1]))"
    outputs = []

    for seed in ("0", "12345"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run([sys.executable, "-c", script, "alice@example.com"],
                                   capture_output=True,
                                   text=True,
                                   check=True,
                                   timeout=120,
                                   env=environment)
        outputs.append(completed.stdout.strip())

    assert outputs[0] != outputs[1], "PYTHONHASHSEED no longer randomizes str hashing; this module's rationale changed"


def test_keys_are_spread_across_the_shards():
    # Not a uniformity proof, and not meant as one: what matters is that no shard is left empty, because an
    # empty branch is parallelism paid for and not used.
    counts = [0] * 16

    for key in PRINCIPALS:
        counts[stable_shard(key, 16)] += 1

    assert min(counts) > 0
    # A loose bound. SHA-256 spreads evenly; an estate's own traffic will not, and that is inherent rather than
    # a defect of this function.
    assert max(counts) < 3 * (len(PRINCIPALS) / 16)


def test_a_null_key_is_not_routed_to_a_fabricated_shard():
    # Pooling them would make one branch the home of every unattributable event in the estate.
    assert stable_shard(None, 8) is None
    assert shard_name(None, 8) is None


def test_the_shard_name_is_what_the_router_expects():
    assert shard_name("alice@example.com", 4) in shard_names(4)
    assert shard_names(3) == ["shard_0", "shard_1", "shard_2"]
    assert shard_names(2, prefix="score_") == ["score_0", "score_1"]


def test_one_shard_is_a_legitimate_configuration():
    # The unsharded scoring path, which is what a deployment starts with.
    assert stable_shard("alice@example.com", 1) == 0
    assert shard_names(1) == ["shard_0"]


@pytest.mark.parametrize("shards", [0, -1], ids=["zero", "negative"])
def test_a_shard_count_that_cannot_hold_anything_is_refused(shards):
    with pytest.raises(ValueError, match="shards must be positive"):
        stable_shard("alice@example.com", shards)

    with pytest.raises(ValueError, match="shards must be positive"):
        shard_names(shards)


def test_changing_the_shard_count_is_a_migration_rather_than_a_setting():
    # Stated as a test because it is the operational consequence a deployment has to plan for: every entity's
    # assignment moves, so per-entity state built on the old count belongs to the wrong branch under the new one.
    moved = sum(1 for key in PRINCIPALS if stable_shard(key, 8) != stable_shard(key, 16))

    assert moved > len(PRINCIPALS) / 3
