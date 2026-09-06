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
Which shard an entity belongs to, decided the same way on every machine and every restart.

This is determinism control 4, which the guide calls the most important control and the one with the best
cost-to-benefit ratio. Any node with `pe_count > 1` emits in completion order, so a stateful stage cannot be
threaded without giving up a total order per entity. Sharding recovers the parallelism without giving up the
order: route by a stable hash of the entity key, run one single-engine branch per shard, and every entity lands
on the same branch on every run.

**Never Python's built-in `hash`.** It is randomized per process by `PYTHONHASHSEED`, so a pipeline sharded with
it produces a different assignment on every restart -- and the assignment decides which stage instance holds an
entity's state, so per-entity history would be split differently every time. The failure is silent: each run is
internally consistent and no two runs agree. Control 13's cross-restart check exists partly to catch exactly
this, since it runs the pipeline twice under different hash seeds and compares byte for byte, and
`test_sharding_survives_a_different_hash_seed` asserts the property directly rather than relying on that check
to notice.

The hash is SHA-256 over the key's UTF-8 bytes, truncated to eight bytes and taken modulo the shard count. Eight
bytes rather than the whole digest because the modulus makes the rest irrelevant, and truncating explicitly is
clearer than relying on Python's arbitrary-precision integers to do it invisibly.

**The distribution is uniform enough and is not the point.** SHA-256 spreads keys evenly, but an estate whose
traffic is dominated by a handful of entities will have hot shards whatever the hash does, because every event
for one entity must reach one shard. That is inherent in the guarantee rather than a defect of this function: a
scheme that balanced better by splitting an entity would have given up the thing sharding is for.
"""

import hashlib
import typing

DEFAULT_PREFIX = "shard_"
"""Prefix for a shard's routing key, matching the guide's own `shard_0`, `shard_1` form."""

_DIGEST_BYTES = 8


def stable_shard(key: typing.Optional[str], shards: int) -> typing.Optional[int]:
    """
    The shard an entity key belongs to, in `[0, shards)`.

    Parameters
    ----------
    key : str or None
        The entity key. `None` returns `None` rather than a shard: a row with no entity has no per-entity state
        to keep together, and routing it to a fabricated shard would make one branch the home of every
        unattributable event in the estate.
    shards : int
        How many shards the scoring path runs.

    Returns
    -------
    int or None
        The shard index, or `None` for a null key.

    Raises
    ------
    ValueError
        If `shards` is not positive.
    """
    if (shards <= 0):
        raise ValueError(f"shards must be positive, received {shards}")

    if (key is None):
        return None

    digest = hashlib.sha256(key.encode("utf-8")).digest()[:_DIGEST_BYTES]

    return int.from_bytes(digest, "big") % shards


def shard_name(key: typing.Optional[str], shards: int, prefix: str = DEFAULT_PREFIX) -> typing.Optional[str]:
    """
    The routing key for an entity, as `RouterStage` names its ports.

    Parameters
    ----------
    key : str or None
        The entity key.
    shards : int
        How many shards the scoring path runs.
    prefix : str, default = "shard_"
        Prefix the port names carry.

    Returns
    -------
    str or None
        The port name, or `None` for a null key, which a caller must route somewhere deliberately.
    """
    index = stable_shard(key, shards)

    return None if index is None else f"{prefix}{index}"


def shard_names(shards: int, prefix: str = DEFAULT_PREFIX) -> list[str]:
    """
    Every routing key, in order, for handing to `RouterStage`'s `keys` argument.

    Parameters
    ----------
    shards : int
        How many shards the scoring path runs.
    prefix : str, default = "shard_"
        Prefix the port names carry.

    Returns
    -------
    list of str
        The port names.

    Raises
    ------
    ValueError
        If `shards` is not positive.
    """
    if (shards <= 0):
        raise ValueError(f"shards must be positive, received {shards}")

    return [f"{prefix}{index}" for index in range(shards)]
