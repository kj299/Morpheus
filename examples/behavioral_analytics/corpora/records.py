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
"""
How the corpora write a collector's record, written once (guide 13 plan item 6).

Every corpus stamps its records with the collector envelope `input_contracts.json` names, and several counted time
the same way; each carried its own copy of the rule until the copies became a second definition.
"""

import random
import typing
import uuid

from morpheus.utils.binding_table import NS_PER_SECOND

DAY_SECONDS = 86400


def at(day: float, hour: float = 0) -> int:
    """An instant `day` days and `hour` hours after the epoch, in nanoseconds."""
    return int((day * DAY_SECONDS + hour * 3600) * NS_PER_SECOND)


def envelope(rows: list, collector: str, schema: str, origin: str, when: int) -> dict:
    """The envelope of the next record in `rows`: its collector, schema, origin, sequence and event time."""
    return {
        "collector_id": collector,
        "schema_version": schema,
        "origin_hash": f"{origin}@{when}",
        "collector_seq": len(rows),
        "event_time": when,
    }


SELF_NAMED = None
"""The `origin` of a record its source names itself: an identity provider's sign-in or session record, which carries
an identifier of its own."""

RECORD_NAMESPACE = uuid.UUID("7d3c9a52-1f0e-4b6d-8a2c-5e4f3b2a1d09")
"""Where the identifier of a self-named record is derived from, so it is a GUID as the provider writes one."""


def seeded_envelope(rng: random.Random, collector: str, schema: str, seq: int, origin: typing.Optional[str]) -> dict:
    """
    The envelope of a record from a corpus drawing its values from a seeded generator.

    `origin_hash` is what identifies the record at its source, as the contract says: `origin`, what the source
    observed it as (a port's poll, a MAC table entry, a session's start), or, for a record its source names itself
    (`SELF_NAMED`), the identifier the source gives it, a GUID drawn from the generator. The draw is taken either way:
    it is where the corpora's identifiers came from before the input samples took one convention (plan section
    4.7a), and every value drawn after it keeps the stream it was planted with.
    """
    bits = rng.getrandbits(64)

    return {
        "collector_id": collector,
        "schema_version": schema,
        "origin_hash": str(uuid.uuid5(RECORD_NAMESPACE, f"{collector}:{bits:016x}")) if origin is None else origin,
        "collector_seq": seq,
    }
