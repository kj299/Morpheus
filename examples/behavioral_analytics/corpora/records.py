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


def seeded_envelope(rng: random.Random, collector: str, schema: str, seq: int) -> dict:
    """The envelope of a record whose origin is drawn from the corpus's seeded generator."""
    return {
        "collector_id": collector,
        "schema_version": schema,
        "origin_hash": f"{rng.getrandbits(64):016x}",
        "collector_seq": seq,
    }
