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
The synthetic corpus `morpheus.compositions.estate` is tested over, and the facts about it the harnesses assert.

Moved out of `tests/morpheus/determinism/estate_pipeline.py` under plan section 4.5, which put the composition in the
package and left the data here, where the tests, the sample and fixture generators and the experiments read it. The
composition module's docstring describes what this corpus plants and why.
"""

import random

import pandas as pd
from corpora import merged  # noqa: F401

from morpheus.compositions import envelope

sp = merged('session')
tp = merged('telemetry')
import typing

from morpheus.compositions.estate import AUTH_CLASS
from morpheus.compositions.estate_context import EstateContext
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.binding_table import BindingTable

CORPUS_SEED = 20260920
UNCHAINED_CLASSES = ("tc1_binding", "tc2_binding")
"""The closed bindings. They are tables rather than observations, so nothing correlates them into a chain."""
DESKS = {
    sp.ALICE: tp.PORTS[0],
    sp.BOB: tp.PORTS[1],
    sp.CAROL: tp.PORTS[2],
}
"""Who sits where. The ports are the ones the telemetry corpus already polls and authenticates."""
REMOTE = sp.DAVE
"""The principal who is never at a desk, and whom the directory therefore does not name."""
DESK_LOGIN_SECONDS = 240
"""First desk sign-in, far enough into the hour that the ports have been polled and authorized first."""
DESK_LOGIN_INTERVAL_SECONDS = 600
"""How often each person authenticates to something during the hour."""
DESK_LOGIN_STAGGER_SECONDS = 37
"""Seconds between one person's sign-in and the next person's, so no two land on one instant."""
APPS = ("wiki", "mail", "crm")
"""What the desk sign-ins are for. Three so `appincrement` has something to count."""


def build_corpus() -> dict[str, pd.DataFrame]:
    """
    The telemetry corpus unchanged, plus the layer 5 authentications made from those desks in the same hour.

    Layers 1 and 2 are `telemetry_pipeline.build_corpus` verbatim. Rebuilding them here would fork a corpus whose
    planted cases are asserted elsewhere, and the point of this harness is the join rather than a second estate.
    """
    corpus = tp.build_corpus()
    corpus[AUTH_CLASS] = envelope.attach(_build_desk_auth(random.Random(CORPUS_SEED)))

    return corpus


def _auth_row(principal: str, time_s: int, app: str) -> dict:
    """One identity provider sign-in, in the shape `session_pipeline` gives them.

    Everyone is in London on their own address, because the geography is not what this harness is about: the
    travel and novelty cases are planted in the layer 5 corpus and asserted there. What matters here is that the
    row is a real layer 5 authentication carrying a principal, not a placeholder shaped like one.
    """
    (country, region, city) = sp.LONDON_PLACE

    return {
        "event_time": time_s * NS_PER_SECOND,
        "user_principal": principal,
        "source_country": country,
        "source_region": region,
        "source_city": city,
        "source_latitude": sp.LONDON[0],
        "source_longitude": sp.LONDON[1],
        "source_asn": sp.ASNS[sp.LONDON_PLACE],
        "source_ip": sp.HOME_IPS[principal],
        "app": app,
        "device_id": sp.DEVICES[principal],
        "auth_result": "success",
        "mfa_used": True,
        "mfa_result": "approved",
        "token_type": "bearer",
    }


CORPUS_SECONDS = tp.CORPUS_SECONDS


def _build_desk_auth(rng: random.Random) -> pd.DataFrame:
    """The hour's sign-ins: three people at their desks and one working from home."""
    principals = list(DESKS) + [REMOTE]
    events: list[dict] = []

    for (index, principal) in enumerate(principals):
        start = DESK_LOGIN_SECONDS + index * DESK_LOGIN_STAGGER_SECONDS

        for (step, time_s) in enumerate(range(start, CORPUS_SECONDS, DESK_LOGIN_INTERVAL_SECONDS)):
            events.append(_auth_row(principal, time_s, APPS[step % len(APPS)]))

    events.sort(key=lambda event: (event["event_time"], event["user_principal"]))
    rows = []

    for (seq, event) in enumerate(events, start=1):
        rows.append({**event, **tp._envelope(rng, "idp", "tc5_auth.v1", seq, None)})  # pylint: disable=protected-access

    return pd.DataFrame(rows)


DIRECTORY = {
    sp.ALICE: tp.IDENTITIES[tp.MAC_A],
    sp.BOB: tp.IDENTITIES[tp.MAC_B],
    sp.CAROL: tp.IDENTITIES[tp.MAC_C],
}
"""The identity provider's own record of which 802.1X identity each person presents.

Read from the telemetry corpus rather than restated, so the two cannot drift apart: if the supplicant on a port
is renamed, this follows it, and a test asserting the ladder cannot pass against a directory that has quietly
stopped describing the estate.
"""


def build_directory_table(principals: typing.Optional[dict] = None) -> BindingTable:
    """
    The supplied half of the ladder: which 802.1X identity each principal presents, and for how long.

    Bounded across the whole corpus rather than left open, because an unbounded binding is the one shape a
    resolver cannot be wrong about and therefore the one that proves nothing. A directory export is a statement
    about an interval; treating it as eternal is how stale attribution outlives the person it described.
    """
    principals = DIRECTORY if principals is None else principals
    rows = []

    for (principal, identity) in sorted(principals.items()):
        rows.append({
            "user_principal": principal,
            "dot1x_identity": identity,
            "bind_start": 0,
            "bind_end": CORPUS_SECONDS * NS_PER_SECOND,
        })

    return BindingTable.from_dataframe(pd.DataFrame(rows),
                                       name="directory",
                                       key_column="user_principal",
                                       value_columns=["dot1x_identity"],
                                       start_column="bind_start",
                                       end_column="bind_end")


def context() -> EstateContext:
    """The estate's context this corpus is run under: the identity provider's 802.1X directory for the three people the chains follow, and the telemetry corpus's gateway."""
    return EstateContext(directory=build_directory_table(), extras=tp.context().extras)
