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
"""
The ladder's third rung: one estate, one hour, the same people at layers 1, 2 and 5.

Chains in this fork spanned two layers and stopped. The reason was not the sealing, which already puts every
class that roots on a port into one chain; it was that layer 5 ran over its own corpus, with its own principals,
in its own week, and nothing connected a person authenticating to an identity provider with the switch port they
were sitting at. `Chain assembly - cross-layer risk` fires on `dc(osi_layer) >= 3` and had nothing to fire on.

This module is the estate where they meet. Layers 1 and 2 are the telemetry corpus unchanged -- the same hour,
the same three ports, the same planted hub, spoof, flood and bypass -- and layer 5 is the desk authentications
the people at those ports made during that hour. Two bindings carry a principal down to a port:

- **The directory** (`user_principal` to `dot1x_identity`) is supplied, not derived. Which 802.1X identity a
  person presents is an employment fact that lives in an identity provider, and no amount of telemetry produces
  it. It is bounded like every other binding here, because people change desks, leave, and are replaced.
- **The supplicant table** (`dot1x_identity` to `auth_port_key`) is derived, by the same
  {py:class}`~morpheus.stages.telemetry.tc2_binding_stage.TC2BindingStage` that closes MAC bindings, keyed on the
  identity instead of the address. An 802.1X exchange says who was on a port and when, which is exactly a
  binding, and the stage was already general over its key.

Walking both, in that order, takes an authentication to the port its principal sat at:

    user_principal -> dot1x_identity -> auth_port_key == the layer 1 entity_key

That last equality is not a coincidence to be relied on quietly, so the harness asserts it. `auth_port_key`,
`port_key` and layer 1's `entity_key` are all `site:switch:port`, which is what lets a chain rooted on any of
them hold members from all three layers.

**Dave is the negative control and is not decoration.** He works remotely, never sits at a desk, and the
directory has no entry for him -- so his authentications resolve to nothing, root on his own principal, and must
share no chain with any port. A ladder that attributed him to a port would be worse than one that reached only
two layers, because it would be confidently wrong rather than visibly incomplete. The `unknown-supplicant` the
bypass presents is the same case from the other end: it binds to a port, but no principal maps to it, so it
carries nobody down.

The chains are sealed the way the two-layer ones are, in one pass over the union of every class in event-time
order, because a Merkle root is computed per sealer and classes sealed separately can never share one.
"""

import random
import typing

import pandas as pd

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.binding_resolver_stage import BindingResolverStage
from morpheus.stages.lineage.chain_anchor_stage import DEFAULT_ANCHOR_COLUMN
from morpheus.stages.lineage.chain_anchor_stage import ChainAnchorStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc2_binding_stage import TC2BindingStage
from morpheus.stages.telemetry.tc5_novelty_stage import TC5NoveltyStage
from morpheus.stages.telemetry.tc5_risk_stage import TC5RiskStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.binding_table import BindingTable
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize

import session_pipeline as sp
import telemetry_pipeline as tp

CORPUS_SEED = 20260920

PERIOD_SECONDS = tp.PERIOD_SECONDS
LATENESS_SECONDS = tp.LATENESS_SECONDS
CORPUS_SECONDS = tp.CORPUS_SECONDS

ID_COLUMNS = tp.ID_COLUMNS
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []

AUTH_CLASS = "tc5_auth"
CHAINED_CLASSES = ("tc1", "tc2_mac", "tc2_arp", "tc2_auth", AUTH_CLASS)
UNCHAINED_CLASSES = ("tc1_binding", "tc2_binding")
"""The closed bindings. They are tables rather than observations, so nothing correlates them into a chain."""

DESKS = {
    sp.ALICE: tp.PORTS[0],
    sp.BOB: tp.PORTS[1],
    sp.CAROL: tp.PORTS[2],
}
"""Who sits where. The ports are the ones the telemetry corpus already polls and authenticates."""

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

REMOTE = sp.DAVE
"""The principal who is never at a desk, and whom the directory therefore does not name."""

DESK_LOGIN_SECONDS = 240
"""First desk sign-in, far enough into the hour that the ports have been polled and authorized first."""

DESK_LOGIN_INTERVAL_SECONDS = 600
"""How often each person authenticates to something during the hour."""

DESK_LOGIN_STAGGER_SECONDS = 37
"""Seconds between one person's sign-in and the next person's, so no two land on one instant."""

DIRECTORY_METHOD_COLUMN = "directory_resolution"
SUPPLICANT_METHOD_COLUMN = "supplicant_resolution"
DESK_IDENTITY_COLUMN = "desk_identity"
DESK_PORT_COLUMN = "desk_port_key"

APPS = ("wiki", "mail", "crm")
"""What the desk sign-ins are for. Three so `appincrement` has something to count."""


def build_corpus() -> dict[str, pd.DataFrame]:
    """
    The telemetry corpus unchanged, plus the layer 5 authentications made from those desks in the same hour.

    Layers 1 and 2 are `telemetry_pipeline.build_corpus` verbatim. Rebuilding them here would fork a corpus whose
    planted cases are asserted elsewhere, and the point of this harness is the join rather than a second estate.
    """
    corpus = tp.build_corpus()
    corpus[AUTH_CLASS] = _build_desk_auth(random.Random(CORPUS_SEED))

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
        rows.append({**event, **tp._envelope(rng, "idp", "TC-5/1.0.0", seq)})  # pylint: disable=protected-access

    return pd.DataFrame(rows)


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


def build_supplicant_table(bindings: pd.DataFrame) -> BindingTable:
    """The derived half: which port an 802.1X identity was authorized on, as the binding stage closed it."""
    return BindingTable.from_dataframe(bindings,
                                       name="dot1x",
                                       key_column="dot1x_identity",
                                       value_columns=["auth_port_key"],
                                       start_column="bind_start",
                                       end_column="bind_end")


def build_pipeline_config(execution_mode=None) -> Config:
    """A pipeline configuration, defaulting to CPU mode and importable without a GPU."""
    return tp.build_pipeline_config(execution_mode)


def _collect(sink: InMemorySinkStage) -> pd.DataFrame:
    from host_frame import to_host_frame

    frames = []

    for message in sink.get_messages():
        meta = message.payload() if isinstance(message, ControlMessage) else message
        frames.append(to_host_frame(meta.copy_dataframe()))

    if (len(frames) == 0):
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _run_class(config: Config, dataframes: list, stages: list, impose_order: bool = True) -> pd.DataFrame:
    """Source, stamp, optional total order, the class's stages, sink. No sealing: the union does that."""
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    for stage in stages:
        pipe.add_stage(stage)

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    return _collect(sink)


def _seal_chains(config: Config, outputs: dict[str, pd.DataFrame], parts: int = 1) -> dict[str, pd.DataFrame]:
    """
    Seal every chained class together, so a port's chain holds its events from all three layers.

    The same pass the telemetry harness makes, over one more class. It is repeated here rather than imported
    because the set of classes differs, and a shared helper parameterized on that set would be a helper whose
    only caller-visible behaviour is the list it is given.
    """
    columns = {name: list(frame.columns) for (name, frame) in outputs.items()}
    tagged = []

    for (name, frame) in outputs.items():
        frame = frame.copy()
        frame["telemetry_class"] = name
        tagged.append(frame)

    union = pd.concat(tagged, ignore_index=True)
    union = union.sort_values(list(DEFAULT_ORDER_COLUMNS), kind="stable").reset_index(drop=True)

    size = max(1, len(union) // max(1, parts))
    dataframes = [union.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(union), size)]

    sealer = WindowSealStage(config,
                             period_seconds=PERIOD_SECONDS,
                             lateness_seconds=LATENESS_SECONDS,
                             order_columns=list(DEFAULT_ORDER_COLUMNS),
                             entity_key_column=DEFAULT_ANCHOR_COLUMN)
    added = [name for name in sealer._needed_columns if name not in union.columns]  # pylint: disable=protected-access

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(sealer)
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    sealed = _collect(sink)
    result = {}

    for name in outputs:
        rows = sealed[sealed["telemetry_class"] == name]
        result[name] = rows[columns[name] + added].reset_index(drop=True)

    return result


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True) -> pd.DataFrame:
    """
    Run the estate: layers 1 and 2 as the telemetry harness runs them, layer 5 resolved down to a port.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    corpus : dict
        The frames from `build_corpus`, possibly permuted.
    batches : dict, optional
        Per class, how the corpus is split across source frames. Defaults to one frame per class.
    impose_order : bool, default = True
        Place `TotalOrderStage` ahead of the stateful stages. The permutation check's negative control turns it
        off.

    Returns
    -------
    `pandas.DataFrame`
        Every class's output, tagged with `telemetry_class`, keyed by `row_key`, canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    telemetry_batches = {name: frames for (name, frames) in batches.items() if name != AUTH_CLASS}
    telemetry = tp.run_classes(config, {name: corpus[name]
                                        for name in corpus if name != AUTH_CLASS},
                               batches=telemetry_batches,
                               impose_order=impose_order)

    # Every telemetry class, not only the chained ones: the binding classes carry no chain and are not sealed,
    # but they are what the `binding:l1` and `binding:l2` sourcetypes are made of, and a harness that dropped
    # them would force whoever renders the estate to run the telemetry pipeline a second time beside it.
    outputs = dict(telemetry)

    # The 802.1X exchanges, closed into identity bindings by the stage that closes MAC ones. Keyed on the
    # identity rather than the address, because the address is the asset and the identity is the person's half of
    # the exchange -- and the directory can only bind a principal to the half a person actually presents.
    identity_bindings = _run_class(
        config, [telemetry["tc2_auth"].copy()],
        [TC2BindingStage(config, key_column="dot1x_identity", attribute_columns=["auth_port_key"])],
        impose_order)

    outputs[AUTH_CLASS] = _run_class(config,
                                     batches[AUTH_CLASS],
                                     [
                                         TC5NoveltyStage(config),
                                         TC5RiskStage(config, min_denominator=1),
                                         BindingResolverStage(config,
                                                              binding_table=build_directory_table(),
                                                              key_column="user_principal",
                                                              output_columns={"dot1x_identity": DESK_IDENTITY_COLUMN},
                                                              method_column=DIRECTORY_METHOD_COLUMN),
                                         BindingResolverStage(config,
                                                              binding_table=build_supplicant_table(identity_bindings),
                                                              key_column=DESK_IDENTITY_COLUMN,
                                                              output_columns={"auth_port_key": DESK_PORT_COLUMN},
                                                              method_column=SUPPLICANT_METHOD_COLUMN),
                                         ChainAnchorStage(config, candidates=[DESK_PORT_COLUMN, "user_principal"]),
                                         EnvelopeStampStage(config, osi_layer=5, entity_columns=["user_principal"]),
                                     ],
                                     impose_order)

    parts = max(len(batches[name]) for name in CHAINED_CLASSES)
    outputs.update(_seal_chains(config, {name: outputs[name] for name in CHAINED_CLASSES}, parts=parts))

    frames = []

    for (name, frame) in outputs.items():
        frame = frame.copy()
        frame["telemetry_class"] = name

        if ("row_key" not in frame.columns):
            frame["row_key"] = frame["event_uid"]

        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    return canonicalize(combined, key_columns=KEY_COLUMNS, ignore_columns=IGNORE_COLUMNS)


def render(result: pd.DataFrame) -> str:
    """The byte-exact rendering compared across restarts and against the golden file."""
    return tp.render(result)
