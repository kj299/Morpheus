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
The TC-0 context store, its two producers, and the point-in-time join, for the determinism harness.

TC-0 is not an OSI layer and has no rules of its own. What it has is a requirement the guide calls hard: every record
bitemporal, so that "was this user in Finance on March 3rd" can be answered both as it is understood now and as it was
understood on March 3rd. A corpus that only ever recorded facts on the day they became true would pass every test here
with a single time axis, so this one is built from the cases where the two axes disagree:

- a **move recorded on time**: a principal changes department and the source says so the same day. Both views agree,
  before and after; this is the control that proves the other cases differ for a reason;
- a **retroactive correction**: a department recorded wrongly from the start and put right three weeks later. Asked
  about the first week, the event-time view says what was believed then and the latest view says what was true;
- a **leaver recorded late**: a termination effective on day twelve that reached the source on day eighteen. For the
  six days between, anything the principal did was done by an active employee with their group memberships, as far
  as anyone could have known;
- a **contract that ended silently**: a principal simply absent from a full snapshot, which is how most HR exports
  report a departure. The snapshot diff turns the absence into a retraction, and nothing else in that snapshot -- a
  restatement of everything else, unchanged -- is recorded twice;
- a **reclassification recorded late**: a database declared restricted a week after it became so;
- a **peer group change recorded late**: a workstation that followed its owner into Finance a day before the
  inventory noticed;
- a **decommissioned host**, absent from a snapshot the way the departed contractor is;
- and two records the store must refuse: one with no instant of recording, and one whose interval ends before it
  begins.

The probes are events at chosen instants, enriched twice -- once with what was known at the event's time and once
with everything -- so each case can be asserted as a pair of answers rather than one.
"""

import typing

import pandas as pd

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc0_asset_stage import ASSET
from morpheus.stages.telemetry.tc0_asset_stage import DEFAULT_ASSET_COLUMNS
from morpheus.stages.telemetry.tc0_asset_stage import TC0AssetStage
from morpheus.stages.telemetry.tc0_enrich_stage import KNOWLEDGE_MODES
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.stages.telemetry.tc0_identity_stage import DEFAULT_PROFILE_COLUMNS
from morpheus.stages.telemetry.tc0_identity_stage import PROFILE
from morpheus.stages.telemetry.tc0_identity_stage import TC0IdentityStage
from morpheus.utils import bitemporal
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.determinism import canonicalize

DAY_SECONDS = 86400

KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []

IDENTITY_CLASS = "tc0_identity"
ASSET_CLASS = "tc0_asset"
IDENTITY_PROBES = "tc0_identity_probes"
ASSET_PROBES = "tc0_asset_probes"

PRODUCER_CLASSES = (IDENTITY_CLASS, ASSET_CLASS)
PROBE_CLASSES = (IDENTITY_PROBES, ASSET_PROBES)
CORPUS_CLASSES = PRODUCER_CLASSES + PROBE_CLASSES

# Principals.
ALICE = "alice@example.com"
BOB = "bob@example.com"
CAROL = "carol@example.com"
DAVE = "dave@example.com"
ERIN = "erin@example.com"
FRANK = "frank@example.com"
GRACE = "grace@example.com"
HEIDI = "heidi@example.com"
MALLORY = "mallory@example.com"

# Hosts.
ALICE_WORKSTATION = "ws-alice"
BOB_WORKSTATION = "ws-bob"
LEDGER = "db-ledger"
BUILDER = "build-07"

MOVE_DAY = 10
"""Bob moves from Engineering to Finance. Recorded the same morning."""

SNAPSHOT_DAY = 15
"""The second full snapshot. Erin and the build server are absent from it."""

TERMINATION_DAY = 12
TERMINATION_RECORDED_DAY = 18
"""Dave leaves on day twelve; the source hears on day eighteen."""

CORRECTION_RECORDED_DAY = 20
"""Carol's department was never Sales. The source is corrected on day twenty, effective from the start."""

RECLASSIFIED_DAY = 5
RECLASSIFIED_RECORDED_DAY = 7
"""The ledger database becomes restricted on day five; the inventory says so on day seven."""

PEER_MOVE_RECORDED_DAY = 11
"""Bob's workstation joins the Finance peer group from day ten; the inventory notices on day eleven."""


def at(day: float, hour: float = 0) -> int:
    """An instant `day` days and `hour` hours into the corpus, in nanoseconds since the epoch."""
    return int((day * DAY_SECONDS + hour * 3600) * NS_PER_SECOND)


SNAPSHOT_HOUR = 6
DELTA_HOUR = 9


def _profile(principal, department, manager, status, valid_from, recorded, change=bitemporal.ASSERT):
    return make_version(PROFILE,
                        principal,
                        valid_from,
                        None,
                        recorded,
                        change, {
                            "department": department, "manager": manager, "employment_status": status
                        })


def _member(principal, group, valid_from, recorded, change=bitemporal.ASSERT):
    return make_version(bitemporal.MEMBERSHIP,
                        principal,
                        valid_from,
                        None,
                        recorded,
                        change, {"group_name": group}, (group, ))


def _asset(host, owner, team, criticality, classification, peer, valid_from, recorded, change=bitemporal.ASSERT):
    return make_version(
        ASSET,
        host,
        valid_from,
        None,
        recorded,
        change,
        {
            "owner": owner,
            "owning_team": team,
            "criticality": criticality,
            "data_classification": classification,
            "peer_group": peer,
        })


def _snapshot(log: list, kind: str, facts: list, recorded_ns: int) -> list:
    """What a full snapshot adds to everything recorded before it, via the store's own diff."""
    known = BitemporalStore("snapshot", [version for version in log if version.recorded_ns < recorded_ns])

    return known.snapshot_changes(kind, facts, recorded_ns)


def _fact(version) -> dict:
    """A version restated as a snapshot row, which is how a full export states what it holds."""
    return {
        "entity": version.entity,
        "valid_from_ns": version.valid_from_ns,
        "valid_to_ns": version.valid_to_ns,
        "values": version.attributes,
        "key_parts": bitemporal.key_parts_of(version),
    }


def identity_log() -> list:
    """Every identity version the source records, in the order it records them."""
    day0 = at(0, SNAPSHOT_HOUR)
    first = [
        _profile(ALICE, "Finance", FRANK, "active", at(0), day0),
        _profile(BOB, "Engineering", GRACE, "active", at(0), day0),
        _profile(CAROL, "Sales", HEIDI, "active", at(0), day0),
        _profile(DAVE, "Finance", FRANK, "active", at(0), day0),
        _profile(ERIN, "Engineering", GRACE, "contractor", at(0), day0),
        _member(ALICE, "finance-users", at(0), day0),
        _member(ALICE, "expense-approvers", at(0), day0),
        _member(BOB, "eng-users", at(0), day0),
        _member(CAROL, "sales-users", at(0), day0),
        _member(DAVE, "finance-users", at(0), day0),
        _member(ERIN, "eng-users", at(0), day0),
    ]

    # The first snapshot goes through the diff too. Against an empty store it asserts everything, which is the
    # point: one code path for every snapshot, including the first.
    log: list = []

    for kind in (PROFILE, bitemporal.MEMBERSHIP):
        log += _snapshot(log, kind, [_fact(version) for version in first if version.kind == kind], day0)

    move = at(MOVE_DAY, DELTA_HOUR)
    log += [
        _profile(BOB, "Finance", FRANK, "active", at(MOVE_DAY), move),
        _member(BOB, "eng-users", at(MOVE_DAY), move, bitemporal.RETRACT),
        _member(BOB, "finance-users", at(MOVE_DAY), move),
    ]

    # The second snapshot restates everyone but Erin, as the source holds them on day fifteen. Dave is still active
    # in it, because the source has not heard yet.
    snapshot = at(SNAPSHOT_DAY, SNAPSHOT_HOUR)
    held = BitemporalStore("held", log)
    still_here = [ALICE, BOB, CAROL, DAVE]

    for kind in (PROFILE, bitemporal.MEMBERSHIP):
        facts = [
            _fact(version) for principal in still_here for version in held.facts_about(principal, snapshot, snapshot)
            if version.kind == kind
        ]
        log += _snapshot(log, kind, facts, snapshot)

    late = at(TERMINATION_RECORDED_DAY, DELTA_HOUR)
    log += [
        _profile(DAVE, "Finance", FRANK, "terminated", at(TERMINATION_DAY), late),
        _member(DAVE, "finance-users", at(TERMINATION_DAY), late, bitemporal.RETRACT),
    ]

    log += [_profile(CAROL, "Marketing", HEIDI, "active", at(0), at(CORRECTION_RECORDED_DAY, DELTA_HOUR))]

    return log


def asset_log() -> list:
    """Every asset version the inventory records, in the order it records them."""
    day0 = at(0, SNAPSHOT_HOUR)
    first = [
        _asset(ALICE_WORKSTATION, ALICE, "finance-it", "medium", "internal", "finance-workstations", at(0), day0),
        _asset(BOB_WORKSTATION, BOB, "eng-it", "medium", "internal", "eng-workstations", at(0), day0),
        _asset(LEDGER, FRANK, "finance-it", "high", "confidential", "databases", at(0), day0),
        _asset(BUILDER, GRACE, "eng-platform", "low", "internal", "build-servers", at(0), day0),
    ]
    log = _snapshot([], ASSET, [_fact(version) for version in first], day0)

    log += [
        _asset(LEDGER,
               FRANK,
               "finance-it",
               "high",
               "restricted",
               "databases",
               at(RECLASSIFIED_DAY),
               at(RECLASSIFIED_RECORDED_DAY, DELTA_HOUR)),
        _asset(BOB_WORKSTATION,
               BOB,
               "eng-it",
               "medium",
               "internal",
               "finance-workstations",
               at(MOVE_DAY),
               at(PEER_MOVE_RECORDED_DAY, DELTA_HOUR)),
    ]

    snapshot = at(SNAPSHOT_DAY, SNAPSHOT_HOUR)
    held = BitemporalStore("held", log)
    facts = [
        _fact(version) for host in (ALICE_WORKSTATION, BOB_WORKSTATION, LEDGER)
        for version in held.facts_about(host, snapshot, snapshot)
    ]
    log += _snapshot(log, ASSET, facts, snapshot)

    return log


def _rows(log: list, entity_column: str, columns: typing.Sequence[str], group_column: str = None) -> list:
    rows = []

    for version in log:
        row = {entity_column: version.entity}

        for name in columns:
            row[name] = version.attributes.get(name) if version.kind != bitemporal.MEMBERSHIP else None

        if (group_column is not None):
            row[group_column] = version.attributes.get(group_column) if version.kind == bitemporal.MEMBERSHIP else None

        row.update({
            bitemporal.VALID_FROM: version.valid_from_ns,
            bitemporal.VALID_TO: version.valid_to_ns,
            bitemporal.RECORDED_AT: version.recorded_ns,
            bitemporal.CHANGE: version.change,
        })
        rows.append(row)

    return rows


def _frame(rows: list) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame[bitemporal.VALID_TO] = frame[bitemporal.VALID_TO].astype("Int64")
    frame[bitemporal.RECORDED_AT] = frame[bitemporal.RECORDED_AT].astype("Int64")
    frame = frame.sort_values([bitemporal.RECORDED_AT], kind="mergesort").reset_index(drop=True)
    frame.insert(0, "source_seq", range(len(frame)))

    return frame


UNRECORDED_ROW = {
    "user_principal": FRANK,
    "group_name": None,
    "department": "Finance",
    "manager": None,
    "employment_status": "active",
    bitemporal.VALID_FROM: at(0),
    bitemporal.VALID_TO: None,
    bitemporal.RECORDED_AT: None,
    bitemporal.CHANGE: bitemporal.ASSERT,
}
"""A record with no instant of recording. Refused rather than stamped with the moment it arrived."""

INVERTED_ROW = {
    "user_principal": GRACE,
    "group_name": "eng-users",
    "department": None,
    "manager": None,
    "employment_status": None,
    bitemporal.VALID_FROM: at(9),
    bitemporal.VALID_TO: at(3),
    bitemporal.RECORDED_AT: at(9, DELTA_HOUR),
    bitemporal.CHANGE: bitemporal.ASSERT,
}
"""A membership that ends before it begins. Refused."""


def _probes(entity_column: str, probes: list) -> pd.DataFrame:
    frame = pd.DataFrame([{
        "probe_id": probe_id, entity_column: entity, "event_time": event_time
    } for (probe_id, entity, event_time) in probes])

    return frame.sort_values("event_time", kind="mergesort").reset_index(drop=True)


IDENTITY_PROBE_EVENTS = [
    ("alice-day3", ALICE, at(3, 12)),
    ("bob-before-move", BOB, at(5, 12)),
    ("bob-after-move", BOB, at(12, 12)),
    ("carol-before-correction", CAROL, at(5, 12)),
    ("carol-after-correction", CAROL, at(22, 12)),
    ("dave-unreported-leaver", DAVE, at(14, 12)),
    ("dave-reported-leaver", DAVE, at(19, 12)),
    ("erin-before-snapshot", ERIN, at(14, 12)),
    ("erin-after-snapshot", ERIN, at(16, 12)),
    ("mallory-unknown", MALLORY, at(5, 12)),
]

ASSET_PROBE_EVENTS = [
    ("alice-workstation-day2", ALICE_WORKSTATION, at(2, 12)),
    ("bob-workstation-unnoticed-move", BOB_WORKSTATION, at(MOVE_DAY, 12)),
    ("ledger-before-reclassification-recorded", LEDGER, at(6, 12)),
    ("ledger-after-reclassification-recorded", LEDGER, at(8, 12)),
    ("builder-after-decommission", BUILDER, at(16, 12)),
]


def build_corpus() -> dict[str, pd.DataFrame]:
    """The fixed corpus: the identity and asset source records, and the probe events for each."""
    identity_rows = _rows(identity_log(), "user_principal", DEFAULT_PROFILE_COLUMNS, group_column="group_name")
    identity_rows += [dict(UNRECORDED_ROW), dict(INVERTED_ROW)]

    return {
        IDENTITY_CLASS: _frame(identity_rows),
        ASSET_CLASS: _frame(_rows(asset_log(), "hostname", DEFAULT_ASSET_COLUMNS)),
        IDENTITY_PROBES: _probes("user_principal", IDENTITY_PROBE_EVENTS),
        ASSET_PROBES: _probes("hostname", ASSET_PROBE_EVENTS),
    }


def build_pipeline_config(execution_mode=None) -> Config:
    """
    A pipeline configuration, defaulting to CPU mode and importable without a GPU.

    Parameters
    ----------
    execution_mode : `morpheus.config.ExecutionMode`, optional
        Mode to build for. Defaults to CPU, which is what the golden file is an artifact of.
    """
    from morpheus.config import CppConfig  # pylint: disable=import-outside-toplevel
    from morpheus.config import ExecutionMode  # pylint: disable=import-outside-toplevel

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.execution_mode = ExecutionMode.CPU if execution_mode is None else execution_mode

    return config


def _collect(sink: InMemorySinkStage) -> pd.DataFrame:
    """Everything the sink received, as one host frame."""
    frames = []

    for message in sink.get_messages():
        meta = message.payload() if hasattr(message, "payload") else message
        frame = meta.copy_dataframe()
        frames.append(frame.to_pandas() if hasattr(frame, "to_pandas") else frame)

    if (len(frames) == 0):
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _run(config: Config, stage, dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(stage)
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    return _collect(sink)


def build_store(name: str, produced: pd.DataFrame) -> BitemporalStore:
    """Rebuild a store from a producer's output, the way a consumer of the sourcetype would."""
    return BitemporalStore.from_records(name, produced.to_dict("records"))


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None) -> pd.DataFrame:
    """
    Run both producers, rebuild both stores from what they emitted, and enrich the probes against each twice.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    corpus : dict
        The frames from `build_corpus`, possibly permuted.
    batches : dict, optional
        Per corpus class, how it is split across source frames. Defaults to one frame per class.

    Returns
    -------
    `pandas.DataFrame`
        The producers' output and the enriched probes, tagged with `telemetry_class`, keyed by `row_key`,
        canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    identity = _run(config, TC0IdentityStage(config), batches[IDENTITY_CLASS])
    asset = _run(config, TC0AssetStage(config), batches[ASSET_CLASS])

    stores = {
        IDENTITY_PROBES: (build_store("identity", identity), "user_principal"),
        ASSET_PROBES: (build_store("asset", asset), "hostname"),
    }

    frames = []

    for (name, produced) in ((IDENTITY_CLASS, identity), (ASSET_CLASS, asset)):
        produced["telemetry_class"] = name
        # A refused row has no identifier of its own, so it is keyed by its position in the source instead.
        produced["row_key"] = [
            uid if isinstance(uid, str) else f"refused-{seq}"
            for (uid, seq) in zip(produced[bitemporal.CONTEXT_UID], produced["source_seq"])
        ]
        frames.append(produced)

    for name in PROBE_CLASSES:
        (store, entity_column) = stores[name]

        for knowledge in KNOWLEDGE_MODES:
            enriched = _run(config,
                            TC0EnrichStage(config, store=store, entity_column=entity_column, knowledge=knowledge),
                            [frame.copy() for frame in batches[name]])
            enriched["telemetry_class"] = name
            enriched["row_key"] = [f"{probe}@{knowledge}" for probe in enriched["probe_id"]]
            frames.append(enriched)

    combined = pd.concat(frames, ignore_index=True)
    # The probes carry no source position, and a column with gaps would otherwise widen to float and render as 4.0.
    combined["source_seq"] = combined["source_seq"].astype("Int64")

    return canonicalize(combined, key_columns=KEY_COLUMNS, ignore_columns=IGNORE_COLUMNS)


def render(result: pd.DataFrame) -> str:
    """The canonical CSV rendering the golden file holds."""
    return result.to_csv(index=False, lineterminator="\n")
