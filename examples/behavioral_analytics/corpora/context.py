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
The synthetic corpus `morpheus.compositions.context` is tested over, and the facts about it the harnesses assert.

Moved out of `tests/morpheus/determinism/context_pipeline.py` under plan section 4.5, which put the composition in the
package and left the data here, where the tests, the sample and fixture generators and the experiments read it. The
composition module's docstring describes what this corpus plants and why.
"""

import typing

import pandas as pd

from morpheus.compositions import envelope
from morpheus.compositions.context import APPLICATION_CLASS
from morpheus.compositions.context import APPLICATION_PROBES
from morpheus.compositions.context import ASSET_CLASS
from morpheus.compositions.context import ASSET_PROBES
from morpheus.compositions.context import IDENTITY_CLASS
from morpheus.compositions.context import IDENTITY_PROBES
from morpheus.compositions.context import PROBE_CLASSES
from morpheus.stages.telemetry.tc0_application_stage import APPLICATION
from morpheus.stages.telemetry.tc0_application_stage import DEFAULT_APPLICATION_COLUMNS
from morpheus.stages.telemetry.tc0_asset_stage import ASSET
from morpheus.stages.telemetry.tc0_asset_stage import DEFAULT_ASSET_COLUMNS
from morpheus.stages.telemetry.tc0_identity_stage import DEFAULT_PROFILE_COLUMNS
from morpheus.stages.telemetry.tc0_identity_stage import PROFILE
from morpheus.utils import bitemporal
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.entity_key import compose_key

DAY_SECONDS = 86400
PRODUCER_CLASSES = (IDENTITY_CLASS, ASSET_CLASS, APPLICATION_CLASS)
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
REPORTER = "svc-reports@example.com"
"""A service principal: a reporting job's identity, which the profile says is not a person."""
# Hosts.
ALICE_WORKSTATION = "ws-alice"
BOB_WORKSTATION = "ws-bob"
LEDGER = "db-ledger"
BUILDER = "build-07"
HOST_KINDS = {
    ALICE_WORKSTATION: ("workstation", "windows", "10.0.19045"),
    BOB_WORKSTATION: ("workstation", "macos", "14.6"),
    LEDGER: ("server", "linux", "9.4"),
    BUILDER: ("server", "linux", "22.04"),
}
"""What the inventory says each host is: its role, and the operating system family and version it runs."""
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
# Servers, as the CMDB keys them: an address and a port.
BILLING_SERVER = ("10.0.2.50", 8443)
LEDGER_API = ("10.0.2.51", 443)
REASSIGN_DAY = 9
REASSIGN_RECORDED_DAY = 10
"""The billing service's port is handed to the reporting service on day nine; the CMDB is updated on day ten."""
APPLICATIONS = {
    "billing": ("finance-apps", "svc-billing", "high", "confidential"),
    "reporting": ("analytics", "svc-reports", "medium", "internal"),
    "ledger-api": ("finance-apps", "svc-ledger", "high", "restricted"),
}
"""Each application's owning team, service account, criticality and data classification."""


def at(day: float, hour: float = 0) -> int:
    """An instant `day` days and `hour` hours into the corpus, in nanoseconds since the epoch."""
    return int((day * DAY_SECONDS + hour * 3600) * NS_PER_SECOND)


SNAPSHOT_HOUR = 6
DELTA_HOUR = 9


def _profile(principal,
             department,
             manager,
             status,
             valid_from,
             recorded,
             change=bitemporal.ASSERT,
             account_type="human",
             privilege="standard"):
    return make_version(
        PROFILE,
        principal,
        valid_from,
        None,
        recorded,
        change,
        {
            "department": department,
            "manager": manager,
            "employment_status": status,
            "account_type": account_type,
            "privilege_level": privilege,
        })


def _member(principal, group, valid_from, recorded, change=bitemporal.ASSERT):
    return make_version(bitemporal.MEMBERSHIP,
                        principal,
                        valid_from,
                        None,
                        recorded,
                        change, {"group_name": group}, (group, ))


def _asset(host, owner, team, criticality, classification, peer, valid_from, recorded, change=bitemporal.ASSERT):
    (role, os_family, os_version) = HOST_KINDS[host]

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
            "device_role": role,
            "os_family": os_family,
            "os_version": os_version,
        })


def _application(server, name, valid_from, recorded, change=bitemporal.ASSERT):
    (team, account, criticality, classification) = APPLICATIONS[name]

    return make_version(
        APPLICATION,
        compose_key(list(server)),
        valid_from,
        None,
        recorded,
        change,
        {
            "name": name,
            "owning_team": team,
            "service_account": account,
            "criticality": criticality,
            "data_classification": classification,
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
        _profile(ALICE, "Finance", FRANK, "active", at(0), day0, privilege="elevated"),
        _profile(BOB, "Engineering", GRACE, "active", at(0), day0),
        _profile(CAROL, "Sales", HEIDI, "active", at(0), day0),
        _profile(DAVE, "Finance", FRANK, "active", at(0), day0),
        _profile(ERIN, "Engineering", GRACE, "contractor", at(0), day0),
        _profile(REPORTER, "Engineering", GRACE, "active", at(0), day0, account_type="service", privilege="elevated"),
        _member(ALICE, "finance-users", at(0), day0),
        _member(ALICE, "expense-approvers", at(0), day0),
        _member(BOB, "eng-users", at(0), day0),
        _member(CAROL, "sales-users", at(0), day0),
        _member(DAVE, "finance-users", at(0), day0),
        _member(ERIN, "eng-users", at(0), day0),
        _member(REPORTER, "reporting-services", at(0), day0),
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
    still_here = [ALICE, BOB, CAROL, DAVE, REPORTER]

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


def application_log() -> list:
    """Every application-to-server version the CMDB records, in the order it records them."""
    day0 = at(0, SNAPSHOT_HOUR)
    log = [_application(BILLING_SERVER, "billing", at(0), day0), _application(LEDGER_API, "ledger-api", at(0), day0)]
    log += [_application(BILLING_SERVER, "reporting", at(REASSIGN_DAY), at(REASSIGN_RECORDED_DAY, DELTA_HOUR))]

    return log


def _application_rows(log: list) -> list:
    """The CMDB's records as its export states them: the server's address and port in two columns."""
    rows = []

    for version in log:
        (address, port) = version.entity.rsplit(":", 1)
        row = {"server_address": address, "server_port": int(port)}
        row.update({name: version.attributes.get(name) for name in DEFAULT_APPLICATION_COLUMNS})
        row.update({
            bitemporal.VALID_FROM: version.valid_from_ns,
            bitemporal.VALID_TO: version.valid_to_ns,
            bitemporal.RECORDED_AT: version.recorded_ns,
            bitemporal.CHANGE: version.change,
        })
        rows.append(row)

    return rows


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
    "account_type": "human",
    "privilege_level": "standard",
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
    "account_type": None,
    "privilege_level": None,
    bitemporal.VALID_FROM: at(9),
    bitemporal.VALID_TO: at(3),
    bitemporal.RECORDED_AT: at(9, DELTA_HOUR),
    bitemporal.CHANGE: bitemporal.ASSERT,
}
"""A membership that ends before it begins. Refused."""


def _server_probes(probes: list) -> pd.DataFrame:
    frame = pd.DataFrame([{
        "probe_id": probe_id, "dst_ip": server[0], "dst_port": server[1], "event_time": event_time
    } for (probe_id, server, event_time) in probes])

    return frame.sort_values("event_time", kind="mergesort").reset_index(drop=True)


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
    ("reporter-service-account", REPORTER, at(3, 12)),
]
ASSET_PROBE_EVENTS = [
    ("alice-workstation-day2", ALICE_WORKSTATION, at(2, 12)),
    ("bob-workstation-unnoticed-move", BOB_WORKSTATION, at(MOVE_DAY, 12)),
    ("ledger-before-reclassification-recorded", LEDGER, at(6, 12)),
    ("ledger-after-reclassification-recorded", LEDGER, at(8, 12)),
    ("builder-after-decommission", BUILDER, at(16, 12)),
]
APPLICATION_PROBE_EVENTS = [
    ("billing-before-reassignment", BILLING_SERVER, at(5, 12)),
    ("billing-port-unrecorded-reassignment", BILLING_SERVER, at(REASSIGN_DAY, 12)),
    ("billing-port-after-recording", BILLING_SERVER, at(12, 12)),
    ("ledger-api-day3", LEDGER_API, at(3, 12)),
    ("unmapped-server", ("10.0.2.99", 443), at(3, 12)),
]


def build_corpus() -> dict[str, pd.DataFrame]:
    """The fixed corpus: the identity, asset and application source records, and the probe events for each."""
    identity_rows = _rows(identity_log(), "user_principal", DEFAULT_PROFILE_COLUMNS, group_column="group_name")
    identity_rows += [dict(UNRECORDED_ROW), dict(INVERTED_ROW)]

    corpus = {
        IDENTITY_CLASS: _frame(identity_rows),
        ASSET_CLASS: _frame(_rows(asset_log(), "hostname", DEFAULT_ASSET_COLUMNS)),
        IDENTITY_PROBES: _probes("user_principal", IDENTITY_PROBE_EVENTS),
        ASSET_PROBES: _probes("hostname", ASSET_PROBE_EVENTS),
        APPLICATION_CLASS: _frame(_application_rows(application_log())),
        APPLICATION_PROBES: _server_probes(APPLICATION_PROBE_EVENTS),
    }

    # A context record is bitemporal: its valid time and its recording time are its clock, and it has no sensor or
    # sampling for the collector envelope to describe. What it shares with every other record is the tenant it
    # belongs to.
    for frame in corpus.values():
        frame["tenant_id"] = envelope.TENANT

    return corpus
