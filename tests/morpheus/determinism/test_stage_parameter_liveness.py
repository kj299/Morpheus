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
Every tunable parameter of every stage in this fork, and proof that changing it changes something.

A parameter that is validated, stored, and never consulted is indistinguishable from one that works. It has a
docstring, a constructor check, and a test asserting the constructor check -- and it does nothing. Four defects
found by audit in this fork were exactly that shape: two expiry timeouts that were stored and never read, an
exchange key that ignored the supplicant it was given, and a key rendering that followed the column's dtype
instead of the value. All four sat behind a green suite, because no test made the parameter's condition arise.

This file is the floor. Each stage's parameters are listed in `REGISTRY`, each with the evidence that it is live:

- `DIFFERS`: run the stage twice over a corpus built to make the parameter bite, and the outputs must differ. A
  parameter nothing reads produces identical output at both values.
- `INPUT_COLUMN`: rename the column in the frame and pass the new name. A stage that ignores the parameter looks
  for the default name, which is no longer there.
- `RAISES`: the value exists to make the stage refuse; it must refuse.
- `INERT`: declared dead, with a reason. Nothing runs. This is the honest escape hatch, and it is deliberately
  ugly to use: the reason is checked for length and shows up in the test name.

`test_every_stage_parameter_is_registered` closes the loop by reading each constructor's signature, so a
parameter added later cannot ship without an entry here.
"""

import dataclasses
import inspect
import typing

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import ExecutionMode
from morpheus.messages import MessageMeta
from morpheus.stages.lineage.binding_resolver_stage import BindingResolverStage
from morpheus.stages.lineage.chain_anchor_stage import ChainAnchorStage
from morpheus.stages.lineage.community_id_stage import CommunityIdStage
from morpheus.stages.lineage.determinism_stamp_stage import DeterminismStampStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.minimization_stage import MinimizationStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.siem_wire_stage import SiemWireStage
from morpheus.stages.telemetry.tc1_binding_stage import TC1BindingStage
from morpheus.stages.telemetry.tc1_change_stage import TC1ChangeStage
from morpheus.stages.telemetry.tc1_feature_stage import TC1FeatureStage
from morpheus.stages.telemetry.tc1_flap_stage import TC1FlapStage
from morpheus.stages.telemetry.tc1_normalize_stage import TC1NormalizeStage
from morpheus.stages.telemetry.tc1_optical_stage import TC1OpticalStage
from morpheus.stages.telemetry.tc2_arp_stage import TC2ArpStage
from morpheus.stages.telemetry.tc2_auth_stage import TC2AuthStage
from morpheus.stages.telemetry.tc2_binding_stage import TC2BindingStage
from morpheus.stages.telemetry.tc2_cardinality_stage import TC2CardinalityStage
from morpheus.stages.telemetry.tc3_beacon_stage import TC3BeaconStage
from morpheus.stages.telemetry.tc3_cardinality_stage import TC3CardinalityStage
from morpheus.stages.telemetry.tc3_reach_stage import TC3ReachStage
from morpheus.stages.telemetry.tc3_ttl_stage import TC3TtlStage
from morpheus.stages.telemetry.tc4_envelope_stage import TC4EnvelopeStage
from morpheus.stages.telemetry.tc4_flow_stage import TC4FlowStage
from morpheus.stages.telemetry.tc6_certificate_stage import TC6CertificateStage
from morpheus.stages.telemetry.tc6_cipher_stage import TC6CipherStage
from morpheus.stages.telemetry.tc6_content_stage import TC6ContentStage
from morpheus.stages.telemetry.tc6_fingerprint_stage import TC6FingerprintStage
from morpheus.stages.telemetry.tc0_asset_stage import TC0AssetStage
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.stages.telemetry.tc0_identity_stage import TC0IdentityStage
from morpheus.stages.telemetry.tc7_dns_stage import TC7DnsStage
from morpheus.stages.telemetry.tc7_http_stage import TC7HttpStage
from morpheus.stages.telemetry.tc7_endpoint_stage import TC7EndpointStage
from morpheus.stages.telemetry.tc7_saas_stage import TC7SaasStage
from morpheus.stages.telemetry.tc5_cadence_stage import TC5CadenceStage
from morpheus.stages.telemetry.tc5_drift_stage import TC5DriftStage
from morpheus.stages.telemetry.tc5_novelty_stage import TC5NoveltyStage
from morpheus.stages.telemetry.tc5_risk_stage import TC5RiskStage
from morpheus.stages.telemetry.tc5_score_stage import TC5ScoreStage
from morpheus.stages.telemetry.tc5_session_stage import TC5SessionStage
from morpheus.stages.telemetry.tc5_travel_stage import TC5TravelStage
from morpheus.utils.binding_table import Binding
from morpheus.utils.binding_table import BindingTable
from morpheus.utils.bitemporal import MEMBERSHIP
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.determinism_envelope import DeterminismEnvelope
from morpheus.utils.model_manifest import ModelManifest
from morpheus.utils.tcp_flags import ACK
from morpheus.utils.tcp_flags import PSH
from morpheus.utils.tcp_flags import RST
from morpheus.utils.tcp_flags import SYN

DIFFERS = "differs"
INPUT_COLUMN = "input_column"
RAISES = "raises"
INERT = "inert"

SECOND = 10**9
MINUTE = 60 * SECOND
HOUR = 60 * MINUTE


@dataclasses.dataclass(frozen=True)
class Knob:
    """One constructor parameter, and how this file proves the stage reads it."""

    name: str
    kind: str
    benign: typing.Any = None
    extreme: typing.Any = None
    reason: str = ""
    frame: typing.Optional[typing.Callable] = None
    """A frame for this parameter alone, where the scenario's own frame cannot make it bite."""

    also: typing.Optional[dict] = None
    """Companion overrides, for a column that more than one parameter names."""

    def __post_init__(self):
        if (self.kind == INERT and len(self.reason) < 30):
            raise ValueError(f"{self.name}: declaring a parameter inert requires saying why, at some length")


@dataclasses.dataclass(frozen=True)
class Scenario:
    """A stage, a frame that makes its parameters bite, and the parameters."""

    stage: type
    frame: typing.Callable
    base: dict
    knobs: tuple


def _config() -> Config:
    CppConfig.set_should_use_cpp(False)
    config = Config()
    config.execution_mode = ExecutionMode.CPU

    return config


def _host_frames(payload) -> list:
    """Everything a stage emitted, as host DataFrames."""
    items = payload if isinstance(payload, list) else [payload]
    frames = []

    for item in items:
        meta = item.payload() if hasattr(item, "payload") else item
        frame = meta.copy_dataframe() if hasattr(meta, "copy_dataframe") else meta
        frames.append(frame.to_pandas() if hasattr(frame, "to_pandas") else frame)

    return frames


def _run(scenario: Scenario, overrides: dict, frame: dict = None) -> list:
    """Construct the stage with these overrides, push one frame through it, and collect everything it emitted."""
    kwargs = dict(scenario.base)
    kwargs.update(overrides)
    stage = scenario.stage(_config(), **kwargs)
    meta = MessageMeta(pd.DataFrame(scenario.frame() if frame is None else frame))
    emitted = stage.on_data(meta)

    if (hasattr(stage, "on_completed")):
        remaining = stage.on_completed()

        if (remaining):
            emitted = (emitted if isinstance(emitted, list) else [emitted]) + list(remaining)

    return _host_frames(emitted if emitted is not None else meta)


def _outcome(scenario: Scenario, overrides: dict, frame: dict = None) -> str:
    """
    What the stage did with this value: its output, or the way it refused.

    Refusing differently is evidence too. A time unit that reads nanoseconds as seconds puts the timestamp past
    what a `Timestamp` can hold, and overflowing is not the same behaviour as succeeding -- a parameter nothing
    consulted could not have caused it.
    """
    try:
        return _text(_run(scenario, overrides, frame=frame))
    except Exception as error:  # pylint: disable=broad-except
        return f"refused: {type(error).__name__}"


def _text(frames: list, rename: dict = None) -> str:
    """Frames as text, columns in a fixed order, so two runs compare without caring about dtypes."""
    rendered = []

    for frame in frames:
        if (rename):
            frame = frame.rename(columns=rename)

        rendered.append(frame.reindex(sorted(frame.columns), axis=1).to_csv(index=False))

    return "\n".join(rendered)


# --- The frames -----------------------------------------------------------------------------------------------


def counters() -> dict:
    # A baseline, a steady increase, a reboot, and a counter that wraps past the 32-bit ceiling.
    return {
        "site_id": ["hq"] * 4,
        "device_id": ["sw1"] * 4,
        "port_id": ["Gi1/0/1"] * 4,
        "event_time": [0, MINUTE, 2 * MINUTE, 3 * MINUTE],
        "uptime": [360000, 366000, 3000, 9000],
        "crc_errors": [100, 142, (1 << 32) - 5, 7],
        "symbol_errors": [0, 0, 0, 3],
        "input_discards": [5, 5, 0, 0],
        "output_discards": [1, 1, 0, 0],
    }


def optics() -> dict:
    # A steady receive level, then a persistent three-decibel step: a tap.
    levels = [-5.0] * 8 + [-8.0] * 4

    return {
        "entity_key": ["hq:sw1:Gi1/0/1"] * len(levels),
        "event_time": [index * MINUTE for index in range(len(levels))],
        "optical_rx_dbm": levels,
        "optical_tx_dbm": [-2.0] * len(levels),
    }


def flaps() -> dict:
    statuses = ["up", "down", "up", "up", "up", "up"]

    # The fourth and fifth polls both read "up", but the device says the link changed between them: a flap the
    # poller never saw. Only a stage reading the last-change column can know it happened.
    return {
        "entity_key": ["hq:sw1:Gi1/0/1"] * len(statuses),
        "event_time": [index * MINUTE for index in range(len(statuses))],
        "oper_status": statuses,
        "if_last_change": [0, 6000, 12000, 18000, 24000, 24000],
    }


def identifiers() -> dict:
    serials = ["sn-a", "sn-b", "sn-a", "sn-c"]

    return {
        "entity_key": ["hq:sw1:Gi1/0/1"] * len(serials),
        "event_time": [index * HOUR for index in range(len(serials))],
        "collector_id": ["poller-1"] * len(serials),
        "collector_seq": list(range(len(serials))),
        "transceiver_serial": serials,
        "lldp_neighbor_chassis_id": ["chassis-a", "chassis-a", "chassis-b", "chassis-b"],
    }


def tied_identifiers() -> dict:
    """Two rows a sort on the order columns cannot separate: the condition `require_total_order` exists for."""
    frame = identifiers()
    frame["collector_seq"] = [0, 0, 1, 2]
    frame["event_time"] = [0, 0, HOUR, 2 * HOUR]

    return frame


def port_inventory() -> dict:
    """Polls of two ports, with one optic swapped and one neighbour repatched, so a binding closes for each cause."""
    serials = ["SN-AAA", "SN-AAA", "SN-BBB", "SN-CCC", "SN-CCC"]
    neighbors = ["chassis-a", "chassis-a", "chassis-a", "chassis-b", "chassis-c"]
    ports = ["Gi1/0/1", "Gi1/0/1", "Gi1/0/1", "Gi1/0/2", "Gi1/0/2"]

    return {
        "site_id": ["hq"] * len(serials),
        "device_id": ["sw1"] * len(serials),
        "port_id": ports,
        "event_time": [index * MINUTE for index in range(len(serials))],
        "transceiver_serial": serials,
        "lldp_neighbor_chassis_id": neighbors,
    }


MINIMIZATION_KEY = b"a-key-long-enough-to-be-accepted"
OTHER_MINIMIZATION_KEY = b"a-different-key-of-the-same-size"


def two_principals_at_layer_five() -> dict:
    """Two principals, each named twice, with a laptop and an address beside them.

    `device_id` is here because it is the column whose category depends on the class: a laptop at layer 5 and a
    switch at layer 1. It is what makes `telemetry_class` observable without contriving anything.
    """
    return {
        "user_principal": ["alice@example.com", "bob@example.com", "alice@example.com"],
        "entity_key": ["alice@example.com", "bob@example.com", "alice@example.com"],
        "device_id": ["laptop-alice", "laptop-bob", "laptop-alice"],
        "source_ip": ["203.0.113.5", "203.0.113.6", "198.51.100.7"],
        "source_country": ["US", "US", "FR"],
        "logcount": [4, 7, 9],
    }


def a_third_name_for_the_same_person() -> dict:
    """The same principals with the 802.1X identity beside them, which the policy below does not cover.

    The condition `allow_unmasked_copies` exists for: a column the policy left alone, still holding a value the
    policy digested somewhere else.
    """
    return {
        "user_principal": ["alice@example.com", "bob@example.com"],
        "entity_key": ["alice@example.com", "bob@example.com"],
        "desk_identity": ["alice@example.com", "bob@example.com"],
        "logcount": [4, 7],
    }


def flow_records() -> dict:
    """Eight flows from two sources, spread far enough apart that a window parameter has something to cut."""
    return {
        "src_ip": ["10.0.0.5"] * 6 + ["10.0.0.6"] * 2,
        "dst_ip": ["10.0.0.9", "10.0.0.10", "8.8.8.8", "10.0.0.11", "8.8.4.4", "10.0.0.12", "10.0.0.9", "8.8.8.8"],
        "dst_port": [443, 445, 443, 22, 443, 445, 443, 80],
        "bgp_as_dst": ["64512", "64512", "15169", "64512", "15169", "64512", "64512", "15169"],
        "bytes_out": [100, 200, 300, 400, 500, 600, 700, 800],
        "bytes_in": [10, 20, 30, 40, 50, 60, 70, 80],
        "ip_ttl": [64, 64, 64, 64, 64, 63, 128, 128],
        "event_time": [10**18 + index * 600 * SECOND for index in range(8)],
    }


def packets() -> dict:
    """Two flows interleaved across four minutes, each with packets either side of a sixty-second boundary.

    Interleaved because a flow cap is only observable mid-stream: run one flow and then the other and a cap of
    one evicts the first after its last row has already been emitted, so the parameter reads as dead when it is
    merely too late to matter. Either side of a boundary because a bin width nothing reads produces one running
    total whatever it is set to.
    """
    rows = []

    for index in range(6):
        base = 10**18 + index * 40 * SECOND
        rows.append((base, "10.0.0.5", 50000, "10.0.0.9", 443, SYN if index == 0 else PSH | ACK, index * 100))
        rows.append((base + 5 * SECOND, "10.0.0.6", 50001, "10.0.0.10", 445, RST | ACK, 0))

    rows.sort()

    return {
        "src_ip": [row[1] for row in rows],
        "src_port": [row[2] for row in rows],
        "dst_ip": [row[3] for row in rows],
        "dst_port": [row[4] for row in rows],
        "tcp_flags": [row[5] for row in rows],
        "data_len": [row[6] for row in rows],
        "event_time": [row[0] for row in rows],
    }


def dns_queries() -> dict:
    """Ten names under one domain spread across twenty minutes, so a window or a cap has something to cut."""
    names = [f"chunk{index}x9q2k7m4v8b3n6c1z5p0w.evil.com" for index in range(10)]

    return {
        "query_name": names,
        "event_time": [10**18 + index * 120 * SECOND for index in range(10)],
    }


def _two_domains() -> dict:
    """Two registered domains interleaved, so a cap of one entity evicts one of them mid-stream."""
    names = []
    times = []

    for index in range(6):
        names.append(f"a{index}x9q2k7m4v8b3n6c1z5p0w.evil.com")
        names.append(f"b{index}x9q2k7m4v8b3n6c1z5p0w.other.com")
        times.extend([10**18 + index * 60 * SECOND, 10**18 + index * 60 * SECOND + SECOND])

    return {"query_name": names, "event_time": times}


def http_requests() -> dict:
    """Two clients interleaved, each requesting distinct paths with a mix of statuses across twenty minutes.

    Interleaved because an entity cap is only observable mid-stream, for the same reason it is in `paired_flows`.
    """
    rows = []

    for index in range(10):
        base = 10**18 + index * 120 * SECOND
        rows.append((base, "10.0.0.5", f"/p{index}", 404 if index % 3 else 200))
        rows.append((base + 5 * SECOND, "10.0.0.6", f"/q{index}", 200))

    rows.sort()

    return {
        "event_time": [row[0] for row in rows],
        "src_ip": [row[1] for row in rows],
        "url_path": [row[2] for row in rows],
        "status_code": [row[3] for row in rows],
    }


DAY = 24 * HOUR


def identity_records() -> dict:
    """Profiles and memberships: one membership retracted, and one move recorded two days late."""
    return {
        "user_principal": ["alice", "alice", "bob", "bob"],
        "group_name": [None, "finance-users", None, None],
        "department": ["Finance", None, "Engineering", "Finance"],
        "manager": ["frank", None, "grace", "frank"],
        "employment_status": ["active", None, "active", "active"],
        "valid_from": [0, 0, 0, 10 * DAY],
        "valid_to": [None, 5 * DAY, None, None],
        "recorded_at": [DAY, DAY, DAY, 12 * DAY],
        "change": ["assert", "retract", "assert", "assert"],
    }


def asset_records() -> dict:
    return {
        "hostname": ["db-ledger", "db-ledger", "ws-01"],
        "owner": ["frank", "frank", "alice"],
        "owning_team": ["finance-it", "finance-it", "finance-it"],
        "criticality": ["high", "high", "medium"],
        "data_classification": ["confidential", "restricted", "internal"],
        "peer_group": ["databases", "databases", "finance-workstations"],
        "valid_from": [0, 5 * DAY, 0],
        "valid_to": [None, None, None],
        "recorded_at": [0, 7 * DAY, 0],
        "change": ["assert", "assert", "retract"],
    }


def _context_store(correction: str = "Marketing") -> BitemporalStore:
    return BitemporalStore(
        "identity",
        [
            make_version("profile", "carol", 0, None, 0, values={"department": "Sales"}),
            make_version("profile", "carol", 0, None, 20 * DAY, values={"department": correction}),
            make_version(
                MEMBERSHIP, "carol", 0, None, 0, values={"group_name": "sales-users"}, key_parts=("sales-users", )),
        ])


def context_probes() -> dict:
    return {"user_principal": ["carol", "carol", "mallory"], "event_time": [5 * DAY, 25 * DAY, 5 * DAY]}


def saas_operations() -> dict:
    """Two principals' exports and queries across two weeks, one export failing, one far above its history."""
    monday = 4 * DAY
    rows = [(principal, operation, count, result, object_type, monday + index * HOUR)
            for (index,
                 (principal, operation, count, result,
                  object_type)) in enumerate([("ann", "Export", 10, "success",
                                               "A"), ("ann", "Query", 3000, "success",
                                                      "B"), ("ann", "Export", 12, "success",
                                                             "A"), ("ann", "Query", 3000, "success",
                                                                    "C"), ("bob", "Export", 10, "success",
                                                                           "A"), ("ann", "Export", 11, "success",
                                                                                  "D"), ("ann", "Export", 400, "denied",
                                                                                         "E"), ("ann", "Export", 200,
                                                                                                "success", "F")])]
    rows.append(("ann", "Export", 30, "success", "A", monday + 6 * DAY + 23 * HOUR))
    rows.append(("ann", "Export", 30, "success", "B", monday + 7 * DAY + HOUR))

    return {
        "user_principal": [row[0] for row in rows],
        "operation": [row[1] for row in rows],
        "record_count": [row[2] for row in rows],
        "result": [row[3] for row in rows],
        "target_object_type": [row[4] for row in rows],
        "event_time": [row[5] for row in rows],
    }


def process_starts() -> dict:
    """Three hosts, two in one peer group, returning to pairs after gaps of a day or two.

    The first host alternates between two pairs, so a recall bound of one forgets each before it returns; the two
    grouped hosts interleave, so an entity bound of one forgets each host between its rows; and the first rows fall
    inside a day of the group's first sighting, so the warm-up decides whether they are judged at all.
    """
    rows = [("h1", "a.exe", 0.0, "g", "High"), ("h2", "b.exe", 0.5, "g", "Medium"), ("h1", "b.exe", 1.5, "g", "Low"),
            ("h1", "a.exe", 2.0, "g", "System"), ("h3", "a.exe", 2.5, None, "High"),
            ("h2", "c.exe", 3.0, "g", "Medium"), ("h1", "b.exe", 3.5, "g", None)]

    return {
        "hostname": [row[0] for row in rows],
        "parent_image_path": [r"C:\Windows\explorer.exe"] * len(rows),
        "image_path": [rf"C:\Tools\{row[1]}" for row in rows],
        "event_time": [int(row[2] * 24 * HOUR) for row in rows],
        "ctx_peer_group": [row[3] for row in rows],
        "integrity_level": [row[4] for row in rows],
    }


def handshakes() -> dict:
    """Twelve handshakes on one pair, then three on another interleaved among them.

    Interleaved because an entity cap is only observable mid-stream: run the pairs one after the other and a cap
    of one evicts the first after its last row has already been emitted, so the parameter reads as dead when it
    is merely too late to matter.

    The issuer settles and then changes, the suite descends partway through, and the content types cross a
    category once, so every parameter these four stages take has something in here to bite on.
    """
    rows = []

    for index in range(12):
        rows.append((10**18 + index * 60 * SECOND,
                     "10.0.0.5",
                     "93.184.216.34",
                     "t13d1516h2_8daaf6152771_b186095e22b6" if index < 10 else "t13d0000h0_aaaa_bbbb",
                     "CN=Corp CA" if index < 10 else "CN=Proxy",
                     "TLS_AES_128_GCM_SHA256" if index < 10 else "TLS_RSA_WITH_3DES_EDE_CBC_SHA",
                     "ok" if index < 10 else "self-signed",
                     "image/png",
                     "image/jpeg" if index < 10 else "application/zip"))

    for offset in (30, 390, 750):
        rows.append((10**18 + offset * SECOND,
                     "10.0.0.6",
                     "93.184.216.35",
                     "t13d0312h2_55b375c5d22e_cd85d2d88918",
                     "CN=Other CA",
                     "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA",
                     "ok",
                     "text/html",
                     "text/html"))

    rows.sort()

    return {
        "event_time": [row[0] for row in rows],
        "src_ip": [row[1] for row in rows],
        "dst_ip": [row[2] for row in rows],
        "ja4_client": [row[3] for row in rows],
        "certificate_issuer": [row[4] for row in rows],
        "cipher_suite": [row[5] for row in rows],
        "validation_result": [row[6] for row in rows],
        "content_type_declared": [row[7] for row in rows],
        "content_type_detected": [row[8] for row in rows],
        "certificate_not_before": [row[0] - 10 * 86400 * SECOND for row in rows],
        "certificate_not_after": [row[0] + 80 * 86400 * SECOND for row in rows],
    }


def transfers() -> dict:
    """Sixteen transfers on one triple with three more beside it, then one far outside what either has done.

    The sixteen are spread rather than uniform so a quantile has somewhere to move between the middle and the
    top, and they descend rather than climb so that the most recent are not also the largest -- against a rising
    series a retention cap is invisible, because the last few samples and the whole history have the same top
    and therefore the same 99th percentile. The three beside them share a source address and differ in
    destination port, which is the only way a key made of three columns can be told from a key made of one.

    One transfer partway down sits at twice the envelope rather than three times it, so a multiplier has a case
    that falls on one side of it at 1.2 and the other at 3.0. Without it every transfer is either under the
    envelope or far over, and the multiplier could be any number at all.
    """
    sizes = [900, 880, 860, 840, 820, 800, 460, 440, 420, 400, 200, 1800, 180, 160, 140, 120, 100]
    rows = [(10**18 + index * 60 * SECOND, "10.0.0.5", "10.0.0.9", 445, size) for (index, size) in enumerate(sizes)]
    rows += [(10**18 + offset * SECOND, "10.0.0.5", "10.0.0.9", 443, 3000) for offset in (30, 90, 150)]
    rows.append((10**18 + 20 * 60 * SECOND, "10.0.0.5", "10.0.0.9", 445, 9000))
    rows.sort()

    return {
        "src_ip": [row[1] for row in rows],
        "dst_ip": [row[2] for row in rows],
        "dst_port": [row[3] for row in rows],
        "flow_data_len": [row[4] for row in rows],
        "flow_bpp": [float(row[4]) for row in rows],
        "event_time": [row[0] for row in rows],
    }


def paired_flows() -> dict:
    """Fourteen flows on one pair with three on another interleaved among them.

    Fourteen because twelve intervals are thirteen arrivals, so a frame one flow shorter could not tell a
    maturity floor of twelve from one of thirteen.

    Interleaved because an entity cap is only observable mid-stream. Run the pairs one after the other and a cap
    of one evicts the first pair after its last row has already been emitted, so the parameter reads as dead when
    it is merely too late to matter.
    """
    rows = [(10**18 + index * 60 * SECOND, "10.0.0.5", "93.184.216.34", 512) for index in range(14)]
    rows += [(10**18 + offset * SECOND, "10.0.0.6", "93.184.216.35", 700) for offset in (30, 390, 750)]
    rows.sort()

    return {
        "src_ip": [row[1] for row in rows],
        "dst_ip": [row[2] for row in rows],
        "bytes_out": [row[3] for row in rows],
        "event_time": [row[0] for row in rows],
    }


def resolved_and_unresolved() -> dict:
    """Two ARP observations the ladder placed on a port and one it could not, each with its own address.

    The candidates disagree on every row where both have a value, so preferring one over the other is visible
    in the output -- which is what makes the order a parameter rather than a set.
    """
    return {
        "resolved_port_key": ["hq:sw1:Gi1/0/1", None, "hq:sw1:Gi1/0/2"],
        "arp_sender_ip": ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
    }


def repeated_days() -> dict:
    """One principal, two scored events per day for three days.

    Observed row by row, the second event of each day is a repeat and reads as nulls; reduced to one observation
    per day, every row carries the day's trajectory. The two modes cannot agree on this frame.
    """
    return {
        "user_principal": ["alice@example.com"] * 6,
        "window_id": [100, 100, 101, 101, 102, 102],
        "mean_abs_z": [1.0, 1.2, 1.4, 1.6, 1.9, 2.1],
    }


def keyed_ports() -> dict:
    """Ports that already carry an `entity_key`, and one that carries only whitespace.

    A stage that reads `overwrite` has to have something to overwrite. The held keys deliberately disagree with
    what the entity columns would compose, so keeping them and replacing them are visibly different outcomes --
    if they agreed, the parameter would look inert whether the stage consulted it or not.
    """
    return {
        "site_id": ["hq", "hq", "hq"],
        "device_id": ["sw1", "sw1", "sw1"],
        "port_id": ["Gi1/0/1", "Gi1/0/2", "Gi1/0/3"],
        "entity_key": ["held:one", "held:two", "   "],
    }


def macs() -> dict:
    addresses = ["aa:00:00:00:00:01", "aa:00:00:00:00:02", "bb:00:00:00:00:03", "aa:00:00:00:00:01"]
    ports = ["Gi1/0/1", "Gi1/0/1", "Gi1/0/1", "Gi1/0/2"]

    return {
        "mac_address": addresses,
        "event_time": [index * MINUTE for index in range(len(addresses))],
        "site_id": ["hq"] * len(addresses),
        "switch_id": ["sw1"] * len(addresses),
        "port_id": ports,
        "vlan_id": [10] * len(addresses),
        "oui": ["one-vendor"] * 4,
    }


def arp() -> dict:
    senders = ["10.0.0.1"] * 6 + ["10.0.0.254"]
    macs_ = ["de:ad:00:00:00:01"] * 3 + ["de:ad:00:00:00:02"] * 3 + ["00:00:5e:00:01:fe"]

    return {
        "arp_sender_ip": senders,
        "arp_sender_mac": macs_,
        "arp_target_ip": ["10.0.0.1"] * 6 + ["10.0.0.254"],
        "arp_operation": ["reply"] * 3 + ["request"] * 4,
        "event_time": [index * SECOND for index in range(len(senders))],
    }


def auth() -> dict:
    return {
        "site_id": ["hq"] * 5,
        "switch_id": ["sw1"] * 5,
        "port_id": ["Gi1/0/1"] * 5,
        "mac_address": [
            "aa:00:00:00:00:01", "aa:00:00:00:00:02", "aa:00:00:00:00:02", "aa:00:00:00:00:01", "de:ad:00:00:00:09"
        ],
        "dot1x_identity": ["phone", "desk", "desk", "phone", "rogue"],
        "dot1x_result": ["started", "started", "success", "success", "success"],
        "event_time": [0, SECOND, 11 * SECOND, 12 * SECOND, 20 * MINUTE],
    }


def flows() -> dict:
    return {
        "src_ip": ["10.0.0.1", "10.0.0.1", "not-an-address"],
        "dest_ip": ["10.0.0.9"] * 3,
        "src_port": [1024, 1025, 1026],
        "dest_port": [443, 443, 443],
        "protocol": [6, 6, 6],
        "event_time": [0, MINUTE, 2 * MINUTE],
        "collector_id": ["poller-1"] * 3,
        "collector_seq": [0, 1, 2],
        "schema_version": ["1.0.0"] * 3,
        "origin_hash": ["abc"] * 3,
    }


def windows() -> dict:
    times = [10, 50, 120, 400, 20]

    return {
        "event_time": [t * SECOND for t in times],
        "entity": ["z", "y", "x", "w", "v"],
        "event_uid": [f"uid-{index}" for index in range(len(times))],
        "collector_id": ["poller-1"] * len(times),
        "collector_seq": list(range(len(times))),
    }


def wire() -> dict:
    return {
        "event_time": [1788114300123456789, 1788114360123456789],
        "event_uid": ["a", "b"],
        "port_key": ["hq:sw1:Gi1/0/1"] * 2,
        "macs_per_port_first_in_window": [True, False],
        "macs_claiming_sender_ip": [1, 2],
        "arp_sender_ip_excluded": [False, False],
        "auth_unpaired": [False, True],
        "auth_port_key": ["hq:sw1:Gi1/0/1"] * 2,
        "bind_start": [1788114000123456789, 1788114060123456789],
        "mac_address": ["aa:00:00:00:00:01", "aa:00:00:00:00:02"],
        "bind_provisional": [True, True],
    }


def wire_without_the_required_columns() -> dict:
    return {"event_time": [1788114300123456789], "event_uid": ["a"]}


def sessions() -> dict:
    # Two sessions open at once, each closing later. Overlapping is what makes max_open_sessions bite, and the gap
    # between a start and its end is what makes the timeout bite.
    return {
        "session_id": ["s-a", "s-b", "s-a", "s-b"],
        "user_principal": ["alice@example.com", "bob@example.com", "alice@example.com", "bob@example.com"],
        "session_action": ["start", "start", "end", "end"],
        "event_time": [0, MINUTE, 2 * MINUTE, 3 * MINUTE],
    }


def logins() -> dict:
    # One principal returning to a location it has used before, with a second principal interleaved. The return is
    # what makes max_values bite; the interleaving is what makes max_entities bite.
    return {
        "user_principal": [
            "alice@example.com", "bob@example.com", "alice@example.com", "bob@example.com", "alice@example.com"
        ],
        "source_country": ["gb", "us", "fr", "us", "gb"],
        "source_region": ["england", "ca", "idf", "ca", "england"],
        "source_city": ["london", "san-jose", "paris", "san-jose", "london"],
        "app": ["vpn", "wiki", "wiki", "wiki", "vpn"],
        "device_id": ["laptop-1", "laptop-2", "laptop-1", "laptop-2", "phone-1"],
        "source_asn": ["as5089", "as7018", "as3215", "as7018", "as5089"],
        "target_host": ["ws-01", "srv-01", "ws-01", "srv-02", "ws-02"],
        "event_time": [0, HOUR, 5 * HOUR, 30 * HOUR, 50 * HOUR],
    }


def journeys() -> dict:
    # Alice crosses the Atlantic in an hour, with a second principal measured between her two records so that
    # evicting her is observable, then a refresh and a failure so each exclusion has something to exclude.
    return {
        "user_principal": [
            "alice@example.com",
            "bob@example.com",
            "alice@example.com",
            "alice@example.com",
            "alice@example.com",
        ],
        "source_latitude": [51.5074, 51.5074, 40.7128, 40.7128, 51.5074],
        "source_longitude": [-0.1278, -0.1278, -74.0060, -74.0060, -0.1278],
        "auth_result": ["success", "success", "success", "success", "failure"],
        "token_type": ["bearer", "bearer", "bearer", "refresh", "bearer"],
        "source_ip": ["203.0.113.10", "203.0.113.11", "203.0.113.12", "198.51.100.7", "203.0.113.13"],
        "event_time": [0, 30 * MINUTE, HOUR, 2 * HOUR, 3 * HOUR],
    }


def denials() -> dict:
    # R-D-L5-004's shape: denials inside ten minutes and an approval at the end, with a record carrying no factor
    # so the proportion is not one for every row, and a second principal so evicting the first is observable.
    return {
        "user_principal": ["alice@example.com"] * 5 + ["bob@example.com", "alice@example.com"],
        "auth_result": ["failure", "failure", "success", "failure", "failure", "success", "success"],
        "mfa_used": [True, True, False, True, True, True, True],
        "mfa_result": ["denied", "denied", None, "denied", "denied", "approved", "approved"],
        "event_time": [0, MINUTE, 2 * MINUTE, 3 * MINUTE, 4 * MINUTE, 5 * MINUTE, 6 * MINUTE],
    }


def scores() -> dict:
    # One principal's score across consecutive windows: a flat stretch and then a climb, which is R-P-L5-006's
    # shape. A second principal is interleaved rather than appended, because evicting the first is only
    # observable if the first is seen again afterwards.
    flat = [1.0, 1.02, 0.98, 1.0, 1.01, 1.0]
    climb = [1.2, 1.6, 2.1, 2.7]

    rows = [("alice@example.com", 100 + index, value) for (index, value) in enumerate(flat)]
    rows.append(("bob@example.com", 100, 1.0))
    rows.extend(("alice@example.com", 106 + index, value) for (index, value) in enumerate(climb))

    return {
        "user_principal": [row[0] for row in rows],
        "window_id": [row[1] for row in rows],
        "mean_abs_z": [row[2] for row in rows],
        "event_time": [index * MINUTE for index in range(len(rows))],
    }


SCORED_WINDOW = 484512


def scored() -> dict:
    # Two principals in one window, one of which the manifest below has no model for.
    return {
        "user_principal": ["alice@example.com", "bob@example.com", "carol@example.com"],
        "window_id": [SCORED_WINDOW] * 3,
        "event_time": [0, MINUTE, 2 * MINUTE],
    }


def scorable() -> dict:
    """The scored frame with features on it. Carol has no model of her own, so a fallback change is observable."""
    return {
        **scored(),
        "logcount": [3.0, 5.0, 7.0],
        "mfa_ratio": [0.5, 0.25, 0.75],
    }


class _LengthScorer:
    """Scores a feature as itself times the length of the pinned model identifier.

    Trivial, and deliberately dependent on `model_version`: a stub ignoring which model it was handed would make
    the manifest knob look inert when it is not, and an inert-looking knob is exactly what this file exists to
    tell apart from a knob nothing reads.
    """

    def __init__(self, scale: float = 1.0):
        self._scale = scale

    def score(self, model_version: str, features: list) -> list:
        weight = float(len(model_version)) * self._scale

        return [{name: value * weight for (name, value) in row.items()} for row in features]


def _envelope(tier: str = "D1", seed: int = 42) -> DeterminismEnvelope:
    return DeterminismEnvelope(tier=tier,
                               fingerprint="a3f9c2e1b8d47506",
                               configuration="7d2e4a1f9c3b5e80",
                               code_commit="c6a3b56",
                               image_digest="sha256:1f0c",
                               feature_schema_version="TC-5/2.1.0",
                               rng_seed=seed)


def _manifest(fallback=None) -> ModelManifest:
    return ModelManifest(window_id=SCORED_WINDOW,
                         models={
                             "alice@example.com": "dfp-alice:14", "bob@example.com": "dfp-bob:3"
                         },
                         fallback=fallback)


def _binding_table(values=("hq:sw1:Gi1/0/1", )) -> BindingTable:
    return BindingTable(
        name="dhcp_lease",
        value_columns=["port_key"],
        bindings=[Binding(key="10.0.0.1", start_ns=0, end_ns=10 * MINUTE, values=tuple(values), uid="u1")],
    )


# --- The registry ---------------------------------------------------------------------------------------------

REGISTRY: dict = {
    "TC1NormalizeStage":
        Scenario(
            stage=TC1NormalizeStage,
            frame=counters,
            base={"uptime_column": "uptime"},
            knobs=(
                Knob("site_column", INPUT_COLUMN, benign="site_id"),
                Knob("device_column", INPUT_COLUMN, benign="device_id"),
                Knob("port_column", INPUT_COLUMN, benign="port_id"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("counter_columns", DIFFERS, benign=["crc_errors"], extreme=["crc_errors", "symbol_errors"]),
                Knob("counter32_columns", DIFFERS, benign=[], extreme=["crc_errors"]),
                Knob("uptime_column", DIFFERS, benign="uptime", extreme=None),
                Knob("uptime_unit", DIFFERS, benign="cs", extreme="s"),
                Knob("delta_suffix", DIFFERS, benign="_delta", extreme="_change"),
                Knob("entity_key_column", DIFFERS, benign="entity_key", extreme="subject_key"),
            ),
        ),
    "TC1OpticalStage":
        Scenario(
            stage=TC1OpticalStage,
            frame=optics,
            base={},
            knobs=(
                Knob("entity_key_column", INPUT_COLUMN, benign="entity_key"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("channel_columns",
                     DIFFERS,
                     benign=["optical_rx_dbm"],
                     extreme=["optical_rx_dbm", "optical_tx_dbm"]),
                Knob("window_seconds", DIFFERS, benign=3600, extreme=120),
                Knob("min_samples", DIFFERS, benign=2, extreme=11),
            ),
        ),
    "TC1FlapStage":
        Scenario(
            stage=TC1FlapStage,
            frame=flaps,
            base={"last_change_column": "if_last_change"},
            knobs=(
                Knob("entity_key_column", INPUT_COLUMN, benign="entity_key"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("status_column", INPUT_COLUMN, benign="oper_status"),
                Knob("last_change_column", DIFFERS, benign=None, extreme="if_last_change"),  # noqa: E501
                Knob("last_change_unit",
                     INERT,
                     reason="Read, and provably unobservable. Every branch of "
                     "LinkFlapCounter._count compares this sample's last-change against the previous sample's -- "
                     "less than, equal, greater -- and never against the event time. A positive rescale preserves "
                     "all three comparisons, so no unit can change any emitted value. It would begin to matter the "
                     "moment anything compares the converted value against an absolute time; nothing does yet."),
                Knob("window_seconds", DIFFERS, benign=3600, extreme=60),
            ),
        ),
    "TC1ChangeStage":
        Scenario(
            stage=TC1ChangeStage,
            frame=identifiers,
            base={},
            knobs=(
                Knob("entity_key_column", INPUT_COLUMN, benign="entity_key"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="s"),
                # Seconds rather than microseconds: this stage has no time window, so a uniform rescale is
                # invisible to it by construction and the observable is that a nanosecond value read as seconds is
                # past what a timestamp can hold. That still only happens if the parameter reaches the conversion.
                Knob("novelty_columns",
                     DIFFERS,
                     benign=["transceiver_serial"],
                     extreme=["transceiver_serial", "lldp_neighbor_chassis_id"]),
                Knob("max_values", DIFFERS, benign=64, extreme=1),
            ),
        ),
    "TC1FeatureStage":
        Scenario(
            stage=TC1FeatureStage,
            frame=identifiers,
            base={},
            knobs=(
                Knob("entity_key_column", INPUT_COLUMN, benign="entity_key"),
                Knob("timestamp_column",
                     INPUT_COLUMN,
                     benign="event_time",
                     also={"order_columns": ["renamed_event_time", "collector_id", "collector_seq"]}),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("transceiver_column", INPUT_COLUMN, benign="transceiver_serial"),
                Knob("neighbor_column", INPUT_COLUMN, benign="lldp_neighbor_chassis_id"),
                Knob("period", DIFFERS, benign="D", extreme="h"),
                Knob("order_columns",
                     DIFFERS,
                     benign=["event_time", "collector_id", "collector_seq"],
                     extreme=["transceiver_serial", "collector_id", "collector_seq"]),
                Knob("require_total_order", RAISES, benign=False, extreme=True, frame=tied_identifiers),
                Knob("preserve_columns",
                     INERT,
                     reason="Carries extra columns into the intermediate feature frame "
                     "only. The stage merges its features back onto the full input frame, so every input column "
                     "survives regardless and the parameter cannot change what this stage emits."),
            ),
        ),
    "TC2CardinalityStage":
        Scenario(
            stage=TC2CardinalityStage,
            frame=macs,
            base={},
            knobs=(
                Knob("mac_column", INPUT_COLUMN, benign="mac_address"),
                Knob("site_column", INPUT_COLUMN, benign="site_id"),
                Knob("switch_column", INPUT_COLUMN, benign="switch_id"),
                Knob("port_column", INPUT_COLUMN, benign="port_id"),
                Knob("vlan_column", INPUT_COLUMN, benign="vlan_id"),
                Knob("oui_column", DIFFERS, benign="oui", extreme="absent_oui"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("window_seconds", DIFFERS, benign=3600, extreme=60),
                Knob("max_samples", DIFFERS, benign=1024, extreme=1),
            ),
        ),
    "TC2ArpStage":
        Scenario(
            stage=TC2ArpStage,
            frame=arp,
            base={},
            knobs=(
                Knob("sender_ip_column", INPUT_COLUMN, benign="arp_sender_ip"),
                Knob("sender_mac_column", INPUT_COLUMN, benign="arp_sender_mac"),
                Knob("target_ip_column", INPUT_COLUMN, benign="arp_target_ip"),
                Knob("operation_column", INPUT_COLUMN, benign="arp_operation"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("window_seconds", DIFFERS, benign=3600, extreme=1),
                Knob("min_denominator", DIFFERS, benign=1, extreme=100),
                Knob("include_gratuitous_requests", DIFFERS, benign=True, extreme=False),
                Knob("excluded_sender_ips", DIFFERS, benign=[], extreme=["10.0.0.254"]),
            ),
        ),
    "TC2AuthStage":
        Scenario(
            stage=TC2AuthStage,
            frame=auth,
            base={},
            knobs=(
                Knob("site_column", INPUT_COLUMN, benign="site_id"),
                Knob("switch_column", INPUT_COLUMN, benign="switch_id"),
                Knob("port_column", INPUT_COLUMN, benign="port_id"),
                Knob("result_column", INPUT_COLUMN, benign="dot1x_result"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("pending_values", DIFFERS, benign=["started"], extreme=["started", "success"]),
                Knob("supplicant_columns", DIFFERS, benign=["mac_address"], extreme=[]),
                Knob("max_clock_skew_seconds", DIFFERS, benign=7 * 24 * 3600, extreme=1),
                Knob("timeout_seconds", DIFFERS, benign=3600, extreme=1),
            ),
        ),
    "TC1BindingStage":
        Scenario(
            stage=TC1BindingStage,
            frame=port_inventory,
            base={},
            knobs=(
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("key_columns", DIFFERS, benign=["site_id", "device_id", "port_id"], extreme=["device_id"]),
                Knob("attribute_columns",
                     DIFFERS,
                     benign=["transceiver_serial", "lldp_neighbor_chassis_id"],
                     extreme=["transceiver_serial"]),
                Knob("max_clock_skew_seconds", DIFFERS, benign=7 * 24 * 3600, extreme=1),
                Knob("idle_timeout_seconds", DIFFERS, benign=7 * 24 * 3600, extreme=1),
                Knob("emit_open_on_complete", DIFFERS, benign=False, extreme=True),
                Knob("emit_open_bindings", DIFFERS, benign=False, extreme=True),
            ),
        ),
    "TC2BindingStage":
        Scenario(
            stage=TC2BindingStage,
            frame=macs,
            base={},
            knobs=(
                Knob("key_column", INPUT_COLUMN, benign="mac_address"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("attribute_columns", DIFFERS, benign=["vlan_id"], extreme=["vlan_id", "oui"]),
                Knob("max_clock_skew_seconds", DIFFERS, benign=7 * 24 * 3600, extreme=1),
                Knob("idle_timeout_seconds", DIFFERS, benign=86400, extreme=60),
                Knob("emit_open_on_complete", DIFFERS, benign=False, extreme=True),
                Knob("emit_open_bindings", DIFFERS, benign=False, extreme=True),
            ),
        ),
    "TC5ScoreStage":
        Scenario(
            stage=TC5ScoreStage,
            frame=scorable,
            base={
                "scorer": _LengthScorer(),
                "manifest": _manifest(fallback="dfp-generic:1"),
                "feature_columns": ["logcount", "mfa_ratio"]
            },
            knobs=(
                Knob("scorer", DIFFERS, benign=_LengthScorer(), extreme=_LengthScorer(scale=2.0)),
                Knob("entity_column", INPUT_COLUMN, benign="user_principal"),
                Knob("window_column", INPUT_COLUMN, benign="window_id"),
                Knob("feature_columns", DIFFERS, benign=["logcount", "mfa_ratio"], extreme=["logcount"]),
                Knob("manifest",
                     DIFFERS,
                     benign=_manifest(fallback="dfp-generic:1"),
                     extreme=_manifest(fallback="dfp-generic-population:1")),
                Knob("decimals", DIFFERS, benign=4, extreme=1),
            ),
        ),
    "TC5SessionStage":
        Scenario(
            stage=TC5SessionStage,
            frame=sessions,
            base={},
            knobs=(
                Knob("session_column", INPUT_COLUMN, benign="session_id"),
                Knob("principal_column", INPUT_COLUMN, benign="user_principal"),
                Knob("action_column", INPUT_COLUMN, benign="session_action"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("start_actions", DIFFERS, benign=("start", ), extreme=("logon", )),
                Knob("end_actions", DIFFERS, benign=("end", ), extreme=("logoff", )),
                Knob("timeout_seconds", DIFFERS, benign=3600, extreme=1),
                Knob("max_clock_skew_seconds", DIFFERS, benign=7 * 24 * 3600, extreme=1),
                Knob("max_open_sessions", DIFFERS, benign=500_000, extreme=1),
            ),
        ),
    "TC5NoveltyStage":
        Scenario(
            stage=TC5NoveltyStage,
            frame=logins,
            base={},
            knobs=(
                Knob("principal_column", INPUT_COLUMN, benign="user_principal"),
                Knob("location_columns",
                     DIFFERS,
                     benign=("source_country", "source_region", "source_city"),
                     extreme=("source_country", )),
                Knob("app_column", INPUT_COLUMN, benign="app"),
                Knob("device_column", INPUT_COLUMN, benign="device_id"),
                Knob("asn_column", INPUT_COLUMN, benign="source_asn"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("window_seconds", DIFFERS, benign=86400, extreme=1),
                Knob("max_samples", DIFFERS, benign=4096, extreme=1),
                Knob("max_values", DIFFERS, benign=256, extreme=1),
                Knob("max_entities", DIFFERS, benign=100_000, extreme=1),
                Knob("target_host_column", DIFFERS, benign=None, extreme="target_host"),
            ),
        ),
    "TC4FlowStage":
        Scenario(stage=TC4FlowStage,
                 frame=packets,
                 base={"bin_seconds": 60},
                 knobs=(
                     Knob("src_ip_column", INPUT_COLUMN, benign="src_ip"),
                     Knob("src_port_column", INPUT_COLUMN, benign="src_port"),
                     Knob("dst_ip_column", INPUT_COLUMN, benign="dst_ip"),
                     Knob("dst_port_column", INPUT_COLUMN, benign="dst_port"),
                     Knob("flags_column", INPUT_COLUMN, benign="tcp_flags"),
                     Knob("data_len_column", INPUT_COLUMN, benign="data_len"),
                     Knob("time_column", INPUT_COLUMN, benign="event_time"),
                     Knob("time_unit", DIFFERS, benign="ns", extreme="s"),
                     Knob("bin_seconds", DIFFERS, benign=60, extreme=600),
                     Knob("max_flows", DIFFERS, benign=500000, extreme=1),
                 )),
    "TC4EnvelopeStage":
        Scenario(stage=TC4EnvelopeStage,
                 frame=transfers,
                 base={
                     "min_samples": 4, "magnitude_columns": ["flow_data_len"]
                 },
                 knobs=(
                     Knob("key_columns", DIFFERS, benign=["src_ip", "dst_ip", "dst_port"], extreme=["src_ip"]),
                     Knob("magnitude_columns", DIFFERS, benign=["flow_data_len"], extreme=["flow_bpp"]),
                     Knob("time_column", INPUT_COLUMN, benign="event_time"),
                     Knob("time_unit", DIFFERS, benign="ns", extreme="s"),
                     Knob("window_seconds", DIFFERS, benign=86400, extreme=120),
                     Knob("quantile", DIFFERS, benign=0.99, extreme=0.5),
                     Knob("multiplier", DIFFERS, benign=3.0, extreme=1.2),
                     Knob("min_samples", DIFFERS, benign=4, extreme=100),
                     Knob("max_samples", DIFFERS, benign=4096, extreme=5),
                 )),
    "TC6FingerprintStage":
        Scenario(stage=TC6FingerprintStage,
                 frame=handshakes,
                 base={},
                 knobs=(
                     Knob("key_columns", DIFFERS, benign=["src_ip"], extreme=["src_ip", "dst_ip"]),
                     Knob("fingerprint_column", INPUT_COLUMN, benign="ja4_client"),
                     Knob("time_column", INPUT_COLUMN, benign="event_time"),
                     Knob("time_unit", DIFFERS, benign="ns", extreme="s"),
                     Knob("max_values", DIFFERS, benign=64, extreme=1),
                     Knob("max_entities", DIFFERS, benign=100000, extreme=1),
                 )),
    "TC6CertificateStage":
        Scenario(
            stage=TC6CertificateStage,
            frame=handshakes,
            base={"min_samples": 3},
            knobs=(
                Knob("key_columns", DIFFERS, benign=["dst_ip"], extreme=["dst_ip", "src_ip"]),
                Knob("issuer_column", INPUT_COLUMN, benign="certificate_issuer"),
                Knob("validation_column", INPUT_COLUMN, benign="validation_result"),
                # The reference is keyed on the destination by default, so renaming the destination
                # column has to move the key with it or the two runs are keyed on different things.
                Knob("destination_column", INPUT_COLUMN, benign="dst_ip", also={"key_columns": ["renamed_dst_ip"]}),
                Knob("not_before_column", INPUT_COLUMN, benign="certificate_not_before"),
                Knob("not_after_column", INPUT_COLUMN, benign="certificate_not_after"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="s"),
                Knob("window_seconds", DIFFERS, benign=2592000, extreme=120),
                Knob("min_samples", DIFFERS, benign=3, extreme=100),
                Knob("max_samples", DIFFERS, benign=512, extreme=3),
                Knob("estate_warmup_seconds", DIFFERS, benign=604800, extreme=0),
            )),
    "TC6CipherStage":
        Scenario(stage=TC6CipherStage,
                 frame=handshakes,
                 base={"min_samples": 3},
                 knobs=(
                     Knob("key_columns", DIFFERS, benign=["src_ip", "dst_ip"], extreme=["src_ip"]),
                     Knob("cipher_column", INPUT_COLUMN, benign="cipher_suite"),
                     Knob("time_column", INPUT_COLUMN, benign="event_time"),
                     Knob("time_unit", DIFFERS, benign="ns", extreme="s"),
                     Knob("window_seconds", DIFFERS, benign=2592000, extreme=120),
                     Knob("min_samples", DIFFERS, benign=3, extreme=100),
                     Knob("max_samples", DIFFERS, benign=512, extreme=2),
                 )),
    "TC6ContentStage":
        Scenario(stage=TC6ContentStage,
                 frame=handshakes,
                 base={},
                 knobs=(
                     Knob("declared_column", INPUT_COLUMN, benign="content_type_declared"),
                     Knob("detected_column", INPUT_COLUMN, benign="content_type_detected"),
                 )),
    "TC0IdentityStage":
        Scenario(
            stage=TC0IdentityStage,
            frame=identity_records,
            base={},
            knobs=(
                Knob("principal_column", INPUT_COLUMN, benign="user_principal"),
                Knob("group_column", INPUT_COLUMN, benign="group_name"),
                Knob("profile_columns", DIFFERS, benign=None, extreme=["department"]),
                # The instants are written back under their canonical names, so renaming an input column would
                # leave both names in the output; swapping which column is read is the evidence instead.
                Knob("valid_from_column", DIFFERS, benign="valid_from", extreme="recorded_at"),
                Knob("valid_to_column", DIFFERS, benign="valid_to", extreme="valid_from"),
                Knob("recorded_column", DIFFERS, benign="recorded_at", extreme="valid_from"),
                Knob("change_column", DIFFERS, benign="change", extreme="no_such_column"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="s"),
            ),
        ),
    "TC0AssetStage":
        Scenario(
            stage=TC0AssetStage,
            frame=asset_records,
            base={},
            knobs=(
                Knob("asset_column", INPUT_COLUMN, benign="hostname"),
                Knob("asset_columns", DIFFERS, benign=None, extreme=["data_classification"]),
                Knob("valid_from_column", DIFFERS, benign="valid_from", extreme="recorded_at"),
                Knob("valid_to_column", DIFFERS, benign="valid_to", extreme="valid_from"),
                Knob("recorded_column", DIFFERS, benign="recorded_at", extreme="valid_from"),
                Knob("change_column", DIFFERS, benign="change", extreme="no_such_column"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="s"),
            ),
        ),
    "TC0EnrichStage":
        Scenario(
            stage=TC0EnrichStage,
            frame=context_probes,
            base={"store": _context_store()},
            knobs=(
                Knob("store", DIFFERS, benign=_context_store(), extreme=_context_store("Support")),
                Knob("entity_column", INPUT_COLUMN, benign="user_principal"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("knowledge", DIFFERS, benign="event", extreme="latest"),
                Knob("prefix", DIFFERS, benign="ctx_", extreme="context_at_event_"),
                Knob("set_kinds", DIFFERS, benign=None, extreme={}),
            ),
        ),
    "TC7DnsStage":
        Scenario(stage=TC7DnsStage,
                 frame=dns_queries,
                 base={},
                 knobs=(
                     Knob("query_column", INPUT_COLUMN, benign="query_name"),
                     Knob("time_column", INPUT_COLUMN, benign="event_time"),
                     Knob("time_unit", DIFFERS, benign="ns", extreme="s"),
                     Knob("window_seconds", DIFFERS, benign=3600, extreme=300),
                     Knob("max_samples", DIFFERS, benign=4096, extreme=3),
                     Knob("max_entities", DIFFERS, benign=500000, extreme=1, frame=_two_domains),
                 )),
    "TC7HttpStage":
        Scenario(stage=TC7HttpStage,
                 frame=http_requests,
                 base={},
                 knobs=(
                     Knob("key_columns", DIFFERS, benign=["src_ip"], extreme=["url_path"]),
                     Knob("status_column", INPUT_COLUMN, benign="status_code"),
                     Knob("path_column", INPUT_COLUMN, benign="url_path"),
                     Knob("time_column", INPUT_COLUMN, benign="event_time"),
                     Knob("time_unit", DIFFERS, benign="ns", extreme="s"),
                     Knob("window_seconds", DIFFERS, benign=600, extreme=60),
                     Knob("max_samples", DIFFERS, benign=4096, extreme=2),
                     Knob("max_entities", DIFFERS, benign=500000, extreme=1),
                 )),
    "TC7SaasStage":
        Scenario(
            stage=TC7SaasStage,
            frame=saas_operations,
            base={"min_samples": 2},
            knobs=(
                Knob("principal_column", INPUT_COLUMN, benign="user_principal"),
                Knob("operation_column", INPUT_COLUMN, benign="operation"),
                Knob("object_type_column", INPUT_COLUMN, benign="target_object_type"),
                Knob("record_count_column", INPUT_COLUMN, benign="record_count"),
                Knob("result_column", INPUT_COLUMN, benign="result"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="s"),
                Knob("baseline_days", DIFFERS, benign=30, extreme=1),
                Knob("quantile", DIFFERS, benign=0.99, extreme=0.0),
                Knob("min_samples", DIFFERS, benign=2, extreme=3),
                Knob("week_epoch", DIFFERS, benign="1970-01-05", extreme="1970-01-04"),
                Knob("max_samples", DIFFERS, benign=4096, extreme=1),
                Knob("max_entities", DIFFERS, benign=500000, extreme=1),
                Knob("decimals", DIFFERS, benign=4, extreme=1),
            ),
        ),
    "TC7EndpointStage":
        Scenario(
            stage=TC7EndpointStage,
            frame=process_starts,
            base={"warmup_days": 1},
            knobs=(
                Knob("host_column", INPUT_COLUMN, benign="hostname"),
                Knob("parent_image_column", INPUT_COLUMN, benign="parent_image_path"),
                Knob("image_column", INPUT_COLUMN, benign="image_path"),
                Knob("integrity_column", INPUT_COLUMN, benign="integrity_level"),
                Knob("peer_group_column", INPUT_COLUMN, benign="ctx_peer_group"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="s"),
                Knob("window_days", DIFFERS, benign=30, extreme=1),
                Knob("warmup_days", DIFFERS, benign=1, extreme=0),
                Knob("max_pairs", DIFFERS, benign=16384, extreme=1),
                Knob("max_entities", DIFFERS, benign=200000, extreme=1),
            ),
        ),
    "TC5CadenceStage":
        Scenario(
            stage=TC5CadenceStage,
            frame=logins,
            base={},
            knobs=(
                Knob("principal_column", INPUT_COLUMN, benign="user_principal"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("utc_offset_minutes", DIFFERS, benign=0, extreme=345),
                Knob("min_samples", DIFFERS, benign=2, extreme=1000),
                Knob("max_entities", DIFFERS, benign=100_000, extreme=1),
                Knob("decimals", DIFFERS, benign=4, extreme=1),
            ),
        ),
    "TC5DriftStage":
        Scenario(
            stage=TC5DriftStage,
            frame=scores,
            base={},
            knobs=(
                Knob("entity_column", INPUT_COLUMN, benign="user_principal"),
                Knob("score_column", INPUT_COLUMN, benign="mean_abs_z"),
                Knob("window_column", INPUT_COLUMN, benign="window_id"),
                Knob("max_windows", DIFFERS, benign=64, extreme=2),
                Knob("min_windows", DIFFERS, benign=4, extreme=100),
                Knob("max_entities", DIFFERS, benign=100_000, extreme=1),
                Knob("decimals", DIFFERS, benign=4, extreme=1),
                Knob("aggregate", DIFFERS, benign="none", extreme="mean", frame=repeated_days),
            ),
        ),
    "TC5TravelStage":
        Scenario(
            stage=TC5TravelStage,
            frame=journeys,
            base={},
            knobs=(
                Knob("principal_column", INPUT_COLUMN, benign="user_principal"),
                Knob("latitude_column", INPUT_COLUMN, benign="source_latitude"),
                Knob("longitude_column", INPUT_COLUMN, benign="source_longitude"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("result_column", INPUT_COLUMN, benign="auth_result"),
                Knob("success_values", DIFFERS, benign=("success", ), extreme=("failure", )),
                Knob("refresh_column", INPUT_COLUMN, benign="token_type"),
                Knob("refresh_values", DIFFERS, benign=("refresh", ), extreme=("bearer", )),
                Knob("source_ip_column", INPUT_COLUMN, benign="source_ip"),
                Knob("excluded_source_networks", DIFFERS, benign=(), extreme=("203.0.113.0/24", )),
                Knob("min_elapsed_seconds", DIFFERS, benign=1, extreme=7200),
                Knob("max_entities", DIFFERS, benign=100_000, extreme=1),
                Knob("decimals", DIFFERS, benign=4, extreme=1),
            ),
        ),
    "TC5RiskStage":
        Scenario(
            stage=TC5RiskStage,
            frame=denials,
            base={"min_denominator": 1},
            knobs=(
                Knob("principal_column", INPUT_COLUMN, benign="user_principal"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("result_column", INPUT_COLUMN, benign="auth_result"),
                Knob("success_values", DIFFERS, benign=("success", ), extreme=("failure", )),
                Knob("mfa_column", INPUT_COLUMN, benign="mfa_used"),
                Knob("mfa_result_column", INPUT_COLUMN, benign="mfa_result"),
                Knob("mfa_success_values", DIFFERS, benign=("approved", ), extreme=("denied", )),
                Knob("run_window_seconds", DIFFERS, benign=600, extreme=1),
                Knob("ratio_window_seconds", DIFFERS, benign=86400, extreme=1),
                Knob("min_denominator", DIFFERS, benign=1, extreme=100),
                Knob("max_samples", DIFFERS, benign=4096, extreme=1),
                Knob("max_entities", DIFFERS, benign=100_000, extreme=1),
            ),
        ),
    "DeterminismStampStage":
        Scenario(
            stage=DeterminismStampStage,
            frame=scored,
            base={
                "envelope": _envelope(), "manifest": _manifest()
            },
            knobs=(
                Knob("envelope", DIFFERS, benign=_envelope(), extreme=_envelope(tier="D0", seed=7)),
                Knob("manifest", DIFFERS, benign=_manifest(), extreme=_manifest(fallback="dfp-generic:2")),
                Knob("entity_column", INPUT_COLUMN, benign="user_principal"),
                Knob("window_column", INPUT_COLUMN, benign="window_id"),
            ),
        ),
    "WindowSealStage":
        Scenario(
            stage=WindowSealStage,
            frame=windows,
            base={
                "period_seconds": 100, "lateness_seconds": 30, "entity_key_column": "entity"
            },
            knobs=(
                Knob("period_seconds", DIFFERS, benign=100, extreme=25),
                Knob("lateness_seconds", DIFFERS, benign=30, extreme=1000),
                Knob("epoch", DIFFERS, benign=0, extreme=37 * SECOND),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("order_columns", DIFFERS, benign=None, extreme=["entity"]),
                Knob("seal_on_complete", DIFFERS, benign=True, extreme=False),
                Knob("entity_key_column", DIFFERS, benign=None, extreme="entity"),
                Knob("uid_column", INPUT_COLUMN, benign="event_uid"),
                Knob("lineage_id_column", DIFFERS, benign="lineage_id", extreme="chain_id"),
                Knob("raise_on_invalid",
                     INERT,
                     reason="Every row in this frame carries a valid event time, and a "
                     "frame that does not is the subject of the window seal stage's own tests; the flag changes "
                     "nothing about a well-formed batch by design."),
                Knob("column_prefix", DIFFERS, benign="", extreme="day_"),
            ),
        ),
    "TotalOrderStage":
        Scenario(
            stage=TotalOrderStage,
            frame=windows,
            base={},
            knobs=(
                Knob("order_columns",
                     DIFFERS,
                     benign=["event_time", "collector_id", "collector_seq"],
                     extreme=["entity", "collector_id", "collector_seq"]),
                Knob("require_total_order",
                     INERT,
                     reason="This frame has a total order under both key sets, and a "
                     "frame with ties is what the stage's own tests use; the flag cannot change a well-ordered batch."),
            ),
        ),
    "CommunityIdStage":
        Scenario(
            stage=CommunityIdStage,
            frame=flows,
            base={
                "dst_ip_column": "dest_ip", "dst_port_column": "dest_port", "raise_on_failure": False
            },
            knobs=(
                Knob("src_ip_column", INPUT_COLUMN, benign="src_ip"),
                Knob("dst_ip_column", INPUT_COLUMN, benign="dest_ip"),
                Knob("protocol_column", INPUT_COLUMN, benign="protocol"),
                Knob("src_port_column", INPUT_COLUMN, benign="src_port"),
                Knob("dst_port_column", INPUT_COLUMN, benign="dest_port"),
                Knob("seed", DIFFERS, benign=0, extreme=1),
                Knob("use_base64", DIFFERS, benign=True, extreme=False),
                Knob("output_column", DIFFERS, benign="community_id", extreme="flow_id"),
                Knob("raise_on_failure", RAISES, benign=False, extreme=True),
            ),
        ),
    "TC3CardinalityStage":
        Scenario(stage=TC3CardinalityStage,
                 frame=flow_records,
                 base={"window_seconds": 3600},
                 knobs=(
                     Knob("src_column", INPUT_COLUMN, benign="src_ip"),
                     Knob("dst_column", INPUT_COLUMN, benign="dst_ip"),
                     Knob("dst_port_column", INPUT_COLUMN, benign="dst_port"),
                     Knob("time_column", INPUT_COLUMN, benign="event_time"),
                     Knob("time_unit", DIFFERS, benign="ns", extreme="s"),
                     Knob("window_seconds", DIFFERS, benign=3600, extreme=60),
                     Knob("max_samples", DIFFERS, benign=4096, extreme=2),
                 )),
    "TC3ReachStage":
        Scenario(stage=TC3ReachStage,
                 frame=flow_records,
                 base={
                     "window_seconds": 3600, "min_denominator": 1
                 },
                 knobs=(
                     Knob("src_column", INPUT_COLUMN, benign="src_ip"),
                     Knob("dst_column", INPUT_COLUMN, benign="dst_ip"),
                     Knob("asn_column", INPUT_COLUMN, benign="bgp_as_dst"),
                     Knob("bytes_out_column", INPUT_COLUMN, benign="bytes_out"),
                     Knob("bytes_in_column", INPUT_COLUMN, benign="bytes_in"),
                     Knob("time_column", INPUT_COLUMN, benign="event_time"),
                     Knob("time_unit", DIFFERS, benign="ns", extreme="s"),
                     Knob("window_seconds", DIFFERS, benign=3600, extreme=60),
                     Knob("min_denominator", DIFFERS, benign=1, extreme=100),
                     Knob("max_samples", DIFFERS, benign=4096, extreme=2),
                 )),
    "TC3BeaconStage":
        Scenario(stage=TC3BeaconStage,
                 frame=paired_flows,
                 base={"window_seconds": 86400},
                 knobs=(
                     Knob("src_column", INPUT_COLUMN, benign="src_ip"),
                     Knob("dst_column", INPUT_COLUMN, benign="dst_ip"),
                     Knob("size_column", INPUT_COLUMN, benign="bytes_out"),
                     Knob("time_column", INPUT_COLUMN, benign="event_time"),
                     Knob("time_unit", DIFFERS, benign="ns", extreme="s"),
                     Knob("window_seconds", DIFFERS, benign=86400, extreme=120),
                     Knob("min_intervals", DIFFERS, benign=12, extreme=3),
                     Knob("max_samples", DIFFERS, benign=4096, extreme=3),
                     Knob("max_entities", DIFFERS, benign=500000, extreme=1),
                 )),
    "TC3TtlStage":
        Scenario(
            stage=TC3TtlStage,
            frame=flow_records,
            base={
                "window_seconds": 86400, "min_samples": 2
            },
            knobs=(
                # A list rather than a name, so the column parameter's rename trick does not apply. Two keys
                # against one is the difference between a reference per source and one per conversation, which
                # is the choice an estate with mixed exporters actually makes.
                Knob("key_columns", DIFFERS, benign=["src_ip"], extreme=["src_ip", "dst_ip"]),
                Knob("ttl_column", INPUT_COLUMN, benign="ip_ttl"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="s"),
                Knob("window_seconds", DIFFERS, benign=86400, extreme=60),
                Knob("min_samples", DIFFERS, benign=2, extreme=100),
                Knob("min_shift", DIFFERS, benign=1, extreme=200),
                Knob("max_samples", DIFFERS, benign=512, extreme=2),
            )),
    "MinimizationStage":
        Scenario(
            stage=MinimizationStage,
            frame=two_principals_at_layer_five,
            base={
                "drop": ["addresses"],
                "pseudonymize": ["user_principal", "entity_key"],
                "key": MINIMIZATION_KEY,
                "telemetry_class": "tc5_auth",
            },
            knobs=(
                Knob("drop", DIFFERS, benign=["addresses"], extreme=["profiles"]),
                Knob("pseudonymize",
                     DIFFERS,
                     benign=["user_principal", "entity_key"],
                     extreme=["user_principal", "entity_key", "logcount"]),
                Knob("key", DIFFERS, benign=MINIMIZATION_KEY, extreme=OTHER_MINIMIZATION_KEY),
                # The one parameter that changes what a category means rather than what is done to it. At layer 5
                # `device_id` is the laptop somebody authenticated from and `addresses` covers it; at layer 1 it
                # is the switch that reported a port, and the same three words leave it alone.
                Knob("telemetry_class", DIFFERS, benign="tc5_auth", extreme="tc1"),
                Knob("digest_length", DIFFERS, benign=32, extreme=8),
                Knob("allow_unmasked_copies",
                     RAISES,
                     benign=True,
                     extreme=False,
                     frame=a_third_name_for_the_same_person),
            )),
    "ChainAnchorStage":
        Scenario(
            stage=ChainAnchorStage,
            frame=resolved_and_unresolved,
            base={
                "candidates": ["resolved_port_key", "arp_sender_ip"],
            },
            knobs=(
                Knob("candidates",
                     DIFFERS,
                     benign=["resolved_port_key", "arp_sender_ip"],
                     extreme=["arp_sender_ip", "resolved_port_key"]),
                Knob("anchor_column", DIFFERS, benign="chain_anchor", extreme="root"),
                Knob("source_column", DIFFERS, benign="chain_anchor_source", extreme="root_from"),
            ),
        ),
    "EnvelopeStampStage":
        Scenario(
            stage=EnvelopeStampStage,
            frame=port_inventory,
            base={
                "osi_layer": 1,
                "entity_columns": ["site_id", "device_id", "port_id"],
            },
            knobs=(
                Knob("osi_layer", DIFFERS, benign=1, extreme=5),
                Knob("entity_columns", DIFFERS, benign=["site_id", "device_id", "port_id"], extreme=["port_id"]),
                Knob("overwrite", DIFFERS, benign=False, extreme=True, frame=keyed_ports),
            ),
        ),
    "LineageStampStage":
        Scenario(
            stage=LineageStampStage,
            frame=flows,
            base={
                "id_columns": ["collector_id", "schema_version", "origin_hash", "collector_seq"],
                "parent_uid_column": "origin_hash",
            },
            knobs=(
                Knob("id_columns",
                     DIFFERS,
                     benign=["collector_id", "collector_seq"],
                     extreme=["collector_id", "schema_version", "origin_hash", "collector_seq"]),
                Knob("event_uid_column", DIFFERS, benign="event_uid", extreme="record_uid"),
                Knob("parent_uid_column", DIFFERS, benign="origin_hash", extreme="collector_id"),
                Knob("link_uid_column", DIFFERS, benign="link_uid", extreme="edge_uid"),
                Knob("relation", DIFFERS, benign="derived_from", extreme="observed_by"),
                Knob("join_method", DIFFERS, benign="direct", extreme="inferred"),
                Knob("join_method_column", DIFFERS, benign="resolution_method", extreme="join_kind"),
                Knob("digest_length", DIFFERS, benign=32, extreme=16),
                Knob("use_gpu_hashing",
                     INERT,
                     reason="Selects the cuDF hashing path, which is unreachable in CPU "
                     "mode; the equivalence of the two paths is asserted by verify_digest_equivalence in "
                     "tests/morpheus/utils/test_lineage_cudf.py, which is where a GPU is available."),
            ),
        ),
    "BindingResolverStage":
        Scenario(
            stage=BindingResolverStage,
            frame=flows,
            base={
                "binding_table": _binding_table(), "key_column": "src_ip"
            },
            knobs=(
                Knob("binding_table", DIFFERS, benign=_binding_table(), extreme=_binding_table(("hq:sw1:Gi9/9/9", ))),
                Knob("key_column", INPUT_COLUMN, benign="src_ip"),
                Knob("time_column", INPUT_COLUMN, benign="event_time"),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
                Knob("output_columns", DIFFERS, benign=None, extreme={"port_key": "resolved_port"}),
                Knob("method_column", DIFFERS, benign="resolution_method", extreme="how_resolved"),
                Knob("uid_column", DIFFERS, benign=None, extreme="binding_uid"),
                Knob("raise_on_unresolved", RAISES, benign=False, extreme=True),
            ),
        ),
    "SiemWireStage":
        Scenario(
            stage=SiemWireStage,
            frame=wire,
            base={"sourcetype": "morpheus:score:l2"},
            knobs=(
                Knob("sourcetype", DIFFERS, benign="morpheus:score:l2", extreme="binding:l2:open"),
                Knob("require_columns", RAISES, benign=False, extreme=True, frame=wire_without_the_required_columns),
                Knob("time_unit", DIFFERS, benign="ns", extreme="us"),
            ),
        ),
}

# --- The checks -----------------------------------------------------------------------------------------------


def _cases(kind: str) -> list:
    return [
        pytest.param(name, knob, id=f"{name}.{knob.name}") for (name, scenario) in sorted(REGISTRY.items())
        for knob in scenario.knobs if knob.kind == kind
    ]


@pytest.mark.parametrize("stage_name", sorted(REGISTRY))
def test_every_stage_parameter_is_registered(stage_name: str):
    """
    The registry has to be total, or it measures only what someone remembered to add.

    A parameter introduced later and never wired to anything is the exact defect this file exists to catch, and it
    would slip through silently if the list were hand-maintained.
    """
    scenario = REGISTRY[stage_name]
    declared = {knob.name for knob in scenario.knobs}
    actual = {name for name in inspect.signature(scenario.stage.__init__).parameters if name not in ("self", "c")}

    assert declared == actual, (f"{stage_name}: unregistered parameters {sorted(actual - declared)}; "
                                f"registered but gone {sorted(declared - actual)}")
    assert len(scenario.knobs) == len(declared), f"{stage_name}: a parameter is registered twice"


FORK_HEADER = "Copyright (c) 2026, NVIDIA CORPORATION."
"""The copyright line every source file this fork added carries.

Upstream files carry a year range ending in 2025, so the 2026 line is what separates them. Note this is the
source convention; the fork's *test* files use the longer SPDX form, and
`tests/morpheus/utils/test_gpu_conformance_targets.py` matches on that one. Two conventions, each internally
consistent, and a scan has to know which tree it is reading."""

STAGE_DIRECTORIES = (("stages", "telemetry"), ("stages", "lineage"), ("stages", "output"))


def _stage_classes_on_disk() -> set:
    """Every stage class this fork added, read from the tree rather than from a list beside it."""
    import os  # pylint: disable=import-outside-toplevel
    import re  # pylint: disable=import-outside-toplevel

    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    package = os.path.join(root, "python", "morpheus", "morpheus")
    found = set()

    for parts in STAGE_DIRECTORIES:
        directory = os.path.join(package, *parts)

        for name in sorted(os.listdir(directory)):
            if (not name.endswith(".py") or name.startswith("_")):
                continue

            with open(os.path.join(directory, name), encoding="utf-8") as handle:
                text = handle.read()

            if (FORK_HEADER not in text[:2000]):
                continue

            found.update(re.findall(r"^class (\w+)\(.*(?:Stage|Mixin).*\):", text, re.MULTILINE))

    return found


def test_every_stage_in_the_fork_is_covered():
    # Scanned from the tree, not pinned to a number. The pinned version read as though it enforced this and did
    # not: a count only notices a stage being removed, never one arriving, so `TC1BindingStage` landed as the
    # twenty-third stage while the assertion `len(REGISTRY) == 22` stayed green and its own comment claimed a new
    # stage could not arrive uncovered. That is the same defect as a tier list that reads as total and is not,
    # which this repository has now found in four separate places.
    on_disk = _stage_classes_on_disk()

    assert len(on_disk) > 15, f"the header scan found only {sorted(on_disk)}; it has stopped identifying stages"

    missing = sorted(on_disk - set(REGISTRY))

    assert missing == [], (f"{missing} ship in this fork and have no entry in REGISTRY, so no test asserts that "
                           f"any of their parameters do anything. Add a Scenario for each.")

    assert {scenario.stage.__name__ for scenario in REGISTRY.values()} == set(REGISTRY)


def test_the_readme_states_the_stage_count_each_telemetry_class_actually_ships():
    # The README tells a reader what to collect and, per telemetry class, how much of it this fork processes
    # today. That is a number in prose beside a number on disk, which is the shape that has drifted three times
    # already in this repository -- most recently a sourcetype count that stayed wrong through a whole
    # increment. A reader deciding what to instrument acts on this table, so it is checked rather than trusted.
    import os  # pylint: disable=import-outside-toplevel
    import re  # pylint: disable=import-outside-toplevel

    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    telemetry = os.path.join(repo_root, "python", "morpheus", "morpheus", "stages", "telemetry")

    with open(os.path.join(repo_root, "README.md"), encoding="utf-8") as handle:
        readme = re.sub(r"\s+", " ", handle.read())

    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}

    # Every class with a producer. TC-3 was not in this loop when its stages landed, and its row went on saying
    # "Schema only" through a whole increment -- the same drift this test exists to catch, in the one class the
    # loop did not name. A list of classes is as capable of being incomplete as a count is.
    for prefix in ("tc0", "tc1", "tc2", "tc3", "tc4", "tc5", "tc6", "tc7"):
        shipped = len(
            [name for name in os.listdir(telemetry) if name.startswith(f"{prefix}_") and name.endswith(".py")])
        label = f"**TC-{prefix[-1]}**"
        row = re.search(rf"\{label[:6]}\*\* [^|]*\|[^|]*\|[^|]*\| (\w+) stages ship", readme)

        assert row is not None, f"the README's collection table no longer states a stage count for TC-{prefix[-1]}"
        assert words[row.group(1).lower()] == shipped, (
            f"the README says {row.group(1)} stages ship for TC-{prefix[-1]}; {shipped} are on disk")


def test_the_readme_uses_links_rather_than_sphinx_roles():
    # The guide is built by Sphinx and `{py:mod}` renders there as a cross-reference. The README is read on
    # GitHub, which renders it as the literal text `{py:mod}`morpheus.utils.event_clock``. Two of these were
    # written into the collection section by habit from editing the guide, and nothing but a reader opening the
    # rendered page would have noticed.
    import os  # pylint: disable=import-outside-toplevel
    import re  # pylint: disable=import-outside-toplevel

    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

    for relative in ("README.md",
                     os.path.join("examples", "layer5_model", "README.md"),
                     os.path.join("examples", "splunk_lineage_app", "README.md")):
        with open(os.path.join(repo_root, relative), encoding="utf-8") as handle:
            found = re.findall(r"\{py:\w+\}`[^`]+`", handle.read())

        assert found == [], f"{relative} uses Sphinx roles, which render as literal text on GitHub: {found}"


@pytest.mark.cpu_mode
@pytest.mark.parametrize(("stage_name", "knob"), _cases(DIFFERS))
def test_changing_the_parameter_changes_the_output(stage_name: str, knob: Knob):
    """Two values, one corpus built to make the parameter bite. A parameter nothing reads produces one answer."""
    scenario = REGISTRY[stage_name]
    frame = knob.frame() if knob.frame is not None else None
    benign = _outcome(scenario, {knob.name: knob.benign}, frame=frame)
    extreme = _outcome(scenario, {knob.name: knob.extreme}, frame=frame)

    assert benign != extreme, (f"{stage_name}.{knob.name} behaved identically at {knob.benign!r} and "
                               f"{knob.extreme!r}. Either the stage does not read it, or this corpus does not "
                               f"make it bite -- both are worth knowing and neither is acceptable.")


@pytest.mark.cpu_mode
@pytest.mark.parametrize(("stage_name", "knob"), _cases(INPUT_COLUMN))
def test_the_column_parameter_is_the_column_that_is_read(stage_name: str, knob: Knob):
    """
    Rename the column and tell the stage its new name. A stage that ignores the parameter reads the old name.

    Asserting the output is unchanged, rather than merely that nothing raised: a stage could fall back to a
    default, log, and emit nulls, which is the quiet version of the same defect.
    """
    scenario = REGISTRY[stage_name]
    renamed = "renamed_" + knob.benign
    frame = {(renamed if name == knob.benign else name): values for (name, values) in scenario.frame().items()}

    overrides = {knob.name: renamed}
    overrides.update(knob.also or {})

    baseline = _text(_run(scenario, {}))
    # Rename the column back on the way out, so the comparison is about the values rather than the heading.
    moved = _text(_run(scenario, overrides, frame=frame), rename={renamed: knob.benign})

    assert baseline == moved, (
        f"{stage_name}.{knob.name}: renaming {knob.benign!r} to {renamed!r} and passing the new name did not "
        f"reproduce the original output, so the parameter is not what the stage reads.")


@pytest.mark.cpu_mode
@pytest.mark.parametrize(("stage_name", "knob"), _cases(RAISES))
def test_the_parameter_makes_the_stage_refuse(stage_name: str, knob: Knob):
    """Some parameters exist to turn a tolerated condition into a refusal. The refusal is the observable."""
    scenario = REGISTRY[stage_name]
    frame = knob.frame() if knob.frame is not None else None

    # It must refuse with the value set, and it must not refuse without it, or the value is not what refused.
    _run(scenario, {knob.name: knob.benign}, frame=frame)

    with pytest.raises((ValueError, KeyError)):
        _run(scenario, {knob.name: knob.extreme}, frame=frame)


@pytest.mark.parametrize(("stage_name", "knob"), _cases(INERT))
def test_a_parameter_declared_inert_says_why(stage_name: str, knob: Knob):
    # Nothing runs. The point is that declaring a parameter untestable is a written, reviewable act rather than an
    # omission, and that the reason travels with the registry instead of living in a commit message.
    assert len(knob.reason) >= 30
    assert knob.name in set(inspect.signature(REGISTRY[stage_name].stage.__init__).parameters)
