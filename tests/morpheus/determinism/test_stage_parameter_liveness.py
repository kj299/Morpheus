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
from morpheus.stages.lineage.community_id_stage import CommunityIdStage
from morpheus.stages.lineage.determinism_stamp_stage import DeterminismStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.siem_wire_stage import SiemWireStage
from morpheus.stages.telemetry.tc1_change_stage import TC1ChangeStage
from morpheus.stages.telemetry.tc1_feature_stage import TC1FeatureStage
from morpheus.stages.telemetry.tc1_flap_stage import TC1FlapStage
from morpheus.stages.telemetry.tc1_normalize_stage import TC1NormalizeStage
from morpheus.stages.telemetry.tc1_optical_stage import TC1OpticalStage
from morpheus.stages.telemetry.tc2_arp_stage import TC2ArpStage
from morpheus.stages.telemetry.tc2_auth_stage import TC2AuthStage
from morpheus.stages.telemetry.tc2_binding_stage import TC2BindingStage
from morpheus.stages.telemetry.tc2_cardinality_stage import TC2CardinalityStage
from morpheus.stages.telemetry.tc5_cadence_stage import TC5CadenceStage
from morpheus.stages.telemetry.tc5_drift_stage import TC5DriftStage
from morpheus.stages.telemetry.tc5_novelty_stage import TC5NoveltyStage
from morpheus.stages.telemetry.tc5_risk_stage import TC5RiskStage
from morpheus.stages.telemetry.tc5_session_stage import TC5SessionStage
from morpheus.stages.telemetry.tc5_travel_stage import TC5TravelStage
from morpheus.utils.binding_table import Binding
from morpheus.utils.binding_table import BindingTable
from morpheus.utils.determinism_envelope import DeterminismEnvelope
from morpheus.utils.model_manifest import ModelManifest

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


def test_every_stage_in_the_fork_is_covered():
    # The registry is checked against each stage's signature above; this checks the set of stages itself, so a new
    # stage cannot arrive with no entry at all.
    assert len(REGISTRY) == 22
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

    for prefix in ("tc1", "tc2", "tc5"):
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
