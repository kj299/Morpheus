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
The snapshot-shaped layer 1 and layer 2 corpus, and the composed telemetry pipeline, for the determinism harness.

The lineage harness proved the lineage substrate deterministic. Nothing had ever run a telemetry stage composed
with another, and the retrospective found that the telemetry stages' own tests were shaped like the tests rather
than like the network. This module is the answer to both. Its corpus is shaped the way collectors actually report:
per-port SNMP polls with uptime and `ifLastChange` in hundredths of a second, a full MAC table snapshot every five
minutes with every address on a port stamped at the snapshot's instant, and ARP at one-second resolution with a
whole burst landing on one tick.

Into that corpus are planted the things the layer 1 and layer 2 features exist to see:

- a **hub**: five MAC addresses behind one access port from the seventh snapshot onward;
- a **spoof**: one MAC reported on two ports in the same snapshot;
- a **cross-switch spoof**: one MAC claimed on a second switch two seconds after it was seen on its own, which is
  the shape a sequentially polled estate actually produces and the one the simultaneous case cannot stand in for;
- a **legitimate move**: one MAC that changes port between snapshots, so a whole cadence separates its two
  sightings and the rule that catches the spoof has something it must not fire on;
- a **flood**: twenty gratuitous ARP replies from one host in one second, claiming the gateway;
- a **reboot**: a device whose uptime and counters restart mid-corpus;
- a **tap**: a step loss of receive power on one port, with transmit power unchanged;
- a **flap**: a link that went down and up between two polls, visible only through `ifLastChange`;
- a **bypass**: an 802.1X success on a port that never started an exchange;
- and one thing that must **not** fire: a VRRP pair whose two MACs legitimately share one address, carried on the
  exclusion list, so the ARP rule's exclusion path is exercised rather than assumed.

The pipeline is one per telemetry class, which is the deployment shape: each class arrives on its own topic and
its stages require its own columns. They compose where the design says they must. Layer 2's binding stage emits
closed bindings; those become a `BindingTable`; layer 2's ARP stream resolves through it; and the port it resolves
to is, byte for byte, the layer 1 `entity_key`. That join is asserted in the harness, not assumed.

Every stateful stage is preceded by `TotalOrderStage`, which is determinism control 8 as a stage. Without it the
permutation check fails, correctly: a counter delta is the difference from the previous sample, and the telemetry
stages flag out-of-order arrival rather than repairing it.
"""

import random
import typing

import pandas as pd

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.binding_resolver_stage import BindingResolverStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc1_change_stage import TC1ChangeStage
from morpheus.stages.telemetry.tc1_flap_stage import TC1FlapStage
from morpheus.stages.telemetry.tc1_normalize_stage import TC1NormalizeStage
from morpheus.stages.telemetry.tc1_optical_stage import TC1OpticalStage
from morpheus.stages.telemetry.tc2_arp_stage import TC2ArpStage
from morpheus.stages.telemetry.tc2_auth_stage import TC2AuthStage
from morpheus.stages.telemetry.tc2_binding_stage import TC2BindingStage
from morpheus.stages.telemetry.tc2_cardinality_stage import TC2CardinalityStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.binding_table import BindingTable
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize
from morpheus.utils.lineage import event_uid

CORPUS_SEED = 20260902

PERIOD_SECONDS = 300
LATENESS_SECONDS = 900
CORPUS_SECONDS = 3600

ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []

SITE = "hq"
SWITCH = "sw1"
REBOOTING_SWITCH = "sw2"
PEER_SWITCH = "sw3"
PORTS = ["Gi1/0/1", "Gi1/0/2", "Gi1/0/3"]
PEER_PORTS = ["Gi3/0/1", "Gi3/0/2", "Gi3/0/3"]
"""A second switch in the MAC-table feed. Without one the corpus could only produce a MAC in two places on a single
switch at a single instant, which is the one shape R-D-L2-004 could already detect and the one shape a real estate
never produces: a poller walks its switches in sequence, so two sightings of a spoofed MAC are seconds apart, not
simultaneous. The peer switch's ports are deliberately outside `SINGLE_HOST_PORTS`, so planting MACs on them says
nothing to R-D-L2-001."""

MAC_A = "aa:bb:cc:00:00:01"
MAC_B = "aa:bb:cc:00:00:02"
MAC_C = "aa:bb:cc:00:00:03"
ROAM_MAC = "aa:bb:cc:00:00:04"
"""A device that legitimately moves ports. The negative control for R-D-L2-004: it is displaced like a spoof, but a
whole poll cadence separates the two sightings, so the gap says it moved rather than that it was in two places."""
HUB_MACS = [f"de:ad:be:ef:00:{index:02x}" for index in range(1, 5)]
ROUTER_MAC = "00:00:5e:00:01:01"
GATEWAY_IP = "10.0.0.1"
VRRP_IP = "10.0.0.254"
VRRP_MACS = ["00:00:5e:00:01:fe", "00:00:5e:00:01:ff"]
"""A first-hop redundancy pair. Two MACs claim one address by design, and the exclusion list says so."""

SINGLE_HOST_PORTS = {f"{SITE}:{SWITCH}:{port}" for port in PORTS}
"""The corpus's own port designations, standing in for the inventory-supplied list R-D-L2-001 reads."""
HOST_IPS = {MAC_A: "10.0.0.11", MAC_B: "10.0.0.12", MAC_C: "10.0.0.13"}

HUB_PORT = "Gi1/0/3"
HUB_FROM_SECONDS = 1800
SPOOF_AT_SECONDS = 2700
FLOOD_AT_SECONDS = 2400
FLOOD_PACKETS = 20
REBOOT_AT_MINUTE = 30
TAP_AT_MINUTE = 40
TAP_LOSS_DB = 3.0
FLAP_AT_MINUTE = 20
BYPASS_AT_SECONDS = 1500
BYPASS_PORT = "Gi1/0/2"

SWEEP_OFFSET_SECONDS = 2
"""How long after the first switch the poller reaches the peer. This is the whole point of the second switch: the
sightings that make up a cross-switch spoof are this far apart, never simultaneous."""
CROSS_SPOOF_AT_SECONDS = 900
"""MAC_B is claimed on the peer switch while it is still live on its own port, inside a single sweep."""
ROAM_AT_SECONDS = 2100
"""ROAM_MAC changes port between two snapshots, so its two sightings are a full cadence apart."""

CS_PER_SECOND = 100

TELEMETRY_CLASSES = ("tc1", "tc2_mac", "tc2_binding", "tc2_arp", "tc2_auth")


def _envelope(rng: random.Random, collector: str, schema: str, seq: int) -> dict:
    return {
        "collector_id": collector,
        "schema_version": schema,
        "origin_hash": f"{rng.getrandbits(64):016x}",
        "collector_seq": seq,
    }


def build_corpus() -> dict[str, pd.DataFrame]:
    """
    Build the fixed corpus, one frame per telemetry class.

    Every value derives from `CORPUS_SEED`, so the corpus is as fixed as a checked-in file while remaining reviewable
    as code. Rows are generated in event order with a monotonic `collector_seq`, which is the envelope's own
    requirement; the harness's permutation check is what scrambles them.
    """
    rng = random.Random(CORPUS_SEED)

    return {
        "tc1": _build_layer_1(rng),
        "tc2_mac": _build_mac_snapshots(rng),
        "tc2_arp": _build_arp(rng),
        "tc2_auth": _build_auth(rng),
    }


def _build_layer_1(rng: random.Random) -> pd.DataFrame:
    """Per-port SNMP polls at one-minute cadence, with a reboot, a tap, and an unpolled flap planted."""
    devices = [(SWITCH, port) for port in PORTS] + [(REBOOTING_SWITCH, "Gi1/0/1")]
    counters = {
        key: {
            "crc_errors": 100, "symbol_errors": 0, "input_discards": 5, "output_discards": 1
        }
        for key in devices
    }
    boot_uptime_cs = 3600 * CS_PER_SECOND
    rows = []
    seq = 0

    for minute in range(0, CORPUS_SECONDS // 60 + 1):
        time_s = minute * 60

        for (device, port) in devices:
            rebooted = device == REBOOTING_SWITCH and minute >= REBOOT_AT_MINUTE

            if (device == REBOOTING_SWITCH and minute == REBOOT_AT_MINUTE):
                counters[(device, port)] = {name: 0 for name in counters[(device, port)]}

            for name in counters[(device, port)]:
                counters[(device, port)][name] += rng.choice([0, 0, 0, 1, 2])

            uptime_cs = (minute - REBOOT_AT_MINUTE) * 60 * CS_PER_SECOND + 30 * CS_PER_SECOND if rebooted else (
                boot_uptime_cs + minute * 60 * CS_PER_SECOND)

            # ifLastChange is relative to the device's own boot. The flap on Gi1/0/1 advances it inside one polling
            # gap while the state reads "up" both sides, which is the case only the device's own record can reveal.
            if (device == SWITCH and port == "Gi1/0/1" and minute >= FLAP_AT_MINUTE):
                last_change_cs = FLAP_AT_MINUTE * 60 * CS_PER_SECOND - 30 * CS_PER_SECOND
            elif (rebooted):
                last_change_cs = 5 * CS_PER_SECOND
            else:
                last_change_cs = 10 * CS_PER_SECOND

            rx_dbm = -7.0 + rng.uniform(-0.05, 0.05)

            if (device == SWITCH and port == HUB_PORT and minute >= TAP_AT_MINUTE):
                rx_dbm -= TAP_LOSS_DB

            seq += 1
            rows.append({
                "event_time": time_s * NS_PER_SECOND,
                "site_id": SITE,
                "device_id": device,
                "port_id": port,
                "uptime": uptime_cs,
                "if_last_change": last_change_cs,
                "oper_status": "up",
                "optical_tx_dbm": round(-2.0 + rng.uniform(-0.05, 0.05), 3),
                "optical_rx_dbm": round(rx_dbm, 3),
                "transceiver_serial": f"XCVR-{device}-{port}",
                "lldp_neighbor_chassis_id": f"nbr-{device}-{port}",
                **counters[(device, port)],
                **_envelope(rng, "snmp-poller", "TC-1/1.0.0", seq),
            })

    return pd.DataFrame(rows)


def _build_mac_snapshots(rng: random.Random) -> pd.DataFrame:
    """A full MAC table snapshot every five minutes, every row stamped at the snapshot's instant."""
    rows = []
    seq = 0

    for time_s in range(0, CORPUS_SECONDS + 1, PERIOD_SECONDS):
        entries = [(MAC_A, "Gi1/0/1"), (MAC_B, "Gi1/0/2"), (MAC_C, "Gi1/0/3")]

        if (time_s >= HUB_FROM_SECONDS):
            entries.extend((mac, HUB_PORT) for mac in HUB_MACS)

        if (time_s == SPOOF_AT_SECONDS):
            entries.append((MAC_A, "Gi1/0/2"))

        for (mac, port) in entries:
            seq += 1
            rows.append({
                "event_time": time_s * NS_PER_SECOND,
                "mac_address": mac,
                "site_id": SITE,
                "switch_id": SWITCH,
                "port_id": port,
                "vlan_id": "10",
                **_envelope(rng, "mac-table", "TC-2/1.0.0", seq),
            })

        # The poller reaches the peer switch a couple of seconds after the first, which is what makes the spoof
        # below `displaced` with a small gap rather than a `conflict`. Emitted here, inside the same snapshot, so
        # the frame stays monotonic in event time the way a collector's stream is: a frame that jumped backwards
        # would make the output depend on where the batch boundaries fell, which control 13 checks for.
        for (mac, port) in ([(ROAM_MAC, PEER_PORTS[2] if time_s >= ROAM_AT_SECONDS else PEER_PORTS[1])] +
                            ([(MAC_B, PEER_PORTS[0])] if time_s == CROSS_SPOOF_AT_SECONDS else [])):
            seq += 1
            rows.append({
                "event_time": (time_s + SWEEP_OFFSET_SECONDS) * NS_PER_SECOND,
                "mac_address": mac,
                "site_id": SITE,
                "switch_id": PEER_SWITCH,
                "port_id": port,
                "vlan_id": "10",
                **_envelope(rng, "mac-table", "TC-2/1.0.0", seq),
            })

    return pd.DataFrame(rows)


def _build_arp(rng: random.Random) -> pd.DataFrame:
    """ARP at one-second resolution: hosts asking for the gateway, the gateway announcing itself, and one flood."""
    events: list[tuple[int, str, str, str, str]] = []

    for time_s in range(5, CORPUS_SECONDS, 10):
        for (mac, ip) in HOST_IPS.items():
            events.append((time_s + rng.randrange(0, 3), ip, mac, GATEWAY_IP, "request"))

    for time_s in range(0, CORPUS_SECONDS, 60):
        events.append((time_s, GATEWAY_IP, ROUTER_MAC, GATEWAY_IP, "reply"))

    # The redundancy pair announces the shared address from alternating MACs. Legitimate, and the reason the
    # exclusion list exists: without it this reads exactly like the flood below.
    for (index, time_s) in enumerate(range(30, CORPUS_SECONDS, 60)):
        events.append((time_s, VRRP_IP, VRRP_MACS[index % 2], VRRP_IP, "reply"))

    # The flood: twenty gratuitous replies claiming the gateway, from the host on Gi1/0/1, inside one second. A
    # source stamping at one-second resolution puts all twenty on one tick.
    for _ in range(FLOOD_PACKETS):
        events.append((FLOOD_AT_SECONDS, GATEWAY_IP, MAC_A, GATEWAY_IP, "reply"))

    events.sort(key=lambda event: event[0])
    rows = []

    for (seq, (time_s, sender_ip, sender_mac, target_ip, operation)) in enumerate(events, start=1):
        rows.append({
            "event_time": time_s * NS_PER_SECOND,
            "arp_sender_ip": sender_ip,
            "arp_sender_mac": sender_mac,
            "arp_target_ip": target_ip,
            "arp_operation": operation,
            **_envelope(rng, "arp-sensor", "TC-2/1.0.0", seq),
        })

    return pd.DataFrame(rows)


def _build_auth(rng: random.Random) -> pd.DataFrame:
    """802.1X exchanges per port, plus one success that nothing preceded."""
    events: list[tuple[int, str, str]] = []

    for (index, port) in enumerate(PORTS):
        for time_s in range(60 + index * 7, CORPUS_SECONDS, 900):
            events.append((time_s, port, "started"))
            events.append((time_s + 3 + index, port, "success"))

    events.append((BYPASS_AT_SECONDS, BYPASS_PORT, "success"))
    events.sort(key=lambda event: event[0])
    rows = []

    for (seq, (time_s, port, result)) in enumerate(events, start=1):
        rows.append({
            "event_time": time_s * NS_PER_SECOND,
            "site_id": SITE,
            "switch_id": SWITCH,
            "port_id": port,
            "dot1x_result": result,
            **_envelope(rng, "radius", "TC-2/1.0.0", seq),
        })

    return pd.DataFrame(rows)


def build_pipeline_config() -> Config:
    """A CPU-mode pipeline configuration, importable without a GPU."""
    from morpheus.config import CppConfig
    from morpheus.config import ExecutionMode

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.execution_mode = ExecutionMode.CPU

    return config


def _collect(sink: InMemorySinkStage) -> pd.DataFrame:
    frames = []

    for message in sink.get_messages():
        meta = message.payload() if isinstance(message, ControlMessage) else message
        df = meta.copy_dataframe()
        frames.append(df.to_pandas() if hasattr(df, "to_pandas") else df)

    if (len(frames) == 0):
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _run_class(config: Config,
               dataframes: list[pd.DataFrame],
               stages: list,
               impose_order: bool,
               seal: bool = True) -> pd.DataFrame:
    """Source → stamp → (total order) → the class's stages → (window seal) → sink, collected to one frame."""
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    for stage in stages:
        pipe.add_stage(stage)

    if (seal):
        pipe.add_stage(
            WindowSealStage(config,
                            period_seconds=PERIOD_SECONDS,
                            lateness_seconds=LATENESS_SECONDS,
                            order_columns=list(DEFAULT_ORDER_COLUMNS)))

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    return _collect(sink)


def build_binding_table(bindings: pd.DataFrame) -> BindingTable:
    """The layer 2 bindings as the resolver consumes them: MAC to the port as layer 1 spells it."""
    return BindingTable.from_dataframe(bindings,
                                       name="mac_table",
                                       key_column="mac_address",
                                       value_columns=["port_key", "vlan_id"],
                                       start_column="bind_start",
                                       end_column="bind_end")


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True) -> pd.DataFrame:
    """
    Run every telemetry class through its pipeline and return one canonicalized frame.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    corpus : dict
        The frames from `build_corpus`, possibly permuted.
    batches : dict, optional
        Per class, how the corpus is split across source frames. Defaults to one frame per class. The batch-split
        sweep is the caller's to vary.
    impose_order : bool, default = True
        Place `TotalOrderStage` ahead of the stateful stages. The permutation check's negative control turns it off
        to reproduce the removed-sort defect and prove the harness catches it.

    Returns
    -------
    `pandas.DataFrame`
        Every class's output, tagged with `telemetry_class`, keyed by `row_key`, canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    outputs = {}

    # Layer 1: counters, optics, flaps, identifier changes.
    outputs["tc1"] = _run_class(config,
                                batches["tc1"],
                                [
                                    TC1NormalizeStage(config, uptime_column="uptime", uptime_unit="cs"),
                                    TC1OpticalStage(config),
                                    TC1FlapStage(config, last_change_column="if_last_change", last_change_unit="cs"),
                                    TC1ChangeStage(config),
                                ],
                                impose_order)

    # Layer 2, from the same snapshots: the cardinality features, and the closed bindings.
    outputs["tc2_mac"] = _run_class(config, batches["tc2_mac"], [TC2CardinalityStage(config)], impose_order)
    bindings = _run_class(config, batches["tc2_mac"], [TC2BindingStage(config)], impose_order, seal=False)

    bindings["row_key"] = [
        event_uid("binding", *values)
        for values in zip(bindings["mac_address"], bindings["port_key"], bindings["bind_start"], bindings["bind_end"],
                          bindings["bind_end_reason"])
    ]
    outputs["tc2_binding"] = bindings

    # Layer 2 ARP, resolved through the bindings the previous pipeline just closed. This is the composition the
    # ladder depends on: a stage's output becomes a table, and another stage resolves through it.
    outputs["tc2_arp"] = _run_class(
        config,
        batches["tc2_arp"],
        [
            TC2ArpStage(config, excluded_sender_ips=[VRRP_IP]),
            BindingResolverStage(config,
                                 binding_table=build_binding_table(bindings),
                                 key_column="arp_sender_mac",
                                 output_columns={
                                     "port_key": "resolved_port_key", "vlan_id": "resolved_vlan_id"
                                 },
                                 uid_column="binding_uid"),
        ],
        impose_order)

    outputs["tc2_auth"] = _run_class(config, batches["tc2_auth"], [TC2AuthStage(config)], impose_order)

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
    return result.to_csv(index=False, lineterminator="\n")
