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

- a **hub**: five MAC addresses behind one access port from the seventh snapshot onward, four above the most the
  port had carried in any of the six before, which is the step R-B-L2-002 fires on;
- a **spoof**: one MAC reported on two ports in the same snapshot;
- a **cross-switch spoof**: one MAC claimed on a second switch two seconds after it was seen on its own, which is
  the shape a sequentially polled estate actually produces and the one the simultaneous case cannot stand in for;
- a **legitimate move**: one MAC that changes port between snapshots, so a whole cadence separates its two
  sightings and the rule that catches the spoof has something it must not fire on;
- a **flood**: twenty gratuitous ARP replies from one host in one second, claiming the gateway;
- a **reboot**: a device whose uptime and counters restart mid-corpus;
- a **tap**: a step loss of receive power on one port, with transmit power unchanged;
- a **failing optic**: a receive level sliding down a few hundredths of a decibel every poll until the maintenance
  swap below replaces it, which is what R-P-L1-004 fires on, where the tap's step must not;
- a **flap**: a link that went down and up between two polls, visible only through `ifLastChange`;
- an **optic swap**: a transceiver serial that changes on a port while the device records no link transition,
  which is what R-D-L1-001 fires on, and beside it a **maintenance swap** whose serial changes with the link's
  drop recorded in `ifLastChange`, which it must not;
- a **bypass**: an 802.1X success on a port that never started an exchange;
- a **flapper**: a link that goes down and up between every pair of polls for the second half of the hour, which
  R-D-L1-003 fires on and the one flap above must not;
- a **failing cable**: a port whose CRC errors climb every poll for the last quarter of the hour, R-B-L1-005's case;
- a **re-patch** and an **inline insertion**: two ports whose LLDP neighbour changes, the first to another of the
  estate's switches and the second to a device the estate has never seen, which R-D-L1-006 reads, beside a switch
  that comes back from its reboot with no neighbours yet, which it must not;
- a **surge** and a **silence**: a port that starts carrying fifteen times what it ever has, and one that stops
  carrying anything, which R-B-L1-007 reads in both directions;
- a **second VLAN** whose cameras share one vendor until a single-board computer appears on it, which R-B-L2-006
  reads, beside a third camera of the known vendor that it must not;
- a **printer unplugged** and a **guest who left**: a MAC the next table walk no longer lists, closed as absent from
  the snapshot, and one the switch's own notification says was removed, closed as an end somebody observed;
- an 802.1X exchange that **took three attempts** and one that **took no time at all**, which R-B-L2-007 and
  R-B-L2-008 read against the port's own exchanges;
- and one thing that must **not** fire: a VRRP pair whose two MACs legitimately share one address, carried on the
  exclusion list, so the ARP rule's exclusion path is exercised rather than assumed.

New plants draw their randomness from a second generator, `EXTRA_SEED`, so that adding one does not move a single
value in the rows that were already here: the ARP and authentication streams, and the counts the documents quote
from them, stay what they were.

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

import stamping
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.binding_resolver_stage import BindingResolverStage
from morpheus.stages.lineage.chain_anchor_stage import DEFAULT_ANCHOR_COLUMN
from morpheus.stages.lineage.chain_anchor_stage import ChainAnchorStage
from morpheus.stages.lineage.determinism_stamp_stage import DeterminismStampStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc1_change_stage import TC1ChangeStage
from morpheus.stages.telemetry.tc1_flap_stage import TC1FlapStage
from morpheus.stages.telemetry.tc1_forecast_stage import TC1ForecastStage
from morpheus.stages.telemetry.tc1_normalize_stage import TC1NormalizeStage
from morpheus.stages.telemetry.tc1_optical_stage import TC1OpticalStage
from morpheus.stages.telemetry.tc1_rate_stage import TC1RateStage
from morpheus.stages.telemetry.tc2_arp_stage import TC2ArpStage
from morpheus.stages.telemetry.tc2_auth_stage import TC2AuthStage
from morpheus.stages.telemetry.tc1_binding_stage import TC1BindingStage
from morpheus.stages.telemetry.tc2_baseline_stage import TC2BaselineStage
from morpheus.stages.telemetry.tc2_binding_stage import TC2BindingStage
from morpheus.stages.telemetry.tc2_cardinality_stage import TC2CardinalityStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.binding_table import BindingTable
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize
from morpheus.utils.lineage import event_uid

CORPUS_SEED = 20260902
EXTRA_SEED = 20261010
"""The second generator, for every row and field added after the first corpus. See the module docstring."""

PERIOD_SECONDS = 300
LATENESS_SECONDS = 900
CORPUS_SECONDS = 3600

ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []

SITE = "hq"
SWITCH = "sw1"
REBOOTING_SWITCH = "sw2"
# A numeric VLAN, which is what a real MAC-table feed sends. Sent as a string it could never exercise the
# widening this corpus exists to catch: `vlan_id` is the entity `ouis_per_vlan` counts by, and one row with a
# null VLAN widens the column to float, so VLAN 10 would render as `10.0` and fork into a second entity whose
# OUI count restarts. Which rows are in which batch would then decide the answer.
VLAN = 10

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

IOT_VLAN = 20
CAMERAS = {"00:40:8c:00:00:01": "Gi1/0/13", "00:40:8c:00:00:02": "Gi1/0/14"}
"""The second VLAN's devices: two cameras of one vendor, the segment's whole population."""
ROGUE_OUI_MAC = "b8:27:eb:00:00:01"
ROGUE_OUI_PORT = "Gi1/0/15"
ROGUE_OUI_AT_SECONDS = 2400
"""A single-board computer plugged into the camera VLAN: a vendor the segment has never carried."""
THIRD_CAMERA_MAC = "00:40:8c:00:00:03"
THIRD_CAMERA_PORT = "Gi1/0/16"
THIRD_CAMERA_AT_SECONDS = 2700
"""Another camera of the known vendor: a new device on the VLAN, and not a new kind of one."""

PRINTER_MAC = "aa:bb:cc:00:00:08"
PRINTER_PORT = "Gi1/0/17"
PRINTER_LAST_SEEN_SECONDS = 1200
"""A printer unplugged between two table walks. The walk at 1500 no longer lists it, which is the only thing a MAC
table says about a device that left."""

GUEST_MAC = "aa:bb:cc:00:00:09"
GUEST_PORT = "Gi1/0/18"
GUEST_FROM_SECONDS = 600
GUEST_LEAVES_AT_SECONDS = 1560
"""A guest laptop, whose departure the switch reports itself: a MAC notification saying the address was removed."""

MAC_ACTION_COLUMN = "mac_action"
"""What a MAC table row says happened: null on a snapshot row, `removed` on the switch's own notification."""

SINGLE_HOST_PORTS = {f"{SITE}:{SWITCH}:{port}" for port in PORTS}
"""The corpus's own port designations, standing in for the inventory-supplied list R-D-L2-001 reads."""

BASELINE_MIN_BUCKETS = 6
"""Snapshot periods a port must have been seen in before R-B-L2-002 has a baseline to measure it against.

Half an hour of five-minute snapshots. A deployment keeps a month of hourly peaks and asks for a day of them, which
is what `TC2BaselineStage` defaults to; this corpus is one hour long, so its periods are the sealing period and its
floor is the six snapshots before the hub arrives. The parameters say so rather than the corpus pretending to be a
month.
"""
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
XCVR_SWAP_AT_MINUTE = 50
XCVR_SWAP_PORT = "Gi1/0/2"
"""Somebody replaces the optic in one port partway through.

The event the `binding_l1` lookup exists to record, and until this was planted the composed corpus never changed
a transceiver at all -- so every port bound once and drained, and the displacement path ran only in unit tests.
Its negative control is already here and needed no planting: the tap on `HUB_PORT` moves that port's receive
power by three decibels without touching its serial, and must leave its binding whole. A binding that split on a
changing optical reading would produce a new interval every poll.

It is also what R-D-L1-001 fires on, as recorded: the serial changes while `oper_status` reads "up" on both polls
and `ifLastChange` never moves, so the device says the link was never down. Replacing an optic means pulling it,
and pulling it takes the link down, so a serial that changes without that transition is a change the port cannot
physically have produced. The swap below is the one that can.
"""

MAINTENANCE_PORT = "Gi1/0/6"
MAINTENANCE_SWAP_AT_MINUTE = 45
"""A second optic replaced, the way a technician replaces one.

The serial changes between two polls that both read "up", exactly as on `XCVR_SWAP_PORT`, with one difference:
the device's `ifLastChange` advanced between them, because the link dropped while the cage was empty and came back
with the new optic. `TC1FlapStage` reads that as two transitions nobody polled, and R-D-L1-001 must stay quiet on
it. Without this port the rule could only be asserted in one direction, and a rule that fires on every optic
swap in the estate is not the rule the guide specifies.
"""

SWAPS = {XCVR_SWAP_PORT: XCVR_SWAP_AT_MINUTE, MAINTENANCE_PORT: MAINTENANCE_SWAP_AT_MINUTE}
"""The minute each replaced optic's new serial first appears, per port."""

FLAPPER_PORT = "Gi1/0/7"
FLAPPER_FROM_MINUTE = 30
"""A link that goes down and comes back between every pair of polls from the half hour.

The device records each drop in `ifLastChange` and the poller sees "up" either side, so every poll counts two
transitions and the hour's count climbs by two a minute. The single flap on Gi1/0/1 is two transitions in the hour,
and the reboot's are labelled as a reset: neither is instability."""

ERROR_PORT = "Gi1/0/8"
ERRORS_FROM_MINUTE = 45
ERRORS_PER_MINUTE_STEP = 30
"""A cable going bad: from minute forty-five the port's CRC errors climb by thirty more each minute than the last.
Every port runs a few errors a minute, which is what the climb is measured against."""

VOLUME_PORT = "Gi1/0/9"
SILENT_PORT = "Gi1/0/10"
VOLUME_FROM_MINUTE = 50
SURGE_BITS_PER_SECOND = 900_000_000
"""From minute fifty, `VOLUME_PORT` sends nine hundred megabits a second where it never sent more than sixty, and
`SILENT_PORT`, which never went quiet, carries nothing at all with its link still up."""

REPATCH_PORT = "Gi1/0/11"
INLINE_PORT = "Gi1/0/12"
REPATCH_AT_MINUTE = 35
INLINE_AT_MINUTE = 38
NEIGHBOURS = {REPATCH_PORT: ("dist-sw3", "Gi3/0/11"), INLINE_PORT: ("dist-sw3", "Gi3/0/12")}
"""What is on the far end of the two uplinks that change. Each change takes the link down, as cabling does."""
REPATCHED_TO = ("dist-sw4", "Gi4/0/11")
INLINE_DEVICE = ("00:1b:21:7f:3a:01", "eth0")
"""A re-patch moves one uplink to another of the estate's distribution switches; an inline insertion puts a device
the estate has never seen between the access switch and its distribution switch, announcing itself by its own MAC
as LLDP chassis IDs usually are."""

REBOOT_PORTS = ["Gi1/0/1", "Gi1/0/2", "Gi1/0/3"]
"""The rebooting switch's ports. Three, so the restart is three ports' flags and one device's event: the shape the
'Device restart' search collapses."""

LINK_SPEED_BPS = 10_000_000_000
"""Every port's speed: the optics are 10GBASE-LR."""

BASE_BITS_PER_SECOND = 40_000_000
"""What an ordinary port carries in each direction, give or take a tenth."""

OPTIC_TYPE = "10GBASE-LR"
OPTIC_FLOOR_DBM = -14.4
OPTIC_FLOORS = {OPTIC_TYPE: OPTIC_FLOOR_DBM}
"""The one optic type this estate runs, and the level its receiver stops working at, from the datasheet.

Supplied to `TC1ForecastStage` the way the estate's site coordinates are supplied to the travel stage: a fact about
the hardware rather than about any poll, so it is not carried on the event.
"""

DEGRADATION_DB_PER_MINUTE = 0.04
"""The optic on `MAINTENANCE_PORT` is failing: its receive level slides down by this much every poll until the swap
replaces it, which is why it was replaced, and which is what R-P-L1-004 fires on.

Forty-four minutes of it is 1.76 dB, and a line through the readings reaches the floor within hours. The tap on
`HUB_PORT` loses more light than that at once and must not fire the forecast, because a step is not a trend; and
the slide is kept shallower than what the baseline stage reports as a step, so the two signals stay distinct. The
degradation is the forecast's and the tap is the baseline's.
"""
BYPASS_AT_SECONDS = 1500
BYPASS_PORT = "Gi1/0/2"
BYPASS_MAC = "de:ad:be:ef:01:01"

BENCH_PORTS = {"Gi1/0/19": "aa:bb:cc:00:01:01", "Gi1/0/20": "aa:bb:cc:00:01:02"}
"""Two shared lab benches whose terminals reauthenticate every two minutes, which is what gives a port a distribution
of its own exchanges inside an hour. The desk ports reauthenticate every fifteen minutes and stop before the hour
ends, which the estate harness depends on: a sign-in after a desk's last exchange must resolve to no port. The
benches' identities are in no directory, so they take nobody down the ladder."""
BENCH_IDENTITIES = {"aa:bb:cc:00:01:01": "bench-01", "aa:bb:cc:00:01:02": "bench-02"}
BENCH_REAUTH_SECONDS = 120
BENCH_ELAPSED_SECONDS = {"Gi1/0/19": 3, "Gi1/0/20": 4}
"""How long each bench's exchanges take, every time but the two below."""
AUTH_BASELINE_MIN_SAMPLES = 20
"""Prior exchanges a port needs before its distribution is published. Twenty, inside an hour; the stage's default is
a hundred, and below a hundred the 99th percentile is the slowest exchange the port has had."""
SLOW_AUTH_PORT = "Gi1/0/19"
SLOW_AUTH_AT_SECONDS = 2700
SLOW_AUTH_ATTEMPTS = 3
"""On the first bench, the exchange at 2700 restarts twice, eight seconds apart, and succeeds nine seconds after
the third attempt: three times the slowest exchange the port has had, and three attempts to get there."""
FAST_AUTH_PORT = "Gi1/0/20"
FAST_AUTH_AT_SECONDS = 2940
"""On the second bench, the exchange at 2940 is accepted in the second it was requested, where every one before it
took four seconds."""

IDENTITIES = {
    MAC_A: "alice-ws",
    MAC_B: "bob-ws",
    MAC_C: "carol-ws",
    "aa:bb:cc:00:00:05": "phone-4021",
    "aa:bb:cc:00:00:06": "desk-4021",
    "de:ad:be:ef:01:01": "unknown-supplicant",
    "de:ad:be:ef:01:02": "unknown-supplicant",
}
"""The identity a supplicant presented, where it presented one. A device doing MAC authentication bypass has no
identity to present, which is what makes `unknown-supplicant` the honest value rather than a blank.

The three workstations present one because the ladder's next rung needs somewhere to stand. An 802.1X identity is
the only thing in this corpus a directory could bind a person to: a MAC is an asset, and asset assignment is a
different fact with a different lifetime. Falling back to the MAC left `dot1x_identity` carrying an address on
most exchanges, which is what the fallback is for and not what an estate with 802.1X deployed actually reports.
Nothing keys on it -- `TC2AuthStage` prefers `mac_address` when both are present, so the exchange key, the port
key and every window are untouched.
"""

AUTH_SUPPLICANTS = {PORTS[0]: MAC_A, PORTS[1]: MAC_B, PORTS[2]: MAC_C}
"""Which device authenticates on each port, matching the MAC table the same corpus reports.

The guide's TC-2 required-field list mandates a supplicant identifier, and without one `TC2AuthStage` falls back to
timing exchanges per port -- a documented degraded mode this corpus used to run in, which meant no composed check
ever exercised the keying that a shared port depends on."""

MULTI_DOMAIN_PORT = "Gi1/0/4"
PHONE_MAC = "aa:bb:cc:00:00:05"
DESK_MAC = "aa:bb:cc:00:00:06"
MULTI_DOMAIN_AT_SECONDS = 1200
"""A Cisco multi-domain access port: a phone with a workstation behind it, which is a standard configuration and
not an anomaly. Both authenticate and their outcomes arrive in the other order. Timed per port, the workstation's
accept closes the phone's exchange and the phone's own accept then has nothing to pair with, so R-D-L2-005 fires on
a legitimate device once per reauthentication."""

CONCURRENT_BYPASS_PORT = "Gi1/0/5"
CONCURRENT_BYPASS_MAC = "de:ad:be:ef:01:02"
LEGIT_SUPPLICANT_MAC = "aa:bb:cc:00:00:07"
CONCURRENT_BYPASS_AT_SECONDS = 2100
"""The harder bypass, and the one that matters more: a rogue authorized while a legitimate exchange on the same
port is still open. Timed per port it pairs with the innocent device's request, reads as an ordinary authorized
session, and the signal the rule exists for disappears. Neither of these ports is in `SINGLE_HOST_PORTS`, so they
say nothing to R-D-L2-001."""

SWEEP_OFFSET_SECONDS = 2
"""How long after the first switch the poller reaches the peer. This is the whole point of the second switch: the
sightings that make up a cross-switch spoof are this far apart, never simultaneous."""
CROSS_SPOOF_AT_SECONDS = 900
"""MAC_B is claimed on the peer switch while it is still live on its own port, inside a single sweep."""
ROAM_AT_SECONDS = 2100
"""ROAM_MAC changes port between two snapshots, so its two sightings are a full cadence apart."""

CS_PER_SECOND = 100

TELEMETRY_CLASSES = ("tc1", "tc1_binding", "tc2_mac", "tc2_binding", "tc2_arp", "tc2_auth")

CLASS_ENVELOPE = {
    "tc1": (1, ["site_id", "device_id", "port_id"]),
    "tc1_binding": (1, ["site_id", "device_id", "port_id"]),
    "tc2_mac": (2, ["mac_address"]),
    "tc2_binding": (2, ["mac_address"]),
    "tc2_arp": (2, ["arp_sender_mac"]),
    "tc2_auth": (2, ["site_id", "switch_id", "port_id"]),
}
"""The layer each class came from, and the columns Part 2 names as its behavioral subject.

Layer 1 keys on the port and layer 2 on the MAC, which is the guide's own split: at layer 1 the port is the
thing that persists, and at layer 2 the MAC is the thing that moves between ports. The 802.1X class is the
exception and keys on the port rather than the supplicant, because an exchange is a question about the port that
authorized it -- the same reasoning that gives `TC2AuthStage` its port-keyed timing.
"""

SETTINGS = {
    "period_seconds": PERIOD_SECONDS,
    "lateness_seconds": LATENESS_SECONDS,
    "baseline_min_buckets": BASELINE_MIN_BUCKETS,
    "optic_floors": OPTIC_FLOORS,
    "auth_baseline_min_samples": AUTH_BASELINE_MIN_SAMPLES,
}
"""The settings that decide this corpus's output, digested into `config_hash` by `stamping.envelope_for`."""
RULES = ("R-D-L1-001",
         "R-D-L1-002",
         "R-D-L1-003",
         "R-P-L1-004",
         "R-B-L1-005",
         "R-D-L1-006",
         "R-B-L1-007",
         "R-B-L2-002",
         "R-D-L2-001",
         "R-D-L2-003",
         "R-D-L2-004",
         "R-D-L2-005",
         "R-B-L2-006",
         "R-B-L2-007",
         "R-B-L2-008")
"""The shipped rules that read this corpus's columns; their thresholds are folded into `pipeline_fingerprint`."""


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
    extra = random.Random(EXTRA_SEED)

    return {
        "tc1": _build_layer_1(rng, extra),
        "tc2_mac": _build_mac_snapshots(rng, extra),
        "tc2_arp": _build_arp(rng),
        "tc2_auth": _build_auth(rng, extra),
    }


def _octets_per_minute(extra: random.Random, bits_per_second: float) -> int:
    """A minute of traffic at a rate, give or take a tenth, in octets."""
    return int(bits_per_second * 60 / 8 * extra.uniform(0.9, 1.1))


def _build_layer_1(rng: random.Random, extra: random.Random) -> pd.DataFrame:
    """Per-port SNMP polls at one-minute cadence, with a reboot, a tap, a failing optic, an unpolled flap, two optic
    swaps, a flapping link, a failing cable, a re-patch, an inline insertion, a surge and a silence.

    The ports and fields the first corpus had draw from `rng` exactly as they always did, so their values are
    unchanged; everything added since draws from `extra`.
    """
    original = [(SWITCH, port) for port in PORTS] + [(SWITCH, MAINTENANCE_PORT), (REBOOTING_SWITCH, "Gi1/0/1")]
    added = [(SWITCH, port) for port in (FLAPPER_PORT, ERROR_PORT, VOLUME_PORT, SILENT_PORT, REPATCH_PORT, INLINE_PORT)]
    added += [(REBOOTING_SWITCH, port) for port in REBOOT_PORTS if (REBOOTING_SWITCH, port) not in original]
    devices = sorted(original + added)
    error_names = ("crc_errors", "symbol_errors", "input_discards", "output_discards")
    counters = {
        key: {
            "crc_errors": 100, "symbol_errors": 0, "input_discards": 5, "output_discards": 1
        }
        for key in devices
    }

    for key in devices:
        counters[key]["if_hc_in_octets"] = 10**12 + extra.randrange(10**9)
        counters[key]["if_hc_out_octets"] = 10**12 + extra.randrange(10**9)

    boot_uptime_cs = 3600 * CS_PER_SECOND
    rows = []
    seq = 0

    for minute in range(0, CORPUS_SECONDS // 60 + 1):
        time_s = minute * 60

        for (device, port) in devices:
            draw = rng if (device, port) in original else extra
            rebooted = device == REBOOTING_SWITCH and minute >= REBOOT_AT_MINUTE

            if (device == REBOOTING_SWITCH and minute == REBOOT_AT_MINUTE):
                counters[(device, port)] = {name: 0 for name in counters[(device, port)]}

            for name in error_names:
                counters[(device, port)][name] += draw.choice([0, 0, 0, 1, 2])

            # The failing cable: thirty more CRC errors each minute than the minute before.
            if (device == SWITCH and port == ERROR_PORT and minute >= ERRORS_FROM_MINUTE):
                counters[(device, port)]["crc_errors"] += ERRORS_PER_MINUTE_STEP * (minute - ERRORS_FROM_MINUTE + 1)

            # Traffic, in both directions. The surge sends, and the silent port's link stays up carrying nothing.
            inbound = _octets_per_minute(extra, BASE_BITS_PER_SECOND)
            outbound = _octets_per_minute(extra, BASE_BITS_PER_SECOND / 2)

            if (device == SWITCH and port == VOLUME_PORT and minute >= VOLUME_FROM_MINUTE):
                outbound = _octets_per_minute(extra, SURGE_BITS_PER_SECOND)

            if (device == SWITCH and port == SILENT_PORT and minute >= VOLUME_FROM_MINUTE):
                (inbound, outbound) = (0, 0)

            if (minute > 0 and not (device == REBOOTING_SWITCH and minute == REBOOT_AT_MINUTE)):
                counters[(device, port)]["if_hc_in_octets"] += inbound
                counters[(device, port)]["if_hc_out_octets"] += outbound

            uptime_cs = (minute - REBOOT_AT_MINUTE) * 60 * CS_PER_SECOND + 30 * CS_PER_SECOND if rebooted else (
                boot_uptime_cs + minute * 60 * CS_PER_SECOND)

            # ifLastChange is relative to the device's own boot. The flap on Gi1/0/1 advances it inside one polling
            # gap while the state reads "up" both sides, which is the case only the device's own record can reveal.
            if (device == SWITCH and port == "Gi1/0/1" and minute >= FLAP_AT_MINUTE):
                last_change_cs = FLAP_AT_MINUTE * 60 * CS_PER_SECOND - 30 * CS_PER_SECOND
            elif (device == SWITCH and port == MAINTENANCE_PORT and minute >= MAINTENANCE_SWAP_AT_MINUTE):
                # The link came back up with the new optic, twenty seconds before the poll that first saw it.
                last_change_cs = MAINTENANCE_SWAP_AT_MINUTE * 60 * CS_PER_SECOND - 20 * CS_PER_SECOND
            elif (device == SWITCH and port == FLAPPER_PORT and minute >= FLAPPER_FROM_MINUTE):
                # Down and back up again in every polling gap: the last change is always twenty seconds ago.
                last_change_cs = minute * 60 * CS_PER_SECOND - 20 * CS_PER_SECOND
            elif (device == SWITCH and port == REPATCH_PORT and minute >= REPATCH_AT_MINUTE):
                last_change_cs = REPATCH_AT_MINUTE * 60 * CS_PER_SECOND - 25 * CS_PER_SECOND
            elif (device == SWITCH and port == INLINE_PORT and minute >= INLINE_AT_MINUTE):
                last_change_cs = INLINE_AT_MINUTE * 60 * CS_PER_SECOND - 15 * CS_PER_SECOND
            elif (rebooted):
                last_change_cs = 5 * CS_PER_SECOND
            else:
                last_change_cs = 10 * CS_PER_SECOND

            # The optic itself is replaced on two ports, which closes each port's binding and opens the next. On
            # one of them the device recorded the link dropping for the swap, above; on the other it did not.
            serial = f"XCVR-{device}-{port}"

            if (device == SWITCH and port in SWAPS and minute >= SWAPS[port]):
                serial = f"{serial}-B"

            rx_dbm = -7.0 + draw.uniform(-0.05, 0.05)

            if (device == SWITCH and port == HUB_PORT and minute >= TAP_AT_MINUTE):
                rx_dbm -= TAP_LOSS_DB

            # The failing optic loses a little more light every poll until it is replaced; its replacement is healthy.
            if (device == SWITCH and port == MAINTENANCE_PORT and minute < MAINTENANCE_SWAP_AT_MINUTE):
                rx_dbm -= DEGRADATION_DB_PER_MINUTE * minute

            # Who is on the other end. The two uplinks that change, change once each; a switch that has just come
            # back from a reboot has not heard from its neighbours yet on its first poll.
            (chassis, neighbour_port) = NEIGHBOURS.get(port, (f"nbr-{device}-{port}", "Gi0/1")) if device == SWITCH \
                else (f"nbr-{device}-{port}", "Gi0/1")

            if (device == SWITCH and port == REPATCH_PORT and minute >= REPATCH_AT_MINUTE):
                (chassis, neighbour_port) = REPATCHED_TO

            if (device == SWITCH and port == INLINE_PORT and minute >= INLINE_AT_MINUTE):
                (chassis, neighbour_port) = INLINE_DEVICE

            if (device == REBOOTING_SWITCH and minute == REBOOT_AT_MINUTE):
                (chassis, neighbour_port) = (None, None)

            tx_dbm = round(-2.0 + draw.uniform(-0.05, 0.05), 3)
            seq += 1
            rows.append({
                "event_time": time_s * NS_PER_SECOND,
                "site_id": SITE,
                "device_id": device,
                "port_id": port,
                "uptime": uptime_cs,
                "if_last_change": last_change_cs,
                "oper_status": "up",
                "link_speed_bps": LINK_SPEED_BPS,
                "optical_tx_dbm": tx_dbm,
                "optical_rx_dbm": round(rx_dbm, 3),
                "transceiver_serial": serial,
                "transceiver_type": OPTIC_TYPE,
                "lldp_neighbor_chassis_id": chassis,
                "lldp_neighbor_port_id": neighbour_port,
                **counters[(device, port)],
                **_envelope(draw, "snmp-poller", "TC-1/1.0.0", seq),
            })

    return pd.DataFrame(rows)


def _mac_row(time_s: int, mac: str, switch: str, port: str, vlan: int, envelope: dict, action: str = None) -> dict:
    return {
        "event_time": time_s * NS_PER_SECOND,
        "mac_address": mac,
        "site_id": SITE,
        "switch_id": switch,
        "port_id": port,
        "vlan_id": vlan,
        MAC_ACTION_COLUMN: action,
        **envelope,
    }


def _build_mac_snapshots(rng: random.Random, extra: random.Random) -> pd.DataFrame:
    """A full MAC table snapshot every five minutes, every row stamped at the snapshot's instant, and between two of
    them the switch's own notification that one address was removed.

    The entries the first corpus had draw their envelopes from `rng` as they always did; the second VLAN, the
    printer and the guest draw from `extra`.
    """
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
            rows.append(_mac_row(time_s, mac, SWITCH, port, VLAN, _envelope(rng, "mac-table", "TC-2/1.0.0", seq)))

        # The same walk of the same switch, reaching the entries planted since: the printer until it is unplugged,
        # the guest while they are here, and the camera VLAN, whose vendor mix the single-board computer changes.
        added = [(mac, port, IOT_VLAN) for (mac, port) in CAMERAS.items()]

        if (time_s <= PRINTER_LAST_SEEN_SECONDS):
            added.append((PRINTER_MAC, PRINTER_PORT, VLAN))

        if (GUEST_FROM_SECONDS <= time_s < GUEST_LEAVES_AT_SECONDS):
            added.append((GUEST_MAC, GUEST_PORT, VLAN))

        if (time_s >= ROGUE_OUI_AT_SECONDS):
            added.append((ROGUE_OUI_MAC, ROGUE_OUI_PORT, IOT_VLAN))

        if (time_s >= THIRD_CAMERA_AT_SECONDS):
            added.append((THIRD_CAMERA_MAC, THIRD_CAMERA_PORT, IOT_VLAN))

        for (mac, port, vlan) in added:
            seq += 1
            rows.append(_mac_row(time_s, mac, SWITCH, port, vlan, _envelope(extra, "mac-table", "TC-2/1.0.0", seq)))

        # The poller reaches the peer switch a couple of seconds after the first, which is what makes the spoof
        # below `displaced` with a small gap rather than a `conflict`. Emitted here, inside the same snapshot, so
        # the frame stays monotonic in event time the way a collector's stream is: a frame that jumped backwards
        # would make the output depend on where the batch boundaries fell, which control 13 checks for.
        for (mac, port) in ([(ROAM_MAC, PEER_PORTS[2] if time_s >= ROAM_AT_SECONDS else PEER_PORTS[1])] +
                            ([(MAC_B, PEER_PORTS[0])] if time_s == CROSS_SPOOF_AT_SECONDS else [])):
            seq += 1
            rows.append(
                _mac_row(time_s + SWEEP_OFFSET_SECONDS,
                         mac,
                         PEER_SWITCH,
                         port,
                         VLAN,
                         _envelope(rng, "mac-table", "TC-2/1.0.0", seq)))

        # The guest's laptop leaves between two walks, and the switch says so: a MAC notification of the removal.
        if (time_s < GUEST_LEAVES_AT_SECONDS < time_s + PERIOD_SECONDS):
            seq += 1
            rows.append(
                _mac_row(GUEST_LEAVES_AT_SECONDS,
                         GUEST_MAC,
                         SWITCH,
                         GUEST_PORT,
                         VLAN,
                         _envelope(extra, "mac-table", "TC-2/1.0.0", seq),
                         action="removed"))

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


def _build_auth(rng: random.Random, extra: random.Random) -> pd.DataFrame:
    """
    802.1X exchanges, every one naming the device being authorized.

    Five shapes: the routine exchange on a single-host port, one authorization nothing preceded, a multi-domain
    port carrying two supplicants whose outcomes interleave, a bypass that lands while a legitimate exchange on
    the same port is still open, and two benches reauthenticating every two minutes. The multi-domain port and the
    bypass are the cases where timing an exchange per port rather than per device is wrong in each direction -- a
    false positive on the phone, and a false negative on the rogue. The benches are the ones with enough exchanges
    for a distribution: one bench's exchange takes three attempts and three times as long, and the other's no time
    at all.
    """
    events: list[tuple[int, str, str, str]] = []

    for (index, port) in enumerate(PORTS):
        supplicant = AUTH_SUPPLICANTS[port]

        for time_s in range(60 + index * 7, CORPUS_SECONDS, 900):
            events.append((time_s, port, "started", supplicant))
            events.append((time_s + 3 + index, port, "success", supplicant))

    # The benches, every two minutes from the first minute. Two of their exchanges are not routine.
    for (port, supplicant) in BENCH_PORTS.items():
        for time_s in range(60, CORPUS_SECONDS, BENCH_REAUTH_SECONDS):
            if (port == SLOW_AUTH_PORT and time_s == SLOW_AUTH_AT_SECONDS):
                # Two restarts eight seconds apart, then a success nine seconds after the third attempt.
                for attempt in range(SLOW_AUTH_ATTEMPTS):
                    events.append((time_s + 8 * attempt, port, "started", supplicant))

                events.append((time_s + 8 * (SLOW_AUTH_ATTEMPTS - 1) + 9, port, "success", supplicant))
                continue

            if (port == FAST_AUTH_PORT and time_s == FAST_AUTH_AT_SECONDS):
                # Accepted in the second it was requested.
                events.append((time_s, port, "started", supplicant))
                events.append((time_s, port, "success", supplicant))
                continue

            events.append((time_s, port, "started", supplicant))
            events.append((time_s + BENCH_ELAPSED_SECONDS[port], port, "success", supplicant))

    # An authorization with nothing in front of it, on a port whose own exchanges are long finished.
    events.append((BYPASS_AT_SECONDS, BYPASS_PORT, "success", BYPASS_MAC))

    # The phone authenticates first and is authorized last, because the workstation behind it answered quicker.
    events.append((MULTI_DOMAIN_AT_SECONDS, MULTI_DOMAIN_PORT, "started", PHONE_MAC))
    events.append((MULTI_DOMAIN_AT_SECONDS + 1, MULTI_DOMAIN_PORT, "started", DESK_MAC))
    events.append((MULTI_DOMAIN_AT_SECONDS + 11, MULTI_DOMAIN_PORT, "success", DESK_MAC))
    events.append((MULTI_DOMAIN_AT_SECONDS + 12, MULTI_DOMAIN_PORT, "success", PHONE_MAC))

    # The rogue is authorized mid-exchange; the legitimate device is authorized afterwards and must still be timed
    # against its own request rather than against whatever happened in between.
    events.append((CONCURRENT_BYPASS_AT_SECONDS, CONCURRENT_BYPASS_PORT, "started", LEGIT_SUPPLICANT_MAC))
    events.append((CONCURRENT_BYPASS_AT_SECONDS + 10, CONCURRENT_BYPASS_PORT, "success", CONCURRENT_BYPASS_MAC))
    events.append((CONCURRENT_BYPASS_AT_SECONDS + 20, CONCURRENT_BYPASS_PORT, "success", LEGIT_SUPPLICANT_MAC))

    events.sort(key=lambda event: event[0])
    rows = []

    for (seq, (time_s, port, result, supplicant)) in enumerate(events, start=1):
        rows.append({
            "event_time": time_s * NS_PER_SECOND,
            "site_id": SITE,
            "switch_id": SWITCH,
            "port_id": port,
            "mac_address": supplicant,
            # The identity half of the supplicant. R-D-L2-005 names it on the alert, and the stage prefers it to
            # the MAC when both are present, so a corpus without one leaves both the fallback and the alert's
            # identity column uncovered.
            "dot1x_identity": IDENTITIES.get(supplicant, BENCH_IDENTITIES.get(supplicant, supplicant)),
            "dot1x_result": result,
            # The benches came later, and draw from the second generator so the rows before them keep theirs.
            **_envelope(extra if supplicant in BENCH_IDENTITIES else rng, "radius", "TC-2/1.0.0", seq),
        })

    return pd.DataFrame(rows)


def build_pipeline_config(execution_mode=None) -> Config:
    """
    A pipeline configuration, defaulting to CPU mode and importable without a GPU.

    Parameters
    ----------
    execution_mode : `morpheus.config.ExecutionMode`, optional
        Mode to build for. Defaults to CPU, which is what every check in this harness has ever run in: the golden
        is a CPU artifact and control 13 has only ever been asserted there. The parameter exists so the same
        corpus and the same golden can be driven in GPU mode and the two compared, since the per-stage `gpu_mode`
        variants are unit tests and none of them composes the pipeline.

        Resolved inside the function rather than as a default argument so that importing this module still does
        not require a GPU.
    """
    from morpheus.config import CppConfig
    from morpheus.config import ExecutionMode

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.execution_mode = ExecutionMode.CPU if execution_mode is None else execution_mode

    return config


def _collect(sink: InMemorySinkStage) -> pd.DataFrame:
    # Sibling module; imported here because this file's own directory is put on the path by whoever imports it.
    from host_frame import to_host_frame

    frames = []

    for message in sink.get_messages():
        meta = message.payload() if isinstance(message, ControlMessage) else message
        df = meta.copy_dataframe()
        frames.append(to_host_frame(df))

    if (len(frames) == 0):
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


CHAIN_ROOTS = {
    "tc1": ["entity_key"],
    "tc2_mac": ["port_key"],
    "tc2_arp": ["resolved_port_key", "arp_sender_ip"],
    "tc2_auth": ["auth_port_key"],
}
"""What each class's correlation chain is rooted on, as candidates in order of preference, decided per row.

A layer 1 sample is about a port, so its chain is rooted on the port's `entity_key`. A MAC table row and an
802.1X exchange each name the port they were observed on directly, so they root on it too. An ARP observation
names only the address being claimed -- unless the ladder resolved its MAC to a port, in which case the
observation is about that port and joins the port's chain, with `chain_anchor_source` recording that it got
there through the binding table. An unresolved one stays rooted on the address, which is what it was rooted on
before and is still true. The binding classes have no entry because they are not sealed into windows at all.

These four classes are sealed together, in one pass over their union in event-time order, because a chain is a
Merkle root over its members and the members of a port's chain now come from two layers. Sealing each class on
its own would give the same root key two different roots, one per class, and nothing downstream could tell that
they were the same chain."""


def _run_class(config: Config,
               dataframes: list[pd.DataFrame],
               stages: list,
               impose_order: bool,
               seal: bool = True,
               anchor: str = None,
               envelope: tuple = None,
               chain: list[str] = None,
               telemetry_class: str = None) -> pd.DataFrame:
    """Source → stamp → (total order) → the class's stages → determinism stamp → (envelope) → (chain anchor) →
    (window seal) → sink.

    The envelope stamp goes after the class's own stages rather than before them, because two of these classes
    are produced by stages that replace the payload wholesale: a binding stage emits one row per interval, not
    one per observation, so anything stamped upstream of it is discarded along with the observations.

    A class given `chain` has its root chosen per row here and is sealed later, in `_seal_chains`, together with
    every other chained class. `anchor` is the older per-class form and seals the class on its own; the two are
    not combined.
    """
    if (chain is not None and anchor is not None):
        raise ValueError("a class is chained per row or anchored per class, not both")

    if (chain is not None and seal):
        raise ValueError("a chained class is sealed with the other chained classes, not on its own")

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    for stage in stages:
        pipe.add_stage(stage)

    if (telemetry_class is None):
        raise ValueError("every class is stamped with the determinism envelope, and the envelope names the class")

    pipe.add_stage(DeterminismStampStage(config, envelope=stamping.envelope_for(telemetry_class, SETTINGS,
                                                                                rules=RULES)))

    if (envelope is not None):
        (osi_layer, entity_columns) = envelope
        pipe.add_stage(EnvelopeStampStage(config, osi_layer=osi_layer, entity_columns=entity_columns))

    if (chain is not None):
        pipe.add_stage(ChainAnchorStage(config, candidates=chain))

    if (seal):
        pipe.add_stage(
            WindowSealStage(config,
                            period_seconds=PERIOD_SECONDS,
                            lateness_seconds=LATENESS_SECONDS,
                            order_columns=list(DEFAULT_ORDER_COLUMNS),
                            entity_key_column=anchor))

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    return _collect(sink)


def _seal_chains(config: Config, outputs: dict[str, pd.DataFrame], parts: int = 1) -> dict[str, pd.DataFrame]:
    """
    Seal the chained classes together, so a port's chain holds its events from every layer that reached it.

    The classes are tagged, aligned to one column set, concatenated, and put in event-time order -- the order a
    deployment would see them in, where a layer 1 sample and the layer 2 observations on the same port arrive
    interleaved rather than one whole class after another. Fed class by class instead, the watermark would advance
    through the first class and declare every row of the second late. `parts` splits that ordered union into
    contiguous source frames so the batch-split sweep exercises this pass as well as the per-class ones.

    Each class comes back with its own columns plus the ones sealing adds, so nothing about a class's shape depends
    on which other classes it was sealed beside.
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


def build_binding_table(bindings: pd.DataFrame) -> BindingTable:
    """The layer 2 bindings as the resolver consumes them: MAC to the port as layer 1 spells it."""
    return BindingTable.from_dataframe(bindings,
                                       name="mac_table",
                                       key_column="mac_address",
                                       value_columns=["port_key", "vlan_id"],
                                       start_column="bind_start",
                                       end_column="bind_end")


PORT_INVENTORY_TABLE = "port_inventory"
"""What the layer 1 binding source calls itself on the shared bucketed sourcetype.

Several binding sources land on `binding:bucketed` and the refresh searches tell them apart with
`binding_table=...`, so this string is the one the app's L1 history refresh selects on.
"""

PORT_INVENTORY_COLUMNS = ["port_id", "switch_id", "site_id", "transceiver_serial", "lldp_neighbor_chassis_id"]
"""What the `binding_l1` lookups return, in the order the app's `fields_list` names them.

`port_id` and `switch_id` are values here rather than the key because the key is the composed `entity_key` and a
Splunk lookup matches on the two columns separately -- the layer 2 lookup hands them on individually, and nothing
on the search head composes them back into a port key.
"""


def build_port_binding_table(port_bindings: pd.DataFrame) -> BindingTable:
    """The layer 1 bindings as an interval table: the port, to the site, optic and neighbour it held.

    The inverse of the layer 2 table above. There the key is the mobile thing and the port is what it resolves to;
    here the port is the key and the optic is what moves.
    """
    return BindingTable.from_dataframe(port_bindings,
                                       name=PORT_INVENTORY_TABLE,
                                       key_column="entity_key",
                                       value_columns=list(PORT_INVENTORY_COLUMNS),
                                       start_column="bind_start",
                                       end_column="bind_end")


def run_classes(config: Config,
                corpus: dict[str, pd.DataFrame],
                batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                impose_order: bool = True) -> dict[str, pd.DataFrame]:
    """
    Run every telemetry class through its own pipeline, unsealed, and return the outputs per class.

    Split out of `run_pipeline` so that a harness composing more layers than this one can add its classes to the
    union before it is sealed. Sealing is where a chain is decided, and a class sealed after the others can never
    join their chains, so the seam has to be here rather than after.

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
    dict
        Class name to its output frame, with `row_key` already set on the classes whose key is not `event_uid`.
        None of them carry window columns yet.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    outputs = {}

    # Layer 1: counters, the rates they become and the port's history of them, optics, flaps, identifier changes.
    # The rate history is kept over the sealing period with the corpus's floor, as the layer 2 baseline is; the
    # volume history is kept per hour of the day, which an hour-long corpus exercises without contradicting.
    outputs["tc1"] = _run_class(
        config,
        batches["tc1"],
        [
            TC1NormalizeStage(config, uptime_column="uptime", uptime_unit="cs"),
            TC1RateStage(config, bucket_seconds=PERIOD_SECONDS, min_buckets=BASELINE_MIN_BUCKETS),
            TC1OpticalStage(config),
            TC1ForecastStage(config, floors=OPTIC_FLOORS),
            TC1FlapStage(config, last_change_column="if_last_change", last_change_unit="cs"),
            TC1ChangeStage(config),
        ],
        impose_order,
        seal=False,
        chain=CHAIN_ROOTS["tc1"],
        envelope=CLASS_ENVELOPE["tc1"],
        telemetry_class="tc1")

    # The same layer 1 snapshots, closed into port bindings. This is the ladder's last rung: the table that takes
    # a switch port to the site, optic and neighbour it held at a given moment. The stage emits its own
    # `binding_uid`, built exactly as `binding_table` builds one, so it serves as the row key directly rather than
    # having a second identifier composed beside it the way the layer 2 bindings do.
    port_bindings = _run_class(config,
                               batches["tc1"], [TC1BindingStage(config)],
                               impose_order,
                               seal=False,
                               envelope=CLASS_ENVELOPE["tc1_binding"],
                               telemetry_class="tc1_binding")
    port_bindings["row_key"] = port_bindings["binding_uid"]
    outputs["tc1_binding"] = port_bindings

    # Layer 2, from the same snapshots: the cardinality features, each port's count and each VLAN's vendor count
    # against the peaks of their own earlier periods, and the closed bindings.
    outputs["tc2_mac"] = _run_class(
        config,
        batches["tc2_mac"],
        [
            TC2CardinalityStage(config),
            TC2BaselineStage(config, bucket_seconds=PERIOD_SECONDS, min_buckets=BASELINE_MIN_BUCKETS),
            TC2BaselineStage(config,
                             entity_column="vlan_key",
                             value_column="ouis_per_vlan",
                             bucket_seconds=PERIOD_SECONDS,
                             min_buckets=BASELINE_MIN_BUCKETS),
        ],
        impose_order,
        seal=False,
        chain=CHAIN_ROOTS["tc2_mac"],
        envelope=CLASS_ENVELOPE["tc2_mac"],
        telemetry_class="tc2_mac")
    # The table walk is a snapshot of each switch, and the switch's own removal notice is a stop: both ends a
    # source reports, beside the ones the closer infers.
    bindings = _run_class(
        config,
        batches["tc2_mac"],
        [TC2BindingStage(config, action_column=MAC_ACTION_COLUMN, snapshot_scope_columns=["site_id", "switch_id"])],
        impose_order,
        seal=False,
        envelope=CLASS_ENVELOPE["tc2_binding"],
        telemetry_class="tc2_binding")

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
        impose_order,
        seal=False,
        chain=CHAIN_ROOTS["tc2_arp"],
        envelope=CLASS_ENVELOPE["tc2_arp"],
        telemetry_class="tc2_arp")

    outputs["tc2_auth"] = _run_class(config,
                                     batches["tc2_auth"],
                                     [TC2AuthStage(config, baseline_min_samples=AUTH_BASELINE_MIN_SAMPLES)],
                                     impose_order,
                                     seal=False,
                                     chain=CHAIN_ROOTS["tc2_auth"],
                                     envelope=CLASS_ENVELOPE["tc2_auth"],
                                     telemetry_class="tc2_auth")

    return outputs


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True) -> pd.DataFrame:
    """
    Run every telemetry class, seal the chained ones together, and return one canonicalized frame.

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

    outputs = run_classes(config, corpus, batches, impose_order)

    # The chained classes are sealed together, in as many contiguous pieces as the widest per-class split, so
    # control 5 sweeps this pass too.
    parts = max(len(batches[name]) for name in CHAIN_ROOTS)
    outputs.update(_seal_chains(config, {name: outputs[name] for name in CHAIN_ROOTS}, parts=parts))

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
