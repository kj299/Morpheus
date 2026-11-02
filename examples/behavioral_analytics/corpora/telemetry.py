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
The synthetic corpus `morpheus.compositions.telemetry` is tested over, and the facts about it the harnesses assert.

Moved out of `tests/morpheus/determinism/telemetry_pipeline.py` under plan section 4.5, which put the composition in the
package and left the data here, where the tests, the sample and fixture generators and the experiments read it. The
composition module's docstring describes what this corpus plants and why.
"""

import hashlib
import random

import pandas as pd

from morpheus.compositions import envelope as universal_envelope
from morpheus.compositions.telemetry import CORPUS_SECONDS
from morpheus.compositions.telemetry import DEVICE_CLASS
from morpheus.compositions.telemetry import IDENTITIES
from morpheus.compositions.telemetry import MAC_A
from morpheus.compositions.telemetry import MAC_ACTION_COLUMN
from morpheus.compositions.telemetry import MAC_B
from morpheus.compositions.telemetry import MAC_C
from morpheus.compositions.telemetry import NETWORK_CONTEXT_CLASS
from morpheus.compositions.telemetry import OPTIC_TYPE
from morpheus.compositions.telemetry import PERIOD_SECONDS
from morpheus.compositions.telemetry import VRRP_IP
from morpheus.stages.telemetry.tc0_network_stage import DEVICE as DEVICE_KIND
from morpheus.stages.telemetry.tc0_network_stage import PORT as PORT_KIND
from morpheus.stages.telemetry.tc0_network_stage import VLAN as VLAN_KIND
from morpheus.utils import bitemporal
from morpheus.utils.binding_table import NS_PER_SECOND

CORPUS_SEED = 20260902
EXTRA_SEED = 20261010
"""The second generator, for every row and field added after the first corpus. See the module docstring."""
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
ROAM_MAC = "aa:bb:cc:00:00:04"
"""A device that legitimately moves ports. The negative control for R-D-L2-004: it is displaced like a spoof, but a
whole poll cadence separates the two sightings, so the gap says it moved rather than that it was in two places."""
HUB_MACS = [f"de:ad:be:ef:00:{index:02x}" for index in range(1, 5)]
ROUTER_MAC = "00:00:5e:00:01:01"
GATEWAY_IP = "10.0.0.1"
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
SINGLE_HOST_PORTS = {f"{SITE}:{SWITCH}:{port}" for port in PORTS}
"""The corpus's own port designations, standing in for the inventory-supplied list R-D-L2-001 reads."""
HOST_IPS = {MAC_A: "10.0.0.11", MAC_B: "10.0.0.12", MAC_C: "10.0.0.13"}
HUB_PORT = "Gi1/0/3"
HUB_FROM_SECONDS = 1800
SPOOF_AT_SECONDS = 2700
FLOOD_AT_SECONDS = 2400
FLOOD_PACKETS = 20
DESK_SPOOF_AT_SECONDS = 2760
DESK_SPOOF_PACKETS = 5
"""The workstation on the tapped port claims the gateway in the window after the tap: its port's chain then carries
R-D-L1-002's 50 and R-D-L2-003's 55 together, which is what `Chain assembly - cross-layer risk` sums."""
ARP_LOSS_AT_SECONDS = 1205
LOST_ARP_RECORDS = 3
"""The ARP sensor numbers three records it never delivers, a third of the way through the hour."""
QUARANTINED_POLLER = "snmp-poller-dmz"
QUARANTINED_SWITCH = "sw3"
QUARANTINED_PORT = "Gi1/0/1"
QUARANTINED_OFFSET_MS = 4500
QUARANTINED_MINUTES = (20, 21, 22)
"""A second poller whose NTP has drifted four and a half seconds, more than `EnvelopeValidateStage` allows, polls one
port three times. Its records are kept and reach no stateful stage."""
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
RENAMED_PORT = ERROR_PORT
RENAMED_TO = "uplink-core-2"
RENAME_AT_MINUTE = 35
"""The failing cable's port is renamed at minute thirty-five, ten minutes before the cable starts to fail. An
error-rate history is mature after `BASELINE_MIN_BUCKETS` five-minute periods, half an hour, so keyed on its reported
name the renamed port would start again and still be immature when the hour ends, and R-B-L1-005 would never fire on
it. Keyed on its `ifIndex`, which the rename does not change, it is the same port with the same history (gap G39)."""


def if_index(device: str, port: str) -> int:
    """The `ifIndex` a port reports: fixed per device and port as first named, the way a persisted index is."""
    return 10000 + int(hashlib.sha256(f"{device}|{port}".encode()).hexdigest()[:6], 16) % 90000


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
SLOW_AUTH_PORT = "Gi1/0/19"
SLOW_AUTH_AT_SECONDS = 2700
SLOW_AUTH_ATTEMPTS = 3
"""On the first bench, the exchange at 2700 restarts twice, eight seconds apart, and succeeds nine seconds after
the third attempt: three times the slowest exchange the port has had, and three attempts to get there."""
FAST_AUTH_PORT = "Gi1/0/20"
FAST_AUTH_AT_SECONDS = 2940
"""On the second bench, the exchange at 2940 is accepted in the second it was requested, where every one before it
took four seconds."""
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
DEVICE_POLLER = "snmp-poller"
"""The class a device's own poll arrives as, and the poller that sends it: `sysUpTime`, a digest of the running
configuration, the routing table's size and the health gauges, once a minute per switch or router. The poller is the
one that polls the ports, numbering the device polls in a sequence of their own, so the estate's collectors are the
ones the clock skew experiment already spreads by name."""
STORM_SWITCH = "sw5"
ROUTER = "rtr1"
DEVICES = [SWITCH, REBOOTING_SWITCH, PEER_SWITCH, STORM_SWITCH, ROUTER]
PUSH_AT_MINUTE = 2
"""`SWITCH`'s configuration is pushed inside the window: a change, and a scheduled one."""
OUT_OF_WINDOW_CHANGE_AT_MINUTE = 40
"""`SWITCH`'s configuration changes again, outside the window, which is what R-D-L1-008 fires on."""
STORM_REBOOT_MINUTES = (36, 40, 43, 45, 47, 49)
"""When the storm switch restarts: once in the period from minute 35, twice in the one from 40, three times in the
one from 45. Each period holds more than the one before, three in a row, which is the trajectory R-P-L1-009 reads;
`REBOOTING_SWITCH` restarts once and holds a run of one."""
ROUTE_DROP_AT_MINUTE = 50
ROUTES_BEFORE = 1200
ROUTES_AFTER = 400
BGP_PEERS_BEFORE = 4
BGP_PEERS_AFTER = 2
"""The router loses two of its four BGP peers, and the routes they carried, at minute fifty."""
COMMISSIONING_RX_DBM = -7.0
"""The receive level every port's link was commissioned at, which is where an optic in this corpus sits when it is
healthy."""
COMMISSIONING_MARGIN_DB = 1.5
"""How far below its commissioning level a port's receive light may fall before R-D-L1-010 reads it as lost light.

The failing optic on `MAINTENANCE_PORT` crosses it at minute thirty-nine, six minutes before it is replaced; the
tap on `HUB_PORT` crosses it at once, at minute forty. Every other port sits within a tenth of a decibel of its
commissioning level. The degradation's 1.76 dB is what `TC1OpticalStage`'s median cannot see, because the median
slides down with it: R-D-L1-002 stays quiet on that port."""
UNRECORDED_PORT = SILENT_PORT
"""A port the inventory never recorded, so its rows carry `ctx_port_found` false and no commissioning level, and
R-D-L1-010 cannot read it, rather than reading it against a guess."""
DEVICE_ROLES = {
    SWITCH: "access-switch",
    REBOOTING_SWITCH: "access-switch",
    PEER_SWITCH: "access-switch",
    STORM_SWITCH: "access-switch",
    ROUTER: "router"
}
"""What each device is, as the inventory records it."""
VLAN_DESIGNATIONS = {VLAN: "desks", IOT_VLAN: "cameras"}
"""What each VLAN carries, as the inventory records it."""
INVENTORY_RECORDED_NS = 0
"""When the inventory recorded the estate as commissioned: the corpus's first instant, which is as early as any of its
records may be dated, since the search-head run moves every event forward from the epoch."""
TELEMETRY_CLASSES = ("tc1",
                     "tc1_binding",
                     DEVICE_CLASS,
                     "tc2_mac",
                     "tc2_binding",
                     "tc2_arp",
                     "tc2_auth",
                     NETWORK_CONTEXT_CLASS)


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

    corpus = {
        "tc1": _build_layer_1(rng, extra),
        DEVICE_CLASS: _build_devices(),
        "tc2_mac": _build_mac_snapshots(rng, extra),
        "tc2_arp": _build_arp(rng),
        "tc2_auth": _build_auth(rng, extra),
    }

    # The rest of the envelope each collector owes: when its sensor saw the record, its clock, its sampling, the
    # tenant. Added after the records are drawn, so it moves none of them.
    corpus = {name: universal_envelope.attach(frame) for (name, frame) in corpus.items()}

    # A context record is bitemporal: its valid time and its recording time are its clock, and it has no sensor or
    # sampling for the collector envelope to describe, as `context_pipeline` says of the other TC-0 kinds. What it
    # shares with every other record is the tenant it belongs to.
    network = _build_network_context()
    network["tenant_id"] = universal_envelope.TENANT
    corpus[NETWORK_CONTEXT_CLASS] = network

    return corpus


def _network_record(kind: str, valid_from_ns: int, recorded_ns: int, **values) -> dict:
    row = {
        "object_kind": kind,
        "site_id": SITE,
        "device_id": None,
        "port_id": None,
        "vlan_id": None,
        "commissioning_rx_dbm": None,
        "designation": None,
        "device_role": None,
        "owner": None,
        "owning_team": None,
        "criticality": None,
        bitemporal.VALID_FROM: valid_from_ns,
        bitemporal.VALID_TO: None,
        bitemporal.RECORDED_AT: recorded_ns,
        bitemporal.CHANGE: bitemporal.ASSERT,
    }
    row.update(values)

    return row


def _build_network_context() -> pd.DataFrame:
    """The network inventory: every recorded port at its commissioning level, every device and both VLANs.

    The replaced optic on `MAINTENANCE_PORT` was recommissioned when it went in, at the level it was measured at,
    which is a second version of the port's record from the minute of the swap. The substituted optic on
    `XCVR_SWAP_PORT` was not, because nobody knew it had been changed; the port keeps its first record.
    """
    commissioned = INVENTORY_RECORDED_NS
    rows = []
    ports = [(SWITCH, port) for port in PORTS + [MAINTENANCE_PORT]]
    ports += [(SWITCH, port) for port in (FLAPPER_PORT, ERROR_PORT, VOLUME_PORT, REPATCH_PORT, INLINE_PORT)]
    ports += [(REBOOTING_SWITCH, port) for port in REBOOT_PORTS]

    for (device, port) in sorted(set(ports)):
        designation = "uplink" if port in NEIGHBOURS and device == SWITCH else "access"
        rows.append(
            _network_record(PORT_KIND,
                            commissioned,
                            commissioned,
                            device_id=device,
                            port_id=port,
                            commissioning_rx_dbm=COMMISSIONING_RX_DBM,
                            designation=designation,
                            owner="netops",
                            criticality="high" if designation == "uplink" else "medium"))

    swap_ns = MAINTENANCE_SWAP_AT_MINUTE * 60 * NS_PER_SECOND
    rows.append(
        _network_record(PORT_KIND,
                        swap_ns,
                        swap_ns + 60 * NS_PER_SECOND,
                        device_id=SWITCH,
                        port_id=MAINTENANCE_PORT,
                        commissioning_rx_dbm=COMMISSIONING_RX_DBM,
                        designation="access",
                        owner="netops",
                        criticality="medium"))

    for device in DEVICES:
        rows.append(
            _network_record(DEVICE_KIND,
                            commissioned,
                            commissioned,
                            device_id=device,
                            device_role=DEVICE_ROLES[device],
                            owner="netops",
                            owning_team="network-engineering",
                            criticality="high" if device == ROUTER else "medium"))

    for (vlan, designation) in sorted(VLAN_DESIGNATIONS.items()):
        rows.append(
            _network_record(VLAN_KIND,
                            commissioned,
                            commissioned,
                            vlan_id=vlan,
                            designation=designation,
                            owner="facilities" if designation == "cameras" else "it-desktop",
                            criticality="medium"))

    frame = pd.DataFrame(rows)
    frame[bitemporal.VALID_TO] = frame[bitemporal.VALID_TO].astype("Int64")
    frame["vlan_id"] = frame["vlan_id"].astype("Int64")
    frame = frame.sort_values([bitemporal.RECORDED_AT], kind="mergesort").reset_index(drop=True)
    frame.insert(0, "source_seq", range(len(frame)))

    return frame


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
            # The rename changes what the port is called and nothing the hardware reports about itself.
            reported_port = RENAMED_TO if (device == SWITCH and port == RENAMED_PORT
                                           and minute >= RENAME_AT_MINUTE) else port
            seq += 1
            rows.append({
                "event_time": time_s * NS_PER_SECOND,
                "site_id": SITE,
                "device_id": device,
                "port_id": reported_port,
                "if_index": if_index(device, port),
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

    # The DMZ poller. Fixed values rather than draws, so planting it moves nothing else; it carries its own clock
    # fields because they are what is wrong with it.
    for (index, minute) in enumerate(QUARANTINED_MINUTES, start=1):
        rows.append({
            "event_time": minute * 60 * NS_PER_SECOND,
            "site_id": SITE,
            "device_id": QUARANTINED_SWITCH,
            "port_id": QUARANTINED_PORT,
            "if_index": if_index(QUARANTINED_SWITCH, QUARANTINED_PORT),
            "uptime": boot_uptime_cs + minute * 60 * CS_PER_SECOND,
            "if_last_change": 10 * CS_PER_SECOND,
            "oper_status": "up",
            "link_speed_bps": LINK_SPEED_BPS,
            "optical_tx_dbm": -2.0,
            "optical_rx_dbm": -7.0,
            "transceiver_serial": f"XCVR-{QUARANTINED_SWITCH}-{QUARANTINED_PORT}",
            "transceiver_type": OPTIC_TYPE,
            "lldp_neighbor_chassis_id": f"nbr-{QUARANTINED_SWITCH}-{QUARANTINED_PORT}",
            "lldp_neighbor_port_id": "Gi0/1",
            "crc_errors": 100,
            "symbol_errors": 0,
            "input_discards": 5,
            "output_discards": 1,
            "if_hc_in_octets": 10**12,
            "if_hc_out_octets": 10**12,
            "collector_id": QUARANTINED_POLLER,
            "schema_version": "TC-1/1.0.0",
            "origin_hash": f"{QUARANTINED_POLLER}-{index:04d}",
            "collector_seq": index,
            "clock_source": "ntp_undisciplined",
            "clock_offset_ms": QUARANTINED_OFFSET_MS,
        })

    # In event order, the DMZ polls after the main poller's in each minute they share.
    rows.sort(key=lambda row: row["event_time"])

    return pd.DataFrame(rows)


def _device_uptime_cs(device: str, minute: int) -> int:
    """A device's `sysUpTime` at a poll: up an hour when the corpus starts, and twenty or thirty seconds after each
    restart, as the port polls of the same switch report it."""
    boot_uptime_cs = 3600 * CS_PER_SECOND

    if (device == REBOOTING_SWITCH and minute >= REBOOT_AT_MINUTE):
        return (minute - REBOOT_AT_MINUTE) * 60 * CS_PER_SECOND + 30 * CS_PER_SECOND

    restarts = [at for at in STORM_REBOOT_MINUTES if at <= minute] if device == STORM_SWITCH else []

    if (len(restarts) > 0):
        return (minute - restarts[-1]) * 60 * CS_PER_SECOND + 20 * CS_PER_SECOND

    return boot_uptime_cs + minute * 60 * CS_PER_SECOND


def _device_config(device: str, minute: int) -> str:
    if (device != SWITCH or minute < PUSH_AT_MINUTE):
        return f"cfg-{device}-1"

    return f"cfg-{device}-2" if minute < OUT_OF_WINDOW_CHANGE_AT_MINUTE else f"cfg-{device}-3"


def _build_devices() -> pd.DataFrame:
    """Each switch and router's own poll, once a minute: a configuration push and a change outside the window, a
    restart, a reboot storm and a routing loss.

    Fixed values rather than draws, as the DMZ poller's are, so adding the class moves nothing in the others. The
    gauges vary with the minute so no column is a constant, and none of them is read by a rule.
    """
    rows = []
    seq = 0

    for minute in range(0, CORPUS_SECONDS // 60 + 1):
        for (index, device) in enumerate(DEVICES):
            routed = device == ROUTER
            dropped = routed and minute >= ROUTE_DROP_AT_MINUTE
            seq += 1
            rows.append({
                "event_time": minute * 60 * NS_PER_SECOND,
                "site_id": SITE,
                "device_id": device,
                "sys_uptime": _device_uptime_cs(device, minute),
                "running_config_hash": _device_config(device, minute),
                "cpu_util": round(0.2 + 0.01 * ((minute * 7 + index * 3) % 11), 2),
                "mem_util": round(0.4 + 0.01 * ((minute * 5 + index) % 7), 2),
                "route_count": (ROUTES_AFTER if dropped else ROUTES_BEFORE) if routed else 40 + index,
                "bgp_peer_count": (BGP_PEERS_AFTER if dropped else BGP_PEERS_BEFORE) if routed else 0,
                "syslog_rate": round(1.0 + 0.1 * ((minute + index) % 5), 1),
                "collector_id": DEVICE_POLLER,
                "schema_version": "TC-1/1.0.0",
                "origin_hash": f"{DEVICE_POLLER}-device-{seq:05d}",
                "collector_seq": seq,
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

    # The desk spoof: the host on the tapped port claims the gateway a few times, one tick each, after the tap. Too
    # few to move the gratuitous proportion, so R-D-L2-003's claimant count is the only layer 2 feature that sees it.
    for offset in range(DESK_SPOOF_PACKETS):
        events.append((DESK_SPOOF_AT_SECONDS + offset, GATEWAY_IP, MAC_C, GATEWAY_IP, "reply"))

    events.sort(key=lambda event: event[0])
    rows = []

    for (seq, (time_s, sender_ip, sender_mac, target_ip, operation)) in enumerate(events, start=1):
        # The records numbered and never delivered: every number from the loss onward is three further on.
        seq += LOST_ARP_RECORDS if time_s >= ARP_LOSS_AT_SECONDS else 0
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
