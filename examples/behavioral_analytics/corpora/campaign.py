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
The synthetic corpus `morpheus.compositions.campaign` is tested over, and the facts about it the harnesses assert.

Moved out of `tests/morpheus/determinism/campaign_pipeline.py` under plan section 4.5, which put the composition in the
package and left the data here, where the tests, the sample and fixture generators and the experiments read it. The
composition module's docstring describes what this corpus plants and why.
"""

import typing

from corpora import merged  # noqa: F401

from morpheus.compositions import envelope

host_identity = merged('host_identity')
import pandas as pd

presentation_pipeline = merged('presentation')
saas_pipeline = merged('saas')
telemetry_pipeline = merged('telemetry')
transport_pipeline = merged('transport')
from morpheus.compositions.campaign import ACTORS
from morpheus.compositions.campaign import AUTH_CLASS
from morpheus.compositions.campaign import BEACONERS
from morpheus.compositions.campaign import C2_ATTACKER
from morpheus.compositions.campaign import CLASSES
from morpheus.compositions.campaign import DAY_SECONDS
from morpheus.compositions.campaign import EDINBURGH
from morpheus.compositions.campaign import EXFILTRATORS
from morpheus.compositions.campaign import FLOW_CLASS
from morpheus.compositions.campaign import HANDSHAKE_CLASS
from morpheus.compositions.campaign import HQ
from morpheus.compositions.campaign import INVENTORY_CLASS
from morpheus.compositions.campaign import KIM
from morpheus.compositions.campaign import LAST_DAY
from morpheus.compositions.campaign import LEASE_CLASS
from morpheus.compositions.campaign import MAC_CLASS
from morpheus.compositions.campaign import PORT_CLASS
from morpheus.compositions.campaign import PORTAL_ISSUER
from morpheus.compositions.campaign import PROCESS_CLASS
from morpheus.compositions.campaign import SAAS_CLASS
from morpheus.compositions.campaign import SESSION_CLASS
from morpheus.compositions.campaign import SESSION_HOLDERS
from morpheus.compositions.campaign import TRANSFER_CLASS
from morpheus.compositions.campaign import Actor
from morpheus.compositions.campaign import at
from morpheus.compositions.campaign import hosts
from morpheus.utils.binding_table import NS_PER_SECOND

SEVERITY = 80
FANOUT_THRESHOLD = 50
"""R-B-L3-001's fan-out threshold, which the attacker stays well under."""
CAMPAIGN_HOUR = 10


def edr_name(host: str) -> str:
    """A host as the EDR reports it: in lower case, where the directory has it as an administrator typed it."""
    return host.lower()


SERVICES = r"C:\Windows\System32\services.exe"
SVCHOST = r"C:\Windows\System32\svchost.exe"
WMI = r"C:\Windows\System32\wbem\WmiPrvSE.exe"
CMD = r"C:\Windows\System32\cmd.exe"
PREVIOUS_HOUR_DESTINATIONS = 3
RISING_DESTINATIONS = 25
FLAT_DESTINATIONS = 20


def novel_image(principal: str) -> str:
    """The process each campaign starts, distinct per actor so one never excuses another in the peer group."""
    return rf"C:\Windows\Temp\{principal.split('@')[0]}-svc.exe"


EXFIL_SEVERITY = 80
PRIOR_EXPORTS = 110
PRIOR_TRANSFERS = 110
SYNC_SERVER = "198.51.100.10"
PORTAL = "198.51.100.20"
LOGON_COLLECTOR_PRINCIPALS = frozenset({KIM})
"""Principals whose sessions come from a collector that says `logon` and `logoff` rather than `start` and `end`.

Kim is the actor R-C-004 fires on, so the rule's one firing depends on the search reading `session_lifecycle`, the
column TC5SessionStage normalizes the vocabulary into, rather than a literal word one collector happens to use.
"""
C2_SEVERITY = 70
BEACON_FLOWS = 20
BEACON_PERIOD_SECONDS = 60
BEACON_BYTES = 512
PUBLIC_ISSUER = "CN=Public Web CA, O=Example Trust"
C2_PORT = 47000
"""The attacker's new stack's connection: the handshake that showed the new fingerprint, and the beacon's first flow
record, which the flow exporter writes when that connection ends. One connection at two layers, one Community ID."""


def _envelope(rows: list, collector: str, schema: str, origin: str, when: int) -> dict:
    return {
        "collector_id": collector,
        "schema_version": schema,
        "origin_hash": f"{origin}@{when}",
        "collector_seq": len(rows),
        "event_time": when,
    }


def _ephemeral_port(rows: list) -> int:
    """A source port, as a client's stack picks one: distinct per connection, and fixed by the corpus."""
    return 49152 + len(rows) % 16384


def _flow(rows: list,
          source: str,
          destination: str,
          when: int,
          *,
          src_port: typing.Optional[int] = None,
          dst_port: int = 445,
          bytes_out: int = 900,
          bytes_in: int = 1400,
          asn: str = "64512"):
    record = _envelope(rows, "netflow-01", "tc3.v1", f"{source}->{destination}", when)
    record.update({
        "src_ip": source,
        "src_port": _ephemeral_port(rows) if src_port is None else src_port,
        "dst_ip": destination,
        "dst_port": dst_port,
        "protocol": "tcp",
        "ip_ttl": 128,
        "bytes_out": bytes_out,
        "bytes_in": bytes_in,
        "bgp_as_dst": asn,
    })
    rows.append(record)


def _login(rows: list, principal: str, source: str, target: str, when: int, *, result: str = "success"):
    record = _envelope(rows, "dc-01", "tc5_auth.v1", f"{principal}>{target}", when)
    record.update({
        "user_principal": principal,
        "source_ip": source,
        "target_host": target,
        "app": "windows-logon",
        "auth_result": result,
    })
    rows.append(record)


def _process(rows: list, host: str, parent: str, image: str, when: int, integrity: str = "System"):
    record = _envelope(rows, "edr-01", "tc7_endpoint.v1", f"{host}>{image}", when)
    record.update({
        "hostname": host,
        "parent_image_path": parent,
        "image_path": image,
        "integrity_level": integrity,
        "signature_status": "Valid",
    })
    rows.append(record)


def _flows(rows: list, actor: Actor, index: int):
    """The hour before the campaign and the campaign's hour, from the actor's source."""
    base = f"10.30.{index}."

    if (actor.rising):
        for (count, minute) in enumerate((10, 20, 30)):
            _flow(rows, actor.source, f"{base}{count + 1}", at(LAST_DAY, CAMPAIGN_HOUR - 1, minute))

        for count in range(RISING_DESTINATIONS):
            _flow(rows,
                  actor.source,
                  f"{base}{100 + count}",
                  at(LAST_DAY, CAMPAIGN_HOUR, actor.fanout_minute, count * 10))
    else:
        for hour in (CAMPAIGN_HOUR - 1, CAMPAIGN_HOUR):
            for count in range(FLAT_DESTINATIONS):
                _flow(rows, actor.source, f"{base}{count + 1}", at(LAST_DAY, hour, actor.fanout_minute, count * 10))


def _session(rows: list, principal: str, address: str, session_id: str, action: str, when: int):
    # A Windows domain controller's 4624 and 4634 records, for the principals whose sessions come from one.
    if (principal in LOGON_COLLECTOR_PRINCIPALS):
        (collector, action) = ("dc-01", {"start": "logon", "end": "logoff"}[action])
    else:
        collector = "vpn-01"

    record = _envelope(rows, collector, "tc5_session.v1", f"{session_id}:{action}", when)
    record.update({
        "user_principal": principal, "source_ip": address, "session_id": session_id, "session_action": action
    })
    rows.append(record)


def _packet(rows: list, source: str, source_port: int, destination: str, data_len: int, when: int):
    record = _envelope(rows, "pcap-01", "tc4.v1", f"{source}:{source_port}={destination}:443", when)
    record.update({
        "src_ip": source,
        "src_port": source_port,
        "dst_ip": destination,
        "dst_port": 443,
        "protocol": "tcp",
        "tcp_flags": transport_pipeline.PSH_ACK,
        "data_len": data_len,
    })
    rows.append(record)


def _handshake(rows: list,
               source: str,
               destination: str,
               issuer: str,
               when: int,
               fingerprint: str = presentation_pipeline.CHROME,
               src_port: typing.Optional[int] = None):
    record = _envelope(rows, "tls-inspect-01", "tc6.v1", f"{source}->{destination}", when)
    record.update({
        "src_ip": source,
        "src_port": _ephemeral_port(rows) if src_port is None else src_port,
        "dst_ip": destination,
        "dst_port": 443,  # The campaign's inspection point records no server name, so each destination is its address.
        "sni": None,
        "protocol": "tcp",
        "tls_version": "TLSv1.3",
        "ja4_client": fingerprint,
        "certificate_issuer": issuer,
        "certificate_fingerprint_sha256": f"{destination}|{issuer}",
        "cipher_suite": presentation_pipeline.MODERN_SUITE,
        "validation_result": presentation_pipeline.VALID,
        "certificate_not_before": when - 10 * DAY_SECONDS * NS_PER_SECOND,
        "certificate_not_after": when + 355 * DAY_SECONDS * NS_PER_SECOND,
        "content_type_declared": None,
        "content_type_detected": None,
    })
    rows.append(record)


def _export(rows: list, principal: str, records: int, when: int, source: str):
    record = _envelope(rows, "saas-audit-01", "tc7_saas.v1", f"{principal}>export@{records}", when)
    record.update({
        "user_principal": principal,
        # The client address the audit log records, which is what places the export in a session.
        "source_ip": source,
        "operation": saas_pipeline.EXPORT,
        "target_object": "customer-list",
        "target_object_type": "Account",
        "record_count": records,
        "result": "success",
        "client_app": "web",
    })
    rows.append(record)


BREACH_PORT = 45000
"""The source port of every breach connection, which is how the layer 3, 4 and 6 records of one connection agree."""


def _build_exfiltration(sessions: list, packets: list, handshakes: list, exports: list, flows: list):
    for (index, (principal, address)) in enumerate(sorted(SESSION_HOLDERS.items())):
        for day in range(LAST_DAY + 1):
            session_id = f"{principal.split('@')[0]}-{day}"
            _session(sessions, principal, address, session_id, "start", at(day, 8, index))

            actor = EXFILTRATORS.get(principal)
            ends_early = day == LAST_DAY and actor is not None and actor.session_end_minute is not None
            end = at(day, CAMPAIGN_HOUR, actor.session_end_minute) if ends_early else at(day, 18, index)
            _session(sessions, principal, address, session_id, "end", end)

            # Every host in the estate reaches the corporate portal every day, so its issuer is the estate's own.
            _handshake(handshakes, address, PORTAL, PORTAL_ISSUER, at(day, 8, 30 + index))

    for (index, actor) in enumerate(EXFILTRATORS.values()):
        # A long, even history of exports, ending the day before, so the baseline is mature and small.
        for count in range(PRIOR_EXPORTS):
            _export(exports,
                    actor.principal,
                    40 + (count * 4) % 21,
                    at(0, 12, index) + count * 90 * 60 * NS_PER_SECOND,
                    actor.address)

        # The breach address's regular uploads to the sync server, inside the envelope's eight-hour window.
        for count in range(PRIOR_TRANSFERS):
            _packet(packets,
                    actor.breach_address,
                    44000 + count,
                    SYNC_SERVER,
                    1200,
                    at(LAST_DAY, 3, index) + count * 3 * 60 * NS_PER_SECOND)

        _export(exports, actor.principal, 5000, at(LAST_DAY, CAMPAIGN_HOUR, actor.export_minute), actor.address)
        breach = at(LAST_DAY, CAMPAIGN_HOUR, actor.breach_minute)
        _packet(packets, actor.breach_address, BREACH_PORT, SYNC_SERVER, 60000, breach)

        # The same connection as the flow exporter saw it, and, for the attacker, as the inspection point saw its
        # handshake: one connection at three layers, which the connection evidence search joins on Community ID.
        # The others' breaches are seen at two layers, which is the control that the search wants all three.
        _flow(flows,
              actor.breach_address,
              SYNC_SERVER,
              breach,
              src_port=BREACH_PORT,
              dst_port=443,
              bytes_out=60000,
              bytes_in=2400,
              asn="64500")

        if (actor.principal == KIM):
            _handshake(handshakes, actor.breach_address, SYNC_SERVER, PORTAL_ISSUER, breach, src_port=BREACH_PORT)
        _handshake(handshakes,
                   actor.breach_address,
                   f"203.0.113.{10 + index}",
                   actor.issuer,
                   at(LAST_DAY, CAMPAIGN_HOUR, actor.handshake_minute))


def _build_command_and_control(flows: list, handshakes: list):
    for (index, actor) in enumerate(BEACONERS.values()):
        # The host's ordinary history: one stack, to the portal, every two minutes from eight.
        for count in range(actor.history):
            _handshake(handshakes, actor.host, PORTAL, PORTAL_ISSUER, at(LAST_DAY, 8, 2 * count, index))

        # Only the attacker's beacon runs over the connection the new stack opened. The controls' fingerprints and
        # beacons are separate connections, so R-C-002 joins them on their addresses, as it always did.
        shared = actor.host == C2_ATTACKER
        _handshake(handshakes,
                   actor.host,
                   actor.fingerprint_destination,
                   PUBLIC_ISSUER,
                   at(LAST_DAY, CAMPAIGN_HOUR, actor.fingerprint_minute, index),
                   fingerprint=presentation_pipeline.NEW_STACK,
                   src_port=C2_PORT if shared else None)

        for count in range(BEACON_FLOWS):
            when = at(LAST_DAY, CAMPAIGN_HOUR, actor.beacon_start_minute,
                      index) + count * BEACON_PERIOD_SECONDS * NS_PER_SECOND
            record = _envelope(flows, "netflow-01", "tc3.v1", f"{actor.host}->{actor.beacon_destination}", when)
            record.update({
                "src_ip": actor.host,
                "src_port": C2_PORT if (shared and count == 0) else _ephemeral_port(flows),
                "dst_ip": actor.beacon_destination,
                "dst_port": 443,
                "protocol": "tcp",
                "ip_ttl": 128,
                "bytes_out": BEACON_BYTES,
                "bytes_in": 128,
                "bgp_as_dst": "64500",
            })
            flows.append(record)


REPLAY_SEVERITY = 90
REPLAY_HOUR = 14
SITE_SWITCHES = {HQ: "hq-sw1", EDINBURGH: "edi-sw1"}
QUIN = "quin@example.com"
RHEA = "rhea@example.com"
SAM = "sam@example.com"
TIA = "tia@example.com"
UMA = "uma@example.com"
VIC = "vic@example.com"


class Desk(typing.NamedTuple):
    site: str
    port: str
    address: str

    @property
    def mac(self) -> str:
        """The desk's workstation, numbered from its address so two desks never share one."""
        octets = [int(part) for part in self.address.split(".")]

        return "aa:30:" + ":".join(f"{octet:02x}" for octet in octets)


class Replay(typing.NamedTuple):
    principal: str
    first: Desk
    second: Desk
    minutes_later: int
    second_result: str = "success"
    second_leased: bool = True
    lease_ended_minutes_before: typing.Optional[int] = None


REMOTE_ADDRESS = "198.51.100.77"
REPLAYERS = {
    QUIN:
        Replay(QUIN, Desk(HQ, "Gi1/0/11", "10.30.1.11"), Desk(EDINBURGH, "Gi2/0/21", "10.30.2.21"), 20),
    RHEA:
        Replay(RHEA, Desk(HQ, "Gi1/0/12", "10.30.1.12"), Desk(HQ, "Gi1/0/13", "10.30.1.13"), 20),
    SAM:
        Replay(SAM, Desk(HQ, "Gi1/0/14", "10.30.1.14"), Desk(EDINBURGH, "Gi2/0/24", "10.30.2.24"), 180),
    TIA:
        Replay(TIA,
               Desk(HQ, "Gi1/0/15", "10.30.1.15"),
               Desk(EDINBURGH, "Gi2/0/25", REMOTE_ADDRESS),
               20,
               second_leased=False),
    UMA:
        Replay(UMA,
               Desk(HQ, "Gi1/0/16", "10.30.1.16"),
               Desk(EDINBURGH, "Gi2/0/26", "10.30.2.26"),
               20,
               lease_ended_minutes_before=60),
    VIC:
        Replay(VIC,
               Desk(HQ, "Gi1/0/17", "10.30.1.17"),
               Desk(EDINBURGH, "Gi2/0/27", "10.30.2.27"),
               20,
               second_result="failure"),
}
"""The attacker, then the five who each fall one step short of R-C-005."""


def _replay_times(index: int, replay: Replay) -> tuple:
    first = at(LAST_DAY, REPLAY_HOUR, index)

    return (first, first + replay.minutes_later * 60 * NS_PER_SECOND)


def _desks() -> list:
    """Every desk the replay corpus has a workstation at, once, in a fixed order."""
    desks = {}

    for replay in REPLAYERS.values():
        for desk in (replay.first, replay.second):
            if (desk.address != REMOTE_ADDRESS):
                desks[(desk.site, desk.port)] = desk

    return [desks[key] for key in sorted(desks)]


def _build_mac_tables(rows: list):
    """Both sites' MAC tables every five minutes, from an hour before the sign-ins until after the last of them:
    R-C-005's desks, and the desk ports R-C-003's devices are plugged into."""
    for when in _walk_times():
        listed = [(desk, desk.mac) for desk in _desks()] + _intrusion_macs(when)

        for (desk, mac) in listed:
            record = _envelope(rows, f"{desk.site}-mac", "tc2_mac.v1", mac, when)
            record.update({
                "mac_address": mac,
                "site_id": desk.site,
                "switch_id": SITE_SWITCHES[desk.site],
                "port_id": desk.port,
                "vlan_id": DESK_VLAN,
            })
            rows.append(record)


CORPUS_CLASSES = CLASSES + (MAC_CLASS, LEASE_CLASS, INVENTORY_CLASS, "tc1")
"""Every collector in the corpus: the emitted classes, the MAC tables, leases and inventory reports the addresses
resolve through, and the headquarters switch's port polls."""
CLIENT_POOLS = ("10.20.1.", "10.20.2.")
"""The workstation subnets the DHCP server hands out: R-C-004's and R-C-002's hosts. The servers on 10.20.0.x are on
static addresses, and the desks R-C-005 signs in from are leased by the acknowledgements above."""
LEASE_COLLECTOR = "dhcp-01"
LEASE_STARTS_HOUR = 8
LEASE_ENDS_HOUR = 20


def _build_leases(rows: list):
    """
    One acknowledgement per desk address, at the start of the working day, granting it to the desk's workstation.

    Each lease runs to the evening, except the one the expired-lease control's second sign-in comes from, whose
    term ends an hour before that sign-in; the remote address was never leased. The relay names the switch and port
    the request arrived on, as DHCP option 82 does, so a lease places its address on a port without the MAC table.
    """
    granted = set()

    for (index, replay) in enumerate(REPLAYERS.values()):
        (_, second_time) = _replay_times(index, replay)

        for (desk, ends) in ((replay.first, None), (replay.second, replay.lease_ended_minutes_before)):
            if (desk.address == REMOTE_ADDRESS or desk.address in granted):
                continue

            granted.add(desk.address)
            start = at(LAST_DAY, LEASE_STARTS_HOUR)
            end = at(LAST_DAY, LEASE_ENDS_HOUR) if ends is None else second_time - ends * 60 * NS_PER_SECOND
            record = _envelope(rows, LEASE_COLLECTOR, "tc3_dhcp.v1", f"{desk.address}>{desk.mac}", start)
            record.update({
                "ip": desk.address,
                "mac": desk.mac,
                "switch_id": SITE_SWITCHES[desk.site],
                "port_id": desk.port,
                "dhcp_message": "ack",
                "lease_seconds": (end - start) // NS_PER_SECOND,
            })
            rows.append(record)


def _build_replays(logins: list):
    for (index, replay) in enumerate(REPLAYERS.values()):
        (first, second) = _replay_times(index, replay)
        _login(logins, replay.principal, replay.first.address, "intranet", first)
        _login(logins, replay.principal, replay.second.address, "intranet", second, result=replay.second_result)


PORT_POLLER = "hq-mac"
"""The headquarters SNMP poller, which walks the access switch's interface table on the cadence it walks the bridge
table on. One poller for both is how SNMP collection is deployed -- `sysUpTime` rides on both walks -- and it keeps
the campaign's collectors the eleven the clock skew experiment spreads by name."""
PHYSICAL_HOUR = 16
PHYSICAL_SEVERITY = 85
MANAGEMENT_VLAN = 99
SWITCH_MANAGEMENT = "10.99.0.10"
"""The headquarters switch's own management interface."""
PRINT_SERVER = "10.20.0.30"
DESK_VLAN = 30
FLEET_OUI = "aa:30:0a"
"""The vendor of every workstation the estate issues: each desk's MAC is numbered from its address, under one OUI."""
BOOT_UPTIME_CS = 30 * DAY_SECONDS * 100
"""The headquarters switch's uptime when the polls begin, in hundredths of a second: up for a month."""


class Intrusion(typing.NamedTuple):
    """Something plugged into a desk port: the desk whose port it is, the device and what it does."""

    desk: Desk
    device_mac: str
    device_address: str
    link_event: bool = True
    reaches_management: bool = True


INTRUSIONS = {
    "rogue":
        Intrusion(Desk(HQ, "Gi1/0/41", "10.30.1.41"), "b8:27:eb:3c:00:41", "10.30.1.141"),
    "known-vendor":
        Intrusion(Desk(HQ, "Gi1/0/42", "10.30.1.42"), "aa:30:0a:1e:02:42", "10.30.1.142"),
    "no-management":
        Intrusion(Desk(HQ, "Gi1/0/43", "10.30.1.43"), "00:1b:a9:3c:00:43", "10.30.1.143", reaches_management=False),
    "no-link-event":
        Intrusion(Desk(HQ, "Gi1/0/44", "10.30.1.44"), "b8:27:eb:3c:00:44", "10.30.1.144", link_event=False),
}
"""The attacker, then the three who each do two of R-C-003's three steps. A single-board computer's OUI, b8:27:eb;
another workstation of the estate's own; a printer's, 00:1b:a9."""


def _intrusion_time(index: int) -> int:
    """When the device goes in: three minutes past a ten-minute mark, between two polls and two table walks."""
    return at(LAST_DAY, PHYSICAL_HOUR, 3 + 10 * index)


def _walk_times() -> range:
    """The table walks and polls, every five minutes from an hour before the sign-ins until after the last of them."""
    return range(at(LAST_DAY, REPLAY_HOUR - 1), at(LAST_DAY, REPLAY_HOUR + 5) + 1, 300 * NS_PER_SECOND)


def _intrusion_macs(when: int) -> list:
    """What each R-C-003 port's walk lists at an instant: the desk's workstation until it is unplugged, and the
    device from the first walk after it went in. Behind a desk switch, both."""
    rows = []

    for (index, intrusion) in enumerate(INTRUSIONS.values()):
        arrived = when > _intrusion_time(index)

        if (not arrived or not intrusion.link_event):
            rows.append((intrusion.desk, intrusion.desk.mac))

        if (arrived):
            rows.append((intrusion.desk, intrusion.device_mac))

    return rows


def _build_port_polls(rows: list):
    """
    The headquarters switch's poll of each R-C-003 port, every five minutes, with the fields the layer 1 stages read.

    The port is up on every poll. Where a workstation is unplugged and a device plugged in, the link dropped and
    came back between two polls, and only the switch's own `ifLastChange` records it: it advances to the instant
    the link came back, which is what `TC1FlapStage` counts as two transitions.
    """
    octets = {intrusion.desk.port: 10**12 for intrusion in INTRUSIONS.values()}
    start = _walk_times()[0]

    for (poll, when) in enumerate(_walk_times()):
        for (index, intrusion) in enumerate(INTRUSIONS.values()):
            # A healthy optic still wanders by thousandths of a decibel; a perfectly flat reading gives the trend fit
            # nothing to estimate its own spread from.
            jitter = ((poll * 7 + index * 3) % 11 - 5) * 0.004
            port = intrusion.desk.port
            uptime_cs = BOOT_UPTIME_CS + (when - start) // (NS_PER_SECOND // 100)
            changed = intrusion.link_event and when > _intrusion_time(index)
            last_change_cs = (BOOT_UPTIME_CS + (_intrusion_time(index) + 20 * NS_PER_SECOND - start) //
                              (NS_PER_SECOND // 100)) if changed else 10 * 100
            octets[port] += 300 * 4_000_000 // 8
            record = _envelope(rows, PORT_POLLER, "TC-1/1.0.0", f"{port}@{when}", when)
            record.update({
                "site_id": HQ,
                "device_id": SITE_SWITCHES[HQ],
                "port_id": port,
                "uptime": uptime_cs,
                "if_last_change": last_change_cs,
                "oper_status": "up",
                "link_speed_bps": 1_000_000_000,
                "optical_tx_dbm": round(-2.0 - jitter / 2, 4),
                "optical_rx_dbm": round(-7.0 + jitter, 4),
                "transceiver_serial": f"XCVR-{SITE_SWITCHES[HQ]}-{port}",
                "transceiver_type": telemetry_pipeline.OPTIC_TYPE,
                "lldp_neighbor_chassis_id": None,
                "lldp_neighbor_port_id": None,
                "crc_errors": 0,
                "symbol_errors": 0,
                "input_discards": 0,
                "output_discards": 0,
                "if_hc_in_octets": octets[port],
                "if_hc_out_octets": octets[port] // 2,
            })
            rows.append(record)


def _build_intrusion_leases(rows: list):
    """The DHCP server's acknowledgement for each device, a minute after it went in, held until the evening."""
    for (index, intrusion) in enumerate(INTRUSIONS.values()):
        start = _intrusion_time(index) + 60 * NS_PER_SECOND
        end = at(LAST_DAY, LEASE_ENDS_HOUR)
        record = _envelope(rows,
                           LEASE_COLLECTOR,
                           "tc3_dhcp.v1",
                           f"{intrusion.device_address}>{intrusion.device_mac}",
                           start)
        record.update({
            "ip": intrusion.device_address,
            "mac": intrusion.device_mac,
            "switch_id": SITE_SWITCHES[HQ],
            "port_id": intrusion.desk.port,
            "dhcp_message": "ack",
            "lease_seconds": (end - start) // NS_PER_SECOND,
        })
        rows.append(record)


def _build_intrusion_flows(flows: list):
    """Three flows from each device a quarter of an hour after it went in: SSH to the switch's management
    interface, or, for the printer, to the print server."""
    for (index, intrusion) in enumerate(INTRUSIONS.values()):
        (destination, port) = (SWITCH_MANAGEMENT, 22) if intrusion.reaches_management else (PRINT_SERVER, 631)

        for minute in range(3):
            _flow(flows,
                  intrusion.device_address,
                  destination,
                  _intrusion_time(index) + (15 + minute) * 60 * NS_PER_SECOND,
                  dst_port=port,
                  bytes_out=2400,
                  bytes_in=3600)


FLOW_EXPORTERS = (("10.20.0.", "rtr-campus-0:256"), ("10.20.1.", "rtr-campus-1:512"), ("10.20.2.", "rtr-campus-2:768"),
                  ("10.30.1.", "rtr-hq-1:1024"))
"""The router that exports a source's flows, and its IPFIX observation domain, by the source's subnet."""


def _flow_exporter(source: str) -> str:
    return next(exporter for (prefix, exporter) in FLOW_EXPORTERS if source.startswith(prefix))


def build_corpus() -> dict[str, pd.DataFrame]:
    """The campaign, split by collector."""
    flows: list = []
    logins: list = []
    processes: list = []

    for day in range(LAST_DAY + 1):
        for (index, actor) in enumerate(ACTORS.values()):
            # Every principal's working day starts with a login to their own server from their own address.
            _login(logins, actor.principal, actor.source, actor.home, at(day, 9, index))

        for host in hosts():
            _process(processes, host, SERVICES, SVCHOST, at(day, 9, 30))
            _process(processes, host, SVCHOST, WMI, at(day, 9, 30))

    for (index, actor) in enumerate(ACTORS.values()):
        _flows(flows, actor, index)
        _login(logins,
               actor.principal,
               actor.login_source,
               actor.target,
               at(LAST_DAY, CAMPAIGN_HOUR, actor.login_minute))
        host = edr_name(actor.target)

        if (actor.process_minute is None):
            _process(processes, host, SVCHOST, WMI, at(LAST_DAY, CAMPAIGN_HOUR, 20))
        else:
            _process(processes,
                     host,
                     SERVICES,
                     novel_image(actor.principal),
                     at(LAST_DAY, CAMPAIGN_HOUR, actor.process_minute))

    def frame(rows: list) -> pd.DataFrame:
        result = pd.DataFrame(rows).sort_values(["event_time", "collector_seq"], kind="mergesort")
        result = result.reset_index(drop=True)
        # Each collector numbers its own records, in the order it saw them. Two collectors writing into one frame --
        # the two session sources, the two sites' MAC tables -- keep a sequence each, so neither one's records read
        # as a gap in the other's.
        result["collector_seq"] = result.groupby("collector_id", sort=False).cumcount()

        return result

    sessions: list = []
    packets: list = []
    handshakes: list = []
    exports: list = []
    _build_exfiltration(sessions, packets, handshakes, exports, flows)
    _build_command_and_control(flows, handshakes)
    _build_replays(logins)
    mac_tables: list = []
    _build_mac_tables(mac_tables)
    leases: list = []
    _build_leases(leases)
    _build_intrusion_leases(leases)
    port_polls: list = []
    _build_port_polls(port_polls)

    # The workstations' daily renewals and inventory reports, which say which machine an address was at layers 3, 4
    # and 6. The renewals come from the same DHCP server as the desks' leases.
    sources = [row["src_ip"] for row in flows + packets + handshakes]
    times = [row["event_time"] for row in flows + packets + handshakes]
    workstations = host_identity.build(sources, min(times), max(times), pools=CLIENT_POOLS)
    # R-C-003's devices are added after: none is a workstation the EDR inventories, and their flows' times must not
    # stretch the span the workstations' renewals cover.
    _build_intrusion_flows(flows)

    # Every flow names the router that metered it, which is what stamped its times.
    flow_frame = frame(flows)
    flow_frame["exporter_id"] = [_flow_exporter(source) for source in flow_frame["src_ip"]]

    return envelope.attach_all({
        FLOW_CLASS: flow_frame,
        AUTH_CLASS: frame(logins),
        PROCESS_CLASS: frame(processes),
        SESSION_CLASS: frame(sessions),
        TRANSFER_CLASS: frame(packets),
        HANDSHAKE_CLASS: frame(handshakes),
        SAAS_CLASS: frame(exports),
        MAC_CLASS: frame(mac_tables),
        PORT_CLASS: frame(port_polls),
        LEASE_CLASS: frame(leases + workstations[LEASE_CLASS].to_dict("records")),
        INVENTORY_CLASS: frame(workstations[INVENTORY_CLASS].to_dict("records")),
    })
