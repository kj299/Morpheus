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
Real-format fixtures, normalized, are the committed samples; and a segment run over them writes what it writes from
the samples.

`examples/behavioral_analytics/fixtures/make_fixtures.py` renders the samples of every class in its `CLASSES` as
the feeds write them -- IPFIX in a packet capture, Entra ID sign-in records, Zeek `ssl.log`/`x509.log`, `http.log` and
`dns.log`, unified audit log records, Sysmon events, Telegraf's SNMP metrics and MAC notification traps, ARP and TCP
captures, FreeRADIUS detail files, a Kea lease log, osquery results, Zeek `conn.log`, a Workday report, Entra ID's
audit log, ServiceNow's CMDB history and NetBox's changelog -- carrying only what each format carries. Reading them back through
`FlowRecordNormalizeStage` and the `collector_schemas` schemas has to give the samples column for column, except
what a format cannot carry, which is named here and has to be null rather than invented.
"""

import filecmp
import json
import math
import os
import pathlib
import re
import shutil
import struct
import sys

import pandas as pd
import pytest

from morpheus.utils import collector_schemas
from morpheus.utils import ipfix
from morpheus.utils.collector_schemas import CollectorProfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
EXAMPLE = os.path.join(REPO_ROOT, "examples", "behavioral_analytics")
FIXTURES = os.path.join(EXAMPLE, "fixtures")

sys.path.insert(0, FIXTURES)
sys.path.insert(0, EXAMPLE)
sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "morpheus", "determinism"))
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "upstream_reuse"))

# pylint: disable=wrong-import-position
import make_fixtures  # noqa: E402

NOT_IN_THE_FORMAT = {
    "tc6": ("content_type_declared", "content_type_detected"),
    "tc1_device": ("running_config_hash", "syslog_rate"),
    "tc0_asset": ("valid_from", ),
    "tc0_application": ("valid_from", ),
    "tc0_network": ("valid_from", ),
}
"""Columns a format cannot carry: what an inspecting proxy saw inside a TLS session is not in Zeek's logs, and no
MIB a switch answers holds a hash of its configuration or the rate of its syslog, and a CMDB's history and NetBox's
changelog keep when a change was entered, not when it took effect."""

NOT_IN_THE_FORMAT_WHERE = {"tc0_identity": ("group_name", ("valid_from", "valid_to"))}
"""Columns a format cannot carry on some of a class's records, named by the column that marks them: a group
membership comes from the directory's audit log, which says when it was made and not when it held."""

NUMBERED_BY_THE_READER = {"tc2_arp"}
"""Classes whose format numbers nothing, so `collector_seq` is the reader's count in file order: a capture does not
show the three records the ARP sensor numbered and never delivered, which its own sequence skips."""

OTHER_FEEDS = {"tc5_session": ("idp-sessions", )}
"""Collectors of a class whose records are another feed's than the fixture's: the identity provider's session
records, each with both ends in it, are not RADIUS accounting."""


def _plain(value):
    if (value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))):
        return None

    if (hasattr(value, "item")):
        value = value.item()

    return int(value) if isinstance(value, float) and value.is_integer() else value


def _records(frame: pd.DataFrame, columns: list) -> list[dict]:
    return [{column: _plain(row.get(column)) for column in columns} for row in frame.to_dict("records")]


@pytest.fixture(name="normalized", scope="module")
def normalized_fixture() -> dict:
    yield make_fixtures.normalize_all()


def test_the_committed_fixtures_are_what_the_samples_render_to(tmp_path):
    make_fixtures.write_all(str(tmp_path))

    for (segment, names) in make_fixtures.CLASSES.values():
        for name in names:
            assert filecmp.cmp(os.path.join(tmp_path, segment, name), os.path.join(FIXTURES, segment, name),
                               shallow=False), f"{segment}/{name} is stale; run make_fixtures.py"


@pytest.mark.parametrize("telemetry_class", sorted(make_fixtures.CLASSES))
def test_a_normalized_fixture_is_its_sample(telemetry_class: str, normalized: dict):
    segment = make_fixtures.CLASSES[telemetry_class][0]
    sample = [
        row for row in make_fixtures.sample(segment, telemetry_class)
        if row.get("collector_id") not in OTHER_FEEDS.get(telemetry_class, ())
    ]
    columns = list(sample[0])
    frame = normalized[telemetry_class]
    missing = NOT_IN_THE_FORMAT.get(telemetry_class, ())
    (marker, missing_where) = NOT_IN_THE_FORMAT_WHERE.get(telemetry_class, (None, ()))
    marked = [marker is not None and row.get(marker) is not None for row in sample]
    expected = [{
        column: (None if column in missing or (is_marked and column in missing_where) else _plain(row.get(column)))
        for column in columns
    } for (row, is_marked) in zip(sample, marked)]

    if (telemetry_class in NUMBERED_BY_THE_READER):
        first = make_fixtures.PROFILES[telemetry_class].first_seq
        assert any(row["collector_seq"] != first + index for (index, row) in enumerate(expected))

        for (index, row) in enumerate(expected):
            row["collector_seq"] = first + index

    assert _records(frame, columns) == expected
    # Whatever the schema writes beyond the sample's columns is an optional column the sample omits; it is null.
    for column in set(frame.columns) - set(columns):
        assert frame[column].isna().all(), column

    # The columns the format cannot carry are not invented, and the sample does have values there to miss.
    for column in missing:
        assert frame[column].isna().all()
        assert any(row[column] is not None for row in sample), column

    for column in missing_where:
        assert frame[column][marked].isna().all()
        assert any(row[column] is not None for (row, is_marked) in zip(sample, marked) if is_marked), column


def test_every_class_column_the_contract_names_is_written(normalized: dict):
    with open(os.path.join(EXAMPLE, "input_contracts.json"), encoding="utf-8") as handle:
        contract = json.load(handle)

    envelope = {column["name"] for column in contract["envelope"]}

    for (telemetry_class, frame) in normalized.items():
        named = [column["name"] for column in contract["classes"][telemetry_class]["columns"]]

        enveloped = envelope if contract["classes"][telemetry_class]["envelope"] else set()

        assert set(frame.columns) == set(named) | enveloped, telemetry_class


def test_the_query_string_is_stripped_and_vocabulary_mapped(normalized: dict):
    with open(os.path.join(FIXTURES, "application", "http.log"), encoding="utf-8") as handle:
        assert "?ref=" in handle.read(), "the fixture has stopped exercising the query string"

    assert not normalized["tc7_http"]["url_path"].str.contains("?", regex=False).any()
    assert set(normalized["tc6"]["tls_version"]) == {"TLSv1.2", "TLSv1.3"}
    assert set(normalized["tc6"]["validation_result"]) == {"ok", "self-signed"}


def test_the_sign_in_mapping_is_the_one_the_upstream_evaluation_measures():
    # `evaluate.py` names the sign-in properties as the upstream stage flattens them, separators stripped; the
    # schema names them as `json_normalize` does. One cannot move without the other.
    import evaluate  # pylint: disable=import-outside-toplevel

    measured = evaluate.SOURCE_MAPPINGS["azure"]
    flattened = {column: re.sub("[_.{}:]", "", field) for (column, field) in collector_schemas.ENTRA_FIELDS.items()}

    assert flattened == measured
    assert set(evaluate.FORK_TC5_INPUTS) == set(collector_schemas.ENTRA_FIELDS)


def test_a_sign_in_is_read_as_the_contract_names_it(tmp_path):
    profile = CollectorProfile("c", "tc5_auth.v1", "corp")
    path = tmp_path / "signins.jsonlines"
    path.write_text(json.dumps({
        "id": "sign-in-1",
        "createdDateTime": "2026-10-17T10:00:00.1234567Z",
        "userPrincipalName": "Alice@Example.com",
        "appDisplayName": "wiki",
        "ipAddress": "203.0.113.11",
        "location": {
            "city": "New York", "state": "New York", "countryOrRegion": "US"
        },
        "status": {
            "errorCode": 50126
        },
        "autonomousSystemNumber": 7018,
        "authenticationRequirement": "multiFactorAuthentication",
        "authenticationDetails": [{
            "authenticationMethod": "Password", "succeeded": True
        }, {
            "authenticationMethod": "Text message", "succeeded": False
        }],
        "incomingTokenType": "saml11",
    }) + "\n",
                    encoding="utf-8")

    row = collector_schemas.entra_signin(str(path), profile).iloc[0]

    assert row["origin_hash"] == "sign-in-1"
    assert row["event_time"] == pd.Timestamp("2026-10-17T10:00:00.1234567Z").value
    assert row["user_principal"] == "alice@example.com"
    assert (row["source_country"], row["source_region"], row["source_city"]) == ("us", "new-york", "new-york")
    assert row["source_latitude"] is pd.NA and row["device_id"] is None and row["target_host"] is None
    assert row["source_asn"] == "as7018"
    assert (row["auth_result"], row["mfa_used"], row["mfa_result"]) == ("failure", True, "denied")
    # A token type the map does not know reaches the SIEM as itself.
    assert row["token_type"] == "saml11"
    assert row["observed_time"] == row["event_time"] and row["tenant_id"] == "corp"


def test_a_passwordless_first_factor_is_not_read_as_a_second_factor(tmp_path):
    # A FIDO2 key or Windows Hello signs a user in with one step and no password; that step is the first factor,
    # so a single-factor sign-in made that way carries no `mfa_result`, and a second factor is whatever follows it.
    profile = CollectorProfile("c", "tc5_auth.v1", "corp")
    path = tmp_path / "signins.jsonlines"
    passwordless = {
        "id": "sign-in-2",
        "createdDateTime": "2026-10-17T10:00:00Z",
        "userPrincipalName": "bob@example.com",
        "appDisplayName": "wiki",
        "ipAddress": "203.0.113.12",
        "status": {
            "errorCode": 0
        },
        "authenticationRequirement": "singleFactorAuthentication",
        "authenticationDetails": [{
            "authenticationMethod": "Windows Hello for Business", "succeeded": True
        }],
        "incomingTokenType": "none",
    }
    stepped_up = {
        **passwordless,
        "id":
            "sign-in-3",
        "authenticationRequirement":
            "multiFactorAuthentication",
        "authenticationDetails": [{
            "authenticationMethod": "FIDO2 security key", "succeeded": True
        }, {
            "authenticationMethod": "Mobile app notification", "succeeded": True
        }],
    }
    path.write_text(json.dumps(passwordless) + "\n" + json.dumps(stepped_up) + "\n", encoding="utf-8")

    frame = collector_schemas.entra_signin(str(path), profile)

    assert (frame.iloc[0]["mfa_used"], frame.iloc[0]["mfa_result"]) == (False, None)
    assert (frame.iloc[1]["mfa_used"], frame.iloc[1]["mfa_result"]) == (True, "approved")


def test_an_unknown_value_passes_through_rather_than_becoming_null(tmp_path):
    profile = CollectorProfile("c", "tc7_saas.v1", "corp")
    path = tmp_path / "audit.jsonlines"
    path.write_text(json.dumps({
        "CreationTime": "2026-10-17T10:00:00",
        "UserId": "a@example.com",
        "Operation": "FileDownloaded",
        "ObjectId": "o",
        "ItemType": "File",
        "ResultStatus": "PartiallySucceeded",
        "ClientInfoString": "x"
    }) + "\n",
                    encoding="utf-8")

    row = collector_schemas.m365_audit(str(path), profile).iloc[0]

    assert row["result"] == "partiallysucceeded"
    assert row["tenant_id"] == "corp"
    assert row["record_count"] is pd.NA
    assert row["event_time"] == pd.Timestamp("2026-10-17T10:00:00Z").value


def test_the_zeek_reader_takes_the_unset_marker_booleans_and_an_unclosed_log(tmp_path):
    # A log Zeek is still writing has no #close line; its last record is a record.
    path = tmp_path / "dns.log"
    path.write_text(
        "#separator \\x09\n#set_separator\t,\n#empty_field\t(empty)\n#unset_field\t-\n#path\tdns\n"
        "#open\t2026-10-17-00-00-00\n#fields\tts\tid.orig_h\tid.orig_p\tquery\tAA\tanswers\n"
        "#types\ttime\taddr\tport\tstring\tbool\tvector[string]\n"
        "1.5\t10.0.0.1\t-\ta.example\tT\t1.2.3.4,5.6.7.8\n"
        "2.5\t10.0.0.2\t5353\tb # c.example\tF\t-\n",
        encoding="utf-8")

    frame = collector_schemas.read_zeek(str(path))

    assert len(frame) == 2
    assert frame["id.orig_p"].tolist()[0] is pd.NA and frame["id.orig_p"].tolist()[1] == 5353
    assert frame["AA"].tolist() == [True, False]
    assert frame["query"].tolist() == ["a.example", "b # c.example"]
    assert pd.isna(frame["answers"].tolist()[1])


def test_the_fingerprint_join_is_used_when_the_logs_carry_no_file_ids(normalized: dict):
    ssl = collector_schemas.read_zeek(os.path.join(FIXTURES, "presentation",
                                                   "ssl.log")).drop(columns=["cert_chain_fuids"])
    x509 = collector_schemas.read_zeek(os.path.join(FIXTURES, "presentation", "x509.log")).drop(columns=["id"])
    joined = collector_schemas.join_ssl_x509(ssl, x509)

    assert joined["certificate.issuer"].tolist() == normalized["tc6"]["certificate_issuer"].tolist()


def test_a_process_whose_image_load_was_not_logged_has_no_signature_status():
    events = pd.DataFrame([
        {
            "EventID": 1, "ProcessGuid": "{a}", "Image": "C:\\a.exe"
        },
        {
            "EventID": 7,
            "ProcessGuid": "{a}",
            "Image": "C:\\a.exe",
            "ImageLoaded": "C:\\x.dll",
            "SignatureStatus": "Valid"
        },
    ])

    assert collector_schemas.join_image_signatures(events)["SignatureStatus"].tolist() == [None]


def _telegraf(path: pathlib.Path, records: list[dict]) -> str:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    return str(path)


def test_a_port_with_no_neighbour_and_an_unknown_state_is_read_as_it_is(tmp_path):
    tags = {"agent_host": "192.0.2.9", "sysName": "sw9", "sysLocation": "lab"}
    port = {**tags, "ifIndex": "7", "ifName": "Gi1/0/7"}
    counters = {
        "ifOperStatus": 9,
        "ifHighSpeed": 1000,
        "ifLastChange": 5,
        "ifHCInOctets": 1,
        "ifHCOutOctets": 2,
        "ifInDiscards": 0,
        "ifOutDiscards": 0,
        "dot3StatsFCSErrors": 0,
        "dot3StatsSymbolErrors": 0,
    }
    optic = {
        "entPhysicalSerialNum": "S",
        "entPhysicalDescr": "1000BASE-SX",
        "txPower": -5125,
        "rxPower": -61,
        "entSensorPrecision": 2
    }
    path = _telegraf(tmp_path / "snmp.jsonlines",
                     [
                         {
                             "name": "snmp", "tags": tags, "fields": {
                                 "sysUpTime": 100
                             }, "timestamp": 60
                         },
                         {
                             "name": "interface", "tags": port, "fields": counters, "timestamp": 60
                         },
                         {
                             "name": "transceiver", "tags": port, "fields": optic, "timestamp": 60
                         },
                     ])

    row = collector_schemas.snmp_ports(path, CollectorProfile("c", "tc1.v1", "corp")).iloc[0]

    assert (row["oper_status"], row["link_speed_bps"], row["uptime"]) == ("9", 1_000_000_000, 100)
    assert (row["optical_tx_dbm"], row["optical_rx_dbm"]) == (-51.25, -0.61)
    assert row["lldp_neighbor_chassis_id"] is None and row["lldp_neighbor_port_id"] is None
    assert (row["origin_hash"], row["event_time"]) == ("sw9/Gi1/0/7@60000000000", 60_000_000_000)


def test_a_mac_notification_names_the_port_its_agent_last_mapped(tmp_path):
    agent = {"agent_host": "192.0.2.9"}
    learnt = "01" + "000a" + "aabbcc000009" + "0002" + "00"
    path = _telegraf(tmp_path / "mac.jsonlines",
                     [
                         {
                             "name": "ifXTable",
                             "tags": {
                                 **agent, "ifIndex": "17"
                             },
                             "fields": {
                                 "ifName": "Gi1/0/17"
                             },
                             "timestamp": 0
                         },
                         {
                             "name": "bridge_port",
                             "tags": {
                                 **agent, "index": "2"
                             },
                             "fields": {
                                 "dot1dBasePortIfIndex": 17
                             },
                             "timestamp": 0
                         },
                         {
                             "name": "fdb",
                             "tags": {
                                 **agent, "sysName": "sw9", "sysLocation": "lab", "index": "10.170.187.204.0.0.1"
                             },
                             "fields": {
                                 "dot1qTpFdbPort": 2, "dot1qTpFdbStatus": 3
                             },
                             "timestamp": 0
                         },
                         {
                             "name": "snmp_trap",
                             "tags": {
                                 "source": "192.0.2.9", "name": "cmnMacChangedNotification"
                             },
                             "fields": {
                                 "cmnHistMacChangedMsg": learnt
                             },
                             "timestamp": 5
                         },
                         {
                             "name": "snmp_trap",
                             "tags": {
                                 "source": "192.0.2.8", "name": "cmnMacChangedNotification"
                             },
                             "fields": {
                                 "cmnHistMacChangedMsg": learnt
                             },
                             "timestamp": 6
                         },
                     ])

    frame = collector_schemas.fdb_entries(path, CollectorProfile("c", "tc2_mac.v1", "corp"))

    assert _records(frame, ["switch_id", "port_id", "vlan_id", "mac_address", "mac_action"]) == [
        {
            "switch_id": "sw9",
            "port_id": "Gi1/0/17",
            "vlan_id": 10,
            "mac_address": "aa:bb:cc:00:00:01",
            "mac_action": None
        },
        {
            "switch_id": "sw9",
            "port_id": "Gi1/0/17",
            "vlan_id": 10,
            "mac_address": "aa:bb:cc:00:00:09",
            "mac_action": "added"
        },
        # An agent no walk has answered for: the address and VLAN are the trap's, the switch and port unknown.
        {
            "switch_id": None,
            "port_id": None,
            "vlan_id": 10,
            "mac_address": "aa:bb:cc:00:00:09",
            "mac_action": "added"
        },
    ]


def _detail(path: pathlib.Path, entries: list[list[tuple[str, str]]]) -> str:
    text = "".join("Thu Jan  1 00:00:00 1970\n" + "".join(f"\t{name} = {value}\n" for (name, value) in entry) + "\n"
                   for entry in entries)
    path.write_text(text, encoding="utf-8")

    return str(path)


def _request(user: str, port: str, mac: str, eap: str, when: int) -> list[tuple[str, str]]:
    return [("Packet-Type", "Access-Request"), ("User-Name", f'"{user}"'), ("NAS-Identifier", '"sw9"'),
            ("NAS-Port-Id", f'"{port}"'), ("Calling-Station-Id", f'"{mac}"'), ("EAP-Message", eap),
            ("Timestamp", str(when))]


def test_an_8021x_reply_is_paired_with_the_request_it_answers_not_the_user_s_last(tmp_path):
    identity = "0x0201000801626e63"
    final = "0x020900061900"
    path = _detail(tmp_path / "auth.detail",
                   [
                       _request("bench", "GigabitEthernet1/0/1", "AA-BB-CC-00-00-01", identity, 10),
                       _request("bench", "TenGigabitEthernet1/1/2", "AA-BB-CC-00-00-02", identity, 11),
                       [("Packet-Type", "Access-Challenge"), ("User-Name", '"bench"'), ("Timestamp", "12")],
                       _request("bench", "GigabitEthernet1/0/1", "AA-BB-CC-00-00-01", final, 13),
                       _request("bench", "TenGigabitEthernet1/1/2", "AA-BB-CC-00-00-02", final, 14),
                       [("Packet-Type", "Access-Reject"), ("User-Name", '"bench"'), ("Timestamp", "14")],
                       [("Packet-Type", "Access-Accept"), ("User-Name", '"bench"'), ("Timestamp", "13")],
                   ])

    frame = collector_schemas.radius_auth(path, CollectorProfile("c", "tc2_auth.v1", "corp"), {"sw9": "lab"})

    assert _records(frame, ["event_time", "site_id", "port_id", "mac_address", "dot1x_result"]) == [
        {
            "event_time": 10 * 10**9,
            "site_id": "lab",
            "port_id": "Gi1/0/1",
            "mac_address": "aa:bb:cc:00:00:01",
            "dot1x_result": "started"
        },
        {
            "event_time": 11 * 10**9,
            "site_id": "lab",
            "port_id": "Te1/1/2",
            "mac_address": "aa:bb:cc:00:00:02",
            "dot1x_result": "started"
        },
        {
            "event_time": 14 * 10**9,
            "site_id": "lab",
            "port_id": "Te1/1/2",
            "mac_address": "aa:bb:cc:00:00:02",
            "dot1x_result": "failure"
        },
        {
            "event_time": 13 * 10**9,
            "site_id": "lab",
            "port_id": "Gi1/0/1",
            "mac_address": "aa:bb:cc:00:00:01",
            "dot1x_result": "success"
        },
    ]


def test_an_interim_update_passes_through_and_a_session_time_is_not_a_start(tmp_path):
    path = _detail(tmp_path / "acct.detail",
                   [[("Acct-Status-Type", status), ("Acct-Session-Id", '"s1"'), ("User-Name", '"a@example.com"'),
                     ("Event-Timestamp", '"Jan  2 1970 00:00:00 UTC"'), ("Acct-Session-Time", "60")]
                    for status in ("Interim-Update", "Stop")] + [[("Packet-Type", "Access-Request")]])

    frame = collector_schemas.radius_accounting(path, CollectorProfile("c", "tc5_session.v1", "corp"))

    assert frame["session_action"].tolist() == ["Interim-Update", "end"]
    assert frame["event_time"].tolist() == [86400 * 10**9] * 2
    assert frame["session_start"].isna().all() and frame["source_ip"].isna().all()


def test_the_arp_reader_takes_a_tagged_frame_and_skips_what_is_not_arp(tmp_path):
    body = struct.pack("!HHBBH", 1, 0x0800, 6, 4, 2) + bytes.fromhex("aabbcc000001") + bytes([10, 0, 0, 1])
    body += bytes.fromhex("aabbcc000002") + bytes([10, 0, 0, 2])
    tagged = bytes(6) + bytes.fromhex("aabbcc000001") + bytes.fromhex("8100000a0806") + body
    ipfix.write_frames(str(tmp_path / "arp.pcap"), [(1000, bytes(12) + b"\x08\x00" + bytes(20)), (2000, tagged)])

    frame = collector_schemas.arp_packets(str(tmp_path / "arp.pcap"), CollectorProfile("c", "tc2_arp.v1", "corp"))

    assert _records(frame, ["event_time", "arp_sender_ip", "arp_sender_mac", "arp_target_ip", "arp_operation"]) == [{
        "event_time": 2000,
        "arp_sender_ip": "10.0.0.1",
        "arp_sender_mac": "aa:bb:cc:00:00:01",
        "arp_target_ip": "10.0.0.2",
        "arp_operation": "reply"
    }]


def test_only_a_client_syn_with_options_is_fingerprinted(tmp_path):

    def frame(flags: int, options: bytes, protocol: int = 6) -> bytes:
        transport = (struct.pack("!HHIIBBHHH", 1000, 443, 0, 0, (20 + len(options)) // 4 << 4, flags, 512, 0, 0) +
                     options if protocol == 6 else struct.pack("!HHHH", 1000, 53, 12, 0) + b"abcd")
        ip = struct.pack("!BBHHHBBH4s4s",
                         0x45,
                         0,
                         20 + len(transport),
                         0,
                         0,
                         64,
                         protocol,
                         0,
                         bytes([10, 0, 0, 1]),
                         bytes([10, 0, 0, 2]))
        return bytes(12) + b"\x08\x00" + ip + transport

    mss = b"\x02\x04\x05\xb4"
    ipfix.write_frames(str(tmp_path / "p.pcap"),
                       [(1000, frame(0x02, mss)), (2000, frame(0x12, mss)), (3000, frame(0x02, b"")),
                        (4000, frame(0, b"", protocol=17))])

    rows = _records(collector_schemas.packets(str(tmp_path / "p.pcap"), CollectorProfile("c", "tc4.v1", "corp")),
                    ["protocol", "tcp_flags", "data_len", "tcp_options_order", "tcp_mss", "tcp_window_initial"])

    assert rows == [
        {
            "protocol": "tcp",
            "tcp_flags": 2,
            "data_len": 0,
            "tcp_options_order": "mss",
            "tcp_mss": 1460,
            "tcp_window_initial": 512
        },
        {
            "protocol": "tcp",
            "tcp_flags": 18,
            "data_len": 0,
            "tcp_options_order": None,
            "tcp_mss": None,
            "tcp_window_initial": None
        },
        {
            "protocol": "tcp",
            "tcp_flags": 2,
            "data_len": 0,
            "tcp_options_order": None,
            "tcp_mss": None,
            "tcp_window_initial": None
        },
        {
            "protocol": "udp",
            "tcp_flags": None,
            "data_len": 4,
            "tcp_options_order": None,
            "tcp_mss": None,
            "tcp_window_initial": None
        },
    ]


def test_a_kea_release_is_read_and_its_other_lines_are_not(tmp_path):
    path = tmp_path / "kea.log"
    path.write_text(
        "1970-01-01 00:00:01.500 INFO  [kea-dhcp4.dhcp4/1.2] DHCP4_STARTED Kea DHCPv4 server version 2.6.1 started\n"
        "1970-01-01 00:00:02.000 INFO  [kea-dhcp4.leases/1.2] DHCP4_LEASE_ALLOC [hwtype=1 aa:bb:cc:00:00:01], "
        "cid=[no info], tid=0x1: lease 10.0.0.5 has been allocated for 3600 seconds\n"
        "1970-01-01 00:00:09.250 INFO  [kea-dhcp4.leases/1.2] DHCP4_RELEASE [hwtype=1 aa:bb:cc:00:00:01], "
        "cid=[no info], tid=0x2: address 10.0.0.5 was released properly.\n",
        encoding="utf-8")

    frame = collector_schemas.kea_leases(str(path), CollectorProfile("c", "tc3_dhcp.v1", "corp"))

    assert _records(frame, ["event_time", "ip", "dhcp_message", "lease_seconds"]) == [
        {
            "event_time": 2 * 10**9, "ip": "10.0.0.5", "dhcp_message": "ack", "lease_seconds": 3600
        },
        {
            "event_time": 9_250_000_000, "ip": "10.0.0.5", "dhcp_message": "release", "lease_seconds": None
        },
    ]


def test_osquery_s_added_rows_are_reports_and_its_removed_rows_are_not(tmp_path):
    lines = [{
        "action": "added", "hostIdentifier": "WS-1", "unixTime": 5, "columns": {
            "interface": "en0", "mac": "m1"
        }
    }, {
        "action": "removed", "hostIdentifier": "WS-1", "unixTime": 6, "columns": {
            "interface": "en0", "mac": "m1"
        }
    }]
    (tmp_path / "results.log").write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")

    frame = collector_schemas.osquery_inventory(str(tmp_path / "results.log"),
                                                CollectorProfile("c", "tc7_host_inventory.v1", "corp"),
                                                60)

    assert _records(frame, ["event_time", "mac", "hostname", "report_valid_seconds"]) == [{
        "event_time": 5 * 10**9, "mac": "m1", "hostname": "WS-1", "report_valid_seconds": 60
    }]


def test_a_connection_with_no_handshake_is_not_a_service_record():
    conn = pd.DataFrame({"uid": ["C1", "C2"], "orig_bytes": [1, 2], "resp_bytes": [3, 4]})
    ssl = pd.DataFrame({"uid": ["C2"], "server_name": ["a.example"], "ja4": ["t13"]})

    joined = collector_schemas.join_conn_ssl(conn, ssl)

    assert (joined["uid"].tolist(), joined["server_name"].tolist()) == (["C2"], ["a.example"])


def test_a_cmdb_version_is_one_checkpoint_and_a_retirement_retracts_it():
    audit = [{
        "tablename": "cmdb_ci_server",
        "documentkey": "d",
        "fieldname": name,
        "newvalue": value,
        "record_checkpoint": checkpoint,
        "sys_created_on": when
    } for (name, value, checkpoint, when) in [("name", "db", "0", "1970-01-01 00:00:01"), (
        "assigned_to", "u-gone", "0", "1970-01-01 00:00:01"), ("os", "Linux", "1", "1970-01-01 00:00:05"),
                                              ("install_status", "7", "2", "1970-01-01 00:00:09")]] + [{
                                                  "tablename": "cmdb_ci_appl",
                                                  "documentkey": "a",
                                                  "fieldname": "name",
                                                  "newvalue": "app",
                                                  "record_checkpoint": "0",
                                                  "sys_created_on": "1970-01-01 00:00:02"
                                              }]

    frame = collector_schemas.normalize(collector_schemas.fold_cmdb_audit(audit, [], [], ["cmdb_ci_server"]),
                                        collector_schemas.servicenow_asset_schema("corp"))

    assert _records(frame, ["source_seq", "hostname", "owner", "os_family", "recorded_at", "change"]) == [
        {
            "source_seq": 0,
            "hostname": "db",
            "owner": None,
            "os_family": None,
            "recorded_at": 10**9,
            "change": "assert"
        },
        {
            "source_seq": 1,
            "hostname": "db",
            "owner": None,
            "os_family": "linux",
            "recorded_at": 5 * 10**9,
            "change": "assert"
        },
        {
            "source_seq": 2,
            "hostname": "db",
            "owner": None,
            "os_family": "linux",
            "recorded_at": 9 * 10**9,
            "change": "retract"
        },
    ]


def test_a_deleted_netbox_interface_retracts_what_it_was_and_takes_its_device_s_owner():

    def change(action: str, kind: str, identifier: int, data: dict) -> dict:
        return {
            "time": "1970-01-01T00:00:01.000000Z",
            "action": {
                "value": action
            },
            "changed_object_type": kind,
            "changed_object_id": identifier,
            "prechange_data": data if action == "delete" else None,
            "postchange_data": None if action == "delete" else data
        }

    port = {"name": "Gi1/0/1", "device": 1, "description": "access", "custom_fields": {"criticality": "low"}}
    changes = [
        change("create", "dcim.site", 1, {"slug": "lab"}),
        change("create", "tenancy.tenant", 1, {"slug": "ops"}),
        change("create", "dcim.device", 1, {
            "name": "sw9", "site": 1, "tenant": 1
        }),
        change("create", "dcim.interface", 1, port),
        change("delete", "dcim.interface", 1, port)
    ]

    frame = collector_schemas.normalize(collector_schemas.fold_netbox_changes(changes),
                                        collector_schemas.netbox_network_schema("corp"))

    assert _records(frame, ["object_kind", "site_id", "device_id", "port_id", "owner", "change"])[1:] == [
        {
            "object_kind": "port",
            "site_id": "lab",
            "device_id": "sw9",
            "port_id": "Gi1/0/1",
            "owner": "ops",
            "change": "assert"
        },
        {
            "object_kind": "port",
            "site_id": "lab",
            "device_id": "sw9",
            "port_id": "Gi1/0/1",
            "owner": "ops",
            "change": "retract"
        },
    ]


def test_identity_is_ordered_by_entry_hr_first_and_an_unentered_change_last():
    workers = [{
        "Worker_Email": "B@example.com", "Effective_Moment": "1970-01-01T00:00:00.000Z", "Worker_Status": "Active"
    },
               {
                   "Worker_Email": "a@example.com",
                   "Effective_Moment": "1970-01-01T00:00:00.000Z",
                   "Entry_Moment": "1970-01-01T00:00:10.000Z",
                   "Worker_Type": "Contingent Worker",
                   "Worker_Status": "Active",
                   "Business_Process_Type": "End Contingent Worker Contract"
               }]
    audit = [{
        "activityDisplayName":
            "Remove member from group",
        "result":
            "success",
        "activityDateTime":
            "1970-01-01T00:00:10Z",
        "targetResources": [{
            "type": "User",
            "userPrincipalName": "a@example.com",
            "modifiedProperties": [{
                "displayName": "Group.DisplayName", "oldValue": "\"eng\"", "newValue": None
            }]
        }]
    },
             {
                 "activityDisplayName": "Update user",
                 "result": "success",
                 "activityDateTime": "1970-01-01T00:00:01Z",
                 "targetResources": []
             }]

    frame = collector_schemas.normalize(collector_schemas.merge_identity(workers, audit),
                                        collector_schemas.workday_entra_identity_schema("corp"))

    assert _records(frame, ["user_principal", "employment_status", "group_name", "recorded_at", "change"]) == [
        {
            "user_principal": "a@example.com",
            "employment_status": "contractor",
            "group_name": None,
            "recorded_at": 10**10,
            "change": "retract"
        },
        {
            "user_principal": "a@example.com",
            "employment_status": None,
            "group_name": "eng",
            "recorded_at": 10**10,
            "change": "retract"
        },
        {
            "user_principal": "b@example.com",
            "employment_status": "active",
            "group_name": None,
            "recorded_at": None,
            "change": "assert"
        },
    ]


SEGMENT_CHECKS = {
    "network": ["tc3", "tc3_dhcp", "tc7_host_inventory"],
    "transport": ["tc4"],
    "session": ["tc5_auth"],
    "application": ["tc7_http", "tc7_dns", "tc7_service"],
    "saas": ["tc7_saas"],
    "endpoint": ["tc7_endpoint"],
}
"""Segments whose every normalized class carries all its columns, so their output can be asked to be identical.
The presentation segment reads the two content columns Zeek cannot supply, and its content rule would differ."""


@pytest.mark.slow
@pytest.mark.parametrize("segment", sorted(SEGMENT_CHECKS))
def test_a_segment_over_the_normalized_feed_writes_what_it_writes_from_the_sample(segment: str,
                                                                                  normalized: dict,
                                                                                  tmp_path):
    import segment_runner  # pylint: disable=import-outside-toplevel

    inputs = tmp_path / "inputs"
    shutil.copytree(os.path.join(EXAMPLE, "samples", segment), inputs)

    for telemetry_class in SEGMENT_CHECKS[segment]:
        frame = normalized[telemetry_class]
        frame = frame[[
            column for column in frame.columns
            if not frame[column].isna().all() or column in make_fixtures.sample(segment, telemetry_class)[0]
        ]]
        rows = _records(frame, list(frame.columns))
        (inputs / f"{telemetry_class}.jsonlines").write_text("".join(json.dumps(row) + "\n" for row in rows),
                                                             encoding="utf-8")

    segment_runner.run(segment, inputs, output_dir=tmp_path / "from_fixture")
    segment_runner.run(segment, pathlib.Path(EXAMPLE) / "samples" / segment, output_dir=tmp_path / "from_sample")

    for path in sorted((tmp_path / "from_sample").glob("*.jsonlines")):
        assert (tmp_path / "from_fixture" / path.name).read_text(encoding="utf-8") == path.read_text(
            encoding="utf-8"), path.name
