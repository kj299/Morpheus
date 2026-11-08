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
Render committed input samples in the formats real feeds write, and read them back.

Each fixture is the same records as `../samples/<segment>/<class>.jsonlines`, written the way the source writes
them: a packet capture of IPFIX from three routers, Entra ID sign-in records as Azure Monitor exports them, Zeek
`ssl.log` and `x509.log`, `http.log` and `dns.log`, unified audit log records, Sysmon events as `wevtutil`
renders them, what two Telegraf SNMP pollers write of their ports and devices, a forwarding-table walk with its
MAC notification traps, a capture of ARP packets, and FreeRADIUS detail files of 802.1X exchanges and of VPN
accounting. Only what the format carries is written; a collector's own facts are the `PROFILES` below.
`normalize_all()` reads them back with `FlowRecordNormalizeStage` and the schemas in
`morpheus.utils.collector_schemas`, and `tests/morpheus/utils/test_collector_schemas.py` holds the result to the
samples, and the segments run over it to what they write from the samples.

Run it after regenerating the samples; the test fails while the fixtures are stale.
"""

import hashlib
import ipaddress
import json
import os
import struct
import sys
import typing
import uuid
from xml.sax.saxutils import escape
from xml.sax.saxutils import quoteattr

import pandas as pd

from morpheus.utils import collector_schemas
from morpheus.utils import ipfix
from morpheus.utils.collector_schemas import CollectorProfile

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(os.path.dirname(HERE), "samples")

EXPORTERS = {"192.0.2.11": "rtr-access-1", "192.0.2.12": "rtr-dc-1", "192.0.2.13": "rtr-edge-1"}
"""Exporter address to name: the inventory a deployment gives `FlowRecordNormalizeStage`."""

COLLECTOR_ADDRESS = "192.0.2.100"
RESOLVER = "10.0.0.53"

ORGANIZATIONS = {"4f5e2b0a-3c1d-4e8f-9a7b-6c5d4e3f2a10": "corp", "9b8a7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d": "labs"}
"""`OrganizationId` to tenant name: what the M365 schema is given."""

PROFILES = {
    "tc3": CollectorProfile("netflow-01", "tc3.v1", "corp", "ntp_disciplined", 17),
    "tc5_auth": CollectorProfile("idp", "tc5_auth.v1", "corp", "ntp_disciplined", -4, first_seq=1),
    "tc6": CollectorProfile("tls-inspect-01", "tc6.v1", "corp", "ptp", -18),
    "tc7_http": CollectorProfile("proxy-01", "tc7_http.v1", "corp", "ntp_disciplined", -15),
    "tc7_dns": CollectorProfile("resolver-01", "tc7_dns.v1", "corp", "ntp_disciplined", 8),
    "tc7_saas": CollectorProfile("saas-audit-01", "tc7_saas.v1", "corp", "ntp_disciplined", -18),
    "tc7_endpoint": CollectorProfile("edr-01", "tc7_endpoint.v1", "corp", "ntp_disciplined", -4),
    "tc1": CollectorProfile("snmp-poller", "tc1.v1", "corp", "ntp_disciplined", 6.0, first_seq=1),
    "tc1_device": CollectorProfile("snmp-poller", "tc1_device.v1", "corp", "ntp_disciplined", 6, first_seq=1),
    "tc2_mac": CollectorProfile("mac-table", "tc2_mac.v1", "corp", "ntp_disciplined", 3, first_seq=1),
    "tc2_arp": CollectorProfile("arp-sensor", "tc2_arp.v1", "corp", "ntp_disciplined", 16, first_seq=1),
    "tc2_auth": CollectorProfile("radius", "tc2_auth.v1", "corp", "ntp_disciplined", 0, first_seq=1),
    "tc5_session": CollectorProfile("radius-accounting", "tc5_session.v1", "corp", "ntp_disciplined", -3, first_seq=1),
    "tc4": CollectorProfile("pcap-01", "tc4.v1", "corp", "ptp", 12),
    "tc3_dhcp": CollectorProfile("dhcp-01", "tc3_dhcp.v1", "corp", "ntp_disciplined", -10),
    "tc7_host_inventory": CollectorProfile("edr-inventory", "tc7_host_inventory.v1", "corp", "ntp_disciplined", -9),
    "tc7_service": CollectorProfile("lb-01", "tc7_service.v1", "corp", "ntp_disciplined", 19),
}
"""What each collector says about itself, which its records do not."""

DMZ_POLLER = CollectorProfile("snmp-poller-dmz", "tc1.v1", "corp", "ntp_undisciplined", 4500.0, first_seq=1)
"""The second collector of `tc1`: the DMZ's poller, whose undisciplined clock runs 4.5 seconds off and which the corpus
quarantines."""

CLASSES = {
    "tc3": ("network", ["tc3.pcap"]),
    "tc5_auth": ("session", ["signins.jsonlines"]),
    "tc6": ("presentation", ["ssl.log", "x509.log"]),
    "tc7_http": ("application", ["http.log"]),
    "tc7_dns": ("application", ["dns.log"]),
    "tc7_saas": ("saas", ["audit.jsonlines"]),
    "tc7_endpoint": ("endpoint", ["sysmon.xml"]),
    "tc1": ("telemetry", ["snmp-poller.jsonlines", "snmp-poller-dmz.jsonlines"]),
    "tc1_device": ("telemetry", ["snmp-poller.jsonlines"]),
    "tc2_mac": ("telemetry", ["mac-table.jsonlines"]),
    "tc2_arp": ("telemetry", ["arp.pcap"]),
    "tc2_auth": ("telemetry", ["radius-auth.detail"]),
    "tc5_session": ("session", ["radius-acct.detail"]),
    "tc4": ("transport", ["tc4.pcap"]),
    "tc3_dhcp": ("network", ["kea-dhcp4.log"]),
    "tc7_host_inventory": ("network", ["osqueryd.results.log"]),
    "tc7_service": ("application", ["conn.log", "ssl.log"]),
}
"""Each class's segment and the files its fixture is."""

FLOW_FIELDS = ("flowStartMilliseconds",
               "sourceIPv4Address",
               "sourceTransportPort",
               "destinationIPv4Address",
               "destinationTransportPort",
               "protocolIdentifier",
               "minimumTTL",
               "octetDeltaCount",
               "reverseOctetDeltaCount",
               "bgpDestinationAsNumber")
PROTOCOL_NUMBERS = {"icmp": 1, "tcp": 6, "udp": 17}
QTYPES = {"A": 1, "AAAA": 28, "CNAME": 5, "MX": 15, "TXT": 16}
RCODES = {"NOERROR": 0, "SERVFAIL": 2, "NXDOMAIN": 3, "REFUSED": 5}
STATUS_MESSAGES = {
    200: "OK",
    204: "No Content",
    301: "Moved Permanently",
    302: "Found",
    304: "Not Modified",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    500: "Internal Server Error"
}
VALIDATION_STRINGS = {value: key for (key, value) in collector_schemas.VALIDATION_RESULTS.items()}
MFA_REQUIREMENTS = {value: key for (key, value) in collector_schemas.ENTRA_MFA_REQUIREMENTS.items()}
TOKEN_TYPES = {"bearer": "none", "refresh": "primaryRefreshToken"}
"""The contract's token types as Entra ID names them; any other value is the provider's own and passes through."""
ENTRA_TENANT = "4f5e2b0a-3c1d-4e8f-9a7b-6c5d4e3f2a10"
ENTRA_FAILURES = {
    # The code Entra ID writes for each way a sign-in fails: the second factor refused, no second factor and the
    # password wrong, or every factor passed and a policy refused the token. The corpus holds the first two.
    "denied": (500121, "Authentication failed during strong authentication request."),
    None: (50126, "Error validating credentials due to invalid username or password."),
    "approved": (53003,
                 "Access has been blocked by Conditional Access policies. The access policy does not allow "
                 "token issuance."),
}
TLS_NAMES = {value: key for (key, value) in collector_schemas.TLS_VERSIONS.items()}
SYSMON_PROVIDER = "<Provider Name='Microsoft-Windows-Sysmon' Guid='{5770385f-c22a-43e0-bf4c-06f5698ffbd9}'/>"


def sample(segment: str, telemetry_class: str) -> list[dict]:
    """The committed sample's records."""
    with open(os.path.join(SAMPLES, segment, f"{telemetry_class}.jsonlines"), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _digest(*parts: typing.Any, length: int = 16) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()[:length]


def _port(*parts: typing.Any) -> int:
    return 49152 + int(_digest(*parts, length=8), 16) % 16384


def _seconds(ns: int) -> str:
    if (ns % 1000 != 0):
        raise ValueError(f"{ns} ns is finer than the microsecond Zeek writes")

    return f"{ns / 1e9:.6f}"


def _zeek_log(path: str, name: str, fields: list[tuple[str, str]], rows: list[list]):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#separator \\x09\n#set_separator\t,\n#empty_field\t(empty)\n#unset_field\t-\n")
        handle.write(f"#path\t{name}\n#open\t1970-01-01-00-00-00\n")
        handle.write("#fields\t" + "\t".join(field for (field, _) in fields) + "\n")
        handle.write("#types\t" + "\t".join(kind for (_, kind) in fields) + "\n")

        for row in rows:
            handle.write("\t".join("-" if value is None else str(value) for value in row) + "\n")

        handle.write("#close\t1970-01-01-00-00-00\n")


def write_flows(directory: str):
    """`tc3.pcap`: one IPFIX message per flow, each exporter announcing its template first."""
    names = {name: address for (address, name) in EXPORTERS.items()}
    fields = ipfix.template_for(FLOW_FIELDS)
    announced = set()
    sequences: dict = {}
    datagrams = []

    for row in sample("network", "tc3"):
        (name, domain) = row["exporter_id"].rsplit(":", 1)
        (address, domain) = (names[name], int(domain))
        record = {
            "flowStartMilliseconds": row["event_time"] // 1_000_000,
            "sourceIPv4Address": row["src_ip"],
            "sourceTransportPort": row["src_port"],
            "destinationIPv4Address": row["dst_ip"],
            "destinationTransportPort": row["dst_port"],
            "protocolIdentifier": PROTOCOL_NUMBERS[row["protocol"]],
            "minimumTTL": row["ip_ttl"],
            "octetDeltaCount": row["bytes_out"],
            "reverseOctetDeltaCount": row["bytes_in"],
            # An exporter with no BGP route to the destination reports AS 0 (RFC 7607).
            "bgpDestinationAsNumber": 0 if row["bgp_as_dst"] is None else int(row["bgp_as_dst"]),
        }

        if (row["event_time"] % 1_000_000 != 0):
            raise ValueError("flowStartMilliseconds cannot carry a sub-millisecond start")

        key = (address, domain)
        message = ipfix.encode_message(domain,
                                       row["event_time"] // 1_000_000_000,
                                       sequences.get(key, 0),
                                       256,
                                       fields, [record],
                                       announce=key not in announced)
        announced.add(key)
        sequences[key] = sequences.get(key, 0) + 1
        datagrams.append((row["observed_time"], address, COLLECTOR_ADDRESS, message))

    ipfix.write_pcap(os.path.join(directory, "tc3.pcap"), datagrams)


def _place_name(slug: str) -> str:
    """A place as Entra ID writes it from the corpus's slug: `north-holland` to `North Holland`, `ny` to `NY`."""
    return " ".join(word.upper() if len(word) <= 2 else word.capitalize() for word in slug.split("-"))


def write_signins(directory: str):
    """
    `signins.jsonlines`: sign-in records as Azure Monitor exports them, each with its `properties`.

    The record's `id` is the corpus's `origin_hash`, since the schema takes a sign-in's own identifier as its
    origin; Entra ID writes a GUID there, and the schema passes either through. Case and separators are the
    provider's: the principal's first letter upper, places as names, and the country as the alpha-2 code upper.
    """
    with open(os.path.join(directory, "signins.jsonlines"), "w", encoding="utf-8") as handle:
        for row in sample("session", "tc5_auth"):
            if (row["event_time"] % 1_000_000_000 != 0):
                raise ValueError("the fixture writes createdDateTime to the second")

            created = pd.Timestamp(row["event_time"], unit="ns", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            principal = row["user_principal"]
            (local, _) = principal.split("@", 1)
            second_factor = row["mfa_result"]
            succeeded = row["auth_result"] == "success"
            (code, reason) = (0, "") if succeeded else ENTRA_FAILURES[second_factor]
            steps = [{"authenticationMethod": "Password", "succeeded": succeeded or second_factor is not None}]

            if (row["mfa_used"]):
                steps.append({
                    "authenticationMethod": "Mobile app notification", "succeeded": second_factor == "approved"
                })

            properties = {
                "id": row["origin_hash"],
                "createdDateTime": created,
                "userDisplayName": _place_name(local),
                "userPrincipalName": principal[0].upper() + principal[1:],
                "appId": str(uuid.uuid5(uuid.NAMESPACE_URL, f"app:{row['app']}")),
                "appDisplayName": row["app"],
                "ipAddress": row["source_ip"],
                "clientAppUsed": "Browser",
                "correlationId": str(uuid.uuid5(uuid.NAMESPACE_URL, row["origin_hash"])),
                "isInteractive": True,
                "riskState": "none",
                "resourceDisplayName": row["app"],
                "resourceTenantId": ENTRA_TENANT,
                "homeTenantId": ENTRA_TENANT,
                "status": {
                    "errorCode": code, "failureReason": reason, "additionalDetails": ""
                },
                "deviceDetail": {
                    "deviceId": row["device_id"],
                    "displayName": row["device_id"],
                    "operatingSystem": "Windows10",
                    "browser": "Edge 120.0.0",
                },
                "location": {
                    "city": _place_name(row["source_city"]),
                    "state": _place_name(row["source_region"]),
                    "countryOrRegion": row["source_country"].upper(),
                    "geoCoordinates": {
                        "latitude": row["source_latitude"], "longitude": row["source_longitude"]
                    },
                },
                "autonomousSystemNumber": int(row["source_asn"].removeprefix("as")),
                "authenticationRequirement": MFA_REQUIREMENTS[row["mfa_used"]],
                "authenticationDetails": steps,
                "mfaDetail": {
                    "authMethod": "Mobile app notification"
                } if row["mfa_used"] else {},
                "incomingTokenType": TOKEN_TYPES.get(row["token_type"], row["token_type"]),
                "tokenIssuerType": "AzureAD",
            }
            record = {
                "time": created,
                "resourceId": f"/tenants/{ENTRA_TENANT}/providers/Microsoft.aadiam",
                "operationName": "Sign-in activity",
                "operationVersion": "1.0",
                "category": "SignInLogs",
                "resultType": str(code),
                "resultSignature": "None",
                "durationMs": 0,
                "callerIpAddress": row["source_ip"],
                "correlationId": properties["correlationId"],
                "identity": properties["userDisplayName"],
                "Level": 4,
                "location": properties["location"]["countryOrRegion"],
                "properties": properties,
            }

            handle.write(json.dumps(record) + "\n")


def write_tls(directory: str):
    """
    `ssl.log` and `x509.log`, one certificate file per session, joined on its file id.

    The corpus gives one fingerprint a different validity period in different sessions, which no real
    certificate has, so the fixture is written the way a Zeek that logs every session's certificate writes it --
    `cert_chain_fuids` against the x509 `id` -- rather than deduplicated by fingerprint, which would keep one period.
    """
    ssl_rows = []
    certificates = []

    for row in sample("presentation", "tc6"):
        fingerprint = row["certificate_fingerprint_sha256"].removeprefix("sha256:")
        file_id = "F" + _digest("x509", row["origin_hash"], row["event_time"])
        ssl_rows.append([
            _seconds(row["event_time"]),
            "C" + _digest("ssl", row["origin_hash"], row["event_time"]),
            row["src_ip"],
            row["src_port"],
            row["dst_ip"],
            row["dst_port"],
            TLS_NAMES[row["tls_version"]],
            row["cipher_suite"],
            row["sni"],
            VALIDATION_STRINGS[row["validation_result"]],
            file_id,
            fingerprint,
            row["ja4_client"]
        ])
        certificates.append([
            _seconds(row["event_time"]),
            file_id,
            fingerprint,
            3,
            _digest("serial", fingerprint, length=20).upper(),
            f"CN={row['sni'] or row['dst_ip']}",
            row["certificate_issuer"],
            _seconds(row["certificate_not_before"]),
            _seconds(row["certificate_not_after"]),
            "T"
        ])

    _zeek_log(os.path.join(directory, "ssl.log"),
              "ssl",
              [("ts", "time"), ("uid", "string"), ("id.orig_h", "addr"), ("id.orig_p", "port"), ("id.resp_h", "addr"),
               ("id.resp_p", "port"), ("version", "string"), ("cipher", "string"), ("server_name", "string"),
               ("validation_status", "string"), ("cert_chain_fuids", "vector[string]"),
               ("cert_chain_fps", "vector[string]"), ("ja4", "string")],
              ssl_rows)
    _zeek_log(os.path.join(directory, "x509.log"),
              "x509",
              [("ts", "time"), ("id", "string"), ("fingerprint", "string"), ("certificate.version", "count"),
               ("certificate.serial", "string"), ("certificate.subject", "string"), ("certificate.issuer", "string"),
               ("certificate.not_valid_before", "time"), ("certificate.not_valid_after", "time"),
               ("host_cert", "bool")],
              certificates)


def write_http(directory: str):
    """`http.log`. Every seventh request carries a query string, which the schema has to strip.

    A posted body is logged as `post_body`, the field Zeek's HTTP post-body package adds; stock Zeek logs only its
    length, and the schema reads a missing field as no body.
    """
    rows = []

    for (index, row) in enumerate(sample("application", "tc7_http")):
        uri = row["url_path"] + (f"?ref={_digest('ref', index, length=6)}" if index % 7 == 0 else "")
        rows.append([
            _seconds(row["event_time"]),
            "C" + _digest("http", row["origin_hash"]),
            row["src_ip"],
            _port("http", row["origin_hash"]),
            row["dst_ip"],
            row["dst_port"],
            1,
            row["http_method"],
            row["dst_ip"],
            uri,
            "1.1",
            row["user_agent"],
            len(row.get("request_body") or ""),
            0,
            row["status_code"],
            STATUS_MESSAGES.get(row["status_code"]),
            row.get("request_body")
        ])

    _zeek_log(os.path.join(directory, "http.log"),
              "http",
              [("ts", "time"), ("uid", "string"), ("id.orig_h", "addr"), ("id.orig_p", "port"), ("id.resp_h", "addr"),
               ("id.resp_p", "port"), ("trans_depth", "count"), ("method", "string"), ("host", "string"),
               ("uri", "string"), ("version", "string"), ("user_agent", "string"), ("request_body_len", "count"),
               ("response_body_len", "count"), ("status_code", "count"), ("status_msg", "string"),
               ("post_body", "string")],
              rows)


def write_dns(directory: str):
    """`dns.log`, every query to the estate's resolver."""
    rows = []

    for row in sample("application", "tc7_dns"):
        rcode = row["response_code"]
        rows.append([
            _seconds(row["event_time"]),
            "C" + _digest("dns", row["origin_hash"]),
            row["src_ip"],
            _port("dns", row["origin_hash"]),
            RESOLVER,
            53,
            "udp",
            int(_digest("id", row["origin_hash"], length=4), 16),
            row["query_name"],
            1,
            "C_INTERNET",
            QTYPES[row["query_type"]],
            row["query_type"],
            RCODES[rcode],
            rcode,
            "F",
            "F",
            "T",
            "T",
            0,
            None,
            None,
            "F"
        ])

    _zeek_log(os.path.join(directory, "dns.log"),
              "dns",
              [("ts", "time"), ("uid", "string"), ("id.orig_h", "addr"), ("id.orig_p", "port"), ("id.resp_h", "addr"),
               ("id.resp_p", "port"), ("proto", "enum"), ("trans_id", "count"), ("query", "string"),
               ("qclass", "count"), ("qclass_name", "string"), ("qtype", "count"), ("qtype_name", "string"),
               ("rcode", "count"), ("rcode_name", "string"), ("AA", "bool"), ("TC", "bool"), ("RD", "bool"),
               ("RA", "bool"), ("Z", "count"), ("answers", "vector[string]"), ("TTLs", "vector[interval]"),
               ("rejected", "bool")],
              rows)


def write_audit(directory: str):
    """`audit.jsonlines`: unified audit log records as the Management Activity API returns them."""
    organizations = {tenant: organization for (organization, tenant) in ORGANIZATIONS.items()}

    with open(os.path.join(directory, "audit.jsonlines"), "w", encoding="utf-8") as handle:
        for row in sample("saas", "tc7_saas"):
            if (row["event_time"] % 1_000_000_000 != 0):
                raise ValueError("CreationTime carries whole seconds")

            created = pd.Timestamp(row["event_time"], unit="ns", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S")
            record = {
                "CreationTime": created,
                "Id": str(uuid.uuid5(uuid.NAMESPACE_URL, row["origin_hash"])),
                "Operation": row["operation"],
                "OrganizationId": organizations[row["tenant_id"]],
                "UserKey": _digest("key", row["user_principal"]),
                "UserType": 0,
                "Version": 1,
                "Workload": "CRM",
                "UserId": row["user_principal"],
                "ObjectId": row["target_object"],
                "ItemType": row["target_object_type"],
                # The corpus's one refused export; a workload reports a refusal in its own word, passed through.
                "ResultStatus": "Succeeded" if row["result"] == "success" else row["result"].capitalize(),
                "ClientInfoString": row["client_app"],
                "RecordCount": row["record_count"],
            }

            if (row.get("source_ip") is not None):
                record["ClientIP"] = row["source_ip"]

            handle.write(json.dumps(record) + "\n")


def _sysmon_event(event_id: int, task: int, record_id: int, computer: str, when_ns: int, data: dict) -> str:
    if (when_ns % 1_000_000 != 0):
        raise ValueError("Sysmon's UtcTime carries milliseconds")

    stamp = pd.Timestamp(when_ns, unit="ns", tz="UTC")
    utc = stamp.strftime("%Y-%m-%d %H:%M:%S.") + f"{stamp.microsecond // 1000:03d}"
    fields = "".join(f"<Data Name={quoteattr(name)}>{escape(str(value))}</Data>" for (name, value) in data.items()
                     if value is not None)

    return ("<Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'><System>" + SYSMON_PROVIDER +
            f"<EventID>{event_id}</EventID><Version>5</Version><Level>4</Level><Task>{task}</Task>"
            "<Opcode>0</Opcode><Keywords>0x8000000000000000</Keywords>"
            f"<TimeCreated SystemTime='{stamp.strftime('%Y-%m-%dT%H:%M:%S.%f')}0Z'/>"
            f"<EventRecordID>{record_id}</EventRecordID><Correlation/>"
            "<Execution ProcessID='3204' ThreadID='4200'/><Channel>Microsoft-Windows-Sysmon/Operational</Channel>"
            f"<Computer>{escape(computer)}</Computer><Security UserID='S-1-5-18'/></System>"
            f"<EventData><Data Name='RuleName'>-</Data><Data Name='UtcTime'>{utc}</Data>{fields}</EventData>"
            "</Event>")


def write_sysmon(directory: str):
    """`sysmon.xml`: each process start, then the load of its own image, which carries the signature check."""
    events = []

    for row in sample("endpoint", "tc7_endpoint"):
        command_line = f'"{row["image_path"]}"'

        if (hashlib.sha256(command_line.encode()).hexdigest() != row["command_line_hash"]):
            raise ValueError(f"the sample's command line for {row['origin_hash']} is not the quoted image")

        process_id = int(_digest("pid", row["process_guid"], length=4), 16)
        events.append(
            _sysmon_event(
                1,
                1,
                len(events) + 1,
                row["hostname"],
                row["event_time"],
                {
                    "ProcessGuid": row["process_guid"],
                    "ProcessId": process_id,
                    "Image": row["image_path"],
                    "CommandLine": command_line,
                    "CurrentDirectory": "C:\\Windows\\system32\\",
                    "User": "NT AUTHORITY\\SYSTEM",
                    "IntegrityLevel": row["integrity_level"],
                    "ParentProcessGuid": row["parent_process_guid"],
                    "ParentImage": row["parent_image_path"],
                }))
        events.append(
            _sysmon_event(
                7,
                7,
                len(events) + 1,
                row["hostname"],
                row["event_time"],
                {
                    "ProcessGuid": row["process_guid"],
                    "ProcessId": process_id,
                    "Image": row["image_path"],
                    "ImageLoaded": row["image_path"],
                    "Signed": "true" if row["signature_status"] == "Valid" else "false",
                    "SignatureStatus": row["signature_status"],
                }))

    with open(os.path.join(directory, "sysmon.xml"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(events) + "\n")


AGENTS = {"sw1": "192.0.2.21", "sw2": "192.0.2.22", "sw3": "192.0.2.23", "sw5": "192.0.2.25", "rtr1": "192.0.2.31"}
"""Each device's management address, which Telegraf tags its records with as `agent_host`."""

SWITCH_SITES = {"sw1": "hq", "sw2": "hq", "sw3": "hq"}
"""`NAS-Identifier` to site: what the 802.1X schema is given, since RADIUS carries no location."""

VPN_GATEWAY = ("192.0.2.41", "vpn-gw-1")
"""The NAS whose accounting the session fixture is: its address and `NAS-Identifier`."""

MEMORY_UNIT = 1 << 20
"""The memory pool's size is a hundred of these, so a pool's used bytes over its total is the sample's fraction."""

OPTICAL_PRECISION = 3
FDB_LEARNED = 3
"""Q-BRIDGE-MIB `dot1qTpFdbStatus` for an address the bridge learnt."""

OPER_STATUS_CODES = {value: key for (key, value) in collector_schemas.SNMP_OPER_STATUS.items()}
NOTIFICATION_CODES = {value: key for (key, value) in collector_schemas.MAC_NOTIFICATION_OPERATIONS.items()}
ARP_CODES = {value: key for (key, value) in collector_schemas.ARP_OPERATIONS.items()}
OUTCOME_PACKETS = {value: key for (key, value) in collector_schemas.RADIUS_OUTCOMES.items()}
ACCOUNTING_TYPES = {value: key for (key, value) in collector_schemas.RADIUS_ACCOUNTING_ACTIONS.items()}
LONG_INTERFACE_NAMES = {value: key for (key, value) in collector_schemas.INTERFACE_ABBREVIATIONS.items()}
BROADCAST = "ff:ff:ff:ff:ff:ff"
NO_ADDRESS = "00:00:00:00:00:00"


def _whole(value: float, scale: int, what: str) -> int:
    scaled = round(value * scale)

    if (scaled / scale != value):
        raise ValueError(f"{what} {value} is not a whole number of 1/{scale}")

    return scaled


def _poll_seconds(ns: int) -> int:
    if (ns % 1_000_000_000 != 0):
        raise ValueError(f"{ns} ns is finer than the second Telegraf's JSON timestamp carries")

    return ns // 1_000_000_000


def _metric(name: str, tags: dict, fields: dict, when_ns: int) -> str:
    return json.dumps({
        "fields": fields, "name": name, "tags": tags, "timestamp": _poll_seconds(when_ns)
    },
                      sort_keys=True)


def _port_lines(row: dict, tags: dict) -> list[str]:
    port_tags = {**tags, "ifIndex": str(row["if_index"]), "ifName": row["port_id"]}
    lines = [
        _metric(
            "interface",
            port_tags,
            {
                "ifOperStatus": OPER_STATUS_CODES[row["oper_status"]],
                "ifHighSpeed": _whole(row["link_speed_bps"], 1, "a link speed") // 1_000_000,
                "ifLastChange": row["if_last_change"],
                "ifHCInOctets": row["if_hc_in_octets"],
                "ifHCOutOctets": row["if_hc_out_octets"],
                "ifInDiscards": row["input_discards"],
                "ifOutDiscards": row["output_discards"],
                "dot3StatsFCSErrors": row["crc_errors"],
                "dot3StatsSymbolErrors": row["symbol_errors"],
            },
            row["event_time"]),
        _metric(
            "transceiver",
            port_tags,
            {
                "entPhysicalSerialNum": row["transceiver_serial"],
                "entPhysicalDescr": row["transceiver_type"],
                "txPower": _whole(row["optical_tx_dbm"], 10**OPTICAL_PRECISION, "a transmit power"),
                "rxPower": _whole(row["optical_rx_dbm"], 10**OPTICAL_PRECISION, "a receive power"),
                "entSensorPrecision": OPTICAL_PRECISION,
            },
            row["event_time"]),
    ]

    if (row["lldp_neighbor_chassis_id"] is not None):
        lines.append(
            _metric(
                "lldp",
                port_tags, {
                    "lldpRemChassisId": row["lldp_neighbor_chassis_id"], "lldpRemPortId": row["lldp_neighbor_port_id"]
                },
                row["event_time"]))

    if (row["link_speed_bps"] % 1_000_000 != 0):
        raise ValueError(f"{row['link_speed_bps']} b/s is not a whole number of the Mb/s ifHighSpeed carries")

    return lines


def write_snmp(directory: str):
    """
    `snmp-poller.jsonlines` and `snmp-poller-dmz.jsonlines`: what two Telegraf `inputs.snmp` agents write, through
    the JSON serializer, each poll.

    The estate's poller walks every device's scalars and BGP peers and the ports it polls; the DMZ's poller, the
    one the corpus quarantines, polls one port and reads only `sysUpTime` besides. Each port is four records of
    one poll, as `join_snmp_ports` reads them back.
    """
    ports = sample("telemetry", "tc1")
    devices = sample("telemetry", "tc1_device")
    by_poll: dict = {}

    for row in ports:
        by_poll.setdefault((row["collector_id"], row["device_id"], row["event_time"]), []).append(row)

    main = []
    written = []

    for device in devices:
        key = (device["collector_id"], device["device_id"], device["event_time"])
        tags = {
            "agent_host": AGENTS[device["device_id"]],
            "host": device["collector_id"],
            "sysLocation": device["site_id"],
            "sysName": device["device_id"]
        }
        used = _whole(device["mem_util"], 100, "a memory fraction")
        main.append(
            _metric(
                "snmp",
                tags,
                {
                    "sysUpTime": device["sys_uptime"],
                    "cpmCPUTotal5minRev": _whole(device["cpu_util"], 100, "a CPU fraction"),
                    "ciscoMemoryPoolUsed": used * MEMORY_UNIT,
                    "ciscoMemoryPoolFree": (100 - used) * MEMORY_UNIT,
                    "inetCidrRouteNumber": device["route_count"],
                },
                device["event_time"]))

        for peer in range(device["bgp_peer_count"]):
            main.append(
                _metric("bgp_peer", {
                    **tags, "index": f"198.51.100.{peer + 1}"
                }, {
                    "bgpPeerState": collector_schemas.BGP_ESTABLISHED, "bgpPeerRemoteAs": 64512 + peer
                },
                        device["event_time"]))

        for row in by_poll.pop(key, []):
            if (row["uptime"] != device["sys_uptime"] or row["site_id"] != device["site_id"]):
                raise ValueError(f"{row['origin_hash']} disagrees with its device's poll")

            main.extend(_port_lines(row, tags))
            written.append(row)

    dmz = []

    for ((collector, device, when), rows) in by_poll.items():
        if (collector != PROFILES["tc1"].collector_id):
            tags = {
                "agent_host": AGENTS[device], "host": collector, "sysLocation": rows[0]["site_id"], "sysName": device
            }
            dmz.append(_metric("snmp", tags, {"sysUpTime": rows[0]["uptime"]}, when))

            for row in rows:
                dmz.extend(_port_lines(row, tags))
                written.append(row)

    if (len(written) != len(ports)):
        raise ValueError("a port was polled with no poll of its device")

    for (name, lines) in (("snmp-poller.jsonlines", main), ("snmp-poller-dmz.jsonlines", dmz)):
        with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
            handle.write("".join(line + "\n" for line in lines))


def _octets(mac: str) -> bytes:
    return bytes(int(octet, 16) for octet in mac.split(":"))


def write_fdb(directory: str):
    """
    `mac-table.jsonlines`: each switch's forwarding table, walked by a Telegraf `inputs.snmp` agent, and the MAC
    notification traps its `inputs.snmp_trap` listener received, through the JSON serializer.

    A walk is the bridge's port map (`ifXTable` and `bridge_port`) and then its `fdb` entries; a removal is the
    switch's `cmnMacChangedNotification`, which names the bridge port the last walk mapped.
    """
    rows = sample("telemetry", "tc2_mac")
    interfaces = {(row["device_id"], row["port_id"]): row["if_index"] for row in sample("telemetry", "tc1")}
    ports: dict = {}

    for row in rows:
        ports.setdefault(row["switch_id"], [])

        if (row["port_id"] not in ports[row["switch_id"]]):
            ports[row["switch_id"]].append(row["port_id"])

    def if_index(switch: str, port: str) -> int:
        return interfaces.get((switch, port), 1000 + int(_digest("ifindex", switch, port, length=4), 16) % 9000)

    bridge_ports = {
        (switch, port): number
        for (switch, names) in ports.items()
        for (number, port) in enumerate(sorted(names), start=1)
    }
    walked = set()
    lines = []

    for (index, row) in enumerate(rows):
        (switch, agent) = (row["switch_id"], AGENTS[row["switch_id"]])
        port = bridge_ports[(switch, row["port_id"])]

        if (row["mac_action"] is not None):
            if ((switch, row["port_id"]) not in walked):
                raise ValueError(f"{row['origin_hash']} names a port no walk has mapped")

            message = (bytes([NOTIFICATION_CODES[row["mac_action"]]]) + row["vlan_id"].to_bytes(2, "big") +
                       _octets(row["mac_address"]) + port.to_bytes(2, "big") + b"\x00")
            lines.append(
                _metric(
                    "snmp_trap",
                    {
                        "host": row["collector_id"],
                        "mib": "CISCO-MAC-NOTIFICATION-MIB",
                        "name": "cmnMacChangedNotification",
                        "oid": ".1.3.6.1.4.1.9.9.215.2.0.1",
                        "source": agent,
                        "version": "2c",
                    }, {"cmnHistMacChangedMsg": message.hex()},
                    row["event_time"]))
            continue

        tags = {"agent_host": agent, "host": row["collector_id"], "sysLocation": row["site_id"], "sysName": switch}
        previous = rows[index - 1] if index > 0 else None

        if (previous is None or previous["mac_action"] is not None
                or (previous["switch_id"], previous["event_time"]) != (switch, row["event_time"])):
            for name in sorted(ports[switch]):
                number = if_index(switch, name)
                lines.append(
                    _metric("ifXTable", {
                        "agent_host": agent, "host": row["collector_id"], "ifIndex": str(number)
                    }, {"ifName": name},
                            row["event_time"]))
                lines.append(
                    _metric("bridge_port",
                            {
                                "agent_host": agent,
                                "host": row["collector_id"],
                                "index": str(bridge_ports[(switch, name)])
                            }, {"dot1dBasePortIfIndex": number},
                            row["event_time"]))
                walked.add((switch, name))

        lines.append(
            _metric("fdb", {
                **tags, "index": ".".join(str(part) for part in (row["vlan_id"], *_octets(row["mac_address"])))
            }, {
                "dot1qTpFdbPort": port, "dot1qTpFdbStatus": FDB_LEARNED
            },
                    row["event_time"]))

    with open(os.path.join(directory, "mac-table.jsonlines"), "w", encoding="utf-8") as handle:
        handle.write("".join(line + "\n" for line in lines))


def write_arp(directory: str):
    """
    `arp.pcap`: every ARP packet the sensor saw, as Ethernet frames.

    A request goes to the broadcast address and asks for no target hardware address; a gratuitous reply announces
    the sender's own address to everyone; any other reply answers the last request from the address it is sent to.
    """
    rows = sample("telemetry", "tc2_arp")
    asked: dict = {}
    frames = []

    for row in rows:
        (sender, target) = (row["arp_sender_ip"], row["arp_target_ip"])

        if (row["arp_operation"] == "request"):
            (destination, target_mac) = (BROADCAST, NO_ADDRESS)
            asked[(sender, target)] = row["arp_sender_mac"]
        elif (sender == target):
            (destination, target_mac) = (BROADCAST, BROADCAST)
        else:
            target_mac = asked.get((target, sender), BROADCAST)
            destination = target_mac

        if (row["event_time"] % 1000 != 0):
            raise ValueError("a capture's timestamp carries microseconds")

        body = struct.pack("!HHBBH", 1, ipfix.ETHERTYPE_IPV4, 6, 4, ARP_CODES[row["arp_operation"]])
        body += _octets(row["arp_sender_mac"]) + ipaddress.IPv4Address(sender).packed
        body += _octets(target_mac) + ipaddress.IPv4Address(target).packed
        frame = (_octets(destination) + _octets(row["arp_sender_mac"]) +
                 collector_schemas.ETHERTYPE_ARP.to_bytes(2, "big") + body)
        frames.append((row["event_time"], frame + b"\x00" * (60 - len(frame))))

    ipfix.write_frames(os.path.join(directory, "arp.pcap"), frames)


def _detail_date(when_ns: int) -> str:
    stamp = pd.Timestamp(_poll_seconds(when_ns), unit="s", tz="UTC")

    return f"{stamp:%a %b} {stamp.day:2d} {stamp:%H:%M:%S %Y}"


def _detail_entry(when_ns: int, attributes: list[tuple[str, typing.Any]]) -> str:
    lines = [_detail_date(when_ns)]

    for (name, value) in attributes:
        text = f'"{value}"' if isinstance(
            value, str) and not value.startswith("0x") and name not in ("Packet-Type",
                                                                        "NAS-IP-Address",
                                                                        "NAS-Port-Type",
                                                                        "Service-Type",
                                                                        "Acct-Status-Type",
                                                                        "Acct-Authentic",
                                                                        "Acct-Terminate-Cause") else str(value)
        lines.append(f"\t{name} = {text}")

    return "\n".join(lines) + "\n\n"


def _eap(code: int, identifier: int, kind: typing.Optional[int] = None, data: bytes = b"") -> str:
    body = b"" if kind is None else bytes([kind]) + data

    return "0x" + (bytes([code, identifier]) + (4 + len(body)).to_bytes(2, "big") + body).hex()


def write_radius_auth(directory: str):
    """
    `radius-auth.detail`: the 802.1X exchanges a FreeRADIUS server logged with `auth_log` and `reply_log` into one
    detail file.

    A start is the supplicant's EAP-Response/Identity; an outcome is the last round's Access-Request and the
    Access-Accept or Access-Reject answering it, which carries the `User-Name` and its request's `Timestamp`,
    as `join_radius_replies` pairs them.
    """
    entries = []
    answered = set()

    for row in sample("telemetry", "tc2_auth"):
        (switch, port) = (row["switch_id"], row["port_id"])
        long_port = next(long + port[len(short):] for (short, long) in LONG_INTERFACE_NAMES.items()
                         if port.startswith(short))
        seconds = _poll_seconds(row["event_time"])
        identity = row["dot1x_identity"]
        request = [
            ("Packet-Type", "Access-Request"),
            ("User-Name", identity),
            ("NAS-IP-Address", AGENTS[switch]),
            ("NAS-Identifier", switch),
            ("NAS-Port-Id", long_port),
            ("NAS-Port-Type", "Ethernet"),
            ("Service-Type", "Framed-User"),
            ("Calling-Station-Id", row["mac_address"].upper().replace(":", "-")),
        ]

        if (row["dot1x_result"] == "started"):
            entries.append(
                _detail_entry(row["event_time"],
                              request + [("EAP-Message", _eap(2, 1, 1, identity.encode())), ("Timestamp", seconds)]))
            continue

        if ((identity, seconds) in answered):
            raise ValueError(f"two outcomes for {identity} at {seconds}: a reply could not be paired")

        answered.add((identity, seconds))
        entries.append(
            _detail_entry(
                row["event_time"],
                request + [("EAP-Message", _eap(2, 9, 25, b"\x00")),
                           ("State", "0x" + _digest("state", row["origin_hash"], length=32)), ("Timestamp", seconds)]))
        entries.append(
            _detail_entry(row["event_time"],
                          [("Packet-Type", OUTCOME_PACKETS[row["dot1x_result"]]), ("User-Name", identity),
                           ("EAP-Message", _eap(3 if row["dot1x_result"] == "success" else 4, 9)),
                           ("Message-Authenticator", "0x" + "0" * 32), ("Timestamp", seconds)]))

    with open(os.path.join(directory, "radius-auth.detail"), "w", encoding="utf-8") as handle:
        handle.write("".join(entries))


def write_radius_accounting(directory: str):
    """
    `radius-acct.detail`: the VPN gateway's accounting Starts and Stops, as FreeRADIUS's `detail` module logs them.

    A Stop carries `Acct-Session-Time` from the last Start of its session, and none where its Start never arrived.
    The identity provider's own session records, which hold both ends in one, are another feed and not here.
    """
    started: dict = {}
    entries = []

    for row in sample("session", "tc5_session"):
        if (row["collector_id"] != PROFILES["tc5_session"].collector_id):
            continue

        stamp = pd.Timestamp(_poll_seconds(row["event_time"]), unit="s", tz="UTC")
        attributes = [
            ("Acct-Status-Type", ACCOUNTING_TYPES[row["session_action"]]),
            ("Acct-Session-Id", row["session_id"]),
            ("User-Name", row["user_principal"]),
            ("NAS-IP-Address", VPN_GATEWAY[0]),
            ("NAS-Identifier", VPN_GATEWAY[1]),
            ("NAS-Port-Type", "Virtual"),
            ("Acct-Authentic", "RADIUS"),
            ("Acct-Delay-Time", 0),
            ("Event-Timestamp", f"{stamp:%b} {stamp.day:2d} {stamp:%Y %H:%M:%S} UTC"),
        ]

        if (row["session_action"] == "start"):
            started[row["session_id"]] = row["event_time"]
        elif (row["session_id"] in started):
            attributes += [("Acct-Session-Time", (row["event_time"] - started.pop(row["session_id"])) // 1_000_000_000),
                           ("Acct-Terminate-Cause", "User-Request")]

        entries.append(_detail_entry(row["event_time"], attributes + [("Timestamp", _poll_seconds(row["event_time"]))]))

    with open(os.path.join(directory, "radius-acct.detail"), "w", encoding="utf-8") as handle:
        handle.write("".join(entries))


TCP_OPTION_BYTES = {"nop": b"\x01", "sok": b"\x04\x02", "ts": b"\x08\x0a" + bytes(8)}
"""The options a stack's SYN carries that hold no value the corpus names; `mss` and `ws` are written with theirs."""

WINDOW_SCALE = 8
CRAFTED_SYN_WINDOW = 1024
"""The window a scanner's raw socket writes into a SYN it crafts without options."""

ESTABLISHED_WINDOW = 502
"""The scaled window every packet after the handshake carries."""

DHCP_SERVER_PID = "1201.139975612345088"
OSQUERY_INTERFACE = "eth0"
REPORT_VALID_SECONDS = 90000
"""How long the inventory's report stands: the deployment's schedule, which osquery's results do not carry."""

CONN_FIELDS = [("ts", "time"), ("uid", "string"), ("id.orig_h", "addr"), ("id.orig_p", "port"), ("id.resp_h", "addr"),
               ("id.resp_p", "port"), ("proto", "enum"), ("service", "string"), ("duration", "interval"),
               ("orig_bytes", "count"), ("resp_bytes", "count"), ("conn_state", "string"), ("history", "string")]
SERVICE_SSL_FIELDS = [("ts", "time"), ("uid", "string"), ("id.orig_h", "addr"), ("id.orig_p", "port"),
                      ("id.resp_h", "addr"), ("id.resp_p", "port"), ("version", "string"), ("server_name", "string"),
                      ("established", "bool"), ("ja4", "string")]


def _tcp_options(order: str, mss: int) -> bytes:
    options = b""

    for name in order.split(","):
        if (name == "mss"):
            options += b"\x02\x04" + mss.to_bytes(2, "big")
        elif (name == "ws"):
            options += b"\x03\x03" + bytes([WINDOW_SCALE])
        else:
            options += TCP_OPTION_BYTES[name]

    if (len(options) % 4 != 0):
        raise ValueError(f"the options {order} are not a whole number of 32-bit words")

    return options


def write_packets(directory: str):
    """
    `tc4.pcap`: every packet the capture point saw, as Ethernet frames of IPv4 and TCP.

    A SYN a stack sent carries the stack's options, MSS and window; a SYN crafted without options carries a raw
    socket's window; every other packet the established window. The payload is zeros of the sample's length.
    """
    frames = []

    for row in sample("transport", "tc4"):
        if (row["protocol"] != "tcp"):
            raise ValueError(f"{row['origin_hash']} is not TCP")

        options = b"" if row["tcp_options_order"] is None else _tcp_options(row["tcp_options_order"], row["tcp_mss"])
        window = row["tcp_window_initial"] if row["tcp_window_initial"] is not None else (
            CRAFTED_SYN_WINDOW if row["tcp_flags"] & collector_schemas.TCP_SYN else ESTABLISHED_WINDOW)
        sequence = int(_digest("seq", row["collector_seq"], length=8), 16)
        tcp = struct.pack("!HHIIBBHHH",
                          row["src_port"],
                          row["dst_port"],
                          sequence,
                          0, (20 + len(options)) // 4 << 4,
                          row["tcp_flags"],
                          window,
                          0,
                          0) + options
        payload = bytes(row["data_len"])
        ip = struct.pack("!BBHHHBBH4s4s",
                         0x45,
                         0,
                         20 + len(tcp) + len(payload),
                         row["collector_seq"] & 0xFFFF,
                         0,
                         64,
                         6,
                         0,
                         ipaddress.IPv4Address(row["src_ip"]).packed,
                         ipaddress.IPv4Address(row["dst_ip"]).packed)

        if (row["event_time"] % 1000 != 0):
            raise ValueError("a capture's timestamp carries microseconds")

        frames.append((row["event_time"],
                       b"\x02\x00\x00\x00\x00\x02\x02\x00\x00\x00\x00\x01" + ipfix.ETHERTYPE_IPV4.to_bytes(2, "big") +
                       ip + tcp + payload))

    ipfix.write_frames(os.path.join(directory, "tc4.pcap"), frames)


def _kea_time(when_ns: int) -> str:
    if (when_ns % 1_000_000 != 0):
        raise ValueError("Kea's log writes the time to the millisecond")

    stamp = pd.Timestamp(when_ns, unit="ns", tz="UTC")

    return stamp.strftime("%Y-%m-%d %H:%M:%S.") + f"{stamp.microsecond // 1000:03d}"


def write_kea(directory: str):
    """`kea-dhcp4.log`: the leases the DHCP server granted, as Kea's `kea-dhcp4.leases` logger writes them."""
    lines = []

    for row in sample("network", "tc3_dhcp"):
        if (row["dhcp_message"] != "ack" or row.get("switch_id") is not None):
            raise ValueError(f"{row['origin_hash']} is not an acknowledgement Kea's lease log can write")

        lines.append(f"{_kea_time(row['event_time'])} INFO  [kea-dhcp4.leases/{DHCP_SERVER_PID}] DHCP4_LEASE_ALLOC "
                     f"[hwtype=1 {row['mac']}], cid=[no info], tid=0x{_digest('tid', row['origin_hash'], length=8)}: "
                     f"lease {row['ip']} has been allocated for {row['lease_seconds']} seconds")

    with open(os.path.join(directory, "kea-dhcp4.log"), "w", encoding="utf-8") as handle:
        handle.write("".join(line + "\n" for line in lines))


def write_osquery(directory: str):
    """`osqueryd.results.log`: each agent's `interface_details` snapshot, one result per report."""
    lines = []

    for (counter, row) in enumerate(sample("network", "tc7_host_inventory")):
        if (row["report_valid_seconds"] != REPORT_VALID_SECONDS):
            raise ValueError(f"{row['origin_hash']} stands for another schedule than the deployment's")

        seconds = _poll_seconds(row["event_time"])
        stamp = pd.Timestamp(seconds, unit="s", tz="UTC")
        lines.append(
            json.dumps({
                "snapshot": [{
                    "interface": OSQUERY_INTERFACE, "mac": row["mac"], "type": "6", "mtu": "1500", "enabled": "1"
                }],
                "action": "snapshot",
                "name": "pack/inventory/interface_details",
                "hostIdentifier": row["hostname"],
                "calendarTime": f"{stamp:%a %b} {stamp.day:2d} {stamp:%H:%M:%S %Y} UTC",
                "unixTime": seconds,
                "epoch": 0,
                "counter": counter,
                "numerics": False,
            }))

    with open(os.path.join(directory, "osqueryd.results.log"), "w", encoding="utf-8") as handle:
        handle.write("".join(line + "\n" for line in lines))


def write_service(directory: str):
    """
    `conn.log` and `ssl.log`: each connection to a `service` the sensor saw, and its handshake, joined on `uid`.

    The bytes are split between the two directions; the schema adds them back.
    """
    (connections, handshakes) = ([], [])

    for row in sample("application", "tc7_service"):
        uid = "C" + _digest("service", row["origin_hash"])
        port = _port("service", row["origin_hash"])
        when = _seconds(row["event_time"])
        sent = row["bytes_total"] // 4
        connections.append([
            when,
            uid,
            row["src_ip"],
            port,
            row["dst_ip"],
            row["dst_port"],
            "tcp",
            "ssl",
            "1.000000",
            sent,
            row["bytes_total"] - sent,
            "SF",
            "ShADadFf"
        ])
        handshakes.append([
            when,
            uid,
            row["src_ip"],
            port,
            row["dst_ip"],
            row["dst_port"],
            "TLSv13",
            row["sni"],
            "T",
            row["ja4_client"]
        ])

    _zeek_log(os.path.join(directory, "conn.log"), "conn", CONN_FIELDS, connections)
    _zeek_log(os.path.join(directory, "ssl.log"), "ssl", SERVICE_SSL_FIELDS, handshakes)


WRITERS = {
    "tc3": write_flows,
    "tc5_auth": write_signins,
    "tc6": write_tls,
    "tc7_http": write_http,
    "tc7_dns": write_dns,
    "tc7_saas": write_audit,
    "tc7_endpoint": write_sysmon,
    "tc1": write_snmp,
    "tc1_device": write_snmp,
    "tc2_mac": write_fdb,
    "tc2_arp": write_arp,
    "tc2_auth": write_radius_auth,
    "tc5_session": write_radius_accounting,
    "tc4": write_packets,
    "tc3_dhcp": write_kea,
    "tc7_host_inventory": write_osquery,
    "tc7_service": write_service,
}


def write_all(root: str = HERE):
    """Write every fixture under `root/<segment>/`."""
    # Two classes read the estate poller's one file, which one writer writes once.
    for (segment, writer) in dict.fromkeys(
        (segment, WRITERS[telemetry_class]) for (telemetry_class, (segment, _)) in CLASSES.items()):
        directory = os.path.join(root, segment)
        os.makedirs(directory, exist_ok=True)
        writer(directory)


def normalize_flows(path: str) -> pd.DataFrame:
    """`tc3` records from a capture, through `FlowRecordNormalizeStage` in a pipeline."""
    # pylint: disable=import-outside-toplevel
    from morpheus.config import Config
    from morpheus.config import CppConfig
    from morpheus.config import ExecutionMode
    from morpheus.pipeline import LinearPipeline
    from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
    from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
    from morpheus.stages.telemetry.flow_record_normalize_stage import FlowRecordNormalizeStage

    CppConfig.set_should_use_cpp(False)
    config = Config()
    config.execution_mode = ExecutionMode.CPU
    profile = PROFILES["tc3"]
    decoded = pd.DataFrame(ipfix.read_pcap(path).records)
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[decoded]))
    pipe.add_stage(
        FlowRecordNormalizeStage(config,
                                 collector_id=profile.collector_id,
                                 tenant_id=profile.tenant_id,
                                 schema_version=profile.schema_version,
                                 clock_source=profile.clock_source,
                                 clock_offset_ms=profile.clock_offset_ms,
                                 sampling_policy=profile.sampling_policy,
                                 exporter_names=EXPORTERS,
                                 first_seq=profile.first_seq))
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    return pd.concat([message.copy_dataframe() for message in sink.get_messages()], ignore_index=True)


def normalize_all(root: str = HERE) -> dict[str, pd.DataFrame]:
    """Every fixture under `root`, normalized, by class."""

    def at(telemetry_class: str, name: str) -> str:
        return os.path.join(root, CLASSES[telemetry_class][0], name)

    return {
        "tc3":
            normalize_flows(at("tc3", "tc3.pcap")),
        "tc5_auth":
            collector_schemas.entra_signin(at("tc5_auth", "signins.jsonlines"), PROFILES["tc5_auth"]),
        "tc6":
            collector_schemas.zeek_ssl(at("tc6", "ssl.log"), at("tc6", "x509.log"), PROFILES["tc6"]),
        "tc7_http":
            collector_schemas.zeek_http(at("tc7_http", "http.log"), PROFILES["tc7_http"]),
        "tc7_dns":
            collector_schemas.zeek_dns(at("tc7_dns", "dns.log"), PROFILES["tc7_dns"]),
        "tc7_saas":
            collector_schemas.m365_audit(at("tc7_saas", "audit.jsonlines"), PROFILES["tc7_saas"], ORGANIZATIONS),
        "tc7_endpoint":
            collector_schemas.sysmon_process(at("tc7_endpoint", "sysmon.xml"), PROFILES["tc7_endpoint"]),
        "tc1":
            collector_schemas.merge_collectors([
                collector_schemas.snmp_ports(at("tc1", "snmp-poller.jsonlines"), PROFILES["tc1"]),
                collector_schemas.snmp_ports(at("tc1", "snmp-poller-dmz.jsonlines"), DMZ_POLLER),
            ]),
        "tc1_device":
            collector_schemas.snmp_devices(at("tc1_device", "snmp-poller.jsonlines"), PROFILES["tc1_device"]),
        "tc2_mac":
            collector_schemas.fdb_entries(at("tc2_mac", "mac-table.jsonlines"), PROFILES["tc2_mac"]),
        "tc2_arp":
            collector_schemas.arp_packets(at("tc2_arp", "arp.pcap"), PROFILES["tc2_arp"]),
        "tc2_auth":
            collector_schemas.radius_auth(at("tc2_auth", "radius-auth.detail"), PROFILES["tc2_auth"], SWITCH_SITES),
        "tc5_session":
            collector_schemas.radius_accounting(at("tc5_session", "radius-acct.detail"), PROFILES["tc5_session"]),
        "tc4":
            collector_schemas.packets(at("tc4", "tc4.pcap"), PROFILES["tc4"]),
        "tc3_dhcp":
            collector_schemas.kea_leases(at("tc3_dhcp", "kea-dhcp4.log"), PROFILES["tc3_dhcp"]),
        "tc7_host_inventory":
            collector_schemas.osquery_inventory(at("tc7_host_inventory", "osqueryd.results.log"),
                                                PROFILES["tc7_host_inventory"],
                                                REPORT_VALID_SECONDS),
        "tc7_service":
            collector_schemas.zeek_service(at("tc7_service", "conn.log"),
                                           at("tc7_service", "ssl.log"),
                                           PROFILES["tc7_service"]),
    }


if (__name__ == "__main__"):
    write_all(sys.argv[1] if len(sys.argv) > 1 else HERE)
