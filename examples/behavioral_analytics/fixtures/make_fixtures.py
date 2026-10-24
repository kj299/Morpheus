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
Render seven of the committed input samples in the formats real feeds write, and read them back.

Each fixture is the same records as `../samples/<segment>/<class>.jsonlines`, written the way the source writes
them: a packet capture of IPFIX from three routers, Entra ID sign-in records as Azure Monitor exports them, Zeek
`ssl.log` and `x509.log`, `http.log` and `dns.log`, unified audit log records, and Sysmon events as `wevtutil`
renders them. Only what the format carries is
written; a collector's own facts are the `PROFILES` below. `normalize_all()` reads them back with
`FlowRecordNormalizeStage` and the schemas in `morpheus.utils.collector_schemas`, and
`tests/morpheus/utils/test_collector_schemas.py` holds the result to the samples, and the segments run over it to
what they write from the samples.

Run it after regenerating the samples; the test fails while the fixtures are stale.
"""

import hashlib
import json
import os
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
    "tc5_auth": CollectorProfile("idp", "TC-5/1.0.0", "corp", "ntp_disciplined", -4, first_seq=1),
    "tc6": CollectorProfile("tls-inspect-01", "tc6.v1", "corp", "ptp", -18),
    "tc7_http": CollectorProfile("proxy-01", "tc7_http.v1", "corp", "ntp_disciplined", -15),
    "tc7_dns": CollectorProfile("resolver-01", "tc7_dns.v1", "corp", "ntp_disciplined", 8),
    "tc7_saas": CollectorProfile("saas-audit-01", "tc7_saas.v1", "corp", "ntp_disciplined", -18),
    "tc7_endpoint": CollectorProfile("edr-01", "tc7_endpoint.v1", "corp", "ntp_disciplined", -4),
}
"""What each collector says about itself, which its records do not."""

CLASSES = {
    "tc3": ("network", ["tc3.pcap"]),
    "tc5_auth": ("session", ["signins.jsonlines"]),
    "tc6": ("presentation", ["ssl.log", "x509.log"]),
    "tc7_http": ("application", ["http.log"]),
    "tc7_dns": ("application", ["dns.log"]),
    "tc7_saas": ("saas", ["audit.jsonlines"]),
    "tc7_endpoint": ("endpoint", ["sysmon.xml"]),
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
    """`http.log`. Every seventh request carries a query string, which the schema has to strip."""
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
            0,
            0,
            row["status_code"],
            STATUS_MESSAGES.get(row["status_code"])
        ])

    _zeek_log(os.path.join(directory, "http.log"),
              "http",
              [("ts", "time"), ("uid", "string"), ("id.orig_h", "addr"), ("id.orig_p", "port"), ("id.resp_h", "addr"),
               ("id.resp_p", "port"), ("trans_depth", "count"), ("method", "string"), ("host", "string"),
               ("uri", "string"), ("version", "string"), ("user_agent", "string"), ("request_body_len", "count"),
               ("response_body_len", "count"), ("status_code", "count"), ("status_msg", "string")],
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


WRITERS = {
    "tc3": write_flows,
    "tc5_auth": write_signins,
    "tc6": write_tls,
    "tc7_http": write_http,
    "tc7_dns": write_dns,
    "tc7_saas": write_audit,
    "tc7_endpoint": write_sysmon,
}


def write_all(root: str = HERE):
    """Write every fixture under `root/<segment>/`."""
    for (telemetry_class, (segment, _)) in CLASSES.items():
        directory = os.path.join(root, segment)
        os.makedirs(directory, exist_ok=True)
        WRITERS[telemetry_class](directory)


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
    }


if (__name__ == "__main__"):
    write_all(sys.argv[1] if len(sys.argv) > 1 else HERE)
