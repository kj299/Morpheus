# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The normalize step of the per-layer pattern for real feeds: one `DataFrameInputSchema` per input class read here, at
every layer, from the format a deployment already writes to the columns the stages read.

Each schema is built from `column_info` primitives and applied with `process_dataframe`, as the guide's per-layer
pattern says, and produces exactly the class's columns from `examples/behavioral_analytics/input_contracts.json`
and the universal envelope:

- `tc5_auth`: Microsoft Entra ID sign-in logs, as Azure Monitor exports them (one record with its `properties`), the
  feed `examples/upstream_reuse/evaluate.py` measures as the `azure` source.
- `tc6`: Zeek `ssl.log` joined to `x509.log` on the leaf certificate's fingerprint, with the JA4 package's `ja4`.
- `tc7_http`: Zeek `http.log`.
- `tc7_dns`: Zeek `dns.log`.
- `tc7_saas`: Microsoft 365 `UnifiedAuditLog` records, as the Management Activity API returns them.
- `tc7_endpoint`: Windows System Monitor (Sysmon) event 1, joined to the event 7 that loaded the process's own
  image for its signature status, as `wevtutil qe /f:xml` renders them.
- `tc1` and `tc1_device`: what a Telegraf `inputs.snmp` agent writes through its JSON serializer, one measurement per
  table it walks -- the `snmp` scalars, `interface` (IF-MIB, EtherLike-MIB), `transceiver` (ENTITY-MIB,
  CISCO-ENTITY-SENSOR-MIB), `lldp` (LLDP-MIB) and `bgp_peer` (BGP4-MIB) -- joined back into one record per port
  and one per device each poll.
- `tc2_mac`: the same agent's walk of Q-BRIDGE-MIB's forwarding table, with the BRIDGE-MIB and IF-MIB tables that
  name its bridge ports, and the CISCO-MAC-NOTIFICATION-MIB traps its `inputs.snmp_trap` listener receives.
- `tc2_arp`: the ARP packets in a packet capture.
- `tc2_auth`: a FreeRADIUS detail file of 802.1X Access-Requests (`auth_log`) and the replies (`reply_log`).
- `tc5_session`: a FreeRADIUS detail file of accounting Starts and Stops.
- `tc4`: the TCP and UDP packets in a packet capture.
- `tc3_dhcp`: a Kea DHCPv4 server's lease log.
- `tc7_host_inventory`: osquery's scheduled `interface_details` results.
- `tc7_service`: Zeek `conn.log` joined to `ssl.log` on the connection's `uid`.
- `tc0_identity`: a Workday report of worker events merged with Entra ID's group-membership audit records.
- `tc0_asset` and `tc0_application`: ServiceNow's `sys_audit` history of CMDB items, folded into versions.
- `tc0_network`: NetBox's object changelog.

Zeek logs are read with {py:func}`morpheus.parsers.zeek.parse`.

**What a format cannot say, the schema does not invent.** Zeek sees a TLS handshake, not what an inspecting proxy
saw inside it, so `content_type_declared` and `content_type_detected` are null from the Zeek schema and the layer 6
content rule cannot fire on a Zeek-only feed; a feed with an inspecting proxy supplies them. A cloud identity
provider has no on-premises target computer, so `target_host` is null from the Entra ID schema. No MIB a switch
answers holds a hash of its running configuration or a rate of its syslog, so `running_config_hash` and
`syslog_rate` are null from SNMP. A packet capture numbers nothing, so the ARP schema's `collector_seq` cannot show a
record its sensor numbered and never delivered. A CMDB's audit history and NetBox's changelog keep when a change
was entered and not when it took effect, and so does Entra ID's audit log for a group membership, so `valid_from`
is null from them; Workday's report keeps both. Kea's lease log does not write the relay's option 82. No format here
records when the collector read a line, so `observed_time` is the record's own time and the collection lag reads
zero; a collector that keeps its read time passes it as `observed_column`. The collector's own facts --
identity, contract version, tenant, clock discipline, sampling -- are a `CollectorProfile`, because no record
carries them, and `collector_seq` numbers the records in file order from the profile's `first_seq`.

**Vocabulary is mapped where the format's differs and passed through where it does not.** Zeek writes `TLSv13`
and OpenSSL's `self signed certificate`; the stages read `TLSv1.3` and `self-signed`. Entra ID writes an error code,
`multiFactorAuthentication` and `primaryRefreshToken`; the stages read `success` or `failure`, a boolean and
`refresh`. SNMP writes `ifOperStatus` as a number, a MAC notification its operation as a byte, ARP an opcode,
RADIUS `Access-Accept` and `Stop`, and a switch's `NAS-Port-Id` the interface's long name. A value the map does not
know is passed through as it is, so a new one reaches the SIEM visibly rather
than as a null.

`origin_hash` is built the way the corpus builders build it for each class, so a normalized record is, to the
lineage stamp, the same record the corpus has; where the record carries its own identifier, as a sign-in's `id`,
that is the origin.
"""

import dataclasses
import hashlib
import ipaddress
import json
import math
import re
import struct
import typing
from xml.etree import ElementTree

import pandas as pd

from morpheus.parsers import zeek
from morpheus.utils import ipfix
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import CustomColumn
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.schema_transforms import process_dataframe

TLS_VERSIONS = {
    "SSLv3": "SSLv3",
    "TLSv10": "TLSv1.0",
    "TLSv11": "TLSv1.1",
    "TLSv12": "TLSv1.2",
    "TLSv13": "TLSv1.3",
    "DTLSv10": "DTLSv1.0",
    "DTLSv12": "DTLSv1.2",
}
"""Zeek's `ssl.log` version names to the ones the layer 6 stages and searches read."""

VALIDATION_RESULTS = {
    "ok": "ok",
    "self signed certificate": "self-signed",
    "self signed certificate in certificate chain": "self-signed-chain",
    "certificate has expired": "expired",
    "unable to get local issuer certificate": "unknown-issuer",
}
"""Zeek's `validation_status` (OpenSSL's verify strings) to the layer 6 vocabulary."""

M365_RESULTS = {"succeeded": "success", "success": "success", "true": "success", "false": "failed"}
"""`ResultStatus` values, lower-cased, that mean something other than themselves; the rest pass through lower-cased."""

ENTRA_FIELDS = {
    "event_time": "createdDateTime",
    "user_principal": "userPrincipalName",
    "source_country": "location.countryOrRegion",
    "source_region": "location.state",
    "source_city": "location.city",
    "source_latitude": "location.geoCoordinates.latitude",
    "source_longitude": "location.geoCoordinates.longitude",
    "source_asn": "autonomousSystemNumber",
    "source_ip": "ipAddress",
    "app": "appDisplayName",
    "device_id": "deviceDetail.deviceId",
    "auth_result": "status.errorCode",
    "mfa_used": "authenticationRequirement",
    "mfa_result": "authenticationDetails",
    "token_type": "incomingTokenType",
}
"""Each `tc5_auth` column to the sign-in property it is read from, as `pandas.json_normalize` names it. This is the
mapping `examples/upstream_reuse/evaluate.py` measures as `SOURCE_MAPPINGS["azure"]`, with the separators the
upstream stage strips kept; `tests/morpheus/utils/test_collector_schemas.py` holds the two to each other."""

ENTRA_MFA_REQUIREMENTS = {"multiFactorAuthentication": True, "singleFactorAuthentication": False}
"""`authenticationRequirement` to whether a second factor was asked for."""

ENTRA_TOKEN_TYPES = {"none": "bearer", "primaryRefreshToken": "refresh", "refreshToken": "refresh"}
"""`incomingTokenType` values that mean something other than themselves; the rest pass through."""

ENTRA_PASSWORD_METHOD = "Password"
"""The `authenticationDetails` step that is the first factor; every other step is a second one."""

SYSMON_NAMESPACE = "{http://schemas.microsoft.com/win/2004/08/events/event}"

ETHERTYPE_ARP = 0x0806


@dataclasses.dataclass(frozen=True)
class CollectorProfile:
    """What a collector says about itself, which no record it reads carries."""

    collector_id: str
    schema_version: str
    tenant_id: str
    clock_source: str = "ntp_disciplined"
    clock_offset_ms: typing.Optional[float] = None
    sampling_policy: str = "full"
    first_seq: int = 0


def _missing(value: typing.Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA


def _text_values(series: pd.Series) -> pd.Series:
    return pd.Series([None if _missing(value) else str(value) for value in series], index=series.index, dtype=object)


def _text(name: str, source: str, mapping: dict = None) -> CustomColumn:
    """A string column read from `source`, nulls as `None`, optionally through a vocabulary map."""

    def column(df: pd.DataFrame) -> pd.Series:
        if (source not in df.columns):
            return pd.Series([None] * len(df), index=df.index, dtype=object)

        values = _text_values(df[source])

        if (mapping is not None):
            values = values.map(lambda value: None if value is None else mapping.get(value, value))

        return values

    return CustomColumn(name=name, dtype="object", process_column_fn=column, input_column_types={source: "object"})


def _through(name: str,
             source: str,
             function: typing.Callable[[str], typing.Any],
             dtype: str = "object") -> CustomColumn:
    """A column read from `source` as text and passed value by value through `function`, nulls staying null."""

    def column(df: pd.DataFrame) -> pd.Series:
        if (source not in df.columns):
            return pd.Series([None] * len(df), index=df.index, dtype=dtype)

        return pd.Series([None if value is None else function(value) for value in _text_values(df[source])],
                         index=df.index,
                         dtype=dtype)

    return CustomColumn(name=name, dtype=dtype, process_column_fn=column, input_column_types={source: "object"})


def _count(name: str, source: str) -> CustomColumn:
    """A nullable integer column read from `source`."""

    def column(df: pd.DataFrame) -> pd.Series:
        if (source not in df.columns):
            return pd.Series([None] * len(df), index=df.index, dtype="Int64")

        return pd.to_numeric(df[source]).astype("Int64")

    return CustomColumn(name=name, dtype="Int64", process_column_fn=column, input_column_types={source: "Int64"})


def _constant(name: str, value: typing.Any) -> CustomColumn:
    dtype = "Int64" if isinstance(value, int) and not isinstance(value, bool) else "object"
    return CustomColumn(name=name,
                        dtype=dtype,
                        process_column_fn=lambda df: pd.Series([value] * len(df), index=df.index, dtype=dtype))


def _seconds_to_ns(series: pd.Series) -> pd.Series:
    # Zeek writes seconds to the microsecond; rounding there first keeps a float's last bits out of the integer.
    return (pd.to_numeric(series) * 1_000_000).round().astype("Int64") * 1000


def _derived(name: str, dtype: str, function: typing.Callable[[pd.DataFrame], pd.Series]) -> CustomColumn:
    return CustomColumn(name=name, dtype=dtype, process_column_fn=function)


def _envelope(profile: CollectorProfile,
              event_time: typing.Callable[[pd.DataFrame], pd.Series],
              origin: typing.Callable[[pd.DataFrame], pd.Series],
              observed_column: typing.Optional[str] = None,
              tenant: typing.Optional[typing.Callable[[pd.DataFrame], pd.Series]] = None) -> tuple[list, list]:
    """The envelope's columns before and after a class's own, in the order the corpus builders write them."""

    def sequence(df: pd.DataFrame) -> pd.Series:
        return pd.Series(range(profile.first_seq, profile.first_seq + len(df)), index=df.index, dtype="Int64")

    def observed(df: pd.DataFrame) -> pd.Series:
        if (observed_column is not None and observed_column in df.columns):
            return pd.to_numeric(df[observed_column]).astype("Int64")

        return event_time(df)

    head = [
        _constant("collector_id", profile.collector_id),
        _constant("schema_version", profile.schema_version),
        _derived("origin_hash", "object", origin),
        _derived("collector_seq", "Int64", sequence),
        _derived("event_time", "Int64", event_time),
    ]
    tail = [
        _derived("observed_time", "Int64", observed),
        _constant("clock_source", profile.clock_source),
        _constant("clock_offset_ms", profile.clock_offset_ms),
        _constant("sampling_policy", profile.sampling_policy),
        _derived("tenant_id", "object", tenant) if tenant is not None else _constant("tenant_id", profile.tenant_id),
    ]

    return (head, tail)


def _zeek_time(df: pd.DataFrame) -> pd.Series:
    return _seconds_to_ns(df["ts"])


def _joined(df: pd.DataFrame, parts: typing.Sequence) -> pd.Series:
    """Concatenate columns and literals row by row, as the corpus builders write `origin_hash`."""
    columns = []

    for part in parts:
        if (isinstance(part, ColumnInfo)):
            columns.append(part._process_column(df).tolist())  # pylint: disable=protected-access
        else:
            columns.append([part] * len(df))

    return pd.Series(["".join(str(value) for value in row) for row in zip(*columns)], index=df.index, dtype=object)


def zeek_ssl_schema(profile: CollectorProfile) -> DataFrameInputSchema:
    """
    `tc6` from Zeek `ssl.log` joined to `x509.log` by `join_ssl_x509`.

    Parameters
    ----------
    profile : `CollectorProfile`
        The collector's own facts.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """
    src = _text("src_ip", "id.orig_h")
    dst = _text("dst_ip", "id.resp_h")
    (head, tail) = _envelope(profile, _zeek_time, lambda df: _joined(df, [src, "->", dst]))

    def fingerprint(df: pd.DataFrame) -> pd.Series:
        return _text_values(df["fingerprint"]).map(lambda value: None if value is None else f"sha256:{value}")

    return DataFrameInputSchema(column_info=head + [
        src,
        _count("src_port", "id.orig_p"),
        dst,
        _count("dst_port", "id.resp_p"),
        _text("sni", "server_name"),
        # ssl.log is the analyzer for TLS over TCP; DTLS and QUIC are logged elsewhere.
        _constant("protocol", "tcp"),
        _text("tls_version", "version", TLS_VERSIONS),
        _text("ja4_client", "ja4"),
        _text("certificate_issuer", "certificate.issuer"),
        _derived("certificate_fingerprint_sha256", "object", fingerprint),
        _text("cipher_suite", "cipher"),
        _text("validation_result", "validation_status", VALIDATION_RESULTS),
        _derived("certificate_not_before", "Int64", lambda df: _seconds_to_ns(df["certificate.not_valid_before"])),
        _derived("certificate_not_after", "Int64", lambda df: _seconds_to_ns(df["certificate.not_valid_after"])),
        # What an inspecting proxy saw inside the session; Zeek did not see it.
        _text("content_type_declared", "content_type_declared"),
        _text("content_type_detected", "content_type_detected"),
    ] + tail)


def zeek_http_schema(profile: CollectorProfile) -> DataFrameInputSchema:
    """
    `tc7_http` from Zeek `http.log`. `url_path` is the `uri` without its query string, and `request_body` is the
    post-body package's `post_body` where the log carries it.

    Parameters
    ----------
    profile : `CollectorProfile`
        The collector's own facts.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """
    src = _text("src_ip", "id.orig_h")

    def path(df: pd.DataFrame) -> pd.Series:
        return _text_values(df["uri"]).map(lambda value: None if value is None else value.split("?", 1)[0])

    url_path = _derived("url_path", "object", path)
    event_time = _derived("event_time", "Int64", _zeek_time)
    (head, tail) = _envelope(profile, _zeek_time, lambda df: _joined(df, [src, ">", url_path, "@", event_time]))

    return DataFrameInputSchema(column_info=head + [
        src,
        _text("dst_ip", "id.resp_h"),
        _count("dst_port", "id.resp_p"),
        _text("http_method", "method"),
        url_path,
        _count("status_code", "status_code"),
        _text("user_agent", "user_agent"),
        # The field Zeek's HTTP post-body package adds; stock Zeek logs only `request_body_len`, and a log without
        # the field gives every request no body.
        _text("request_body", "post_body"),
    ] + tail)


def zeek_dns_schema(profile: CollectorProfile) -> DataFrameInputSchema:
    """
    `tc7_dns` from Zeek `dns.log`.

    Parameters
    ----------
    profile : `CollectorProfile`
        The collector's own facts.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """
    src = _text("src_ip", "id.orig_h")
    query = _text("query_name", "query")
    event_time = _derived("event_time", "Int64", _zeek_time)
    (head, tail) = _envelope(profile, _zeek_time, lambda df: _joined(df, [src, "?", query, "@", event_time]))

    return DataFrameInputSchema(column_info=head + [
        src,
        query,
        _text("query_type", "qtype_name"),
        _text("response_code", "rcode_name"),
    ] + tail)


def m365_audit_schema(profile: CollectorProfile,
                      organizations: typing.Optional[dict] = None,
                      record_count_field: str = "RecordCount") -> DataFrameInputSchema:
    """
    `tc7_saas` from Microsoft 365 `UnifiedAuditLog` records.

    Parameters
    ----------
    profile : `CollectorProfile`
        The collector's own facts. Its `tenant_id` is used for an `OrganizationId` not in `organizations`.
    organizations : dict, optional
        `OrganizationId` to the tenant name the searches group by.
    record_count_field : str, default = "RecordCount"
        Where the workload reports how many records an operation touched. The audit schema has no common field
        for it; name the one the workload writes.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """
    organizations = dict(organizations or {})

    def creation(df: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(df["CreationTime"], utc=True).astype("int64").astype("Int64")

    def tenant(df: pd.DataFrame) -> pd.Series:
        if ("OrganizationId" not in df.columns):
            return pd.Series([profile.tenant_id] * len(df), index=df.index, dtype=object)

        return _text_values(df["OrganizationId"]).map(lambda value: organizations.get(value, profile.tenant_id))

    def result(df: pd.DataFrame) -> pd.Series:
        return _text_values(df["ResultStatus"]).map(
            lambda value: None if value is None else M365_RESULTS.get(value.lower(), value.lower()))

    principal = _text("user_principal", "UserId")
    operation = _text("operation", "Operation")
    target = _text("target_object", "ObjectId")
    event_time = _derived("event_time", "Int64", creation)
    (head, tail) = _envelope(profile,
                             creation,
                             lambda df: _joined(df, [principal, ">", operation, ">", target, "@", event_time]),
                             tenant=tenant)

    return DataFrameInputSchema(column_info=head + [
        principal,
        operation,
        target,
        _text("target_object_type", "ItemType"),
        _count("record_count", record_count_field),
        _derived("result", "object", result),
        _text("client_app", "ClientInfoString"),
        _text("source_ip", "ClientIP"),
    ] + tail)


def _place(value: str) -> str:
    """A place name as the layer 5 searches group it: lower case, one hyphen for each run of white space."""
    return "-".join(value.lower().split())


def _mfa_result(steps: typing.Any) -> typing.Optional[str]:
    """
    `approved` or `denied` from a sign-in's second-factor steps; null where none was asked for.

    The first step of `authenticationDetails` is the first factor, whatever method it used: a password, a FIDO2 key or
    Windows Hello sign a passwordless user in with one step, and that step is not a second factor. Every step after
    the first is one, and the result is `approved` only if all of them succeeded.
    """
    if (_missing(steps) or isinstance(steps, str)):
        return None

    second = list(steps)[1:]

    if (len(second) == 0):
        return None

    return "approved" if all(bool(step.get("succeeded")) for step in second) else "denied"


def entra_signin_schema(profile: CollectorProfile) -> DataFrameInputSchema:
    """
    `tc5_auth` from Microsoft Entra ID sign-in records, their `properties` flattened as `read_entra` reads them.

    `auth_result` is `success` for `status.errorCode` 0 and `failure` for any other; `mfa_used` is whether
    `authenticationRequirement` asked for a second factor; `mfa_result` is whether every step of
    `authenticationDetails` after the password succeeded, and null where there was none; `source_asn` is
    `autonomousSystemNumber` in the `as5089` form the contract gives; places are lower-cased and hyphenated, and
    the principal lower-cased, as the contract asks.

    Parameters
    ----------
    profile : `CollectorProfile`
        The collector's own facts.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """
    fields = ENTRA_FIELDS

    def created(df: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(df[fields["event_time"]], utc=True).astype("int64").astype("Int64")

    def mfa_result(df: pd.DataFrame) -> pd.Series:
        source = fields["mfa_result"]
        values = df[source] if source in df.columns else [None] * len(df)

        return pd.Series([_mfa_result(value) for value in values], index=df.index, dtype=object)

    (head, tail) = _envelope(profile, created, lambda df: _text_values(df["id"]))

    return DataFrameInputSchema(column_info=head + [
        _through("user_principal", fields["user_principal"], str.lower),
        _text("source_ip", fields["source_ip"]),
        _text("app", fields["app"]),
        _through("auth_result", fields["auth_result"], lambda code: "success" if int(float(code)) == 0 else "failure"),
        # A domain controller's target computer; a cloud identity provider has none.
        _text("target_host", "target_host"),
        _through("source_country", fields["source_country"], str.lower),
        _through("source_region", fields["source_region"], _place),
        _through("source_city", fields["source_city"], _place),
        _number("source_latitude", fields["source_latitude"]),
        _number("source_longitude", fields["source_longitude"]),
        _through("source_asn", fields["source_asn"], lambda number: f"as{int(float(number))}"),
        _text("device_id", fields["device_id"]),
        _through("mfa_used", fields["mfa_used"], lambda value: ENTRA_MFA_REQUIREMENTS.get(value), dtype="boolean"),
        _derived("mfa_result", "object", mfa_result),
        _text("token_type", fields["token_type"], ENTRA_TOKEN_TYPES),
    ] + tail)


def sysmon_process_schema(profile: CollectorProfile) -> DataFrameInputSchema:
    """
    `tc7_endpoint` from Sysmon event 1 records joined by `join_image_signatures`.

    `command_line_hash` is the SHA-256 of `CommandLine`, so the command line itself, which often carries a
    credential, goes no further than this schema.

    Parameters
    ----------
    profile : `CollectorProfile`
        The collector's own facts.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """

    def utc(df: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(df["UtcTime"], utc=True, format="%Y-%m-%d %H:%M:%S.%f").astype("int64").astype("Int64")

    def command_line(df: pd.DataFrame) -> pd.Series:
        return _text_values(df["CommandLine"]).map(
            lambda value: None if value is None else hashlib.sha256(value.encode("utf-8")).hexdigest())

    host = _text("hostname", "Computer")
    image = _text("image_path", "Image")
    event_time = _derived("event_time", "Int64", utc)
    (head, tail) = _envelope(profile, utc, lambda df: _joined(df, [host, ">", image, "@", event_time]))

    return DataFrameInputSchema(column_info=head + [
        host,
        _text("process_guid", "ProcessGuid"),
        _text("parent_process_guid", "ParentProcessGuid"),
        _text("parent_image_path", "ParentImage"),
        image,
        _derived("command_line_hash", "object", command_line),
        _text("integrity_level", "IntegrityLevel"),
        _text("signature_status", "SignatureStatus"),
    ] + tail)


def _leaf(series: pd.Series) -> pd.Series:
    return _text_values(series).map(lambda value: None if value in (None, "(empty)") else value.split(",")[0])


def join_ssl_x509(ssl: pd.DataFrame, x509: pd.DataFrame) -> pd.DataFrame:
    """
    Each `ssl.log` row with its leaf certificate's `x509.log` row.

    Two ways to match, and the one the logs allow is used. Where `ssl.log` carries `cert_chain_fuids` and
    `x509.log` its file `id`, the leaf is the first file of the session's chain, one `x509.log` row per session.
    Otherwise -- Zeek 5 and later by default, which logs each certificate once -- it is the first of
    `cert_chain_fps` against `fingerprint`, which is exact for a real certificate, since one fingerprint is one
    certificate and one validity period.

    Parameters
    ----------
    ssl, x509 : `pandas.DataFrame`
        As `morpheus.parsers.zeek.parse` reads them.

    Returns
    -------
    `pandas.DataFrame`
        `ssl` in its own order, with the leaf's x509 columns and `fingerprint` beside it; null where the leaf was
        not logged.
    """
    if ("cert_chain_fuids" in ssl.columns and "id" in x509.columns):
        leaf = _leaf(ssl["cert_chain_fuids"])
        certificates = x509.drop_duplicates("id").set_index("id")
    else:
        leaf = _leaf(ssl["cert_chain_fps"])
        certificates = x509.drop_duplicates("fingerprint").set_index("fingerprint", drop=False)

    joined = ssl.copy()

    for column in ("fingerprint", "certificate.issuer", "certificate.not_valid_before", "certificate.not_valid_after"):
        joined[column] = leaf.map(certificates[column]).values

    return joined


def join_image_signatures(events: pd.DataFrame) -> pd.DataFrame:
    """
    Each Sysmon event 1 with the `SignatureStatus` of the event 7 that loaded its own image.

    Parameters
    ----------
    events : `pandas.DataFrame`
        As `read_sysmon` reads them, events 1 and 7 together.

    Returns
    -------
    `pandas.DataFrame`
        The event 1 rows in their own order, with `SignatureStatus`; null where no such event 7 was logged.
    """
    starts = events[events["EventID"] == 1].reset_index(drop=True)
    loads = events[events["EventID"] == 7]
    status = {
        (guid, loaded): value
        for (guid, loaded, value) in zip(loads["ProcessGuid"], loads["ImageLoaded"], loads["SignatureStatus"])
    }
    starts = starts.drop(columns=[column for column in ("SignatureStatus", "ImageLoaded") if column in starts.columns])
    starts["SignatureStatus"] = [
        status.get((guid, image)) for (guid, image) in zip(starts["ProcessGuid"], starts["Image"])
    ]

    return starts


def _number(name: str, source: str) -> CustomColumn:
    """A nullable float column read from `source`."""

    def column(df: pd.DataFrame) -> pd.Series:
        if (source not in df.columns):
            return pd.Series([None] * len(df), index=df.index, dtype="Float64")

        return pd.to_numeric(df[source]).astype("Float64")

    return CustomColumn(name=name, dtype="Float64", process_column_fn=column, input_column_types={source: "Float64"})


def _null_count(name: str) -> CustomColumn:
    return _derived(name, "Int64", lambda df: pd.Series([None] * len(df), index=df.index, dtype="Int64"))


def _whole_seconds(source: str) -> typing.Callable[[pd.DataFrame], pd.Series]:

    def column(df: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(df[source]).astype("Int64") * 1_000_000_000

    return column


SNMP_OPER_STATUS = {1: "up", 2: "down", 3: "testing", 4: "unknown", 5: "dormant", 6: "notPresent", 7: "lowerLayerDown"}
"""IF-MIB `ifOperStatus` as the contract writes it."""

BGP_ESTABLISHED = 6
"""BGP4-MIB `bgpPeerState` for a session that is up."""

MAC_NOTIFICATION_OPERATIONS = {1: "added", 2: "removed"}
"""CISCO-MAC-NOTIFICATION-MIB's operation byte in `cmnHistMacChangedMsg`: an address learnt, or one removed."""

ARP_OPERATIONS = {1: "request", 2: "reply"}
"""The ARP opcode (RFC 826) as the contract writes it."""

RADIUS_OUTCOMES = {"Access-Accept": "success", "Access-Reject": "failure"}
"""The reply that ends an 802.1X exchange, as the contract writes it."""

RADIUS_ACCOUNTING_ACTIONS = {"Start": "start", "Stop": "end"}
"""`Acct-Status-Type` as the contract's start-and-end vocabulary writes it; an `Interim-Update` passes through."""

INTERFACE_ABBREVIATIONS = {
    "HundredGigE": "Hu",
    "FortyGigabitEthernet": "Fo",
    "TwentyFiveGigE": "Twe",
    "TenGigabitEthernet": "Te",
    "GigabitEthernet": "Gi",
    "FastEthernet": "Fa",
}
"""The long interface names a switch sends in `NAS-Port-Id`, to the short ones its `ifName` reports."""

EAP_RESPONSE = 2
EAP_IDENTITY = 1


def _scale(value: typing.Any, precision: typing.Any) -> typing.Optional[float]:
    if (_missing(value)):
        return None

    return int(value) / 10**int(precision)


def _snmp_key(record: dict) -> tuple:
    return (record["tags"]["agent_host"], record["timestamp"])


def join_snmp_ports(records: list[dict]) -> pd.DataFrame:
    """
    One row per `interface` record of a Telegraf SNMP feed, with the poll's `sysUpTime` and the port's transceiver
    and LLDP neighbour beside it.

    Telegraf writes one measurement per table it walks, so a port's poll is four records with the same agent and
    timestamp: the `snmp` scalars, its `interface` row (IF-MIB and EtherLike-MIB, tagged `ifIndex` and `ifName`),
    its `transceiver` row (ENTITY-MIB and CISCO-ENTITY-SENSOR-MIB, joined to the port by Telegraf's secondary
    index) and its `lldp` row (LLDP-MIB, whose local port number these switches give as the `ifIndex`). A port
    with no neighbour that poll has no `lldp` row, and its neighbour columns are null.

    Parameters
    ----------
    records : list of dict
        As `read_telegraf` reads them.

    Returns
    -------
    `pandas.DataFrame`
        The `interface` records in their own order, tags and fields as columns.
    """
    scalars = {_snmp_key(record): record["fields"] for record in records if record["name"] == "snmp"}
    by_port = {
        (record["name"], ) + _snmp_key(record) + (record["tags"]["ifIndex"], ): record["fields"]
        for record in records if record["name"] in ("transceiver", "lldp")
    }
    rows = []

    for record in records:
        if (record["name"] != "interface"):
            continue

        key = _snmp_key(record) + (record["tags"]["ifIndex"], )
        rows.append({
            "timestamp": record["timestamp"],
            **record["tags"],
            "sysUpTime": scalars.get(_snmp_key(record), {}).get("sysUpTime"),
            **record["fields"],
            **by_port.get(("transceiver", ) + key, {}),
            **by_port.get(("lldp", ) + key, {}),
        })

    return pd.DataFrame(rows)


def join_snmp_devices(records: list[dict]) -> pd.DataFrame:
    """
    One row per `snmp` record of a Telegraf SNMP feed, with the number of its BGP peers established that poll.

    Parameters
    ----------
    records : list of dict
        As `read_telegraf` reads them.

    Returns
    -------
    `pandas.DataFrame`
        The `snmp` records in their own order, tags and fields as columns, and `bgp_established`, counted from the
        poll's `bgp_peer` records (BGP4-MIB `bgpPeerTable`).
    """
    established: dict = {}

    for record in records:
        if (record["name"] == "bgp_peer" and record["fields"].get("bgpPeerState") == BGP_ESTABLISHED):
            established[_snmp_key(record)] = established.get(_snmp_key(record), 0) + 1

    return pd.DataFrame([{
        "timestamp": record["timestamp"],
        **record["tags"],
        **record["fields"],
        "bgp_established": established.get(_snmp_key(record), 0),
    } for record in records if record["name"] == "snmp"])


def _mac(octets: typing.Sequence[int]) -> str:
    return ":".join(f"{octet:02x}" for octet in octets)


def join_fdb_ports(records: list[dict]) -> pd.DataFrame:
    """
    One row per forwarding-table entry and per MAC notification in a Telegraf SNMP feed, each with its port's name.

    An entry is a `fdb` record (Q-BRIDGE-MIB `dot1qTpFdbTable`), whose index is the VLAN and the address's six
    octets and whose `dot1qTpFdbPort` is a bridge port. A notification is a `snmp_trap` record of
    CISCO-MAC-NOTIFICATION-MIB's `cmnMacChangedNotification`, whose `cmnHistMacChangedMsg` packs the operation,
    the VLAN, the address and the bridge port. A bridge port is named through the poll's `bridge_port` records
    (BRIDGE-MIB `dot1dBasePortIfIndex`) and `ifXTable` records (IF-MIB `ifName`), the latest the agent answered;
    a trap names its agent by address only, so its switch and site are the ones that agent's last poll was tagged
    with.

    Parameters
    ----------
    records : list of dict
        As `read_telegraf` reads them.

    Returns
    -------
    `pandas.DataFrame`
        In file order: `timestamp`, `sysName`, `sysLocation`, `ifName`, `vlan`, `mac` and `operation`, the last
        null for a table entry.
    """
    interfaces: dict = {}
    bridge_ports: dict = {}
    agents: dict = {}
    rows = []

    def port(agent: str, bridge_port: int) -> typing.Optional[str]:
        if_index = bridge_ports.get((agent, bridge_port))

        return None if if_index is None else interfaces.get((agent, if_index))

    for record in records:
        tags = record["tags"]

        if (record["name"] == "ifXTable"):
            interfaces[(tags["agent_host"], int(tags["ifIndex"]))] = record["fields"]["ifName"]
        elif (record["name"] == "bridge_port"):
            bridge_ports[(tags["agent_host"], int(tags["index"]))] = record["fields"]["dot1dBasePortIfIndex"]
        elif (record["name"] == "fdb"):
            agents[tags["agent_host"]] = (tags.get("sysName"), tags.get("sysLocation"))
            (vlan, *octets) = (int(part) for part in tags["index"].split("."))
            rows.append({
                "timestamp": record["timestamp"],
                "sysName": tags.get("sysName"),
                "sysLocation": tags.get("sysLocation"),
                "ifName": port(tags["agent_host"], record["fields"]["dot1qTpFdbPort"]),
                "vlan": vlan,
                "mac": _mac(octets),
                "operation": None,
            })
        elif (record["name"] == "snmp_trap" and tags.get("name") == "cmnMacChangedNotification"):
            message = bytes.fromhex(record["fields"]["cmnHistMacChangedMsg"])
            (sys_name, location) = agents.get(tags["source"], (None, None))
            rows.append({
                "timestamp": record["timestamp"],
                "sysName": sys_name,
                "sysLocation": location,
                "ifName": port(tags["source"], int.from_bytes(message[9:11], "big")),
                "vlan": int.from_bytes(message[1:3], "big"),
                "mac": _mac(message[3:9]),
                "operation": MAC_NOTIFICATION_OPERATIONS.get(message[0], str(message[0])),
            })

    return pd.DataFrame(rows)


def _eap(value: typing.Any) -> typing.Optional[bytes]:
    if (_missing(value) or not str(value).startswith("0x")):
        return None

    return bytes.fromhex(str(value)[2:])


def join_radius_replies(entries: list[dict]) -> pd.DataFrame:
    """
    The steps of each 802.1X exchange in a FreeRADIUS detail file of requests (`auth_log`) and replies
    (`reply_log`).

    An exchange starts with the Access-Request whose EAP-Message is the supplicant's EAP-Response/Identity, and
    ends with the Access-Accept or Access-Reject the server sent. A reply carries the `User-Name` the EAP module
    puts in it but not the port or the address; the detail module stamps both a request and its reply with
    `Timestamp`, when the server received the request, so the reply's port and address are its request's: the
    request with the same `User-Name` and `Timestamp`. The rounds of the exchange between the two are not steps.

    Parameters
    ----------
    entries : list of dict
        As `read_radius_detail` reads them.

    Returns
    -------
    `pandas.DataFrame`
        One row per step in file order: the request's attributes, the reply's `Packet-Type` as `outcome` (null for
        the start), and `Timestamp`.
    """
    requests: dict = {}
    rows = []

    for entry in entries:
        kind = entry.get("Packet-Type")

        if (kind == "Access-Request"):
            requests[(entry.get("User-Name"), entry.get("Timestamp"))] = entry
            eap = _eap(entry.get("EAP-Message"))

            if (eap is not None and len(eap) > 4 and eap[0] == EAP_RESPONSE and eap[4] == EAP_IDENTITY):
                rows.append({**entry, "outcome": None})
        elif (kind in RADIUS_OUTCOMES):
            request = requests.get((entry.get("User-Name"), entry.get("Timestamp")), {})
            rows.append({**request, "Timestamp": entry.get("Timestamp"), "outcome": kind})

    return pd.DataFrame(rows)


def _interface_name(value: str) -> str:
    for (long_name, short_name) in INTERFACE_ABBREVIATIONS.items():
        if (value.startswith(long_name)):
            return short_name + value[len(long_name):]

    return value


def _station_mac(value: str) -> str:
    return value.lower().replace("-", ":")


def _event_timestamp(df: pd.DataFrame) -> pd.Series:
    # The detail module writes an `Event-Timestamp` as the date in UTC, `Jan  1 1970 00:00:00 UTC`.
    return pd.Series([
        pd.Timestamp(pd.to_datetime(" ".join(str(value).split()), format="%b %d %Y %H:%M:%S UTC", utc=True)).value
        for value in df["Event-Timestamp"]
    ],
                     index=df.index,
                     dtype="Int64")


def snmp_port_schema(profile: CollectorProfile) -> DataFrameInputSchema:
    """
    `tc1` from a Telegraf SNMP feed's port polls, joined by `join_snmp_ports`.

    The site is `sysLocation` and the device `sysName`, both tagged on every record; `uptime` and
    `if_last_change` are TimeTicks as the device reports them; `link_speed_bps` is `ifHighSpeed`, in Mb/s; the
    optical powers are CISCO-ENTITY-SENSOR-MIB values scaled by their `entSensorPrecision`; `crc_errors` is
    EtherLike-MIB's `dot3StatsFCSErrors`. Telegraf's JSON timestamp is the poll's, in seconds.

    Parameters
    ----------
    profile : `CollectorProfile`
        The collector's own facts.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """
    event_time = _whole_seconds("timestamp")

    def optical(name: str, source: str) -> CustomColumn:
        return _derived(
            name,
            "Float64", lambda df: pd.Series(
                [_scale(value, precision)
                 for (value, precision) in zip(df[source], df["entSensorPrecision"])], index=df.index, dtype="Float64"))

    (head, tail) = _envelope(
        profile,
        event_time, lambda df: _joined(df, [_text("sysName", "sysName"), "/", _text("ifName", "ifName"), "@"]) + pd.
        Series([str(value) for value in event_time(df)], index=df.index))

    return DataFrameInputSchema(column_info=head + [
        _text("site_id", "sysLocation"),
        _text("device_id", "sysName"),
        _text("port_id", "ifName"),
        _count("if_index", "ifIndex"),
        _count("uptime", "sysUpTime"),
        _count("if_last_change", "ifLastChange"),
        _through("oper_status", "ifOperStatus", lambda value: SNMP_OPER_STATUS.get(int(value), value)),
        _derived("link_speed_bps", "Int64", lambda df: pd.to_numeric(df["ifHighSpeed"]).astype("Int64") * 1_000_000),
        optical("optical_tx_dbm", "txPower"),
        optical("optical_rx_dbm", "rxPower"),
        _text("transceiver_serial", "entPhysicalSerialNum"),
        _text("transceiver_type", "entPhysicalDescr"),
        _text("lldp_neighbor_chassis_id", "lldpRemChassisId"),
        _text("lldp_neighbor_port_id", "lldpRemPortId"),
        _count("crc_errors", "dot3StatsFCSErrors"),
        _count("symbol_errors", "dot3StatsSymbolErrors"),
        _count("input_discards", "ifInDiscards"),
        _count("output_discards", "ifOutDiscards"),
        _count("if_hc_in_octets", "ifHCInOctets"),
        _count("if_hc_out_octets", "ifHCOutOctets"),
    ] + tail)


def snmp_device_schema(profile: CollectorProfile) -> DataFrameInputSchema:
    """
    `tc1_device` from a Telegraf SNMP feed's scalars, joined by `join_snmp_devices`.

    `cpu_util` is CISCO-PROCESS-MIB's `cpmCPUTotal5minRev`, a percentage, as a fraction; `mem_util` is
    CISCO-MEMORY-POOL-MIB's used bytes over used and free; `route_count` is IP-FORWARD-MIB's
    `inetCidrRouteNumber`. No MIB a switch answers gives a hash of its running configuration or a rate of the
    syslog messages it sends -- CISCO-CONFIG-MAN-MIB says when the configuration last changed, not what it is --
    so `running_config_hash` and `syslog_rate` are null from SNMP; a configuration archive and a syslog
    collector supply them.

    Parameters
    ----------
    profile : `CollectorProfile`
        The collector's own facts.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """
    event_time = _whole_seconds("timestamp")

    def memory(df: pd.DataFrame) -> pd.Series:
        used = pd.to_numeric(df["ciscoMemoryPoolUsed"]).astype("Float64")

        return used / (used + pd.to_numeric(df["ciscoMemoryPoolFree"]).astype("Float64"))

    (head, tail) = _envelope(
        profile,
        event_time, lambda df: _joined(df, [_text("sysName", "sysName"), "@"]) + pd.Series(
            [str(value) for value in event_time(df)], index=df.index))

    return DataFrameInputSchema(column_info=head + [
        _text("site_id", "sysLocation"),
        _text("device_id", "sysName"),
        _count("sys_uptime", "sysUpTime"),
        _text("running_config_hash", "running_config_hash"),
        _derived("cpu_util", "Float64", lambda df: pd.to_numeric(df["cpmCPUTotal5minRev"]).astype("Float64") / 100),
        _derived("mem_util", "Float64", memory),
        _count("route_count", "inetCidrRouteNumber"),
        _count("bgp_peer_count", "bgp_established"),
        _number("syslog_rate", "syslog_rate"),
    ] + tail)


def fdb_schema(profile: CollectorProfile) -> DataFrameInputSchema:
    """
    `tc2_mac` from a Telegraf SNMP feed's forwarding tables and MAC notifications, joined by `join_fdb_ports`.

    Parameters
    ----------
    profile : `CollectorProfile`
        The collector's own facts.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """
    event_time = _whole_seconds("timestamp")
    (head, tail) = _envelope(
        profile,
        event_time, lambda df: _joined(df, [
            _text("sysName", "sysName"), "/", _text("ifName", "ifName"), "/", _text("mac", "mac"), "@"
        ]) + pd.Series([str(value) for value in event_time(df)], index=df.index))

    return DataFrameInputSchema(column_info=head + [
        _text("mac_address", "mac"),
        _text("site_id", "sysLocation"),
        _text("switch_id", "sysName"),
        _text("port_id", "ifName"),
        _count("vlan_id", "vlan"),
        _text("mac_action", "operation"),
    ] + tail)


def arp_schema(profile: CollectorProfile) -> DataFrameInputSchema:
    """
    `tc2_arp` from the ARP packets of a capture, as `read_arp` reads them.

    Parameters
    ----------
    profile : `CollectorProfile`
        The collector's own facts.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """

    def captured(df: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(df["received_ns"]).astype("Int64")

    (head, tail) = _envelope(
        profile,
        captured, lambda df: _joined(df, [_text("spa", "spa"), ">", _text("sha", "sha"), "?", _text("tpa", "tpa"), "@"])
        + pd.Series([str(value) for value in captured(df)], index=df.index))

    return DataFrameInputSchema(column_info=head + [
        _text("arp_sender_ip", "spa"),
        _text("arp_sender_mac", "sha"),
        _text("arp_target_ip", "tpa"),
        _through("arp_operation", "operation", lambda value: ARP_OPERATIONS.get(int(value), value)),
    ] + tail)


def radius_auth_schema(profile: CollectorProfile, sites: dict) -> DataFrameInputSchema:
    """
    `tc2_auth` from a FreeRADIUS detail file of 802.1X requests and replies, joined by `join_radius_replies`.

    The switch is `NAS-Identifier` and its site is the deployment's, since RADIUS carries no location; the port
    is `NAS-Port-Id` under the short name the switch's `ifName` gives it; the address is `Calling-Station-Id`
    with colons, lower case.

    Parameters
    ----------
    profile : `CollectorProfile`
        The collector's own facts.
    sites : dict
        `NAS-Identifier` to site, as the deployment's inventory gives it.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """
    event_time = _whole_seconds("Timestamp")
    port = _through("port_id", "NAS-Port-Id", _interface_name)
    mac = _through("mac_address", "Calling-Station-Id", _station_mac)

    def result(df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            ["started" if _missing(value) else RADIUS_OUTCOMES.get(value, value) for value in df["outcome"]],
            index=df.index,
            dtype=object)

    (head, tail) = _envelope(
        profile,
        event_time, lambda df: _joined(df, [_text("switch", "NAS-Identifier"), "/", port, "/", mac, "@"]) + pd.Series(
            [str(value) for value in event_time(df)], index=df.index))

    return DataFrameInputSchema(column_info=head + [
        _text("site_id", "NAS-Identifier", sites),
        _text("switch_id", "NAS-Identifier"),
        port,
        mac,
        _text("dot1x_identity", "User-Name"),
        _derived("dot1x_result", "object", result),
    ] + tail)


def radius_accounting_schema(profile: CollectorProfile) -> DataFrameInputSchema:
    """
    `tc5_session` from a FreeRADIUS detail file of accounting requests.

    Each Start and Stop is one record, in the contract's start-and-end vocabulary, timed by `Event-Timestamp`,
    when the NAS says it happened. A Stop's `Acct-Session-Time` is not read as the session's start: the stage
    pairs the start and end records, and a Stop whose Start was lost has to read as one.

    Parameters
    ----------
    profile : `CollectorProfile`
        The collector's own facts.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """
    action = _text("session_action", "Acct-Status-Type", RADIUS_ACCOUNTING_ACTIONS)
    (head, tail) = _envelope(profile,
                             _event_timestamp,
                             lambda df: _joined(df, [_text("session", "Acct-Session-Id"), ":", action]))

    return DataFrameInputSchema(column_info=head + [
        _text("session_id", "Acct-Session-Id"),
        _text("user_principal", "User-Name"),
        action,
        _text("source_ip", "Framed-IP-Address"),
        _null_count("session_start"),
        _null_count("session_end"),
    ] + tail)


IP_PROTOCOLS = {6: "tcp", 17: "udp"}
"""The IP protocol numbers a packet record names."""

TCP_OPTIONS = {0: "eol", 1: "nop", 2: "mss", 3: "ws", 4: "sok", 5: "sack", 8: "ts"}
"""TCP option kinds (RFC 9293, 7323, 2018) by the names `tcp_options_order` writes them in."""

TCP_SYN = 0x02
TCP_ACK = 0x10

KEA_MESSAGES = {"DHCP4_LEASE_ALLOC": "ack", "DHCP4_RELEASE": "release", "DHCP4_DECLINE": "decline"}
"""The Kea DHCPv4 server's log codes for what it granted, released and had declined, as the contract writes them."""

KEA_LINE = re.compile(r"^(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}) +\w+ +\[[^\]]*\] (?P<code>DHCP4_\w+) "
                      r"\[hwtype=\d+ (?P<mac>[0-9a-f:]+)\][^:]*: (?:lease (?P<lease_ip>[\d.]+) has been allocated "
                      r"for (?P<seconds>\d+) seconds|address (?P<released>[\d.]+) .*)$")


def _tcp_options(data: bytes) -> tuple[list[str], typing.Optional[int]]:
    (names, mss, offset) = ([], None, 0)

    while (offset < len(data)):
        kind = data[offset]
        names.append(TCP_OPTIONS.get(kind, str(kind)))

        if (kind == 0):
            break

        if (kind == 1):
            offset += 1
            continue

        length = data[offset + 1] if offset + 1 < len(data) else 0

        if (kind == 2 and length == 4):
            mss = int.from_bytes(data[offset + 2:offset + 4], "big")

        offset += max(length, 2)

    return (names, mss)


def read_packets(path: str) -> pd.DataFrame:
    """
    Every IPv4 TCP and UDP packet in a capture, read with `morpheus.utils.ipfix.read_frames`.

    The payload length is the IP datagram's, less its headers, so a capture that snapped each frame short reads
    the length the packet had.

    Returns
    -------
    `pandas.DataFrame`
        One row per packet in capture order: `received_ns`, the addresses and ports, `protocol` (the number),
        `flags`, `window` and `options` (the option names in order, comma separated) for TCP, `mss`, `data_len`.
    """
    rows = []

    for (received_ns, frame) in ipfix.read_frames(path):
        (ethertype, offset) = (int.from_bytes(frame[12:14], "big"), 14)

        if (ethertype == ipfix.ETHERTYPE_VLAN):
            (ethertype, offset) = (int.from_bytes(frame[16:18], "big"), 18)

        if (ethertype != ipfix.ETHERTYPE_IPV4 or len(frame) < offset + 20):
            continue

        header = (frame[offset] & 0x0F) * 4
        (total, protocol) = (int.from_bytes(frame[offset + 2:offset + 4], "big"), frame[offset + 9])
        transport = offset + header

        if (protocol not in IP_PROTOCOLS or len(frame) < transport + 8):
            continue

        row = {
            "received_ns": received_ns,
            "src": str(ipaddress.IPv4Address(frame[offset + 12:offset + 16])),
            "dst": str(ipaddress.IPv4Address(frame[offset + 16:offset + 20])),
            "sport": int.from_bytes(frame[transport:transport + 2], "big"),
            "dport": int.from_bytes(frame[transport + 2:transport + 4], "big"),
            "protocol": protocol,
            "flags": None,
            "window": None,
            "options": None,
            "mss": None,
        }

        if (protocol == 6):
            length = (frame[transport + 12] >> 4) * 4
            (names, mss) = _tcp_options(frame[transport + 20:transport + length])
            row.update({
                "flags": frame[transport + 13],
                "window": int.from_bytes(frame[transport + 14:transport + 16], "big"),
                "options": ",".join(names) if names else None,
                "mss": mss,
                "data_len": total - header - length,
            })
        else:
            row["data_len"] = total - header - 8

        rows.append(row)

    return pd.DataFrame(rows,
                        columns=[
                            "received_ns",
                            "src",
                            "dst",
                            "sport",
                            "dport",
                            "protocol",
                            "flags",
                            "window",
                            "options",
                            "mss",
                            "data_len"
                        ])


def packet_schema(profile: CollectorProfile) -> DataFrameInputSchema:
    """
    `tc4` from the packets of a capture, as `read_packets` reads them.

    The three fingerprint columns are one fingerprint, as `TC4FlowStage` reads them: they are written for a client
    `SYN` (`SYN` without `ACK`) that carries options, and null otherwise. A `SYN` crafted with no options, as a
    scanner's raw socket writes it, still has a window, but no stack wrote it to fingerprint.

    Parameters
    ----------
    profile : `CollectorProfile`
        The collector's own facts.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """

    def captured(df: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(df["received_ns"]).astype("Int64")

    def fingerprinted(df: pd.DataFrame) -> pd.Series:
        return pd.Series([
            not _missing(flags) and int(flags) & (TCP_SYN | TCP_ACK) == TCP_SYN and not _missing(options)
            for (flags, options) in zip(df["flags"], df["options"])
        ],
                         index=df.index)

    def fingerprint(name: str, source: str, dtype: str) -> CustomColumn:
        return _derived(
            name,
            dtype, lambda df: pd.Series([
                value if keep and not _missing(value) else None for (keep, value) in zip(fingerprinted(df), df[source])
            ], index=df.index, dtype=dtype))

    (head, tail) = _envelope(
        profile,
        captured, lambda df: _joined(df, [
            _text("src", "src"), ":", _count("sport", "sport"), "=", _text("dst", "dst"), ":", _count("dport", "dport")
        ]))

    return DataFrameInputSchema(column_info=head + [
        _text("src_ip", "src"),
        _count("src_port", "sport"),
        _text("dst_ip", "dst"),
        _count("dst_port", "dport"),
        _through("protocol", "protocol", lambda value: IP_PROTOCOLS.get(int(value), value)),
        _count("tcp_flags", "flags"),
        _count("data_len", "data_len"),
        fingerprint("tcp_options_order", "options", "object"),
        fingerprint("tcp_mss", "mss", "Int64"),
        fingerprint("tcp_window_initial", "window", "Int64"),
    ] + tail)


def read_kea(path: str) -> pd.DataFrame:
    """
    The leases a Kea DHCPv4 server's log records: each `DHCP4_LEASE_ALLOC`, with the lease time it granted, and
    each `DHCP4_RELEASE` and `DHCP4_DECLINE`, from lines in Kea's default layout, the time in UTC.

    Returns
    -------
    `pandas.DataFrame`
        One row per such line in file order: `time` (ns), `code`, `mac`, `ip` and `lease_seconds` (null but for an
        allocation). Every other line is skipped.
    """
    rows = []

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            match = KEA_LINE.match(line.rstrip("\n"))

            if (match is None):
                continue

            rows.append({
                "time": pd.Timestamp(match["time"], tz="UTC").value,
                "code": match["code"],
                "mac": match["mac"],
                "ip": match["lease_ip"] or match["released"],
                "lease_seconds": None if match["seconds"] is None else int(match["seconds"]),
            })

    return pd.DataFrame(rows, columns=["time", "code", "mac", "ip", "lease_seconds"])


def kea_schema(profile: CollectorProfile) -> DataFrameInputSchema:
    """
    `tc3_dhcp` from a Kea DHCPv4 server's lease log, as `read_kea` reads it.

    An allocation is the server's acknowledgement, whether of a new lease or a renewal. Kea's lease log does not
    write the relay's option 82, so `switch_id` and `port_id` are null from it; a deployment that needs them reads
    the server's packet log or its lease database's `relay_agent_info`.

    Parameters
    ----------
    profile : `CollectorProfile`
        The collector's own facts.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """

    def when(df: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(df["time"]).astype("Int64")

    (head, tail) = _envelope(
        profile,
        when, lambda df: _joined(df, [_text("ip", "ip"), ">", _text("mac", "mac"), "@"]) + pd.Series(
            [str(value) for value in when(df)], index=df.index))

    return DataFrameInputSchema(column_info=head + [
        _text("ip", "ip"),
        _text("mac", "mac"),
        _text("dhcp_message", "code", KEA_MESSAGES),
        _count("lease_seconds", "lease_seconds"),
        _text("switch_id", "switch_id"),
        _text("port_id", "port_id"),
    ] + tail)


def read_osquery(path: str) -> pd.DataFrame:
    """
    The interfaces in osquery's scheduled-query results log, one JSON object per line: each snapshot's rows, or a
    differential result's `columns` where its `action` is `added`.

    Returns
    -------
    `pandas.DataFrame`
        One row per interface in file order: `unixTime`, `hostIdentifier`, `interface` and `mac`.
    """
    rows = []

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if (not line.strip()):
                continue

            result = json.loads(line)
            reported = result.get("snapshot") if result.get("action") == "snapshot" else (
                [result["columns"]] if result.get("action") == "added" else [])

            for interface in reported:
                rows.append({
                    "unixTime": result["unixTime"],
                    "hostIdentifier": result["hostIdentifier"],
                    "interface": interface.get("interface"),
                    "mac": interface.get("mac"),
                })

    return pd.DataFrame(rows, columns=["unixTime", "hostIdentifier", "interface", "mac"])


def osquery_inventory_schema(profile: CollectorProfile, report_valid_seconds: int) -> DataFrameInputSchema:
    """
    `tc7_host_inventory` from osquery's `interface_details` results, as `read_osquery` reads them.

    The machine is the agent's `hostIdentifier`, configured as its hostname. How long a report stands is the
    deployment's schedule, which no result carries, so it is given.

    Parameters
    ----------
    profile : `CollectorProfile`
        The collector's own facts.
    report_valid_seconds : int
        How long one report stands: the query's interval and the misses the deployment tolerates.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """
    event_time = _whole_seconds("unixTime")
    (head, tail) = _envelope(
        profile,
        event_time,
        lambda df: _joined(df, [_text("mac", "mac"), "@"]) + pd.Series([str(value)
                                                                        for value in event_time(df)], index=df.index))

    return DataFrameInputSchema(column_info=head + [
        _text("mac", "mac"),
        _text("hostname", "hostIdentifier"),
        _constant("report_valid_seconds", report_valid_seconds),
    ] + tail)


def join_conn_ssl(conn: pd.DataFrame, ssl: pd.DataFrame) -> pd.DataFrame:
    """
    Each Zeek `conn.log` connection that `ssl.log` holds a handshake for, with its `server_name` and `ja4`, joined
    on `uid`.

    Parameters
    ----------
    conn, ssl : `pandas.DataFrame`
        As `morpheus.parsers.zeek.parse` reads them.

    Returns
    -------
    `pandas.DataFrame`
        Those `conn` rows in their own order, with `server_name` and `ja4`.
    """
    handshakes = ssl.drop_duplicates("uid").set_index("uid")
    joined = conn[conn["uid"].isin(handshakes.index)].reset_index(drop=True)

    for column in ("server_name", "ja4"):
        joined[column] = joined["uid"].map(handshakes[column]).values if column in handshakes.columns else None

    return joined


def zeek_service_schema(profile: CollectorProfile) -> DataFrameInputSchema:
    """
    `tc7_service` from Zeek `conn.log` joined to `ssl.log` by `join_conn_ssl`.

    `bytes_total` is `orig_bytes` and `resp_bytes` together. The origin names the `service` as its client asked for
    it: by its `server_name` where the handshake had one, otherwise by its address.

    Parameters
    ----------
    profile : `CollectorProfile`
        The collector's own facts.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """
    sni = _text("sni", "server_name")

    def origin(df: pd.DataFrame) -> pd.Series:
        names = _text_values(df["server_name"]) if "server_name" in df.columns else [None] * len(df)

        return pd.Series([
            f"{client}->{name or server}:{port}@{when}"
            for (client, name, server, port, when) in zip(_text_values(df["id.orig_h"]), names, _text_values(
                df["id.resp_h"]), pd.to_numeric(df["id.resp_p"]).astype("Int64"), _zeek_time(df))
        ],
                         index=df.index,
                         dtype=object)

    def total(df: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(df["orig_bytes"]).astype("Int64") + pd.to_numeric(df["resp_bytes"]).astype("Int64")

    (head, tail) = _envelope(profile, _zeek_time, origin)

    return DataFrameInputSchema(column_info=head + [
        _text("src_ip", "id.orig_h"),
        _text("dst_ip", "id.resp_h"),
        _count("dst_port", "id.resp_p"),
        sni,
        _derived("bytes_total", "Int64", total),
        _text("ja4_client", "ja4"),
    ] + tail)


SERVICENOW_CRITICALITY = {
    "1 - most critical": "high",
    "2 - somewhat critical": "medium",
    "3 - less critical": "low",
    "4 - not critical": "none"
}
"""ServiceNow's `business_criticality` choices as the contract writes them."""

SERVICENOW_RETIRED = "7"
"""The `install_status` of a retired configuration item: the CMDB's way of saying a machine or service is gone."""

SERVICENOW_ASSET_ROLES = {"cmdb_ci_server": "server", "cmdb_ci_pc_hardware": "workstation"}
"""A machine's CMDB class as the role the contract names it by."""

SERVICENOW_APPLICATIONS = ("cmdb_ci_appl", )

NETBOX_KINDS = {"dcim.device": "device", "ipam.vlan": "vlan", "dcim.interface": "port"}
"""NetBox's object types for what `tc0_network` describes."""

WORKDAY_RETRACTIONS = frozenset({"End Contingent Worker Contract"})
"""The Workday business processes after which the worker is no longer in the directory's population."""

ENTRA_GROUP_CHANGES = {"Add member to group": "assert", "Remove member from group": "retract"}
"""Entra ID's audit activities for group membership, as the change each records."""


def _utc(value: typing.Any) -> typing.Optional[int]:
    if (_missing(value) or value == ""):
        return None

    return pd.Timestamp(pd.to_datetime(value, utc=True)).value


def _context_columns(tenant_id: str) -> tuple[list, list]:
    """A context record's sequence before its class's columns, and its clocks, change and tenant after."""
    head = [
        _derived("source_seq", "Int64", lambda df: pd.Series(range(len(df)), index=df.index, dtype="Int64")),
    ]
    tail = [
        _derived("valid_from",
                 "Int64",
                 lambda df: pd.Series([_utc(value) for value in df["valid_from"]], index=df.index, dtype="Int64")),
        _derived("valid_to",
                 "Int64",
                 lambda df: pd.Series([_utc(value) for value in df["valid_to"]], index=df.index, dtype="Int64")),
        _derived("recorded_at",
                 "Int64",
                 lambda df: pd.Series([_utc(value) for value in df["recorded_at"]], index=df.index, dtype="Int64")),
        _derived(
            "change",
            "object", lambda df: pd.Series(["retract" if gone else "assert"
                                            for gone in df["retracted"]], index=df.index, dtype=object)),
        _constant("tenant_id", tenant_id),
    ]

    return (head, tail)


def read_servicenow(path: str) -> list[dict]:
    """The records of a ServiceNow Table API response, `{"result": [...]}`, as written."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["result"]


def fold_cmdb_audit(audit: list[dict], users: list[dict], groups: list[dict],
                    tables: typing.Iterable[str]) -> pd.DataFrame:
    """
    Each version of the configuration items in `tables`, folded from ServiceNow's `sys_audit` history.

    An audit row is one field's change; the rows of one item that share a `record_checkpoint` are one update, and
    the item after it is one version. Reference fields hold a `sys_id`, so `assigned_to` is named by the user's
    `email` and `support_group` by the group's `name`. A version that sets `install_status` to retired is the item
    leaving the CMDB, with what it was.

    Parameters
    ----------
    audit : list of dict
        `sys_audit` rows, in the order the CMDB wrote them.
    users, groups : list of dict
        `sys_user` and `sys_user_group` records.
    tables : iterable of str
        The CMDB classes to fold.

    Returns
    -------
    `pandas.DataFrame`
        One row per version in order: the item's fields as of that version, `tablename`, `recorded_at` (the
        update's `sys_created_on`), `retracted`, and the `valid_from` and `valid_to` the CMDB does not keep, null.
    """
    (emails, names) = ({
        user["sys_id"]: user.get("email")
        for user in users
    }, {
        group["sys_id"]: group.get("name")
        for group in groups
    })
    wanted = set(tables)
    (items, rows, last) = ({}, [], None)

    for change in audit:
        if (change["tablename"] not in wanted):
            continue

        key = (change["documentkey"], change["record_checkpoint"])
        item = items.setdefault(change["documentkey"], {})
        item[change["fieldname"]] = change["newvalue"]

        if (key != last):
            rows.append({"tablename": change["tablename"], "recorded_at": change["sys_created_on"], "key": key})
            last = key

        rows[-1]["state"] = dict(item)
        rows[-1]["retracted"] = (change["fieldname"] == "install_status" and change["newvalue"] == SERVICENOW_RETIRED
                                 or rows[-1].get("retracted", False))

    return pd.DataFrame([{
        "tablename": row["tablename"],
        **row["state"],
        "assigned_to": emails.get(row["state"].get("assigned_to")),
        "support_group": names.get(row["state"].get("support_group")),
        "recorded_at": row["recorded_at"],
        "retracted": row["retracted"],
        "valid_from": None,
        "valid_to": None,
    } for row in rows])


def servicenow_asset_schema(tenant_id: str) -> DataFrameInputSchema:
    """
    `tc0_asset` from ServiceNow CMDB versions of servers and workstations, folded by `fold_cmdb_audit`.

    The CMDB keeps when a change was entered, not when it took effect, so `valid_from` is null from it; a change
    entered late reads as entered. The machine's role is its CMDB class; `peer_group` and `data_classification`
    are the estate's custom fields `u_peer_group` and `u_data_classification`; the operating system's family is
    `os`, lower-cased.

    Parameters
    ----------
    tenant_id : str
        The CMDB's tenant.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """
    (head, tail) = _context_columns(tenant_id)

    return DataFrameInputSchema(column_info=head + [
        _text("hostname", "name"),
        _text("owner", "assigned_to"),
        _text("owning_team", "support_group"),
        _text("criticality", "business_criticality", SERVICENOW_CRITICALITY),
        _text("data_classification", "u_data_classification"),
        _text("peer_group", "u_peer_group"),
        _text("device_role", "tablename", SERVICENOW_ASSET_ROLES),
        _through("os_family", "os", str.lower),
        _text("os_version", "os_version"),
    ] + tail)


def servicenow_application_schema(tenant_id: str) -> DataFrameInputSchema:
    """
    `tc0_application` from ServiceNow CMDB versions of applications, folded by `fold_cmdb_audit`.

    As for assets, `valid_from` is null from the CMDB. The address and port are the application's `ip_address`
    and `tcp_port`; the account it runs as and its data classification are the custom fields `u_service_account`
    and `u_data_classification`.

    Parameters
    ----------
    tenant_id : str
        The CMDB's tenant.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """
    (head, tail) = _context_columns(tenant_id)

    return DataFrameInputSchema(column_info=head + [
        _text("server_address", "ip_address"),
        _count("server_port", "tcp_port"),
        _text("name", "name"),
        _text("owning_team", "support_group"),
        _text("service_account", "u_service_account"),
        _text("criticality", "business_criticality", SERVICENOW_CRITICALITY),
        _text("data_classification", "u_data_classification"),
    ] + tail)


def read_netbox_changes(path: str) -> list[dict]:
    """The object changes of a NetBox `/api/core/object-changes/` response, oldest first, as written."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["results"]


def fold_netbox_changes(changes: list[dict]) -> pd.DataFrame:
    """
    Each change NetBox's changelog holds to a device, a VLAN or an interface, with what it refers to named.

    A change's `postchange_data` (its `prechange_data` for a deletion) refers to other objects by id, so every
    object the log creates is kept, and a device's site, role and tenant are named by their slugs, an interface's
    device by its name. An interface has no tenant; its owner is its device's.

    Parameters
    ----------
    changes : list of dict
        NetBox object changes, oldest first.

    Returns
    -------
    `pandas.DataFrame`
        One row per change to the three kinds, in order: `object_kind`, the site, device, port and VLAN, the
        description, role, tenant and custom fields, `recorded_at` (the change's `time`), `retracted`, and the
        `valid_from` and `valid_to` NetBox does not keep, null.
    """
    objects: dict = {}
    rows = []

    def slug(kind: str, identifier: typing.Any) -> typing.Optional[str]:
        return objects.get((kind, identifier), {}).get("slug")

    for change in changes:
        kind = change["changed_object_type"]
        deleted = change["action"]["value"] == "delete"
        data = change["prechange_data"] if deleted else change["postchange_data"]
        objects[(kind, change["changed_object_id"])] = data

        if (kind not in NETBOX_KINDS):
            continue

        device = objects.get(("dcim.device", data.get("device")), {}) if kind == "dcim.interface" else data
        fields = data.get("custom_fields") or {}
        rows.append({
            "object_kind":
                NETBOX_KINDS[kind],
            "site":
                slug("dcim.site", device.get("site")),
            "device":
                device.get("name") if kind != "ipam.vlan" else None,
            "port":
                data.get("name") if kind == "dcim.interface" else None,
            "vid":
                data.get("vid"),
            "commissioning_rx_dbm":
                fields.get("commissioning_rx_dbm"),
            "designation":
                data.get("description") if kind == "dcim.interface" else
                (data.get("name") if kind == "ipam.vlan" else None),
            "role":
                slug("dcim.devicerole", data.get("role")) if kind == "dcim.device" else None,
            "tenant":
                slug("tenancy.tenant", device.get("tenant") if kind == "dcim.interface" else data.get("tenant")),
            "owning_team":
                fields.get("owning_team"),
            "criticality":
                fields.get("criticality"),
            "recorded_at":
                change["time"],
            "retracted":
                deleted,
            "valid_from":
                None,
            "valid_to":
                None,
        })

    return pd.DataFrame(rows)


def netbox_network_schema(tenant_id: str) -> DataFrameInputSchema:
    """
    `tc0_network` from NetBox's changelog, folded by `fold_netbox_changes`.

    The changelog keeps when a change was made, not when it took effect, so `valid_from` is null from it. A
    port's designation is its interface's `description`, a VLAN's its `name`; the owner is the tenant's slug; the
    commissioning level, the owning team and the criticality are the estate's custom fields.

    Parameters
    ----------
    tenant_id : str
        The inventory's tenant.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """
    (head, tail) = _context_columns(tenant_id)

    return DataFrameInputSchema(column_info=head + [
        _text("object_kind", "object_kind"),
        _text("site_id", "site"),
        _text("device_id", "device"),
        _text("port_id", "port"),
        _count("vlan_id", "vid"),
        _number("commissioning_rx_dbm", "commissioning_rx_dbm"),
        _text("designation", "designation"),
        _text("device_role", "role"),
        _text("owner", "tenant"),
        _text("owning_team", "owning_team"),
        _text("criticality", "criticality"),
    ] + tail)


def read_workday(path: str) -> list[dict]:
    """The rows of a Workday report-as-a-service JSON response, `{"Report_Entry": [...]}`, as written."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["Report_Entry"]


def read_entra_audit(path: str) -> list[dict]:
    """Entra ID audit records as Azure Monitor exports them, one JSON object per line, each record's `properties`."""
    with open(path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    return [record.get("properties", record) for record in records]


def _quoted(value: typing.Any) -> typing.Optional[str]:
    # Entra ID writes a modified property's values as JSON text: a group's name arrives as "\"finance-users\"".
    return None if _missing(value) else json.loads(value)


def merge_identity(workers: list[dict], audit: list[dict]) -> pd.DataFrame:
    """
    The directory's changes to principals from two sources, as one stream ordered by when each was entered.

    A principal's profile is the HR system's: a Workday report of worker events, each with the moment it took
    effect and the moment it was entered, the worker's organization, manager, worker type and status, and the two
    fields the report calculates from the directory, the account's type and privilege. A group membership is the
    directory's: Entra ID's `Add member to group` and `Remove member from group` audit records, which say when they
    were made and not when they took effect. The stream is ordered by entry time, a change with none last, and at
    one moment the HR system's change before the directory's, as the directory provisions from it.

    Parameters
    ----------
    workers : list of dict
        Workday report rows, in entry order.
    audit : list of dict
        Entra ID audit records' `properties`, in the order logged.

    Returns
    -------
    `pandas.DataFrame`
        One row per change: the report's fields or the group's name, `valid_from`, `valid_to`, `recorded_at` and
        `retracted`.
    """
    rows = []

    for worker in workers:
        status = (worker.get("Worker_Status") or "").lower()
        rows.append({
            "user_principal":
                worker.get("Worker_Email"),
            "department":
                worker.get("Supervisory_Organization"),
            "manager":
                worker.get("Manager_Email"),
            "employment_status":
                "terminated" if status == "terminated" else
                ("contractor" if worker.get("Worker_Type") == "Contingent Worker" else status or None),
            "account_type": (worker.get("Account_Type") or "").lower() or None,
            "privilege_level": (worker.get("Privilege_Level") or "").lower() or None,
            "group_name":
                None,
            "valid_from":
                worker.get("Effective_Moment"),
            "valid_to":
                None,
            "recorded_at":
                worker.get("Entry_Moment"),
            "retracted":
                worker.get("Business_Process_Type") in WORKDAY_RETRACTIONS,
            "source":
                0,
        })

    for record in audit:
        if (record.get("activityDisplayName") not in ENTRA_GROUP_CHANGES or record.get("result") != "success"):
            continue

        user = next(target for target in record["targetResources"] if target.get("type") == "User")
        named = {prop["displayName"]: prop for prop in user.get("modifiedProperties", [])}["Group.DisplayName"]
        retract = ENTRA_GROUP_CHANGES[record["activityDisplayName"]] == "retract"
        rows.append({
            "user_principal": user.get("userPrincipalName"),
            "department": None,
            "manager": None,
            "employment_status": None,
            "account_type": None,
            "privilege_level": None,
            "group_name": _quoted(named["oldValue"] if retract else named["newValue"]),
            "valid_from": None,
            "valid_to": None,
            "recorded_at": record.get("activityDateTime"),
            "retracted": retract,
            "source": 1,
        })

    rows.sort(key=lambda row: (_utc(row["recorded_at"]) is None, _utc(row["recorded_at"]) or 0, row["source"]))

    return pd.DataFrame(rows).drop(columns=["source"])


def workday_entra_identity_schema(tenant_id: str) -> DataFrameInputSchema:
    """
    `tc0_identity` from Workday worker events and Entra ID group membership, merged by `merge_identity`.

    A profile carries both of the HR system's clocks; a group membership only the directory's entry time, so its
    `valid_from` and `valid_to` are null.

    Parameters
    ----------
    tenant_id : str
        The directory's tenant.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`
        The schema.
    """
    (head, tail) = _context_columns(tenant_id)

    return DataFrameInputSchema(column_info=head + [
        _through("user_principal", "user_principal", str.lower),
        _text("department", "department"),
        _through("manager", "manager", str.lower),
        _text("employment_status", "employment_status"),
        _text("account_type", "account_type"),
        _text("privilege_level", "privilege_level"),
        _text("group_name", "group_name"),
    ] + tail)


def read_zeek(path: str) -> pd.DataFrame:
    """A Zeek ASCII log, through `morpheus.parsers.zeek.parse`."""
    return zeek.parse(path, df_type="pandas")


def read_m365(path: str) -> pd.DataFrame:
    """Unified audit log records, one JSON object per line or one JSON array, values as written."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read().strip()

    records = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line]

    return pd.DataFrame(records)


def read_entra(path: str) -> pd.DataFrame:
    """
    Sign-in records as Azure Monitor exports them, one JSON object per line or one JSON array, each record's
    `properties` flattened to dotted columns as `AzureSourceStage` flattens them. A file of bare sign-in objects,
    as the Graph API's `signIns` returns them, is flattened the same way.
    """
    with open(path, encoding="utf-8") as handle:
        text = handle.read().strip()

    records = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line]

    if (len(records) > 0 and all("properties" in record for record in records)):
        records = [record["properties"] for record in records]

    return pd.json_normalize(records)


def read_sysmon(path: str) -> pd.DataFrame:
    """
    Sysmon events as `wevtutil qe Microsoft-Windows-Sysmon/Operational /f:xml` writes them: `<Event>` elements one
    after another, with no enclosing root.

    Returns
    -------
    `pandas.DataFrame`
        One row per event: `EventID`, `Computer`, and each `EventData/Data` by its `Name`.
    """
    with open(path, encoding="utf-8") as handle:
        root = ElementTree.fromstring(f"<Events>{handle.read()}</Events>")

    rows = []

    for event in root.iter(f"{SYSMON_NAMESPACE}Event"):
        system = event.find(f"{SYSMON_NAMESPACE}System")
        row = {
            "EventID": int(system.findtext(f"{SYSMON_NAMESPACE}EventID")),
            "Computer": system.findtext(f"{SYSMON_NAMESPACE}Computer"),
        }

        for data in event.iter(f"{SYSMON_NAMESPACE}Data"):
            row[data.get("Name")] = data.text

        rows.append(row)

    return pd.DataFrame(rows)


def read_telegraf(path: str) -> list[dict]:
    """
    Metrics as Telegraf's JSON serializer writes them, one object per line: `name`, `tags`, `fields` and
    `timestamp`, the last in seconds, Telegraf's default `json_timestamp_units`.
    """
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_arp(path: str) -> pd.DataFrame:
    """
    Every ARP packet over Ethernet and IPv4 in a capture, read with `morpheus.utils.ipfix.read_frames`.

    Returns
    -------
    `pandas.DataFrame`
        One row per packet in capture order: `received_ns`, `operation` (the opcode), `sha`, `spa`, `tha`, `tpa`.
    """
    rows = []

    for (received_ns, frame) in ipfix.read_frames(path):
        offset = 14
        ethertype = int.from_bytes(frame[12:14], "big")

        if (ethertype == ipfix.ETHERTYPE_VLAN):
            (ethertype, offset) = (int.from_bytes(frame[16:18], "big"), 18)

        if (ethertype != ETHERTYPE_ARP or len(frame) < offset + 28):
            continue

        (hardware, protocol, hardware_length, protocol_length, operation) = struct.unpack_from("!HHBBH", frame, offset)

        if ((hardware, protocol, hardware_length, protocol_length) != (1, ipfix.ETHERTYPE_IPV4, 6, 4)):
            continue

        body = frame[offset + 8:offset + 28]
        rows.append({
            "received_ns": received_ns,
            "operation": operation,
            "sha": _mac(body[0:6]),
            "spa": str(ipaddress.IPv4Address(body[6:10])),
            "tha": _mac(body[10:16]),
            "tpa": str(ipaddress.IPv4Address(body[16:20])),
        })

    return pd.DataFrame(rows, columns=["received_ns", "operation", "sha", "spa", "tha", "tpa"])


def _radius_value(text: str) -> str:
    text = text.strip()

    if (len(text) >= 2 and text[0] == text[-1] == '"'):
        return text[1:-1].replace('\\"', '"').replace("\\\\", "\\")

    return text


def read_radius_detail(path: str) -> list[dict]:
    """
    The entries of a FreeRADIUS detail file: each a date line, then one tab-indented `Attribute = value` per line,
    then a blank line.

    Returns
    -------
    list of dict
        Each entry's attributes by name, quotes removed, in file order; an attribute written twice keeps its last
        value.
    """
    entries = []
    entry: typing.Optional[dict] = None

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if (not line.strip()):
                if (entry is not None):
                    entries.append(entry)

                entry = None
            elif (line[0] not in " \t"):
                entry = {}
            elif (entry is not None and "=" in line):
                (name, value) = line.split("=", 1)
                # The branch's own `is not None` is the guard pylint does not follow.
                entry[name.strip()] = _radius_value(value)  # pylint: disable=unsupported-assignment-operation

    if (entry is not None):
        entries.append(entry)

    return entries


def merge_collectors(frames: typing.Sequence[pd.DataFrame]) -> pd.DataFrame:
    """
    One class's records from several collectors as one stream: by event time, then collector, then each
    collector's own sequence, the order the corpus builders write a class that two collectors feed.
    """
    merged = pd.concat(list(frames), ignore_index=True)

    return merged.sort_values(["event_time", "collector_id", "collector_seq"], kind="mergesort").reset_index(drop=True)


def normalize(frame: pd.DataFrame, schema: DataFrameInputSchema) -> pd.DataFrame:
    """Apply a schema with `process_dataframe`, which the per-layer pattern names for this step."""
    return process_dataframe(frame, schema)


def zeek_ssl(ssl_path: str, x509_path: str, profile: CollectorProfile) -> pd.DataFrame:
    """`tc6` records from a Zeek `ssl.log` and its `x509.log`."""
    return normalize(join_ssl_x509(read_zeek(ssl_path), read_zeek(x509_path)), zeek_ssl_schema(profile))


def zeek_http(path: str, profile: CollectorProfile) -> pd.DataFrame:
    """`tc7_http` records from a Zeek `http.log`."""
    return normalize(read_zeek(path), zeek_http_schema(profile))


def zeek_dns(path: str, profile: CollectorProfile) -> pd.DataFrame:
    """`tc7_dns` records from a Zeek `dns.log`."""
    return normalize(read_zeek(path), zeek_dns_schema(profile))


def m365_audit(path: str, profile: CollectorProfile, organizations: typing.Optional[dict] = None) -> pd.DataFrame:
    """`tc7_saas` records from unified audit log records."""
    return normalize(read_m365(path), m365_audit_schema(profile, organizations))


def entra_signin(path: str, profile: CollectorProfile) -> pd.DataFrame:
    """`tc5_auth` records from Entra ID sign-in records."""
    return normalize(read_entra(path), entra_signin_schema(profile))


def sysmon_process(path: str, profile: CollectorProfile) -> pd.DataFrame:
    """`tc7_endpoint` records from Sysmon events 1 and 7."""
    return normalize(join_image_signatures(read_sysmon(path)), sysmon_process_schema(profile))


def snmp_ports(path: str, profile: CollectorProfile) -> pd.DataFrame:
    """`tc1` records from a Telegraf SNMP feed."""
    return normalize(join_snmp_ports(read_telegraf(path)), snmp_port_schema(profile))


def snmp_devices(path: str, profile: CollectorProfile) -> pd.DataFrame:
    """`tc1_device` records from a Telegraf SNMP feed."""
    return normalize(join_snmp_devices(read_telegraf(path)), snmp_device_schema(profile))


def fdb_entries(path: str, profile: CollectorProfile) -> pd.DataFrame:
    """`tc2_mac` records from a Telegraf SNMP feed of forwarding tables and MAC notifications."""
    return normalize(join_fdb_ports(read_telegraf(path)), fdb_schema(profile))


def arp_packets(path: str, profile: CollectorProfile) -> pd.DataFrame:
    """`tc2_arp` records from a packet capture."""
    return normalize(read_arp(path), arp_schema(profile))


def radius_auth(path: str, profile: CollectorProfile, sites: dict) -> pd.DataFrame:
    """`tc2_auth` records from a FreeRADIUS detail file of 802.1X requests and replies."""
    return normalize(join_radius_replies(read_radius_detail(path)), radius_auth_schema(profile, sites))


def radius_accounting(path: str, profile: CollectorProfile) -> pd.DataFrame:
    """`tc5_session` records from a FreeRADIUS detail file of accounting requests."""
    entries = [entry for entry in read_radius_detail(path) if "Acct-Status-Type" in entry]

    return normalize(pd.DataFrame(entries), radius_accounting_schema(profile))


def packets(path: str, profile: CollectorProfile) -> pd.DataFrame:
    """`tc4` records from a packet capture."""
    return normalize(read_packets(path), packet_schema(profile))


def kea_leases(path: str, profile: CollectorProfile) -> pd.DataFrame:
    """`tc3_dhcp` records from a Kea DHCPv4 server's log."""
    return normalize(read_kea(path), kea_schema(profile))


def osquery_inventory(path: str, profile: CollectorProfile, report_valid_seconds: int) -> pd.DataFrame:
    """`tc7_host_inventory` records from osquery's results log."""
    return normalize(read_osquery(path), osquery_inventory_schema(profile, report_valid_seconds))


def zeek_service(conn_path: str, ssl_path: str, profile: CollectorProfile) -> pd.DataFrame:
    """`tc7_service` records from a Zeek `conn.log` and its `ssl.log`."""
    return normalize(join_conn_ssl(read_zeek(conn_path), read_zeek(ssl_path)), zeek_service_schema(profile))


def servicenow_assets(audit_path: str, users_path: str, groups_path: str, tenant_id: str) -> pd.DataFrame:
    """`tc0_asset` records from ServiceNow's `sys_audit`, `sys_user` and `sys_user_group`."""
    versions = fold_cmdb_audit(read_servicenow(audit_path),
                               read_servicenow(users_path),
                               read_servicenow(groups_path),
                               SERVICENOW_ASSET_ROLES)

    return normalize(versions, servicenow_asset_schema(tenant_id))


def servicenow_applications(audit_path: str, users_path: str, groups_path: str, tenant_id: str) -> pd.DataFrame:
    """`tc0_application` records from ServiceNow's `sys_audit`, `sys_user` and `sys_user_group`."""
    versions = fold_cmdb_audit(read_servicenow(audit_path),
                               read_servicenow(users_path),
                               read_servicenow(groups_path),
                               SERVICENOW_APPLICATIONS)

    return normalize(versions, servicenow_application_schema(tenant_id))


def netbox_network(path: str, tenant_id: str) -> pd.DataFrame:
    """`tc0_network` records from NetBox's changelog."""
    return normalize(fold_netbox_changes(read_netbox_changes(path)), netbox_network_schema(tenant_id))


def workday_entra_identity(workday_path: str, audit_path: str, tenant_id: str) -> pd.DataFrame:
    """`tc0_identity` records from a Workday worker-event report and Entra ID's audit log."""
    return normalize(merge_identity(read_workday(workday_path), read_entra_audit(audit_path)),
                     workday_entra_identity_schema(tenant_id))
