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
The normalize step of the per-layer pattern for six real feeds: one `DataFrameInputSchema` per layer 5, 6 and 7
sub-class, from the format a deployment already writes to the columns the stages read.

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

Zeek logs are read with {py:func}`morpheus.parsers.zeek.parse`.

**What a format cannot say, the schema does not invent.** Zeek sees a TLS handshake, not what an inspecting proxy
saw inside it, so `content_type_declared` and `content_type_detected` are null from the Zeek schema and the layer 6
content rule cannot fire on a Zeek-only feed; a feed with an inspecting proxy supplies them. A cloud identity
provider has no on-premises target computer, so `target_host` is null from the Entra ID schema. No format here
records when the collector read a line, so `observed_time` is the record's own time and the collection lag reads
zero; a collector that keeps its read time passes it as `observed_column`. The collector's own facts --
identity, contract version, tenant, clock discipline, sampling -- are a `CollectorProfile`, because no record
carries them, and `collector_seq` numbers the records in file order from the profile's `first_seq`.

**Vocabulary is mapped where the format's differs and passed through where it does not.** Zeek writes `TLSv13`
and OpenSSL's `self signed certificate`; the stages read `TLSv1.3` and `self-signed`. Entra ID writes an error code,
`multiFactorAuthentication` and `primaryRefreshToken`; the stages read `success` or `failure`, a boolean and
`refresh`. A value the map does not know is passed through as it is, so a new one reaches the SIEM visibly rather
than as a null.

`origin_hash` is built the way the corpus builders build it for each class, so a normalized record is, to the
lineage stamp, the same record the corpus has; where the record carries its own identifier, as a sign-in's `id`,
that is the origin.
"""

import dataclasses
import hashlib
import json
import math
import typing
from xml.etree import ElementTree

import pandas as pd

from morpheus.parsers import zeek
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
    `tc7_http` from Zeek `http.log`. `url_path` is the `uri` without its query string.

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
    """`approved` or `denied` from a sign-in's second-factor steps; null where none was asked for."""
    if (_missing(steps) or isinstance(steps, str)):
        return None

    second = [step for step in steps if step.get("authenticationMethod") != ENTRA_PASSWORD_METHOD]

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

    def coordinate(name: str) -> CustomColumn:
        source = fields[name]

        def column(df: pd.DataFrame) -> pd.Series:
            if (source not in df.columns):
                return pd.Series([None] * len(df), index=df.index, dtype="Float64")

            return pd.to_numeric(df[source]).astype("Float64")

        return CustomColumn(name=name,
                            dtype="Float64",
                            process_column_fn=column,
                            input_column_types={source: "Float64"})

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
        coordinate("source_latitude"),
        coordinate("source_longitude"),
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
