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
One campaign told at three layers, and the composed pipelines that measure it, for the chained rules.

Every other corpus in this fork is one layer's, and they share no entity: an address in the layer 3 corpus is not a
principal in the layer 5 one or a host in the layer 7 one. A chained rule needs the opposite -- the same source, the
same person and the same host seen by three collectors -- so this corpus is written as one estate over nine days and
split by collector. Each layer's records then go through that layer's own stages, exactly as a deployment would run
them, and the chained rule is evaluated over what comes out.

**R-C-001 - Lateral movement chain.** Within 30 minutes, in order: a source's fan-out rises above its own previous
hour; a principal authenticates *from that source* to a host they have never logged into; and that host starts a
process whose ancestry neither it nor its peer group has run in thirty days. No step has to breach a threshold of its
own. The attacker here reaches twenty-five new addresses, half what R-B-L3-001 fires on, and neither a first login to
a server nor one new process is an alert by itself.

The attacker does all three, and six others each do all but one:

- **wrong order**: the new process runs before the login that would have to have started it;
- **too slow**: every step happens, the last one thirty-five minutes after the first;
- **known host**: the login is to a server the principal logs into every day;
- **other source**: the login comes from an address other than the one whose fan-out rose;
- **flat fan-out**: a busy host reaching the same twenty servers every hour, so nothing rose;
- **no new process**: the host only runs what it always runs.

**The steps are joined on values, not on `lineage_id`.** The three layers are sealed on three different entities --
a source address, a principal, a host -- so no lineage chain holds all three, and the rule joins the source address
to the login's source and the login's target to the process's host, carrying each step's lineage identifiers as
evidence. Hosts are compared case-folded, because a login log and an EDR name the same machine differently.

**Each step may precede the one it follows by up to two minutes.** That is the rule's declared join tolerance, twice
a worst-case clock offset of sixty seconds between collectors, as Part 3's governance asks.

**R-C-004 - Staged exfiltration.** Within two hours, in order: a principal's SaaS export more than five times their
own baseline; a transfer envelope breach from an address the principal holds a session on at that moment; and a
connection from that address to a destination whose certificate issuer nobody in the estate has seen in thirty days.
The same collectors' shapes carry it -- SaaS audit, session start and stop, packets, TLS handshakes -- each through
its own layer's composed pipeline, and the attacker does all three while four others each do all but one:

- **someone else's host**: the breach and the handshake come from an address another principal's session holds;
- **outside the session**: the breach comes from the principal's own address twenty minutes after they logged off;
- **known issuer**: the last connection's certificate is the corporate one every host in the estate sees daily;
- **wrong order**: the breach and the handshake both happen before the export.

**R-C-002 - Command and control establishment.** A settled host presents a TLS client fingerprint it never has, to a
destination, and within the hour starts beaconing to that same destination. The attacker does both; four others do
both with one condition broken:

- **beacon first**: the host was already beaconing to the destination an hour before the new fingerprint;
- **other destination**: the beacon goes somewhere other than where the new fingerprint went;
- **too slow**: the beacon matures sixty-seven minutes after the fingerprint;
- **unsettled host**: five handshakes of history, so the new fingerprint is only new because the host is.
"""

import typing

import pandas as pd
import presentation_pipeline
import saas_pipeline
import transport_pipeline

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.stages.telemetry.tc3_beacon_stage import TC3BeaconStage
from morpheus.stages.telemetry.tc3_cardinality_stage import TC3CardinalityStage
from morpheus.stages.telemetry.tc3_reach_stage import TC3ReachStage
from morpheus.stages.telemetry.tc3_ttl_stage import TC3TtlStage
from morpheus.stages.telemetry.tc5_novelty_stage import TC5NoveltyStage
from morpheus.stages.telemetry.tc5_session_stage import TC5SessionStage
from morpheus.stages.telemetry.tc7_endpoint_stage import TC7EndpointStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900
DAY_SECONDS = 86400

ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []

FLOW_CLASS = "tc3"
AUTH_CLASS = "tc5_auth"
PROCESS_CLASS = "tc7_endpoint"
SESSION_CLASS = "tc5_session"
TRANSFER_CLASS = transport_pipeline.TELEMETRY_CLASS
HANDSHAKE_CLASS = presentation_pipeline.TELEMETRY_CLASS
SAAS_CLASS = saas_pipeline.SAAS_CLASS
CLASSES = (FLOW_CLASS, AUTH_CLASS, PROCESS_CLASS, SESSION_CLASS, TRANSFER_CLASS, HANDSHAKE_CLASS, SAAS_CLASS)
"""The collectors, in the order they are run. The last four are R-C-004's; each of those runs through its layer's own
composed pipeline, which is the one its single-layer rules are asserted over."""

# The rule's own figures, stated once so the corpus is built around them deliberately.
CHAIN_WINDOW_SECONDS = 30 * 60
JOIN_TOLERANCE_SECONDS = 120
SEVERITY = 80
FANOUT_THRESHOLD = 50
"""R-B-L3-001's fan-out threshold, which the attacker stays well under."""

LAST_DAY = 8
"""The day the campaign happens. The eight before it are the history every step is novel against."""
CAMPAIGN_HOUR = 10
SERVER_GROUP = "servers"


class Actor(typing.NamedTuple):
    """One source, the principal who logs in from it, and the server they reach, with what each step does."""

    source: str
    principal: str
    home: str
    """The server the principal logs into every day."""
    target: str
    """The server the campaign's login reaches. The home server for the known-host control."""
    login_source: str
    """Where the login comes from. The fan-out source except in the other-source control."""
    rising: bool
    fanout_minute: float
    login_minute: float
    process_minute: typing.Optional[float]
    """When the novel process runs on the target, or None when it only runs its routine."""


ATTACKER = "dana@example.com"
WRONG_ORDER = "eli@example.com"
TOO_SLOW = "fay@example.com"
KNOWN_HOST = "gus@example.com"
OTHER_SOURCE = "hal@example.com"
FLAT_FANOUT = "ivan@example.com"
NO_NEW_PROCESS = "judy@example.com"

ACTORS = {
    ATTACKER: Actor("10.20.0.5", ATTACKER, "srv-dana", "SRV-DB-02", "10.20.0.5", True, 5, 12, 20),
    WRONG_ORDER: Actor("10.20.0.6", WRONG_ORDER, "srv-eli", "srv-app-03", "10.20.0.6", True, 5, 14, 8),
    TOO_SLOW: Actor("10.20.0.7", TOO_SLOW, "srv-fay", "srv-fs-04", "10.20.0.7", True, 5, 20, 40),
    KNOWN_HOST: Actor("10.20.0.8", KNOWN_HOST, "srv-gus", "srv-gus", "10.20.0.8", True, 5, 12, 20),
    OTHER_SOURCE: Actor("10.20.0.9", OTHER_SOURCE, "srv-hal", "srv-web-05", "10.20.0.99", True, 5, 12, 20),
    FLAT_FANOUT: Actor("10.20.0.10", FLAT_FANOUT, "srv-ivan", "srv-mq-06", "10.20.0.10", False, 5, 12, 20),
    NO_NEW_PROCESS: Actor("10.20.0.11", NO_NEW_PROCESS, "srv-judy", "srv-ci-07", "10.20.0.11", True, 5, 12, None),
}

SERVICES = r"C:\Windows\System32\services.exe"
SVCHOST = r"C:\Windows\System32\svchost.exe"
WMI = r"C:\Windows\System32\wbem\WmiPrvSE.exe"
CMD = r"C:\Windows\System32\cmd.exe"

PREVIOUS_HOUR_DESTINATIONS = 3
RISING_DESTINATIONS = 25
FLAT_DESTINATIONS = 20


def at(day: float, hour: float = 0, minute: float = 0, second: float = 0) -> int:
    """An instant in the corpus, in nanoseconds since the epoch."""
    return int((day * DAY_SECONDS + hour * 3600 + minute * 60 + second) * NS_PER_SECOND)


def novel_image(principal: str) -> str:
    """The process each campaign starts, distinct per actor so one never excuses another in the peer group."""
    return rf"C:\Windows\Temp\{principal.split('@')[0]}-svc.exe"


# --- R-C-004's actors ---------------------------------------------------------------------------------------------

EXFIL_WINDOW_SECONDS = 2 * 3600
RECORD_MULTIPLE = 5.0
EXFIL_SEVERITY = 80
PRIOR_EXPORTS = 110
PRIOR_TRANSFERS = 110
SYNC_SERVER = "198.51.100.10"
PORTAL = "198.51.100.20"
PORTAL_ISSUER = presentation_pipeline.CORP_CA


class Exfiltrator(typing.NamedTuple):
    """A principal, the address their session holds, and when each of R-C-004's steps happens on the last day."""

    principal: str
    address: str
    """The address the principal's sessions come from."""
    breach_address: str
    """Where the breach and the handshake come from. Another principal's address for the someone-else control."""
    session_end_minute: typing.Optional[float]
    """When the last day's session ends, minutes after the campaign hour starts; None for the evening, as usual."""
    export_minute: float
    breach_minute: float
    handshake_minute: float
    issuer: str


KIM = "kim@example.com"
LEE = "lee@example.com"
MAX = "max@example.com"
NIA = "nia@example.com"
OTO = "oto@example.com"
PIA = "pia@example.com"


def relay_issuer(principal: str) -> str:
    """An authority nobody in the estate has seen, distinct per actor so one's first sighting never covers another's."""
    return f"CN={principal.split('@')[0].title()} Relay CA, O=Unknown"


EXFILTRATORS = {
    KIM: Exfiltrator(KIM, "10.20.1.5", "10.20.1.5", None, 5, 30, 50, relay_issuer(KIM)),
    LEE: Exfiltrator(LEE, "10.20.1.6", "10.20.1.7", None, 5, 30, 50, relay_issuer(LEE)),
    NIA: Exfiltrator(NIA, "10.20.1.8", "10.20.1.8", 10, 5, 30, 50, relay_issuer(NIA)),
    OTO: Exfiltrator(OTO, "10.20.1.10", "10.20.1.10", None, 5, 30, 50, PORTAL_ISSUER),
    PIA: Exfiltrator(PIA, "10.20.1.9", "10.20.1.9", None, 30, 5, 15, relay_issuer(PIA)),
}
SESSION_HOLDERS = {**{actor.principal: actor.address for actor in EXFILTRATORS.values()}, MAX: "10.20.1.7"}
"""Every principal with a daily session and the address it comes from. Max exports nothing; Lee's breach comes from
Max's address."""

# --- R-C-002's actors ---------------------------------------------------------------------------------------------

C2_WINDOW_SECONDS = 3600
C2_SEVERITY = 70
SETTLED_HANDSHAKES = 24
"""Past the twenty R-B-L6-001's floor asks for."""
UNSETTLED_HANDSHAKES = 5
BEACON_FLOWS = 20
BEACON_PERIOD_SECONDS = 60
BEACON_BYTES = 512
PUBLIC_ISSUER = "CN=Public Web CA, O=Example Trust"


class Beaconer(typing.NamedTuple):
    """A host, how settled its TLS history is, and when and where its new fingerprint and its beacon happen."""

    host: str
    history: int
    fingerprint_destination: str
    fingerprint_minute: float
    beacon_destination: str
    beacon_start_minute: float
    """Minutes after the campaign hour starts; negative for the hour before."""


C2_ATTACKER = "10.20.2.5"
BEACON_FIRST = "10.20.2.6"
OTHER_DESTINATION = "10.20.2.7"
C2_TOO_SLOW = "10.20.2.8"
UNSETTLED = "10.20.2.9"

BEACONERS = {
    C2_ATTACKER: Beaconer(C2_ATTACKER, SETTLED_HANDSHAKES, "203.0.113.100", 0, "203.0.113.100", 2),
    BEACON_FIRST: Beaconer(BEACON_FIRST, SETTLED_HANDSHAKES, "203.0.113.101", 0, "203.0.113.101", -60),
    OTHER_DESTINATION: Beaconer(OTHER_DESTINATION, SETTLED_HANDSHAKES, "203.0.113.102", 0, "203.0.113.112", 2),
    C2_TOO_SLOW: Beaconer(C2_TOO_SLOW, SETTLED_HANDSHAKES, "203.0.113.103", 0, "203.0.113.103", 55),
    UNSETTLED: Beaconer(UNSETTLED, UNSETTLED_HANDSHAKES, "203.0.113.104", 0, "203.0.113.104", 2),
}


def servers() -> list:
    """Every server a principal logs into or a campaign reaches, in one peer group."""
    names = set()

    for actor in ACTORS.values():
        names.update((actor.home, actor.target.lower()))

    return sorted(names)


def asset_versions() -> list:
    return [
        make_version("asset",
                     host,
                     0,
                     None,
                     0,
                     values={
                         "owner": "it-ops@example.com",
                         "owning_team": "platform",
                         "criticality": "high",
                         "data_classification": "internal",
                         "peer_group": SERVER_GROUP,
                     }) for host in servers()
    ]


def build_store() -> BitemporalStore:
    """The inventory the process enrichment reads: every server in one peer group."""
    return BitemporalStore("asset", asset_versions())


def _envelope(rows: list, collector: str, schema: str, origin: str, when: int) -> dict:
    return {
        "collector_id": collector,
        "schema_version": schema,
        "origin_hash": f"{origin}@{when}",
        "collector_seq": len(rows),
        "event_time": when,
    }


def _flow(rows: list, source: str, destination: str, when: int):
    record = _envelope(rows, "netflow-01", "tc3.v1", f"{source}->{destination}", when)
    record.update({
        "src_ip": source,
        "dst_ip": destination,
        "dst_port": 445,
        "protocol": "tcp",
        "ip_ttl": 128,
        "bytes_out": 900,
        "bytes_in": 1400,
        "bgp_as_dst": "64512",
    })
    rows.append(record)


def _login(rows: list, principal: str, source: str, target: str, when: int):
    record = _envelope(rows, "dc-01", "tc5_auth.v1", f"{principal}>{target}", when)
    record.update({
        "user_principal": principal,
        "source_ip": source,
        "target_host": target,
        "app": "windows-logon",
        "auth_result": "success",
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
    record = _envelope(rows, "vpn-01", "tc5_session.v1", f"{session_id}:{action}", when)
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
               fingerprint: str = presentation_pipeline.CHROME):
    record = _envelope(rows, "tls-inspect-01", "tc6.v1", f"{source}->{destination}", when)
    record.update({
        "src_ip": source,
        "dst_ip": destination,
        "dst_port": 443,
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


def _export(rows: list, principal: str, records: int, when: int):
    record = _envelope(rows, "saas-audit-01", "tc7_saas.v1", f"{principal}>export@{records}", when)
    record.update({
        "user_principal": principal,
        "operation": saas_pipeline.EXPORT,
        "target_object": "customer-list",
        "target_object_type": "Account",
        "record_count": records,
        "result": "success",
        "client_app": "web",
    })
    rows.append(record)


def _build_exfiltration(sessions: list, packets: list, handshakes: list, exports: list):
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
            _export(exports, actor.principal, 40 + (count * 4) % 21, at(0, 12, index) + count * 90 * 60 * NS_PER_SECOND)

        # The breach address's regular uploads to the sync server, inside the envelope's eight-hour window.
        for count in range(PRIOR_TRANSFERS):
            _packet(packets,
                    actor.breach_address,
                    44000 + count,
                    SYNC_SERVER,
                    1200,
                    at(LAST_DAY, 3, index) + count * 3 * 60 * NS_PER_SECOND)

        _export(exports, actor.principal, 5000, at(LAST_DAY, CAMPAIGN_HOUR, actor.export_minute))
        _packet(packets,
                actor.breach_address,
                45000,
                SYNC_SERVER,
                60000,
                at(LAST_DAY, CAMPAIGN_HOUR, actor.breach_minute))
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

        _handshake(handshakes,
                   actor.host,
                   actor.fingerprint_destination,
                   PUBLIC_ISSUER,
                   at(LAST_DAY, CAMPAIGN_HOUR, actor.fingerprint_minute, index),
                   fingerprint=presentation_pipeline.NEW_STACK)

        for count in range(BEACON_FLOWS):
            when = at(LAST_DAY, CAMPAIGN_HOUR, actor.beacon_start_minute,
                      index) + count * BEACON_PERIOD_SECONDS * NS_PER_SECOND
            record = _envelope(flows, "netflow-01", "tc3.v1", f"{actor.host}->{actor.beacon_destination}", when)
            record.update({
                "src_ip": actor.host,
                "dst_ip": actor.beacon_destination,
                "dst_port": 443,
                "protocol": "tcp",
                "ip_ttl": 128,
                "bytes_out": BEACON_BYTES,
                "bytes_in": 128,
                "bgp_as_dst": "64500",
            })
            flows.append(record)


def build_corpus() -> dict[str, pd.DataFrame]:
    """The campaign, split by collector."""
    flows: list = []
    logins: list = []
    processes: list = []

    for day in range(LAST_DAY + 1):
        for (index, actor) in enumerate(ACTORS.values()):
            # Every principal's working day starts with a login to their own server from their own address.
            _login(logins, actor.principal, actor.source, actor.home, at(day, 9, index))

        for host in servers():
            _process(processes, host, SERVICES, SVCHOST, at(day, 9, 30))
            _process(processes, host, SVCHOST, WMI, at(day, 9, 30))

    for (index, actor) in enumerate(ACTORS.values()):
        _flows(flows, actor, index)
        _login(logins,
               actor.principal,
               actor.login_source,
               actor.target,
               at(LAST_DAY, CAMPAIGN_HOUR, actor.login_minute))
        host = actor.target.lower()

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
        result["collector_seq"] = range(len(result))

        return result

    sessions: list = []
    packets: list = []
    handshakes: list = []
    exports: list = []
    _build_exfiltration(sessions, packets, handshakes, exports)
    _build_command_and_control(flows, handshakes)

    return {
        FLOW_CLASS: frame(flows),
        AUTH_CLASS: frame(logins),
        PROCESS_CLASS: frame(processes),
        SESSION_CLASS: frame(sessions),
        TRANSFER_CLASS: frame(packets),
        HANDSHAKE_CLASS: frame(handshakes),
        SAAS_CLASS: frame(exports),
    }


def build_pipeline_config(execution_mode=None) -> Config:
    """
    A pipeline configuration, defaulting to CPU mode and importable without a GPU.

    Parameters
    ----------
    execution_mode : `morpheus.config.ExecutionMode`, optional
        Mode to build for. Defaults to CPU, which is what the golden file is an artifact of.
    """
    from morpheus.config import CppConfig  # pylint: disable=import-outside-toplevel
    from morpheus.config import ExecutionMode  # pylint: disable=import-outside-toplevel

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.execution_mode = ExecutionMode.CPU if execution_mode is None else execution_mode

    return config


def _stages(config: Config, telemetry_class: str) -> tuple:
    """Each layer's own stages, its OSI layer and the entity it is sealed on."""
    if (telemetry_class == FLOW_CLASS):
        return ([
            TC3CardinalityStage(config, window_seconds=PERIOD_SECONDS),
            TC3ReachStage(config, window_seconds=PERIOD_SECONDS),
            TC3BeaconStage(config, window_seconds=2 * PERIOD_SECONDS),
            TC3TtlStage(config, window_seconds=2 * PERIOD_SECONDS),
        ],
                3, ["src_ip"])

    if (telemetry_class == SESSION_CLASS):
        return ([TC5SessionStage(config)], 5, ["user_principal"])

    if (telemetry_class == AUTH_CLASS):
        return ([TC5NoveltyStage(config, target_host_column="target_host")], 5, ["user_principal"])

    return ([TC0EnrichStage(config, store=build_store(), entity_column="hostname"), TC7EndpointStage(config)],
            7, ["hostname"])


def _collect(sink: InMemorySinkStage) -> pd.DataFrame:
    frames = []

    for message in sink.get_messages():
        meta = message.payload() if hasattr(message, "payload") else message
        frame = meta.copy_dataframe()
        frames.append(frame.to_pandas() if hasattr(frame, "to_pandas") else frame)

    if (len(frames) == 0):
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def run_class(config: Config,
              telemetry_class: str,
              dataframes: list[pd.DataFrame],
              impose_order: bool = True) -> pd.DataFrame:
    """Source, stamp, total order, the layer's stages, envelope, seal hourly, sink, for one collector."""
    (stages, osi_layer, entity_columns) = _stages(config, telemetry_class)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    for stage in stages:
        pipe.add_stage(stage)

    pipe.add_stage(EnvelopeStampStage(config, osi_layer=osi_layer, entity_columns=entity_columns))
    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=PERIOD_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        entity_key_column="entity_key"))

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    frame = _collect(sink)
    frame["telemetry_class"] = telemetry_class
    frame["row_key"] = frame["event_uid"]

    return frame


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True) -> pd.DataFrame:
    """
    Each collector through its own layer's pipeline, and the three outputs as one canonical frame.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    corpus : dict
        The frames from `build_corpus`, possibly permuted.
    batches : dict, optional
        How each collector's frame is split across source frames. Defaults to one frame each.
    impose_order : bool, default = True
        Place `TotalOrderStage` ahead of the stateful stages.

    Returns
    -------
    `pandas.DataFrame`
        The three classes' output, tagged with `telemetry_class`, keyed by `row_key`, canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    # R-C-004's last three collectors go through the composed pipelines their own rules are asserted over, so the
    # chain is measured on exactly what R-B-L4-005, the layer 6 stages and R-B-L7-002 see.
    delegated = {TRANSFER_CLASS: transport_pipeline, HANDSHAKE_CLASS: presentation_pipeline, SAAS_CLASS: saas_pipeline}
    frames = []

    for name in CLASSES:
        if (name in delegated):
            frames.append(delegated[name].run_pipeline(config, {name: batches[name][0]},
                                                       batches={name: batches[name]},
                                                       impose_order=impose_order))
        else:
            frames.append(run_class(config, name, batches[name], impose_order=impose_order))

    return canonicalize(pd.concat(frames, ignore_index=True), key_columns=KEY_COLUMNS, ignore_columns=IGNORE_COLUMNS)


def render(result: pd.DataFrame) -> str:
    """The canonical CSV rendering the golden file holds."""
    return result.to_csv(index=False, lineterminator="\n")


# --- R-C-001, evaluated the way its search evaluates it ------------------------------------------------------------


def _truthy(value) -> bool:
    return False if pd.isna(value) else bool(value)


def fanout_rises(result: pd.DataFrame) -> pd.DataFrame:
    """Step 1: the first flow per source and hour whose fan-out exceeds the source's peak in the hour before.

    A source with no flows in the hour before has nothing to rise above and is not counted.
    """
    flows = result.loc[(result["telemetry_class"] == FLOW_CLASS) & result["dsts_per_src"].notna(),
                       ["src_ip", "window_id", "dsts_per_src", "event_time", "lineage_id"]].copy()
    flows["window_id"] = flows["window_id"].astype("int64")
    flows["dsts_per_src"] = flows["dsts_per_src"].astype("int64")
    peaks = flows.groupby(["src_ip", "window_id"])["dsts_per_src"].max().rename("previous_peak").reset_index()
    peaks["window_id"] = peaks["window_id"] + 1
    flows = flows.merge(peaks, on=["src_ip", "window_id"], how="inner")
    flows = flows[flows["dsts_per_src"] > flows["previous_peak"]]

    return (flows.sort_values("event_time").groupby(["src_ip", "window_id"], as_index=False).first()[[
        "src_ip", "event_time", "dsts_per_src", "previous_peak", "lineage_id"
    ]])


def lateral_movement(result: pd.DataFrame,
                     rising: bool = True,
                     new_host: bool = True,
                     novel_process: bool = True,
                     same_source: bool = True,
                     ordered: bool = True,
                     window_seconds: typing.Optional[int] = CHAIN_WINDOW_SECONDS,
                     tolerance_seconds: int = JOIN_TOLERANCE_SECONDS) -> dict:
    """(source, principal, host) chains R-C-001 fires on, each with its three step times.

    Every condition can be switched off, which is how the counterfactual checks ask what a control would have done
    without the one condition that stops it.
    """
    tolerance_ns = tolerance_seconds * NS_PER_SECOND

    if (rising):
        starts = fanout_rises(result)
    else:
        # Without the rise, a source's first flow in every hour starts a chain.
        flows = result[result["telemetry_class"] == FLOW_CLASS]
        starts = (flows.sort_values("event_time").groupby(["src_ip", "window_id"],
                                                          as_index=False).first()[["src_ip", "event_time"]])

    logins = result[(result["telemetry_class"] == AUTH_CLASS) & (result["auth_result"] == "success")]

    if (new_host):
        logins = logins[logins["target_host_first_seen"].map(_truthy)]

    processes = result[result["telemetry_class"] == PROCESS_CLASS]

    if (novel_process):
        processes = processes[processes["endpoint_pair_novel"].map(_truthy)]

    fired: dict = {}

    for (_, start) in starts.iterrows():
        t1 = int(start["event_time"])
        candidates = logins if not same_source else logins[logins["source_ip"] == start["src_ip"]]

        for (_, login) in candidates.iterrows():
            t2 = int(login["event_time"])

            if (ordered and t2 < t1 - tolerance_ns):
                continue

            host = str(login["target_host"]).lower()

            for (_, process) in processes[processes["hostname"].str.lower() == host].iterrows():
                t3 = int(process["event_time"])

                if (ordered and t3 < t2 - tolerance_ns):
                    continue

                if (window_seconds is not None and max(t2, t3) - t1 > window_seconds * NS_PER_SECOND):
                    continue

                key = (start["src_ip"], login["user_principal"], host)
                fired.setdefault(key, (t1, t2, t3))

    return fired


# --- R-C-004, evaluated the way its search evaluates it ------------------------------------------------------------


def session_intervals(result: pd.DataFrame) -> pd.DataFrame:
    """Each session's principal, address and the interval it was open over. An unended session is open at the end."""
    rows = result[result["telemetry_class"] == SESSION_CLASS].copy()
    rows["event_time"] = rows["event_time"].astype("int64")
    starts = rows[rows["session_action"] == "start"].groupby(["session_key", "user_principal",
                                                              "source_ip"])["event_time"].min().rename("opened")
    ends = rows[rows["session_action"] == "end"].groupby("session_key")["event_time"].max().rename("closed")

    return starts.reset_index().merge(ends.reset_index(), on="session_key", how="left")


def staged_exfiltration(result: pd.DataFrame,
                        in_session: bool = True,
                        session_bound: bool = True,
                        new_issuer: bool = True,
                        ordered: bool = True,
                        window_seconds: typing.Optional[int] = EXFIL_WINDOW_SECONDS,
                        tolerance_seconds: int = JOIN_TOLERANCE_SECONDS) -> dict:
    """(principal, address) chains R-C-004 fires on, each with its three step times.

    `session_bound` off lets a breach from any address count; `in_session` off keeps the address a session of the
    principal's but drops the requirement that it was open at the breach.
    """
    tolerance_ns = tolerance_seconds * NS_PER_SECOND
    window_ns = None if window_seconds is None else window_seconds * NS_PER_SECOND

    exports = result[(result["telemetry_class"] == SAAS_CLASS)
                     & result["saas_baseline_mature"].map(_truthy)
                     & (result["saas_record_ratio"].astype("Float64") > RECORD_MULTIPLE).fillna(False)]
    breaches = result[(result["telemetry_class"] == TRANSFER_CLASS)
                      & (result["flow_data_len_envelope_breached"].map(_truthy)
                         | result["flow_bpp_envelope_breached"].map(_truthy))]
    handshakes = result[result["telemetry_class"] == HANDSHAKE_CLASS]

    if (new_issuer):
        handshakes = handshakes[handshakes["cert_issuer_new_to_estate"].map(_truthy)]

    sessions = session_intervals(result)
    fired: dict = {}

    for (_, export) in exports.iterrows():
        principal = export["user_principal"]
        t1 = int(export["event_time"])
        own = sessions[sessions["user_principal"] == principal]

        for (_, breach) in breaches.iterrows():
            t2 = int(breach["event_time"])
            address = breach["src_ip"]

            if (ordered and t2 < t1 - tolerance_ns) or (window_ns is not None and t2 - t1 > window_ns):
                continue

            if (session_bound):
                held = own[own["source_ip"] == address]

                if (in_session):
                    held = held[(held["opened"] <= t2) & (held["closed"].isna() | (held["closed"] >= t2))]

                if (len(held) == 0):
                    continue

            for (_, handshake) in handshakes[handshakes["src_ip"] == address].iterrows():
                t3 = int(handshake["event_time"])

                if (ordered and t3 < t2 - tolerance_ns) or (window_ns is not None and t3 - t1 > window_ns):
                    continue

                fired.setdefault((principal, address), (t1, t2, t3))

    return fired


# --- R-C-002, evaluated the way its search evaluates it ------------------------------------------------------------


def tls_before_beaconing(result: pd.DataFrame,
                         settled: bool = True,
                         same_destination: bool = True,
                         ordered: bool = True,
                         window_seconds: typing.Optional[int] = C2_WINDOW_SECONDS,
                         tolerance_seconds: int = JOIN_TOLERANCE_SECONDS) -> dict:
    """(host, destination) pairs R-C-002 fires on, with the fingerprint's time and the beacon's.

    The beacon is dated when it first matures, which is when R-B-L3-002 would first report it.
    """
    tolerance_ns = tolerance_seconds * NS_PER_SECOND
    fingerprints = result[(result["telemetry_class"] == HANDSHAKE_CLASS) & result["ja4_client_first_seen"].map(_truthy)]

    if (settled):
        fingerprints = fingerprints[fingerprints["ja4_client_observations"].astype("Int64").fillna(0) >= 20]

    flows = result[(result["telemetry_class"] == FLOW_CLASS) & result["flow_regularity_mature"].map(_truthy)]
    flows = flows[(flows["flow_interval_cv"].astype(float) < 0.15) & (flows["flow_size_cv"].astype(float) < 0.15)]
    beacons = flows.groupby(["src_ip", "dst_ip"])["event_time"].min().astype("int64")
    fired: dict = {}

    for (_, fingerprint) in fingerprints.iterrows():
        t1 = int(fingerprint["event_time"])

        for ((source, destination), t2) in beacons.items():
            if (source != fingerprint["src_ip"]) or (same_destination and destination != fingerprint["dst_ip"]):
                continue

            if (ordered and t2 < t1 - tolerance_ns):
                continue

            if (window_seconds is not None and t2 - t1 > window_seconds * NS_PER_SECOND):
                continue

            fired.setdefault((source, destination), (t1, int(t2)))

    return fired
