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
A flow-shaped layer 3 corpus, and the composed TC-3 pipeline, for the determinism harness.

The shape is what a flow exporter actually sends: one record per flow, carrying the addresses and ports, the byte
counts in both directions, the TTL the exporter saw, and the destination's autonomous system. Records arrive in
event-time order and are keyed on the source address, which is what Part 2 names as the TC-3 entity.

**Fourteen days of history come before four scored hours.** R-B-L3-001 measures a source against its own fortnight,
and a corpus four hours long could only ever test it against a literal. The history is what the estate's ordinary
hosts did on ordinary days -- a workstation reaching its three servers in the working hours, a DHCP server pinging
the addresses it is about to lease, clients querying the DNS server -- and is quiet by construction: nothing in it
should fire, and the harness asserts that nothing does.

Into the scored hours are planted the things the eight layer 3 rules exist to see, each with the case beside it
that must stay quiet:

- a **scan**: one workstation reaching more internal addresses in each successive hour, far beyond anything its
  fortnight holds, which is R-B-L3-001's condition and, because the growth is monotonic, R-P-L3-005's as well;
- a **DHCP server** whose Monday-morning conflict checks reach sixty addresses -- more than the literal R-B-L3-001
  used to read, and fewer than its own Mondays reach -- which is the control that makes the step load-bearing;
- a **browser**: another host reaching many addresses, all of them on the public internet, which is the control that
  makes "predominantly internal" load-bearing rather than decorative;
- a **workstation suddenly reached by many**, fourteen internal sources where a fortnight held one, which is
  R-B-L3-006, against a **DNS server** whose ordinary hours are busier and an **application server** taking on the
  clients of a new service, which steps and is a server;
- an **upload to a network the host has never reached**, more lopsided than anything in its fortnight, which is
  R-B-L3-007, against the same host's first download from another new network and its heavy upload to the backup
  provider it reaches daily;
- a **port sweep**: one host trying forty ports on one internal server, which is R-D-L3-008, against a conferencing
  client's media streams across a range of ports on an external relay;
- a **beacon**: a pair exchanging a fixed-size message on a strict timer, which R-B-L3-002 reads;
- a **worker**: a host talking to one file server at the ragged intervals a person produces, which must not read
  as a beacon however many flows it makes;
- a **stray**: a single flow to a reserved address, which R-D-L3-003 catches and which is low volume by nature;
- a **tap**: a host whose TTL drops by exactly one hop partway through, which is what an interposed device does
  and what R-B-L3-004 reads;
- and a **quiet server** whose TTL never moves, so the TTL feature is a statement about a change rather than
  about a value.

Every stateful stage is preceded by `TotalOrderStage`, which is determinism control 8 as a stage. The stages
here are cumulative in the same way the layer 1 and 2 ones are -- a count, a proportion, a rhythm, a reference and a
history are all functions of what came before -- so the permutation check has something to catch.
"""

import random
import typing

import pandas as pd

import stamping
from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.community_id_stage import CommunityIdStage
from morpheus.stages.lineage.determinism_stamp_stage import DeterminismStampStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.stages.telemetry.tc3_baseline_stage import TC3BaselineStage
from morpheus.stages.telemetry.tc3_beacon_stage import TC3BeaconStage
from morpheus.stages.telemetry.tc3_cardinality_stage import TC3CardinalityStage
from morpheus.stages.telemetry.tc3_reach_stage import TC3ReachStage
from morpheus.stages.telemetry.tc3_ttl_stage import TC3TtlStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize

CORPUS_SEED = 20260921

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900
DAY_SECONDS = 24 * PERIOD_SECONDS

HISTORY_DAYS = 14
"""The fortnight R-B-L3-001 measures a source against, which is `TC3BaselineStage`'s default window."""

SCORED_START_HOUR = 8
"""The scored hours begin at eight on the fifteenth day, the hour the DHCP server's Monday mornings begin."""

CORPUS_HOURS = 4
"""Scored hours, after the history. The rhythm and TTL references are taken over these."""
CORPUS_SECONDS = CORPUS_HOURS * PERIOD_SECONDS

HISTORY_SECONDS = HISTORY_DAYS * DAY_SECONDS + SCORED_START_HOUR * PERIOD_SECONDS
"""Seconds of corpus before the first scored hour."""

BASELINE_MIN_BUCKETS = 24
"""Active hours a host needs before its own history is a baseline. A day's worth, which is the stage's default and
which every host with a history here clears by the fourth day."""

ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []

TELEMETRY_CLASS = "tc3"
OSI_LAYER = 3
ENTITY_COLUMNS = ["src_ip"]
"""Part 2 names `src_ip` as the TC-3 entity key, and the directed pair separately. The envelope carries the
source, because that is the subject a layer 3 score is about; the pair is the beacon stage's own key and rides
on the row as `flow_pair_key`."""
SETTINGS = {
    "period_seconds": PERIOD_SECONDS,
    "lateness_seconds": LATENESS_SECONDS,
    "corpus_seconds": CORPUS_SECONDS,
    "history_days": HISTORY_DAYS,
    "baseline_min_buckets": BASELINE_MIN_BUCKETS,
}
"""The settings that decide this corpus's output, digested into `config_hash` by `stamping.envelope_for`."""
RULES = ("R-B-L3-001", "R-B-L3-002", "R-D-L3-003", "R-B-L3-004", "R-P-L3-005", "R-B-L3-006", "R-B-L3-007", "R-D-L3-008")
"""The shipped rules that read this corpus's columns; their thresholds are folded into `pipeline_fingerprint`."""

COLLECTOR = "netflow-01"
SCHEMA_VERSION = "tc3.v1"

SCANNER = "10.0.0.50"
BROWSER = "10.0.0.60"
BEACONER = "10.0.0.70"
WORKER = "10.0.0.80"
STRAY = "10.0.0.90"
TAPPED = "10.0.0.100"
QUIET = "10.0.0.110"
UPLOADER = "10.0.0.120"
SWEEPER = "10.0.0.130"
CONFERENCING = "10.0.0.140"

FILE_SERVER = "10.0.1.200"
BEACON_SERVER = "93.184.216.34"
TAP_SERVER = "10.0.1.210"
QUIET_SERVER = "10.0.1.220"
SWEPT_SERVER = "10.0.1.240"
MEDIA_RELAY = "93.184.216.90"
RESERVED_DESTINATION = "240.0.0.1"

DHCP_SERVER = "10.0.1.67"
DNS_SERVER = "10.0.1.53"
APP_SERVER = "10.0.1.230"
REACHED_WORKSTATION = "10.0.2.15"
"""A finance workstation, which in the scored hours is reached by fourteen internal sources where its fortnight held
one: its owner's jump host. R-B-L3-006's case."""
JUMP_HOST = "10.0.1.5"

SCANNER_SERVERS = (FILE_SERVER, "10.0.1.201", "10.0.1.202")
"""The three servers the scanning workstation reaches in its ordinary hours, so its fortnight's peak is three."""

SCAN_PER_HOUR = (5, 20, 45, 80)
"""Distinct internal destinations the scanner reaches in each successive scored hour.

Monotonic on purpose. R-B-L3-001 reads the level and R-P-L3-005 reads the shape, and a corpus whose scan arrived
all at once would satisfy the first rule and say nothing about the second.

Every hour is a step above the three servers the workstation's fortnight holds; only the last clears the literal
R-B-L3-001 keeps as its floor, which is the rule's own statement that a workstation reaching five addresses is not
yet a scan however few it reached before."""

BROWSE_PER_HOUR = (30, 30, 30, 30)
"""Destinations the browser reaches, at a level the scanner only passes in its last hour. Steady rather than
rising, so it is a control for the trajectory rule as well as for the fan-out one."""

DHCP_CHECKS_MONDAY = 64
"""Addresses the DHCP server pings before leasing them on a Monday morning, when the estate's laptops return. Above
the literal R-B-L3-001 reads, and the reason a literal cannot be the rule."""
DHCP_CHECKS_TODAY = 60
"""This Monday's checks: as many as the literal ever fired on and fewer than the server's own Mondays."""
DHCP_RENEWALS = 2
"""Addresses checked in an ordinary hour."""

DNS_CLIENTS_MONDAY = 20
DNS_CLIENTS_TODAY = 18
DNS_CLIENTS_ORDINARY = 3
"""Distinct clients querying the DNS server in its busiest hours, this morning, and an ordinary hour."""

APP_CLIENTS_ORDINARY = 2
APP_CLIENTS_TODAY = 15
"""Clients of the application server before and after a new service is pointed at it. A step, on a server."""

REACHING_SOURCES = 14
"""Internal sources that reach the finance workstation in the scored hours."""

BACKUP_ASN = "64502"
"""The backup provider the uploading host sends a nightly copy to, which is its fortnight's most lopsided traffic."""
UPLOAD_ASN = "64666"
DOWNLOAD_ASN = "64667"
"""Networks the uploading host has never reached: one it sends a large upload to, one it downloads from."""

SWEEP_PORTS = 40
"""Ports the sweeper tries on one server, which is R-D-L3-008's case."""
MEDIA_PORTS = 30
"""Ports a conferencing client's media streams use on its relay, which is R-D-L3-008's control: many ports, none of
them inside the estate."""

BEACON_PERIOD_SECONDS = 300
BEACON_BYTES = 512
"""A fixed-size check-in every five minutes. Twelve an hour, so the twelve intervals R-B-L3-002 wants exist
within the first two hours and the rule has three hours of corpus to be right about."""

WORKER_GAPS = (7, 412, 33, 900, 61, 18, 745, 5, 203, 88, 1300, 12, 470, 29, 151, 640)
"""Seconds between one person's flows to the file server. Ragged at the scale of the gaps themselves, which is
what the coefficient of variation measures."""

DEFAULT_TTL = 64
TAP_AT_HOUR = 2
"""The hour an interposed device starts forwarding the tapped host's traffic, costing it exactly one hop."""

INTERNAL_PREFIX = "10.0.9."
EXTERNAL_HOSTS = tuple(f"93.184.216.{index}" for index in range(1, 61))
"""Addresses the classifier calls global, which the documentation ranges are not.

Worth knowing before writing any corpus for this feature: `192.0.2.0/24`, `198.51.100.0/24` and `203.0.113.0/24`
are reserved for documentation, and `parsers/ip.py` therefore classifies every one of them as private and none of
them as global. A corpus built from the addresses the RFCs set aside for examples cannot exercise the external
path at all -- it silently reports a browser reaching the public internet as a host that never left the estate,
which is the exact condition R-B-L3-001 distinguishes a scan by. The first draft of this corpus did that."""

EXTERNAL_ASNS = ("64500", "64501", "64502", "64503")
INTERNAL_ASN = "64512"

DEVICE_ROLES = {
    DHCP_SERVER: "server",
    DNS_SERVER: "server",
    APP_SERVER: "server",
    FILE_SERVER: "server",
    SWEPT_SERVER: "server",
    REACHED_WORKSTATION: "workstation",
    SCANNER: "workstation",
    UPLOADER: "workstation",
}
"""What the asset inventory says each address is. R-B-L3-006 reads the destination's role."""

CONTEXT_PREFIX = "dst_ctx_"
"""The destination's context, as `TC0EnrichStage` attaches it. Prefixed, because the row's subject is the source and
a `ctx_device_role` on a flow would read as the source's."""


def at(hour: int, second: int = 0) -> int:
    """Event time for an offset into the scored hours, in nanoseconds since the epoch."""
    return (HISTORY_SECONDS + hour * PERIOD_SECONDS + second) * NS_PER_SECOND


def history_at(day: int, hour: int, second: int = 0) -> int:
    """Event time for an offset into the fortnight before them."""
    return (day * DAY_SECONDS + hour * PERIOD_SECONDS + second) * NS_PER_SECOND


def asset_versions() -> list:
    """The inventory's word on what each address is, recorded before the corpus begins."""
    return [
        make_version("asset", address, 0, None, 0, values={"device_role": role})
        for (address, role) in sorted(DEVICE_ROLES.items())
    ]


def build_store() -> BitemporalStore:
    """The asset store the destination enrichment reads."""
    return BitemporalStore("asset", asset_versions())


def _flow(source: str,
          destination: str,
          event_time_ns: int,
          seq: int,
          dst_port: int = 443,
          ttl: int = DEFAULT_TTL,
          bytes_out: int = 1200,
          bytes_in: int = 4800,
          asn: str = INTERNAL_ASN,
          protocol: str = "tcp",
          src_port: typing.Optional[int] = None) -> dict:
    """One flow record, with the envelope every record in this fork carries."""
    return {
        "collector_id": COLLECTOR,
        "schema_version": SCHEMA_VERSION,
        "origin_hash": f"{source}->{destination}",
        "collector_seq": seq,
        "event_time": event_time_ns,
        "src_ip": source,
        "src_port": 49152 + seq % 16384 if src_port is None else src_port,
        "dst_ip": destination,
        "dst_port": dst_port,
        "protocol": protocol,
        "ip_ttl": ttl,
        "bytes_out": bytes_out,
        "bytes_in": bytes_in,
        "bgp_as_dst": asn,
    }


def _history(add) -> None:
    """The fortnight: ordinary hours of the hosts that have one, and nothing a rule should read."""
    for day in range(HISTORY_DAYS):
        monday = day % 7 == 0

        # The scanning workstation, before it scanned: three servers in each working hour.
        for hour in (9, 10, 11):
            for (index, server) in enumerate(SCANNER_SERVERS):
                add(source=SCANNER,
                    destination=server,
                    event_time_ns=history_at(day, hour, 300 + index * 300),
                    dst_port=445,
                    bytes_out=900,
                    bytes_in=24000)

        # The DHCP server's conflict checks: a burst on Monday mornings, a trickle otherwise.
        for hour in (8, 9):
            checks = DHCP_CHECKS_MONDAY if (monday and hour == 8) else DHCP_RENEWALS

            for index in range(checks):
                add(source=DHCP_SERVER,
                    destination=f"10.0.10.{index + 1}",
                    event_time_ns=history_at(day, hour, 30 + index * 20),
                    dst_port=0,
                    protocol="icmp",
                    src_port=8,
                    bytes_out=84,
                    bytes_in=0)

        # Clients querying the DNS server.
        for hour in (8, 9):
            clients = DNS_CLIENTS_MONDAY if (monday and hour == 8) else DNS_CLIENTS_ORDINARY

            for index in range(clients):
                add(source=f"10.0.11.{index + 1}",
                    destination=DNS_SERVER,
                    event_time_ns=history_at(day, hour, 45 + index * 60),
                    dst_port=53,
                    protocol="udp",
                    bytes_out=74,
                    bytes_in=180)

        # The application server's two clients, and the finance workstation's one visitor.
        for hour in (10, 11):
            for index in range(APP_CLIENTS_ORDINARY):
                add(source=f"10.0.12.{index + 1}",
                    destination=APP_SERVER,
                    event_time_ns=history_at(day, hour, 120 + index * 90),
                    dst_port=8443)

            add(source=JUMP_HOST,
                destination=REACHED_WORKSTATION,
                event_time_ns=history_at(day, hour, 600),
                dst_port=3389,
                bytes_out=40000,
                bytes_in=900000)

        # The uploading host: downloads from two networks it knows, and a nightly copy to its backup provider.
        for hour in (9, 10):
            for (index, asn) in enumerate(EXTERNAL_ASNS[:2]):
                add(source=UPLOADER,
                    destination=EXTERNAL_HOSTS[index],
                    event_time_ns=history_at(day, hour, 200 + index * 400),
                    bytes_out=1200,
                    bytes_in=48000,
                    asn=asn)

        add(source=UPLOADER,
            destination=EXTERNAL_HOSTS[2],
            event_time_ns=history_at(day, 10, 2400),
            bytes_out=600000,
            bytes_in=100000,
            asn=BACKUP_ASN)


def _scored(add, rng: random.Random) -> None:
    """The four scored hours, with every planted case and its control."""
    for hour in range(CORPUS_HOURS):
        # The scan. Every destination is inside the estate, and each hour reaches more of them than the last.
        for index in range(SCAN_PER_HOUR[hour]):
            add(source=SCANNER,
                destination=f"{INTERNAL_PREFIX}{index + 1}",
                event_time_ns=at(hour, 60 + index * 7),
                dst_port=445,
                bytes_out=180,
                bytes_in=0)

        # The browser, reaching just as many places on the public internet.
        for index in range(BROWSE_PER_HOUR[hour]):
            add(source=BROWSER,
                destination=EXTERNAL_HOSTS[index % len(EXTERNAL_HOSTS)],
                event_time_ns=at(hour, 90 + index * 11),
                bytes_out=rng.randint(400, 3000),
                bytes_in=rng.randint(4000, 90000),
                asn=EXTERNAL_ASNS[index % len(EXTERNAL_ASNS)])

        # The beacon, on its timer, carrying the same number of bytes every time.
        for index in range(PERIOD_SECONDS // BEACON_PERIOD_SECONDS):
            add(source=BEACONER,
                destination=BEACON_SERVER,
                event_time_ns=at(hour, index * BEACON_PERIOD_SECONDS),
                bytes_out=BEACON_BYTES,
                bytes_in=96,
                asn=EXTERNAL_ASNS[0])

        # The quiet server conversation, whose TTL is the control for the tap below.
        for index in range(6):
            add(source=QUIET, destination=QUIET_SERVER, event_time_ns=at(hour, 120 + index * 400), dst_port=22)

        # The tap. One hop is lost from the hour the device is interposed, and not before.
        for index in range(6):
            add(source=TAPPED,
                destination=TAP_SERVER,
                event_time_ns=at(hour, 150 + index * 380),
                dst_port=3389,
                ttl=DEFAULT_TTL - 1 if hour >= TAP_AT_HOUR else DEFAULT_TTL)

        # The DHCP server: this Monday's checks, then the trickle.
        checks = DHCP_CHECKS_TODAY if hour == 0 else DHCP_RENEWALS

        for index in range(checks):
            add(source=DHCP_SERVER,
                destination=f"10.0.10.{index + 1}",
                event_time_ns=at(hour, 30 + index * 20),
                dst_port=0,
                protocol="icmp",
                src_port=8,
                bytes_out=84,
                bytes_in=0)

        # The DNS server's clients, busy this morning and ordinary after.
        clients = DNS_CLIENTS_TODAY if hour == 0 else DNS_CLIENTS_ORDINARY

        for index in range(clients):
            add(source=f"10.0.11.{index + 1}",
                destination=DNS_SERVER,
                event_time_ns=at(hour, 45 + index * 60),
                dst_port=53,
                protocol="udp",
                bytes_out=74,
                bytes_in=180)

    # The application server takes on a new service's clients in the third hour: a step, on a server.
    for index in range(APP_CLIENTS_TODAY):
        add(source=f"10.0.12.{index + 1}", destination=APP_SERVER, event_time_ns=at(2, 300 + index * 40), dst_port=8443)

    # The finance workstation is reached by fourteen internal sources in the third hour, where its fortnight held one.
    for index in range(REACHING_SOURCES):
        add(source=f"10.0.20.{index + 1}",
            destination=REACHED_WORKSTATION,
            event_time_ns=at(2, 900 + index * 30),
            dst_port=445,
            bytes_out=2400,
            bytes_in=600)

    # The uploading host: its backup provider sent more than ever, then a large upload to a network it has never
    # reached, then a download from another. The backup comes first because each hour joins the envelope the next
    # is measured against, and the upload after it would otherwise hide it.
    add(source=UPLOADER,
        destination=EXTERNAL_HOSTS[2],
        event_time_ns=at(0, 1200),
        bytes_out=5000000,
        bytes_in=100000,
        asn=BACKUP_ASN)
    add(source=UPLOADER,
        destination="93.184.216.66",
        event_time_ns=at(1, 1200),
        bytes_out=9000000,
        bytes_in=3000,
        asn=UPLOAD_ASN)
    add(source=UPLOADER,
        destination="93.184.216.67",
        event_time_ns=at(2, 1200),
        bytes_out=1500,
        bytes_in=80000,
        asn=DOWNLOAD_ASN)

    # The sweep: forty ports on one internal server, each answered, at the uneven pace a scanner keeps when it
    # waits on what each port says back. Neither the gaps nor the sizes are regular, so the pair is a sweep and not a
    # beacon.
    elapsed = 2400

    for index in range(SWEEP_PORTS):
        elapsed += rng.randint(1, 40)
        add(source=SWEEPER,
            destination=SWEPT_SERVER,
            event_time_ns=at(2, elapsed),
            dst_port=1000 + index * 37,
            bytes_out=rng.randint(80, 400),
            bytes_in=rng.randint(40, 2000))

    # A conferencing client's media streams, each on its own port at the relay, outside the estate, opened as the
    # call's participants join and each carrying what its participant sent.
    elapsed = 600

    for index in range(MEDIA_PORTS):
        elapsed += rng.randint(1, 60)
        add(source=CONFERENCING,
            destination=MEDIA_RELAY,
            event_time_ns=at(1, elapsed),
            dst_port=50000 + index,
            protocol="udp",
            bytes_out=rng.randint(20000, 400000),
            bytes_in=rng.randint(20000, 400000),
            asn=EXTERNAL_ASNS[1])

    # The worker, at a person's intervals, across the scored hours rather than per hour.
    elapsed = 30

    for (index, gap) in enumerate(WORKER_GAPS * 3):
        elapsed += gap

        if (elapsed >= CORPUS_SECONDS):
            break

        add(source=WORKER,
            destination=FILE_SERVER,
            event_time_ns=at(0, elapsed),
            dst_port=445,
            bytes_out=rng.randint(200, 40000),
            bytes_in=rng.randint(200, 900000))

    # The stray: one flow to a reserved address, which is the whole of R-D-L3-003's evidence.
    add(source=STRAY, destination=RESERVED_DESTINATION, event_time_ns=at(1, 1800), dst_port=53, asn=None)


def build_corpus() -> dict[str, pd.DataFrame]:
    """The seeded flow corpus, one frame for the single TC-3 class."""
    rng = random.Random(CORPUS_SEED)
    rows: list[dict] = []

    def add(**kwargs):
        rows.append(_flow(seq=len(rows), **kwargs))

    _history(add)
    _scored(add, rng)

    frame = pd.DataFrame(rows).sort_values("event_time", kind="mergesort").reset_index(drop=True)
    frame["collector_seq"] = range(len(frame))

    return {TELEMETRY_CLASS: frame}


def build_pipeline_config(execution_mode=None) -> Config:
    """
    A pipeline configuration, defaulting to CPU mode and importable without a GPU.

    Parameters
    ----------
    execution_mode : `morpheus.config.ExecutionMode`, optional
        Mode to build for. Defaults to CPU, which is what the golden file is an artifact of. The parameter exists
        so the same corpus and the same golden can be driven in GPU mode and the two compared. Resolved inside the
        function rather than as a default argument so that importing this module still does not require a GPU.
    """
    from morpheus.config import CppConfig  # pylint: disable=import-outside-toplevel
    from morpheus.config import ExecutionMode  # pylint: disable=import-outside-toplevel

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.execution_mode = ExecutionMode.CPU if execution_mode is None else execution_mode

    return config


def _collect(sink: InMemorySinkStage) -> pd.DataFrame:
    """Everything the sink received, as one host frame."""
    from host_frame import to_host_frame

    frames = []

    for message in sink.get_messages():
        meta = message.payload() if hasattr(message, "payload") else message
        frame = meta.copy_dataframe()
        # Integer columns are carried as nullable integers in both modes before anything is joined, so the
        # concatenation cannot widen them to float in one mode and not the other; see `host_frame`.
        frames.append(to_host_frame(frame))

    if (len(frames) == 0):
        return pd.DataFrame()

    # pandas warns here, and at the same operation inside `WindowSealStage`, that a future version will stop
    # excluding all-NA columns when it infers the concatenated dtypes. Both warnings are real and neither is
    # acted on yet: the dtypes this produces today are the ones the golden was generated from, and the
    # batch-split sweep compares the by-row split against the single-frame run byte for byte, so a future pandas
    # changing them would fail a test rather than drift. The corpus reaches the case the other harnesses do not
    # because a single flow's row legitimately has no coefficient and no TTL shift to report.
    return pd.concat(frames, ignore_index=True)


def build_stages(config: Config, min_denominator: int = 5) -> list:
    """The TC-3 stages, in the order a deployment would compose them.

    Cardinality first because it needs nothing from the others, then reach, which classifies the destination the
    counts were taken over, then the baseline, which measures two of the counts and reach's asymmetry against each
    host's fortnight, then the rhythm and the reference, which are per-pair and per-source rather than per-window
    and are independent of the rest. The destination's context comes last of the feature stages; nothing before it
    reads it.
    """
    return [
        TC3CardinalityStage(config, window_seconds=PERIOD_SECONDS),
        TC3ReachStage(config, window_seconds=PERIOD_SECONDS, min_denominator=min_denominator),
        TC3BaselineStage(config, bucket_seconds=PERIOD_SECONDS, min_buckets=BASELINE_MIN_BUCKETS),
        TC3BeaconStage(config, window_seconds=CORPUS_SECONDS, min_intervals=12),
        TC3TtlStage(config, window_seconds=CORPUS_SECONDS, min_samples=5),
        TC0EnrichStage(config, store=build_store(), entity_column="dst_ip", prefix=CONTEXT_PREFIX),
    ]


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True) -> pd.DataFrame:
    """
    Run the TC-3 class through its composed pipeline and return one canonicalized frame.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    corpus : dict
        The frames from `build_corpus`, possibly permuted.
    batches : dict, optional
        How the corpus is split across source frames. Defaults to one frame. The batch-split sweep is the
        caller's to vary.
    impose_order : bool, default = True
        Place `TotalOrderStage` ahead of the stateful stages. The permutation check's negative control turns it
        off, and every stage here is cumulative, so the difference is visible.

    Returns
    -------
    `pandas.DataFrame`
        The class's output, tagged with `telemetry_class`, keyed by `row_key`, canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=batches[TELEMETRY_CLASS]))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    # The flow's Community ID, so a layer 3 record joins the layer 4 and 6 records of the same connection by
    # equality rather than by a time window. Stateless, so its place relative to the order makes no difference.
    pipe.add_stage(CommunityIdStage(config, dst_ip_column="dst_ip", dst_port_column="dst_port"))

    for stage in build_stages(config):
        pipe.add_stage(stage)

    pipe.add_stage(DeterminismStampStage(config, envelope=stamping.envelope_for(TELEMETRY_CLASS, SETTINGS,
                                                                                rules=RULES)))
    pipe.add_stage(EnvelopeStampStage(config, osi_layer=OSI_LAYER, entity_columns=ENTITY_COLUMNS))
    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=PERIOD_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        entity_key_column="entity_key"))

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    result = _collect(sink)
    result["telemetry_class"] = TELEMETRY_CLASS
    result["row_key"] = result["event_uid"]

    return canonicalize(result, key_columns=KEY_COLUMNS, ignore_columns=IGNORE_COLUMNS)


def render(result: pd.DataFrame) -> str:
    """The canonical CSV rendering the golden file holds."""
    return result.to_csv(index=False, lineterminator="\n")
