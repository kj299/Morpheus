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
The synthetic corpus `morpheus.compositions.network` is tested over, and the facts about it the harnesses assert.

Moved out of `tests/morpheus/determinism/network_pipeline.py` under plan section 4.5, which put the composition in the
package and left the data here, where the tests, the sample and fixture generators and the experiments read it. The
composition module's docstring describes what this corpus plants and why.
"""

import random
import typing

from corpora import merged  # noqa: F401

from morpheus.compositions import envelope

host_identity = merged('host_identity')
import pandas as pd

from morpheus.compositions.network import APP_SERVER
from morpheus.compositions.network import CORPUS_HOURS
from morpheus.compositions.network import CORPUS_SECONDS
from morpheus.compositions.network import DHCP_SERVER
from morpheus.compositions.network import DNS_SERVER
from morpheus.compositions.network import FILE_SERVER
from morpheus.compositions.network import HISTORY_DAYS
from morpheus.compositions.network import LATE_DELIVERY_CLASS
from morpheus.compositions.network import PERIOD_SECONDS
from morpheus.compositions.network import REACHED_WORKSTATION
from morpheus.compositions.network import SCANNER
from morpheus.compositions.network import SWEPT_SERVER
from morpheus.compositions.network import TELEMETRY_CLASS
from morpheus.compositions.network import UPLOADER
from morpheus.utils.binding_table import NS_PER_SECOND

CORPUS_SEED = 20260921
DAY_SECONDS = 24 * PERIOD_SECONDS
SCORED_START_HOUR = 8
"""The scored hours begin at eight on the fifteenth day, the hour the DHCP server's Monday mornings begin."""
HISTORY_SECONDS = HISTORY_DAYS * DAY_SECONDS + SCORED_START_HOUR * PERIOD_SECONDS
"""Seconds of corpus before the first scored hour."""
COLLECTOR = "netflow-01"
SCHEMA_VERSION = "tc3.v1"
EXPORTERS = (("10.0.0.", "rtr-access-1:256"), ("10.0.2.", "rtr-access-2:768"), ("10.0.1.", "rtr-dc-1:512"),
             ("10.0.9.", "rtr-access-1:256"))
"""The router that exports a source's flows, and its IPFIX observation domain, by the source's subnet."""
EDGE_EXPORTER = "rtr-edge-1:1024"
"""Every source outside the estate's subnets is seen first at the edge."""
BROWSER = "10.0.0.60"
BEACONER = "10.0.0.70"
WORKER = "10.0.0.80"
STRAY = "10.0.0.90"
TAPPED = "10.0.0.100"
QUIET = "10.0.0.110"
SWEEPER = "10.0.0.130"
CONFERENCING = "10.0.0.140"
BEACON_SERVER = "93.184.216.34"
TAP_SERVER = "10.0.1.210"
QUIET_SERVER = "10.0.1.220"
MEDIA_RELAY = "93.184.216.90"
RESERVED_DESTINATION = "240.0.0.1"
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


def at(hour: int, second: int = 0) -> int:
    """Event time for an offset into the scored hours, in nanoseconds since the epoch."""
    return (HISTORY_SECONDS + hour * PERIOD_SECONDS + second) * NS_PER_SECOND


def history_at(day: int, hour: int, second: int = 0) -> int:
    """Event time for an offset into the fortnight before them."""
    return (day * DAY_SECONDS + hour * PERIOD_SECONDS + second) * NS_PER_SECOND


def exporter_for(source: str) -> str:
    """The exporter that sees a source's flows first."""
    for (prefix, exporter) in EXPORTERS:
        if (source.startswith(prefix)):
            return exporter

    return EDGE_EXPORTER


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
        "exporter_id": exporter_for(source),
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

    # The late flow: the quiet host's half-hour connection in the second scored hour, exported from the access
    # router's cache at the end of the fourth. Its event time is when the flow began, its observed time when the
    # exporter let it go, and it is the collector's last record because it was the last to arrive.
    late = _flow(source=QUIET, destination=QUIET_SERVER, event_time_ns=at(1, 1800), seq=len(frame))
    late["observed_time"] = at(CORPUS_HOURS - 1, 3000)

    # Which machine each client address was: the DHCP server's leases and the EDR agent's inventory, over the span.
    hosts = host_identity.build(frame["src_ip"], int(frame["event_time"].min()), int(frame["event_time"].max()))

    return envelope.attach_all({TELEMETRY_CLASS: frame, LATE_DELIVERY_CLASS: pd.DataFrame([late]), **hosts})
