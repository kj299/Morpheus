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
The synthetic corpus `morpheus.compositions.transport` is tested over, and the facts about it the harnesses assert.

Moved out of `tests/morpheus/determinism/transport_pipeline.py` under plan section 4.5, which put the composition in the
package and left the data here, where the tests, the sample and fixture generators and the experiments read it. The
composition module's docstring describes what this corpus plants and why.
"""

import random

from corpora import merged  # noqa: F401

from morpheus.compositions import envelope

host_identity = merged('host_identity')
import pandas as pd

from morpheus.compositions.transport import CORPUS_HOURS
from morpheus.compositions.transport import PERIOD_SECONDS
from morpheus.compositions.transport import TELEMETRY_CLASS
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.tcp_flags import ACK
from morpheus.utils.tcp_flags import FIN
from morpheus.utils.tcp_flags import PSH
from morpheus.utils.tcp_flags import RST
from morpheus.utils.tcp_flags import SYN

CORPUS_SEED = 20260921
COLLECTOR = "pcap-01"
SCHEMA_VERSION = "tc4.v1"
SCANNER = "10.0.0.50"
WORKSTATION = "10.0.0.60"
SECOND_STACK = "10.0.0.61"
"""A workstation that starts presenting a stack it has never presented, under the same browser."""
TOOL_HOST = "10.0.0.62"
"""A Linux machine whose requests come from `curl`, which says nothing about an operating system."""
SWEEPER = "10.0.0.70"
BACKUP_CLIENT = "10.0.0.80"
BUSY_CLIENT = "10.0.0.90"
OUTAGE_CLIENTS = tuple(f"10.0.0.1{index:02d}" for index in range(1, 21))
TARGET = "10.0.1.9"
FILE_SERVER = "10.0.1.20"
BACKUP_SERVER = "10.0.1.30"
BUSY_SERVER = "10.0.1.40"
FAILING_SERVER = "10.0.1.50"
SCAN_PORTS = 60
"""Destination ports the scan touches inside one minute. Above R-D-L4-002's fifty, and the workstation below
never comes close, which is what makes the rule a test of the corpus rather than of the threshold."""
SCAN_AT_MINUTE = 5
SWEEP_TARGETS = 30
OUTAGE_AT_MINUTE = 40
BACKUP_HISTORY = 120
"""Prior transfers on the exfiltration triple. Above the hundred a 99th percentile needs to be a percentile."""
BACKUP_BYTES = 1200
EXFIL_BYTES = 60000
BUSY_BYTES = (400, 900)
WINDOWS_STACK = ("mss,nop,ws,nop,nop,sok", 1460, 64240)
LINUX_STACK = ("mss,sok,ts,nop,ws", 1460, 29200)
"""The SYN options, maximum segment size and initial window of current Windows and Linux stacks."""
SYN_ACK = SYN | ACK
PSH_ACK = PSH | ACK
FIN_ACK = FIN | ACK
RST_ACK = RST | ACK


def at(hour: int, second: int = 0) -> int:
    """Event time for an offset into the corpus, in nanoseconds since the epoch."""
    return (hour * PERIOD_SECONDS + second) * NS_PER_SECOND


def build_corpus() -> dict[str, pd.DataFrame]:
    """The seeded packet corpus, one frame for the single TC-4 class."""
    rng = random.Random(CORPUS_SEED)
    rows: list[dict] = []

    def add(src, sport, dst, dport, flags, event_time_ns, data_len=0, stack=None):
        (options, mss, window) = stack if stack is not None else (None, None, None)
        rows.append({
            "collector_id": COLLECTOR,
            "schema_version": SCHEMA_VERSION,
            "origin_hash": f"{src}:{sport}={dst}:{dport}",
            "collector_seq": len(rows),
            "event_time": event_time_ns,
            "src_ip": src,
            "src_port": sport,
            "dst_ip": dst,
            "dst_port": dport,
            "protocol": "tcp",
            "tcp_flags": flags,
            "data_len": data_len,
            "tcp_options_order": options,
            "tcp_mss": mss,
            "tcp_window_initial": window,
        })

    def respond(src, sport, dst, dport, flags, event_time_ns, data_len=0):
        """A packet travelling the other way down the same conversation.

        A capture sees the answer with the server as its source, so it belongs to the reverse `flow_id` rather
        than to the one the request opened. Both rules that read a refusal depend on that: the flags of a
        refused connection land on the reverse flow, and which side of it fans out is what separates a sweep
        from an outage.
        """
        add(src=dst, sport=dport, dst=src, dport=sport, flags=flags, event_time_ns=event_time_ns, data_len=data_len)

    # The scan: one SYN per destination port, inside a single minute, and nothing answering.
    for index in range(SCAN_PORTS):
        add(SCANNER, 40000 + index, TARGET, 1000 + index, SYN, at(0, SCAN_AT_MINUTE * 60 + index // 2))

    def handshake(client, sport, port, base, stack):
        add(client, sport, FILE_SERVER, port, SYN, base, stack=stack)
        respond(client, sport, FILE_SERVER, port, SYN_ACK, base + NS_PER_SECOND)
        add(client, sport, FILE_SERVER, port, PSH_ACK, base + 2 * NS_PER_SECOND, data_len=1400)
        add(client, sport, FILE_SERVER, port, FIN_ACK, base + 3 * NS_PER_SECOND)

    # The workstation: complete handshakes, which is what `syn/all` alone would flag and what must not be.
    for hour in range(CORPUS_HOURS):
        for (index, port) in enumerate((443, 445, 22)):
            base = at(hour, 600 + index * 120)
            sport = 51000 + index

            add(WORKSTATION, sport, FILE_SERVER, port, SYN, base, stack=WINDOWS_STACK)
            respond(WORKSTATION, sport, FILE_SERVER, port, SYN_ACK, base + NS_PER_SECOND)
            add(WORKSTATION, sport, FILE_SERVER, port, PSH_ACK, base + 2 * NS_PER_SECOND, data_len=1400)
            add(WORKSTATION, sport, FILE_SERVER, port, PSH_ACK, base + 3 * NS_PER_SECOND, data_len=900)
            add(WORKSTATION, sport, FILE_SERVER, port, FIN_ACK, base + 4 * NS_PER_SECOND)

    # The second workstation: Windows's stack for an hour, then a Linux one it has never presented.
    for hour in range(CORPUS_HOURS):
        for (index, port) in enumerate((443, 445)):
            handshake(SECOND_STACK,
                      52000 + hour * 10 + index,
                      port,
                      at(hour, 900 + index * 120),
                      WINDOWS_STACK if hour == 0 else LINUX_STACK)

    # The tool host: Linux's stack throughout.
    for hour in range(CORPUS_HOURS):
        handshake(TOOL_HOST, 53000 + hour, 443, at(hour, 1100), LINUX_STACK)

    # The closed-port sweep: one source, many destinations, all refusing.
    for index in range(SWEEP_TARGETS):
        add(SWEEPER, 42000 + index, f"10.0.2.{index + 1}", 3389, SYN, at(0, 1800 + index))
        respond(SWEEPER, 42000 + index, f"10.0.2.{index + 1}", 3389, RST_ACK, at(0, 1801 + index))

    # The outage: many sources, one destination, the same ratio pointed the other way.
    for (index, client) in enumerate(OUTAGE_CLIENTS):
        add(client, 43000 + index, FAILING_SERVER, 8080, SYN, at(0, OUTAGE_AT_MINUTE * 60 + index))
        respond(client, 43000 + index, FAILING_SERVER, 8080, RST_ACK, at(0, OUTAGE_AT_MINUTE * 60 + index + 1))

    # The backup triple: a long, regular history and then one transfer far outside it.
    for index in range(BACKUP_HISTORY):
        add(BACKUP_CLIENT, 44000 + index, BACKUP_SERVER, 445, PSH_ACK, at(0, 10 + index * 20), data_len=BACKUP_BYTES)

    add(BACKUP_CLIENT, 45000, BACKUP_SERVER, 445, PSH_ACK, at(1, 1200), data_len=EXFIL_BYTES)

    # The busy but bounded triple: transfers varying by a factor of two, which must not breach.
    for index in range(BACKUP_HISTORY):
        add(BUSY_CLIENT,
            46000 + index,
            BUSY_SERVER,
            443,
            PSH_ACK,
            at(0, 15 + index * 20),
            data_len=rng.randint(*BUSY_BYTES))

    add(BUSY_CLIENT, 47000, BUSY_SERVER, 443, PSH_ACK, at(1, 1200), data_len=BUSY_BYTES[1])

    frame = pd.DataFrame(rows).sort_values("event_time", kind="mergesort").reset_index(drop=True)
    frame["collector_seq"] = range(len(frame))
    # Only a SYN carries its options, so these are integers with a gap on every other packet. Held as nullable
    # integers, as a collector sends them, rather than the float pandas widens a gapped column to.
    for column in ("tcp_mss", "tcp_window_initial"):
        frame[column] = frame[column].astype("Int64")

    # Which machine each client address was: the DHCP server's leases and the EDR agent's inventory, over the span.
    hosts = host_identity.build(frame["src_ip"], int(frame["event_time"].min()), int(frame["event_time"].max()))

    return envelope.attach_all({TELEMETRY_CLASS: frame, **hosts})
