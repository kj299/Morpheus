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
A packet-shaped layer 4 corpus, and the composed TC-4 pipeline, for the determinism harness.

The shape is what a capture emits: one record per packet, carrying the five-tuple, the TCP flags byte and the
payload length. Records arrive in event-time order across two hours and are keyed on `flow_id`, which is what
Part 2 names as the TC-4 entity, with `community_id` beside it for the cross-tool join the guide asks for --
layer 4 is where that identifier finally has a natural home rather than being computed for its own sake.

Into that corpus are planted the things the rules exist to see, each with the case beside it that must stay
quiet:

- a **port scan**: SYN to sixty destination ports from one source inside a minute and nothing coming back, which
  is R-D-L4-002;
- an **ordinary workstation**: complete handshakes to a handful of servers, which is what most of an estate does
  and what a rule reading `syn/all` alone would flag;
- a **closed-port sweep**: one source collecting RST from many destinations, which is R-D-L4-003 pointed one way;
- a **service outage**: many sources collecting RST from one destination, which is the same ratio pointed the
  other way and needs a different response, so the corpus has to contain both or the rule cannot be shown to
  distinguish them;
- an **exfiltration**: a backup triple with a long, regular history and then one transfer far outside it, which
  is R-B-L4-005;
- and a **busy but bounded** triple whose transfers vary by a factor of two, which must not breach.

**The exfiltration triple is given a hundred and twenty prior transfers on purpose.** A 99th percentile by
nearest rank over fewer than a hundred samples is simply the maximum, so a corpus that gave the triple twenty
transfers would be demonstrating "three times the largest ever seen" while the rule and its documentation both
say "three times the 99th percentile". The corpus is built to let the rule be what it claims to be.

Every stateful stage is preceded by `TotalOrderStage`, which is determinism control 8 as a stage. Both TC-4
stages are cumulative -- a running bin total and a trailing quantile -- so the permutation check has something to
catch.
"""

import random
import typing

import pandas as pd

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.community_id_stage import CommunityIdStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc4_envelope_stage import TC4EnvelopeStage
from morpheus.stages.telemetry.tc4_flow_stage import TC4FlowStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize
from morpheus.utils.tcp_flags import ACK
from morpheus.utils.tcp_flags import FIN
from morpheus.utils.tcp_flags import PSH
from morpheus.utils.tcp_flags import RST
from morpheus.utils.tcp_flags import SYN

CORPUS_SEED = 20260921

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900
BIN_SECONDS = 60
CORPUS_HOURS = 2
CORPUS_SECONDS = CORPUS_HOURS * PERIOD_SECONDS

ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []

TELEMETRY_CLASS = "tc4"
OSI_LAYER = 4
ENTITY_COLUMNS = ["flow_id"]

COLLECTOR = "pcap-01"
SCHEMA_VERSION = "tc4.v1"

SCANNER = "10.0.0.50"
WORKSTATION = "10.0.0.60"
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

    def add(src, sport, dst, dport, flags, event_time_ns, data_len=0):
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

    # The workstation: complete handshakes, which is what `syn/all` alone would flag and what must not be.
    for hour in range(CORPUS_HOURS):
        for (index, port) in enumerate((443, 445, 22)):
            base = at(hour, 600 + index * 120)
            sport = 51000 + index

            add(WORKSTATION, sport, FILE_SERVER, port, SYN, base)
            respond(WORKSTATION, sport, FILE_SERVER, port, SYN_ACK, base + NS_PER_SECOND)
            add(WORKSTATION, sport, FILE_SERVER, port, PSH_ACK, base + 2 * NS_PER_SECOND, data_len=1400)
            add(WORKSTATION, sport, FILE_SERVER, port, PSH_ACK, base + 3 * NS_PER_SECOND, data_len=900)
            add(WORKSTATION, sport, FILE_SERVER, port, FIN_ACK, base + 4 * NS_PER_SECOND)

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

    return {TELEMETRY_CLASS: frame}


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


def _collect(sink: InMemorySinkStage) -> pd.DataFrame:
    """Everything the sink received, as one host frame."""
    frames = []

    for message in sink.get_messages():
        meta = message.payload() if hasattr(message, "payload") else message
        frame = meta.copy_dataframe()
        frames.append(frame.to_pandas() if hasattr(frame, "to_pandas") else frame)

    if (len(frames) == 0):
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def build_stages(config: Config) -> list:
    """The TC-4 stages, in the order a deployment would compose them.

    The flow rollup first, because the envelope reads the magnitudes it produces; `community_id` beside them,
    computed from the five-tuple the records already carry.
    """
    return [
        TC4FlowStage(config, bin_seconds=BIN_SECONDS),
        TC4EnvelopeStage(config, window_seconds=CORPUS_SECONDS * 4),
        CommunityIdStage(config, dst_ip_column="dst_ip", dst_port_column="dst_port"),
    ]


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True) -> pd.DataFrame:
    """
    Run the TC-4 class through its composed pipeline and return one canonicalized frame.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    corpus : dict
        The frames from `build_corpus`, possibly permuted.
    batches : dict, optional
        How the corpus is split across source frames. Defaults to one frame.
    impose_order : bool, default = True
        Place `TotalOrderStage` ahead of the stateful stages.

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

    for stage in build_stages(config):
        pipe.add_stage(stage)

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
