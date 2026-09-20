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

The shape is what a flow exporter actually sends: one record per flow, carrying the addresses, the destination
port, the byte counts in both directions, the TTL the exporter saw, and the destination's autonomous system.
Records arrive in event-time order across four hours and are keyed on the source address, which is what Part 2
names as the TC-3 entity.

Into that corpus are planted the things the five layer 3 rules exist to see, each with the case beside it that
must stay quiet:

- a **scan**: one host reaching more internal addresses in each successive hour, which is R-B-L3-001's condition
  and, because the growth is monotonic, R-P-L3-005's as well;
- a **browser**: another host reaching just as many addresses, all of them on the public internet, which is the
  negative control that makes "predominantly internal" load-bearing rather than decorative;
- a **beacon**: a pair exchanging a fixed-size message on a strict timer, which R-B-L3-002 reads;
- a **worker**: a host talking to one file server at the ragged intervals a person produces, which must not read
  as a beacon however many flows it makes;
- a **stray**: a single flow to a reserved address, which R-D-L3-003 catches and which is low volume by nature;
- a **tap**: a host whose TTL drops by exactly one hop partway through, which is what an interposed device does
  and what R-B-L3-004 reads;
- and a **quiet server** whose TTL never moves, so the TTL feature is a statement about a change rather than
  about a value.

Every stateful stage is preceded by `TotalOrderStage`, which is determinism control 8 as a stage. The stages
here are cumulative in the same way the layer 1 and 2 ones are -- a count, a proportion, a rhythm and a
reference are all functions of what came before -- so the permutation check has something to catch.
"""

import random
import typing

import pandas as pd

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc3_beacon_stage import TC3BeaconStage
from morpheus.stages.telemetry.tc3_cardinality_stage import TC3CardinalityStage
from morpheus.stages.telemetry.tc3_reach_stage import TC3ReachStage
from morpheus.stages.telemetry.tc3_ttl_stage import TC3TtlStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize

CORPUS_SEED = 20260921

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900
CORPUS_HOURS = 4
CORPUS_SECONDS = CORPUS_HOURS * PERIOD_SECONDS

ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []

TELEMETRY_CLASS = "tc3"
OSI_LAYER = 3
ENTITY_COLUMNS = ["src_ip"]
"""Part 2 names `src_ip` as the TC-3 entity key, and the directed pair separately. The envelope carries the
source, because that is the subject a layer 3 score is about; the pair is the beacon stage's own key and rides
on the row as `flow_pair_key`."""

COLLECTOR = "netflow-01"
SCHEMA_VERSION = "tc3.v1"

SCANNER = "10.0.0.50"
BROWSER = "10.0.0.60"
BEACONER = "10.0.0.70"
WORKER = "10.0.0.80"
STRAY = "10.0.0.90"
TAPPED = "10.0.0.100"
QUIET = "10.0.0.110"

FILE_SERVER = "10.0.1.200"
BEACON_SERVER = "93.184.216.34"
TAP_SERVER = "10.0.1.210"
QUIET_SERVER = "10.0.1.220"
RESERVED_DESTINATION = "240.0.0.1"

SCAN_PER_HOUR = (5, 20, 45, 80)
"""Distinct internal destinations the scanner reaches in each successive hour.

Monotonic on purpose. R-B-L3-001 reads the level and R-P-L3-005 reads the shape, and a corpus whose scan arrived
all at once would satisfy the first rule and say nothing about the second.

The last hour clears the shipped rule's threshold and the browser below never does, which is what makes the
corpus a test of the rule rather than of the corpus. That threshold is a fixed number standing in for the
per-source fourteen-day percentile the guide actually asks for; the app cannot compute one without a history it
does not keep, and the substitution is recorded in the search's own comment rather than left for a reader to
notice."""

BROWSE_PER_HOUR = (30, 30, 30, 30)
"""Destinations the browser reaches, at a level the scanner only passes in its last hour. Steady rather than
rising, so it is a control for the trajectory rule as well as for the fan-out one."""

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
    """Event time for an offset into the corpus, in nanoseconds since the epoch."""
    return (hour * PERIOD_SECONDS + second) * NS_PER_SECOND


def _flow(source: str,
          destination: str,
          event_time_ns: int,
          seq: int,
          dst_port: int = 443,
          ttl: int = DEFAULT_TTL,
          bytes_out: int = 1200,
          bytes_in: int = 4800,
          asn: str = INTERNAL_ASN) -> dict:
    """One flow record, with the envelope every record in this fork carries."""
    return {
        "collector_id": COLLECTOR,
        "schema_version": SCHEMA_VERSION,
        "origin_hash": f"{source}->{destination}",
        "collector_seq": seq,
        "event_time": event_time_ns,
        "src_ip": source,
        "dst_ip": destination,
        "dst_port": dst_port,
        "protocol": "tcp",
        "ip_ttl": ttl,
        "bytes_out": bytes_out,
        "bytes_in": bytes_in,
        "bgp_as_dst": asn,
    }


def build_corpus() -> dict[str, pd.DataFrame]:
    """The seeded flow corpus, one frame for the single TC-3 class."""
    rng = random.Random(CORPUS_SEED)
    rows: list[dict] = []

    def add(**kwargs):
        rows.append(_flow(seq=len(rows), **kwargs))

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

    # The worker, at a person's intervals, across the whole corpus rather than per hour.
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
    frames = []

    for message in sink.get_messages():
        meta = message.payload() if hasattr(message, "payload") else message
        frame = meta.copy_dataframe()
        frames.append(frame.to_pandas() if hasattr(frame, "to_pandas") else frame)

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
    """The four TC-3 stages, in the order a deployment would compose them.

    Cardinality first because it needs nothing from the others, then reach, which classifies the destination the
    counts were taken over, then the rhythm and the reference, which are per-pair and per-source rather than
    per-window and are independent of both.
    """
    return [
        TC3CardinalityStage(config, window_seconds=PERIOD_SECONDS),
        TC3ReachStage(config, window_seconds=PERIOD_SECONDS, min_denominator=min_denominator),
        TC3BeaconStage(config, window_seconds=CORPUS_SECONDS, min_intervals=12),
        TC3TtlStage(config, window_seconds=CORPUS_SECONDS, min_samples=5),
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
