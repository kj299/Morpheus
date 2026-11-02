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

Every flow names the exporter that sent it, `exporter_id`: the router and the IPFIX observation domain, from the
source's subnet. One collector, `netflow-01`, receives them all, and that is exactly why the exporter has to be on
the record -- a flow's timestamp is the exporter's clock, not the collector's, and the clock-skew experiment and the
quarantine both need to know whose clock it is.

And one flow is **late**: the access router held a record in its cache and exported it at the end of the scored
hours, long after the hour it belongs to was sealed. It is delivered as its own trailing frame, after every other
record, because that is when it reached the pipeline; the sealer marks it `is_late` and not `window_complete`, it
reaches the sample events, and no detection reads it.

Every stateful stage is preceded by `TotalOrderStage`, which is determinism control 8 as a stage. The stages
here are cumulative in the same way the layer 1 and 2 ones are -- a count, a proportion, a rhythm, a reference and a
history are all functions of what came before -- so the permutation check has something to catch.
"""

import typing

import pandas as pd

from morpheus.compositions import envelope
from morpheus.compositions import host_identity
from morpheus.compositions import stamping
from morpheus.compositions.corpus_pipeline import _collect
from morpheus.compositions.corpus_pipeline import (  # noqa: F401 pylint: disable=unused-import  # re-exported for callers
    build_pipeline_config, )
from morpheus.compositions.resumable import ResumableLinearPipeline
from morpheus.config import Config
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.chain_anchor_stage import DEFAULT_ANCHOR_COLUMN
from morpheus.stages.lineage.collector_sequence_stage import CollectorSequenceStage
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
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900
HISTORY_DAYS = 14
"""The fortnight R-B-L3-001 measures a source against, which is `TC3BaselineStage`'s default window."""
CORPUS_HOURS = 4
"""Scored hours, after the history. The rhythm and TTL references are taken over these."""
CORPUS_SECONDS = CORPUS_HOURS * PERIOD_SECONDS
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
LATE_DELIVERY_CLASS = "tc3_late_delivery"
"""The frame the late flow arrives in, after everything else. Its records are TC-3 flows like any other."""
SCANNER = "10.0.0.50"
UPLOADER = "10.0.0.120"
FILE_SERVER = "10.0.1.200"
SWEPT_SERVER = "10.0.1.240"
DHCP_SERVER = "10.0.1.67"
DNS_SERVER = "10.0.1.53"
APP_SERVER = "10.0.1.230"
REACHED_WORKSTATION = "10.0.2.15"
"""A finance workstation, which in the scored hours is reached by fourteen internal sources where its fortnight held
one: its owner's jump host. R-B-L3-006's case."""
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


def asset_versions() -> list:
    """The inventory's word on what each address is, recorded before the corpus begins."""
    return [
        make_version("asset", address, 0, None, 0, values={"device_role": role})
        for (address, role) in sorted(DEVICE_ROLES.items())
    ]


def build_store() -> BitemporalStore:
    """The asset store the destination enrichment reads."""
    return BitemporalStore("asset", asset_versions())


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

    # Validated once per class, ahead of every stage: ingest_time stamped, the envelope checked, untrusted clocks
    # taken out of the stream and kept.
    (batches, quarantined) = envelope.validate(config, batches, ID_COLUMNS)
    (leases, inventory) = host_identity.tables(config, batches, impose_order)

    pipe = ResumableLinearPipeline(config)
    # The late delivery arrives after every other record, however the rest were split, because that is when it came.
    pipe.set_source(
        InMemorySourceStage(config, dataframes=batches[TELEMETRY_CLASS] + batches.get(LATE_DELIVERY_CLASS, [])))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    pipe.add_stage(CollectorSequenceStage(config))

    # The flow's Community ID, so a layer 3 record joins the layer 4 and 6 records of the same connection by
    # equality rather than by a time window. Stateless, so its place relative to the order makes no difference.
    pipe.add_stage(CommunityIdStage(config, dst_ip_column="dst_ip", dst_port_column="dst_port"))

    for stage in build_stages(config):
        pipe.add_stage(stage)

    # The source address to its interface and its machine, and the chain rooted on the machine where there is one.
    # A host's baselines stay keyed on its address; what changes is that its story can be told across layers.
    for stage in host_identity.resolver_stages(config, leases, inventory):
        pipe.add_stage(stage)

    pipe.add_stage(DeterminismStampStage(config, envelope=stamping.envelope_for(TELEMETRY_CLASS, SETTINGS,
                                                                                rules=RULES)))
    pipe.add_stage(EnvelopeStampStage(config, osi_layer=OSI_LAYER, entity_columns=ENTITY_COLUMNS))
    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=PERIOD_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        entity_key_column=DEFAULT_ANCHOR_COLUMN))

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    # pandas warns inside `_collect`, and at the same operation inside `WindowSealStage`, that a future version will
    # stop excluding all-NA columns when it infers the concatenated dtypes. Both warnings are real and neither is
    # acted on yet: the dtypes this produces today are the ones the golden was generated from, and the batch-split
    # sweep compares the by-row split against the single-frame run byte for byte, so a future pandas changing them
    # would fail a test rather than drift. The corpus reaches the case the other harnesses do not because a single
    # flow's row legitimately has no coefficient and no TTL shift to report.
    result = _collect(sink)

    # A collector every record of which was quarantined delivered nothing, and the class is absent rather than an
    # empty frame -- which, with the DHCP server and the EDR inventory beside it, is now a spread the sweeps reach.
    if (len(result) > 0):
        result["telemetry_class"] = TELEMETRY_CLASS
        result["row_key"] = result["event_uid"]

    return canonicalize(envelope.with_quarantine(result, quarantined),
                        key_columns=KEY_COLUMNS,
                        ignore_columns=IGNORE_COLUMNS)


def render(result: pd.DataFrame) -> str:
    """The canonical CSV rendering the golden file holds."""
    return result.to_csv(index=False, lineterminator="\n")
