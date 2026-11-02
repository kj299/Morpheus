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

The handshakes carry the TCP options their SYN advertised, which is the stack fingerprint R-B-L4-004 reads. The
ordinary workstation's are Windows's throughout; a **second workstation** presents Windows's for an hour and then a
Linux stack it has never presented, under a browser that still says Windows in the application corpus's proxy log;
and a **tool host** presents Linux's throughout under `curl`, which claims no operating system. Every SYN to the file
server lands on a port the CMDB maps to an application, which `TC0EnrichStage` attaches as `ctx_application_*`.

**The exfiltration triple is given a hundred and twenty prior transfers on purpose.** A 99th percentile by
nearest rank over fewer than a hundred samples is simply the maximum, so a corpus that gave the triple twenty
transfers would be demonstrating "three times the largest ever seen" while the rule and its documentation both
say "three times the 99th percentile". The corpus is built to let the rule be what it claims to be.

Every stateful stage is preceded by `TotalOrderStage`, which is determinism control 8 as a stage. Both TC-4
stages are cumulative -- a running bin total and a trailing quantile -- so the permutation check has something to
catch.
"""

import typing

import pandas as pd

from morpheus.compositions import application_context
from morpheus.compositions import envelope
from morpheus.compositions import host_identity
from morpheus.compositions import stamping
from morpheus.compositions.corpus_pipeline import _collect
from morpheus.compositions.corpus_pipeline import (  # noqa: F401 pylint: disable=unused-import  # re-exported for callers
    build_pipeline_config, )
from morpheus.compositions.repository import repository_path
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
from morpheus.stages.telemetry.tc4_abp_score_stage import TC4AbpScoreStage
from morpheus.stages.telemetry.tc4_envelope_stage import TC4EnvelopeStage
from morpheus.stages.telemetry.tc4_flow_stage import TC4FlowStage
from morpheus.utils.binding_table import BindingTable
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900
BIN_SECONDS = 60
MODEL_INVENTORY = repository_path("models", "inventory", "inventory.json")
"""The repository's model inventory, which pins the ABP booster's bytes."""
CORPUS_HOURS = 2
CORPUS_SECONDS = CORPUS_HOURS * PERIOD_SECONDS
ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []
TELEMETRY_CLASS = "tc4"
OSI_LAYER = 4
ENTITY_COLUMNS = ["flow_id"]
SETTINGS = {
    "period_seconds": PERIOD_SECONDS,
    "lateness_seconds": LATENESS_SECONDS,
    "bin_seconds": BIN_SECONDS,
    "envelope_window_seconds": CORPUS_SECONDS * 4,
}
"""The settings that decide this corpus's output, digested into `config_hash` by `stamping.envelope_for`."""
RULES = ("R-D-L4-002", "R-D-L4-003", "R-B-L4-005", "R-B-L4-004")
"""The shipped rules that read this corpus's columns; their thresholds are folded into `pipeline_fingerprint`."""


def build_stages(config: Config, applications=None) -> list:
    """The TC-4 stages, in the order a deployment would compose them.

    The flow rollup first, because the envelope reads the magnitudes it produces; `community_id` beside them,
    computed from the five-tuple the records already carry; and the application the destination port belongs to, from
    the CMDB. `applications` is the store, defaulting to the estate's.
    """
    store = application_context.store(config) if applications is None else applications

    return [
        TC4FlowStage(config, bin_seconds=BIN_SECONDS),
        # R-B-L4-001's model over the thirteen features the rollup writes. Every flow carries the score; no search
        # reads it, because the booster scores every flow in this corpus between 0.89 and 0.93 and so would fire on
        # all of them at the upstream threshold (see the model inventory's README).
        TC4AbpScoreStage(config, inventory_path=MODEL_INVENTORY),
        TC4EnvelopeStage(config, window_seconds=CORPUS_SECONDS * 4),
        CommunityIdStage(config, dst_ip_column="dst_ip", dst_port_column="dst_port"),
        application_context.enrich_stage(config, store),
    ]


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True,
                 hosts: typing.Optional[tuple] = None,
                 sessions: typing.Optional[BindingTable] = None) -> pd.DataFrame:
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
    hosts : tuple, optional
        The lease and inventory tables the source addresses resolve through, from `host_identity.tables`. Defaults
        to the ones this corpus's own DHCP and inventory records close into; a corpus composing this pipeline over
        its own estate passes that estate's.
    sessions : `morpheus.utils.binding_table.BindingTable`, optional
        Which principal's session held each address, from `host_identity.session_table`. Given, each packet carries
        the `session_key` of the session that held its source address; this corpus has no sessions of its own.

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
    (leases, inventory) = host_identity.tables(config, batches, impose_order) if hosts is None else hosts

    pipe = ResumableLinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=batches[TELEMETRY_CLASS]))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    pipe.add_stage(CollectorSequenceStage(config))

    for stage in build_stages(config):
        pipe.add_stage(stage)

    # The source address to its interface and its machine, and the chain rooted on the machine where there is one.
    for stage in host_identity.resolver_stages(config, leases, inventory, fallback_anchor="flow_id"):
        pipe.add_stage(stage)

    if (sessions is not None):
        pipe.add_stage(host_identity.session_resolver(config, sessions))

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
