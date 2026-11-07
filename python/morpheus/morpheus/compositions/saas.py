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
The layer 7 SaaS sub-class, `tc7_saas`, and its composed pipeline, for the determinism harness.

The two SaaS rules are the first in this fork that read the TC-0 context store, so the pipeline enriches every
operation twice before measuring it -- the principal's profile and group memberships, and the target object's data
classification -- both as known at the operation's own time. It then seals hourly windows as the other layer 7
classes do, and weekly windows behind them, Monday to Monday UTC, over which
{py:class}`~morpheus.stages.telemetry.tc5_drift_stage.TC5DriftStage` measures whether a principal's breadth is
rising, exactly as it measures a layer 5 score over days.

**Each condition of both rules has a benign case in this corpus where it is the one condition that keeps the rule
quiet.** R-B-L7-002 is a multiple of the principal's own history for the operation, and the history must exist:

- an **exfiltrator** exports eighty times their own 99th percentile from a restricted object, and fires -- while
  running queries of two to three thousand records all month, which a baseline pooling operations would have let
  excuse the export;
- the **same multiple from a public object** fires too, at a quarter of the severity -- the classification is a
  weight on the notable, not a bar the export has to clear;
- the same multiple from an **object the inventory has never heard of** fires at the default severity, flagged;
- a principal whose **large exports are routine** exports just as much and is well inside their own baseline;
- a principal **exporting just under five times** their baseline stays quiet;
- a **newcomer** with twenty exports behind them has no baseline, because by nearest rank a 99th percentile of
  twenty values is the largest of them;
- and an export that **failed** read nothing and is not measured.

R-P-L7-006 is breadth rising for four consecutive weeks with the role unchanged:

- a **creeper** reaches one more object type each week for four weeks, groups untouched, and is watchlisted;
- a principal with the **same rise and a role change** in the middle of it is quiet -- new duties explain new
  objects;
- a **short rise** of two weeks is quiet;
- a principal who has **always reached everything** has breadth and no rise;
- and a principal whose **role change was recorded late** is watchlisted, because on every week of the rise the
  store said the role was unchanged. That is the event-time knowledge the enrichment defaults to, and the harness
  asserts what the latest knowledge would have said instead.

**A second tenant shares the audit collector.** Its sync service exports the same small dataset twice a day all
month, and for the last two days exports it twice as often. Neither rule has anything to say about that -- the
records and the object type never change -- and it is not a principal's behaviour at all: it is the platform's. The
health pass over this corpus is where it shows, as the one step above its own history among every collector, class
and tenant, which is what the 'Platform baseline step' search reads.
"""

import typing

import pandas as pd

from morpheus.compositions import envelope
from morpheus.compositions import host_identity
from morpheus.compositions import stamping
from morpheus.compositions.corpus_pipeline import _collect
from morpheus.compositions.corpus_pipeline import build_pipeline_config  # noqa: F401 pylint: disable=unused-import
from morpheus.compositions.corpus_pipeline import render  # noqa: F401 pylint: disable=unused-import
from morpheus.compositions.estate_context import EstateContext
from morpheus.compositions.resumable import ResumableLinearPipeline
from morpheus.config import Config
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.collector_sequence_stage import CollectorSequenceStage
from morpheus.stages.lineage.determinism_stamp_stage import DeterminismStampStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.stages.telemetry.tc5_drift_stage import TC5DriftStage
from morpheus.stages.telemetry.tc7_saas_stage import DEFAULT_WEEK_EPOCH
from morpheus.stages.telemetry.tc7_saas_stage import WEEK_SECONDS
from morpheus.stages.telemetry.tc7_saas_stage import TC7SaasStage
from morpheus.utils.binding_table import BindingTable
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900
ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []
OSI_LAYER = 7
ENTITY_COLUMNS = ["user_principal"]
SAAS_CLASS = "tc7_saas"
SETTINGS = {"period_seconds": PERIOD_SECONDS, "lateness_seconds": LATENESS_SECONDS, "week_seconds": WEEK_SECONDS}
"""The settings that decide this corpus's output, with the knowledge mode `run_pipeline` is given added per run."""
RULES = ("R-B-L7-002", "R-P-L7-006")
"""The shipped rules that read this corpus's columns; their thresholds are folded into `pipeline_fingerprint`."""
WEEK_PREFIX = "week_"
"""Prefix of the weekly seal's columns, so they sit beside the hourly seal's rather than over them."""


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True,
                 knowledge: str = "event",
                 sessions: typing.Optional[BindingTable] = None,
                 *,
                 context: EstateContext = None) -> pd.DataFrame:
    """
    Source, stamp, total order, enrich twice, measure, envelope, seal hourly, seal weekly, trajectory, sink.

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
    knowledge : str, default = "event"
        What the enrichment knows: the context as recorded by each operation's time, or everything.
    sessions : `morpheus.utils.binding_table.BindingTable`, optional
        Which principal's session held each address, from `host_identity.session_table`. Given, each operation
        carries the `session_key` of the session that held the address it came from, its `source_ip`; this corpus
        records no address and has no sessions.
    context : `morpheus.compositions.estate_context.EstateContext`
        The estate's context; the identity directory and the asset inventory are both required.

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

    (identity, asset) = (context or EstateContext()).require("the SaaS composition", "identity", "asset")

    pipe = ResumableLinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=batches[SAAS_CLASS]))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    pipe.add_stage(CollectorSequenceStage(config))

    pipe.add_stage(TC0EnrichStage(config, store=identity, entity_column="user_principal", knowledge=knowledge))
    pipe.add_stage(
        TC0EnrichStage(config, store=asset, entity_column="target_object", knowledge=knowledge, prefix="ctx_object_"))
    pipe.add_stage(TC7SaasStage(config))

    if (sessions is not None):
        pipe.add_stage(host_identity.session_resolver(config, sessions, address_column="source_ip"))

    settings = {**SETTINGS, "knowledge": knowledge}
    pipe.add_stage(DeterminismStampStage(config, envelope=stamping.envelope_for(SAAS_CLASS, settings, rules=RULES)))
    pipe.add_stage(EnvelopeStampStage(config, osi_layer=OSI_LAYER, entity_columns=ENTITY_COLUMNS))
    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=PERIOD_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        entity_key_column="entity_key"))
    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=WEEK_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        epoch=DEFAULT_WEEK_EPOCH,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        column_prefix=WEEK_PREFIX))
    pipe.add_stage(
        TC5DriftStage(config,
                      score_column="saas_object_types_in_week",
                      window_column=f"{WEEK_PREFIX}window_id",
                      aggregate="max"))

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    frame = _collect(sink)
    frame["telemetry_class"] = SAAS_CLASS
    frame["row_key"] = frame["event_uid"]

    return canonicalize(envelope.with_quarantine(frame, quarantined),
                        key_columns=KEY_COLUMNS,
                        ignore_columns=IGNORE_COLUMNS)
