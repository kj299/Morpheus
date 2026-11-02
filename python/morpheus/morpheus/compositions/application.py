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
Three layer 7 sub-classes -- DNS, HTTP and the services they reach -- and their composed pipeline, for the
determinism harness.

Layer 7 is not one telemetry class. The guide gives it four sub-classes keyed on four different entities, and this
file builds the ones whose rules read nothing but the traffic and the estate's own records: `tc7_dns` and `tc7_http`,
sealed on the client, and `tc7_service`, sealed on the service a connection reached, all sharing `morpheus:score:l7`
the way layer 2's classes share `morpheus:score:l2`. The SaaS and endpoint sub-classes have corpora of their own.

**A week of each client's life comes before the day the rules are read on.** Both DNS and HTTP keep every client's
busiest window of each day, and a rule that reads "twice this client's busiest day" needs days to have been busy on.
The corpus therefore runs eight days: seven of ordinary history, at mid-morning each day, and the eighth on which the
actors act. A client with no history is the case the floor exists for, and several are left without one on purpose.

**Each condition of every rule has a benign case in this corpus where it is the one condition that keeps the rule
quiet.** R-B-L7-001 is four conditions -- entropy, label length, the count of qualifying subdomains, and twice the
client's busiest day -- and the guide warns that entropy alone flags every content delivery network; a corpus
demonstrating the rule only against ordinary browsing would pass with any one of the four removed. So:

- a **tunnel** clears all four -- random payload, long labels, a new name per chunk, from a workstation whose days
  have been ordinary -- and fires;
- a **content delivery network** clears entropy and the count and fails only on label length;
- a **long tenant hostname**, queried over and over, clears entropy and label length and fails only on the count;
- a **SaaS provider** with a hundred ordinary tenant subdomains clears the count and fails on both of the others;
- a **file reputation agent**, looking up every file hash it sees by DNS, clears entropy, label length and the count
  every day of its life, and is stopped by its own history alone;
- and ordinary browsing clears nothing.

The domain generation rule, R-B-L7-008, is three: many registered domains from one client, most of them answered
`NXDOMAIN`, and twice the client's busiest day. A **generator** clears all three; a **laptop with a stale search
suffix** is told no constantly about one domain; a **busy browser** reaches as many domains and is answered for all
of them; and a **link checker** walks as many dead links every day of its life.

R-D-L7-005 is three conditions, and the same holds: an **enumerator** clears all of them and fires; a **crawler**
requests as many distinct paths and is refused almost nothing; a **broken client** hammers one missing resource and is
refused constantly on a single path; and a **vulnerability scanner** walks the web server every morning and is stopped
by its own days. A second enumerator has **found nothing at all** -- every response a 404, no 2xx -- which is the case
that exists to prove the rule is evaluated as `4xx > 0.7 * 2xx` rather than as a ratio that is undefined for exactly
the client the rule is for; it is also new to the estate, and is held to the floor alone.

The service rule, R-B-L7-007, is four: a service reached this hour by at least twice the clients of its busiest day,
at least ten of them, at least one with client software the service has never seen, and a week of days to measure
against. **The payroll API**, called by three application servers every hour of its life and this hour by a dozen
workstations running a scripting library, clears all four. **The wiki** behind the same address on the same port --
which only its name tells apart from the API -- is reached by twice its usual readers, all in the browser it knows;
**the metrics collector** is reached by an agent that upgraded, and no more agents than usual; **the CI server** is
reached by a new tool from five machines, too few to matter; and **a service deployed this morning** has no days.

Every stateful stage is preceded by `TotalOrderStage`. Every stage is windowed and cumulative, so the permutation
check has something to catch.
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
from morpheus.stages.lineage.chain_anchor_stage import ChainAnchorStage
from morpheus.stages.lineage.collector_sequence_stage import CollectorSequenceStage
from morpheus.stages.lineage.determinism_stamp_stage import DeterminismStampStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc5_drift_stage import TC5DriftStage
from morpheus.stages.telemetry.tc7_dns_stage import TC7DnsStage
from morpheus.stages.telemetry.tc7_http_stage import TC7HttpStage
from morpheus.stages.telemetry.tc7_service_stage import SERVICE_KEY
from morpheus.stages.telemetry.tc7_service_stage import TC7ServiceStage
from morpheus.stages.telemetry.tc7_sid_score_stage import TC7SidScoreStage
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900
DAY_HOURS = 24
ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []
OSI_LAYER = 7
ENTITY_COLUMNS = ["src_ip"]
SERVICE_ENTITY_COLUMNS = [SERVICE_KEY]
DNS_CLASS = "tc7_dns"
HTTP_CLASS = "tc7_http"
BODY_COLUMN = "request_body"
"""The HTTP column the SID model reads: the request body a content-inspecting proxy records."""
MODEL_INVENTORY = repository_path("models", "inventory", "inventory.json")
"""The repository's model inventory, which pins the SID model's bytes."""
SERVICE_CLASS = "tc7_service"
TELEMETRY_CLASSES = (DNS_CLASS, HTTP_CLASS, SERVICE_CLASS)
DAY_SECONDS = DAY_HOURS * PERIOD_SECONDS
DRIFT_PREFIX = "day_"
SETTINGS = {"period_seconds": PERIOD_SECONDS, "lateness_seconds": LATENESS_SECONDS, "day_seconds": DAY_SECONDS}
"""The settings that decide this corpus's output, digested into `config_hash` by `stamping.envelope_for`."""
RULES = ("R-B-L7-001", "R-B-L7-003", "R-D-L7-005", "R-B-L7-007", "R-B-L7-008", "R-B-L4-004", "R-P-L7-009", "R-P-L7-010")
"""The shipped rules that read this corpus's columns; their thresholds are folded into `pipeline_fingerprint`."""
SUBDOMAIN_THRESHOLD = 100
PATH_THRESHOLD = 200


def build_stages(config: Config, telemetry_class: str, applications=None) -> list:
    """
    The stages for one layer 7 class.

    A request and a connection reach a server's port, and which application answered is read from the CMDB through
    `TC0EnrichStage`; a query reaches the resolver, which is not the application the client wanted, so DNS is not
    enriched. `applications` is the store, defaulting to the estate's.
    """
    if (telemetry_class == DNS_CLASS):
        return [TC7DnsStage(config)]

    store = application_context.store(config) if applications is None else applications

    if (telemetry_class == HTTP_CLASS):
        return [
            TC7HttpStage(config),
            TC7SidScoreStage(config, inventory_path=MODEL_INVENTORY, text_column=BODY_COLUMN),
            application_context.enrich_stage(config, store)
        ]

    if (telemetry_class == SERVICE_CLASS):
        return [TC7ServiceStage(config), application_context.enrich_stage(config, store)]

    raise ValueError(f"no stages for {telemetry_class!r}")


def _run_class(config: Config,
               telemetry_class: str,
               dataframes: list[pd.DataFrame],
               impose_order: bool,
               hosts: tuple = None,
               applications=None) -> pd.DataFrame:
    """Source, stamp, total order, the class's stage, determinism stamp, envelope, window seal, sink."""
    pipe = ResumableLinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    pipe.add_stage(CollectorSequenceStage(config))

    for stage in build_stages(config, telemetry_class, applications):
        pipe.add_stage(stage)

    # The source address to its interface and its machine, and the chain rooted on the machine where there is one.
    # A service is the subject of its own class, so its chain is rooted on the service rather than on the client.
    (leases, inventory) = hosts if hosts is not None else host_identity.tables(config, {})
    resolvers = host_identity.resolver_stages(config, leases, inventory)

    if (telemetry_class == SERVICE_CLASS):
        resolvers = resolvers[:-1] + [ChainAnchorStage(config, candidates=[SERVICE_KEY])]

    for stage in resolvers:
        pipe.add_stage(stage)

    entities = SERVICE_ENTITY_COLUMNS if telemetry_class == SERVICE_CLASS else ENTITY_COLUMNS

    pipe.add_stage(DeterminismStampStage(config, envelope=stamping.envelope_for(telemetry_class, SETTINGS,
                                                                                rules=RULES)))
    pipe.add_stage(EnvelopeStampStage(config, osi_layer=OSI_LAYER, entity_columns=entities))
    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=PERIOD_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        entity_key_column=DEFAULT_ANCHOR_COLUMN))

    if (telemetry_class == DNS_CLASS):
        # The daily trajectory R-P-L7-009 reads: a second seal over days behind the hourly one, and the drift stage
        # over each client's busiest domain of the day -- the most distinct subdomains under one registered domain
        # in an hour -- which is a running count within the hour and so is measured by its largest value.
        pipe.add_stage(
            WindowSealStage(config,
                            period_seconds=DAY_SECONDS,
                            lateness_seconds=LATENESS_SECONDS,
                            order_columns=list(DEFAULT_ORDER_COLUMNS),
                            column_prefix=DRIFT_PREFIX))
        pipe.add_stage(
            TC5DriftStage(config,
                          entity_column="src_ip",
                          score_column="dns_subdomains_per_domain",
                          window_column=f"{DRIFT_PREFIX}window_id",
                          aggregate="max",
                          ceiling=SUBDOMAIN_THRESHOLD))

    if (telemetry_class == HTTP_CLASS):
        # The daily trajectory R-P-L7-010 reads, built as the DNS one is: each client's busiest ten minutes of distinct
        # paths in the day, projected to R-D-L7-005's floor.
        pipe.add_stage(
            WindowSealStage(config,
                            period_seconds=DAY_SECONDS,
                            lateness_seconds=LATENESS_SECONDS,
                            order_columns=list(DEFAULT_ORDER_COLUMNS),
                            column_prefix=DRIFT_PREFIX))
        pipe.add_stage(
            TC5DriftStage(config,
                          entity_column="src_ip",
                          score_column="http_distinct_paths",
                          window_column=f"{DRIFT_PREFIX}window_id",
                          aggregate="max",
                          ceiling=PATH_THRESHOLD))

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    return _collect(sink)


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True) -> pd.DataFrame:
    """
    Run the three layer 7 classes and return one canonicalized frame.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    corpus : dict
        The frames from `build_corpus`, possibly permuted.
    batches : dict, optional
        Per class, how the corpus is split across source frames. Defaults to one frame per class.
    impose_order : bool, default = True
        Place `TotalOrderStage` ahead of the stateful stages.

    Returns
    -------
    `pandas.DataFrame`
        Every class's output, tagged with `telemetry_class`, keyed by `row_key`, canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    # Validated once per class, ahead of every stage: ingest_time stamped, the envelope checked, untrusted clocks
    # taken out of the stream and kept.
    (batches, quarantined) = envelope.validate(config, batches, ID_COLUMNS)
    hosts = host_identity.tables(config, batches, impose_order)
    applications = application_context.store(config)

    frames = []

    for name in TELEMETRY_CLASSES:
        frame = _run_class(config, name, batches[name], impose_order, hosts, applications)

        # A class every record of which was quarantined produced nothing, and is absent rather than an empty frame.
        if (len(frame) == 0):
            continue

        frame["telemetry_class"] = name
        frame["row_key"] = frame["event_uid"]
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if len(frames) > 0 else pd.DataFrame()

    return canonicalize(envelope.with_quarantine(combined, quarantined),
                        key_columns=KEY_COLUMNS,
                        ignore_columns=IGNORE_COLUMNS)


def render(result: pd.DataFrame) -> str:
    """The canonical CSV rendering the golden file holds."""
    return result.to_csv(index=False, lineterminator="\n")
