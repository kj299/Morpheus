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
A handshake-shaped layer 6 corpus, and the composed TC-6 pipeline, for the determinism harness.

The shape is what a TLS inspection point emits: one record per connection, carrying the client fingerprint, the
negotiated suite, the certificate the server presented and how the chain validated. Records arrive in event-time
order across a week: six days on which two hosts acquire client stacks, and a seventh whose three hours hold
everything the five per-event rules read.

**The sealing entity is the host, not the fingerprint, and that is a departure from Part 2.** The guide names
`ja4_client` and `certificate_fingerprint_sha256` as TC-6's entity key, and both are carried on every row here
because both are what an analyst pivots on. Neither is a thing that has a history. A JA4 fingerprint is a
property of a TLS stack, so thousands of unrelated hosts running the same browser build share one; rooting a
lineage chain on it would group them into a single entity, and `Behavior summary - per-layer scores` would
aggregate the estate's browser population rather than anything that behaves. The host is what acquires a new
stack, what gets intercepted and what gets downgraded, so the host is what the envelope seals on -- which is
also what makes R-C-002 expressible, since it correlates a fingerprint on a host with beaconing from that same
host one layer down.

Into the corpus are planted the things the five rules exist to see, each with the case beside it that must stay
quiet:

- a **managed endpoint acquiring a new TLS stack**, which is R-B-L6-001, against a host that alternates between
  two stacks it has always had -- the case a rule reading "the fingerprint changed" would flag every time -- and
  against a **personal device** with just as settled a history and just as new a stack, which the asset inventory
  has never heard of: the guide's argument for this rule is that there are few innocent ways for a managed endpoint
  to acquire a stack, and a phone updating its browser is one of the many for a device nobody manages;
- an **interception**, a destination presenting an issuer it never has before, which is R-D-L6-002, against a
  content delivery host legitimately behind four authorities, against a destination whose authority genuinely
  rotates and then settles, which must stop being reported once it is the destination's new normal, and against a
  **shared host** -- one address terminating two names whose certificates come from two authorities -- which is two
  destinations, because a destination is the name the client asked for at the address that answered, and whose
  shop is then intercepted: keyed on the address, the second name's authority would hide it;
- a **self-signed certificate on a connection leaving the estate**, which is R-D-L6-003, against the internal
  appliance whose self-signed certificate is ordinary;
- a **cipher downgrade**, a pair landing on a broken suite it has never negotiated, which is R-B-L6-004, against
  a pair whose own suite varies routinely and against a legacy appliance whose floor is low and stays there;
- a **file moved behind a declared image**, which is R-D-L6-005, against the re-encodings an estate produces
  in thousands;
- and a **host acquiring a new client stack every day for a week**, which is R-P-L6-006's trajectory -- the count
  of distinct fingerprints the host has presented rising on four consecutive days -- against a host that acquires
  two and stops. R-B-L6-001 reports each acquisition on its own; the trajectory is what says six of them are one
  story, and it is measured over daily windows a second `WindowSealStage` seals behind the hourly one, as the
  principals' and the host segment's trajectories are.

Three of the five rules need a reference and two do not, which the corpus makes visible: the self-signed and
content rules fire on a destination's first handshake, while the issuer and cipher rules stay quiet until their
entity has a history. "The rule is quiet" means different things in those two cases and the harness asserts both.

Each handshake's source is enriched from the TC-0 asset inventory, through the machine its address resolved to, and its
destination from the CMDB's application-to-server mapping on `dst_ip:dst_port`; and each carries its Community ID, so a
handshake joins the flow and the packets of the same connection by equality. Every handshake but the internal
appliance's names the server it asked for in `sni`.

Every stateful stage is preceded by `TotalOrderStage`, which is determinism control 8 as a stage. Three of the
four TC-6 stages are cumulative -- a novelty set, a mode and a running minimum are all functions of what came
before -- so the permutation check has something to catch.
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
from morpheus.stages.telemetry.tc5_drift_stage import TC5DriftStage
from morpheus.stages.telemetry.tc6_certificate_stage import TC6CertificateStage
from morpheus.stages.telemetry.tc6_cipher_stage import TC6CipherStage
from morpheus.stages.telemetry.tc6_content_stage import TC6ContentStage
from morpheus.stages.telemetry.tc6_fingerprint_stage import TC6FingerprintStage
from morpheus.utils.binding_table import BindingTable
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize
from morpheus.utils.entity_key import normalize_hostname

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900
DAY_SECONDS = 86400
DRIFT_PREFIX = "day_"
ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []
TELEMETRY_CLASS = "tc6"
OSI_LAYER = 6
ENTITY_COLUMNS = ["src_ip"]
MIN_SAMPLES = 5
"""Prior handshakes a reference needs. The stages' own default, repeated so the corpus can be built to clear it
deliberately rather than by accident."""
SETTINGS = {
    "period_seconds": PERIOD_SECONDS,
    "lateness_seconds": LATENESS_SECONDS,
    "min_samples": MIN_SAMPLES,
    "day_seconds": DAY_SECONDS,
}
"""The settings that decide this corpus's output, digested into `config_hash` by `stamping.envelope_for`."""
RULES = ("R-B-L6-001", "R-D-L6-002", "R-D-L6-003", "R-B-L6-004", "R-D-L6-005", "R-P-L6-006")
"""The shipped rules that read this corpus's columns; their thresholds are folded into `pipeline_fingerprint`."""
# Hosts.
MANAGED = "10.0.0.20"
VARIED = "10.0.0.21"
LEGACY_CLIENT = "10.0.0.22"
CREEPER = "10.0.0.24"
"""A managed workstation that presents one more client stack every day of the week: R-P-L6-006's trajectory."""
UPDATED = "10.0.0.25"
"""A managed workstation that acquires a stack on each of two days and then stops: a rise too short to watchlist."""
INVENTORY = {
    MANAGED: ("workstation", "medium", "alice@example.com"),
    VARIED: ("workstation", "medium", "bob@example.com"),
    LEGACY_CLIENT: ("server", "high", "it-ops@example.com"),
    CREEPER: ("workstation", "medium", "carol@example.com"),
    UPDATED: ("workstation", "low", "dave@example.com"),
}
"""The managed hosts, as the asset inventory has them: role, criticality and owner. The personal device is absent."""
# Issuers.
CORP_CA = "CN=Corp Issuing CA, O=Example, C=GB"


def asset_versions() -> list:
    """The inventory's record of every managed host, recorded before the corpus begins, keyed by host name."""
    return [
        make_version("asset",
                     normalize_hostname(host_identity.hostname_for(address)),
                     0,
                     None,
                     0,
                     values={
                         "device_role": role, "criticality": criticality, "owner": owner
                     }) for (address, (role, criticality, owner)) in sorted(INVENTORY.items())
    ]


def build_store() -> BitemporalStore:
    """The asset store the source enrichment reads."""
    return BitemporalStore("asset", asset_versions())


def build_stages(config: Config,
                 store: typing.Optional[BitemporalStore] = None,
                 hosts: typing.Optional[tuple] = None,
                 applications: typing.Optional[BitemporalStore] = None) -> list:
    """The TC-6 stages, in the order a deployment would compose them, then the source's machine and its context, and
    the application the destination's port belongs to.

    The order of the four is not load-bearing -- none reads a column another writes -- so they are composed in the
    order the rules are numbered, which is the order a reader will look for them in. The source address is then
    resolved to its machine through the leases and the EDR inventory in `hosts`, and the machine is what the asset
    inventory is looked up on: an address names a host only for as long as its lease does, and a host name for as
    long as the host does. `applications` is the CMDB's store, defaulting to the estate's.
    """
    (leases, inventory) = host_identity.tables(config, {}) if hosts is None else hosts
    applications = application_context.store(config) if applications is None else applications

    return [
        TC6FingerprintStage(config),
        TC6CertificateStage(config, min_samples=MIN_SAMPLES),
        TC6CipherStage(config, min_samples=MIN_SAMPLES),
        TC6ContentStage(config),
        *host_identity.resolver_stages(config, leases, inventory, fallback_anchor="src_ip"),
        TC0EnrichStage(config,
                       store=build_store() if store is None else store,
                       entity_column=host_identity.HOST_KEY_COLUMN,
                       entity_normalization="hostname"),
        application_context.enrich_stage(config, applications),
    ]


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True,
                 store: typing.Optional[BitemporalStore] = None,
                 hosts: typing.Optional[tuple] = None,
                 sessions: typing.Optional[BindingTable] = None) -> pd.DataFrame:
    """
    Run the TC-6 class through its composed pipeline and return one canonicalized frame.

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
    store : `morpheus.utils.bitemporal.BitemporalStore`, optional
        The asset inventory the sources are enriched from. Defaults to this corpus's; another corpus running its
        handshakes through this pipeline supplies its own.
    hosts : tuple, optional
        The lease and inventory tables the source addresses resolve through, from `host_identity.tables`. Defaults
        to the ones this corpus's own DHCP and inventory records close into; a corpus composing this pipeline over
        its own estate passes that estate's.
    sessions : `morpheus.utils.binding_table.BindingTable`, optional
        Which principal's session held each address, from `host_identity.session_table`. Given, each handshake
        carries the `session_key` of the session that held its source address; this corpus has no sessions of its
        own.

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

    pipe.add_stage(CommunityIdStage(config, dst_ip_column="dst_ip", dst_port_column="dst_port"))

    # The handshake stages, the source address to its interface and its machine with the chain rooted on the
    # machine where there is one, and the machine's context.
    for stage in build_stages(config, store, (leases, inventory)):
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
    # The daily trajectory R-P-L6-006 reads: a second seal over days, its columns prefixed so the hourly windows
    # keep their identity, and the drift stage over each host's count of distinct client stacks. The count is
    # cumulative, so a day is measured by its largest value, as the SaaS corpus's weekly breadth is.
    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=DAY_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        column_prefix=DRIFT_PREFIX))
    pipe.add_stage(
        TC5DriftStage(config,
                      entity_column="src_ip",
                      score_column="ja4_client_distinct",
                      window_column=f"{DRIFT_PREFIX}window_id",
                      aggregate="max"))

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
