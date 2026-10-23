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
A host scored as a host: one feature vector per machine per hour, from three layers, against a learned model of its
peer group.

Every other score in this fork is a principal's. A host has columns at layers 3, 6 and 7 -- its fan-out step and
reach on its flows, its TLS client fingerprints on its handshakes, its novel process pairs on its EDR records -- and
since host identity those rows name the same machine in `host_key`. This corpus is the first place they are put
together and scored.

**The estate.** Ten workstations over three weeks: four in finance, four in engineering, one finance laptop that
joins in the scored week, and a kiosk the inventory gives no peer group. Each works eight hours a day; each working
hour it reaches a few of the estate's servers and now and then a site on the public internet, opens a TLS session or
two -- the browser's fingerprint, sometimes the updater's -- and starts its usual processes, with an occasional
software update that no host of its group has run. The first fourteen days are the training fortnight, the last
seven are scored.

**The pipeline.** Each class goes through its own layer's stages as its segment runs them: the flows through
`TC3CardinalityStage`, `TC3ReachStage` and `TC3BaselineStage`; the handshakes through `TC6FingerprintStage`; the
processes through `TC7EndpointStage`. The flows and handshakes resolve their source address through the DHCP leases
and the EDR inventory to `host_key`, as every layer does since host identity, and are keyed on it from there; the
EDR names its own machine. All three are enriched from the asset inventory, which is where the peer group comes
from. The three outputs then meet in a host segment: one stream, sealed by the hour on `host_key`, reduced by
`EntityFrameStage` to one row per host per hour, scored by `TC5ScoreStage` on `host_key`, stamped by
`DeterminismStampStage`, sealed again by the day, and given its trajectory by `TC5DriftStage`.

**The features** are the five the plan names: `dsts_per_src_step`, the hour's furthest fan-out past anything the host
reached in its own fortnight; `internal_dst_ratio`, the share of the hour's destinations inside the estate;
`byte_asymmetry`, the most lopsided flow it sent; `ja4_client_distinct`, how many client fingerprints the machine has
ever offered; and `endpoint_novel_pairs`, how many of its process pairs neither it nor its group had run in thirty
days.

**The models are learned, and shared by peer group.** A host with a fortnight of its own is still a thin training
set, and a host that joins tomorrow has none; a machine's peer group, read from the asset inventory, is what it is
expected to behave like. `examples/layer5_model/train_models.py --hosts` fits one autoencoder per peer group on its
hosts' fortnight, and one on every host's fortnight as the population, and commits them under
`examples/layer5_model/models/host_models.json`. The host manifest pins every inventoried host to its group's
model and declares the population model as the fallback, so the kiosk is scored against the estate and says so in
`model_fallback_used`, and the joiner is scored by its group's model from its first complete hour. Scoring the hosts
with `ReferenceScorer` first would have put a number on every host row and a claim behind none of them.

**The plants.** One engineering workstation drifts: from the scored week's second day it reaches a few more servers
and a few more public addresses each day, sends more than it receives, offers a new client fingerprint and runs new
process pairs, each day a little more than the last, and no hour of it is extreme. One finance workstation steps:
from the third scored day it behaves differently and keeps behaving that way, which raises its score once and does
not keep raising it. The daily trajectory is what tells the two apart.
"""

import functools
import os
import random
import typing

import envelope
import host_identity
import pandas as pd
import stamping
from corpus_pipeline import _collect
from corpus_pipeline import build_pipeline_config  # pylint: disable=unused-import  # re-exported for callers
from resumable import ResumableLinearPipeline

from morpheus.config import Config
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.collector_sequence_stage import CollectorSequenceStage
from morpheus.stages.lineage.determinism_stamp_stage import DeterminismStampStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.entity_frame_stage import EntityFrameStage
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.stages.telemetry.tc3_baseline_stage import TC3BaselineStage
from morpheus.stages.telemetry.tc3_cardinality_stage import TC3CardinalityStage
from morpheus.stages.telemetry.tc3_reach_stage import TC3ReachStage
from morpheus.stages.telemetry.tc5_drift_stage import TC5DriftStage
from morpheus.stages.telemetry.tc5_score_stage import TC5ScoreStage
from morpheus.stages.telemetry.tc6_fingerprint_stage import TC6FingerprintStage
from morpheus.stages.telemetry.tc7_endpoint_stage import TC7EndpointStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize
from morpheus.utils.dfencoder_scorer import DfencoderScorer
from morpheus.utils.dfencoder_scorer import load_models
from morpheus.utils.model_manifest import ModelManifest

CORPUS_SEED = 20261021

HOUR_S = 3600
DAY_S = 24 * HOUR_S
PERIOD_SECONDS = HOUR_S
LATENESS_SECONDS = 900

TRAINING_DAYS = 14
SCORED_DAYS = 7
CORPUS_DAYS = TRAINING_DAYS + SCORED_DAYS

EPOCH_S = 39 * DAY_S
"""1970-02-09T00:00:00Z, a Monday: the first day of the training fortnight.

Counted from the epoch, as every other corpus is, because the validation package feeds all of them to one search
head and the runner moves every timestamp by one offset: a corpus dated in another year would land fifty years from
the rest. Placed so the scored week ends after every other corpus's last event, which puts the host watchlist's
entries inside their thirty days when the expiry job runs.
"""

SCORES_FROM_NS = (EPOCH_S + TRAINING_DAYS * DAY_S) * NS_PER_SECOND
"""The end of the fortnight the models were fitted on, and the first instant they may score."""

WORK_HOURS = tuple(range(9, 17))
"""Every host's working hours, every day: eight hourly windows a day per host."""

ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []

FLOW_CLASS = "tc3"
HANDSHAKE_CLASS = "tc6"
PROCESS_CLASS = "tc7_endpoint"
FEATURE_CLASSES = (FLOW_CLASS, HANDSHAKE_CLASS, PROCESS_CLASS)
HOST_CLASS = "host_score"
"""The class of the host segment's rows, and the only class this corpus emits."""

FINANCE = "finance-workstations"
ENGINEERING = "engineering-workstations"
POPULATION = "all-hosts"
"""The key of the model fitted on every host's fortnight, which is the fallback."""

HOSTS = {
    "10.0.1.11": FINANCE,
    "10.0.1.12": FINANCE,
    "10.0.1.13": FINANCE,
    "10.0.1.14": FINANCE,
    "10.0.2.21": ENGINEERING,
    "10.0.2.22": ENGINEERING,
    "10.0.2.23": ENGINEERING,
    "10.0.2.24": ENGINEERING,
    "10.0.1.15": FINANCE,
    "10.0.3.31": None,
}
"""Each workstation's address, which its DHCP lease fixes for the corpus, and its peer group in the inventory."""

CLIENT_POOLS = ("10.0.1.", "10.0.2.", "10.0.3.")

DRIFTING = "10.0.2.24"
"""The engineering workstation whose behaviour moves a little further each day of the scored week."""
DRIFT_FROM_DAY = 1
"""The scored day the drift starts on; day 0 of the scored week is still the fortnight's habits."""

STEPPED = "10.0.1.14"
"""The finance workstation that changes once and stays changed."""
STEP_DAY = 2

JOINER = "10.0.1.15"
"""The finance laptop issued on the scored week's first day, with no fortnight of its own."""

KIOSK = "10.0.3.31"
"""The lobby kiosk: inventoried, and in no peer group."""

SERVERS = tuple(f"10.0.0.{index}" for index in range(1, 25))
"""The estate's servers. Static addresses, outside every DHCP pool."""

PUBLIC = tuple(f"93.184.216.{index}" for index in range(1, 61))
"""Addresses on the public internet."""

BROWSER = "t13d1516h2_8daaf6152771_b186095e22b6"
BROWSER_UPDATED = "t13d1516h2_8daaf6152771_02713d6af862"
UPDATER = "t13d0312h2_55b375c5d22e_cd85d2d88918"

EXPLORER = r"C:\Windows\explorer.exe"
SERVICES = r"C:\Windows\System32\services.exe"
SVCHOST = r"C:\Windows\System32\svchost.exe"
USUAL_PAIRS = ((EXPLORER, r"C:\Program Files\Browser\browser.exe"), (SERVICES, SVCHOST),
               (EXPLORER, r"C:\Program Files\Office\outlook.exe"))

REACH_MIN_DENOMINATOR = 3

ENDPOINT_WINDOW_DAYS = 30
ENDPOINT_WARMUP_DAYS = 7

HOST_FEATURES = {
    "dsts_per_src_step": ["dsts_per_src_step", "max"],
    "internal_dst_ratio": ["internal_dst_ratio", "last"],
    "byte_asymmetry": ["byte_asymmetry", "max"],
    "ja4_client_distinct": ["ja4_client_distinct", "max"],
    "endpoint_novel_pairs": ["endpoint_pair_novel", "count_true"],
}
"""Each host feature and the reduction of the class column it is built from, for `EntityFrameStage`."""

SCORED_FEATURES = list(HOST_FEATURES)

CARRY_COLUMNS = ["ctx_peer_group", "tenant_id"]

MANIFEST_WINDOW = 0
"""The window the host manifest is pinned for: the scored week, as the session manifest's is.

A frame row's `window_id` is its hour, which is not that span, so the score and stamp stages are given no window
column (`window_column=None`) and take every row as the manifest's own -- the single-window case they document.
"""

DRIFT_PREFIX = "day_"

SETTINGS = {
    "period_seconds": PERIOD_SECONDS,
    "lateness_seconds": LATENESS_SECONDS,
    "day_seconds": DAY_S,
    "training_days": TRAINING_DAYS,
    "scores_from_ns": SCORES_FROM_NS,
    "features": HOST_FEATURES,
    "reach_min_denominator": REACH_MIN_DENOMINATOR,
    "endpoint_window_days": ENDPOINT_WINDOW_DAYS,
    "endpoint_warmup_days": ENDPOINT_WARMUP_DAYS,
}
"""The settings that decide this corpus's output, digested into `config_hash` by `stamping.envelope_for`."""

RULES = ("R-P-L3-009", )
"""The shipped rule that reads these rows: the host drift trajectory, on the `morpheus:score:host` sourcetype."""

OSI_LAYER = 3
ENTITY_COLUMNS = ["host_key"]

MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..",
                           "..",
                           "..",
                           "examples",
                           "layer5_model",
                           "models",
                           "host_models.json")
"""The committed per-peer-group models, trained on the fortnight by `train_models.py --hosts`."""


def at(day: int, hour: int, minute: int = 0, second: int = 0) -> int:
    """An instant in the corpus, in epoch nanoseconds, `day` counted from the first day of the fortnight."""
    return (EPOCH_S + day * DAY_S + hour * HOUR_S + minute * 60 + second) * NS_PER_SECOND


def host_key(address: str) -> str:
    """The machine an address is leased to, as `host_key` names it."""
    return host_identity.hostname_for(address).lower()


def asset_versions() -> list:
    """The inventory: every workstation, with its peer group where it has one, recorded before the corpus starts."""
    return [
        make_version("asset",
                     host_key(address),
                     0,
                     None,
                     at(0, 0),
                     values={
                         "owner": "it-ops@example.com",
                         "peer_group": group,
                         "device_role": "kiosk" if group is None else "workstation",
                         "os_family": "windows",
                         "criticality": "low" if group is None else "medium",
                     }) for (address, group) in sorted(HOSTS.items())
    ]


def build_store() -> BitemporalStore:
    """The asset store every class is enriched from."""
    return BitemporalStore("asset", asset_versions())


def _envelope(rows: list, collector: str, schema: str, origin: str, when: int) -> dict:
    return {
        "collector_id": collector,
        "schema_version": schema,
        "origin_hash": f"{origin}@{when}",
        "collector_seq": len(rows),
        "event_time": when,
    }


def _flow(rows: list, source: str, destination: str, when: int, bytes_out: int, bytes_in: int):
    record = _envelope(rows, "netflow-01", "tc3.v1", f"{source}->{destination}", when)
    record.update({
        "src_ip": source,
        "src_port": 49152 + len(rows) % 16384,
        "dst_ip": destination,
        "dst_port": 443,
        "protocol": "tcp",
        "ip_ttl": 128,
        "bytes_out": bytes_out,
        "bytes_in": bytes_in,
        "bgp_as_dst": None if destination in SERVERS else "64500",
        "exporter_id": f"rtr-campus-{source.split('.')[2]}:256",
    })
    rows.append(record)


def _handshake(rows: list, source: str, destination: str, when: int, fingerprint: str):
    record = _envelope(rows, "tls-inspect-01", "tc6.v1", f"{source}->{destination}", when)
    record.update({
        "src_ip": source,
        "src_port": 49152 + len(rows) % 16384,
        "dst_ip": destination,
        "dst_port": 443,
        "sni": None,
        "protocol": "tcp",
        "tls_version": "TLSv1.3",
        "ja4_client": fingerprint,
        "certificate_issuer": "CN=Public Web CA, O=Example Trust",
        "certificate_fingerprint_sha256": f"{destination}|public",
        "cipher_suite": "TLS_AES_256_GCM_SHA384",
        "validation_result": "ok",
        "certificate_not_before": when - 10 * DAY_S * NS_PER_SECOND,
        "certificate_not_after": when + 355 * DAY_S * NS_PER_SECOND,
        "content_type_declared": None,
        "content_type_detected": None,
    })
    rows.append(record)


def _process(rows: list, address: str, parent: str, image: str, when: int):
    host = host_identity.hostname_for(address)
    record = _envelope(rows, "edr-01", "tc7_endpoint.v1", f"{host}>{image}", when)
    record.update({
        "hostname": host,
        "parent_image_path": parent,
        "image_path": image,
        "integrity_level": "Medium",
        "signature_status": "Valid",
    })
    rows.append(record)


def intensity(address: str, day: int) -> int:
    """How far a host's behaviour has moved from its fortnight on a corpus day: 0 for every ordinary hour."""
    scored = day - TRAINING_DAYS

    if (address == DRIFTING and scored >= DRIFT_FROM_DAY):
        return scored - DRIFT_FROM_DAY + 1

    if (address == STEPPED and scored >= STEP_DAY):
        return 4

    return 0


def active(address: str, day: int) -> bool:
    """Whether a host is in use on a corpus day: every host every day, except the joiner before it is issued."""
    return address != JOINER or day >= TRAINING_DAYS


def browser_update_day(index: int) -> int:
    """
    The corpus day a host's browser updates to its next release, which offers a client stack it never has.

    Staggered over the fortnight's second week, as an estate's updates are, so the training rows hold hosts that
    have offered two client stacks and hosts that have offered three. Were every host's count the same on every
    training row, the model would learn that the count never moves, and the first new stack anywhere would score in
    the billions -- which is what the first fit of these models did.
    """
    return TRAINING_DAYS // 2 + 1 + index % 5


def _hour(rng: random.Random,
          flows: list,
          handshakes: list,
          processes: list,
          address: str,
          day: int,
          hour: int,
          index: int):
    """One working hour of one host: its flows, its handshakes and its processes."""
    level = intensity(address, day)
    servers = rng.sample(SERVERS, rng.randint(3, 6) + 2 * level)
    public = rng.sample(PUBLIC, (1 if rng.random() < 0.4 else 0) + 2 * level)

    for (position, destination) in enumerate(servers + public):
        bytes_in = rng.randint(1200, 3200)
        bytes_out = rng.randint(500, 1500) * (1 + level)
        _flow(flows, address, destination, at(day, hour, 2 + position, index), bytes_out, bytes_in)

    browser = BROWSER_UPDATED if day >= browser_update_day(index) else BROWSER
    fingerprints = [browser] + ([UPDATER] if rng.random() < 0.15 else [])

    if (level > 0 and hour == WORK_HOURS[0]):
        # The drifting machine offers a client stack it has never offered, each day the change goes on; the stepped
        # one took up one new stack the day it changed, and keeps using it.
        stamp = day if address == DRIFTING else TRAINING_DAYS + STEP_DAY
        fingerprints.append(f"t13d{level:02d}11h2_{address.replace('.', '')[-6:]}a0b1c2_{stamp:012d}")

    for (position, fingerprint) in enumerate(fingerprints):
        _handshake(handshakes, address, rng.choice(PUBLIC), at(day, hour, 40 + position, index), fingerprint)

    for (position, (parent, image)) in enumerate(USUAL_PAIRS):
        _process(processes, address, parent, image, at(day, hour, 50 + position, index))

    novel = (1 if rng.random() < 0.05 else 0) + (level if hour in (WORK_HOURS[0], WORK_HOURS[4]) else 0)

    for count in range(novel):
        # A program no machine in its group has run: a software update, or something that is not one.
        image = rf"C:\Users\Public\{host_key(address)}-{day}-{hour}-{count}.exe"
        _process(processes, address, EXPLORER, image, at(day, hour, 56, index + 10 * count))


def build_corpus() -> dict[str, pd.DataFrame]:
    """Three weeks of ten workstations, split by collector, with the leases and inventory reports that name them."""
    rng = random.Random(CORPUS_SEED)
    flows: list = []
    handshakes: list = []
    processes: list = []

    for day in range(CORPUS_DAYS):
        for hour in WORK_HOURS:
            for (index, address) in enumerate(sorted(HOSTS)):
                if (active(address, day)):
                    _hour(rng, flows, handshakes, processes, address, day, hour, index)

    def frame(rows: list) -> pd.DataFrame:
        result = pd.DataFrame(rows).sort_values(["event_time", "collector_seq"], kind="mergesort")
        result = result.reset_index(drop=True)
        result["collector_seq"] = result.groupby("collector_id", sort=False).cumcount()

        return result

    workstations = host_identity.build(sorted(HOSTS), at(0, 0), at(CORPUS_DAYS, 0) - 1, pools=CLIENT_POOLS)

    return envelope.attach_all({
        FLOW_CLASS: frame(flows),
        HANDSHAKE_CLASS: frame(handshakes),
        PROCESS_CLASS: frame(processes),
        host_identity.LEASE_CLASS: workstations[host_identity.LEASE_CLASS],
        host_identity.INVENTORY_CLASS: workstations[host_identity.INVENTORY_CLASS],
    })


def class_stages(config: Config, telemetry_class: str, leases, inventory) -> list:
    """Each class's own layer stages, resolved to the machine and enriched from the inventory."""

    def enrich() -> TC0EnrichStage:
        return TC0EnrichStage(config, store=build_store(), entity_column="host_key", entity_normalization="hostname")

    if (telemetry_class == FLOW_CLASS):
        return [
            *host_identity.resolver_stages(config, leases, inventory),
            TC3CardinalityStage(config, window_seconds=PERIOD_SECONDS),
            # A workstation reaches three to six servers in an hour; the default floor of ten would leave its ratio
            # unpublished in every ordinary hour, and a feature that is only there when something is wrong is not one.
            TC3ReachStage(config, window_seconds=PERIOD_SECONDS, min_denominator=REACH_MIN_DENOMINATOR),
            TC3BaselineStage(config, bucket_seconds=PERIOD_SECONDS),
            enrich(),
        ]

    if (telemetry_class == HANDSHAKE_CLASS):
        # Keyed on the machine rather than the address: the ladder the stage's own documentation recommends.
        return [
            *host_identity.resolver_stages(config, leases, inventory),
            TC6FingerprintStage(config, key_columns=["host_key"]),
            enrich(),
        ]

    return [
        TC0EnrichStage(config, store=build_store(), entity_column="hostname", entity_normalization="hostname"),
        TC7EndpointStage(config, window_days=ENDPOINT_WINDOW_DAYS, warmup_days=ENDPOINT_WARMUP_DAYS),
    ]


def run_class(config: Config,
              telemetry_class: str,
              dataframes: list[pd.DataFrame],
              leases,
              inventory,
              impose_order: bool = True) -> pd.DataFrame:
    """Source, stamp, total order, sequence, the class's stages, sink: what the host segment reads from one class."""
    pipe = ResumableLinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    pipe.add_stage(CollectorSequenceStage(config))

    for stage in class_stages(config, telemetry_class, leases, inventory):
        pipe.add_stage(stage)

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    frame = _collect(sink, drop_empty=True)
    frame["telemetry_class"] = telemetry_class

    return frame


def merge_classes(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    The classes' rows as one stream, in the order the host segment's sealer reads them.

    A deployment's host segment reads a topic three segments write to, and the lateness horizon absorbs how they
    interleave. A replay has each class whole, so it is put in event order first; otherwise every handshake would
    arrive after three weeks of flows and be late.
    """
    present = [frame for frame in frames if len(frame) > 0]

    if (len(present) == 0):
        # Every class held back, which the clock quarantine does to a corpus whose collectors all sit beyond its
        # bound: the segment has nothing to frame, and an empty stream is the answer rather than an error.
        return pd.DataFrame()

    merged = pd.concat(present, ignore_index=True)
    merged = merged.sort_values(list(DEFAULT_ORDER_COLUMNS), kind="mergesort").reset_index(drop=True)

    return merged


def host_stages(config: Config, scorer, manifest: ModelManifest) -> list:
    """The host segment: hourly seal on the machine, frame, score, determinism stamp, envelope, daily seal, drift."""
    return [
        WindowSealStage(config,
                        period_seconds=PERIOD_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        entity_key_column="host_key"),
        EntityFrameStage(config,
                         features=HOST_FEATURES,
                         entity_column="host_key",
                         carry_columns=CARRY_COLUMNS,
                         collector_id="host-frame",
                         schema_version="host_frame.v1"),
        TC5ScoreStage(config,
                      scorer=scorer,
                      manifest=manifest,
                      feature_columns=SCORED_FEATURES,
                      entity_column="host_key",
                      window_column=None),
        DeterminismStampStage(config,
                              envelope=stamping.envelope_for(HOST_CLASS, SETTINGS, rules=RULES),
                              manifest=manifest,
                              entity_column="host_key",
                              window_column=None),
        EnvelopeStampStage(config, osi_layer=OSI_LAYER, entity_columns=ENTITY_COLUMNS),
        WindowSealStage(config,
                        period_seconds=DAY_S,
                        lateness_seconds=LATENESS_SECONDS,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        column_prefix=DRIFT_PREFIX),
        TC5DriftStage(config, entity_column="host_key", window_column=f"{DRIFT_PREFIX}window_id", aggregate="mean"),
    ]


def build_scoring(models: dict) -> tuple[DfencoderScorer, ModelManifest]:
    """
    The scorer and the host manifest for a set of peer-group models, built together.

    Parameters
    ----------
    models : dict
        Peer group (or `POPULATION`) to `(pinned version, fitted model)`.

    Every inventoried host whose peer group has a model is pinned to that model; the population model is the
    declared fallback, so a host with no group, or a group with no model, is scored against the estate and says so.
    Every version the manifest resolves is a learned model, so `DfencoderScorer` answers for all of them; the
    session corpus's `PinnedScorer` splices in its arithmetic fallback, and nothing here has one to splice.
    """
    if (POPULATION not in models):
        raise ValueError(f"the host models have no {POPULATION!r} model; it is the fallback the manifest declares")

    pinned = {
        host_key(address): models[group][0]
        for (address, group) in sorted(HOSTS.items()) if group is not None and group in models
    }
    scorer = DfencoderScorer(dict(models.values()), SCORED_FEATURES)
    manifest = ModelManifest(window_id=MANIFEST_WINDOW,
                             models=pinned,
                             fallback=models[POPULATION][0],
                             scores_from_ns=SCORES_FROM_NS)

    return (scorer, manifest)


@functools.lru_cache(maxsize=1)
def committed_scoring() -> tuple[DfencoderScorer, ModelManifest]:
    """The committed host models, read and checked against their recorded versions once per process."""
    return build_scoring(load_models(MODELS_PATH))


class _Unscored:
    """The scorer of a run whose manifest scores nothing: asked anything, it is a wiring mistake."""

    def score(self, model_version: str, features: list) -> list:
        raise AssertionError(f"asked to score {len(features)} rows against {model_version}; nothing is scored here")


def unscored() -> tuple[_Unscored, ModelManifest]:
    """
    A scorer and manifest that score nothing, for the run the training frames are taken from.

    The features are what the stages emit and do not depend on any score, so the frames exist before the models do.
    """
    return (_Unscored(),
            ModelManifest(window_id=MANIFEST_WINDOW,
                          models={},
                          fallback=f"{POPULATION}:untrained",
                          scores_from_ns=at(CORPUS_DAYS + 1, 0)))


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True,
                 scorer=None,
                 manifest: typing.Optional[ModelManifest] = None,
                 host_batches: int = 1) -> pd.DataFrame:
    """
    Each class through its own stages, then the host segment over all three, and its frame rows as one frame.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    corpus : dict
        The frames from `build_corpus`, possibly permuted.
    batches : dict, optional
        Per class, how the corpus is split across source frames. Defaults to one frame per class.
    impose_order : bool, default = True
        Place `TotalOrderStage` ahead of every class's stateful stages.
    scorer, manifest : optional
        What scores the host frame and what pins each host to a model. Default to the committed models; passed
        together or not at all.
    host_batches : int, default = 1
        How many source frames the merged stream is cut into for the host segment, by event order.

    Returns
    -------
    `pandas.DataFrame`
        The host rows, tagged `host_score`, keyed by `row_key`, canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    (batches, quarantined) = envelope.validate(config, batches, ID_COLUMNS)

    if ((scorer is None) != (manifest is None)):
        raise ValueError("scorer and manifest are passed together or not at all")

    if (scorer is None):
        (scorer, manifest) = committed_scoring()

    (leases, inventory) = host_identity.tables(config, batches, impose_order=impose_order)
    frames = [
        run_class(config, name, batches[name], leases, inventory, impose_order=impose_order) for name in FEATURE_CLASSES
        if len(batches.get(name, [])) > 0
    ]
    merged = merge_classes(frames)

    if (len(merged) == 0):
        # Nothing reached the segment. Its rows would be none, and the quarantine is what the run produced.
        result = pd.DataFrame(
            {column: pd.Series(dtype="object")
             for column in ("event_uid", "telemetry_class", "row_key")})

        return canonicalize(envelope.with_quarantine(result, quarantined),
                            key_columns=KEY_COLUMNS,
                            ignore_columns=IGNORE_COLUMNS)

    size = -(-len(merged) // max(1, host_batches))
    sources = [merged.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(merged), size)]

    pipe = ResumableLinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=sources))

    for stage in host_stages(config, scorer, manifest):
        pipe.add_stage(stage)

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    result = _collect(sink, drop_empty=True)
    result["telemetry_class"] = HOST_CLASS
    result["row_key"] = result["event_uid"]

    return canonicalize(envelope.with_quarantine(result, quarantined),
                        key_columns=KEY_COLUMNS,
                        ignore_columns=IGNORE_COLUMNS)


def training_frames(result: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    What each peer group's model is trained on: its hosts' hourly frames from the fortnight, as the five features;
    and the population's, every host's.

    Frames with a gap in any feature are dropped rather than imputed: the first day of a host's flows has no
    baseline to step past, and the first week of its processes no history to be novel against.
    """
    rows = result[(result["telemetry_class"] == HOST_CLASS) & (result["event_time"].astype("int64") < SCORES_FROM_NS)]
    rows = rows.dropna(subset=SCORED_FEATURES)
    frames = {POPULATION: rows[SCORED_FEATURES].astype("float64").reset_index(drop=True)}

    for group in sorted(set(rows["ctx_peer_group"].dropna())):
        members = rows[rows["ctx_peer_group"] == group]
        frames[group] = members[SCORED_FEATURES].astype("float64").reset_index(drop=True)

    return frames


def render(result: pd.DataFrame) -> str:
    """The golden file's rendering: every column, rows in canonical order."""
    return result.to_csv(index=False, lineterminator="\n")
