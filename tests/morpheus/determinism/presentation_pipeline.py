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
A handshake-shaped layer 6 corpus, and the composed TC-6 pipeline, for the determinism harness.

The shape is what a TLS inspection point emits: one record per connection, carrying the client fingerprint, the
negotiated suite, the certificate the server presented and how the chain validated. Records arrive in event-time
order across three hours.

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
  content delivery host legitimately behind four authorities and against a destination whose authority genuinely
  rotates and then settles, which must stop being reported once it is the destination's new normal;
- a **self-signed certificate on a connection leaving the estate**, which is R-D-L6-003, against the internal
  appliance whose self-signed certificate is ordinary;
- a **cipher downgrade**, a pair landing on a broken suite it has never negotiated, which is R-B-L6-004, against
  a pair whose own suite varies routinely and against a legacy appliance whose floor is low and stays there;
- and a **file moved behind a declared image**, which is R-D-L6-005, against the re-encodings an estate produces
  in thousands.

Three of the five rules need a reference and two do not, which the corpus makes visible: the self-signed and
content rules fire on a destination's first handshake, while the issuer and cipher rules stay quiet until their
entity has a history. "The rule is quiet" means different things in those two cases and the harness asserts both.

Each handshake's source is enriched from the TC-0 asset inventory, keyed on the address until hosts have an identity
of their own across layers, and carries its Community ID, so a handshake joins the flow and the packets of the same
connection by equality.

Every stateful stage is preceded by `TotalOrderStage`, which is determinism control 8 as a stage. Three of the
four TC-6 stages are cumulative -- a novelty set, a mode and a running minimum are all functions of what came
before -- so the permutation check has something to catch.
"""

import hashlib
import typing

import pandas as pd

import stamping
from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.community_id_stage import CommunityIdStage
from morpheus.stages.lineage.determinism_stamp_stage import DeterminismStampStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.stages.telemetry.tc6_certificate_stage import TC6CertificateStage
from morpheus.stages.telemetry.tc6_cipher_stage import TC6CipherStage
from morpheus.stages.telemetry.tc6_content_stage import TC6ContentStage
from morpheus.stages.telemetry.tc6_fingerprint_stage import TC6FingerprintStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900
CORPUS_HOURS = 3
CORPUS_SECONDS = CORPUS_HOURS * PERIOD_SECONDS

ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []

TELEMETRY_CLASS = "tc6"
OSI_LAYER = 6
ENTITY_COLUMNS = ["src_ip"]

COLLECTOR = "tls-inspect-01"
SCHEMA_VERSION = "tc6.v1"

MIN_SAMPLES = 5
"""Prior handshakes a reference needs. The stages' own default, repeated so the corpus can be built to clear it
deliberately rather than by accident."""
SETTINGS = {"period_seconds": PERIOD_SECONDS, "lateness_seconds": LATENESS_SECONDS, "min_samples": MIN_SAMPLES}
"""The settings that decide this corpus's output, digested into `config_hash` by `stamping.envelope_for`."""
RULES = ("R-B-L6-001", "R-D-L6-002", "R-D-L6-003", "R-B-L6-004", "R-D-L6-005")
"""The shipped rules that read this corpus's columns; their thresholds are folded into `pipeline_fingerprint`."""

# Hosts.
MANAGED = "10.0.0.20"
VARIED = "10.0.0.21"
LEGACY_CLIENT = "10.0.0.22"
PERSONAL_DEVICE = "10.0.0.23"
"""A phone on the corporate wireless. Settled, and then a new stack; and in no inventory."""

INVENTORY = {
    MANAGED: ("workstation", "medium", "alice@example.com"),
    VARIED: ("workstation", "medium", "bob@example.com"),
    LEGACY_CLIENT: ("server", "high", "it-ops@example.com"),
}
"""The managed hosts, as the asset inventory has them: role, criticality and owner. The personal device is absent."""

# Destinations. Outside the ranges the RFCs reserve for documentation, which `parsers/ip.py` classifies as
# private -- the layer 3 corpus was built on those once and reported a host on the public internet as one that
# never left the estate.
PORTAL = "93.184.216.34"
CDN = "93.184.216.35"
ROTATED = "93.184.216.36"
ATTACKER = "93.184.216.37"
APPLIANCE = "10.0.1.50"

# Fingerprints. The shape JA4 produces, which is what a reader will be comparing against their own feed.
CHROME = "t13d1516h2_8daaf6152771_b186095e22b6"
UPDATER = "t13d0312h2_55b375c5d22e_cd85d2d88918"
NEW_STACK = "t13d3104h1_9b1e5a0f4c22_7e0fd1a83b55"

# Issuers.
CORP_CA = "CN=Corp Issuing CA, O=Example, C=GB"
PUBLIC_CA = "CN=Public Trust CA G2, O=Trust Services, C=US"
ROTATED_CA = "CN=Public Trust CA G3, O=Trust Services, C=US"
PROXY_CA = "CN=Inspection Proxy, O=Unknown"
SELF_CA = "CN=appliance.local, O=self"

CDN_ISSUERS = (
    "CN=Edge CA A, O=Delivery",
    "CN=Edge CA B, O=Delivery",
    "CN=Edge CA C, O=Delivery",
    "CN=Edge CA D, O=Delivery",
)
"""Four authorities behind one content delivery host, which is ordinary and must not read as interception."""

# Cipher suites, by the tier they sit in.
MODERN_SUITE = "TLS_AES_256_GCM_SHA384"
MODERN_ALT = "TLS_AES_128_GCM_SHA256"
FORWARD_SUITE = "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
LEGACY_SUITE = "TLS_RSA_WITH_AES_128_CBC_SHA"
BROKEN_SUITE = "TLS_RSA_WITH_3DES_EDE_CBC_SHA"

VALID = "ok"
SELF_SIGNED = "self-signed"

PNG = "image/png"
JPEG = "image/jpeg"
ZIP = "application/zip"

DAY_NS = 86400 * NS_PER_SECOND
CORP_VALIDITY_DAYS = 365
SHORT_VALIDITY_DAYS = 3
"""The attacker certificate's lifetime. Short lifetimes are the guide's named signal and this one is far below
anything an estate issues, so a search can put a threshold anywhere sensible and still separate them."""


def at(hour: int, second: int = 0) -> int:
    """Event time for an offset into the corpus, in nanoseconds since the epoch."""
    return (hour * PERIOD_SECONDS + second) * NS_PER_SECOND


def asset_versions() -> list:
    """The inventory's record of every managed host, recorded before the corpus begins."""
    return [
        make_version("asset",
                     address,
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


def _fingerprint_of(destination: str, issuer: str) -> str:
    """A stable stand-in for a certificate's SHA-256 fingerprint.

    `hash()` is salted by `PYTHONHASHSEED` for strings, so a corpus built on it would differ between
    interpreters -- which is precisely what the cross-restart check exists to catch, and there is no reason to
    make it catch something the corpus can simply not do.
    """
    digest = hashlib.sha256(f"{destination}|{issuer}".encode("utf-8")).hexdigest()

    return f"sha256:{digest[:32]}"


def build_corpus() -> dict[str, pd.DataFrame]:
    """The seeded handshake corpus, one frame for the single TC-6 class."""
    rows: list[dict] = []

    def add(src,
            dst,
            fingerprint,
            issuer,
            suite,
            event_time_ns,
            validation=VALID,
            validity_days=CORP_VALIDITY_DAYS,
            declared=None,
            detected=None):
        rows.append({
            "collector_id": COLLECTOR,
            "schema_version": SCHEMA_VERSION,
            "origin_hash": f"{src}->{dst}",
            "collector_seq": len(rows),
            "event_time": event_time_ns,
            "src_ip": src,
            "src_port": 49152 + len(rows),
            "dst_ip": dst,
            "dst_port": 443,
            "protocol": "tcp",
            "tls_version": "TLSv1.3" if suite.startswith("TLS_AES") or suite.startswith("TLS_CHACHA") else "TLSv1.2",
            "ja4_client": fingerprint,
            "certificate_issuer": issuer,
            "certificate_fingerprint_sha256": _fingerprint_of(dst, issuer),
            "cipher_suite": suite,
            "validation_result": validation,
            "certificate_not_before": event_time_ns - 10 * DAY_NS,
            "certificate_not_after": event_time_ns - 10 * DAY_NS + validity_days * DAY_NS,
            "content_type_declared": declared,
            "content_type_detected": detected,
        })

    # The managed endpoint: one stack for two hours, then a second one it has never presented. Its destination
    # presents one authority throughout, so the fingerprint is the only thing that changes about it.
    for index in range(24):
        add(MANAGED, PORTAL, CHROME, CORP_CA, MODERN_SUITE, at(0, 60 + index * 120), declared=PNG, detected=JPEG)

    add(MANAGED, PORTAL, NEW_STACK, CORP_CA, MODERN_SUITE, at(1, 300), declared=PNG, detected=JPEG)

    # The personal device: the same settled history and the same departure from it, on a device no inventory holds.
    for index in range(24):
        add(PERSONAL_DEVICE, PORTAL, CHROME, CORP_CA, MODERN_SUITE, at(0, 75 + index * 120))

    add(PERSONAL_DEVICE, PORTAL, NEW_STACK, CORP_CA, MODERN_SUITE, at(1, 420))

    # The varied host: two stacks it has always had, alternating. A rule reading `changed` would fire on every
    # one of these, which is why the rule reads `first_seen`.
    for index in range(24):
        stack = CHROME if index % 2 == 0 else UPDATER
        suite = MODERN_ALT if index % 3 else FORWARD_SUITE
        issuer = CDN_ISSUERS[index % len(CDN_ISSUERS)]

        add(VARIED, CDN, stack, issuer, suite, at(0, 90 + index * 120), declared=PNG, detected=JPEG)

    # The interception: a destination that has presented one authority all corpus, then presents a proxy's.
    for index in range(12):
        add(VARIED, PORTAL, CHROME, CORP_CA, MODERN_SUITE, at(0, 30 + index * 240))

    add(VARIED, PORTAL, CHROME, PROXY_CA, MODERN_SUITE, at(2, 600))

    # The rotation: an authority migration that is a change and then a fact, which must stop being reported.
    for index in range(8):
        add(MANAGED, ROTATED, CHROME, PUBLIC_CA, MODERN_SUITE, at(0, 45 + index * 300))

    for index in range(16):
        add(MANAGED, ROTATED, CHROME, ROTATED_CA, MODERN_SUITE, at(1, 45 + index * 180))

    # The self-signed pair: one certificate leaving the estate, one staying inside it. Both fire the same flag
    # and only one of them is a finding, which is the whole of R-D-L6-003.
    add(MANAGED,
        ATTACKER,
        CHROME,
        PROXY_CA,
        FORWARD_SUITE,
        at(2, 900),
        validation=SELF_SIGNED,
        validity_days=SHORT_VALIDITY_DAYS)

    for index in range(6):
        add(LEGACY_CLIENT, APPLIANCE, UPDATER, SELF_CA, LEGACY_SUITE, at(0, 120 + index * 400), validation=SELF_SIGNED)

    # The downgrade: a pair that has only ever negotiated modern suites landing on a broken one.
    for index in range(10):
        add(LEGACY_CLIENT, PORTAL, UPDATER, CORP_CA, MODERN_SUITE, at(1, 30 + index * 200))

    add(LEGACY_CLIENT, PORTAL, UPDATER, CORP_CA, BROKEN_SUITE, at(2, 1200))

    # The file behind a declared image, and the re-encodings it has to be told apart from.
    add(MANAGED,
        ATTACKER,
        CHROME,
        PROXY_CA,
        FORWARD_SUITE,
        at(2, 1500),
        validation=SELF_SIGNED,
        validity_days=SHORT_VALIDITY_DAYS,
        declared=PNG,
        detected=ZIP)

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
    from host_frame import to_host_frame

    frames = []

    for message in sink.get_messages():
        meta = message.payload() if hasattr(message, "payload") else message
        frame = meta.copy_dataframe()
        # Integer columns are carried as nullable integers in both modes before anything is joined, so the
        # concatenation cannot widen them to float in one mode and not the other; see `host_frame`.
        frames.append(to_host_frame(frame))

    if (len(frames) == 0):
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def build_stages(config: Config, store: typing.Optional[BitemporalStore] = None) -> list:
    """The TC-6 stages, in the order a deployment would compose them, and the source's context after them.

    The order is not load-bearing -- none of the four reads a column another writes -- so they are composed in
    the order the rules are numbered, which is the order a reader will look for them in.
    """
    return [
        TC6FingerprintStage(config),
        TC6CertificateStage(config, min_samples=MIN_SAMPLES),
        TC6CipherStage(config, min_samples=MIN_SAMPLES),
        TC6ContentStage(config),
        TC0EnrichStage(config, store=build_store() if store is None else store, entity_column="src_ip"),
    ]


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True,
                 store: typing.Optional[BitemporalStore] = None) -> pd.DataFrame:
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

    pipe.add_stage(CommunityIdStage(config, dst_ip_column="dst_ip", dst_port_column="dst_port"))

    for stage in build_stages(config, store):
        pipe.add_stage(stage)

    pipe.add_stage(DeterminismStampStage(config, envelope=stamping.envelope_for(TELEMETRY_CLASS, SETTINGS,
                                                                                rules=RULES)))
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
