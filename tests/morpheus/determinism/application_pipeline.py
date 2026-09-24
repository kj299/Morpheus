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
The first two layer 7 sub-classes, DNS and HTTP, and their composed pipeline, for the determinism harness.

Layer 7 is not one telemetry class. The guide gives it four sub-classes keyed on four different entities, and this
file builds the two whose rules need nothing else to exist: `tc7_dns` and `tc7_http`, sharing `morpheus:score:l7` the
way layer 2's classes share `morpheus:score:l2`. The SaaS and endpoint sub-classes wait for their rules, which read
the data classification, role assignment and peer groups that only the TC-0 context store can supply.

Both classes seal on the **client**. For HTTP the guide gives no entity key and the source is what behaves. For DNS the
guide says `hostname`, which is ambiguous between the host that asked and the name it asked for; the host that asked is
what behaves, so it is what a chain at this layer roots on, and the registered domain is carried as the grouping key
R-B-L7-001 aggregates by.

**Each condition of both rules has a benign case in this corpus where it is the one condition that keeps the rule
quiet.** R-B-L7-001 is three conditions, and the guide warns that entropy alone flags every content delivery network;
a corpus demonstrating the rule only against ordinary browsing would pass with any one of the three removed. So:

- a **tunnel** clears all three -- random payload, long labels, a new name per chunk -- and fires;
- a **content delivery network** clears entropy and the distinct count and fails only on label length;
- a **long tenant hostname**, queried over and over, clears entropy and label length and fails only on the count;
- a **SaaS provider** with a hundred ordinary tenant subdomains clears the count and fails on both of the others;
- and ordinary browsing clears nothing.

R-D-L7-005 is two conditions, and the same holds: an **enumerator** clears both and fires; a **crawler** requests as
many distinct paths and is refused almost nothing; a **broken client** hammers one missing resource and is refused
constantly on a single path. A second enumerator has **found nothing at all** -- every response a 404, no 2xx -- which
is the case that exists to prove the rule is evaluated as `4xx > 0.7 * 2xx` rather than as a ratio that is undefined
for exactly the client the rule is for.

Every stateful stage is preceded by `TotalOrderStage`. Both stages are windowed and cumulative, so the permutation
check has something to catch.
"""

import base64
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
from morpheus.stages.telemetry.tc7_dns_stage import TC7DnsStage
from morpheus.stages.telemetry.tc7_http_stage import TC7HttpStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize

CORPUS_SEED = 20260923

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900

ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []

OSI_LAYER = 7
ENTITY_COLUMNS = ["src_ip"]

DNS_CLASS = "tc7_dns"
HTTP_CLASS = "tc7_http"
TELEMETRY_CLASSES = (DNS_CLASS, HTTP_CLASS)

# The rules' own thresholds, stated once so the corpus is built to clear them deliberately.
ENTROPY_THRESHOLD = 4.0
LABEL_LENGTH_THRESHOLD = 30
SUBDOMAIN_THRESHOLD = 100
PATH_THRESHOLD = 200
ERROR_RATIO_THRESHOLD = 0.7

# DNS clients and the domains they query.
TUNNELLER = "10.0.0.30"
CDN_CLIENT = "10.0.0.31"
TENANT_CLIENT = "10.0.0.32"
SAAS_CLIENT = "10.0.0.33"
BROWSER = "10.0.0.34"

TUNNEL_DOMAIN = "evil-tunnel.com"
CDN_DOMAIN = "cdn-assets.net"
TENANT_DOMAIN = "tenant-saas.com"
SAAS_DOMAIN = "saas-provider.com"

DNS_VOLUME = 130
"""Distinct names each high-volume DNS actor queries. Comfortably above R-B-L7-001's hundred, so the count is not what
separates them; what separates them is which of the other two conditions they fail.

Comfortably, because random names do not all clear 4.0 bits: about one in twelve of the CDN's twenty-nine-character
names falls just under. At a hundred and ten names that left the CDN's entropy-clearing names one above the count's
threshold, so its control -- that entropy and the count alone would have fired -- held by a single name."""

CDN_LABEL_LENGTH = 29
"""One short of the rule's thirty, and long enough that random characters clear 4.0 bits. The label-length condition
is therefore the only thing between this CDN and the rule."""

# HTTP clients.
ENUMERATOR = "10.0.0.40"
EMPTY_HANDED = "10.0.0.41"
CRAWLER = "10.0.0.42"
BROKEN_CLIENT = "10.0.0.43"
READER = "10.0.0.44"

WEB_SERVER = "10.0.1.80"

HTTP_VOLUME = 220
"""Distinct paths each high-volume HTTP actor requests. Above R-D-L7-005's two hundred."""

ENUMERATOR_HIT_EVERY = 10
"""One request in this many finds something. The enumerator's ratio is nine refusals to one success."""

CRAWLER_MISS_EVERY = 20
"""One request in this many is refused. The crawler's ratio is one refusal to nineteen successes."""

BROKEN_REQUESTS = 60


def at(hour: int, second: float = 0) -> int:
    """Event time for an offset into the corpus, in nanoseconds since the epoch."""
    return int((hour * PERIOD_SECONDS + second) * NS_PER_SECOND)


def _random_label(rng: random.Random, length: int, alphabet: str = "abcdefghijklmnopqrstuvwxyz0123456789") -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def _tunnel_name(rng: random.Random) -> str:
    """A chunk of exfiltrated data, base32-encoded the way tunnelling tools encode it -- DNS is case-insensitive,
    which is why they use base32 rather than base64 -- and split into two long labels."""
    payload = base64.b32encode(bytes(rng.randrange(256) for _ in range(45))).decode("ascii").rstrip("=").lower()

    return f"{payload[:40]}.{payload[40:]}.{TUNNEL_DOMAIN}"


def _envelope(collector: str, schema: str, seq: int, origin: str) -> dict:
    return {"collector_id": collector, "schema_version": schema, "origin_hash": origin, "collector_seq": seq}


def _build_dns(rng: random.Random) -> pd.DataFrame:
    rows: list[dict] = []

    def add(client, name, event_time_ns):
        rows.append({
            **_envelope("resolver-01", "tc7_dns.v1", len(rows), f"{client}?{name}"),
            "event_time": event_time_ns,
            "src_ip": client,
            "query_name": name,
            "query_type": "TXT" if name.endswith(TUNNEL_DOMAIN) else "A",
        })

    # The tunnel: a new, random, long-labelled name for every chunk, all under one domain, inside twenty minutes.
    for index in range(DNS_VOLUME):
        add(TUNNELLER, _tunnel_name(rng), at(0, 300 + index * 10))

    # The content delivery network: random asset hostnames, many of them, one label just short of thirty.
    for index in range(DNS_VOLUME):
        add(CDN_CLIENT, f"{_random_label(rng, CDN_LABEL_LENGTH)}.{CDN_DOMAIN}", at(0, 305 + index * 10))

    # The tenant: one long random hostname, asked for over and over. Every condition but the count.
    tenant = f"{_random_label(rng, 40)}.{_random_label(rng, 32)}.{TENANT_DOMAIN}"

    for index in range(30):
        add(TENANT_CLIENT, tenant, at(0, 310 + index * 30))

    # The SaaS provider: a hundred and ten ordinary tenant subdomains. Only the count.
    for index in range(DNS_VOLUME):
        add(SAAS_CLIENT, f"team{index}.{SAAS_DOMAIN}", at(0, 315 + index * 10))

    # Ordinary browsing.
    for (index, name) in enumerate([
            "www.example.com",
            "mail.globex-corporation.com",
            "login.microsoftonline.com",
            "api.stripe.com",
            "outlook.office365.com"
    ] * 4):
        add(BROWSER, name, at(1, 60 + index * 90))

    frame = pd.DataFrame(rows).sort_values("event_time", kind="mergesort").reset_index(drop=True)
    frame["collector_seq"] = range(len(frame))

    return frame


def _build_http(rng: random.Random) -> pd.DataFrame:
    rows: list[dict] = []

    def add(client, path, status, event_time_ns):
        rows.append({
            **_envelope("proxy-01", "tc7_http.v1", len(rows), f"{client}>{path}"),
            "event_time": event_time_ns,
            "src_ip": client,
            "dst_ip": WEB_SERVER,
            "http_method": "GET",
            "url_path": path,
            "status_code": status,
            "user_agent": "curl/8.5.0" if client in (ENUMERATOR, EMPTY_HANDED) else "Mozilla/5.0",
        })

    words = ["admin", "backup", "config", "api", "login", "old", "test", "dev", "private", "upload"]

    # The enumerator: a different path every second, nine in ten not there.
    for index in range(HTTP_VOLUME):
        status = 200 if index % ENUMERATOR_HIT_EVERY == 0 else 404
        add(ENUMERATOR, f"/{words[index % len(words)]}/{index}", status, at(0, 600 + index))

    # The enumerator that has found nothing: every response a refusal, so its 4xx:2xx ratio is undefined.
    for index in range(HTTP_VOLUME):
        add(EMPTY_HANDED, f"/{words[index % len(words)]}-{index}.bak", 404, at(0, 1200 + index))

    # The crawler: as many distinct paths, nearly all of them there.
    for index in range(HTTP_VOLUME):
        status = 404 if index % CRAWLER_MISS_EVERY == 0 else 200
        add(CRAWLER, f"/articles/{index}", status, at(0, 1800 + index))

    # The broken client: one missing resource, asked for again and again.
    for index in range(BROKEN_REQUESTS):
        add(BROKEN_CLIENT, "/static/app.js.map", 404, at(0, 2400 + index * 5))

    # Ordinary reading.
    for index in range(20):
        add(READER, f"/news/{rng.randrange(5)}", 200 if index % 7 else 304, at(1, 120 + index * 60))

    frame = pd.DataFrame(rows).sort_values("event_time", kind="mergesort").reset_index(drop=True)
    frame["collector_seq"] = range(len(frame))

    return frame


def build_corpus() -> dict[str, pd.DataFrame]:
    """The seeded corpus, one frame per layer 7 class."""
    rng = random.Random(CORPUS_SEED)

    return {DNS_CLASS: _build_dns(rng), HTTP_CLASS: _build_http(rng)}


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


def build_stages(config: Config, telemetry_class: str) -> list:
    """The stages for one layer 7 class."""
    if (telemetry_class == DNS_CLASS):
        return [TC7DnsStage(config)]

    if (telemetry_class == HTTP_CLASS):
        return [TC7HttpStage(config)]

    raise ValueError(f"no stages for {telemetry_class!r}")


def _run_class(config: Config, telemetry_class: str, dataframes: list[pd.DataFrame],
               impose_order: bool) -> pd.DataFrame:
    """Source, stamp, total order, the class's stage, envelope, window seal, sink."""
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    for stage in build_stages(config, telemetry_class):
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

    return _collect(sink)


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True) -> pd.DataFrame:
    """
    Run both layer 7 classes and return one canonicalized frame.

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
        Both classes' output, tagged with `telemetry_class`, keyed by `row_key`, canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    frames = []

    for name in TELEMETRY_CLASSES:
        frame = _run_class(config, name, batches[name], impose_order)
        frame["telemetry_class"] = name
        frame["row_key"] = frame["event_uid"]
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    return canonicalize(combined, key_columns=KEY_COLUMNS, ignore_columns=IGNORE_COLUMNS)


def render(result: pd.DataFrame) -> str:
    """The canonical CSV rendering the golden file holds."""
    return result.to_csv(index=False, lineterminator="\n")
