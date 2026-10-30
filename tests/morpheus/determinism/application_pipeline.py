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

import base64
import random
import typing

import application_context
import envelope
import host_identity
import pandas as pd
import stamping
from corpus_pipeline import _collect
from corpus_pipeline import build_pipeline_config  # pylint: disable=unused-import  # re-exported for callers
from resumable import ResumableLinearPipeline

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
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize

CORPUS_SEED = 20260923

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
SERVICE_CLASS = "tc7_service"
TELEMETRY_CLASSES = (DNS_CLASS, HTTP_CLASS, SERVICE_CLASS)

DAY_SECONDS = DAY_HOURS * PERIOD_SECONDS
DRIFT_PREFIX = "day_"
SETTINGS = {"period_seconds": PERIOD_SECONDS, "lateness_seconds": LATENESS_SECONDS, "day_seconds": DAY_SECONDS}
"""The settings that decide this corpus's output, digested into `config_hash` by `stamping.envelope_for`."""
RULES = ("R-B-L7-001", "R-D-L7-005", "R-B-L7-007", "R-B-L7-008", "R-B-L4-004", "R-P-L7-009", "R-P-L7-010")
"""The shipped rules that read this corpus's columns; their thresholds are folded into `pipeline_fingerprint`."""

# The rules' own thresholds, stated once so the corpus is built to clear them deliberately.
ENTROPY_THRESHOLD = 4.0
LABEL_LENGTH_THRESHOLD = 30
SUBDOMAIN_THRESHOLD = 100
PATH_THRESHOLD = 200
ERROR_RATIO_THRESHOLD = 0.7
BASELINE_MULTIPLE = 2.0
"""How many times a client's busiest day the DNS, HTTP and service rules require above their floors."""
DGA_DOMAIN_THRESHOLD = 50
NXDOMAIN_THRESHOLD = 0.5
SERVICE_CLIENT_FLOOR = 10

HISTORY_DAYS = 7
"""Days of ordinary life before the day the actors act. A week, which is what a client needs for a baseline."""
EVENT_DAY = HISTORY_DAYS
HISTORY_HOUR = 9
"""The hour of each history day the ordinary traffic happens in."""

# DNS clients and the domains they query.
TUNNELLER = "10.0.0.30"
CDN_CLIENT = "10.0.0.31"
TENANT_CLIENT = "10.0.0.32"
SAAS_CLIENT = "10.0.0.33"
BROWSER = "10.0.0.34"
HASH_LOOKUP = "10.0.0.35"
"""An endpoint agent that asks a file reputation service about every file hash it sees, by DNS, every day."""
GENERATOR = "10.0.0.36"
"""Malware finding its controller among names it generates."""
STALE_SUFFIX = "10.0.0.37"
BUSY_BROWSER = "10.0.0.38"
LINK_CHECKER = "10.0.0.39"
SLOW_TUNNELLER = "10.0.0.46"
"""A client whose tunnel ramps: a few more chunks under one domain every day, never reaching R-B-L7-001's floor and
never twice its own busiest day, which is R-P-L7-009's trajectory."""

TUNNEL_DOMAIN = "evil-tunnel.com"
SLOW_DOMAIN = "slow-tunnel.net"
CDN_DOMAIN = "cdn-assets.net"
TENANT_DOMAIN = "tenant-saas.com"
SAAS_DOMAIN = "saas-provider.com"
REPUTATION_DOMAIN = "filerep-av.net"
SUFFIX_DOMAIN = "example-corp.com"

DNS_VOLUME = 130
"""Distinct names each high-volume DNS actor queries. Comfortably above R-B-L7-001's hundred, so the count is not what
separates them; what separates them is which of the other conditions they fail.

Comfortably, because random names do not all clear 4.0 bits: about one in twelve of the CDN's twenty-nine-character
names falls just under. At a hundred and ten names that left the CDN's entropy-clearing names one above the count's
threshold, so its control -- that entropy and the count alone would have fired -- held by a single name."""
SLOW_START = 10
SLOW_STEP = 6
"""The slow tunnel's chunks on its first day and the day-on-day increase: ten to fifty-two over the eight days, each
day a little more than the one before and the last barely a tenth above its busiest."""
SLOW_SEED = CORPUS_SEED + 1
"""The slow tunnel's names come from a generator of their own, so adding it moved no other actor's names."""

REPUTATION_HISTORY = 100
REPUTATION_TODAY = 120
"""Lookups the reputation agent makes each history day, and on the day the rules are read: at the count's hundred every
day of its first week, above it on the eighth, and nowhere near twice its own busiest.

At the floor rather than above it, because for its first week the agent has no baseline and is held to the floor alone,
as every client is -- had its first week cleared the hundred, the rule would have fired on it then, which is what the
floor is for and not what this control is about."""

CDN_LABEL_LENGTH = 29
"""One short of the rule's thirty, and long enough that random characters clear 4.0 bits. The label-length condition
is therefore the only thing between this CDN and the rule."""

GENERATED = 80
"""Registered domains the generator asks for in its hour, nine in ten of which do not exist."""
GENERATOR_RESOLVES_EVERY = 10
LINKS_HISTORY = 50
LINKS_TODAY = 60
"""Domains the link checker resolves each morning of its first week -- at the rule's fifty, for the reason the
reputation agent's week is at its hundred -- and on the eighth, seven in ten of them dead."""
BROWSED = 70
"""Domains the busy browser reaches in its hour, every one of them answered."""

# HTTP clients.
ENUMERATOR = "10.0.0.40"
EMPTY_HANDED = "10.0.0.41"
CRAWLER = "10.0.0.42"
BROKEN_CLIENT = "10.0.0.43"
READER = "10.0.0.44"
SCANNER = "10.0.0.45"
"""The vulnerability scanner that walks the web server every morning."""
SLOW_ENUMERATOR = "10.0.0.47"
"""A client whose probing ramps: a few more paths every morning, never reaching R-D-L7-005's two hundred and never
twice its own busiest day, which is R-P-L7-010's trajectory."""

# The transport corpus's clients, whose stacks R-B-L4-004 sets against the user agents they send here.
STACK_CONSISTENT = "10.0.0.60"
STACK_SPOOFER = "10.0.0.61"
STACK_TOOL = "10.0.0.62"

WEB_SERVER = "10.0.1.80"
WEB_PORT = 443

HTTP_VOLUME = 220
"""Distinct paths each high-volume HTTP actor requests. Above R-D-L7-005's two hundred."""

SLOW_PATH_START = 40
SLOW_PATH_STEP = 10
"""The slow enumerator's distinct paths on its first morning and the day-on-day increase: forty to a hundred and ten
over the eight days, the last barely a tenth above its busiest and ninety short of the floor."""

SCANNER_HISTORY = 200
"""Paths the scanner walks each morning of its first week: at the rule's two hundred, for the reason the reputation
agent's week is at its hundred."""

ENUMERATOR_HIT_EVERY = 10
"""One request in this many finds something. The enumerator's ratio is nine refusals to one success."""

CRAWLER_MISS_EVERY = 20
"""One request in this many is refused. The crawler's ratio is one refusal to nineteen successes."""

BROKEN_REQUESTS = 60

CHROME_WINDOWS = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/126.0.0.0 Safari/537.36")
FIREFOX_LINUX = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
CURL = "curl/8.5.0"
SCANNER_AGENT = "Mozilla/5.0 (compatible; VulnScanner/10.4)"
CRAWLER_AGENT = "IntranetIndexer/3.1"

# Services.
EDGE_PROXY = "10.0.2.10"
"""One address terminating two names: the payroll API and the wiki."""
METRICS_SERVER = "10.0.2.20"
NEW_SERVER = "10.0.2.30"
CI_SERVER = "10.0.2.40"

PAYROLL = "payroll-api.corp.example"
WIKI = "wiki.corp.example"
CI = "ci.corp.example"
NEW_APP = "newapp.corp.example"
METRICS_PORT = 8443

APP_SERVERS = tuple(f"10.0.3.{index}" for index in (11, 12, 13))
WORKSTATIONS = tuple(f"10.0.5.{index}" for index in range(10, 60))
METRICS_AGENTS = tuple(f"10.0.6.{index}" for index in range(10, 22))

FP_BROWSER = "t13d1516h2_8daaf6152771_e5627efa2ab1"
FP_APP = "t13d1715h2_5b57614c22b0_3d5424432f57"
FP_SCRIPT = "t13d0912h1_c866b44c5a26_b0c3d6c5e2a1"
FP_AGENT = "t12d2008h1_a09f3c656075_9b3c2b5d1e0f"
FP_AGENT_UPGRADED = "t13d2008h1_a09f3c656075_7e1d4c2a9b88"
FP_CI = "t13d1411h2_1d37bd780c83_3f5c6b2e8d10"
FP_CI_TOOL = "t13d0611h1_1d37bd780c83_aa61e0c3d7f2"

PAYROLL_WORKSTATIONS = 12
WIKI_HISTORY_READERS = 10
WIKI_TODAY_READERS = 25
CI_HISTORY_CLIENTS = 2
CI_TODAY_CLIENTS = 5
NEW_APP_CLIENTS = 15


def at(hour: int, second: float = 0) -> int:
    """Event time for an offset into the corpus, in nanoseconds since the epoch."""
    return int((hour * PERIOD_SECONDS + second) * NS_PER_SECOND)


def on(day: int, hour: int, second: float = 0) -> int:
    """Event time for an hour of a day of the corpus."""
    return at(day * DAY_HOURS + hour, second)


def event(hour: int, second: float = 0) -> int:
    """Event time for an hour of the day the actors act."""
    return on(EVENT_DAY, hour, second)


def _random_label(rng: random.Random, length: int, alphabet: str = "abcdefghijklmnopqrstuvwxyz0123456789") -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def _tunnel_name(rng: random.Random, domain: str = TUNNEL_DOMAIN) -> str:
    """A chunk of exfiltrated data, base32-encoded the way tunnelling tools encode it -- DNS is case-insensitive,
    which is why they use base32 rather than base64 -- and split into two long labels."""
    payload = base64.b32encode(bytes(rng.randrange(256) for _ in range(45))).decode("ascii").rstrip("=").lower()

    return f"{payload[:40]}.{payload[40:]}.{domain}"


def _reputation_name(rng: random.Random) -> str:
    """A file's hash, base32 as the reputation service takes it, as one long label under the service's domain."""
    digest = base64.b32encode(bytes(rng.randrange(256) for _ in range(32))).decode("ascii").rstrip("=").lower()

    return f"{digest}.{REPUTATION_DOMAIN}"


def _envelope(collector: str, schema: str, seq: int, origin: str) -> dict:
    return {"collector_id": collector, "schema_version": schema, "origin_hash": origin, "collector_seq": seq}


def _ordered(rows: list) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values("event_time", kind="mergesort").reset_index(drop=True)
    frame["collector_seq"] = range(len(frame))

    return frame


def _build_dns(rng: random.Random) -> pd.DataFrame:
    rows: list[dict] = []

    def add(client, name, event_time_ns, response="NOERROR"):
        rows.append({
            **_envelope("resolver-01", "tc7_dns.v1", len(rows), f"{client}?{name}@{event_time_ns}"),
            "event_time":
                event_time_ns,
            "src_ip":
                client,
            "query_name":
                name,
            "query_type":
                "TXT" if name.endswith((TUNNEL_DOMAIN, SLOW_DOMAIN)) else "A",
            "response_code":
                response,
        })

    # The week before: the tunneller and the generator browse, the reputation agent looks up the day's files, and
    # the link checker walks its links. The other DNS clients are new to the estate on the day the rules are read.
    for day in range(HISTORY_DAYS):
        for index in range(6):
            add(TUNNELLER, f"www.news-site{index}.com", on(day, HISTORY_HOUR, 60 + index * 120))
            add(GENERATOR, f"www.shop-site{index}.com", on(day, HISTORY_HOUR, 90 + index * 120))

        for index in range(REPUTATION_HISTORY):
            add(HASH_LOOKUP, _reputation_name(rng), on(day, HISTORY_HOUR, 30 + index * 10))

        for index in range(LINKS_HISTORY):
            response = "NXDOMAIN" if index % 10 < 7 else "NOERROR"
            add(LINK_CHECKER, f"www.linked-site{index}.org", on(day, HISTORY_HOUR, 45 + index * 15), response)

    # The slow tunnel: the same shape as the tunnel below, a few chunks at a time, a few more every day of the week
    # and on the event day. No day reaches the floor and no day is twice the one before, so R-B-L7-001 is quiet on
    # every one; the daily trajectory is what sees it. Its names come from their own generator.
    slow = random.Random(SLOW_SEED)

    for day in range(HISTORY_DAYS + 1):
        for index in range(SLOW_START + SLOW_STEP * day):
            when = event(0, 400 + index * 30) if day == EVENT_DAY else on(day, HISTORY_HOUR, 20 + index * 30)
            add(SLOW_TUNNELLER, _tunnel_name(slow, SLOW_DOMAIN), when)

    # The tunnel: a new, random, long-labelled name for every chunk, all under one domain, inside twenty minutes.
    for index in range(DNS_VOLUME):
        add(TUNNELLER, _tunnel_name(rng), event(0, 300 + index * 10))

    # The content delivery network: random asset hostnames, many of them, one label just short of thirty.
    for index in range(DNS_VOLUME):
        add(CDN_CLIENT, f"{_random_label(rng, CDN_LABEL_LENGTH)}.{CDN_DOMAIN}", event(0, 305 + index * 10))

    # The tenant: one long random hostname, asked for over and over. Every condition but the count.
    tenant = f"{_random_label(rng, 40)}.{_random_label(rng, 32)}.{TENANT_DOMAIN}"

    for index in range(30):
        add(TENANT_CLIENT, tenant, event(0, 310 + index * 30))

    # The SaaS provider: a hundred and thirty ordinary tenant subdomains. Only the count.
    for index in range(DNS_VOLUME):
        add(SAAS_CLIENT, f"team{index}.{SAAS_DOMAIN}", event(0, 315 + index * 10))

    # The reputation agent's day, as busy as any other.
    for index in range(REPUTATION_TODAY):
        add(HASH_LOOKUP, _reputation_name(rng), event(0, 320 + index * 10))

    # Ordinary browsing.
    for (index, name) in enumerate([
            "www.example.com",
            "mail.globex-corporation.com",
            "login.microsoftonline.com",
            "api.stripe.com",
            "outlook.office365.com"
    ] * 4):
        add(BROWSER, name, event(1, 60 + index * 90))

    # The generator: eighty registered domains it has never asked for, nine in ten of them nowhere.
    for index in range(GENERATED):
        response = "NOERROR" if index % GENERATOR_RESOLVES_EVERY == 0 else "NXDOMAIN"
        tld = ("com", "net", "org", "info")[index % 4]
        add(GENERATOR,
            f"{_random_label(rng, 12, 'abcdefghijklmnopqrstuvwxyz')}.{tld}",
            event(2, 30 + index * 20),
            response)

    # The stale suffix: told no about one domain all hour.
    for index in range(GENERATED):
        name = ("wpad", "isatap", "fileserver", "printer")[index % 4]
        add(STALE_SUFFIX, f"{name}.{SUFFIX_DOMAIN}", event(2, 35 + index * 20), "NXDOMAIN")

    # The busy browser: as many domains as the generator, every one answered.
    for index in range(BROWSED):
        add(BUSY_BROWSER, f"www.morning-read{index}.com", event(2, 40 + index * 20))

    # The link checker's morning, no different from any other.
    for index in range(LINKS_TODAY):
        response = "NXDOMAIN" if index % 10 < 7 else "NOERROR"
        add(LINK_CHECKER, f"www.linked-site{index}.org", event(2, 45 + index * 15), response)

    return _ordered(rows)


def _build_http(rng: random.Random) -> pd.DataFrame:
    rows: list[dict] = []

    def add(client, path, status, event_time_ns, agent=CHROME_WINDOWS):
        rows.append({
            **_envelope("proxy-01", "tc7_http.v1", len(rows), f"{client}>{path}@{event_time_ns}"),
            "event_time":
                event_time_ns,
            "src_ip":
                client,
            "dst_ip":
                WEB_SERVER,
            "dst_port":
                WEB_PORT,
            "http_method":
                "GET",
            "url_path":
                path,
            "status_code":
                status,
            "user_agent":
                agent,
        })

    words = ["admin", "backup", "config", "api", "login", "old", "test", "dev", "private", "upload"]

    # The week before: the enumerator reads the news in its browser, the reader reads it too, and the scanner walks
    # the server as it does every morning.
    for day in range(HISTORY_DAYS):
        for index in range(10):
            add(ENUMERATOR, f"/news/{index}", 200, on(day, HISTORY_HOUR, 60 + index * 30))
            add(READER, f"/news/{index}", 200, on(day, HISTORY_HOUR, 75 + index * 30))

        for index in range(SCANNER_HISTORY):
            status = 200 if index % ENUMERATOR_HIT_EVERY == 0 else 404
            add(SCANNER,
                f"/{words[index % len(words)]}/{index}",
                status,
                on(day, HISTORY_HOUR, 600 + index),
                SCANNER_AGENT)

    # The slow enumerator: the same probing as the enumerator below, in the browser it always uses, ten more paths
    # every morning of the week and the day after it. Every one of its windows is under both of R-D-L7-005's floors.
    for day in range(HISTORY_DAYS + 1):
        for index in range(SLOW_PATH_START + SLOW_PATH_STEP * day):
            status = 200 if index % ENUMERATOR_HIT_EVERY == 0 else 404
            add(SLOW_ENUMERATOR,
                f"/{words[index % len(words)]}/{index}",
                status,
                on(day, HISTORY_HOUR, 1800 + index * 2))

    # The enumerator: a different path every second, nine in ten not there, from a tool it has never run.
    for index in range(HTTP_VOLUME):
        status = 200 if index % ENUMERATOR_HIT_EVERY == 0 else 404
        add(ENUMERATOR, f"/{words[index % len(words)]}/{index}", status, event(0, 600 + index), CURL)

    # The enumerator that has found nothing: every response a refusal, so its 4xx:2xx ratio is undefined.
    for index in range(HTTP_VOLUME):
        add(EMPTY_HANDED, f"/{words[index % len(words)]}-{index}.bak", 404, event(0, 1200 + index), CURL)

    # The crawler: as many distinct paths, nearly all of them there.
    for index in range(HTTP_VOLUME):
        status = 404 if index % CRAWLER_MISS_EVERY == 0 else 200
        add(CRAWLER, f"/articles/{index}", status, event(0, 1800 + index), CRAWLER_AGENT)

    # The broken client: one missing resource, asked for again and again.
    for index in range(BROKEN_REQUESTS):
        add(BROKEN_CLIENT, "/static/app.js.map", 404, event(0, 2400 + index * 5))

    # The scanner's morning, no busier than its others.
    for index in range(HTTP_VOLUME):
        status = 200 if index % ENUMERATOR_HIT_EVERY == 0 else 404
        add(SCANNER, f"/{words[index % len(words)]}/{index}", status, event(0, 2700 + index), SCANNER_AGENT)

    # Ordinary reading.
    for index in range(20):
        add(READER, f"/news/{rng.randrange(5)}", 200 if index % 7 else 304, event(1, 120 + index * 60))

    # The transport corpus's clients, saying which operating system they run. The spoofer's says Windows, and the
    # stack its connections carry says otherwise; the tool says nothing about an operating system at all.
    for (client, agent) in ((STACK_CONSISTENT, CHROME_WINDOWS), (STACK_SPOOFER, CHROME_WINDOWS), (STACK_TOOL, CURL)):
        for index in range(5):
            add(client, f"/portal/{index}", 200, event(3, 60 + index * 60), agent)

    return _ordered(rows)


def _build_services() -> pd.DataFrame:
    rows: list[dict] = []

    def add(client, address, port, name, fingerprint, event_time_ns, size=4096):
        rows.append({
            **_envelope("lb-01", "tc7_service.v1", len(rows), f"{client}->{name or address}:{port}@{event_time_ns}"),
            "event_time":
                event_time_ns,
            "src_ip":
                client,
            "dst_ip":
                address,
            "dst_port":
                port,
            "sni":
                name,
            "bytes_total":
                size,
            "ja4_client":
                fingerprint,
        })

    # The week before. The payroll API is called by its three application servers every hour of the working morning;
    # the wiki is read by ten people a day; the metrics collector hears from twelve agents; the CI server from two.
    for day in range(HISTORY_DAYS):
        for hour in (HISTORY_HOUR, HISTORY_HOUR + 1, HISTORY_HOUR + 2):
            for (index, server) in enumerate(APP_SERVERS):
                add(server, EDGE_PROXY, 443, PAYROLL, FP_APP, on(day, hour, 60 + index * 300), 8192)

        for (index, reader) in enumerate(WORKSTATIONS[:WIKI_HISTORY_READERS]):
            add(reader, EDGE_PROXY, 443, WIKI, FP_BROWSER, on(day, HISTORY_HOUR, 120 + index * 60))

        for (index, agent) in enumerate(METRICS_AGENTS):
            add(agent, METRICS_SERVER, METRICS_PORT, None, FP_AGENT, on(day, HISTORY_HOUR, 30 + index * 30), 1024)

        for (index, builder) in enumerate(WORKSTATIONS[40:40 + CI_HISTORY_CLIENTS]):
            add(builder, CI_SERVER, 443, CI, FP_CI, on(day, HISTORY_HOUR, 200 + index * 60))

    # The day. The application servers call the payroll API as always -- and then a dozen workstations do, with a
    # scripting library the API has never been called by.
    for (index, server) in enumerate(APP_SERVERS):
        add(server, EDGE_PROXY, 443, PAYROLL, FP_APP, event(HISTORY_HOUR, 60 + index * 300), 8192)

    for (index, workstation) in enumerate(WORKSTATIONS[:PAYROLL_WORKSTATIONS]):
        add(workstation, EDGE_PROXY, 443, PAYROLL, FP_SCRIPT, event(HISTORY_HOUR, 1200 + index * 30), 65536)

    # The wiki, on the same address and port: twice its usual readers, all in the browser it knows.
    for (index, reader) in enumerate(WORKSTATIONS[:WIKI_TODAY_READERS]):
        add(reader, EDGE_PROXY, 443, WIKI, FP_BROWSER, event(HISTORY_HOUR, 120 + index * 60))

    # The metrics collector: its usual twelve agents, one of which upgraded overnight.
    for (index, agent) in enumerate(METRICS_AGENTS):
        fingerprint = FP_AGENT_UPGRADED if index == 0 else FP_AGENT
        add(agent, METRICS_SERVER, METRICS_PORT, None, fingerprint, event(HISTORY_HOUR, 30 + index * 30), 1024)

    # The CI server: a new tool on five machines, more than twice its usual two and still a handful.
    for (index, builder) in enumerate(WORKSTATIONS[40:40 + CI_TODAY_CLIENTS]):
        add(builder, CI_SERVER, 443, CI, FP_CI_TOOL, event(HISTORY_HOUR, 200 + index * 60))

    # The service deployed this morning: fifteen clients, every fingerprint new, and no days to measure them against.
    for (index, client) in enumerate(WORKSTATIONS[20:20 + NEW_APP_CLIENTS]):
        fingerprint = (FP_BROWSER, FP_SCRIPT, FP_CI_TOOL)[index % 3]
        add(client, NEW_SERVER, 443, NEW_APP, fingerprint, event(HISTORY_HOUR, 300 + index * 60))

    return _ordered(rows)


def build_corpus() -> dict[str, pd.DataFrame]:
    """The seeded corpus, one frame per layer 7 class."""
    rng = random.Random(CORPUS_SEED)

    (dns, http, services) = (_build_dns(rng), _build_http(rng), _build_services())
    every = pd.concat([dns, http, services], ignore_index=True)

    # Which machine each client address was: the DHCP server's leases and the EDR agent's inventory, over the span.
    hosts = host_identity.build(every["src_ip"], int(every["event_time"].min()), int(every["event_time"].max()))

    return envelope.attach_all({DNS_CLASS: dns, HTTP_CLASS: http, SERVICE_CLASS: services, **hosts})


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
        return [TC7HttpStage(config), application_context.enrich_stage(config, store)]

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
