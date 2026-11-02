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
The synthetic corpus `morpheus.compositions.presentation` is tested over, and the facts about it the harnesses assert.

Moved out of `tests/morpheus/determinism/presentation_pipeline.py` under plan section 4.5, which put the composition in
the package and left the data here, where the tests, the sample and fixture generators and the experiments read it. The
composition module's docstring describes what this corpus plants and why.
"""

import hashlib

from corpora import merged  # noqa: F401

from morpheus.compositions import envelope

host_identity = merged('host_identity')
import pandas as pd

from morpheus.compositions.presentation import CORP_CA
from morpheus.compositions.presentation import CREEPER
from morpheus.compositions.presentation import DAY_SECONDS
from morpheus.compositions.presentation import LEGACY_CLIENT
from morpheus.compositions.presentation import MANAGED
from morpheus.compositions.presentation import PERIOD_SECONDS
from morpheus.compositions.presentation import TELEMETRY_CLASS
from morpheus.compositions.presentation import UPDATED
from morpheus.compositions.presentation import VARIED
from morpheus.utils.binding_table import NS_PER_SECOND

CORPUS_HOURS = 3
CORPUS_SECONDS = CORPUS_HOURS * PERIOD_SECONDS
"""The event day's span: the three hours every per-event rule is read over."""
HISTORY_DAYS = 6
"""Days before the event day on which the two stack-acquiring hosts are seen, so a daily trajectory has days."""
EVENT_DAY = HISTORY_DAYS
COLLECTOR = "tls-inspect-01"
SCHEMA_VERSION = "tc6.v1"
PERSONAL_DEVICE = "10.0.0.23"
"""A phone on the corporate wireless. Settled, and then a new stack; and in no inventory."""
# Destinations. Outside the ranges the RFCs reserve for documentation, which `parsers/ip.py` classifies as
# private -- the layer 3 corpus was built on those once and reported a host on the public internet as one that
# never left the estate.
PORTAL = "93.184.216.34"
CDN = "93.184.216.35"
ROTATED = "93.184.216.36"
ATTACKER = "93.184.216.37"
APPLIANCE = "10.0.1.50"
SHARED = "93.184.216.38"
"""A hosting provider's address, terminating two customers' names."""
VENDOR = "93.184.216.39"
"""A software vendor's update service, which the two stack-acquiring hosts reach every day behind one authority."""
SHOP = "shop.example-hosting.net"
MAIL = "mail.example-hosting.net"
NAMES = {
    PORTAL: "portal.example.com",
    CDN: "static.example-delivery.net",
    ROTATED: "updates.example.org",
    ATTACKER: "cdn-sync.example.net",
    VENDOR: "updates.example-vendor.net",
}
"""The server name each destination's handshakes ask for. The internal appliance is reached by its address."""
# Fingerprints. The shape JA4 produces, which is what a reader will be comparing against their own feed.
CHROME = "t13d1516h2_8daaf6152771_b186095e22b6"
UPDATER = "t13d0312h2_55b375c5d22e_cd85d2d88918"
NEW_STACK = "t13d3104h1_9b1e5a0f4c22_7e0fd1a83b55"
STACKS = (
    CHROME,
    "t13d1517h2_8daaf6152771_02713d6af862",
    "t13d1715h2_5b57614c22b0_3d5424432f57",
    "t13d0912h1_c866b44c5a26_b0c3d6c5e2a1",
    "t13d1411h2_1d37bd780c83_3f5c6b2e8d10",
    "t12d2008h1_a09f3c656075_9b3c2b5d1e0f",
    "t13d0611h1_1d37bd780c83_aa61e0c3d7f2",
)
"""The stacks the creeping host presents, one more each day; the host that stops acquires the first three."""
DAILY_HANDSHAKES = 24
"""Each stack-acquiring host's handshakes a day, enough that every acquisition after the first day is a settled
host's new stack to R-B-L6-001."""
UPDATED_STACKS = 3
PUBLIC_CA = "CN=Public Trust CA G2, O=Trust Services, C=US"
ROTATED_CA = "CN=Public Trust CA G3, O=Trust Services, C=US"
HOSTING_CA = "CN=Hosting Mail CA, O=Hosting"
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


def on(day: int, hour: int, second: int = 0) -> int:
    """Event time for an hour of a day of the corpus, in nanoseconds since the epoch."""
    return (day * DAY_SECONDS + hour * PERIOD_SECONDS + second) * NS_PER_SECOND


def at(hour: int, second: int = 0) -> int:
    """Event time for an offset into the event day, in nanoseconds since the epoch."""
    return on(EVENT_DAY, hour, second)


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
            detected=None,
            sni=None):
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
            "sni": sni if sni is not None else NAMES.get(dst),
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

    # The creeping host: a new client stack every day for a week, each presented among the ones it already has, so
    # its count of distinct stacks rises by one a day. Every acquisition after the first day is a settled host's new
    # stack and R-B-L6-001 reports it; the trajectory is what reads the seven as one. Beside it, a host that acquires
    # a stack on each of two days and then settles: a rise of three windows, one short of the rule. Both reach a
    # vendor's update service that presents one authority throughout, so nothing else about them changes.
    for day in range(HISTORY_DAYS + 1):
        for index in range(DAILY_HANDSHAKES):
            add(CREEPER, VENDOR, STACKS[index % (day + 1)], PUBLIC_CA, MODERN_SUITE, on(day, 1, 60 + index * 120))
            add(UPDATED,
                VENDOR,
                STACKS[index % min(day + 1, UPDATED_STACKS)],
                PUBLIC_CA,
                MODERN_SUITE,
                on(day, 1, 75 + index * 120))

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

    # The shared host: two names on one address, each with its own authority, visited in turn -- and then the shop
    # intercepted. Keyed on the address alone this is a destination with two authorities already, and the third is
    # one more among several rather than the first the shop has ever presented.
    for index in range(16):
        (name, issuer) = (SHOP, PUBLIC_CA) if index % 2 == 0 else (MAIL, HOSTING_CA)
        add(MANAGED, SHARED, CHROME, issuer, MODERN_SUITE, at(0, 100 + index * 400), sni=name)

    add(MANAGED, SHARED, CHROME, PROXY_CA, MODERN_SUITE, at(2, 700), sni=SHOP)

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

    # Which machine each client address was: the DHCP server's leases and the EDR agent's inventory, over the span.
    # The personal device is leased an address like any client and runs no EDR agent, so its address resolves to
    # an interface and to no machine -- which is why the inventory has no context for it.
    hosts = host_identity.build(frame["src_ip"],
                                int(frame["event_time"].min()),
                                int(frame["event_time"].max()),
                                unmanaged=(PERSONAL_DEVICE, ))

    return envelope.attach_all({TELEMETRY_CLASS: frame, **hosts})
