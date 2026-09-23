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
An explicit strength ordering for TLS cipher suites, which R-B-L6-004 cannot be written without.

The rule is that a pair's negotiated suite has fallen below the weakest that pair has previously settled on. That
comparison needs an order, and the order is a judgement rather than a fact: there is no field in the handshake
that says how strong a suite is, and no two published tables agree in every position. The guide says so and asks
for the table to be maintained alongside the rule, which is what this module is.

**Six tiers, ordered by what an attacker has to defeat.** The properties that decide the tier are, in order of
how much they matter: whether the suite is broken outright, whether the key exchange is forward-secret, and
whether the bulk cipher is authenticated.

- `BROKEN` -- null, anonymous, export-grade, RC4, DES and 3DES. Not weak but wrong: a negotiation landing here
  is a finding on its own, whatever the pair has done before.
- `LEGACY` -- RSA key exchange with a CBC bulk cipher. No forward secrecy, and a construction with a long
  history of padding-oracle attacks.
- `LEGACY_AEAD` -- RSA key exchange with an authenticated cipher. The bulk cipher is sound; recording the
  session key still exposes every past session to whoever later obtains the server's private key.
- `FORWARD_CBC` -- ephemeral key exchange with a CBC bulk cipher. Forward-secret, but unauthenticated
  encryption with a MAC applied in the wrong order.
- `FORWARD_AEAD` -- ephemeral key exchange with an authenticated cipher. What a healthy TLS 1.2 negotiation
  looks like.
- `MODERN` -- the TLS 1.3 suites, which are forward-secret and authenticated by construction.

**An unrecognized suite has no rank, and that is the whole point.** Giving an unknown name a default rank would
do one of two harmful things depending on where the default sat: a low default manufactures a downgrade every
time an estate deploys a suite this table has not heard of, and a high default hides a real downgrade to
something obscure. `rank` returns `None` instead, the rule declines to fire, and the stage counts the
unrecognized names so an estate can see that its table needs an entry rather than discovering it from a
detection that never fires. A table that silently covers everything is worse than one that says where it ends.

**The rank is an ordinal, not a score.** The distance between two tiers means nothing; only which is larger
does. A rule reading "half the strength" off these numbers would be reading something that is not there.
"""

import re
import typing

BROKEN = 0
LEGACY = 1
LEGACY_AEAD = 2
FORWARD_CBC = 3
FORWARD_AEAD = 4
MODERN = 5

TIER_NAMES = {
    BROKEN: "broken",
    LEGACY: "legacy",
    LEGACY_AEAD: "legacy_aead",
    FORWARD_CBC: "forward_cbc",
    FORWARD_AEAD: "forward_aead",
    MODERN: "modern",
}
"""The tier each rank names, for a row a person has to read."""

_BROKEN_MARKERS = ("NULL", "ANON", "EXPORT", "RC4", "DES_CBC", "3DES", "DES40", "IDEA", "SEED", "RC2", "MD5")
"""Substrings that put a suite in `BROKEN` whatever else it carries. Checked first, so an otherwise modern-
looking name that negotiates RC4 does not reach a higher tier on the strength of its key exchange."""

SUITES: dict = {
    # TLS 1.3. The key exchange is not in the name because the protocol fixes it.
    "TLS_AES_128_GCM_SHA256": MODERN,
    "TLS_AES_256_GCM_SHA384": MODERN,
    "TLS_CHACHA20_POLY1305_SHA256": MODERN,
    "TLS_AES_128_CCM_SHA256": MODERN,
    "TLS_AES_128_CCM_8_SHA256": MODERN,

  # Ephemeral key exchange, authenticated cipher.
    "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256": FORWARD_AEAD,
    "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384": FORWARD_AEAD,
    "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256": FORWARD_AEAD,
    "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256": FORWARD_AEAD,
    "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384": FORWARD_AEAD,
    "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256": FORWARD_AEAD,
    "TLS_DHE_RSA_WITH_AES_128_GCM_SHA256": FORWARD_AEAD,
    "TLS_DHE_RSA_WITH_AES_256_GCM_SHA384": FORWARD_AEAD,
    "TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256": FORWARD_AEAD,

  # Ephemeral key exchange, CBC.
    "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA": FORWARD_CBC,
    "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256": FORWARD_CBC,
    "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA": FORWARD_CBC,
    "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384": FORWARD_CBC,
    "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA": FORWARD_CBC,
    "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256": FORWARD_CBC,
    "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA": FORWARD_CBC,
    "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384": FORWARD_CBC,
    "TLS_DHE_RSA_WITH_AES_128_CBC_SHA": FORWARD_CBC,
    "TLS_DHE_RSA_WITH_AES_128_CBC_SHA256": FORWARD_CBC,
    "TLS_DHE_RSA_WITH_AES_256_CBC_SHA": FORWARD_CBC,
    "TLS_DHE_RSA_WITH_AES_256_CBC_SHA256": FORWARD_CBC,

  # RSA key exchange, authenticated cipher.
    "TLS_RSA_WITH_AES_128_GCM_SHA256": LEGACY_AEAD,
    "TLS_RSA_WITH_AES_256_GCM_SHA384": LEGACY_AEAD,

  # RSA key exchange, CBC.
    "TLS_RSA_WITH_AES_128_CBC_SHA": LEGACY,
    "TLS_RSA_WITH_AES_128_CBC_SHA256": LEGACY,
    "TLS_RSA_WITH_AES_256_CBC_SHA": LEGACY,
    "TLS_RSA_WITH_AES_256_CBC_SHA256": LEGACY,

    # Broken outright, listed rather than left to the marker scan so the common ones are explicit.
    "TLS_RSA_WITH_3DES_EDE_CBC_SHA": BROKEN,
    "TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA": BROKEN,
    "TLS_RSA_WITH_RC4_128_SHA": BROKEN,
    "TLS_RSA_WITH_RC4_128_MD5": BROKEN,
    "TLS_RSA_WITH_NULL_SHA256": BROKEN,
}
"""Every suite this table ranks, by its IANA name.

Not every suite the registry holds, and it cannot be: the registry has hundreds, most of which no endpoint has
negotiated this decade. What is here is what an enterprise estate actually presents, plus the broken ones a
downgrade would aim at. Anything absent is reported as unrecognized rather than guessed at.
"""

_SEPARATORS = re.compile(r"[\s-]+")


def normalize(name: typing.Any) -> typing.Optional[str]:
    """
    A suite name in the form this table is keyed by, or `None` where there is no name.

    Parameters
    ----------
    name : any
        The negotiated suite as the collector reported it.

    Returns
    -------
    str or None

    Notes
    -----
    Upper-cased, with hyphens and whitespace folded to underscores, so a feed that spells the IANA name in
    lower case or with hyphens lands on the same key.

    This is spelling, not translation. OpenSSL's short names are a different naming scheme rather than a
    different spelling of the same one -- `ECDHE-RSA-AES128-GCM-SHA256` is not a hyphenated
    `TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256`, it drops the `WITH` and punctuates the key size differently -- so
    a feed emitting them is reported as unrecognized rather than quietly mapped. That is the visible failure
    this module prefers: an estate sees its whole feed land in the unrecognized count and adds an alias table
    at the collector, instead of a rule that silently never fires.
    """
    if (name is None):
        return None

    text = _SEPARATORS.sub("_", str(name).strip()).upper()

    return text or None


def rank(name: typing.Any) -> typing.Optional[int]:
    """
    The strength tier of a cipher suite, or `None` where the table does not recognize it.

    Parameters
    ----------
    name : any
        The negotiated suite as the collector reported it.

    Returns
    -------
    int or None
        One of the six tier constants, or `None`.
    """
    normalized = normalize(name)

    if (normalized is None):
        return None

    if (any(marker in normalized for marker in _BROKEN_MARKERS)):
        return BROKEN

    return SUITES.get(normalized)


def tier_name(value: typing.Any) -> typing.Optional[str]:
    """
    The name of a tier, or `None` for anything that is not one.

    Parameters
    ----------
    value : any
        A rank.

    Returns
    -------
    str or None
    """
    return TIER_NAMES.get(value) if isinstance(value, int) and not isinstance(value, bool) else None
