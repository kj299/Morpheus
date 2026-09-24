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
What a DNS query name is made of: who registered it, what sits below that, and how random the part below looks.

R-B-L7-001 is three conditions and the guide insists on all three: query name entropy above 4.0 bits per character,
mean label length above 30, and more than 100 distinct subdomains under one registered domain in an hour. "Entropy
alone flags every content delivery network." Each condition needs a piece this module supplies.

**The registered domain is found by the Public Suffix List's own algorithm, not by splitting on dots.** Taking the
last two labels makes `example.co.uk` a subdomain of `co.uk`, which would put every British company under one
registered domain and give the distinct-subdomain count to the country. The list is the one upstream Morpheus
already bundles for its URL parser, so there is no new dependency. Its matching rules are applied properly here --
the longest matching rule wins, `*.ck` matches any label under `ck`, and `!www.ck` is an exception that makes
`www.ck` registrable -- which the upstream loader does not do: it reads the file as a list of literal suffixes, so a
wildcard rule matches only a label literally spelled `*`.

**The private section of the list is included.** It is what makes `attacker.github.io` a registered domain in its
own right rather than a subdomain of `github.io`. Tunnelling through a hosting platform's customer subdomains is
the case that matters, and treating the platform as the registrant would merge every customer into one domain and
put a tunnel's hundred subdomains among a million legitimate ones.

**Entropy is taken over the part below the registered domain, not the whole name.** The registered domain is
constant across every query to it, so over the whole name the figure is mostly a property of how the registrant
spelled their domain rather than of what varies from one query to the next -- and the variation is where a
tunnel's data rides. Measured, the difference is the width of the gap the threshold has to sit in. Ordinary names
score between 3.0 and 3.7 bits per character as whole names (`www.northwind-traders-inc.com` is 3.66) and between
0 and 2.3 below their registered domain (`www` is 0). A tunnel's payload scores above 4.0 either way. Measuring
the whole name therefore puts ordinary traffic within a few tenths of the threshold, where a longer or more varied
domain name tips it over; measuring below the domain leaves it nearly two bits clear. An earlier draft of this
paragraph said the constant part *dilutes* the measurement, which is backwards -- appending a domain's characters
usually adds symbols and raises the figure -- and the test that caught it is kept. A name with nothing below its
registered domain has no such part and gets no entropy, rather than the entropy of the registered domain itself.

**Shannon entropy is bounded by the length of the string it is taken over**, which is worth knowing when reading
the threshold. A string of `n` characters cannot exceed `log2(n)` bits per character, so no label shorter than
seventeen characters can clear 4.0 whatever it contains. That is not a defect of the measurement; it is why the
rule pairs it with a label-length condition.
"""

import collections
import math
import os
import typing

SUFFIX_LIST_NAME = "public_suffix_list.dat"
"""The file upstream Morpheus bundles in its data directory."""

_RULES: typing.Optional[tuple] = None


def _load_rules() -> tuple:
    """The list's rules as three sets: exact suffixes, wildcard parents, and exceptions.

    Read once per process. Both the ICANN and the private sections are kept.
    """
    global _RULES  # pylint: disable=global-statement

    if (_RULES is not None):
        return _RULES

    import morpheus  # pylint: disable=import-outside-toplevel

    exact: set = set()
    wildcards: set = set()
    exceptions: set = set()

    with open(os.path.join(morpheus.DATA_DIR, SUFFIX_LIST_NAME), encoding="utf-8") as handle:
        for line in handle:
            # The list's own format: one rule per line, the rule ends at the first whitespace, `//` starts a comment.
            rule = line.strip().split()[0] if line.strip() else ""

            if (not rule or rule.startswith("//")):
                continue

            rule = rule.lower()

            if (rule.startswith("!")):
                exceptions.add(rule[1:])
            elif (rule.startswith("*.")):
                wildcards.add(rule[2:])
            else:
                exact.add(rule)

    _RULES = (frozenset(exact), frozenset(wildcards), frozenset(exceptions))

    return _RULES


def normalize(name: typing.Any) -> typing.Optional[str]:
    """
    A query name lower-cased with its trailing root dot removed, or `None` where there is none.

    Parameters
    ----------
    name : any
        The name as the resolver logged it.

    Returns
    -------
    str or None
    """
    if (name is None):
        return None

    text = str(name).strip().lower().rstrip(".")

    return text or None


def public_suffix(name: typing.Any) -> typing.Optional[str]:
    """
    The public suffix of a name by the list's algorithm, or `None` where the name has no labels.

    Parameters
    ----------
    name : any
        The query name.

    Returns
    -------
    str or None
        The longest matching suffix. A name matching no rule takes its last label, which is the list's
        documented default rule `*`.
    """
    normalized = normalize(name)

    if (normalized is None):
        return None

    (exact, wildcards, exceptions) = _load_rules()
    labels = normalized.split(".")

    # Longest first, so the first match is the prevailing rule.
    for start in range(len(labels)):
        candidate = ".".join(labels[start:])

        if (candidate in exceptions):
            # An exception rule's suffix is the rule minus its leftmost label.
            return ".".join(labels[start + 1:]) or None

        if (candidate in exact):
            return candidate

        parent = ".".join(labels[start + 1:])

        if (parent and parent in wildcards):
            return candidate

    return labels[-1]


def registered_domain(name: typing.Any) -> typing.Optional[str]:
    """
    The public suffix plus the one label before it, or `None` where the name is itself a public suffix.

    Parameters
    ----------
    name : any
        The query name.

    Returns
    -------
    str or None
    """
    normalized = normalize(name)

    if (normalized is None):
        return None

    suffix = public_suffix(normalized)

    if (suffix is None or normalized == suffix):
        return None

    remainder = normalized.removesuffix(f".{suffix}")

    return f"{remainder.split('.')[-1]}.{suffix}"


def subdomain(name: typing.Any) -> typing.Optional[str]:
    """
    Everything below the registered domain, or `None` where there is nothing below it.

    Parameters
    ----------
    name : any
        The query name.

    Returns
    -------
    str or None
    """
    normalized = normalize(name)

    if (normalized is None):
        return None

    registered = registered_domain(normalized)

    if (registered is None or normalized == registered):
        return None

    return normalized.removesuffix(f".{registered}")


def shannon_entropy(text: typing.Any) -> typing.Optional[float]:
    """
    Shannon entropy of a string's characters, in bits per character, or `None` for an empty or absent string.

    Parameters
    ----------
    text : any
        The string to measure.

    Returns
    -------
    float or None

    Notes
    -----
    Bounded above by `log2(len(text))`, so a short string cannot score high however random it is.
    """
    if (text is None):
        return None

    value = str(text)

    if (len(value) == 0):
        return None

    total = len(value)

    # Written as p * log2(1/p) so every term is non-negative, and a one-symbol string is 0.0 rather than -0.0.
    return sum((count / total) * math.log2(total / count) for count in collections.Counter(value).values())


def subdomain_entropy(name: typing.Any) -> typing.Optional[float]:
    """
    Entropy of the part of a name below its registered domain, with the dots between labels left out.

    The one definition the stage and its tests share, so "the query name entropy" means the same thing
    everywhere it is read.

    Parameters
    ----------
    name : any
        The query name.

    Returns
    -------
    float or None
        `None` where there is nothing below the registered domain.

    Notes
    -----
    The dots are structure rather than payload: counting them adds a symbol every multi-label name shares.
    The name is lower-cased first, as resolvers log it -- DNS is case-insensitive, which is why tunnelling
    tools encode in base32 rather than base64, and measuring a case the wire does not preserve would measure
    the logger.
    """
    below = subdomain(name)

    if (below is None):
        return None

    return shannon_entropy(below.replace(".", ""))


def label_lengths(text: typing.Any) -> list:
    """
    The length of each dot-separated label in a string, empty labels excluded.

    Parameters
    ----------
    text : any
        Usually the subdomain portion of a query name.

    Returns
    -------
    list of int
    """
    if (text is None):
        return []

    return [len(label) for label in str(text).split(".") if label]


def mean_label_length(text: typing.Any) -> typing.Optional[float]:
    """
    The mean label length of a string, or `None` where it has no labels.

    Parameters
    ----------
    text : any
        Usually the subdomain portion of a query name.

    Returns
    -------
    float or None
    """
    lengths = label_lengths(text)

    return sum(lengths) / len(lengths) if lengths else None
