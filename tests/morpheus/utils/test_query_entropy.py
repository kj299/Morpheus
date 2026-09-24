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

import math

import pytest

from morpheus.utils.query_entropy import label_lengths
from morpheus.utils.query_entropy import mean_label_length
from morpheus.utils.query_entropy import normalize
from morpheus.utils.query_entropy import public_suffix
from morpheus.utils.query_entropy import registered_domain
from morpheus.utils.query_entropy import shannon_entropy
from morpheus.utils.query_entropy import subdomain
from morpheus.utils.query_entropy import subdomain_entropy

# --- The registered domain -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [("www.example.com", "example.com"), ("a.b.c.example.com", "example.com"), ("example.com", "example.com"),
     ("www.example.co.uk", "example.co.uk"), ("deep.www.example.co.uk", "example.co.uk")])
def test_the_registered_domain_is_what_someone_registered(name: str, expected: str):
    assert registered_domain(name) == expected


def test_a_two_label_suffix_is_not_mistaken_for_a_registrant():
    # The failure splitting on dots produces. Taking the last two labels makes every British company a
    # subdomain of `co.uk`, and the distinct-subdomain count would go to the country.
    assert public_suffix("www.example.co.uk") == "co.uk"
    assert registered_domain("www.example.co.uk") != "co.uk"


def test_a_public_suffix_has_no_registered_domain():
    assert registered_domain("co.uk") is None
    assert registered_domain("com") is None


def test_a_wildcard_rule_matches_any_label_beneath_it():
    # `*.ck` is in the list. The upstream loader reads the file as literal suffixes, so this rule would match
    # only a label spelled `*`; the algorithm applied here is the list's own.
    assert public_suffix("foo.bar.ck") == "bar.ck"
    assert registered_domain("foo.bar.ck") == "foo.bar.ck"


def test_an_exception_rule_overrides_the_wildcard():
    # `!www.ck` makes `www.ck` registrable despite `*.ck`.
    assert public_suffix("www.ck") == "ck"
    assert registered_domain("www.ck") == "www.ck"
    assert registered_domain("sub.www.ck") == "www.ck"


def test_a_hosting_platform_s_customers_are_registrants_in_their_own_right():
    # The private section is kept for this. Tunnelling through a platform's customer subdomains is the case that
    # matters, and treating the platform as the registrant would put a tunnel's subdomains among millions of
    # legitimate ones.
    assert registered_domain("x.attacker.github.io") == "attacker.github.io"
    assert registered_domain("x.someone-else.github.io") == "someone-else.github.io"


def test_a_name_matching_no_rule_takes_its_last_label_as_the_suffix():
    # The list's documented default rule, `*`.
    assert public_suffix("host.internal-only-tld") == "internal-only-tld"
    assert registered_domain("a.host.internal-only-tld") == "host.internal-only-tld"


@pytest.mark.parametrize("spelling", ["WWW.EXAMPLE.COM", "www.example.com.", "  www.example.com  "])
def test_case_and_the_root_dot_are_normalized(spelling: str):
    assert registered_domain(spelling) == "example.com"
    assert subdomain(spelling) == "www"


def test_an_absent_name_has_no_parts():
    assert normalize(None) is None
    assert normalize("  ") is None
    assert registered_domain(None) is None
    assert subdomain(None) is None
    assert public_suffix(None) is None


# --- The subdomain and its entropy -----------------------------------------------------------------------------


def test_the_subdomain_is_everything_below_the_registered_domain():
    assert subdomain("a.b.example.co.uk") == "a.b"
    assert subdomain("example.co.uk") is None


def test_entropy_is_zero_for_one_repeated_symbol_and_never_negative_zero():
    value = shannon_entropy("aaaa")

    assert value == 0.0
    assert math.copysign(1.0, value) == 1.0


def test_entropy_of_evenly_used_symbols_is_the_log_of_their_count():
    assert shannon_entropy("abcd") == pytest.approx(2.0)
    assert shannon_entropy("0123456789abcdef") == pytest.approx(4.0)


def test_a_short_string_cannot_clear_the_rule_s_threshold():
    # The bound worth knowing when reading R-B-L7-001: `n` characters cannot exceed log2(n) bits per character,
    # so no label shorter than seventeen characters reaches 4.0 whatever it contains.
    sixteen_distinct = "abcdefghijklmnop"

    assert shannon_entropy(sixteen_distinct) == pytest.approx(4.0)
    assert shannon_entropy(sixteen_distinct[:15]) < 4.0


def test_an_empty_string_has_no_entropy():
    assert shannon_entropy("") is None
    assert shannon_entropy(None) is None


ORDINARY_NAMES = (
    "www.northwind-traders-inc.com",
    "mail.globex-corporation.com",
    "login.microsoftonline.com",
    "api.stripe.com",
    "outlook.office365.com",
)

TUNNEL_NAME = "iqqieph543y4e2zq7ehmpxib4sehknfcb4fq2bgdn3ma44pa.7v33a5tq5okaxvjt.evil-tunnel.com"


def _whole(name: str) -> float:
    return shannon_entropy(name.replace(".", ""))


def test_measuring_below_the_registered_domain_widens_the_gap_the_threshold_sits_in():
    # Why entropy is taken below the registered domain, measured rather than asserted. This test replaced one
    # that asserted the domain *dilutes* the figure, which is backwards: appending a domain's characters usually
    # adds symbols and raises it. What the constant part actually does is lift ordinary names toward 4.0 -- they
    # sit between 3.0 and 3.7 whole and between 0 and 2.3 below the domain -- which narrows the margin between
    # ordinary traffic and a tunnel to a few tenths of a bit.
    ordinary_whole = max(_whole(name) for name in ORDINARY_NAMES)
    ordinary_below = max(subdomain_entropy(name) for name in ORDINARY_NAMES)

    assert ordinary_whole > 3.0
    assert ordinary_below < 2.5
    assert subdomain_entropy(TUNNEL_NAME) > 4.0

    margin_below = subdomain_entropy(TUNNEL_NAME) - ordinary_below
    margin_whole = _whole(TUNNEL_NAME) - ordinary_whole

    assert margin_below > margin_whole + 1.0


def test_the_dots_between_labels_are_not_counted_as_payload():
    # A dot is structure every multi-label name shares, so counting it adds a symbol that says nothing.
    assert subdomain_entropy("abcd.abcd.example.com") == pytest.approx(shannon_entropy("abcdabcd"))


def test_a_name_with_nothing_below_its_registered_domain_has_no_subdomain_entropy():
    # No part below means no measurement, rather than the entropy of the registered domain standing in for it.
    assert subdomain_entropy("example.com") is None


# --- Label lengths ---------------------------------------------------------------------------------------------


def test_label_lengths_skip_the_empty_label_a_stray_dot_would_make():
    assert label_lengths("abc..de") == [3, 2]


def test_the_mean_label_length_is_over_the_labels_present():
    assert mean_label_length("abcd.ab") == 3.0
    assert mean_label_length(None) is None
    assert mean_label_length("") is None
