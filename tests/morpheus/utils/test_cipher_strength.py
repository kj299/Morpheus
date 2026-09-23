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

import pytest

from morpheus.utils.cipher_strength import BROKEN
from morpheus.utils.cipher_strength import FORWARD_AEAD
from morpheus.utils.cipher_strength import FORWARD_CBC
from morpheus.utils.cipher_strength import LEGACY
from morpheus.utils.cipher_strength import LEGACY_AEAD
from morpheus.utils.cipher_strength import MODERN
from morpheus.utils.cipher_strength import SUITES
from morpheus.utils.cipher_strength import TIER_NAMES
from morpheus.utils.cipher_strength import normalize
from morpheus.utils.cipher_strength import rank
from morpheus.utils.cipher_strength import tier_name


def test_the_tiers_are_ordered_the_way_the_module_claims():
    # The ordering is the whole module. If these ever stopped being ascending, every comparison built on them
    # would keep working and mean the opposite.
    assert BROKEN < LEGACY < LEGACY_AEAD < FORWARD_CBC < FORWARD_AEAD < MODERN


def test_every_tier_has_a_name_and_every_name_a_tier():
    assert set(TIER_NAMES) == {BROKEN, LEGACY, LEGACY_AEAD, FORWARD_CBC, FORWARD_AEAD, MODERN}
    assert len(set(TIER_NAMES.values())) == len(TIER_NAMES)


@pytest.mark.parametrize(("suite", "expected"),
                         [("TLS_AES_128_GCM_SHA256", MODERN), ("TLS_CHACHA20_POLY1305_SHA256", MODERN),
                          ("TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256", FORWARD_AEAD),
                          ("TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256", FORWARD_AEAD),
                          ("TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA", FORWARD_CBC),
                          ("TLS_RSA_WITH_AES_128_GCM_SHA256", LEGACY_AEAD), ("TLS_RSA_WITH_AES_128_CBC_SHA", LEGACY),
                          ("TLS_RSA_WITH_3DES_EDE_CBC_SHA", BROKEN), ("TLS_RSA_WITH_RC4_128_SHA", BROKEN)])
def test_a_suite_lands_in_the_tier_its_properties_imply(suite: str, expected: int):
    assert rank(suite) == expected


def test_forward_secrecy_outranks_an_authenticated_cipher_without_it():
    # The ordering's central judgement, asserted rather than left in the prose. Recording a session and later
    # obtaining the server's key reads every past session of the RSA suite and none of the ephemeral one, which
    # is worth more than the bulk cipher's mode.
    assert rank("TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA") > rank("TLS_RSA_WITH_AES_256_GCM_SHA384")


def test_a_broken_marker_beats_an_otherwise_respectable_name():
    # The marker scan runs first for this reason. A suite negotiating RC4 with an ephemeral key exchange is
    # broken whatever the key exchange does for it, and matching the name to a tier table alone would have put
    # it above a sound TLS 1.2 suite.
    assert rank("TLS_ECDHE_RSA_WITH_RC4_128_SHA") == BROKEN
    assert rank("TLS_DHE_RSA_WITH_DES_CBC_SHA") == BROKEN
    assert rank("TLS_ECDH_ANON_WITH_AES_128_CBC_SHA") == BROKEN


def test_an_unrecognized_suite_has_no_rank():
    # The decision this module exists to make visible. Any default would either manufacture a downgrade for
    # every suite the table has not heard of or hide a real one, so the answer is that there is no answer.
    assert rank("TLS_SOMETHING_NOBODY_HAS_SHIPPED") is None
    assert rank("") is None
    assert rank(None) is None


def test_an_openssl_short_name_is_unrecognized_rather_than_guessed_at():
    # Spelling is folded; naming schemes are not translated. `ECDHE-RSA-AES128-GCM-SHA256` is not a hyphenated
    # IANA name -- it drops the `WITH` and punctuates the key size differently -- so it lands in the
    # unrecognized count, where an estate can see its whole feed needs an alias table at the collector.
    assert rank("ECDHE-RSA-AES128-GCM-SHA256") is None


@pytest.mark.parametrize("spelling", ["tls_aes_128_gcm_sha256", "TLS-AES-128-GCM-SHA256", "  TLS_AES_128_GCM_SHA256  "])
def test_spelling_is_folded(spelling: str):
    assert rank(spelling) == MODERN


def test_normalize_reports_no_name_rather_than_an_empty_one():
    assert normalize(None) is None
    assert normalize("   ") is None


def test_tier_name_refuses_anything_that_is_not_a_tier():
    assert tier_name(MODERN) == "modern"
    assert tier_name(None) is None
    assert tier_name(99) is None
    assert tier_name(True) is None


def test_the_table_only_holds_tiers_it_declares():
    assert set(SUITES.values()) <= set(TIER_NAMES)


def test_the_table_is_keyed_in_normalized_form():
    # A key that does not survive its own normalization is a key nothing can ever look up.
    assert all(name == normalize(name) for name in SUITES)
