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
Whether the inventory of personal data still describes the pipeline that produces it.

An inventory is the artifact counsel needs to answer the question this fork's guide defers to them, and an
inventory nobody checks is worse than none: it reads as authoritative and it is a snapshot of whatever the
pipeline looked like on the day somebody wrote it down. The completeness check below is the whole point of the
module -- a new feature column fails a test until somebody has said what it tells a reader about a person.
"""

import csv
import glob
import hashlib
import hmac
import os
import re

import pytest

from morpheus.utils.personal_data import ADDRESSES
from morpheus.utils.personal_data import AMBIGUOUS
from morpheus.utils.personal_data import BOUNDED_DOMAIN
from morpheus.utils.personal_data import CATEGORIES
from morpheus.utils.personal_data import COLUMNS
from morpheus.utils.personal_data import IDENTIFIES
from morpheus.utils.personal_data import LOCATES
from morpheus.utils.personal_data import OPERATIONAL
from morpheus.utils.personal_data import PERSONAL_CATEGORIES
from morpheus.utils.personal_data import PER_ROW
from morpheus.utils.personal_data import PROFILES
from morpheus.utils.personal_data import PSEUDONYMS
from morpheus.utils.personal_data import classify
from morpheus.utils.personal_data import columns_in
from morpheus.utils.personal_data import personal_columns
from morpheus.utils.personal_data import pseudonymize

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
GOLDENS = os.path.join(REPO_ROOT, "tests", "morpheus", "determinism", "*.csv")

KEY = b"a-key-long-enough-to-be-accepted"


def golden_columns() -> set:
    """Every column the four reference pipelines emit, read from the goldens rather than from a list."""
    found = set()

    for path in sorted(glob.glob(GOLDENS)):
        with open(path, encoding="utf-8") as handle:
            found.update(next(csv.reader(handle)))

    return found


def test_the_goldens_are_where_we_think_they_are():
    # Without this the completeness check compares two empty sets and reports a clean inventory of nothing.
    columns = golden_columns()

    assert len(columns) > 150
    assert "user_principal" in columns


def test_every_column_the_pipeline_emits_is_classified():
    # The reason the module exists. A feature column that reaches a SIEM before anybody has decided what it says
    # about a person is the whole failure, and it is a silent one: the column just appears in an index.
    unclassified = sorted(golden_columns() - set(COLUMNS) - set(AMBIGUOUS) - set(PER_ROW))

    assert unclassified == [], (f"{unclassified} reach a SIEM and nothing says what they tell a reader about a "
                                f"person. Add each to morpheus.utils.personal_data.")


def test_nothing_is_classified_that_the_pipeline_does_not_emit():
    # The other direction, which keeps the inventory from accumulating columns that were renamed or removed. A
    # stale entry is not harmless: it makes a policy written by category quietly narrower than it reads.
    stale = sorted((set(COLUMNS) | set(AMBIGUOUS) | set(PER_ROW)) - golden_columns())

    assert stale == [], f"{stale} are classified and nothing emits them any more"


def test_a_column_is_classified_once():
    assert sorted(set(COLUMNS) & set(AMBIGUOUS)) == []
    assert sorted(set(COLUMNS) & set(PER_ROW)) == []
    assert sorted(set(AMBIGUOUS) & set(PER_ROW)) == []


def test_every_category_used_is_one_of_the_six():
    assert set(COLUMNS.values()) <= set(CATEGORIES)

    for meanings in AMBIGUOUS.values():
        assert set(meanings.values()) <= set(CATEGORIES)


def test_the_switch_and_the_laptop_are_the_same_column_name():
    # The collision a policy written by column name alone gets wrong in both directions: strip `device_id` and
    # the identifier ladder loses the switch it walks through; keep it and a person's laptop ships unminimized.
    assert classify("device_id", "tc1") == OPERATIONAL
    assert classify("device_id", "tc5_auth") != OPERATIONAL


def test_the_subject_of_a_row_is_a_port_or_a_person_depending_on_the_layer():
    # Part 2 defines a different subject per layer, and `entity_key` holds whichever it is.
    assert classify("entity_key", "tc1") == LOCATES
    assert classify("entity_key", "tc5_auth") == IDENTIFIES


def test_the_anchor_is_decided_per_row_and_is_not_given_a_category():
    # A third kind of ambiguity, and the one that only turned up when a policy was run against the estate.
    # `ChainAnchorStage` copies the first candidate that matched, so within a single layer 5 frame the anchor is
    # a desk port on the rows the ladder resolved and the principal on the rows it did not. There is no per-class
    # answer, and picking the one that is right more often would be wrong on the rest.
    with pytest.raises(ValueError, match="chain_anchor_source"):
        classify("chain_anchor", "tc5_auth")

    assert "chain_anchor" not in columns_in(LOCATES, "tc5_auth")
    assert "chain_anchor" not in columns_in(IDENTIFIES, "tc5_auth")

    # It still counts as personal, because every candidate it can hold is.
    assert personal_columns(["chain_anchor", "window_id"]) == ["chain_anchor"]


def test_an_ambiguous_column_is_not_guessed_at():
    with pytest.raises(ValueError, match="different things"):
        classify("entity_key")

    with pytest.raises(ValueError, match="no recorded meaning"):
        classify("entity_key", "tc9_imaginary")


def test_an_unclassified_column_raises_rather_than_defaulting():
    # Defaulting to operational is the dangerous direction: it would let an unconsidered column through a policy
    # that names every personal category.
    with pytest.raises(KeyError):
        classify("some_new_feature")


def test_the_profile_the_design_manufactures_is_the_largest_category():
    # Worth asserting rather than assuming. The columns an estate collects are a minority of what it ends up
    # holding about a person; most of it is derived here, which is the part nobody thinks to ask about.
    sizes = {category: len(columns_in(category)) for category in CATEGORIES}

    assert sizes[PROFILES] > sizes[IDENTIFIES] + sizes[LOCATES]
    assert sizes[PROFILES] > 60


def test_the_counts_the_documents_quote_are_the_counts_here():
    # The top-level README and the guide's Part 6 both spell these out, because they are the most surprising
    # thing in the inventory and a reader takes them at face value. Pinning them here means a reclassification
    # fails a test rather than quietly making two documents wrong.
    assert [len(columns_in(category)) for category in (IDENTIFIES, ADDRESSES, LOCATES, PSEUDONYMS, PROFILES)] == [
        9, 17, 14, 8, 160
    ], "the category sizes have moved; README.md and the guide's Part 6 quote them and need the same edit"

    for name in ("README.md",
                 os.path.join("docs",
                              "source",
                              "developer_guide",
                              "guides",
                              "11_predictive_behavioral_analytics_osi.md")):
        with open(os.path.join(REPO_ROOT, name), encoding="utf-8") as handle:
            assert "a hundred and sixty are profile, a hundred and fifty-four of them behavioural" in re.sub(
                r"\s+", " ", handle.read()), name


def test_columns_in_resolves_the_ambiguous_names_for_a_class():
    assert "device_id" not in columns_in(OPERATIONAL)
    assert "device_id" in columns_in(OPERATIONAL, "tc1")
    assert "entity_key" in columns_in(IDENTIFIES, "tc5_auth")
    assert "entity_key" not in columns_in(IDENTIFIES, "tc1")


def test_columns_in_refuses_a_category_it_does_not_have():
    with pytest.raises(ValueError):
        columns_in("interesting")


def test_personal_columns_picks_out_what_a_frame_says_about_a_person():
    present = ["user_principal", "logcount", "window_id", "source_country", "collector_id"]

    assert personal_columns(present) == ["logcount", "source_country", "user_principal"]

    for column in personal_columns(present):
        assert classify(column) in PERSONAL_CATEGORIES


def test_a_pseudonym_is_stable_and_keyed():
    # Stable because every stateful stage keys per-entity history on the identifier, and an unstable pseudonym
    # would split one person's history into as many people as there were runs. Keyed because a bare digest of a
    # principal is recovered from the estate's own directory.
    once = pseudonymize(["alice@example.com"], KEY)
    again = pseudonymize(["alice@example.com"], KEY)
    other = pseudonymize(["alice@example.com"], b"a-different-key-entirely-here")

    assert once == again
    assert once != other
    assert once[0] != hashlib.sha256(b"alice@example.com").hexdigest()[:32]
    assert once[0] == hmac.new(KEY, b"alice@example.com", hashlib.sha256).hexdigest()[:32]


def test_a_pseudonym_preserves_a_join():
    # The property that makes this usable at all: the same person reached through two different column names
    # still joins to themselves afterwards.
    digests = pseudonymize(["alice@x", "bob@x", "alice@x"], KEY)

    assert digests[0] == digests[2]
    assert digests[0] != digests[1]


def test_a_missing_identifier_is_not_given_one():
    # Hashing a null would manufacture a value, and every row without an identifier would acquire one identity
    # in common -- a person who does not exist, with everybody else's unattributed activity.
    assert pseudonymize([None, "alice@x", float("nan")], KEY) == [None, pseudonymize(["alice@x"], KEY)[0], None]


def test_a_short_key_is_refused():
    with pytest.raises(ValueError, match="at least"):
        pseudonymize(["alice@x"], "short")


def test_a_key_of_the_wrong_type_is_refused():
    with pytest.raises(TypeError):
        pseudonymize(["alice@x"], 12345678901234567890)


def test_a_non_positive_digest_length_is_refused():
    with pytest.raises(ValueError):
        pseudonymize(["alice@x"], KEY, digest_length=0)


def test_the_bounded_domains_are_columns_we_classify():
    # A name here that nothing emits would be a refusal nobody could trigger, which reads as a protection and
    # is not one.
    unknown = sorted(BOUNDED_DOMAIN - set(COLUMNS) - set(AMBIGUOUS))

    assert unknown == []


def test_the_frequencies_give_a_bounded_domain_away():
    # Why `BOUNDED_DOMAIN` is a refusal rather than a warning, demonstrated rather than asserted in prose. The
    # key is secret and it does not help: the attacker never guesses a value, they count digests and read the
    # mapping off the shape of the distribution, which the pseudonym preserves exactly.
    countries = ["US"] * 80 + ["GB"] * 15 + ["FR"] * 5
    digests = pseudonymize(countries, KEY)

    ranked = sorted({digest: digests.count(digest) for digest in set(digests)}.items(), key=lambda pair: -pair[1])
    recovered = [digest for (digest, _) in ranked]

    assert [digests[countries.index(country)] for country in ("US", "GB", "FR")] == recovered
