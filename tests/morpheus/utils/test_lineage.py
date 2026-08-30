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

import hashlib
import random

import numpy as np
import pytest

from morpheus.utils.lineage import DEFAULT_DIGEST_LENGTH
from morpheus.utils.lineage import UNIT_SEPARATOR
from morpheus.utils.lineage import event_uid
from morpheus.utils.lineage import event_uid_series
from morpheus.utils.lineage import lineage_id
from morpheus.utils.lineage import link_uid
from morpheus.utils.lineage import link_uid_series
from morpheus.utils.lineage import merkle_root
from morpheus.utils.lineage import window_id_from_timestamp


def test_event_uid_is_sha256_of_separated_fields():
    parts = ("collector-a", "TC-5/2.1.0", "abc123", "42")
    expected = hashlib.sha256(UNIT_SEPARATOR.join(parts).encode("utf-8")).hexdigest()[:DEFAULT_DIGEST_LENGTH]

    assert event_uid(*parts) == expected


def test_event_uid_is_stable():
    results = {event_uid("collector-a", "TC-5/2.1.0", "abc123", 42) for _ in range(100)}

    assert len(results) == 1


def test_event_uid_separator_prevents_field_boundary_collisions():
    # Without a delimiter these two field sets concatenate identically.
    assert event_uid("ab", "c") != event_uid("a", "bc")


def test_event_uid_is_order_sensitive():
    assert event_uid("a", "b") != event_uid("b", "a")


def test_event_uid_coerces_non_strings():
    assert event_uid(1, 2) == event_uid("1", "2")


def test_event_uid_requires_a_field():
    with pytest.raises(ValueError):
        event_uid()


@pytest.mark.parametrize("digest_length", [1, 8, 16, 32, 64])
def test_event_uid_digest_length(digest_length: int):
    result = event_uid("a", "b", digest_length=digest_length)

    assert len(result) == digest_length


@pytest.mark.parametrize("digest_length", [0, -1, 65])
def test_event_uid_rejects_bad_digest_length(digest_length: int):
    with pytest.raises(ValueError):
        event_uid("a", "b", digest_length=digest_length)


def test_link_uid_depends_on_every_field():
    base = link_uid("parent", "child", "carried_by", "hard:flow_id")

    assert base != link_uid("other", "child", "carried_by", "hard:flow_id")
    assert base != link_uid("parent", "other", "carried_by", "hard:flow_id")
    assert base != link_uid("parent", "child", "other", "hard:flow_id")
    assert base != link_uid("parent", "child", "carried_by", "soft:dhcp_lease")


def test_link_uid_is_directional():
    assert link_uid("a", "b", "r", "m") != link_uid("b", "a", "r", "m")


def test_merkle_root_is_order_independent():
    uids = [event_uid("collector", str(i)) for i in range(9)]
    shuffled = list(uids)
    random.Random(1337).shuffle(shuffled)

    assert merkle_root(uids) == merkle_root(shuffled)


def test_merkle_root_is_duplicate_insensitive():
    uids = [event_uid("collector", str(i)) for i in range(5)]

    assert merkle_root(uids) == merkle_root(uids + uids)


def test_merkle_root_changes_with_membership():
    uids = [event_uid("collector", str(i)) for i in range(5)]

    assert merkle_root(uids) != merkle_root(uids[:-1])
    assert merkle_root(uids) != merkle_root(uids + [event_uid("collector", "extra")])


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 64, 65])
def test_merkle_root_shape(count: int):
    uids = [event_uid("collector", str(i)) for i in range(count)]
    root = merkle_root(uids)

    assert len(root) == 64
    assert int(root, 16) >= 0


def test_merkle_root_single_leaf_is_domain_separated():
    # A one element tree must be the leaf digest, not the raw identifier, so a leaf can never be mistaken for a node.
    uid = event_uid("collector", "1")
    expected = hashlib.sha256(b"\x00" + uid.encode("utf-8")).hexdigest()

    assert merkle_root([uid]) == expected


def test_merkle_root_promotes_the_odd_node():
    # An odd trailing node is carried up unchanged rather than paired with a copy of itself. Duplicating it would make
    # a tree of N leaves indistinguishable from one of N+1 whose last two leaves are equal.
    uids = sorted({event_uid("c", str(i)) for i in range(3)})
    leaves = [hashlib.sha256(b"\x00" + uid.encode("utf-8")).digest() for uid in uids]

    expected = hashlib.sha256(b"\x01" + hashlib.sha256(b"\x01" + leaves[0] + leaves[1]).digest() +
                              leaves[2]).hexdigest()

    assert merkle_root(uids) == expected


def test_lineage_id_depends_on_every_field():
    root = merkle_root([event_uid("c", "1")])
    base = lineage_id("jdoe", 484512, root)

    assert base != lineage_id("asmith", 484512, root)
    assert base != lineage_id("jdoe", 484513, root)
    assert base != lineage_id("jdoe", 484512, merkle_root([event_uid("c", "2")]))


@pytest.mark.parametrize("event_time_ns, period_ns, epoch_ns, expected",
                         [
                             (0, 3600 * 10**9, 0, 0),
                             (3599 * 10**9, 3600 * 10**9, 0, 0),
                             (3600 * 10**9, 3600 * 10**9, 0, 1),
                             (7200 * 10**9, 3600 * 10**9, 0, 2),
                             (-1, 3600 * 10**9, 0, -1),
                             (3600 * 10**9, 3600 * 10**9, 1800 * 10**9, 0),
                             (5400 * 10**9, 3600 * 10**9, 1800 * 10**9, 1),
                         ])
def test_window_id_from_timestamp(event_time_ns: int, period_ns: int, epoch_ns: int, expected: int):
    assert window_id_from_timestamp(event_time_ns, period_ns, epoch_ns=epoch_ns) == expected


def test_window_id_boundaries_are_half_open():
    period = 60 * 10**9

    assert window_id_from_timestamp(period - 1, period) == 0
    assert window_id_from_timestamp(period, period) == 1


def test_window_id_rejects_non_positive_period():
    with pytest.raises(ValueError):
        window_id_from_timestamp(0, 0)


def test_event_uid_series_matches_scalar():
    collectors = ["a", "b", "c"]
    seqs = [1, 2, 3]

    assert event_uid_series([collectors, seqs]) == [event_uid(c, s) for (c, s) in zip(collectors, seqs)]


def test_event_uid_series_requires_equal_lengths():
    with pytest.raises(ValueError):
        event_uid_series([["a", "b"], [1]])


def test_event_uid_series_requires_a_column():
    with pytest.raises(ValueError):
        event_uid_series([])


def test_event_uid_series_empty_rows():
    assert len(event_uid_series([[], []])) == 0


def test_link_uid_series_matches_scalar():
    parents = ["p1", "p2"]
    children = ["c1", "c2"]

    expected = [link_uid(p, c, "carried_by", "hard:flow_id") for (p, c) in zip(parents, children)]

    assert link_uid_series(parents, children, "carried_by", "hard:flow_id") == expected


@pytest.mark.parametrize("empty_parent", [None, "", np.nan])
def test_link_uid_series_treats_missing_parent_as_chain_root(empty_parent):
    results = link_uid_series([empty_parent, "p2"], ["c1", "c2"], "carried_by", "hard:flow_id")

    assert results[0] is None
    assert results[1] == link_uid("p2", "c2", "carried_by", "hard:flow_id")


def test_link_uid_series_requires_equal_lengths():
    with pytest.raises(ValueError):
        link_uid_series(["p1"], ["c1", "c2"], "r", "m")
