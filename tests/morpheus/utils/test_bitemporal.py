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

import itertools

import pytest

from morpheus.utils import bitemporal
from morpheus.utils.bitemporal import ASSERT
from morpheus.utils.bitemporal import MEMBERSHIP
from morpheus.utils.bitemporal import RETRACT
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version

DAY = 86400 * 10**9


def profile(entity, department, valid_from, recorded, valid_to=None, change=ASSERT):
    return make_version("profile", entity, valid_from, valid_to, recorded, change, {"department": department})


def member(entity, group, valid_from, recorded, valid_to=None, change=ASSERT):
    return make_version(MEMBERSHIP, entity, valid_from, valid_to, recorded, change, {"group_name": group}, (group, ))


def department_of(store, entity, valid_ns, known_ns=None):
    version = store.resolve(entity, valid_ns, known_ns)

    return None if version is None else version.attributes["department"]


# --- Versions ----------------------------------------------------------------------------------------------------


def test_a_version_is_content_addressed():
    first = profile("alice", "Finance", 0, DAY)

    assert first.uid == profile("alice", "Finance", 0, DAY).uid
    assert first.uid != profile("alice", "Finance", 0, 2 * DAY).uid
    assert first.uid != profile("alice", "Sales", 0, DAY).uid
    assert first.uid != profile("alice", "Finance", 0, DAY, valid_to=5 * DAY).uid


def test_the_valid_interval_is_half_open():
    version = profile("alice", "Finance", DAY, 0, valid_to=3 * DAY)

    assert not version.covers(DAY - 1)
    assert version.covers(DAY)
    assert version.covers(3 * DAY - 1)
    assert not version.covers(3 * DAY)
    assert profile("alice", "Finance", DAY, 0).covers(10**6 * DAY)


def test_a_membership_is_keyed_by_principal_and_group():
    version = member("alice", "finance-users", 0, 0)

    assert version.entity == "alice"
    assert version.key == "alice:finance-users"
    assert bitemporal.key_parts_of(version) == ["finance-users"]
    assert not bitemporal.key_parts_of(profile("alice", "Finance", 0, 0))


@pytest.mark.parametrize(
    "arguments, reason",
    [
        ((None, 0, None, 0, ASSERT), bitemporal.NO_ENTITY),
        (("  ", 0, None, 0, ASSERT), bitemporal.NO_ENTITY),
        (("alice", None, None, 0, ASSERT), bitemporal.NO_VALID_FROM),
        (("alice", 0, None, None, ASSERT), bitemporal.NO_RECORDED_AT),
        (("alice", DAY, DAY, 0, ASSERT), bitemporal.INVERTED_INTERVAL),
        (("alice", DAY, 0, 0, ASSERT), bitemporal.INVERTED_INTERVAL),
        (("alice", 0, None, 0, "update"), bitemporal.UNKNOWN_CHANGE),
        (("alice", 0, None, 0, ASSERT), None),
    ],
)
def test_refusal_reasons(arguments, reason):
    assert bitemporal.refusal_reason(*arguments) == reason


def test_a_version_without_a_recorded_instant_cannot_be_made():
    with pytest.raises(ValueError, match="no_recorded_at"):
        make_version("profile", "alice", 0, None, None)


# --- Resolution --------------------------------------------------------------------------------------------------


def test_a_correction_changes_the_present_view_of_the_past_and_not_the_past_view():
    store = BitemporalStore("t", [profile("carol", "Sales", 0, 0), profile("carol", "Marketing", 0, 20 * DAY)])

    assert department_of(store, "carol", 5 * DAY, known_ns=5 * DAY) == "Sales"
    assert department_of(store, "carol", 5 * DAY) == "Marketing"
    assert department_of(store, "carol", 5 * DAY, known_ns=20 * DAY) == "Marketing"
    assert department_of(store, "carol", 5 * DAY, known_ns=20 * DAY - 1) == "Sales"


def test_a_later_version_covering_part_of_an_earlier_one_leaves_the_rest_standing():
    store = BitemporalStore("t", [profile("bob", "Engineering", 0, 0), profile("bob", "Finance", 10 * DAY, 10 * DAY)])

    assert department_of(store, "bob", 5 * DAY) == "Engineering"
    assert department_of(store, "bob", 12 * DAY) == "Finance"


def test_a_retraction_ends_a_fact_from_its_valid_start():
    store = BitemporalStore("t",
                            [member("dave", "fin", 0, 0), member("dave", "fin", 12 * DAY, 18 * DAY, change=RETRACT)])

    assert store.resolve("dave:fin", 11 * DAY) is not None
    assert store.resolve("dave:fin", 14 * DAY) is None
    assert store.resolve("dave:fin", 14 * DAY, known_ns=14 * DAY) is not None
    assert store.winner("dave:fin", 14 * DAY).change == RETRACT


def test_nothing_recorded_by_the_horizon_is_nothing():
    store = BitemporalStore("t", [profile("alice", "Finance", 0, DAY)])

    assert store.resolve("alice", 0, known_ns=DAY - 1) is None
    assert store.winner("alice", 0, known_ns=DAY - 1) is None
    assert store.resolve("nobody", 0) is None


def test_resolution_does_not_depend_on_the_order_versions_arrive_in():
    versions = [
        profile("carol", "Sales", 0, 0),
        profile("carol", "Marketing", 0, 20 * DAY),
        profile("carol", "Support", 30 * DAY, 30 * DAY),
        profile("carol", "Sales", 0, 20 * DAY, valid_to=DAY),
    ]
    probes = [(valid, known) for valid in (0, 5 * DAY, 31 * DAY) for known in (None, 0, 20 * DAY, 30 * DAY)]
    expected = [department_of(BitemporalStore("t", versions), "carol", *probe) for probe in probes]

    for order in itertools.permutations(versions):
        store = BitemporalStore("t", order)

        assert [department_of(store, "carol", *probe) for probe in probes] == expected


def test_ties_are_broken_by_the_later_valid_start_then_the_later_end():
    # Recorded together: the one starting later wins where both cover, then the one ending later.
    store = BitemporalStore("t", [profile("erin", "A", 0, DAY), profile("erin", "B", DAY, DAY)])

    assert department_of(store, "erin", 2 * DAY) == "B"
    assert department_of(store, "erin", 0) == "A"

    store = BitemporalStore(
        "t", [profile("erin", "A", 0, DAY, valid_to=5 * DAY), profile("erin", "B", 0, DAY, valid_to=9 * DAY)])

    assert department_of(store, "erin", 2 * DAY) == "B"


def test_facts_about_an_entity_are_everything_that_held_sorted():
    store = BitemporalStore("t",
                            [
                                profile("alice", "Finance", 0, 0),
                                member("alice", "zeta", 0, 0),
                                member("alice", "alpha", 0, 0),
                                member("alice", "gone", 0, 0, valid_to=DAY),
                                member("bob", "alpha", 0, 0),
                            ])

    held = store.facts_about("alice", 2 * DAY)

    assert [version.key for version in held] == ["alice:alpha", "alice:zeta", "alice"]
    assert [version.kind for version in held] == [MEMBERSHIP, MEMBERSHIP, "profile"]
    assert store.facts_about(None, 0) == []


def test_a_key_cannot_change_kind():
    store = BitemporalStore("t", [profile("alice", "Finance", 0, 0)])

    with pytest.raises(ValueError, match="cannot also be"):
        store.add(make_version("asset", "alice", 0, None, 0, ASSERT, {"owner": "x"}))


def test_contradictions_are_counted_and_still_resolved():
    store = BitemporalStore("t", [profile("alice", "Finance", 0, DAY), profile("alice", "Sales", 0, DAY)])

    assert store.contradictions() == 1
    assert department_of(store, "alice", 0) == "Sales"

    agreeing = BitemporalStore("t", [profile("alice", "Finance", 0, DAY), profile("alice", "Finance", 0, 2 * DAY)])

    assert agreeing.contradictions() == 0


# --- Snapshots ---------------------------------------------------------------------------------------------------


def fact(entity, department_name, valid_from=0):
    return {"entity": entity, "valid_from_ns": valid_from, "values": {"department": department_name}}


def test_a_snapshot_against_an_empty_store_asserts_everything():
    changes = BitemporalStore("t").snapshot_changes("profile", [fact("a", "X"), fact("b", "Y")], DAY)

    assert [(version.key, version.change, version.recorded_ns) for version in changes] == [("a", ASSERT, DAY),
                                                                                           ("b", ASSERT, DAY)]


def test_a_snapshot_records_only_what_changed_and_retracts_what_it_omits():
    store = BitemporalStore("t", [profile("a", "X", 0, 0), profile("b", "Y", 0, 0), profile("c", "Z", 0, 0)])
    changes = store.snapshot_changes("profile", [fact("a", "X"), fact("b", "Moved"), fact("d", "New")], 5 * DAY)

    assert [(version.key, version.change) for version in changes] == [("b", ASSERT), ("c", RETRACT), ("d", ASSERT)]

    retraction = changes[1]

    assert retraction.valid_from_ns == 5 * DAY
    assert retraction.valid_to_ns is None


def test_a_snapshot_changing_only_an_effective_date_is_a_change():
    store = BitemporalStore("t", [profile("a", "X", 0, 0)])

    assert store.snapshot_changes("profile", [fact("a", "X")], DAY) == []
    assert len(store.snapshot_changes("profile", [fact("a", "X", valid_from=0)], DAY)) == 0
    assert len(store.snapshot_changes("profile", [{**fact("a", "X"), "valid_to_ns": 9 * DAY}], DAY)) == 1


def test_a_snapshot_never_retracts_another_kind_or_something_already_ended():
    store = BitemporalStore("t", [member("a", "g", 0, 0), profile("ended", "X", 0, 0, valid_to=DAY)])

    assert store.snapshot_changes("profile", [], 5 * DAY) == []


def test_a_snapshot_does_not_change_the_store():
    store = BitemporalStore("t", [profile("a", "X", 0, 0)])
    store.snapshot_changes("profile", [], DAY)

    assert store.size == 1


# --- Round trip through a producer's columns ----------------------------------------------------------------------


def test_columns_round_trip_into_an_identical_store():
    (columns, refused) = bitemporal.context_columns([MEMBERSHIP, "profile", "profile"], ["alice", "alice", None],
                                                    [("fin", ), (), ()], [{
                                                        "group_name": "fin"
                                                    }, {
                                                        "department": "Finance"
                                                    }, {}], [0, 0, 0], [None, 3 * DAY, None], [DAY, DAY, DAY],
                                                    [None, "RETRACT", None])

    assert refused == {bitemporal.NO_ENTITY: 1}
    assert columns[bitemporal.CHANGE] == [ASSERT, RETRACT, ASSERT]
    assert columns[bitemporal.CONTEXT_KEY] == ["alice:fin", "alice", None]
    assert columns[bitemporal.CONTEXT_ATTRIBUTES] == ["group_name", "department", None]

    records = [{name: values[row] for (name, values) in columns.items()} for row in range(3)]
    records[0]["group_name"] = "fin"
    records[1]["department"] = "Finance"

    store = BitemporalStore.from_records("t", records)

    assert store.size == 2
    assert {version.uid
            for version in store.history("alice:fin") + store.history("alice")
            } == {columns[bitemporal.CONTEXT_UID][0], columns[bitemporal.CONTEXT_UID][1]}


def test_a_record_altered_after_it_was_produced_is_rejected():
    (columns, _) = bitemporal.context_columns("profile", ["alice"], [()], [{
        "department": "Finance"
    }], [0], [None], [DAY], [None])
    record = {name: values[0] for (name, values) in columns.items()}
    record["department"] = "Sales"

    with pytest.raises(ValueError, match="does not reproduce its own identifier"):
        BitemporalStore.from_records("t", [record])


def test_unparsable_instants_are_refused_rather_than_raised():
    (columns, refused) = bitemporal.context_columns("profile", ["alice"], [()], [{
        "department": "Finance"
    }], ["2026-01-01T00:00:00Z"], [None], ["not a time"], [None])

    assert refused == {bitemporal.NO_RECORDED_AT: 1}
    assert columns[bitemporal.VALID_FROM][0] is not None
    assert columns[bitemporal.CONTEXT_UID] == [None]
