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

import datetime
import logging

import pandas as pd
import pytest

from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.binding_table import Binding
from morpheus.utils.binding_table import BindingTable
from morpheus.utils.binding_table import to_epoch_ns

HOUR_NS = 3600 * NS_PER_SECOND

# Two consecutive DHCP leases on one address, plus an unrelated address.
LEASES = pd.DataFrame({
    "ip": ["10.0.0.5", "10.0.0.5", "10.0.0.9"],
    "mac": ["aa:bb:cc:00:00:01", "aa:bb:cc:00:00:02", "aa:bb:cc:00:00:03"],
    "hostname": ["laptop-a", "laptop-b", "printer"],
    "bind_start": [0, 2 * HOUR_NS, 0],
    "bind_end": [HOUR_NS, 3 * HOUR_NS, 4 * HOUR_NS],
})


def build_leases(**kwargs) -> BindingTable:
    defaults = {
        "name": "dhcp_lease",
        "key_column": "ip",
        "value_columns": ["mac", "hostname"],
        "start_column": "bind_start",
        "end_column": "bind_end",
    }
    defaults.update(kwargs)

    frame = defaults.pop("df", LEASES)

    return BindingTable.from_dataframe(frame, **defaults)


@pytest.fixture(name="leases")
def leases_fixture() -> BindingTable:
    yield build_leases()


def test_to_epoch_ns_accepts_several_forms():
    assert to_epoch_ns(1_700_000_000, time_unit="s") == 1_700_000_000 * NS_PER_SECOND
    assert to_epoch_ns(1_700_000_000 * NS_PER_SECOND) == 1_700_000_000 * NS_PER_SECOND
    assert to_epoch_ns("1970-01-01T01:00:00") == HOUR_NS
    assert to_epoch_ns(datetime.datetime(1970, 1, 1, 1, 0, 0)) == HOUR_NS
    assert to_epoch_ns(pd.Timestamp("1970-01-01T01:00:00")) == HOUR_NS


def test_to_epoch_ns_normalizes_timezones():
    # Both name the same instant, so both must land on the same integer.
    assert to_epoch_ns("1970-01-01T01:00:00+00:00") == to_epoch_ns("1970-01-01T02:00:00+01:00")


def test_to_epoch_ns_nulls():
    assert to_epoch_ns(None) is None
    assert to_epoch_ns(float("nan")) is None
    assert to_epoch_ns(pd.NaT) is None


def test_to_epoch_ns_rejects_garbage():
    with pytest.raises(ValueError):
        to_epoch_ns("not-a-time")


def test_to_epoch_ns_honors_unit_for_numpy_scalars():
    # A DataFrame column yields NumPy scalars, which do not subclass int; the unit must still be applied.
    value = pd.Series([1_700_000_000], dtype="int64").iloc[0]

    assert to_epoch_ns(value, time_unit="s") == 1_700_000_000 * NS_PER_SECOND


def test_table_shape(leases: BindingTable):
    assert leases.name == "dhcp_lease"
    assert leases.value_columns == ["mac", "hostname"]
    assert leases.size == 3
    assert leases.key_count == 2
    assert leases.overlapping_key_count == 0


def test_resolve_picks_the_covering_interval(leases: BindingTable):
    assert leases.resolve("10.0.0.5", HOUR_NS // 2).values == ("aa:bb:cc:00:00:01", "laptop-a")
    assert leases.resolve("10.0.0.5", int(2.5 * HOUR_NS)).values == ("aa:bb:cc:00:00:02", "laptop-b")


def test_resolve_is_half_open(leases: BindingTable):
    assert leases.resolve("10.0.0.5", 0) is not None
    # The instant the first lease ends belongs to no lease; the second has not started.
    assert leases.resolve("10.0.0.5", HOUR_NS) is None
    assert leases.resolve("10.0.0.5", 2 * HOUR_NS) is not None
    assert leases.resolve("10.0.0.5", 3 * HOUR_NS) is None


def test_resolve_gap_between_leases_is_unresolved(leases: BindingTable):
    # This is the property that matters: an address with no lease resolves to nothing rather than to whoever held it
    # most recently.
    assert leases.resolve("10.0.0.5", int(1.5 * HOUR_NS)) is None


def test_resolve_unknown_key_and_null_inputs(leases: BindingTable):
    assert leases.resolve("192.0.2.1", HOUR_NS // 2) is None
    assert leases.resolve(None, HOUR_NS // 2) is None
    assert leases.resolve("10.0.0.5", None) is None


def test_resolve_coerces_the_key():
    table = BindingTable("vlan", ["site"], [Binding("42", 0, HOUR_NS, ("hq", ), "u1")])

    assert table.resolve(42, 1).values == ("hq", )


def test_overlapping_intervals_resolve_to_the_later_start(caplog: pytest.LogCaptureFixture):
    overlapping = pd.DataFrame({
        "ip": ["10.0.0.5", "10.0.0.5"],
        "mac": ["aa:bb:cc:00:00:01", "aa:bb:cc:00:00:02"],
        "hostname": ["stale", "current"],
        "bind_start": [0, HOUR_NS],
        "bind_end": [4 * HOUR_NS, 3 * HOUR_NS],
    })

    # The configured `morpheus` logger does not propagate to the root logger, where caplog listens, so the handler is
    # attached to the emitting logger directly.
    module_logger = logging.getLogger("morpheus.utils.binding_table")
    module_logger.addHandler(caplog.handler)

    try:
        table = build_leases(df=overlapping)
    finally:
        module_logger.removeHandler(caplog.handler)

    assert table.overlapping_key_count == 1
    assert "overlapping intervals on 1 of 1 keys" in caplog.text
    # Both cover this instant. The more recent start wins, which is the documented rule.
    assert table.resolve("10.0.0.5", 2 * HOUR_NS).values == ("aa:bb:cc:00:00:02", "current")
    # Only the earlier binding is open here.
    assert table.resolve("10.0.0.5", int(3.5 * HOUR_NS)).values == ("aa:bb:cc:00:00:01", "stale")


def test_overlapping_resolution_is_independent_of_row_order():
    rows = pd.DataFrame({
        "ip": ["10.0.0.5"] * 3,
        "mac": ["m1", "m2", "m3"],
        "hostname": ["h1", "h2", "h3"],
        "bind_start": [0, HOUR_NS, HOUR_NS],
        "bind_end": [4 * HOUR_NS, 3 * HOUR_NS, 2 * HOUR_NS],
    })

    forward = build_leases(df=rows)
    reversed_rows = build_leases(df=rows.iloc[::-1].reset_index(drop=True))

    for instant in (0, HOUR_NS, int(1.5 * HOUR_NS), int(2.5 * HOUR_NS), int(3.5 * HOUR_NS)):
        assert forward.resolve("10.0.0.5", instant) == reversed_rows.resolve("10.0.0.5", instant)


def test_long_interval_behind_short_ones_is_still_found():
    # Exercises the bounded backward scan: the covering binding is not the nearest one by start.
    rows = pd.DataFrame({
        "ip": ["10.0.0.5"] * 3,
        "mac": ["long", "short-a", "short-b"],
        "hostname": ["h", "h", "h"],
        "bind_start": [0, HOUR_NS, 2 * HOUR_NS],
        "bind_end": [10 * HOUR_NS, HOUR_NS + 1, 2 * HOUR_NS + 1],
    })

    table = build_leases(df=rows)

    assert table.resolve("10.0.0.5", 5 * HOUR_NS).values == ("long", "h")


def test_datetime_columns():
    frame = pd.DataFrame({
        "ip": ["10.0.0.5"],
        "mac": ["aa"],
        "hostname": ["h"],
        "bind_start": pd.to_datetime(["2026-01-01T00:00:00"]),
        "bind_end": pd.to_datetime(["2026-01-01T01:00:00"]),
    })

    table = build_leases(df=frame)

    assert table.resolve("10.0.0.5", to_epoch_ns("2026-01-01T00:30:00")) is not None
    assert table.resolve("10.0.0.5", to_epoch_ns("2026-01-01T01:30:00")) is None


def test_second_resolution_columns():
    frame = pd.DataFrame({"ip": ["10.0.0.5"], "mac": ["aa"], "hostname": ["h"], "bind_start": [100], "bind_end": [200]})

    table = build_leases(df=frame, time_unit="s")

    assert table.resolve("10.0.0.5", 150 * NS_PER_SECOND) is not None
    assert table.resolve("10.0.0.5", 150) is None


def test_open_ended_binding_is_rejected_by_default():
    frame = pd.DataFrame({"ip": ["10.0.0.5"], "mac": ["aa"], "hostname": ["h"], "bind_start": [0], "bind_end": [None]})

    with pytest.raises(ValueError, match="no end"):
        build_leases(df=frame)


def test_open_ended_binding_can_be_capped():
    frame = pd.DataFrame({"ip": ["10.0.0.5"], "mac": ["aa"], "hostname": ["h"], "bind_start": [0], "bind_end": [None]})

    table = build_leases(df=frame, open_end_duration_ns=HOUR_NS)

    assert table.resolve("10.0.0.5", HOUR_NS - 1) is not None
    assert table.resolve("10.0.0.5", HOUR_NS) is None


def test_open_end_duration_must_be_positive():
    with pytest.raises(ValueError):
        build_leases(open_end_duration_ns=0)


def test_inverted_interval_is_rejected():
    frame = pd.DataFrame({
        "ip": ["10.0.0.5"], "mac": ["aa"], "hostname": ["h"], "bind_start": [HOUR_NS], "bind_end": [0]
    })

    with pytest.raises(ValueError, match="ends at or before"):
        build_leases(df=frame)


def test_null_key_is_rejected():
    frame = pd.DataFrame({"ip": [None], "mac": ["aa"], "hostname": ["h"], "bind_start": [0], "bind_end": [HOUR_NS]})

    with pytest.raises(ValueError, match="non-null key"):
        build_leases(df=frame)


def test_null_start_is_rejected():
    frame = pd.DataFrame({
        "ip": ["10.0.0.5"], "mac": ["aa"], "hostname": ["h"], "bind_start": [None], "bind_end": [HOUR_NS]
    })

    with pytest.raises(ValueError, match="null start"):
        build_leases(df=frame)


def test_missing_column_raises():
    with pytest.raises(KeyError, match="switch_id"):
        build_leases(value_columns=["mac", "switch_id"])


def test_constructor_validation():
    with pytest.raises(ValueError):
        BindingTable("", ["mac"], [])

    with pytest.raises(ValueError):
        BindingTable("t", [], [])

    with pytest.raises(ValueError):
        BindingTable("t", ["mac"], [Binding("k", 0, 1, ("a", "b"), "u")])


def test_uid_is_content_addressed_and_stable():
    first = build_leases()
    second = build_leases()

    assert first.resolve("10.0.0.5", 1).uid == second.resolve("10.0.0.5", 1).uid
    assert first.resolve("10.0.0.5", 1).uid != first.resolve("10.0.0.9", 1).uid


def test_uid_changes_with_the_binding():
    shifted = LEASES.copy()
    shifted.loc[0, "hostname"] = "renamed"

    assert build_leases().resolve("10.0.0.5", 1).uid != build_leases(df=shifted).resolve("10.0.0.5", 1).uid


def test_resolve_many_matches_resolve(leases: BindingTable):
    keys = ["10.0.0.5", "10.0.0.5", "10.0.0.9", "192.0.2.1"]
    times = [HOUR_NS // 2, int(2.5 * HOUR_NS), HOUR_NS, HOUR_NS]

    assert leases.resolve_many(keys, times) == [leases.resolve(k, t) for (k, t) in zip(keys, times)]


def test_resolve_many_memoization_is_transparent(leases: BindingTable):
    results = leases.resolve_many(["10.0.0.5"] * 20, [HOUR_NS // 2] * 20)

    assert len(results) == 20
    assert len({binding.uid for binding in results}) == 1


def test_resolve_many_requires_equal_lengths(leases: BindingTable):
    with pytest.raises(ValueError):
        leases.resolve_many(["10.0.0.5"], [1, 2])


def test_resolve_many_empty(leases: BindingTable):
    assert len(leases.resolve_many([], [])) == 0


def test_bucketed_records_cover_the_interval(leases: BindingTable):
    records = leases.to_bucketed_records(bucket_seconds=1800, key_name="ip")

    laptop_a = [r for r in records if r["hostname"] == "laptop-a"]

    # A one hour lease starting at the epoch touches buckets 0 and 1 at half hour granularity.
    assert [r["bucket"] for r in laptop_a] == [0, 1]
    assert all(r["ip"] == "10.0.0.5" for r in laptop_a)
    assert all(r["mac"] == "aa:bb:cc:00:00:01" for r in laptop_a)
    assert all("binding_uid" in r for r in records)


def test_bucketed_records_are_single_valued_per_key_and_bucket():
    overlapping = pd.DataFrame({
        "ip": ["10.0.0.5", "10.0.0.5"],
        "mac": ["stale", "current"],
        "hostname": ["a", "b"],
        "bind_start": [0, HOUR_NS],
        "bind_end": [4 * HOUR_NS, 3 * HOUR_NS],
    })

    records = build_leases(df=overlapping).to_bucketed_records(bucket_seconds=3600, key_name="ip")
    slots = [(r["ip"], r["bucket"]) for r in records]

    # A SIEM lookup returns one row per key, so the expansion must not emit two.
    assert len(slots) == len(set(slots))
    # Buckets 1 and 2 are covered by both; the later start wins, matching resolve().
    by_bucket = {r["bucket"]: r["mac"] for r in records}
    assert by_bucket == {0: "stale", 1: "current", 2: "current", 3: "stale"}


def test_bucketed_records_are_sorted(leases: BindingTable):
    records = leases.to_bucketed_records(bucket_seconds=1800, key_name="ip")
    slots = [(r["ip"], r["bucket"]) for r in records]

    assert slots == sorted(slots)


def test_bucketed_records_can_omit_the_uid(leases: BindingTable):
    records = leases.to_bucketed_records(bucket_seconds=1800, include_uid=False)

    assert all("binding_uid" not in r for r in records)
    assert all("key" in r for r in records)


def test_bucketed_records_reject_a_non_positive_bucket(leases: BindingTable):
    with pytest.raises(ValueError):
        leases.to_bucketed_records(bucket_seconds=0)


def test_bucketed_records_guard_against_explosion(leases: BindingTable):
    with pytest.raises(ValueError, match="over the limit"):
        leases.to_bucketed_records(bucket_seconds=1, max_buckets_per_binding=100)


def test_bucketed_frame(leases: BindingTable):
    frame = leases.to_bucketed_frame(bucket_seconds=1800, key_name="ip")

    # `bucket_start` sits with the key and the bucket because it is the row's identity in time, not one of the
    # binding's values: a bucketed row's only timestamp is the bucket it stands for.
    assert list(frame.columns) == ["ip", "bucket", "bucket_start", "mac", "hostname", "binding_uid"]
    assert len(frame) == len(leases.to_bucketed_records(bucket_seconds=1800, key_name="ip"))


def test_to_epoch_ns_understands_snmp_timeticks():
    # sysUpTime and ifLastChange are TimeTicks, hundredths of a second. One day is 8,640,000 of them.
    assert to_epoch_ns(8_640_000, time_unit="cs") == 24 * HOUR_NS
    assert to_epoch_ns(pd.Series([8_640_000], dtype="int64").iloc[0], time_unit="cs") == 24 * HOUR_NS


def test_tie_break_direction_is_pinned():
    # Two bindings identical in every respect but their attributes. The choice must be a function of the data, and
    # the documented direction, the greater tuple compared as strings, is the one the code actually takes. This
    # existed undocumented and the docstring said the opposite.
    lower = Binding(key="k", start_ns=0, end_ns=100, values=("aaa", ), uid="1")
    higher = Binding(key="k", start_ns=0, end_ns=100, values=("zzz", ), uid="2")

    assert BindingTable("t", ["v"], [lower, higher]).resolve("k", 50).values == ("zzz", )
    assert BindingTable("t", ["v"], [higher, lower]).resolve("k", 50).values == ("zzz", )


def test_the_tie_break_does_not_depend_on_how_the_input_was_batched():
    # The tie-break exists so the winner is a function of the data rather than of input order. Rendering attributes
    # with bare `str` broke exactly that: the same attribute arriving as 10 in one batch and 10.0 in the next --
    # which is all it takes for one row elsewhere in the batch to be null -- ordered differently and could pick a
    # different winner.
    from morpheus.utils.binding_table import _sort_key

    def binding(vlan):
        return Binding(key="10.0.0.1", start_ns=0, end_ns=NS_PER_SECOND, values=(vlan, "a"), uid="u")

    assert _sort_key(binding(10)) == _sort_key(binding(10.0))

    # A missing attribute has to stay comparable with a present one, or the sort raises rather than choosing.
    assert _sort_key(binding(None)) < _sort_key(binding(10))
