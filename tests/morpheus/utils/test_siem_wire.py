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
The event-time contract between the pipeline and the shipped Splunk app.

These tests read the app's actual `props.conf` rather than a copy of its values, so the Python rendering and the
Splunk parsing configuration cannot drift apart. The drift they exist to prevent is silent and severe: an
`event_time` Splunk cannot parse does not raise, it falls back to index time, and every windowed rule quietly
becomes a rule about when the pipeline was busy.
"""

import configparser
import datetime
import os
import re

import pandas as pd
import pytest

from morpheus.io import serializers
from morpheus.utils.siem_wire import SPLUNK_TIME_FORMAT
from morpheus.utils.binding_table import TABLE_NAME_COLUMN
from morpheus.utils.binding_table import BindingTable
from morpheus.utils.siem_wire import render_event_time
from morpheus.utils.siem_wire import render_event_time_series

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PROPS_PATH = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "TA-morpheus-lineage", "default", "props.conf")

SAVEDSEARCHES_PATH = PROPS_PATH.replace("props.conf", "savedsearches.conf")
# A timestamp with a non-zero microsecond component and sub-microsecond digits that must be discarded.
SAMPLE_NS = 1788114300123456789
SAMPLE_RENDERED = "2026-08-30T18:25:00.123456UTC"


def load_props() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(PROPS_PATH)

    return parser


def splunk_format_to_strptime(time_format: str) -> str:
    """Translate a Splunk `TIME_FORMAT` to a `strptime` format: `%<n>N` subseconds become `%f`."""
    return re.sub(r"%\d?N", "%f", time_format)


def event_time_stanzas() -> list[str]:
    parser = load_props()

    return [name for name in parser.sections() if parser[name].get("TIME_PREFIX", "").startswith('"event_time"')]


def test_the_app_defines_event_time_stanzas():
    # Guards the other tests: if the stanza naming changes, they must not silently pass over an empty list.
    stanzas = event_time_stanzas()

    assert len(stanzas) >= 8, stanzas
    assert "morpheus:edge" in stanzas


@pytest.mark.parametrize("stanza", event_time_stanzas())
def test_every_event_time_stanza_uses_the_shared_format(stanza: str):
    settings = load_props()[stanza]

    assert settings["TIME_FORMAT"] == SPLUNK_TIME_FORMAT
    # The trailing quote is what requires event_time to be a JSON string rather than a bare number.
    assert settings["TIME_PREFIX"].endswith('"')


def test_rendering_matches_the_shipped_prefix_and_format():
    # The whole contract in one assertion chain: render, serialize exactly as the Kafka sink does, then apply the
    # app's own TIME_PREFIX and TIME_FORMAT to what lands on the wire.
    settings = load_props()["morpheus:score:l3"]
    line = serializers.df_to_json(pd.DataFrame({
        "event_time": [render_event_time(SAMPLE_NS)], "src_ip": ["10.0.1.10"]
    }),
                                  strip_newlines=True)[0]

    match = re.search(settings["TIME_PREFIX"], line)

    assert match is not None, f"TIME_PREFIX did not match the serialized event: {line}"

    remainder = line[match.end():]
    parsed = datetime.datetime.strptime(remainder[:len(SAMPLE_RENDERED)],
                                        splunk_format_to_strptime(settings["TIME_FORMAT"]))

    assert parsed == datetime.datetime(2026, 8, 30, 18, 25, 0, 123456)


def test_unrendered_nanoseconds_do_not_match():
    # The negative control, and the defect this contract exists to prevent: an integer event_time serializes
    # without the quote the prefix requires, Splunk matches nothing, and _time silently becomes index time.
    settings = load_props()["morpheus:score:l3"]
    line = serializers.df_to_json(pd.DataFrame({
        "event_time": [SAMPLE_NS], "src_ip": ["10.0.1.10"]
    }),
                                  strip_newlines=True)[0]

    assert re.search(settings["TIME_PREFIX"], line) is None


def test_prefix_tolerates_pretty_printed_json():
    settings = load_props()["morpheus:score:l3"]

    assert re.search(settings["TIME_PREFIX"], f'{{"event_time": "{SAMPLE_RENDERED}"}}') is not None


def test_render_truncates_rather_than_rounds():
    # Rounding could move an event across a window boundary it never crossed. Every value from the microsecond
    # boundary through the last nanosecond before the next one must render as the same microsecond.
    exact_microsecond = 1788114300123456000

    assert render_event_time(exact_microsecond) == SAMPLE_RENDERED
    assert render_event_time(SAMPLE_NS) == SAMPLE_RENDERED
    assert render_event_time(exact_microsecond + 999) == SAMPLE_RENDERED
    assert render_event_time(exact_microsecond + 1000) == "2026-08-30T18:25:00.123457UTC"


def test_render_is_precise_above_the_float_mantissa():
    # Nanosecond epochs exceed 2**53, so any implementation routing through float seconds loses microseconds.
    late = 4102444800_000001000  # 2100-01-01T00:00:00.000001Z

    assert render_event_time(late) == "2100-01-01T00:00:00.000001UTC"


def test_render_accepts_the_other_forms_to_epoch_ns_accepts():
    assert render_event_time("2026-08-30T18:25:00.123456") == SAMPLE_RENDERED
    assert render_event_time(datetime.datetime(2026, 8, 30, 18, 25, 0, 123456)) == SAMPLE_RENDERED
    assert render_event_time(1788114300, time_unit="s") == "2026-08-30T18:25:00.000000UTC"


def test_render_is_utc_regardless_of_host_zone():
    # A local-time rendering would make _time depend on which host ran the pipeline.
    assert render_event_time(0) == "1970-01-01T00:00:00.000000UTC"


def test_render_passes_nulls_through():
    assert render_event_time(None) is None
    assert render_event_time_series([None, SAMPLE_NS]) == [None, SAMPLE_RENDERED]


def timestamped_stanzas() -> list[str]:
    """Every stanza that anchors `_time` on a field, whatever that field is called."""
    parser = load_props()

    return [name for name in parser.sections() if parser[name].get("TIME_PREFIX")]


# The field each sourcetype's TIME_PREFIX names, and what in this repo is responsible for emitting it. A stanza
# anchored on a field its producer never writes is silently severe: Splunk falls back to ingest time, or inherits the
# previous event's timestamp, and the row lands at a time that looks plausible and is wrong.
TIMESTAMP_FIELD_OWNERS = {
    "event_time": "morpheus.utils.siem_wire.render_event_time_series, before the sink",
    "bind_start": "TC2BindingStage, on every closed and provisional binding record",
    "bind_end": "TC2BindingStage, on closed binding records",
    "bucket_start": "BindingTable.to_bucketed_records, on every bucketed row",
    # The TC-0 context store is in Part 6's "Must be built" table. These two stanzas are the contract it will have to
    # meet, deliberately shipped ahead of it. Listing the field here keeps this test honest about the difference
    # between "nothing emits it yet, by design" and "nothing will ever emit it, by mistake".
    "valid_from": "not built: the bitemporal TC-0 context store, per Part 6",
}

UNBUILT_TIMESTAMP_FIELDS = {"valid_from"}


@pytest.mark.parametrize("stanza", timestamped_stanzas())
def test_every_stanza_is_timed_on_a_field_something_actually_emits(stanza: str):
    settings = load_props()[stanza]
    field = re.match(r'"([a-z_]+)"', settings["TIME_PREFIX"])

    assert field is not None, f"{stanza} has a TIME_PREFIX this test cannot read: {settings['TIME_PREFIX']}"
    name = field.group(1)

    assert name in TIMESTAMP_FIELD_OWNERS, (
        f"{stanza} anchors _time on {name!r}, which nothing in this repo is known to emit. Either the producer "
        f"gained the field and this map needs it, or the stanza is timed on a field that will never arrive.")

    # Whatever the field is called, it is rendered by the one shared renderer, so the format must be the shared one.
    assert settings["TIME_FORMAT"] == SPLUNK_TIME_FORMAT
    assert settings["TIME_PREFIX"].endswith('"')


def test_the_unbuilt_timestamp_fields_are_still_unbuilt():
    # A tripwire on the exemption above: when the TC-0 store lands and starts writing `valid_from`, this fails and
    # the field moves from the unbuilt set to an owner, rather than the exemption quietly outliving its reason.
    for field in UNBUILT_TIMESTAMP_FIELDS:
        assert "not built" in TIMESTAMP_FIELD_OWNERS[field]


def test_the_bucketed_stanza_is_timed_on_the_field_the_expansion_writes():
    # The specific pairing that was wrong: bucketed rows carry no bind_start, only the bucket's own start.
    settings = load_props()["binding:bucketed"]

    assert '"bucket_start"' in settings["TIME_PREFIX"]

    table = BindingTable.from_dataframe(
        pd.DataFrame({
            "key": ["10.0.0.5"],
            "bind_start": [SAMPLE_NS],
            "bind_end": [SAMPLE_NS + 600 * 1_000_000_000],
            "mac": ["aa:bb:cc:dd:ee:ff"],
        }),
        "dhcp",
        "key", ["mac"],
        "bind_start",
        "bind_end")
    record = table.to_bucketed_records(bucket_seconds=300, key_name="ip")[0]

    assert "bucket_start" in record
    line = serializers.df_to_json(pd.DataFrame([record]), strip_newlines=True)[0]
    match = re.search(settings["TIME_PREFIX"], line)

    assert match is not None
    stamp = re.match(r'[^"]+', line[match.end():]).group(0)
    parsed = datetime.datetime.strptime(stamp, splunk_format_to_strptime(settings["TIME_FORMAT"]).replace("%Z", "UTC"))
    epoch_seconds = (parsed - datetime.datetime(1970, 1, 1)).total_seconds()

    # The bucket's start, not the binding's: the row stands for the whole bucket, so its time is a bucket boundary.
    assert epoch_seconds % 300 == 0
    assert epoch_seconds == (SAMPLE_NS // 1_000_000_000) // 300 * 300


def test_the_bucketed_expansion_can_name_its_source_table():
    # The lookup refresh discriminates on `binding_table`; several sources share the sourcetype.
    table = BindingTable.from_dataframe(
        pd.DataFrame({
            "key": ["10.0.0.5"], "bind_start": [0], "bind_end": [600 * 1_000_000_000], "mac": ["aa:bb:cc:dd:ee:ff"]
        }),
        "dhcp",
        "key", ["mac"],
        "bind_start",
        "bind_end")

    named = table.to_bucketed_records(bucket_seconds=300, key_name="ip", table_name="dhcp_lease")
    assert {record["binding_table"] for record in named} == {"dhcp_lease"}

    # Unnamed tables carry no such field, so a single-source estate is not made to invent one.
    assert all("binding_table" not in record for record in table.to_bucketed_records(bucket_seconds=300))


def test_the_lookup_refresh_filters_on_a_field_the_expansion_can_emit():
    # The search reads `binding_table=dhcp_lease`; the expansion must be able to write exactly that field name.
    with open(SAVEDSEARCHES_PATH, encoding="utf-8") as handle:
        searches = handle.read()

    assert "binding_table=dhcp_lease" in searches
    assert TABLE_NAME_COLUMN == "binding_table"
