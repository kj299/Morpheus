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
from morpheus.utils.siem_wire import render_event_time
from morpheus.utils.siem_wire import render_event_time_series

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PROPS_PATH = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "TA-morpheus-lineage", "default", "props.conf")

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
