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
The stage that puts a parsable timestamp on the wire, asserted through the app's own parsing configuration.

The assertions here do not compare against a hard-coded rendering. They serialize exactly as the Kafka sink does,
apply the `TIME_PREFIX` regex and `TIME_FORMAT` read out of the shipped `props.conf`, and parse the result. A
rendering that satisfies these tests is one Splunk will read; a rendering that merely looks right is not enough,
which is the whole reason the defect this stage prevents went unnoticed for so long.
"""

import configparser
import datetime
import os
import re

import pandas as pd
import pytest

from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.io import serializers
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.stages.output.siem_wire_stage import SiemWireStage
from morpheus.utils.type_utils import get_df_class

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PROPS_PATH = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "TA-morpheus-lineage", "default", "props.conf")

# Sub-microsecond digits that must be truncated rather than rounded, so a rendering never moves an event.
SAMPLE_NS = 1788114300123456789
SAMPLE_TIME = datetime.datetime(2026, 8, 30, 18, 25, 0, 123456)
LATER_NS = SAMPLE_NS + 61 * 10**9

SCORE_L2_COLUMNS = {
    "event_uid": ["a", "b"],
    "port_key": ["site-1:sw1:Gi1/0/1", "site-1:sw1:Gi1/0/2"],
    "macs_per_port_first_in_window": [True, False],
    "macs_claiming_sender_ip": [1, 2],
    "arp_sender_ip_excluded": [False, False],
    "auth_unpaired": [False, True],
    "auth_port_key": ["site-1:sw1:Gi1/0/1", "site-1:sw1:Gi1/0/2"],
}


def load_props() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(PROPS_PATH)

    return parser


def splunk_format_to_strptime(time_format: str) -> str:
    """Translate a Splunk `TIME_FORMAT` to a `strptime` format: `%<n>N` subseconds become `%f`."""
    return re.sub(r"%\d?N", "%f", time_format)


def parse_as_splunk_would(line: str, stanza: str) -> datetime.datetime:
    """Apply one stanza's own TIME_PREFIX and TIME_FORMAT to a serialized record, the way Splunk does."""
    settings = load_props()[stanza]
    match = re.search(settings["TIME_PREFIX"], line)

    assert match is not None, f"TIME_PREFIX for {stanza} did not match the serialized event: {line}"

    remainder = line[match.end():]
    rendered = remainder[:remainder.index('"')]

    return datetime.datetime.strptime(rendered, splunk_format_to_strptime(settings["TIME_FORMAT"]))


def as_lines(df) -> list:
    host = df.to_pandas() if hasattr(df, "to_pandas") else df

    return serializers.df_to_json(host, strip_newlines=True)


def run_stage(config: Config, frame: dict, sourcetype: str, **kwargs):
    stage = SiemWireStage(config, sourcetype=sourcetype, **kwargs)
    meta = MessageMeta(get_df_class(config.execution_mode)(frame))

    return stage.on_data(meta).copy_dataframe()


@pytest.mark.gpu_and_cpu_mode
def test_the_rendered_event_time_is_what_splunk_parses(config: Config):
    # The contract in one chain: render, serialize as the sink does, then parse with the app's own configuration.
    result = run_stage(config, {"event_time": [SAMPLE_NS, LATER_NS], **SCORE_L2_COLUMNS}, "morpheus:score:l2")
    lines = as_lines(result)

    assert parse_as_splunk_would(lines[0], "morpheus:score:l2") == SAMPLE_TIME
    assert parse_as_splunk_would(lines[1], "morpheus:score:l2") == SAMPLE_TIME + datetime.timedelta(seconds=61)


@pytest.mark.gpu_and_cpu_mode
def test_without_the_stage_splunk_parses_nothing(config: Config):
    # The negative control, and the defect this stage exists to prevent. An integer event_time serializes without
    # the quote the prefix requires, Splunk matches nothing, and _time silently becomes index time.
    frame = get_df_class(config.execution_mode)({
        "event_time": [SAMPLE_NS], **{
            name: values[:1]
            for (name, values) in SCORE_L2_COLUMNS.items()
        }
    })
    settings = load_props()["morpheus:score:l2"]

    assert re.search(settings["TIME_PREFIX"], as_lines(frame)[0]) is None


@pytest.mark.gpu_and_cpu_mode
def test_every_time_column_is_rendered_not_only_the_anchor(config: Config):
    # binding:l2 anchors on bind_end, but a consumer reading bind_start off the same record should not find one
    # field a string and its sibling a nineteen-digit integer.
    result = run_stage(
        config,
        {
            "bind_end": [LATER_NS],
            "bind_start": [SAMPLE_NS],
            "event_time": [LATER_NS],
            "mac_address": ["aa:bb:cc:dd:ee:ff"],
            "port_key": ["site-1:sw1:Gi1/0/1"],
            "bind_end_reason": ["displaced"],
            "bind_gap_ns": [61 * 10**9],
            "bind_observations": [4],
        },
        "binding:l2")
    line = as_lines(result)[0]

    assert parse_as_splunk_would(line, "binding:l2") == SAMPLE_TIME + datetime.timedelta(seconds=61)
    assert parse_as_splunk_would(line, "binding:l2:open") == SAMPLE_TIME, "bind_start was left unrendered"
    assert parse_as_splunk_would(line, "morpheus:score:l2") == SAMPLE_TIME + datetime.timedelta(seconds=61)


@pytest.mark.gpu_and_cpu_mode
def test_a_nanosecond_column_is_left_as_a_number(config: Config):
    # bind_gap_ns is the value R-D-L2-004 compares against a threshold. Rounding it to microseconds to fit a
    # timestamp format would change that arithmetic, so it is deliberately not a time column.
    result = run_stage(
        config,
        {
            "bind_end": [LATER_NS],
            "bind_start": [SAMPLE_NS],
            "mac_address": ["aa:bb:cc:dd:ee:ff"],
            "port_key": ["site-1:sw1:Gi1/0/1"],
            "bind_end_reason": ["displaced"],
            "bind_gap_ns": [61 * 10**9],
            "bind_observations": [4],
        },
        "binding:l2")

    assert result["bind_gap_ns"].dtype.kind in ("i", "u")
    assert '"bind_gap_ns":61000000000' in as_lines(result)[0].replace(" ", "")


@pytest.mark.gpu_and_cpu_mode
def test_a_null_timestamp_stays_null(config: Config):
    # An open binding has no end. Inventing one would be a worse answer than the SIEM seeing an absent field.
    result = run_stage(
        config,
        {
            "bind_start": [SAMPLE_NS],
            "event_time": [SAMPLE_NS],
            "mac_address": ["aa:bb:cc:dd:ee:ff"],
            "port_key": ["site-1:sw1:Gi1/0/1"],
            "bind_provisional": [True],
        },
        "binding:l2:open")

    assert parse_as_splunk_would(as_lines(result)[0], "binding:l2:open") == SAMPLE_TIME


@pytest.mark.cpu_mode
def test_a_column_the_batch_does_not_carry_is_skipped(config: Config):
    # binding:l2:open renders bind_start and event_time; a batch without event_time is still renderable.
    result = run_stage(config,
                       {
                           "bind_start": [SAMPLE_NS],
                           "mac_address": ["aa"],
                           "port_key": ["p"],
                           "bind_provisional": [True],
                       },
                       "binding:l2:open")

    assert "event_time" not in result.columns
    assert parse_as_splunk_would(as_lines(result)[0], "binding:l2:open") == SAMPLE_TIME


@pytest.mark.cpu_mode
def test_a_missing_required_column_raises(config: Config):
    with pytest.raises(KeyError, match="auth_unpaired"):
        run_stage(config, {"event_time": [SAMPLE_NS], "event_uid": ["a"]}, "morpheus:score:l2")


@pytest.mark.cpu_mode
def test_a_missing_anchor_column_names_the_consequence(config: Config):
    with pytest.raises(KeyError, match="stamped at index"):
        run_stage(config, {"bind_start": [SAMPLE_NS]}, "binding:l2")


@pytest.mark.cpu_mode
def test_the_requirement_can_be_relaxed_for_exploration(config: Config, capfd: pytest.CaptureFixture):
    # Read from the file descriptor rather than caplog: Morpheus configures its own handlers and the record does
    # not propagate to the root logger pytest attaches to.
    result = run_stage(config, {"event_time": [SAMPLE_NS]}, "morpheus:score:l2", require_columns=False)
    warned = capfd.readouterr().err

    assert "auth_unpaired" in warned
    # The anchor is present here, so the sentence about index-time stamping does not belong in this warning.
    assert "index time" not in warned
    assert parse_as_splunk_would(as_lines(result)[0], "morpheus:score:l2") == SAMPLE_TIME


@pytest.mark.cpu_mode
def test_an_unproduced_sourcetype_cannot_be_configured(config: Config):
    with pytest.raises(ValueError, match="TC-0 context store"):
        SiemWireStage(config, sourcetype="context:asset")


@pytest.mark.cpu_mode
def test_an_unknown_sourcetype_cannot_be_configured(config: Config):
    with pytest.raises(KeyError):
        SiemWireStage(config, sourcetype="morpheus:score:l9")


@pytest.mark.cpu_mode
def test_the_needed_columns_are_declared_as_strings(config: Config):
    stage = SiemWireStage(config, sourcetype="binding:l2")

    assert stage.sourcetype == "binding:l2"
    assert stage._needed_columns["bind_end"] == TypeId.STRING  # pylint: disable=protected-access
    assert stage._needed_columns["bind_start"] == TypeId.STRING  # pylint: disable=protected-access


@pytest.mark.cpu_mode
def test_a_control_message_stays_a_control_message(config: Config):
    stage = SiemWireStage(config, sourcetype="binding:l2:open")
    message = ControlMessage()
    message.payload(
        MessageMeta(
            get_df_class(config.execution_mode)({
                "bind_start": [SAMPLE_NS],
                "mac_address": ["aa"],
                "port_key": ["p"],
                "bind_provisional": [True],
            })))

    result = stage.on_data(message)

    assert isinstance(result, ControlMessage)
    assert parse_as_splunk_would(as_lines(result.payload().copy_dataframe())[0], "binding:l2:open") == SAMPLE_TIME


@pytest.mark.cpu_mode
def test_an_empty_batch_is_returned_unchanged(config: Config):
    stage = SiemWireStage(config, sourcetype="morpheus:score:l2")
    meta = MessageMeta(get_df_class(config.execution_mode)(pd.DataFrame({"event_time": []})))

    assert stage.on_data(meta) is meta
