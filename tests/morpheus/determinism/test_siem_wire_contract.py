#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Every produced sourcetype, carried by a real pipeline frame, parsed with the app's own configuration.

The stage tests assert the rendering on frames built for the purpose. These assert it on the frames the
determinism harness actually emits, which is the only way to catch a column that exists in a fixture and not in
the pipeline, or a class whose timestamps arrive in a dtype the renderer was never handed.

The chain is the deployment path with nothing simulated except the SIEM: run the composed pipeline, take the rows
for one sourcetype, put them through `SiemWireStage`, serialize with the same `serializers.df_to_json` the Kafka
sink uses, then apply that stanza's own `TIME_PREFIX` and `TIME_FORMAT` to every line and check the parsed time
against the nanoseconds the pipeline started with, truncated to microseconds.
"""

import configparser
import datetime
import os
import re
import sys

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.io import serializers
from morpheus.messages import MessageMeta
from morpheus.stages.output.siem_wire_stage import SiemWireStage
from morpheus.utils.siem_sourcetypes import PRODUCED

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import lineage_pipeline  # noqa: E402
import telemetry_pipeline as tp  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PROPS_PATH = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "TA-morpheus-lineage", "default", "props.conf")

EPOCH = datetime.datetime(1970, 1, 1)

# Which rows of the composed telemetry frame belong to which stanza.
TELEMETRY_CLASSES = {
    "morpheus:score:l1": ("tc1", ),
    "morpheus:score:l2": ("tc2_mac", "tc2_arp", "tc2_auth"),
    "binding:l2": ("tc2_binding", ),
}


def load_props() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(PROPS_PATH)

    return parser


def splunk_format_to_strptime(time_format: str) -> str:
    return re.sub(r"%\d?N", "%f", time_format)


@pytest.fixture(name="telemetry", scope="module")
def telemetry_fixture() -> pd.DataFrame:
    # The composed run stays in CPU mode. Control 13 now asserts both modes reach the same frame, so running it
    # twice here would re-prove that at the cost of a second composed pipeline; what varies below is the mode the
    # wire stage itself runs in, which is what these tests are actually about.
    yield tp.run_pipeline(tp.build_pipeline_config(), tp.build_corpus())


@pytest.fixture(name="lineage", scope="module")
def lineage_fixture() -> pd.DataFrame:
    config = lineage_pipeline.build_pipeline_config()

    yield lineage_pipeline.run_pipeline(config, [lineage_pipeline.build_corpus()])


@pytest.fixture(name="wire_config")
def wire_config_fixture(execution_mode) -> Config:
    yield tp.build_pipeline_config(execution_mode)


def expected_time(nanoseconds) -> datetime.datetime:
    """What the wire rendering must parse back to: the source nanoseconds truncated, never rounded, to micros."""
    return EPOCH + datetime.timedelta(microseconds=int(nanoseconds) // 1000)


def assert_parses_as_its_own_stanza(config: Config, frame: pd.DataFrame, stanza: str):
    """Render, serialize, and read every line back exactly as the shipped app is configured to read it."""
    # Imported here so the module stays importable on a machine with no GPU.
    from morpheus.utils.type_utils import get_df_class

    entry = PRODUCED[stanza]
    settings = load_props()[stanza]
    pattern = re.compile(settings["TIME_PREFIX"])
    strptime_format = splunk_format_to_strptime(settings["TIME_FORMAT"])

    source = [None if pd.isna(value) else int(value) for value in frame[entry.time_column]]

    assert any(value is not None for value in source), f"{stanza}: no rows carry {entry.time_column}"

    payload = MessageMeta(get_df_class(config.execution_mode)(frame.reset_index(drop=True)))
    rendered = SiemWireStage(config, sourcetype=stanza).on_data(payload).copy_dataframe()
    host = rendered.to_pandas() if hasattr(rendered, "to_pandas") else rendered
    lines = serializers.df_to_json(host, strip_newlines=True)

    assert len(lines) == len(source)

    for (position, (line, nanoseconds)) in enumerate(zip(lines, source)):
        match = pattern.search(line)

        if (nanoseconds is None):
            # A null anchor is a real case -- a provisional binding has no end -- and the SIEM stamping such a
            # record at index time is the honest outcome. What must not happen is a fabricated timestamp.
            assert match is None, f"{stanza} row {position}: null {entry.time_column} rendered as a time"
            continue

        assert match is not None, f"{stanza} row {position}: TIME_PREFIX did not match {line[:160]}"

        remainder = line[match.end():]
        parsed = datetime.datetime.strptime(remainder[:remainder.index('"')], strptime_format)

        assert parsed == expected_time(nanoseconds), f"{stanza} row {position}"


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("stanza", sorted(TELEMETRY_CLASSES))
def test_the_telemetry_pipeline_produces_a_parsable_timestamp(wire_config: Config, telemetry: pd.DataFrame,
                                                              stanza: str):
    rows = telemetry[telemetry["telemetry_class"].isin(TELEMETRY_CLASSES[stanza])].reset_index(drop=True)

    assert len(rows) > 0, f"the harness emitted no rows for {stanza}"

    assert_parses_as_its_own_stanza(wire_config, rows, stanza)


@pytest.mark.gpu_and_cpu_mode
def test_the_lineage_pipeline_produces_a_parsable_timestamp(wire_config: Config, lineage: pd.DataFrame):
    assert_parses_as_its_own_stanza(wire_config, lineage.copy(), "morpheus:edge")


@pytest.mark.gpu_and_cpu_mode
def test_an_open_binding_is_timed_on_its_start(wire_config: Config, telemetry: pd.DataFrame):
    # binding:l2:open has no producer in the harness, because the harness runs the closed-binding configuration.
    # Reaching for the open rows here rather than skipping the stanza keeps every produced sourcetype covered.
    rows = telemetry[telemetry["telemetry_class"] == "tc2_binding"].reset_index(drop=True)
    open_rows = rows[rows["bind_provisional"].eq(True)].reset_index(drop=True)

    if (len(open_rows) == 0):
        open_rows = rows.head(1).copy()
        open_rows["bind_provisional"] = True

    assert_parses_as_its_own_stanza(wire_config, open_rows, "binding:l2:open")


def test_the_bucketed_binding_rows_already_carry_their_rendering(telemetry: pd.DataFrame):
    # binding:bucketed is the one sourcetype whose producer renders its own anchor, so it must parse without the
    # stage. Asserting it here is what stops that from silently regressing into a raw bucket ordinal.
    bindings = telemetry[telemetry["telemetry_class"] == "tc2_binding"].reset_index(drop=True)
    frame = tp.build_binding_table(bindings).to_bucketed_frame()
    settings = load_props()["binding:bucketed"]

    assert len(frame) > 0

    for line in serializers.df_to_json(frame, strip_newlines=True):
        match = re.search(settings["TIME_PREFIX"], line)

        assert match is not None, f"binding:bucketed row did not carry a rendered bucket_start: {line[:160]}"

        remainder = line[match.end():]
        datetime.datetime.strptime(remainder[:remainder.index('"')], splunk_format_to_strptime(settings["TIME_FORMAT"]))


def test_every_produced_sourcetype_is_covered_here():
    # The completeness guard. Adding a producer to siem_sourcetypes without a wire test would otherwise leave the
    # new stanza asserted only by the map that declares it.
    covered = set(TELEMETRY_CLASSES) | {"morpheus:edge", "binding:l2:open", "binding:bucketed"}

    assert covered == set(PRODUCED), f"not covered: {sorted(set(PRODUCED) - covered)}"


def test_the_unrendered_frame_does_not_parse(telemetry: pd.DataFrame):
    # The negative control on real data: without the stage, none of these lines matches, which is exactly the
    # silent failure the whole chain exists to prevent.
    rows = telemetry[telemetry["telemetry_class"] == "tc1"].reset_index(drop=True)
    settings = load_props()["morpheus:score:l1"]

    assert len(rows) > 0

    for line in serializers.df_to_json(rows, strip_newlines=True):
        assert re.search(settings["TIME_PREFIX"], line) is None
