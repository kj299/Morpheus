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
Six real-format fixtures, normalized, are the committed samples; and a segment run over them writes what it writes
from the samples.

`examples/behavioral_analytics/fixtures/make_fixtures.py` renders the samples of six input classes as the feeds
write them -- IPFIX in a packet capture, Zeek `ssl.log`/`x509.log`, `http.log` and `dns.log`, unified audit log
records and Sysmon events -- carrying only what each format carries. Reading them back through
`FlowRecordNormalizeStage` and the `collector_schemas` schemas has to give the samples column for column, except
what a format cannot carry, which is named here and has to be null rather than invented.
"""

import filecmp
import json
import math
import os
import pathlib
import shutil
import sys

import pandas as pd
import pytest

from morpheus.utils import collector_schemas
from morpheus.utils.collector_schemas import CollectorProfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
EXAMPLE = os.path.join(REPO_ROOT, "examples", "behavioral_analytics")
FIXTURES = os.path.join(EXAMPLE, "fixtures")

sys.path.insert(0, FIXTURES)
sys.path.insert(0, EXAMPLE)
sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "morpheus", "determinism"))

# pylint: disable=wrong-import-position
import make_fixtures  # noqa: E402

NOT_IN_THE_FORMAT = {"tc6": ("content_type_declared", "content_type_detected")}
"""Columns a format cannot carry: what an inspecting proxy saw inside a TLS session is not in Zeek's logs."""


def _plain(value):
    if (value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))):
        return None

    if (hasattr(value, "item")):
        value = value.item()

    return int(value) if isinstance(value, float) and value.is_integer() else value


def _records(frame: pd.DataFrame, columns: list) -> list[dict]:
    return [{column: _plain(row.get(column)) for column in columns} for row in frame.to_dict("records")]


@pytest.fixture(name="normalized", scope="module")
def normalized_fixture() -> dict:
    yield make_fixtures.normalize_all()


def test_the_committed_fixtures_are_what_the_samples_render_to(tmp_path):
    make_fixtures.write_all(str(tmp_path))

    for (segment, names) in make_fixtures.CLASSES.values():
        for name in names:
            assert filecmp.cmp(os.path.join(tmp_path, segment, name), os.path.join(FIXTURES, segment, name),
                               shallow=False), f"{segment}/{name} is stale; run make_fixtures.py"


@pytest.mark.parametrize("telemetry_class", sorted(make_fixtures.CLASSES))
def test_a_normalized_fixture_is_its_sample(telemetry_class: str, normalized: dict):
    segment = make_fixtures.CLASSES[telemetry_class][0]
    sample = make_fixtures.sample(segment, telemetry_class)
    columns = list(sample[0])
    frame = normalized[telemetry_class]
    missing = NOT_IN_THE_FORMAT.get(telemetry_class, ())
    expected = [{
        column: (None if column in missing else _plain(row.get(column)))
        for column in columns
    } for row in sample]

    assert _records(frame, columns) == expected
    # Whatever the schema writes beyond the sample's columns is an optional column the sample omits; it is null.
    for column in set(frame.columns) - set(columns):
        assert frame[column].isna().all(), column

    # The columns the format cannot carry are not invented, and the sample does have values there to miss.
    for column in missing:
        assert frame[column].isna().all()
        assert any(row[column] is not None for row in sample), column


def test_every_class_column_the_contract_names_is_written(normalized: dict):
    with open(os.path.join(EXAMPLE, "input_contracts.json"), encoding="utf-8") as handle:
        contract = json.load(handle)

    envelope = [column["name"] for column in contract["envelope"]]

    for (telemetry_class, frame) in normalized.items():
        named = [column["name"] for column in contract["classes"][telemetry_class]["columns"]]

        assert set(frame.columns) == set(named) | set(envelope), telemetry_class


def test_the_query_string_is_stripped_and_vocabulary_mapped(normalized: dict):
    with open(os.path.join(FIXTURES, "application", "http.log"), encoding="utf-8") as handle:
        assert "?ref=" in handle.read(), "the fixture has stopped exercising the query string"

    assert not normalized["tc7_http"]["url_path"].str.contains("?", regex=False).any()
    assert set(normalized["tc6"]["tls_version"]) == {"TLSv1.2", "TLSv1.3"}
    assert set(normalized["tc6"]["validation_result"]) == {"ok", "self-signed"}


def test_an_unknown_value_passes_through_rather_than_becoming_null(tmp_path):
    profile = CollectorProfile("c", "tc7_saas.v1", "corp")
    path = tmp_path / "audit.jsonlines"
    path.write_text(json.dumps({
        "CreationTime": "2026-10-17T10:00:00",
        "UserId": "a@example.com",
        "Operation": "FileDownloaded",
        "ObjectId": "o",
        "ItemType": "File",
        "ResultStatus": "PartiallySucceeded",
        "ClientInfoString": "x"
    }) + "\n",
                    encoding="utf-8")

    row = collector_schemas.m365_audit(str(path), profile).iloc[0]

    assert row["result"] == "partiallysucceeded"
    assert row["tenant_id"] == "corp"
    assert row["record_count"] is pd.NA
    assert row["event_time"] == pd.Timestamp("2026-10-17T10:00:00Z").value


def test_the_zeek_reader_takes_the_unset_marker_booleans_and_an_unclosed_log(tmp_path):
    # A log Zeek is still writing has no #close line; its last record is a record.
    path = tmp_path / "dns.log"
    path.write_text(
        "#separator \\x09\n#set_separator\t,\n#empty_field\t(empty)\n#unset_field\t-\n#path\tdns\n"
        "#open\t2026-10-17-00-00-00\n#fields\tts\tid.orig_h\tid.orig_p\tquery\tAA\tanswers\n"
        "#types\ttime\taddr\tport\tstring\tbool\tvector[string]\n"
        "1.5\t10.0.0.1\t-\ta.example\tT\t1.2.3.4,5.6.7.8\n"
        "2.5\t10.0.0.2\t5353\tb # c.example\tF\t-\n",
        encoding="utf-8")

    frame = collector_schemas.read_zeek(str(path))

    assert len(frame) == 2
    assert frame["id.orig_p"].tolist()[0] is pd.NA and frame["id.orig_p"].tolist()[1] == 5353
    assert frame["AA"].tolist() == [True, False]
    assert frame["query"].tolist() == ["a.example", "b # c.example"]
    assert pd.isna(frame["answers"].tolist()[1])


def test_the_fingerprint_join_is_used_when_the_logs_carry_no_file_ids(normalized: dict):
    ssl = collector_schemas.read_zeek(os.path.join(FIXTURES, "presentation",
                                                   "ssl.log")).drop(columns=["cert_chain_fuids"])
    x509 = collector_schemas.read_zeek(os.path.join(FIXTURES, "presentation", "x509.log")).drop(columns=["id"])
    joined = collector_schemas.join_ssl_x509(ssl, x509)

    assert joined["certificate.issuer"].tolist() == normalized["tc6"]["certificate_issuer"].tolist()


def test_a_process_whose_image_load_was_not_logged_has_no_signature_status():
    events = pd.DataFrame([
        {
            "EventID": 1, "ProcessGuid": "{a}", "Image": "C:\\a.exe"
        },
        {
            "EventID": 7,
            "ProcessGuid": "{a}",
            "Image": "C:\\a.exe",
            "ImageLoaded": "C:\\x.dll",
            "SignatureStatus": "Valid"
        },
    ])

    assert collector_schemas.join_image_signatures(events)["SignatureStatus"].tolist() == [None]


SEGMENT_CHECKS = {
    "network": ["tc3"], "application": ["tc7_http", "tc7_dns"], "saas": ["tc7_saas"], "endpoint": ["tc7_endpoint"]
}
"""Segments whose every normalized class carries all its columns, so their output can be asked to be identical.
The presentation segment reads the two content columns Zeek cannot supply, and its content rule would differ."""


@pytest.mark.slow
@pytest.mark.parametrize("segment", sorted(SEGMENT_CHECKS))
def test_a_segment_over_the_normalized_feed_writes_what_it_writes_from_the_sample(segment: str,
                                                                                  normalized: dict,
                                                                                  tmp_path):
    import segment_runner  # pylint: disable=import-outside-toplevel

    inputs = tmp_path / "inputs"
    shutil.copytree(os.path.join(EXAMPLE, "samples", segment), inputs)

    for telemetry_class in SEGMENT_CHECKS[segment]:
        frame = normalized[telemetry_class]
        frame = frame[[
            column for column in frame.columns
            if not frame[column].isna().all() or column in make_fixtures.sample(segment, telemetry_class)[0]
        ]]
        rows = _records(frame, list(frame.columns))
        (inputs / f"{telemetry_class}.jsonlines").write_text("".join(json.dumps(row) + "\n" for row in rows),
                                                             encoding="utf-8")

    segment_runner.run(segment, inputs, output_dir=tmp_path / "from_fixture")
    segment_runner.run(segment, pathlib.Path(EXAMPLE) / "samples" / segment, output_dir=tmp_path / "from_sample")

    for path in sorted((tmp_path / "from_sample").glob("*.jsonlines")):
        assert (tmp_path / "from_fixture" / path.name).read_text(encoding="utf-8") == path.read_text(
            encoding="utf-8"), path.name
