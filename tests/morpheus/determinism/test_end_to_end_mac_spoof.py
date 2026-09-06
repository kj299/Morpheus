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
R-D-L2-004 against a producer's actual bytes, for the first time.

Every other test in this repository applies a rule's predicate to a DataFrame. That answers whether the analytics
are right, and says nothing about whether what a SIEM receives carries what the rule reads -- which is where two
of this app's defects lived, invisible from any frame.

So nothing here touches a frame. The example writes JSON lines to disk; this reads those bytes back, stamps
`_time` the way Splunk would by applying the shipped `props.conf`'s own `TIME_PREFIX` regex and `TIME_FORMAT` to
the raw line, and evaluates R-D-L2-004's predicate against the parsed JSON. What it asserts is what an analyst
would see in their queue.
"""

import datetime
import json
import os
import re
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
EXAMPLE = os.path.join(REPO_ROOT, "examples", "behavioral_analytics")
RUNNER = os.path.join(EXAMPLE, "run_mac_spoof.py")
SOURCE = os.path.join(EXAMPLE, "mac_table_sample.jsonlines")
EXPECTED = os.path.join(EXAMPLE, "expected_notables.json")
PROPS = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "TA-morpheus-lineage", "default", "props.conf")
SAVEDSEARCHES = PROPS.replace("props.conf", "savedsearches.conf")

SOURCETYPE = "binding:l2"
RULE = "R-D-L2-004"
GAP_THRESHOLD_NS = 60 * 10**9
NS_PER_SECOND = 10**9


def _conf(path: str) -> dict:
    """A Splunk `.conf` as a dict of dicts, with its backslash continuations folded first."""
    import configparser

    with open(path, encoding="utf-8") as handle:
        folded = re.sub(r"\\\s*\r?\n\s*", " ", handle.read())

    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read_string(folded)

    return {name: dict(parser[name]) for name in parser.sections()}


def _splunk_time(line: str) -> datetime.datetime:
    """
    Stamp one raw line the way Splunk would: its stanza's own prefix regex, its own format, no shortcuts.

    Deliberately reading the bytes rather than the parsed object. A JSON parser is more forgiving than a regex
    anchored on a quote, and the whole failure this guards against is a value that parses perfectly as JSON and
    matches nothing as a timestamp.
    """
    settings = _conf(PROPS)[SOURCETYPE]
    match = re.search(settings["time_prefix"], line)

    assert match is not None, f"TIME_PREFIX for {SOURCETYPE} matched nothing in: {line[:160]}"

    remainder = line[match.end():]
    rendered = remainder[:remainder.index('"')]

    return datetime.datetime.strptime(rendered, re.sub(r"%\d?N", "%f", settings["time_format"]))


def _read_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_lines(path: str) -> list:
    with open(path, encoding="utf-8") as handle:
        return [line for line in handle.read().splitlines() if line.strip()]


def _fires(record: dict) -> bool:
    """R-D-L2-004's predicate, read off the saved search rather than restated from memory."""
    return (record.get("bind_end_reason") in ("conflict", "displaced") and record.get("bind_gap_ns") is not None
            and record["bind_gap_ns"] <= GAP_THRESHOLD_NS)


@pytest.fixture(name="notables", scope="module")
def notables_fixture(tmp_path_factory) -> list:
    """Run the shipped example as a subprocess and hand back the lines it wrote."""
    output = tmp_path_factory.mktemp("mac_spoof") / "notables.jsonlines"
    completed = subprocess.run([sys.executable, RUNNER, "--source", SOURCE, "--output", str(output)],
                               capture_output=True,
                               text=True,
                               check=False,
                               timeout=600,
                               cwd=REPO_ROOT)

    assert completed.returncode == 0, f"the example failed:\n{completed.stderr[-4000:]}"
    assert output.exists(), "the example reported success and wrote no file"

    yield [line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_the_example_runs_and_writes_records(notables: list):
    # If this fails, nothing below means anything: the assertions would be about an empty file.
    assert len(notables) == 6, "the sample should close six bindings"


def test_exactly_one_notable_and_it_is_the_spoof(notables: list):
    firing = [json.loads(line) for line in notables if _fires(json.loads(line))]

    assert len(firing) == 1, f"expected one notable, got {[record['port_key'] for record in firing]}"

    hit = firing[0]
    expected = _read_json(EXPECTED)["notables"][0]

    assert hit["mac_address"] == expected["mac_address"]
    assert hit["port_key"] == expected["port_key"], "the notable must name the port the address was taken from"
    assert hit["bind_end_reason"] == "conflict"
    assert hit["bind_gap_ns"] == 0, "two sightings in one instant are zero apart; that is what makes it a conflict"
    assert hit["bind_observations"] == expected["bind_observations"]


def test_the_notable_is_stamped_by_its_own_event_time(notables: list):
    # The hop every other test skips. A record whose timestamp Splunk cannot parse is stamped at index time, and
    # a 20-minute windowed search then reports on when the pipeline was busy.
    line = next(line for line in notables if _fires(json.loads(line)))
    expected = _read_json(EXPECTED)["notables"][0]

    stamped = _splunk_time(line)
    source_times = [json.loads(row)["event_time"] for row in _read_lines(SOURCE)]
    spoof_instant = sorted(set(source_times))[2]

    # The binding ends one tick after the instant that closed it: the interval is half-open, so the shortest
    # interval that actually contains what was seen ends just past the last sighting.
    expected_instant = datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=(spoof_instant + 1) // 1000)

    assert stamped == expected_instant
    assert expected["_time"] == expected["bind_end"], "the notable's time is its binding's end, by this stanza"


def test_a_device_that_merely_moved_is_not_a_notable(notables: list):
    # The corpus contains a legitimate move as well as a spoof, and the rule separates them on the gap between
    # sightings rather than on a suppression list. Without this the test above would pass on a rule that fired on
    # every displacement, which is the false-positive shape the threshold exists to prevent.
    displaced = [json.loads(line) for line in notables if json.loads(line)["bind_end_reason"] == "displaced"]

    assert len(displaced) == 1
    assert displaced[0]["bind_gap_ns"] == 300 * NS_PER_SECOND
    assert not _fires(displaced[0]), "a 300-second gap is a move, and the rule's threshold is 60 seconds"


def test_the_predicate_matches_the_shipped_saved_search():
    # The predicate above is a restatement, and a restatement drifts. This pins it to the search's own text.
    search = _conf(SAVEDSEARCHES)[f"{RULE} - MAC in two places at once"]["search"]

    assert f"sourcetype={SOURCETYPE}" in search
    assert "bind_end_reason=conflict" in search and "bind_end_reason=displaced" in search
    assert str(GAP_THRESHOLD_NS) in search, "the gap threshold here no longer matches the saved search"
    assert "bind_gap_ns <= gap_threshold" in search


def test_every_record_carries_what_the_alert_shows(notables: list):
    # The `| table` clause names the fields an analyst reads. A blank column on a notable is the defect the field
    # linter exists for, checked here against records that actually reached a file.
    search = _conf(SAVEDSEARCHES)[f"{RULE} - MAC in two places at once"]["search"]
    tabled = re.search(r"\|\s*table\s+(.*)$", search).group(1).split()
    evaluated = {"_time", "rule_id", "risk_score", "osi_layer", "entity_key", "gap_seconds"}

    for line in notables:
        record = json.loads(line)

        for field in tabled:
            if (field in evaluated):
                continue

            assert field in record, f"{field} is on the alert and absent from the record"
