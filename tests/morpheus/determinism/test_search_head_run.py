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
The search-head run: what its runner does to the events before Splunk sees them, and what Splunk said back.

`run_search_head.py` runs inside a Splunk container this repository's CI cannot start, so the parts of it that
decide whether the run means anything are tested here without one: that moving the 1970 timestamps forward
changes nothing but the timestamps, by a whole number of weeks; that every shipped search is dispatched exactly
once, in the order VALIDATION.md prescribes; that each sourcetype goes to the index the procedure names; and that
the validation-only settings cover every sourcetype the app defines.

The run's own result, `validate/search_head_results.json`, is compared with `expected_results.json`, risk index
included. The runs before it are kept in `validate/search_head_runs/`, each with the differences it found; were the
result removed, the comparison would skip and name the missing file rather than pass on nothing.
"""

import importlib.util
import json
import os
import re

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
VALIDATE = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "validate")
RUNNER = os.path.join(VALIDATE, "run_search_head.py")
RESULTS = os.path.join(VALIDATE, "search_head_results.json")
EXPECTED = os.path.join(VALIDATE, "expected_results.json")
VALIDATION = os.path.join(VALIDATE, "VALIDATION.md")
VALIDATION_PROPS = os.path.join(VALIDATE, "validation_app", "local", "props.conf")
APP_PROPS = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "TA-morpheus-lineage", "default", "props.conf")


@pytest.fixture(name="runner", scope="module")
def runner_fixture():
    """Import the runner as a module. Nothing runs at import; `main` is guarded."""
    spec = importlib.util.spec_from_file_location("run_search_head", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def _event_files() -> list[str]:
    """The files the runner indexes: every rendered event file, and not the manifest beside them."""
    return sorted(name for name in os.listdir(os.path.join(VALIDATE, "sample_events")) if name.endswith(".jsonlines"))


def _sample_lines(runner) -> list[str]:
    lines = []

    for name in _event_files():
        with open(os.path.join(VALIDATE, "sample_events", name), encoding="utf-8") as handle:
            lines.extend(handle.read().splitlines())

    assert len(lines) > 0
    del runner

    return lines


def test_the_events_are_dated_where_the_runner_assumes(runner):
    # The reason the runner moves them at all: no search head indexes a 1970 timestamp as itself.
    (newest, oldest) = runner.newest_and_oldest(_sample_lines(runner))

    assert oldest >= 0 and newest < 365 * 86400, "the corpora no longer count from the epoch; revisit the shift"


def test_the_shift_is_a_whole_number_of_weeks_landing_before_the_run(runner):
    now = 1_790_000_000.0
    (newest, _) = runner.newest_and_oldest(_sample_lines(runner))
    offset = runner.week_offset(newest, now)

    assert offset % runner.WEEK_SECONDS == 0, "anything but whole weeks moves an hour-of-day or weekday decision"
    assert newest + offset <= now - runner.LANDING_SECONDS
    assert newest + offset > now - runner.LANDING_SECONDS - runner.WEEK_SECONDS


def test_the_shift_moves_timestamps_and_nothing_else(runner):
    weeks = 2900 * runner.WEEK_SECONDS

    for line in _sample_lines(runner)[::50]:
        shifted = runner.shift_line(line, weeks)

        assert runner.TIMESTAMP.sub('"T"', shifted) == runner.TIMESTAMP.sub('"T"', line), \
            "something other than a timestamp changed"
        assert len(runner.TIMESTAMP.findall(shifted)) == len(runner.TIMESTAMP.findall(line))

        before = [runner._epoch(match) for match in runner.TIMESTAMP.finditer(line)]  # pylint: disable=protected-access
        after = [runner._epoch(match) for match in runner.TIMESTAMP.finditer(shifted)]  # pylint: disable=protected-access

        assert all(abs((b - a) - weeks) < 1e-6 for (a, b) in zip(before, after))


def test_every_search_is_dispatched_once_in_the_prescribed_order(runner):
    names = runner.stanzas()
    order = [name for (name, _) in runner.ordered(names)]
    phases = dict(runner.ordered(names))

    assert sorted(order) == sorted(names) and len(order) == len(set(order)) == 48

    position = {name: index for (index, name) in enumerate(order)}
    refreshes = [name for name in names if phases[name] == "refresh"]
    expiries = [name for name in names if phases[name] == "expiry"]

    assert len(refreshes) == 3 and len(expiries) == 3
    assert max(position[name] for name in refreshes) < min(position[name] for name in runner.PREDICTIVE)

    for writer in runner.PREDICTIVE:
        assert position[writer] < position["R-B-L7-002 - Bulk data access"]

    # Every detection writes its risk record before the search that sums them, and the summary is collected before
    # the trajectory rule reads it.
    writers = [name for name in names if runner.writes_risk(name)]

    assert len(writers) == 38
    assert max(position[name] for name in writers) < position[runner.CHAIN]
    assert position[runner.SUMMARY] < position[runner.TRAJECTORY] < position[runner.CHAIN]
    assert min(position[name] for name in expiries) > max(position[name] for name in names if name not in expiries)


def test_the_chain_risk_check_is_chain_assembly_up_to_its_threshold(runner):
    # The check reports how close each chain came; it must be the shipped search's own arithmetic, not a copy.
    shipped = runner.search_text(runner.CHAIN)
    check = runner.chain_risk_query()

    assert check.startswith(shipped.split("| where total_risk", 1)[0])
    assert "| where layer_span >= 3" in check and check.endswith("| stats count BY total_risk")


def test_each_sourcetype_goes_where_the_procedure_sends_it(runner):
    # VALIDATION.md's indexing loop, as a table: scored events and the edge stream are separate indexes, bindings
    # and context have their own.
    files = _event_files()
    where = {runner.sourcetype_of(name): runner.index_of(runner.sourcetype_of(name)) for name in files}

    for (sourcetype, index) in where.items():
        if (sourcetype == "morpheus:edge"):
            assert index == "behavior_lineage"
        elif (sourcetype.startswith("morpheus:score:")):
            assert index == "behavior_events", sourcetype
        elif (sourcetype.startswith("binding:")):
            assert index == "behavior_bindings", sourcetype
        else:
            assert sourcetype.startswith("context:") and index == "behavior_context", sourcetype


def test_the_validation_settings_cover_every_sourcetype_the_app_defines():
    # One sourcetype left on the shipped thirty days would have its oldest events silently retimed.
    def stanzas(path):
        with open(path, encoding="utf-8") as handle:
            return set(re.findall(r"^\[(.+)\]\s*$", handle.read(), re.MULTILINE))

    with open(VALIDATION_PROPS, encoding="utf-8") as handle:
        values = re.findall(r"^MAX_DAYS_AGO\s*=\s*(\d+)", handle.read(), re.MULTILINE)

    assert stanzas(VALIDATION_PROPS) == stanzas(APP_PROPS)
    assert len(values) == len(stanzas(APP_PROPS)) and all(int(value) >= 60 for value in values)


def test_the_recorded_run_returned_what_is_written():
    if (not os.path.exists(RESULTS)):
        pytest.skip("validate/search_head_results.json is not committed: the search-head run has not been recorded. "
                    "examples/splunk_lineage_app/validate/run_search_head.sh records it.")

    with open(RESULTS, encoding="utf-8") as handle:
        run = json.load(handle)

    with open(EXPECTED, encoding="utf-8") as handle:
        expected = json.load(handle)["searches"]

    assert run["indexed"] == run["expected_indexed"], "not every sample event was indexed"
    assert set(run["searches"]) == set(expected), "the run and the expectation name different searches"

    errored = {name: entry["error"] for (name, entry) in run["searches"].items() if "error" in entry}
    assert errored == {}, f"searches that did not run on the search head: {sorted(errored)}"

    differing = {
        name: (entry["rows"], expected[name]["expected_rows"])
        for (name, entry) in run["searches"].items() if entry["rows"] != expected[name]["expected_rows"]
    }
    assert differing == {}, f"rows returned on the search head versus written: {differing}"

    # The write path, end to end: every row a detection returned is a record in the risk index, and the chains
    # that span three layers carry the risk the expectation says.
    written = sum(entry["rows"] for (name, entry) in run["searches"].items() if name.startswith("R-"))
    risk = run["checks"]["risk_records"]

    assert risk["written"] == risk["indexed"] == written
    assert {row["total_risk"]: int(row["count"]) for row in run["checks"]["three_layer_chain_risk"]} == \
        expected["Chain assembly - cross-layer risk"]["three_layer_chain_risk"]


FIRST_RUN = os.path.join(VALIDATE, "search_head_runs", "2026-10-05T0114Z.json")
"""The first search-head run, kept because it is the evidence for the two changes it caused."""


def test_the_first_run_differed_from_what_was_written_in_exactly_the_two_ways_it_caused_changes():
    # Splunk 10.2.8, 2026-10-05: forty-six of forty-eight searches returned what was then written. The other two are
    # why the wire format leaves nulls out and why the watchlist expiry expects two rows: R-B-L2-002 let four ports
    # through whose `macs_per_port_step` was sent as null, and the expiry kept R-P-L7-006's entries because the
    # runner dates the events within their thirty days. Reading the nulls out of the wire then showed a third that
    # had matched only because the expectation shared the defect: Binding health counted four classes whose
    # `resolution_method` was null on every row. The run itself was of the events as they were then, so it
    # is not compared with the regenerated ones; `search_head_results.json` is for that.
    with open(FIRST_RUN, encoding="utf-8") as handle:
        run = json.load(handle)

    with open(EXPECTED, encoding="utf-8") as handle:
        expected = json.load(handle)["searches"]

    assert run["splunk_version"].startswith("Splunk 10.2")
    assert run["indexed"] == run["expected_indexed"] and sum(run["indexed"].values()) == 8400
    assert not any("error" in entry for entry in run["searches"].values())

    differing = {
        name: entry["rows"]
        for (name, entry) in run["searches"].items() if entry["rows"] != expected[name]["expected_rows"]
    }

    assert differing == {
        "R-B-L2-002 - Port-to-MAC binding novelty": 6,
        "Binding health - unresolved rate": 5,
        "R-P-L3-005 - Fan-out trajectory": 0,
    }, ("the watchlist expiry now expects what the run found; two are nulls that reached the wire; the trajectory "
        "rule's own SPL could not fire, which the third run found")
    assert expected["R-B-L2-002 - Port-to-MAC binding novelty"]["expected_rows"] == 2
    assert expected["Binding health - unresolved rate"]["expected_rows"] == 1


SECOND_RUN = os.path.join(VALIDATE, "search_head_runs", "2026-10-09T1400Z.json")
"""The run over the regenerated events, which matched every expectation then written, and found nothing."""


def test_the_second_run_differed_from_what_is_written_only_where_the_trajectory_rule_was_broken():
    # Splunk 10.2.8, 2026-10-09: all forty-eight returned what was then written, R-P-L3-005's zero among them. A
    # third run, on a search head started in the development container, found the zero was the search and not the
    # data: `last(previous_destinations)` read a field its own streamstats was creating, which is null on every row.
    with open(SECOND_RUN, encoding="utf-8") as handle:
        run = json.load(handle)

    with open(EXPECTED, encoding="utf-8") as handle:
        expected = json.load(handle)["searches"]

    assert run["splunk_version"].startswith("Splunk 10.2")
    assert run["indexed"] == run["expected_indexed"] and sum(run["indexed"].values()) == 8400

    differing = {
        name: entry["rows"]
        for (name, entry) in run["searches"].items() if entry["rows"] != expected[name]["expected_rows"]
    }

    assert differing == {"R-P-L3-005 - Fan-out trajectory": 0}
    assert "risk_records" not in run["checks"], "the write path did not exist when this run was made"


def test_no_sample_event_sends_a_null():
    # The repair, held: an absent field cannot be compared, a null one was.
    for name in _event_files():
        with open(os.path.join(VALIDATE, "sample_events", name), encoding="utf-8") as handle:
            for line in handle:
                assert all(value is not None for value in json.loads(line).values()), name
