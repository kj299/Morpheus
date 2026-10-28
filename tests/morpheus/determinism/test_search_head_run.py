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

    assert sorted(order) == sorted(names) and len(order) == len(set(order)) == 85

    position = {name: index for (index, name) in enumerate(order)}
    refreshes = [name for name in names if phases[name] == "refresh"]
    expiries = [name for name in names if phases[name] == "expiry"]

    assert len(refreshes) == 5 and len(expiries) == 6
    assert max(position[name] for name in refreshes) < min(position[name] for name in runner.PREDICTIVE)

    for writer in runner.PREDICTIVE:
        assert position[writer] < position["R-B-L7-002 - Bulk data access"]
        assert position[writer] < position["R-B-L7-004 - Process ancestry novelty"]

    # Every detection writes its risk record before the search that sums them, and the summary is collected before
    # the trajectory rule reads it.
    writers = [name for name in names if runner.writes_risk(name)]

    assert len(writers) == 61
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
        elif (sourcetype in ("morpheus:health", "morpheus:quarantine", "morpheus:late")):
            assert index == "behavior_" + sourcetype.split(":")[1], sourcetype
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


BEFORE_THE_MODELS = {
    "R-D-L5-007 - Off-hours authentication": 2,
    "R-B-L5-001 - Composite authentication anomaly": 0,
    "R-B-L5-002 - Location novelty anomaly": 0,
    "Behavior summary - per-layer scores": 4335,
}
"""What every run before the learned layer 5 models found where the models changed the answer.

The two model rules were empty by their gate, the off-hours rule had no takeover afternoon to find, and the summary
had no fortnight of training rows to group. Every earlier run is compared with what is written now, so these four
differ in each of them for that reason and no other.
"""

BEFORE_THE_HOST_BASELINES = {
    "R-P-L3-005 - Fan-out trajectory": 15,
    "R-B-L7-004 - Process ancestry novelty": 10,
    "Behavior summary - per-layer scores": 4615,
}
"""What every run before step 9's host baselines found where they changed the answer.

The layer 3 corpus gained a fortnight of ordinary history, whose mornings and Monday DHCP checks the trajectory rule
reads as rises; the campaign gained a principal whose novel process lands on a workstation; and the summary groups
both. The four searches step 9 added are absent from the earlier runs and are not compared.
"""

BEFORE_THE_NETWORK_OBJECTS = {
    "Binding lookup - L1 refresh": 7,
    "Binding lookup - L1 history refresh": 2,
    "Behavior summary - per-layer scores": 5173,
}
"""What every run before step 10's network-object detections found where its corpus changed the answer.

The layer 1 corpus gained the re-patched, inserted and restarted ports, whose bindings the two refreshes now carry,
and the polls, exchanges and MAC rows step 10 planted, which the summary groups. The ten searches step 10 added are
absent from the earlier runs and are not compared.
"""

BEFORE_THE_LEASES = {
    "Binding lookup - L2/L3 refresh": 0,
    "Binding lookup - L2/L3 expiry": 0,
    "Binding lookup - L1 history expiry": 0,
}
"""What every run before step 12's leases found where they changed the answer.

The L2/L3 refresh had no lease to select, so it and its expiry were empty; and the layer 1 history's expiry kept
nothing, because its refresh keyed each row on a day counted from 1970, which the run's `now()` placed past the
retention window. Every refresh now keys its rows on the bucket their own time falls in.
"""

BEFORE_THE_SERVICES = {
    "R-D-L6-002 - Certificate issuer anomaly": 2,
    "Host evidence - one machine across layers": 17,
    "Behavior summary - per-layer scores": 5447,
}
"""What every run before step 13's services and applications found where they changed the answer.

The issuer anomaly is grouped on the address and server name together, so an interception of one name on a shared
address is a third notable; two of R-B-L4-004's hosts are seen at layers 4 and 7, which the host evidence report
counts; and the summary groups the week of DNS and HTTP history and the service connections. The three searches step
13 added are absent from the earlier runs and are not compared, and neither is the host evidence report in a run
older than it.
"""

BEFORE_THE_DEVICES = {
    "R-B-L2-002 - Port-to-MAC binding novelty": 2,
    "R-B-L2-006 - New vendor on a VLAN": 2,
    "Device restart": 1,
    "Behavior summary - per-layer scores": 5743,
    "Binding lookup - L2/L3 refresh": 1505,
    "Binding lookup - L2/L3 expiry": 1505,
    "Binding lookup - L2 refresh": 927,
    "Binding lookup - L2 expiry": 927,
}
"""What every run before step 15's devices and physical chain found where they changed the answer.

The campaign's headquarters desk ports reach layer 2 for the first time, and each of the four intrusions is a
second address on its port and two of them a vendor new to the desk VLAN; the restart report reads the devices'
own polls, where the storm switch restarts in three periods; the intruders' leases and their ports' MAC bindings
fill the two lower rungs; and the summary groups the device polls, the port polls, the walks and the intruders'
flows. The three searches step 15 added are absent from the earlier runs and are not compared.
"""

BEFORE_THE_TRAJECTORIES = {
    "Behavior summary - per-layer scores": 8801,
    "R-B-L6-001 - New TLS client fingerprint": 5,
}
"""What the run before the daily drift reached the TLS and DNS columns found where it changed the answer.

The summary gained the presentation corpus's week of 336 handshakes and the application corpus's 248 slow-tunnel
queries; the fingerprint rule gained the creeping host's six acquisitions and its control's two. Every earlier run is
compared with what is written now, so the fingerprint rule's five rows differ in each of them for that reason and no
other; each older step's summary figure overrides this one's. The two searches step 8's third part added, the stack and
the subdomain trajectories, are absent from the earlier runs and are not compared.
"""

BEFORE_THE_DESK_SPOOF = {
    "Chain assembly - cross-layer risk": 0,
}
"""What every run before the desk spoof found where it changed the answer.

The telemetry corpus gained a workstation on the tapped port that claims the gateway in the window after the tap, so
one three-layer chain carries R-D-L1-002's 50 and R-D-L2-003's 55 together and the chain search returns it. Every
earlier run found the two accusations on different chains and the search empty. The ARP anomaly's own row count did
not move, because it groups by the contested address, and neither did the summary's, because the spoof's rows join a
group the window already had.
"""

BEFORE_THE_HTTP_AND_PROCESS_TRAJECTORIES = {
    "Behavior summary - per-layer scores": 8969,
}
"""What every run before plan section 3.3's trajectories found where its corpora changed the answer.

The application corpus gained a client that probes the web server a little more every morning and the endpoint corpus
a build server that takes up its peer's tools one a day, and the summary groups both; the two searches section 3.3
added are absent from the earlier runs and are not compared. R-B-L7-004 returns the same eleven rows and raises one by
the watchlist's boost, which a row count does not see. Each older step's summary figure overrides this one's.
"""


def _present(changes: dict, run: dict) -> dict:
    """The changes a later step made, restricted to the searches this run dispatched."""
    return {name: rows for (name, rows) in changes.items() if name in run["searches"]}


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

    # R-B-L2-002's six were the two then written and four ports whose null step reached the wire. Six are written
    # again since step 15, for a different four -- the campaign's desk ports -- so the count no longer tells them
    # apart: the search matches by coincidence, and the run's figure is pinned on its own below.
    devices = {
        name: rows
        for (name, rows) in _present(BEFORE_THE_DEVICES, run).items()
        if name != "R-B-L2-002 - Port-to-MAC binding novelty"
    }

    assert differing == {
        **_present(BEFORE_THE_DESK_SPOOF, run), **_present(BEFORE_THE_TRAJECTORIES, run),
        **devices,
        **_present(BEFORE_THE_SERVICES, run),
        **BEFORE_THE_LEASES,
        **BEFORE_THE_NETWORK_OBJECTS,
        **BEFORE_THE_HOST_BASELINES,
        **BEFORE_THE_MODELS,
        "Binding health - unresolved rate": 5,
        "R-P-L3-005 - Fan-out trajectory": 0,
    }, ("the watchlist expiry now expects what the run found; two are nulls that reached the wire; the trajectory "
        "rule's own SPL could not fire, which the third run found")
    assert run["searches"]["R-B-L2-002 - Port-to-MAC binding novelty"]["rows"] == 6
    assert expected["R-B-L2-002 - Port-to-MAC binding novelty"]["expected_rows"] == 6
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

    assert differing == {
        **_present(BEFORE_THE_DESK_SPOOF, run),
        **_present(BEFORE_THE_TRAJECTORIES, run),
        **_present(BEFORE_THE_DEVICES, run),
        **_present(BEFORE_THE_SERVICES, run),
        **BEFORE_THE_LEASES,
        **BEFORE_THE_NETWORK_OBJECTS,
        **BEFORE_THE_HOST_BASELINES,
        **BEFORE_THE_MODELS,
        "R-P-L3-005 - Fan-out trajectory": 0,
    }
    assert "risk_records" not in run["checks"], "the write path did not exist when this run was made"


def test_no_sample_event_sends_a_null():
    # The repair, held: an absent field cannot be compared, a null one was.
    for name in _event_files():
        with open(os.path.join(VALIDATE, "sample_events", name), encoding="utf-8") as handle:
            for line in handle:
                assert all(value is not None for value in json.loads(line).values()), name


THIRD_RUN = os.path.join(VALIDATE, "search_head_runs", "2026-10-09T1641Z.json")
"""The first run with the risk write path, on Splunk installed from its tarball in the development container."""

FOURTH_RUN = os.path.join(VALIDATE, "search_head_runs", "2026-10-09T2006Z.json")
"""The same tree as the third, run with the package as shipped on the Docker image, on another machine."""


def test_the_tarball_and_the_docker_image_agree_on_every_search():
    # Two installs of the same version, on two machines, over the same events: agreeing on every row count and on
    # every check is what makes either one more than an anecdote.
    with open(THIRD_RUN, encoding="utf-8") as handle:
        tarball = json.load(handle)

    with open(FOURTH_RUN, encoding="utf-8") as handle:
        docker = json.load(handle)

    assert tarball["splunk_version"] == docker["splunk_version"]
    assert {name: entry.get("rows") for (name, entry) in tarball["searches"].items()} == \
        {name: entry.get("rows") for (name, entry) in docker["searches"].items()}

    for check in ("three_layer_chain_risk", "risk_records", "principal_watchlist", "binding_tables"):
        assert tarball["checks"][check] == docker["checks"][check], check


def test_the_fourth_run_differs_from_what_is_written_only_where_the_learned_models_arrived():
    # Made before the layer 5 models, on 2026-10-09: all forty-eight then matched. Against the events the models
    # now score, it differs in exactly the searches that read them or the events they added -- the two model
    # rules, which were empty by their gate; the off-hours rule, which finds the takeover afternoon; and the
    # summary, which groups the fortnight the models were trained on -- and its risk index is the 94 rows the
    # detections then wrote.
    with open(FOURTH_RUN, encoding="utf-8") as handle:
        run = json.load(handle)

    with open(EXPECTED, encoding="utf-8") as handle:
        expected = json.load(handle)["searches"]

    differing = {
        name: entry["rows"]
        for (name, entry) in run["searches"].items() if entry["rows"] != expected[name]["expected_rows"]
    }

    assert differing == {
        **_present(BEFORE_THE_DESK_SPOOF, run),
        **_present(BEFORE_THE_TRAJECTORIES, run),
        **_present(BEFORE_THE_DEVICES, run),
        **_present(BEFORE_THE_SERVICES, run),
        **BEFORE_THE_LEASES,
        **BEFORE_THE_NETWORK_OBJECTS,
        **BEFORE_THE_HOST_BASELINES,
        **BEFORE_THE_MODELS,
    }
    assert run["checks"]["risk_records"]["written"] == run["checks"]["risk_records"]["indexed"] == 94


FIFTH_RUN = os.path.join(VALIDATE, "search_head_runs", "2026-10-09T2326Z.json")
"""The run over the events the learned layer 5 models score, which matched every expectation then written."""


def test_the_fifth_run_differs_from_what_is_written_only_where_the_host_baselines_arrived():
    # Made on 2026-10-09 with the learned layer 5 models: all forty-eight then matched, with 166 risk records.
    # Against step 9's events it differs in exactly the three searches the host baselines changed.
    with open(FIFTH_RUN, encoding="utf-8") as handle:
        run = json.load(handle)

    with open(EXPECTED, encoding="utf-8") as handle:
        expected = json.load(handle)["searches"]

    differing = {
        name: entry["rows"]
        for (name, entry) in run["searches"].items() if entry["rows"] != expected[name]["expected_rows"]
    }

    assert differing == {
        **_present(BEFORE_THE_DESK_SPOOF, run),
        **_present(BEFORE_THE_TRAJECTORIES, run),
        **_present(BEFORE_THE_DEVICES, run),
        **_present(BEFORE_THE_SERVICES, run),
        **BEFORE_THE_LEASES,
        **BEFORE_THE_NETWORK_OBJECTS,
        **BEFORE_THE_HOST_BASELINES,
    }
    assert run["checks"]["risk_records"]["written"] == run["checks"]["risk_records"]["indexed"] == 166


SIXTH_RUN = os.path.join(VALIDATE, "search_head_runs", "2026-10-10T0315Z.json")
"""The run over step 9's events, which matched every expectation then written."""


def test_the_sixth_run_differs_from_what_is_written_only_where_the_network_objects_arrived():
    # Made on 2026-10-10 with step 9's host baselines: all fifty-two then matched, with 198 risk records. Against
    # step 10's events it differs in exactly the two binding refreshes and the summary, and has none of the ten
    # searches step 10 added.
    with open(SIXTH_RUN, encoding="utf-8") as handle:
        run = json.load(handle)

    with open(EXPECTED, encoding="utf-8") as handle:
        expected = json.load(handle)["searches"]

    differing = {
        name: entry["rows"]
        for (name, entry) in run["searches"].items() if entry["rows"] != expected[name]["expected_rows"]
    }

    assert differing == {
        **_present(BEFORE_THE_DESK_SPOOF, run),
        **_present(BEFORE_THE_TRAJECTORIES, run),
        **_present(BEFORE_THE_DEVICES, run),
        **_present(BEFORE_THE_SERVICES, run),
        **BEFORE_THE_LEASES,
        **BEFORE_THE_NETWORK_OBJECTS
    }
    assert len(run["searches"]) == 52 and len(expected) == 85
    assert run["checks"]["risk_records"]["written"] == run["checks"]["risk_records"]["indexed"] == 198


BEFORE_THE_PLATFORM = {
    "Behavior summary - per-layer scores": 5345,
}
"""What every run before step 11's platform plants found where they changed the answer.

The SaaS corpus gained a second tenant's sync exports, which the summary groups. The four searches step 11 added are
absent from the earlier runs and are not compared.
"""

SEVENTH_RUN = os.path.join(VALIDATE, "search_head_runs", "2026-10-10T0451Z.json")
"""The run over step 10's events, which matched every expectation then written."""


def test_the_seventh_run_differs_from_what_is_written_only_where_the_platform_arrived():
    # Made on 2026-10-10 with step 10's network objects: all sixty-two then matched, with 209 risk records. Against
    # step 11's events it differs in the summary alone, and has none of the four searches step 11 added. Every
    # detection's base search reading window_complete=true changed nothing, because no scored event is incomplete.
    with open(SEVENTH_RUN, encoding="utf-8") as handle:
        run = json.load(handle)

    with open(EXPECTED, encoding="utf-8") as handle:
        expected = json.load(handle)["searches"]

    differing = {
        name: entry["rows"]
        for (name, entry) in run["searches"].items() if entry["rows"] != expected[name]["expected_rows"]
    }

    assert differing == {
        **_present(BEFORE_THE_DESK_SPOOF, run),
        **_present(BEFORE_THE_TRAJECTORIES, run),
        **_present(BEFORE_THE_DEVICES, run),
        **_present(BEFORE_THE_SERVICES, run),
        **BEFORE_THE_LEASES,
        **BEFORE_THE_PLATFORM
    }
    assert len(run["searches"]) == 62 and len(expected) == 85
    assert run["checks"]["risk_records"]["written"] == run["checks"]["risk_records"]["indexed"] == 209


EIGHTH_RUN = os.path.join(VALIDATE, "search_head_runs", "2026-10-10T2235Z.json")
"""The run over step 11's events, which matched every expectation then written."""


def test_the_eighth_run_differs_from_what_is_written_only_where_the_leases_arrived():
    # Made on 2026-10-10 when the pipeline began reporting on itself: all sixty-six then matched, with 209 risk
    # records. Against step 12's events it differs in the L2/L3 refresh, which had no lease to select, and the two
    # expiry jobs whose refreshes keyed rows on a bucket counted from 1970; it has none of the six searches step 12
    # added. The summary groups the same rows: the chains rooted on machines regroup lineage, not windows.
    with open(EIGHTH_RUN, encoding="utf-8") as handle:
        run = json.load(handle)

    with open(EXPECTED, encoding="utf-8") as handle:
        expected = json.load(handle)["searches"]

    differing = {
        name: entry["rows"]
        for (name, entry) in run["searches"].items() if entry["rows"] != expected[name]["expected_rows"]
    }

    assert differing == {
        **_present(BEFORE_THE_DESK_SPOOF, run),
        **_present(BEFORE_THE_TRAJECTORIES, run),
        **_present(BEFORE_THE_DEVICES, run),
        **_present(BEFORE_THE_SERVICES, run),
        **BEFORE_THE_LEASES,
    }
    assert len(run["searches"]) == 66 and len(expected) == 85
    assert run["checks"]["risk_records"]["written"] == run["checks"]["risk_records"]["indexed"] == 209


NINTH_RUN = os.path.join(VALIDATE, "search_head_runs", "2026-10-15T1208Z.json")
"""The run over step 12's events, which matched every expectation then written."""


def test_the_ninth_run_differs_from_what_is_written_only_where_the_services_arrived():
    # Made on 2026-10-15 when hosts gained an identity across layers: all seventy-two then matched, with 209 risk
    # records. Against step 13's events it differs in the issuer anomaly, now grouped per server name, the host
    # evidence report, which gained two of R-B-L4-004's hosts, and the summary; it has none of the three searches
    # step 13 added.
    with open(NINTH_RUN, encoding="utf-8") as handle:
        run = json.load(handle)

    with open(EXPECTED, encoding="utf-8") as handle:
        expected = json.load(handle)["searches"]

    differing = {
        name: entry["rows"]
        for (name, entry) in run["searches"].items() if entry["rows"] != expected[name]["expected_rows"]
    }

    assert differing == {
        **_present(BEFORE_THE_DESK_SPOOF, run),
        **_present(BEFORE_THE_TRAJECTORIES, run),
        **_present(BEFORE_THE_DEVICES, run),
        **BEFORE_THE_SERVICES
    }
    assert len(run["searches"]) == 72 and len(expected) == 85
    assert run["checks"]["risk_records"]["written"] == run["checks"]["risk_records"]["indexed"] == 209


TENTH_RUN = os.path.join(VALIDATE, "search_head_runs", "2026-10-16T1841Z.json")
"""The run over step 13's events, which matched every expectation then written."""


def test_the_tenth_run_differs_from_what_is_written_only_where_the_devices_arrived():
    # Made on 2026-10-16 when a service and an application became entities: all seventy-five then matched, with 213
    # risk records. Against step 15's events it differs in the two layer 2 rules the campaign's desk ports now reach,
    # the restart report, which reads the devices' own polls, the two lower binding rungs and their expiries, and the
    # summary; it has none of the three searches step 15 added.
    with open(TENTH_RUN, encoding="utf-8") as handle:
        run = json.load(handle)

    with open(EXPECTED, encoding="utf-8") as handle:
        expected = json.load(handle)["searches"]

    differing = {
        name: entry["rows"]
        for (name, entry) in run["searches"].items() if entry["rows"] != expected[name]["expected_rows"]
    }

    assert differing == {
        **_present(BEFORE_THE_DESK_SPOOF, run), **_present(BEFORE_THE_TRAJECTORIES, run), **BEFORE_THE_DEVICES
    }
    assert len(run["searches"]) == 75 and len(expected) == 85
    assert run["checks"]["risk_records"]["written"] == run["checks"]["risk_records"]["indexed"] == 213


BEFORE_THE_HOSTS = {
    "Behavior summary - per-layer scores": 7233,
    "R-B-L6-001 - New TLS client fingerprint": 5,
}
"""What every run before the host segment's sourcetype found where it changed the answer.

The summary gained the host segment's 1568 frames, one per workstation per hour and each its own group, and then the
week of handshakes and the slow tunnel's queries the daily drift over the TLS and DNS columns needed; the fingerprint
rule gained the creeping host's six acquisitions and its control's two on that week. The two searches step 8's second
part added, the host drift rule and the host watchlist's expiry, and the two its third part added, the stack and the
subdomain trajectories, are absent from the earlier runs and are not compared.
"""

ELEVENTH_RUN = os.path.join(VALIDATE, "search_head_runs", "2026-10-17T0126Z.json")
"""The run over step 15's events, which matched every expectation then written."""


def test_the_eleventh_run_differs_from_what_is_written_only_where_the_host_segment_arrived():
    # Made on 2026-10-17 when the switch and the router became entities: all seventy-eight then matched, with 221
    # risk records. Against the host segment's events and the trajectory week's it differs in the summary and the
    # fingerprint rule, and has none of the four searches step 8's host half brought. Nothing else moved: the host
    # corpus's machines share nothing with any other corpus's, and R-B-L7-004's lookup against the hosts' watchlist
    # finds none of its hosts there.
    with open(ELEVENTH_RUN, encoding="utf-8") as handle:
        run = json.load(handle)

    with open(EXPECTED, encoding="utf-8") as handle:
        expected = json.load(handle)["searches"]

    differing = {
        name: entry["rows"]
        for (name, entry) in run["searches"].items() if entry["rows"] != expected[name]["expected_rows"]
    }

    assert differing == {**BEFORE_THE_DESK_SPOOF, **BEFORE_THE_HOSTS}
    assert len(run["searches"]) == 78 and len(expected) == 85
    assert run["checks"]["risk_records"]["written"] == run["checks"]["risk_records"]["indexed"] == 221


TWELFTH_RUN = os.path.join(VALIDATE, "search_head_runs", "2026-10-22T1603Z.json")
"""The run over the host segment's events, which matched every expectation then written."""


def test_the_twelfth_run_differs_from_what_is_written_only_where_the_trajectory_week_arrived():
    # Made on 2026-10-22 when the host segment gained a sourcetype: all eighty then matched, with 224 risk records.
    # Against the events the daily drift over ja4_client_distinct and dns_subdomains_per_domain needed, it differs in
    # the summary and the fingerprint rule, and has neither of the two trajectory searches. Nothing else moved: the
    # slow tunnel never reaches R-B-L7-001's count, the creeping host's stacks are new to it and to nothing else, and
    # both rules' watchlist entries expire before R-B-L7-004 could read them, whose hosts are not theirs anyway.
    with open(TWELFTH_RUN, encoding="utf-8") as handle:
        run = json.load(handle)

    with open(EXPECTED, encoding="utf-8") as handle:
        expected = json.load(handle)["searches"]

    differing = {
        name: entry["rows"]
        for (name, entry) in run["searches"].items() if entry["rows"] != expected[name]["expected_rows"]
    }

    assert differing == {**BEFORE_THE_HTTP_AND_PROCESS_TRAJECTORIES, **BEFORE_THE_DESK_SPOOF, **BEFORE_THE_TRAJECTORIES}
    assert len(run["searches"]) == 80 and len(expected) == 85
    assert run["checks"]["risk_records"]["written"] == run["checks"]["risk_records"]["indexed"] == 224
    assert {row["rule_id"] for row in run["checks"]["host_watchlist"]} == {"R-P-L3-009"}


THIRTEENTH_RUN = os.path.join(VALIDATE, "search_head_runs", "2026-10-23T1125Z.json")
"""The run over the events the daily drift over the TLS and DNS columns added; it matched everything then written."""


def test_the_thirteenth_run_differs_from_what_is_written_only_where_the_desk_spoof_arrived():
    # Made on 2026-10-23 when the daily drift reached the TLS and DNS columns: all eighty-two then matched, with 233
    # risk records, and the host watchlist held R-P-L3-009's three, R-P-L6-006's three and R-P-L7-009's four. Against
    # the events the desk spoof regenerated it differs in the chain search, which it found empty for the last of its
    # four reasons: the two detections that accused a three-layer chain accused different chains; and against the
    # events section 3.3 added, in the summary.
    with open(THIRTEENTH_RUN, encoding="utf-8") as handle:
        run = json.load(handle)

    with open(EXPECTED, encoding="utf-8") as handle:
        expected = json.load(handle)["searches"]

    differing = {
        name: entry["rows"]
        for (name, entry) in run["searches"].items() if entry["rows"] != expected[name]["expected_rows"]
    }

    assert differing == {**BEFORE_THE_HTTP_AND_PROCESS_TRAJECTORIES, **BEFORE_THE_DESK_SPOOF}
    assert len(run["searches"]) == 82 and len(expected) == 85
    assert run["checks"]["risk_records"]["written"] == run["checks"]["risk_records"]["indexed"] == 233
    assert {
        row["total_risk"]: int(row["count"])
        for row in run["checks"]["three_layer_chain_risk"]
    } == {
        "0": 13, "50": 1, "55": 1
    }
    assert {row["rule_id"] for row in run["checks"]["host_watchlist"]} == {"R-P-L3-009", "R-P-L6-006", "R-P-L7-009"}


FOURTEENTH_RUN = os.path.join(VALIDATE, "search_head_runs", "2026-10-26T1224Z.json")
"""The run over the events the desk spoof and the network inventory added; it matched everything then written."""


def test_the_fourteenth_run_differs_from_what_is_written_only_where_the_http_and_process_trajectories_arrived():
    # Made on 2026-10-26 when the network inventory arrived: all eighty-three then matched, with 235 risk records, and
    # the host watchlist held the three host rules' entries. Against the events section 3.3 added it differs in the
    # summary alone, and has neither R-P-L7-010 nor R-P-L7-011.
    with open(FOURTEENTH_RUN, encoding="utf-8") as handle:
        run = json.load(handle)

    with open(EXPECTED, encoding="utf-8") as handle:
        expected = json.load(handle)["searches"]

    differing = {
        name: entry["rows"]
        for (name, entry) in run["searches"].items() if entry["rows"] != expected[name]["expected_rows"]
    }

    assert differing == BEFORE_THE_HTTP_AND_PROCESS_TRAJECTORIES
    assert len(run["searches"]) == 83 and len(expected) == 85
    assert set(expected) - set(run["searches"]) == {
        "R-P-L7-010 - Path count trajectory", "R-P-L7-011 - Process breadth trajectory"
    }
    assert run["checks"]["risk_records"]["written"] == run["checks"]["risk_records"]["indexed"] == 235
    assert {
        row["total_risk"]: int(row["count"])
        for row in run["checks"]["three_layer_chain_risk"]
    } == {
        "0": 13, "135": 1, "55": 1
    }
    assert {row["rule_id"] for row in run["checks"]["host_watchlist"]} == {"R-P-L3-009", "R-P-L6-006", "R-P-L7-009"}
