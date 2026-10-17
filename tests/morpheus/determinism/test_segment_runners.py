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
Each segment's runner, from its sample files to the bytes a SIEM receives, reproduces its golden.

Two hops, each asserted where it can be: the composition over the in-memory corpus reaches its golden file (every
harness asserts that), and here the runner -- `FileSourceStage` over the committed samples, the same composition,
`SiemWireStage` and `SiemWireSinkStage` -- writes exactly the lines that in-memory result renders to through the
same wire. Then a restart: a runner stopped after the first half of the data and started again on the second,
with only its checkpoint directory between them, writes what one uninterrupted run writes.
"""

import json
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = HERE.parents[2] / "examples" / "behavioral_analytics"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXAMPLE))

# pylint: disable=wrong-import-position
import segment_runner  # noqa: E402

from morpheus.messages import MessageMeta  # noqa: E402
from morpheus.stages.output.siem_wire_stage import SiemWireStage  # noqa: E402
from morpheus.utils.determinism_envelope import COLD_START  # noqa: E402
from morpheus.utils.siem_wire import to_wire_lines  # noqa: E402
from morpheus.utils.stage_checkpoint import snapshot_digest  # noqa: E402
from morpheus.utils.tracker_state import from_bytes  # noqa: E402

SEGMENTS = sorted(segment_runner.SEGMENTS)


def _expected(segment: str) -> dict[str, list[str]]:
    """The in-memory composition's rows, rendered through the same wire, per sourcetype."""
    import importlib  # pylint: disable=import-outside-toplevel

    module = importlib.import_module(segment_runner.SEGMENTS[segment]["module"])
    result = module.run_pipeline(module.build_pipeline_config(), module.build_corpus())
    lines: dict[str, list[str]] = {}

    for (_, sourcetype, rows) in segment_runner.route(segment, result):
        config = segment_runner._config()  # pylint: disable=protected-access
        stage = SiemWireStage(config, sourcetype=sourcetype, require_columns=False)
        rendered = stage.on_data(MessageMeta(rows.reset_index(drop=True))).copy_dataframe()
        lines.setdefault(sourcetype, []).extend(to_wire_lines(rendered))

    return lines


def _by_file(lines: dict[str, list[str]]) -> dict[str, list[str]]:
    return {sourcetype.replace(":", "_"): sorted(values) for (sourcetype, values) in lines.items()}


@pytest.mark.cpu_mode
@pytest.mark.parametrize("segment", SEGMENTS)
def test_the_runner_writes_what_the_composition_renders(segment: str, tmp_path):
    written = segment_runner.run(segment, EXAMPLE / "samples" / segment, output_dir=tmp_path / "out")
    on_disk = {
        path.stem: sorted(path.read_text(encoding="utf-8").splitlines())
        for path in (tmp_path / "out").glob("*.jsonlines")
    }

    assert sum(written.values()) > 0
    assert on_disk == _by_file(_expected(segment))


def _split_inputs(segment: str, target: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """The segment's samples cut at one instant shared by every class: before it, and from it on."""
    records = {}

    for path in sorted((EXAMPLE / "samples" / segment).glob("*.jsonlines")):
        records[path.name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    times = sorted(record["event_time"] for rows in records.values() for record in rows)
    cut = times[len(times) // 2]
    (first, second) = (target / "first", target / "second")

    for directory in (first, second):
        directory.mkdir(parents=True)

    for (name, rows) in records.items():
        for (directory, keep) in ((first, lambda t: t < cut), (second, lambda t: t >= cut)):
            with open(directory / name, "w", encoding="utf-8") as out:
                for record in rows:
                    if (keep(record["event_time"])):
                        out.write(json.dumps(record) + "\n")

    return (first, second)


RESTARTABLE = ("endpoint", "saas", "session")
"""Segments whose stopped-and-restarted output can equal one uninterrupted run's.

The others rebuild a lookup from bindings their own run closes -- the DHCP server's leases, the EDR inventory, the
switch's MAC table -- and a binding still open when the first run stops is closed, and so in the lookup, only in the
second. A deployment holds those bindings in the SIEM's lookups, refreshed from `binding:*`; the compositions build
them per run, so for them this equality is not the property to ask for. Their stages restore all the same."""


def _without_resume_marker(lines: dict[str, list[str]]) -> dict[str, list[str]]:
    stripped = {}

    for (name, values) in lines.items():
        records = [json.loads(value) for value in values]

        for record in records:
            record.pop("resumed_from_state", None)

        stripped[name] = sorted(json.dumps(record, sort_keys=True) for record in records)

    return stripped


@pytest.mark.slow
@pytest.mark.cpu_mode
@pytest.mark.parametrize("segment", RESTARTABLE)
def test_a_restarted_runner_writes_what_one_run_writes(segment: str, tmp_path):
    (first, second) = _split_inputs(segment, tmp_path / "inputs")
    state = tmp_path / "state"

    segment_runner.run(segment, first, output_dir=tmp_path / "restarted", state_dir=state, final=False)
    checkpoint = sorted(state.glob("checkpoint-*.json"))

    assert len(checkpoint) == 1

    segment_runner.run(segment, second, output_dir=tmp_path / "restarted", state_dir=state, final=True)
    segment_runner.run(segment, EXAMPLE / "samples" / segment, output_dir=tmp_path / "once")

    restarted = {
        path.stem: path.read_text(encoding="utf-8").splitlines()
        for path in (tmp_path / "restarted").glob("*.jsonlines")
    }
    once = {path.stem: path.read_text(encoding="utf-8").splitlines() for path in (tmp_path / "once").glob("*.jsonlines")}

    assert _without_resume_marker(restarted) == _without_resume_marker(once)

    # And the rows the second process scored name the checkpoint it resumed from; the first process's say it began
    # cold.
    digest = snapshot_digest(from_bytes(checkpoint[0].read_bytes()))
    markers = {json.loads(line).get("resumed_from_state") for lines in restarted.values() for line in lines} - {None}

    assert markers == {COLD_START, digest}


POLICY = EXAMPLE / "minimization_policy.example.yaml"
KEY = "an-estate-key-of-sufficient-length"


def _principals(segment: str) -> set[str]:
    """Every principal any of the segment's inputs names."""
    found = set()

    for path in (EXAMPLE / "samples" / segment).glob("*.jsonlines"):
        for line in path.read_text(encoding="utf-8").splitlines():
            for (name, value) in json.loads(line).items():
                if (name in ("user_principal", "owner", "manager", "service_account") and isinstance(value, str)):
                    found.add(value.lower())

    return found


@pytest.mark.cpu_mode
@pytest.mark.parametrize("segment", SEGMENTS)
def test_under_the_example_policy_no_principal_reaches_the_wire(segment: str, tmp_path, monkeypatch):
    monkeypatch.setenv("MORPHEUS_MINIMIZATION_KEY", KEY)
    segment_runner.run(segment, EXAMPLE / "samples" / segment, output_dir=tmp_path / "out", policy_path=POLICY)

    principals = _principals(segment)

    if (segment in ("context", "saas", "session")):
        assert principals, "the check has nobody to look for"

    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in (tmp_path / "out").glob("*.jsonlines"))

    assert text, "the runner wrote nothing"
    # Substrings, not field values: a principal inside a model version or an entity key is still the principal.
    assert sorted(principal for principal in principals if principal in text) == []


@pytest.mark.cpu_mode
def test_a_policy_without_its_key_does_not_run(tmp_path, monkeypatch):
    monkeypatch.delenv("MORPHEUS_MINIMIZATION_KEY", raising=False)

    with pytest.raises(KeyError, match="MORPHEUS_MINIMIZATION_KEY"):
        segment_runner.run("saas", EXAMPLE / "samples" / "saas", output_dir=tmp_path / "out", policy_path=POLICY)


@pytest.mark.cpu_mode
def test_a_cold_start_refuses_a_directory_that_already_holds_records(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "morpheus_score_l7.jsonlines").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        segment_runner.run("saas", EXAMPLE / "samples" / "saas", output_dir=tmp_path / "out")


def test_every_segment_has_a_runner_and_a_sample():
    for segment in SEGMENTS:
        assert (EXAMPLE / f"run_{segment}.py").is_file(), segment
        assert any((EXAMPLE / "samples" / segment).glob("*.jsonlines")), segment
