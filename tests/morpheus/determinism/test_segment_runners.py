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
same wire. The two branches beside the scored stream are held the same way: the runner's `morpheus:health` file is
`envelope.health` over the composition's result, and its `morpheus:edge` file, for a segment whose module names
`EDGE_HOPS`, is `edge_stream.emit` with them, each rendered through the wire. Then a restart: a runner stopped after
the first half of the data and started again on the second, with only its checkpoint directory between them, writes
what one uninterrupted run writes.
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
import edge_stream  # noqa: E402
import envelope  # noqa: E402
import host_identity  # noqa: E402
import segment_runner  # noqa: E402

from morpheus.messages import MessageMeta  # noqa: E402
from morpheus.stages.output.siem_wire_stage import SiemWireStage  # noqa: E402
from morpheus.utils.bucket_peak import DEFAULT_MIN_BUCKETS  # noqa: E402
from morpheus.utils.determinism_envelope import COLD_START  # noqa: E402
from morpheus.utils.siem_wire import to_wire_lines  # noqa: E402
from morpheus.utils.stage_checkpoint import snapshot_digest  # noqa: E402
from morpheus.utils.tracker_state import from_bytes  # noqa: E402

SEGMENTS = sorted(segment_runner.SEGMENTS)


def _wire(sourcetype: str, rows) -> list[str]:
    config = segment_runner._config()  # pylint: disable=protected-access
    stage = SiemWireStage(config, sourcetype=sourcetype, require_columns=False)

    return to_wire_lines(stage.on_data(MessageMeta(rows.reset_index(drop=True))).copy_dataframe())


def _branches(module, config, result) -> dict[str, list[str]]:
    """What the harnesses produce beside the scored rows: `envelope.health`, and the edges where hops are named."""
    lines: dict[str, list[str]] = {}

    if (hasattr(module, "PERIOD_SECONDS")):
        records = envelope.health(config,
                                  result,
                                  module.PERIOD_SECONDS,
                                  getattr(module, "HEALTH_MIN_BUCKETS", DEFAULT_MIN_BUCKETS))

        if (len(records) > 0):
            lines[segment_runner.HEALTH_SOURCETYPE] = _wire(segment_runner.HEALTH_SOURCETYPE, records)
    else:
        assert "window_id" not in result.columns or result["window_id"].isna().all(), "sealed rows and no period"

    if (hasattr(module, "EDGE_HOPS")):
        records = edge_stream.emit(config, result, **module.EDGE_HOPS)

        if (len(records) > 0):
            lines[segment_runner.EDGE_SOURCETYPE] = _wire(segment_runner.EDGE_SOURCETYPE, records)

    return lines


def _expected(segment: str) -> dict[str, list[str]]:
    """The in-memory composition's rows and its two branches, rendered through the same wire, per sourcetype."""
    import importlib  # pylint: disable=import-outside-toplevel

    module = importlib.import_module(segment_runner.SEGMENTS[segment]["module"])
    config = module.build_pipeline_config()
    result = module.run_pipeline(config, module.build_corpus())
    lines: dict[str, list[str]] = {}

    for (_, sourcetype, rows) in segment_runner.route(segment, result):
        lines.setdefault(sourcetype, []).extend(_wire(sourcetype, rows))

    lines.update(_branches(module, config, result))

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

    # Every segment that seals writes its health; the context segment seals nothing. No segment's module names hops
    # yet, so none writes an edge file: the test below turns the branch on for one and holds it to the harness.
    assert ("morpheus_health" in on_disk) == (segment != "context"), sorted(on_disk)
    assert "morpheus_edge" not in on_disk, sorted(on_disk)


NETWORK_HOPS = {
    "hop_method_columns": [host_identity.LEASE_METHOD_COLUMN, host_identity.INVENTORY_METHOD_COLUMN],
    "hop_uid_columns": [host_identity.LEASE_UID_COLUMN, host_identity.INVENTORY_UID_COLUMN],
    "hop_layers": [3, 7],
}
"""The network segment's resolution hops, were its module to name them: a flow's source address through the DHCP
lease at layer 3 and through the EDR inventory at layer 7."""


@pytest.mark.cpu_mode
def test_a_segment_naming_its_hops_writes_the_edges_the_harness_emits(tmp_path, monkeypatch):
    import network_pipeline  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(network_pipeline, "EDGE_HOPS", NETWORK_HOPS, raising=False)
    written = segment_runner.run("network", EXAMPLE / "samples" / "network", output_dir=tmp_path / "out")
    on_disk = sorted((tmp_path / "out" / "morpheus_edge.jsonlines").read_text(encoding="utf-8").splitlines())

    config = network_pipeline.build_pipeline_config()
    result = network_pipeline.run_pipeline(config, network_pipeline.build_corpus())
    edges = edge_stream.emit(config, result, **NETWORK_HOPS)

    assert written[segment_runner.EDGE_SOURCETYPE] == len(edges) > 0
    assert on_disk == sorted(_wire(segment_runner.EDGE_SOURCETYPE, edges))


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

def _files(directory: pathlib.Path) -> dict[str, list[str]]:
    return {path.stem: path.read_text(encoding="utf-8").splitlines() for path in directory.glob("*.jsonlines")}


def _without_resume_marker(lines: dict[str, list[str]]) -> dict[str, list[str]]:
    stripped = {}

    for (name, values) in lines.items():
        records = [json.loads(value) for value in values]

        # Every health count is compared as written. The fallback count was the one exception until plan section
        # 3.7: it read null from a run stopped before the scored week and zero from a run over the whole replay, and
        # it now answers from the window's own rows, which are the same either way.
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

    restarted = _files(tmp_path / "restarted")
    once = _files(tmp_path / "once")

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
