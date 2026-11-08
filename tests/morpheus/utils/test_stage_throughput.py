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
A change that halves a stage's throughput is visible (guide 13, plan item 15).

`ci/scripts/stage_throughput.py` times every telemetry and lineage stage's `on_data` over the segments' samples and
compares the result with `ci/artifacts/stage_throughput_baseline.json`. These hold it to measuring the stage that did
the work, to normalizing by the machine, and to flagging a halving and nothing within the noise.
"""

import importlib.util
import json
import os
import time

import pandas as pd

from morpheus.messages import MessageMeta

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
MODULE_PATH = os.path.join(REPO_ROOT, "ci", "scripts", "stage_throughput.py")


def _load():
    spec = importlib.util.spec_from_file_location("stage_throughput", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


stage_throughput = _load()


class _Base:

    def on_data(self, message):
        time.sleep(0.001)
        return message


class _Inherits(_Base):
    pass


class _Overrides(_Base):

    def on_data(self, message):
        return super().on_data(message)


def test_a_stage_is_measured_as_itself_and_a_call_to_its_base_once():
    message = MessageMeta(pd.DataFrame({"a": range(5)}))
    original = vars(_Base)["on_data"]
    totals: dict = {}

    with stage_throughput.instrumented([_Base, _Overrides], totals):
        _Inherits().on_data(message)
        _Overrides().on_data(message)

    assert {name: (entry["rows"], entry["calls"]) for (name, entry) in totals.items()} == {
        "_Inherits": (5, 1), "_Overrides": (5, 1)
    }
    assert vars(_Base)["on_data"] is original


def test_every_telemetry_and_lineage_stage_is_measured():
    names = {cls.__name__ for cls in stage_throughput.measured_classes()}

    assert {"TelemetryStage", "TC1NormalizeStage", "LineageStampStage", "WindowSealStage"} <= names


def _run(seconds: dict, rows: int = 1000, unit: float = 0.1) -> dict:
    stages = {name: {"seconds": value, "rows": rows, "calls": 1} for (name, value) in seconds.items()}

    return {"unit_seconds": unit, "stages": stage_throughput.throughput(stages, unit)}


def test_a_halved_stage_is_flagged_and_one_within_the_noise_is_not():
    baseline = _run({"Slower": 0.1, "Noisy": 0.1, "Steady": 0.1})
    current = _run({"Slower": 0.25, "Noisy": 0.15, "Steady": 0.1})

    comparison = stage_throughput.compare(current, baseline)

    assert comparison["halved"] == ["Slower"]
    assert round(comparison["ratios"]["Noisy"], 2) == 0.67


def test_a_slower_machine_is_not_a_slower_stage():
    baseline = _run({"Stage": 0.1}, unit=0.1)
    current = _run({"Stage": 0.2}, unit=0.2)

    assert stage_throughput.compare(current, baseline)["ratios"] == {"Stage": 1.0}


def test_a_stage_too_quick_to_measure_is_reported_but_not_compared():
    baseline = _run({"Quick": 0.001, "Gone": 0.1})
    current = _run({"Quick": 0.004, "New": 0.1})

    comparison = stage_throughput.compare(current, baseline)

    assert (comparison["ratios"], comparison["new"], comparison["gone"]) == ({}, ["New"], ["Gone"])
    assert "`Quick`" in stage_throughput.render(current, comparison)


def test_the_summary_marks_a_halving():
    baseline = _run({"Slower": 0.1})
    current = _run({"Slower": 0.3})

    assert "**halved**" in stage_throughput.render(current, stage_throughput.compare(current, baseline))


def test_the_baseline_names_every_stage_the_segments_run():
    with open(os.path.join(REPO_ROOT, stage_throughput.BASELINE), encoding="utf-8") as handle:
        baseline = json.load(handle)

    measured = {cls.__name__ for cls in stage_throughput.measured_classes()}

    assert baseline["unit_seconds"] > 0
    assert set(baseline["stages"]) <= measured
    assert {"LineageStampStage", "EnvelopeStampStage", "TC1NormalizeStage", "TC3BaselineStage"} <= set(
        baseline["stages"])
