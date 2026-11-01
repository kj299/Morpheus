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
The telemetry base class, and the event-clock guard it carries (guide 13, plan item 8; plan section 4.3).

The plan's measure of done is that one far-future row cannot expire every window in any stage. That is tested
directly, over every stage that keeps a clock and over the corpus the parameter-liveness registry already builds to
make that stage's parameters bite: the same frame is run twice, once as it is and once with a copy of its middle row
moved ten years ahead, and every other row must come out of the second run exactly as it came out of the first.
"""

import importlib
import inspect
import os
import pkgutil
import re
import sys

import pandas as pd
import pytest

from morpheus.messages import MessageMeta
from morpheus.stages import telemetry
from morpheus.stages.telemetry.telemetry_base import TelemetryStage
from morpheus.utils.event_clock import CLOCK_REFUSED_COLUMN

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "determinism"))

# pylint: disable=wrong-import-position
import test_stage_parameter_liveness as liveness  # noqa: E402

TEN_YEARS_NS = 3650 * 24 * 3600 * 10**9

NS_PER_UNIT = {"ns": 1, "us": 10**3, "ms": 10**6, "s": 10**9}

KEEPS_NO_EVENT_TIME_STATE = {
    "TC5DriftStage": "its history is a sequence of windows keyed by window_id, and it reads no event time",
    "TC5ScoreStage": "it scores each row against the model its window's manifest names, and holds no history",
    "TC4AbpScoreStage": "it scores each flow on that flow's own features and holds no history",
    "TC6ContentStage": "it compares each response's declared and sniffed types, row by row, with no history",
    "TC7SidScoreStage": "it scores each request body on its own text and holds no history",
}
"""The TC-3 to TC-7 stages without a clock, each with the reason it needs none.

A stage belongs here only if nothing it remembers moves with event time; a stage that takes an event-time column
and keeps any per-entity state must take `max_clock_skew_seconds` instead."""


def _layer_stages() -> dict:
    """Every TC-3 to TC-7 stage class, by name, from the modules on disk rather than a list kept here."""
    found = {}

    for module_info in pkgutil.iter_modules(telemetry.__path__):
        if (re.match(r"tc[3-7]_.*_stage$", module_info.name)):
            module = importlib.import_module(f"{telemetry.__name__}.{module_info.name}")

            for (name, value) in vars(module).items():
                if (inspect.isclass(value) and value.__module__ == module.__name__ and name.endswith("Stage")):
                    found[name] = value

    return found


def _guarded_scenarios() -> list:
    return sorted(name for (name, scenario) in liveness.REGISTRY.items()
                  if "max_clock_skew_seconds" in inspect.signature(scenario.stage.__init__).parameters)


def _parameter(scenario, name: str):
    if (name in scenario.base):
        return scenario.base[name]

    return inspect.signature(scenario.stage.__init__).parameters[name].default


def _ahead(value, time_unit: str):
    """`value` moved ten years later, in the representation it arrived in."""
    if (isinstance(value, (int, float)) and not isinstance(value, bool)):
        return type(value)(value + TEN_YEARS_NS // NS_PER_UNIT[time_unit])

    moved = pd.Timestamp(value) + pd.Timedelta(TEN_YEARS_NS, unit="ns")

    return moved.isoformat() if isinstance(value, str) else moved


def _with_one_row_from_the_future(frame: dict, time_column: str, time_unit: str) -> tuple[dict, int]:
    """The frame with a copy of its middle row, ten years ahead, inserted after it."""
    length = len(frame[time_column])
    middle = length // 2
    injected = {}

    for (column, values) in frame.items():
        values = list(values)
        copy = _ahead(values[middle], time_unit) if column == time_column else values[middle]
        injected[column] = values[:middle + 1] + [copy] + values[middle + 1:]

    return (injected, middle + 1)


def _output(scenario, frame: dict) -> pd.DataFrame:
    frames = liveness._run(scenario, {}, frame=frame)  # pylint: disable=protected-access

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _comparable(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[CLOCK_REFUSED_COLUMN], errors="ignore").reset_index(drop=True).astype(str)


def test_every_layer_three_to_seven_stage_is_a_telemetry_stage():
    stages = _layer_stages()

    assert len(stages) >= 25
    assert sorted(name for (name, cls) in stages.items() if not issubclass(cls, TelemetryStage)) == []


def test_every_stage_that_reads_event_time_keeps_a_clock_or_says_why_not():
    # The gap plan item 8 named: four layer 5 stages kept windows, histories and a last sighting on event time with
    # no clock. A stage added later with the same omission fails here rather than in an incident.
    unguarded = []

    for (name, cls) in sorted(_layer_stages().items()):
        parameters = inspect.signature(cls.__init__).parameters

        if ("max_clock_skew_seconds" in parameters):
            assert name not in KEEPS_NO_EVENT_TIME_STATE, f"{name} keeps a clock; remove it from the exemptions"
        elif (name not in KEEPS_NO_EVENT_TIME_STATE):
            unguarded.append(name)

    assert unguarded == []


def test_every_exemption_is_a_stage_that_exists():
    assert set(KEEPS_NO_EVENT_TIME_STATE) <= set(_layer_stages())


@pytest.mark.parametrize("stage_name", _guarded_scenarios())
def test_one_row_from_the_future_changes_no_other_row(stage_name: str):
    scenario = liveness.REGISTRY[stage_name]
    time_column = _parameter(scenario, "time_column")
    time_unit = _parameter(scenario, "time_unit") if "time_unit" in inspect.signature(
        scenario.stage.__init__).parameters else "ns"
    frame = scenario.frame()

    (injected, _) = _with_one_row_from_the_future(frame, time_column, time_unit)

    before = _output(scenario, frame)
    after = _output(scenario, injected)

    if (CLOCK_REFUSED_COLUMN not in after.columns):
        # A stage whose output is not one row per input -- a binding record per lease -- has nowhere to mark the row,
        # so the refused row must leave no trace at all.
        pd.testing.assert_frame_equal(_comparable(after), _comparable(before))

        return

    flagged = after[CLOCK_REFUSED_COLUMN].fillna(False).astype(bool)

    # The guard's own record: the row is marked. Where there is one output row per input row, it is the only one.
    assert int(flagged.sum()) >= 1

    if (len(after) == len(injected[time_column])):
        assert int(flagged.sum()) == 1

    # Every other row is what it was. A stage that let the row through would have expired its windows, closed its
    # sessions or reset its baselines behind it, and the rows after it would differ.
    pd.testing.assert_frame_equal(_comparable(after[~flagged]), _comparable(before))


def test_the_check_has_teeth():
    # Negative control: the same comparison over a stage whose clock is effectively removed must fail, or the test
    # above could pass because the corpus never let a far-future row matter.
    scenario = liveness.REGISTRY["TC3BaselineStage"]
    frame = scenario.frame()
    (injected, _) = _with_one_row_from_the_future(frame, _parameter(scenario, "time_column"), "ns")

    loose = liveness.Scenario(stage=scenario.stage,
                              frame=scenario.frame,
                              base={
                                  **scenario.base, "max_clock_skew_seconds": 2 * TEN_YEARS_NS // 10**9
                              },
                              knobs=())
    after = _output(loose, injected)
    before = _output(loose, frame)

    without_injected = after.drop(index=len(frame[_parameter(scenario, "time_column")]) // 2 + 1)

    assert not _comparable(without_injected).equals(_comparable(before))


def test_an_empty_message_is_returned_untouched():
    stage = liveness.REGISTRY["TC5RiskStage"].stage(liveness._config())  # pylint: disable=protected-access
    meta = MessageMeta(pd.DataFrame({"event_time": pd.Series([], dtype="int64")}))

    assert stage.on_data(meta) is meta


def test_a_missing_column_is_refused_with_the_stage_s_name():
    stage = liveness.REGISTRY["TC5RiskStage"].stage(liveness._config())  # pylint: disable=protected-access

    with pytest.raises(KeyError, match="TC5RiskStage requires columns"):
        stage.on_data(MessageMeta(pd.DataFrame({"unrelated": [1]})))
