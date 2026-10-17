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
Every stage that remembers says what it remembers, and resumes from it as if it had never stopped.

The stages are driven by the frames `test_stage_parameter_liveness.py` already builds to make each stage's parameters
bite, which is to say frames that reach its state. Two properties are checked per stage:

- **The list is total.** After one frame, every attribute whose value changed is in `_checkpoint_attributes`. An
  attribute missing from the list is memory a restarted process would lose, and this is the only place that would
  notice before an operator did.
- **The list is enough.** A stage stopped after half the frame, snapshotted to bytes and restored into a new
  instance, emits for the second half exactly what one instance emits for both halves.
"""

import os
import sys
import typing

import pandas as pd
import pytest

from morpheus.messages import MessageMeta
from morpheus.pipeline.checkpoint_mixin import CheckpointMixin
from morpheus.utils.stage_checkpoint import CheckpointDirectory
from morpheus.utils.stage_checkpoint import restore_stages
from morpheus.utils.stage_checkpoint import snapshot_digest
from morpheus.utils.stage_checkpoint import snapshot_stages
from morpheus.utils.tracker_state import StateError
from morpheus.utils.tracker_state import encode
from morpheus.utils.tracker_state import from_bytes
from morpheus.utils.tracker_state import to_bytes

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "determinism"))

# pylint: disable=wrong-import-position
import test_stage_parameter_liveness as liveness  # noqa: E402

WARN_ONCE = ("a flag that keeps a configuration warning to one log line per process; a restarted process logging it "
             "once more is what an operator reading that process's log should see")

NOT_STATE = {
    "TC1ForecastStage": {
        "_warned_typeless": WARN_ONCE
    },
    "TC2AuthStage": {
        "_warned_portwise": WARN_ONCE
    },
    "TC0EnrichStage": {
        "_store": "the store sorts a fact's versions on first read and caches that; the store itself is a "
                  "constructor argument, rebuilt from the same context source on restart",
    },
}
"""Attributes that change under a frame and are deliberately not memory, with why."""

NOT_ENCODABLE = {"_input_ports", "_output_ports"}
"""Morpheus's own wiring, rebuilt by the pipeline; never state."""


def _variants(name: str) -> list[tuple[str, dict, typing.Optional[typing.Callable]]]:
    """The stage's own frame, then every parameter's two values over the frame built to make it bite.

    One frame reaches only the state its defaults reach -- `TC2BindingStage`'s snapshot memory is untouched unless
    snapshot mode is on -- so each check runs under every configuration the liveness registry already knows makes a
    difference.
    """
    scenario = liveness.REGISTRY[name]
    variants = [("base", {}, None)]

    for knob in scenario.knobs:
        if (knob.kind != liveness.DIFFERS):
            continue

        for (label, value) in (("benign", knob.benign), ("extreme", knob.extreme)):
            variants.append((f"{knob.name}={label}", {knob.name: value, **(knob.also or {})}, knob.frame))

    return variants


def _build(name: str, overrides: typing.Optional[dict] = None) -> typing.Any:
    scenario = liveness.REGISTRY[name]

    return scenario.stage(liveness._config(), **{**scenario.base, **(overrides or {})})  # pylint: disable=protected-access


def _frame(name: str, frame: typing.Optional[typing.Callable] = None) -> pd.DataFrame:
    return pd.DataFrame((frame or liveness.REGISTRY[name].frame)())


def _fingerprint(stage: typing.Any) -> dict:
    """Each attribute's value as encoded text, or its identity where it has no encoding."""
    found = {}

    for (attribute, value) in vars(stage).items():
        if (attribute in NOT_ENCODABLE):
            continue

        try:
            found[attribute] = repr(encode(value))
        except Exception:  # pylint: disable=broad-except
            found[attribute] = f"identity:{id(value)}"

    return found


def _emit(stage: typing.Any, frame: pd.DataFrame) -> list:
    meta = MessageMeta(frame.copy())
    emitted = stage.on_data(meta)

    return liveness._host_frames(emitted if emitted is not None else meta)  # pylint: disable=protected-access


def _complete(stage: typing.Any) -> list:
    if (not hasattr(stage, "on_completed")):
        return []

    remaining = stage.on_completed()

    return liveness._host_frames(list(remaining)) if remaining else []  # pylint: disable=protected-access


def _text(frames: list) -> str:
    return liveness._text(frames)  # pylint: disable=protected-access


STAGES = sorted(liveness.REGISTRY)


@pytest.mark.parametrize("name", STAGES)
def test_every_attribute_a_frame_changes_is_declared(name: str):
    changed = set()
    ran = 0

    for (_, overrides, frame) in _variants(name):
        try:
            stage = _build(name, overrides)
            before = _fingerprint(stage)
            _emit(stage, _frame(name, frame))
        except Exception:  # pylint: disable=broad-except
            # A value the stage refuses reaches no state; the liveness file asserts the refusal itself.
            continue

        ran += 1
        after = _fingerprint(stage)
        changed |= {attribute for (attribute, value) in after.items() if before.get(attribute) != value}

    cls = liveness.REGISTRY[name].stage
    declared = set(cls._checkpoint_attributes) if issubclass(cls, CheckpointMixin) else set()  # pylint: disable=protected-access
    excused = set(NOT_STATE.get(name, {}))

    assert ran > 0
    assert changed - excused <= declared, (f"{name} changed {sorted(changed - excused - declared)} and does not "
                                           f"list them in _checkpoint_attributes, so a restart would lose them")


@pytest.mark.parametrize("name",
                         [name for name in STAGES if issubclass(liveness.REGISTRY[name].stage, CheckpointMixin)])
def test_every_declared_attribute_exists(name: str):
    stage = _build(name)

    for attribute in stage._checkpoint_attributes:  # pylint: disable=protected-access
        assert hasattr(stage, attribute), f"{name} declares {attribute} and has no such attribute"


def _straight(name: str, overrides: dict, first: pd.DataFrame, second: pd.DataFrame) -> list:
    stage = _build(name, overrides)

    return _emit(stage, first) + _emit(stage, second) + _complete(stage)


def _stopped_and_resumed(name: str, overrides: dict, first: pd.DataFrame, second: pd.DataFrame) -> list:
    stopped = _build(name, overrides)
    output = _emit(stopped, first)

    if (isinstance(stopped, CheckpointMixin)):
        stopped.suspend_end_of_stream()
        # A stop for a checkpoint flushes nothing: what would have been flushed is the successor's to publish.
        assert not _complete(stopped)

    snapshot = from_bytes(to_bytes(snapshot_stages([stopped])))
    resumed = _build(name, overrides)
    restore_stages([resumed], snapshot)

    return output + _emit(resumed, second) + _complete(resumed)


@pytest.mark.parametrize("name", STAGES)
def test_a_restored_stage_continues_as_if_it_had_never_stopped(name: str):
    differing = []
    ran = 0

    for (label, overrides, frame_builder) in _variants(name):
        frame = _frame(name, frame_builder)
        half = len(frame) // 2
        (first, second) = (frame.iloc[:half].reset_index(drop=True), frame.iloc[half:].reset_index(drop=True))

        try:
            expected = _text(_straight(name, overrides, first, second))
        except Exception:  # pylint: disable=broad-except
            continue

        ran += 1

        if (_text(_stopped_and_resumed(name, overrides, first, second)) != expected):
            differing.append(label)

    assert ran > 0
    assert not differing, f"{name} resumed differently from an uninterrupted run under {differing}"


@pytest.mark.parametrize("name",
                         [name for name in STAGES if issubclass(liveness.REGISTRY[name].stage, CheckpointMixin)])
def test_the_same_rows_give_the_same_snapshot(name: str):
    (first, second) = (_build(name), _build(name))
    _emit(first, _frame(name))
    _emit(second, _frame(name))

    assert to_bytes(snapshot_stages([first])) == to_bytes(snapshot_stages([second]))


def test_a_snapshot_of_another_composition_is_refused():
    names = [name for name in STAGES if issubclass(liveness.REGISTRY[name].stage, CheckpointMixin)][:2]
    snapshot = snapshot_stages([_build(names[0]), _build(names[1])])

    with pytest.raises(StateError, match="different composition"):
        restore_stages([_build(names[1]), _build(names[0])], snapshot)


def test_a_snapshot_names_the_window_it_was_sealed_through():
    stage = _build("WindowSealStage")
    _emit(stage, _frame("WindowSealStage"))
    snapshot = snapshot_stages([stage])

    assert snapshot["sealed_through_window_id"] == stage.sealed_through_window_id
    assert snapshot["sealed_through_window_id"] is not None


def test_the_directory_returns_the_latest_checkpoint(tmp_path):
    directory = CheckpointDirectory(str(tmp_path / "state"))

    assert directory.latest() is None

    stage = _build("TC3TtlStage")
    directory.write(snapshot_stages([stage]))
    _emit(stage, _frame("TC3TtlStage"))
    second = snapshot_stages([stage])
    path = directory.write(second)

    assert os.path.basename(path) == "checkpoint-00000001.json"
    assert snapshot_digest(directory.latest()) == snapshot_digest(second)
    # Nothing but the checkpoints themselves is left behind.
    assert sorted(os.listdir(directory.path)) == ["checkpoint-00000000.json", "checkpoint-00000001.json"]


GUIDE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..",
                     "..",
                     "..",
                     "docs",
                     "source",
                     "developer_guide",
                     "guides",
                     "11_predictive_behavioral_analytics_osi.md")

NUMBER_WORDS = {
    "twenty": 20,
    "twenty-one": 21,
    "twenty-two": 22,
    "thirty-seven": 37,
    "thirty-eight": 38,
    "thirty-nine": 39,
    "forty": 40
}


def test_the_guide_counts_the_trackers_and_stages_that_remember():
    # The Part 6 row states both numbers; a stage or tracker added without a snapshot would leave it claiming more
    # coverage than exists, and one added with a snapshot would leave it claiming less.
    import re  # pylint: disable=import-outside-toplevel

    with open(GUIDE, encoding="utf-8") as handle:
        text = handle.read()

    trackers = re.search(r"All ([a-z-]+) trackers under `morpheus.utils`", text)
    stages = re.search(r"the ([a-z-]+) stages that remember", text)
    stateful = {name for name in STAGES if issubclass(liveness.REGISTRY[name].stage, CheckpointMixin)}
    utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..",
                             "..",
                             "..",
                             "python",
                             "morpheus",
                             "morpheus",
                             "utils")
    tracker_count = 0

    for file_name in sorted(os.listdir(utils_dir)):
        if (file_name.endswith(".py")):
            with open(os.path.join(utils_dir, file_name), encoding="utf-8") as handle:
                tracker_count += len(re.findall(r"^class \w+\(StateMixin\):", handle.read(), re.MULTILINE))

    assert NUMBER_WORDS.get(trackers.group(1)) == tracker_count
    assert NUMBER_WORDS.get(stages.group(1)) == len(stateful)
