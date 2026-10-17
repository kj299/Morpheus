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
Control 13's seventh check: a pipeline stopped halfway and resumed from its snapshot reaches its golden.

The cross-restart check (the third) proves a fresh process over the same corpus reaches the same output, which is a
statement about replay. A deployment does not replay; it stops, and the process that replaces it carries on from
wherever the old one got to. This is that statement, over all twelve composed pipelines: every linear segment in
each is stopped at its halfway row with open windows and bindings kept, snapshotted to bytes, restored into stages
built afresh from the same arguments, and finished -- and the result must be the same golden the uninterrupted run
is held to. `resumable.py` describes the mechanics.
"""

import os
import sys

import pytest

from morpheus.config import ExecutionMode
from morpheus.stages.lineage.window_seal_stage import WindowSealStage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import resumable  # noqa: E402
import test_gpu_parity  # noqa: E402

CORPORA = test_gpu_parity.CORPORA

STATELESS = {
    "context": "the context pipeline resolves identity, asset and application records against bitemporal stores "
               "built from its own corpus, and none of its stages keeps a history of its own",
}
"""Compositions with no stage that remembers, so a cut carries nothing across. Listed so one gaining memory is seen."""


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("corpus_name", CORPORA)
def test_a_stopped_and_resumed_pipeline_reaches_its_golden(corpus_name: str, execution_mode: ExecutionMode):
    with resumable.stop_and_resume() as record:
        test_gpu_parity._reaches_golden(corpus_name, execution_mode)  # pylint: disable=protected-access

    # Not vacuous: every segment was cut, and at least one carried memory across the cut.
    assert record.segments > 0
    assert (record.stateful_segments > 0) == (corpus_name not in STATELESS)


@pytest.mark.cpu_mode
def test_the_check_fails_when_a_stage_forgets_part_of_its_memory(monkeypatch):
    # The negative control. A window seal that resumes without the windows it was holding loses their rows, and
    # the comparison must notice; if it does not, the check above is passing without looking.
    monkeypatch.setattr(WindowSealStage, "_checkpoint_attributes", ("_sealer", "_emit_control_messages"))

    with resumable.stop_and_resume():
        with pytest.raises(AssertionError):
            test_gpu_parity._reaches_golden("lineage", ExecutionMode.CPU)  # pylint: disable=protected-access


@pytest.mark.cpu_mode
def test_outside_the_block_a_pipeline_runs_once():
    # The same compositions serve every other check, so the resumable pipeline must be inert unless asked.
    with resumable.stop_and_resume() as record:
        pass

    test_gpu_parity._reaches_golden("lineage", ExecutionMode.CPU)  # pylint: disable=protected-access

    assert record.segments == 0
