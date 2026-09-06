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

import dataclasses

import pytest

from morpheus.utils.model_manifest import ModelManifest

ALICE = "alice@example.com"
BOB = "bob@example.com"

WINDOW = 484512


def manifest(**kwargs) -> ModelManifest:
    defaults = {"window_id": WINDOW, "models": {ALICE: "dfp-alice:14", BOB: "dfp-bob:3"}}
    defaults.update(kwargs)

    return ModelManifest(**defaults)


def test_an_entity_is_scored_against_its_own_model():
    resolution = manifest().resolve(ALICE, WINDOW)

    assert resolution.model_version == "dfp-alice:14"
    assert resolution.fallback_used is False


def test_the_manifest_refuses_a_window_it_was_not_pinned_for():
    # The control itself. A structure that will answer for the next window makes "held for the whole window" an
    # instruction rather than a guarantee, and a retraining event between the two is exactly what moves a score
    # for reasons that have nothing to do with the entity.
    with pytest.raises(ValueError, match="was resolved for window"):
        manifest().resolve(ALICE, WINDOW + 1)


def test_a_new_window_means_a_new_manifest():
    # Which is the moment a newly trained model is allowed in, and the only one.
    following = manifest(window_id=WINDOW + 1, models={ALICE: "dfp-alice:15"})

    assert following.resolve(ALICE, WINDOW + 1).model_version == "dfp-alice:15"


def test_an_entity_with_no_model_is_refused_when_no_fallback_is_declared():
    # DFPInferenceStage substitutes generic_user silently. Reporting a number produced by a population model as
    # though it were about this entity is worse than reporting none.
    with pytest.raises(ValueError, match="no fallback is declared"):
        manifest().resolve("carol@example.com", WINDOW)


def test_a_declared_fallback_is_used_and_says_so():
    resolution = manifest(fallback="dfp-generic:2").resolve("carol@example.com", WINDOW)

    assert resolution.model_version == "dfp-generic:2"
    assert resolution.fallback_used is True


def test_a_null_entity_takes_the_fallback_rather_than_somebody_elses_model():
    resolution = manifest(fallback="dfp-generic:2").resolve(None, WINDOW)

    assert resolution.fallback_used is True


def test_a_null_entity_with_no_fallback_is_refused():
    with pytest.raises(ValueError, match="no fallback is declared"):
        manifest().resolve(None, WINDOW)


@pytest.mark.parametrize("version", ["dfp-alice", "dfp-alice:", "", 14],
                         ids=["bare_name", "empty_version", "empty", "not_a_string"])
def test_a_model_that_does_not_pin_a_version_is_refused(version):
    # A bare name resolves to whatever is current, which is the "latest" resolution control 1 forbids. Caught
    # when the manifest is built rather than when a score turns out to be irreproducible six months later.
    with pytest.raises(ValueError, match="does not pin a version"):
        ModelManifest(window_id=WINDOW, models={ALICE: version})


def test_an_unpinned_fallback_is_refused_too():
    with pytest.raises(ValueError, match="does not pin a version"):
        ModelManifest(window_id=WINDOW, models={}, fallback="dfp-generic")


def test_the_manifest_cannot_be_changed_after_it_is_built():
    # Frozen, because "resolved once" is the control and a mutable manifest is a manifest that can be resolved
    # twice.
    subject = manifest()

    with pytest.raises(dataclasses.FrozenInstanceError):
        subject.window_id = WINDOW + 1


def test_the_entity_count_is_reported():
    assert manifest().entities == 2
    assert ModelManifest(window_id=WINDOW, models={}, fallback="dfp-generic:2").entities == 0


def test_resolving_twice_gives_the_same_answer():
    subject = manifest()

    assert subject.resolve(ALICE, WINDOW) == subject.resolve(ALICE, WINDOW)
