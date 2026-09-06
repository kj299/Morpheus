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

from morpheus.config import Config
from morpheus.utils.determinism_envelope import TIERS
from morpheus.utils.determinism_envelope import UNKNOWN
from morpheus.utils.determinism_envelope import DeterminismEnvelope
from morpheus.utils.determinism_envelope import config_hash
from morpheus.utils.determinism_envelope import pipeline_fingerprint

COMPONENTS = {
    "configuration": "7d2e4a1f9c3b5e80",
    "schema_versions": {
        "TC-5": "2.1.0"
    },
    "thresholds": {
        "R-D-L5-003": 900, "R-D-L5-004": 4
    },
    "code_commit": "c6a3b56",
    "image_digest": "sha256:1f0c",
}


def envelope(**kwargs) -> DeterminismEnvelope:
    defaults = {
        "tier": "D1",
        "fingerprint": "a3f9c2e1b8d47506",
        "configuration": "7d2e4a1f9c3b5e80",
        "code_commit": "c6a3b56",
        "image_digest": "sha256:1f0c",
        "feature_schema_version": "TC-5/2.1.0",
        "rng_seed": 42,
    }
    defaults.update(kwargs)

    return DeterminismEnvelope(**defaults)


def test_the_configuration_hashes_to_the_same_value_twice():
    config = Config()

    assert config_hash(config) == config_hash(config)
    assert len(config_hash(config)) == 16


def test_a_changed_configuration_hashes_differently():
    config = Config()
    before = config_hash(config)
    config.pipeline_batch_size = config.pipeline_batch_size * 2

    assert config_hash(config) != before


def test_the_fingerprint_is_stable_across_dictionary_order():
    # The components arrive as dictionaries, and two runs meaning the same thing must not disagree because one
    # of them happened to build a dictionary in a different order.
    reordered = dict(COMPONENTS)
    reordered["thresholds"] = {"R-D-L5-004": 4, "R-D-L5-003": 900}

    assert pipeline_fingerprint(**COMPONENTS) == pipeline_fingerprint(**reordered)


def test_a_threshold_carried_in_a_float_does_not_fork_the_fingerprint():
    # Reading a column that holds a gap gives floats on the host, so the same threshold reaches this as 900 in
    # one run and 900.0 in another. Forking on that would make every event before the change incomparable with
    # every event after it, for no reason at all.
    as_float = {**COMPONENTS, "thresholds": {"R-D-L5-003": 900.0, "R-D-L5-004": 4.0}}

    assert pipeline_fingerprint(**as_float) == pipeline_fingerprint(**COMPONENTS)


def test_a_boolean_threshold_does_not_collide_with_one():
    # A bool is an int in Python, so rendering True as 1 would make a flag and a threshold of one hash alike.
    flagged = {**COMPONENTS, "thresholds": {"R-D-L5-003": True}}
    numeric = {**COMPONENTS, "thresholds": {"R-D-L5-003": 1}}

    assert pipeline_fingerprint(**flagged) != pipeline_fingerprint(**numeric)


@pytest.mark.parametrize("component", sorted(COMPONENTS), ids=sorted(COMPONENTS))
def test_changing_any_component_changes_the_fingerprint(component):
    # Every one of them decides what a score means, so a fingerprint that ignored one would claim two events are
    # comparable when they are not.
    changed = dict(COMPONENTS)
    changed[component] = ({
        "changed": "changed"
    } if isinstance(COMPONENTS[component], dict) else COMPONENTS[component] + "-changed")

    assert pipeline_fingerprint(**changed) != pipeline_fingerprint(**COMPONENTS)


@pytest.mark.parametrize("component", sorted(COMPONENTS), ids=sorted(COMPONENTS))
@pytest.mark.parametrize("missing", [None, "", {}], ids=["none", "empty_string", "empty_dict"])
def test_an_omitted_component_is_refused_rather_than_skipped(component, missing):
    # Silently skipping what is not known is what the fingerprint exists to prevent.
    with pytest.raises(ValueError, match="Every component is required"):
        pipeline_fingerprint(**{**COMPONENTS, component: missing})


def test_an_unknown_component_is_hashed_rather_than_hidden():
    # A deployment that genuinely cannot determine the commit says so, and its events are then visibly not
    # comparable with events produced where it was known. That is the honest outcome.
    unknown = {**COMPONENTS, "code_commit": UNKNOWN}

    assert pipeline_fingerprint(**unknown) != pipeline_fingerprint(**COMPONENTS)
    assert len(pipeline_fingerprint(**unknown)) == 16


def test_the_envelope_reaches_a_siem_as_flat_columns():
    columns = envelope().to_columns()

    assert columns["determinism_tier"] == "D1"
    assert columns["pipeline_fingerprint"] == "a3f9c2e1b8d47506"
    assert columns["config_hash"] == "7d2e4a1f9c3b5e80"
    assert columns["rng_seed"] == 42
    assert columns["feature_schema_version"] == "TC-5/2.1.0"


@pytest.mark.parametrize("tier", TIERS)
def test_every_declared_tier_is_accepted(tier):
    assert envelope(tier=tier).to_columns()["determinism_tier"] == tier


@pytest.mark.parametrize("tier", ["D4", "d1", "", "bit-exact"], ids=["unknown", "lowercase", "empty", "prose"])
def test_a_tier_a_consumer_cannot_interpret_is_refused(tier):
    with pytest.raises(ValueError, match="is not one of"):
        envelope(tier=tier)


def test_the_envelope_cannot_be_changed_after_it_is_built():
    with pytest.raises(dataclasses.FrozenInstanceError):
        envelope().tier = "D0"
