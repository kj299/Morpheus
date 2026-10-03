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
The determinism envelope every composed corpus stamps on its rows: control 12, and control 1's model columns.

Every pipeline under this directory builds its envelope here, so the twelve agree on what the fields mean.
`configuration` digests the settings that decide the output rather than the `Config` (which carries the execution
mode and would fork the envelope between the CPU and GPU runs control 13's parity check compares);
`fingerprint` folds in the feature schema version and the thresholds the layer's shipped rules state, read from the
app's stanzas so a retuned rule changes the fingerprint of every event scored under it; the code commit and image
digest are `UNKNOWN`, honestly, because a corpus is not a deployment and a golden that changed on every commit
would prove nothing. A deployment supplies all three.
"""

import configparser
import os
import re
import typing

from morpheus.utils.determinism_envelope import UNKNOWN
from morpheus.utils.determinism_envelope import DeterminismEnvelope
from morpheus.utils.determinism_envelope import pipeline_fingerprint
from morpheus.utils.determinism_envelope import settings_hash

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
SAVED_SEARCHES = os.path.join(REPO_ROOT,
                              "examples",
                              "splunk_lineage_app",
                              "TA-morpheus-lineage",
                              "default",
                              "savedsearches.conf")

SCHEMA_VERSION = "1.0.0"
"""The first versioned feature schema. Every class is at it; a class whose columns change meaning bumps its own."""

DEFAULT_TIER = "D1"
"""Decision-stable, which is what the harnesses assert: the same rules fire on the same rows."""

_EVAL_ASSIGNMENT = re.compile(r"\|\s*eval\s+([^|]*)")
_NUMBER_ASSIGNMENT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)(?![\w.])")
_COMPARISON = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)(?![\w.])")


def _searches() -> dict:
    with open(SAVED_SEARCHES, encoding="utf-8") as handle:
        folded = re.sub(r"\\\s*\r?\n\s*", " ", handle.read())

    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read_string(folded)

    return {name: parser[name]["search"] for name in parser.sections() if parser.has_option(name, "search")}


def rule_thresholds(rule_ids: typing.Sequence[str]) -> dict:
    """
    Every numeric threshold the named rules' stanzas state, keyed by rule and then by the name it is stated under.

    An `eval` assignment of a number (`kmh_threshold = 900`) and a comparison against one (`dsts_per_src>50`)
    both count; a threshold a rule states both ways is recorded once under each name, which is harmless to a
    digest. Read from the app rather than restated, so the fingerprint on an event follows the rule that would
    read it.
    """
    searches = _searches()
    thresholds: dict = {}

    for rule_id in rule_ids:
        stanza = next((name for name in searches if name.startswith(rule_id + " ")), None)

        if (stanza is None):
            raise ValueError(f"no saved search for {rule_id}; the fingerprint names rules the app ships")

        found: dict = {}
        search = searches[stanza]

        for clause in _EVAL_ASSIGNMENT.findall(search):
            for (name, value) in _NUMBER_ASSIGNMENT.findall(clause):
                found[name] = float(value)

        for (name, operator, value) in _COMPARISON.findall(search):
            found[f"{name}{operator}"] = float(value)

        thresholds[rule_id] = found

    return thresholds


def envelope_for(telemetry_class: str,
                 settings: dict,
                 rules: typing.Sequence[str] = (),
                 tier: str = DEFAULT_TIER,
                 schema_version: str = SCHEMA_VERSION,
                 rng_seed: int = 0) -> DeterminismEnvelope:
    """
    The envelope one corpus stamps: its class, the settings that decide its output, and its rules' thresholds.

    Parameters
    ----------
    telemetry_class : str
        The class whose feature schema this is, as Part 2 names it (`TC-3`), or a name for a composition of
        several (`estate`).
    settings : dict
        The parameters that decide the output: seal period, lateness horizon, and the stages' own.
    rules : sequence of str
        The identifiers of the shipped rules that read this corpus's columns.
    tier : str
        The determinism tier the harness asserts.
    schema_version : str
        The feature schema version.
    rng_seed : int
        The seed a model path would be handed; the arithmetic corpora have none and record zero.
    """
    configuration = settings_hash(settings)
    fingerprint = pipeline_fingerprint(configuration=configuration,
                                       schema_versions={telemetry_class: schema_version},
                                       thresholds=rule_thresholds(rules) or {"none": {}},
                                       code_commit=UNKNOWN,
                                       image_digest=UNKNOWN)

    return DeterminismEnvelope(tier=tier,
                               fingerprint=fingerprint,
                               configuration=configuration,
                               code_commit=UNKNOWN,
                               image_digest=UNKNOWN,
                               feature_schema_version=f"{telemetry_class}/{schema_version}",
                               rng_seed=rng_seed)
