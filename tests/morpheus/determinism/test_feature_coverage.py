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
Every feature is read by a shipped search, is an input of a committed model, or is listed with the reason it is neither.

The retrospective found a dozen host features emitted and read by nothing, and no way to know it short of reading
every stage against every search. This makes the question a test. The feature stages of every composed corpus --
the layer 1 and 2 telemetry compositions, the layer 3, 4 and 6 host compositions, the layer 5 session composition,
and the layer 7 DNS, HTTP, service, SaaS and endpoint compositions -- are built, the columns they declare are read
off them, and each must have one of three dispositions: a `search =` body in `savedsearches.conf` names it; a model
under `examples/layer5_model/models/` lists it among its features, since a column the autoencoder is trained on is
read by the score rather than by a rule; or the "Features without a rule" table, which the design guide's Part 6 and
the app README both carry, lists it with a reason. A column that is listed and has since found a reader, or become a
model input, fails too, so the table cannot outlive its reasons.

A search reads a column when the column's name is in a `search =` body. A mention in a comment or a description is
an explanation of why a search does not read it, which is the opposite, and the comment on R-D-L7-005 is the case
in point: it names `http_4xx_to_2xx_ratio` to say why the search thresholds the counts instead.

Context, join and identity columns are outside it: `TC0EnrichStage` writes what the inventory holds, and the rules
that read it name it; `CommunityIdStage` writes a join key, which the connection evidence search reads; and the
host resolvers and `ChainAnchorStage` write which machine a row came from and how it was found, which are lineage
rather than behaviour and are asserted by the host identity harness.
"""

import glob
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import application_pipeline  # noqa: E402
import network_pipeline  # noqa: E402
import presentation_pipeline  # noqa: E402
import session_pipeline  # noqa: E402
import telemetry_pipeline  # noqa: E402
import transport_pipeline  # noqa: E402

from morpheus.stages.lineage.binding_resolver_stage import BindingResolverStage  # noqa: E402
from morpheus.stages.lineage.chain_anchor_stage import ChainAnchorStage  # noqa: E402
from morpheus.stages.lineage.community_id_stage import CommunityIdStage  # noqa: E402
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage  # noqa: E402
from morpheus.stages.telemetry.tc1_binding_stage import TC1BindingStage  # noqa: E402
from morpheus.stages.telemetry.tc2_arp_stage import TC2ArpStage  # noqa: E402
from morpheus.stages.telemetry.tc2_auth_stage import TC2AuthStage  # noqa: E402
from morpheus.stages.telemetry.tc2_binding_stage import TC2BindingStage  # noqa: E402
from morpheus.stages.telemetry.tc5_cadence_stage import TC5CadenceStage  # noqa: E402
from morpheus.stages.telemetry.tc5_drift_stage import TC5DriftStage  # noqa: E402
from morpheus.stages.telemetry.tc5_novelty_stage import TC5NoveltyStage  # noqa: E402
from morpheus.stages.telemetry.tc5_risk_stage import TC5RiskStage  # noqa: E402
from morpheus.stages.telemetry.tc5_score_stage import TC5ScoreStage  # noqa: E402
from morpheus.stages.telemetry.tc5_session_stage import TC5SessionStage  # noqa: E402
from morpheus.stages.telemetry.tc5_travel_stage import TC5TravelStage  # noqa: E402
from morpheus.stages.telemetry.tc7_endpoint_stage import TC7EndpointStage  # noqa: E402
from morpheus.stages.telemetry.tc7_saas_stage import TC7SaasStage  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SAVED_SEARCHES = os.path.join(REPO_ROOT,
                              "examples",
                              "splunk_lineage_app",
                              "TA-morpheus-lineage",
                              "default",
                              "savedsearches.conf")
MODELS = os.path.join(REPO_ROOT, "examples", "layer5_model", "models", "*.json")
DOCUMENTS = {
    "guide":
        os.path.join(REPO_ROOT,
                     "docs",
                     "source",
                     "developer_guide",
                     "guides",
                     "11_predictive_behavioral_analytics_osi.md"),
    "app README":
        os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "README.md"),
}
HEADING = re.compile(r"^#+ Features without a rule\s*$", re.M)
SEARCH_BODY = re.compile(r"^search\s*=((?:[^\n]*\\\n)*[^\n]*)$", re.M)
"""A `search =` line and the continuation lines its trailing backslashes join to it."""

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

CONTEXT_STAGES = (TC0EnrichStage, CommunityIdStage, BindingResolverStage, ChainAnchorStage)
"""The stages whose columns are context, a join key or identity rather than behaviour. See the module docstring."""


def _compositions() -> dict:
    """
    The feature stages of every composed corpus, by the corpus's name.

    The compositions that expose a stage list are taken from it. The telemetry and session compositions build their
    stage lists inside their runners, per class, so the stages are built here as the runners build them; the
    parameters that differ decide values, not the columns a stage declares.
    """
    config = network_pipeline.build_pipeline_config()
    (scorer, manifest) = session_pipeline.committed_scoring()

    return {
        "telemetry":
            telemetry_pipeline.layer_1_stages(config) + telemetry_pipeline.device_stages(config) +
            [TC1BindingStage(config)] + telemetry_pipeline.mac_table_stages(config) +
            [TC2BindingStage(config), TC2ArpStage(config), TC2AuthStage(config)],
        "network":
            network_pipeline.build_stages(config),
        "transport":
            transport_pipeline.build_stages(config),
        "session": [
            TC5NoveltyStage(config),
            TC5CadenceStage(config),
            TC5TravelStage(config),
            TC5RiskStage(config),
            TC5ScoreStage(config, scorer=scorer, manifest=manifest, feature_columns=session_pipeline.SCORED_FEATURES),
            TC5DriftStage(config),
            TC5SessionStage(config),
        ],
        "presentation":
            presentation_pipeline.build_stages(config),
        "application":
            application_pipeline.build_stages(config, application_pipeline.DNS_CLASS) +
            application_pipeline.build_stages(config, application_pipeline.HTTP_CLASS) +
            application_pipeline.build_stages(config, application_pipeline.SERVICE_CLASS),
        "saas": [TC7SaasStage(config)],
        "endpoint": [TC7EndpointStage(config)],
    }


def _feature_stages() -> list:
    return [stage for stages in _compositions().values() for stage in stages if not isinstance(stage, CONTEXT_STAGES)]


def _emitted() -> dict:
    """Every column a feature stage declares, mapped to the names of the stages that declare it."""
    emitted: dict = {}

    for stage in _feature_stages():
        for column in stage.get_needed_columns():
            emitted[column] = emitted.get(column, frozenset()) | {type(stage).__name__}

    return emitted


def _searched() -> set:
    with open(SAVED_SEARCHES, encoding="utf-8") as handle:
        bodies = SEARCH_BODY.findall(handle.read())

    return set(IDENTIFIER.findall("\n".join(bodies)))


def _model_inputs() -> dict:
    """Every feature a committed model is trained on, mapped to the model files that list it."""
    inputs: dict = {}

    for path in sorted(glob.glob(MODELS)):
        with open(path, encoding="utf-8") as handle:
            models = json.load(handle)["models"]

        for entry in models.values():
            for feature in entry["model"]["features"]:
                inputs[feature] = inputs.get(feature, frozenset()) | {os.path.basename(path)}

    return inputs


def _listed(path: str) -> dict:
    """The table under the "Features without a rule" heading: each listed column, mapped to the stages it names."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    match = HEADING.search(text)

    assert match is not None, f"{path} has no 'Features without a rule' section"

    listed = {}

    for line in text[match.end():].splitlines():
        if (line.startswith("#")):
            break

        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]

        if (len(cells) != 3 or cells[0] in ("Feature", "---")):
            continue

        stages = frozenset(re.findall(r"`([^`]+)`", cells[1]))

        for column in re.findall(r"`([^`]+)`", cells[0]):
            listed[column] = stages

    return listed


def test_the_stages_emit_what_this_test_expects_to_read():
    emitted = _emitted()

    # One column from each composition, so a regression in a stage list cannot pass quietly.
    for column in ("bits_per_second",
                   "ports_per_mac",
                   "dsts_per_src_step",
                   "byte_asymmetry_step",
                   "weekday_surprise_bits",
                   "ja4_client_changed",
                   "dns_client_key",
                   "http_4xx_to_2xx_ratio",
                   "service_key",
                   "saas_week_id",
                   "endpoint_host"):
        assert column in emitted, column


def test_a_search_body_is_what_counts_as_a_read():
    searched = _searched()

    # Read in a body, so the matcher finds it.
    assert "locincrement_z_loss" in searched
    # Named in R-D-L7-005's comment to say why the search thresholds the counts instead; not a read.
    assert "http_4xx_to_2xx_ratio" not in searched


def test_the_scored_stage_is_fed_what_the_committed_models_were_trained_on():
    inputs = _model_inputs()
    emitted = _emitted()

    assert set(session_pipeline.SCORED_FEATURES) <= set(inputs)

    # The session models' inputs are columns the session composition emits, so the model cannot be fed a column
    # that has been renamed under it without this saying so.
    session_inputs = sorted(feature for (feature, files) in inputs.items() if "session_models.json" in files)
    assert session_inputs, "no committed session model lists its features"
    assert all(feature in emitted for feature in session_inputs), session_inputs


@pytest.mark.parametrize("document", sorted(DOCUMENTS))
def test_every_feature_is_read_listed_or_a_model_input(document: str):
    searched = _searched()
    inputs = _model_inputs()
    unread = {
        column: stages
        for (column, stages) in _emitted().items() if column not in searched and column not in inputs
    }
    listed = _listed(DOCUMENTS[document])

    missing = sorted(set(unread) - set(listed))
    assert not missing, f"read by no search, input to no model and not listed in the {document}'s table: {missing}"

    misattributed = sorted(column for column in unread if listed[column] != unread[column])
    assert not misattributed, f"the {document}'s table names the wrong stage for {misattributed}"


@pytest.mark.parametrize("document", sorted(DOCUMENTS))
def test_nothing_listed_has_found_a_reader_or_stopped_existing(document: str):
    emitted = _emitted()
    searched = _searched()
    inputs = _model_inputs()
    listed = _listed(DOCUMENTS[document])

    stale = sorted(column for column in listed if column in searched or column in inputs or column not in emitted)
    assert not stale, (f"the {document}'s table lists columns a search now reads, a model now takes or no stage "
                       f"emits: {stale}")


def test_the_two_tables_agree():
    (guide, readme) = (_listed(DOCUMENTS["guide"]), _listed(DOCUMENTS["app README"]))

    assert guide == readme
