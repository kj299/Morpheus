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
Every host feature is read by a shipped search, or is listed with the reason it is not.

The retrospective found a dozen host features emitted and read by nothing, and no way to know it short of reading
every stage against every search. This makes the question a test. The host-centric stages -- the ones the layer 3,
4, 6 and endpoint layer 7 corpora compose -- are built, the columns they declare are read off them, and each must
either appear in `savedsearches.conf` or be listed in the "Features without a rule" table, which the design guide's
Part 6 and the app README both carry. A column that is listed and has since found a reader fails too, so the table
cannot outlive its reasons.

Context and join columns are outside it: `TC0EnrichStage` writes what the inventory holds, and the rules that read it
name it; `CommunityIdStage` writes a join key, which the connection evidence search reads.
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import network_pipeline  # noqa: E402
import presentation_pipeline  # noqa: E402
import transport_pipeline  # noqa: E402

from morpheus.stages.lineage.community_id_stage import CommunityIdStage  # noqa: E402
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage  # noqa: E402
from morpheus.stages.telemetry.tc7_endpoint_stage import TC7EndpointStage  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SAVED_SEARCHES = os.path.join(REPO_ROOT,
                              "examples",
                              "splunk_lineage_app",
                              "TA-morpheus-lineage",
                              "default",
                              "savedsearches.conf")
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


def _host_stages() -> list:
    config = network_pipeline.build_pipeline_config()
    stages = (network_pipeline.build_stages(config) + transport_pipeline.build_stages(config) +
              presentation_pipeline.build_stages(config) + [TC7EndpointStage(config)])

    return [stage for stage in stages if not isinstance(stage, (TC0EnrichStage, CommunityIdStage))]


def _emitted() -> dict:
    """Every column a host stage declares, mapped to the stage that declares it."""
    return {column: type(stage).__name__ for stage in _host_stages() for column in stage.get_needed_columns()}


def _searched() -> set:
    with open(SAVED_SEARCHES, encoding="utf-8") as handle:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", handle.read()))


def _listed(path: str) -> dict:
    """The table under the "Features without a rule" heading: each listed column, mapped to the stage it names."""
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

        stage = cells[1].strip("`")

        for column in re.findall(r"`([^`]+)`", cells[0]):
            listed[column] = stage

    return listed


def test_the_host_stages_emit_what_this_test_expects_to_read():
    emitted = _emitted()

    # The stages this step built or rewired are among them, so a regression in the stage list cannot pass quietly.
    for column in ("dsts_per_src_step",
                   "srcs_per_dst_step",
                   "byte_asymmetry_step",
                   "ja4_client_changed",
                   "endpoint_host",
                   "dst_ports_per_src"):
        assert column in emitted, column


@pytest.mark.parametrize("document", sorted(DOCUMENTS))
def test_every_host_feature_is_read_or_listed(document: str):
    unread = {column: stage for (column, stage) in _emitted().items() if column not in _searched()}
    listed = _listed(DOCUMENTS[document])

    missing = sorted(set(unread) - set(listed))
    assert not missing, f"read by no search and not listed in the {document}'s table: {missing}"

    misattributed = sorted(column for column in unread if listed[column] != unread[column])
    assert not misattributed, f"the {document}'s table names the wrong stage for {misattributed}"


@pytest.mark.parametrize("document", sorted(DOCUMENTS))
def test_nothing_listed_has_found_a_reader_or_stopped_existing(document: str):
    emitted = _emitted()
    searched = _searched()
    listed = _listed(DOCUMENTS[document])

    stale = sorted(column for column in listed if column in searched or column not in emitted)
    assert not stale, f"the {document}'s table lists columns a search now reads or no stage emits: {stale}"


def test_the_two_tables_agree():
    (guide, readme) = (_listed(DOCUMENTS["guide"]), _listed(DOCUMENTS["app README"]))

    assert guide == readme
