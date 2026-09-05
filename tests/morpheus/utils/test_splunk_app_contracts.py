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
The numeric contracts shared between the pipeline, the Splunk app, and the design guide.

Four values have to agree across artifacts that no single test suite covers: the bucket width, the binding
retention, the Community ID seed, and the `event_time` rendering. Each of them fails silently when it drifts --
a lookup that matches nothing, a lookup that outlives or predeceases its index, flow hashes that no other tool
in the estate agrees with. Prose in three documents is not enforcement, so these tests read the shipped
configuration directly and compare it against the values in code.

The fourth contract, the `event_time` rendering, lives in `test_siem_wire.py` beside the renderer it constrains.
"""

import configparser
import inspect
import os
import re

import pytest

from morpheus.stages.lineage.community_id_stage import CommunityIdStage
from morpheus.utils.binding_table import DEFAULT_BUCKET_SECONDS
from morpheus.utils.community_id import community_id
from morpheus.utils.community_id import community_id_series

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
APP_ROOT = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "TA-morpheus-lineage")
SAVEDSEARCHES_PATH = os.path.join(APP_ROOT, "default", "savedsearches.conf")
INDEXES_PATH = os.path.join(APP_ROOT, "default", "indexes.conf")
GUIDE_PATH = os.path.join(REPO_ROOT,
                          "docs",
                          "source",
                          "developer_guide",
                          "guides",
                          "11_predictive_behavioral_analytics_osi.md")

# `| where bucket >= floor((now() - 34560000) / 300)` -- the expiry job, carrying both contracts at once.
EXPIRY_PATTERN = re.compile(r"floor\(\(now\(\)\s*-\s*(\d+)\)\s*/\s*(\d+)\)")

# `| eval bucket=floor(_time/300)` -- the query-side discretization.
DISCRETIZE_PATTERN = re.compile(r"floor\(_time\s*/\s*(\d+)\)")

BINDING_INDEXES = ["behavior_bindings"]


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def expiry_pairs(path: str) -> list[tuple[int, int]]:
    """Every `(retention_seconds, bucket_seconds)` pair appearing in an expiry expression."""
    return [(int(retention), int(bucket)) for (retention, bucket) in EXPIRY_PATTERN.findall(read_text(path))]


def test_the_patterns_still_find_something():
    # Guards every other test here: a stanza rewrite that breaks these patterns must fail loudly rather than
    # silently reduce the assertions below to vacuous truths over empty lists.
    assert len(expiry_pairs(SAVEDSEARCHES_PATH)) > 0
    assert len(expiry_pairs(GUIDE_PATH)) > 0
    assert len(DISCRETIZE_PATTERN.findall(read_text(GUIDE_PATH))) > 0


@pytest.mark.parametrize("path", [SAVEDSEARCHES_PATH, GUIDE_PATH], ids=["app", "guide"])
def test_bucket_width_matches_the_pipeline(path: str):
    # Contract 1. The pipeline expands bindings across buckets of DEFAULT_BUCKET_SECONDS; the SIEM rediscretizes
    # event times with the same divisor. Disagreement means every lookup misses, with no error.
    divisors = {bucket for (_, bucket) in expiry_pairs(path)}
    divisors.update(int(value) for value in DISCRETIZE_PATTERN.findall(read_text(path)))

    assert divisors == {DEFAULT_BUCKET_SECONDS}, f"{path} discretizes on {divisors}"


def test_binding_retention_matches_between_index_and_expiry_job():
    # Contract 2. outputlookup never removes rows, so a separate job expires them. If the job's cutoff outlives
    # the index the lookup keeps rows whose events are gone; if it predeceases the index, events become
    # unattributable while still queryable. The README warns that these must move together; this enforces it.
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(INDEXES_PATH)

    for index_name in BINDING_INDEXES:
        index_retention = int(parser[index_name]["frozenTimePeriodInSecs"])
        job_cutoffs = {retention for (retention, _) in expiry_pairs(SAVEDSEARCHES_PATH)}

        assert job_cutoffs == {index_retention}, (f"{index_name} retains {index_retention}s but the expiry job "
                                                  f"cuts at {job_cutoffs}")


def test_the_guide_and_the_app_agree_on_both_numbers():
    # The guide quotes the app's stanzas; when the two disagree the app is normative, but disagreement is still a
    # documentation defect and it has happened before on this branch.
    assert set(expiry_pairs(GUIDE_PATH)) == set(expiry_pairs(SAVEDSEARCHES_PATH))


def test_community_id_seed_defaults_to_zero():
    # Contract 3. The seed is part of the hash input, so a non-default value produces flow identifiers that no
    # other tool in the estate agrees with, forfeiting the entire reason the field exists.
    for callable_under_test in (community_id, community_id_series, CommunityIdStage.__init__):
        default = inspect.signature(callable_under_test).parameters["seed"].default

        assert default == 0, f"{callable_under_test.__qualname__} defaults its seed to {default}"


def test_kvstore_lookups_expose_the_key_they_are_written_by():
    # A KV Store lookup can only address a specific record when `_key` is in its fields_list. Without it the refresh
    # jobs' `key_field=_key` matches nothing, every overlapping window appends duplicates instead of overwriting,
    # and determinism control 11's idempotent sink does not hold at the SIEM boundary.
    transforms = configparser.ConfigParser(interpolation=None)
    transforms.read(os.path.join(APP_ROOT, "default", "transforms.conf"))

    for lookup in ("binding_l2_l3", "binding_l1"):
        fields = [field.strip() for field in transforms[lookup]["fields_list"].split(",")]

        assert "_key" in fields, f"{lookup} cannot be written by key without _key in fields_list"

    # Every job that writes a KV Store lookup writes it by key, the refresh jobs and the expiry job alike. The conf
    # is read as text because a saved search's `search` spans continuation lines.
    with open(SAVEDSEARCHES_PATH, encoding="utf-8") as handle:
        searches = handle.read()

    writes = re.findall(r"outputlookup\s+binding_\S+(.*)", searches)

    assert len(writes) == 3, writes

    for tail in writes:
        assert "key_field=_key" in tail, f"a lookup is written without a key: {tail!r}"
