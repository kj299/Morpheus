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
The reference datasets pinned in the determinism envelope (guide 13, plan item 3; plan section 4.2).
"""

import csv
import glob
import hashlib
import os
import re
import shutil
import sys

import pytest

import morpheus
from morpheus.utils import query_entropy
from morpheus.utils.determinism_envelope import DIGEST_CHARACTERS
from morpheus.utils.determinism_envelope import REFERENCE_DATA_COLUMN
from morpheus.utils.reference_data import PUBLIC_SUFFIX_LIST
from morpheus.utils.reference_data import REFERENCE_DATASETS
from morpheus.utils.reference_data import clear_cache
from morpheus.utils.reference_data import dataset_path
from morpheus.utils.reference_data import read_dataset
from morpheus.utils.reference_data import reference_digests
from morpheus.utils.reference_data import render_reference_data

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PACKAGE = os.path.join(REPO_ROOT, "python", "morpheus", "morpheus")
DETERMINISM = os.path.join(REPO_ROOT, "tests", "morpheus", "determinism")

TUNNEL_NAME = "aGVsbG8gd29ybGQ.t.exfil-relay.net"
"""A name whose registered domain the shipped list puts at `exfil-relay.net`."""

ZONE_DATABASE = re.compile(
    r"^\s*(?:import|from)\s+(zoneinfo|pytz|tzdata|dateutil\.tz|backports\.zoneinfo)\b"
    r"|tz_(?:convert|localize)\(\s*[\"'](?!UTC[\"'])[^\"']+[\"']"
    r"|ZoneInfo\(",
    re.MULTILINE)
"""What resolving a named zone looks like: importing a zone database, or converting to a zone other than UTC."""


def _change_list(directory, monkeypatch):
    """Make the shipped suffix list plus one private rule this process's current list."""
    for name in REFERENCE_DATASETS.values():
        shutil.copy(os.path.join(morpheus.DATA_DIR, name), directory / name)

    with open(directory / REFERENCE_DATASETS[PUBLIC_SUFFIX_LIST], "a", encoding="utf-8") as handle:
        handle.write("\n// A hosting platform that started selling customer subdomains.\nexfil-relay.net\n")

    monkeypatch.setattr(morpheus, "DATA_DIR", str(directory))
    monkeypatch.setattr(query_entropy, "_RULES", None)
    clear_cache()


@pytest.fixture(name="changed_list")
def changed_list_fixture(tmp_path, monkeypatch):
    _change_list(tmp_path, monkeypatch)

    yield tmp_path

    clear_cache()


def _envelope_for():
    sys.path.insert(0, DETERMINISM)

    try:
        import stamping  # pylint: disable=import-outside-toplevel,import-error
    finally:
        sys.path.remove(DETERMINISM)

    return stamping.envelope_for("TC-7", {"seal_period_ns": 1}, rules=("R-B-L7-001", ))


def test_the_digest_is_of_the_bytes_the_package_ships():
    for (name, file_name) in REFERENCE_DATASETS.items():
        with open(os.path.join(morpheus.DATA_DIR, file_name), "rb") as handle:
            expected = hashlib.sha256(handle.read()).hexdigest()[:DIGEST_CHARACTERS]

        assert reference_digests()[name] == expected


def test_the_rendering_names_every_dataset_in_order():
    assert render_reference_data({"b": "2", "a": "1"}) == "a:1;b:2"
    assert render_reference_data(reference_digests()).startswith(f"{PUBLIC_SUFFIX_LIST}:")

    with pytest.raises(ValueError):
        render_reference_data({})


def test_an_unregistered_dataset_is_refused():
    # A stage reading a file this module does not know would be reading something the envelope cannot name.
    with pytest.raises(KeyError, match="not a registered reference dataset"):
        dataset_path("tzdata")


def test_a_changed_list_changes_the_envelope(tmp_path, monkeypatch):
    # The plan item's own test: the same pipeline over a different suffix list stamps a different envelope.
    clear_cache()
    shipped = _envelope_for()

    _change_list(tmp_path, monkeypatch)

    try:
        changed = _envelope_for()
    finally:
        clear_cache()

    assert changed.to_columns()[REFERENCE_DATA_COLUMN] != shipped.to_columns()[REFERENCE_DATA_COLUMN]
    assert changed.fingerprint != shipped.fingerprint
    # Nothing else moved: the configuration is the same configuration.
    assert changed.configuration == shipped.configuration


def test_the_change_is_one_that_moves_a_feature(changed_list):
    # Why the digest belongs in the envelope: one added rule takes the registered domain a tunnel's subdomains are
    # counted under from the platform to its customer, and with it every count R-B-L7-001 reads.
    del changed_list

    assert query_entropy.registered_domain(TUNNEL_NAME) == "t.exfil-relay.net"


def test_the_shipped_list_splits_the_same_name_differently():
    assert query_entropy.registered_domain(TUNNEL_NAME) == "exfil-relay.net"


def test_a_list_replaced_under_a_running_process_changes_nothing_until_it_restarts(changed_list):
    # The rules a stage parsed and the digest the envelope carries are read from the same bytes, once. A copy
    # replaced on disk afterwards must change neither, or a row would name a list its features were not split by.
    path = changed_list / REFERENCE_DATASETS[PUBLIC_SUFFIX_LIST]
    (text, digest) = read_dataset(PUBLIC_SUFFIX_LIST)
    domain = query_entropy.registered_domain(TUNNEL_NAME)

    with open(path, "a", encoding="utf-8") as handle:
        handle.write("t.exfil-relay.net\n")

    assert read_dataset(PUBLIC_SUFFIX_LIST) == (text, digest)
    assert query_entropy.registered_domain(TUNNEL_NAME) == domain

    clear_cache()

    assert read_dataset(PUBLIC_SUFFIX_LIST)[1] != digest


def test_every_golden_row_names_the_list_the_package_ships():
    # A golden regenerated before a list update, or a composition that stamped no reference data, fails here. A
    # quarantined row was refused before scoring and carries no envelope at all, which its blank fingerprint says.
    expected = render_reference_data(reference_digests())
    goldens = sorted(glob.glob(os.path.join(DETERMINISM, "golden_*_expected.csv")))

    assert goldens

    for path in goldens:
        with open(path, encoding="utf-8", newline="") as handle:
            values = {row[REFERENCE_DATA_COLUMN] for row in csv.DictReader(handle) if row["pipeline_fingerprint"]}

        assert values == {expected}, os.path.basename(path)


def test_no_module_resolves_a_named_time_zone():
    # The time zone database is not a reference dataset of these pipelines because no stage reads it; TC-5 takes a
    # fixed offset so a tzdata update cannot move its features. The first module to resolve a named zone fails
    # here, and must register the database so its digest reaches the envelope with it.
    found = []

    for (directory, _, files) in os.walk(PACKAGE):
        for name in files:
            if (name.endswith(".py")):
                with open(os.path.join(directory, name), encoding="utf-8") as handle:
                    if (ZONE_DATABASE.search(handle.read())):
                        found.append(os.path.relpath(os.path.join(directory, name), PACKAGE))

    assert not found


@pytest.mark.parametrize("line",
                         [
                             "import zoneinfo",
                             "from pytz import timezone",
                             "from dateutil.tz import gettz",
                             "stamp.tz_convert('America/New_York')",
                             "zone = ZoneInfo(name)",
                         ])
def test_the_time_zone_check_has_teeth(line: str):
    assert ZONE_DATABASE.search(line)


@pytest.mark.parametrize("line", ["stamp.tz_convert(\"UTC\")", "stamp.tz_localize(None)", "tz_convert(state['tz'])"])
def test_the_time_zone_check_passes_utc_and_restored_zones(line: str):
    assert not ZONE_DATABASE.search(line)
