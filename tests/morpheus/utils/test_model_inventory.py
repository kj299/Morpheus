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
The inventory refuses a model file that is not the one it records, and says so when the file was never fetched.
"""

import json
import os
import shutil

import pytest

from morpheus.utils.model_inventory import REPOSITORY_INVENTORY
from morpheus.utils.model_inventory import ModelFileUnavailable
from morpheus.utils.model_inventory import file_sha256
from morpheus.utils.model_inventory import load_entry

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INVENTORY = os.path.join(REPO_ROOT, REPOSITORY_INVENTORY)
LFS_POINTER = ("version https://git-lfs.github.com/spec/v1\n"
               "oid sha256:d12f31d6dcebd03dab1824c850ebe928b5f5b76d3e9029e9b160f593c12bea9e\n"
               "size 44793456\n")


def copy_abp(tmp_path) -> str:
    """The inventory with only its ABP entry, copied where a test may change it."""
    with open(INVENTORY, encoding="utf-8") as handle:
        inventory = json.load(handle)

    inventory["models"] = {"abp-pcap-xgb": inventory["models"]["abp-pcap-xgb"]}
    shutil.copytree(os.path.join(os.path.dirname(INVENTORY), "abp-pcap-xgb"), tmp_path / "abp-pcap-xgb")
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")

    return str(path)


def test_the_abp_entry_loads_with_its_files_verified():
    entry = load_entry(INVENTORY, "abp-pcap-xgb")

    assert entry.version == "abp-pcap-xgb:c3e05a33fdc02a04"
    assert entry.model_path.endswith(os.path.join("abp-pcap-xgb", "abp-pcap-xgb.json"))
    assert len(entry.settings["features"]) == 13


def test_every_entry_names_its_rule_and_where_it_came_from():
    with open(INVENTORY, encoding="utf-8") as handle:
        inventory = json.load(handle)

    assert sorted(entry["rule"] for entry in inventory["models"].values()) == ["R-B-L4-001", "R-B-L7-003"]

    for (name, entry) in inventory["models"].items():
        assert entry["provenance"]["upstream_commit"].startswith("c6a3b56d"), name
        assert entry["version"] == f"{name}:{entry['files']['model']['sha256'][:16]}"


def test_a_changed_model_file_is_refused(tmp_path):
    path = copy_abp(tmp_path)
    model = tmp_path / "abp-pcap-xgb" / "abp-pcap-xgb.json"
    model.write_bytes(model.read_bytes().replace(b'"num_trees":"20"', b'"num_trees":"19"'))

    with pytest.raises(ValueError, match="changed without its entry"):
        load_entry(path, "abp-pcap-xgb")


def test_a_recorded_version_that_is_not_the_file_s_digest_is_refused(tmp_path):
    path = copy_abp(tmp_path)
    inventory = json.loads((tmp_path / "inventory.json").read_text(encoding="utf-8"))
    inventory["models"]["abp-pcap-xgb"]["version"] = "abp-pcap-xgb:latest"
    (tmp_path / "inventory.json").write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ValueError, match="abp-pcap-xgb:latest"):
        load_entry(path, "abp-pcap-xgb")


def test_an_lfs_pointer_is_named_as_one_with_the_command_that_fetches_it(tmp_path):
    pointer = tmp_path / "model.onnx"
    pointer.write_text(LFS_POINTER, encoding="utf-8")

    # Without --exclude="" the repository's .lfsconfig, which excludes every path, makes the pull a silent no-op.
    with pytest.raises(ModelFileUnavailable, match='git lfs pull --exclude="" --include="models/inventory/\\*\\*"'):
        file_sha256(str(pointer))


def test_a_missing_file_and_a_missing_entry_are_refused(tmp_path):
    with pytest.raises(ModelFileUnavailable):
        file_sha256(str(tmp_path / "absent"))

    with pytest.raises(KeyError, match="no model 'triton'"):
        load_entry(INVENTORY, "triton")
