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
Each of the fork's CI environments has one definition (guide 13, plan items 10, 11 and 14).

`dependencies.yaml` and the CPU lock each named the test environment's packages and channels, and disagreed. A file
key of `dependencies.yaml` is now each environment's definition -- `cpu-test` for the test tiers, `cpu-docs` for the
documentation job, `cpu-build` for the C++ job -- and `ci/scripts/cpu_env/make_lock.py` generates the request file
from it and solves the request into the lock. These hold the three to each other, for every environment, without
solving anything: the request is what the definition says, the lock records the request it was solved from, the lock
holds every package the request names at the version it pins, from the channels it lists, the packages only pip
carries are pinned beside it, and CI installs that lock.
"""

import importlib.util
import json
import os
import re

import pytest
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
MODULE_PATH = os.path.join(REPO_ROOT, "ci", "scripts", "cpu_env", "make_lock.py")
WORKFLOW = os.path.join(REPO_ROOT, ".github", "workflows", "fork-cpu.yaml")
PACKAGE_URL = re.compile(r"^https://conda\.anaconda\.org/(?P<channel>.+)/(?:linux-64|noarch)/"
                         r"(?P<name>.+)-(?P<version>[^-]+)-(?P<build>[^-]+)\.(?:conda|tar\.bz2)#(?P<md5>[0-9a-f]{32})$")


def _load():
    spec = importlib.util.spec_from_file_location("make_lock", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


make_lock = _load()
KEYS = sorted(make_lock.ENVIRONMENTS)


def _yaml(*path: str) -> dict:
    with open(os.path.join(REPO_ROOT, *path), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _lock_lines(key: str) -> list[str]:
    with open(os.path.join(REPO_ROOT, make_lock.ENVIRONMENTS[key]), encoding="utf-8") as handle:
        return handle.read().splitlines()


def _locked(key: str) -> dict:
    """Every package in a lock: name to its version and channel."""
    found = {}

    for line in _lock_lines(key):
        if (line.startswith("https://")):
            match = PACKAGE_URL.match(line)

            assert match is not None, f"not a package URL with its checksum: {line}"
            found[match["name"]] = (match["version"], match["channel"])

    return found


def _requested(key: str) -> tuple[list[str], list[str]]:
    """A definition's packages: every list its file key includes, as conda packages and as pip's."""
    definition = _yaml("dependencies.yaml")
    (conda, pip) = ([], [])

    # A file key's matrix has one value per axis here, so a `specific` entry applies when its matrix asks for
    # nothing else: the build key's CUDA version selects the toolkit's packages.
    matrix = {axis: str(values[0]) for (axis, values) in definition["files"][key].get("matrix", {}).items()}

    for name in definition["files"][key]["includes"]:
        section = definition["dependencies"][name]
        entries = list(section.get("common", []))

        for specific in section.get("specific", []):
            assert specific["output_types"] == ["conda"], name
            entries += [{
                "output_types": specific["output_types"], "packages": chosen["packages"]
            } for chosen in specific["matrices"] if (chosen.get("matrix") or {}).items() <= matrix.items()]

        for entry in entries:
            assert entry["output_types"] == ["conda"], name

            for package in entry["packages"]:
                if (isinstance(package, dict)):
                    pip.extend(package["pip"])
                else:
                    conda.append(package)

    return (conda, pip)


@pytest.mark.parametrize("key", KEYS)
def test_the_request_is_what_the_definition_says(key: str):
    definition = _yaml("dependencies.yaml")
    request = _yaml(make_lock.request_path(key))
    (conda, pip) = _requested(key)

    assert definition["files"][key]["output"] == "conda"
    assert [package for package in request["dependencies"] if isinstance(package, str)] == sorted(set(conda))
    assert [package["pip"] for package in request["dependencies"]
            if isinstance(package, dict)] == ([sorted(pip)] if pip else [])
    assert request["channels"] == definition["channels"]


@pytest.mark.parametrize("key", KEYS)
def test_the_lock_records_the_request_it_was_solved_from(key: str):
    stated = [line[len(make_lock.HASH_PREFIX):] for line in _lock_lines(key) if line.startswith(make_lock.HASH_PREFIX)]

    assert stated == [make_lock.request_digest(REPO_ROOT, key)
                      ], (f"the request changed without a new lock; run ci/scripts/cpu_env/make_lock.py {key}")


@pytest.mark.parametrize("key", KEYS)
def test_the_lock_holds_every_requested_package_at_its_pin(key: str):
    locked = _locked(key)

    assert len(locked) > 200

    for spec in _requested(key)[0]:
        (name, pin) = re.match(r"^([A-Za-z0-9_.-]+?)(?:[<>=!].*)?$", spec).group(1), spec.partition("=")[2]

        assert name in locked, f"{spec} is requested and not in the lock"

        if (">" not in spec and "<" not in spec and pin):
            assert locked[name][0] == pin or locked[name][0].startswith(pin + "."), (spec, locked[name])


@pytest.mark.parametrize("key", KEYS)
def test_every_locked_package_comes_from_a_channel_the_definition_lists(key: str):
    channels = set(_yaml("dependencies.yaml")["channels"])

    assert {channel for (_, channel) in _locked(key).values()} <= channels


@pytest.mark.parametrize("key", KEYS)
def test_what_only_pip_carries_is_pinned_beside_the_lock(key: str):
    requested = _requested(key)[1]
    path = os.path.join(REPO_ROOT, make_lock.ENVIRONMENTS[key] + make_lock.PIP_SUFFIX)

    if (not requested):
        assert not os.path.exists(path)
        return

    with open(path, encoding="utf-8") as handle:
        pinned = dict(line.split("==") for line in handle.read().split())

    normal = {name.lower().replace("_", "-") for name in pinned}

    assert {re.split(r"[<>=!]", spec)[0] for spec in requested} <= normal
    assert all(re.fullmatch(r"[0-9][0-9A-Za-z.]*", version) for version in pinned.values()), pinned


@pytest.mark.parametrize("key", KEYS)
def test_the_lock_says_it_is_generated_and_names_no_packages_of_its_own(key: str):
    comments = [line for line in _lock_lines(key) if line.startswith("#")]

    assert any("GENERATED by ci/scripts/cpu_env/make_lock.py" in line for line in comments)
    # The old header carried its own `micromamba create ... python=3.12 morpheus-core=25.06 ...`: a second definition.
    assert not any("micromamba create" in line or "morpheus-core=" in line for line in comments)


def test_ci_installs_every_lock_and_nothing_else():
    with open(WORKFLOW, encoding="utf-8") as handle:
        workflow = yaml.safe_load(handle)

    installed = sorted(step["with"]["environment-file"] for job in workflow["jobs"].values() for step in job["steps"]
                       if "environment-file" in step.get("with", {}))

    assert installed == sorted(path.replace(os.sep, "/") for path in make_lock.ENVIRONMENTS.values())


def test_the_pip_pins_are_everything_pip_installed():
    inspected = json.dumps({
        "installed": [
            {
                "metadata": {
                    "name": "exhale", "version": "0.3.7"
                }, "installer": "pip"
            },
            {
                "metadata": {
                    "name": "nvidia_sphinx_theme", "version": "0.0.9"
                }, "installer": "pip"
            },
            # What a requested package brought with it: CI installs with --no-deps, so it has to be listed.
            {
                "metadata": {
                    "name": "pydata_sphinx_theme", "version": "0.24.0"
                }, "installer": "pip"
            },
            {
                "metadata": {
                    "name": "numpy", "version": "1.26.4"
                }, "installer": "conda"
            }
        ]
    })

    assert make_lock.pip_requests("dependencies:\n- sphinx>=8.2\n- pip:\n  - exhale>=0.3.7\n"
                                  "  - nvidia-sphinx-theme>=0.0.7\nname: x\n") == ["exhale", "nvidia-sphinx-theme"]
    assert make_lock.render_pip(inspected,
                                ["exhale", "nvidia-sphinx-theme"
                                 ]) == ("exhale==0.3.7\nnvidia_sphinx_theme==0.0.9\npydata_sphinx_theme==0.24.0\n")

    with pytest.raises(ValueError, match="exhale"):
        make_lock.render_pip(json.dumps({"installed": []}), ["exhale"])


def test_render_keeps_the_explicit_list_and_drops_the_exporter_s_preamble():
    explicit = "# This file may be used to create an environment\n# platform: linux-64\n@EXPLICIT\nhttps://a/b#c\n"
    text = make_lock.render(explicit, "0" * 64)

    assert text.endswith("# platform: linux-64\n@EXPLICIT\nhttps://a/b#c\n")
    assert f"{make_lock.HASH_PREFIX}{'0' * 64}\n" in text
    assert "This file may be used" not in text
