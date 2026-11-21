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
The conda recipe, the Dockerfile and the container scripts agree with the tree they build.

An image build is the one path nothing in CI runs: it needs a Docker daemon, most of an hour and, for the tests,
a card. So it drifted. The recipe's run requirements said "this should be synced with `runtime` in
dependencies.yaml" and were not; the build script passed a build argument the Dockerfile did not declare; the
runtime image shipped the stages and not the models they score with; and the package's version came from a git tag
this fork does not have. These hold the parts that can be checked without building anything: the lists that must
match do, what the Dockerfile names exists, and the pins a build cannot do without are stated.

What they do not check is that the image builds. `ci/conda/recipes/run_conda_build.sh morpheus-core` and
`docker/build_container_release.sh` are the checks of that.
"""

import importlib.util
import os
import re
import subprocess

import pytest
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
RECIPE_DIR = os.path.join("ci", "conda", "recipes", "morpheus-core")
RECIPE = os.path.join(RECIPE_DIR, "meta.yaml")
SMOKE_TEST = os.path.join(RECIPE_DIR, "morpheus_core_test.sh")
CONDA_BUILD = os.path.join("ci", "conda", "recipes", "run_conda_build.sh")
DOCKERFILE = os.path.join("docker", "Dockerfile")
BUILD_SCRIPT = os.path.join("docker", "build_container.sh")
VERSION_SCRIPT = os.path.join("ci", "scripts", "project_version.sh")
"""The files an image build reads, relative to the repository's root."""

REQUIREMENT_NAME = re.compile(r"^(?:&\w+\s+)?([A-Za-z0-9_.-]+)")


def _text(*parts: str) -> str:
    """A file of the repository, by its path from the root."""
    with open(os.path.join(REPO_ROOT, *parts), encoding="utf-8") as handle:
        return handle.read()


def _recipe_section(name: str) -> list[str]:
    """
    One requirements list of the recipe, as its entries are written.

    Read by line rather than as YAML: the recipe is a Jinja template, and rendering it needs conda-build. An entry is
    a `- ` line under `<name>:`; an alias (`*mlflow`) is resolved to the entry that anchors it.
    """
    text = _text(RECIPE)
    anchors = dict(re.findall(r"^\s+- &(\w+) (.+?)\s*(?:#.*)?$", text, re.MULTILINE))
    match = re.search(rf"^      {name}:\n((?:        .*\n|\s*\n)+)", text, re.MULTILINE)

    assert match is not None, f"the recipe has no {name} requirements"

    entries = []

    for line in match.group(1).splitlines():
        line = line.strip()

        if (not line.startswith("- ")):
            continue

        entry = line[2:].split("#", 1)[0].strip()

        if (entry.startswith("*")):
            entry = anchors[entry[1:]]
        elif (entry.startswith("&")):
            entry = entry.split(" ", 1)[1]

        entries.append(entry)

    return entries


def _names(entries: list[str]) -> set:
    return {REQUIREMENT_NAME.match(entry).group(1) for entry in entries}


def _runtime_group() -> list[str]:
    """The conda packages of `runtime` in dependencies.yaml: what the runtime environment file installs."""
    with open(os.path.join(REPO_ROOT, "dependencies.yaml"), encoding="utf-8") as handle:
        runtime = yaml.safe_load(handle)["dependencies"]["runtime"]

    return [
        package for block in runtime["common"] if "conda" in block["output_types"] for package in block["packages"]
        if isinstance(package, str)
    ]


BROUGHT_BY_ANOTHER_GROUP = {"cuda-version", "python"}
"""Requirements the runtime environment takes from the groups it includes beside `runtime`: `cudatoolkit` and
`python`."""

INSTALLER_ONLY = {"pip"}
"""What the runtime group names to install the environment's pip packages with, which the package does not run on."""


def test_the_package_runs_on_what_the_runtime_environment_installs():
    # The recipe's own comment asked for this and nothing held it: the package still required the Triton client and
    # the RSS reader after their stages were removed, and did not require the two libraries the model stages score
    # with. A package added to one list and not the other fails here.
    recipe = _names(_recipe_section("run"))
    runtime = _names(_runtime_group())

    assert recipe - BROUGHT_BY_ANOTHER_GROUP == runtime - INSTALLER_ONLY, (
        f"only the recipe requires {sorted(recipe - BROUGHT_BY_ANOTHER_GROUP - runtime)}; "
        f"only the runtime group installs {sorted(runtime - INSTALLER_ONLY - recipe)}")


def test_the_two_pins_a_build_cannot_do_without_are_stated():
    # Both were found by a build that failed. The channel solves an open sysroot to one whose libc the compiler's
    # linker cannot read, and an open gflags to one libarrow does not link, so cudf does not import.
    assert "sysroot_linux-64 =2.28" in _recipe_section("build")

    for section in ("host", "run"):
        assert "gflags 2.2.*" in _recipe_section(section), section

    assert "gflags=2.2" in _runtime_group()


def test_the_recipe_names_no_package_type():
    # `type: conda_v2` was how an output asked for the `.conda` format. Current conda-build takes the format from its
    # configuration and fails on the key with KeyError 'conda_v2', after it has made the build environment.
    instructions = [line for line in _text(RECIPE).splitlines() if not line.lstrip().startswith("#")]

    assert not any(re.match(r"\s+type:", line) for line in instructions)


def test_the_smoke_test_reads_only_what_the_recipe_ships_with_it():
    # conda-build runs the test in a directory holding the recipe's `source_files` and nothing else of the tree.
    recipe = _text(RECIPE)
    shipped = re.search(r"^      source_files:\n((?:        - .*\n)+)", recipe, re.MULTILINE)

    assert shipped is not None

    files = [line.strip()[2:] for line in shipped.group(1).splitlines()]
    sample = re.search(r'^SAMPLE="(.+)"$', _text(SMOKE_TEST), re.MULTILINE).group(1)

    assert sample in files

    for path in files:
        with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as handle:
            first = handle.readline()

        assert not first.startswith("version https://git-lfs"), f"{path} is a Git LFS pointer in a plain clone"


def test_the_smoke_test_needs_no_card():
    # An image build has no GPU, so the package's test cannot be the test suite. It runs a CPU pipeline.
    script = _text(SMOKE_TEST)

    assert "--use_cpu_only" in script
    assert "pytest" not in script


def _cmake_version() -> str:
    return re.search(r"^project\(morpheus\s+VERSION\s+([0-9.]+)", _text("CMakeLists.txt"), re.MULTILINE).group(1)


def test_the_version_has_a_source_when_the_clone_has_no_tag():
    # This fork carries no tags. The conda build and the release image both took their version from
    # `git describe --tags`, which in a clone of it is an error.
    version = subprocess.run([os.path.join(REPO_ROOT, VERSION_SCRIPT)], check=True, capture_output=True,
                             text=True).stdout.strip()
    described = subprocess.run(["git", "-C", REPO_ROOT, "describe", "--tags", "--abbrev=0"],
                               check=False,
                               capture_output=True,
                               text=True)

    if (described.returncode == 0):
        assert version == described.stdout.strip()
    else:
        assert version == f"v{_cmake_version()}"

    assert "project_version.sh" in _text(CONDA_BUILD)

    for script in ("build_container_release.sh", "run_container_release.sh"):
        text = _text("docker", script)

        assert "project_version.sh" in text, script
        assert "git describe" not in text, script


def _stages() -> list[str]:
    return re.findall(r"^FROM .+ AS (\w+)$", _text(DOCKERFILE), re.MULTILINE)


def test_every_stage_the_dockerfile_builds_from_is_one_it_defines():
    text = _text(DOCKERFILE)
    stages = _stages()

    assert len(stages) == len(set(stages)) > 5

    for source in re.findall(r"--from=([\w./:${}-]+)", text):
        assert source in stages, f"--from={source} names no stage; an image named here is not pinned by CONDA_IMAGE"

    for (position,
         parent) in enumerate(re.findall(r"^FROM --platform=\$TARGETPLATFORM (\S+) AS \w+$", text, re.MULTILINE)):
        if (not parent.startswith("${")):
            assert parent in stages[:position], f"stage {stages[position]} builds from {parent} before it is defined"


def test_the_image_starts_from_a_pinned_conda_and_verifies_what_it_downloads():
    text = _text(DOCKERFILE)
    image = re.search(r'^ARG CONDA_IMAGE="(.+)"$', text, re.MULTILINE).group(1)

    assert re.fullmatch(r"[\w./-]+:\d[\w.-]*", image), f"{image} is not a version tag"

    instructions = [line for line in text.splitlines() if not line.lstrip().startswith("#")]

    assert not any(":latest" in line for line in instructions)
    assert not any("ssl_verify" in line for line in instructions)


def test_the_image_installs_the_package_from_the_channels_it_was_built_against():
    built_against = re.findall(r'^CONDA_ARGS_ARRAY\+=\("-c" ".*\}([\w/-]+)"\)$', _text(CONDA_BUILD), re.MULTILINE)

    assert len(built_against) >= 3

    installs = re.findall(r"mamba install -n morpheus \\\n((?:\s+-c [\w/-]+ \\\n)+)\s+morpheus-core", _text(DOCKERFILE))

    assert len(installs) == 2, "the docs stage and the runtime stage each install the built package"

    for block in installs:
        assert re.findall(r"-c ([\w/-]+)", block) == ["local"] + built_against


def test_every_build_argument_the_script_passes_is_one_the_dockerfile_takes():
    declared = set(re.findall(r"^ARG (\w+)", _text(DOCKERFILE), re.MULTILINE))
    passed = set(re.findall(r"--build-arg (\w+)=", _text(BUILD_SCRIPT)))

    assert len(passed) > 5
    assert passed <= declared, f"build_container.sh passes {sorted(passed - declared)}, which the Dockerfile ignores"


def _fetch_data():
    spec = importlib.util.spec_from_file_location("fetch_data", os.path.join(REPO_ROOT, "scripts", "fetch_data.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_the_runtime_image_ships_the_models_the_stages_score_with():
    # `TC4AbpScoreStage` and `TC7SidScoreStage` read the model inventory from the repository. The image installs the
    # package outside one, so it has to carry the inventory, fetch its Git LFS file and say where it is.
    text = _text(DOCKERFILE)
    runtime = text[text.index("# ============ Stage: runtime ============"):text.
                   index("# ============ Stage: development ============")]

    assert 'COPY --from=git_clone "/tmp/morpheus_repo/models/inventory" "./models/inventory"' in runtime
    assert "ENV MORPHEUS_REPOSITORY=${MORPHEUS_ROOT}" in runtime

    fetched = re.search(r"\./scripts/fetch_data\.py fetch ([\w ]+)$", text, re.MULTILINE).group(1).split()
    datasets = _fetch_data().LFS_DATASETS

    assert "inventory" in fetched
    assert set(fetched) <= set(datasets)
    assert datasets["inventory"] == "models/inventory/**"
    assert os.path.isfile(os.path.join(REPO_ROOT, "models", "inventory", "inventory.json"))


@pytest.mark.parametrize("name", ["dev", "runtime"])
def test_the_environment_file_each_image_copies_exists(name: str):
    # The Dockerfile copies `<name>_cuda-<major><minor>_arch-<arch>.yaml`. The script that builds it once offered
    # arm64, whose files the audit removed.
    text = _text(DOCKERFILE)
    major = re.search(r"^ARG CUDA_MAJOR_VER=(\d+)$", text, re.MULTILINE).group(1)
    minor = re.search(r"^ARG CUDA_MINOR_VER=(\d+)$", text, re.MULTILINE).group(1)
    arch = re.search(r"^ARG REAL_ARCH=(\w+)$", text, re.MULTILINE).group(1)

    assert os.path.isfile(
        os.path.join(REPO_ROOT, "conda", "environments", f"{name}_cuda-{major}{minor}_arch-{arch}.yaml"))
    assert "aarch64" not in _text(BUILD_SCRIPT)
