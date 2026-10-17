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
The CLI half of the `@register_stage` contract, for every stage this fork registers.

A stage that registers with the CLI promises that a click command can be built from its constructor signature.
That build happens lazily, so a constructor annotation the CLI cannot express (a `Union`, an `Optional`, an
ignored argument without a default) crashes `morpheus run` for anyone who touches the command, while every
pipeline-level test stays green. This module forces the build for each registered stage -- which caught two such
crashes after four otherwise-green sessions when it covered four stages, and three more (`Optional[str]` on two
layer 5 stages, a manifest on the determinism stamp) when it was widened to all of them -- checks that each is a
`morpheus run pipeline-other` command, and holds the compositions the example README documents to the CLI.
"""

import importlib
import importlib.util
import inspect
import json
import os
import pkgutil
import re
import shlex

import click
import pytest
from click.testing import CliRunner

import morpheus.stages.lineage
import morpheus.stages.telemetry
from morpheus.cli import commands
from morpheus.cli.stage_registry import GlobalStageRegistry
from morpheus.config import PipelineModes
from morpheus.stages.lineage.binding_resolver_stage import BindingResolverStage
from morpheus.stages.lineage.determinism_stamp_stage import DeterminismStampStage

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
EXAMPLE = os.path.join(REPO_ROOT, "examples", "behavioral_analytics")
README = os.path.join(EXAMPLE, "README.md")
INVENTORY = os.path.join(REPO_ROOT, "tests", "morpheus", "determinism", "test_readme_inventory.py")

OUTPUT_MODULES = ("morpheus.stages.output.siem_wire_stage", "morpheus.stages.output.siem_wire_sink_stage")
PYTHON_ONLY = ("determinism-stamp", "tc0-enrich", "binding-resolve", "tc5-score")
"""The commands whose required inputs are Python objects, which the README has to say."""

COMMAND_BLOCK = re.compile(r"```bash\n(morpheus run .*?)```", re.DOTALL)


def registered_stages() -> list:
    """Every class under the lineage, telemetry and SIEM wire modules that carries a CLI registration."""
    names = [
        module.name for package in (morpheus.stages.lineage, morpheus.stages.telemetry)
        for module in pkgutil.iter_modules(package.__path__, package.__name__ + ".")
    ] + list(OUTPUT_MODULES)
    found = []

    for name in sorted(names):
        module = importlib.import_module(name)

        for (_, cls) in inspect.getmembers(module, inspect.isclass):
            if (cls.__module__ == name and hasattr(cls, "_morpheus_registered_stage")):
                found.append(cls)

    return found


STAGES = registered_stages()


def _inventory():
    # The README inventory test knows how to find this fork's stage files and read the README's stated count; it is
    # loaded under its own name so pytest does not take it for a second copy of that test module.
    spec = importlib.util.spec_from_file_location("readme_inventory_helpers", INVENTORY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.mark.parametrize("stage_class", STAGES, ids=lambda cls: cls.__name__)
def test_cli_command_builds_and_renders_help(stage_class: type):
    command = stage_class._morpheus_registered_stage.build_command()  # pylint: disable=protected-access
    result = CliRunner().invoke(command, ["--help"])

    assert result.exit_code == 0, result.output


def test_every_registered_stage_this_fork_wrote_is_covered():
    inventory = _inventory()

    assert {cls.__name__ for cls in STAGES} == set(inventory.fork_stage_classes())
    assert len(STAGES) == inventory.claim_in("README")[0]


def test_every_stage_is_a_pipeline_other_command():
    names = set(GlobalStageRegistry.get().get_registered_names(PipelineModes.OTHER))
    missing = [cls.__name__ for cls in STAGES if cls._morpheus_registered_stage.name not in names]  # pylint: disable=protected-access

    assert missing == [], f"not registered in morpheus.cli.commands: {missing}"


def test_ignored_arguments_have_defaults():
    # An ignored argument without a default is the crash the module docstring describes; the build above would
    # raise, and this says which argument.
    for cls in STAGES:
        registration = cls._morpheus_registered_stage  # pylint: disable=protected-access
        signature = inspect.signature(cls.__init__)

        for name in getattr(registration, "ignore_args", None) or ():
            assert signature.parameters[name].default is not inspect.Parameter.empty, f"{cls.__name__}.{name}"


def test_ignored_arguments_stay_required_programmatically(config):
    # The None defaults exist only to satisfy the CLI's ignored-argument rule; the constructors must still refuse.
    with pytest.raises(ValueError, match="binding_table is required"):
        BindingResolverStage(config)

    with pytest.raises(ValueError, match="envelope is required"):
        DeterminismStampStage(config)


def _readme_commands() -> list:
    with open(README, encoding="utf-8") as handle:
        text = handle.read()

    blocks = [shlex.split(block.replace("\\\n", " ")) for block in COMMAND_BLOCK.findall(text)]

    assert len(blocks) == 9, "one composition per segment"

    return blocks


def test_the_readme_names_the_python_only_stages():
    with open(README, encoding="utf-8") as handle:
        text = handle.read()

    for name in PYTHON_ONLY:
        assert f"`{name}`" in text, name


def _rewrite(argv: list, out_dir: str) -> list:
    return [os.path.join(out_dir, token[len("out/"):]) if token.startswith("out/") else token for token in argv]


@pytest.fixture(name="parse_only")
def parse_only_fixture():
    """Replace the callback that runs the pipeline with one that records the stages the CLI built, then restore."""
    original = commands.pipeline_other._result_callback  # pylint: disable=protected-access
    built = {}

    @commands.pipeline_other.result_callback(replace=True)
    @click.pass_context
    def record(ctx, stages, *_, **__):
        built["stages"] = stages
        ctx.exit(47)

    yield built

    commands.pipeline_other._result_callback = original  # pylint: disable=protected-access


@pytest.mark.parametrize("index", range(9))
def test_each_documented_composition_builds_through_the_cli(index: int, parse_only, tmp_path, monkeypatch):
    argv = _rewrite(_readme_commands()[index], str(tmp_path))
    monkeypatch.chdir(EXAMPLE)

    result = CliRunner().invoke(commands.cli, argv[1:])

    assert result.exit_code == 47, result.output
    # Every command named after `pipeline-other` became one stage.
    registered = set(GlobalStageRegistry.get().get_registered_names(PipelineModes.OTHER))
    commands_named = [token for token in argv[argv.index("pipeline-other") + 1:] if token in registered]
    assert len(parse_only["stages"]) == len(commands_named) > 3


@pytest.mark.slow
def test_the_context_composition_writes_records(tmp_path, monkeypatch):
    argv = _rewrite(_readme_commands()[-1], str(tmp_path))
    monkeypatch.chdir(EXAMPLE)

    result = CliRunner().invoke(commands.cli, argv[1:])

    assert result.exit_code == 0, result.output

    with open(os.path.join(tmp_path, "context_identity.jsonlines"), encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]

    assert len(records) > 0
    assert all("user_principal" in record and None not in record.values() for record in records)
