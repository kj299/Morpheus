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
The CLI half of the `@register_stage` contract.

A stage that registers with the CLI promises that a click command can be built from its constructor signature.
That build happens lazily, so a constructor annotation the CLI cannot express (a `Union`, an `Optional`, an
ignored argument without a default) crashes `morpheus run` for anyone who touches the command, while every
pipeline-level test stays green. This module forces the build for each lineage stage, which is exactly the check
that caught two such crashes after four otherwise-green test sessions.
"""

import pytest
from click.testing import CliRunner

from morpheus.stages.lineage.binding_resolver_stage import BindingResolverStage
from morpheus.stages.lineage.community_id_stage import CommunityIdStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage

LINEAGE_STAGES = [BindingResolverStage, CommunityIdStage, LineageStampStage, WindowSealStage]


@pytest.mark.parametrize("stage_class", LINEAGE_STAGES, ids=lambda cls: cls.__name__)
def test_cli_command_builds_and_renders_help(stage_class: type):
    registration = getattr(stage_class, "_morpheus_registered_stage", None)

    assert registration is not None, f"{stage_class.__name__} is not registered with the CLI"

    command = registration.build_command()
    result = CliRunner().invoke(command, ["--help"])

    assert result.exit_code == 0, result.output


def test_ignored_arguments_stay_required_programmatically(config):
    # binding_table's None default exists only to satisfy the CLI's ignored-argument rule; the constructor must
    # still refuse it.
    with pytest.raises(ValueError, match="binding_table is required"):
        BindingResolverStage(config)
