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

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.utils.bitemporal import MEMBERSHIP
from morpheus.utils.bitemporal import RETRACT
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.type_utils import get_df_class

DAY = 86400 * 10**9


def cell(value):
    return None if pd.isna(value) else value


def identity_store() -> BitemporalStore:
    return BitemporalStore(
        "identity",
        [
            make_version("profile", "carol", 0, None, 0, values={"department": "Sales"}),
            make_version("profile", "carol", 0, None, 20 * DAY, values={"department": "Marketing"}),
            make_version(MEMBERSHIP, "carol", 0, None, 0, values={"group_name": "zeta"}, key_parts=("zeta", )),
            make_version(MEMBERSHIP, "carol", 0, None, 0, values={"group_name": "alpha"}, key_parts=("alpha", )),
            make_version(MEMBERSHIP,
                         "carol",
                         10 * DAY,
                         None,
                         12 * DAY,
                         RETRACT,
                         values={"group_name": "zeta"},
                         key_parts=("zeta", )),
        ])


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("store", identity_store())
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC0EnrichStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


def events(times, principals=None) -> dict:
    return {
        "user_principal": list(principals) if principals is not None else ["carol"] * len(times),
        "event_time": list(times),
    }


@pytest.mark.gpu_and_cpu_mode
def test_the_default_is_what_was_known_at_the_event(config: Config):
    result = run(config, events([5 * DAY, 25 * DAY]))

    assert list(result["ctx_department"]) == ["Sales", "Marketing"]
    assert set(result["ctx_knowledge"]) == {"event"}


@pytest.mark.gpu_and_cpu_mode
def test_latest_knowledge_rewrites_the_past(config: Config):
    result = run(config, events([5 * DAY]), knowledge="latest")

    assert list(result["ctx_department"]) == ["Marketing"]


@pytest.mark.gpu_and_cpu_mode
def test_memberships_are_one_sorted_column(config: Config):
    # Two principals with the same groups carry the same string, which is what lets a search compare them.
    result = run(config, events([5 * DAY, 11 * DAY, 13 * DAY]))

    assert list(result["ctx_groups"]) == ["alpha|zeta", "alpha|zeta", "alpha"]


@pytest.mark.gpu_and_cpu_mode
def test_an_unknown_principal_is_not_found(config: Config):
    result = run(config, events([5 * DAY, 5 * DAY], principals=["carol", "mallory"]))

    assert list(result["ctx_found"]) == [True, False]
    assert cell(result["ctx_department"][1]) is None
    assert cell(result["ctx_groups"][1]) is None
    assert cell(result["ctx_version_uids"][1]) is None


@pytest.mark.gpu_and_cpu_mode
def test_every_answer_names_the_versions_it_rests_on_and_when_they_were_recorded(config: Config):
    store = identity_store()
    result = run(config, events([25 * DAY]), store=store)

    cited = set(result["ctx_version_uids"][0].split("|"))
    held = {version.uid for version in store.facts_about("carol", 25 * DAY, known_ns=25 * DAY)}

    assert cited == held
    assert result["ctx_recorded_at"][0] == 20 * DAY


@pytest.mark.gpu_and_cpu_mode
def test_an_untimed_row_gets_no_context(config: Config):
    result = run(config, events([None]))

    assert list(result["ctx_found"]) == [False]


@pytest.mark.gpu_and_cpu_mode
def test_asset_context_needs_no_membership_column(config: Config):
    store = BitemporalStore("asset", [make_version("asset", "db-ledger", 0, None, 0, values={"peer_group": "db"})])
    result = run(config, {"hostname": ["db-ledger"], "event_time": [DAY]}, store=store, entity_column="hostname")

    assert list(result["ctx_peer_group"]) == ["db"]
    assert "ctx_groups" not in result.columns


def test_constructor_rejects_bad_configuration(config: Config):
    with pytest.raises(ValueError, match="store"):
        TC0EnrichStage(config)

    with pytest.raises(ValueError, match="knowledge"):
        TC0EnrichStage(config, store=identity_store(), knowledge="tomorrow")

    with pytest.raises(ValueError, match="set_kinds"):
        TC0EnrichStage(config, store=identity_store(), set_kinds={MEMBERSHIP: "groups"})
