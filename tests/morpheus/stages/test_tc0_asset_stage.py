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
from morpheus.stages.telemetry.tc0_asset_stage import ASSET
from morpheus.stages.telemetry.tc0_asset_stage import TC0AssetStage
from morpheus.utils import bitemporal
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.type_utils import get_df_class

DAY = 86400 * 10**9


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC0AssetStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


def inventory(**overrides) -> dict:
    payload = {
        "hostname": ["db-ledger", "db-ledger", "ws-01"],
        "owner": ["frank", "frank", "alice"],
        "owning_team": ["finance-it", "finance-it", "finance-it"],
        "criticality": ["high", "high", "medium"],
        "data_classification": ["confidential", "restricted", "internal"],
        "peer_group": ["databases", "databases", None],
        "valid_from": [0, 5 * DAY, 0],
        "recorded_at": [0, 7 * DAY, 0],
    }
    payload.update(overrides)

    return payload


@pytest.mark.gpu_and_cpu_mode
def test_every_record_is_an_asset_version_keyed_on_its_host(config: Config):
    result = run(config, inventory())

    assert set(result[bitemporal.CONTEXT_KIND]) == {ASSET}
    assert list(result[bitemporal.CONTEXT_KEY]) == ["db-ledger", "db-ledger", "ws-01"]
    assert list(result[bitemporal.CONTEXT_ATTRIBUTES])[0] == \
        "criticality,data_classification,owner,owning_team,peer_group"


@pytest.mark.gpu_and_cpu_mode
def test_a_reclassification_recorded_late_is_both_answers(config: Config):
    store = BitemporalStore.from_records("t", run(config, inventory()).to_dict("records"))

    assert store.resolve("db-ledger", 6 * DAY, known_ns=6 * DAY).attributes["data_classification"] == "confidential"
    assert store.resolve("db-ledger", 6 * DAY).attributes["data_classification"] == "restricted"


@pytest.mark.gpu_and_cpu_mode
def test_a_host_with_no_peer_group_gets_none_rather_than_a_guess(config: Config):
    store = BitemporalStore.from_records("t", run(config, inventory()).to_dict("records"))

    assert store.resolve("ws-01", DAY).attributes["peer_group"] is None


@pytest.mark.gpu_and_cpu_mode
def test_a_row_without_a_host_or_its_recorded_instant_is_refused(config: Config):
    result = run(config, inventory(hostname=[None, "db-ledger", "ws-01"], recorded_at=[0, None, 0]))

    assert list(result[bitemporal.CONTEXT_REFUSED].fillna("")) == [bitemporal.NO_ENTITY, bitemporal.NO_RECORDED_AT, ""]


@pytest.mark.gpu_and_cpu_mode
def test_the_attributes_are_configurable(config: Config):
    payload = {"asset_id": ["a1"], "tier": ["gold"], "valid_from": [0], "recorded_at": [0]}
    result = run(config, payload, asset_column="asset_id", asset_columns=["tier"])

    assert list(result[bitemporal.CONTEXT_ATTRIBUTES]) == ["tier"]


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_attribute_column_names_itself(config: Config):
    payload = inventory()
    del payload["peer_group"]

    with pytest.raises(KeyError, match="peer_group"):
        run(config, payload)
