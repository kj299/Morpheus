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
import json
import os
import shutil

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.stages.telemetry.tc4_abp_score_stage import ABP_ANOMALOUS
from morpheus.stages.telemetry.tc4_abp_score_stage import ABP_MODEL_VERSION
from morpheus.stages.telemetry.tc4_abp_score_stage import ABP_SCORE
from morpheus.stages.telemetry.tc4_abp_score_stage import TC4AbpScoreStage
from morpheus.stages.telemetry.tc4_flow_stage import MODEL_FEATURES
from morpheus.utils.model_inventory import REPOSITORY_INVENTORY
from morpheus.utils.type_utils import get_df_class

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INVENTORY = os.path.join(REPO_ROOT, REPOSITORY_INVENTORY)


def check_rows() -> list:
    with open(os.path.join(os.path.dirname(INVENTORY), "abp-pcap-xgb", "conversion_check.json"),
              encoding="utf-8") as handle:
        document = json.load(handle)

    return document["rows"][:3] + document["probes"]["rows"]


def flows(rows: list) -> dict:
    """Flow records with each feature under the column TC4FlowStage writes it to."""
    return {column: [row["features"][index] for row in rows] for (index, column) in enumerate(MODEL_FEATURES.values())}


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC4AbpScoreStage(config, inventory_path=kwargs.pop("inventory_path", INVENTORY), **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_each_flow_is_scored_as_the_upstream_booster_scored_it(config: Config):
    rows = check_rows()
    result = run(config, flows(rows))

    assert list(result[ABP_SCORE]) == pytest.approx([row["probability"] for row in rows], abs=0)
    assert list(result[ABP_ANOMALOUS]) == [row["probability"] > 0.5 for row in rows]
    assert set(result[ABP_ANOMALOUS]) == {True, False}
    assert set(result[ABP_MODEL_VERSION]) == {"abp-pcap-xgb:c3e05a33fdc02a04"}


@pytest.mark.gpu_and_cpu_mode
def test_a_frame_without_the_flow_features_is_refused(config: Config):
    payload = flows(check_rows()[:1])
    del payload["flow_ppm"]

    with pytest.raises(KeyError, match="flow_ppm"):
        run(config, payload)


@pytest.mark.gpu_and_cpu_mode
def test_a_model_trained_on_another_feature_order_is_refused(config: Config, tmp_path):
    with open(INVENTORY, encoding="utf-8") as handle:
        inventory = json.load(handle)

    features = inventory["models"]["abp-pcap-xgb"]["settings"]["features"]
    features[0], features[1] = features[1], features[0]
    shutil.copytree(os.path.join(os.path.dirname(INVENTORY), "abp-pcap-xgb"), tmp_path / "abp-pcap-xgb")
    (tmp_path / "inventory.json").write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ValueError, match="orders must agree"):
        TC4AbpScoreStage(config, inventory_path=str(tmp_path / "inventory.json"))
