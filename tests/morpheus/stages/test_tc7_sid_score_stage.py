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
The SID stage writes what the scorer found, and tells "found nothing" apart from "had nothing to read".

The model is stored in Git LFS; see `tests/morpheus/utils/test_sid_scorer.py` for how an unfetched file is treated.
"""

import os

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.stages.telemetry.tc7_sid_score_stage import SID_LABELS
from morpheus.stages.telemetry.tc7_sid_score_stage import SID_MODEL_VERSION
from morpheus.stages.telemetry.tc7_sid_score_stage import TC7SidScoreStage
from morpheus.stages.telemetry.tc7_sid_score_stage import score_column
from morpheus.utils.model_inventory import REPOSITORY_INVENTORY
from morpheus.utils.model_inventory import ModelFileUnavailable
from morpheus.utils.model_inventory import load_entry
from morpheus.utils.type_utils import get_df_class

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INVENTORY = os.path.join(REPO_ROOT, REPOSITORY_INVENTORY)
KEY = "aws_secret_access_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"


@pytest.fixture(name="fetched", autouse=True, scope="module")
def fetched_fixture():
    try:
        load_entry(INVENTORY, "sid-minibert-20230424")
    except ModelFileUnavailable as error:
        if (os.environ.get("CI")):
            pytest.fail(f"the workflow fetches the inventory's LFS files before the tiers, and {error}")

        pytest.skip(str(error))


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC7SidScoreStage(config, inventory_path=INVENTORY, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_a_key_is_labelled_and_plain_text_is_not(config: Config):
    result = run(config, {"body": [KEY, "the weather report said clouds again today"]})

    assert "secret_keys" in list(result[SID_LABELS])[0].split(",")
    assert list(result[SID_LABELS])[1] == ""
    assert list(result[score_column("secret_keys")])[0] > 0.5
    assert set(result[SID_MODEL_VERSION]) == {"sid-minibert-20230424:d12f31d6dcebd03d"}


@pytest.mark.gpu_and_cpu_mode
def test_a_row_with_nothing_to_read_has_no_score_rather_than_an_empty_finding(config: Config):
    result = run(config, {"body": ["", None, KEY]})

    assert pd.isna(list(result[SID_LABELS])[0])
    assert pd.isna(list(result[SID_LABELS])[1])
    assert pd.isna(list(result[score_column("password")])[0])
    assert not pd.isna(list(result[SID_LABELS])[2])


@pytest.mark.cpu_mode
def test_a_missing_body_read_from_a_file_is_not_the_text_nan(config: Config):
    # A JSON lines reader gives a null string column as NaN, and `str(nan)` is "nan", which the model would read and
    # score: every bodyless request in a file-fed segment came out with probabilities and an empty finding.
    result = run(config, {"body": [float("nan"), KEY]})

    assert pd.isna(list(result[SID_LABELS])[0])
    assert pd.isna(list(result[score_column("password")])[0])
    assert "secret_keys" in list(result[SID_LABELS])[1].split(",")


@pytest.mark.gpu_and_cpu_mode
def test_the_text_column_is_a_parameter(config: Config):
    result = run(config, {"response_content": [KEY]}, text_column="response_content")

    assert "secret_keys" in list(result[SID_LABELS])[0].split(",")

    with pytest.raises(KeyError, match="'body'"):
        run(config, {"response_content": [KEY]})
