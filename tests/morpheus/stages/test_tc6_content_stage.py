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
from morpheus.stages.telemetry.tc6_content_stage import TC6ContentStage
from morpheus.utils.type_utils import get_df_class

SECOND = 10**9
START = 10**18


def transfers(declared, detected) -> dict:
    return {
        "content_type_declared": list(declared),
        "content_type_detected": list(detected),
        "event_time": [START + index * SECOND for index in range(len(declared))],
    }


def truthy(value) -> bool:
    """A column value as a plain bool, with a null reading as False.

    `bool(pd.NA)` raises rather than returning False, and every flag on these rows is nullable because "not yet
    answerable" is a distinct state from "no". Tests that mean "this did not fire" say so through this.
    """
    return False if pd.isna(value) else bool(value)


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC6ContentStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_an_archive_behind_a_declared_image_crosses_a_boundary(config: Config):
    result = run(config, transfers(["image/png"], ["application/zip"]))

    assert list(result["content_category_declared"]) == ["image"]
    assert list(result["content_category_detected"]) == ["archive"]
    assert truthy(list(result["content_category_crossed"])[0]) is True


@pytest.mark.gpu_and_cpu_mode
def test_a_re_encoding_does_not(config: Config):
    # The negative half, and the reason the rule compares categories rather than types. An estate has
    # thousands of these and a rule reporting them all would bury the one that matters.
    result = run(config, transfers(["image/png"], ["image/jpeg"]))

    assert truthy(list(result["content_category_crossed"])[0]) is False


@pytest.mark.gpu_and_cpu_mode
def test_a_charset_parameter_is_not_a_content_change(config: Config):
    result = run(config, transfers(["text/html; charset=utf-8"], ["text/html"]))

    assert truthy(list(result["content_category_crossed"])[0]) is False


@pytest.mark.gpu_and_cpu_mode
def test_an_unrecognized_type_yields_no_verdict_and_is_counted(config: Config):
    result = run(config, transfers(["image/png"], ["application/x-invented-here"]))

    assert pd.isna(list(result["content_category_crossed"])[0])
    assert truthy(list(result["content_category_unclassified"])[0]) is True


@pytest.mark.gpu_and_cpu_mode
def test_a_record_carrying_no_types_is_not_a_gap_in_the_map(config: Config):
    # The distinction the unclassified count depends on. A record with nothing to compare is not a type the
    # map failed to place, and counting it would make the gap look larger than it is.
    result = run(config, transfers([None], [None]))

    assert pd.isna(list(result["content_category_crossed"])[0])
    assert truthy(list(result["content_category_unclassified"])[0]) is False


@pytest.mark.gpu_and_cpu_mode
def test_a_frame_carrying_one_side_answers_no_verdict_rather_than_failing(config: Config):
    payload = {
        "content_type_declared": ["image/png"],
        "event_time": [START],
    }
    result = run(config, payload)

    assert list(result["content_category_declared"]) == ["image"]
    assert pd.isna(list(result["content_category_crossed"])[0])


@pytest.mark.cpu_mode
def test_a_frame_carrying_neither_side_is_refused(config: Config):
    # A feed that cannot support this rule at all is worth failing on, rather than filling with nulls that
    # read as an absence of mismatches.
    with pytest.raises(KeyError, match="content_type"):
        run(config, {"event_time": [START]})
