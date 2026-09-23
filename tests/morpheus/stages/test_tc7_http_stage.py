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
from morpheus.stages.telemetry.tc7_http_stage import TC7HttpStage
from morpheus.utils.type_utils import get_df_class

SECOND = 10**9
START = 10**18


def truthy(value) -> bool:
    """A column value as a plain bool, with a null reading as False.

    `bool(pd.NA)` raises rather than returning False, and every flag on these rows is nullable because "not
    answerable" is a distinct state from "no".
    """
    return False if pd.isna(value) else bool(value)


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC7HttpStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


def requests(statuses, paths=None, sources=None, times=None) -> dict:
    count = len(statuses)

    return {
        "src_ip": list(sources) if sources is not None else ["10.0.0.5"] * count,
        "status_code": list(statuses),
        "url_path": list(paths) if paths is not None else [f"/p{index}" for index in range(count)],
        "event_time": list(times) if times is not None else [START + index * SECOND for index in range(count)],
    }


@pytest.mark.gpu_and_cpu_mode
def test_the_windowed_counts_and_their_ratio_follow_the_responses(config: Config):
    result = run(config, requests([200, 404, 404, 200, 404]))

    assert list(result["http_4xx_in_window"]) == [0, 1, 2, 2, 3]
    assert list(result["http_2xx_in_window"]) == [1, 1, 1, 2, 2]
    assert list(result["http_4xx_to_2xx_ratio"])[-1] == 1.5


@pytest.mark.gpu_and_cpu_mode
def test_a_client_that_has_found_nothing_has_no_ratio_and_both_counts(config: Config):
    # The purest enumerator: nothing it asked for exists. The ratio is undefined, and a search thresholding the
    # ratio column would silently exclude it -- which is why the counts ride beside it and the shipped search
    # tests `4xx > 0.7 * 2xx` instead.
    result = run(config, requests([404, 404, 404]))

    assert list(result["http_4xx_in_window"])[-1] == 3
    assert list(result["http_2xx_in_window"])[-1] == 0
    assert pd.isna(list(result["http_4xx_to_2xx_ratio"])[-1])


@pytest.mark.gpu_and_cpu_mode
def test_redirects_and_server_errors_stay_out_of_the_ratio(config: Config):
    # The rule names 2xx and 4xx and nothing else. A 301 is neither a success nor a refusal of the path asked
    # for, and a 5xx is about the server rather than about whether the path exists.
    result = run(config, requests([200, 301, 503, 404]))

    assert list(result["http_status_class"]) == ["2xx", "3xx", "5xx", "4xx"]
    assert pd.isna(list(result["http_4xx_in_window"])[1])
    assert pd.isna(list(result["http_4xx_in_window"])[2])
    assert list(result["http_4xx_in_window"])[-1] == 1
    assert list(result["http_2xx_in_window"])[-1] == 1


@pytest.mark.gpu_and_cpu_mode
def test_every_request_counts_toward_distinct_paths_whatever_its_status(config: Config):
    result = run(config, requests([200, 301, 503, 404]))

    assert list(result["http_distinct_paths"]) == [1, 2, 3, 4]


@pytest.mark.gpu_and_cpu_mode
def test_a_repeated_path_is_not_a_new_one(config: Config):
    # The broken client hammering one missing resource: all refusals, one path. The distinct count is what keeps
    # it out of the rule.
    result = run(config, requests([404] * 4, paths=["/missing"] * 4))

    assert list(result["http_distinct_paths"])[-1] == 1
    assert list(result["http_4xx_in_window"])[-1] == 4


@pytest.mark.gpu_and_cpu_mode
def test_a_burst_on_one_timestamp_is_counted_rather_than_refused(config: Config):
    # A proxy logging at one-second resolution puts an enumerator's burst on one tick.
    result = run(config, requests([404] * 5, times=[START] * 5))

    assert list(result["http_4xx_in_window"]) == [1, 2, 3, 4, 5]
    assert list(result["http_distinct_paths"]) == [1, 2, 3, 4, 5]


@pytest.mark.gpu_and_cpu_mode
def test_two_clients_keep_separate_windows(config: Config):
    result = run(config, requests([404, 404, 200], sources=["10.0.0.5", "10.0.0.6", "10.0.0.5"]))

    assert list(result["http_4xx_in_window"]) == [1, 1, 1]
    assert list(result["http_2xx_in_window"]) == [0, 0, 1]


@pytest.mark.gpu_and_cpu_mode
def test_an_old_request_leaves_the_window(config: Config):
    result = run(config, requests([404, 404], times=[START, START + 3600 * SECOND]))

    assert list(result["http_4xx_in_window"]) == [1, 1]
    assert list(result["http_distinct_paths"]) == [1, 1]


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("status", [None, "not-a-status", 42, 999, 404.5])
def test_something_that_is_not_a_status_has_no_class(config: Config, status):
    result = run(config, requests([status]))

    assert pd.isna(list(result["http_status_class"])[0])
    assert pd.isna(list(result["http_4xx_in_window"])[0])


@pytest.mark.gpu_and_cpu_mode
def test_a_saturated_window_says_so(config: Config):
    result = run(config, requests([404] * 6), max_samples=3)

    assert truthy(list(result["http_window_saturated"])[-1]) is True


@pytest.mark.cpu_mode
def test_a_missing_column_is_refused(config: Config):
    with pytest.raises(KeyError, match="status_code"):
        run(config, {"src_ip": ["10.0.0.5"], "url_path": ["/"], "event_time": [START]})


@pytest.mark.cpu_mode
def test_an_empty_key_list_is_refused(config: Config):
    with pytest.raises(ValueError, match="key_columns"):
        TC7HttpStage(config, key_columns=[])
