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
from morpheus.stages.telemetry.tc7_dns_stage import TC7DnsStage
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
    TC7DnsStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


TUNNEL_LABEL = "iqqieph543y4e2zq7ehmpxib4sehknfcb4fq2bgdn3ma44pa"


def queries(names, times=None) -> dict:
    return {
        "query_name": list(names),
        "event_time": list(times) if times is not None else [START + index * SECOND for index in range(len(names))],
    }


@pytest.mark.gpu_and_cpu_mode
def test_a_query_is_split_into_its_registered_domain_and_the_part_below(config: Config):
    result = run(config, queries(["a.b.example.co.uk"]))

    assert list(result["dns_registered_domain"]) == ["example.co.uk"]
    assert list(result["dns_subdomain"]) == ["a.b"]


@pytest.mark.gpu_and_cpu_mode
def test_a_tunnel_label_clears_the_entropy_and_length_conditions(config: Config):
    result = run(config, queries([f"{TUNNEL_LABEL}.{TUNNEL_LABEL[::-1]}.evil-tunnel.com"]))

    assert list(result["dns_subdomain_entropy"])[0] > 4.0
    assert list(result["dns_mean_label_length"])[0] > 30


@pytest.mark.gpu_and_cpu_mode
def test_an_ordinary_name_clears_neither(config: Config):
    result = run(config, queries(["www.northwind-traders-inc.com"]))

    assert list(result["dns_subdomain_entropy"])[0] < 4.0
    assert list(result["dns_mean_label_length"])[0] < 30


@pytest.mark.gpu_and_cpu_mode
def test_distinct_subdomains_are_counted_per_registered_domain(config: Config):
    result = run(config, queries(["a.evil.com", "b.evil.com", "a.evil.com", "a.other.com"]))

    assert list(result["dns_subdomains_per_domain"]) == [1, 2, 2, 1]


@pytest.mark.gpu_and_cpu_mode
def test_an_apex_query_contributes_nothing_to_the_count(config: Config):
    # An apex query is not a subdomain, and counting it would let a domain reach the threshold with one fewer real
    # name than the rule requires.
    result = run(config, queries(["a.evil.com", "evil.com", "b.evil.com"]))

    assert pd.isna(list(result["dns_subdomains_per_domain"])[1])
    assert list(result["dns_subdomains_per_domain"])[2] == 2


@pytest.mark.gpu_and_cpu_mode
def test_a_burst_on_one_timestamp_is_counted_rather_than_refused(config: Config):
    # A resolver logging at one-second resolution puts a whole tunnel burst on one tick. Refusing equal timestamps
    # would count one name per second, which is the burst the rule exists to see.
    names = [f"chunk{index}.evil.com" for index in range(5)]
    result = run(config, queries(names, times=[START] * 5))

    assert list(result["dns_subdomains_per_domain"]) == [1, 2, 3, 4, 5]


@pytest.mark.gpu_and_cpu_mode
def test_an_old_subdomain_leaves_the_window(config: Config):
    result = run(config, queries(["a.evil.com", "b.evil.com"], times=[START, START + 7200 * SECOND]))

    assert list(result["dns_subdomains_per_domain"]) == [1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_name_yields_no_measurements(config: Config):
    result = run(config, queries([None, "a.evil.com"]))

    assert pd.isna(list(result["dns_registered_domain"])[0])
    assert pd.isna(list(result["dns_subdomain_entropy"])[0])
    assert list(result["dns_subdomains_per_domain"])[1] == 1


@pytest.mark.gpu_and_cpu_mode
def test_a_saturated_window_says_so(config: Config):
    result = run(config, queries([f"s{index}.evil.com" for index in range(6)]), max_samples=3)

    assert truthy(list(result["dns_subdomains_saturated"])[-1]) is True


@pytest.mark.cpu_mode
def test_a_missing_column_is_refused(config: Config):
    with pytest.raises(KeyError, match="query_name"):
        run(config, {"event_time": [START]})


@pytest.mark.cpu_mode
def test_a_non_positive_window_is_refused(config: Config):
    with pytest.raises(ValueError, match="window_seconds"):
        TC7DnsStage(config, window_seconds=0)
