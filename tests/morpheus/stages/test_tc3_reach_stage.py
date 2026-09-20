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
from morpheus.stages.telemetry.tc3_reach_stage import TC3ReachStage
from morpheus.utils.type_utils import get_df_class

SECOND = 10**9
START = 10**18


def flows(destinations, sources=None, asns=None, out_bytes=None, in_bytes=None) -> dict:
    count = len(destinations)

    return {
        "src_ip": list(sources) if sources is not None else ["10.0.0.5"] * count,
        "dst_ip": list(destinations),
        "bgp_as_dst": list(asns) if asns is not None else ["64512"] * count,
        "bytes_out": list(out_bytes) if out_bytes is not None else [100] * count,
        "bytes_in": list(in_bytes) if in_bytes is not None else [100] * count,
        "event_time": [START + index * SECOND for index in range(count)],
    }


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("min_denominator", 1)
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC3ReachStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_the_destination_is_classified_by_the_shared_parsers(config: Config):
    # Not a private copy of the rules. `parsers/ip.py` is already in Morpheus, is vectorized over both frame
    # libraries, and follows Python's own `ipaddress` semantics rather than an intuition about what looks
    # internal -- which includes counting loopback and the reserved ranges as private.
    result = run(config, flows(["10.0.0.9", "8.8.8.8", "224.0.0.1", "240.0.0.1", "127.0.0.1"]))

    assert list(result["dst_is_private"]) == [True, False, False, True, True]
    assert list(result["dst_is_multicast"]) == [False, False, True, False, False]
    assert list(result["dst_is_reserved"]) == [False, False, False, True, False]


@pytest.mark.gpu_and_cpu_mode
def test_the_internal_proportion_is_what_turns_a_count_into_a_rule(config: Config):
    # Four hundred destinations is a scan when they are inside the estate and a busy browser when they are not,
    # which is why R-B-L3-001 requires both conditions.
    inside = run(config, flows(["10.0.0.9", "10.0.0.10", "10.0.0.11", "10.0.0.12"]))
    outside = run(config, flows(["8.8.8.8", "1.1.1.1", "9.9.9.9", "208.67.222.222"]))

    assert list(inside["internal_dst_ratio"])[-1] == pytest.approx(1.0)
    assert list(outside["internal_dst_ratio"])[-1] == pytest.approx(0.0)


@pytest.mark.gpu_and_cpu_mode
def test_no_proportion_is_published_from_one_flow(config: Config):
    # A source's first destination makes it either wholly internal or wholly external, and a rule reading that
    # fires on every host's first minute.
    result = run(config, flows(["10.0.0.9", "8.8.8.8"]), min_denominator=5)

    assert result["internal_dst_ratio"].isna().all()


@pytest.mark.gpu_and_cpu_mode
def test_byte_asymmetry_survives_a_flow_that_got_nothing_back(config: Config):
    # The plus one in the guide's formula is load-bearing: exfiltration's defining shape is that nothing came
    # back, so the denominator is zero exactly when the feature matters most.
    result = run(config, flows(["8.8.8.8", "8.8.4.4"], out_bytes=[5000, 100], in_bytes=[0, 100]))

    assert list(result["byte_asymmetry"])[0] == pytest.approx(5000.0)
    assert list(result["byte_asymmetry"])[1] == pytest.approx(100.0 / 101.0)


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_byte_count_is_null_rather_than_zero(config: Config):
    # Zero is the value for a flow that sent nothing. Reporting it for a field the collector omitted would make
    # an incomplete record look like a quiet one.
    result = run(config, flows(["8.8.8.8"], out_bytes=[None], in_bytes=[10]))

    assert pd.isna(list(result["byte_asymmetry"])[0])


@pytest.mark.gpu_and_cpu_mode
def test_a_networks_novelty_is_permanent_rather_than_windowed(config: Config):
    # "Has this source ever reached this network" is the question worth asking about a first contact, and it is
    # a different question from "in the last hour".
    result = run(config, flows(["8.8.8.8", "1.1.1.1", "8.8.4.4"], asns=["15169", "13335", "15169"]))
    novelty = list(result["dst_asn_first_seen"])

    assert pd.isna(novelty[0])
    assert (novelty[1], novelty[2]) == (True, False)


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_network_is_not_a_new_one(config: Config):
    # A collector that stopped populating the field would otherwise look like a host that started roaming.
    result = run(config, flows(["8.8.8.8", "1.1.1.1"], asns=["15169", None]))

    assert pd.isna(list(result["dst_asn_first_seen"])[1])


@pytest.mark.gpu_and_cpu_mode
def test_two_sources_keep_separate_proportions(config: Config):
    result = run(config, flows(["10.0.0.9", "8.8.8.8", "10.0.0.10"], sources=["10.0.0.5", "10.0.0.6", "10.0.0.5"]))
    ratios = list(result["internal_dst_ratio"])

    assert ratios[0] == pytest.approx(1.0)
    assert ratios[1] == pytest.approx(0.0)
    assert ratios[2] == pytest.approx(1.0)


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_column_is_refused(config: Config):
    meta = MessageMeta(get_df_class(config.execution_mode)({"src_ip": ["10.0.0.5"], "event_time": [START]}))

    with pytest.raises(KeyError, match="dst_ip"):
        TC3ReachStage(config).on_data(meta)


def test_a_non_positive_window_is_refused(config: Config):
    with pytest.raises(ValueError, match="window_seconds"):
        TC3ReachStage(config, window_seconds=0)
