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
from morpheus.stages.telemetry.tc4_envelope_stage import TC4EnvelopeStage
from morpheus.utils.type_utils import get_df_class

HOUR = 3600 * 10**9
START = 10**18


def transfers(values, dst_port=445, dst_ip="10.0.1.9") -> dict:
    count = len(values)

    return {
        "src_ip": ["10.0.0.5"] * count,
        "dst_ip": [dst_ip] * count,
        "dst_port": [dst_port] * count,
        "flow_bpp": [float(value) for value in values],
        "flow_data_len": [int(value) * 10 for value in values],
        "event_time": [START + index * HOUR for index in range(count)],
    }


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("min_samples", 5)
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC4EnvelopeStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_a_transfer_far_above_the_triples_own_normal_is_a_breach(config: Config):
    result = run(config, transfers([100] * 10 + [500]))
    last = result.iloc[-1]

    assert last["flow_bpp_envelope"] == 100.0
    assert last["flow_bpp_envelope_ratio"] == pytest.approx(5.0)
    assert bool(last["flow_bpp_envelope_breached"]) is True
    assert bool(last["flow_bpp_envelope_mature"]) is True


@pytest.mark.gpu_and_cpu_mode
def test_the_key_is_the_triple_and_not_the_flow(config: Config):
    # A flow identifier carries the ephemeral source port, so every conversation would be a new entity and no
    # baseline would ever mature. The triple is the unit a transfer size is a property of.
    result = run(config, transfers([100] * 8))

    assert set(result["transfer_triple"]) == {"10.0.0.5:10.0.1.9:445"}


@pytest.mark.gpu_and_cpu_mode
def test_the_same_pair_on_another_port_is_another_conversation(config: Config):
    payload = transfers([100] * 12)
    payload["dst_port"] = [445] * 6 + [9100] * 6
    result = run(config, payload, min_samples=3)

    assert result["transfer_triple"].nunique() == 2

    # The second port starts from nothing: its first transfers have no envelope, while the first port's are
    # already mature. Two conversations, two histories.
    assert pd.isna(result.iloc[6]["flow_bpp_envelope"])
    assert bool(result.iloc[5]["flow_bpp_envelope_mature"]) is True
    assert bool(result.iloc[-1]["flow_bpp_envelope_mature"]) is True


@pytest.mark.gpu_and_cpu_mode
def test_both_magnitudes_are_tracked_separately(config: Config):
    # The rule names two, and a transfer can breach either without the other: one is the shape of the traffic
    # and the other is its volume.
    result = run(config, transfers([100] * 10 + [500]))

    for column in ("flow_bpp", "flow_data_len"):
        assert f"{column}_envelope" in result.columns
        assert bool(result.iloc[-1][f"{column}_envelope_breached"]) is True


@pytest.mark.gpu_and_cpu_mode
def test_no_envelope_is_published_from_too_few_transfers(config: Config):
    # Below the floor a 99th percentile by nearest rank is simply the maximum, so publishing one would put a
    # figure in the output that is not what its name says.
    result = run(config, transfers([100] * 3), min_samples=100)

    assert result["flow_bpp_envelope"].isna().all()
    assert not any(bool(value) for value in result["flow_bpp_envelope_mature"])


@pytest.mark.gpu_and_cpu_mode
def test_a_record_without_the_magnitude_is_left_alone(config: Config):
    payload = transfers([100] * 10)
    payload["flow_bpp"][4] = None
    result = run(config, payload)

    assert pd.isna(list(result["flow_bpp_envelope_ratio"])[4])
    assert bool(result.iloc[-1]["flow_bpp_envelope_mature"]) is True


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_key_column_is_refused(config: Config):
    meta = MessageMeta(get_df_class(config.execution_mode)({"src_ip": ["10.0.0.5"], "event_time": [START]}))

    with pytest.raises(KeyError, match="dst_ip"):
        TC4EnvelopeStage(config).on_data(meta)


def test_an_empty_magnitude_list_is_refused(config: Config):
    with pytest.raises(ValueError, match="magnitude_columns"):
        TC4EnvelopeStage(config, magnitude_columns=[])


def test_an_empty_key_is_refused(config: Config):
    with pytest.raises(ValueError, match="key_columns"):
        TC4EnvelopeStage(config, key_columns=[])
