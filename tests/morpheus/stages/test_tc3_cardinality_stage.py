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
from morpheus.stages.telemetry.tc3_cardinality_stage import TC3CardinalityStage
from morpheus.utils.type_utils import get_df_class

SECOND = 10**9
START = 10**18


def flows(sources, destinations, ports=None, times=None) -> dict:
    count = len(sources)

    return {
        "src_ip": list(sources),
        "dst_ip": list(destinations),
        "dst_port": list(ports) if ports is not None else [443] * count,
        "event_time": list(times) if times is not None else [START + index * SECOND for index in range(count)],
    }


def run(config: Config, payload: dict, stage: TC3CardinalityStage = None, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    (stage or TC3CardinalityStage(config, **kwargs)).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_fan_out_counts_the_places_one_source_reached(config: Config):
    result = run(config, flows(["10.0.0.5"] * 4, ["10.0.0.9", "10.0.0.10", "10.0.0.11", "10.0.0.9"]))

    assert list(result["dsts_per_src"]) == [1, 2, 3, 3]
    assert list(result["dsts_per_src_first_in_window"]) == [True, True, True, False]


@pytest.mark.gpu_and_cpu_mode
def test_fan_in_is_the_same_arithmetic_pointed_the_other_way(config: Config):
    # A rise on a server is ordinary and a rise on a workstation is not, which is why the count is published
    # rather than thresholded in the stage.
    result = run(config, flows(["10.0.0.5", "10.0.0.6", "10.0.0.7"], ["10.0.0.9"] * 3))

    assert list(result["srcs_per_dst"]) == [1, 2, 3]
    assert list(result["dsts_per_src"]) == [1, 1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_the_two_shapes_of_a_scan_are_separated(config: Config):
    # Many addresses on one port is a hunt for a service; many ports on one address is a hunt for a way in.
    service = run(config, flows(["10.0.0.5"] * 3, ["10.0.0.9", "10.0.0.10", "10.0.0.11"], ports=[445] * 3))
    entry = run(config, flows(["10.0.0.5"] * 3, ["10.0.0.9"] * 3, ports=[22, 80, 445]))

    assert list(service["dsts_per_src"])[-1] == 3
    assert list(service["dst_ports_per_src"])[-1] == 1
    assert list(entry["dsts_per_src"])[-1] == 1
    assert list(entry["dst_ports_per_src"])[-1] == 3


@pytest.mark.gpu_and_cpu_mode
def test_the_current_flow_is_counted_inside_its_own_window(config: Config):
    # A threshold has to trip on the flow that crosses it rather than on the one after, or every alert is one
    # flow late and the last flow of a burst is never counted at all.
    result = run(config, flows(["10.0.0.5"], ["10.0.0.9"]))

    assert list(result["dsts_per_src"]) == [1]


@pytest.mark.gpu_and_cpu_mode
def test_a_flow_leaving_the_window_stops_counting(config: Config):
    times = [START, START + 10 * SECOND, START + 7200 * SECOND]
    result = run(config,
                 flows(["10.0.0.5"] * 3, ["10.0.0.9", "10.0.0.10", "10.0.0.11"], times=times),
                 window_seconds=3600)

    assert list(result["dsts_per_src"]) == [1, 2, 1]


@pytest.mark.gpu_and_cpu_mode
def test_a_port_widened_to_float_does_not_fork_its_own_count(config: Config):
    # The defect the shared normalization exists to prevent: one null row widens the column, 443 renders as
    # "443.0", and the port count doubles without anything having changed on the wire.
    payload = flows(["10.0.0.5"] * 3, ["10.0.0.9"] * 3, ports=[443, None, 443])
    result = run(config, payload)
    counts = result["dst_ports_per_src"]

    assert [counts[0], counts[2]] == [1, 1]
    assert pd.isna(counts[1])


@pytest.mark.gpu_and_cpu_mode
def test_a_flow_missing_an_address_counts_towards_nothing(config: Config):
    # A flow with no destination is not a flow to nowhere; it is a record the collector did not finish, and
    # counting it under a fabricated key would put that record in somebody's fan-out.
    result = run(config, flows(["10.0.0.5", None, "10.0.0.5"], ["10.0.0.9", "10.0.0.10", "10.0.0.11"]))
    counts = result["dsts_per_src"]

    assert [counts[0], counts[2]] == [1, 2]
    assert pd.isna(counts[1])


@pytest.mark.gpu_and_cpu_mode
def test_saturation_says_the_count_is_a_floor(config: Config):
    # A scan is both the thing the feature exists to see and the thing that would exhaust the window, so a count
    # that stopped rising has to say whether the source stopped or the tracker did.
    destinations = [f"10.0.{index // 250}.{index % 250}" for index in range(12)]
    result = run(config, flows(["10.0.0.5"] * 12, destinations), max_samples=5)

    assert list(result["dsts_per_src"])[-1] == 5
    assert bool(list(result["dsts_per_src_saturated"])[-1]) is True


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_column_is_refused(config: Config):
    frame = get_df_class(config.execution_mode)({"src_ip": ["10.0.0.5"], "event_time": [START]})
    meta = MessageMeta(frame)

    with pytest.raises(KeyError, match="dst_ip"):
        TC3CardinalityStage(config).on_data(meta)


def test_a_non_positive_window_is_refused(config: Config):
    with pytest.raises(ValueError, match="window_seconds"):
        TC3CardinalityStage(config, window_seconds=0)


@pytest.mark.gpu_and_cpu_mode
def test_an_empty_batch_passes_through(config: Config):
    meta = MessageMeta(get_df_class(config.execution_mode)(flows([], [])))
    TC3CardinalityStage(config).on_data(meta)

    assert meta.count == 0
