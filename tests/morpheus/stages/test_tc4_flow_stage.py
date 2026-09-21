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
from morpheus.stages.telemetry.tc4_flow_stage import MODEL_FEATURES
from morpheus.stages.telemetry.tc4_flow_stage import TC4FlowStage
from morpheus.utils.type_utils import get_df_class

SECOND = 10**9
MINUTE = 60 * SECOND
START = 10**18 - (10**18 % MINUTE)
"""A whole minute, so a bin boundary is exactly where the tests say it is."""

SYN = 0x02
SYN_ACK = 0x12
PSH_ACK = 0x18
FIN_ACK = 0x11
RST = 0x04


def records(flags, times=None, payloads=None, src_port=50001, dst_port=443, dst_ip="10.0.1.9") -> dict:
    count = len(flags)

    return {
        "src_ip": ["10.0.0.5"] * count,
        "src_port": [src_port] * count,
        "dst_ip": [dst_ip] * count,
        "dst_port": [dst_port] * count,
        "tcp_flags": list(flags),
        "data_len": list(payloads) if payloads is not None else [0] * count,
        "event_time": list(times) if times is not None else [START + index * SECOND for index in range(count)],
    }


def run(config: Config, payload: dict, stage: TC4FlowStage = None, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    (stage or TC4FlowStage(config, **kwargs)).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_the_flow_identifier_is_the_format_part_two_names(config: Config):
    result = run(config, records([SYN]))

    assert list(result["flow_id"]) == ["10.0.0.5:50001=10.0.1.9:443"]


@pytest.mark.gpu_and_cpu_mode
def test_the_thirteen_features_are_all_emitted(config: Config):
    # The mapping is what a deployment feeding the shipped model builds its input frame from, so every entry in
    # it has to name a column that exists.
    result = run(config, records([SYN, SYN_ACK, PSH_ACK, FIN_ACK], payloads=[0, 0, 1200, 0]))

    assert len(MODEL_FEATURES) == 13

    for column in MODEL_FEATURES.values():
        assert column in result.columns, column


@pytest.mark.gpu_and_cpu_mode
def test_the_rollup_is_a_running_total_within_the_bin(config: Config):
    # Not a per-batch group. The row that closes a bin carries the complete figure, and a search takes the
    # maximum per bin, which is what makes the answer independent of how the stream was chunked.
    result = run(config, records([SYN, SYN_ACK, PSH_ACK, FIN_ACK], payloads=[0, 0, 1200, 0]))

    assert list(result["flow_ppm"]) == [1, 2, 3, 4]
    assert list(result["flow_data_len"]) == [0, 0, 1200, 1200]
    assert list(result["flow_all"]) == [1, 3, 5, 7]
    assert list(result["flow_bpp"]) == [0.0, 0.0, 400.0, 300.0]


@pytest.mark.gpu_and_cpu_mode
def test_a_bin_split_across_messages_still_ends_on_the_right_number(config: Config):
    # The correctness point the running total exists for. The reference groups within the message it was handed,
    # so a bin spanning two messages produces two partial aggregates and the model sees both as complete.
    payload = records([SYN, SYN_ACK, PSH_ACK, FIN_ACK], payloads=[0, 0, 1200, 0])
    whole = run(config, payload)

    stage = TC4FlowStage(config)
    frame = get_df_class(config.execution_mode)(payload)
    halves = [
        run(config, {name: values[:2]
                     for (name, values) in payload.items()}, stage=stage),
        run(config, {name: values[2:]
                     for (name, values) in payload.items()}, stage=stage),
    ]
    split = pd.concat(halves, ignore_index=True)

    assert frame is not None
    assert list(split["flow_ppm"]) == list(whole["flow_ppm"])
    assert list(split["flow_data_len"]) == list(whole["flow_data_len"])
    assert list(split["flow_all"]) == list(whole["flow_all"])


@pytest.mark.gpu_and_cpu_mode
def test_a_bin_is_half_open_and_labelled_by_its_start(config: Config):
    # The departure from the reference's rounding kernel, which labels a bin by its end and puts a record
    # landing exactly on a boundary into the following bin. Every window in this fork is [start, end), and a
    # layer 4 bin closing the other side of a boundary would disagree with every other layer about which side
    # an event falls on -- for exactly the events most likely to be on one.
    times = [START, START + MINUTE - 1, START + MINUTE]
    result = run(config, records([SYN, SYN, SYN], times=times))
    bins = list(result["rollup_time_ns"])

    assert bins[0] == START
    assert bins[1] == START
    assert bins[2] == START + MINUTE
    assert list(result["flow_ppm"]) == [1, 2, 1]


@pytest.mark.gpu_and_cpu_mode
def test_a_new_bin_starts_the_totals_again(config: Config):
    result = run(config, records([SYN] * 4, times=[START, START + SECOND, START + MINUTE, START + MINUTE + SECOND]))

    assert list(result["flow_ppm"]) == [1, 2, 1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_the_ratios_are_the_references_arithmetic(config: Config):
    result = run(config, records([SYN, SYN_ACK, PSH_ACK, FIN_ACK]))
    last = result.iloc[-1]

    assert last["flow_ack"] == 3
    assert last["flow_syn"] == 2
    assert last["flow_all"] == 7
    assert last["flow_syn_ratio"] == pytest.approx(2 / 7)
    assert last["flow_ackpush_ratio"] == pytest.approx((3 + 1) / 7)
    assert last["flow_fin_ratio"] == pytest.approx(1 / 7)
    assert last["flow_rst_ratio"] == pytest.approx(0.0)


@pytest.mark.gpu_and_cpu_mode
def test_a_bin_with_no_flags_has_no_ratio(config: Config):
    # A ratio of zero would claim the flow had flags and none of them were RST, rather than that it had none.
    result = run(config, records([0x00, 0x00]))

    assert result["flow_rst_ratio"].isna().all()
    assert list(result["flow_all"]) == [0, 0]


@pytest.mark.gpu_and_cpu_mode
def test_two_flows_do_not_share_a_bin(config: Config):
    payload = records([SYN, SYN])
    payload["dst_port"] = [443, 8443]
    result = run(config, payload)

    assert result["flow_id"].nunique() == 2
    assert list(result["flow_ppm"]) == [1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_the_flags_can_come_as_five_columns_instead_of_a_byte(config: Config):
    # Zeek and the flow exporters supply the counters rather than the raw byte, and a stage that only read the
    # byte would be unusable on the sources the guide lists first.
    payload = records([SYN, SYN_ACK])
    del payload["tcp_flags"]
    payload.update({"ack": [0, 1], "psh": [0, 0], "rst": [0, 0], "syn": [1, 1], "fin": [0, 0]})

    result = run(config, payload)

    assert list(result["flow_syn"]) == [1, 2]
    assert list(result["flow_ack"]) == [0, 1]


@pytest.mark.gpu_and_cpu_mode
def test_a_record_with_unreadable_flags_joins_no_bin(config: Config):
    result = run(config, records([SYN, 300, SYN]))

    assert pd.isna(list(result["flow_ppm"])[1])
    assert list(result["flow_ppm"])[2] == 2


@pytest.mark.gpu_and_cpu_mode
def test_a_port_widened_to_float_does_not_fork_a_flow(config: Config):
    payload = records([SYN, SYN, SYN])
    payload["src_port"] = [50001, None, 50001]
    result = run(config, payload)
    identifiers = [value for value in result["flow_id"] if isinstance(value, str)]

    assert len(set(identifiers)) == 1
    assert "50001.0" not in "".join(identifiers)


@pytest.mark.gpu_and_cpu_mode
def test_a_frame_with_no_flags_at_all_is_refused(config: Config):
    payload = records([SYN])
    del payload["tcp_flags"]

    with pytest.raises(KeyError, match="tcp_flags"):
        run(config, payload)


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_address_column_is_refused(config: Config):
    meta = MessageMeta(get_df_class(config.execution_mode)({"src_ip": ["10.0.0.5"], "event_time": [START]}))

    with pytest.raises(KeyError, match="src_port"):
        TC4FlowStage(config).on_data(meta)


def test_a_non_positive_bin_is_refused(config: Config):
    with pytest.raises(ValueError, match="bin_seconds"):
        TC4FlowStage(config, bin_seconds=0)
