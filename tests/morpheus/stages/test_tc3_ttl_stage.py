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
from morpheus.stages.telemetry.tc3_ttl_stage import TC3TtlStage
from morpheus.utils.type_utils import get_df_class

SECOND = 10**9
START = 10**18


def flows(ttls, sources=None) -> dict:
    count = len(ttls)

    return {
        "src_ip": list(sources) if sources is not None else ["10.0.0.5"] * count,
        "ip_ttl": list(ttls),
        "event_time": [START + index * SECOND for index in range(count)],
    }


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC3TtlStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_an_interposed_device_shows_as_one_hop_lost(config: Config):
    # Something that forwards a packet decrements the field exactly once, which is why one hop counts.
    result = run(config, flows([64] * 8 + [63]))

    assert list(result["ip_ttl_established"])[-1] == 64
    assert list(result["ip_ttl_shift"])[-1] == -1
    assert bool(list(result["ip_ttl_shifted"])[-1]) is True


@pytest.mark.gpu_and_cpu_mode
def test_a_steady_source_reports_no_shift(config: Config):
    result = run(config, flows([64] * 9))

    assert list(result["ip_ttl_shift"])[-1] == 0
    assert bool(list(result["ip_ttl_shifted"])[-1]) is False
    assert list(result["ip_ttl_distinct"])[-1] == 1


@pytest.mark.gpu_and_cpu_mode
def test_two_hosts_behind_one_address_are_reported_rather_than_averaged(config: Config):
    # A mean over a bimodal source lands between 128 and 64 and describes neither.
    result = run(config, flows([64, 64, 64, 64, 64, 64, 128, 128, 64]))

    assert list(result["ip_ttl_established"])[-1] == 64
    assert list(result["ip_ttl_distinct"])[-1] == 2


@pytest.mark.gpu_and_cpu_mode
def test_no_reference_is_published_before_there_is_one(config: Config):
    result = run(config, flows([64] * 3), min_samples=5)

    assert result["ip_ttl_established"].isna().all()
    assert result["ip_ttl_shift"].isna().all()
    assert [bool(value) for value in result["ip_ttl_mature"]] == [False] * 3


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_ttl_is_left_alone_rather_than_defaulted(config: Config):
    # Cloud flow logs mostly omit the field, and substituting 64 would build a reference out of an assumption
    # and then report every real packet as a shift away from it.
    result = run(config, flows([64, 64, None, 64, 64, 64, 63]))

    assert pd.isna(list(result["ip_ttl_established"])[2])
    assert list(result["ip_ttl_shift"])[-1] == -1


@pytest.mark.gpu_and_cpu_mode
def test_a_value_outside_the_field_is_not_profiled(config: Config):
    result = run(config, flows([64, 64, 300, 64, 64, 64, 64]))

    assert pd.isna(list(result["ip_ttl_shift"])[2])
    assert list(result["ip_ttl_established"])[-1] == 64


@pytest.mark.gpu_and_cpu_mode
def test_two_sources_keep_separate_references(config: Config):
    sources = ["10.0.0.5"] * 6 + ["10.0.0.6"] * 6
    result = run(config, flows([64] * 6 + [128] * 6, sources=sources), min_samples=3)

    assert list(result["ip_ttl_established"])[5] == 64
    assert list(result["ip_ttl_established"])[-1] == 128


@pytest.mark.gpu_and_cpu_mode
def test_the_profile_can_be_kept_per_collector_as_well(config: Config):
    # Exporters disagree about which packet of a flow the TTL comes from, so an estate mixing them keys the
    # profile on the collector too rather than reading the disagreement as an interposition.
    payload = flows([64] * 6 + [63] * 6)
    payload["collector_id"] = ["a"] * 6 + ["b"] * 6

    result = run(config, payload, key_columns=["src_ip", "collector_id"], min_samples=3)

    assert bool(list(result["ip_ttl_shifted"])[-1]) is False


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_column_is_refused(config: Config):
    meta = MessageMeta(get_df_class(config.execution_mode)({"src_ip": ["10.0.0.5"], "event_time": [START]}))

    with pytest.raises(KeyError, match="ip_ttl"):
        TC3TtlStage(config).on_data(meta)


def test_an_empty_key_is_refused(config: Config):
    with pytest.raises(ValueError, match="key_columns"):
        TC3TtlStage(config, key_columns=[])


# --- The clock guard ------------------------------------------------------------------------------------------

YEARS_AHEAD_NS = 10 * 365 * 86400 * 10**9
"""A collector clock wrong by a decade: the failure an unset RTC produces, far beyond any gap a real feed has."""


def _without_row(payload: dict, position: int) -> dict:
    """The same frame with one row left out, which is what a refused row must be indistinguishable from."""
    return {
        name: [value for (index, value) in enumerate(values) if index != position]
        for (name, values) in payload.items()
    }


def _pandas(meta: MessageMeta) -> pd.DataFrame:
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _drive(config: Config, stage, payload: dict) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    stage.on_data(meta)

    return _pandas(meta)


@pytest.mark.gpu_and_cpu_mode
def test_a_time_years_ahead_of_the_stream_is_refused_and_moves_nothing(config: Config):
    # Row 2 is stamped a decade ahead. Admitted, it would drive every horizon in the stage past the rows before it
    # and refuse the real row after it as out of order; refused, the stream reads exactly as if it had never arrived.
    payload = flows([64, 64, 64, 64])
    payload["event_time"] = [START, START + SECOND, START + YEARS_AHEAD_NS, START + 2 * SECOND]

    with_bad = _drive(config, TC3TtlStage(config), payload)
    control = _drive(config, TC3TtlStage(config), _without_row(payload, 2))

    assert [bool(value) for value in with_bad["clock_refused"]] == [False, False, True, False]
    assert pd.isna(with_bad["ip_ttl_distinct"][2])

    kept = with_bad.drop(index=2).drop(columns=["clock_refused"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, control, check_dtype=False)


@pytest.mark.gpu_and_cpu_mode
def test_a_frame_with_nothing_refused_does_not_gain_the_column(config: Config):
    # The column is written only when a row is refused: the compositions carrying this stage have goldens without it,
    # and a null an upstream guard left is not this stage's to turn into False.
    payload = flows([64, 64, 64, 64])

    assert "clock_refused" not in _drive(config, TC3TtlStage(config), payload).columns


def test_a_clock_bound_that_admits_nothing_is_refused(config: Config):
    with pytest.raises(ValueError, match="max_clock_skew_seconds"):
        TC3TtlStage(config, max_clock_skew_seconds=0)


@pytest.mark.gpu_and_cpu_mode
def test_a_stop_and_restore_round_trip_keeps_the_clock(config: Config):
    # A fresh instance believes the first time it is shown, so a decade-ahead row that opens a resumed process would
    # set its clock. The clock is part of the snapshot, and the restored instance refuses it as the stopped one would.
    payload = flows([64, 64, 64, 64])
    payload["event_time"] = [START, START + SECOND, START + YEARS_AHEAD_NS, START + 2 * SECOND]
    (before, after) = ({
        name: values[:2]
        for (name, values) in payload.items()
    }, {
        name: values[2:]
        for (name, values) in payload.items()
    })

    stopped = TC3TtlStage(config)
    _drive(config, stopped, before)

    resumed = TC3TtlStage(config)
    resumed.restore_state(stopped.checkpoint_state())

    assert [bool(value) for value in _drive(config, resumed, after)["clock_refused"]] == [True, False]
