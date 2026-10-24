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
from morpheus.stages.telemetry.tc3_beacon_stage import TC3BeaconStage
from morpheus.utils.type_utils import get_df_class

SECOND = 10**9
START = 10**18
PERIOD = 60 * SECOND
BEACON_THRESHOLD = 0.15
"""The figure R-B-L3-002 names, so the tests evaluate the rule's predicate rather than one near it."""

RAGGED_GAPS = [7, 412, 33, 900, 61, 18, 745, 5, 203, 88, 1300, 12, 470, 29]
"""Gaps in seconds between one person's flows. Ragged at the scale of the gaps themselves, which is what the
coefficient measures and what an implant on a timer is not."""


def flows(count: int,
          destination: str = "198.51.100.7",
          ragged: bool = False,
          sizes=None,
          source: str = "10.0.0.5") -> dict:
    if (ragged):
        times = []
        now = START

        for index in range(count):
            times.append(now)
            now += RAGGED_GAPS[index % len(RAGGED_GAPS)] * SECOND
    else:
        times = [START + index * PERIOD for index in range(count)]

    return {
        "src_ip": [source] * count,
        "dst_ip": [destination] * count,
        "bytes_out": list(sizes) if sizes is not None else [512] * count,
        "event_time": times,
    }


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC3BeaconStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_a_timer_reads_as_a_beacon_and_a_person_does_not(config: Config):
    steady = run(config, flows(15))
    ragged = run(config, flows(15, ragged=True))

    assert bool(list(steady["flow_regularity_mature"])[-1]) is True
    assert list(steady["flow_interval_cv"])[-1] == pytest.approx(0.0)
    assert list(ragged["flow_interval_cv"])[-1] > BEACON_THRESHOLD


@pytest.mark.gpu_and_cpu_mode
def test_the_pair_is_the_entity_and_not_the_source(config: Config):
    # A workstation that beacons to one address and browses to five hundred others has a perfectly regular
    # conversation buried in a wholly irregular stream. Keyed on the source it would be averaged away.
    payload = flows(15)
    noise = flows(15, destination="203.0.113.9", ragged=True)

    for name in payload:
        payload[name] = [value for pair in zip(payload[name], noise[name]) for value in pair]

    result = run(config, payload)
    beaconing = result[result["dst_ip"] == "198.51.100.7"]
    browsing = result[result["dst_ip"] == "203.0.113.9"]

    assert list(beaconing["flow_interval_cv"])[-1] == pytest.approx(0.0)
    assert list(browsing["flow_interval_cv"])[-1] > BEACON_THRESHOLD
    assert set(result["flow_pair_key"]) == {"10.0.0.5:198.51.100.7", "10.0.0.5:203.0.113.9"}


@pytest.mark.gpu_and_cpu_mode
def test_twelve_intervals_are_thirteen_flows(config: Config):
    result = run(config, flows(13))
    intervals = list(result["flow_intervals"])
    mature = [bool(value) for value in result["flow_regularity_mature"]]

    assert intervals[-1] == 12
    assert mature[-2] is False
    assert mature[-1] is True
    assert pd.isna(list(result["flow_interval_cv"])[-2])


@pytest.mark.gpu_and_cpu_mode
def test_size_regularity_is_reported_beside_the_timing(config: Config):
    # An operator triaging an alert needs to see which half of R-B-L3-002 fired.
    steady = run(config, flows(15))
    varied = run(config, flows(15, sizes=[100 * (index + 1) for index in range(15)]))

    assert list(steady["flow_size_cv"])[-1] == pytest.approx(0.0)
    assert list(varied["flow_interval_cv"])[-1] == pytest.approx(0.0)
    assert list(varied["flow_size_cv"])[-1] > BEACON_THRESHOLD


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_address_leaves_the_row_unmeasured(config: Config):
    payload = flows(13)
    payload["dst_ip"][4] = None
    result = run(config, payload)

    assert pd.isna(list(result["flow_pair_key"])[4])
    assert pd.isna(list(result["flow_intervals"])[4])


@pytest.mark.gpu_and_cpu_mode
def test_the_mean_interval_is_reported_in_nanoseconds(config: Config):
    result = run(config, flows(13))

    assert list(result["flow_mean_interval_ns"])[-1] == PERIOD


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_column_is_refused(config: Config):
    meta = MessageMeta(get_df_class(config.execution_mode)({"src_ip": ["10.0.0.5"], "event_time": [START]}))

    with pytest.raises(KeyError, match="dst_ip"):
        TC3BeaconStage(config).on_data(meta)


def test_a_non_positive_window_is_refused(config: Config):
    with pytest.raises(ValueError, match="window_seconds"):
        TC3BeaconStage(config, window_seconds=0)


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
    payload = flows(4)
    payload["event_time"] = [START, START + PERIOD, START + YEARS_AHEAD_NS, START + 2 * PERIOD]

    with_bad = _drive(config, TC3BeaconStage(config), payload)
    control = _drive(config, TC3BeaconStage(config), _without_row(payload, 2))

    assert [bool(value) for value in with_bad["clock_refused"]] == [False, False, True, False]
    assert pd.isna(with_bad["flow_intervals"][2])

    kept = with_bad.drop(index=2).drop(columns=["clock_refused"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, control, check_dtype=False)


@pytest.mark.gpu_and_cpu_mode
def test_a_frame_with_nothing_refused_does_not_gain_the_column(config: Config):
    # The column is written only when a row is refused: the compositions carrying this stage have goldens without it,
    # and a null an upstream guard left is not this stage's to turn into False.
    payload = flows(4)

    assert "clock_refused" not in _drive(config, TC3BeaconStage(config), payload).columns


def test_a_clock_bound_that_admits_nothing_is_refused(config: Config):
    with pytest.raises(ValueError, match="max_clock_skew_seconds"):
        TC3BeaconStage(config, max_clock_skew_seconds=0)


@pytest.mark.gpu_and_cpu_mode
def test_a_stop_and_restore_round_trip_keeps_the_clock(config: Config):
    # A fresh instance believes the first time it is shown, so a decade-ahead row that opens a resumed process would
    # set its clock. The clock is part of the snapshot, and the restored instance refuses it as the stopped one would.
    payload = flows(4)
    payload["event_time"] = [START, START + PERIOD, START + YEARS_AHEAD_NS, START + 2 * PERIOD]
    (before, after) = ({
        name: values[:2]
        for (name, values) in payload.items()
    }, {
        name: values[2:]
        for (name, values) in payload.items()
    })

    stopped = TC3BeaconStage(config)
    _drive(config, stopped, before)

    resumed = TC3BeaconStage(config)
    resumed.restore_state(stopped.checkpoint_state())

    assert [bool(value) for value in _drive(config, resumed, after)["clock_refused"]] == [True, False]
