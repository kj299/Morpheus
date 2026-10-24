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
from morpheus.stages.telemetry.tc7_service_stage import TC7ServiceStage
from morpheus.utils.type_utils import get_df_class

SECOND = 10**9
DAY = 86400 * SECOND
START = 10**18


def truthy(value) -> bool:
    return False if pd.isna(value) else bool(value)


def run(config: Config, payload: dict, stage: TC7ServiceStage = None, **kwargs) -> pd.DataFrame:
    stage = stage if stage is not None else TC7ServiceStage(config, **kwargs)
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    stage.on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


def connections(clients, names=None, fingerprints=None, times=None, address="10.0.2.10", port=443, sizes=None) -> dict:
    count = len(clients)

    return {
        "src_ip": list(clients),
        "dst_ip": [address] * count,
        "dst_port": [port] * count,
        "sni": list(names) if names is not None else ["api.corp.example"] * count,
        "bytes_total": list(sizes) if sizes is not None else [100] * count,
        "ja4_client": list(fingerprints) if fingerprints is not None else ["fp-browser"] * count,
        "event_time": list(times) if times is not None else [START + index * SECOND for index in range(count)],
    }


@pytest.mark.gpu_and_cpu_mode
def test_a_service_is_its_name_where_there_is_one_and_its_address_where_not(config: Config):
    result = run(config, connections(["a", "b", "c"], names=["API.corp.example", "wiki.corp.example", None]))

    assert list(result["service_key"]) == ["api.corp.example:443", "wiki.corp.example:443", "10.0.2.10:443"]
    assert list(result["service_key_source"]) == ["sni", "sni", "address"]


@pytest.mark.gpu_and_cpu_mode
def test_the_window_counts_connections_bytes_and_distinct_clients(config: Config):
    result = run(config, connections(["a", "b", "a", "c"], sizes=[10, 20, 30, 40]))

    assert list(result["service_connections_in_window"]) == [1, 2, 3, 4]
    assert list(result["service_bytes_in_window"]) == [10.0, 30.0, 60.0, 100.0]
    assert list(result["service_clients_in_window"]) == [1, 2, 2, 3]


@pytest.mark.gpu_and_cpu_mode
def test_an_old_connection_leaves_the_window(config: Config):
    times = [START, START + 10 * SECOND, START + 2 * 3600 * SECOND]
    result = run(config, connections(["a", "b", "c"], times=times))

    assert list(result["service_connections_in_window"]) == [1, 2, 1]
    assert list(result["service_clients_in_window"]) == [1, 2, 1]


@pytest.mark.gpu_and_cpu_mode
def test_client_software_is_novel_to_the_service_and_not_to_the_client(config: Config):
    fingerprints = ["fp-browser", "fp-browser", "fp-script", "fp-script"]
    result = run(config, connections(["a", "b", "c", "d"], fingerprints=fingerprints))

    assert pd.isna(result["service_client_fingerprint_first_seen"].iloc[0])
    assert [truthy(value) for value in result["service_client_fingerprint_first_seen"]][1:] == [False, True, False]
    assert list(result["service_client_fingerprints"]) == [1, 1, 2, 2]


@pytest.mark.gpu_and_cpu_mode
def test_the_clients_are_measured_against_the_service_s_busiest_day(config: Config):
    clients = []
    times = []

    for day in range(3):
        for index in range(2):
            clients.append(f"c{index}")
            times.append(START + day * DAY + index * SECOND)

    for index in range(6):
        clients.append(f"new{index}")
        times.append(START + 3 * DAY + index * SECOND)

    result = run(config, connections(clients, times=times), baseline_min_days=3)
    today = result.iloc[-1]

    assert not truthy(result["service_clients_baseline_mature"].iloc[5])
    assert truthy(today["service_clients_baseline_mature"])
    assert today["service_clients_baseline"] == 2
    assert today["service_clients_ratio"] == pytest.approx(3.0)


@pytest.mark.gpu_and_cpu_mode
def test_state_carries_across_batches(config: Config):
    stage = TC7ServiceStage(config)
    run(config, connections(["a", "b"]), stage)
    later = run(config, connections(["c"], times=[START + 5 * SECOND]), stage)

    assert list(later["service_clients_in_window"]) == [3]


@pytest.mark.gpu_and_cpu_mode
def test_a_connection_without_a_port_reaches_no_service(config: Config):
    payload = connections(["a"])
    payload["dst_port"] = [None]
    result = run(config, payload)

    assert pd.isna(result["service_key"].iloc[0])
    assert pd.isna(result["service_clients_in_window"].iloc[0])


@pytest.mark.cpu_mode
def test_optional_columns_may_be_absent(config: Config):
    payload = connections(["a", "b"])

    for column in ("sni", "bytes_total", "ja4_client"):
        del payload[column]

    result = run(config, payload)

    assert list(result["service_key"]) == ["10.0.2.10:443", "10.0.2.10:443"]
    assert result["service_client_fingerprint_first_seen"].isna().all()
    assert list(result["service_bytes_in_window"]) == [0.0, 0.0]


@pytest.mark.cpu_mode
def test_a_missing_column_names_itself(config: Config):
    payload = connections(["a"])
    del payload["dst_ip"]

    with pytest.raises(KeyError, match="dst_ip"):
        run(config, payload)


@pytest.mark.cpu_mode
def test_constructor_rejects_bad_configuration(config: Config):
    with pytest.raises(ValueError, match="window_seconds"):
        TC7ServiceStage(config, window_seconds=0)

    with pytest.raises(ValueError, match="max_samples"):
        TC7ServiceStage(config, max_samples=0)

    with pytest.raises(ValueError, match="max_entities"):
        TC7ServiceStage(config, max_entities=0)


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
    payload = connections(["a", "b", "c", "d"])
    payload["event_time"] = [START, START + SECOND, START + YEARS_AHEAD_NS, START + 2 * SECOND]

    with_bad = _drive(config, TC7ServiceStage(config), payload)
    control = _drive(config, TC7ServiceStage(config), _without_row(payload, 2))

    assert [bool(value) for value in with_bad["clock_refused"]] == [False, False, True, False]
    assert pd.isna(with_bad["service_connections_in_window"][2])

    kept = with_bad.drop(index=2).drop(columns=["clock_refused"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, control, check_dtype=False)


@pytest.mark.gpu_and_cpu_mode
def test_a_frame_with_nothing_refused_does_not_gain_the_column(config: Config):
    # The column is written only when a row is refused: the compositions carrying this stage have goldens without it,
    # and a null an upstream guard left is not this stage's to turn into False.
    payload = connections(["a", "b", "c", "d"])

    assert "clock_refused" not in _drive(config, TC7ServiceStage(config), payload).columns


def test_a_clock_bound_that_admits_nothing_is_refused(config: Config):
    with pytest.raises(ValueError, match="max_clock_skew_seconds"):
        TC7ServiceStage(config, max_clock_skew_seconds=0)


@pytest.mark.gpu_and_cpu_mode
def test_a_stop_and_restore_round_trip_keeps_the_clock(config: Config):
    # A fresh instance believes the first time it is shown, so a decade-ahead row that opens a resumed process would
    # set its clock. The clock is part of the snapshot, and the restored instance refuses it as the stopped one would.
    payload = connections(["a", "b", "c", "d"])
    payload["event_time"] = [START, START + SECOND, START + YEARS_AHEAD_NS, START + 2 * SECOND]
    (before, after) = ({
        name: values[:2]
        for (name, values) in payload.items()
    }, {
        name: values[2:]
        for (name, values) in payload.items()
    })

    stopped = TC7ServiceStage(config)
    _drive(config, stopped, before)

    resumed = TC7ServiceStage(config)
    resumed.restore_state(stopped.checkpoint_state())

    assert [bool(value) for value in _drive(config, resumed, after)["clock_refused"]] == [True, False]
