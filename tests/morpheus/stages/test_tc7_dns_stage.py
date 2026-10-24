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


def queries(names, times=None, sources=None, responses=None) -> dict:
    count = len(names)
    payload = {
        "src_ip": list(sources) if sources is not None else ["10.0.0.5"] * count,
        "query_name": list(names),
        "event_time": list(times) if times is not None else [START + index * SECOND for index in range(count)],
    }

    if (responses is not None):
        payload["response_code"] = list(responses)

    return payload


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


DAY = 86400 * SECOND


@pytest.mark.gpu_and_cpu_mode
def test_a_client_counts_the_registered_domains_it_asked_for(config: Config):
    names = ["a.one.com", "b.one.com", "a.two.com", "a.three.org"]
    result = run(config, queries(names, sources=["10.0.0.5", "10.0.0.5", "10.0.0.5", "10.0.0.6"]))

    assert list(result["dns_client_key"]) == ["10.0.0.5", "10.0.0.5", "10.0.0.5", "10.0.0.6"]
    assert list(result["dns_domains_per_client"]) == [1, 1, 2, 1]


@pytest.mark.gpu_and_cpu_mode
def test_the_nxdomain_share_waits_for_enough_answers(config: Config):
    names = [f"x{index}.example.com" for index in range(4)]
    result = run(config, queries(names, responses=["NXDOMAIN", "NOERROR", "nxdomain", None]), min_queries=3)

    assert list(result["dns_nxdomain_in_window"])[:3] == [1, 1, 2]
    assert list(result["dns_answered_in_window"])[:3] == [1, 2, 3]
    assert pd.isna(result["dns_nxdomain_ratio"].iloc[1])
    assert result["dns_nxdomain_ratio"].iloc[2] == pytest.approx(2 / 3)
    # A query the record has no answer for is neither a refusal nor an answer.
    assert pd.isna(result["dns_answered_in_window"].iloc[3])


@pytest.mark.cpu_mode
def test_without_a_response_column_the_share_is_null(config: Config):
    result = run(config, queries(["a.example.com", "b.example.com"]))

    assert result["dns_nxdomain_ratio"].isna().all()
    assert result["dns_answered_in_window"].isna().all()


@pytest.mark.gpu_and_cpu_mode
def test_a_client_is_measured_against_its_own_busiest_day(config: Config):
    # Two subdomains a day for three days, then eight in a day: four times the busiest day the client has had.
    names = []
    times = []

    for day in range(3):
        for index in range(2):
            names.append(f"d{day}n{index}.example.com")
            times.append(START + day * DAY + index * SECOND)

    for index in range(8):
        names.append(f"today{index}.example.com")
        times.append(START + 3 * DAY + index * SECOND)

    result = run(config, queries(names, times=times), baseline_min_days=3)
    today = result.iloc[-1]

    assert not truthy(result["dns_subdomains_per_domain_baseline_mature"].iloc[5])
    assert truthy(today["dns_subdomains_per_domain_baseline_mature"])
    assert today["dns_subdomains_per_domain_baseline"] == 2
    assert today["dns_subdomains_per_domain_ratio"] == pytest.approx(4.0)
    assert today["dns_domains_per_client_baseline"] == 1


@pytest.mark.cpu_mode
def test_a_missing_client_column_is_refused(config: Config):
    payload = queries(["a.example.com"])
    del payload["src_ip"]

    with pytest.raises(KeyError, match="src_ip"):
        run(config, payload)


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
    payload = queries([f"c{index}.evil.example" for index in range(4)])
    payload["event_time"] = [START, START + SECOND, START + YEARS_AHEAD_NS, START + 2 * SECOND]

    with_bad = _drive(config, TC7DnsStage(config), payload)
    control = _drive(config, TC7DnsStage(config), _without_row(payload, 2))

    assert [bool(value) for value in with_bad["clock_refused"]] == [False, False, True, False]
    assert pd.isna(with_bad["dns_subdomains_per_domain"][2])

    kept = with_bad.drop(index=2).drop(columns=["clock_refused"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, control, check_dtype=False)


@pytest.mark.gpu_and_cpu_mode
def test_a_frame_with_nothing_refused_does_not_gain_the_column(config: Config):
    # The column is written only when a row is refused: the compositions carrying this stage have goldens without it,
    # and a null an upstream guard left is not this stage's to turn into False.
    payload = queries([f"c{index}.evil.example" for index in range(4)])

    assert "clock_refused" not in _drive(config, TC7DnsStage(config), payload).columns


def test_a_clock_bound_that_admits_nothing_is_refused(config: Config):
    with pytest.raises(ValueError, match="max_clock_skew_seconds"):
        TC7DnsStage(config, max_clock_skew_seconds=0)


@pytest.mark.gpu_and_cpu_mode
def test_a_stop_and_restore_round_trip_keeps_the_clock(config: Config):
    # A fresh instance believes the first time it is shown, so a decade-ahead row that opens a resumed process would
    # set its clock. The clock is part of the snapshot, and the restored instance refuses it as the stopped one would.
    payload = queries([f"c{index}.evil.example" for index in range(4)])
    payload["event_time"] = [START, START + SECOND, START + YEARS_AHEAD_NS, START + 2 * SECOND]
    (before, after) = ({
        name: values[:2]
        for (name, values) in payload.items()
    }, {
        name: values[2:]
        for (name, values) in payload.items()
    })

    stopped = TC7DnsStage(config)
    _drive(config, stopped, before)

    resumed = TC7DnsStage(config)
    resumed.restore_state(stopped.checkpoint_state())

    assert [bool(value) for value in _drive(config, resumed, after)["clock_refused"]] == [True, False]
