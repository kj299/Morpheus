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
@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0)])
def test_flows_at_one_instant_do_not_excuse_each_other(config: Config, order):
    # Three flows in one timestamp, two to a network the source has never reached. Neither makes the other look
    # familiar, and the answer is the same whichever order the collector listed them in.
    burst = [("8.8.8.8", "15169"), ("8.8.4.4", "15169"), ("1.1.1.1", "13335")]
    rows = [("9.9.9.9", "19281")] + [burst[index] for index in order]
    payload = flows([row[0] for row in rows], asns=[row[1] for row in rows])
    payload["event_time"] = [START] + [START + SECOND] * 3
    result = run(config, payload)

    novel = dict(zip(result["dst_ip"], result["dst_asn_first_seen"]))

    assert (novel["8.8.8.8"], novel["8.8.4.4"], novel["1.1.1.1"]) == (True, True, True)


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


class _DeviceSeries:
    """What a cuDF series does when iterated: refuse, and point at `to_pandas`."""

    def __init__(self, values: list):
        self._values = values

    def __iter__(self):
        raise TypeError("Series object is not iterable. Consider using `.to_pandas()`")

    def to_pandas(self) -> pd.Series:
        return pd.Series(self._values)


def test_the_destination_classifiers_are_read_on_the_host(monkeypatch):
    # On a GPU the shared IP classifiers return cuDF series, and `list()` of one raises -- which failed every
    # layer 3 test in GPU mode and, through them, the network and campaign harnesses.
    import morpheus.stages.telemetry.tc3_reach_stage as reach

    for name in ("is_private", "is_reserved", "is_multicast"):
        monkeypatch.setattr(reach.ip, name, lambda series: _DeviceSeries([True, False]))

    classified = TC3ReachStage._classify({"dst_ip": None}, "dst_ip")  # pylint: disable=protected-access

    assert all(values == [True, False] for values in classified.values())


def _nullable(values) -> list:
    return [None if pd.isna(value) else bool(value) for value in values]


@pytest.mark.gpu_and_cpu_mode
def test_a_management_destination_is_the_estates_to_name(config: Config):
    # R-C-003's last step. Both destinations are private; only the estate's list says which one is its control plane.
    result = run(config,
                 flows(["10.99.0.1", "10.20.0.5", "10.99.0.254", "8.8.8.8"]),
                 management_prefixes=["10.99.0.0/24", "fd00:99::/64"])

    assert _nullable(result["dst_is_management"]) == [True, False, True, False]


@pytest.mark.gpu_and_cpu_mode
def test_without_a_list_management_is_unclassified_rather_than_false(config: Config):
    result = run(config, flows(["10.99.0.1", "8.8.8.8"]))

    assert _nullable(result["dst_is_management"]) == [None, None]


@pytest.mark.gpu_and_cpu_mode
def test_a_destination_that_is_not_an_address_is_unclassified(config: Config):
    result = run(config, flows(["10.99.0.1", "not-an-address"]), management_prefixes=["10.99.0.0/24"])

    assert _nullable(result["dst_is_management"]) == [True, None]


@pytest.mark.parametrize("prefixes", [["10.99.0.1/24"], ["management"], ["10.99.0.0/33"]])
def test_a_prefix_that_is_not_a_network_is_refused(config: Config, prefixes):
    # Host bits set is a typing error in the estate's list; accepting it would classify a network nobody meant.
    with pytest.raises(ValueError, match="management_prefixes"):
        TC3ReachStage(config, management_prefixes=prefixes)


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
    payload = flows(["10.0.0.9", "10.0.0.10", "10.0.0.11", "10.0.0.12"])
    payload["event_time"] = [START, START + SECOND, START + YEARS_AHEAD_NS, START + 2 * SECOND]

    with_bad = _drive(config, TC3ReachStage(config, min_denominator=1), payload)
    control = _drive(config, TC3ReachStage(config, min_denominator=1), _without_row(payload, 2))

    assert [bool(value) for value in with_bad["clock_refused"]] == [False, False, True, False]
    assert pd.isna(with_bad["internal_dsts_in_window"][2])

    kept = with_bad.drop(index=2).drop(columns=["clock_refused"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, control, check_dtype=False)


@pytest.mark.gpu_and_cpu_mode
def test_a_frame_with_nothing_refused_does_not_gain_the_column(config: Config):
    # The column is written only when a row is refused: the compositions carrying this stage have goldens without it,
    # and a null an upstream guard left is not this stage's to turn into False.
    payload = flows(["10.0.0.9", "10.0.0.10", "10.0.0.11", "10.0.0.12"])

    assert "clock_refused" not in _drive(config, TC3ReachStage(config, min_denominator=1), payload).columns


def test_a_clock_bound_that_admits_nothing_is_refused(config: Config):
    with pytest.raises(ValueError, match="max_clock_skew_seconds"):
        TC3ReachStage(config, max_clock_skew_seconds=0)


@pytest.mark.gpu_and_cpu_mode
def test_a_stop_and_restore_round_trip_keeps_the_clock(config: Config):
    # A fresh instance believes the first time it is shown, so a decade-ahead row that opens a resumed process would
    # set its clock. The clock is part of the snapshot, and the restored instance refuses it as the stopped one would.
    payload = flows(["10.0.0.9", "10.0.0.10", "10.0.0.11", "10.0.0.12"])
    payload["event_time"] = [START, START + SECOND, START + YEARS_AHEAD_NS, START + 2 * SECOND]
    (before, after) = ({
        name: values[:2]
        for (name, values) in payload.items()
    }, {
        name: values[2:]
        for (name, values) in payload.items()
    })

    stopped = TC3ReachStage(config, min_denominator=1)
    _drive(config, stopped, before)

    resumed = TC3ReachStage(config, min_denominator=1)
    resumed.restore_state(stopped.checkpoint_state())

    assert [bool(value) for value in _drive(config, resumed, after)["clock_refused"]] == [True, False]
