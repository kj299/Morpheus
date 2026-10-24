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
from morpheus.stages.telemetry.tc6_cipher_stage import TC6CipherStage
from morpheus.utils.type_utils import get_df_class

SECOND = 10**9
START = 10**18

MODERN_SUITE = "TLS_AES_128_GCM_SHA256"
GOOD_SUITE = "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
LEGACY_SUITE = "TLS_RSA_WITH_AES_128_CBC_SHA"
BROKEN_SUITE = "TLS_RSA_WITH_3DES_EDE_CBC_SHA"


def handshakes(suites, sources=None, destinations=None) -> dict:
    count = len(suites)

    return {
        "src_ip": list(sources) if sources is not None else ["10.0.0.5"] * count,
        "dst_ip": list(destinations) if destinations is not None else ["93.184.216.34"] * count,
        "cipher_suite": list(suites),
        "event_time": [START + index * SECOND for index in range(count)],
    }


def truthy(value) -> bool:
    """A column value as a plain bool, with a null reading as False.

    `bool(pd.NA)` raises rather than returning False, and every flag on these rows is nullable because "not yet
    answerable" is a distinct state from "no". Tests that mean "this did not fire" say so through this.
    """
    return False if pd.isna(value) else bool(value)


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC6CipherStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_a_negotiation_below_the_pair_s_own_floor_is_a_downgrade(config: Config):
    result = run(config, handshakes([GOOD_SUITE] * 6 + [BROKEN_SUITE]), min_samples=3)

    assert list(result["cipher_floor_tier"])[-1] == "forward_aead"
    assert list(result["cipher_tier"])[-1] == "broken"
    assert truthy(list(result["cipher_downgraded"])[-1]) is True


@pytest.mark.gpu_and_cpu_mode
def test_a_pair_negotiating_what_it_always_negotiates_is_quiet(config: Config):
    result = run(config, handshakes([GOOD_SUITE] * 8), min_samples=3)

    assert not any(truthy(value) for value in list(result["cipher_downgraded"])[3:])


@pytest.mark.gpu_and_cpu_mode
def test_the_pair_s_own_variation_is_not_a_downgrade(config: Config):
    # Why the reference is a minimum rather than a mode. A pair that usually negotiates a modern suite and
    # sometimes an older one has a mode of the modern one, and every ordinary older negotiation would sit
    # below it. Only the floor separates routine from attack.
    result = run(config,
                 handshakes([MODERN_SUITE] * 5 + [GOOD_SUITE] + [MODERN_SUITE] * 3 + [GOOD_SUITE]),
                 min_samples=3)

    assert truthy(list(result["cipher_downgraded"])[-1]) is False


@pytest.mark.gpu_and_cpu_mode
def test_an_unrecognized_suite_gets_no_rank_and_no_verdict(config: Config):
    result = run(config, handshakes([GOOD_SUITE] * 5 + ["TLS_NOBODY_SHIPS_THIS"]), min_samples=3)

    assert truthy(list(result["cipher_unrecognized"])[-1]) is True
    assert pd.isna(list(result["cipher_rank"])[-1])
    assert pd.isna(list(result["cipher_downgraded"])[-1])


@pytest.mark.gpu_and_cpu_mode
def test_an_unrecognized_suite_does_not_enter_the_floor(config: Config):
    # The subtler half. If an unranked suite were admitted under any default, the floor would become that
    # default and every later negotiation would be measured against a number nobody chose. The pair's floor
    # here must be the one its recognized handshakes imply.
    result = run(config, handshakes([GOOD_SUITE] * 4 + ["TLS_NOBODY_SHIPS_THIS"] * 3 + [GOOD_SUITE]), min_samples=3)

    assert list(result["cipher_floor_tier"])[-1] == "forward_aead"
    assert truthy(list(result["cipher_downgraded"])[-1]) is False


@pytest.mark.gpu_and_cpu_mode
def test_two_pairs_keep_separate_floors(config: Config):
    # A legacy appliance one host talks to must not lower the floor for a different host's conversations.
    payload = handshakes([LEGACY_SUITE] * 4 + [GOOD_SUITE] * 4, sources=["10.0.0.5"] * 4 + ["10.0.0.6"] * 4)
    result = run(config, payload, min_samples=3)

    assert list(result["cipher_floor_tier"])[3] == "legacy"
    assert list(result["cipher_floor_tier"])[-1] == "forward_aead"


@pytest.mark.gpu_and_cpu_mode
def test_no_floor_is_published_before_the_history_supports_one(config: Config):
    result = run(config, handshakes([GOOD_SUITE] * 5), min_samples=3)

    assert [truthy(value) for value in result["cipher_mature"]] == [False, False, False, True, True]


@pytest.mark.gpu_and_cpu_mode
def test_the_pair_key_is_directed(config: Config):
    # A downgrade is something done to one direction of a conversation, so the two directions are separate
    # entities rather than one.
    result = run(
        config,
        handshakes([GOOD_SUITE] * 2, sources=["10.0.0.5", "93.184.216.34"], destinations=["93.184.216.34", "10.0.0.5"]))

    assert len(set(result["tls_pair_key"])) == 2


@pytest.mark.cpu_mode
def test_a_missing_column_is_refused(config: Config):
    with pytest.raises(KeyError, match="cipher_suite"):
        run(config, {"src_ip": ["10.0.0.5"], "dst_ip": ["10.0.1.5"], "event_time": [START]})


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
    # Row 6 is stamped a decade ahead. Admitted, it would drive every horizon in the stage past the rows before it
    # and refuse the real row after it as out of order; refused, the stream reads exactly as if it had never arrived.
    payload = handshakes([GOOD_SUITE] * 8)
    payload["event_time"] = [START + index * SECOND
                             for index in range(6)] + [START + YEARS_AHEAD_NS, START + 6 * SECOND]

    with_bad = _drive(config, TC6CipherStage(config), payload)
    control = _drive(config, TC6CipherStage(config), _without_row(payload, 6))

    assert [bool(value)
            for value in with_bad["clock_refused"]] == [False, False, False, False, False, False, True, False]
    assert pd.isna(with_bad["cipher_floor"][6])

    kept = with_bad.drop(index=6).drop(columns=["clock_refused"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, control, check_dtype=False)


@pytest.mark.gpu_and_cpu_mode
def test_a_frame_with_nothing_refused_does_not_gain_the_column(config: Config):
    # The column is written only when a row is refused: the compositions carrying this stage have goldens without it,
    # and a null an upstream guard left is not this stage's to turn into False.
    payload = handshakes([GOOD_SUITE] * 8)

    assert "clock_refused" not in _drive(config, TC6CipherStage(config), payload).columns


def test_a_clock_bound_that_admits_nothing_is_refused(config: Config):
    with pytest.raises(ValueError, match="max_clock_skew_seconds"):
        TC6CipherStage(config, max_clock_skew_seconds=0)


@pytest.mark.gpu_and_cpu_mode
def test_a_stop_and_restore_round_trip_keeps_the_clock(config: Config):
    # A fresh instance believes the first time it is shown, so a decade-ahead row that opens a resumed process would
    # set its clock. The clock is part of the snapshot, and the restored instance refuses it as the stopped one would.
    payload = handshakes([GOOD_SUITE] * 8)
    payload["event_time"] = [START + index * SECOND
                             for index in range(6)] + [START + YEARS_AHEAD_NS, START + 6 * SECOND]
    (before, after) = ({
        name: values[:6]
        for (name, values) in payload.items()
    }, {
        name: values[6:]
        for (name, values) in payload.items()
    })

    stopped = TC6CipherStage(config)
    _drive(config, stopped, before)

    resumed = TC6CipherStage(config)
    resumed.restore_state(stopped.checkpoint_state())

    assert [bool(value) for value in _drive(config, resumed, after)["clock_refused"]] == [True, False]
