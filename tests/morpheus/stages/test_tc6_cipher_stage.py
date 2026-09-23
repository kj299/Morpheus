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
