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
from morpheus.stages.telemetry.tc6_fingerprint_stage import TC6FingerprintStage
from morpheus.utils.type_utils import get_df_class

SECOND = 10**9
START = 10**18

CHROME = "t13d1516h2_8daaf6152771_b186095e22b6"
CURL = "t13d0312h2_55b375c5d22e_cd85d2d88918"
NEW_STACK = "t13d0000h0_000000000000_000000000000"


def handshakes(fingerprints, sources=None) -> dict:
    count = len(fingerprints)

    return {
        "src_ip": list(sources) if sources is not None else ["10.0.0.5"] * count,
        "ja4_client": list(fingerprints),
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
    TC6FingerprintStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_a_new_stack_on_a_known_host_is_first_seen(config: Config):
    result = run(config, handshakes([CHROME] * 5 + [NEW_STACK]))

    assert truthy(list(result["ja4_client_first_seen"])[-1]) is True
    assert list(result["ja4_client_distinct"])[-1] == 2


@pytest.mark.gpu_and_cpu_mode
def test_a_host_using_the_stacks_it_always_uses_is_quiet(config: Config):
    # The negative half. A host alternating between a browser and a command-line client is doing nothing
    # unusual, and a rule reading `changed` rather than `first_seen` would fire on every alternation.
    result = run(config, handshakes([CHROME, CURL] * 4))

    assert not any(truthy(value) for value in list(result["ja4_client_first_seen"])[2:])
    assert list(result["ja4_client_distinct"])[-1] == 2


@pytest.mark.gpu_and_cpu_mode
def test_changed_and_first_seen_are_different_questions(config: Config):
    # Why both columns are on the row. Alternating between two known stacks changes the value every time and
    # is never novel; the rule wants the second question and a search reading the first would be unusable.
    result = run(config, handshakes([CHROME, CURL, CHROME]))

    assert truthy(list(result["ja4_client_changed"])[-1]) is True
    assert truthy(list(result["ja4_client_first_seen"])[-1]) is False


@pytest.mark.gpu_and_cpu_mode
def test_the_first_handshake_a_host_makes_answers_neither(config: Config):
    # There is no previous sample to differ from, and the question is not answerable rather than answered "no".
    result = run(config, handshakes([CHROME]))

    assert pd.isna(list(result["ja4_client_first_seen"])[0])
    assert pd.isna(list(result["ja4_client_changed"])[0])


@pytest.mark.gpu_and_cpu_mode
def test_two_hosts_keep_separate_histories(config: Config):
    # The fingerprint a second host has never presented is novel to it even though the estate has seen it.
    result = run(config, handshakes([CHROME, CHROME, CURL], sources=["10.0.0.5", "10.0.0.6", "10.0.0.6"]))

    assert list(result["ja4_client_distinct"])[1] == 1


@pytest.mark.gpu_and_cpu_mode
def test_a_handshake_without_a_fingerprint_teaches_the_history_nothing(config: Config):
    # A null is not a stack. Learning it would make the next real fingerprint read as a change from nothing.
    result = run(config, handshakes([CHROME, None, CHROME]))

    assert pd.isna(list(result["ja4_client_first_seen"])[1])
    assert truthy(list(result["ja4_client_first_seen"])[2]) is False
    assert list(result["ja4_client_distinct"])[2] == 1


@pytest.mark.gpu_and_cpu_mode
def test_the_prior_handshake_count_is_the_history_behind_the_row(config: Config):
    # Read before this handshake is counted, so it describes what the host had done before rather than a figure
    # this row has already joined.
    result = run(config, handshakes([CHROME] * 4))

    assert list(result["ja4_client_observations"]) == [0, 1, 2, 3]


@pytest.mark.gpu_and_cpu_mode
def test_a_host_the_estate_has_just_started_seeing_is_novel_in_every_direction(config: Config):
    # What the corpus caught. Every fingerprint is new to a host that has just appeared, so novelty alone fires
    # on a laptop back from repair, a new starter, anything behind a fresh lease. The count is what a rule uses
    # to tell that from a host whose stack has been stable and then changed, and the two are indistinguishable
    # without it -- both rows below say `first_seen`.
    fresh = run(config, handshakes([CHROME, CURL]))
    settled = run(config, handshakes([CHROME] * 20 + [NEW_STACK]))

    assert truthy(list(fresh["ja4_client_first_seen"])[-1]) is True
    assert truthy(list(settled["ja4_client_first_seen"])[-1]) is True

    assert list(fresh["ja4_client_observations"])[-1] == 1
    assert list(settled["ja4_client_observations"])[-1] == 20


@pytest.mark.gpu_and_cpu_mode
def test_the_count_is_per_host(config: Config):
    result = run(config, handshakes([CHROME] * 4, sources=["10.0.0.5", "10.0.0.6", "10.0.0.5", "10.0.0.6"]))

    assert list(result["ja4_client_observations"]) == [0, 0, 1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_an_unusable_handshake_is_not_counted(config: Config):
    # A record with no fingerprint told the history nothing, so counting it would inflate the settledness of a
    # host on the strength of records that taught nothing.
    result = run(config, handshakes([CHROME, None, CHROME]))

    assert list(result["ja4_client_observations"])[2] == 1


@pytest.mark.gpu_and_cpu_mode
def test_the_host_key_is_composed_from_the_columns_it_is_told(config: Config):
    result = run(config, handshakes([CHROME] * 2))

    assert set(result["tls_client_key"]) == {"10.0.0.5"}


@pytest.mark.gpu_and_cpu_mode
def test_a_saturated_history_says_so(config: Config):
    # A host that has genuinely presented more stacks than the cap is already the anomaly the feature is for,
    # and the flag is what stops a search reading the capped count as the whole truth.
    result = run(config, handshakes([f"stack-{index}" for index in range(6)]), max_values=3)

    assert truthy(list(result["ja4_client_saturated"])[-1]) is True


@pytest.mark.cpu_mode
def test_a_missing_column_is_refused(config: Config):
    with pytest.raises(KeyError, match="ja4_client"):
        run(config, {"src_ip": ["10.0.0.5"], "event_time": [START]})


@pytest.mark.cpu_mode
def test_an_empty_key_list_is_refused(config: Config):
    with pytest.raises(ValueError, match="key_columns"):
        TC6FingerprintStage(config, key_columns=[])
