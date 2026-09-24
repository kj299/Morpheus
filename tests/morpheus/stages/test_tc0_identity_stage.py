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
from morpheus.stages.telemetry.tc0_identity_stage import PROFILE
from morpheus.stages.telemetry.tc0_identity_stage import TC0IdentityStage
from morpheus.utils import bitemporal
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.type_utils import get_df_class

DAY = 86400 * 10**9


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC0IdentityStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


def records(**overrides) -> dict:
    payload = {
        "user_principal": ["alice", "alice", "bob"],
        "group_name": [None, "finance-users", None],
        "department": ["Finance", None, "Engineering"],
        "manager": ["frank", None, "grace"],
        "employment_status": ["active", None, "active"],
        "valid_from": [0, 0, 0],
        "valid_to": [None, None, 5 * DAY],
        "recorded_at": [DAY, DAY, DAY],
        "change": ["assert", "assert", "assert"],
    }
    payload.update(overrides)

    return payload


@pytest.mark.gpu_and_cpu_mode
def test_a_row_naming_a_group_is_a_membership_and_any_other_is_a_profile(config: Config):
    result = run(config, records())

    assert list(result[bitemporal.CONTEXT_KIND]) == [PROFILE, bitemporal.MEMBERSHIP, PROFILE]
    assert list(result[bitemporal.CONTEXT_KEY]) == ["alice", "alice:finance-users", "bob"]
    assert list(result[bitemporal.CONTEXT_ENTITY]) == ["alice", "alice", "bob"]
    assert list(result[bitemporal.CONTEXT_ATTRIBUTES]) == [
        "department,employment_status,manager", "group_name", "department,employment_status,manager"
    ]
    assert result[bitemporal.CONTEXT_REFUSED].isna().all()
    assert result[bitemporal.CONTEXT_UID].notna().all()


@pytest.mark.gpu_and_cpu_mode
def test_the_output_rebuilds_a_store_that_answers_as_the_records_said(config: Config):
    store = BitemporalStore.from_records("t", run(config, records()).to_dict("records"))

    assert store.resolve("alice", DAY).attributes["department"] == "Finance"
    assert store.resolve("alice:finance-users", DAY) is not None
    assert store.resolve("bob", 4 * DAY).attributes["manager"] == "grace"
    assert store.resolve("bob", 5 * DAY) is None


@pytest.mark.gpu_and_cpu_mode
def test_a_row_without_its_recorded_instant_is_refused_not_stamped(config: Config):
    # Stamping the arrival time would make every as-known-at answer depend on when the pipeline ran.
    result = run(config, records(recorded_at=[DAY, None, DAY]))

    assert list(result[bitemporal.CONTEXT_REFUSED].fillna("")) == ["", bitemporal.NO_RECORDED_AT, ""]
    assert pd.isna(result[bitemporal.CONTEXT_UID][1])
    assert pd.isna(result[bitemporal.RECORDED_AT][1])


@pytest.mark.gpu_and_cpu_mode
def test_an_interval_ending_before_it_begins_is_refused(config: Config):
    result = run(config, records(valid_to=[None, None, 0]))

    assert list(result[bitemporal.CONTEXT_REFUSED].fillna("")) == ["", "", bitemporal.INVERTED_INTERVAL]


@pytest.mark.gpu_and_cpu_mode
def test_string_instants_are_accepted(config: Config):
    result = run(
        config,
        records(valid_from=["2026-01-01T00:00:00Z"] * 3,
                valid_to=[None] * 3,
                recorded_at=["2026-01-02T00:00:00+00:00"] * 3))

    assert list(result[bitemporal.VALID_FROM]) == [pd.Timestamp("2026-01-01").value] * 3
    assert list(result[bitemporal.RECORDED_AT]) == [pd.Timestamp("2026-01-02").value] * 3


@pytest.mark.gpu_and_cpu_mode
def test_optional_columns_may_be_absent(config: Config):
    payload = records()

    for name in ("group_name", "valid_to", "change"):
        del payload[name]

    result = run(config, payload)

    assert set(result[bitemporal.CONTEXT_KIND]) == {PROFILE}
    assert set(result[bitemporal.CHANGE]) == {bitemporal.ASSERT}
    assert result[bitemporal.VALID_TO].isna().all()


@pytest.mark.gpu_and_cpu_mode
def test_a_retraction_is_carried_through(config: Config):
    result = run(config, records(change=["assert", "RETRACT", "assert"]))

    assert list(result[bitemporal.CHANGE]) == ["assert", "retract", "assert"]


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_required_column_names_itself(config: Config):
    payload = records()
    del payload["recorded_at"]

    with pytest.raises(KeyError, match="recorded_at"):
        run(config, payload)


def test_constructor_rejects_empty_configuration(config: Config):
    with pytest.raises(ValueError):
        TC0IdentityStage(config, principal_column="")

    with pytest.raises(ValueError):
        TC0IdentityStage(config, profile_columns=[])
