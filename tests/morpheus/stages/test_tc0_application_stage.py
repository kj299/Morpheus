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
from morpheus.stages.telemetry.tc0_application_stage import APPLICATION
from morpheus.stages.telemetry.tc0_application_stage import TC0ApplicationStage
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.utils import bitemporal
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.type_utils import get_df_class

DAY = 86400 * 10**9


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC0ApplicationStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


def mapping(**overrides) -> dict:
    payload = {
        "server_address": ["10.0.2.50", "10.0.2.50", "10.0.2.51"],
        "server_port": [8443, 8443, 443],
        "name": ["billing", "reporting", "ledger-api"],
        "owning_team": ["finance-apps", "analytics", "finance-apps"],
        "service_account": ["svc-billing", "svc-reports", "svc-ledger"],
        "criticality": ["high", "medium", "high"],
        "data_classification": ["confidential", "internal", "restricted"],
        "valid_from": [0, 9 * DAY, 0],
        "recorded_at": [0, 10 * DAY, 0],
    }
    payload.update(overrides)

    return payload


@pytest.mark.gpu_and_cpu_mode
def test_every_record_is_an_application_version_keyed_on_its_server_s_port(config: Config):
    result = run(config, mapping())

    assert set(result[bitemporal.CONTEXT_KIND]) == {APPLICATION}
    assert list(result[bitemporal.CONTEXT_ENTITY]) == ["10.0.2.50:8443", "10.0.2.50:8443", "10.0.2.51:443"]
    assert result[bitemporal.CONTEXT_UID].notna().all()


@pytest.mark.cpu_mode
def test_a_reassigned_server_answers_both_ways_once_it_is_rebuilt_and_joined(config: Config):
    store = BitemporalStore.from_records("application", run(config, mapping()).to_dict("records"))
    probe = pd.DataFrame({"dst_ip": ["10.0.2.50"], "dst_port": [8443], "event_time": [9 * DAY + 3600 * 10**9]})
    answers = {}

    for knowledge in ("event", "latest"):
        meta = MessageMeta(probe.copy())
        TC0EnrichStage(config,
                       store=store,
                       entity_columns=["dst_ip", "dst_port"],
                       prefix="ctx_application_",
                       knowledge=knowledge).on_data(meta)
        answers[knowledge] = meta.copy_dataframe()["ctx_application_name"].iloc[0]

    assert answers == {"event": "billing", "latest": "reporting"}


@pytest.mark.cpu_mode
def test_a_record_without_a_recording_instant_is_refused_and_kept(config: Config):
    result = run(config, mapping(recorded_at=[0, None, 0]))

    assert result[bitemporal.CONTEXT_UID].isna().sum() == 1


@pytest.mark.cpu_mode
def test_a_record_without_a_port_names_no_server(config: Config):
    result = run(config, mapping(server_port=[8443, None, 443]))

    assert result[bitemporal.CONTEXT_UID].isna().sum() == 1


@pytest.mark.cpu_mode
def test_a_missing_column_names_itself(config: Config):
    payload = mapping()
    del payload["service_account"]

    with pytest.raises(KeyError, match="service_account"):
        run(config, payload)


@pytest.mark.cpu_mode
def test_constructor_rejects_bad_configuration(config: Config):
    with pytest.raises(ValueError, match="address_column"):
        TC0ApplicationStage(config, address_column="")

    with pytest.raises(ValueError, match="application_columns"):
        TC0ApplicationStage(config, application_columns=[])
