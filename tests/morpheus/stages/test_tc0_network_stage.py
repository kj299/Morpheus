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
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.stages.telemetry.tc0_network_stage import DEVICE
from morpheus.stages.telemetry.tc0_network_stage import PORT
from morpheus.stages.telemetry.tc0_network_stage import VLAN
from morpheus.stages.telemetry.tc0_network_stage import TC0NetworkStage
from morpheus.utils import bitemporal
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.type_utils import get_df_class

MINUTE = 60 * 10**9


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC0NetworkStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


def inventory(**overrides) -> dict:
    """A port, its switch and a VLAN, then the port recommissioned with a new optic at minute 45."""
    payload = {
        "object_kind": [PORT, DEVICE, VLAN, PORT],
        "site_id": ["hq", "hq", "hq", "hq"],
        "device_id": ["sw1", "sw1", None, "sw1"],
        "port_id": ["Gi1/0/6", None, None, "Gi1/0/6"],
        "vlan_id": [None, None, 10, None],
        "commissioning_rx_dbm": [-7.0, None, None, -6.5],
        "designation": ["access", None, "desks", "access"],
        "device_role": [None, "access-switch", None, None],
        "owner": ["netops", "netops", "facilities", "netops"],
        "owning_team": [None, "network-engineering", None, None],
        "criticality": ["medium", "high", "medium", "medium"],
        "valid_from": [-MINUTE, -MINUTE, -MINUTE, 45 * MINUTE],
        "recorded_at": [-MINUTE, -MINUTE, -MINUTE, 45 * MINUTE],
    }
    payload.update(overrides)

    return payload


@pytest.mark.gpu_and_cpu_mode
def test_each_object_is_keyed_as_the_telemetry_keys_it(config: Config):
    result = run(config, inventory())

    assert list(result[bitemporal.CONTEXT_KIND]) == [PORT, DEVICE, VLAN, PORT]
    assert list(result[bitemporal.CONTEXT_ENTITY]) == ["hq:sw1:Gi1/0/6", "hq:sw1", "hq:10", "hq:sw1:Gi1/0/6"]
    assert result[bitemporal.CONTEXT_UID].notna().all()


@pytest.mark.gpu_and_cpu_mode
def test_a_record_carries_only_its_own_kind_s_attributes(config: Config):
    result = run(config, inventory())

    assert list(result[bitemporal.CONTEXT_ATTRIBUTES]) == [
        "commissioning_rx_dbm,criticality,designation,owner",
        "criticality,device_role,owner,owning_team",
        "criticality,designation,owner",
        "commissioning_rx_dbm,criticality,designation,owner",
    ]


@pytest.mark.gpu_and_cpu_mode
def test_an_unknown_kind_or_a_missing_key_part_is_refused_and_kept(config: Config):
    result = run(config, inventory(object_kind=[PORT, "router", VLAN, PORT], vlan_id=[None, None, None, None]))

    assert list(result[bitemporal.CONTEXT_REFUSED].isna()) == [True, False, False, True]
    assert result[bitemporal.CONTEXT_UID].isna().tolist() == [False, True, True, False]


@pytest.mark.gpu_and_cpu_mode
def test_a_batch_without_a_kind_s_columns_is_refused_only_for_that_kind(config: Config):
    devices_only = {name: values[1:2] for (name, values) in inventory().items()}
    del devices_only["commissioning_rx_dbm"]
    del devices_only["port_id"]

    assert run(config, devices_only)[bitemporal.CONTEXT_UID].notna().all()

    with pytest.raises(KeyError, match="commissioning_rx_dbm"):
        run(config, {**devices_only, "object_kind": [PORT], "port_id": ["Gi1/0/6"]})


@pytest.mark.cpu_mode
def test_the_port_is_recommissioned_from_the_moment_its_optic_was_replaced(config: Config):
    store = BitemporalStore.from_records("network", run(config, inventory()).to_dict("records"))
    probe = pd.DataFrame({"entity_key": ["hq:sw1:Gi1/0/6"] * 2, "event_time": [44 * MINUTE, 46 * MINUTE]})
    meta = MessageMeta(probe)
    TC0EnrichStage(config, store=store, entity_column="entity_key", prefix="ctx_port_").on_data(meta)
    enriched = meta.copy_dataframe()

    assert list(enriched["ctx_port_commissioning_rx_dbm"]) == ["-7", "-6.5"]
    assert list(enriched["ctx_port_designation"]) == ["access", "access"]
    assert enriched["ctx_port_found"].all()


def test_kinds_must_have_both_a_key_and_attributes(config: Config):
    with pytest.raises(ValueError, match="every kind needs both"):
        TC0NetworkStage(config, key_columns={PORT: ["port_id"]}, kind_attributes={DEVICE: ["owner"]})

    with pytest.raises(ValueError, match="at least one column"):
        TC0NetworkStage(config, key_columns={PORT: []}, kind_attributes={PORT: ["owner"]})

    with pytest.raises(ValueError, match="kind_column"):
        TC0NetworkStage(config, kind_column="")
