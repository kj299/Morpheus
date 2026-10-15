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
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.stages.lineage.edge_emit_stage import CARRIED_BY
from morpheus.stages.lineage.edge_emit_stage import EDGE_COLUMNS
from morpheus.stages.lineage.edge_emit_stage import RESOLVED_THROUGH
from morpheus.stages.lineage.edge_emit_stage import EdgeEmitStage
from morpheus.utils.lineage import link_uid
from morpheus.utils.type_utils import get_df_class

HOPS = {"hop_method_columns": ["host_lease_resolution", "host_inventory_resolution"],
        "hop_uid_columns": ["host_lease_uid", "host_inventory_uid"],
        "hop_layers": [3, 7]}


def resolved_flows() -> dict:
    """Two flows: one whose address resolved through both hops, one no lease names."""
    return {
        "event_uid": ["flow-1", "flow-2"],
        "osi_layer": [3, 3],
        "lineage_id": ["chain-a", "chain-b"],
        "event_time": [100, 200],
        "community_id": ["1:abc", None],
        "host_lease_resolution": ["soft:dhcp_lease", "unresolved"],
        "host_lease_uid": ["lease-9", None],
        "host_inventory_resolution": ["soft:host_inventory", "unresolved"],
        "host_inventory_uid": ["report-4", None],
    }


def connection() -> dict:
    """One connection seen by a flow exporter, a packet sensor twice and an inspection point, and one seen once."""
    return {
        "event_uid": ["flow", "packet-1", "packet-2", "handshake", "lonely"],
        "osi_layer": [3, 4, 4, 6, 4],
        "lineage_id": ["l3", "l4", "l4", "l6", "l4b"],
        "event_time": [10, 11, 12, 13, 14],
        "community_id": ["1:abc", "1:abc", "1:abc", "1:abc", "1:zzz"],
    }


def emit(config: Config, payload: dict, stage: EdgeEmitStage = None, **kwargs) -> pd.DataFrame:
    stage = stage if stage is not None else EdgeEmitStage(config, **kwargs)
    result = stage.on_data(MessageMeta(get_df_class(config.execution_mode)(payload)))
    df = result.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_each_resolved_hop_is_an_edge_from_its_binding(config: Config):
    edges = emit(config, resolved_flows(), **HOPS)

    assert list(edges.columns) == list(EDGE_COLUMNS)
    assert list(edges["parent_event_uid"]) == ["lease-9", "report-4"]
    assert set(edges["child_event_uid"]) == {"flow-1"}
    assert set(edges["relation_type"]) == {RESOLVED_THROUGH}
    assert list(edges["join_method"]) == ["soft:dhcp_lease", "soft:host_inventory"]
    assert list(edges["parent_osi_layer"]) == [3, 7]
    assert set(edges["child_osi_layer"]) == {3}
    assert set(edges["lineage_id"]) == {"chain-a"}
    assert set(edges["community_id"]) == {"1:abc"}
    assert set(edges["event_time"]) == {100}


@pytest.mark.gpu_and_cpu_mode
def test_an_unresolved_hop_is_no_edge(config: Config):
    # The second flow's address is in no lease. An edge to nothing would be a relation the pipeline never made.
    edges = emit(config, resolved_flows(), **HOPS)

    assert "flow-2" not in set(edges["child_event_uid"])


@pytest.mark.gpu_and_cpu_mode
def test_an_edge_has_an_identity_of_its_own(config: Config):
    edges = emit(config, resolved_flows(), **HOPS)
    again = emit(config, resolved_flows(), **HOPS)

    assert list(edges["link_uid"]) == list(again["link_uid"])
    assert edges["link_uid"].iloc[0] == link_uid("lease-9", "flow-1", RESOLVED_THROUGH, "soft:dhcp_lease")


@pytest.mark.gpu_and_cpu_mode
def test_a_connection_is_carried_by_the_nearest_layer_below(config: Config):
    # The packets join to the flow and the handshake to the first packet, not to the flow: each sighting is joined
    # to the one beneath it, which is what lets a chain be walked a layer at a time.
    edges = emit(config, connection(), join_column="community_id")

    pairs = list(zip(edges["parent_event_uid"], edges["child_event_uid"]))

    assert pairs == [("flow", "packet-1"), ("flow", "packet-2"), ("packet-1", "handshake")]
    assert set(edges["relation_type"]) == {CARRIED_BY}
    assert set(edges["join_method"]) == {"hard:community_id"}
    assert list(edges["parent_osi_layer"]) == [3, 3, 4]
    assert list(edges["child_osi_layer"]) == [4, 4, 6]
    assert list(edges["lineage_id"]) == ["l4", "l4", "l6"]


@pytest.mark.gpu_and_cpu_mode
def test_a_connection_seen_at_one_layer_is_no_edge(config: Config):
    edges = emit(config, connection(), join_column="community_id")

    assert "lonely" not in set(edges["child_event_uid"])


@pytest.mark.gpu_and_cpu_mode
def test_a_connection_is_remembered_across_batches(config: Config):
    stage = EdgeEmitStage(config, join_column="community_id")
    payload = connection()
    first = {name: values[:1] for (name, values) in payload.items()}
    rest = {name: values[1:] for (name, values) in payload.items()}

    assert len(emit(config, first, stage)) == 0
    assert len(emit(config, rest, stage)) == 3


@pytest.mark.gpu_and_cpu_mode
def test_an_evicted_connection_misses_an_edge_and_never_makes_a_wrong_one(config: Config):
    stage = EdgeEmitStage(config, join_column="community_id", max_keys=1)
    payload = {
        "event_uid": ["flow-a", "flow-b", "packet-a"],
        "osi_layer": [3, 3, 4],
        "lineage_id": ["a", "b", "a4"],
        "event_time": [1, 2, 3],
        "community_id": ["1:a", "1:b", "1:a"],
    }

    assert len(emit(config, payload, stage)) == 0


@pytest.mark.gpu_and_cpu_mode
def test_both_relations_from_one_stream(config: Config):
    payload = resolved_flows()
    payload["community_id"] = ["1:abc", "1:abc"]
    payload["osi_layer"] = [3, 4]
    edges = emit(config, payload, join_column="community_id", **HOPS)

    assert list(edges["relation_type"]) == [RESOLVED_THROUGH, RESOLVED_THROUGH, CARRIED_BY]
    assert list(edges["child_event_uid"]) == ["flow-1", "flow-1", "flow-2"]


@pytest.mark.gpu_and_cpu_mode
def test_a_control_message_carries_its_edges(config: Config):
    message = ControlMessage()
    message.payload(MessageMeta(get_df_class(config.execution_mode)(resolved_flows())))
    result = EdgeEmitStage(config, **HOPS).on_data(message)
    df = result.payload().copy_dataframe()
    df = df.to_pandas() if hasattr(df, "to_pandas") else df

    assert len(df) == 2


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_column_names_itself(config: Config):
    payload = resolved_flows()
    del payload["host_lease_uid"]

    with pytest.raises(KeyError, match="host_lease_uid"):
        emit(config, payload, **HOPS)


@pytest.mark.gpu_and_cpu_mode
def test_constructor_rejects_bad_configuration(config: Config):
    with pytest.raises(ValueError, match="same hops"):
        EdgeEmitStage(config, hop_method_columns=["a"], hop_uid_columns=[])

    with pytest.raises(ValueError, match="at least one hop"):
        EdgeEmitStage(config)

    with pytest.raises(ValueError, match="max_keys"):
        EdgeEmitStage(config, join_column="community_id", max_keys=0)
