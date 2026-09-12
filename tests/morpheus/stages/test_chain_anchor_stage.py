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
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.chain_anchor_stage import ChainAnchorStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.type_utils import get_df_class

CANDIDATES = ["resolved_port_key", "arp_sender_ip"]


def observations() -> dict:
    """Three ARP observations: two the ladder resolved to a port, one it could not."""
    return {
        "resolved_port_key": ["hq:sw1:Gi1/0/1", None, "hq:sw1:Gi1/0/2"],
        "arp_sender_ip": ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
    }


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("candidates", CANDIDATES)
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    ChainAnchorStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_a_resolved_row_roots_on_the_port_and_an_unresolved_one_on_itself(config: Config):
    # The reason the stage exists. Before it, an ARP observation was anchored on the address being claimed and
    # nothing else, so it could never share a chain with the layer 1 samples of the port it came from.
    result = run(config, observations())

    assert list(result["chain_anchor"]) == ["hq:sw1:Gi1/0/1", "10.0.0.2", "hq:sw1:Gi1/0/2"]
    assert list(result["chain_anchor_source"]) == ["resolved_port_key", "arp_sender_ip", "resolved_port_key"]


@pytest.mark.gpu_and_cpu_mode
def test_the_source_tells_a_soft_join_from_a_direct_observation(config: Config):
    # A chain reached through the binding table is weaker evidence than one that was not, and an analyst reading
    # `lineage_id` later has to be able to tell which. The source column is that record, the way `join_method`
    # is on an edge.
    result = run(config, observations())

    resolved = result[result["chain_anchor_source"] == "resolved_port_key"]
    fallback = result[result["chain_anchor_source"] == "arp_sender_ip"]

    assert len(resolved) == 2
    assert len(fallback) == 1
    assert fallback["chain_anchor"].iloc[0] == "10.0.0.2"


@pytest.mark.gpu_and_cpu_mode
def test_a_row_with_no_candidate_value_carries_no_root(config: Config):
    # No root rather than a fabricated one. A default would pool every rootless row in the estate under one
    # invented chain, which is exactly what the null-key rule exists to prevent.
    payload = observations()
    payload["arp_sender_ip"][1] = None

    result = run(config, payload)

    assert pd.isna(result["chain_anchor"].iloc[1])
    assert pd.isna(result["chain_anchor_source"].iloc[1])
    assert result["chain_anchor"].iloc[0] == "hq:sw1:Gi1/0/1"


@pytest.mark.gpu_and_cpu_mode
def test_whitespace_and_empty_strings_are_not_values(config: Config):
    # `normalize_text` decides what counts as present, so a padded or empty resolution does not win over a real
    # fallback. The same rule the entity keys use, applied to the same kind of decision.
    payload = observations()
    payload["resolved_port_key"] = ["   ", "", "hq:sw1:Gi1/0/2"]

    result = run(config, payload)

    assert list(result["chain_anchor"]) == ["10.0.0.1", "10.0.0.2", "hq:sw1:Gi1/0/2"]
    assert list(result["chain_anchor_source"]) == ["arp_sender_ip", "arp_sender_ip", "resolved_port_key"]


@pytest.mark.gpu_and_cpu_mode
def test_candidate_order_is_preference_order(config: Config):
    # Reversing the list reverses the decision on every row where both have a value. If it did not, the
    # parameter would be a set and the "most preferred first" contract would be decoration.
    result = run(config, observations(), candidates=list(reversed(CANDIDATES)))

    assert list(result["chain_anchor"]) == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    assert list(result["chain_anchor_source"]) == ["arp_sender_ip"] * 3


@pytest.mark.gpu_and_cpu_mode
def test_a_single_candidate_is_the_old_per_class_anchor(config: Config):
    # A class with one candidate behaves exactly as the per-class constant did, so nothing that was chained
    # before is unchained by moving to a per-row decision.
    result = run(config, {"entity_key": ["hq:sw1:Gi1/0/1", None]}, candidates=["entity_key"])

    assert result["chain_anchor"].iloc[0] == "hq:sw1:Gi1/0/1"
    assert result["chain_anchor_source"].iloc[0] == "entity_key"
    assert pd.isna(result["chain_anchor"].iloc[1])


@pytest.mark.gpu_and_cpu_mode
def test_the_output_columns_are_configurable(config: Config):
    result = run(config, observations(), anchor_column="root", source_column="root_from")

    assert list(result["root"]) == ["hq:sw1:Gi1/0/1", "10.0.0.2", "hq:sw1:Gi1/0/2"]
    assert list(result["root_from"]) == ["resolved_port_key", "arp_sender_ip", "resolved_port_key"]
    assert "chain_anchor" not in result.columns


@pytest.mark.gpu_and_cpu_mode
def test_an_absent_candidate_is_a_wiring_error_not_a_missing_value(config: Config):
    # Silently skipping an absent column would root every row on the fallback and say nothing, which is the
    # failure mode that makes a soft join look like a direct one.
    with pytest.raises(KeyError, match="resolved_port_key"):
        run(config, {"arp_sender_ip": ["10.0.0.1"]})


@pytest.mark.gpu_and_cpu_mode
def test_an_empty_frame_passes_through(config: Config):
    meta = MessageMeta(get_df_class(config.execution_mode)({"resolved_port_key": [], "arp_sender_ip": []}))
    ChainAnchorStage(config, candidates=CANDIDATES).on_data(meta)

    assert meta.count == 0


@pytest.mark.gpu_and_cpu_mode
def test_a_control_message_is_handled_the_same_way(config: Config):
    message = ControlMessage()
    message.payload(MessageMeta(get_df_class(config.execution_mode)(observations())))

    ChainAnchorStage(config, candidates=CANDIDATES).on_data(message)
    df = message.payload().copy_dataframe()
    result = df.to_pandas() if hasattr(df, "to_pandas") else df

    assert list(result["chain_anchor"]) == ["hq:sw1:Gi1/0/1", "10.0.0.2", "hq:sw1:Gi1/0/2"]


def test_no_candidates_is_refused(config: Config):
    with pytest.raises(ValueError, match="at least one column"):
        ChainAnchorStage(config, candidates=[])


def test_a_repeated_candidate_is_refused(config: Config):
    with pytest.raises(ValueError, match="repeat"):
        ChainAnchorStage(config, candidates=["a", "a"])


def test_colliding_output_columns_are_refused(config: Config):
    with pytest.raises(ValueError, match="must differ"):
        ChainAnchorStage(config, candidates=["a"], anchor_column="same", source_column="same")


@pytest.mark.gpu_and_cpu_mode
def test_the_stage_runs_in_a_pipeline(config: Config):
    # The needed columns are declared, so a pipeline preallocates them; the decision is still made on the value,
    # not on the column's presence, or a preallocated empty column would read as "already anchored".
    frame = get_df_class(config.execution_mode)(observations())
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[frame]))
    pipe.add_stage(ChainAnchorStage(config, candidates=CANDIDATES))
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1

    df = messages[0].copy_dataframe()
    result = df.to_pandas() if hasattr(df, "to_pandas") else df

    assert list(result["chain_anchor"]) == ["hq:sw1:Gi1/0/1", "10.0.0.2", "hq:sw1:Gi1/0/2"]
