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

import pytest

from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc2_binding_stage import TC2BindingStage
from morpheus.utils.binding_closer import DISPLACED
from morpheus.utils.binding_closer import DRAINED
from morpheus.utils.binding_closer import NS_PER_SECOND
from morpheus.utils.binding_table import BindingTable
from morpheus.utils.type_utils import get_df_class

MINUTE_NS = 60 * NS_PER_SECOND
MAC_A = "00:11:22:33:44:55"
MAC_B = "aa:bb:cc:dd:ee:ff"


def frame(macs: list, ports: list, times: list = None, switches: list = None, vlans: list = None) -> dict:
    count = len(macs)

    return {
        "mac_address": macs,
        "event_time": [index * MINUTE_NS for index in range(count)] if times is None else times,
        "site_id": ["hq"] * count,
        "switch_id": ["sw1"] * count if switches is None else switches,
        "port_id": ports,
        "vlan_id": ["10"] * count if vlans is None else vlans,
    }


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return series.tolist()


def feed(config: Config, payload: dict, **kwargs) -> tuple:
    """Run one batch through a stage and return (emitted frames, the stage)."""
    stage = TC2BindingStage(config, **kwargs)
    emitted = stage.on_data(MessageMeta(get_df_class(config.execution_mode)(payload)))

    return (emitted, stage)


def test_execution_modes(config: Config):
    assert issubclass(TC2BindingStage, GpuAndCpuMixin)

    assert set(TC2BindingStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    needed = TC2BindingStage(config).get_needed_columns()

    assert needed["bind_start"] == TypeId.INT64
    assert needed["bind_end"] == TypeId.INT64
    assert needed["bind_end_reason"] == TypeId.STRING
    assert needed["bind_end_observed"] == TypeId.BOOL8


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(TC2BindingStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.gpu_and_cpu_mode
def test_a_stable_estate_emits_nothing(config: Config):
    # The normal case. Nothing moved, so no binding ended, so there is nothing to emit yet.
    (emitted, stage) = feed(config, frame([MAC_A] * 4, ["Gi1/0/1"] * 4))

    assert emitted == []
    assert stage.open_count == 1


@pytest.mark.gpu_and_cpu_mode
def test_a_move_emits_a_closed_binding(config: Config):
    (emitted, _) = feed(config, frame([MAC_A, MAC_A, MAC_A], ["Gi1/0/1", "Gi1/0/1", "Gi1/0/2"]))

    assert len(emitted) == 1
    binding = emitted[0]

    assert _as_list(binding, "mac_address") == [MAC_A]
    assert _as_list(binding, "port_id") == ["Gi1/0/1"]
    assert _as_list(binding, "bind_start") == [0]
    # Closed one tick past the last sighting on the old port, not at the sighting on the new one.
    assert _as_list(binding, "bind_end") == [MINUTE_NS + 1]
    assert _as_list(binding, "bind_end_reason") == [DISPLACED]
    assert _as_list(binding, "bind_end_observed") == [False]
    assert _as_list(binding, "bind_observations") == [2]


@pytest.mark.gpu_and_cpu_mode
def test_the_output_resolves_through_a_binding_table(config: Config):
    """The whole point: what this emits has to be consumable by the resolver, not merely well-shaped."""
    (emitted, stage) = feed(config, frame([MAC_A, MAC_A, MAC_A], ["Gi1/0/1", "Gi1/0/1", "Gi1/0/2"]))
    closed = emitted + stage.on_completed()

    import pandas as pd
    records = pd.concat([
        meta.copy_dataframe() if not hasattr(meta.copy_dataframe(), "to_pandas") else meta.copy_dataframe().to_pandas()
        for meta in closed
    ],
                        ignore_index=True)

    table = BindingTable.from_dataframe(records,
                                        name="mac_to_port",
                                        key_column="mac_address",
                                        value_columns=["switch_id", "port_id", "vlan_id"],
                                        start_column="bind_start",
                                        end_column="bind_end")

    assert table.resolve(MAC_A, 0).values[1] == "Gi1/0/1"
    assert table.resolve(MAC_A, 2 * MINUTE_NS).values[1] == "Gi1/0/2"


@pytest.mark.gpu_and_cpu_mode
def test_the_gap_between_bindings_resolves_to_nothing(config: Config):
    (emitted, _) = feed(config, frame([MAC_A, MAC_A], ["Gi1/0/1", "Gi1/0/2"], times=[0, 10 * MINUTE_NS]))

    import pandas as pd
    records = emitted[0].copy_dataframe()
    records = records.to_pandas() if hasattr(records, "to_pandas") else records

    table = BindingTable.from_dataframe(pd.DataFrame(records),
                                        name="mac_to_port",
                                        key_column="mac_address",
                                        value_columns=["switch_id", "port_id", "vlan_id"],
                                        start_column="bind_start",
                                        end_column="bind_end")

    # Five minutes after the MAC was last seen on Gi1/0/1 and before it appeared on Gi1/0/2, nobody knows where it
    # was. Resolving to nothing says so; stretching the binding would name a port confidently and possibly wrongly.
    assert table.resolve(MAC_A, 0) is not None
    assert table.resolve(MAC_A, 5 * MINUTE_NS) is None


@pytest.mark.gpu_and_cpu_mode
def test_still_open_bindings_are_emitted_at_the_end(config: Config):
    # Without this, a replay over a finite corpus would lose every MAC that never moved, which is most of them.
    (emitted, stage) = feed(config, frame([MAC_A] * 3, ["Gi1/0/1"] * 3))

    assert emitted == []

    final = stage.on_completed()

    assert len(final) == 1
    assert _as_list(final[0], "mac_address") == [MAC_A]
    assert _as_list(final[0], "bind_end_reason") == [DRAINED]


@pytest.mark.gpu_and_cpu_mode
def test_opting_out_of_the_final_flush(config: Config):
    (_, stage) = feed(config, frame([MAC_A] * 3, ["Gi1/0/1"] * 3), emit_open_on_complete=False)

    assert stage.on_completed() == []


@pytest.mark.gpu_and_cpu_mode
def test_macs_are_tracked_separately(config: Config):
    payload = frame([MAC_A, MAC_B, MAC_A, MAC_B], ["Gi1/0/1", "Gi1/0/2", "Gi1/0/3", "Gi1/0/2"],
                    times=[0, 0, MINUTE_NS, MINUTE_NS])

    (emitted, stage) = feed(config, payload)

    # Only the MAC that moved closed a binding; the stationary one is still open.
    assert _as_list(emitted[0], "mac_address") == [MAC_A]
    assert stage.open_count == 2


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    df_class = get_df_class(config.execution_mode)
    stage = TC2BindingStage(config)

    stage.on_data(MessageMeta(df_class(frame([MAC_A], ["Gi1/0/1"], times=[0]))))
    emitted = stage.on_data(MessageMeta(df_class(frame([MAC_A], ["Gi1/0/2"], times=[MINUTE_NS]))))

    # The displacement spans the message boundary, which is the whole reason this stage is stateful.
    assert len(emitted) == 1
    assert _as_list(emitted[0], "port_id") == ["Gi1/0/1"]


@pytest.mark.gpu_and_cpu_mode
def test_untracked_columns_do_not_split_a_binding(config: Config):
    payload = frame([MAC_A] * 3, ["Gi1/0/1"] * 3)
    payload["wireless_rssi"] = [-50, -60, -80]

    (emitted, stage) = feed(config, payload, attribute_columns=["switch_id", "port_id"])

    assert emitted == []
    assert stage.open_count == 1


@pytest.mark.gpu_and_cpu_mode
def test_a_vlan_change_is_a_displacement(config: Config):
    payload = frame([MAC_A, MAC_A], ["Gi1/0/1", "Gi1/0/1"], vlans=["10", "20"])

    (emitted, _) = feed(config, payload)

    assert _as_list(emitted[0], "vlan_id") == ["10"]


@pytest.mark.gpu_and_cpu_mode
def test_out_of_order_observation_is_skipped(config: Config):
    payload = frame([MAC_A, MAC_A, MAC_A], ["Gi1/0/1", "Gi1/0/1", "Gi1/0/2"], times=[0, 5 * MINUTE_NS, MINUTE_NS])

    (emitted, stage) = feed(config, payload)

    # The late sighting on the other port must not displace a binding it predates.
    assert emitted == []
    assert _as_list(stage.on_completed()[0], "port_id") == ["Gi1/0/1"]


@pytest.mark.gpu_and_cpu_mode
def test_tc2_binding_stage_pipe(config: Config):
    source_df = get_df_class(config.execution_mode)(frame([MAC_A, MAC_A, MAC_A], ["Gi1/0/1", "Gi1/0/1", "Gi1/0/2"]))

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC2BindingStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()

    # One binding closed by the move, and one drained at the end of the stream.
    reasons = sorted(reason for message in messages for reason in _as_list(message, "bind_end_reason"))
    assert reasons == [DISPLACED, DRAINED]


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    message = ControlMessage()
    payload = frame([MAC_A, MAC_A], ["Gi1/0/1", "Gi1/0/2"])
    message.payload(MessageMeta(get_df_class(config.execution_mode)(payload)))

    emitted = TC2BindingStage(config).on_data(message)

    assert _as_list(emitted[0], "port_id") == ["Gi1/0/1"]


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = frame([MAC_A], ["Gi1/0/1"])
    del payload["vlan_id"]

    with pytest.raises(KeyError, match="vlan_id"):
        feed(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError):
        TC2BindingStage(config, idle_timeout_seconds=0)


@pytest.mark.gpu_and_cpu_mode
def test_a_closed_binding_carries_the_port_as_layer_1_spells_it(config: Config):
    # The ladder's first arrow: a MAC resolved through this binding must land on the string the TC-1 stages key on.
    # Layer 1 calls the middle segment device_id and layer 2 calls it switch_id; the composed key is what joins.
    from morpheus.stages.telemetry.tc1_normalize_stage import TC1NormalizeStage

    (_, stage) = feed(config, frame([MAC_A], ["Gi1/0/1"]))
    binding = stage.on_completed()[0]

    layer_1 = MessageMeta(
        get_df_class(config.execution_mode)({
            "site_id": ["hq"],
            "device_id": ["sw1"],
            "port_id": ["Gi1/0/1"],
            "event_time": [0],
            "crc_errors": [0],
            "symbol_errors": [0],
            "input_discards": [0],
            "output_discards": [0],
        }))
    TC1NormalizeStage(config).on_data(layer_1)

    assert _as_list(binding, "port_key") == _as_list(layer_1, "entity_key") == ["hq:sw1:Gi1/0/1"]


@pytest.mark.gpu_and_cpu_mode
def test_the_site_is_part_of_the_default_binding_target(config: Config):
    # The telemetry class's entity key is site:switch:port:vlan. A target without the site cannot join to layer 1.
    payload = frame([MAC_A], ["Gi1/0/1"])
    del payload["site_id"]

    with pytest.raises(KeyError, match="site_id"):
        feed(config, payload)


@pytest.mark.gpu_and_cpu_mode
def test_port_key_is_null_when_the_target_does_not_name_a_full_port(config: Config):
    # A caller binding MACs to VLANs alone gets bindings, but no port key, rather than a partial one.
    (_, stage) = feed(config, frame([MAC_A], ["Gi1/0/1"]), attribute_columns=["switch_id", "vlan_id"])
    binding = stage.on_completed()[0]

    assert _as_list(binding, "port_key") == [None]
