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

from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc2_cardinality_stage import TC2CardinalityStage
from morpheus.utils.distinct_window import NS_PER_SECOND
from morpheus.utils.type_utils import get_df_class

MINUTE_NS = 60 * NS_PER_SECOND


def frame(macs: list, ports: list = None, vlans: list = None, ouis: list = None, times: list = None) -> dict:
    count = len(macs)
    payload = {
        "mac_address": macs,
        "site_id": ["hq"] * count,
        "switch_id": ["sw1"] * count,
        "port_id": ["Gi1/0/1"] * count if ports is None else ports,
        "vlan_id": ["10"] * count if vlans is None else vlans,
        "event_time": [index * MINUTE_NS for index in range(count)] if times is None else times,
    }

    if (ouis is not None):
        payload["oui"] = ouis

    return payload


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [None if value is None or value is pd.NA or value != value else value for value in series.tolist()]


def run(config: Config, payload: dict, **kwargs) -> MessageMeta:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC2CardinalityStage(config, **kwargs).on_data(meta)

    return meta


def test_execution_modes(config: Config):
    assert issubclass(TC2CardinalityStage, GpuAndCpuMixin)

    assert set(TC2CardinalityStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    needed = TC2CardinalityStage(config).get_needed_columns()

    assert needed["macs_per_port"] == TypeId.INT64
    assert needed["ports_per_mac"] == TypeId.INT64
    assert needed["ouis_per_vlan"] == TypeId.INT64
    assert needed["macs_per_port_first_in_window"] == TypeId.BOOL8
    assert needed["macs_per_port_saturated"] == TypeId.BOOL8


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(TC2CardinalityStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.gpu_and_cpu_mode
def test_a_single_device_port_stays_at_one(config: Config):
    meta = run(config, frame(["00:11:22:33:44:55"] * 4))

    assert _as_list(meta, "macs_per_port") == [1, 1, 1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_an_unauthorized_switch_raises_the_port_count(config: Config):
    # Several MACs behind one access port is the classic signature.
    meta = run(config, frame(["00:11:22:33:44:55", "00:11:22:33:44:66", "00:11:22:33:44:77"]))

    assert _as_list(meta, "macs_per_port") == [1, 2, 3]
    assert _as_list(meta, "macs_per_port_first_in_window") == [True, True, True]


@pytest.mark.gpu_and_cpu_mode
def test_a_mac_on_several_ports_raises_the_mac_count(config: Config):
    # One MAC belongs to one interface; on several it is spoofing, or a device being moved.
    meta = run(config, frame(["00:11:22:33:44:55"] * 3, ports=["Gi1/0/1", "Gi1/0/2", "Gi1/0/3"]))

    assert _as_list(meta, "ports_per_mac") == [1, 2, 3]
    # Each port sees the MAC for the first time, so the per-port count stays at one.
    assert _as_list(meta, "macs_per_port") == [1, 1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_ports_are_keyed_by_site_and_switch_too(config: Config):
    # The same interface name exists on every switch in the estate; keying on it alone would pool them.
    payload = frame(["00:11:22:33:44:55", "00:11:22:33:44:66"])
    payload["switch_id"] = ["sw1", "sw2"]

    meta = run(config, payload)

    assert _as_list(meta, "port_key") == ["hq:sw1:Gi1/0/1", "hq:sw2:Gi1/0/1"]
    assert _as_list(meta, "macs_per_port") == [1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_a_new_oui_on_a_vlan_is_flagged(config: Config):
    meta = run(
        config,
        frame(["00:11:22:33:44:55", "00:11:22:33:44:66", "aa:bb:cc:00:00:01"], ports=["Gi1/0/1", "Gi1/0/2", "Gi1/0/3"]))

    # Two devices from one vendor then one from another: the third is an unmanaged device class on the segment.
    assert _as_list(meta, "ouis_per_vlan") == [1, 1, 2]
    assert _as_list(meta, "ouis_per_vlan_first_in_window") == [True, False, True]


@pytest.mark.gpu_and_cpu_mode
def test_the_oui_is_derived_when_the_collector_omits_it(config: Config):
    # Without deriving it, an estate whose collectors omit the field would report one OUI per VLAN forever.
    payload = frame(["00:11:22:33:44:55", "aa:bb:cc:00:00:01"], ports=["Gi1/0/1", "Gi1/0/2"])

    assert "oui" not in payload

    meta = run(config, payload)

    assert _as_list(meta, "ouis_per_vlan") == [1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_a_supplied_oui_is_used_as_given(config: Config):
    payload = frame(["00:11:22:33:44:55", "00:11:22:33:44:66"],
                    ports=["Gi1/0/1", "Gi1/0/2"],
                    ouis=["vendor-a", "vendor-b"])

    meta = run(config, payload)

    # The collector's own classification wins where it exists; the MAC prefix is only the fallback.
    assert _as_list(meta, "ouis_per_vlan") == [1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_vlans_are_counted_separately(config: Config):
    payload = frame(["00:11:22:33:44:55", "aa:bb:cc:00:00:01"], ports=["Gi1/0/1", "Gi1/0/2"], vlans=["10", "20"])

    meta = run(config, payload)

    # A different vendor on a different VLAN is not novelty on the first one.
    assert _as_list(meta, "ouis_per_vlan") == [1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_values_leave_the_window(config: Config):
    payload = frame(["00:11:22:33:44:55", "00:11:22:33:44:66"], times=[0, 120 * MINUTE_NS])

    meta = run(config, payload, window_seconds=300)

    # Two hours later the first MAC is long gone from the window, so the port is back to carrying one device.
    assert _as_list(meta, "macs_per_port") == [1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_saturation_is_reported_rather_than_hidden(config: Config):
    # A MAC flood is what this feature exists to notice and what would exhaust memory; the cap makes the count a
    # floor and says so.
    macs = [f"00:11:22:33:44:{index:02x}" for index in range(6)]

    meta = run(config, frame(macs), max_samples=3)

    assert _as_list(meta, "macs_per_port") == [1, 2, 3, 3, 3, 3]
    assert _as_list(meta, "macs_per_port_saturated") == [False, False, False, True, True, True]


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    df_class = get_df_class(config.execution_mode)
    stage = TC2CardinalityStage(config)

    stage.on_data(MessageMeta(df_class(frame(["00:11:22:33:44:55"], times=[0]))))
    second = MessageMeta(df_class(frame(["00:11:22:33:44:66"], times=[MINUTE_NS])))
    stage.on_data(second)

    # The window spans the message boundary, which is the whole reason this stage is stateful.
    assert _as_list(second, "macs_per_port") == [2]


@pytest.mark.gpu_and_cpu_mode
def test_the_current_sample_is_counted(config: Config):
    meta = run(config, frame(["00:11:22:33:44:55"]))

    # A threshold has to trip on the row that crosses it, not on the row after.
    assert _as_list(meta, "macs_per_port") == [1]


@pytest.mark.gpu_and_cpu_mode
def test_warm_up_columns_are_typed(config: Config):
    meta = run(config, frame(["00:11:22:33:44:55", "00:11:22:33:44:66"]))

    assert pd.api.types.is_integer_dtype(meta.get_data("macs_per_port").dtype)
    assert pd.api.types.is_bool_dtype(meta.get_data("macs_per_port_saturated").dtype)


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    message = ControlMessage()
    payload = frame(["00:11:22:33:44:55", "00:11:22:33:44:66"])
    message.payload(MessageMeta(get_df_class(config.execution_mode)(payload)))

    TC2CardinalityStage(config).on_data(message)

    assert _as_list(message.payload(), "macs_per_port") == [1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_tc2_cardinality_stage_pipe(config: Config):
    payload = frame(["00:11:22:33:44:55", "00:11:22:33:44:66", "00:11:22:33:44:77"])
    source_df = get_df_class(config.execution_mode)(payload)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC2CardinalityStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "macs_per_port") == [1, 2, 3]


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = frame(["00:11:22:33:44:55"])
    del payload["vlan_id"]

    with pytest.raises(KeyError, match="vlan_id"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError):
        TC2CardinalityStage(config, window_seconds=0)

    with pytest.raises(ValueError):
        TC2CardinalityStage(config, max_samples=0)


@pytest.mark.gpu_and_cpu_mode
def test_a_snapshot_lists_every_mac_on_a_port_at_one_instant(config: Config):
    # A MAC table snapshot reports every address on a port with the snapshot's own timestamp. Five hosts behind a
    # hub arrive as five rows at one instant, and every one must count; this stage read them as one host.
    macs = [f"00:11:22:33:44:{index:02x}" for index in range(5)]
    meta = run(config, frame(macs, times=[MINUTE_NS] * 5))

    assert _as_list(meta, "macs_per_port") == [1, 2, 3, 4, 5]


@pytest.mark.gpu_and_cpu_mode
def test_a_null_port_yields_null_counts_rather_than_a_fabricated_key(config: Config):
    # Two observations with no port. Pooled under the string "None:..." they would read as two MACs on one very
    # busy port; they are nothing of the kind.
    payload = frame(["00:11:22:33:44:55", "00:11:22:33:44:66", "00:11:22:33:44:77"],
                    ports=[None, None, "Gi1/0/1"],
                    times=[0, MINUTE_NS, 2 * MINUTE_NS])
    meta = run(config, payload)

    assert _as_list(meta, "port_key") == [None, None, "hq:sw1:Gi1/0/1"]
    assert _as_list(meta, "macs_per_port") == [None, None, 1]
    assert _as_list(meta, "ports_per_mac") == [None, None, 1]
    # The VLAN count does not depend on the port and still runs; all three MACs share one OUI.
    assert _as_list(meta, "ouis_per_vlan") == [1, 1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_port_key_is_composed_like_the_layer_1_entity_key(config: Config):
    from morpheus.utils.entity_key import compose_key

    meta = run(config, frame(["00:11:22:33:44:55"]))

    assert _as_list(meta, "port_key") == [compose_key(("hq", "sw1", "Gi1/0/1"))]


@pytest.mark.cpu_mode
def test_a_vlan_stays_one_vlan_when_its_column_widens(config: Config):
    # `vlan_id` is an entity here, so the rendering rule that protects `entity_key` has to reach it too. pandas
    # widens the column to float as soon as one row in the batch has no VLAN, and rendering that as "10.0" forks
    # VLAN 10 into a second VLAN whose OUI history starts empty -- so a flood sits under the threshold because
    # some unrelated row was null. Which batch the null lands in depends on where the stream was divided, so this
    # is the batch-split invariance control 13 checks, not only a counting error.
    stage = TC2CardinalityStage(config)

    complete = pd.DataFrame(frame(["00:aa:aa:00:00:01", "00:bb:bb:00:00:01"], vlans=[10, 10], times=[0, MINUTE_NS]))
    stage.on_data(MessageMeta(complete))

    widened = pd.DataFrame(
        frame(["00:cc:cc:00:00:01", "00:dd:dd:00:00:01"], vlans=[10, None], times=[2 * MINUTE_NS, 3 * MINUTE_NS]))
    stage.on_data(MessageMeta(widened))

    assert complete["vlan_id"].dtype != widened["vlan_id"].dtype, "pandas no longer widens; this proves nothing"

    # Three OUIs have now been seen on VLAN 10. The fourth row has no VLAN and so has no count at all.
    assert list(complete["ouis_per_vlan"]) == [1, 2]
    assert list(widened["ouis_per_vlan"])[0] == 3


def test_the_stage_renders_keys_by_the_shared_rule_not_its_own():
    # This stage used to carry a private copy of the normalization rule, and the copy drifted. Delegation is the
    # fix; this pins it, because a reimplementation that looks equivalent is how the drift happened the first time.
    from morpheus.utils.entity_key import normalize_text

    for value in (10, 10.0, "10", " 10 "):
        assert TC2CardinalityStage._text(value) == normalize_text(value) == "10"

    for missing in (None, float("nan"), ""):
        assert TC2CardinalityStage._text(missing) is normalize_text(missing) is None
