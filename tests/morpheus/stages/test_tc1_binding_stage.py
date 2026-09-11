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
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc1_binding_stage import DEFAULT_IDLE_TIMEOUT_SECONDS
from morpheus.stages.telemetry.tc1_binding_stage import TC1BindingStage
from morpheus.utils.type_utils import get_df_class

SECOND = 10**9
HOUR = 3600 * SECOND
DAY = 24 * HOUR

PORT = ("hq", "sw1", "Gi1/0/1")
OTHER = ("hq", "sw1", "Gi1/0/2")


def samples(rows: list) -> dict:
    """One row per poll: (port tuple, time, transceiver, neighbor)."""
    return {
        "site_id": [row[0][0] for row in rows],
        "device_id": [row[0][1] for row in rows],
        "port_id": [row[0][2] for row in rows],
        "event_time": [row[1] for row in rows],
        "transceiver_serial": [row[2] for row in rows],
        "lldp_neighbor_chassis_id": [row[3] for row in rows],
    }


def _frame(meta: MessageMeta):
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    """Feed one batch, then complete, and return every emitted binding as one host frame."""
    stage = TC1BindingStage(config, **kwargs)
    frames = [_frame(meta) for meta in stage.on_data(MessageMeta(get_df_class(config.execution_mode)(payload)))]
    frames.extend(_frame(meta) for meta in stage.on_completed())

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@pytest.mark.gpu_and_cpu_mode
def test_a_stable_port_is_emitted_once_at_the_end(config: Config):
    # Nearly every port in a healthy estate never changes, and those are exactly the rows the lookup exists to
    # hold. A stage that only emitted on change would produce an almost empty table.
    result = run(config, samples([(PORT, t * HOUR, "SN-AAA", "chassis-a") for t in range(4)]))

    assert len(result) == 1
    assert result["entity_key"].iloc[0] == "hq:sw1:Gi1/0/1"
    assert result["transceiver_serial"].iloc[0] == "SN-AAA"
    assert result["bind_end_reason"].iloc[0] == "drained"


@pytest.mark.gpu_and_cpu_mode
def test_swapping_the_optic_closes_the_binding_and_opens_another(config: Config):
    # The event this layer exists to see. The port is the key and the optic is what moved, so a changed serial
    # ends one interval and starts the next.
    result = run(config, samples([(PORT, 0, "SN-AAA", "chassis-a"), (PORT, HOUR, "SN-BBB", "chassis-a")]))

    assert len(result) == 2
    assert list(result["transceiver_serial"]) == ["SN-AAA", "SN-BBB"]
    assert result["bind_end_reason"].iloc[0] == "displaced"


@pytest.mark.gpu_and_cpu_mode
def test_a_binding_ends_just_after_its_last_sighting_not_when_the_change_was_noticed(config: Config):
    # The property that makes this table answer "what was in this port at time T" honestly. A port polled at
    # noon and found changed at one o'clock was not observed holding the old optic for that hour: the swap could
    # have happened at any point inside it. The old binding therefore ends just after the last poll that saw it,
    # and the hour of silence resolves to nothing rather than to a claim nobody made.
    rows = [(PORT, 0, "SN-AAA", "chassis-a"), (PORT, HOUR, "SN-AAA", "chassis-a"),
            (PORT, 2 * HOUR, "SN-BBB", "chassis-a")]
    result = run(config, samples(rows))
    old = result.iloc[0]

    assert old["bind_end_reason"] == "displaced"
    assert old["bind_start"] == 0
    assert old["bind_end"] == HOUR + 1, "the interval must stop at the last sighting, not stretch to the swap"
    assert old["bind_observations"] == 2


@pytest.mark.gpu_and_cpu_mode
def test_a_repatched_neighbor_is_a_change_too(config: Config):
    # The far end moving matters as much as the near end. Somebody repatched the fibre.
    result = run(config, samples([(PORT, 0, "SN-AAA", "chassis-a"), (PORT, HOUR, "SN-AAA", "chassis-b")]))

    assert len(result) == 2
    assert list(result["lldp_neighbor_chassis_id"]) == ["chassis-a", "chassis-b"]


@pytest.mark.gpu_and_cpu_mode
def test_a_moving_optical_reading_does_not_split_a_binding(config: Config):
    # Columns outside the attribute list are ignored. Optical power varies every poll; a binding that split on it
    # would produce a new interval per sample and a lookup nobody could use.
    payload = samples([(PORT, t * HOUR, "SN-AAA", "chassis-a") for t in range(4)])
    payload["optical_rx_dbm"] = [-3.1, -3.2, -3.15, -3.4]

    assert len(run(config, payload)) == 1


@pytest.mark.gpu_and_cpu_mode
def test_ports_do_not_contaminate_each_other(config: Config):
    result = run(
        config,
        samples([(PORT, 0, "SN-AAA", "chassis-a"), (OTHER, 0, "SN-BBB", "chassis-b"),
                 (PORT, HOUR, "SN-CCC", "chassis-a")]))

    assert set(result["entity_key"]) == {"hq:sw1:Gi1/0/1", "hq:sw1:Gi1/0/2"}
    assert len(result[result["entity_key"] == "hq:sw1:Gi1/0/2"]) == 1


@pytest.mark.gpu_and_cpu_mode
def test_the_key_is_the_string_layer_1_and_layer_2_both_compose(config: Config):
    # The ladder's first arrow is a join on this string. If it were composed differently here than in
    # TC1NormalizeStage or in TC2BindingStage's port_key, the hop would be a reconstruction rather than a join.
    from morpheus.utils.entity_key import compose_key  # pylint: disable=import-outside-toplevel

    result = run(config, samples([(PORT, 0, "SN-AAA", "chassis-a")]))

    assert result["entity_key"].iloc[0] == compose_key(list(PORT))


@pytest.mark.gpu_and_cpu_mode
def test_the_device_is_emitted_under_both_names(config: Config):
    # Layer 1 says device_id, layer 2 says switch_id, and the shipped binding_l1 refresh search keys on
    # switch_id. Emitting both is what lets that search work without renaming a column on a search head.
    result = run(config, samples([(PORT, 0, "SN-AAA", "chassis-a")]))

    assert result["device_id"].iloc[0] == "sw1"
    assert result["switch_id"].iloc[0] == "sw1"


@pytest.mark.gpu_and_cpu_mode
def test_the_key_parts_are_recoverable_as_their_own_columns(config: Config):
    # The refresh search builds its lookup key from port_id and switch_id, so a consumer must never have to split
    # the composed string itself.
    result = run(config, samples([(PORT, 0, "SN-AAA", "chassis-a")]))

    assert result["site_id"].iloc[0] == "hq"
    assert result["port_id"].iloc[0] == "Gi1/0/1"


@pytest.mark.gpu_and_cpu_mode
def test_the_uid_is_built_the_way_the_binding_table_builds_it(config: Config):
    # The bucketed layer 2 path gets its uid from the bucketing step. There is no bucketing step here, so this
    # stage computes it -- and it has to agree with the other path or the two layers identify a binding
    # differently and an analyst cannot recover either from the other.
    from morpheus.utils.lineage import event_uid  # pylint: disable=import-outside-toplevel

    result = run(config, samples([(PORT, 0, "SN-AAA", "chassis-a"), (PORT, HOUR, "SN-BBB", "chassis-a")]))
    first = result.iloc[0]

    assert first["binding_uid"] == event_uid("port_inventory",
                                             "hq:sw1:Gi1/0/1",
                                             int(first["bind_start"]),
                                             int(first["bind_end"]),
                                             "SN-AAA",
                                             "chassis-a")


@pytest.mark.gpu_and_cpu_mode
def test_a_port_that_stops_being_polled_ages_out(config: Config):
    # Decommissioned rather than merely quiet. The advance stays inside the clock guard's skew bound, because a
    # jump past that is refused as an implausible timestamp and would age nothing out -- which is the guard
    # working, and was this test's first failure.
    result = run(config,
                 samples([(PORT, 0, "SN-AAA", "chassis-a"), (OTHER, 3 * DAY, "SN-BBB", "chassis-b")]),
                 idle_timeout_seconds=DAY // SECOND)
    aged = result[result["entity_key"] == "hq:sw1:Gi1/0/1"]

    assert aged["bind_end_reason"].iloc[0] == "idle_timeout"


@pytest.mark.gpu_and_cpu_mode
def test_the_idle_timeout_is_days_rather_than_minutes(config: Config):
    # The decision this stage exists to get right, asserted rather than left in a docstring. A poller outage must
    # not close every binding in the estate: at layer 2 a quiet MAC has left, but a quiet port has only stopped
    # being asked. With the layer 2 default of thirty minutes, a six-hour collector gap would age this out.
    assert DEFAULT_IDLE_TIMEOUT_SECONDS >= 24 * 3600

    gap = samples([(PORT, 0, "SN-AAA", "chassis-a"), (PORT, 6 * HOUR, "SN-AAA", "chassis-a")])
    result = run(config, gap)

    assert len(result) == 1, "a six-hour polling gap must not age out a port that is still there"
    assert result["bind_end_reason"].iloc[0] == "drained"


@pytest.mark.gpu_and_cpu_mode
def test_a_row_missing_a_key_part_binds_to_nothing(config: Config):
    payload = samples([(PORT, 0, "SN-AAA", "chassis-a")])
    payload["port_id"] = [None]

    assert len(run(config, payload)) == 0


@pytest.mark.gpu_and_cpu_mode
def test_a_provisional_record_is_available_before_the_binding_closes(config: Config):
    result = run(config, samples([(PORT, 0, "SN-AAA", "chassis-a")]), emit_open_bindings=True)
    provisional = result[result["bind_provisional"]]

    assert len(provisional) == 1
    assert provisional["bind_end_reason"].iloc[0] == "open"
    assert pd.isna(provisional["bind_end"].iloc[0])


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    stage = TC1BindingStage(config)
    first = stage.on_data(MessageMeta(get_df_class(config.execution_mode)(samples([(PORT, 0, "SN-AAA", "c-a")]))))
    second = stage.on_data(MessageMeta(get_df_class(config.execution_mode)(samples([(PORT, HOUR, "SN-BBB", "c-a")]))))

    assert not first
    assert len(second) == 1
    assert _frame(second[0])["bind_end_reason"].iloc[0] == "displaced"


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    stage = TC1BindingStage(config)
    message = ControlMessage()
    message.payload(MessageMeta(get_df_class(config.execution_mode)(samples([(PORT, 0, "SN-AAA", "c-a")]))))

    assert not stage.on_data(message)
    assert len(stage.on_completed()) == 1


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = samples([(PORT, 0, "SN-AAA", "chassis-a")])
    del payload["transceiver_serial"]

    with pytest.raises(KeyError, match="TC1BindingStage requires columns.*transceiver_serial"):
        run(config, payload)


@pytest.mark.gpu_and_cpu_mode
def test_tc1_binding_stage_pipe(config: Config):
    frame = get_df_class(config.execution_mode)(samples([(PORT, 0, "SN-AAA", "chassis-a"),
                                                         (PORT, HOUR, "SN-BBB", "chassis-a")]))
    pipeline = LinearPipeline(config)
    pipeline.set_source(InMemorySourceStage(config, [frame]))
    pipeline.add_stage(TC1BindingStage(config))
    sink = pipeline.add_stage(InMemorySinkStage(config))
    pipeline.run()

    assert len(sink.get_messages()) >= 1


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError, match="idle_timeout_seconds must be positive"):
        TC1BindingStage(config, idle_timeout_seconds=0)

    with pytest.raises(ValueError, match="attribute_columns must name at least one column"):
        TC1BindingStage(config, attribute_columns=[])

    with pytest.raises(ValueError, match="key_columns must name at least one column"):
        TC1BindingStage(config, key_columns=[])
