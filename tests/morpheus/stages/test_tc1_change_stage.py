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
from morpheus.stages.telemetry.tc1_change_stage import TC1ChangeStage
from morpheus.utils.type_utils import get_df_class

DAY_NS = 24 * 3600 * 10**9


def frame(serials: list, neighbors: list = None, times: list = None, entity: str = "hq:sw1:Gi1/0/1") -> dict:
    count = len(serials)

    return {
        "entity_key": [entity] * count,
        "event_time": [index * DAY_NS for index in range(count)] if times is None else times,
        "transceiver_serial": serials,
        "lldp_neighbor_chassis_id": ["chassis-a"] * count if neighbors is None else neighbors,
    }


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [None if value is None or value is pd.NA or value != value else value for value in series.tolist()]


def run(config: Config, payload: dict, **kwargs) -> MessageMeta:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC1ChangeStage(config, **kwargs).on_data(meta)

    return meta


def test_execution_modes(config: Config):
    assert issubclass(TC1ChangeStage, GpuAndCpuMixin)

    assert set(TC1ChangeStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    needed = TC1ChangeStage(config).get_needed_columns()

    assert needed["transceiver_serial_changed"] == TypeId.BOOL8
    assert needed["transceiver_serial_first_seen"] == TypeId.BOOL8
    assert needed["transceiver_serial_distinct_count"] == TypeId.INT64


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(TC1ChangeStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.gpu_and_cpu_mode
def test_a_substitution_is_detected(config: Config):
    meta = run(config, frame(["SN-AAA", "SN-AAA", "SN-BBB"]))

    assert _as_list(meta, "transceiver_serial_changed") == [None, False, True]
    assert _as_list(meta, "transceiver_serial_first_seen") == [None, False, True]


@pytest.mark.gpu_and_cpu_mode
def test_a_change_across_a_month_boundary_is_detected(config: Config):
    # The case the period-bucketed feature cannot see. Two samples forty days apart fall in different monthly
    # buckets, each holding one distinct value, so a distinct count reads them as no change at all.
    meta = run(config, frame(["SN-AAA", "SN-BBB"], times=[0, 40 * DAY_NS]))

    assert _as_list(meta, "transceiver_serial_changed") == [None, True]


@pytest.mark.gpu_and_cpu_mode
def test_it_sees_what_the_period_bucketed_feature_cannot(config: Config):
    """The reason this stage exists, demonstrated against the stage it supersedes rather than asserted."""
    from morpheus.stages.telemetry.tc1_feature_stage import TC1FeatureStage

    # One collector, strictly increasing sequence numbers, and a swap that straddles a month boundary.
    payload = frame(["SN-AAA", "SN-BBB"], times=[0, 40 * DAY_NS])
    payload["collector_id"] = ["poller-1"] * 2
    payload["collector_seq"] = [0, 1]

    df_class = get_df_class(config.execution_mode)

    # TC1FeatureStage re-orders rows, so it returns a new payload rather than mutating the one it was given.
    bucketed = TC1FeatureStage(config).on_data(MessageMeta(df_class(dict(payload))))

    compared = MessageMeta(df_class(dict(payload)))
    TC1ChangeStage(config).on_data(compared)

    # The distinct count reads 1 on each side of the boundary, so the substitution leaves no trace in it.
    assert _as_list(bucketed, "transceiver_increment") == [1, 1]
    # The same two rows, compared rather than bucketed.
    assert _as_list(compared, "transceiver_serial_changed") == [None, True]


@pytest.mark.gpu_and_cpu_mode
def test_a_change_across_any_gap_is_detected(config: Config):
    # No calendar is consulted, so more than a year between polls behaves exactly like a minute.
    meta = run(config, frame(["SN-AAA", "SN-BBB"], times=[0, 400 * DAY_NS]))

    assert _as_list(meta, "transceiver_serial_changed") == [None, True]


@pytest.mark.gpu_and_cpu_mode
def test_the_first_sample_answers_neither_question(config: Config):
    meta = run(config, frame(["SN-AAA"]))

    # A null, not False. A rule reading "did not change" must not count a port's first appearance as evidence of
    # stability that was never established.
    assert _as_list(meta, "transceiver_serial_changed") == [None]
    assert _as_list(meta, "transceiver_serial_first_seen") == [None]
    assert _as_list(meta, "transceiver_serial_distinct_count") == [1]


@pytest.mark.gpu_and_cpu_mode
def test_nulls_survive_as_nulls_not_false(config: Config):
    meta = run(config, frame(["SN-AAA", "SN-BBB"]))
    changed = meta.get_data("transceiver_serial_changed")

    assert int(changed.isna().sum()) == 1

    if (config.execution_mode == ExecutionMode.CPU):
        # Under pandas that takes the nullable boolean type; a plain object column would let a consumer coerce the
        # unanswerable first row into False.
        assert pd.api.types.is_extension_array_dtype(changed.dtype)


@pytest.mark.gpu_and_cpu_mode
def test_returning_to_a_previous_optic_is_a_change_but_not_new(config: Config):
    meta = run(config, frame(["SN-AAA", "SN-BBB", "SN-AAA"]))

    # Swapped out and swapped back is two substitutions, and the second is explained by the estate's own history.
    assert _as_list(meta, "transceiver_serial_changed") == [None, True, True]
    assert _as_list(meta, "transceiver_serial_first_seen") == [None, True, False]
    assert _as_list(meta, "transceiver_serial_distinct_count") == [1, 2, 2]


@pytest.mark.gpu_and_cpu_mode
def test_an_empty_cage_is_a_value(config: Config):
    meta = run(config, frame([None, None, "SN-AAA", None]))

    # Installing an optic and pulling it are both changes; a null is a physical state, not missing data.
    assert _as_list(meta, "transceiver_serial_changed") == [None, False, True, True]


@pytest.mark.gpu_and_cpu_mode
def test_identifiers_are_tracked_independently(config: Config):
    meta = run(config, frame(["SN-AAA", "SN-AAA"], neighbors=["chassis-a", "chassis-b"]))

    # Swapping the optic and re-cabling to a different neighbor are different events.
    assert _as_list(meta, "transceiver_serial_changed") == [None, False]
    assert _as_list(meta, "lldp_neighbor_chassis_id_changed") == [None, True]


@pytest.mark.gpu_and_cpu_mode
def test_ports_do_not_contaminate_each_other(config: Config):
    payload = {
        "entity_key": ["hq:sw1:Gi1/0/1", "hq:sw1:Gi1/0/2"] * 2,
        "event_time": [0, 0, DAY_NS, DAY_NS],
        "transceiver_serial": ["SN-AAA", "SN-BBB", "SN-AAA", "SN-BBB"],
        "lldp_neighbor_chassis_id": ["chassis-a"] * 4,
    }

    meta = run(config, payload)

    # Two ports legitimately hold different optics; only a change within one port is a substitution.
    assert _as_list(meta, "transceiver_serial_changed") == [None, None, False, False]


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    df_class = get_df_class(config.execution_mode)
    stage = TC1ChangeStage(config)

    def message(index: int, serial: str):
        return MessageMeta(df_class(frame([serial], times=[index * DAY_NS])))

    stage.on_data(message(0, "SN-AAA"))
    second = message(1, "SN-BBB")
    stage.on_data(second)

    # The comparison spans the message boundary, which is the whole reason this stage is stateful.
    assert _as_list(second, "transceiver_serial_changed") == [True]


@pytest.mark.gpu_and_cpu_mode
def test_out_of_order_sample_is_flagged_not_scored(config: Config):
    payload = frame(["SN-AAA", "SN-AAA", "SN-ZZZ", "SN-AAA"], times=[0, DAY_NS, 0, 2 * DAY_NS])

    meta = run(config, payload)

    # The late sample carries no verdict, and must not have become the state the next one is compared against.
    assert _as_list(meta, "transceiver_serial_changed") == [None, False, None, False]


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    message = ControlMessage()
    message.payload(MessageMeta(get_df_class(config.execution_mode)(frame(["SN-AAA", "SN-BBB"]))))

    TC1ChangeStage(config).on_data(message)

    assert _as_list(message.payload(), "transceiver_serial_changed") == [None, True]


@pytest.mark.gpu_and_cpu_mode
def test_tc1_change_stage_pipe(config: Config):
    source_df = get_df_class(config.execution_mode)(frame(["SN-AAA", "SN-AAA", "SN-BBB"]))

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC1ChangeStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "transceiver_serial_changed") == [None, False, True]


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = frame(["SN-AAA"])
    del payload["transceiver_serial"]

    with pytest.raises(KeyError, match="transceiver_serial"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError):
        TC1ChangeStage(config, novelty_columns=[])

    with pytest.raises(ValueError):
        TC1ChangeStage(config, max_values=0)
