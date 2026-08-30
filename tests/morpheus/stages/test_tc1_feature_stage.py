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
from morpheus.stages.telemetry.tc1_feature_stage import TC1FeatureStage
from morpheus.utils.counter_delta import NS_PER_SECOND
from morpheus.utils.type_utils import get_df_class

HOUR_NS = 3600 * NS_PER_SECOND
DAY_NS = 24 * HOUR_NS


def samples(entity_keys: list, serials: list, neighbors: list, times: list = None) -> dict:
    """Build a well-formed TC-1 frame: one collector, strictly increasing sequence numbers."""
    count = len(entity_keys)
    times = [index * HOUR_NS for index in range(count)] if times is None else times

    return {
        "entity_key": entity_keys,
        "event_time": times,
        "collector_id": ["poller-1"] * count,
        "collector_seq": list(range(count)),
        "transceiver_serial": serials,
        "lldp_neighbor_chassis_id": neighbors,
    }


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return series.tolist()


def run(config: Config, frame: dict, **kwargs) -> MessageMeta:
    meta = MessageMeta(get_df_class(config.execution_mode)(frame))

    return TC1FeatureStage(config, **kwargs).on_data(meta)


def test_execution_modes(config: Config):
    assert issubclass(TC1FeatureStage, GpuAndCpuMixin)

    assert set(TC1FeatureStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    needed = TC1FeatureStage(config).get_needed_columns()

    assert needed["transceiver_increment"] == TypeId.INT64
    assert needed["lldp_neighbor_increment"] == TypeId.INT64


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(TC1FeatureStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.gpu_and_cpu_mode
def test_steady_state_never_increments(config: Config):
    # A port that reports the same optic and the same neighbor on every poll is the normal case, and normal has to
    # read as 1 or every port on the estate is an alert.
    result = run(config, samples(["hq:sw1:Gi1/0/1"] * 4, ["SN-AAA"] * 4, ["chassis-a"] * 4))

    assert _as_list(result, "transceiver_increment") == [1, 1, 1, 1]
    assert _as_list(result, "lldp_neighbor_increment") == [1, 1, 1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_transceiver_substitution_increments(config: Config):
    result = run(config, samples(["hq:sw1:Gi1/0/1"] * 4, ["SN-AAA", "SN-AAA", "SN-BBB", "SN-BBB"], ["chassis-a"] * 4))

    # The count rises at the swap and stays up: the port has seen two distinct optics today, which remains true for
    # the rest of the period.
    assert _as_list(result, "transceiver_increment") == [1, 1, 2, 2]
    # Swapping an optic does not change who is on the far end.
    assert _as_list(result, "lldp_neighbor_increment") == [1, 1, 1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_neighbor_change_increments_independently(config: Config):
    result = run(config, samples(["hq:sw1:Gi1/0/1"] * 3, ["SN-AAA"] * 3, ["chassis-a", "chassis-b", "chassis-a"]))

    # Returning to the original neighbor does not decrement: two distinct values have still been seen.
    assert _as_list(result, "lldp_neighbor_increment") == [1, 2, 2]
    assert _as_list(result, "transceiver_increment") == [1, 1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_ports_do_not_contaminate_each_other(config: Config):
    # Every port on a switch has its own optic. Grouped any coarser than the port, this feature would report a
    # forty-port switch as forty substitutions.
    frame = samples(["hq:sw1:Gi1/0/1", "hq:sw1:Gi1/0/2", "hq:sw1:Gi1/0/1", "hq:sw1:Gi1/0/2"],
                    ["SN-AAA", "SN-BBB", "SN-AAA", "SN-BBB"], ["chassis-a", "chassis-b", "chassis-a", "chassis-b"])

    result = run(config, frame)

    assert _as_list(result, "transceiver_increment") == [1, 1, 1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_an_empty_cage_counts_as_a_distinct_state(config: Config):
    # No transceiver is a real physical state, not missing data, so installing one is a change worth counting.
    result = run(config, samples(["hq:sw1:Gi1/0/1"] * 3, [None, None, "SN-AAA"], ["chassis-a"] * 3))

    assert _as_list(result, "transceiver_increment") == [1, 1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_features_are_invariant_to_input_order(config: Config):
    frame = samples(["hq:sw1:Gi1/0/1"] * 5, ["SN-AAA", "SN-AAA", "SN-BBB", "SN-BBB", "SN-BBB"], ["chassis-a"] * 5)
    shuffled = {column: [values[index] for index in (3, 0, 4, 2, 1)] for (column, values) in frame.items()}

    in_order = run(config, frame)
    out_of_order = run(config, shuffled)

    # This is the property the sort exists for. Both batches carry the same rows, so both must produce the same
    # features in the same order, whatever arrangement they arrived in.
    assert _as_list(in_order, "transceiver_increment") == _as_list(out_of_order, "transceiver_increment")
    assert _as_list(in_order, "collector_seq") == _as_list(out_of_order, "collector_seq") == [0, 1, 2, 3, 4]


def test_without_the_sort_a_permuted_batch_scores_differently():
    # The negative control for the test above. Applying the same schema to the same rows in a different order, with
    # no total order imposed, silently produces different features. Nothing here raises; the output just answers a
    # different question. That is the defect the stage's sort removes.
    from morpheus.utils.column_info import process_dataframe
    from morpheus.utils.tc1_features import build_tc1_feature_schema

    schema = build_tc1_feature_schema(timestamp_column="event_time")
    frame = pd.DataFrame(
        samples(["hq:sw1:Gi1/0/1"] * 5, ["SN-AAA", "SN-AAA", "SN-BBB", "SN-BBB", "SN-BBB"], ["chassis-a"] * 5))
    frame["event_time"] = pd.to_datetime(frame["event_time"], unit="ns")

    in_order = process_dataframe(frame, schema)
    permuted = process_dataframe(frame.iloc[[3, 0, 4, 2, 1]].reset_index(drop=True), schema)

    assert in_order["transceiver_increment"].tolist() == [1, 1, 2, 2, 2]
    assert permuted["transceiver_increment"].tolist() != [1, 1, 2, 2, 2]


@pytest.mark.gpu_and_cpu_mode
def test_period_boundary_is_a_blind_spot(config: Config):
    # An honest test of a known limitation rather than a claim of coverage. The count resets at the period boundary,
    # so a swap between the last poll of one day and the first of the next is one distinct value on each day and is
    # invisible to this feature. Detecting it needs a period longer than the gap, or state carried across days.
    frame = samples(["hq:sw1:Gi1/0/1"] * 2, ["SN-AAA", "SN-BBB"], ["chassis-a"] * 2,
                    times=[23 * HOUR_NS, DAY_NS + HOUR_NS])

    result = run(config, frame)

    assert _as_list(result, "transceiver_increment") == [1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_the_envelope_timestamp_is_not_rewritten(config: Config):
    # The period needs a datetime, but the envelope specifies nanoseconds in the pipeline. Deriving the period into
    # a private column keeps that contract instead of converting the column everything downstream reads.
    frame = samples(["hq:sw1:Gi1/0/1"] * 2, ["SN-AAA"] * 2, ["chassis-a"] * 2)

    result = run(config, frame)

    assert _as_list(result, "event_time") == [0, HOUR_NS]
    assert "_tc1_period_time" not in result.get_column_names()


@pytest.mark.gpu_and_cpu_mode
def test_other_columns_survive(config: Config):
    frame = samples(["hq:sw1:Gi1/0/1"] * 2, ["SN-AAA"] * 2, ["chassis-a"] * 2)
    frame["crc_errors_delta"] = [4, 9]

    result = run(config, frame)

    assert _as_list(result, "crc_errors_delta") == [4, 9]
    assert result.count == 2


@pytest.mark.gpu_and_cpu_mode
def test_tc1_feature_stage_pipe(config: Config):
    frame = samples(["hq:sw1:Gi1/0/1"] * 3, ["SN-AAA", "SN-BBB", "SN-BBB"], ["chassis-a"] * 3)
    source_df = get_df_class(config.execution_mode)(frame)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC1FeatureStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "transceiver_increment") == [1, 2, 2]


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    frame = samples(["hq:sw1:Gi1/0/1"] * 2, ["SN-AAA", "SN-BBB"], ["chassis-a"] * 2)
    message = ControlMessage()
    message.payload(MessageMeta(get_df_class(config.execution_mode)(frame)))

    result = TC1FeatureStage(config).on_data(message)

    assert _as_list(result.payload(), "transceiver_increment") == [1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    frame = samples(["hq:sw1:Gi1/0/1"], ["SN-AAA"], ["chassis-a"])
    del frame["lldp_neighbor_chassis_id"]

    with pytest.raises(KeyError, match="lldp_neighbor_chassis_id"):
        run(config, frame)


@pytest.mark.gpu_and_cpu_mode
def test_tied_order_columns_raise(config: Config):
    frame = samples(["hq:sw1:Gi1/0/1"] * 2, ["SN-AAA", "SN-BBB"], ["chassis-a"] * 2, times=[0, 0])
    frame["collector_seq"] = [1, 1]

    with pytest.raises(ValueError, match="tied"):
        run(config, frame)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError):
        TC1FeatureStage(config, order_columns=[])

    with pytest.raises(ValueError):
        TC1FeatureStage(config, period="")

    with pytest.raises(ValueError, match="identical"):
        TC1FeatureStage(config, transceiver_column="same", neighbor_column="same")
