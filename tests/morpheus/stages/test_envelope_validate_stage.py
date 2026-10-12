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
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.envelope_validate_stage import CLOCK_OFFSET_EXCEEDED
from morpheus.stages.lineage.envelope_validate_stage import CLOCK_UNSYNCHRONIZED
from morpheus.stages.lineage.envelope_validate_stage import EnvelopeValidateStage
from morpheus.stages.lineage.envelope_validate_stage import parse_null_reasons
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.type_utils import get_df_class

SECOND_NS = 10**9
DELIVERY_NS = 2 * SECOND_NS


def delivered(observed: list) -> list:
    """A replay's ingest clock: two seconds after the sensor saw it."""
    return [None if value is None else value + DELIVERY_NS for value in observed]


def records(count: int = 2, **overrides) -> dict:
    """A complete envelope on every row; an override replaces one field's column."""
    payload = {
        "schema_version": ["TC-1/1.0.0"] * count,
        "collector_id": ["snmp-poller"] * count,
        "collector_seq": list(range(count)),
        "observed_time": [index * SECOND_NS for index in range(count)],
        "event_time": [index * SECOND_NS for index in range(count)],
        "clock_source": ["ntp_disciplined"] * count,
        "clock_offset_ms": [3] * count,
        "sampling_policy": ["full"] * count,
        "origin_hash": [f"{index:016x}" for index in range(count)],
        "tenant_id": ["corp"] * count,
    }
    payload.update(overrides)

    return payload


def _as_list(frame, column: str) -> list:
    series = frame[column]

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [None if value is None or value is pd.NA or value != value else value for value in series.tolist()]


def run(config: Config, payload: dict, **kwargs):
    kwargs.setdefault("ingest_clock", delivered)
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    result = EnvelopeValidateStage(config, **kwargs).on_data(meta)

    return result.copy_dataframe()


def test_execution_modes(config: Config):
    assert issubclass(EnvelopeValidateStage, GpuAndCpuMixin)

    assert set(EnvelopeValidateStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    needed = EnvelopeValidateStage(config).get_needed_columns()

    # Not declared: a declared column arrives preallocated with zeros, which would read as stamped at the epoch.
    assert "ingest_time" not in needed
    assert needed["envelope_valid"] == TypeId.BOOL8
    assert needed["envelope_defects"] == TypeId.STRING
    assert needed["clock_quarantined"] == TypeId.BOOL8
    assert needed["quarantine_reason"] == TypeId.STRING


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(EnvelopeValidateStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.parametrize("kwargs, message", [({
    "required_columns": []
}, "required_columns"), ({
    "max_offset_ms": -1
}, "max_offset_ms")])
def test_a_configuration_that_cannot_mean_anything_is_refused(config: Config, kwargs, message):
    with pytest.raises(ValueError, match=message):
        EnvelopeValidateStage(config, **kwargs)


@pytest.mark.gpu_and_cpu_mode
def test_a_complete_envelope_is_valid_and_gains_its_ingest_time(config: Config):
    result = run(config, records())

    assert _as_list(result, "envelope_valid") == [True, True]
    assert _as_list(result, "envelope_defects") == [None, None]
    assert _as_list(result, "clock_quarantined") == [False, False]
    assert _as_list(result, "quarantine_reason") == [None, None]
    assert _as_list(result, "ingest_time") == [DELIVERY_NS, SECOND_NS + DELIVERY_NS]


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_field_is_a_defect_and_a_stated_null_is_not(config: Config):
    payload = records(sampling_policy=[None, None])
    payload["envelope_null_reasons"] = [None, "sampling_policy=not_reported_by_source"]
    result = run(config, payload)

    # The rule the guide states: unavailable is explicitly null with a reason, never a plausible default.
    assert _as_list(result, "envelope_valid") == [False, True]
    assert _as_list(result, "envelope_defects") == ["sampling_policy", None]


@pytest.mark.gpu_and_cpu_mode
def test_a_field_the_frame_does_not_carry_at_all_is_a_defect_on_every_row(config: Config):
    payload = records()
    del payload["tenant_id"]
    result = run(config, payload)

    assert _as_list(result, "envelope_defects") == ["tenant_id", "tenant_id"]


@pytest.mark.gpu_and_cpu_mode
def test_values_outside_the_guides_vocabulary_are_defects(config: Config):
    result = run(config, records(clock_source=["gps", "ptp"], sampling_policy=["full", "most"]))

    assert _as_list(result, "envelope_defects") == ["clock_source:invalid", "sampling_policy:invalid"]


@pytest.mark.gpu_and_cpu_mode
def test_every_sampling_shape_the_guide_names_is_accepted(config: Config):
    result = run(config, records(3, sampling_policy=["full", "1:100", "adaptive:target=5000"]))

    assert _as_list(result, "envelope_valid") == [True, True, True]


@pytest.mark.gpu_and_cpu_mode
def test_an_unsynchronized_clock_or_one_beyond_the_bound_is_quarantined(config: Config):
    payload = records(4,
                      clock_source=["unsynchronized", "ntp_disciplined", "ntp_disciplined", "ptp"],
                      clock_offset_ms=[0, 1000, -1001, 5000])
    result = run(config, payload)

    # A thousand milliseconds is the bound and is inside it; one more either side of true is not.
    assert _as_list(result, "clock_quarantined") == [True, False, True, True]
    assert _as_list(result,
                    "quarantine_reason") == [CLOCK_UNSYNCHRONIZED, None, CLOCK_OFFSET_EXCEEDED, CLOCK_OFFSET_EXCEEDED]


@pytest.mark.gpu_and_cpu_mode
def test_an_offset_nobody_measured_is_not_a_clock_known_to_be_wrong(config: Config):
    payload = records(clock_offset_ms=[None, None])
    payload["envelope_null_reasons"] = ["clock_offset_ms=not_measured", None]
    result = run(config, payload)

    assert _as_list(result, "clock_quarantined") == [False, False]
    assert _as_list(result, "envelope_valid") == [True, False]


@pytest.mark.gpu_and_cpu_mode
def test_with_a_route_the_quarantined_records_leave_the_stream_and_are_kept(config: Config):
    held = []
    payload = records(3, clock_source=["ntp_disciplined", "unsynchronized", "ntp_disciplined"])
    result = run(config, payload, on_quarantine=held.append)

    assert _as_list(result, "collector_seq") == [0, 2]
    assert len(held) == 1
    assert list(held[0]["collector_seq"]) == [1]
    assert list(held[0]["quarantine_reason"]) == [CLOCK_UNSYNCHRONIZED]


@pytest.mark.gpu_and_cpu_mode
def test_without_a_route_nothing_is_lost(config: Config):
    result = run(config, records(clock_source=["unsynchronized", "ptp"]))

    assert _as_list(result, "collector_seq") == [0, 1]
    assert _as_list(result, "clock_quarantined") == [True, False]


@pytest.mark.gpu_and_cpu_mode
def test_event_time_defaults_to_observed_time(config: Config):
    result = run(config, records(event_time=[None, 7 * SECOND_NS]))

    assert _as_list(result, "event_time") == [0, 7 * SECOND_NS]
    assert _as_list(result, "envelope_valid") == [True, True]


@pytest.mark.gpu_and_cpu_mode
def test_an_ingest_time_the_source_stamped_is_kept(config: Config):
    result = run(config, records(ingest_time=[5 * SECOND_NS, None]))

    assert _as_list(result, "ingest_time") == [5 * SECOND_NS, SECOND_NS + DELIVERY_NS]


@pytest.mark.cpu_mode
def test_without_a_replay_clock_ingest_time_is_the_wall_clock(config: Config):
    import time

    before = time.time_ns()
    meta = MessageMeta(pd.DataFrame(records()))
    result = EnvelopeValidateStage(config).on_data(meta).copy_dataframe()

    assert all(value >= before for value in result["ingest_time"])


def test_null_reasons_are_codes_and_anything_else_is_ignored():
    assert parse_null_reasons("clock_offset_ms=not_measured; sampling_policy=unknown") == {
        "clock_offset_ms": "not_measured", "sampling_policy": "unknown"
    }
    assert not parse_null_reasons("tenant_id=Somebody forgot")
    assert not parse_null_reasons("=orphan;nonsense")
    assert not parse_null_reasons(None)


@pytest.mark.gpu_and_cpu_mode
def test_it_composes_in_a_pipeline_and_routes_there_too(config: Config):
    held = []
    df_class = get_df_class(config.execution_mode)
    sources = [
        df_class(records(2, clock_source=["unsynchronized", "ptp"])),
        df_class(records(1, clock_offset_ms=[9000])),
    ]

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=sources))
    pipe.add_stage(EnvelopeValidateStage(config, ingest_clock=delivered, on_quarantine=held.append))
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    messages = sink.get_messages()

    assert sum(message.count for message in messages) == 1
    assert sum(len(frame) for frame in held) == 2

    # Composed, the stamp is the replay clock's, which the stage called directly would not show: a pipeline
    # preallocates the columns a stage declares, and a preallocated zero must not read as already stamped.
    stamped = [value for message in messages for value in _as_list(message.copy_dataframe(), "ingest_time")]

    assert stamped == [SECOND_NS + DELIVERY_NS]
