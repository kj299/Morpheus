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

import math

import pandas as pd
import pytest

from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc1_normalize_stage import TC1NormalizeStage
from morpheus.stages.telemetry.tc1_rate_stage import TC1RateStage
from morpheus.utils.bucket_peak import NS_PER_SECOND
from morpheus.utils.type_utils import get_df_class

MINUTE_NS = 60 * NS_PER_SECOND
PERIOD_NS = 5 * MINUTE_NS
PORT = "hq:sw1:Gi1/0/1"
TEN_GIGABIT = 10_000_000_000


def frame(errors: list, octets_in: list = None, octets_out: list = None, times: list = None, **columns) -> dict:
    """One port's polls, already normalized: deltas over sixty seconds, one poll per five-minute period."""
    count = len(errors)

    return {
        "entity_key": [PORT] * count,
        "event_time": [index * PERIOD_NS for index in range(count)] if times is None else times,
        "interval_seconds": [60.0] * count,
        "counter_reset": [False] * count,
        "sample_out_of_order": [False] * count,
        "crc_errors_delta": errors,
        "symbol_errors_delta": [0] * count,
        "input_discards_delta": [6] * count,
        "output_discards_delta": [0] * count,
        "if_hc_in_octets_delta": [750_000] * count if octets_in is None else octets_in,
        "if_hc_out_octets_delta": [375_000] * count if octets_out is None else octets_out,
        "link_speed_bps": [TEN_GIGABIT] * count,
        **columns,
    }


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [
        None if (value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))) else value
        for value in series.tolist()
    ]


def run(config: Config, payload: dict, **kwargs) -> MessageMeta:
    defaults = {"min_buckets": 4, "volume_seasonality": "none"}
    defaults.update(kwargs)
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC1RateStage(config, **defaults).on_data(meta)

    return meta


def test_execution_modes(config: Config):
    assert issubclass(TC1RateStage, GpuAndCpuMixin)

    assert set(TC1RateStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    needed = TC1RateStage(config).get_needed_columns()

    for name in ("error_rate",
                 "discard_rate",
                 "bits_in_per_second",
                 "bits_out_per_second",
                 "bits_per_second",
                 "utilization",
                 "error_rate_baseline_max",
                 "error_rate_step",
                 "bits_per_second_baseline_max",
                 "bits_per_second_baseline_min",
                 "bits_per_second_step"):
        assert needed[name] == TypeId.FLOAT64, name

    assert needed["error_rate_baseline_buckets"] == TypeId.INT64
    assert needed["bits_per_second_baseline_mature"] == TypeId.BOOL8


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(TC1RateStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.parametrize("kwargs, message",
                         [({
                             "volume_seasonality": "weekly"
                         }, "volume_seasonality"), ({
                             "error_columns": []
                         }, "error_columns"), ({
                             "bucket_seconds": 0
                         }, "bucket_seconds")])
def test_a_configuration_that_cannot_mean_anything_is_refused(config: Config, kwargs, message):
    with pytest.raises(ValueError, match=message):
        TC1RateStage(config, **kwargs)


@pytest.mark.gpu_and_cpu_mode
def test_a_rate_is_the_delta_over_the_interval_it_covers(config: Config):
    meta = run(config, frame([30, 60]))

    # Thirty CRC errors over sixty seconds, plus no symbol errors.
    assert _as_list(meta, "error_rate") == [0.5, 1.0]
    assert _as_list(meta, "discard_rate") == [0.1, 0.1]
    # Octets are bits once multiplied by eight, and the busier direction is what fills the link.
    assert _as_list(meta, "bits_in_per_second") == [100_000.0, 100_000.0]
    assert _as_list(meta, "bits_out_per_second") == [50_000.0, 50_000.0]
    assert _as_list(meta, "bits_per_second") == [150_000.0, 150_000.0]
    assert _as_list(meta, "utilization") == [1e-05, 1e-05]


@pytest.mark.gpu_and_cpu_mode
def test_a_reset_or_a_late_sample_is_not_a_rate(config: Config):
    payload = frame([4, 900, 4, 4])
    payload["counter_reset"] = [False, True, False, False]
    payload["sample_out_of_order"] = [False, False, True, False]
    meta = run(config, payload)

    # The reboot's delta covers the uptime, not a period the history can compare with; it would otherwise read as
    # a burst on every port of the switch.
    assert _as_list(meta, "error_rate") == [4 / 60, None, None, 4 / 60]
    assert _as_list(meta, "bits_per_second") == [150_000.0, None, None, 150_000.0]


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_counter_is_no_rate_rather_than_a_smaller_one(config: Config):
    meta = run(config, frame([4, None], octets_in=[750_000, None]))

    assert _as_list(meta, "error_rate") == [4 / 60, None]
    assert _as_list(meta, "bits_in_per_second") == [100_000.0, None]
    assert _as_list(meta, "bits_per_second") == [150_000.0, None]


@pytest.mark.gpu_and_cpu_mode
def test_a_climbing_error_rate_is_a_step_above_the_ports_own_history(config: Config):
    # Four quiet periods, then one with a hundred and twenty errors a minute.
    meta = run(config, frame([2, 3, 1, 3, 120]))

    assert _as_list(meta, "error_rate_baseline_mature") == [False, False, False, False, True]
    assert _as_list(meta, "error_rate_baseline_max")[4] == pytest.approx(3 / 60)
    assert _as_list(meta, "error_rate_step")[4] == pytest.approx(117 / 60)


@pytest.mark.gpu_and_cpu_mode
def test_a_port_that_always_ran_errors_is_measured_against_that(config: Config):
    meta = run(config, frame([90, 120, 100, 110, 115]))

    assert _as_list(meta, "error_rate_step")[4] < 0


@pytest.mark.gpu_and_cpu_mode
def test_volume_is_measured_against_its_peak_and_its_trough(config: Config):
    octets = [750_000, 700_000, 800_000, 760_000, 7_500_000, 0]
    meta = run(config, frame([0] * 6, octets_in=octets, octets_out=[0] * 6))

    assert _as_list(meta, "bits_per_second_baseline_max")[4] == pytest.approx(800_000 * 8 / 60)
    assert _as_list(meta, "bits_per_second_baseline_min")[4] == pytest.approx(700_000 * 8 / 60)
    assert _as_list(meta, "bits_per_second_step")[4] == pytest.approx((7_500_000 - 800_000) * 8 / 60)
    # The silent period is measured against a history the burst has now joined at the top and not at the bottom.
    assert _as_list(meta, "bits_per_second")[5] == 0.0
    assert _as_list(meta, "bits_per_second_baseline_min")[5] == pytest.approx(700_000 * 8 / 60)


@pytest.mark.gpu_and_cpu_mode
def test_a_port_that_has_been_idle_has_a_trough_of_zero(config: Config):
    meta = run(config, frame([0] * 5, octets_in=[0, 750_000, 0, 750_000, 0], octets_out=[0] * 5))

    assert _as_list(meta, "bits_per_second_baseline_min")[4] == 0.0


@pytest.mark.gpu_and_cpu_mode
def test_traffic_is_measured_against_the_same_hour_when_asked(config: Config):
    hour_ns = 3600 * NS_PER_SECOND
    day_ns = 24 * hour_ns
    # Nine o'clock on four days, busy, and three o'clock on the same four days, quiet; then a fifth three o'clock
    # carrying what nine o'clock always carries.
    times = [day * day_ns + 9 * hour_ns for day in range(4)] + [day * day_ns + 3 * hour_ns for day in range(5)]
    times = sorted(times)
    busy = {time: 6_000_000 for time in times if (time // hour_ns) % 24 == 9}
    octets = [busy.get(time, 600_000) for time in times]
    octets[-1] = 6_000_000
    payload = frame([0] * len(times), octets_in=octets, octets_out=[0] * len(times), times=times)

    seasonal = run(config, payload, volume_seasonality="hour_of_day")
    pooled = run(config, payload, volume_seasonality="none")

    # Against three o'clock, the fifth three o'clock is a step; against the whole day it is an ordinary morning.
    assert _as_list(seasonal, "bits_per_second_step")[-1] > 0
    assert _as_list(pooled, "bits_per_second_step")[-1] == 0.0


@pytest.mark.gpu_and_cpu_mode
def test_two_ports_keep_two_histories(config: Config):
    payload = frame([2, 2, 2, 2, 2, 2])
    payload["entity_key"] = [PORT, "hq:sw1:Gi1/0/2"] * 3
    payload["event_time"] = [0, 0, PERIOD_NS, PERIOD_NS, 2 * PERIOD_NS, 2 * PERIOD_NS]
    meta = run(config, payload, min_buckets=2)

    assert _as_list(meta, "error_rate_baseline_buckets") == [0, 0, 1, 1, 2, 2]


@pytest.mark.gpu_and_cpu_mode
def test_rates_compose_behind_the_normalize_stage(config: Config):
    df_class = get_df_class(config.execution_mode)
    raw = df_class({
        "site_id": ["hq"] * 3,
        "device_id": ["sw1"] * 3,
        "port_id": ["Gi1/0/1"] * 3,
        "event_time": [0, MINUTE_NS, 2 * MINUTE_NS],
        "crc_errors": [100, 130, 190],
        "symbol_errors": [0, 0, 0],
        "input_discards": [0, 0, 0],
        "output_discards": [0, 0, 0],
        "if_hc_in_octets": [0, 75_000, 150_000],
        "if_hc_out_octets": [0, 0, 0],
        "link_speed_bps": [TEN_GIGABIT] * 3,
    })

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[raw]))
    pipe.add_stage(TC1NormalizeStage(config))
    pipe.add_stage(TC1RateStage(config, min_buckets=1))
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    meta = sink.get_messages()[0]

    assert _as_list(meta, "error_rate") == [None, 0.5, 1.0]
    assert _as_list(meta, "bits_per_second") == [None, 10_000.0, 10_000.0]
