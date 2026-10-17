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
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.stages.telemetry.tc1_device_stage import TC1DeviceStage
from morpheus.stages.telemetry.tc1_device_stage import in_change_window
from morpheus.stages.telemetry.tc1_device_stage import parse_change_window
from morpheus.utils.counter_delta import TIMETICKS_CEILING_NS
from morpheus.utils.type_utils import get_df_class

NS_PER_SECOND = 10**9
MINUTE_NS = 60 * NS_PER_SECOND
CS_PER_SECOND = 100
BOOT_CS = 3600 * CS_PER_SECOND

YEARS_AHEAD_NS = 10 * 365 * 86400 * NS_PER_SECOND
"""A device clock wrong by a decade: the failure an unset RTC produces, far beyond any gap a real feed has."""


def uptimes(minutes: int, reboots: tuple = ()) -> list:
    """A device polled every minute, up an hour at the first poll and restarting twenty seconds before each reboot."""
    values = []
    booted_at = None

    for minute in range(minutes):
        if (minute in reboots):
            booted_at = minute

        if (booted_at is None):
            values.append(BOOT_CS + minute * 60 * CS_PER_SECOND)
        else:
            values.append((minute - booted_at) * 60 * CS_PER_SECOND + 20 * CS_PER_SECOND)

    return values


def frame(minutes: int, reboots: tuple = (), device: str = "sw1", **columns) -> dict:
    payload = {
        "site_id": ["hq"] * minutes,
        "device_id": [device] * minutes,
        "event_time": [minute * MINUTE_NS for minute in range(minutes)],
        "sys_uptime": uptimes(minutes, reboots),
    }
    payload.update(columns)

    return payload


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [None if pd.isna(value) else value for value in series.tolist()]


def run(config: Config, payload: dict, stage: TC1DeviceStage = None, **kwargs) -> MessageMeta:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    (stage or TC1DeviceStage(config, **kwargs)).on_data(meta)

    return meta


@pytest.mark.gpu_and_cpu_mode
def test_execution_modes(config: Config):
    assert issubclass(TC1DeviceStage, GpuAndCpuMixin)

    assert set(TC1DeviceStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


@pytest.mark.gpu_and_cpu_mode
def test_needed_columns(config: Config):
    needed = TC1DeviceStage(config).get_needed_columns()

    assert needed["device_key"] == TypeId.STRING
    assert needed["device_rebooted"] == TypeId.BOOL8
    assert needed["device_reboot_rising_buckets"] == TypeId.INT64
    assert needed["config_change_in_window"] == TypeId.BOOL8
    assert needed["route_count_delta"] == TypeId.INT64
    assert needed["clock_refused"] == TypeId.BOOL8


@pytest.mark.gpu_and_cpu_mode
def test_the_device_key_is_site_and_device(config: Config):
    meta = run(config, frame(2))

    assert _as_list(meta, "device_key") == ["hq:sw1", "hq:sw1"]


@pytest.mark.gpu_and_cpu_mode
def test_a_restart_is_read_from_the_uptime(config: Config):
    meta = run(config, frame(5, reboots=(3, )))

    # The first poll establishes the uptime; the restart is the poll whose uptime went backwards.
    assert _as_list(meta, "device_rebooted") == [None, False, False, True, False]


@pytest.mark.gpu_and_cpu_mode
def test_an_uptime_shorter_than_the_gap_is_a_restart_even_when_it_advanced(config: Config):
    # Polled an hour apart, the device was up ten minutes at the second poll and five at the first: it advanced,
    # and it still came up in between.
    payload = {
        "site_id": ["hq", "hq"],
        "device_id": ["sw1", "sw1"],
        "event_time": [0, 3600 * NS_PER_SECOND],
        "sys_uptime": [300 * CS_PER_SECOND, 600 * CS_PER_SECOND],
    }

    assert _as_list(run(config, payload), "device_rebooted") == [None, True]


@pytest.mark.gpu_and_cpu_mode
def test_the_timeticks_ceiling_is_not_a_restart(config: Config):
    ceiling_cs = TIMETICKS_CEILING_NS // (10 * 10**6)
    payload = {
        "site_id": ["hq", "hq"],
        "device_id": ["sw1", "sw1"],
        "event_time": [0, MINUTE_NS],
        "sys_uptime": [ceiling_cs - 30 * CS_PER_SECOND, 30 * CS_PER_SECOND],
    }

    assert _as_list(run(config, payload), "device_rebooted") == [None, False]
    assert _as_list(run(config, payload, uptime_unit="s"), "device_rebooted") == [None, True]


@pytest.mark.gpu_and_cpu_mode
def test_restarts_are_counted_per_period(config: Config):
    meta = run(config, frame(10, reboots=(2, 4, 7)), bucket_seconds=300, min_buckets=1)

    assert _as_list(meta, "device_reboots_in_bucket") == [0, 0, 1, 1, 2, 0, 0, 1, 1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_a_rising_run_counts_periods_each_with_more_restarts_than_the_last(config: Config):
    # One restart in the second period, two in the third, three in the fourth: a trajectory of three.
    meta = run(config, frame(20, reboots=(6, 10, 12, 15, 17, 19)), bucket_seconds=300, min_buckets=1)
    rising = _as_list(meta, "device_reboot_rising_buckets")

    assert rising[:5] == [0] * 5
    assert rising[6] == 1
    assert rising[10] == 0 and rising[12] == 2
    assert rising[15] == 0 and rising[17] == 0 and rising[19] == 3


@pytest.mark.gpu_and_cpu_mode
def test_a_period_with_no_poll_ends_the_run(config: Config):
    stage = TC1DeviceStage(config, bucket_seconds=60, min_buckets=1)
    times = [0, 1, 2, 4]
    payload = {
        "site_id": ["hq"] * 4,
        "device_id": ["sw1"] * 4,
        "event_time": [minute * MINUTE_NS for minute in times],
        # Restarts at the second and third polls, then a period with no poll before the fourth restarts again.
        "sys_uptime": [BOOT_CS, 20 * CS_PER_SECOND, 20 * CS_PER_SECOND, 20 * CS_PER_SECOND],
    }

    assert _as_list(run(config, payload, stage=stage), "device_reboot_rising_buckets") == [0, 1, 0, 1]


@pytest.mark.gpu_and_cpu_mode
def test_the_baseline_is_the_most_any_earlier_period_held(config: Config):
    meta = run(config, frame(20, reboots=(6, 10, 12)), bucket_seconds=300, min_buckets=2)

    assert _as_list(meta, "device_reboot_baseline_mature")[:5] == [False] * 5
    assert _as_list(meta, "device_reboot_baseline")[10] == 1
    assert _as_list(meta, "device_reboot_baseline")[15] == 2


@pytest.mark.gpu_and_cpu_mode
def test_a_configuration_change_and_whether_it_is_new(config: Config):
    hashes = ["a", "a", "b", "b", "a"]
    meta = run(config, frame(5, running_config_hash=hashes))

    assert _as_list(meta, "config_changed") == [None, False, True, False, True]
    assert _as_list(meta, "config_first_seen") == [None, False, True, False, False]


@pytest.mark.gpu_and_cpu_mode
def test_a_change_window_says_where_a_change_landed(config: Config):
    hashes = ["a", "a", "b", "b", "b", "c"]
    meta = run(config, frame(6, running_config_hash=hashes), change_window=["00:00", "00:03"])

    # Only a change is placed in the window; a poll that changed nothing has nothing to place.
    assert _as_list(meta, "config_change_in_window") == [None, None, True, None, None, False]


@pytest.mark.gpu_and_cpu_mode
def test_without_a_window_nothing_is_placed(config: Config):
    meta = run(config, frame(3, running_config_hash=["a", "b", "c"]))

    assert _as_list(meta, "config_change_in_window") == [None, None, None]


@pytest.mark.parametrize(
    ("window", "minute_of_day", "inside"),
    [(("22:00", "02:00"), 23 * 60, True), (("22:00", "02:00"), 60, True), (("22:00", "02:00"), 2 * 60, False),
     (("01:00", "02:00"), 60, True), (("01:00", "02:00"), 2 * 60, False)])
def test_a_window_may_wrap_midnight(window: tuple, minute_of_day: int, inside: bool):
    event_time_ns = (3 * 86400 + minute_of_day * 60) * NS_PER_SECOND

    assert in_change_window(event_time_ns, parse_change_window(list(window))) is inside


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("window", [["00:00"], ["00:00", "00:00"], ["24:00", "01:00"], ["noon", "01:00"], "00:00"])
def test_an_invalid_window_is_refused(config: Config, window):
    with pytest.raises(ValueError, match="change_window"):
        TC1DeviceStage(config, change_window=window)


@pytest.mark.gpu_and_cpu_mode
def test_route_count_delta(config: Config):
    meta = run(config, frame(4, route_count=[1200, 1200, None, 400]))

    # A poll that did not report its table carries no delta, and the next is measured against the last one that did.
    assert _as_list(meta, "route_count_delta") == [None, 0, None, -800]


@pytest.mark.gpu_and_cpu_mode
def test_devices_are_separate(config: Config):
    first = frame(3, reboots=(2, ), device="sw1")
    second = frame(3, device="sw2")
    payload = {key: [value for pair in zip(column, second[key]) for value in pair] for (key, column) in first.items()}

    meta = run(config, payload)

    assert _as_list(meta, "device_key") == ["hq:sw1", "hq:sw2"] * 3
    assert _as_list(meta, "device_rebooted") == [None, None, False, False, True, False]


@pytest.mark.gpu_and_cpu_mode
def test_state_carries_across_messages(config: Config):
    stage = TC1DeviceStage(config)
    payload = frame(4, reboots=(3, ), running_config_hash=["a", "a", "a", "b"])
    head = {name: values[:2] for (name, values) in payload.items()}
    tail = {name: values[2:] for (name, values) in payload.items()}

    run(config, head, stage=stage)
    meta = run(config, tail, stage=stage)

    assert _as_list(meta, "device_rebooted") == [False, True]
    assert _as_list(meta, "config_changed") == [False, True]


@pytest.mark.gpu_and_cpu_mode
def test_a_late_poll_carries_nothing_and_moves_nothing(config: Config):
    payload = frame(3, running_config_hash=["a", "b", "c"])
    payload = {name: [values[0], values[2], values[1]] for (name, values) in payload.items()}

    meta = run(config, payload)

    assert _as_list(meta, "config_changed") == [None, True, None]
    assert _as_list(meta, "device_rebooted") == [None, False, None]


@pytest.mark.gpu_and_cpu_mode
def test_a_keyless_poll_carries_nothing(config: Config):
    payload = frame(2)
    payload["site_id"] = [None, "hq"]

    meta = run(config, payload)

    assert _as_list(meta, "device_key") == [None, "hq:sw1"]
    assert _as_list(meta, "device_reboots_in_bucket") == [None, 0]


@pytest.mark.gpu_and_cpu_mode
def test_a_poll_from_the_future_is_refused(config: Config):
    payload = frame(3, running_config_hash=["a", "a", "b"])
    payload["event_time"][1] += YEARS_AHEAD_NS

    meta = run(config, payload)

    assert _as_list(meta, "clock_refused") == [False, True, False]
    assert _as_list(meta, "config_changed") == [None, None, True]


@pytest.mark.gpu_and_cpu_mode
def test_without_an_uptime_no_restart_is_read(config: Config):
    payload = frame(3)
    del payload["sys_uptime"]

    meta = run(config, payload)

    assert _as_list(meta, "device_rebooted") == [None] * 3
    assert _as_list(meta, "device_reboots_in_bucket") == [None] * 3


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("missing", ["site_id", "device_id", "event_time"])
def test_missing_required_column(config: Config, missing: str):
    payload = frame(2)
    del payload[missing]

    with pytest.raises(KeyError, match=missing):
        run(config, payload)
