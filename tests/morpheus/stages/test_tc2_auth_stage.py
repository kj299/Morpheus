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
from morpheus.stages.telemetry.tc2_auth_stage import TC2AuthStage
from morpheus.utils.session_timer import NS_PER_SECOND
from morpheus.utils.type_utils import get_df_class


def frame(results: list, times: list = None, ports: list = None) -> dict:
    count = len(results)

    return {
        "site_id": ["hq"] * count,
        "switch_id": ["sw1"] * count,
        "port_id": ["Gi1/0/1"] * count if ports is None else ports,
        "dot1x_result": results,
        "event_time": [index * NS_PER_SECOND for index in range(count)] if times is None else times,
    }


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [None if value is None or value is pd.NA or value != value else value for value in series.tolist()]


def run(config: Config, payload: dict, **kwargs) -> MessageMeta:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC2AuthStage(config, **kwargs).on_data(meta)

    return meta


def test_execution_modes(config: Config):
    assert issubclass(TC2AuthStage, GpuAndCpuMixin)

    assert set(TC2AuthStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    needed = TC2AuthStage(config).get_needed_columns()

    assert needed["auth_elapsed_seconds"] == TypeId.FLOAT64
    assert needed["auth_attempts"] == TypeId.INT64
    assert needed["auth_unpaired"] == TypeId.BOOL8


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(TC2AuthStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.gpu_and_cpu_mode
def test_a_paired_exchange_is_timed(config: Config):
    meta = run(config, frame(["started", "success"], times=[0, 3 * NS_PER_SECOND]))

    assert _as_list(meta, "auth_elapsed_seconds") == [None, 3.0]
    assert _as_list(meta, "auth_attempts") == [None, 1]
    assert _as_list(meta, "auth_unpaired") == [None, False]


@pytest.mark.gpu_and_cpu_mode
def test_a_start_row_carries_nulls_not_zeros(config: Config):
    # A start establishes what the outcome will be measured against; it is not itself an authorization. A zero
    # here would match a rule looking for "authorized instantly".
    meta = run(config, frame(["started"]))

    assert _as_list(meta, "auth_elapsed_seconds") == [None]
    assert _as_list(meta, "auth_unpaired") == [None]


@pytest.mark.gpu_and_cpu_mode
def test_authorization_with_no_exchange_is_flagged(config: Config):
    # The bypass signal. MAC authentication bypass and a device bridged behind an authorized supplicant both look
    # like this from the switch.
    meta = run(config, frame(["success"]))

    assert _as_list(meta, "auth_unpaired") == [True]
    assert _as_list(meta, "auth_elapsed_seconds") == [None]


@pytest.mark.gpu_and_cpu_mode
def test_a_null_result_starts_the_clock(config: Config):
    # A source that emits the request with no result column populated still opens an exchange.
    meta = run(config, frame([None, "success"], times=[0, 2 * NS_PER_SECOND]))

    assert _as_list(meta, "auth_elapsed_seconds") == [None, 2.0]
    assert _as_list(meta, "auth_unpaired") == [None, False]


@pytest.mark.gpu_and_cpu_mode
def test_retries_are_counted(config: Config):
    meta = run(config, frame(["started", "started", "started", "success"], times=[0, 5, 8, 9]))

    # A success after three attempts is not a first-time one, and timing from the last attempt alone would hide
    # the two before it.
    assert _as_list(meta, "auth_attempts") == [None, None, None, 3]


@pytest.mark.gpu_and_cpu_mode
def test_a_failure_is_timed_like_any_other_outcome(config: Config):
    meta = run(config, frame(["started", "failure"], times=[0, 4 * NS_PER_SECOND]))

    assert _as_list(meta, "auth_elapsed_seconds") == [None, 4.0]
    assert _as_list(meta, "auth_unpaired") == [None, False]


@pytest.mark.gpu_and_cpu_mode
def test_ports_are_timed_separately(config: Config):
    payload = frame(["started", "started", "success", "success"],
                    times=[0, 0, 2 * NS_PER_SECOND, 9 * NS_PER_SECOND],
                    ports=["Gi1/0/1", "Gi1/0/2", "Gi1/0/1", "Gi1/0/2"])

    meta = run(config, payload)

    assert _as_list(meta, "auth_elapsed_seconds") == [None, None, 2.0, 9.0]
    assert _as_list(meta, "auth_port_key")[0] == "hq:sw1:Gi1/0/1"


@pytest.mark.gpu_and_cpu_mode
def test_pending_values_are_configurable(config: Config):
    meta = run(config, frame(["radius-request", "accept"], times=[0, NS_PER_SECOND]), pending_values=["radius-request"])

    assert _as_list(meta, "auth_elapsed_seconds") == [None, 1.0]


@pytest.mark.gpu_and_cpu_mode
def test_an_outcome_before_its_start_is_flagged_not_negated(config: Config):
    payload = frame(["started", "success"], times=[10 * NS_PER_SECOND, 2 * NS_PER_SECOND])

    meta = run(config, payload)

    # A naive subtraction would report an authorization that took minus eight seconds.
    assert _as_list(meta, "auth_elapsed_seconds") == [None, None]


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    df_class = get_df_class(config.execution_mode)
    stage = TC2AuthStage(config)

    stage.on_data(MessageMeta(df_class(frame(["started"], times=[0]))))
    second = MessageMeta(df_class(frame(["success"], times=[6 * NS_PER_SECOND])))
    stage.on_data(second)

    # The exchange spans the message boundary, which is the whole reason this stage is stateful.
    assert _as_list(second, "auth_elapsed_seconds") == [6.0]
    assert _as_list(second, "auth_unpaired") == [False]


@pytest.mark.gpu_and_cpu_mode
def test_an_open_exchange_is_still_pending(config: Config):
    stage = TC2AuthStage(config)
    stage.on_data(MessageMeta(get_df_class(config.execution_mode)(frame(["started"]))))

    assert stage.pending_count == 1


@pytest.mark.gpu_and_cpu_mode
def test_nulls_survive_as_nulls_not_false(config: Config):
    meta = run(config, frame(["started", "success"], times=[0, NS_PER_SECOND]))
    unpaired = meta.get_data("auth_unpaired")

    # The start row answers neither question. Coerced to False it would read as "an exchange we confirmed was
    # properly paired", which nobody established.
    assert int(unpaired.isna().sum()) == 1

    if (config.execution_mode == ExecutionMode.CPU):
        assert pd.api.types.is_extension_array_dtype(unpaired.dtype)


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    message = ControlMessage()
    payload = frame(["started", "success"], times=[0, 2 * NS_PER_SECOND])
    message.payload(MessageMeta(get_df_class(config.execution_mode)(payload)))

    TC2AuthStage(config).on_data(message)

    assert _as_list(message.payload(), "auth_elapsed_seconds") == [None, 2.0]


@pytest.mark.gpu_and_cpu_mode
def test_tc2_auth_stage_pipe(config: Config):
    payload = frame(["started", "success", "success"], times=[0, 2 * NS_PER_SECOND, 4 * NS_PER_SECOND])
    source_df = get_df_class(config.execution_mode)(payload)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC2AuthStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    # The second success has no exchange left in front of it, which is the bypass shape.
    assert _as_list(messages[0], "auth_unpaired") == [None, False, True]


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = frame(["started"])
    del payload["dot1x_result"]

    with pytest.raises(KeyError, match="dot1x_result"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError):
        TC2AuthStage(config, timeout_seconds=0)


@pytest.mark.gpu_and_cpu_mode
def test_a_null_port_carries_no_timing_and_pairs_with_nothing(config: Config):
    # A start with no port followed by a success with no port. Keyed on the string "None:..." the second would be
    # timed against the first, pairing an exchange nobody can locate.
    payload = frame(["started", "success"], times=[0, NS_PER_SECOND], ports=[None, None])
    meta = run(config, payload)

    assert _as_list(meta, "auth_port_key") == [None, None]
    assert _as_list(meta, "auth_elapsed_seconds") == [None, None]
    assert _as_list(meta, "auth_unpaired") == [None, None]


@pytest.mark.cpu_mode
def test_a_stale_exchange_does_not_swallow_a_later_bypass(config: Config):
    # An exchange that starts and never finishes, then a success far past the timeout. The success has no
    # authentication in front of it, which is what a MAB bypass looks like from the switch. If the abandoned
    # exchange were still pending it would pair with this outcome and the bypass would read as authorized.
    timeout = 300
    times = [0, (timeout + 60) * NS_PER_SECOND]
    meta = run(config, frame(["start", "success"], times=times), timeout_seconds=timeout)

    assert _as_list(meta, "auth_unpaired") == [None, True]
    # Nothing was measured against the abandoned start, so no duration is reported for it.
    assert _as_list(meta, "auth_elapsed_seconds") == [None, None]


@pytest.mark.cpu_mode
def test_an_exchange_inside_the_timeout_still_pairs(config: Config):
    # The counterpart: expiry must not eat a live exchange. One second inside the horizon still pairs and times.
    timeout = 300
    times = [0, (timeout - 1) * NS_PER_SECOND]
    meta = run(config, frame(["start", "success"], times=times), timeout_seconds=timeout)

    assert _as_list(meta, "auth_unpaired") == [None, False]
    assert _as_list(meta, "auth_elapsed_seconds") == [None, float(timeout - 1)]


@pytest.mark.cpu_mode
def test_expiry_falls_in_the_same_place_however_the_stream_is_batched(config: Config):
    # Expiry keys on each row's own event time, not on a batch boundary, so splitting the stream cannot change
    # which exchanges were still pending when an outcome arrived.
    timeout = 300
    times = [0, (timeout + 60) * NS_PER_SECOND, (timeout + 61) * NS_PER_SECOND]
    payload = frame(["start", "success", "success"], times=times)

    whole = _as_list(run(config, payload, timeout_seconds=timeout), "auth_unpaired")

    stage = TC2AuthStage(config, timeout_seconds=timeout)
    split = []

    for start in range(len(times)):
        piece = {name: values[start:start + 1] for (name, values) in payload.items()}
        meta = MessageMeta(get_df_class(config.execution_mode)(piece))
        stage.on_data(meta)
        split.extend(_as_list(meta, "auth_unpaired"))

    assert whole == split == [None, True, True]


@pytest.mark.gpu_and_cpu_mode
def test_two_supplicants_on_one_port_do_not_close_each_others_exchanges(config: Config):
    # Cisco multi-domain seats a phone and a workstation on one interface, and both run their own 802.1X. Keyed on
    # the port alone the second one to finish finds its slot already taken, so a routine authorization is written
    # as authorization-without-authentication and R-D-L2-005 fires on every phone port in the estate, forever.
    payload = frame(["started", "started", "success", "success"], times=[t * NS_PER_SECOND for t in (0, 1, 3, 5)])
    payload["mac_address"] = ["00:00:00:00:0a:aa", "00:00:00:00:0b:bb", "00:00:00:00:0a:aa", "00:00:00:00:0b:bb"]

    meta = run(config, payload)

    assert _as_list(meta, "auth_unpaired") == [None, None, False, False]
    # Each device is timed against its own start -- 0s to 3s and 1s to 5s -- not against whichever start came last.
    assert _as_list(meta, "auth_elapsed_seconds") == [None, None, 3.0, 4.0]
    assert _as_list(meta, "auth_attempts") == [None, None, 1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_a_bypass_beside_a_live_exchange_is_still_reported(config: Config):
    # The direction that matters. A device authorized with nothing in front of it -- MAB, or one bridged behind an
    # already authorized supplicant -- is exactly what R-D-L2-005 exists to catch. Keyed on the port alone it took
    # the pending slot of the device it is bridged behind, reported an ordinary elapsed time, and the alert that
    # did fire named the legitimate supplicant instead.
    payload = frame(["started", "success", "success"], times=[t * NS_PER_SECOND for t in (0, 2, 6)])
    payload["mac_address"] = ["00:00:00:00:1e:61", "de:ad:be:ef:00:01", "00:00:00:00:1e:61"]

    meta = run(config, payload)

    unpaired = _as_list(meta, "auth_unpaired")

    assert unpaired[1] is True, "the bypass must be flagged"
    assert unpaired[2] is False, "the legitimate supplicant must not be"
    assert _as_list(meta, "auth_elapsed_seconds") == [None, None, 6.0]


@pytest.mark.gpu_and_cpu_mode
def test_an_identity_pairs_an_exchange_when_no_mac_is_reported(config: Config):
    # Not every source reports a MAC on both halves. `dot1x_identity` is in the TC-2 required-field list, so it is
    # the documented fallback rather than a guess.
    payload = frame(["started", "started", "success", "success"], times=[t * NS_PER_SECOND for t in (0, 1, 3, 5)])
    payload["dot1x_identity"] = ["host/a.corp", "host/b.corp", "host/a.corp", "host/b.corp"]

    meta = run(config, payload)

    assert _as_list(meta, "auth_unpaired") == [None, None, False, False]


@pytest.mark.gpu_and_cpu_mode
def test_a_source_with_no_supplicant_still_times_per_port(config: Config):
    # The fallback, and the reason the golden corpus is unchanged by supplicant keying: a frame carrying no
    # supplicant column behaves exactly as it did when the key was the port.
    payload = frame(["started", "success"], times=[t * NS_PER_SECOND for t in (0, 4)])

    meta = run(config, payload)

    assert _as_list(meta, "auth_unpaired") == [None, False]
    assert _as_list(meta, "auth_elapsed_seconds") == [None, 4.0]


@pytest.mark.gpu_and_cpu_mode
def test_one_supplicant_on_two_ports_is_two_exchanges(config: Config):
    # The key is the port extended with the supplicant, not the supplicant alone: a laptop moving between two
    # ports has two exchanges, and the second must not be paired against the first port's start.
    payload = frame(["started", "success"], times=[t * NS_PER_SECOND for t in (0, 4)], ports=["Gi1/0/1", "Gi1/0/2"])
    payload["mac_address"] = ["00:00:00:00:0a:aa"] * 2

    meta = run(config, payload)

    assert _as_list(meta, "auth_unpaired") == [None, True]


@pytest.mark.cpu_mode
def test_one_row_from_a_broken_clock_does_not_abandon_every_pending_exchange(config: Config):
    # Same defect as the binding stage's, with a worse consequence. Abandoning a pending exchange means the real
    # outcome, when it arrives, pairs with nothing and is written auth_unpaired=true -- so one row from a device
    # whose clock is wrong by years would make R-D-L2-005 fire on every legitimate supplicant in the estate.
    stage = TC2AuthStage(config, timeout_seconds=300)
    ten_years = 10 * 365 * 24 * 3600 * NS_PER_SECOND

    started = frame(["started"] * 3, times=[0, 0, 0], ports=["Gi1/0/1", "Gi1/0/2", "Gi1/0/3"])
    started["mac_address"] = ["00:00:00:00:00:01", "00:00:00:00:00:02", "00:00:00:00:00:03"]
    stage.on_data(MessageMeta(get_df_class(config.execution_mode)(started)))

    assert stage._timer.pending_count == 3

    poisoned = frame(["started"], times=[ten_years], ports=["Gi1/0/9"])
    poisoned["mac_address"] = ["00:00:00:00:00:ff"]
    meta = MessageMeta(get_df_class(config.execution_mode)(poisoned))
    stage.on_data(meta)

    assert stage._timer.pending_count == 3, "the refused row abandons nothing and opens nothing"
    assert _as_list(meta, "auth_unpaired") == [None], "and carries no timing of its own"

    # The legitimate outcome still pairs with its own start rather than reading as a bypass.
    finished = frame(["success"], times=[2 * NS_PER_SECOND], ports=["Gi1/0/1"])
    finished["mac_address"] = ["00:00:00:00:00:01"]
    meta = MessageMeta(get_df_class(config.execution_mode)(finished))
    stage.on_data(meta)

    assert _as_list(meta, "auth_unpaired") == [False]
    assert _as_list(meta, "auth_elapsed_seconds") == [2.0]
