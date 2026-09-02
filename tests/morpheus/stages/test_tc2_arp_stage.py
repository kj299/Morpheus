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
from morpheus.stages.telemetry.tc2_arp_stage import TC2ArpStage
from morpheus.utils.ratio_window import NS_PER_SECOND
from morpheus.utils.type_utils import get_df_class

GATEWAY = "10.0.0.1"
VICTIM = "10.0.0.9"
ATTACKER_MAC = "00:11:22:33:44:55"
ROUTER_MAC = "aa:bb:cc:dd:ee:ff"


def frame(senders: list, targets: list, macs: list = None, ops: list = None, times: list = None) -> dict:
    count = len(senders)

    return {
        "arp_sender_ip": senders,
        "arp_target_ip": targets,
        "arp_sender_mac": [ATTACKER_MAC] * count if macs is None else macs,
        "arp_operation": ["reply"] * count if ops is None else ops,
        "event_time": [index * NS_PER_SECOND for index in range(count)] if times is None else times,
    }


def gratuitous(count: int, sender: str = GATEWAY, **kwargs) -> dict:
    """A run of packets in which the sender claims its own address."""
    return frame([sender] * count, [sender] * count, **kwargs)


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [None if value is None or value is pd.NA or value != value else value for value in series.tolist()]


def run(config: Config, payload: dict, **kwargs) -> MessageMeta:
    defaults = {"min_denominator": 4}
    defaults.update(kwargs)
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC2ArpStage(config, **defaults).on_data(meta)

    return meta


def test_execution_modes(config: Config):
    assert issubclass(TC2ArpStage, GpuAndCpuMixin)

    assert set(TC2ArpStage(config).supported_execution_modes()) == {ExecutionMode.GPU, ExecutionMode.CPU}


def test_needed_columns(config: Config):
    needed = TC2ArpStage(config).get_needed_columns()

    assert needed["arp_is_gratuitous"] == TypeId.BOOL8
    assert needed["gratuitous_arp_ratio"] == TypeId.FLOAT64
    assert needed["macs_claiming_sender_ip"] == TypeId.INT64


def test_cli_command_builds():
    from click.testing import CliRunner

    registration = getattr(TC2ArpStage, "_morpheus_registered_stage", None)

    assert registration is not None
    assert CliRunner().invoke(registration.build_command(), ["--help"]).exit_code == 0


@pytest.mark.gpu_and_cpu_mode
def test_gratuitous_is_sender_equals_target(config: Config):
    payload = frame([GATEWAY, GATEWAY], [GATEWAY, VICTIM])

    meta = run(config, payload)

    # Claiming your own address is gratuitous; asking about someone else's is ordinary resolution.
    assert _as_list(meta, "arp_is_gratuitous") == [True, False]


@pytest.mark.gpu_and_cpu_mode
def test_a_null_address_is_not_a_claim_of_equality(config: Config):
    payload = frame([None, GATEWAY], [None, None])

    meta = run(config, payload)

    assert _as_list(meta, "arp_is_gratuitous") == [False, False]


@pytest.mark.gpu_and_cpu_mode
def test_no_ratio_below_the_denominator_floor(config: Config):
    # A proportion over two packets is noise, and one gratuitous packet out of one reads as 1.0.
    meta = run(config, gratuitous(4), min_denominator=4)

    assert _as_list(meta, "gratuitous_arp_ratio") == [None, None, None, 1.0]


@pytest.mark.gpu_and_cpu_mode
def test_a_poisoning_stream_reads_high(config: Config):
    meta = run(config, gratuitous(6))

    assert _as_list(meta, "gratuitous_arp_ratio")[-1] == pytest.approx(1.0)
    assert _as_list(meta, "gratuitous_arp_count")[-1] == 6


@pytest.mark.gpu_and_cpu_mode
def test_ordinary_resolution_reads_low(config: Config):
    # A host resolving its neighbours all day is not poisoning anyone.
    payload = frame([VICTIM] * 6, [f"10.0.0.{index + 20}" for index in range(6)])

    meta = run(config, payload)

    assert _as_list(meta, "gratuitous_arp_ratio")[-1] == pytest.approx(0.0)


@pytest.mark.gpu_and_cpu_mode
def test_a_chatty_host_is_not_penalised_for_volume(config: Config):
    # The reason this is a proportion rather than a count. One announcement in a stream of resolution is normal.
    payload = frame([VICTIM] * 8, [VICTIM] + [f"10.0.0.{index + 20}" for index in range(7)])

    meta = run(config, payload)

    assert _as_list(meta, "gratuitous_arp_count")[-1] == 1
    assert _as_list(meta, "gratuitous_arp_ratio")[-1] == pytest.approx(0.125)


@pytest.mark.gpu_and_cpu_mode
def test_only_replies_count_by_default(config: Config):
    # The guide specifies gratuitous replies; an RFC 5227 announcement is a request and is not counted unless asked.
    payload = gratuitous(4, ops=["request"] * 4)

    meta = run(config, payload)

    assert _as_list(meta, "arp_is_gratuitous") == [True] * 4
    assert _as_list(meta, "gratuitous_arp_ratio")[-1] == pytest.approx(0.0)


@pytest.mark.gpu_and_cpu_mode
def test_gratuitous_requests_can_be_included(config: Config):
    payload = gratuitous(4, ops=["request"] * 4)

    meta = run(config, payload, include_gratuitous_requests=True)

    assert _as_list(meta, "gratuitous_arp_ratio")[-1] == pytest.approx(1.0)


@pytest.mark.gpu_and_cpu_mode
def test_opcode_renderings_are_understood(config: Config):
    # Collectors render the opcode as a number, a word, or the tcpdump spelling.
    payload = gratuitous(4, ops=["2", "reply", "is-at", "REPLY"])

    meta = run(config, payload)

    assert _as_list(meta, "gratuitous_arp_ratio")[-1] == pytest.approx(1.0)


@pytest.mark.gpu_and_cpu_mode
def test_the_proportion_is_taken_per_sender(config: Config):
    payload = frame([GATEWAY] * 8, [GATEWAY] * 4 + [VICTIM] * 4, macs=[ATTACKER_MAC] * 4 + [ROUTER_MAC] * 4)

    meta = run(config, payload)

    # The device doing the claiming is what a responder needs to identify, so the share is per sender MAC.
    assert _as_list(meta, "gratuitous_arp_ratio")[3] == pytest.approx(1.0)
    assert _as_list(meta, "gratuitous_arp_ratio")[-1] == pytest.approx(0.0)


@pytest.mark.gpu_and_cpu_mode
def test_two_macs_claiming_one_address_are_counted(config: Config):
    # R-D-L2-003: an address resolving to more than one MAC inside the window.
    payload = frame([GATEWAY, GATEWAY], [GATEWAY, GATEWAY], macs=[ROUTER_MAC, ATTACKER_MAC])

    meta = run(config, payload)

    assert _as_list(meta, "macs_claiming_sender_ip") == [1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_a_redundancy_address_is_marked_rather_than_dropped(config: Config):
    # HSRP and VRRP virtuals legitimately move between MACs. The guide says the rule is unusable without the
    # exclusion list, so the stage marks the row and leaves the decision to the rule rather than silently
    # discarding evidence.
    payload = frame([GATEWAY, GATEWAY], [GATEWAY, GATEWAY], macs=[ROUTER_MAC, ATTACKER_MAC])

    meta = run(config, payload, excluded_sender_ips=[GATEWAY])

    assert _as_list(meta, "arp_sender_ip_excluded") == [True, True]
    assert _as_list(meta, "macs_claiming_sender_ip") == [1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_an_unlisted_address_is_not_excluded(config: Config):
    payload = frame([VICTIM], [VICTIM])

    meta = run(config, payload, excluded_sender_ips=[GATEWAY])

    assert _as_list(meta, "arp_sender_ip_excluded") == [False]


@pytest.mark.gpu_and_cpu_mode
def test_events_leave_the_window(config: Config):
    payload = gratuitous(5, times=[0, 1, 2, 3, 3600 * NS_PER_SECOND])

    meta = run(config, payload, window_seconds=10, min_denominator=1)

    # An hour later the burst has expired, so the host reads as a single announcement rather than a flood.
    assert _as_list(meta, "arp_count_in_window")[-1] == 1


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    df_class = get_df_class(config.execution_mode)
    stage = TC2ArpStage(config, min_denominator=4)

    stage.on_data(MessageMeta(df_class(gratuitous(3, times=[0, 1, 2]))))
    second = MessageMeta(df_class(gratuitous(1, times=[3 * NS_PER_SECOND])))
    stage.on_data(second)

    # The window spans the message boundary, which is the whole reason this stage is stateful.
    assert _as_list(second, "gratuitous_arp_ratio") == [1.0]


@pytest.mark.gpu_and_cpu_mode
def test_warm_up_gaps_are_nulls_rather_than_nan(config: Config):
    meta = run(config, gratuitous(4), min_denominator=4)
    ratio = meta.get_data("gratuitous_arp_ratio")

    assert int(ratio.isna().sum()) == 3

    if (config.execution_mode == ExecutionMode.CPU):
        assert pd.api.types.is_extension_array_dtype(ratio.dtype)


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    message = ControlMessage()
    message.payload(MessageMeta(get_df_class(config.execution_mode)(gratuitous(4))))

    TC2ArpStage(config, min_denominator=4).on_data(message)

    assert _as_list(message.payload(), "gratuitous_arp_ratio")[-1] == pytest.approx(1.0)


@pytest.mark.gpu_and_cpu_mode
def test_tc2_arp_stage_pipe(config: Config):
    source_df = get_df_class(config.execution_mode)(gratuitous(5))

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC2ArpStage(config, min_denominator=4))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "gratuitous_arp_ratio")[-1] == pytest.approx(1.0)


@pytest.mark.gpu_and_cpu_mode
def test_the_target_address_is_required(config: Config):
    # The guide's TC-2 field list omits it, which would make the feature uncomputable; the error says why.
    payload = gratuitous(2)
    del payload["arp_target_ip"]

    with pytest.raises(KeyError, match="arp_target_ip"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError):
        TC2ArpStage(config, window_seconds=0)

    with pytest.raises(ValueError):
        TC2ArpStage(config, min_denominator=0)
