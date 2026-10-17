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

from morpheus.messages import MessageMeta
from morpheus.stages.telemetry.flow_record_normalize_stage import OUTPUT_COLUMNS
from morpheus.stages.telemetry.flow_record_normalize_stage import FlowRecordNormalizeStage
from morpheus.utils.tracker_state import from_bytes
from morpheus.utils.tracker_state import to_bytes


def _records(**overrides) -> pd.DataFrame:
    base = {
        "flowStartMilliseconds": [1000, 2000],
        "sourceIPv4Address": ["10.0.0.1", "10.0.0.2"],
        "sourceTransportPort": [49152, 49153],
        "destinationIPv4Address": ["10.0.1.1", "10.0.1.2"],
        "destinationTransportPort": [443, 53],
        "protocolIdentifier": [6, 17],
        "minimumTTL": [64, 128],
        "octetDeltaCount": [100, 74],
        "reverseOctetDeltaCount": [2000, 180],
        "bgpDestinationAsNumber": [64500, 0],
        "exporter_address": ["192.0.2.11", "192.0.2.12"],
        "observation_domain_id": [256, 512],
        "export_time": [1, 2],
        "received_ns": [1_500_000_000, 2_500_000_000],
    }
    base.update(overrides)

    return pd.DataFrame(base)


def _stage(config, **kwargs) -> FlowRecordNormalizeStage:
    return FlowRecordNormalizeStage(config,
                                    collector_id=kwargs.pop("collector_id", "netflow-01"),
                                    tenant_id=kwargs.pop("tenant_id", "corp"),
                                    **kwargs)


@pytest.mark.cpu_mode
def test_a_record_becomes_the_tc3_contract_and_nothing_else(config):
    out = _stage(config, clock_offset_ms=17, exporter_names={"192.0.2.11": "rtr-access-1"}).normalize(_records())

    assert list(out.columns) == list(OUTPUT_COLUMNS)
    assert out.to_dict("records")[0] == {
        "collector_id": "netflow-01",
        "schema_version": "tc3.v1",
        "origin_hash": "10.0.0.1->10.0.1.1",
        "collector_seq": 0,
        "exporter_id": "rtr-access-1:256",
        "event_time": 1_000_000_000,
        "src_ip": "10.0.0.1",
        "src_port": 49152,
        "dst_ip": "10.0.1.1",
        "dst_port": 443,
        "protocol": "tcp",
        "ip_ttl": 64,
        "bytes_out": 100,
        "bytes_in": 2000,
        "bgp_as_dst": "64500",
        "observed_time": 1_500_000_000,
        "clock_source": "ntp_disciplined",
        "clock_offset_ms": 17,
        "sampling_policy": "full",
        "tenant_id": "corp",
    }


@pytest.mark.cpu_mode
def test_as_zero_is_no_as_and_an_unnamed_exporter_keeps_its_address(config):
    second = _stage(config).normalize(_records()).iloc[1]

    assert second["bgp_as_dst"] is None
    assert second["exporter_id"] == "192.0.2.12:512"
    assert second["protocol"] == "udp"


@pytest.mark.cpu_mode
def test_ipv6_and_an_unnamed_protocol_number(config):
    out = _stage(config).normalize(
        _records(sourceIPv4Address=[None, None],
                 destinationIPv4Address=[None, None],
                 sourceIPv6Address=["2001:db8::1", "2001:db8::2"],
                 destinationIPv6Address=["2001:db8::9", "2001:db8::a"],
                 protocolIdentifier=[132, 89]))

    assert list(out["src_ip"]) == ["2001:db8::1", "2001:db8::2"]
    assert list(out["protocol"]) == ["sctp", "89"]


@pytest.mark.cpu_mode
def test_a_record_without_a_start_is_timed_by_its_export_and_without_addresses_refused(config):
    stage = _stage(config)
    timed = stage.normalize(_records(flowStartMilliseconds=[None, None], received_ns=[None, None]))

    assert list(timed["event_time"]) == [1_000_000_000, 2_000_000_000]
    assert list(timed["observed_time"]) == list(timed["event_time"])

    with pytest.raises(KeyError, match="no source or no destination"):
        stage.normalize(_records(sourceIPv4Address=[None, None]))


@pytest.mark.cpu_mode
def test_the_sequence_continues_across_frames_and_a_restart(config):
    stage = _stage(config, first_seq=10)

    assert list(stage.normalize(_records())["collector_seq"]) == [10, 11]

    resumed = _stage(config)
    resumed.restore_state(from_bytes(to_bytes(stage.checkpoint_state())))

    assert list(resumed.normalize(_records())["collector_seq"]) == [12, 13]


@pytest.mark.cpu_mode
def test_a_collector_must_say_who_it_is(config):
    with pytest.raises(ValueError, match="collector_id and a tenant_id"):
        _stage(config, tenant_id="")

    with pytest.raises(ValueError, match="first_seq"):
        _stage(config, first_seq=-1)


@pytest.mark.cpu_mode
def test_on_data_replaces_the_frame(config):
    out = _stage(config).on_data(MessageMeta(_records()))

    assert list(out.copy_dataframe().columns) == list(OUTPUT_COLUMNS)
