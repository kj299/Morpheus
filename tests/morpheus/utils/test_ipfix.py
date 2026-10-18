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

import struct

import pytest

from morpheus.utils import ipfix

NAMES = ("flowStartMilliseconds",
         "sourceIPv4Address",
         "sourceTransportPort",
         "destinationIPv4Address",
         "destinationTransportPort",
         "protocolIdentifier",
         "minimumTTL",
         "octetDeltaCount",
         "reverseOctetDeltaCount",
         "bgpDestinationAsNumber")


def _record(index: int) -> dict:
    return {
        "flowStartMilliseconds": 1_000 * index,
        "sourceIPv4Address": f"10.0.0.{index}",
        "sourceTransportPort": 49152 + index,
        "destinationIPv4Address": "10.0.1.1",
        "destinationTransportPort": 443,
        "protocolIdentifier": 6,
        "minimumTTL": 64,
        "octetDeltaCount": 100 * index,
        "reverseOctetDeltaCount": 2**40 + index,
        "bgpDestinationAsNumber": 64500,
    }


def test_a_capture_reads_back_what_was_written(tmp_path):
    fields = ipfix.template_for(NAMES)
    datagrams = [
        (5 * 10**9 + 250_000,
         "192.0.2.11",
         "192.0.2.100",
         ipfix.encode_message(256, 5, 0, 256, fields, [_record(1), _record(2)], announce=True)),
        (6 * 10**9, "192.0.2.11", "192.0.2.100", ipfix.encode_message(256, 6, 2, 256, fields, [_record(3)], False)),
    ]
    path = str(tmp_path / "flows.pcap")
    ipfix.write_pcap(path, datagrams)

    capture = ipfix.read_pcap(path)

    assert capture.messages == 2
    assert capture.skipped_sets == 0
    assert [{name: record[name] for name in NAMES} for record in capture.records] == [_record(i) for i in (1, 2, 3)]
    assert [record["received_ns"] for record in capture.records] == [5 * 10**9 + 250_000] * 2 + [6 * 10**9]
    assert {(record["exporter_address"], record["observation_domain_id"])
            for record in capture.records} == {("192.0.2.11", 256)}
    # The reverse counter is RFC 5103's, enterprise 29305, and a 64-bit value survives.
    assert [field.enterprise for field in fields if field.element_id == 1] == [None, ipfix.REVERSE_PEN]


def test_a_data_set_before_its_template_is_skipped_and_counted(tmp_path):
    fields = ipfix.template_for(NAMES)
    path = str(tmp_path / "flows.pcap")
    ipfix.write_pcap(
        path,
        [
            (10**9, "192.0.2.11", "192.0.2.100", ipfix.encode_message(256, 1, 0, 256, fields, [_record(1)], False)),
            (2 * 10**9, "192.0.2.11", "192.0.2.100", ipfix.encode_message(256, 2, 1, 256, fields, [_record(2)], True)),
        ])

    capture = ipfix.read_pcap(path)

    assert capture.skipped_sets == 1
    assert [record["sourceIPv4Address"] for record in capture.records] == ["10.0.0.2"]


def test_templates_are_scoped_to_their_exporter_and_domain():
    fields = ipfix.template_for(NAMES)
    templates: dict = {}
    ipfix.decode_message(ipfix.encode_message(256, 1, 0, 256, fields, [_record(1)], True), "192.0.2.11", templates)

    other_domain = ipfix.encode_message(512, 1, 0, 256, fields, [_record(2)], False)
    other_exporter = ipfix.encode_message(256, 1, 0, 256, fields, [_record(3)], False)
    capture = ipfix.Capture(records=[])

    assert not ipfix.decode_message(other_domain, "192.0.2.11", templates, capture=capture)
    assert not ipfix.decode_message(other_exporter, "192.0.2.12", templates, capture=capture)
    assert capture.skipped_sets == 2


def test_an_element_nobody_reads_is_skipped_by_its_length_variable_ones_included():
    # A template with an unknown fixed element (id 300, 4 bytes), a variable-length one (id 82, interfaceName) and
    # then a known one: the known one must still decode.
    template = struct.pack("!HH", 256, 3) + struct.pack("!HH", 300, 4) + struct.pack("!HH", 82, 65535) + struct.pack(
        "!HH", 8, 4)
    data = b"\x00\x00\x00\x07" + bytes([5]) + b"eth0/" + bytes([10, 0, 0, 9])
    sets = struct.pack("!HH", 2, 4 + len(template)) + template + struct.pack("!HH", 256, 4 + len(data)) + data
    message = struct.pack("!HHIII", 10, 16 + len(sets), 1, 0, 1) + sets

    records = ipfix.decode_message(message, "192.0.2.11", {})

    assert len(records) == 1
    record = records[0]

    assert record["sourceIPv4Address"] == "10.0.0.9"
    assert "interfaceName" not in record


def test_a_message_that_is_not_ipfix_is_refused():
    fields = ipfix.template_for(NAMES)
    message = bytearray(ipfix.encode_message(256, 1, 0, 256, fields, [_record(1)], True))
    message[1] = 9

    with pytest.raises(ValueError, match="version 9"):
        ipfix.decode_message(bytes(message), "192.0.2.11", {})

    with pytest.raises(ValueError, match="disagrees"):
        ipfix.decode_message(ipfix.encode_message(256, 1, 0, 256, fields, [_record(1)], True)[:-1], "x", {})


def test_a_file_that_is_not_a_pcap_is_refused(tmp_path):
    path = tmp_path / "not.pcap"
    path.write_bytes(b"\x0a\x0d\x0d\x0a" + b"\x00" * 40)

    with pytest.raises(ValueError, match="pcapng"):
        ipfix.read_pcap(str(path))


def test_traffic_to_another_port_is_ignored(tmp_path):
    fields = ipfix.template_for(NAMES)
    path = str(tmp_path / "flows.pcap")
    ipfix.write_pcap(
        path, [(10**9, "192.0.2.11", "192.0.2.100", ipfix.encode_message(256, 1, 0, 256, fields, [_record(1)], True))],
        port=2055)

    capture = ipfix.read_pcap(path)

    assert (capture.records, capture.ignored_packets) == ([], 1)
    assert len(ipfix.read_pcap(path, port=2055).records) == 1
