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

import logging

import numpy as np
import pytest

from morpheus.utils.community_id import PROTO_ICMP
from morpheus.utils.community_id import PROTO_ICMP6
from morpheus.utils.community_id import PROTO_SCTP
from morpheus.utils.community_id import PROTO_TCP
from morpheus.utils.community_id import PROTO_UDP
from morpheus.utils.community_id import community_id
from morpheus.utils.community_id import community_id_series
from morpheus.utils.community_id import get_protocol_number

# Values produced by the reference implementation at https://github.com/corelight/pycommunityid, which is the
# normative implementation of https://github.com/corelight/community-id-spec. The first three are the worked examples
# published in the specification itself.
REFERENCE_VECTORS = [
    # (proto, src_ip, dst_ip, src_port, dst_port, expected)
    (PROTO_TCP, "128.232.110.120", "66.35.250.204", 34855, 80, "1:LQU9qZlK+B5F3KDmev6m5PMibrg="),
    (PROTO_UDP, "192.168.1.52", "8.8.8.8", 54585, 53, "1:d/FP5EW3wiY1vCndhwleRRKHowQ="),
    (PROTO_ICMP, "192.168.0.89", "192.168.0.1", 8, 0, "1:X0snYXpgwiv9TZtqg64sgzUn6Dk="),
    (PROTO_SCTP, "192.168.170.8", "192.168.170.56", 7, 7, "1:MP2EtRCAUIZvTw6MxJHLV7N7JDs="),
    (PROTO_ICMP6, "fe80::200:86ff:fe05:80da", "fe80::260:97ff:fe07:69ea", 135, 0, "1:dGHyGvjMfljg6Bppwm3bg0LO8TY="),
    (PROTO_TCP, "2001:db8::1", "2001:db8::2", 443, 51000, "1:2TiKiiiSIuDma9IxOEOpx4wCXWM="),
]

# Pairs that must collapse onto the same ID because they are the two directions of one flow.
BIDIRECTIONAL_PAIRS = [
    ((PROTO_TCP, "128.232.110.120", "66.35.250.204", 34855, 80),
     (PROTO_TCP, "66.35.250.204", "128.232.110.120", 80, 34855)),
    ((PROTO_UDP, "192.168.1.52", "8.8.8.8", 54585, 53), (PROTO_UDP, "8.8.8.8", "192.168.1.52", 53, 54585)),
    # ICMP echo request pairs with echo reply via the type mapping, not via the code.
    ((PROTO_ICMP, "192.168.0.89", "192.168.0.1", 8, 0), (PROTO_ICMP, "192.168.0.1", "192.168.0.89", 0, 0)),
    ((PROTO_ICMP6, "fe80::200:86ff:fe05:80da", "fe80::260:97ff:fe07:69ea", 135, 0),
     (PROTO_ICMP6, "fe80::260:97ff:fe07:69ea", "fe80::200:86ff:fe05:80da", 136, 0)),
]


@pytest.mark.parametrize("proto, src_ip, dst_ip, src_port, dst_port, expected", REFERENCE_VECTORS)
def test_matches_reference_vectors(proto: int, src_ip: str, dst_ip: str, src_port: int, dst_port: int, expected: str):
    assert community_id(src_ip, dst_ip, proto, src_port, dst_port) == expected


@pytest.mark.parametrize("forward, reverse", BIDIRECTIONAL_PAIRS)
def test_bidirectional_flows_agree(forward: tuple, reverse: tuple):
    (fwd_proto, fwd_src, fwd_dst, fwd_sport, fwd_dport) = forward
    (rev_proto, rev_src, rev_dst, rev_sport, rev_dport) = reverse

    assert (community_id(fwd_src, fwd_dst, fwd_proto, fwd_sport,
                         fwd_dport) == community_id(rev_src, rev_dst, rev_proto, rev_sport, rev_dport))


def test_portless_protocol_is_bidirectional():
    # GRE, which has no ports.
    assert community_id("10.1.2.3", "10.4.5.6", 47) == community_id("10.4.5.6", "10.1.2.3", 47)
    assert community_id("10.1.2.3", "10.4.5.6", 47) == "1:hjmHR9XFBjHCDWCa1JnWlrW+aJY="


def test_one_way_icmp_is_not_reordered():
    # Type 20 has no counterpart in the mapping table, so the tuple is one-way and must not be flipped.
    forward = community_id("192.168.0.89", "192.168.0.1", PROTO_ICMP, 20, 0)
    reverse = community_id("192.168.0.1", "192.168.0.89", PROTO_ICMP, 20, 0)

    assert forward == "1:3o2RFccXzUgjl7zDpqmY7yJi8rI="
    assert forward != reverse


def test_seed_changes_the_result():
    args = ("128.232.110.120", "66.35.250.204", PROTO_TCP, 34855, 80)

    assert community_id(*args, seed=0) != community_id(*args, seed=1)
    assert community_id(*args, seed=1) == "1:3V71V58M3Ksw/yuFALMcW0LAHvc="


def test_hex_rendering():
    result = community_id("128.232.110.120", "66.35.250.204", PROTO_TCP, 34855, 80, use_base64=False)

    assert result == "1:2d053da9994af81e45dca0e67afea6e4f3226eb8"


def test_packed_addresses_accepted():
    packed = community_id(b"\x80\xe8\x6ex", b"\x42\x23\xfa\xcc", PROTO_TCP, 34855, 80)

    assert packed == "1:LQU9qZlK+B5F3KDmev6m5PMibrg="


def test_is_deterministic_across_calls():
    results = {community_id("10.0.0.1", "10.0.0.2", PROTO_TCP, 1234, 80) for _ in range(100)}

    assert len(results) == 1


@pytest.mark.parametrize("value, expected",
                         [(6, 6), ("6", 6), ("tcp", PROTO_TCP), ("TCP", PROTO_TCP), ("udp", PROTO_UDP),
                          ("icmp", PROTO_ICMP), ("icmp6", PROTO_ICMP6), ("ICMPv6", PROTO_ICMP6),
                          ("ipv6-icmp", PROTO_ICMP6), ("sctp", PROTO_SCTP), (" tcp ", PROTO_TCP), (47, 47)])
def test_get_protocol_number(value, expected: int):
    assert get_protocol_number(value) == expected


@pytest.mark.parametrize("value", ["not-a-protocol", None, True])
def test_get_protocol_number_rejects_invalid(value):
    with pytest.raises(ValueError):
        get_protocol_number(value)


def test_protocol_name_matches_number():
    by_name = community_id("128.232.110.120", "66.35.250.204", "tcp", 34855, 80)
    by_number = community_id("128.232.110.120", "66.35.250.204", PROTO_TCP, 34855, 80)

    assert by_name == by_number


def test_missing_ports_for_port_protocol_raises():
    with pytest.raises(ValueError):
        community_id("10.0.0.1", "10.0.0.2", PROTO_TCP)


def test_half_specified_ports_raise():
    with pytest.raises(ValueError):
        community_id("10.0.0.1", "10.0.0.2", 47, src_port=80)


def test_invalid_address_raises():
    with pytest.raises(ValueError):
        community_id("not-an-address", "10.0.0.2", PROTO_TCP, 80, 443)


def test_series_matches_scalar():
    src_ips = ["128.232.110.120", "192.168.1.52", "10.1.2.3"]
    dst_ips = ["66.35.250.204", "8.8.8.8", "10.4.5.6"]
    protos = [PROTO_TCP, PROTO_UDP, PROTO_TCP]
    src_ports = [34855, 54585, 1024]
    dst_ports = [80, 53, 443]

    results = community_id_series(src_ips, dst_ips, protos, src_ports, dst_ports)

    expected = [
        community_id(src, dst, proto, sport, dport)
        for (src, dst, proto, sport, dport) in zip(src_ips, dst_ips, protos, src_ports, dst_ports)
    ]

    assert results == expected


def test_series_memoization_is_transparent():
    # The same tuple repeated must produce the same value every time, which is what the memoization relies on.
    row_count = 50
    results = community_id_series(["10.0.0.1"] * row_count, ["10.0.0.2"] * row_count, [PROTO_TCP] * row_count,
                                  [1234] * row_count, [80] * row_count)

    assert len(results) == row_count
    assert len(set(results)) == 1
    assert results[0] == community_id("10.0.0.1", "10.0.0.2", PROTO_TCP, 1234, 80)


def test_series_without_ports():
    results = community_id_series(["10.1.2.3"], ["10.4.5.6"], [47])

    assert results == [community_id("10.1.2.3", "10.4.5.6", 47)]


def test_series_null_ports_treated_as_portless():
    # A mixed batch where the GRE row has no ports but the port columns still exist.
    results = community_id_series(["10.1.2.3", "10.1.2.3"], ["10.4.5.6", "10.4.5.6"], [47, PROTO_TCP], [np.nan, 1024],
                                  [None, 443])

    assert results[0] == community_id("10.1.2.3", "10.4.5.6", 47)
    assert results[1] == community_id("10.1.2.3", "10.4.5.6", PROTO_TCP, 1024, 443)


def test_series_bad_row_yields_none_by_default(caplog: pytest.LogCaptureFixture):
    # The configured `morpheus` logger does not propagate to the root logger, where caplog listens, so the handler is
    # attached to the emitting logger directly.
    module_logger = logging.getLogger("morpheus.utils.community_id")
    module_logger.addHandler(caplog.handler)

    try:
        results = community_id_series(["10.0.0.1", "bogus", None], ["10.0.0.2", "10.0.0.2", "10.0.0.2"],
                                      [PROTO_TCP, PROTO_TCP, PROTO_TCP], [1234, 1234, 1234], [80, 80, 80])
    finally:
        module_logger.removeHandler(caplog.handler)

    assert results[0] == community_id("10.0.0.1", "10.0.0.2", PROTO_TCP, 1234, 80)
    assert results[1] is None
    assert results[2] is None
    assert "Community ID could not be computed for 2 of 3 rows" in caplog.text


def test_series_bad_row_raises_when_requested():
    with pytest.raises(ValueError):
        community_id_series(["bogus"], ["10.0.0.2"], [PROTO_TCP], [1234], [80], raise_on_failure=True)


def test_series_empty_input():
    assert len(community_id_series([], [], [])) == 0
