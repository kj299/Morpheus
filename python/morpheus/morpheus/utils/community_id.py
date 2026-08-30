# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Community ID flow hashing.

The Community ID is a standard, tool-independent identifier for a network flow. Because the endpoint pair is placed in
a canonical order before hashing, both directions of a bidirectional flow produce the same value, which makes it usable
as an exact join key between Morpheus output and network telemetry produced by other tooling.

This module implements version 1 of the specification at https://github.com/corelight/community-id-spec including the
ICMP and ICMPv6 message-type mapping that establishes request/response directionality.
"""

import base64
import hashlib
import logging
import socket
import struct
import typing

import pandas as pd

logger = logging.getLogger(__name__)

PROTO_ICMP = 1
PROTO_TCP = 6
PROTO_UDP = 17
PROTO_ICMP6 = 58
PROTO_SCTP = 132

PORT_PROTOS = frozenset((PROTO_ICMP, PROTO_TCP, PROTO_UDP, PROTO_ICMP6, PROTO_SCTP))
"""IP protocols for which the Community ID computation requires a port pair."""

VERSION = "1"
"""Community ID specification version implemented by this module."""

# Mapping of an ICMP message type to the type of its expected counterpart. A type present in the mapping identifies a
# two-way interaction, which means the endpoints may be flipped into canonical order. A type absent from the mapping is
# one-way and its tuple is always left as observed.
_ICMP_TYPE_MAPPER = {
    0: 8,  # echo reply <-> echo request
    8: 0,
    9: 10,  # router advertisement <-> router solicitation
    10: 9,
    13: 14,  # timestamp request <-> timestamp reply
    14: 13,
    15: 16,  # information request <-> information reply
    16: 15,
    17: 18,  # address mask request <-> address mask reply
    18: 17,
}

_ICMP6_TYPE_MAPPER = {
    128: 129,  # echo request <-> echo reply
    129: 128,
    130: 131,  # multicast listener query <-> report
    131: 130,
    133: 134,  # router solicitation <-> router advertisement
    134: 133,
    135: 136,  # neighbor solicitation <-> neighbor advertisement
    136: 135,
    139: 140,  # who-are-you request <-> reply
    140: 139,
    144: 145,  # home agent address discovery request <-> reply
    145: 144,
}

AddressType = typing.Union[str, bytes]
PortType = typing.Optional[typing.Union[int, str]]

_PROTOCOL_NAMES = {
    "icmp": PROTO_ICMP,
    "tcp": PROTO_TCP,
    "udp": PROTO_UDP,
    "icmp6": PROTO_ICMP6,
    "icmpv6": PROTO_ICMP6,
    "ipv6-icmp": PROTO_ICMP6,
    "sctp": PROTO_SCTP,
}


def get_protocol_number(value: typing.Union[int, str]) -> int:
    """
    Resolve an IP protocol to its numeric value.

    Telemetry sources are inconsistent about this: NetFlow exporters emit the number while Zeek and most cloud flow
    logs emit a name. Both are accepted so that the same stage configuration works against either.

    Parameters
    ----------
    value : int or str
        Protocol number, a string holding a number, or a protocol name such as `tcp` or `ipv6-icmp`. Names are matched
        case-insensitively.

    Returns
    -------
    int
        The protocol number.

    Raises
    ------
    ValueError
        If the value is neither a known name nor an integer.
    """
    if (isinstance(value, bool)):
        raise ValueError(f"Invalid IP protocol: {value!r}")

    if (isinstance(value, int)):
        return value

    if (isinstance(value, str)):
        name = value.strip().lower()

        if (name in _PROTOCOL_NAMES):
            return _PROTOCOL_NAMES[name]

        try:
            return int(name)
        except ValueError as exc:
            raise ValueError(f"Unknown IP protocol: {value!r}") from exc

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown IP protocol: {value!r}") from exc


def _addr_to_nbo(addr: AddressType) -> bytes:
    """
    Convert an IPv4 or IPv6 address to its packed network byte order representation.

    Parameters
    ----------
    addr : str or bytes
        Address in presentation form, or already packed.

    Returns
    -------
    bytes
        Four bytes for IPv4, sixteen bytes for IPv6.

    Raises
    ------
    ValueError
        If `addr` is neither a parsable address string nor a packed address.
    """
    if (isinstance(addr, (bytes, bytearray))):
        if (len(addr) in (4, 16)):
            return bytes(addr)

        raise ValueError(f"Packed address must be 4 or 16 bytes, received {len(addr)}")

    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            return socket.inet_pton(family, addr)
        except (OSError, TypeError):
            pass

    raise ValueError(f"Unable to parse IP address: {addr!r}")


def _icmp_port_equivalents(mtype: int, mcode: int, mapper: dict[int, int]) -> tuple[int, int, bool]:
    """
    Map an ICMP message type and code onto the port pair used by the hash, mirroring Zeek's behavior.

    Returns
    -------
    tuple
        `(sport, dport, is_one_way)`. When the type has a defined counterpart the interaction is two-way and the
        counterpart type takes the place of the destination port. Otherwise the code is used and the tuple is treated
        as one-way, which suppresses canonical reordering.
    """
    counterpart = mapper.get(mtype)
    if (counterpart is None):
        return (mtype, mcode, True)

    return (mtype, counterpart, False)


def community_id(src_ip: AddressType,
                 dst_ip: AddressType,
                 protocol: int,
                 src_port: PortType = None,
                 dst_port: PortType = None,
                 seed: int = 0,
                 use_base64: bool = True) -> str:
    """
    Compute the version 1 Community ID for a single flow.

    Parameters
    ----------
    src_ip : str or bytes
        Source address, in presentation form or packed.
    dst_ip : str or bytes
        Destination address, in presentation form or packed.
    protocol : int
        IP protocol number, for example 6 for TCP.
    src_port : int, optional
        Source port. For ICMP and ICMPv6 this is the message type. Required when `protocol` is in `PORT_PROTOS`.
    dst_port : int, optional
        Destination port. For ICMP and ICMPv6 this is the message code. Required when `src_port` is supplied.
    seed : int, default = 0
        Two byte seed prefixed to the hash input. Must match across every producer that is expected to agree.
    use_base64 : bool, default = True
        When True the digest is base64 encoded, matching the conventional rendering. When False it is hex encoded.

    Returns
    -------
    str
        The Community ID, for example `1:LQU9qZlK+B5F3KDmev6m5PMibrg=`.

    Raises
    ------
    ValueError
        If the addresses cannot be parsed, if only one of the two ports is supplied, or if a port-enabled protocol is
        given without ports.
    """
    proto = get_protocol_number(protocol)

    if ((src_port is None) != (dst_port is None)):
        raise ValueError("Either both or neither of src_port and dst_port must be supplied")

    if (proto in PORT_PROTOS and src_port is None):
        raise ValueError(f"Ports are required for protocol {proto}")

    sport = int(src_port) if src_port is not None else None
    dport = int(dst_port) if dst_port is not None else None

    # The ICMP mapping is applied before ordering, so that a request and its reply order identically.
    is_one_way = False
    if (sport is not None):
        if (proto == PROTO_ICMP):
            (sport, dport, is_one_way) = _icmp_port_equivalents(sport, dport, _ICMP_TYPE_MAPPER)
        elif (proto == PROTO_ICMP6):
            (sport, dport, is_one_way) = _icmp_port_equivalents(sport, dport, _ICMP6_TYPE_MAPPER)

    saddr = _addr_to_nbo(src_ip)
    daddr = _addr_to_nbo(dst_ip)

    # Canonical order: the smaller endpoint becomes the source. Addresses compare as packed bytes; comparing the ports
    # as integers is equivalent to comparing their two byte big-endian representations.
    is_ordered = (is_one_way or saddr < daddr or (saddr == daddr and sport is not None and sport < dport))

    if (not is_ordered):
        (saddr, daddr) = (daddr, saddr)
        (sport, dport) = (dport, sport)

    hasher = hashlib.sha1()
    hasher.update(struct.pack("!H", seed))
    hasher.update(saddr)
    hasher.update(daddr)
    hasher.update(struct.pack("B", proto))
    hasher.update(b"\x00")  # padding, keeps the input 32-bit aligned

    if (sport is not None):
        hasher.update(struct.pack("!H", sport))
        hasher.update(struct.pack("!H", dport))

    if (use_base64):
        return f"{VERSION}:{base64.b64encode(hasher.digest()).decode('ascii')}"

    return f"{VERSION}:{hasher.hexdigest()}"


def _is_null(value: typing.Any) -> bool:
    """Return True for `None` and for the null sentinels used by pandas and cuDF."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        # Array-like and other values that `isna` cannot reduce to a single boolean are never null scalars.
        return False


def community_id_series(src_ip: typing.Sequence,
                        dst_ip: typing.Sequence,
                        protocol: typing.Sequence,
                        src_port: typing.Optional[typing.Sequence] = None,
                        dst_port: typing.Optional[typing.Sequence] = None,
                        seed: int = 0,
                        use_base64: bool = True,
                        raise_on_failure: bool = False) -> list[typing.Optional[str]]:
    """
    Compute Community IDs for a batch of flows.

    Results are memoized on the flow tuple, so repeated tuples within a batch, which are common once flows have been
    rolled up into time bins, cost only a dictionary lookup.

    Parameters
    ----------
    src_ip : sequence
        Per-row source addresses.
    dst_ip : sequence
        Per-row destination addresses.
    protocol : sequence
        Per-row IP protocols, as numbers or names.
    src_port : sequence, optional
        Per-row source ports. When omitted, every row is treated as port-less, which is only valid for protocols
        outside of `PORT_PROTOS`.
    dst_port : sequence, optional
        Per-row destination ports. Must be supplied whenever `src_port` is.
    seed : int, default = 0
        Two byte hash seed.
    use_base64 : bool, default = True
        Digest rendering, see `community_id`.
    raise_on_failure : bool, default = False
        When True, a row that cannot be hashed raises. When False the row yields `None` and the failure is logged once
        per batch, so a single malformed record does not discard the batch.

    Returns
    -------
    list
        One Community ID per input row, or `None` for rows that could not be computed.
    """
    row_count = len(src_ip)
    src_ports = src_port if src_port is not None else [None] * row_count
    dst_ports = dst_port if dst_port is not None else [None] * row_count

    cache: dict[tuple, typing.Optional[str]] = {}
    results: list[typing.Optional[str]] = []
    failures = 0
    first_failure: typing.Optional[Exception] = None

    missing = object()

    for row in zip(src_ip, dst_ip, protocol, src_ports, dst_ports):
        result = cache.get(row, missing)

        if (result is missing):
            try:
                if (_is_null(row[0]) or _is_null(row[1]) or _is_null(row[2])):
                    raise ValueError(f"Null address or protocol in flow tuple: {row!r}")

                row_sport = None if _is_null(row[3]) else row[3]
                row_dport = None if _is_null(row[4]) else row[4]

                result = community_id(row[0],
                                      row[1],
                                      row[2],
                                      src_port=row_sport,
                                      dst_port=row_dport,
                                      seed=seed,
                                      use_base64=use_base64)
            except (ValueError, TypeError, struct.error) as exc:
                if (raise_on_failure):
                    raise

                failures += 1
                if (first_failure is None):
                    first_failure = exc

                result = None

            cache[row] = result

        results.append(result)

    if (failures > 0):
        logger.warning("Community ID could not be computed for %d of %d rows; first failure: %s",
                       failures,
                       row_count,
                       first_failure)

    return results
