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
IPFIX messages, read from a packet capture of what exporters sent, as one record per data record.

A flow exporter speaks IPFIX (RFC 7011) over UDP, and the first thing a deployment can show it receives is a
capture of that traffic. This module reads one: a classic libpcap file of Ethernet frames, of which the UDP
datagrams to the collector's port are IPFIX messages. Templates are kept per exporter address, observation
domain and template, as the RFC requires, so a data set is decoded only with the template its own exporter
announced for that domain; a data set whose template has not been seen is counted and skipped rather than guessed
at, because an exporter re-announces its templates and a collector that guesses decodes garbage.

Every data record becomes a dict keyed by the information element's name from the IANA registry, with the
reverse direction of a biflow (RFC 5103, enterprise 29305) as `reverse` plus the capitalized name, and four
fields about where it came from: `exporter_address`, `observation_domain_id`, `export_time` (the message
header's, in seconds) and `received_ns`, the capture timestamp, which is when the collector received it.

Only the elements the TC-3 normalization reads are named; any other element is skipped by its length, including
variable-length ones, so a template carrying more than these decodes and loses nothing anyone reads.

`write_pcap` and `encode_message` are the other direction, so a fixture can be produced from records and read
back by the same rules a real capture is.
"""

import dataclasses
import ipaddress
import struct
import typing

IPFIX_VERSION = 10
IPFIX_PORT = 4739
"""The IANA port for IPFIX over UDP."""

TEMPLATE_SET_ID = 2
OPTIONS_TEMPLATE_SET_ID = 3
VARIABLE_LENGTH = 65535
REVERSE_PEN = 29305
"""The private enterprise number RFC 5103 assigns to reverse information elements."""

PCAP_MAGIC = 0xA1B2C3D4
PCAP_MAGIC_NANOSECOND = 0xA1B23C4D
LINKTYPE_ETHERNET = 1
ETHERTYPE_IPV4 = 0x0800
ETHERTYPE_VLAN = 0x8100
IPPROTO_UDP = 17


@dataclasses.dataclass(frozen=True)
class Element:
    """An information element this module decodes: its registry name and how its bytes read."""

    name: str
    kind: str
    """`unsigned`, `ipv4`, `ipv6` or `milliseconds` (dateTimeMilliseconds, read as an unsigned count)."""


ELEMENTS: dict[int, Element] = {
    1: Element("octetDeltaCount", "unsigned"),
    2: Element("packetDeltaCount", "unsigned"),
    4: Element("protocolIdentifier", "unsigned"),
    7: Element("sourceTransportPort", "unsigned"),
    8: Element("sourceIPv4Address", "ipv4"),
    11: Element("destinationTransportPort", "unsigned"),
    12: Element("destinationIPv4Address", "ipv4"),
    16: Element("bgpSourceAsNumber", "unsigned"),
    17: Element("bgpDestinationAsNumber", "unsigned"),
    27: Element("sourceIPv6Address", "ipv6"),
    28: Element("destinationIPv6Address", "ipv6"),
    52: Element("minimumTTL", "unsigned"),
    53: Element("maximumTTL", "unsigned"),
    150: Element("flowStartSeconds", "unsigned"),
    152: Element("flowStartMilliseconds", "milliseconds"),
}
"""The IANA IPFIX information elements the TC-3 normalization reads, by element id."""

REVERSIBLE = frozenset({1, 2})
"""Elements whose RFC 5103 reverse is decoded too: the counters, which are what a biflow's other direction is."""

_IDS_BY_NAME = {element.name: element_id for (element_id, element) in ELEMENTS.items()}


def reverse_name(name: str) -> str:
    """The RFC 5103 name of an element's reverse: `reverseOctetDeltaCount` for `octetDeltaCount`."""
    return "reverse" + name[0].upper() + name[1:]


@dataclasses.dataclass(frozen=True)
class Field:
    """One field specifier of a template: element, length, and enterprise number where there is one."""

    element_id: int
    length: int
    enterprise: typing.Optional[int] = None


@dataclasses.dataclass
class Capture:
    """What a capture held: the decoded records, and what could not be decoded and why."""

    records: list
    messages: int = 0
    skipped_sets: int = 0
    """Data sets whose template had not been announced by that exporter for that domain."""

    ignored_packets: int = 0
    """Frames that were not IPv4 UDP to the collector's port."""


def _value(element: Element, data: bytes) -> typing.Any:
    if (element.kind == "ipv4"):
        return str(ipaddress.IPv4Address(data))

    if (element.kind == "ipv6"):
        return str(ipaddress.IPv6Address(data))

    # Reduced-size encoding (RFC 7011 section 6.2) shortens an unsigned field; big-endian reads it either way.
    return int.from_bytes(data, "big")


def _parse_template_set(body: bytes) -> list[tuple[int, list[Field]]]:
    templates = []
    offset = 0

    while (offset + 4 <= len(body)):
        (template_id, count) = struct.unpack_from("!HH", body, offset)
        offset += 4

        if (template_id < 256):
            # The rest of the set is padding.
            break

        fields = []

        for _ in range(count):
            (raw_id, length) = struct.unpack_from("!HH", body, offset)
            offset += 4
            enterprise = None

            if (raw_id & 0x8000):
                (enterprise, ) = struct.unpack_from("!I", body, offset)
                offset += 4

            fields.append(Field(raw_id & 0x7FFF, length, enterprise))

        templates.append((template_id, fields))

    return templates


def _record_length(fields: list[Field]) -> typing.Optional[int]:
    if (any(field.length == VARIABLE_LENGTH for field in fields)):
        return None

    return sum(field.length for field in fields)


def _parse_data_set(body: bytes, fields: list[Field]) -> list[dict]:
    records = []
    offset = 0
    fixed = _record_length(fields)

    while (offset < len(body)):
        if (fixed is not None and offset + fixed > len(body)):
            # Padding after the last record.
            break

        record = {}

        for field in fields:
            length = field.length

            if (length == VARIABLE_LENGTH):
                length = body[offset]
                offset += 1

                if (length == 255):
                    (length, ) = struct.unpack_from("!H", body, offset)
                    offset += 2

            data = body[offset:offset + length]
            offset += length

            if (field.enterprise is None and field.element_id in ELEMENTS):
                element = ELEMENTS[field.element_id]
                record[element.name] = _value(element, data)
            elif (field.enterprise == REVERSE_PEN and field.element_id in REVERSIBLE):
                element = ELEMENTS[field.element_id]
                record[reverse_name(element.name)] = _value(element, data)

        records.append(record)

    return records


def decode_message(message: bytes,
                   exporter_address: str,
                   templates: dict,
                   received_ns: int = None,
                   capture: Capture = None) -> list[dict]:
    """
    The data records of one IPFIX message.

    Parameters
    ----------
    message : bytes
        One message, header included.
    exporter_address : str
        Where it came from; templates are scoped to it.
    templates : dict
        The templates seen so far, updated in place with any this message announces.
    received_ns : int, optional
        When the collector received it, in ns since the Unix epoch.
    capture : `Capture`, optional
        Where to count a data set skipped for want of its template.

    Returns
    -------
    list of dict
        One dict per data record.

    Raises
    ------
    ValueError
        If the message is not IPFIX version 10 or its length disagrees with its header.
    """
    if (len(message) < 16):
        raise ValueError(f"an IPFIX message header is 16 bytes, received {len(message)}")

    (version, length, export_time, _, domain) = struct.unpack_from("!HHIII", message, 0)

    if (version != IPFIX_VERSION):
        raise ValueError(f"not an IPFIX message: version {version}")

    if (length != len(message)):
        raise ValueError(f"IPFIX message length {length} disagrees with the {len(message)} bytes received")

    records = []
    offset = 16

    while (offset + 4 <= length):
        (set_id, set_length) = struct.unpack_from("!HH", message, offset)

        if (set_length < 4):
            raise ValueError(f"IPFIX set length {set_length} is shorter than its own header")

        body = message[offset + 4:offset + set_length]
        offset += set_length

        if (set_id == TEMPLATE_SET_ID):
            for (template_id, fields) in _parse_template_set(body):
                templates[(exporter_address, domain, template_id)] = fields
        elif (set_id == OPTIONS_TEMPLATE_SET_ID):
            # Options describe the exporter rather than flows; nothing the normalization reads.
            continue
        elif (set_id >= 256):
            fields = templates.get((exporter_address, domain, set_id))

            if (fields is None):
                if (capture is not None):
                    capture.skipped_sets += 1

                continue

            for record in _parse_data_set(body, fields):
                record["exporter_address"] = exporter_address
                record["observation_domain_id"] = domain
                record["export_time"] = export_time
                record["received_ns"] = received_ns
                records.append(record)

    return records


def _udp_payload(frame: bytes, port: int) -> typing.Optional[tuple[str, bytes]]:
    if (len(frame) < 14):
        return None

    ethertype = struct.unpack_from("!H", frame, 12)[0]
    offset = 14

    if (ethertype == ETHERTYPE_VLAN and len(frame) >= 18):
        ethertype = struct.unpack_from("!H", frame, 16)[0]
        offset = 18

    if (ethertype != ETHERTYPE_IPV4 or len(frame) < offset + 20):
        return None

    header_length = (frame[offset] & 0x0F) * 4
    (total_length, ) = struct.unpack_from("!H", frame, offset + 2)

    if (frame[offset + 9] != IPPROTO_UDP):
        return None

    source = str(ipaddress.IPv4Address(frame[offset + 12:offset + 16]))
    udp = offset + header_length
    (destination_port, udp_length) = struct.unpack_from("!2xHH", frame, udp)

    if (destination_port != port):
        return None

    end = min(udp + udp_length, offset + total_length, len(frame))

    return (source, frame[udp + 8:end])


def read_pcap(path: str, port: int = IPFIX_PORT) -> Capture:
    """
    Every data record in a capture of IPFIX over UDP, in the order received.

    Parameters
    ----------
    path : str
        A classic libpcap file (microsecond or nanosecond timestamps) of Ethernet frames.
    port : int, default = 4739
        The UDP port the collector listens on.

    Returns
    -------
    `Capture`
        The records and what was skipped.

    Raises
    ------
    ValueError
        If the file is not a libpcap capture of Ethernet frames.
    """
    with open(path, "rb") as handle:
        data = handle.read()

    if (len(data) < 24):
        raise ValueError(f"{path} is too short to be a pcap file")

    for (endian, scale) in (("<", 1000), (">", 1000), ("<", 1), (">", 1)):
        magic = struct.unpack_from(endian + "I", data, 0)[0]

        if (magic == (PCAP_MAGIC if scale == 1000 else PCAP_MAGIC_NANOSECOND)):
            break
    else:
        raise ValueError(f"{path} is not a libpcap file (pcapng is not read here; convert it with editcap -F pcap)")

    linktype = struct.unpack_from(endian + "I", data, 20)[0]

    if (linktype != LINKTYPE_ETHERNET):
        raise ValueError(f"{path} has link type {linktype}; only Ethernet captures are read")

    capture = Capture(records=[])
    templates: dict = {}
    offset = 24

    while (offset + 16 <= len(data)):
        (seconds, fraction, included, _) = struct.unpack_from(endian + "IIII", data, offset)
        frame = data[offset + 16:offset + 16 + included]
        offset += 16 + included
        payload = _udp_payload(frame, port)

        if (payload is None):
            capture.ignored_packets += 1
            continue

        (exporter, message) = payload
        received_ns = seconds * 1_000_000_000 + fraction * scale
        capture.messages += 1
        capture.records.extend(decode_message(message, exporter, templates, received_ns, capture))

    return capture


def template_for(names: typing.Sequence[str]) -> list[Field]:
    """
    Field specifiers for a template carrying the named elements, reverse ones included, at their natural lengths.

    Parameters
    ----------
    names : sequence of str
        Element names from `ELEMENTS`, or their `reverse` forms.

    Returns
    -------
    list of `Field`
        The specifiers, in order.
    """
    lengths = {"unsigned": 8, "ipv4": 4, "ipv6": 16, "milliseconds": 8}
    narrow = {
        "protocolIdentifier": 1,
        "minimumTTL": 1,
        "maximumTTL": 1,
        "sourceTransportPort": 2,
        "destinationTransportPort": 2,
        "bgpSourceAsNumber": 4,
        "bgpDestinationAsNumber": 4,
        "flowStartSeconds": 4
    }
    fields = []

    for name in names:
        enterprise = None
        forward = name

        if (name.startswith("reverse")):
            forward = name[len("reverse")].lower() + name[len("reverse") + 1:]
            enterprise = REVERSE_PEN

        element_id = _IDS_BY_NAME[forward]
        element = ELEMENTS[element_id]
        fields.append(Field(element_id, narrow.get(forward, lengths[element.kind]), enterprise))

    return fields


def _encode_value(field: Field, value: typing.Any) -> bytes:
    element = ELEMENTS[field.element_id]

    if (element.kind == "ipv4"):
        return ipaddress.IPv4Address(value).packed

    if (element.kind == "ipv6"):
        return ipaddress.IPv6Address(value).packed

    return int(value).to_bytes(field.length, "big")


def encode_message(domain: int,
                   export_time: int,
                   sequence: int,
                   template_id: int,
                   fields: list[Field],
                   records: list[dict],
                   announce: bool) -> bytes:
    """
    One IPFIX message: the template set when `announce`, then a data set of `records` in `fields` order.

    Parameters
    ----------
    domain : int
        The observation domain.
    export_time : int
        Seconds since the Unix epoch.
    sequence : int
        Data records sent in this domain before this message.
    template_id : int
        At least 256.
    fields : list of `Field`
        From `template_for`.
    records : list of dict
        Keyed by element name, reverse names for reverse fields.
    announce : bool
        Send the template set first.

    Returns
    -------
    bytes
        The message.
    """
    sets = b""

    if (announce):
        specifiers = b"".join(
            struct.pack("!HH", field.element_id | (0x8000 if field.enterprise is not None else 0), field.length) +
            (struct.pack("!I", field.enterprise) if field.enterprise is not None else b"") for field in fields)
        template = struct.pack("!HH", template_id, len(fields)) + specifiers
        sets += struct.pack("!HH", TEMPLATE_SET_ID, 4 + len(template)) + template

    data = b""

    for record in records:
        for field in fields:
            name = ELEMENTS[field.element_id].name
            data += _encode_value(field, record[reverse_name(name) if field.enterprise is not None else name])

    sets += struct.pack("!HH", template_id, 4 + len(data)) + data

    return struct.pack("!HHIII", IPFIX_VERSION, 16 + len(sets), export_time, sequence, domain) + sets


def _checksum(header: bytes) -> int:
    total = sum(struct.unpack(f"!{len(header) // 2}H", header))

    while (total >> 16):
        total = (total & 0xFFFF) + (total >> 16)

    return ~total & 0xFFFF


def write_pcap(path: str, datagrams: typing.Iterable[tuple[int, str, str, bytes]], port: int = IPFIX_PORT):
    """
    Write UDP datagrams to a classic libpcap file as Ethernet frames.

    Parameters
    ----------
    path : str
        Where to write.
    datagrams : iterable of (int, str, str, bytes)
        Capture time in ns (written to the microsecond), source address, destination address, payload.
    port : int, default = 4739
        The destination port.
    """
    with open(path, "wb") as handle:
        handle.write(struct.pack("<IHHiIII", PCAP_MAGIC, 2, 4, 0, 0, 65535, LINKTYPE_ETHERNET))

        for (identifier, (when_ns, source, destination, payload)) in enumerate(datagrams):
            udp = struct.pack("!HHHH", 49152 + identifier % 16384, port, 8 + len(payload), 0) + payload
            ip = struct.pack("!BBHHHBBH4s4s",
                             0x45,
                             0,
                             20 + len(udp),
                             identifier & 0xFFFF,
                             0,
                             64,
                             IPPROTO_UDP,
                             0,
                             ipaddress.IPv4Address(source).packed,
                             ipaddress.IPv4Address(destination).packed)
            ip = ip[:10] + struct.pack("!H", _checksum(ip)) + ip[12:]
            frame = b"\x02\x00\x00\x00\x00\x02" + b"\x02\x00\x00\x00\x00\x01" + struct.pack("!H",
                                                                                            ETHERTYPE_IPV4) + ip + udp
            (seconds, remainder) = divmod(when_ns, 1_000_000_000)
            handle.write(struct.pack("<IIII", seconds, remainder // 1000, len(frame), len(frame)) + frame)
