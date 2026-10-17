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
Turn IPFIX flow records into the TC-3 flow record the layer 3 stages read, envelope included.

The input is what {py:func}`~morpheus.utils.ipfix.read_pcap` decodes: one row per data record, columns named by
the IANA information element, plus where it came from. The output is the TC-3 field list of
`examples/behavioral_analytics/input_contracts.json` and the universal envelope, and nothing else, so the stages
downstream read exactly what the corpus builders hand them.

What comes from the record, and what a collector has to say for itself:

- **From the record.** The addresses (IPv4 or IPv6), the ports, the protocol named from its number, the TTL from
  `minimumTTL`, `bytes_out` from `octetDeltaCount` and `bytes_in` from its RFC 5103 reverse, the destination AS,
  and `event_time` from `flowStartMilliseconds` (or `flowStartSeconds`). `exporter_id` is the exporter's name and
  its observation domain, `rtr-dc-1:512`, because one router exports several domains and each is its own clock
  and its own sequence. `observed_time` is when the collector received the message, the capture timestamp.
- **From the collector.** `collector_id`, `schema_version`, `tenant_id`, `clock_source`, `clock_offset_ms` and
  `sampling_policy` are facts about the collector and the exporter's configuration, which no flow record carries,
  and are parameters. `collector_seq` counts the records this collector has normalized, from `first_seq`, and is
  carried across a restart by the stage's checkpoint, so it stays strictly increasing as the envelope requires.

**An AS of zero is no AS.** RFC 7607 reserves AS 0, and an exporter reports it when the destination has no BGP
route; it becomes a null, which is what the reach stage treats as an unknown destination network, rather than a
network called `0`.

`origin_hash` is the flow's two addresses, `src->dst`, as the corpus builder gives it, so a normalized record and
the corpus's record for the same flow are the same record to the lineage stamp.
"""

import logging
import math
import typing

import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.checkpoint_mixin import CheckpointMixin
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.column_assign import to_host_frame
from morpheus.utils.ipfix import reverse_name
from morpheus.utils.type_utils import is_cudf_type

logger = logging.getLogger(__name__)

PROTOCOL_NAMES = {1: "icmp", 6: "tcp", 17: "udp", 47: "gre", 50: "esp", 58: "ipv6-icmp", 132: "sctp"}
"""IANA protocol numbers by the names the TC-3 stages and the app's searches use; any other is its number."""

OUTPUT_COLUMNS = ("collector_id",
                  "schema_version",
                  "origin_hash",
                  "collector_seq",
                  "exporter_id",
                  "event_time",
                  "src_ip",
                  "src_port",
                  "dst_ip",
                  "dst_port",
                  "protocol",
                  "ip_ttl",
                  "bytes_out",
                  "bytes_in",
                  "bgp_as_dst",
                  "observed_time",
                  "clock_source",
                  "clock_offset_ms",
                  "sampling_policy",
                  "tenant_id")
"""The TC-3 contract's columns and the envelope's, in the order the corpus builder writes them."""

INTEGER_COLUMNS = ("collector_seq",
                   "event_time",
                   "src_port",
                   "dst_port",
                   "ip_ttl",
                   "bytes_out",
                   "bytes_in",
                   "observed_time")


def _present(value: typing.Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _first(record: dict, *names: str) -> typing.Any:
    for name in names:
        value = record.get(name)

        if (_present(value)):
            return value

    return None


@register_stage("flow-normalize")
class FlowRecordNormalizeStage(CheckpointMixin, GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Map decoded IPFIX records onto the TC-3 flow record and the universal envelope.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    collector_id : str
        This collector's stable identity.
    tenant_id : str
        The scope every record belongs to.
    schema_version : str, default = "tc3.v1"
        The contract version the records are written to.
    clock_source : str, default = "ntp_disciplined"
        How the exporters' clocks are disciplined.
    clock_offset_ms : float, default = None
        The exporters' last measured clock offset, where the collector knows it.
    sampling_policy : str, default = "full"
        `full`, or the exporters' sampling, for example `1:100`.
    exporter_names : dict, default = None
        Exporter address to the name `exporter_id` carries. An address with no name is used as it is.
    first_seq : int, default = 0
        The `collector_seq` of the first record a cold start normalizes.
    """

    _checkpoint_attributes = ("_next_seq", )

    def __init__(self,
                 c: Config,
                 collector_id: str,
                 tenant_id: str,
                 schema_version: str = "tc3.v1",
                 clock_source: str = "ntp_disciplined",
                 clock_offset_ms: float = None,
                 sampling_policy: str = "full",
                 exporter_names: dict = None,
                 first_seq: int = 0):
        super().__init__(c)

        if (not collector_id or not tenant_id):
            raise ValueError("FlowRecordNormalizeStage needs a collector_id and a tenant_id; the envelope requires "
                             "both and no flow record carries either")

        if (first_seq < 0):
            raise ValueError(f"first_seq must not be negative, received {first_seq}")

        self._collector_id = collector_id
        self._tenant_id = tenant_id
        self._schema_version = schema_version
        self._clock_source = clock_source
        self._clock_offset_ms = clock_offset_ms
        self._sampling_policy = sampling_policy
        self._exporter_names = dict(exporter_names or {})
        self._next_seq = first_seq

    @property
    def name(self) -> str:
        """Stage name."""
        return "flow-normalize"

    def accepted_types(self) -> tuple:
        """
        Accepted input types for this stage.

        Returns
        -------
        tuple
            Accepted input types.
        """
        return (ControlMessage, MessageMeta)

    def supports_cpp_node(self) -> bool:
        """Whether this stage supports a C++ node."""
        return False

    def normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        The TC-3 records for a frame of decoded IPFIX records, numbered after the last frame's.

        Parameters
        ----------
        frame : `pandas.DataFrame`
            One row per data record, as `read_pcap` decodes them.

        Returns
        -------
        `pandas.DataFrame`
            Exactly `OUTPUT_COLUMNS`.

        Raises
        ------
        KeyError
            If a record carries neither address family, or no start time and no export time.
        """
        rows = []
        unnamed = set()

        for record in frame.to_dict("records"):
            src = _first(record, "sourceIPv4Address", "sourceIPv6Address")
            dst = _first(record, "destinationIPv4Address", "destinationIPv6Address")

            if (src is None or dst is None):
                raise KeyError("an IPFIX record carries no source or no destination address; its template lacks "
                               "elements 8/27 or 12/28, which no TC-3 stage can do without")

            if (_present(record.get("flowStartMilliseconds"))):
                event_time = int(record["flowStartMilliseconds"]) * 1_000_000
            elif (_present(record.get("flowStartSeconds"))):
                event_time = int(record["flowStartSeconds"]) * 1_000_000_000
            elif (_present(record.get("export_time"))):
                event_time = int(record["export_time"]) * 1_000_000_000
            else:
                raise KeyError("an IPFIX record carries no flow start and no message export time")

            address = record.get("exporter_address")
            name = self._exporter_names.get(address)

            if (name is None):
                unnamed.add(address)
                name = address

            protocol = _first(record, "protocolIdentifier")
            asn = _first(record, "bgpDestinationAsNumber")
            received = _first(record, "received_ns")

            rows.append({
                "collector_id": self._collector_id,
                "schema_version": self._schema_version,
                "origin_hash": f"{src}->{dst}",
                "collector_seq": self._next_seq,
                "exporter_id": f"{name}:{int(record['observation_domain_id'])}",
                "event_time": event_time,
                "src_ip": src,
                "src_port": _first(record, "sourceTransportPort"),
                "dst_ip": dst,
                "dst_port": _first(record, "destinationTransportPort"),
                "protocol": None if protocol is None else PROTOCOL_NAMES.get(int(protocol), str(int(protocol))),
                "ip_ttl": _first(record, "minimumTTL"),
                "bytes_out": _first(record, "octetDeltaCount"),
                "bytes_in": _first(record, reverse_name("octetDeltaCount")),
                "bgp_as_dst": None if asn is None or int(asn) == 0 else str(int(asn)),
                "observed_time": event_time if received is None else int(received),
                "clock_source": self._clock_source,
                "clock_offset_ms": self._clock_offset_ms,
                "sampling_policy": self._sampling_policy,
                "tenant_id": self._tenant_id,
            })
            self._next_seq += 1

        if (len(unnamed) > 0):
            logger.warning(
                "FlowRecordNormalizeStage has no name for exporters %s; their exporter_id carries the "
                "address. Give exporter_names so a renumbered router keeps its identity.",
                sorted(str(address) for address in unnamed))

        out = pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))

        for column in INTEGER_COLUMNS:
            out[column] = out[column].astype("Int64" if out[column].isna().any() else "int64")

        return out

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Replace each message's records with their TC-3 form.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Decoded IPFIX records.

        Returns
        -------
        A message of the same type holding the TC-3 records.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        frame = meta.copy_dataframe()
        normalized = self.normalize(to_host_frame(frame))

        if (is_cudf_type(frame)):
            import cudf  # pylint: disable=import-outside-toplevel

            normalized = cudf.from_pandas(normalized)

        out = MessageMeta(normalized)

        if (isinstance(message, ControlMessage)):
            message.payload(out)
            return message

        return out

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
