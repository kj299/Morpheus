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
"""Computes the Community ID flow hash for network telemetry."""

import logging
import typing

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.community_id import community_id_series

logger = logging.getLogger(__name__)


@register_stage("community-id")
class CommunityIdStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Add a Community ID flow hash column to network telemetry.

    The Community ID places the endpoint pair in a canonical order before hashing, so both directions of a
    bidirectional flow produce the same value. That makes it an exact join key between Morpheus output and telemetry
    from Zeek, Suricata, and most other network tooling, which converts what would otherwise be a time-bounded
    approximate join into an equality join.

    Set `src_port_column` and `dst_port_column` to `None` for protocols that have no ports. When the port columns are
    configured but null for a given row, that row is hashed without ports, which is correct for protocols outside of
    TCP, UDP, SCTP, ICMP, and ICMPv6 and an error for those within it.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    src_ip_column : str, default = "src_ip"
        Column holding the source address, in presentation form.
    dst_ip_column : str, default = "dest_ip"
        Column holding the destination address, in presentation form.
    protocol_column : str, default = "protocol"
        Column holding the IP protocol, either as a number or as a name such as `tcp`.
    src_port_column : str, default = "src_port"
        Column holding the source port. For ICMP and ICMPv6 this is the message type. Set to `None` for port-less
        telemetry.
    dst_port_column : str, default = "dest_port"
        Column holding the destination port. For ICMP and ICMPv6 this is the message code. Set to `None` for port-less
        telemetry.
    seed : int, default = 0
        Two byte hash seed. Must match across every producer expected to agree, including the tooling Morpheus output
        is joined against. Leave at the default unless the rest of the estate uses a different seed.
    use_base64 : bool, default = True
        When True the digest is base64 encoded, which is the conventional rendering. When False it is hex encoded.
    output_column : str, default = "community_id"
        Column to write the Community ID to. Overwritten if it already exists.
    raise_on_failure : bool, default = False
        When True a row that cannot be hashed raises, failing the batch. When False the row receives a null value and
        the failure count is logged, so a single malformed record does not discard the batch.
    """

    def __init__(self,
                 c: Config,
                 src_ip_column: str = "src_ip",
                 dst_ip_column: str = "dest_ip",
                 protocol_column: str = "protocol",
                 src_port_column: str = "src_port",
                 dst_port_column: str = "dest_port",
                 seed: int = 0,
                 use_base64: bool = True,
                 output_column: str = "community_id",
                 raise_on_failure: bool = False):
        super().__init__(c)

        if ((src_port_column is None) != (dst_port_column is None)):
            raise ValueError("Either both or neither of src_port_column and dst_port_column must be supplied")

        if (not 0 <= seed <= 0xFFFF):
            raise ValueError(f"seed must fit in two bytes, received {seed}")

        self._src_ip_column = src_ip_column
        self._dst_ip_column = dst_ip_column
        self._protocol_column = protocol_column
        self._src_port_column = src_port_column
        self._dst_port_column = dst_port_column
        self._seed = seed
        self._use_base64 = use_base64
        self._output_column = output_column
        self._raise_on_failure = raise_on_failure

        self._needed_columns[output_column] = TypeId.STRING

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "community-id"

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

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Compute the Community ID for every row of the message payload.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming message.

        Returns
        -------
        The input message, with `output_column` populated.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        required = [self._src_ip_column, self._dst_ip_column, self._protocol_column]
        if (self._src_port_column is not None):
            required += [self._src_port_column, self._dst_port_column]

        with meta.mutable_dataframe() as df:
            missing = [col for col in required if col not in df.columns]
            if (len(missing) > 0):
                raise KeyError(f"CommunityIdStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            src_ports = to_host_list(df, self._src_port_column) if self._src_port_column is not None else None
            dst_ports = to_host_list(df, self._dst_port_column) if self._dst_port_column is not None else None

            values = community_id_series(to_host_list(df, self._src_ip_column),
                                         to_host_list(df, self._dst_ip_column),
                                         to_host_list(df, self._protocol_column),
                                         src_port=src_ports,
                                         dst_port=dst_ports,
                                         seed=self._seed,
                                         use_base64=self._use_base64,
                                         raise_on_failure=self._raise_on_failure)

            assign_str_column(df, self._output_column, values)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
