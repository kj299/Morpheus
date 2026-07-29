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
"""Stamps deterministic lineage identifiers onto messages."""

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
from morpheus.stages.lineage._column_utils import assign_str_column
from morpheus.stages.lineage._column_utils import to_host_list
from morpheus.utils.lineage import DEFAULT_DIGEST_LENGTH
from morpheus.utils.lineage import event_uid_series
from morpheus.utils.lineage import link_uid_series

logger = logging.getLogger(__name__)

DEFAULT_ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
"""Default fields feeding the event identifier, matching the universal telemetry envelope."""


@register_stage("lineage-stamp")
class LineageStampStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Attach deterministic lineage identifiers to every row of a message.

    The stage always writes an `event_uid` derived from `id_columns`. When `parent_uid_column` is configured it
    additionally writes a `link_uid` describing the parent-child edge, along with the `join_method` that established
    it. Recording the join method is what later lets an analyst distinguish an exact attribution from an inferred one.

    Every identifier is a pure function of values already in the message, so replaying the same input reproduces the
    same lineage. Hashing is performed on the host in both GPU and CPU execution modes so that an identifier never
    depends on which mode produced it.

    Place this stage immediately after normalization, before any windowing or scoring, so that downstream stages carry
    the identifiers and any late-arriving copy of a record resolves to the same `event_uid`.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    id_columns : list of str, optional
        Columns feeding the event identifier, in a significant order. Defaults to
        `["collector_id", "schema_version", "origin_hash", "collector_seq"]`. Every producer expected to agree on
        identifiers must use the same columns in the same order.
    event_uid_column : str, default = "event_uid"
        Column to write the event identifier to. Overwritten if it already exists.
    parent_uid_column : str, optional
        Column holding the parent record's event identifier. When `None`, no edge is emitted. Rows with a null or
        empty parent are treated as chain roots and receive a null `link_uid`.
    link_uid_column : str, default = "link_uid"
        Column to write the edge identifier to. Only used when `parent_uid_column` is set.
    relation : str, default = "derived_from"
        Semantic relationship recorded on every edge, for example `carried_by` or `authenticated_via`.
    join_method : str, default = "hard"
        How the edge was established, for example `hard:flow_id` or `soft:dhcp_lease`.
    join_method_column : str, default = "join_method"
        Column to write `join_method` to. Only used when `parent_uid_column` is set.
    digest_length : int, default = 32
        Number of hexadecimal characters retained from each SHA-256 digest.
    """

    def __init__(self,
                 c: Config,
                 id_columns: list[str] = None,
                 event_uid_column: str = "event_uid",
                 parent_uid_column: str = None,
                 link_uid_column: str = "link_uid",
                 relation: str = "derived_from",
                 join_method: str = "hard",
                 join_method_column: str = "join_method",
                 digest_length: int = DEFAULT_DIGEST_LENGTH):
        super().__init__(c)

        if (id_columns is None):
            id_columns = list(DEFAULT_ID_COLUMNS)

        id_columns = list(id_columns)

        if (len(id_columns) == 0):
            raise ValueError("id_columns must contain at least one column")

        if (not 1 <= digest_length <= 64):
            raise ValueError(f"digest_length must be between 1 and 64, received {digest_length}")

        self._id_columns = id_columns
        self._event_uid_column = event_uid_column
        self._parent_uid_column = parent_uid_column
        self._link_uid_column = link_uid_column
        self._relation = relation
        self._join_method = join_method
        self._join_method_column = join_method_column
        self._digest_length = digest_length

        self._needed_columns[event_uid_column] = TypeId.STRING

        if (parent_uid_column is not None):
            self._needed_columns[link_uid_column] = TypeId.STRING
            self._needed_columns[join_method_column] = TypeId.STRING

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "lineage-stamp"

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
        Stamp lineage identifiers onto the message payload.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming message.

        Returns
        -------
        The input message, with the identifier columns populated.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            missing = [col for col in self._id_columns if col not in df.columns]
            if (len(missing) > 0):
                raise KeyError(f"LineageStampStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            columns = [to_host_list(df, col) for col in self._id_columns]
            event_uids = event_uid_series(columns, digest_length=self._digest_length)
            assign_str_column(df, self._event_uid_column, event_uids)

            if (self._parent_uid_column is not None):
                if (self._parent_uid_column not in df.columns):
                    raise KeyError(f"LineageStampStage was configured with parent_uid_column="
                                   f"'{self._parent_uid_column}' which is not present in the DataFrame. "
                                   f"Available columns: {sorted(df.columns)}")

                parent_uids = to_host_list(df, self._parent_uid_column)
                link_uids = link_uid_series(parent_uids,
                                            event_uids,
                                            self._relation,
                                            self._join_method,
                                            digest_length=self._digest_length)

                assign_str_column(df, self._link_uid_column, link_uids)
                assign_str_column(df, self._join_method_column, [self._join_method] * len(event_uids))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
