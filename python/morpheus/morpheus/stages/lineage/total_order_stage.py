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
"""Imposes determinism control 8's total row order on every batch."""

import logging
import typing

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import sort_for_cumulative_features

logger = logging.getLogger(__name__)


@register_stage("total-order", ignore_args=["order_columns"])
class TotalOrderStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Sort every batch into determinism control 8's total row order before any stateful stage sees it.

    The telemetry stages are stateful in arrival order. A counter delta is the difference from the previous sample,
    a binding closes when the next sighting is elsewhere, a distinct count depends on which values came before. Each
    of them flags a sample that arrives out of order rather than reordering it, because reordering inside a stage
    would hide the collector defect the flag exists to report. That leaves the question of who does impose the
    order, and the answer is this stage, placed once, ahead of the first stateful stage.

    Within a batch it is control 8 exactly: sort by `event_time`, then by `collector_id` and `collector_seq` so that
    two samples stamped in the same second still have one order. Across batches it can do nothing, which is the
    reason the guide says to shard by entity and preserve per-entity ordering upstream: a row that arrives a batch
    late is still late here, and the stateful stages will still flag it.

    The sort is total by default. If the order columns leave ties the stage raises rather than falling back on the
    input's own arrangement, since that arrangement is exactly what determinism exists to remove. The envelope
    requires `collector_seq` to be strictly monotonic per collector, so ties mean the envelope is being violated.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    order_columns : list of str, optional
        Sort key, most significant first. Defaults to `event_time`, `collector_id`, `collector_seq`.
    require_total_order : bool, default = True
        Raise when the order columns leave ties.
    """

    def __init__(self, c: Config, order_columns: list[str] = None, require_total_order: bool = True):
        super().__init__(c)

        self._order_columns = list(DEFAULT_ORDER_COLUMNS) if order_columns is None else list(order_columns)
        self._require_total_order = require_total_order

        if (len(self._order_columns) == 0):
            raise ValueError("order_columns must name at least one column")

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "total-order"

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
        Return the message with its rows in total order.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming batch.

        Returns
        -------
        The message, carrying a new payload with the rows sorted and the index reset. A new payload rather than an
        in-place sort, because the cumulative primitives downstream realign by index and need it fresh.

        Raises
        ------
        KeyError
            If an order column is absent.
        ValueError
            If `require_total_order` is set and the order columns leave ties.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        # Imported here so that this module remains importable in CPU-only environments where cuDF is absent.
        from morpheus.utils.type_utils import get_df_class

        ordered = sort_for_cumulative_features(meta.copy_dataframe(),
                                               order_columns=self._order_columns,
                                               require_total_order=self._require_total_order)
        new_meta = MessageMeta(get_df_class(self._config.execution_mode)(ordered))

        if (isinstance(message, ControlMessage)):
            message.payload(new_meta)
            return message

        return new_meta

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
