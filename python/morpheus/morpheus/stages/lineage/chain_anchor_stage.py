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
Decides, per row, what a record's correlation chain is rooted on.

Part 4 defines a chain as one root entity's events inside one window, and its identifier as a Merkle root over the
members. That makes the chain a property of *membership*, and membership is decided by what each record is
anchored on. Until now that anchor was a per-class constant: a layer 1 sample on its port, an ARP observation on
the address being claimed. Two records from two layers could never share a chain, because they could never share
a root -- which is why every chain in the corpus spanned exactly one layer, and the cross-layer rule had nothing
to fire on.

The ladder exists to change that. When `BindingResolverStage` resolves an ARP observation's MAC to the port that
held it, the observation *is about that port* for correlation purposes, and its chain should be the port's chain.
This stage makes that decision explicit and per row: it takes the candidate columns in order of preference and
writes the first one that has a value. A resolved observation roots on the port; an unresolved one falls back to
its own subject, which is what it was anchored on before and is still true.

**Which candidate supplied the root is recorded beside it.** A chain that reaches across layers through a soft
join is not the same evidence as one that does not, and an analyst reading `lineage_id` months later has to be
able to tell which they are looking at. `chain_anchor_source` names the column, the same way `join_method` on an
edge names the join. A row where no candidate has a value carries no anchor rather than an invented one, on the
rule every key in this repository follows.
"""

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
from morpheus.utils.entity_key import normalize_text

logger = logging.getLogger(__name__)

DEFAULT_ANCHOR_COLUMN = "chain_anchor"
DEFAULT_SOURCE_COLUMN = "chain_anchor_source"


@register_stage("chain-anchor", modes=[])
class ChainAnchorStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write the column a row's correlation chain is rooted on, chosen per row from candidates in order.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    candidates : list of str
        Columns to draw the root from, most preferred first. A resolved layer 2 observation names
        `["resolved_port_key", "arp_sender_ip"]`: the port when the ladder reached it, the address otherwise.
    anchor_column : str, default = "chain_anchor"
        Column the chosen root is written to. `WindowSealStage` reads it as its `entity_key_column`.
    source_column : str, default = "chain_anchor_source"
        Column recording which candidate supplied the root, so a chain reached through a soft join can be told
        from one that was not.

    Raises
    ------
    ValueError
        If no candidates are named, a candidate is repeated, or the two output columns collide.
    """

    def __init__(self,
                 c: Config,
                 candidates: list[str],
                 anchor_column: str = DEFAULT_ANCHOR_COLUMN,
                 source_column: str = DEFAULT_SOURCE_COLUMN):
        super().__init__(c)

        if (not candidates):
            raise ValueError("candidates must name at least one column; a chain with nothing to root on is not a "
                             "chain, and a row with no root should carry none rather than a default")

        if (len(set(candidates)) != len(candidates)):
            raise ValueError(f"candidates must not repeat a column, received {candidates}")

        if (not anchor_column or not source_column):
            raise ValueError("anchor_column and source_column must both be named")

        if (anchor_column == source_column):
            raise ValueError(f"anchor_column and source_column must differ, both are {anchor_column!r}")

        self._candidates = list(candidates)
        self._anchor_column = anchor_column
        self._source_column = source_column

        self._needed_columns[anchor_column] = TypeId.STRING
        self._needed_columns[source_column] = TypeId.STRING

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "chain-anchor"

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
        Choose each row's chain root from the candidates, in order, and record which one supplied it.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming records.

        Returns
        -------
        The same message, with the anchor and its source attached.

        Raises
        ------
        KeyError
            If a candidate column is absent. A candidate that is not there is a wiring mistake, not a row with no
            value, and silently skipping it would root every row on the fallback without saying so.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            missing = [name for name in self._candidates if name not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"ChainAnchorStage candidates {missing} are absent. Available columns: "
                               f"{sorted(df.columns)}")

            columns = {name: to_host_list(df, name) for name in self._candidates}
            row_count = len(df)
            anchors: list[typing.Optional[str]] = []
            sources: list[typing.Optional[str]] = []

            for position in range(row_count):
                chosen = None
                source = None

                for name in self._candidates:
                    value = normalize_text(columns[name][position])

                    if (value is not None):
                        chosen = value
                        source = name
                        break

                anchors.append(chosen)
                sources.append(source)

            rootless = sum(1 for anchor in anchors if anchor is None)

            assign_str_column(df, self._anchor_column, anchors)
            assign_str_column(df, self._source_column, sources)

            if (rootless > 0):
                logger.warning(
                    "ChainAnchorStage found no root for %d of %d rows: none of %s had a value. They carry no "
                    "%s and will belong to no chain.",
                    rootless,
                    row_count,
                    self._candidates,
                    self._anchor_column)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
