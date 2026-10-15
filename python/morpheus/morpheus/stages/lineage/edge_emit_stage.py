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
One `morpheus:edge` record per relation a record was joined by: what it joined to, how, and in which chain.

Part 4 describes lineage as a graph -- a record and the records it was joined to, each edge saying how it was
established -- and every stage that establishes a join already writes its half of that: `BindingResolverStage`
names the binding it resolved through and the method, `CommunityIdStage` names the connection. What nothing did
was write the edge itself, so the `morpheus:edge` sourcetype carried flow records standing in for edges, with no
parent, no child, no join method and no chain, and the chain assembly search read a `lineage_id` from it that was
never there.

This stage writes the edges. It reads two kinds of relation off the rows it is given:

- **A resolution hop.** For each hop it is told about -- a resolver's method column and its binding-identifier
  column -- a row the hop resolved yields one edge whose parent is the binding (`parent_event_uid` is its
  `binding_uid`, the content-addressed identifier of the closed interval) and whose child is the record. The
  relation is `resolved_through` and the join method is the one the resolver wrote, `soft:dhcp_lease` or
  `hard:...`. A row the hop did not resolve has no parent there and yields nothing for it.
- **A shared connection.** With `join_column`, rows of different layers that carry the same value -- a Community ID
  the flow exporter, the packet sensor and the inspection point each stamped -- are one connection seen several
  times. Each row is joined to the first row, seen before it, of the nearest layer below it that carried the value,
  as `carried_by`, with the join method `hard:<join_column>`. Rows of one layer are not joined to each other: two
  sightings at one layer are the same evidence twice, not a relation.

Every edge carries `link_uid`, from `morpheus.utils.lineage.link_uid`, so an edge has an identity of its own and
re-emitting it on replay produces the same record; the OSI layer at each end, the parent's as configured for the
hop or as the parent row carried it; the child's `lineage_id`, which is what lets the chain assembly search count a
chain's join methods beside its layers; the child's `community_id` where it has one; and the child's
`event_time`, so the edge is indexed when the joined record happened.

**Placement.** Behind the resolvers and `WindowSealStage`, reading the sealed rows: the edge carries the chain the
child was sealed into. It replaces each batch with that batch's edges, so in a deployment it sits on its own branch
of the sealed stream, beside the sink the scored rows go to, as `PipelineHealthStage` does.

**State.** A shared-connection join remembers, per value and per layer, the first row that carried it. That memory is
bounded by `max_keys`, oldest value first, because a Community ID is a five-tuple's hash and an estate makes new
ones forever. A value evicted and seen again starts over, and its later rows join to the first row seen after
that, which is a missed edge and never a wrong one. The stage is stateful and must run single-engine --
determinism control 4.
"""

import collections
import logging
import typing

import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.column_assign import to_host_frame
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.entity_key import render_integral
from morpheus.utils.lineage import link_uid

logger = logging.getLogger(__name__)

RESOLVED_THROUGH = "resolved_through"
"""A record and the binding a resolver placed it through."""

CARRIED_BY = "carried_by"
"""A record and the lower layer's record of the same connection."""

DEFAULT_MAX_KEYS = 1_000_000

EDGE_COLUMNS = ("link_uid",
                "parent_event_uid",
                "child_event_uid",
                "relation_type",
                "join_method",
                "parent_osi_layer",
                "child_osi_layer",
                "lineage_id",
                "community_id",
                "event_time")
"""What every edge record carries, in this order."""

_RESOLVED_PREFIXES = ("hard:", "soft:")


def _integral(value: typing.Any) -> typing.Optional[int]:
    """A whole number however it is carried, a rendered one included, or `None`."""
    rendered = render_integral(value)

    if (rendered is None):
        text = normalize_text(value)
        rendered = text if text is not None and text.lstrip("-").isdigit() else None

    return None if rendered is None else int(rendered)


@register_stage("edge-emit", modes=[], ignore_args=["hop_method_columns", "hop_uid_columns", "hop_layers"])
class EdgeEmitStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Replace each batch of sealed rows with one edge per resolution hop and per shared connection they carry.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    hop_method_columns : list of str, optional
        One per resolution hop: the column the hop's resolver wrote its method to.
    hop_uid_columns : list of str, optional
        One per hop, in the same order: the column the resolver wrote the binding's identifier to.
    hop_layers : list of int, optional
        One per hop, in the same order: the OSI layer of the source the binding was closed from -- 3 for a DHCP
        server's leases, 2 for a MAC table or an 802.1X supplicant, 7 for an EDR agent's inventory -- or `None` for
        a binding with no layer, such as a directory entry. Defaults to `None` for every hop.
    join_column : str, optional
        A column whose value names one connection across layers, `community_id` in practice. Omitted, no
        shared-connection edges are emitted.
    uid_column : str, default = "event_uid"
        The record's own identifier, the child of every edge.
    layer_column : str, default = "osi_layer"
        The record's OSI layer.
    lineage_column : str, default = "lineage_id"
        The chain the record was sealed into.
    time_column : str, default = "event_time"
        When the record happened, which is when its edges are indexed.
    community_column : str, default = "community_id"
        Copied onto each edge where the child carries it.
    max_keys : int, default = 1000000
        Join values remembered for shared-connection edges, oldest evicted first.

    Raises
    ------
    ValueError
        If the hop lists differ in length, if there are no hops and no `join_column`, or if `max_keys` is not
        positive.
    """

    def __init__(self,
                 c: Config,
                 hop_method_columns: list[str] = None,
                 hop_uid_columns: list[str] = None,
                 hop_layers: list[typing.Optional[int]] = None,
                 join_column: str = None,
                 uid_column: str = "event_uid",
                 layer_column: str = "osi_layer",
                 lineage_column: str = "lineage_id",
                 time_column: str = "event_time",
                 community_column: str = "community_id",
                 max_keys: int = DEFAULT_MAX_KEYS):
        super().__init__(c)

        methods = list(hop_method_columns or [])
        uids = list(hop_uid_columns or [])
        layers = list(hop_layers) if hop_layers is not None else [None] * len(methods)

        if (len(methods) != len(uids) or len(methods) != len(layers)):
            raise ValueError(f"hop_method_columns, hop_uid_columns and hop_layers must name the same hops, received "
                             f"{len(methods)}, {len(uids)} and {len(layers)}")

        if (len(methods) == 0 and not join_column):
            raise ValueError("EdgeEmitStage needs at least one hop or a join_column; with neither there is no relation "
                             "to write")

        if (max_keys <= 0):
            raise ValueError(f"max_keys must be positive, received {max_keys}")

        self._hops = list(zip(methods, uids, layers))
        self._join_column = join_column
        self._uid_column = uid_column
        self._layer_column = layer_column
        self._lineage_column = lineage_column
        self._time_column = time_column
        self._community_column = community_column
        self._max_keys = max_keys

        # Per join value, the identifier of the first row at each layer that carried it.
        self._first: collections.OrderedDict[str, dict[int, str]] = collections.OrderedDict()

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "edge-emit"

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

    def _carried_by(self, key: str, layer: int, uid: str) -> typing.Optional[tuple]:
        """The parent a row of `layer` carrying `key` joins to, remembering the row if it is its layer's first."""
        seen = self._first.get(key)

        if (seen is None):
            seen = {}
            self._first[key] = seen

            while (len(self._first) > self._max_keys):
                self._first.popitem(last=False)
        else:
            self._first.move_to_end(key)

        below = [other for other in seen if other < layer]
        parent = None if len(below) == 0 else (seen[max(below)], max(below))
        seen.setdefault(layer, uid)

        return parent

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Replace the batch with its edges.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Sealed rows.

        Returns
        -------
        A message whose payload is the batch's edges, in row order and, within a row, hops first.

        Raises
        ------
        KeyError
            If a column the configured relations read is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        frame = to_host_frame(meta.copy_dataframe())
        needed = [self._uid_column, self._layer_column, self._lineage_column, self._time_column]
        needed += [column for (method, uid, _) in self._hops for column in (method, uid)]

        if (self._join_column):
            needed.append(self._join_column)

        missing = [column for column in needed if column not in frame.columns]

        if (missing):
            raise KeyError(f"EdgeEmitStage reads {missing}, which these rows do not carry. Available columns: "
                           f"{sorted(frame.columns)}")

        def column(name: str) -> list:
            return [normalize_text(value) for value in frame[name]]

        children = column(self._uid_column)
        layers = [_integral(value) for value in frame[self._layer_column]]
        lineages = column(self._lineage_column)
        times = [_integral(value) for value in frame[self._time_column]]
        communities = column(self._community_column) if self._community_column in frame.columns else [None] * len(
            frame)
        hops = [(column(method), column(uid), layer) for (method, uid, layer) in self._hops]
        joins = column(self._join_column) if self._join_column else None
        join_method = f"hard:{self._join_column}" if self._join_column else None

        edges = []

        def edge(parent: str, child: str, relation: str, method: str, parent_layer, position: int):
            edges.append({
                "link_uid": link_uid(parent, child, relation, method),
                "parent_event_uid": parent,
                "child_event_uid": child,
                "relation_type": relation,
                "join_method": method,
                "parent_osi_layer": parent_layer,
                "child_osi_layer": layers[position],
                "lineage_id": lineages[position],
                "community_id": communities[position],
                "event_time": times[position],
            })

        skipped = 0

        for (position, child) in enumerate(children):
            if (child is None):
                skipped += 1
                continue

            for (methods, uids, parent_layer) in hops:
                (method, parent) = (methods[position], uids[position])

                if (parent is not None and method is not None and method.startswith(_RESOLVED_PREFIXES)):
                    edge(parent, child, RESOLVED_THROUGH, method, parent_layer, position)

            if (joins is not None and joins[position] is not None and layers[position] is not None):
                carried = self._carried_by(joins[position], layers[position], child)

                if (carried is not None):
                    edge(carried[0], child, CARRIED_BY, join_method, carried[1], position)

        if (skipped > 0):
            logger.warning("EdgeEmitStage wrote no edge for %d of %d rows, which carry no %s to be the child of one",
                           skipped,
                           len(children),
                           self._uid_column)

        # Imported here so that this module remains importable in CPU-only environments where cuDF is absent.
        from morpheus.utils.type_utils import get_df_class

        result_frame = pd.DataFrame(edges, columns=list(EDGE_COLUMNS))

        for name in ("parent_osi_layer", "child_osi_layer", "event_time"):
            result_frame[name] = result_frame[name].astype("Int64")

        result = MessageMeta(get_df_class(self._config.execution_mode)(result_frame))

        if (isinstance(message, ControlMessage)):
            message.payload(result)
            return message

        return result

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
