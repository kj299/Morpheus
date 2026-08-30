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
"""Resolves observations against time-bounded bindings, turning a soft join into concrete columns."""

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
from morpheus.utils.binding_table import BindingTable
from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list

logger = logging.getLogger(__name__)

UNRESOLVED = "unresolved"
"""Value written to the method column when no binding covered the row."""


@register_stage("binding-resolve", ignore_args=["binding_table", "output_columns"])
class BindingResolverStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Resolve each row against a `BindingTable` and write the resolved attributes as columns.

    This is the stage that makes the layer 1 through 3 half of the lineage ladder runnable. A layer 3 observation
    carries an IP address; the physical port it came from is only knowable through a DHCP lease and a switch
    forwarding-table entry, each valid for an interval. Running this stage once per rung resolves the chain in the
    pipeline, at event time, rather than leaving it to a SIEM query to approximate later.

    Every row gets a value in `method_column`: either `soft:<table name>` or `unresolved`. That field is what lets an
    analyst tell an inferred attribution from an exact one without re-deriving the join, and it is deliberately not
    optional -- an unresolved row that looks identical to a resolved one is how soft joins produce confidently wrong
    attribution.

    Resolution is a pure function of the row and the table, so a replay against the same table produces the same
    columns. It is *not* a pure function of wall-clock time, which means the table must be snapshotted alongside the
    model and configuration for a replay to be meaningful. Part 5 of the OSI behavioral analytics guide covers that.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    binding_table : `morpheus.utils.binding_table.BindingTable`
        The facts to resolve against. Required; the default of `None` exists only so the CLI can register the
        stage, and construction rejects it.
    key_column : str, default = "src_ip"
        Column holding the value to resolve.
    time_column : str, default = "event_time"
        Column holding the event time. Must be event time, never ingest time: resolving a lease against the moment the
        record reached the pipeline attributes late-arriving telemetry to whoever holds the address now.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    output_columns : dict, optional
        Mapping of binding attribute name to output column name. Defaults to writing each attribute under its own
        name. Existing columns are overwritten.
    method_column : str, default = "resolution_method"
        Column recording how each row was resolved.
    uid_column : str, optional
        When set, the winning binding's content-addressed identifier is written here, so an attribution can be traced
        back to the exact lease or forwarding-table entry behind it.
    raise_on_unresolved : bool, default = False
        When True an unresolved row fails the batch. When False it receives nulls, is marked `unresolved`, and the
        count is logged. The default is deliberate: an unresolved row is normal at the edges of a collection window
        and should be visible rather than fatal.
    """

    def __init__(self,
                 c: Config,
                 binding_table: BindingTable = None,
                 key_column: str = "src_ip",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 output_columns: dict = None,
                 method_column: str = "resolution_method",
                 uid_column: str = None,
                 raise_on_unresolved: bool = False):
        super().__init__(c)

        if (binding_table is None):
            raise ValueError("binding_table is required")

        if (not key_column):
            raise ValueError("key_column is required")

        if (not method_column):
            raise ValueError("method_column is required")

        known = binding_table.value_columns
        output_columns = dict(output_columns) if output_columns is not None else {name: name for name in known}

        unknown = [name for name in output_columns if name not in known]

        if (len(unknown) > 0):
            raise ValueError(f"output_columns references attributes {unknown} which the binding table does not "
                             f"provide. Available attributes: {known}")

        if (len(output_columns) == 0):
            raise ValueError("output_columns must map at least one attribute")

        self._table = binding_table
        self._key_column = key_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._output_columns = output_columns
        self._method_column = method_column
        self._uid_column = uid_column
        self._raise_on_unresolved = raise_on_unresolved

        for target in output_columns.values():
            self._needed_columns[target] = TypeId.STRING

        self._needed_columns[method_column] = TypeId.STRING

        if (uid_column is not None):
            self._needed_columns[uid_column] = TypeId.STRING

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "binding-resolve"

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
        Resolve every row of the message payload against the binding table.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming message.

        Returns
        -------
        The input message, with the resolved attribute columns, the method column, and optionally the binding
        identifier populated.

        Raises
        ------
        KeyError
            If the key or time column is absent.
        ValueError
            If `raise_on_unresolved` is set and a row does not resolve.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            missing = [column for column in (self._key_column, self._time_column) if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"BindingResolverStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            keys = to_host_list(df, self._key_column)
            times = [to_epoch_ns(value, time_unit=self._time_unit) for value in to_host_list(df, self._time_column)]

            resolved = self._table.resolve_many(keys, times)
            unresolved_count = sum(1 for binding in resolved if binding is None)

            if (unresolved_count > 0 and self._raise_on_unresolved):
                raise ValueError(f"{unresolved_count} of {len(resolved)} rows did not resolve against binding table "
                                 f"{self._table.name!r}")

            attribute_index = {name: position for (position, name) in enumerate(self._table.value_columns)}

            for (attribute, target) in self._output_columns.items():
                position = attribute_index[attribute]
                assign_str_column(df,
                                  target,
                                  [None if binding is None else str(binding.values[position]) for binding in resolved])

            assign_str_column(df,
                              self._method_column,
                              [UNRESOLVED if binding is None else f"soft:{self._table.name}" for binding in resolved])

            if (self._uid_column is not None):
                assign_str_column(df,
                                  self._uid_column, [None if binding is None else binding.uid for binding in resolved])

        if (unresolved_count > 0):
            logger.warning("Binding table %r did not resolve %d of %d rows on column %r",
                           self._table.name,
                           unresolved_count,
                           len(resolved),
                           self._key_column)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
