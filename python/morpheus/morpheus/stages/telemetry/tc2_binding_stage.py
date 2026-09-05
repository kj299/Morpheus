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
"""Turns layer 2 binding observations into closed, resolvable intervals."""

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
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.binding_closer import DEFAULT_IDLE_TIMEOUT_NS
from morpheus.utils.binding_closer import NS_PER_SECOND
from morpheus.utils.binding_closer import BindingCloser
from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import compose_key

logger = logging.getLogger(__name__)

DEFAULT_ATTRIBUTE_COLUMNS = ["site_id", "switch_id", "port_id", "vlan_id"]
"""What a MAC address is bound to in the TC-2 telemetry class: the class's own `site_id:switch_id:port_id:vlan_id`."""

PORT_KEY_COLUMN = "port_key"
PORT_KEY_ATTRIBUTES = ("site_id", "switch_id", "port_id")
"""The attributes that compose `port_key`, in the order layer 1 composes `entity_key`. When every one of them is a
binding attribute, each closed binding carries the port as the same string the TC-1 stages key on, which is the join
the identifier ladder's first arrow depends on. `switch_id` here and `device_id` at layer 1 name the same thing.
"""

DEFAULT_IDLE_TIMEOUT_SECONDS = DEFAULT_IDLE_TIMEOUT_NS // NS_PER_SECOND

BIND_START_COLUMN = "bind_start"
BIND_END_COLUMN = "bind_end"
END_REASON_COLUMN = "bind_end_reason"
END_OBSERVED_COLUMN = "bind_end_observed"
OBSERVATIONS_COLUMN = "bind_observations"
PROVISIONAL_COLUMN = "bind_provisional"
OPEN_REASON = "open"
"""`bind_end_reason` on a provisional record: the binding has not ended, and the end is null."""


@register_stage("tc2-binding", ignore_args=["attribute_columns"])
class TC2BindingStage(GpuAndCpuMixin, SinglePortStage):
    """
    Emit closed layer 2 bindings from a stream of observations.

    Layer 2 is where the identifier ladder crosses from a physical port to a MAC address, and it carries that
    weight only if its bindings are time-bounded. `morpheus.utils.binding_table` resolves against half-open
    intervals, so a binding with no end resolves nothing and layer 2 stops being a usable lineage hop. The TC-2
    telemetry class states this as a requirement: emit an explicit end on expiry rather than relying on the next
    binding's start.

    Nothing upstream provides that end. A switch MAC table says what is bound now, accounting stops go missing, and
    releases are advisory. This stage infers ends and records how it arrived at each one, so a consumer can tell an
    observed end from a deduced one by filtering on `bind_end_observed`.

    Unlike the TC-1 stages this is not a pass-through. Its input is one row per observation and its output is one
    row per *closed* binding, so a message in may produce no message at all, which is the normal case for a stable
    estate where nothing moved. A binding that is still open has not been emitted yet and is held in memory until
    something ends it.

    Ends are inferred at the earliest time consistent with the observations, which leaves gaps between bindings
    rather than stretching one to meet the next. That is the point: a gap resolves to nothing and tells an analyst
    the answer is unknown, whereas a stretched binding covers a period the MAC may already have left and answers
    confidently and wrongly.

    The stage is stateful and must run single-engine. Shard by switch upstream for parallelism, which is
    determinism control 4, and note that sharding by MAC would be wrong here: displacement is only detectable when
    both sightings of a MAC reach the same instance.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    key_column : str, default = "mac_address"
        Column holding what is bound.
    time_column : str, default = "event_time"
        Column holding the observation's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    attribute_columns : list of str, optional
        Columns making up the binding target. Defaults to `["site_id", "switch_id", "port_id", "vlan_id"]`, the
        telemetry class's own entity key. A sample whose target differs from the open binding's displaces it; one
        whose target matches extends it. Columns outside this list are ignored, so a changing signal strength does
        not split a binding. When `site_id`, `switch_id` and `port_id` are all present, each emitted binding also
        carries `port_key`, the port as `site_id:switch_id:port_id`, identical to the layer 1 `entity_key`.
    idle_timeout_seconds : int, default = 1800
        Silence after which an open binding is presumed aged out. Set it from the source's own aging interval where
        that is known; switch MAC tables commonly age at five minutes.
    emit_open_on_complete : bool, default = True
        Close and emit every still-open binding when the stream ends. Without this, a binding that never moved is
        never emitted at all, so a replay over a finite corpus would silently lose every stable MAC.
    emit_open_bindings : bool, default = False
        Also emit a provisional record the moment a binding opens, with a null `bind_end`,
        `bind_end_reason = "open"` and `bind_provisional = true`. Off, a device plugged in now cannot be resolved
        until its binding closes, which is up to `idle_timeout_seconds` of "unknown" during an incident. On, live
        attribution has an answer immediately; the consumer caps the open interval with an explicit assumed
        duration (`BindingTable.from_dataframe(open_end_duration_ns=...)`), and the closed record that follows,
        carrying the same key and `bind_start`, supersedes it. One record per binding, not per sample: a sample
        that merely extends an open binding emits nothing.
    """

    def __init__(self,
                 c: Config,
                 key_column: str = "mac_address",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 attribute_columns: list[str] = None,
                 idle_timeout_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS,
                 emit_open_on_complete: bool = True,
                 emit_open_bindings: bool = False):
        super().__init__(c)

        attribute_columns = list(DEFAULT_ATTRIBUTE_COLUMNS) if attribute_columns is None else list(attribute_columns)

        if (idle_timeout_seconds <= 0):
            raise ValueError(f"idle_timeout_seconds must be positive, received {idle_timeout_seconds}")

        self._key_column = key_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._attribute_columns = attribute_columns
        self._emit_open_on_complete = emit_open_on_complete
        self._emit_open_bindings = emit_open_bindings

        self._closer = BindingCloser(attribute_names=attribute_columns,
                                     idle_timeout_ns=idle_timeout_seconds * NS_PER_SECOND)

        self._needed_columns[BIND_START_COLUMN] = TypeId.INT64
        self._needed_columns[BIND_END_COLUMN] = TypeId.INT64
        self._needed_columns[END_REASON_COLUMN] = TypeId.STRING
        self._needed_columns[END_OBSERVED_COLUMN] = TypeId.BOOL8
        self._needed_columns[OBSERVATIONS_COLUMN] = TypeId.INT64
        self._needed_columns[PORT_KEY_COLUMN] = TypeId.STRING
        self._needed_columns[PROVISIONAL_COLUMN] = TypeId.BOOL8

        self._emits_port_key = all(name in attribute_columns for name in PORT_KEY_ATTRIBUTES)

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc2-binding"

    def accepted_types(self) -> tuple:
        """
        Accepted input types for this stage.

        Returns
        -------
        tuple
            Accepted input types.
        """
        return (ControlMessage, MessageMeta)

    def compute_schema(self, schema: StageSchema):
        """
        Declare the output type, which is a frame of closed bindings rather than the observations that came in.
        """
        schema.output_schema.set_type(MessageMeta)

    def supports_cpp_node(self) -> bool:
        """Whether this stage supports a C++ node."""
        return False

    @property
    def open_count(self) -> int:
        """Bindings currently open and therefore not yet emitted."""
        return self._closer.open_count

    def _records(self, closed: list, opened: list) -> dict:
        """Render closed bindings, then provisional open ones, as columns, one row each. `bind_end` is left out and
        assigned separately so that its nulls are nulls in both execution modes."""
        records = list(closed) + list(opened)
        columns: dict[str, list] = {self._key_column: [record.key for record in records]}

        for name in self._attribute_columns:
            columns[name] = [record.attributes.get(name) for record in records]

        columns[BIND_START_COLUMN] = [record.bind_start_ns for record in records]
        columns[END_REASON_COLUMN] = [record.end_reason for record in closed] + [OPEN_REASON] * len(opened)
        columns[END_OBSERVED_COLUMN] = [record.end_observed for record in closed] + [False] * len(opened)
        columns[OBSERVATIONS_COLUMN] = [record.observations for record in records]
        columns[PROVISIONAL_COLUMN] = [False] * len(closed) + [True] * len(opened)

        return columns

    def _emit(self, closed: list, opened: list = ()) -> list:
        """Wrap closed and provisional bindings in a frame of the execution mode's own type, or nothing when there
        are none."""
        opened = list(opened)

        if (len(closed) == 0 and len(opened) == 0):
            return []

        # Imported here so that this module remains importable in CPU-only environments where cuDF is absent.
        from morpheus.utils.type_utils import get_df_class

        df = get_df_class(self._config.execution_mode)(self._records(closed, opened))

        # A provisional record has no end. Null here, in both modes, is what tells the consumer to apply its own
        # assumed duration rather than read a fabricated one.
        ends = [record.bind_end_ns for record in closed] + [None] * len(opened)
        assign_nullable_int_column(df, BIND_END_COLUMN, ends)

        # The port as layer 1 spells it, so a MAC resolved through this binding lands on a TC-1 `entity_key`. Null
        # when the target does not name a full port, or when any part of it was missing.
        port_keys = [
            compose_key([record.attributes.get(name) for name in PORT_KEY_ATTRIBUTES]) if self._emits_port_key else None
            for record in list(closed) + opened
        ]
        assign_str_column(df, PORT_KEY_COLUMN, port_keys)

        return [MessageMeta(df)]

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]) -> list:
        """
        Feed a batch of observations to the closer and emit whatever bindings that ended.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming observations.

        Returns
        -------
        list of `morpheus.messages.MessageMeta`
            One frame of closed bindings, or an empty list when this batch ended none, which is the normal case
            for an estate where nothing moved.

        Raises
        ------
        KeyError
            If the key column, the time column, or a declared attribute column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return []

        source = meta.copy_dataframe()

        required = [self._key_column, self._time_column] + self._attribute_columns
        missing = [column for column in required if column not in source.columns]

        if (len(missing) > 0):
            raise KeyError(f"TC2BindingStage requires columns {missing} which are not present in the DataFrame. "
                           f"Available columns: {sorted(source.columns)}")

        keys = to_host_list(source, self._key_column)
        raw_times = to_host_list(source, self._time_column)
        attributes = {name: to_host_list(source, name) for name in self._attribute_columns}

        closed = []
        opened_keys: list[str] = []
        unordered = 0

        for (position, key) in enumerate(keys):
            try:
                event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
            except ValueError:
                event_time_ns = None

            if (event_time_ns is None or key is None):
                unordered += 1
                continue

            # A MAC that has gone quiet for longer than the idle timeout has left, and its binding has to end at
            # the point it went quiet. Without this the next sighting simply extends the old binding across the
            # silence, so a soft join in the middle of that gap attributes an event to a port the device was not
            # on. Expiry runs on this row's own event time, which keeps the closure in the same place however the
            # stream is divided into batches.
            closed.extend(self._closer.expire(event_time_ns))

            result = self._closer.observe(str(key),
                                          event_time_ns,
                                          {name: attributes[name][position]
                                           for name in self._attribute_columns})

            closed.extend(result.closed)
            unordered += int(result.out_of_order)

            if (result.opened and self._emit_open_bindings):
                opened_keys.append(str(key))

        if (unordered > 0):
            logger.warning(
                "TC2BindingStage skipped %d of %d observations that were out of order, keyless, or without a "
                "usable event time; they did not advance any binding. Shard by switch and preserve per-key "
                "ordering upstream.",
                unordered,
                len(keys))

        # One provisional record per binding that opened in this batch, in its current state. A key that opened
        # twice, displaced within the batch, yields one record for the binding now open; the earlier one is among the
        # closed records.
        opened = []
        seen: set[str] = set()

        for key in opened_keys:
            if (key in seen):
                continue

            seen.add(key)
            record = self._closer.open_binding(key)

            if (record is not None):
                opened.append(record)

        return self._emit(closed, opened)

    def on_completed(self) -> list:
        """
        Close whatever is still open when the stream ends.

        A binding that never moved is otherwise never emitted, so a replay over a finite corpus would lose every
        stable MAC in the estate, which is most of them.
        """
        if (not self._emit_open_on_complete):
            return []

        return self._emit(self._closer.drain())

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name,
                                 ops.map(self.on_data),
                                 ops.filter(lambda frames: len(frames) > 0),
                                 ops.on_completed(self.on_completed),
                                 ops.flatten())
        builder.make_edge(input_node, node)

        return node
