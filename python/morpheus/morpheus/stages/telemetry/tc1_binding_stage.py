# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Turns layer 1 observations into the interval records the `binding_l1` lookup is built from.

This is the last rung of the identifier ladder and the only one with no producer until now. The ladder resolves an
address to a physical place one hop at a time: `binding_l2_l3` takes an IP to a MAC and a switch port, and
`binding_l1` takes that port to a site, a transceiver and an LLDP neighbor. Without this stage a chain query stops
at a port name, which is a label; with it the same query reaches a building, an optic and whatever is on the other
end of the fibre. The rules the guide says are not expressible any other way -- R-C-003 and R-C-005 -- are the ones
that need the last hop.

**The key is the port and the attributes are what is in it, which is the inverse of layer 2.** There the key is a
MAC, the mobile thing, bound to a location; a sample whose location differs means the device moved. Here the port
is fixed and the optic is what moves, so the port is the key and `transceiver_serial` and
`lldp_neighbor_chassis_id` are the attributes. A sample whose attributes differ means somebody changed the optic or
repatched the far end. Getting this backwards produces a table that answers "where is this transceiver", which is
not the question the ladder asks.

**The idle timeout is days rather than minutes, and that is a decision rather than a copied default.**
`TC2BindingStage` ages a binding out after thirty minutes because a MAC that has gone quiet has left, and switch
MAC tables age at about five minutes anyway. A switch port is not like that: it is polled on a cadence, a
transceiver sits in it for months, and silence means the poller stopped rather than that the optic left. A
thirty-minute horizon would close every binding in the estate during a collector outage and destroy the historical
attribution for that window -- the same catastrophic shape {py:mod}`~morpheus.utils.event_clock` guards against
from a different cause. Seven days is long enough that only a genuinely decommissioned port ages out, and an estate
that knows its polling cadence should set it from that.

**There is no bucketing here, deliberately.** The layer 2 lookup is bucketed at five minutes because a DHCP lease
moves and "which MAC held this address" is a question with a time argument. A transceiver in a switch port is
stable for months, so `binding_l1` has no bucket column and this stage emits none. Bucket only what moves.

The emitted records carry `binding_uid` built exactly as `morpheus.utils.binding_table` builds it, so a layer 1
binding and a layer 2 one are identified the same way and an analyst can recover the exact record behind either.
The layer 2 path gets its uid from the bucketing step; the unbucketed layer 1 path has no such step, so the uid is
computed here.
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
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.binding_closer import NS_PER_SECOND
from morpheus.utils.binding_closer import BindingCloser
from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import compose_key
from morpheus.utils.event_clock import DEFAULT_MAX_SKEW_SECONDS
from morpheus.utils.event_clock import EventClock
from morpheus.utils.lineage import event_uid

logger = logging.getLogger(__name__)

DEFAULT_ATTRIBUTE_COLUMNS = ["transceiver_serial", "lldp_neighbor_chassis_id"]
"""What is bound to a port: the optic in it and the neighbor on the other end.

Both are what the `binding_l1` lookup returns, and both are things an estate changes deliberately. Optical power
and error counters are deliberately *not* here: they vary continuously, and a binding that split whenever a
receive level moved by a tenth of a decibel would produce a new interval every poll.
"""

KEY_ATTRIBUTES = ("site_id", "device_id", "port_id")
"""The parts that compose the port key, in the order layer 1 composes `entity_key` through `entity_key.compose_key`.

The string is identical to the one `TC1NormalizeStage` emits and to the `port_key` `TC2BindingStage` carries on a
closed binding, which is what makes the ladder's first arrow a join rather than a reconstruction.
"""

SWITCH_COLUMN = "switch_id"
"""Emitted alongside `device_id`, carrying the same value.

Layer 1 calls the device `device_id` and layer 2 calls it `switch_id`. They are one identifier under two names,
and the shipped `binding_l1` refresh search keys on `switch_id`. Emitting both is what lets that search work
unchanged rather than renaming a column on a search head, which the guide is explicit about not doing.
"""

DEFAULT_IDLE_TIMEOUT_SECONDS = 7 * 24 * 3600
BINDING_TABLE_NAME = "port_inventory"

KEY_COLUMN = "entity_key"
UID_COLUMN = "binding_uid"
BIND_START_COLUMN = "bind_start"
BIND_END_COLUMN = "bind_end"
END_REASON_COLUMN = "bind_end_reason"
END_OBSERVED_COLUMN = "bind_end_observed"
OBSERVATIONS_COLUMN = "bind_observations"
PROVISIONAL_COLUMN = "bind_provisional"
OPEN_REASON = "open"


@register_stage("tc1-binding", modes=[])
class TC1BindingStage(GpuAndCpuMixin, SinglePortStage):
    """
    Close layer 1 port observations into resolvable intervals.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    key_columns : list of str, optional
        Columns composing the port. Defaults to `["site_id", "device_id", "port_id"]`, composed into `entity_key`
        exactly as the other TC-1 stages compose it. A row missing any part carries no key and is skipped rather
        than binding to a fabricated one.
    time_column : str, default = "event_time"
        Column holding the observation's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    attribute_columns : list of str, optional
        Columns making up what is bound to the port. Defaults to `["transceiver_serial",
        "lldp_neighbor_chassis_id"]`. A sample whose attributes differ from the open binding's closes it and opens
        a new one; one whose attributes match extends it. Columns outside this list are ignored, which is what
        keeps a moving optical power reading from splitting a binding every poll.
    max_clock_skew_seconds : int, default = 604800
        How far ahead of the stream's own progress a row's event time may be before it is refused. Expiry runs on
        event time, so a device whose clock is wrong by years would otherwise drive the horizon past every open
        binding at once. See `morpheus.utils.event_clock`.
    idle_timeout_seconds : int, default = 604800
        Silence after which an open binding is presumed to have ended. Days rather than the minutes layer 2 uses:
        a port is polled rather than chatty, so silence means the poller stopped rather than that the optic left,
        and a short horizon would close the whole estate during a collector outage.
    emit_open_on_complete : bool, default = True
        Close and emit every still-open binding when the stream ends. Without it a port whose optic never changed
        is never emitted at all -- which is nearly every port in a healthy estate, and exactly the rows the lookup
        exists to hold.
    emit_open_bindings : bool, default = False
        Also emit a provisional record the moment a binding opens, with a null `bind_end` and
        `bind_provisional = true`, so a port brought into service now is resolvable before its binding closes.
    """

    def __init__(self,
                 c: Config,
                 key_columns: list[str] = None,
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 attribute_columns: list[str] = None,
                 max_clock_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS,
                 idle_timeout_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS,
                 emit_open_on_complete: bool = True,
                 emit_open_bindings: bool = False):
        super().__init__(c)

        key_columns = list(KEY_ATTRIBUTES) if key_columns is None else list(key_columns)
        attribute_columns = list(DEFAULT_ATTRIBUTE_COLUMNS) if attribute_columns is None else list(attribute_columns)

        if (len(key_columns) == 0):
            raise ValueError("key_columns must name at least one column")

        if (len(attribute_columns) == 0):
            raise ValueError("attribute_columns must name at least one column; a binding to nothing has no content")

        if (idle_timeout_seconds <= 0):
            raise ValueError(f"idle_timeout_seconds must be positive, received {idle_timeout_seconds}")

        if (max_clock_skew_seconds <= 0):
            raise ValueError(f"max_clock_skew_seconds must be positive, received {max_clock_skew_seconds}")

        self._key_columns = key_columns
        self._time_column = time_column
        self._time_unit = time_unit
        self._attribute_columns = attribute_columns
        self._emit_open_on_complete = emit_open_on_complete
        self._emit_open_bindings = emit_open_bindings

        self._clock = EventClock(max_skew_ns=max_clock_skew_seconds * NS_PER_SECOND)
        self._closer = BindingCloser(attribute_names=attribute_columns,
                                     idle_timeout_ns=idle_timeout_seconds * NS_PER_SECOND)

        self._needed_columns[KEY_COLUMN] = TypeId.STRING
        self._needed_columns[UID_COLUMN] = TypeId.STRING
        self._needed_columns[BIND_START_COLUMN] = TypeId.INT64
        self._needed_columns[BIND_END_COLUMN] = TypeId.INT64
        self._needed_columns[END_REASON_COLUMN] = TypeId.STRING
        self._needed_columns[END_OBSERVED_COLUMN] = TypeId.BOOL8
        self._needed_columns[OBSERVATIONS_COLUMN] = TypeId.INT64
        self._needed_columns[PROVISIONAL_COLUMN] = TypeId.BOOL8

        for name in self._key_columns:
            self._needed_columns[name] = TypeId.STRING

        self._needed_columns[SWITCH_COLUMN] = TypeId.STRING

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc1-binding"

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
        """Declare the output type: a frame of port bindings rather than the observations that came in."""
        schema.output_schema.set_type(MessageMeta)

    def supports_cpp_node(self) -> bool:
        """Whether this stage supports a C++ node."""
        return False

    @property
    def open_count(self) -> int:
        """Bindings currently open and therefore not yet emitted."""
        return self._closer.open_count

    def _records(self, closed: list, opened: list) -> dict:
        """Render closed bindings, then provisional open ones, as columns, one row each.

        `bind_end` is assigned separately so that a provisional record's absent end is null in both execution
        modes rather than a fabricated number.
        """
        records = list(closed) + list(opened)
        ends = [record.bind_end_ns for record in closed] + [None] * len(opened)

        columns: dict[str, list] = {KEY_COLUMN: [record.key for record in records]}

        # The key's parts, recovered from the key itself so that a consumer never has to split the string. The
        # order is the composition order, which is why it can be recovered at all.
        parts = [record.key.split(":") for record in records]

        for (index, name) in enumerate(self._key_columns):
            columns[name] = [part[index] if len(part) > index else None for part in parts]

        # Layer 2's name for the same identifier, so the shipped refresh search keys on a column that is there.
        if ("device_id" in self._key_columns):
            columns[SWITCH_COLUMN] = list(columns["device_id"])

        for name in self._attribute_columns:
            columns[name] = [record.attributes.get(name) for record in records]

        columns[BIND_START_COLUMN] = [record.bind_start_ns for record in records]
        columns[END_REASON_COLUMN] = [record.end_reason for record in closed] + [OPEN_REASON] * len(opened)
        columns[END_OBSERVED_COLUMN] = [record.end_observed for record in closed] + [False] * len(opened)
        columns[OBSERVATIONS_COLUMN] = [record.observations for record in records]
        columns[PROVISIONAL_COLUMN] = [False] * len(closed) + [True] * len(opened)

        # Built exactly as `binding_table` builds it, so the two layers identify a binding the same way.
        columns[UID_COLUMN] = [
            event_uid(BINDING_TABLE_NAME,
                      record.key,
                      record.bind_start_ns,
                      ends[position],
                      *[record.attributes.get(name) for name in self._attribute_columns])
            for (position, record) in enumerate(records)
        ]

        return columns

    def _emit(self, closed: list, opened: list = ()) -> list:
        """Wrap closed and provisional bindings in a frame, or nothing when there are none."""
        opened = list(opened)

        if (len(closed) == 0 and len(opened) == 0):
            return []

        # Imported here so that this module remains importable in CPU-only environments where cuDF is absent.
        from morpheus.utils.type_utils import get_df_class  # pylint: disable=import-outside-toplevel

        df = get_df_class(self._config.execution_mode)(self._records(closed, opened))

        # A provisional record has no end. Null here, in both modes, is what tells a consumer to apply its own
        # assumed duration rather than read a fabricated one, and it is assigned separately because a plain
        # column of Python `None` widens an integer column to float in one mode and not the other.
        assign_nullable_int_column(df,
                                   BIND_END_COLUMN, [record.bind_end_ns for record in closed] + [None] * len(opened))

        return [MessageMeta(df)]

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]) -> list:
        """
        Feed a batch of port observations to the closer and emit whatever bindings ended.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming observations.

        Returns
        -------
        list of `morpheus.messages.MessageMeta`
            One frame of port bindings, or an empty list when this batch ended none, which is the normal case for
            an estate where nobody changed an optic.

        Raises
        ------
        KeyError
            If a key column, the time column, or a declared attribute column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return []

        source = meta.copy_dataframe()

        required = list(self._key_columns) + [self._time_column] + self._attribute_columns
        missing = [column for column in required if column not in source.columns]

        if (len(missing) > 0):
            raise KeyError(f"TC1BindingStage requires columns {missing} which are not present in the DataFrame. "
                           f"Available columns: {sorted(source.columns)}")

        key_parts = {name: to_host_list(source, name) for name in self._key_columns}
        raw_times = to_host_list(source, self._time_column)
        attributes = {name: to_host_list(source, name) for name in self._attribute_columns}
        row_count = len(raw_times)

        closed = []
        opened_keys: list[str] = []
        unordered = 0
        implausible = 0

        for position in range(row_count):
            key = compose_key([key_parts[name][position] for name in self._key_columns])

            try:
                event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
            except ValueError:
                event_time_ns = None

            if (event_time_ns is None or key is None):
                unordered += 1
                continue

            if (not self._clock.accept(event_time_ns)):
                implausible += 1
                continue

            # A port that has not been polled for longer than the idle timeout is presumed out of service, and its
            # binding ends where the silence began. Expiry runs on this row's own event time, which keeps the
            # closure in the same place however the stream is divided into batches.
            closed.extend(self._closer.expire(event_time_ns))

            result = self._closer.observe(key,
                                          event_time_ns,
                                          {name: attributes[name][position]
                                           for name in self._attribute_columns})

            closed.extend(result.closed)
            unordered += int(result.out_of_order)

            if (result.opened and self._emit_open_bindings):
                opened_keys.append(key)

        if (unordered > 0):
            logger.warning(
                "TC1BindingStage skipped %d of %d observations that were out of order, missing a key part, or "
                "without a usable event time; they did not advance any binding. Shard by device and preserve "
                "per-port ordering upstream.",
                unordered,
                row_count)

        if (implausible > 0):
            logger.warning(
                "TC1BindingStage refused %d of %d rows whose event time was further ahead of the stream than "
                "max_clock_skew_seconds allows; they opened no binding and did not drive expiry.",
                implausible,
                row_count)

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

        A port whose optic never changed is otherwise never emitted, and in a healthy estate that is nearly every
        port -- which is to say, nearly the whole lookup.
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
