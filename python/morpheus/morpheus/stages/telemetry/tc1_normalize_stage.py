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
"""Normalizes layer 1 collector output into the TC-1 telemetry envelope."""

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
from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.counter_delta import NS_PER_SECOND
from morpheus.utils.counter_delta import CounterTracker
from morpheus.utils.counter_delta import TIMETICKS_CEILING_NS
from morpheus.utils.entity_key import KEY_SEPARATOR
from morpheus.utils.entity_key import compose_key

logger = logging.getLogger(__name__)

DEFAULT_COUNTER_COLUMNS = ["crc_errors", "symbol_errors", "input_discards", "output_discards"]
"""Counters the TC-1 telemetry class requires as deltas."""

ENTITY_KEY_SEPARATOR = KEY_SEPARATOR
"""Separator for the `site_id:device_id:port_id` entity key. Shared with the layer 2 stages, whose `port_key` is
the same three values composed the same way, so a resolved MAC lands on exactly this string.
"""


@register_stage("tc1-normalize")
class TC1NormalizeStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Turn raw layer 1 collector records into the TC-1 envelope.

    Collection itself sits outside Morpheus: an SNMP and LLDP poller publishes per-port records to a topic and this
    stage consumes them. What it adds is the part that cannot be done statelessly at the collector, because it
    requires the previous sample: interface counters arrive as monotonic totals, and a total is useless as a
    feature. It grows without bound, it wraps at its ceiling, and it restarts at zero when the device reboots, so a
    rule written against one fires on the device's age rather than on its behavior.

    Each row therefore gains a delta per counter, the interval that delta actually covers, and flags saying whether
    the counter wrapped, whether the device reset, and whether the sample arrived out of order. A wrap and a reboot
    look identical in the counter alone, so `uptime_column` is what tells them apart; without it a decrease yields
    no delta rather than a guess, because both candidate answers would be fabrications that read like measurements.

    Each row also gains `entity_key`, the `site_id:device_id:port_id` identity that the per-port models key on and
    that anchors the bottom of the lineage ladder.

    This stage is stateful and must run single-engine, which is the Morpheus default. For parallelism, shard by
    device upstream and give each shard its own instance, which is determinism control 4. Rows are processed in the
    order given: a delta depends on the sample before it, so out-of-order input is flagged rather than silently
    turned into a negative or inflated count.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    site_column : str, default = "site_id"
        Column holding the site identifier.
    device_column : str, default = "device_id"
        Column holding the device identifier.
    port_column : str, default = "port_id"
        Column holding the port identifier.
    time_column : str, default = "event_time"
        Column holding the sample's event time. Event time, never ingest time: a delta computed against ingest
        order is a measurement of the collector's scheduling, not of the interface.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    counter_columns : list of str, optional
        Raw counter columns to difference. Defaults to the four the telemetry class requires.
    counter32_columns : list of str, optional
        Which of those are 32-bit. SNMP `Counter32` values wrap far more often than 64-bit ones, and a wrap that is
        not declared is mistaken for a reset.
    uptime_column : str, optional
        Column holding device uptime, in `uptime_unit`. Strongly recommended: it is the only thing that
        distinguishes a counter wrap from a reboot.
    uptime_unit : str, default = "s"
        Unit of `uptime_column`. SNMP `sysUpTime` is `TimeTicks`, hundredths of a second: pass `"cs"` for it, since
        leaving it at seconds inflates every uptime a hundredfold and reboots stop being detected. Passing `"cs"`
        also tells the tracker that the uptime counter itself rolls over at 2**32 centiseconds, about 497 days, so a
        device that has been up longer than that is not mistaken for one that just restarted.
    delta_suffix : str, default = "_delta"
        Suffix for the emitted delta columns.
    entity_key_column : str, default = "entity_key"
        Column to write the composite entity identity to.
    """

    def __init__(self,
                 c: Config,
                 site_column: str = "site_id",
                 device_column: str = "device_id",
                 port_column: str = "port_id",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 counter_columns: list[str] = None,
                 counter32_columns: list[str] = None,
                 uptime_column: str = None,
                 uptime_unit: str = "s",
                 delta_suffix: str = "_delta",
                 entity_key_column: str = "entity_key"):
        super().__init__(c)

        # An empty list is a configuration error, not a request for the defaults, so `None` is what selects them.
        counter_columns = list(DEFAULT_COUNTER_COLUMNS) if counter_columns is None else list(counter_columns)
        counter32_columns = [] if counter32_columns is None else list(counter32_columns)

        if (len(counter_columns) == 0):
            raise ValueError("counter_columns must contain at least one counter")

        if (not delta_suffix):
            raise ValueError("delta_suffix is required, or the deltas would overwrite the raw counters")

        unknown = [name for name in counter32_columns if name not in counter_columns]

        if (len(unknown) > 0):
            raise ValueError(f"counter32_columns names {unknown}, which are not in counter_columns")

        self._key_columns = [site_column, device_column, port_column]
        self._time_column = time_column
        self._time_unit = time_unit
        self._counter_columns = counter_columns
        self._uptime_column = uptime_column
        self._uptime_unit = uptime_unit
        self._delta_suffix = delta_suffix
        self._entity_key_column = entity_key_column

        # SNMP reports `sysUpTime` as `TimeTicks`, which is a 32-bit counter of centiseconds and rolls over after
        # about 497 days. Telling the tracker where that ceiling is lets it separate a rollover, where the device has
        # been up longer than the counter can express, from a genuine restart. Without it a rollover reads as a
        # reboot and the port's whole accumulated error total is emitted as one interval's delta.
        self._tracker = CounterTracker(
            counter_names=counter_columns,
            counter_bits={name: 32 if name in counter32_columns else 64
                          for name in counter_columns},
            uptime_ceiling_ns=TIMETICKS_CEILING_NS if uptime_unit == "cs" else None)

        self._needed_columns[entity_key_column] = TypeId.STRING
        self._needed_columns["interval_seconds"] = TypeId.FLOAT64

        for flag in ("counter_reset", "counter_wrapped", "sample_out_of_order"):
            self._needed_columns[flag] = TypeId.BOOL8

        for name in counter_columns:
            self._needed_columns[f"{name}{delta_suffix}"] = TypeId.INT64

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc1-normalize"

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

    def _uptime_ns(self, value: typing.Any) -> typing.Optional[int]:
        if (value is None):
            return None

        try:
            return to_epoch_ns(value, time_unit=self._uptime_unit)
        except ValueError:
            return None

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Normalize every row of the message payload.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming message.

        Returns
        -------
        The input message, with the entity key, the counter deltas, the covered interval, and the reset, wrap, and
        ordering flags populated.

        Raises
        ------
        KeyError
            If a key column, the time column, or a declared counter column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = self._key_columns + [self._time_column] + self._counter_columns
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC1NormalizeStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            key_parts = [to_host_list(df, column) for column in self._key_columns]
            # A missing part yields a null key rather than the string "None": half an identity is not an identity.
            entity_keys = [compose_key(row) for row in zip(*key_parts)]

            raw_times = to_host_list(df, self._time_column)
            counters = {name: to_host_list(df, name) for name in self._counter_columns}

            if (self._uptime_column is not None and self._uptime_column in df.columns):
                uptimes = [self._uptime_ns(value) for value in to_host_list(df, self._uptime_column)]
            else:
                uptimes = [None] * len(entity_keys)

            deltas: dict[str, list] = {name: [] for name in self._counter_columns}
            intervals: list[typing.Optional[float]] = []
            flags: dict[str, list[bool]] = {"counter_reset": [], "counter_wrapped": [], "sample_out_of_order": []}
            unordered = 0
            keyless = 0

            for (position, entity_key) in enumerate(entity_keys):
                if (entity_key is None):
                    # No identity, so no previous sample to difference against. The row passes through with the
                    # deltas null rather than being attributed to a fabricated port.
                    for name in self._counter_columns:
                        deltas[name].append(None)

                    intervals.append(None)
                    flags["counter_reset"].append(False)
                    flags["counter_wrapped"].append(False)
                    flags["sample_out_of_order"].append(False)
                    keyless += 1
                    continue

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                if (event_time_ns is None):
                    for name in self._counter_columns:
                        deltas[name].append(None)

                    intervals.append(None)
                    flags["counter_reset"].append(False)
                    flags["counter_wrapped"].append(False)
                    flags["sample_out_of_order"].append(True)
                    unordered += 1
                    continue

                result = self._tracker.observe(entity_key,
                                               event_time_ns,
                                               {name: counters[name][position]
                                                for name in self._counter_columns},
                                               uptime_ns=uptimes[position])

                for name in self._counter_columns:
                    deltas[name].append(result.deltas[name])

                intervals.append(None if result.interval_ns is None else result.interval_ns / NS_PER_SECOND)
                flags["counter_reset"].append(result.counter_reset)
                flags["counter_wrapped"].append(result.counter_wrapped)
                flags["sample_out_of_order"].append(result.out_of_order)
                unordered += int(result.out_of_order)

            assign_str_column(df, self._entity_key_column, entity_keys)

            for name in self._counter_columns:
                assign_nullable_int_column(df, f"{name}{self._delta_suffix}", deltas[name])

            assign_nullable_float_column(df, "interval_seconds", intervals)

            for (flag, values) in flags.items():
                df[flag] = values

        if (keyless > 0):
            logger.warning(
                "TC1NormalizeStage saw %d of %d samples with a null site, device, or port; they carry no entity key "
                "and no deltas. A collector that omits any of the three cannot anchor the lineage ladder.",
                keyless,
                len(entity_keys))

        if (unordered > 0):
            logger.warning(
                "TC1NormalizeStage saw %d of %d samples out of order or without a usable event time; their deltas "
                "are null. Shard by device and preserve per-port ordering upstream.",
                unordered,
                len(entity_keys))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
