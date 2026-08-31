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
"""Counts link flaps per interval, including those hidden between polls."""

import logging
import math
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
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.link_flap import NS_PER_SECOND
from morpheus.utils.link_flap import LinkFlapTracker

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 3600
"""Interval the windowed flap count covers, in seconds of event time."""

FLAP_COLUMN = "link_flaps"
"""Lower bound on transitions since the previous sample."""

WINDOW_COLUMN = "link_flaps_in_window"
"""Lower bound on transitions over the trailing window."""

FLAG_COLUMNS = ("link_flap_unpolled", "link_flap_device_reset", "link_flap_last_change_inconsistent")
"""Boolean columns describing how each count was arrived at."""


@register_stage("tc1-flap")
class TC1FlapStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Count link state transitions per port, including the ones polling cannot see.

    A port that changes operational state repeatedly is unstable, and instability at layer 1 is what precedes a
    layer 2 loop: every transition re-triggers spanning tree, and enough of them in a short window is a fault about
    to become an outage. What matters is therefore the count of transitions, not the current state.

    Comparing `oper_status` between polls counts only transitions visible at a poll boundary, and misses in the
    worst direction: a port that drops and recovers inside one sixty-second gap looks perfectly stable, and that is
    precisely the flapping port. `last_change_column` closes it. If the device's own record of when the interface
    last changed advanced between two polls, the interface transitioned even when both polls saw the same status,
    and it did so an even number of times, so at least twice.

    Every count is a lower bound, because the device records only the most recent transition: a port that flapped
    nine times between polls still reports two. That is a floor rather than an estimate, which is the right way
    round here, since an under-counted flapping port is still flagged while an interpolated guess would put a
    number nobody measured in front of an analyst.

    The stage is stateful across messages and must run single-engine. For parallelism, shard by device upstream and
    give each shard its own instance, which is determinism control 4. Place it after
    `morpheus.stages.telemetry.tc1_normalize_stage.TC1NormalizeStage`, which supplies `entity_key`.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    entity_key_column : str, default = "entity_key"
        Column holding the port identity transitions are counted per.
    time_column : str, default = "event_time"
        Column holding the sample's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    status_column : str, default = "oper_status"
        Column holding the interface's operational state. Compared only for equality, so any stable rendering
        works as long as one collector does not switch renderings mid-stream.
    last_change_column : str, optional
        Column holding when the interface last changed state, as an absolute time. Strongly recommended: without
        it only transitions visible at a poll boundary are counted, and a flap contained inside the polling gap is
        invisible. Devices report this relative to their own uptime, so the collector must normalize it; a value
        that goes backwards is read as a device restart, which is itself a transition.
    last_change_unit : str, default = "ns"
        Unit for numeric timestamps in `last_change_column`.
    window_seconds : int, default = 3600
        Interval the windowed count covers.
    """

    def __init__(self,
                 c: Config,
                 entity_key_column: str = "entity_key",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 status_column: str = "oper_status",
                 last_change_column: str = None,
                 last_change_unit: str = "ns",
                 window_seconds: int = DEFAULT_WINDOW_SECONDS):
        super().__init__(c)

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        self._entity_key_column = entity_key_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._status_column = status_column
        self._last_change_column = last_change_column
        self._last_change_unit = last_change_unit

        self._tracker = LinkFlapTracker(window_ns=window_seconds * NS_PER_SECOND)

        self._needed_columns[FLAP_COLUMN] = TypeId.INT64
        self._needed_columns[WINDOW_COLUMN] = TypeId.INT64

        for flag in FLAG_COLUMNS:
            self._needed_columns[flag] = TypeId.BOOL8

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc1-flap"

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

    def _last_change_ns(self, value: typing.Any) -> typing.Optional[int]:
        if (value is None):
            return None

        try:
            return to_epoch_ns(value, time_unit=self._last_change_unit)
        except ValueError:
            return None

    @staticmethod
    def _status(value: typing.Any) -> typing.Optional[str]:
        if (value is None):
            return None

        # A null in a string column can arrive as a float NaN, which must not compare unequal to itself and be
        # mistaken for a transition on every sample.
        if (isinstance(value, float) and math.isnan(value)):
            return None

        return str(value)

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the per-sample and windowed flap counts, and the flags describing how they were reached.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming message.

        Returns
        -------
        The input message, with the flap columns populated.

        Raises
        ------
        KeyError
            If the entity key, the time column, or the status column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._entity_key_column, self._time_column, self._status_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC1FlapStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            entity_keys = to_host_list(df, self._entity_key_column)
            raw_times = to_host_list(df, self._time_column)
            statuses = to_host_list(df, self._status_column)

            if (self._last_change_column is not None and self._last_change_column in df.columns):
                changes = [self._last_change_ns(value) for value in to_host_list(df, self._last_change_column)]
            else:
                changes = [None] * len(entity_keys)

            flaps: list = []
            windowed: list = []
            flags: dict[str, list] = {name: [] for name in FLAG_COLUMNS}
            unordered = 0

            for (position, entity_key) in enumerate(entity_keys):
                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                if (event_time_ns is None):
                    flaps.append(None)
                    windowed.append(None)

                    for name in FLAG_COLUMNS:
                        flags[name].append(False)

                    unordered += 1
                    continue

                result = self._tracker.observe(str(entity_key),
                                               event_time_ns,
                                               self._status(statuses[position]),
                                               last_change_ns=changes[position])

                flaps.append(result.flaps)
                windowed.append(result.flaps_in_window)
                flags["link_flap_unpolled"].append(result.last_change_advanced and (result.flaps or 0) > 1)
                flags["link_flap_device_reset"].append(result.device_reset)
                flags["link_flap_last_change_inconsistent"].append(result.last_change_inconsistent)
                unordered += int(result.out_of_order)

            assign_nullable_int_column(df, FLAP_COLUMN, flaps)
            assign_nullable_int_column(df, WINDOW_COLUMN, windowed)

            for (name, values) in flags.items():
                df[name] = values

        if (unordered > 0):
            logger.warning(
                "TC1FlapStage saw %d of %d samples out of order or without a usable event time; they carry no "
                "count and did not advance any port's state. Shard by device and preserve per-port ordering "
                "upstream.",
                unordered,
                len(entity_keys))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
