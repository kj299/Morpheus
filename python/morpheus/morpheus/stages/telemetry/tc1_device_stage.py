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
"""Derives restarts, configuration changes and routing churn per switch or router, keyed on the device itself."""

import dataclasses
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
from morpheus.pipeline.checkpoint_mixin import CheckpointMixin
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.bucket_peak import BucketPeakTracker
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.counter_delta import TIMETICKS_CEILING_NS
from morpheus.utils.counter_delta import device_restarted
from morpheus.utils.entity_key import compose_key
from morpheus.utils.event_clock import CLOCK_REFUSED_COLUMN
from morpheus.utils.event_clock import DEFAULT_MAX_SKEW_SECONDS
from morpheus.utils.event_clock import NS_PER_SECOND
from morpheus.utils.event_clock import EventClock
from morpheus.utils.event_clock import mark_refused
from morpheus.utils.tracker_state import StateMixin
from morpheus.utils.value_novelty import ValueNoveltyTracker

logger = logging.getLogger(__name__)

DEFAULT_BUCKET_SECONDS = 86400
"""The period restarts are counted per. A day: a switch that restarts once a month and one that restarts every
night differ in their days, not their hours."""

DEFAULT_WINDOW_SECONDS = 30 * 86400
"""How far back the per-device restart baseline reaches."""

DEFAULT_MIN_BUCKETS = 7
"""Committed periods required before the baseline is published. A week of days."""

DEFAULT_MAX_DEVICES = 100_000
"""Devices retained before the least recently seen is forgotten. An estate has far fewer switches than ports."""

SECONDS_PER_DAY = 86400

DEVICE_KEY = "device_key"
REBOOTED = "device_rebooted"
REBOOTS_IN_BUCKET = "device_reboots_in_bucket"
REBOOT_BASELINE = "device_reboot_baseline"
REBOOT_BASELINE_MATURE = "device_reboot_baseline_mature"
REBOOT_RISING = "device_reboot_rising_buckets"
CONFIG_CHANGED = "config_changed"
CONFIG_FIRST_SEEN = "config_first_seen"
CONFIG_IN_WINDOW = "config_change_in_window"
ROUTE_DELTA = "route_count_delta"

CONFIG_FIELD = "config"


def _parse_clock(text: str) -> int:
    (hours, minutes) = (int(part) for part in text.split(":"))

    if (not (0 <= hours < 24 and 0 <= minutes < 60)):
        raise ValueError(f"change_window times must be HH:MM in UTC, received {text!r}")

    return (hours * 60 + minutes) * 60


def parse_change_window(window: typing.Optional[typing.Sequence[str]]) -> typing.Optional[tuple[int, int]]:
    """
    Read a change window of two UTC clock times into seconds after midnight.

    Parameters
    ----------
    window : sequence of two str, optional
        `["HH:MM", "HH:MM"]`, the start inclusive and the end exclusive. A window whose end is earlier than its start
        wraps midnight: `["22:00", "02:00"]` is four hours.

    Returns
    -------
    tuple of two int, or None
        The start and end in seconds after midnight, or `None` when no window was given.

    Raises
    ------
    ValueError
        If the window is not two clock times, or its two ends are the same.
    """
    if (window is None):
        return None

    if (isinstance(window, str) or len(window) != 2):
        raise ValueError(f"change_window must be two HH:MM times, received {window!r}")

    try:
        (start, end) = (_parse_clock(str(part)) for part in window)
    except (TypeError, ValueError) as error:
        raise ValueError(f"change_window must be two HH:MM times, received {window!r}") from error

    if (start == end):
        raise ValueError(f"change_window {window!r} is empty; a window that admits nothing is no window")

    return (start, end)


def in_change_window(event_time_ns: int, window: tuple[int, int]) -> bool:
    """Whether an event time's UTC clock time falls inside a window `parse_change_window` returned."""
    (start, end) = window
    second = (event_time_ns // NS_PER_SECOND) % SECONDS_PER_DAY

    if (start < end):
        return start <= second < end

    return second >= start or second < end


@dataclasses.dataclass
class _DeviceState:
    last_time_ns: typing.Optional[int] = None
    last_uptime_ns: typing.Optional[int] = None
    last_routes: typing.Any = None
    bucket: typing.Optional[int] = None
    reboots: int = 0
    previous_reboots: int = 0
    previous_rising: int = 0


class RebootCounter(StateMixin):
    """
    Per-device restarts per period, and how many periods in a row each held more than the one before.

    A device that restarts once is an event; one whose restarts are each closer to the last is a trajectory, and
    the trajectory is the predictive signal. It is a count of consecutive periods, each with more restarts than the
    one before, ending at the period this observation is in: one, two and three restarts in three successive days
    is three. A period with no observation in it held no restarts, so a gap ends the run.

    The open period's figure is the count so far, so it rises as the period's restarts arrive and every row is
    measured with what the stream had shown by then. That is what makes it a pure function of the rows in order.

    Parameters
    ----------
    bucket_ns : int
        The period restarts are counted per.
    max_devices : int
        Devices retained before the least recently seen is forgotten.
    """

    def __init__(self, bucket_ns: int, max_devices: int):
        self._bucket_ns = bucket_ns
        self._max_devices = max_devices
        self._states: dict[str, _DeviceState] = {}

    def state(self, device_key: str) -> _DeviceState:
        """The device's state, created on first sight and marked most recently seen."""
        state = self._states.pop(device_key, None)

        if (state is None):
            state = _DeviceState()

        self._states[device_key] = state

        while (len(self._states) > self._max_devices):
            del self._states[next(iter(self._states))]

        return state

    def count(self, state: _DeviceState, event_time_ns: int, rebooted: bool) -> tuple[int, int]:
        """Add one observation to its period and return the period's restarts so far and the rising run."""
        bucket = event_time_ns // self._bucket_ns

        if (state.bucket is None):
            state.bucket = bucket
        elif (bucket > state.bucket):
            # The period this observation opens closes the one before it. Whether that one was rising is settled
            # now, against the one before it; an empty period between them held nothing, so nothing rises past it.
            closing_rising = (state.previous_rising + 1) if state.reboots > state.previous_reboots else 0

            if (bucket == state.bucket + 1):
                (state.previous_reboots, state.previous_rising) = (state.reboots, closing_rising)
            else:
                (state.previous_reboots, state.previous_rising) = (0, 0)

            state.bucket = bucket
            state.reboots = 0

        state.reboots += int(rebooted)
        rising = (state.previous_rising + 1) if state.reboots > state.previous_reboots else 0

        return (state.reboots, rising)


@register_stage("tc1-device", ignore_args=["change_window"])
class TC1DeviceStage(CheckpointMixin, GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Derive what happened to a switch or router as a whole: restarts, configuration changes and routing churn.

    A port key carries its device only as a middle segment, so a restart surfaces once per port and a configuration
    push nowhere at all. This stage reads the device's own poll -- `sysUpTime`, a digest of its running configuration,
    the size of its routing table -- keyed `site_id:device_id`, and gives the device features of its own.

    **A restart is read from the uptime alone**, the way `morpheus.utils.counter_delta` reads it for port counters:
    an uptime that went backwards, or advanced by less than the time between the two polls, means the device came up
    in between, unless the uptime counter itself rolled over, which the time between the polls distinguishes. A
    device's first poll establishes its uptime and says nothing about a restart.

    **The restart rate has a baseline per device.** Restarts are counted per period, `bucket_seconds`, and each
    period's count is measured against the most any earlier period inside `window_seconds` held, with
    `morpheus.utils.bucket_peak`, the tracker the port baselines use. `device_reboot_rising_buckets` is how many
    periods in a row, ending at this one, each held more restarts than the one before -- one, then two, then three
    -- which is the trajectory R-P-L1-009 reads: a device failing gets worse, where one that restarts for
    maintenance does it once.

    **A configuration change is a change in the digest**, `config_changed`, with `config_first_seen` saying whether
    the device has ever run that configuration before, from `morpheus.utils.value_novelty` as the port identifiers
    are. Where the estate declares a change window, `config_change_in_window` says whether a change landed inside
    it: R-D-L1-008 reads a change outside the window, on a device whose changes are supposed to be scheduled. The
    window is a pair of UTC clock times and is the estate's to declare, so without one the column is null rather
    than a guess at when somebody meant to work.

    **Routing churn is the change in the routing table's size**, `route_count_delta`, from the device's previous
    poll. It is context on the restart and the change rather than a signal of its own: a router that loses two thirds
    of its routes has lost a peer, and whether that was the change somebody pushed is what the analyst asks next.

    The stage is stateful across messages and must run single-engine, behind a total order. For parallelism, shard by
    device upstream and give each shard its own instance, which is determinism control 4.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    site_column : str, default = "site_id"
        Column holding the site, the first part of the device key.
    device_column : str, default = "device_id"
        Column holding the device, the second part.
    time_column : str, default = "event_time"
        Column holding the poll's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    uptime_column : str, default = "sys_uptime"
        Column holding the device's uptime. Without it no restart is read and the restart columns are null.
    uptime_unit : str, default = "cs"
        Unit of `uptime_column`. SNMP `sysUpTime` is `TimeTicks`, hundredths of a second, and with `"cs"` the counter's
        own 32-bit ceiling is applied, so a device up for more than 497 days is not read as restarting.
    config_column : str, default = "running_config_hash"
        Column holding a digest of the running configuration. Without it the configuration columns are null. Not
        `config_hash`, which is the pipeline's own configuration digest and is what `DeterminismStampStage` writes.
    route_column : str, default = "route_count"
        Column holding the size of the routing table. Without it `route_count_delta` is null.
    change_window : list of two str, optional
        The estate's change window, `["HH:MM", "HH:MM"]` in UTC, start inclusive and end exclusive; it may wrap
        midnight. Without one, `config_change_in_window` is null.
    bucket_seconds : int, default = 86400
        The period restarts are counted per.
    window_seconds : int, default = 30 days
        How far back the restart baseline reaches.
    min_buckets : int, default = 7
        Committed periods required before the restart baseline is published.
    max_devices : int, default = 100000
        Devices retained before the least recently seen is forgotten. A forgotten device starts from no history.
    max_clock_skew_seconds : int, default = 604800
        How far ahead of the stream's own progress a row's event time may be before it is refused. A refused row
        carries no device features, sets `clock_refused`, and moves no state. See `morpheus.utils.event_clock`.
    """

    _checkpoint_attributes = ("_baseline", "_reboots", "_configs", "_clock")

    def __init__(self,
                 c: Config,
                 site_column: str = "site_id",
                 device_column: str = "device_id",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 uptime_column: str = "sys_uptime",
                 uptime_unit: str = "cs",
                 config_column: str = "running_config_hash",
                 route_column: str = "route_count",
                 change_window: list[str] = None,
                 bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 min_buckets: int = DEFAULT_MIN_BUCKETS,
                 max_devices: int = DEFAULT_MAX_DEVICES,
                 max_clock_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS):
        super().__init__(c)

        if (max_clock_skew_seconds <= 0):
            raise ValueError(f"max_clock_skew_seconds must be positive, received {max_clock_skew_seconds}")

        if (max_devices <= 0):
            raise ValueError(f"max_devices must be positive, received {max_devices}")

        self._site_column = site_column
        self._device_column = device_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._uptime_column = uptime_column
        self._uptime_unit = uptime_unit
        self._config_column = config_column
        self._route_column = route_column
        self._change_window = parse_change_window(change_window)
        self._uptime_ceiling_ns = TIMETICKS_CEILING_NS if uptime_unit == "cs" else None

        # The tracker validates the period, window and floor, so a bad one is refused before any row is read.
        self._baseline = BucketPeakTracker(bucket_ns=bucket_seconds * NS_PER_SECOND,
                                           window_ns=window_seconds * NS_PER_SECOND,
                                           min_buckets=min_buckets,
                                           max_entities=max_devices)
        self._reboots = RebootCounter(bucket_ns=bucket_seconds * NS_PER_SECOND, max_devices=max_devices)
        self._configs = ValueNoveltyTracker([CONFIG_FIELD], max_entities=max_devices)

        self._needed_columns[DEVICE_KEY] = TypeId.STRING
        self._needed_columns[REBOOTED] = TypeId.BOOL8
        self._needed_columns[REBOOTS_IN_BUCKET] = TypeId.INT64
        self._needed_columns[REBOOT_BASELINE] = TypeId.INT64
        self._needed_columns[REBOOT_BASELINE_MATURE] = TypeId.BOOL8
        self._needed_columns[REBOOT_RISING] = TypeId.INT64
        self._needed_columns[CONFIG_CHANGED] = TypeId.BOOL8
        self._needed_columns[CONFIG_FIRST_SEEN] = TypeId.BOOL8
        self._needed_columns[CONFIG_IN_WINDOW] = TypeId.BOOL8
        self._needed_columns[ROUTE_DELTA] = TypeId.INT64

        self._clock = EventClock(max_skew_ns=max_clock_skew_seconds * NS_PER_SECOND)
        self._needed_columns[CLOCK_REFUSED_COLUMN] = TypeId.BOOL8

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc1-device"

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
        if (value is None or (isinstance(value, float) and math.isnan(value))):
            return None

        try:
            return to_epoch_ns(value, time_unit=self._uptime_unit)
        except ValueError:
            return None

    @staticmethod
    def _routes(value: typing.Any) -> typing.Optional[int]:
        if (value is None or (isinstance(value, float) and math.isnan(value))):
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _config_digest(value: typing.Any) -> typing.Any:
        # A null in a string column can arrive as a float NaN, which must not compare unequal to itself and be read
        # as a change on every poll.
        if (isinstance(value, float) and math.isnan(value)):
            return None

        return None if value is None else str(value)

    def _optional_column(self, df, column: typing.Optional[str], rows: int) -> list:
        if (column is not None and column in df.columns):
            return to_host_list(df, column)

        return [None] * rows

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write each poll's device key and its restart, configuration and routing features.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming message.

        Returns
        -------
        The input message, with the device columns populated.

        Raises
        ------
        KeyError
            If the site, device or time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._site_column, self._device_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC1DeviceStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            rows = meta.count
            keys = [
                compose_key(parts)
                for parts in zip(to_host_list(df, self._site_column), to_host_list(df, self._device_column))
            ]
            raw_times = to_host_list(df, self._time_column)
            has_uptime = self._uptime_column is not None and self._uptime_column in df.columns
            has_config = self._config_column is not None and self._config_column in df.columns
            uptimes = [self._uptime_ns(value) for value in self._optional_column(df, self._uptime_column, rows)]
            configs = [self._config_digest(value) for value in self._optional_column(df, self._config_column, rows)]
            routes = [self._routes(value) for value in self._optional_column(df, self._route_column, rows)]

            columns: dict[str, list] = {
                name: [None] * rows
                for name in (REBOOTED, REBOOTS_IN_BUCKET, REBOOT_BASELINE, REBOOT_BASELINE_MATURE, REBOOT_RISING,
                             CONFIG_CHANGED, CONFIG_FIRST_SEEN, CONFIG_IN_WINDOW, ROUTE_DELTA)
            }
            refused = [False] * rows
            keyless = 0
            unordered = 0

            for (position, key) in enumerate(keys):
                if (key is None):
                    # A device with no site or no name has no identity to hold state against; pooling such rows
                    # would make every one of them one device restarting on every poll.
                    keyless += 1
                    continue

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                if (event_time_ns is None):
                    unordered += 1
                    continue

                if (not self._clock.accept(event_time_ns)):
                    refused[position] = True
                    continue

                state = self._reboots.state(key)

                if (state.last_time_ns is not None and event_time_ns <= state.last_time_ns):
                    # A late or duplicate poll would be measured against a poll that came after it, so a restart's
                    # uptime would read as running backwards. It carries nothing and moves nothing.
                    unordered += 1
                    continue

                uptime_ns = uptimes[position]
                rebooted = None

                if (has_uptime and uptime_ns is not None and state.last_time_ns is not None):
                    rebooted = device_restarted(state.last_uptime_ns,
                                                uptime_ns,
                                                event_time_ns - state.last_time_ns,
                                                self._uptime_ceiling_ns)

                if (has_uptime):
                    (in_bucket, rising) = self._reboots.count(state, event_time_ns, bool(rebooted))
                    baseline = self._baseline.observe(key, event_time_ns, in_bucket)

                    columns[REBOOTED][position] = rebooted
                    columns[REBOOTS_IN_BUCKET][position] = in_bucket
                    columns[REBOOT_RISING][position] = rising
                    columns[REBOOT_BASELINE][position] = baseline.reference
                    columns[REBOOT_BASELINE_MATURE][position] = baseline.mature

                if (has_config):
                    novelty = self._configs.observe(key, event_time_ns, {CONFIG_FIELD: configs[position]})
                    changed = novelty.changed[CONFIG_FIELD]

                    columns[CONFIG_CHANGED][position] = changed
                    columns[CONFIG_FIRST_SEEN][position] = novelty.first_seen[CONFIG_FIELD]

                    if (changed and self._change_window is not None):
                        columns[CONFIG_IN_WINDOW][position] = in_change_window(event_time_ns, self._change_window)

                if (routes[position] is not None and state.last_routes is not None):
                    columns[ROUTE_DELTA][position] = routes[position] - state.last_routes

                state.last_time_ns = event_time_ns
                state.last_uptime_ns = uptime_ns if uptime_ns is not None else state.last_uptime_ns
                state.last_routes = routes[position] if routes[position] is not None else state.last_routes

            assign_str_column(df, DEVICE_KEY, keys)

            for name in (REBOOTED, REBOOT_BASELINE_MATURE, CONFIG_CHANGED, CONFIG_FIRST_SEEN, CONFIG_IN_WINDOW):
                assign_nullable_bool_column(df, name, columns[name])

            for name in (REBOOTS_IN_BUCKET, REBOOT_BASELINE, REBOOT_RISING, ROUTE_DELTA):
                assign_nullable_int_column(df, name, columns[name])

            mark_refused(df, refused)

        if (any(refused)):
            logger.warning(
                "TC1DeviceStage refused %d of %d polls whose event time was further ahead of the stream than "
                "max_clock_skew_seconds allows; they carry no device features and set clock_refused.",
                sum(refused),
                rows)

        if (keyless > 0):
            logger.warning(
                "TC1DeviceStage saw %d of %d polls with no site or no device; they carry no device features. A null "
                "part means a missing site or hostname upstream, not a device named \"None\".",
                keyless,
                rows)

        if (unordered > 0):
            logger.warning(
                "TC1DeviceStage saw %d of %d polls out of order, duplicated, or without a usable event time; they "
                "carry no device features and moved no device's state. Shard by device and preserve per-device "
                "ordering upstream.",
                unordered,
                rows)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
