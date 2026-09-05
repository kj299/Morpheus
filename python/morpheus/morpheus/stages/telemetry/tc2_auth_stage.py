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
"""Times 802.1X authorization per port."""

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
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import KEY_SEPARATOR
from morpheus.utils.entity_key import compose_key
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.event_clock import DEFAULT_MAX_SKEW_SECONDS
from morpheus.utils.event_clock import EventClock
from morpheus.utils.session_timer import NS_PER_SECOND
from morpheus.utils.session_timer import SessionTimer

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300
"""Silence after which a pending exchange is abandoned."""

DEFAULT_PENDING_VALUES = ["started", "start", "in-progress", "in_progress", "pending", "request"]
"""Result values that mean the exchange is still running rather than finished."""

DEFAULT_SUPPLICANT_COLUMNS = ["mac_address", "dot1x_identity"]
"""Columns naming the device being authorized, in preference order. The MAC comes first because it is present on
both halves of an exchange, while an identity is often unknown until the supplicant has answered."""

PORT_KEY_SEPARATOR = KEY_SEPARATOR

ELAPSED_COLUMN = "auth_elapsed_seconds"
ATTEMPTS_COLUMN = "auth_attempts"
UNPAIRED_COLUMN = "auth_unpaired"
PORT_KEY_COLUMN = "auth_port_key"


@register_stage("tc2-auth", ignore_args=["pending_values"])
class TC2AuthStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Time 802.1X authorization per port, and flag authorization that nobody authenticated for.

    The telemetry class asks for the time-to-authorize distribution, and it is a distribution rather than a
    threshold because both tails carry meaning. A slow authorization is a supplicant retrying, a RADIUS server
    under load, or credentials being guessed. A very fast one can be a replayed or cached success.

    The most useful case is neither tail. An outcome that arrives with no exchange in front of it is what a bypass
    looks like from the switch: MAC authentication bypass and a device bridged behind an already authorized
    supplicant both produce authorization without anybody authenticating. `auth_unpaired` marks those rows rather
    than leaving them as a null elapsed time, which would read as missing data instead of as an event.

    `auth_attempts` travels with the timing, because a success after three retries is not the same as a first-time
    one and the elapsed time from the last attempt alone would hide the two before it.

    Rows are classified by `result_column`: a value listed in `pending_values`, or a null, starts the clock, and
    anything else stops it. That matches how the common sources report, with a RADIUS Access-Request or an EAPOL
    start preceding an Accept or Reject.

    An exchange is timed per supplicant on a port, not per port. A port is routinely shared: Cisco multi-domain
    seats a phone and a workstation on one interface, multi-auth seats more, and a switch reload restarts every
    session on every port at once. Keyed on the port alone, one device's outcome closes another device's exchange,
    and the rule inverts in both directions at once -- the second legitimate supplicant is reported as authorized
    without authentication, while a genuine bypass arriving mid-exchange takes the pending slot of the device it is
    bridged behind and reads as an ordinary timed session. That second direction is the one that matters: it is
    exactly the signal R-D-L2-005 exists to catch, and it disappears silently.

    Where no supplicant column is present the key falls back to the port, which is the old behavior and the best a
    source that does not report the device can support. The stage says so once, because on a shared port that
    fallback mispairs.

    The stage is stateful across messages and must run single-engine. Shard by switch upstream, which keeps every
    supplicant on a port together on one instance; sharding by identity would not, since the same supplicant can
    appear on several ports.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    site_column : str, default = "site_id"
        Column holding the site identifier, used to build the port key.
    switch_column : str, default = "switch_id"
        Column holding the switch identifier.
    port_column : str, default = "port_id"
        Column holding the port identifier. Exchanges are timed per port, since that is the physical thing being
        authorized onto the network.
    result_column : str, default = "dot1x_result"
        Column holding the outcome. Nulls and `pending_values` start the clock; anything else stops it.
    supplicant_columns : list of str, optional
        Columns identifying the device being authorized, in preference order; the first one present in the frame
        with a value for that row is used. An exchange is paired per supplicant per port, so that two devices
        authenticating on one interface do not close each other's exchanges. Defaults to
        `DEFAULT_SUPPLICANT_COLUMNS`. Pass an empty list to time per port alone.
    time_column : str, default = "event_time"
        Column holding the event time. Event time, never ingest time: an elapsed time measured against ingest
        order describes the collector's queue rather than the exchange.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    pending_values : list of str, optional
        Result values meaning the exchange is still running. Compared case-insensitively.
    max_clock_skew_seconds : int, default = 604800
        How far ahead of the stream's own progress a row's event time may be before it is refused. Expiry runs on
        event time, so a device whose clock is wrong by years would otherwise abandon every pending exchange at
        once. A refused row carries no timing and is counted. See `morpheus.utils.event_clock`.
    timeout_seconds : int, default = 300
        Silence after which a pending exchange is abandoned, so a port whose result never arrived does not hold
        state forever and its next outcome is correctly reported as unpaired.
    """

    def __init__(self,
                 c: Config,
                 site_column: str = "site_id",
                 switch_column: str = "switch_id",
                 port_column: str = "port_id",
                 result_column: str = "dot1x_result",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 pending_values: list[str] = None,
                 supplicant_columns: list[str] = None,
                 max_clock_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS,
                 timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS):
        super().__init__(c)

        if (timeout_seconds <= 0):
            raise ValueError(f"timeout_seconds must be positive, received {timeout_seconds}")

        if (max_clock_skew_seconds <= 0):
            raise ValueError(f"max_clock_skew_seconds must be positive, received {max_clock_skew_seconds}")

        pending = DEFAULT_PENDING_VALUES if pending_values is None else pending_values

        self._site_column = site_column
        self._switch_column = switch_column
        self._port_column = port_column
        self._result_column = result_column
        self._supplicant_columns = (DEFAULT_SUPPLICANT_COLUMNS
                                    if supplicant_columns is None else list(supplicant_columns))
        self._warned_portwise = False
        self._time_column = time_column
        self._time_unit = time_unit
        self._pending_values = {str(value).strip().lower() for value in pending}

        self._clock = EventClock(max_skew_ns=max_clock_skew_seconds * NS_PER_SECOND)
        self._timer = SessionTimer(timeout_ns=timeout_seconds * NS_PER_SECOND)

        self._needed_columns[PORT_KEY_COLUMN] = TypeId.STRING
        self._needed_columns[ELAPSED_COLUMN] = TypeId.FLOAT64
        self._needed_columns[ATTEMPTS_COLUMN] = TypeId.INT64
        self._needed_columns[UNPAIRED_COLUMN] = TypeId.BOOL8

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc2-auth"

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

    @property
    def pending_count(self) -> int:
        """Exchanges currently open and awaiting an outcome."""
        return self._timer.pending_count

    @staticmethod
    def _text(value: typing.Any) -> typing.Optional[str]:
        """Normalize a host value, collapsing every flavor of missing to `None`."""
        if (value is None):
            return None

        if (isinstance(value, float) and math.isnan(value)):
            return None

        return str(value).strip()

    def _is_start(self, result: typing.Optional[str]) -> bool:
        """Whether this row opens an exchange rather than closing one."""
        return result is None or result.lower() in self._pending_values

    def _exchange_key(self, port_key: str, supplicant_sources: list[list], position: int) -> str:
        """The key an exchange is paired on: the port, extended with the device being authorized on it."""
        for source in supplicant_sources:
            supplicant = normalize_text(source[position])

            if (supplicant is not None):
                return compose_key([port_key, supplicant])

        if (not self._warned_portwise):
            self._warned_portwise = True
            logger.warning(
                "TC2AuthStage found no supplicant in columns %s, so exchanges are timed per port. On a port "
                "carrying more than one supplicant -- a phone and a workstation, or anything behind a hub -- one "
                "device's outcome will close another device's exchange.",
                self._supplicant_columns)

        return port_key

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the elapsed time, the attempt count, and the unpaired flag.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming authentication events.

        Returns
        -------
        The input message, with the timing columns populated.

        Raises
        ------
        KeyError
            If the port, result, or time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._switch_column, self._port_column, self._result_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC2AuthStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            switches = to_host_list(df, self._switch_column)
            ports = to_host_list(df, self._port_column)
            results = to_host_list(df, self._result_column)
            raw_times = to_host_list(df, self._time_column)

            sites = to_host_list(df, self._site_column) if self._site_column in df.columns else [None] * len(ports)

            # In preference order, so a row missing its identity still pairs on its MAC.
            supplicant_sources = [to_host_list(df, name) for name in self._supplicant_columns if name in df.columns]

            port_keys: list = []
            elapsed: list = []
            attempts: list = []
            unpaired: list = []
            unordered = 0
            keyless = 0
            abandoned = 0
            implausible = 0
            has_site = self._site_column in df.columns

            for (position, raw_port) in enumerate(ports):
                # The same composition the layer 1 stages use for `entity_key`. A null part yields a null key.
                location = (sites[position], switches[position], raw_port) if has_site else (switches[position],
                                                                                             raw_port)
                port_key = compose_key(location)
                port_keys.append(port_key)

                if (port_key is None):
                    # No port to time an exchange for. Pairing this against a fabricated key would let one port's
                    # start close another port's outcome.
                    elapsed.append(None)
                    attempts.append(None)
                    unpaired.append(None)
                    keyless += 1
                    continue

                result = self._text(results[position])

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                if (event_time_ns is None):
                    elapsed.append(None)
                    attempts.append(None)
                    unpaired.append(None)
                    unordered += 1
                    continue

                # An exchange that never resolved has to stop being pending, or a later outcome pairs with it and
                # is timed against an exchange it has nothing to do with. That is not merely a wrong duration: a
                # bypass, which is an outcome with no exchange in front of it, reads as an ordinary authorized
                # session and the signal R-D-L2-005 exists to catch disappears. Expiry runs on this row's own
                # event time rather than a wall clock or a batch boundary, so it falls in the same place however
                # the stream is divided.
                # A time the stream cannot believe must not reach the expiry below. One row from a device whose
                # clock is wrong by years would otherwise abandon every pending exchange in the estate at once,
                # and each of those exchanges' real outcomes would then read as an unpaired authorization.
                if (not self._clock.accept(event_time_ns)):
                    elapsed.append(None)
                    attempts.append(None)
                    unpaired.append(None)
                    implausible += 1
                    continue

                abandoned += len(self._timer.expire(event_time_ns))

                # The port is what is being authorized onto the network, but the exchange belongs to a device on
                # it. Extending the key with the supplicant is what keeps two devices on one interface from
                # closing each other's exchanges; without a supplicant this is the port key unchanged.
                exchange_key = self._exchange_key(port_key, supplicant_sources, position)

                if (self._is_start(result)):
                    self._timer.begin(exchange_key, event_time_ns)
                    # A start is not itself an event to score; it establishes what the outcome will be measured
                    # against. Nulls rather than zeros, so a rule reading "authorized instantly" cannot match here.
                    elapsed.append(None)
                    attempts.append(None)
                    unpaired.append(None)
                    continue

                timing = self._timer.complete(exchange_key, event_time_ns, outcome=result)

                elapsed.append(None if timing.elapsed_ns is None else timing.elapsed_ns / NS_PER_SECOND)
                attempts.append(timing.attempts)
                unpaired.append(timing.unpaired)
                unordered += int(timing.out_of_order)

            assign_str_column(df, PORT_KEY_COLUMN, port_keys)
            assign_nullable_float_column(df, ELAPSED_COLUMN, elapsed)
            assign_nullable_int_column(df, ATTEMPTS_COLUMN, attempts)
            assign_nullable_bool_column(df, UNPAIRED_COLUMN, unpaired)

        if (keyless > 0):
            logger.warning("TC2AuthStage saw %d of %d events with a null site, switch, or port; they carry no timing.",
                           keyless,
                           len(ports))

        if (implausible > 0):
            logger.warning(
                "TC2AuthStage refused %d of %d events whose event time was further ahead of the stream than "
                "max_clock_skew_seconds allows; they carry no timing and did not abandon any exchange. A device "
                "whose clock is wrong by years is the usual cause.",
                implausible,
                len(ports))

        if (unordered > 0):
            logger.warning(
                "TC2AuthStage saw %d of %d events out of order or without a usable event time; they carry no "
                "timing. Shard by switch and preserve per-port ordering upstream.",
                unordered,
                len(ports))

        if (abandoned > 0):
            logger.info(
                "TC2AuthStage abandoned %d exchanges that went unresolved for longer than the timeout. An exchange "
                "that never completes is its own signal; a later outcome on the same port is now correctly read as "
                "unpaired rather than timed against it.",
                abandoned)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
