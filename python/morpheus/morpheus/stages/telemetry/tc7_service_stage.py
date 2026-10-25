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
A service as the subject of behaviour: who reaches it, how much, and with which client software.

Every other class in this fork is keyed on something that initiates -- a port, a host, a principal, a client. A
service is reached rather than reaching, and the questions worth asking about it are the ones nobody's own history
can answer: an internal API that three application servers call every hour, called this hour by a dozen workstations
running a client it has never been called by, is a change in the service's life that each of those workstations
sees as one ordinary request.

**The service is the name where there is one, and the address where there is not.** A connection's `sni` names the
service the client asked for; one address terminating two names is two services, and one name served from a pool of
addresses is one. The key is therefore `sni:dst_port` where the record carries a name and `dst_ip:dst_port` where it
does not, and `service_key_source` says which, so a search can tell a name-keyed service from an address standing in
for one.

**Per window, per service:** connections, bytes and distinct clients over the trailing window, and whether the
connection's client fingerprint has ever reached this service before, through
{py:mod}`~morpheus.utils.value_novelty`. A fingerprint is the client software, not the client: a new workstation
running the estate's browser reaches the service with a fingerprint it already knows.

**The distinct client count is measured against the service's own days.** The busiest window of each day is kept
through {py:mod}`~morpheus.utils.bucket_peak`, and `service_clients_ratio` is this window's count as a multiple of the
highest of a month of them. A service with less than a week of days has no baseline, and a rule reading the ratio
stays quiet on it: a service deployed this morning has no life to change yet.

The stage is stateful across messages and must run single-engine, or sharded by service.
"""

import collections
import dataclasses
import logging
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
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.bucket_peak import BucketPeakTracker
from morpheus.utils.bucket_peak import measure_rows
from morpheus.utils.bucket_peak import ratios_to_reference
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.distinct_window import DistinctWindowTracker
from morpheus.utils.entity_key import compose_key
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.event_clock import CLOCK_REFUSED_COLUMN
from morpheus.utils.event_clock import DEFAULT_MAX_SKEW_SECONDS
from morpheus.utils.event_clock import clock_for_skew_seconds
from morpheus.utils.event_clock import event_times_ns
from morpheus.utils.event_clock import guard_times
from morpheus.utils.event_clock import refuse_rows
from morpheus.utils.value_novelty import ValueNoveltyTracker

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 3600
"""Trailing window the per-service counts are taken over. An hour, the period the layer 7 classes seal on."""

DEFAULT_BASELINE_BUCKET_SECONDS = 86400
"""The period a service's busiest window is kept per. A day."""

DEFAULT_BASELINE_WINDOW_SECONDS = 30 * 86400
"""How far back a service's days reach."""

DEFAULT_BASELINE_MIN_DAYS = 7
"""Days of history a service needs before it has a baseline."""

SERVICE_KEY = "service_key"
SERVICE_KEY_SOURCE = "service_key_source"
CONNECTIONS = "service_connections_in_window"
BYTES = "service_bytes_in_window"
CLIENTS = "service_clients_in_window"
CLIENTS_SATURATED = "service_clients_saturated"
FINGERPRINT_FIRST_SEEN = "service_client_fingerprint_first_seen"
FINGERPRINTS = "service_client_fingerprints"
CLIENTS_BASELINE = "service_clients_baseline"
CLIENTS_BASELINE_DAYS = "service_clients_baseline_days"
CLIENTS_BASELINE_MATURE = "service_clients_baseline_mature"
CLIENTS_RATIO = "service_clients_ratio"

FINGERPRINT_FIELD = "client_fingerprint"
"""The field the per-service novelty tracker watches."""

KEYED_BY_NAME = "sni"
KEYED_BY_ADDRESS = "address"


@dataclasses.dataclass
class _Window:
    entries: collections.deque = dataclasses.field(default_factory=collections.deque)
    total: float = 0.0
    last_ns: typing.Optional[int] = None


class _WindowSums:
    """Per-service connection counts and byte sums over a trailing window, bounded per service and in services."""

    def __init__(self, window_ns: int, max_samples: int, max_entities: int):
        self._window_ns = window_ns
        self._max_samples = max_samples
        self._max_entities = max_entities
        self._windows: collections.OrderedDict[str, _Window] = collections.OrderedDict()

    def observe(self, key: str, event_time_ns: int, value: typing.Optional[float]) -> typing.Optional[tuple]:
        """`(connections, bytes)` in the window ending at this connection, or `None` for an out-of-order one."""
        window = self._windows.get(key)

        if (window is None):
            window = _Window()
            self._windows[key] = window

            while (len(self._windows) > self._max_entities):
                self._windows.popitem(last=False)
        else:
            self._windows.move_to_end(key)

        if (window.last_ns is not None and event_time_ns < window.last_ns):
            return None

        horizon = event_time_ns - self._window_ns

        while (len(window.entries) > 0
               and (window.entries[0][0] <= horizon or len(window.entries) >= self._max_samples)):
            window.total -= window.entries.popleft()[1]

        amount = 0.0 if value is None else float(value)
        window.entries.append((event_time_ns, amount))
        window.total += amount
        window.last_ns = event_time_ns

        return (len(window.entries), window.total)


def _number(value: typing.Any) -> typing.Optional[float]:
    try:
        if (value is None or value != value):  # pylint: disable=comparison-with-itself
            return None

        return float(value)
    except (TypeError, ValueError):
        return None


@register_stage("tc7-service")
class TC7ServiceStage(CheckpointMixin, GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Key each connection on the service it reached, and write that service's window counts and client novelty.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    client_column : str, default = "src_ip"
        Column holding the client.
    address_column : str, default = "dst_ip"
        Column holding the service's address.
    port_column : str, default = "dst_port"
        Column holding the service's port.
    name_column : str, default = "sni"
        Column holding the name the client asked for. The column may be absent, in which case every service is keyed
        on its address.
    bytes_column : str, default = "bytes_total"
        Column holding the connection's bytes in both directions. The column may be absent.
    fingerprint_column : str, default = "ja4_client"
        Column holding the client software's fingerprint. The column may be absent, in which case the novelty
        columns are null.
    time_column : str, default = "event_time"
        Column holding the connection's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    window_seconds : int, default = 3600
        Trailing window the counts are taken over.
    max_samples : int, default = 4096
        Connections retained per service regardless of the window.
    max_entities : int, default = 100000
        Services tracked before the least recently seen is forgotten.
    baseline_bucket_seconds : int, default = 86400
        The period a service's busiest window of distinct clients is kept per.
    baseline_window_seconds : int, default = 2592000
        How far back those periods reach.
    baseline_min_days : int, default = 7
        Periods of history a service needs before `service_clients_baseline` is published.
    max_clock_skew_seconds : int, default = 604800
        How far ahead of the stream's own progress a row's event time may be before it is refused. Every window and
        every day here expires on event time, so one connection from a sensor whose clock is wrong by years would
        empty each service's connections, clients and days at once and refuse every real connection after it as out
        of order. A refused row carries no measurement from this stage, sets `clock_refused`, and moves no horizon;
        its service key is still composed, since that needs no history. See `morpheus.utils.event_clock`.
    """

    _checkpoint_attributes = ("_clients", "_fingerprints", "_baseline", "_sums", "_clock")

    def __init__(self,
                 c: Config,
                 client_column: str = "src_ip",
                 address_column: str = "dst_ip",
                 port_column: str = "dst_port",
                 name_column: str = "sni",
                 bytes_column: str = "bytes_total",
                 fingerprint_column: str = "ja4_client",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 max_samples: int = 4096,
                 max_entities: int = 100_000,
                 baseline_bucket_seconds: int = DEFAULT_BASELINE_BUCKET_SECONDS,
                 baseline_window_seconds: int = DEFAULT_BASELINE_WINDOW_SECONDS,
                 baseline_min_days: int = DEFAULT_BASELINE_MIN_DAYS,
                 max_clock_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS):
        super().__init__(c)

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        if (max_samples <= 0):
            raise ValueError(f"max_samples must be positive, received {max_samples}")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._client_column = client_column
        self._address_column = address_column
        self._port_column = port_column
        self._name_column = name_column
        self._bytes_column = bytes_column
        self._fingerprint_column = fingerprint_column
        self._time_column = time_column
        self._time_unit = time_unit

        window_ns = window_seconds * NS_PER_SECOND

        self._sums = _WindowSums(window_ns, max_samples, max_entities)
        self._clients = DistinctWindowTracker(window_ns=window_ns, max_samples=max_samples, max_entities=max_entities)
        # A service is reached by many clients in one second, so connections at one instant are measured, not
        # refused.
        self._fingerprints = ValueNoveltyTracker([FINGERPRINT_FIELD],
                                                 max_values=1024,
                                                 max_entities=max_entities,
                                                 simultaneous=True)
        self._baseline = BucketPeakTracker(bucket_ns=baseline_bucket_seconds * NS_PER_SECOND,
                                           window_ns=baseline_window_seconds * NS_PER_SECOND,
                                           min_buckets=baseline_min_days,
                                           max_entities=max_entities)

        self._needed_columns[SERVICE_KEY] = TypeId.STRING
        self._needed_columns[SERVICE_KEY_SOURCE] = TypeId.STRING
        self._needed_columns[CONNECTIONS] = TypeId.INT64
        self._needed_columns[BYTES] = TypeId.FLOAT64
        self._needed_columns[CLIENTS] = TypeId.INT64
        self._needed_columns[CLIENTS_SATURATED] = TypeId.BOOL8
        self._needed_columns[FINGERPRINT_FIRST_SEEN] = TypeId.BOOL8
        self._needed_columns[FINGERPRINTS] = TypeId.INT64
        self._needed_columns[CLIENTS_BASELINE] = TypeId.INT64
        self._needed_columns[CLIENTS_BASELINE_DAYS] = TypeId.INT64
        self._needed_columns[CLIENTS_BASELINE_MATURE] = TypeId.BOOL8
        self._needed_columns[CLIENTS_RATIO] = TypeId.FLOAT64

        self._clock = clock_for_skew_seconds(max_clock_skew_seconds)
        self._needed_columns[CLOCK_REFUSED_COLUMN] = TypeId.BOOL8

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc7-service"

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

    @staticmethod
    def service_key(address: typing.Any, port: typing.Any, name: typing.Any) -> tuple:
        """
        The service a connection reached, and whether its name or its address identifies it.

        Parameters
        ----------
        address, port, name
            The connection's destination address, port and requested name.

        Returns
        -------
        tuple
            `(key, source)`, or `(None, None)` without a port, or without both a name and an address.
        """
        named = normalize_text(name)

        if (named is not None):
            key = compose_key([named.lower(), port])

            return (key, KEYED_BY_NAME) if key is not None else (None, None)

        key = compose_key([address, port])

        return (key, KEYED_BY_ADDRESS) if key is not None else (None, None)

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the service key and the service's window counts, client novelty and baseline.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming connection records.

        Returns
        -------
        The input message, with the service columns populated.

        Raises
        ------
        KeyError
            If the client, address, port or time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._client_column, self._address_column, self._port_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC7ServiceStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            rows = len(df)
            clients = to_host_list(df, self._client_column)
            addresses = to_host_list(df, self._address_column)
            ports = to_host_list(df, self._port_column)
            names = to_host_list(df, self._name_column) if self._name_column in df.columns else [None] * rows
            sizes = to_host_list(df, self._bytes_column) if self._bytes_column in df.columns else [None] * rows
            fingerprints = (to_host_list(df, self._fingerprint_column)
                            if self._fingerprint_column in df.columns else [None] * rows)
            # Guarded once, before any window or day is shown a time: a refused time becomes `None`, which the rows
            # below already treat as a connection with nothing to count.
            (times, refused) = guard_times(self._clock, event_times_ns(df, self._time_column, self._time_unit))

            keys: list = []
            sources: list = []
            connections: list = []
            volumes: list = []
            reached_by: list = []
            saturated: list = []
            first_seen: list = []
            known: list = []
            unusable = 0
            unordered = 0

            for position in range(rows):
                (key, source) = self.service_key(addresses[position], ports[position], names[position])
                event_time_ns = times[position]

                keys.append(key)
                sources.append(source)

                client = normalize_text(clients[position])

                if (key is None or event_time_ns is None):
                    unusable += int(not refused[position])

                    for column in (connections, volumes, reached_by, first_seen, known):
                        column.append(None)

                    saturated.append(False)
                    continue

                sums = self._sums.observe(key, event_time_ns, _number(sizes[position]))
                unordered += int(sums is None)
                connections.append(None if sums is None else sums[0])
                volumes.append(None if sums is None else sums[1])

                if (client is None):
                    reached_by.append(None)
                    saturated.append(False)
                else:
                    result = self._clients.observe(key, event_time_ns, client)
                    reached_by.append(result.distinct)
                    saturated.append(result.saturated)

                fingerprint = normalize_text(fingerprints[position])

                if (fingerprint is None):
                    first_seen.append(None)
                    known.append(None)
                else:
                    novelty = self._fingerprints.observe(key, event_time_ns, {FINGERPRINT_FIELD: fingerprint})
                    first_seen.append(novelty.first_seen[FINGERPRINT_FIELD])
                    known.append(None if novelty.out_of_order else novelty.distinct_counts[FINGERPRINT_FIELD])

            measured = measure_rows(self._baseline, keys, reached_by, times)

            assign_str_column(df, SERVICE_KEY, keys)
            assign_str_column(df, SERVICE_KEY_SOURCE, sources)
            assign_nullable_int_column(df, CONNECTIONS, connections)
            assign_nullable_float_column(df, BYTES, volumes)
            assign_nullable_int_column(df, CLIENTS, reached_by)
            assign_nullable_bool_column(df, CLIENTS_SATURATED, saturated)
            assign_nullable_bool_column(df, FINGERPRINT_FIRST_SEEN, first_seen)
            assign_nullable_int_column(df, FINGERPRINTS, known)
            assign_nullable_int_column(df, CLIENTS_BASELINE, measured.references)
            assign_nullable_int_column(df, CLIENTS_BASELINE_DAYS, measured.buckets)
            assign_nullable_bool_column(df, CLIENTS_BASELINE_MATURE, measured.mature)
            assign_nullable_float_column(df, CLIENTS_RATIO, ratios_to_reference(measured, reached_by))

            refuse_rows(df,
                        refused,
                        logger=logger,
                        stage="TC7ServiceStage",
                        what="connections",
                        outcome="they entered no window, set clock_refused and expired nothing.")

            if (unusable > 0):
                logger.warning(
                    "TC7ServiceStage left %d of %d connections out of every service for want of a port, a name or "
                    "address, or a usable event time.",
                    unusable,
                    rows)

            if (unordered > 0):
                logger.warning(
                    "TC7ServiceStage saw %d connections arrive earlier than their service's previous one; they were "
                    "not counted.",
                    unordered)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
