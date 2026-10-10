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
Turns layer 1 counter deltas into rates, and measures each port's rates against its own history.

`morpheus.stages.telemetry.tc1_normalize_stage.TC1NormalizeStage` turns a counter into the change since the last
poll and the interval that change actually covers. Neither is a feature on its own: forty CRC errors is a quiet hour
on one link and a failing cable on another, and a delta over ninety seconds is not comparable with one over sixty.
The quantity a rule can threshold is the rate, and the quantity a rule should threshold is the rate against what the
same port has done before. This stage writes both, for the two things a port does that an operator can see.

**Errors.** `error_rate` is CRC and symbol errors per second. A cable going bad, a dirty connector or a duplex
mismatch climbs; a port that has always run a few errors an hour does not. The rate is measured against the highest
five-minute peak the port has reached in its own history, and `error_rate_step` is how far above it this poll sits,
positive only when the port has never been this bad. `discard_rate` travels beside it as context, without a history:
discards are congestion as often as fault, and a rule on them alone would page on every busy afternoon.

**Traffic.** `bits_in_per_second`, `bits_out_per_second` and their sum, from the 64-bit octet counters, and
`utilization`, the busier direction against `link_speed_bps`. The history here keeps two figures rather than one,
because a port can depart from itself in two directions: `bits_per_second_baseline_max`, the most it has carried in
any five-minute period, and `bits_per_second_baseline_min`, the least. A port carrying twice its record is moving
something it never has; a port that has never been quiet and now carries nothing has lost whatever was on it, which
is how a disconnected server or a removed device looks from the switch. A port whose quietest period was zero can
never read as silent, which is the right answer for a desk port that empties every night.

**Traffic has a time of day, so its history can be kept per hour.** With `volume_seasonality="hour_of_day"` the
volume history is kept per port per UTC hour, so nine o'clock is measured against nine o'clocks and the overnight
backup against overnight. Errors are not seasonal and are measured against the port's whole history.

**A rate is null where the delta is not a measurement of the interval.** A device reset restarts the counters and
caps the interval at the uptime; the delta is real, but it measures a period the history has no comparison for, and
a reboot would otherwise read as a burst on every port of the switch. An out-of-order sample carries no delta. Null
rows enter no history.

The stage is stateful across messages and must run single-engine, or sharded by device, which is determinism
control 4. Place it after the normalize stage, which supplies the deltas, the interval and the flags.
"""

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
from morpheus.utils.bucket_peak import NS_PER_SECOND
from morpheus.utils.bucket_peak import BucketPeakTracker
from morpheus.utils.bucket_peak import measure_rows
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import KEY_SEPARATOR
from morpheus.utils.entity_key import normalize_text

logger = logging.getLogger(__name__)

DEFAULT_ERROR_COLUMNS = ["crc_errors", "symbol_errors"]
"""The counters whose sum is the error rate: physical-layer errors, which a cable, an optic or a duplex fault drives."""

DEFAULT_DISCARD_COLUMNS = ["input_discards", "output_discards"]
"""The counters whose sum is the discard rate, carried as context."""

SEASONALITIES = ("none", "hour_of_day")
"""How the volume history is divided: one history per port, or one per port per UTC hour."""

DEFAULT_BUCKET_SECONDS = 300
"""The period a peak is kept per. Five minutes: long enough that one poll's jitter is not a period of its own, short
enough that a burst is not averaged away."""

DEFAULT_WINDOW_SECONDS = 14 * 24 * 3600
"""How far back the periods reach. A fortnight, so every weekday has been seen twice."""

DEFAULT_MIN_BUCKETS = 36
"""Committed periods required before a history is published: three hours of a port's whole history, or three days
of one hour's twelve periods when the volume history is kept per hour."""

DEFAULT_MAX_BUCKETS = 4096
"""Periods retained per history. A fortnight of five-minute periods is 4032."""

ERROR_RATE = "error_rate"
DISCARD_RATE = "discard_rate"
BITS_IN = "bits_in_per_second"
BITS_OUT = "bits_out_per_second"
BITS = "bits_per_second"
UTILIZATION = "utilization"

BASELINE_MAX_SUFFIX = "_baseline_max"
BASELINE_MIN_SUFFIX = "_baseline_min"
BUCKETS_SUFFIX = "_baseline_buckets"
MATURE_SUFFIX = "_baseline_mature"
STEP_SUFFIX = "_step"

BITS_PER_OCTET = 8


def _number(value: typing.Any) -> typing.Optional[float]:
    """A host value as a float, or `None` for every flavour of missing."""
    if (value is None or isinstance(value, bool)):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return None if math.isnan(number) else number


def _flag(value: typing.Any) -> bool:
    """A host value as a flag, with every flavour of missing read as not set."""
    if (value is None or (isinstance(value, float) and math.isnan(value))):
        return False

    if (isinstance(value, str)):
        return value.strip().lower() == "true"

    try:
        return bool(value)
    except TypeError:
        # `pandas.NA` refuses truthiness, which is the honest answer and here means the flag was not reported.
        return False


@register_stage("tc1-rate", ignore_args=["error_columns", "discard_columns"])
class TC1RateStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write each port's error, discard and traffic rates, and measure the error rate and the traffic against the port's
    own history.

    See the module docstring for what each column means and why the rates are null where they are.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    entity_key_column : str, default = "entity_key"
        Column holding the port the histories are kept per.
    time_column : str, default = "event_time"
        Column holding the poll's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    delta_suffix : str, default = "_delta"
        Suffix the normalize stage gave the delta columns.
    error_columns : list of str, optional
        Counters summed into `error_rate`. Defaults to CRC and symbol errors.
    discard_columns : list of str, optional
        Counters summed into `discard_rate`. Defaults to input and output discards.
    in_octets_column : str, default = "if_hc_in_octets"
        The inbound 64-bit octet counter, by its raw name; its delta is read.
    out_octets_column : str, default = "if_hc_out_octets"
        The outbound 64-bit octet counter, by its raw name.
    link_speed_column : str, default = "link_speed_bps"
        Column holding the interface's speed in bits per second, which `utilization` is taken against. Absent or
        zero, utilization is null.
    volume_seasonality : str, default = "hour_of_day"
        `hour_of_day` keeps the volume history per port per UTC hour; `none` keeps one per port.
    bucket_seconds : int, default = 300
        The period a peak, and for traffic a trough, is kept per.
    window_seconds : int, default = 1209600
        How far back the periods reach. A fortnight.
    min_buckets : int, default = 36
        Committed periods a history needs before it is published.
    max_buckets : int, default = 4096
        Periods retained per history.
    """

    def __init__(self,
                 c: Config,
                 entity_key_column: str = "entity_key",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 delta_suffix: str = "_delta",
                 error_columns: list[str] = None,
                 discard_columns: list[str] = None,
                 in_octets_column: str = "if_hc_in_octets",
                 out_octets_column: str = "if_hc_out_octets",
                 link_speed_column: str = "link_speed_bps",
                 volume_seasonality: str = "hour_of_day",
                 bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 min_buckets: int = DEFAULT_MIN_BUCKETS,
                 max_buckets: int = DEFAULT_MAX_BUCKETS):
        super().__init__(c)

        error_columns = list(DEFAULT_ERROR_COLUMNS) if error_columns is None else list(error_columns)
        discard_columns = list(DEFAULT_DISCARD_COLUMNS) if discard_columns is None else list(discard_columns)

        if (len(error_columns) == 0):
            raise ValueError("error_columns must name at least one counter; an error rate over nothing is zero")

        if (volume_seasonality not in SEASONALITIES):
            raise ValueError(f"volume_seasonality must be one of {SEASONALITIES}, received {volume_seasonality!r}")

        if (bucket_seconds <= 0):
            raise ValueError(f"bucket_seconds must be positive, received {bucket_seconds}")

        self._entity_key_column = entity_key_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._error_deltas = [f"{name}{delta_suffix}" for name in error_columns]
        self._discard_deltas = [f"{name}{delta_suffix}" for name in discard_columns]
        self._octet_deltas = (f"{in_octets_column}{delta_suffix}", f"{out_octets_column}{delta_suffix}")
        self._link_speed_column = link_speed_column
        self._seasonal = volume_seasonality == "hour_of_day"

        def tracker() -> BucketPeakTracker:
            return BucketPeakTracker(bucket_ns=bucket_seconds * NS_PER_SECOND,
                                     window_ns=window_seconds * NS_PER_SECOND,
                                     min_buckets=min_buckets,
                                     max_buckets=max_buckets)

        self._errors = tracker()
        # The trough is kept as the peak of the negated value, which is the same history read the other way up.
        self._volume_peaks = tracker()
        self._volume_troughs = tracker()

        for name in (ERROR_RATE, DISCARD_RATE, BITS_IN, BITS_OUT, BITS, UTILIZATION):
            self._needed_columns[name] = TypeId.FLOAT64

        self._needed_columns[f"{ERROR_RATE}{BASELINE_MAX_SUFFIX}"] = TypeId.FLOAT64
        self._needed_columns[f"{ERROR_RATE}{BUCKETS_SUFFIX}"] = TypeId.INT64
        self._needed_columns[f"{ERROR_RATE}{MATURE_SUFFIX}"] = TypeId.BOOL8
        self._needed_columns[f"{ERROR_RATE}{STEP_SUFFIX}"] = TypeId.FLOAT64

        self._needed_columns[f"{BITS}{BASELINE_MAX_SUFFIX}"] = TypeId.FLOAT64
        self._needed_columns[f"{BITS}{BASELINE_MIN_SUFFIX}"] = TypeId.FLOAT64
        self._needed_columns[f"{BITS}{BUCKETS_SUFFIX}"] = TypeId.INT64
        self._needed_columns[f"{BITS}{MATURE_SUFFIX}"] = TypeId.BOOL8
        self._needed_columns[f"{BITS}{STEP_SUFFIX}"] = TypeId.FLOAT64

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc1-rate"

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

    def _event_time_ns(self, raw: typing.Any) -> typing.Optional[int]:
        try:
            return to_epoch_ns(raw, time_unit=self._time_unit)
        except ValueError:
            return None

    @staticmethod
    def _rate(deltas: list, interval: typing.Optional[float]) -> typing.Optional[float]:
        """The sum of the deltas per second, or `None` if any is missing: a partial sum reads as a measured drop."""
        if (interval is None or interval <= 0 or any(delta is None for delta in deltas)):
            return None

        return sum(deltas) / interval

    def _volume_key(self, key: typing.Optional[str], event_time_ns: typing.Optional[int]) -> typing.Optional[str]:
        if (key is None or event_time_ns is None or not self._seasonal):
            return key

        hour = (event_time_ns // (3600 * NS_PER_SECOND)) % 24

        return f"{key}{KEY_SEPARATOR}h{hour:02d}"

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the rates and the histories they are measured against.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming message.

        Returns
        -------
        The input message, with the rate columns populated.

        Raises
        ------
        KeyError
            If the entity key, the time, the interval or an error delta column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._entity_key_column, self._time_column, "interval_seconds"] + self._error_deltas
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC1RateStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            row_count = len(df)

            def column(name: str) -> list:
                return to_host_list(df, name) if name in df.columns else [None] * row_count

            keys = [normalize_text(value) for value in column(self._entity_key_column)]
            times = [self._event_time_ns(raw) for raw in column(self._time_column)]
            intervals = [_number(value) for value in column("interval_seconds")]
            resets = [_flag(value) for value in column("counter_reset")]
            unordered = [_flag(value) for value in column("sample_out_of_order")]
            speeds = [_number(value) for value in column(self._link_speed_column)]

            def deltas_of(names: list) -> list:
                values = [[_number(value) for value in column(name)] for name in names]

                return [list(row) for row in zip(*values)] if len(values) > 0 else [[] for _ in range(row_count)]

            error_deltas = deltas_of(self._error_deltas)
            discard_deltas = deltas_of(self._discard_deltas)
            octet_deltas = deltas_of(list(self._octet_deltas))

            error_rates = []
            discard_rates = []
            bits_in = []
            bits_out = []
            bits = []
            utilization = []

            for position in range(row_count):
                # A reset's delta covers the uptime rather than the gap, and an out-of-order sample carries none; in
                # either case the interval is not one the history can be compared with.
                usable = (keys[position] is not None and times[position] is not None and not resets[position]
                          and not unordered[position])
                interval = intervals[position] if usable else None

                error_rates.append(self._rate(error_deltas[position], interval))
                discard_rates.append(
                    self._rate(discard_deltas[position], interval) if len(discard_deltas[position]) > 0 else None)

                (inbound, outbound) = octet_deltas[position]
                inbound_bps = self._rate([inbound], interval)
                outbound_bps = self._rate([outbound], interval)
                inbound_bps = None if inbound_bps is None else inbound_bps * BITS_PER_OCTET
                outbound_bps = None if outbound_bps is None else outbound_bps * BITS_PER_OCTET

                bits_in.append(inbound_bps)
                bits_out.append(outbound_bps)

                if (inbound_bps is None or outbound_bps is None):
                    bits.append(None)
                    utilization.append(None)
                    continue

                bits.append(inbound_bps + outbound_bps)
                speed = speeds[position]
                # Full duplex carries the speed in each direction, so the busier one is the one that fills the link.
                utilization.append(max(inbound_bps, outbound_bps) / speed if speed is not None and speed > 0 else None)

            errors = measure_rows(self._errors, keys, error_rates, times)

            volume_keys = [self._volume_key(key, time) for (key, time) in zip(keys, times)]
            peaks = measure_rows(self._volume_peaks, volume_keys, bits, times)
            troughs = measure_rows(self._volume_troughs,
                                   volume_keys, [None if value is None else -value for value in bits],
                                   times)

            assign_nullable_float_column(df, ERROR_RATE, error_rates)
            assign_nullable_float_column(df, DISCARD_RATE, discard_rates)
            assign_nullable_float_column(df, BITS_IN, bits_in)
            assign_nullable_float_column(df, BITS_OUT, bits_out)
            assign_nullable_float_column(df, BITS, bits)
            assign_nullable_float_column(df, UTILIZATION, utilization)

            assign_nullable_float_column(df, f"{ERROR_RATE}{BASELINE_MAX_SUFFIX}", errors.references)
            assign_nullable_int_column(df, f"{ERROR_RATE}{BUCKETS_SUFFIX}", errors.buckets)
            assign_nullable_bool_column(df, f"{ERROR_RATE}{MATURE_SUFFIX}", errors.mature)
            assign_nullable_float_column(df, f"{ERROR_RATE}{STEP_SUFFIX}", errors.steps)

            assign_nullable_float_column(df, f"{BITS}{BASELINE_MAX_SUFFIX}", peaks.references)
            assign_nullable_float_column(df,
                                         f"{BITS}{BASELINE_MIN_SUFFIX}",
                                         [None if value is None else -value for value in troughs.references])
            assign_nullable_int_column(df, f"{BITS}{BUCKETS_SUFFIX}", peaks.buckets)
            assign_nullable_bool_column(df, f"{BITS}{MATURE_SUFFIX}", peaks.mature)
            assign_nullable_float_column(df, f"{BITS}{STEP_SUFFIX}", peaks.steps)

        refused = errors.unordered + peaks.unordered

        if (refused > 0):
            logger.warning(
                "TC1RateStage saw %d rates earlier than their port's previous one; they entered no history. Preserve "
                "per-port ordering upstream.",
                refused)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
