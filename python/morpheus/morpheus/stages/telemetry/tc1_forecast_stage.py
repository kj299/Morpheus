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
"""Extrapolates each port's receive power to the level its optic stops working at."""

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
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.determinism import DEFAULT_FLOAT_DECIMALS
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.optical_forecast import DEFAULT_MAX_RESIDUAL_DB
from morpheus.utils.optical_forecast import DEFAULT_MAX_SAMPLES
from morpheus.utils.optical_forecast import DEFAULT_MIN_SAMPLES
from morpheus.utils.optical_forecast import DEFAULT_MIN_SIGNIFICANCE
from morpheus.utils.optical_forecast import NS_PER_SECOND
from morpheus.utils.optical_forecast import OpticalForecastTracker

logger = logging.getLogger(__name__)

DEFAULT_CHANNEL_COLUMN = "optical_rx_dbm"
"""The level that degrades with the path. Transmit power is the local laser's and is not forecast here."""

DEFAULT_TYPE_COLUMN = "transceiver_type"
"""The optic's type, which is what its minimum receive level is a property of."""

DEFAULT_OPTIC_COLUMN = "transceiver_serial"
"""The optic's identity, whose change starts the port's history over."""

DEFAULT_WINDOW_SECONDS = 7 * 24 * 3600
"""Trailing window the line is fitted over, in seconds of event time."""

TREND_SUFFIX = "_trend_db_per_day"
RESIDUAL_SUFFIX = "_trend_residual_db"
SIGNIFICANCE_SUFFIX = "_trend_significance"
SAMPLES_SUFFIX = "_trend_samples"
FLOOR_SUFFIX = "_floor_dbm"
DAYS_SUFFIX = "_days_to_floor"
STATUS_SUFFIX = "_forecast_status"


@register_stage("tc1-forecast", ignore_args=["floors"])
class TC1ForecastStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Fit each port's receive level over its recent history and project when it reaches its optic's floor.

    This is what R-P-L1-004 reads. `morpheus.stages.telemetry.tc1_optical_stage.TC1OpticalStage` says how far a
    port's level sits from what it was; this stage says where it is going, which is the operations question a layer
    1 pipeline is cheapest at answering. The arithmetic and its three refusals -- a trend that is not a line, a slope
    the noise could have produced, a replaced optic whose predecessor's readings are not its own -- are in
    `morpheus.utils.optical_forecast`, and the columns here are its figures:

    - `<channel>_trend_db_per_day`, the fitted slope, negative for a falling level;
    - `<channel>_trend_residual_db` and `<channel>_trend_significance`, how well the line fits and how far the
      slope is from flat in its own standard errors, which are the figures to read when tuning the two refusals;
    - `<channel>_trend_samples`, the readings the fit covers;
    - `<channel>_floor_dbm`, the floor this row was projected to;
    - `<channel>_days_to_floor`, set when the status is `projected`, and the quantity the rule thresholds;
    - `<channel>_forecast_status`, one of the values `morpheus.utils.optical_forecast.STATUSES` names, saying why
      a row carries a projection or does not.

    The floor is the transceiver's, looked up by type from `floors`: a 10GBASE-LR receiver works to about -14 dBm
    and a 1000BASE-LX one to about -19, so no single figure serves an estate. A type the mapping does not name
    takes `default_floor_dbm`, and with neither the trend is reported under `no_floor`. The mapping is supplied to
    the stage rather than carried on the event because it is a fact about the optic's datasheet, not about the poll.

    The stage is stateful across messages, holding a trailing window of readings per port, and must run
    single-engine. For parallelism, shard by device upstream and give each shard its own instance, which is
    determinism control 4. Place it after
    `morpheus.stages.telemetry.tc1_normalize_stage.TC1NormalizeStage`, which supplies `entity_key`.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    entity_key_column : str, default = "entity_key"
        Column holding the port identity the trend is kept per.
    time_column : str, default = "event_time"
        Column holding the sample's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    channel_column : str, default = "optical_rx_dbm"
        Column holding the level to extrapolate, in dBm. Also the prefix of every column written.
    type_column : str, default = "transceiver_type"
        Column holding the optic's type, which `floors` is keyed by. Compared case-insensitively. A frame without
        it takes `default_floor_dbm` for every row.
    floors : mapping of str to float, optional
        Minimum receive level per transceiver type, in dBm. Supplied in code rather than on the command line.
    default_floor_dbm : float, optional
        Floor for a type the mapping does not name. Left unset, such a row is reported `no_floor`.
    optic_column : str, optional
        Column holding the optic's identity, whose change discards the port's readings before it. Defaults to
        `transceiver_serial`; pass `None` to fit across replacements, which is wrong for a forecast and right for
        almost nothing.
    window_seconds : int, default = 604800
        Trailing window the line is fitted over. A week.
    min_samples : int, default = 12
        Readings required before a line is fitted.
    max_samples : int, default = 2048
        Readings retained per port regardless of the window.
    max_residual_db : float, default = 0.5
        Root mean square residual above which the readings are not on a line and nothing is projected.
    min_significance : float, default = 4.0
        Standard errors a falling slope must be from flat before it is projected.
    decimals : int, default = 4
        Decimal places every reported figure is rounded to, under determinism control 9.
    """

    def __init__(self,
                 c: Config,
                 entity_key_column: str = "entity_key",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 channel_column: str = DEFAULT_CHANNEL_COLUMN,
                 type_column: str = DEFAULT_TYPE_COLUMN,
                 floors: typing.Mapping[str, float] = None,
                 default_floor_dbm: float = None,
                 optic_column: str = DEFAULT_OPTIC_COLUMN,
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 max_samples: int = DEFAULT_MAX_SAMPLES,
                 max_residual_db: float = DEFAULT_MAX_RESIDUAL_DB,
                 min_significance: float = DEFAULT_MIN_SIGNIFICANCE,
                 decimals: int = DEFAULT_FLOAT_DECIMALS):
        super().__init__(c)

        if (not channel_column):
            raise ValueError("channel_column is required; it names the level to fit and prefixes every column written")

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        self._entity_key_column = entity_key_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._channel_column = channel_column
        self._type_column = type_column
        self._optic_column = optic_column
        self._default_floor = None if default_floor_dbm is None else float(default_floor_dbm)

        floors = {} if floors is None else floors
        self._floors = {self._type_key(name): float(level) for (name, level) in floors.items()}

        if (None in self._floors):
            raise ValueError("floors names a blank transceiver type; the default floor is default_floor_dbm")

        self._tracker = OpticalForecastTracker(window_ns=window_seconds * NS_PER_SECOND,
                                               min_samples=min_samples,
                                               max_samples=max_samples,
                                               max_residual_db=max_residual_db,
                                               min_significance=min_significance,
                                               decimals=decimals)

        self._warned_typeless = False

        self._needed_columns[f"{channel_column}{TREND_SUFFIX}"] = TypeId.FLOAT64
        self._needed_columns[f"{channel_column}{RESIDUAL_SUFFIX}"] = TypeId.FLOAT64
        self._needed_columns[f"{channel_column}{SIGNIFICANCE_SUFFIX}"] = TypeId.FLOAT64
        self._needed_columns[f"{channel_column}{SAMPLES_SUFFIX}"] = TypeId.INT64
        self._needed_columns[f"{channel_column}{FLOOR_SUFFIX}"] = TypeId.FLOAT64
        self._needed_columns[f"{channel_column}{DAYS_SUFFIX}"] = TypeId.FLOAT64
        self._needed_columns[f"{channel_column}{STATUS_SUFFIX}"] = TypeId.STRING

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc1-forecast"

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
    def tracked_entities(self) -> int:
        """Ports currently holding readings."""
        return self._tracker.tracked_entities

    @staticmethod
    def _type_key(name: typing.Any) -> typing.Optional[str]:
        """The mapping key a transceiver type is looked up by: trimmed and case-folded, so `10GBASE-LR` and
        `10gbase-lr` are one optic."""
        normalized = normalize_text(name)

        return None if normalized is None else normalized.lower()

    @staticmethod
    def _reading(value: typing.Any) -> typing.Optional[float]:
        """Return a host value as a float, or `None` where the port reported no level."""
        if (value is None):
            return None

        try:
            reading = float(value)
        except (TypeError, ValueError):
            return None

        # A null in a float column arrives as NaN, which is not a level and must not enter a fit.
        return None if math.isnan(reading) else reading

    def _floor_for(self, optic_type: typing.Any) -> typing.Optional[float]:
        return self._floors.get(self._type_key(optic_type), self._default_floor)

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the trend, the fit's quality, the floor and the projection for every row.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming message.

        Returns
        -------
        The input message, with the forecast columns populated.

        Raises
        ------
        KeyError
            If the entity key, the time column, or the channel column is absent. The type and optic columns may be
            absent: a frame without the type takes the default floor, and one without the optic never resets.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._entity_key_column, self._time_column, self._channel_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC1ForecastStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            entity_keys = to_host_list(df, self._entity_key_column)
            raw_times = to_host_list(df, self._time_column)
            levels = to_host_list(df, self._channel_column)
            row_count = len(entity_keys)

            has_types = self._type_column in df.columns
            types = to_host_list(df, self._type_column) if has_types else [None] * row_count
            has_optics = self._optic_column is not None and self._optic_column in df.columns
            optics = to_host_list(df, self._optic_column) if has_optics else [None] * row_count

            trends: list = []
            residuals: list = []
            significances: list = []
            samples: list = []
            floors: list = []
            days: list = []
            statuses: list = []
            unordered = 0
            keyless = 0

            for position in range(row_count):
                # A row whose key is null has no identity to hold state against. The contract in
                # `morpheus.utils.entity_key` is that a null key gets no per-entity features and the stage says how
                # many rows that happened to.
                key = normalize_text(entity_keys[position])

                try:
                    event_time_ns = None if key is None else to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                if (key is None or event_time_ns is None):
                    trends.append(None)
                    residuals.append(None)
                    significances.append(None)
                    samples.append(None)
                    floors.append(None)
                    days.append(None)
                    statuses.append(None)
                    keyless += int(key is None)
                    unordered += int(key is not None)
                    continue

                result = self._tracker.observe(key,
                                               event_time_ns,
                                               self._reading(levels[position]),
                                               floor_dbm=self._floor_for(types[position]),
                                               optic=normalize_text(optics[position]))

                trends.append(result.trend_db_per_day)
                residuals.append(result.residual_db)
                significances.append(result.significance)
                samples.append(result.samples)
                floors.append(result.floor_dbm)
                days.append(result.days_to_floor)
                statuses.append(result.status)
                unordered += int(result.out_of_order)

            prefix = self._channel_column
            assign_nullable_float_column(df, f"{prefix}{TREND_SUFFIX}", trends)
            assign_nullable_float_column(df, f"{prefix}{RESIDUAL_SUFFIX}", residuals)
            assign_nullable_float_column(df, f"{prefix}{SIGNIFICANCE_SUFFIX}", significances)
            assign_nullable_int_column(df, f"{prefix}{SAMPLES_SUFFIX}", samples)
            assign_nullable_float_column(df, f"{prefix}{FLOOR_SUFFIX}", floors)
            assign_nullable_float_column(df, f"{prefix}{DAYS_SUFFIX}", days)
            assign_str_column(df, f"{prefix}{STATUS_SUFFIX}", statuses)

        if (not has_types and self._default_floor is None and not self._warned_typeless):
            self._warned_typeless = True
            logger.warning(
                "TC1ForecastStage found no %s column and has no default_floor_dbm, so every row is reported "
                "no_floor: the trend is fitted and nothing is projected. Supply the optic type on the row, or a "
                "default floor. Said once rather than per batch.",
                self._type_column)

        if (keyless > 0):
            logger.warning(
                "TC1ForecastStage saw %d of %d rows with a null entity key; they carry no forecast. A null key "
                "means a missing site, device, or port upstream, not a port named \"None\".",
                keyless,
                row_count)

        if (unordered > 0):
            logger.warning(
                "TC1ForecastStage saw %d of %d samples out of order or without a usable event time; they carry no "
                "forecast and did not enter any fit. Shard by device and preserve per-port ordering upstream.",
                unordered,
                row_count)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
