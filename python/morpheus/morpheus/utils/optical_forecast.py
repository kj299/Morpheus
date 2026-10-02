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
Where a port's receive power is heading, and when it gets to the level the optic stops working at.

This is what R-P-L1-004 reads: a linear extrapolation of `optical_rx_dbm` per port, projecting the crossing of the
transceiver's minimum receive level, and firing when that crossing is within fourteen days. The guide calls it an
operations rule rather than a security one, and says it earns the layer 1 pipeline its budget: an optic that is
going to fail is cheap to replace on a Tuesday afternoon and expensive to replace at three in the morning.

**The line is fitted, not assumed.** Ordinary least squares over the port's readings in a trailing window gives a
slope in decibels per day and the value the line takes now; the floor minus that value, over the slope, is the time
to the floor. The slope is reported as fitted, and the forecast is published only where the line describes the
readings:

- **The trend has to be a line.** An inline tap steps the level down by one to three decibels at once; a re-patch
  steps it up. A step inside the window tilts a fitted line steeply, and a forecast read off that line would say the
  link reaches the floor in hours. The residual, how far the readings sit from the line on average, is what tells a
  step from a slope -- a degradation leaves residuals the size of the diagnostics' noise, a step leaves residuals
  the size of the step -- so a fit whose residual exceeds `max_residual_db` is reported as `nonlinear` rather than
  projected. The step itself is `morpheus.utils.optical_baseline`'s to report, and it does.
- **The slope has to be real.** A dozen readings of a healthy optic fit a line whose slope is whatever the noise
  happened to lean, and extrapolating that over fourteen days manufactures a failure out of nothing. The slope is
  therefore measured against its own standard error, and a slope fewer than `min_significance` errors from flat
  is reported as `not_degrading`, however steep its extrapolation. With a day of readings a real degradation is
  hundreds of errors from flat and noise is one or two.
- **The fit belongs to the optic, not the port.** A replaced transceiver has a new link budget, and the readings
  of the one it replaced say nothing about it. When the identity the caller supplies changes, the history is
  cleared and the new optic begins at `immature`.

The floor is the transceiver's, not the estate's: a 10GBASE-LR receiver works to about -14 dBm and a 1000BASE-LX
one to about -19, so one threshold would alarm on half the estate or none of it. The caller supplies the floor per
reading, looked up from whatever it knows about the optic; where it knows nothing the trend is still reported and
the status says there was nothing to project to.

Readings are in dBm, the slope in dB per day, and the time in days, which is the unit the rule is written in.
"""

import collections
import dataclasses
import math
import typing

from morpheus.utils.determinism import DEFAULT_FLOAT_DECIMALS
from morpheus.utils.determinism import quantize_value

NS_PER_SECOND = 10**9
NS_PER_DAY = 24 * 3600 * NS_PER_SECOND

DEFAULT_WINDOW_NS = 7 * NS_PER_DAY
"""Trailing window the line is fitted over. A week is long enough for a degradation of half a decibel a day to
stand well clear of the diagnostics' noise, and short enough that the reference is the link as it is now."""

DEFAULT_MIN_SAMPLES = 12
"""Readings required before a line is fitted. Fewer than this and the slope is the noise's."""

DEFAULT_MAX_SAMPLES = 2048
"""Readings retained per entity, whatever the window implies. A week of five-minute polls, and a bound on memory
when a collector is polling far faster than that."""

DEFAULT_MAX_RESIDUAL_DB = 0.5
"""How far the readings may sit from the fitted line, on average, before the trend is not a line. Optical
diagnostics report to a tenth of a decibel; the common inline taps cost one to three."""

DEFAULT_MIN_SIGNIFICANCE = 4.0
"""Standard errors a slope must be from flat before it is a degradation rather than noise."""

DEFAULT_MAX_ENTITIES = 100_000

STATUS_PROJECTED = "projected"
"""The line describes the readings, it falls, and the floor is where it is heading. `days_to_floor` is set."""

STATUS_NOT_DEGRADING = "not_degrading"
"""The level is steady, rising, or falling by less than the noise can vouch for."""

STATUS_BELOW_FLOOR = "below_floor"
"""The current reading is already at or under the floor. Nothing to forecast; the link is in trouble now."""

STATUS_NONLINEAR = "nonlinear"
"""The readings do not sit on a line, so a line extrapolated from them would be a fiction. A step, usually."""

STATUS_IMMATURE = "immature"
"""Too few readings for a fit."""

STATUS_NO_FLOOR = "no_floor"
"""The trend is fitted but the caller supplied no floor to project it to."""

STATUS_NO_READING = "no_reading"
"""This sample carried no level, which is a port with no optic in the cage."""

STATUSES = (STATUS_PROJECTED,
            STATUS_NOT_DEGRADING,
            STATUS_BELOW_FLOOR,
            STATUS_NONLINEAR,
            STATUS_IMMATURE,
            STATUS_NO_FLOOR,
            STATUS_NO_READING)
"""Every value `ForecastResult.status` takes, which is the bounded domain the column carries."""


@dataclasses.dataclass(frozen=True)
class ForecastResult:
    """
    The outcome of observing one reading.

    Attributes
    ----------
    status : str
        One of `STATUSES`. Which of the figures below are set depends on it.
    trend_db_per_day : float or None
        The fitted slope, negative for a falling level. `None` until the fit is mature or when the sample carried
        no reading.
    residual_db : float or None
        Root mean square distance of the readings from the fitted line. `None` where the slope is.
    significance : float or None
        How many of its own standard errors the slope is from flat. `None` where the slope is; infinite where the
        readings sit exactly on a line.
    samples : int
        Readings the fit covers, counting this one.
    floor_dbm : float or None
        The floor this reading was measured against, as supplied.
    days_to_floor : float or None
        Days until the fitted line reaches the floor. Set when projected; zero when already below the floor;
        `None` otherwise.
    saturated : bool
        Readings were dropped to stay inside `max_samples`, so the fit covers a suffix of the window.
    history_reset : bool
        The optic changed on this sample and the readings before it were discarded.
    out_of_order : bool
        The sample's event time was not after the previous sample's. State is left untouched.
    """

    status: str
    trend_db_per_day: typing.Optional[float]
    residual_db: typing.Optional[float]
    significance: typing.Optional[float]
    samples: int
    floor_dbm: typing.Optional[float]
    days_to_floor: typing.Optional[float]
    saturated: bool
    history_reset: bool
    out_of_order: bool


@dataclasses.dataclass(frozen=True)
class _Fit:
    slope_per_day: float
    value_now: float
    residual: float
    significance: float


@dataclasses.dataclass
class _EntityTrack:
    history: collections.deque = dataclasses.field(default_factory=collections.deque)
    last_time_ns: typing.Optional[int] = None
    optic: typing.Any = None
    seen_optic: bool = False
    saturated: bool = False


class OpticalForecastTracker:
    """
    Per-entity linear trend of a receive level, extrapolated to a floor.

    The tracker holds a trailing window of readings per entity and nothing else, so its memory is bounded by the
    entity count times the window. Results depend only on the sequence of samples it has been shown, so replaying
    a stream reproduces every forecast.

    Parameters
    ----------
    window_ns : int, default = 7 days
        Trailing window, in nanoseconds of event time, the line is fitted over.
    min_samples : int, default = 12
        Readings required before a line is fitted.
    max_samples : int, default = 2048
        Readings retained per entity regardless of the window.
    max_residual_db : float, default = 0.5
        Root mean square residual above which the readings are not on a line.
    min_significance : float, default = 4.0
        Standard errors a falling slope must be from flat before it is projected.
    max_entities : int, default = 100000
        Entities retained before the least recently seen is dropped. A dropped entity starts again, immature.
    decimals : int, default = 4
        Decimal places every reported figure is rounded to, under determinism control 9.
    """

    def __init__(self,
                 window_ns: int = DEFAULT_WINDOW_NS,
                 *,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 max_samples: int = DEFAULT_MAX_SAMPLES,
                 max_residual_db: float = DEFAULT_MAX_RESIDUAL_DB,
                 min_significance: float = DEFAULT_MIN_SIGNIFICANCE,
                 max_entities: int = DEFAULT_MAX_ENTITIES,
                 decimals: int = DEFAULT_FLOAT_DECIMALS):
        if (window_ns <= 0):
            raise ValueError(f"window_ns must be positive, received {window_ns}")

        if (min_samples < 3):
            raise ValueError(f"min_samples must be at least 3, received {min_samples}; two readings fit any line "
                             f"exactly and leave nothing to measure the fit by")

        if (max_samples < min_samples):
            raise ValueError(f"max_samples ({max_samples}) must be at least min_samples ({min_samples}), or a line "
                             f"could never be fitted")

        if (max_residual_db <= 0):
            raise ValueError(f"max_residual_db must be positive, received {max_residual_db}")

        if (min_significance < 0):
            raise ValueError(f"min_significance must not be negative, received {min_significance}")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._window_ns = window_ns
        self._min_samples = min_samples
        self._max_samples = max_samples
        self._max_residual_db = max_residual_db
        self._min_significance = min_significance
        self._max_entities = max_entities
        self._decimals = decimals

        self._tracks: collections.OrderedDict[str, _EntityTrack] = collections.OrderedDict()

    @property
    def tracked_entities(self) -> int:
        """Entities currently holding readings."""
        return len(self._tracks)

    def observe(self,
                entity_key: str,
                event_time_ns: int,
                reading: typing.Optional[float],
                floor_dbm: typing.Optional[float] = None,
                optic: typing.Any = None) -> ForecastResult:
        """
        Record one reading and return where the port's level is heading.

        Parameters
        ----------
        entity_key : str
            The port the reading belongs to, typically `site_id:device_id:port_id`.
        event_time_ns : int
            When the sample was taken, in nanoseconds since the epoch. Event time, never ingest time: a trend
            fitted against arrival order describes the collector's scheduling rather than the link.
        reading : float, optional
            The receive level in dBm. `None` for a port with no optic in the cage, which contributes nothing.
        floor_dbm : float, optional
            The level this optic stops working at, in dBm. `None` where the caller does not know it, in which case
            the trend is reported and nothing is projected.
        optic : any, optional
            The identity of the thing being fitted, typically the transceiver serial. A change discards the
            readings before it, since they were of a different optic. `None` is a value, so a cage that empties
            and is refilled starts over too.

        Returns
        -------
        `ForecastResult`
        """
        track = self._tracks.get(entity_key)

        if (track is None):
            track = _EntityTrack()
            self._tracks[entity_key] = track
            self._evict()
        else:
            self._tracks.move_to_end(entity_key)

        if (track.last_time_ns is not None and event_time_ns <= track.last_time_ns):
            # Admitting this would let a late reading rewrite a fit later samples were already judged against. An
            # equal timestamp is a duplicate poll: the level is polled per port, and one port has one level at
            # one instant.
            return self._result(STATUS_NO_READING if reading is None else STATUS_IMMATURE,
                                fit=None,
                                samples=len(track.history),
                                floor_dbm=floor_dbm,
                                days_to_floor=None,
                                track=track,
                                history_reset=False,
                                out_of_order=True)

        reset = track.seen_optic and optic != track.optic

        if (reset):
            # A new optic is a new link budget. The readings of the one before it are about something else.
            track.history.clear()
            track.saturated = False

        track.optic = optic
        track.seen_optic = True
        track.last_time_ns = event_time_ns

        horizon = event_time_ns - self._window_ns

        while (len(track.history) > 0 and track.history[0][0] <= horizon):
            track.history.popleft()

        fit = None
        days = None

        if (reading is None):
            status = STATUS_NO_READING
        else:
            track.history.append((event_time_ns, float(reading)))

            if (len(track.history) > self._max_samples):
                track.saturated = True

                while (len(track.history) > self._max_samples):
                    track.history.popleft()

            if (len(track.history) < self._min_samples):
                status = STATUS_IMMATURE
            else:
                fit = self._fit(track.history, event_time_ns)
                (status, days) = self._judge(fit, float(reading), floor_dbm)

        return self._result(status,
                            fit=fit,
                            samples=len(track.history),
                            floor_dbm=floor_dbm,
                            days_to_floor=days,
                            track=track,
                            history_reset=reset,
                            out_of_order=False)

    def _judge(self, fit: _Fit, reading: float, floor_dbm: typing.Optional[float]) -> tuple:
        """What the fitted line says about this reading, and how long it gives the optic."""
        if (floor_dbm is None):
            return (STATUS_NO_FLOOR, None)

        floor = float(floor_dbm)

        if (reading <= floor):
            return (STATUS_BELOW_FLOOR, 0.0)

        if (fit.residual > self._max_residual_db):
            return (STATUS_NONLINEAR, None)

        if (fit.slope_per_day >= 0 or fit.significance < self._min_significance):
            return (STATUS_NOT_DEGRADING, None)

        return (STATUS_PROJECTED, max((fit.value_now - floor) / (-fit.slope_per_day), 0.0))

    @staticmethod
    def _fit(history: collections.deque, now_ns: int) -> _Fit:
        """
        Ordinary least squares over the retained readings, with time measured in days before `now_ns`.

        Centering on the newest sample keeps the arithmetic in small numbers -- epoch nanoseconds squared would
        not survive a double -- and makes the intercept the line's value now, which is the figure the projection
        starts from.
        """
        times = [(stamp - now_ns) / NS_PER_DAY for (stamp, _) in history]
        levels = [level for (_, level) in history]
        count = len(times)

        mean_time = sum(times) / count
        mean_level = sum(levels) / count

        spread = sum((time - mean_time)**2 for time in times)
        covariance = sum((time - mean_time) * (level - mean_level) for (time, level) in zip(times, levels))

        slope = covariance / spread
        value_now = mean_level - slope * mean_time

        residuals = [level - (mean_level + slope * (time - mean_time)) for (time, level) in zip(times, levels)]
        squared = sum(residual**2 for residual in residuals)
        residual = math.sqrt(squared / count)

        # The slope's standard error, from the residual variance with the two fitted parameters taken out. Readings
        # exactly on a line leave no error to divide by, and the slope is as significant as a slope can be.
        error = math.sqrt(squared / (count - 2) / spread)
        significance = math.inf if error == 0 else abs(slope) / error

        return _Fit(slope_per_day=slope, value_now=value_now, residual=residual, significance=significance)

    def _result(self,
                status: str,
                *,
                fit: typing.Optional[_Fit],
                samples: int,
                floor_dbm: typing.Optional[float],
                days_to_floor: typing.Optional[float],
                track: _EntityTrack,
                history_reset: bool,
                out_of_order: bool) -> ForecastResult:
        return ForecastResult(status=status,
                              trend_db_per_day=None if fit is None else self._round(fit.slope_per_day),
                              residual_db=None if fit is None else self._round(fit.residual),
                              significance=None if fit is None else self._round(fit.significance),
                              samples=samples,
                              floor_dbm=None if floor_dbm is None else self._round(float(floor_dbm)),
                              days_to_floor=None if days_to_floor is None else self._round(days_to_floor),
                              saturated=track.saturated,
                              history_reset=history_reset,
                              out_of_order=out_of_order)

    def _round(self, value: float) -> float:
        # Infinity has no decimals to keep, and quantizing it would raise.
        return value if math.isinf(value) else quantize_value(value, decimals=self._decimals)

    def _evict(self) -> None:
        while (len(self._tracks) > self._max_entities):
            self._tracks.popitem(last=False)
