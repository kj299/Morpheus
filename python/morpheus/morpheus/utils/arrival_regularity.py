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
How regular a flow of events is, which is the signal beaconing consists of.

People are irregular. A workstation talks to a file server when somebody opens a file, and the gaps between those
conversations are as ragged as the working day. Implants are not irregular: they ask for instructions on a timer,
and the timer is the tell. R-B-L3-002 thresholds that tell as a coefficient of variation of the inter-arrival
time below 0.15 over at least twelve intervals, with the per-flow byte counts in a narrow band.

The coefficient of variation rather than the standard deviation, because regularity is scale-free. A beacon every
sixty seconds jittering by a second and one every hour jittering by a minute are equally regular, and only the
ratio says so. A threshold on the deviation alone would be a threshold on the beacon period.

Four decisions in here are decisions rather than details.

**An interval is between two arrivals, so `n` arrivals give `n - 1` intervals.** The guide's twelve intervals are
thirteen flows, and reporting the interval count rather than the sample count is what keeps that from being
rounded in somebody's head. Below the minimum the coefficient is reported as `None` rather than as a number,
because a ratio over two intervals is arithmetic rather than evidence, and a detector that emitted it would fire
on any pair of evenly spaced flows.

**A mean interval of zero yields no coefficient, not a coefficient of zero.** Several flows stamped at one instant
-- a collector that batches, or a source that opens a connection per destination in a burst -- have no spacing to
be regular about. Zero is the most beacon-like value this can produce, so returning it for the one case where the
quantity is undefined would make the flattest possible input the most alarming, which is the mistake
`morpheus.utils.drift_trajectory` already refuses to make for the same reason.

**The window is trailing and a pause is not an interval.** A beacon that stops for a week and restarts is still a
beacon; treating the week as one enormous interval would say the opposite, and keeping it forever would let one
ancient arrival hold the mean up indefinitely. Arrivals older than the window are dropped, so what is measured is
the rhythm of the recent past, and a pause longer than the window simply leaves less to measure.

**Size regularity is the same computation over a different series**, and it is kept beside the timing rather than
folded into it. A pair that is regular in time and wildly variable in size is a poll; one regular in both is a
poll carrying a fixed-length message. The rule wants both and the two are reported separately, because an
operator triaging an alert needs to see which half fired.
"""

import collections
import dataclasses
import statistics
import typing

NS_PER_SECOND = 10**9

DEFAULT_WINDOW_NS = 24 * 3600 * NS_PER_SECOND
"""Trailing window arrivals are retained for.

A day, because a beacon period is minutes to hours and twelve intervals of an hourly beacon need half of one. A
shorter window cannot see a slow beacon at all; a longer one measures a rhythm the host may have finished with.
"""

DEFAULT_MIN_INTERVALS = 12
"""Intervals required before a coefficient is published, which is the figure R-B-L3-002 names.

Thirteen arrivals. Below it the coefficient exists arithmetically and means nothing: two flows a minute apart and
a third a minute after that are perfectly regular by this measure and are not a beacon.
"""

DEFAULT_MAX_SAMPLES = 4096
"""Arrivals retained per entity whatever the window implies, so memory is bounded by entities rather than traffic.

When it binds the coefficient describes the most recent `max_samples` arrivals rather than the whole window, which
is a narrower claim and is marked as one.
"""

DEFAULT_MAX_ENTITIES = 500_000
"""Pairs retained before the least recently seen is dropped. A dropped pair starts again from no history."""


@dataclasses.dataclass(frozen=True)
class RegularityResult:
    """
    The outcome of observing one arrival.

    Attributes
    ----------
    arrivals : int
        Arrivals in the window, counting this one.
    intervals : int
        Gaps between them, which is `arrivals - 1`.
    mean_interval_ns : int or None
        Mean gap, rounded to whole nanoseconds. `None` before there is an interval to average.
    interval_cv : float or None
        Standard deviation of the gaps over their mean. `None` below `min_intervals`, and `None` when the mean is
        zero, which is undefined rather than perfectly regular.
    size_cv : float or None
        The same ratio over the sizes supplied with each arrival. `None` when no sizes were supplied, when there
        are fewer than `min_intervals` intervals, or when the mean size is zero.
    mature : bool
        There are at least `min_intervals` intervals, so the coefficients are published.
    saturated : bool
        The sample cap is binding, so the coefficients describe the retained tail rather than the window.
    out_of_order : bool
        This arrival's event time was not after the previous one's. State is left untouched, and every field
        describes the window as it stood before the arrival.
    """

    arrivals: int
    intervals: int
    mean_interval_ns: typing.Optional[int]
    interval_cv: typing.Optional[float]
    size_cv: typing.Optional[float]
    mature: bool
    saturated: bool
    out_of_order: bool


@dataclasses.dataclass
class _EntityWindow:
    """One pair's retained arrivals."""

    times: collections.deque = dataclasses.field(default_factory=collections.deque)
    sizes: collections.deque = dataclasses.field(default_factory=collections.deque)
    last_time_ns: typing.Optional[int] = None
    saturated: bool = False


def coefficient_of_variation(values: typing.Sequence[float]) -> typing.Optional[float]:
    """
    Standard deviation over mean, or `None` where that is not a quantity.

    The population standard deviation rather than the sample one. What is being described is the spread of the
    intervals actually observed, not an estimate of the spread of a process that produced them, and Bessel's
    correction answers the second question. At twelve intervals the two differ by about four percent, which is
    enough to move a value across a threshold of 0.15, so the choice is recorded rather than left to the reader
    of `statistics`.

    Parameters
    ----------
    values : sequence of float
        The series. Fewer than two values, or a mean of zero, yields `None`.

    Returns
    -------
    float or None
    """
    if (len(values) < 2):
        return None

    mean = statistics.fmean(values)

    if (mean == 0):
        return None

    return statistics.pstdev(values) / mean


class ArrivalRegularityTracker:
    """
    Per-entity regularity of arrival times, and of the sizes that came with them.

    The entity is whatever the caller keys on. R-B-L3-002 keys on the directed pair `src_ip:dst_ip`, which is the
    unit a beacon actually exists between; keying on the source alone would mix a beacon in with everything else
    that host does and hide it in the variance.

    Results depend only on the sequence of arrivals the tracker has been shown, so replaying a stream reproduces
    them.

    Parameters
    ----------
    window_ns : int, default = 1 day
        Trailing window arrivals are retained for.
    min_intervals : int, default = 12
        Intervals required before the coefficients are published.
    max_samples : int, default = 4096
        Arrivals retained per entity regardless of the window.
    max_entities : int, default = 500000
        Entities retained before the least recently seen is dropped.
    """

    def __init__(self,
                 window_ns: int = DEFAULT_WINDOW_NS,
                 min_intervals: int = DEFAULT_MIN_INTERVALS,
                 max_samples: int = DEFAULT_MAX_SAMPLES,
                 max_entities: int = DEFAULT_MAX_ENTITIES):
        if (window_ns <= 0):
            raise ValueError(f"window_ns must be positive, received {window_ns}")

        if (min_intervals < 2):
            raise ValueError(f"min_intervals must be at least 2, received {min_intervals}; a coefficient over "
                             f"one interval is not a coefficient")

        if (max_samples < 2):
            raise ValueError(f"max_samples must be at least 2, received {max_samples}")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._window_ns = window_ns
        self._min_intervals = min_intervals
        self._max_samples = max_samples
        self._max_entities = max_entities

        self._windows: collections.OrderedDict[str, _EntityWindow] = collections.OrderedDict()

    @property
    def tracked_entities(self) -> int:
        """Entities currently holding a window."""
        return len(self._windows)

    def _window_for(self, entity_key: str) -> _EntityWindow:
        window = self._windows.get(entity_key)

        if (window is None):
            window = _EntityWindow()
            self._windows[entity_key] = window

            while (len(self._windows) > self._max_entities):
                self._windows.popitem(last=False)
        else:
            self._windows.move_to_end(entity_key)

        return window

    def _describe(self, window: _EntityWindow, out_of_order: bool) -> RegularityResult:
        """Render one pair's retained arrivals as a result."""
        times = list(window.times)
        intervals = [float(later - earlier) for (earlier, later) in zip(times, times[1:])]
        mature = len(intervals) >= self._min_intervals

        mean_interval_ns = round(statistics.fmean(intervals)) if len(intervals) > 0 else None
        interval_cv = coefficient_of_variation(intervals) if mature else None

        sizes = [size for size in window.sizes if size is not None]
        # Sizes are compared over the same maturity bar as the intervals, so the two halves of R-B-L3-002 are
        # published together. A size series shorter than the arrivals means the caller supplied some and not
        # others, and a coefficient over the ones that happened to have a value would describe that accident.
        size_cv = coefficient_of_variation(sizes) if (mature and len(sizes) == len(times)) else None

        return RegularityResult(arrivals=len(times),
                                intervals=len(intervals),
                                mean_interval_ns=mean_interval_ns,
                                interval_cv=interval_cv,
                                size_cv=size_cv,
                                mature=mature,
                                saturated=window.saturated,
                                out_of_order=out_of_order)

    def observe(self, entity_key: str, event_time_ns: int, size: typing.Optional[float] = None) -> RegularityResult:
        """
        Record one arrival and describe the entity's rhythm including it.

        Parameters
        ----------
        entity_key : str
            What the regularity is measured for, usually a directed address pair.
        event_time_ns : int
            The arrival's event time, in nanoseconds since the epoch. Event time, never ingest time.
        size : float, optional
            A magnitude to measure the regularity of alongside the timing, usually the flow's byte count.

        Returns
        -------
        `RegularityResult`

        Raises
        ------
        ValueError
            If `entity_key` is empty or `event_time_ns` is not an integer.
        """
        if (not entity_key):
            raise ValueError("entity_key is required; a rhythm has to be a rhythm of something")

        if (not isinstance(event_time_ns, int) or isinstance(event_time_ns, bool)):
            raise ValueError(f"event_time_ns must be an int, received {event_time_ns!r}")

        window = self._window_for(entity_key)

        # Out-of-order arrivals are refused rather than repaired, exactly as the other trackers refuse them: an
        # interval computed from a timestamp that went backwards is negative, and a negative interval in the mean
        # would make a shuffled stream look more regular than the ordered one it came from.
        if (window.last_time_ns is not None and event_time_ns <= window.last_time_ns):
            return self._describe(window, out_of_order=True)

        window.times.append(event_time_ns)
        window.sizes.append(None if size is None else float(size))
        window.last_time_ns = event_time_ns

        horizon = event_time_ns - self._window_ns

        while (len(window.times) > 0 and window.times[0] < horizon):
            window.times.popleft()
            window.sizes.popleft()

        if (len(window.times) > self._max_samples):
            window.saturated = True

            while (len(window.times) > self._max_samples):
                window.times.popleft()
                window.sizes.popleft()

        return self._describe(window, out_of_order=False)
