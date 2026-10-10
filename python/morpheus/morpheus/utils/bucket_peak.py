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
The most of something an entity has had in any one period of its own history, so this period can be measured
against it.

R-B-L2-002 reads this: a port's distinct-MAC count making a step change relative to the port's thirty-day baseline.
The count itself is `morpheus.utils.distinct_window`'s, over a trailing hour. What that count was on every other
hour of the last month is what this module keeps, reduced to the one figure the rule compares against -- the most
the port has ever carried in an hour -- so that a step is a count the port has never reached rather than a count
above some threshold somebody typed. An access port that has carried one device for a month and now carries five
has stepped; a trunk that carries three hundred and now carries three hundred and four has not.

**The baseline is a history of periods, not of rows.** A trailing window of rows saturates on a busy port in two
snapshots, and a month of them is unbounded. One figure per period per entity -- the peak the count reached in that
hour -- is bounded by time alone: seven hundred and twenty hours in thirty days, however many devices sit behind the
port. The period is the caller's to choose and should match the window the count itself is taken over, since the
peak of an hourly count means most when the periods are hours.

**The reference is the peaks before this period, never including it.** A count that could raise the baseline it is
measured against would never step; the open period's own peak joins the history when the next period opens. The
figure is therefore the same for every row of a period, however the rows are batched, which is what determinism
control 5 asks of it.

**Maturity is a count of periods, not of rows.** A port seen in two hours has no month to be measured against, and
a reference over two peaks is the larger of two numbers wearing a baseline's name. Below the floor the tracker
publishes nothing, and the rule stays quiet on a port the estate has only just met -- which is the honest answer,
and the one R-D-L2-001, with its designation list, exists to give instead.
"""

import collections
import dataclasses
import typing

NS_PER_SECOND = 10**9

DEFAULT_BUCKET_NS = 3600 * NS_PER_SECOND
"""The period a peak is kept per. An hour, which is the window the layer 2 cardinality count is taken over."""

DEFAULT_WINDOW_NS = 30 * 24 * 3600 * NS_PER_SECOND
"""How far back the peaks reach. Thirty days, which is the baseline R-B-L2-002 names."""

DEFAULT_MIN_BUCKETS = 24
"""Committed periods required before a reference is published. A day of hours."""

DEFAULT_MAX_BUCKETS = 1024
"""Periods retained per entity whatever the window implies. Thirty days of hours is seven hundred and twenty; the
cap is a bound on memory for a caller who sets the period shorter than the window warrants."""

DEFAULT_MAX_ENTITIES = 500_000
"""Entities retained before the least recently seen is dropped. Sized for ports, which run to the hundreds of
thousands on a large estate."""


@dataclasses.dataclass(frozen=True)
class PeakResult:
    """
    One observation, and the history it is measured against.

    Attributes
    ----------
    reference : number or None
        The highest peak among the committed periods inside the window, or `None` before the tracker is mature.
    buckets : int
        Committed periods the reference was taken over.
    mature : bool
        Whether a reference was published.
    step : number or None
        This value minus the reference, or `None` where the reference is. Positive means the entity has never, in
        any period of its history, had this many.
    saturated : bool
        Periods were dropped to stay inside `max_buckets`, so the reference covers a suffix of the history rather
        than the whole window.
    out_of_order : bool
        The observation is earlier than the entity's previous one, in which case it was refused.
    """

    reference: typing.Any
    buckets: int
    mature: bool
    step: typing.Any
    saturated: bool
    out_of_order: bool


@dataclasses.dataclass
class _EntityHistory:
    committed: collections.deque = dataclasses.field(default_factory=collections.deque)
    open_bucket: typing.Optional[int] = None
    open_peak: typing.Any = None
    last_time_ns: typing.Optional[int] = None
    reference: typing.Any = None
    saturated: bool = False


class BucketPeakTracker:
    """
    Per-entity history of per-period peaks, and each observation measured against the periods before its own.

    Results depend only on the sequence of observations the tracker has been shown, so replaying a stream
    reproduces them, and the reference every row of a period sees is the same whichever batch carried the row.

    Parameters
    ----------
    bucket_ns : int, default = 1 hour
        The period a peak is kept per, in nanoseconds of event time.
    window_ns : int, default = 30 days
        How far back the committed periods reach.
    min_buckets : int, default = 24
        Committed periods required before a reference is published.
    max_buckets : int, default = 1024
        Periods retained per entity whatever the window implies.
    max_entities : int, default = 500000
        Entities retained before the least recently seen is dropped. A dropped entity starts from no history.
    """

    def __init__(self,
                 bucket_ns: int = DEFAULT_BUCKET_NS,
                 window_ns: int = DEFAULT_WINDOW_NS,
                 min_buckets: int = DEFAULT_MIN_BUCKETS,
                 max_buckets: int = DEFAULT_MAX_BUCKETS,
                 max_entities: int = DEFAULT_MAX_ENTITIES):
        if (bucket_ns <= 0):
            raise ValueError(f"bucket_ns must be positive, received {bucket_ns}")

        if (window_ns < bucket_ns):
            raise ValueError(f"window_ns ({window_ns}) must be at least bucket_ns ({bucket_ns}), or no committed "
                             f"period could ever be inside the window")

        if (min_buckets < 1):
            raise ValueError(f"min_buckets must be at least 1, received {min_buckets}; a reference over no periods "
                             f"is nothing wearing a baseline's name")

        if (max_buckets < min_buckets):
            raise ValueError(f"max_buckets ({max_buckets}) must be at least min_buckets ({min_buckets}), or the "
                             f"tracker discards the periods it needs to mature")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._bucket_ns = bucket_ns
        self._window_ns = window_ns
        self._min_buckets = min_buckets
        self._max_buckets = max_buckets
        self._max_entities = max_entities

        self._histories: collections.OrderedDict[str, _EntityHistory] = collections.OrderedDict()

    @property
    def tracked_entities(self) -> int:
        """Entities currently holding a history."""
        return len(self._histories)

    def observe(self, entity_key: str, event_time_ns: int, value: typing.Any) -> PeakResult:
        """
        Record one observation and describe the history its entity's earlier periods supply.

        Parameters
        ----------
        entity_key : str
            What the peaks are kept per: a port, for R-B-L2-002.
        event_time_ns : int
            The observation's event time, in nanoseconds since the epoch. Event time, never ingest time: a period
            is a span of the estate's clock, not of the collector's.
        value : number
            The count this observation carries, typically a trailing-window distinct count.

        Returns
        -------
        `PeakResult`
        """
        history = self._histories.get(entity_key)

        if (history is None):
            history = _EntityHistory()
            self._histories[entity_key] = history
            self._evict()
        else:
            self._histories.move_to_end(entity_key)

        if (history.last_time_ns is not None and event_time_ns < history.last_time_ns):
            # A late observation belongs to a period whose peak may already be committed, and raising that peak
            # now would change the reference later rows were already measured against. Equal timestamps are not
            # late: a MAC table snapshot stamps every address on a port at one instant.
            return self._result(history, None, out_of_order=True)

        bucket = event_time_ns // self._bucket_ns

        if (history.open_bucket is None):
            history.open_bucket = bucket
            history.open_peak = value
        elif (bucket > history.open_bucket):
            # The period this observation opens closes the one before it, whose peak now joins the history.
            self._commit(history, event_time_ns)
            history.open_bucket = bucket
            history.open_peak = value
        elif (history.open_peak is None or value > history.open_peak):
            history.open_peak = value

        history.last_time_ns = event_time_ns

        return self._result(history, value, out_of_order=False)

    def _commit(self, history: _EntityHistory, now_ns: int) -> None:
        history.committed.append((history.open_bucket, history.open_peak))

        if (len(history.committed) > self._max_buckets):
            history.saturated = True

            while (len(history.committed) > self._max_buckets):
                history.committed.popleft()

        # A period is inside the window while any of it is: one that ended before the horizon has aged out.
        horizon = now_ns - self._window_ns

        while (len(history.committed) > 0 and (history.committed[0][0] + 1) * self._bucket_ns <= horizon):
            history.committed.popleft()

        # Recomputed once per period rather than once per row, which is what keeps a three-hundred-address trunk
        # snapshot from costing the whole history each row.
        peaks = [peak for (_, peak) in history.committed if peak is not None]
        history.reference = max(peaks) if len(peaks) > 0 else None

    def _result(self, history: _EntityHistory, value: typing.Any, out_of_order: bool) -> PeakResult:
        mature = len(history.committed) >= self._min_buckets and history.reference is not None
        reference = history.reference if mature else None
        step = None if (reference is None or value is None or out_of_order) else value - reference

        return PeakResult(reference=reference,
                          buckets=len(history.committed),
                          mature=mature,
                          step=step,
                          saturated=history.saturated,
                          out_of_order=out_of_order)

    def _evict(self) -> None:
        while (len(self._histories) > self._max_entities):
            self._histories.popitem(last=False)


@dataclasses.dataclass
class MeasuredRows:
    """
    A batch of observations measured against their entities' histories, one list entry per row.

    Attributes
    ----------
    references, buckets, mature, steps : list
        `PeakResult`'s fields for each row, or `None` throughout for a row the tracker did not see.
    keyless : int
        Rows with no entity or no value, which entered no history.
    unordered : int
        Rows with an entity and a value but no usable event time, or earlier than their entity's previous row.
    """

    references: list
    buckets: list
    mature: list
    steps: list
    keyless: int = 0
    unordered: int = 0


def measure_rows(tracker: BucketPeakTracker, keys: list, values: list, event_times_ns: list) -> MeasuredRows:
    """
    Show a tracker a batch in row order and collect what each row is measured against.

    A row whose key or value is `None` has nothing to hold a history against or nothing to measure. Pooling the
    keyless under a fabricated entity would make every bad row one very busy entity, so they carry nulls and are
    counted. So are rows with no event time, which cannot be placed in a period.

    Parameters
    ----------
    tracker : `BucketPeakTracker`
        The histories, which this call advances.
    keys : list
        Each row's entity, already normalized, or `None`.
    values : list
        Each row's value, already converted to a number, or `None`.
    event_times_ns : list
        Each row's event time in nanoseconds since the epoch, or `None`.

    Returns
    -------
    `MeasuredRows`
    """
    measured = MeasuredRows(references=[], buckets=[], mature=[], steps=[])

    for (key, value, event_time_ns) in zip(keys, values, event_times_ns):
        if (key is None or value is None or event_time_ns is None):
            measured.references.append(None)
            measured.buckets.append(None)
            measured.mature.append(None)
            measured.steps.append(None)
            measured.keyless += int(key is None or value is None)
            measured.unordered += int(key is not None and value is not None)
            continue

        result = tracker.observe(key, event_time_ns, value)

        measured.references.append(result.reference)
        measured.buckets.append(result.buckets)
        measured.mature.append(result.mature)
        measured.steps.append(result.step)
        measured.unordered += int(result.out_of_order)

    return measured
