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
A stream clock that one bad timestamp cannot run away with.

The stateful telemetry stages expire their own state on event time: a binding that has gone quiet closes, an 802.1X
exchange that never resolved is abandoned. Expiry runs per row on that row's own event time rather than on a batch
boundary or a wall clock, because either of those would make the output depend on how the stream happened to be
divided, which is exactly what determinism control 13's batch-split sweep exists to catch.

That leaves the expiry horizon trusting whatever the collector stamped on a row. It should not. A switch whose NTP
has failed can report a time years ahead, and a single such row drives the horizon past every open binding in the
estate at once, closing all of them as idle timeouts and destroying the historical attribution with them. An
injected record does the same thing deliberately.

The bound here is deliberately not a wall clock. Comparing event time against the time of day would make a replay
of last week's data expire everything on sight, and would make the same corpus produce different output depending
on when it was run, which is the property the entire design rests on. The only reference available that keeps
replay honest is the stream's own progress: a time is implausible when it is further ahead of everything seen so
far than any real gap could account for.

The default skew is a week, which is chosen to catch the failure that actually happens -- a clock wrong by years,
from an unset RTC or a rollover -- while admitting any operational gap a real feed produces, including a pipeline
that was stopped overnight or a replay of an archive with holes in it. It does not catch a clock that is subtly
wrong, and it is not meant to; a device an hour ahead of its neighbours is a monitoring problem rather than a
correctness one for these stages.

A clock running *behind* needs no bound here. The stages already refuse an observation that would move a binding's
end backwards, and a stale row simply fails to advance the clock.
"""

import logging
import typing

from morpheus.utils.tracker_state import StateMixin

NS_PER_SECOND = 10**9

DEFAULT_MAX_SKEW_SECONDS = 7 * 24 * 3600
"""How far ahead of the stream's own progress a single row may be and still be believed."""

CLOCK_REFUSED_COLUMN = "clock_refused"
"""Per row: a guarded stage refused this row's event time as further ahead of the stream than it allows.

One column shared by every guarded stage rather than one per stage, because the question an operator asks is whether
the row was believed, and a row refused by any stage carries no per-entity features from that stage. Each stage ORs
its own refusals in, so a later stage cannot clear an earlier one's. It is a column rather than a log line so the
pipeline's health record can count it per collector and window, which is where a clock wrong by years shows up.
"""


class EventClock(StateMixin):
    """
    The largest event time the stream has been willing to believe.

    Parameters
    ----------
    max_skew_ns : int, default = 7 days
        How far ahead of the current clock a row's event time may be before it is refused. Must be positive.

    Notes
    -----
    The clock is a pure function of the sequence of times it has been shown, so two runs over the same rows reach
    the same value however the rows were batched.
    """

    def __init__(self, max_skew_ns: int = DEFAULT_MAX_SKEW_SECONDS * NS_PER_SECOND):
        if (max_skew_ns <= 0):
            raise ValueError(f"max_skew_ns must be positive, received {max_skew_ns}")

        self._max_skew_ns = max_skew_ns
        self._value_ns: typing.Optional[int] = None

    @property
    def value_ns(self) -> typing.Optional[int]:
        """The current clock, or `None` before any time has been accepted."""
        return self._value_ns

    def accept(self, event_time_ns: int) -> bool:
        """
        Judge a row's event time, advancing the clock when it is believable.

        The first time seen establishes the clock. This is the one place a single row still sets it, and the
        stateful stages run behind a total order, so the row that establishes it is the earliest in the stream
        rather than an outlier that sorted to the end.

        Parameters
        ----------
        event_time_ns : int
            The row's event time, in nanoseconds since the Unix epoch.

        Returns
        -------
        bool
            `True` when the time is plausible and the clock now covers it. `False` when it is further ahead than
            `max_skew_ns`, in which case the clock is left untouched and the caller should give the row no
            per-entity features rather than letting it drive expiry.
        """
        if (self._value_ns is None):
            self._value_ns = event_time_ns

            return True

        if (event_time_ns > self._value_ns + self._max_skew_ns):
            return False

        if (event_time_ns > self._value_ns):
            self._value_ns = event_time_ns

        return True


def clock_for_skew_seconds(max_clock_skew_seconds: int) -> EventClock:
    """
    Build a stage's clock from its `max_clock_skew_seconds` parameter.

    The one place the parameter is validated, so the error names the parameter the operator set rather than the
    nanosecond bound the clock holds inside.

    Raises
    ------
    ValueError
        If `max_clock_skew_seconds` is not positive.
    """
    if (max_clock_skew_seconds <= 0):
        raise ValueError(f"max_clock_skew_seconds must be positive, received {max_clock_skew_seconds}")

    return EventClock(max_skew_ns=max_clock_skew_seconds * NS_PER_SECOND)


def event_times_ns(df, column: str, time_unit: str = "ns") -> list[typing.Optional[int]]:
    """
    Read a frame's event-time column as nanoseconds since the epoch, one entry per row.

    Parameters
    ----------
    df : `pandas.DataFrame` or `cudf.DataFrame`
        The frame the stage is reading.
    column : str
        The event-time column.
    time_unit : str, default = "ns"
        Unit for numeric values, as `morpheus.utils.binding_table.to_epoch_ns` takes it.

    Returns
    -------
    list of int or None
        The row's time, or `None` where the value was null or could not be interpreted as a timestamp. A tracker
        treats `None` as a row with nothing to measure, which is also what `guard_times` turns a refused time into.
    """
    # Imported here so that this module stays free of a frame library for the stages that only need the clock.
    from morpheus.utils.binding_table import to_epoch_ns
    from morpheus.utils.column_assign import to_host_list

    times = []

    for raw in to_host_list(df, column):
        try:
            times.append(to_epoch_ns(raw, time_unit=time_unit))
        except ValueError:
            times.append(None)

    return times


def _truthy(value: typing.Any) -> bool:
    try:
        # A nullable boolean, compared by value rather than identity.
        return value is not None and bool(value == True)  # noqa: E712 pylint: disable=singleton-comparison
    except (TypeError, ValueError):
        return False


def mark_refused(df, refused: typing.Sequence[bool]) -> None:
    """
    OR one stage's refusals into `CLOCK_REFUSED_COLUMN`.

    Parameters
    ----------
    df : `pandas.DataFrame` or `cudf.DataFrame`
        The frame the stage is writing, in place.
    refused : sequence of bool
        One entry per row: whether this stage refused it.
    """
    # Imported here so that this module stays free of a frame library for the stages that only need the clock.
    from morpheus.utils.column_assign import assign_nullable_bool_column
    from morpheus.utils.column_assign import to_host_list

    earlier = to_host_list(df, CLOCK_REFUSED_COLUMN) if CLOCK_REFUSED_COLUMN in df.columns else [None] * len(refused)

    assign_nullable_bool_column(df,
                                CLOCK_REFUSED_COLUMN,
                                [_truthy(prior) or bool(now) for (prior, now) in zip(earlier, refused)])


def refuse_rows(df, refused: typing.Sequence[bool], *, logger: logging.Logger, stage: str, what: str,
                outcome: str) -> int:
    """
    Record one stage's refusals on the frame, and say so when there were any.

    The mark is written on every frame, refused or not, so that whether a frame carries `CLOCK_REFUSED_COLUMN` is a
    property of the stage and not of which rows happened to share a batch: a `False` on a believed row answers the
    health record's count as 0 however the stream was split, where a missing column would answer it null.

    Parameters
    ----------
    df : `pandas.DataFrame` or `cudf.DataFrame`
        The frame the stage is writing, in place.
    refused : sequence of bool
        One entry per row: whether this stage refused it, as `guard_times` reports.
    logger : `logging.Logger`
        The stage's logger, so the warning is filed under the stage's module.
    stage : str
        The stage's name, as the warning's subject.
    what : str
        What a row of this stage's input is, in the plural: "flows", "handshakes", "queries".
    outcome : str
        What became of the refused rows, completing the sentence after "max_clock_skew_seconds allows;".

    Returns
    -------
    int
        How many rows were refused.
    """
    mark_refused(df, refused)
    count = sum(1 for flag in refused if flag)

    if (count > 0):
        logger.warning(
            "%s refused %d of %d %s whose event time was further ahead of the stream than max_clock_skew_seconds "
            "allows; %s",
            stage,
            count,
            len(refused),
            what,
            outcome)

    return count


def guard_times(clock: EventClock, times: typing.Sequence[typing.Optional[int]]) -> tuple[list, list]:
    """
    Pass a column of event times through a clock, in row order.

    For the stages that hand a whole column of times to a tracker rather than walking the rows themselves. A
    refused time becomes `None`, which every tracker already treats as a row with nothing to measure, so the row
    carries no per-entity features and moves no horizon.

    Parameters
    ----------
    clock : `EventClock`
        The stage's clock; advanced as a side effect.
    times : sequence of int or None
        Event times in nanoseconds, `None` where the row had none.

    Returns
    -------
    tuple of (list, list)
        The times with refused ones replaced by `None`, and one boolean per row saying which were refused.
    """
    guarded = []
    refused = []

    for value in times:
        believed = value is None or clock.accept(value)
        guarded.append(value if believed else None)
        refused.append(not believed)

    return (guarded, refused)
