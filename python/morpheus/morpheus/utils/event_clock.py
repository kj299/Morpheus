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

import typing

NS_PER_SECOND = 10**9

DEFAULT_MAX_SKEW_SECONDS = 7 * 24 * 3600
"""How far ahead of the stream's own progress a single row may be and still be believed."""


class EventClock:
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
