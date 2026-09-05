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
How long an exchange took, by pairing its start with its outcome.

The TC-2 telemetry class asks for the time-to-authorize distribution per port, and the reason it is a distribution
rather than a threshold is that both tails mean something. A slow authorization is a supplicant retrying, a RADIUS
server under load, or credentials being guessed. A very fast one can be a replayed or cached success. And the
interesting case is neither tail: an outcome with no exchange in front of it at all, which is what an 802.1X
bypass looks like from the switch, since MAC authentication bypass and a device bridged behind an already
authorized supplicant both produce authorization without anybody authenticating.

That last case is why an unpaired outcome is reported rather than dropped. Dropping it would make the feature
silent on exactly the condition it exists to detect, and a null elapsed time with no explanation reads as missing
data rather than as an event.

Restarts are counted too. A supplicant that begins three times before succeeding did not take one attempt, and the
elapsed time from the last begin would hide the two before it, so the count travels with the timing.
"""

import collections
import dataclasses
import typing

NS_PER_SECOND = 10**9

DEFAULT_TIMEOUT_NS = 300 * NS_PER_SECOND
"""Silence after which a pending exchange is abandoned. An 802.1X exchange that has not resolved in five minutes
has failed in some way the switch never reported.
"""

DEFAULT_MAX_PENDING = 500_000
"""Exchanges held open before the least recently started is dropped."""


@dataclasses.dataclass(frozen=True)
class SessionTiming:
    """
    The outcome of completing one exchange.

    Attributes
    ----------
    key : str
        What the exchange belongs to, typically a port.
    elapsed_ns : int or None
        Time from the most recent begin to this outcome. `None` when no begin was seen, which is the unpaired case.
    outcome : str or None
        Whatever the source reported as the result.
    attempts : int
        Begins observed for this exchange, so a success after three retries is distinguishable from a first-time
        one. Zero when unpaired.
    unpaired : bool
        An outcome arrived with no exchange in front of it. Reported rather than discarded, because authorization
        without authentication is the bypass signal.
    out_of_order : bool
        The outcome predates the begin it would be paired with. Nothing is timed.
    """

    key: str
    elapsed_ns: typing.Optional[int]
    outcome: typing.Optional[str]
    attempts: int
    unpaired: bool
    out_of_order: bool


@dataclasses.dataclass
class _Pending:
    started_ns: int
    attempts: int


class SessionTimer:
    """
    Per-key pairing of an exchange's start with its outcome.

    The timer holds one pending exchange per key, so its memory is bounded by the number of simultaneously open
    exchanges rather than by the stream. Results depend only on the sequence of calls it has been shown.

    Parameters
    ----------
    timeout_ns : int, default = 5 minutes
        Silence after which `expire` abandons a pending exchange.
    max_pending : int, default = 500000
        Exchanges held open before the least recently started is dropped. A dropped exchange's outcome arrives
        unpaired, which over-reports the bypass signal rather than hiding it.
    """

    def __init__(self, timeout_ns: int = DEFAULT_TIMEOUT_NS, max_pending: int = DEFAULT_MAX_PENDING):
        if (timeout_ns <= 0):
            raise ValueError(f"timeout_ns must be positive, received {timeout_ns}")

        if (max_pending <= 0):
            raise ValueError(f"max_pending must be positive, received {max_pending}")

        self._timeout_ns = timeout_ns
        self._max_pending = max_pending

        self._pending: collections.OrderedDict[str, _Pending] = collections.OrderedDict()
        self._expired_through_ns: typing.Optional[int] = None

    @property
    def pending_count(self) -> int:
        """Exchanges currently open."""
        return len(self._pending)

    def begin(self, key: str, event_time_ns: int) -> None:
        """
        Record the start of an exchange, or another attempt at one already open.

        Parameters
        ----------
        key : str
            What the exchange belongs to.
        event_time_ns : int
            When it started, in nanoseconds since the epoch.
        """
        state = self._pending.get(key)

        if (state is None):
            self._pending[key] = _Pending(started_ns=event_time_ns, attempts=1)
        else:
            # The clock restarts from the latest attempt, but the attempt count carries, so a success after three
            # retries is not mistaken for a first-time one.
            state.started_ns = max(state.started_ns, event_time_ns)
            state.attempts += 1

        self._pending.move_to_end(key)

        while (len(self._pending) > self._max_pending):
            self._pending.popitem(last=False)

    def complete(self, key: str, event_time_ns: int, outcome: typing.Optional[str] = None) -> SessionTiming:
        """
        Record an outcome and return what it took to get there.

        Parameters
        ----------
        key : str
            What the exchange belongs to.
        event_time_ns : int
            When the outcome arrived.
        outcome : str, optional
            Whatever the source reported.

        Returns
        -------
        `SessionTiming`
        """
        state = self._pending.pop(key, None)

        if (state is None):
            # Authorization with no authentication in front of it. This is the signal, not a gap in the data.
            return SessionTiming(key=key,
                                 elapsed_ns=None,
                                 outcome=outcome,
                                 attempts=0,
                                 unpaired=True,
                                 out_of_order=False)

        if (event_time_ns < state.started_ns):
            return SessionTiming(key=key,
                                 elapsed_ns=None,
                                 outcome=outcome,
                                 attempts=state.attempts,
                                 unpaired=False,
                                 out_of_order=True)

        return SessionTiming(key=key,
                             elapsed_ns=event_time_ns - state.started_ns,
                             outcome=outcome,
                             attempts=state.attempts,
                             unpaired=False,
                             out_of_order=False)

    def expire(self, now_ns: int) -> list[str]:
        """
        Abandon exchanges that never resolved.

        Parameters
        ----------
        now_ns : int
            Current event time.

        Returns
        -------
        list of str
            The keys abandoned, oldest first. An exchange that never resolved is its own signal, so the caller is
            told which rather than the state simply disappearing.
        """
        horizon = now_ns - self._timeout_ns

        # Called once per row on that row's own event time, so a per-row scan of every pending exchange is
        # quadratic in batch size. A horizon that has not advanced cannot abandon anything the previous call did
        # not already abandon, since every exchange begun since was stamped at or after it, so the skip is exact.
        if (self._expired_through_ns is not None and horizon <= self._expired_through_ns):
            return []

        self._expired_through_ns = horizon
        stale = [key for (key, state) in self._pending.items() if state.started_ns < horizon]

        for key in stale:
            del self._pending[key]

        return stale
