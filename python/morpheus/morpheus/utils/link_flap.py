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
Link flap counting, including the flaps that happen between polls.

A port that changes operational state repeatedly is unstable, and instability at layer 1 is what precedes a layer 2
loop: each transition re-triggers spanning tree, and enough of them in a short window is the shape of a fault
about to become an outage. The feature is therefore the count of transitions per interval, not the current state.

Comparing `oper_status` between consecutive polls only counts transitions that happen to be visible at the poll
boundary. That is the wrong count, and wrong in the worst direction: a port that drops and recovers inside one
sixty-second polling gap looks perfectly stable, which is exactly the port that is flapping. `last_change_time`
fixes it. Devices maintain it as the moment the interface last changed state, so if it advanced between two polls
the interface transitioned even when both polls saw the same status, and it transitioned an even number of times,
so at least twice.

Every count this produces is therefore a **lower bound**. `last_change_time` records only the most recent
transition, so a port that flapped nine times between polls still reports two. That is a floor on the truth rather
than an estimate of it, which is the right way round for a signal whose purpose is to notice instability at all:
under-counting a flapping port still leaves it flagged, whereas an interpolated guess would put a number that was
never measured in front of an analyst.

The collector is expected to normalize `last_change_time` to an absolute time before it gets here. Devices report
it relative to their own uptime, which restarts on reboot; a value that goes backwards is read as exactly that, and
a reboot is itself a transition, since the link went down with the device and came back with it.
"""

import collections
import dataclasses
import typing

NS_PER_SECOND = 10**9

DEFAULT_WINDOW_NS = 3600 * NS_PER_SECOND
"""Interval the windowed flap count covers. An hour is long enough to accumulate a pattern and short enough that
the count still describes the port's present behavior.
"""

DEFAULT_MAX_EVENTS = 4096
"""Flap records retained per entity, whatever the window implies. A bound on memory for a port flapping hard."""

DEFAULT_MAX_ENTITIES = 100_000
"""Entities tracked before the least recently seen is forgotten."""

REBOOT_FLAPS = 2
"""Transitions attributed to a device restart: the link went down with the device and came back up with it."""


@dataclasses.dataclass(frozen=True)
class FlapResult:
    """
    The outcome of observing one sample of a port's operational state.

    Attributes
    ----------
    flaps : int or None
        Lower bound on state transitions since the previous sample. `None` for the first sample of an entity and
        for an out-of-order one, where the count is not computable rather than zero.
    flaps_in_window : int or None
        Lower bound on transitions over the trailing window, including this sample's. `None` where `flaps` is.
    last_change_advanced : bool
        The device reported a state change since the previous sample. When this is set and the status is unchanged,
        the transitions were invisible to polling and were counted anyway.
    device_reset : bool
        `last_change_time` went backwards, which happens when the device restarted and its clock began again.
    last_change_inconsistent : bool
        The status changed while `last_change_time` did not, which the device should never report. The transition
        is counted and the disagreement is flagged, because a device that does not maintain the field is one whose
        sub-poll flaps cannot be seen at all.
    out_of_order : bool
        The sample's event time was not after the previous sample's. State is left untouched.
    """

    flaps: typing.Optional[int]
    flaps_in_window: typing.Optional[int]
    last_change_advanced: bool
    device_reset: bool
    last_change_inconsistent: bool
    out_of_order: bool


@dataclasses.dataclass
class _EntityState:
    event_time_ns: int
    oper_status: typing.Optional[str]
    last_change_ns: typing.Optional[int]


class LinkFlapTracker:
    """
    Per-entity link stability state.

    The tracker holds one sample per entity plus a trailing window of flap records, so its memory is bounded by the
    entity count rather than by the stream. Results depend only on the sequence of samples it has been shown, so
    replaying a stream reproduces the counts.

    Parameters
    ----------
    window_ns : int, default = 1 hour
        Interval, in nanoseconds of event time, the windowed count covers.
    max_events : int, default = 4096
        Flap records retained per entity regardless of the window.
    max_entities : int, default = 100000
        Entities retained before the least recently seen is dropped. A dropped entity's next sample is treated as
        its first, yielding no count rather than a wrong one.
    """

    def __init__(self,
                 window_ns: int = DEFAULT_WINDOW_NS,
                 max_events: int = DEFAULT_MAX_EVENTS,
                 max_entities: int = DEFAULT_MAX_ENTITIES):
        if (window_ns <= 0):
            raise ValueError(f"window_ns must be positive, received {window_ns}")

        if (max_events <= 0):
            raise ValueError(f"max_events must be positive, received {max_events}")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._window_ns = window_ns
        self._max_events = max_events
        self._max_entities = max_entities

        self._states: collections.OrderedDict[str, _EntityState] = collections.OrderedDict()
        self._events: dict[str, collections.deque] = {}

    @property
    def tracked_entities(self) -> int:
        """Entities currently holding state."""
        return len(self._states)

    def observe(self,
                entity_key: str,
                event_time_ns: int,
                oper_status: typing.Optional[str],
                last_change_ns: typing.Optional[int] = None) -> FlapResult:
        """
        Record one sample and return the transitions it implies.

        Parameters
        ----------
        entity_key : str
            The port the sample belongs to, typically `site_id:device_id:port_id`.
        event_time_ns : int
            When the sample was taken, in nanoseconds since the epoch. Event time, never ingest time.
        oper_status : str, optional
            The interface's operational state, for example `"up"` or `"down"`. Only compared for equality, so any
            stable rendering works as long as one collector does not switch between renderings mid-stream.
        last_change_ns : int, optional
            When the interface last changed state, as an absolute time. Supplying it is what lets a flap that
            began and ended between two polls be counted; without it, only transitions visible at a poll boundary
            are seen.

        Returns
        -------
        `FlapResult`
        """
        previous = self._states.get(entity_key)
        current = _EntityState(event_time_ns=event_time_ns, oper_status=oper_status, last_change_ns=last_change_ns)

        if (previous is None):
            self._remember(entity_key, current)

            return FlapResult(flaps=None,
                              flaps_in_window=None,
                              last_change_advanced=False,
                              device_reset=False,
                              last_change_inconsistent=False,
                              out_of_order=False)

        if (event_time_ns <= previous.event_time_ns):
            # Accepting this would let a late sample rewrite the state a later one was already compared against. An
            # equal timestamp is rejected on purpose: operational state is polled per port, and two states for one
            # port at one instant is a duplicate poll, not a transition.
            return FlapResult(flaps=None,
                              flaps_in_window=None,
                              last_change_advanced=False,
                              device_reset=False,
                              last_change_inconsistent=False,
                              out_of_order=True)

        (flaps, advanced, reset, inconsistent) = self._count(previous, current)

        self._remember(entity_key, current)

        history = self._events.setdefault(entity_key, collections.deque())

        if (flaps > 0):
            history.append((event_time_ns, flaps))

            while (len(history) > self._max_events):
                history.popleft()

        horizon = event_time_ns - self._window_ns

        while (len(history) > 0 and history[0][0] <= horizon):
            history.popleft()

        return FlapResult(flaps=flaps,
                          flaps_in_window=sum(count for (_, count) in history),
                          last_change_advanced=advanced,
                          device_reset=reset,
                          last_change_inconsistent=inconsistent,
                          out_of_order=False)

    def _count(self, previous: _EntityState, current: _EntityState) -> tuple[int, bool, bool, bool]:
        """Return the transition floor plus what the two samples disagreed about."""
        status_changed = (previous.oper_status is not None and current.oper_status is not None
                          and previous.oper_status != current.oper_status)

        if (current.last_change_ns is None or previous.last_change_ns is None):
            # Only what polling happened to see. A flap contained inside the gap leaves no trace here.
            return (int(status_changed), False, False, False)

        if (current.last_change_ns < previous.last_change_ns):
            # The device's clock restarted, so the device restarted, so the link went down and came back.
            return (REBOOT_FLAPS, True, True, False)

        if (current.last_change_ns == previous.last_change_ns):
            if (status_changed):
                # The device says nothing changed while its own status says otherwise. Trust the status, and note
                # that this device cannot be relied on to reveal the flaps that happen between polls.
                return (1, False, False, True)

            return (0, False, False, False)

        # The interface changed state. Ending on a different status means an odd number of transitions, so at least
        # one; ending on the same status means an even number, so at least two.
        return (1 if status_changed else 2, True, False, False)

    def _remember(self, entity_key: str, state: _EntityState) -> None:
        self._states[entity_key] = state
        self._states.move_to_end(entity_key)

        while (len(self._states) > self._max_entities):
            (dropped, _) = self._states.popitem(last=False)
            self._events.pop(dropped, None)
