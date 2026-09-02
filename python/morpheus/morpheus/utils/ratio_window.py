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
The share of an entity's events in a trailing window that are of one kind.

Some behavioral signals are not a count but a proportion, because the raw count tracks how busy something is
rather than how odd it is. ARP is the TC-2 case: a host on a chatty segment sends far more ARP than one on a quiet
segment, so a count of gratuitous replies mostly measures traffic volume. The share of that host's ARP which is
gratuitous does not, and it is what separates a device announcing itself after a failover from one flooding
announcements to poison a neighbor's cache.

A ratio over a small denominator is noise dressed as a measurement: one event out of one is 1.0, which reads as
total saturation and means almost nothing. `min_denominator` is therefore a floor below which no ratio is
published at all, on the same reasoning as the warm-up in `morpheus.utils.optical_baseline`, and for the same
reason: a reference that nothing supports is worse than no reference.

The current event is counted inside its own window, so a proportion moves on the event that changes it.
"""

import collections
import dataclasses
import typing

NS_PER_SECOND = 10**9

DEFAULT_WINDOW_NS = 300 * NS_PER_SECOND
"""Trailing window the proportion is taken over. Five minutes matches the reconciliation cadence the TC-2
telemetry class specifies, and is short enough that a burst is not diluted by an hour of normal traffic.
"""

DEFAULT_MIN_DENOMINATOR = 10
"""Events required in the window before a ratio is published."""

DEFAULT_MAX_SAMPLES = 4096
"""Events retained per entity, whatever the window implies. A flood is both the thing being measured and the thing
that would exhaust memory, so the cap turns the proportion into an estimate over the retained tail and says so.
"""

DEFAULT_MAX_ENTITIES = 500_000
"""Entities tracked before the least recently seen is forgotten."""


@dataclasses.dataclass(frozen=True)
class RatioResult:
    """
    The outcome of observing one event.

    Attributes
    ----------
    numerator : int
        Events of the kind being measured in the window, counting this one if it is of that kind.
    denominator : int
        Events in the window, counting this one.
    ratio : float or None
        `numerator / denominator`, or `None` while the window holds fewer than `min_denominator` events, where a
        proportion would be noise rather than a measurement.
    saturated : bool
        The sample cap is binding, so the proportion describes the retained tail rather than the whole window.
    out_of_order : bool
        The event's time was not after the previous one's. State is left untouched.
    """

    numerator: int
    denominator: int
    ratio: typing.Optional[float]
    saturated: bool
    out_of_order: bool


@dataclasses.dataclass
class _EntityWindow:
    history: collections.deque = dataclasses.field(default_factory=collections.deque)
    numerator: int = 0
    last_seen_ns: int = 0
    started: bool = False


class RatioWindowTracker:
    """
    Per-entity proportion of one kind of event over a trailing window.

    The tracker holds one window per entity, bounded by both the window duration and a sample cap, so its memory is
    bounded by the entity count rather than by the stream. Results depend only on the sequence of events it has
    been shown, so replaying a stream reproduces the proportions.

    Parameters
    ----------
    window_ns : int, default = 5 minutes
        Trailing window, in nanoseconds of event time.
    min_denominator : int, default = 10
        Events required in the window before a ratio is published. Below this the ratio is `None`.
    max_samples : int, default = 4096
        Events retained per entity regardless of the window. When this binds the result is marked saturated.
    max_entities : int, default = 500000
        Entities retained before the least recently seen is dropped.
    """

    def __init__(self,
                 window_ns: int = DEFAULT_WINDOW_NS,
                 min_denominator: int = DEFAULT_MIN_DENOMINATOR,
                 max_samples: int = DEFAULT_MAX_SAMPLES,
                 max_entities: int = DEFAULT_MAX_ENTITIES):
        if (window_ns <= 0):
            raise ValueError(f"window_ns must be positive, received {window_ns}")

        if (min_denominator < 1):
            raise ValueError(f"min_denominator must be at least 1, received {min_denominator}")

        if (max_samples < min_denominator):
            raise ValueError(f"max_samples ({max_samples}) must be at least min_denominator ({min_denominator}), "
                             "or a ratio could never be published")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._window_ns = window_ns
        self._min_denominator = min_denominator
        self._max_samples = max_samples
        self._max_entities = max_entities

        self._windows: collections.OrderedDict[str, _EntityWindow] = collections.OrderedDict()

    @property
    def tracked_entities(self) -> int:
        """Entities currently holding a window."""
        return len(self._windows)

    def observe(self, entity_key: str, event_time_ns: int, counts_toward_numerator: bool) -> RatioResult:
        """
        Record one event and return the proportion the window now shows.

        Parameters
        ----------
        entity_key : str
            What the proportion is taken per, typically the sender of the events.
        event_time_ns : int
            When the event happened, in nanoseconds since the epoch. Event time, never ingest time.
        counts_toward_numerator : bool
            Whether this event is of the kind being measured. Every event counts toward the denominator.

        Returns
        -------
        `RatioResult`
        """
        window = self._windows.get(entity_key)

        if (window is None):
            window = _EntityWindow()
            self._windows[entity_key] = window

        self._windows.move_to_end(entity_key)

        if (window.started and event_time_ns <= window.last_seen_ns):
            # A late event would be expired against the wrong horizon and would change what a later ratio sees.
            return RatioResult(numerator=window.numerator,
                               denominator=len(window.history),
                               ratio=None,
                               saturated=False,
                               out_of_order=True)

        # Expire against this event's horizon first, so the proportion describes the window as it stands now.
        horizon = event_time_ns - self._window_ns

        while (len(window.history) > 0 and window.history[0][0] <= horizon):
            self._drop_oldest(window)

        window.history.append((event_time_ns, counts_toward_numerator))
        window.numerator += int(counts_toward_numerator)

        saturated = len(window.history) > self._max_samples

        while (len(window.history) > self._max_samples):
            self._drop_oldest(window)

        window.last_seen_ns = event_time_ns
        window.started = True
        self._evict()

        denominator = len(window.history)
        # Below the floor a proportion is noise: one event out of one reads as 1.0 and means nothing.
        ratio = (window.numerator / denominator) if denominator >= self._min_denominator else None

        return RatioResult(numerator=window.numerator,
                           denominator=denominator,
                           ratio=ratio,
                           saturated=saturated,
                           out_of_order=False)

    @staticmethod
    def _drop_oldest(window: _EntityWindow) -> None:
        (_, was_numerator) = window.history.popleft()
        window.numerator -= int(was_numerator)

    def _evict(self) -> None:
        while (len(self._windows) > self._max_entities):
            self._windows.popitem(last=False)
