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
Distinct values per entity over a trailing window.

Three of the TC-2 behavioral features are the same question asked with the entity and the value swapped around, and
each is diagnostic in a different direction:

- **Distinct MACs per port.** An access port serves one device, or a handful behind an IP phone. A count that
  climbs is an unauthorized hub or switch, or a MAC flood against the forwarding table.
- **Distinct ports per MAC.** One MAC belongs to one interface. Seeing it on several ports at once is spoofing;
  seeing it move between them over time is a device being carried around, which is benign in an office and much
  less so in a datacentre.
- **Distinct OUIs per VLAN.** An OUI that a VLAN has not carried before is an unmanaged device class appearing on
  a segment that was supposed to be homogeneous.

The window is trailing and the current sample is counted inside it. "How many distinct MACs has this port seen in
the last hour" plainly includes the one that just arrived, so a threshold trips on the sample that crosses it
rather than on the one after. That is the opposite of `morpheus.utils.optical_baseline`, where the current reading
is deliberately excluded so that it cannot anchor the reference it is being judged against. The two conventions
look alike and mean different things, which is why each is written out where it is used rather than shared behind
a flag that would have to name the difference anyway.

Counting distinct values is kept to constant work per sample by carrying an occurrence count alongside the window
and decrementing it on eviction, rather than scanning the window on every row. That matters here: unlike the layer
1 windows, which hold one reading per port per poll, a busy trunk port can carry thousands of MACs.
"""

import collections
import dataclasses
import typing

NS_PER_SECOND = 10**9

DEFAULT_WINDOW_NS = 3600 * NS_PER_SECOND
"""Trailing window distinct values are counted over."""

DEFAULT_MAX_SAMPLES = 4096
"""Observations retained per entity, whatever the window implies.

A MAC flood is precisely the condition this feature exists to notice, and precisely the condition that would let an
unbounded window exhaust memory. The cap keeps the count a lower bound instead, and a saturated entity is reported
as such, so the floor is never mistaken for the true figure.
"""

DEFAULT_MAX_ENTITIES = 500_000
"""Entities tracked before the least recently seen is forgotten. Sized for the TC-2 telemetry class, where keying
by MAC address runs to low hundreds of thousands, rather than for layer 1's port counts.
"""


@dataclasses.dataclass(frozen=True)
class DistinctWindowResult:
    """
    The outcome of observing one value.

    Attributes
    ----------
    distinct : int
        Distinct values in the window, counting this one. A lower bound when `saturated` is set.
    total : int
        Observations in the window, counting this one.
    first_in_window : bool
        This value was not in the window before this sample. Window-scoped novelty, which is a weaker and more
        perishable claim than never having been seen at all; `morpheus.utils.value_novelty` answers the latter.
    saturated : bool
        The sample cap is binding, so observations are being evicted before the window would have expired them and
        `distinct` is a floor rather than the figure.
    out_of_order : bool
        The sample's event time was not after the previous one's. State is left untouched.
    """

    distinct: int
    total: int
    first_in_window: bool
    saturated: bool
    out_of_order: bool


@dataclasses.dataclass
class _EntityWindow:
    history: collections.deque = dataclasses.field(default_factory=collections.deque)
    occurrences: collections.Counter = dataclasses.field(default_factory=collections.Counter)
    last_seen_ns: int = 0
    started: bool = False


class DistinctWindowTracker:
    """
    Per-entity distinct-value counting over a trailing window.

    The tracker holds one window per entity, bounded by both the window duration and a sample cap, so its memory is
    bounded by the entity count rather than by the stream. Results depend only on the sequence of samples it has
    been shown, so replaying a stream reproduces the counts.

    Parameters
    ----------
    window_ns : int, default = 1 hour
        Trailing window, in nanoseconds of event time.
    max_samples : int, default = 4096
        Observations retained per entity regardless of the window. When this binds, the result is marked saturated
        and the distinct count is a lower bound.
    max_entities : int, default = 500000
        Entities retained before the least recently seen is dropped. A dropped entity starts from an empty window,
        so its next count is a fresh floor rather than a stale figure.
    """

    def __init__(self,
                 window_ns: int = DEFAULT_WINDOW_NS,
                 max_samples: int = DEFAULT_MAX_SAMPLES,
                 max_entities: int = DEFAULT_MAX_ENTITIES):
        if (window_ns <= 0):
            raise ValueError(f"window_ns must be positive, received {window_ns}")

        if (max_samples <= 0):
            raise ValueError(f"max_samples must be positive, received {max_samples}")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._window_ns = window_ns
        self._max_samples = max_samples
        self._max_entities = max_entities

        self._windows: collections.OrderedDict[str, _EntityWindow] = collections.OrderedDict()

    @property
    def tracked_entities(self) -> int:
        """Entities currently holding a window."""
        return len(self._windows)

    def observe(self, entity_key: str, event_time_ns: int, value: typing.Any) -> DistinctWindowResult:
        """
        Record one observation and return what the window now holds.

        Parameters
        ----------
        entity_key : str
            What the values are being counted per: a port, a MAC address, a VLAN.
        event_time_ns : int
            When the value was observed, in nanoseconds since the epoch. Event time, never ingest time: a window
            measured in arrival order describes the collector's scheduling rather than the segment.
        value : any
            The value to count. Must be hashable. `None` is counted like any other value, since "a port reporting
            no OUI" is a state rather than an absence of data.

        Returns
        -------
        `DistinctWindowResult`
        """
        window = self._windows.get(entity_key)

        if (window is None):
            window = _EntityWindow()
            self._windows[entity_key] = window

        self._windows.move_to_end(entity_key)

        if (window.started and event_time_ns <= window.last_seen_ns):
            # A late sample would be evicted against the wrong horizon and would change what a later count sees,
            # making the result depend on delivery order rather than on the segment.
            return DistinctWindowResult(distinct=len(window.occurrences),
                                        total=len(window.history),
                                        first_in_window=False,
                                        saturated=False,
                                        out_of_order=True)

        # Expire against this sample's horizon before anything else. Membership has to be judged against the window
        # as it stands now, or a value that left an hour ago would still be found and would not read as new.
        horizon = event_time_ns - self._window_ns

        while (len(window.history) > 0 and window.history[0][0] <= horizon):
            self._drop_oldest(window)

        # The current sample is then counted. "Distinct values in the last hour" includes the one that just
        # arrived, so a threshold trips on the sample that crosses it rather than on the one after. That is what
        # separates this from `optical_baseline`, which excludes the current reading so it cannot anchor the
        # baseline it is being judged against.
        first_in_window = window.occurrences[value] == 0
        window.history.append((event_time_ns, value))
        window.occurrences[value] += 1

        saturated = len(window.history) > self._max_samples

        while (len(window.history) > self._max_samples):
            self._drop_oldest(window)

        window.last_seen_ns = event_time_ns
        window.started = True
        self._evict()

        return DistinctWindowResult(distinct=len(window.occurrences),
                                    total=len(window.history),
                                    first_in_window=first_in_window,
                                    saturated=saturated,
                                    out_of_order=False)

    @staticmethod
    def _drop_oldest(window: _EntityWindow) -> None:
        (_, dropped) = window.history.popleft()
        window.occurrences[dropped] -= 1

        if (window.occurrences[dropped] == 0):
            # Removed rather than left at zero, so that `len(occurrences)` is the distinct count.
            del window.occurrences[dropped]

    def _evict(self) -> None:
        while (len(self._windows) > self._max_entities):
            self._windows.popitem(last=False)
