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
Per-entity counts and sums over a trailing window of event time.

{py:mod}`~morpheus.utils.distinct_window` answers how many different values an entity has shown in the last hour.
This answers the two plainer questions asked beside it: how many observations, and what they add up to -- a
service's connections and bytes in the hour, which `TC7ServiceStage` writes next to the count of its clients.

The conventions are the distinct window's, so the two can be read side by side. The window is trailing and the
current observation is counted inside it. An observation earlier than the entity's previous one is refused and
leaves the window untouched, so a result depends on the stream and not on how it was delivered; one at the same
instant is accepted. Each window is bounded by a sample cap as well as by its duration, and the entities by a
count, the least recently seen going first.

When the sample cap evicts an observation the window would still have held, the result says so: the count and
the sum are then floors rather than the figures, and a reader should be told as it is for a distinct count.
"""

import collections
import dataclasses
import typing

from morpheus.utils.distinct_window import DEFAULT_MAX_ENTITIES
from morpheus.utils.distinct_window import DEFAULT_MAX_SAMPLES
from morpheus.utils.distinct_window import DEFAULT_WINDOW_NS
from morpheus.utils.tracker_state import StateMixin


@dataclasses.dataclass(frozen=True)
class SumWindowResult:
    """
    The outcome of observing one amount.

    Attributes
    ----------
    count : int
        Observations in the window, counting this one. A lower bound when `saturated` is set.
    total : float
        Their amounts added up, counting this one. A lower bound when `saturated` is set and amounts are not
        negative.
    saturated : bool
        The sample cap is binding: this observation displaced one the window's duration would have kept.
    out_of_order : bool
        The observation's event time was before the entity's previous one. State is left untouched, and `count`
        and `total` are the window as it stood.
    """

    count: int
    total: float
    saturated: bool
    out_of_order: bool


@dataclasses.dataclass
class _Window:
    entries: collections.deque = dataclasses.field(default_factory=collections.deque)
    total: float = 0.0
    last_ns: typing.Optional[int] = None


class SumWindowTracker(StateMixin):
    """
    Per-entity observation counts and amount sums over a trailing window.

    Results depend only on the sequence of observations the tracker has been shown, so replaying a stream
    reproduces them.

    Parameters
    ----------
    window_ns : int, default = 1 hour
        Trailing window, in nanoseconds of event time.
    max_samples : int, default = 4096
        Observations retained per entity regardless of the window. When this binds, the result is marked saturated.
    max_entities : int, default = 500000
        Entities retained before the least recently seen is dropped. A dropped entity starts from an empty window.
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
        self._windows: collections.OrderedDict[str, _Window] = collections.OrderedDict()

    @property
    def tracked_entities(self) -> int:
        """Entities currently holding a window."""
        return len(self._windows)

    def observe(self, entity_key: str, event_time_ns: int, value: typing.Optional[float]) -> SumWindowResult:
        """
        Record one observation and return what the window now holds.

        Parameters
        ----------
        entity_key : str
            What is being counted per: a service.
        event_time_ns : int
            When the observation was made, in nanoseconds since the epoch. Event time, never ingest time.
        value : float, optional
            The amount to add. `None` counts as an observation of nothing: it is counted and adds zero.

        Returns
        -------
        `SumWindowResult`
        """
        window = self._windows.get(entity_key)

        if (window is None):
            window = _Window()
            self._windows[entity_key] = window

            while (len(self._windows) > self._max_entities):
                self._windows.popitem(last=False)
        else:
            self._windows.move_to_end(entity_key)

        if (window.last_ns is not None and event_time_ns < window.last_ns):
            return SumWindowResult(count=len(window.entries), total=window.total, saturated=False, out_of_order=True)

        horizon = event_time_ns - self._window_ns
        saturated = False

        while (len(window.entries) > 0
               and (window.entries[0][0] <= horizon or len(window.entries) >= self._max_samples)):
            # Displaced by the cap rather than by its age: the window would still have held it.
            saturated = saturated or window.entries[0][0] > horizon
            window.total -= window.entries.popleft()[1]

        amount = 0.0 if value is None else float(value)
        window.entries.append((event_time_ns, amount))
        window.total += amount
        window.last_ns = event_time_ns

        return SumWindowResult(count=len(window.entries), total=window.total, saturated=saturated, out_of_order=False)
