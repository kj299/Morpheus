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
Interface counter deltas, with wrap and reset told apart.

Network devices report error and discard counters as monotonically increasing totals. Those totals are useless as
features: they grow without bound, they wrap at 2^32 or 2^64, and they restart at zero when the device reboots. A
rule written against a raw counter fires on the device's age, not on its behavior.

The difference between two samples is the useful quantity, and computing it correctly means telling three cases
apart when a counter appears to decrease:

- a **wrap**, where the counter passed its ceiling and the true delta is `current + 2**bits - previous`;
- a **reset**, where the device rebooted and the true delta is `current`, accumulated over the uptime rather than
  over the full sampling interval;
- an **out-of-order sample**, where nothing is wrong with the counter and the delta is simply not computable.

Only the device's uptime distinguishes a wrap from a reset. Where it is available this module uses it; where it is
not, a decrease is reported as a reset with no delta rather than guessed at, because guessing in either direction
fabricates a number that looks like a measurement.

Every result carries the interval it covers, so a consumer computing a rate divides by what actually elapsed
rather than by the nominal polling period.
"""

import collections
import dataclasses
import typing

NS_PER_SECOND = 10**9

DEFAULT_COUNTER_BITS = 64
"""Counter width assumed when none is given. SNMP Counter64 is the common case; Counter32 must be declared."""

DEFAULT_MAX_ENTITIES = 100_000
"""Entities tracked before the least recently seen are forgotten. Layer 1 port counts sit far below this."""


@dataclasses.dataclass(frozen=True)
class DeltaResult:
    """
    The outcome of observing one sample.

    Attributes
    ----------
    deltas : dict
        Per-counter change since the previous sample. A value is `None` when no delta is computable: the first
        sample for an entity, an out-of-order sample, or a decrease that could not be resolved.
    interval_ns : int or None
        Time the deltas cover. This is the gap since the previous sample, except after a reset, where it is capped
        at the device's uptime, since the counter only began accumulating when the device came up.
    counter_reset : bool
        The device restarted its counters between samples.
    counter_wrapped : bool
        At least one counter passed its ceiling and was corrected for.
    out_of_order : bool
        The sample's event time was not after the previous sample's. State is left untouched.
    """

    deltas: dict[str, typing.Optional[int]]
    interval_ns: typing.Optional[int]
    counter_reset: bool
    counter_wrapped: bool
    out_of_order: bool


@dataclasses.dataclass
class _EntityState:
    event_time_ns: int
    uptime_ns: typing.Optional[int]
    counters: dict[str, int]


class CounterTracker:
    """
    Per-entity counter state, turning raw totals into deltas.

    The tracker holds one sample per entity and nothing else, so its memory is bounded by the entity count rather
    than by the stream. Results depend only on the sequence of samples it has been shown, which is what makes a
    replay of the same stream produce the same deltas.

    Parameters
    ----------
    counter_names : list of str
        Counters to track.
    counter_bits : int or dict, default = 64
        Counter width, either one value for all counters or a per-counter mapping. SNMP `Counter32` values must be
        declared as 32 or their wraps are silently mistaken for resets.
    max_entities : int, default = 100000
        Entities retained before the least recently seen is dropped. A dropped entity's next sample is treated as
        its first, yielding no deltas rather than a wrong one.
    """

    def __init__(self,
                 counter_names: typing.Sequence[str],
                 counter_bits: typing.Union[int, dict[str, int]] = DEFAULT_COUNTER_BITS,
                 max_entities: int = DEFAULT_MAX_ENTITIES):
        if (len(counter_names) == 0):
            raise ValueError("At least one counter name is required")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._counter_names = list(counter_names)

        if (isinstance(counter_bits, int)):
            widths = {name: counter_bits for name in self._counter_names}
        else:
            widths = {name: counter_bits.get(name, DEFAULT_COUNTER_BITS) for name in self._counter_names}

        for (name, bits) in widths.items():
            if (bits <= 0):
                raise ValueError(f"Counter width for {name!r} must be positive, received {bits}")

        self._ceilings = {name: 1 << bits for (name, bits) in widths.items()}
        self._max_entities = max_entities
        self._states: collections.OrderedDict[str, _EntityState] = collections.OrderedDict()

    @property
    def counter_names(self) -> list[str]:
        """Counters this tracker computes deltas for."""
        return list(self._counter_names)

    @property
    def tracked_entities(self) -> int:
        """Entities currently holding state."""
        return len(self._states)

    def _none_result(self, **flags) -> DeltaResult:
        return DeltaResult(deltas={name: None
                                   for name in self._counter_names},
                           interval_ns=None,
                           counter_reset=flags.get("counter_reset", False),
                           counter_wrapped=False,
                           out_of_order=flags.get("out_of_order", False))

    def observe(self,
                entity_key: str,
                event_time_ns: int,
                counters: dict[str, typing.Optional[int]],
                uptime_ns: typing.Optional[int] = None) -> DeltaResult:
        """
        Record one sample and return the deltas it implies.

        Parameters
        ----------
        entity_key : str
            The entity the counters belong to, typically `site_id:device_id:port_id`.
        event_time_ns : int
            When the sample was taken, in nanoseconds since the epoch. Event time, never ingest time.
        counters : dict
            Raw counter totals. A counter missing or `None` yields a `None` delta for that counter alone.
        uptime_ns : int, optional
            How long the device had been running when sampled. Supplying it is what lets a wrap be told from a
            reboot; without it, any decrease is reported as a reset with no delta.

        Returns
        -------
        `DeltaResult`
        """
        previous = self._states.get(entity_key)

        if (previous is not None):
            self._states.move_to_end(entity_key)

        current = _EntityState(
            event_time_ns=event_time_ns,
            uptime_ns=uptime_ns,
            counters={name: counters.get(name)
                      for name in self._counter_names if counters.get(name) is not None})

        if (previous is None):
            self._remember(entity_key, current)

            return self._none_result()

        if (event_time_ns <= previous.event_time_ns):
            # Leave state on the later sample: accepting this one would make the next delta span backwards.
            return self._none_result(out_of_order=True)

        interval_ns = event_time_ns - previous.event_time_ns
        reset = self._is_reset(previous, current, interval_ns)

        deltas: dict[str, typing.Optional[int]] = {}
        wrapped = False

        for name in self._counter_names:
            (delta, counter_wrapped) = self._delta_for(name, previous, current, reset, uptime_ns)
            deltas[name] = delta
            wrapped = wrapped or counter_wrapped

        if (reset and uptime_ns is not None):
            # The counter only accumulated since the device came up, so that, not the sampling gap, is the window
            # the deltas cover.
            interval_ns = min(interval_ns, uptime_ns)

        self._remember(entity_key, current)

        return DeltaResult(deltas=deltas,
                           interval_ns=interval_ns,
                           counter_reset=reset,
                           counter_wrapped=wrapped,
                           out_of_order=False)

    def _is_reset(self, previous: _EntityState, current: _EntityState, interval_ns: int) -> bool:
        """Whether the device restarted between the two samples."""
        if (current.uptime_ns is None or previous.uptime_ns is None):
            # Without uptime a decrease is unresolvable; `_delta_for` reports it as a reset with no delta.
            return any(
                current.counters.get(name) is not None and previous.counters.get(name) is not None
                and current.counters[name] < previous.counters[name] for name in self._counter_names)

        # An uptime that went backwards is unambiguous. An uptime that advanced by less than the sampling gap means
        # the device came up during it, which a plain comparison of uptimes would miss.
        return current.uptime_ns < previous.uptime_ns or current.uptime_ns < interval_ns

    def _delta_for(self,
                   name: str,
                   previous: _EntityState,
                   current: _EntityState,
                   reset: bool,
                   uptime_ns: typing.Optional[int]) -> tuple[typing.Optional[int], bool]:
        current_value = current.counters.get(name)
        previous_value = previous.counters.get(name)

        if (current_value is None or previous_value is None):
            return (None, False)

        if (reset):
            # Counting restarted at zero, so everything present now accumulated after the restart.
            return (current_value if uptime_ns is not None else None, False)

        if (current_value >= previous_value):
            return (current_value - previous_value, False)

        return (current_value + self._ceilings[name] - previous_value, True)

    def _remember(self, entity_key: str, state: _EntityState) -> None:
        self._states[entity_key] = state
        self._states.move_to_end(entity_key)

        while (len(self._states) > self._max_entities):
            self._states.popitem(last=False)
