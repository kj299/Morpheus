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
Identifier change detection with no period boundary.

`DistinctIncrementColumn` counts distinct values per group per period, and the period is the problem. The count
resets at each boundary, so a change from the last sample before one to the first sample after it is a single
distinct value on each side and reads as no change at all. Lengthening the period makes boundaries rarer without
removing them, which narrows the blind spot rather than closing it.

Carrying the previous value per entity closes it outright. A change is detected the moment the new value arrives,
whatever the calendar says, because nothing here is bucketed by time. This is the same shape as
`morpheus.utils.counter_delta`: hold one sample per entity, compare, and report what the comparison implies.

Two questions get separate answers, because they are separately actionable:

- **Did it change from the previous sample?** A transceiver serial that differs from the one seen a minute ago is a
  substitution that just happened. This is the alerting signal.
- **Has this entity ever reported this value before?** An optic that has never been in this cage is a different
  situation from one being rotated back in after maintenance. Both are changes; only the first is unexplained by
  the estate's own history.

A null is a value, not missing data. A port with an empty cage genuinely reports no serial, and going from no optic
to an optic is a change worth seeing, so `None` is tracked like any other value rather than skipped.
"""

import collections
import dataclasses
import typing

DEFAULT_MAX_VALUES = 64
"""Distinct values recalled per field per entity. A port that has legitimately held more optics than this is
already the anomaly the feature exists to surface.
"""

DEFAULT_MAX_ENTITIES = 100_000
"""Entities tracked before the least recently seen is forgotten."""


@dataclasses.dataclass(frozen=True)
class NoveltyResult:
    """
    The outcome of observing one sample's identifiers.

    Attributes
    ----------
    changed : dict
        Per field, whether the value differs from the previous sample for this entity. `None` on the entity's
        first sample and on an out-of-order one, where the question is not answerable rather than answered "no".
    first_seen : dict
        Per field, whether this entity has never reported this value before. `None` wherever `changed` is.
    distinct_counts : dict
        Per field, how many distinct values this entity has reported, counting the current one.
    out_of_order : bool
        The sample's event time was not after the previous sample's. State is left untouched.
    """

    changed: dict[str, typing.Optional[bool]]
    first_seen: dict[str, typing.Optional[bool]]
    distinct_counts: dict[str, int]
    out_of_order: bool


@dataclasses.dataclass
class _FieldState:
    last: typing.Any = None
    has_last: bool = False
    seen: collections.OrderedDict = dataclasses.field(default_factory=collections.OrderedDict)
    distinct: int = 0


class ValueNoveltyTracker:
    """
    Per-entity identifier state, detecting a change whenever it happens.

    The tracker holds the previous value and a bounded recall set per field per entity, so its memory is bounded by
    the entity count rather than by the stream, and by construction it has no period boundary for a change to hide
    behind. Results depend only on the sequence of samples it has been shown, so replaying a stream reproduces
    them.

    Parameters
    ----------
    field_names : list of str
        Identifiers to track, typically `["transceiver_serial", "lldp_neighbor_chassis_id"]`.
    max_values : int, default = 64
        Distinct values recalled per field per entity before the least recently seen is dropped.
    max_entities : int, default = 100000
        Entities retained before the least recently seen is dropped. A dropped entity's next sample is treated as
        its first, answering `None` rather than reporting a change it cannot substantiate.

    Notes
    -----
    `changed` has no blind spot: it compares consecutive samples and nothing else. `first_seen` does have one, and
    it is the recall bound rather than a period. Once a value has been evicted from an entity's recall set, its
    return reads as first seen again. The bound errs toward over-reporting novelty, which is the safe direction
    for a signal whose purpose is to surface the unexplained; under-reporting would hide it. For the same reason
    `distinct_counts` counts values that were new when observed rather than the size of the recall set, so it does
    not silently stop rising once the set is full.

    Returning to a previous value is a change, and deliberately so. A transceiver going A to B to A is two
    substitutions, not one, and an optic swapped out and swapped back is a more interesting sequence than one that
    simply changed.
    """

    def __init__(self,
                 field_names: typing.Sequence[str],
                 max_values: int = DEFAULT_MAX_VALUES,
                 max_entities: int = DEFAULT_MAX_ENTITIES):
        if (len(field_names) == 0):
            raise ValueError("At least one field name is required")

        if (max_values <= 0):
            raise ValueError(f"max_values must be positive, received {max_values}")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._field_names = list(field_names)
        self._max_values = max_values
        self._max_entities = max_entities

        self._states: collections.OrderedDict[str, dict[str, _FieldState]] = collections.OrderedDict()
        self._last_seen: dict[str, int] = {}

    @property
    def field_names(self) -> list[str]:
        """Identifiers this tracker watches."""
        return list(self._field_names)

    @property
    def tracked_entities(self) -> int:
        """Entities currently holding state."""
        return len(self._states)

    def observe(self, entity_key: str, event_time_ns: int, values: dict[str, typing.Any]) -> NoveltyResult:
        """
        Record one sample and return what changed.

        Parameters
        ----------
        entity_key : str
            The entity the identifiers belong to, typically `site_id:device_id:port_id`.
        event_time_ns : int
            When the sample was taken, in nanoseconds since the epoch. Event time, never ingest time.
        values : dict
            The identifier values. `None` is a value, not an omission: a port with nothing installed reports
            nothing, and the transition into or out of that state is a change.

        Returns
        -------
        `NoveltyResult`
        """
        previous_time = self._last_seen.get(entity_key)

        if (previous_time is not None and event_time_ns <= previous_time):
            # Admitting this would make the next sample's comparison run against a value that arrived late, so the
            # answer would depend on delivery order rather than on the estate. An equal timestamp is rejected on
            # purpose: a port has one transceiver and one neighbor at any instant, so two values at one time is a
            # duplicate poll, and "changed" has no meaning between simultaneous samples.
            return NoveltyResult(changed={name: None
                                          for name in self._field_names},
                                 first_seen={name: None
                                             for name in self._field_names},
                                 distinct_counts={name: 0
                                                  for name in self._field_names},
                                 out_of_order=True)

        fields = self._states.get(entity_key)
        is_first_sample = fields is None

        if (fields is None):
            fields = {name: _FieldState() for name in self._field_names}
            self._states[entity_key] = fields

        self._states.move_to_end(entity_key)

        changed: dict[str, typing.Optional[bool]] = {}
        first_seen: dict[str, typing.Optional[bool]] = {}
        distinct: dict[str, int] = {}

        for name in self._field_names:
            state = fields[name]
            value = values.get(name)

            was_new = value not in state.seen

            if (is_first_sample or not state.has_last):
                # The first sample establishes what normal looks like for this entity; it is not itself an event.
                changed[name] = None
                first_seen[name] = None
            else:
                changed[name] = value != state.last
                first_seen[name] = was_new

            if (was_new):
                state.distinct += 1

            state.seen[value] = None
            state.seen.move_to_end(value)

            while (len(state.seen) > self._max_values):
                state.seen.popitem(last=False)

            state.last = value
            state.has_last = True
            distinct[name] = state.distinct

        self._last_seen[entity_key] = event_time_ns
        self._evict()

        return NoveltyResult(changed=changed, first_seen=first_seen, distinct_counts=distinct, out_of_order=False)

    def _evict(self) -> None:
        while (len(self._states) > self._max_entities):
            (dropped, _) = self._states.popitem(last=False)
            self._last_seen.pop(dropped, None)
