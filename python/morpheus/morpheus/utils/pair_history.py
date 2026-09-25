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
Whether an entity has done this before, recently: the question R-B-L7-004 asks of a process pair.

`morpheus.utils.value_novelty` answers "has this entity ever reported this value", with no clock, because an optic
that has never been in a cage is news however long ago the cage was installed. Process ancestry needs a clock. A
`(parent, child)` pair last seen on a host eleven months ago is as unexplained today as one never seen at all, and
the rule says so: not seen *in thirty days*. So each entity here keeps, per pair, when it was last seen, and a pair
counts as seen only if that was inside the window.

**Seen means seen strictly before this instant.** An EDR logs a dozen process starts in one second. None of them is
prior to another, so each is compared with the pairs seen before that second, and the answer is the same whichever
order the log lists them in. Every pair keeps the latest time it was seen before its most recent sighting for
exactly this: a second sighting at one instant must not hide the first one's history.

**No answer before a warm-up.** Everything is new to an entity on its first day. Until the entity's history spans
the warm-up period the tracker says the question is not yet answerable, rather than answering "novel" for every pair
the entity has, which would be most of them.

**An out-of-order arrival is refused rather than repaired**, as the other trackers refuse it: whether a pair was seen
before is exactly what an arrival from the past disagrees about.

Memory is bounded twice, by pairs per entity and by entities, and both evict the least recently seen. An evicted
pair's return reads as novel. That errs toward over-reporting, which is the safe direction for a signal whose purpose
is to surface the unexplained.
"""

import collections
import dataclasses
import typing

NS_PER_SECOND = 10**9
DAY_NS = 24 * 3600 * NS_PER_SECOND

DEFAULT_WINDOW_NS = 30 * DAY_NS
"""How recently a pair must have been seen to count as seen. Thirty days, which is what R-B-L7-004 names."""

DEFAULT_WARMUP_NS = 7 * DAY_NS
"""History an entity needs before a pair can be called novel."""

DEFAULT_MAX_PAIRS = 16_384
"""Pairs recalled per entity. A peer group of build servers runs a few thousand distinct pairs a month."""

DEFAULT_MAX_ENTITIES = 200_000
"""Entities recalled before the least recently seen is forgotten. A forgotten entity starts its warm-up again."""


@dataclasses.dataclass(frozen=True)
class PairSighting:
    """
    The outcome of observing one pair on one entity.

    Attributes
    ----------
    seen : bool or None
        Whether the entity saw this pair strictly before this instant and inside the window. `None` on an
        out-of-order arrival, where the question has no stable answer.
    mature : bool
        The entity's history spans the warm-up, so `seen` being False means novel rather than merely new.
    history_ns : int or None
        How far back the entity's history reaches from this instant. `None` on an out-of-order arrival.
    saturated : bool
        The pair recall bound has evicted pairs from this entity, so `seen` may be False for a pair seen long ago.
    out_of_order : bool
        This sighting was earlier than the entity's previous one. State is left untouched.
    """

    seen: typing.Optional[bool]
    mature: bool
    history_ns: typing.Optional[int]
    saturated: bool
    out_of_order: bool


@dataclasses.dataclass
class _EntityPairs:
    first_ns: int
    last_ns: int
    # Per pair: the latest time it was seen, and the latest time before that.
    pairs: collections.OrderedDict = dataclasses.field(default_factory=collections.OrderedDict)
    saturated: bool = False


class PairHistoryTracker:
    """
    Per-entity recall of the pairs it has produced, and when each was last seen.

    Results depend only on the sequence of sightings the tracker has been shown, and not on the order of sightings at
    one instant, so replaying a stream reproduces them.

    Parameters
    ----------
    window_ns : int, optional
        How recently a pair must have been seen to count as seen.
    warmup_ns : int, optional
        History an entity needs before `mature` is True.
    max_pairs : int, optional
        Pairs recalled per entity before the least recently seen is dropped.
    max_entities : int, optional
        Entities recalled before the least recently seen is dropped.
    """

    def __init__(self,
                 window_ns: int = DEFAULT_WINDOW_NS,
                 warmup_ns: int = DEFAULT_WARMUP_NS,
                 max_pairs: int = DEFAULT_MAX_PAIRS,
                 max_entities: int = DEFAULT_MAX_ENTITIES):
        if (window_ns <= 0):
            raise ValueError(f"window_ns must be positive, received {window_ns}")

        if (warmup_ns < 0):
            raise ValueError(f"warmup_ns must not be negative, received {warmup_ns}")

        if (max_pairs <= 0):
            raise ValueError(f"max_pairs must be positive, received {max_pairs}")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._window_ns = window_ns
        self._warmup_ns = warmup_ns
        self._max_pairs = max_pairs
        self._max_entities = max_entities

        self._entities: collections.OrderedDict[str, _EntityPairs] = collections.OrderedDict()

    @property
    def tracked_entities(self) -> int:
        """Entities currently holding a history."""
        return len(self._entities)

    def observe(self, entity_key: str, event_time_ns: int, pair: typing.Hashable) -> PairSighting:
        """
        Record one sighting of a pair on an entity and say whether the entity had seen it recently.

        Parameters
        ----------
        entity_key : str
            Whose history: a host, or a peer group.
        event_time_ns : int
            The sighting's event time, in nanoseconds since the epoch. Event time, never ingest time.
        pair : hashable
            What was seen, usually a normalized `(parent, child)` tuple.

        Returns
        -------
        `PairSighting`

        Raises
        ------
        ValueError
            If `entity_key` is empty or `event_time_ns` is not an integer.
        """
        if (not entity_key):
            raise ValueError("entity_key is required; a history has to be a history of something")

        if (not isinstance(event_time_ns, int) or isinstance(event_time_ns, bool)):
            raise ValueError(f"event_time_ns must be an int, received {event_time_ns!r}")

        entity = self._entities.get(entity_key)

        if (entity is None):
            entity = _EntityPairs(first_ns=event_time_ns, last_ns=event_time_ns)
            self._entities[entity_key] = entity

            while (len(self._entities) > self._max_entities):
                self._entities.popitem(last=False)
        elif (event_time_ns < entity.last_ns):
            return PairSighting(seen=None, mature=False, history_ns=None, saturated=entity.saturated, out_of_order=True)
        else:
            self._entities.move_to_end(entity_key)

        horizon = event_time_ns - self._window_ns
        times = entity.pairs.get(pair)
        seen = False

        if (times is not None):
            (last, before) = times
            # The latest sighting strictly before this instant: the last one, unless it is this instant.
            previous = last if last < event_time_ns else before
            seen = previous is not None and previous >= horizon

            if (last < event_time_ns):
                times[0] = event_time_ns
                times[1] = last

            entity.pairs.move_to_end(pair)
        else:
            entity.pairs[pair] = [event_time_ns, None]

            if (len(entity.pairs) > self._max_pairs):
                entity.saturated = True

                while (len(entity.pairs) > self._max_pairs):
                    entity.pairs.popitem(last=False)

        entity.last_ns = event_time_ns
        history_ns = event_time_ns - entity.first_ns

        return PairSighting(seen=seen,
                            mature=history_ns >= self._warmup_ns,
                            history_ns=history_ns,
                            saturated=entity.saturated,
                            out_of_order=False)
