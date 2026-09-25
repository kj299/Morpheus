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
What an entity's value usually is, so that this observation can be compared against it.

Two layer 6 rules are the same question asked about different fields. R-D-L6-002 asks whether a destination's
certificate issuer differs from the issuer that destination usually presents, which catches interception --
including the well-intentioned interception that broke a policy. R-B-L6-004 asks whether a pair's negotiated
cipher suite is weaker than the weakest that pair has ever settled on, which catches an active downgrade. Both
need a reference taken over the entity's own prior observations, and neither can use a global one: an estate
with a dozen certificate authorities has no single correct issuer, and a fleet spanning two decades of operating
systems has no single acceptable cipher.

The reference is the only thing that differs between them, so it is a parameter. `mode_of` answers "what does
this entity normally present", which is what an issuer comparison wants. `minimum_of` answers "what is the worst
this entity has ever accepted", which is what a downgrade comparison wants -- a mode would be wrong there, since
a pair that negotiates a strong suite nine times in ten has a mode that a tenth, weaker, routine negotiation
sits below without anything being under attack.

**The reference is taken over prior observations, before this one joins them.** A value that counted towards its
own reference would drag the reference towards itself, and the first observation of a genuinely new issuer would
be compared against a history it had already entered.

**Maturity is not a formality.** A mode over two observations is the first value seen twice, and a minimum over
one observation is that observation, which every subsequent value either matches or sits above. Below the floor
the tracker publishes no reference and the rules that read it stay quiet, which is the correct behaviour for a
destination nobody has seen enough of to have expectations about.

**An out-of-order arrival is refused rather than repaired.** The reference is over prior observations, and which
observations are prior is exactly what an out-of-order arrival disagrees about. The result carries the flag so a
caller can tell a refusal from an answer. Observations at the same instant are not out of order: none of them is
prior to another, so each is measured against the observations strictly before that instant, and the answer does
not depend on the order they arrive in.

This module overlaps `morpheus.utils.ttl_profile`, which keeps the same trailing window and takes the same mode
over it. They are separate because their answers are different in kind: a TTL profile reports a magnitude -- how
many hops this packet sits from the reference -- and that arithmetic is meaningless for an issuer name. Rebasing
the TTL tracker onto this one would be a real simplification and is deliberately not done here, because it would
put a merged layer 3 golden at risk for a tidiness this layer does not need. Recorded rather than left for a
reader to notice, so that whoever does it knows it was seen.
"""

import collections
import dataclasses
import typing

NS_PER_SECOND = 10**9

DEFAULT_WINDOW_NS = 30 * 24 * 3600 * NS_PER_SECOND
"""Trailing window the reference is taken over. Thirty days, which is the period R-B-L6-001 names and the one an
estate's certificate rotation is measured against."""

DEFAULT_MIN_SAMPLES = 5
"""Prior observations required before a reference is published. Below this a mode is the first value seen twice
and a minimum is whatever arrived first."""

DEFAULT_MAX_SAMPLES = 512
"""Observations retained per entity whatever the window implies, bounding memory by entities rather than by
traffic."""

DEFAULT_MAX_ENTITIES = 500_000
"""Entities retained before the least recently seen is dropped. A dropped entity starts from no reference."""


def mode_of(values: typing.Sequence) -> typing.Any:
    """
    The commonest value in a series, ties going to the largest.

    Parameters
    ----------
    values : sequence
        The retained observations. Empty yields `None`.

    Returns
    -------
    The commonest value, or `None`.

    Notes
    -----
    The tie-break is what makes this deterministic. `Counter.most_common` orders ties by insertion, so two
    issuers each seen three times would produce whichever arrived first -- a reference that depends on arrival
    order, which is the property determinism control 8 exists to remove.
    """
    if (len(values) == 0):
        return None

    counts = collections.Counter(values)
    best = max(counts.values())

    return max(value for (value, count) in counts.items() if count == best)


def minimum_of(values: typing.Sequence) -> typing.Any:
    """
    The smallest value in a series.

    Parameters
    ----------
    values : sequence
        The retained observations. Empty yields `None`.

    Returns
    -------
    The smallest value, or `None`.
    """
    return min(values) if len(values) > 0 else None


@dataclasses.dataclass(frozen=True)
class HistoryResult:
    """
    One observation, and the reference its entity's history supplies.

    Attributes
    ----------
    value : any
        What was observed.
    reference : any
        What the entity's prior observations reduce to, or `None` before the tracker is mature.
    samples : int
        Prior observations the reference was taken over.
    distinct : int
        Distinct values among those. One means the entity has never presented anything else, which is what makes
        a difference worth reporting; a destination already presenting four issuers is a weaker signal and the
        count is carried so a search can say so.
    mature : bool
        Whether a reference was published.
    saturated : bool
        Whether observations were dropped to stay inside `max_samples`, making the reference a reduction over a
        suffix of the entity's history rather than over all of it.
    out_of_order : bool
        Whether this observation is earlier than the entity's previous one, in which case it was refused.
    """

    value: typing.Any
    reference: typing.Any
    samples: int
    distinct: int
    mature: bool
    saturated: bool
    out_of_order: bool


@dataclasses.dataclass
class _EntityWindow:
    """One entity's retained observations."""

    times: collections.deque = dataclasses.field(default_factory=collections.deque)
    values: collections.deque = dataclasses.field(default_factory=collections.deque)
    last_time_ns: typing.Optional[int] = None
    saturated: bool = False
    # The observations strictly before `last_time_ns`, so every observation at that instant shares one reference.
    instant_prior: list = dataclasses.field(default_factory=list)


class ValueHistoryTracker:
    """
    Per-entity reference over a trailing window, and each observation measured against it.

    Results depend only on the sequence of observations the tracker has been shown, so replaying a stream
    reproduces them.

    Parameters
    ----------
    reduction : callable, optional
        What the prior observations reduce to. Defaults to `mode_of`.
    window_ns : int, optional
        Trailing window the reference is taken over.
    min_samples : int, optional
        Prior observations required before a reference is published.
    max_samples : int, optional
        Observations retained per entity whatever the window implies.
    max_entities : int, optional
        Entities retained before the least recently seen is dropped.
    """

    def __init__(self,
                 reduction: typing.Callable = mode_of,
                 window_ns: int = DEFAULT_WINDOW_NS,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 max_samples: int = DEFAULT_MAX_SAMPLES,
                 max_entities: int = DEFAULT_MAX_ENTITIES):
        if (window_ns <= 0):
            raise ValueError(f"window_ns must be positive, received {window_ns}")

        if (min_samples < 1):
            raise ValueError(f"min_samples must be at least 1, received {min_samples}; a reference over no "
                             f"observations is the value itself wearing a reference's name")

        if (max_samples < min_samples):
            raise ValueError(f"max_samples ({max_samples}) must be at least min_samples ({min_samples}), or the "
                             f"tracker discards the observations it needs to mature")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._reduction = reduction
        self._window_ns = window_ns
        self._min_samples = min_samples
        self._max_samples = max_samples
        self._max_entities = max_entities

        self._windows: dict = {}

    @property
    def tracked_entities(self) -> int:
        """How many entities currently hold a history."""
        return len(self._windows)

    def _window_for(self, entity_key: str) -> _EntityWindow:
        window = self._windows.pop(entity_key, None)

        if (window is None):
            window = _EntityWindow()

            while (len(self._windows) >= self._max_entities):
                self._windows.pop(next(iter(self._windows)))

        self._windows[entity_key] = window

        return window

    def observe(self, entity_key: str, event_time_ns: int, value: typing.Any) -> HistoryResult:
        """
        Record one observation and describe the reference its entity's history supplies.

        Parameters
        ----------
        entity_key : str
            What the reference belongs to: a destination for an issuer, a directed pair for a cipher.
        event_time_ns : int
            The observation's event time, in nanoseconds since the epoch. Event time, never ingest time.
        value : any
            The observed value. Must be hashable, and comparable if the reduction orders its input.

        Returns
        -------
        `HistoryResult`

        Raises
        ------
        ValueError
            If `entity_key` is empty or `event_time_ns` is not an integer.
        """
        if (not entity_key):
            raise ValueError("entity_key is required; a reference has to be a reference for something")

        if (not isinstance(event_time_ns, int) or isinstance(event_time_ns, bool)):
            raise ValueError(f"event_time_ns must be an int, received {event_time_ns!r}")

        window = self._window_for(entity_key)

        if (window.last_time_ns is not None and event_time_ns < window.last_time_ns):
            return self._result(value, list(window.values), window.saturated, out_of_order=True)

        if (event_time_ns == window.last_time_ns):
            prior = window.instant_prior
        else:
            horizon = event_time_ns - self._window_ns

            while (len(window.times) > 0 and window.times[0] < horizon):
                window.times.popleft()
                window.values.popleft()

            # Taken before this observation joins them, so the reference is the entity's history rather than a
            # figure this observation has already pulled towards itself.
            prior = list(window.values)
            window.instant_prior = prior

        window.times.append(event_time_ns)
        window.values.append(value)
        window.last_time_ns = event_time_ns

        if (len(window.times) > self._max_samples):
            window.saturated = True

            while (len(window.times) > self._max_samples):
                window.times.popleft()
                window.values.popleft()

        return self._result(value, prior, window.saturated, out_of_order=False)

    def _result(self, value: typing.Any, prior: list, saturated: bool, out_of_order: bool) -> HistoryResult:
        reference = self._reduction(prior) if len(prior) >= self._min_samples else None

        return HistoryResult(value=value,
                             reference=reference,
                             samples=len(prior),
                             distinct=len(set(prior)),
                             mature=reference is not None,
                             saturated=saturated,
                             out_of_order=out_of_order)
