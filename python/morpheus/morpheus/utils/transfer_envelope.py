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
How large a transfer is against the largest this pair has normally made, which is what R-B-L4-005 thresholds.

A byte count means nothing on its own. Forty megabytes between a workstation and a backup server is a Tuesday,
and the same forty megabytes between that workstation and a printer is the most interesting thing in the estate.
The guide is explicit about the consequence: baseline per `(src_ip, dst_ip, dst_port)` triple, not globally,
because a global threshold on transfer volume is useless in a heterogeneous environment.

**The baseline is a quantile of the triple's own prior transfers, taken by nearest rank.** No interpolation: with
a bounded sample, an interpolated 99th percentile is a number no transfer ever had, and a detector whose
reference is a value the entity never produced is harder to explain to the person triaging it than one that
points at a real transfer. Nearest rank also has no floating-point tie-break to get wrong on replay.

**A quantile over too few samples is the maximum, and saying otherwise would be a lie about what is being
measured.** By nearest rank the 99th percentile of anything up to a hundred samples *is* the largest of them, so a
triple with twenty transfers has a "99th percentile" that is simply its record, and three times a record is a far
weaker test than the rule intends. `min_samples` therefore defaults to a hundred, which is the point at which the
quantile begins to be a quantile rather than an extreme, and below it no baseline is published at all. An estate
whose triples are too sparse for that should read this feature as "three times the largest ever seen" and set the
parameter deliberately, rather than inheriting a number that reads as a percentile and is not one.

**Prior transfers rather than all of them.** A transfer folded into its own baseline raises the very reference it
is being compared against, which is the same reason {py:mod}`~morpheus.utils.optical_baseline` excludes the
current reading -- and it matters more here, because one enormous transfer would otherwise partly excuse itself.

**A baseline of zero yields no ratio.** A triple that has only ever sent empty payloads has no scale for
"three times bigger" to mean anything against, and returning an infinity would make the quietest pair in the
estate the loudest alert in it.
"""

import collections
import dataclasses
import math
import typing

NS_PER_SECOND = 10**9

DEFAULT_WINDOW_NS = 30 * 24 * 3600 * NS_PER_SECOND
"""Trailing window the baseline is taken over. Thirty days, which is the period R-B-L4-005 names."""

DEFAULT_QUANTILE = 0.99
"""Quantile the baseline is taken at, which is the one R-B-L4-005 names."""

DEFAULT_MULTIPLIER = 3.0
"""How many times the baseline counts as a breach. Three, which is the figure R-B-L4-005 names."""

DEFAULT_MIN_SAMPLES = 100
"""Prior transfers required before a baseline is published.

A hundred because that is where a 99th percentile by nearest rank stops being the maximum. Below it the figure
exists and means something different from what its name says, which is worse than having no figure.
"""

DEFAULT_MAX_SAMPLES = 4096
"""Transfers retained per entity whatever the window implies, bounding memory by entities rather than traffic."""

DEFAULT_MAX_ENTITIES = 500_000
"""Entities retained before the least recently seen is dropped. A dropped entity starts from no baseline."""


@dataclasses.dataclass(frozen=True)
class EnvelopeResult:
    """
    The outcome of observing one transfer.

    Attributes
    ----------
    value : float
        The magnitude observed.
    baseline : float or None
        The quantile of this entity's prior transfers in the window. `None` below `min_samples`.
    ratio : float or None
        `value / baseline`. `None` without a baseline, and `None` when the baseline is zero, which is undefined
        rather than infinite.
    samples : int
        Transfers retained for this entity, counting this one.
    breached : bool
        `ratio` is at least `multiplier`. False whenever there is no ratio.
    mature : bool
        A baseline was published, so `ratio` is a comparison rather than an absence.
    saturated : bool
        The sample cap is binding, so the baseline describes the retained tail rather than the window.
    out_of_order : bool
        This transfer's event time was not after the previous one's. State is left untouched.
    """

    value: float
    baseline: typing.Optional[float]
    ratio: typing.Optional[float]
    samples: int
    breached: bool
    mature: bool
    saturated: bool
    out_of_order: bool


@dataclasses.dataclass
class _EntityWindow:
    """One entity's retained transfers."""

    times: collections.deque = dataclasses.field(default_factory=collections.deque)
    values: collections.deque = dataclasses.field(default_factory=collections.deque)
    last_time_ns: typing.Optional[int] = None
    saturated: bool = False


def nearest_rank(values: typing.Sequence[float], quantile: float) -> typing.Optional[float]:
    """
    The value at a quantile by nearest rank, which is always one of the values.

    The rank is `ceil(quantile * n)`, clamped into the sequence, and the result is the value at that rank in
    ascending order. For `n` below `1 / (1 - quantile)` this is the maximum, which is a property of the estimator
    rather than a defect of this implementation and is the reason `TransferEnvelopeTracker` refuses to publish a
    baseline from too few samples.

    Parameters
    ----------
    values : sequence of float
        The sample. Empty yields `None`.
    quantile : float
        Between 0 and 1 inclusive.

    Returns
    -------
    float or None
    """
    if (not 0.0 <= quantile <= 1.0):
        raise ValueError(f"quantile must be between 0 and 1, received {quantile}")

    if (len(values) == 0):
        return None

    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))

    return ordered[min(rank, len(ordered)) - 1]


class TransferEnvelopeTracker:
    """
    Per-entity quantile baseline of a magnitude, and each transfer's size against it.

    The entity is whatever the caller keys on. R-B-L4-005 keys on the `(src_ip, dst_ip, dst_port)` triple, which
    is the unit a transfer envelope is a property of: the same pair on a different port is a different
    conversation with a different normal size.

    Results depend only on the sequence of transfers the tracker has been shown, so replaying a stream reproduces
    them.

    Parameters
    ----------
    window_ns : int, default = 30 days
        Trailing window the baseline is taken over.
    quantile : float, default = 0.99
        Quantile the baseline is taken at.
    multiplier : float, default = 3.0
        How many times the baseline counts as a breach.
    min_samples : int, default = 100
        Prior transfers required before a baseline is published.
    max_samples : int, default = 4096
        Transfers retained per entity regardless of the window.
    max_entities : int, default = 500000
        Entities retained before the least recently seen is dropped.
    """

    def __init__(self,
                 window_ns: int = DEFAULT_WINDOW_NS,
                 quantile: float = DEFAULT_QUANTILE,
                 multiplier: float = DEFAULT_MULTIPLIER,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 max_samples: int = DEFAULT_MAX_SAMPLES,
                 max_entities: int = DEFAULT_MAX_ENTITIES):
        if (window_ns <= 0):
            raise ValueError(f"window_ns must be positive, received {window_ns}")

        if (not 0.0 <= quantile <= 1.0):
            raise ValueError(f"quantile must be between 0 and 1, received {quantile}")

        if (multiplier <= 1.0):
            raise ValueError(f"multiplier must be above 1, received {multiplier}; a breach at or below the "
                             f"baseline is every transfer the entity has ever made")

        if (min_samples < 1):
            raise ValueError(f"min_samples must be at least 1, received {min_samples}")

        if (max_samples < 1):
            raise ValueError(f"max_samples must be at least 1, received {max_samples}")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._window_ns = window_ns
        self._quantile = quantile
        self._multiplier = multiplier
        self._min_samples = min_samples
        self._max_samples = max_samples
        self._max_entities = max_entities

        self._windows: collections.OrderedDict[str, _EntityWindow] = collections.OrderedDict()

    @property
    def tracked_entities(self) -> int:
        """Entities currently holding a window."""
        return len(self._windows)

    def _window_for(self, entity_key: str) -> _EntityWindow:
        window = self._windows.get(entity_key)

        if (window is None):
            window = _EntityWindow()
            self._windows[entity_key] = window

            while (len(self._windows) > self._max_entities):
                self._windows.popitem(last=False)
        else:
            self._windows.move_to_end(entity_key)

        return window

    def observe(self, entity_key: str, event_time_ns: int, value: float) -> EnvelopeResult:
        """
        Record one transfer and describe it against the entity's own envelope.

        Parameters
        ----------
        entity_key : str
            What the envelope belongs to, usually a source, destination and port triple.
        event_time_ns : int
            The transfer's event time, in nanoseconds since the epoch. Event time, never ingest time.
        value : float
            The magnitude, usually a byte count or a bytes-per-packet figure.

        Returns
        -------
        `EnvelopeResult`

        Raises
        ------
        ValueError
            If `entity_key` is empty, `event_time_ns` is not an integer, or `value` is not a finite number.
        """
        if (not entity_key):
            raise ValueError("entity_key is required; an envelope has to be an envelope for something")

        if (not isinstance(event_time_ns, int) or isinstance(event_time_ns, bool)):
            raise ValueError(f"event_time_ns must be an int, received {event_time_ns!r}")

        try:
            magnitude = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"value must be a number, received {value!r}") from exc

        if (math.isnan(magnitude) or math.isinf(magnitude)):
            raise ValueError(f"value must be finite, received {value!r}; a baseline built from an infinity "
                             f"excuses every transfer after it")

        window = self._window_for(entity_key)

        if (window.last_time_ns is not None and event_time_ns <= window.last_time_ns):
            # Refused rather than repaired, as the other trackers refuse it. The baseline is over prior
            # transfers, and which transfers are prior is exactly what an out-of-order arrival disagrees about.
            prior = list(window.values)
            baseline = nearest_rank(prior, self._quantile) if len(prior) >= self._min_samples else None

            return EnvelopeResult(value=magnitude,
                                  baseline=baseline,
                                  ratio=None,
                                  samples=len(prior),
                                  breached=False,
                                  mature=baseline is not None,
                                  saturated=window.saturated,
                                  out_of_order=True)

        horizon = event_time_ns - self._window_ns

        while (len(window.times) > 0 and window.times[0] < horizon):
            window.times.popleft()
            window.values.popleft()

        # Taken before this transfer joins them, so one enormous transfer cannot partly excuse itself.
        prior = list(window.values)
        baseline = nearest_rank(prior, self._quantile) if len(prior) >= self._min_samples else None
        ratio = None if (baseline is None or baseline == 0) else magnitude / baseline

        window.times.append(event_time_ns)
        window.values.append(magnitude)
        window.last_time_ns = event_time_ns

        if (len(window.times) > self._max_samples):
            window.saturated = True

            while (len(window.times) > self._max_samples):
                window.times.popleft()
                window.values.popleft()

        return EnvelopeResult(value=magnitude,
                              baseline=baseline,
                              ratio=ratio,
                              samples=len(window.values),
                              breached=ratio is not None and ratio >= self._multiplier,
                              mature=baseline is not None,
                              saturated=window.saturated,
                              out_of_order=False)
