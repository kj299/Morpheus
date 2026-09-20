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
What a source's IP time-to-live usually is, and by how many hops this packet differs from it.

A TTL is set by the sending host to a value its operating system picks -- 64, 128 and 255 are the common ones --
and decremented once by every router on the way. The value a collector sees is therefore the initial value minus
the hop count, and for a given source seen at a given point in the network it is close to constant. R-B-L3-004
reads a change in it as an interposed device or as spoofing, and both readings follow from the same arithmetic:
something that forwards packets decrements the field, and something that forges them has to guess what the real
host would have sent.

**The reference is the mode of the source's prior packets, not their mean.** A mean over a source that is really
two hosts behind one address, one Windows and one Linux, lands between 128 and 64 and describes neither. The mode
describes the commonest path, `distinct` says whether there is more than one, and a bimodal source is a fact
about the estate rather than an error to average away.

**Prior packets rather than all of them**, for the reason
{py:mod}`~morpheus.utils.optical_baseline` uses the same rule: a sample folded into its own reference partly
anchors it, which damps the step the feature exists to expose.

**Ties break towards the larger TTL.** Where an interposed device has been present for exactly half the window,
63 and 64 have equal support, and calling 63 the reference would report the device's absence as the anomaly. The
un-decremented value is the one a host sends, so it is the one treated as the baseline.

**A shift of exactly one hop counts.** The guide's wording for R-B-L3-004 is "shifts by more than one
hop-equivalent", and read strictly that excludes a shift of one -- which is precisely what a single interposed
device produces, since it forwards the packet once. Requiring strictly more would miss the case the rule exists
for and catch only the ones with two or more devices in the path, so `min_shift` defaults to one and the
comparison is inclusive. An estate that finds one hop too noisy, because its routing genuinely moves, raises the
parameter deliberately rather than inheriting a threshold that reads as caution and is a blind spot.

This module reports one packet's deviation from its source's reference. It does not decide that a distribution
has moved, because that is a question about a window of packets rather than about a packet, and the shipped
detection asks it the way R-D-L2-001 asks its own: by aggregating the per-observation figure over a window on
the search head, where the threshold can be tuned without redeploying a pipeline.
"""

import collections
import dataclasses
import typing

NS_PER_SECOND = 10**9

DEFAULT_WINDOW_NS = 24 * 3600 * NS_PER_SECOND
"""Trailing window the reference is taken over.

A day. Long enough that a source seen a few times an hour has a reference at all, short enough that a genuine
re-route stops being called an anomaly once the new path is what the source has been doing all day.
"""

DEFAULT_MIN_SAMPLES = 5
"""Prior packets required before a reference is published. Below this a mode is the first value seen twice."""

DEFAULT_MIN_SHIFT = 1
"""Hops of difference that count as a shift. One, because one is what an interposed device adds."""

DEFAULT_MAX_SAMPLES = 512
"""Packets retained per source whatever the window implies, bounding memory by sources rather than by traffic."""

DEFAULT_MAX_ENTITIES = 500_000
"""Sources retained before the least recently seen is dropped. A dropped source starts from no reference."""

MAX_TTL = 255
"""The field is eight bits. A value outside it did not come off a wire and is refused rather than profiled."""


@dataclasses.dataclass(frozen=True)
class TtlResult:
    """
    The outcome of observing one packet's TTL.

    Attributes
    ----------
    ttl : int
        The value observed.
    established : int or None
        The source's reference, being the mode of its prior packets in the window. `None` before `min_samples`.
    shift : int or None
        `ttl - established`, in hops. Negative means this packet travelled through more devices than the source's
        packets usually do, which is the direction an interposition produces. `None` without a reference.
    distinct : int
        Distinct TTLs among the retained packets, counting this one. Above one means the source is more than one
        host, or its path is changing.
    samples : int
        Packets retained for this source, counting this one.
    shifted : bool
        `shift` is at least `min_shift` in magnitude. False whenever there is no reference yet.
    mature : bool
        A reference was published, so `shift` is a comparison rather than an absence.
    saturated : bool
        The sample cap is binding, so the reference describes the retained tail rather than the window.
    out_of_order : bool
        This packet's event time was not after the previous one's. State is left untouched.
    """

    ttl: int
    established: typing.Optional[int]
    shift: typing.Optional[int]
    distinct: int
    samples: int
    shifted: bool
    mature: bool
    saturated: bool
    out_of_order: bool


@dataclasses.dataclass
class _SourceWindow:
    """One source's retained packets."""

    times: collections.deque = dataclasses.field(default_factory=collections.deque)
    ttls: collections.deque = dataclasses.field(default_factory=collections.deque)
    last_time_ns: typing.Optional[int] = None
    saturated: bool = False


def established_ttl(values: typing.Sequence[int]) -> typing.Optional[int]:
    """
    The commonest TTL in a series, ties going to the larger value.

    Parameters
    ----------
    values : sequence of int
        The retained TTLs. Empty yields `None`.

    Returns
    -------
    int or None
    """
    if (len(values) == 0):
        return None

    counts = collections.Counter(values)
    best = max(counts.values())

    return max(ttl for (ttl, count) in counts.items() if count == best)


class TtlProfileTracker:
    """
    Per-source TTL reference, and each packet's deviation from it.

    The entity is whatever the caller keys on, which for R-B-L3-004 is the source address. Keying on the directed
    pair instead would be defensible for an estate with asymmetric routing, and would need a reference per
    destination to be worth anything; the source is the unit the rule names.

    Results depend only on the sequence of packets the tracker has been shown, so replaying a stream reproduces
    them.

    Parameters
    ----------
    window_ns : int, default = 1 day
        Trailing window the reference is taken over.
    min_samples : int, default = 5
        Prior packets required before a reference is published.
    min_shift : int, default = 1
        Hops of difference that count as a shift.
    max_samples : int, default = 512
        Packets retained per source regardless of the window.
    max_entities : int, default = 500000
        Sources retained before the least recently seen is dropped.
    """

    def __init__(self,
                 window_ns: int = DEFAULT_WINDOW_NS,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 min_shift: int = DEFAULT_MIN_SHIFT,
                 max_samples: int = DEFAULT_MAX_SAMPLES,
                 max_entities: int = DEFAULT_MAX_ENTITIES):
        if (window_ns <= 0):
            raise ValueError(f"window_ns must be positive, received {window_ns}")

        if (min_samples < 1):
            raise ValueError(f"min_samples must be at least 1, received {min_samples}")

        if (min_shift < 1):
            raise ValueError(f"min_shift must be at least 1, received {min_shift}; a shift of zero hops is the "
                             f"source behaving normally")

        if (max_samples < 1):
            raise ValueError(f"max_samples must be at least 1, received {max_samples}")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._window_ns = window_ns
        self._min_samples = min_samples
        self._min_shift = min_shift
        self._max_samples = max_samples
        self._max_entities = max_entities

        self._windows: collections.OrderedDict[str, _SourceWindow] = collections.OrderedDict()

    @property
    def tracked_entities(self) -> int:
        """Sources currently holding a window."""
        return len(self._windows)

    def _window_for(self, entity_key: str) -> _SourceWindow:
        window = self._windows.get(entity_key)

        if (window is None):
            window = _SourceWindow()
            self._windows[entity_key] = window

            while (len(self._windows) > self._max_entities):
                self._windows.popitem(last=False)
        else:
            self._windows.move_to_end(entity_key)

        return window

    def observe(self, entity_key: str, event_time_ns: int, ttl: int) -> TtlResult:
        """
        Record one packet's TTL and describe how far it sits from its source's reference.

        Parameters
        ----------
        entity_key : str
            What the reference belongs to, usually the source address.
        event_time_ns : int
            The packet's event time, in nanoseconds since the epoch. Event time, never ingest time.
        ttl : int
            The observed time-to-live, between 0 and 255.

        Returns
        -------
        `TtlResult`

        Raises
        ------
        ValueError
            If `entity_key` is empty, `event_time_ns` is not an integer, or `ttl` is outside the field's range.
        """
        if (not entity_key):
            raise ValueError("entity_key is required; a reference has to be a reference for something")

        if (not isinstance(event_time_ns, int) or isinstance(event_time_ns, bool)):
            raise ValueError(f"event_time_ns must be an int, received {event_time_ns!r}")

        if (not isinstance(ttl, int) or isinstance(ttl, bool) or not 0 <= ttl <= MAX_TTL):
            raise ValueError(f"ttl must be an int between 0 and {MAX_TTL}, received {ttl!r}; a value outside the "
                             f"field's range is a parsing fault rather than a packet, and profiling it would put "
                             f"the fault in the reference")

        window = self._window_for(entity_key)

        if (window.last_time_ns is not None and event_time_ns <= window.last_time_ns):
            # Refused rather than repaired, as the other trackers refuse it. The reference is over prior packets,
            # and which packets are prior is exactly what an out-of-order arrival disagrees about.
            prior = list(window.ttls)
            reference = established_ttl(prior) if len(prior) >= self._min_samples else None

            return TtlResult(ttl=ttl,
                             established=reference,
                             shift=None,
                             distinct=len(set(prior)),
                             samples=len(prior),
                             shifted=False,
                             mature=reference is not None,
                             saturated=window.saturated,
                             out_of_order=True)

        horizon = event_time_ns - self._window_ns

        while (len(window.times) > 0 and window.times[0] < horizon):
            window.times.popleft()
            window.ttls.popleft()

        # Taken before this packet joins them, so the reference is the source's history rather than a figure this
        # packet has already pulled towards itself.
        prior = list(window.ttls)
        reference = established_ttl(prior) if len(prior) >= self._min_samples else None
        shift = None if reference is None else ttl - reference

        window.times.append(event_time_ns)
        window.ttls.append(ttl)
        window.last_time_ns = event_time_ns

        if (len(window.times) > self._max_samples):
            window.saturated = True

            while (len(window.times) > self._max_samples):
                window.times.popleft()
                window.ttls.popleft()

        return TtlResult(ttl=ttl,
                         established=reference,
                         shift=shift,
                         distinct=len(set(window.ttls)),
                         samples=len(window.ttls),
                         shifted=shift is not None and abs(shift) >= self._min_shift,
                         mature=reference is not None,
                         saturated=window.saturated,
                         out_of_order=False)
