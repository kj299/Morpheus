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
Runs of failures, and the success that ends one, over a trailing window per entity.

Two of the TC-5 behavioral features are the same shape: "failure-then-success sequences within a short window",
and the denial pattern R-D-L5-004 reads. Both are about a success that arrives after failures rather than about
either on its own, which is why they are one primitive.

**The trailing approval is what makes the pattern actionable rather than noisy.** Failed authentications are the
most ordinary event in an estate -- people mistype passwords -- and counting them alone produces a rule that
fires on the whole workforce every Monday. A run of denials that then stops being denied is a different claim:
somebody either remembered the password or approved the prompt, and in the MFA case the guide is explicit that
the approval is the part worth paging on.

**Two counts, because the two rules ask different questions.** `failures` is every failure in the window,
whatever order they came in, which is what R-D-L5-004 wants: five challenges, four denials, an approval, and the
denials need not be contiguous because a fatigue attack interleaves with the victim's own traffic.
`consecutive_failures` is the unbroken run immediately before this event, which is what the plain
failure-then-success feature wants. Reporting only one of them would leave a rule to reconstruct the other from
a column that cannot carry it.

The run is capped by what the window holds. A run that started before the window opened is reported at the
length the window can account for rather than at its true length, which under-reports rather than inventing
history the window has already forgotten.
"""

import collections
import dataclasses

NS_PER_SECOND = 10**9

DEFAULT_WINDOW_NS = 600 * NS_PER_SECOND
"""Trailing window, ten minutes: the interval R-D-L5-004 names."""

DEFAULT_MAX_SAMPLES = 4096
"""Events retained per entity regardless of the window.

A credential-stuffing run against one principal is both what this feature exists to notice and what would let an
unbounded window exhaust memory. The cap makes the counts lower bounds instead, and says so.
"""

DEFAULT_MAX_ENTITIES = 100_000
"""Entities tracked before the least recently seen is forgotten."""


@dataclasses.dataclass(frozen=True)
class OutcomeRunResult:
    """
    The outcome of observing one attempt.

    Attributes
    ----------
    attempts : int
        Attempts in the window, counting this one.
    failures : int
        Failed attempts in the window, counting this one if it failed. Not necessarily contiguous.
    consecutive_failures : int
        The unbroken run of failures immediately before this attempt, capped by what the window holds. Zero when
        the previous attempt in the window succeeded.
    failure_then_success : bool
        This attempt succeeded and at least one failure came immediately before it. The plain form of the
        pattern; a rule wanting a longer run reads `consecutive_failures`.
    saturated : bool
        The sample cap is binding, so events are being evicted before the window would have expired them and both
        counts are floors rather than figures.
    out_of_order : bool
        The attempt's event time was not after the previous one's. State is left untouched.
    """

    attempts: int
    failures: int
    consecutive_failures: int
    failure_then_success: bool
    saturated: bool
    out_of_order: bool


@dataclasses.dataclass
class _EntityWindow:
    history: collections.deque = dataclasses.field(default_factory=collections.deque)
    failures: int = 0
    trailing_failures: int = 0
    last_seen_ns: int = 0
    started: bool = False


class OutcomeRunTracker:
    """
    Per-entity failure counting and run length over a trailing window.

    The tracker holds one window per entity, bounded by both the window duration and a sample cap, so its memory
    is bounded by the entity count rather than by the stream. Results depend only on the sequence of attempts it
    has been shown, so replaying a stream reproduces them.

    Parameters
    ----------
    window_ns : int, default = 10 minutes
        Trailing window the counts cover.
    max_samples : int, default = 4096
        Attempts retained per entity regardless of the window. When this binds the result is marked saturated and
        the counts are lower bounds.
    max_entities : int, default = 100000
        Entities retained before the least recently seen is dropped. A dropped entity starts from an empty
        window, so its next attempt reads as the first rather than as ending a run it cannot see.
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

    def observe(self, entity_key: str, event_time_ns: int, succeeded: bool) -> OutcomeRunResult:
        """
        Record one attempt and return what the window now holds.

        Parameters
        ----------
        entity_key : str
            The entity the attempt belongs to, typically a principal.
        event_time_ns : int
            When it happened, in nanoseconds since the epoch. Event time, never ingest time.
        succeeded : bool
            Whether the attempt succeeded. A caller that cannot tell should not guess: an unknown outcome counted
            as a failure inflates a run, and counted as a success ends one that is still going.

        Returns
        -------
        `OutcomeRunResult`
        """
        window = self._windows.get(entity_key)

        if (window is None):
            window = _EntityWindow()
            self._windows[entity_key] = window
            self._evict()
        else:
            self._windows.move_to_end(entity_key)

        if (window.started and event_time_ns < window.last_seen_ns):
            # Admitting this would put an attempt into the middle of a run it did not belong to, so the next
            # result would depend on delivery order rather than on what the principal did. An equal timestamp is
            # not late: identity providers stamp to the second, and a fatigue burst puts several challenges on one
            # tick. Rejecting those would read the attack this feature exists to catch as a single prompt.
            return OutcomeRunResult(attempts=len(window.history),
                                    failures=window.failures,
                                    consecutive_failures=min(window.trailing_failures, len(window.history)),
                                    failure_then_success=False,
                                    saturated=False,
                                    out_of_order=True)

        # Expire against this attempt's horizon before anything else, so the run is judged against the window as
        # it stands now rather than as it stood when the previous attempt arrived.
        horizon_ns = event_time_ns - self._window_ns

        while (len(window.history) > 0 and window.history[0][0] <= horizon_ns):
            self._drop_oldest(window)

        # Read the run before this attempt joins it. The cap is what the window can still account for: a run that
        # began before the window opened is reported short rather than at a length the window has forgotten.
        preceding_run = min(window.trailing_failures, len(window.history))

        window.history.append((event_time_ns, succeeded))
        window.last_seen_ns = event_time_ns
        window.started = True

        if (succeeded):
            window.trailing_failures = 0
        else:
            window.failures += 1
            window.trailing_failures += 1

        saturated = len(window.history) > self._max_samples

        while (len(window.history) > self._max_samples):
            self._drop_oldest(window)

        return OutcomeRunResult(attempts=len(window.history),
                                failures=window.failures,
                                consecutive_failures=preceding_run,
                                failure_then_success=succeeded and preceding_run > 0,
                                saturated=saturated,
                                out_of_order=False)

    @staticmethod
    def _drop_oldest(window: _EntityWindow) -> None:
        """Evict the oldest attempt, keeping the failure count honest."""
        (_, succeeded) = window.history.popleft()

        if (not succeeded):
            window.failures -= 1

    def _evict(self) -> None:
        """Forget the least recently seen entities until the cap holds."""
        while (len(self._windows) > self._max_entities):
            self._windows.popitem(last=False)
