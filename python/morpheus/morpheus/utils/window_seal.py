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
Deterministic window sealing.

Windows are half-open intervals `[epoch + k*period, epoch + (k+1)*period)` anchored to a fixed absolute epoch, never
to the first observed event. A window is *sealed* once the event-time watermark, the largest event time observed so
far, reaches the window's end plus a lateness horizon; only sealed windows should be scored. Rows that arrive for an
already sealed window are *late*: they belong to a separate late-arrival stream and never mutate a published window.

Sealing is driven by the watermark rather than by the wall clock, which is what makes it reproducible: replaying the
same rows in the same order seals the same windows at the same points in the stream, no matter when or how fast the
replay runs. Row membership in a window is a pure function of the row's event time alone; only the on-time versus
late classification depends on stream order, which is exactly the property the late-arrival stream records.

The `WindowSealer` here tracks watermark, open windows, and per-window revision numbers. It deliberately holds no row
data, so the caller owns buffering and the sealer's own memory stays bounded.
"""

import collections
import dataclasses
import typing

from morpheus.utils.lineage import window_id_from_timestamp

SEALED_BY_WATERMARK = "watermark"
"""The window's seal was triggered by the event-time watermark passing its lateness horizon."""

SEALED_BY_FLUSH = "flush"
"""The window was still open when the stream ended and was sealed by the final flush."""

DEFAULT_REVISION_MEMORY = 100_000
"""Number of windows for which revision counters are retained before the oldest are forgotten."""


@dataclasses.dataclass(frozen=True)
class RowAssignment:
    """
    Where one row landed.

    Attributes
    ----------
    window_id : int or None
        The window ordinal, or `None` when the row's event time could not be interpreted.
    late : bool
        True when the row's window was already sealed at the moment the row was observed.
    """

    window_id: typing.Optional[int]
    late: bool


@dataclasses.dataclass(frozen=True)
class BatchResult:
    """
    Outcome of observing one batch of rows.

    Attributes
    ----------
    assignments : list of `RowAssignment`
        One entry per input row, in input order.
    sealed : list of int
        Window ordinals sealed while observing this batch, in ascending window order. Only windows that received at
        least one on-time row appear; an empty window has nothing to publish.
    """

    assignments: list[RowAssignment]
    sealed: list[int]


class WindowSealer:
    """
    Watermark-driven window assignment and sealing.

    Rows are processed strictly in the order given, one at a time: a row is first classified against the watermark as
    it stood before the row, then the watermark advances, then any open windows whose horizon the new watermark passed
    are sealed. Because every step is a function of the row sequence alone, the outcome is independent of how the rows
    were batched, which is determinism control 5 of the OSI behavioral analytics guide applied to windowing.

    Parameters
    ----------
    period_ns : int
        Window length in nanoseconds. Must be positive.
    lateness_ns : int
        Lateness horizon in nanoseconds: how far the watermark must pass a window's end before the window seals.
        Must not be negative. Larger values trade detection latency for fewer late rows.
    epoch_ns : int, default = 0
        Absolute anchor for window boundaries, in nanoseconds since the Unix epoch.
    revision_memory : int, default = 100000
        Number of windows for which revision counters are kept. Counters for the least recently touched windows
        beyond this are forgotten, after which a late arrival for such a window restarts its numbering; the
        `window_complete` flag, not the revision number, is what marks those rows as non-authoritative.
    """

    def __init__(self,
                 period_ns: int,
                 lateness_ns: int,
                 epoch_ns: int = 0,
                 revision_memory: int = DEFAULT_REVISION_MEMORY):
        if (period_ns <= 0):
            raise ValueError(f"period_ns must be positive, received {period_ns}")

        if (lateness_ns < 0):
            raise ValueError(f"lateness_ns must not be negative, received {lateness_ns}")

        if (revision_memory <= 0):
            raise ValueError(f"revision_memory must be positive, received {revision_memory}")

        self._period_ns = period_ns
        self._lateness_ns = lateness_ns
        self._epoch_ns = epoch_ns
        self._revision_memory = revision_memory

        self._watermark_ns: typing.Optional[int] = None
        self._open: set[int] = set()
        self._revisions: collections.OrderedDict[int, int] = collections.OrderedDict()

    @property
    def watermark_ns(self) -> typing.Optional[int]:
        """The largest event time observed so far, or `None` before the first row."""
        return self._watermark_ns

    @property
    def open_window_ids(self) -> list[int]:
        """Ordinals of windows currently holding on-time rows, ascending."""
        return sorted(self._open)

    def window_id(self, event_time_ns: int) -> int:
        """The ordinal of the window containing `event_time_ns`."""
        return window_id_from_timestamp(event_time_ns, self._period_ns, epoch_ns=self._epoch_ns)

    def window_bounds(self, window_id: int) -> tuple[int, int]:
        """
        The half-open interval covered by a window.

        Parameters
        ----------
        window_id : int
            Window ordinal.

        Returns
        -------
        tuple
            `(start_ns, end_ns)`; the start is inclusive and the end is exclusive.
        """
        start_ns = self._epoch_ns + window_id * self._period_ns

        return (start_ns, start_ns + self._period_ns)

    def is_sealed(self, window_id: int) -> bool:
        """Whether the watermark has passed this window's lateness horizon."""
        if (self._watermark_ns is None):
            return False

        return self.window_bounds(window_id)[1] + self._lateness_ns <= self._watermark_ns

    def next_revision(self, window_id: int) -> int:
        """
        Allocate the next revision number for a window.

        The first publication of a window is revision 0; every subsequent publication for the same window, which in
        practice means late-arrival batches and explicit backfills, increments from there. Consumers keep every
        revision and filter to the maximum per window, which is why a republication must never reuse a number.
        """
        revision = self._revisions.pop(window_id, -1) + 1

        self._revisions[window_id] = revision

        while (len(self._revisions) > self._revision_memory):
            self._revisions.popitem(last=False)

        return revision

    def observe(self, event_times_ns: typing.Sequence[typing.Optional[int]]) -> BatchResult:
        """
        Assign a batch of rows to windows, advancing the watermark and sealing as it goes.

        Parameters
        ----------
        event_times_ns : sequence of int or None
            Per-row event times in nanoseconds since the epoch, in stream order. `None` marks a row whose time could
            not be interpreted; it receives a `None` window and does not advance the watermark.

        Returns
        -------
        `BatchResult`
        """
        assignments: list[RowAssignment] = []
        sealed: list[int] = []

        for event_time_ns in event_times_ns:
            if (event_time_ns is None):
                assignments.append(RowAssignment(window_id=None, late=False))
                continue

            window_id = self.window_id(event_time_ns)
            late = self.is_sealed(window_id)

            assignments.append(RowAssignment(window_id=window_id, late=late))

            if (not late):
                self._open.add(window_id)

            if (self._watermark_ns is None or event_time_ns > self._watermark_ns):
                self._watermark_ns = event_time_ns

                for open_id in sorted(self._open):
                    if (self.is_sealed(open_id)):
                        self._open.discard(open_id)
                        sealed.append(open_id)

        return BatchResult(assignments=assignments, sealed=sealed)

    def flush(self) -> list[int]:
        """
        Seal every remaining open window, in ascending order.

        Called when the stream ends. Windows sealed this way carry `sealed_by = "flush"` so a consumer can tell a
        watermark-complete window from one cut short by shutdown.
        """
        remaining = sorted(self._open)
        self._open.clear()

        return remaining
