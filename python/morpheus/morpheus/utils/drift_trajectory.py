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
Whether an entity's reconstruction error is climbing, and by how much against its own variation.

This is what R-P-L5-006 reads, and R-P-L5-006 is the rule the word "predictive" in this fork's title rests on:
`mean_abs_z` rising monotonically across four consecutive windows, by more than one and a half standard
deviations in total, without any single window crossing the alerting threshold. The guide is explicit that it
should never page -- it places a principal on a watchlist and raises the sensitivity of layer 7 rules for that
principal -- and equally explicit that the premise behind it, that reconstruction error rises during
reconnaissance and staging, is a hypothesis this work does not establish. Nothing here establishes it either.
What this does is measure the trajectory exactly, so that a deployment can test the premise against its own
incident history rather than taking it on trust.

**This module does not know what produced the score.** It takes a number per entity per window. That the number
usually comes from a per-user autoencoder is the caller's business, and keeping it out of here is what lets the
trajectory be tested without one.

**A run needs consecutive windows.** A principal who did not authenticate for two days has not been "rising for
four consecutive windows", and treating their next observation as the fourth would report drift where there was
absence. A gap therefore restarts the run rather than extending it, and the restart is reported.

**The standard deviation is the entity's own, over its prior windows only.** Including the current value in the
spread it is being measured against would let a rising sequence inflate its own denominator and suppress the
signal it exists to produce -- the same convention, and for the same reason, as
`morpheus.utils.cyclic_histogram`. The spread is of the values rather than of the differences: the guide says
"a total increase above 1.5 standard deviations", and the quantity a reader will picture is the spread of the
score itself. A deployment preferring the spread of the differences is measuring something defensible and
different, and should say so rather than reinterpreting this column.

Velocity and acceleration are the first and second differences the guide names as its second predictive
mechanism, reported alongside. A large positive second difference is drift becoming a spike.
"""

import collections
import dataclasses
import statistics
import typing

from morpheus.utils.determinism import DEFAULT_FLOAT_DECIMALS
from morpheus.utils.determinism import quantize_value

DEFAULT_MAX_WINDOWS = 64
"""Windows retained per entity.

The trajectory is a short-run question -- the rule asks about four -- and the spread it is measured against
should describe the entity's recent normal rather than its entire history. Sixty-four daily windows is two
months, which is long enough to be a baseline and short enough that a principal who changed roles a year ago is
not still being compared with who they were.
"""

DEFAULT_MIN_WINDOWS = 4
"""Prior windows before a trajectory is called mature.

A standard deviation over two points is not a description of anything. Reported rather than withheld, so a rule
can decide: a brand new principal whose score is climbing is not obviously the less interesting case.
"""

DEFAULT_MAX_ENTITIES = 100_000


@dataclasses.dataclass(frozen=True)
class DriftResult:
    """
    The outcome of observing one window's score for one entity.

    Attributes
    ----------
    windows : int
        Windows observed for this entity, counting this one.
    velocity : float or None
        This window's score minus the previous window's. `None` on the first window of a run.
    acceleration : float or None
        This window's velocity minus the previous window's. `None` until there are three windows in a run.
        A large positive value is drift becoming a spike.
    rising_windows : int
        Length of the current strictly-increasing run, counting this window. One means the run just started or
        just broke.
    total_rise : float or None
        The score now minus the score at the start of the current run. `None` for a run of one.
    baseline_sigma : float or None
        The entity's own standard deviation over its prior retained windows, excluding this one. `None` with
        fewer than two priors.
    rise_sigmas : float or None
        `total_rise` in units of `baseline_sigma`. This is the quantity R-P-L5-006 thresholds at 1.5. `None`
        where either input is, and `None` rather than infinity where the spread is zero -- a principal whose
        score has never varied has no scale to express a rise in, and inventing one would make the flattest
        history the most alarming.
    mature : bool
        The entity has at least `min_windows` prior observations.
    run_restarted : bool
        The window is not the one after the previous observation, so the run restarted rather than continuing.
        A gap in a principal's activity is absence, not drift.
    out_of_order : bool
        The window is not after the previous one. Nothing is recorded.
    """

    windows: int
    velocity: typing.Optional[float]
    acceleration: typing.Optional[float]
    rising_windows: int
    total_rise: typing.Optional[float]
    baseline_sigma: typing.Optional[float]
    rise_sigmas: typing.Optional[float]
    mature: bool
    run_restarted: bool
    out_of_order: bool


@dataclasses.dataclass
class _EntityTrack:
    values: collections.deque = dataclasses.field(default_factory=collections.deque)
    last_window: typing.Optional[int] = None
    last_value: typing.Optional[float] = None
    last_velocity: typing.Optional[float] = None
    run_start_value: typing.Optional[float] = None
    rising: int = 0


class DriftTrajectoryTracker:
    """
    Per-entity trajectory of a per-window score.

    Memory is bounded by the entity count times the retained window count, both fixed. Results depend only on
    the sequence of observations it has been shown, so replaying a stream reproduces them.

    Parameters
    ----------
    max_windows : int, default = 64
        Windows retained per entity, which is what the standard deviation is computed over.
    min_windows : int, default = 4
        Prior windows before a trajectory is reported as mature.
    max_entities : int, default = 100000
        Entities retained before the least recently seen is dropped.
    decimals : int, default = 4
        Decimal places every reported figure is rounded to, under determinism control 9.
    """

    def __init__(self,
                 max_windows: int = DEFAULT_MAX_WINDOWS,
                 min_windows: int = DEFAULT_MIN_WINDOWS,
                 max_entities: int = DEFAULT_MAX_ENTITIES,
                 decimals: int = DEFAULT_FLOAT_DECIMALS):
        if (max_windows < 2):
            raise ValueError(f"max_windows must be at least 2, received {max_windows}")

        if (min_windows < 0):
            raise ValueError(f"min_windows must not be negative, received {min_windows}")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._max_windows = max_windows
        self._min_windows = min_windows
        self._max_entities = max_entities
        self._decimals = decimals

        self._tracks: collections.OrderedDict[str, _EntityTrack] = collections.OrderedDict()

    @property
    def tracked_entities(self) -> int:
        """Entities currently holding a trajectory."""
        return len(self._tracks)

    def observe(self, entity_key: str, window_id: int, value: float) -> DriftResult:
        """
        Record one window's score and return what the trajectory now looks like.

        Parameters
        ----------
        entity_key : str
            The entity whose own trajectory this is.
        window_id : int
            The window the score belongs to. Consecutive identifiers are what make a run a run.
        value : float
            The score, typically `mean_abs_z`.

        Returns
        -------
        `DriftResult`
        """
        track = self._tracks.get(entity_key)

        if (track is None):
            track = _EntityTrack()
            self._tracks[entity_key] = track
            self._evict()
        else:
            self._tracks.move_to_end(entity_key)

        if (track.last_window is not None and window_id <= track.last_window):
            # A window that arrives after a later one cannot join the run: admitting it would make the next
            # result depend on delivery order rather than on the entity.
            return DriftResult(windows=len(track.values),
                               velocity=None,
                               acceleration=None,
                               rising_windows=track.rising,
                               total_rise=None,
                               baseline_sigma=self._sigma(track),
                               rise_sigmas=None,
                               mature=len(track.values) >= self._min_windows,
                               run_restarted=False,
                               out_of_order=True)

        priors = list(track.values)
        sigma = self._sigma(track)
        mature = len(priors) >= self._min_windows
        consecutive = track.last_window is not None and window_id == track.last_window + 1
        restarted = track.last_window is not None and not consecutive

        velocity = None
        acceleration = None

        if (consecutive and track.last_value is not None):
            velocity = value - track.last_value

            if (track.last_velocity is not None):
                acceleration = velocity - track.last_velocity

        if (consecutive and velocity is not None and velocity > 0):
            track.rising += 1

            if (track.rising == 2):
                # The run begins at the window before this one, which is the last value that was not a rise.
                track.run_start_value = track.last_value
        else:
            # A break, a gap, or the first observation. The run begins here rather than continuing, and this
            # window is its first member.
            track.rising = 1
            track.run_start_value = value

        total_rise = None if track.rising < 2 else value - track.run_start_value
        rise_sigmas = None

        if (total_rise is not None and sigma is not None and sigma > 0):
            rise_sigmas = total_rise / sigma

        track.values.append(value)

        while (len(track.values) > self._max_windows):
            track.values.popleft()

        track.last_window = window_id
        track.last_value = value
        track.last_velocity = velocity

        return DriftResult(windows=len(track.values),
                           velocity=self._round(velocity),
                           acceleration=self._round(acceleration),
                           rising_windows=track.rising,
                           total_rise=self._round(total_rise),
                           baseline_sigma=self._round(sigma),
                           rise_sigmas=self._round(rise_sigmas),
                           mature=mature,
                           run_restarted=restarted,
                           out_of_order=False)

    def _round(self, value: typing.Optional[float]) -> typing.Optional[float]:
        return None if value is None else quantize_value(value, decimals=self._decimals)

    @staticmethod
    def _sigma(track: _EntityTrack) -> typing.Optional[float]:
        """The entity's own spread over its prior windows. `None` with fewer than two."""
        return statistics.stdev(track.values) if len(track.values) >= 2 else None

    def _evict(self) -> None:
        while (len(self._tracks) > self._max_entities):
            self._tracks.popitem(last=False)
