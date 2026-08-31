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
Optical power deviation from a per-port rolling baseline.

An absolute optical power reading says almost nothing on its own. A receive level of -7 dBm is healthy on one link
and alarming on another, because the correct value is a function of the optic, the fibre length, the splice count
and the patch path. What is diagnostic is the *change* from what this particular port has been reporting, which
makes the useful quantity a deviation from a per-port baseline rather than a threshold.

Two things move it, and they are the two the TC-1 telemetry class cares about:

- **Degradation** drifts the receive level down slowly, over days, as a connector fouls or a bend tightens.
- **An inline tap** steps it down at once. A passive splitter has to divert light to see it, and the light it
  diverts is missing from the far end: a fraction of a dB for a high-split tap, one to three dB for the common
  ones. That step is the security-relevant signal, and it is invisible to any absolute threshold that was set
  loosely enough not to alarm on healthy links.

Transmit and receive are kept separate because they fail differently. A falling transmit level is the local laser
ageing. A falling receive level with a steady transmit level at the far end is something that happened to the path
between them.

The baseline is the median of the port's prior readings inside a trailing window. The median rather than the mean
because optical diagnostics are noisy at the tenth-of-a-dB level and occasionally report a wild value, and a
reference a single bad reading can drag is not a reference. Prior readings rather than all of them because a sample
included in its own baseline partly anchors its own reference, damping the very step this is meant to expose.

Readings are in dBm and the deviation is in dB, and both stay that way: dBm is logarithmic, so a difference of
levels is already a ratio of powers, which is the quantity that means "how much light went missing".
"""

import collections
import dataclasses
import statistics
import typing

NS_PER_SECOND = 10**9

DEFAULT_WINDOW_NS = 6 * 3600 * NS_PER_SECOND
"""Trailing window the baseline is taken over. Six hours is long enough to average out diurnal thermal drift and
short enough that the reference still describes the link as it is now.
"""

DEFAULT_MIN_SAMPLES = 5
"""Prior readings required before a baseline is published. Below this the median is not a description of anything."""

DEFAULT_MAX_SAMPLES = 512
"""Readings retained per channel per entity, whatever the window implies. A bound on memory when a collector is
polling far faster than it was meant to.
"""

DEFAULT_MAX_ENTITIES = 100_000
"""Entities tracked before the least recently seen is forgotten."""


@dataclasses.dataclass(frozen=True)
class BaselineResult:
    """
    The outcome of observing one set of optical readings.

    Attributes
    ----------
    baselines : dict
        Per-channel median of the prior readings in the window. `None` until `min_samples` of them exist.
    deviations : dict
        Per-channel current reading minus its baseline, in dB. Negative means light went missing. `None` wherever
        the baseline is `None` or the current reading is absent.
    sample_counts : dict
        Prior readings each baseline was taken over, so a consumer can tell a well-supported reference from a
        barely-supported one.
    out_of_order : bool
        The sample's event time was not after the previous sample's. State is left untouched.
    """

    baselines: dict[str, typing.Optional[float]]
    deviations: dict[str, typing.Optional[float]]
    sample_counts: dict[str, int]
    out_of_order: bool


class OpticalBaselineTracker:
    """
    Per-entity rolling optical baselines.

    The tracker keeps a trailing window of readings per channel per entity and nothing else, so its memory is
    bounded by the entity count times the window rather than by the stream. Results depend only on the sequence of
    samples it has been shown, so replaying a stream reproduces the deviations.

    Parameters
    ----------
    channel_names : list of str
        Optical channels to track, typically `["optical_tx_dbm", "optical_rx_dbm"]`.
    window_ns : int, default = 6 hours
        Trailing window, in nanoseconds of event time, that the baseline is taken over.
    min_samples : int, default = 5
        Prior readings required before a baseline is published.
    max_samples : int, default = 512
        Readings retained per channel per entity regardless of the window.
    max_entities : int, default = 100000
        Entities retained before the least recently seen is dropped. A dropped entity starts accumulating again,
        publishing no baseline until it has `min_samples`, rather than a baseline built from nothing.

    Notes
    -----
    The baseline follows the link. Once the window has rolled past the last pre-event reading, the median describes
    the new level and the deviation returns to zero, so a step is a transient signal and not a persistent state: it
    has to be caught within `window_ns` of the step that caused it. Lengthening the window holds the evidence
    longer at the cost of a reference that lags a legitimate re-patch for just as long. A slow enough degradation
    is invisible for the same reason, since the baseline drifts down with it; catching that needs a comparison
    against a commissioning value, which is asset context and belongs in TC-0, not here.
    """

    def __init__(self,
                 channel_names: typing.Sequence[str],
                 window_ns: int = DEFAULT_WINDOW_NS,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 max_samples: int = DEFAULT_MAX_SAMPLES,
                 max_entities: int = DEFAULT_MAX_ENTITIES):
        if (len(channel_names) == 0):
            raise ValueError("At least one channel name is required")

        if (window_ns <= 0):
            raise ValueError(f"window_ns must be positive, received {window_ns}")

        if (min_samples < 1):
            raise ValueError(f"min_samples must be at least 1, received {min_samples}")

        if (max_samples < min_samples):
            raise ValueError(f"max_samples ({max_samples}) must be at least min_samples ({min_samples}), or a "
                             "baseline could never be published")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._channel_names = list(channel_names)
        self._window_ns = window_ns
        self._min_samples = min_samples
        self._max_samples = max_samples
        self._max_entities = max_entities

        # entity -> channel -> deque of (event_time_ns, reading)
        self._readings: collections.OrderedDict[str, dict[str, collections.deque]] = collections.OrderedDict()
        self._last_seen: dict[str, int] = {}

    @property
    def channel_names(self) -> list[str]:
        """Channels this tracker baselines."""
        return list(self._channel_names)

    @property
    def tracked_entities(self) -> int:
        """Entities currently holding state."""
        return len(self._readings)

    def observe(self, entity_key: str, event_time_ns: int, readings: dict[str,
                                                                          typing.Optional[float]]) -> BaselineResult:
        """
        Record one set of readings and return the deviations they imply.

        Parameters
        ----------
        entity_key : str
            The port the readings belong to, typically `site_id:device_id:port_id`.
        event_time_ns : int
            When the sample was taken, in nanoseconds since the epoch. Event time, never ingest time: a window
            measured in arrival order describes the collector's scheduling rather than the link.
        readings : dict
            Per-channel optical power in dBm. A channel missing or `None` contributes nothing to its own baseline
            and receives a `None` deviation, which is the case for a port with no optic in the cage.

        Returns
        -------
        `BaselineResult`
        """
        previous_time = self._last_seen.get(entity_key)

        if (previous_time is not None and event_time_ns <= previous_time):
            # Admitting this would let a late reading rewrite a baseline that later samples were already scored
            # against, which would make the result depend on arrival order.
            return BaselineResult(baselines={name: None
                                             for name in self._channel_names},
                                  deviations={name: None
                                              for name in self._channel_names},
                                  sample_counts={name: 0
                                                 for name in self._channel_names},
                                  out_of_order=True)

        channels = self._readings.get(entity_key)

        if (channels is None):
            channels = {name: collections.deque() for name in self._channel_names}
            self._readings[entity_key] = channels

        self._readings.move_to_end(entity_key)

        horizon = event_time_ns - self._window_ns
        baselines: dict[str, typing.Optional[float]] = {}
        deviations: dict[str, typing.Optional[float]] = {}
        counts: dict[str, int] = {}

        for name in self._channel_names:
            history = channels[name]

            while (len(history) > 0 and history[0][0] <= horizon):
                history.popleft()

            prior = [reading for (_, reading) in history]
            counts[name] = len(prior)

            baseline = statistics.median(prior) if len(prior) >= self._min_samples else None
            baselines[name] = baseline

            current = readings.get(name)
            deviations[name] = None if (baseline is None or current is None) else float(current) - baseline

            if (current is not None):
                history.append((event_time_ns, float(current)))

                while (len(history) > self._max_samples):
                    history.popleft()

        self._last_seen[entity_key] = event_time_ns
        self._evict()

        return BaselineResult(baselines=baselines, deviations=deviations, sample_counts=counts, out_of_order=False)

    def _evict(self) -> None:
        while (len(self._readings) > self._max_entities):
            (dropped, _) = self._readings.popitem(last=False)
            self._last_seen.pop(dropped, None)
