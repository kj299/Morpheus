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
How unusual a cyclic bucket -- an hour of the day, a day of the week -- is for one entity's own history.

The TC-5 telemetry class asks for "hour-of-day and day-of-week deviation from the user's own histogram". The
comparison that matters is against the entity itself: 03:00 is unremarkable for the batch account that has run at
03:00 nightly for a year, and is the whole signal for the analyst who has never once authenticated outside office
hours. A population baseline answers a different and much weaker question.

**The current sample is excluded from the history it is judged against.** That is the opposite convention to
`morpheus.utils.distinct_window`, where the sample is counted inside its own window so a threshold trips on the row
that crosses it, and the same convention as `morpheus.utils.optical_baseline`, where a reading must not anchor the
reference it is measured against. Here the exclusion is not a preference: including the sample would give a user's
first ever 03:00 authentication a share of one in one, which is the least surprising value the measure can take,
at the exact moment it should be the most.

**Unseen buckets are smoothed rather than special-cased.** An entity's share of a bucket it has never used is zero,
and the surprise of a zero-probability event is infinite, which is not a number a threshold can be written against.
Add-one smoothing gives every bucket a floor of `1 / (samples + buckets)`, so a bucket first seen after two years of
history is reported as far more surprising than the same bucket first seen in a new joiner's second week. That is
the correct ordering, and it falls out of the history length rather than having to be configured.

**Both renderings of the same number are emitted.** `smoothed_share` is one integer division and is what the
arithmetic actually is; `surprise_bits` is its negative base-2 logarithm, because a threshold reads better in bits
than in a share that runs to four leading zeros. Both are quantized under determinism control 9
(`morpheus.utils.determinism.quantize_value`), which is what keeps the logarithm's last bits from reaching a golden
file.

**Buckets are supplied by the caller, not derived here.** This module never looks at a wall clock or a timezone. A
caller converting event time to an hour of the day has to choose an offset, and the choice is a determinism
question rather than a convenience: resolving a named zone through the IANA database makes every feature in the
pipeline depend on a data file that is revised several times a year, so a re-run after a `tzdata` update can move
scores with nothing in the pipeline having changed. Control 2 would have to freeze and hash that file. A fixed
offset has no such dependency, and `morpheus.stages.telemetry.tc5_cadence_stage` therefore takes one.
"""

import collections
import dataclasses
import math

from morpheus.utils.determinism import DEFAULT_FLOAT_DECIMALS
from morpheus.utils.determinism import quantize_value

DEFAULT_MAX_ENTITIES = 100_000
"""Entities tracked before the least recently seen is forgotten.

Sized for the TC-5 telemetry class, which the guide puts at tens of thousands of users -- an order of magnitude
below the MAC-keyed layer 2 trackers, and the reason this default is not theirs.
"""

DEFAULT_MIN_SAMPLES = 32
"""Prior observations before an entity's histogram is called mature.

A histogram with three entries in it says almost nothing, and a rule that reads a share off one is measuring how
new the account is. The tracker reports maturity rather than withholding a value, so a caller can choose between
suppressing an immature entity and alerting on it -- a brand new account authenticating at 03:00 is not obviously
the less interesting case.
"""


@dataclasses.dataclass(frozen=True)
class CyclicHistogramResult:
    """
    The outcome of observing one sample, judged against everything that came before it.

    Attributes
    ----------
    bucket_count : int
        Prior observations in this bucket, not counting this one.
    samples : int
        Prior observations across every bucket, not counting this one.
    smoothed_share : float
        `(bucket_count + 1) / (samples + buckets)`, the entity's add-one smoothed share of this bucket. Quantized.
    surprise_bits : float
        `-log2(smoothed_share)`. Larger is more unusual. Quantized. Zero prior samples gives `log2(buckets)`, the
        value a uniform prior implies, so a first-ever observation is neither surprising nor reassuring.
    bucket_unseen : bool
        This entity has never been observed in this bucket before.
    mature : bool
        The entity has at least `min_samples` prior observations, so the share is worth reading.
    out_of_order : bool
        The sample's event time was not after the previous one's. State is left untouched and the result describes
        the history as it stands, because a sample that arrives late must not be judged against events that
        happened after it.
    """

    bucket_count: int
    samples: int
    smoothed_share: float
    surprise_bits: float
    bucket_unseen: bool
    mature: bool
    out_of_order: bool


@dataclasses.dataclass
class _EntityHistogram:
    counts: collections.Counter = dataclasses.field(default_factory=collections.Counter)
    samples: int = 0
    last_seen_ns: int = 0
    started: bool = False


class CyclicHistogramTracker:
    """
    Per-entity counts over a fixed set of cyclic buckets, reporting how unusual each new sample is.

    Memory is bounded by the entity count times the bucket count, both of which are fixed, so the tracker cannot
    grow with the stream. Results depend only on the sequence of samples it has been shown, so replaying a stream
    reproduces them exactly.

    Parameters
    ----------
    buckets : int
        Size of the cycle: 24 for hour of day, 7 for day of week. Bucket indices must fall in `[0, buckets)`.
    min_samples : int, default = 32
        Prior observations before an entity's histogram is reported as mature.
    max_entities : int, default = 100000
        Entities retained before the least recently seen is dropped. A dropped entity starts from an empty
        histogram, so its next sample reads as immature rather than as carrying a stale share.
    decimals : int, default = 4
        Decimal places both quantized outputs are rounded to, under determinism control 9.
    """

    def __init__(self,
                 buckets: int,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 max_entities: int = DEFAULT_MAX_ENTITIES,
                 decimals: int = DEFAULT_FLOAT_DECIMALS):
        if (buckets <= 0):
            raise ValueError(f"buckets must be positive, received {buckets}")

        if (min_samples < 0):
            raise ValueError(f"min_samples must not be negative, received {min_samples}")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._buckets = buckets
        self._min_samples = min_samples
        self._max_entities = max_entities
        self._decimals = decimals

        self._histograms: collections.OrderedDict[str, _EntityHistogram] = collections.OrderedDict()

    @property
    def buckets(self) -> int:
        """Size of the cycle this tracker counts over."""
        return self._buckets

    @property
    def tracked_entities(self) -> int:
        """Entities currently holding a histogram."""
        return len(self._histograms)

    def observe(self, entity_key: str, event_time_ns: int, bucket: int) -> CyclicHistogramResult:
        """
        Record one observation and return how unusual it was against everything before it.

        Parameters
        ----------
        entity_key : str
            The entity whose own history this is judged against.
        event_time_ns : int
            Event time, used only to reject samples that arrive out of order.
        bucket : int
            Cyclic bucket index, in `[0, buckets)`.

        Returns
        -------
        `CyclicHistogramResult`
            The prior history's verdict on this sample.

        Raises
        ------
        ValueError
            If `bucket` is outside `[0, buckets)`. A caller that cannot determine a bucket should not call this
            with a sentinel: a bucket the cycle does not have would be counted as a real one, and would then make
            every genuine observation of it look less unusual than it is.
        """
        if (bucket < 0 or bucket >= self._buckets):
            raise ValueError(f"bucket must fall in [0, {self._buckets}), received {bucket}")

        histogram = self._histograms.get(entity_key)

        if (histogram is None):
            histogram = _EntityHistogram()
            self._histograms[entity_key] = histogram
            self._evict()
        else:
            self._histograms.move_to_end(entity_key)

        out_of_order = histogram.started and event_time_ns <= histogram.last_seen_ns

        # The reading is taken before the sample lands, whether or not the sample is going to land. An out-of-order
        # sample still gets an answer -- the history as it stood is a defensible thing to judge it against -- but it
        # must not change that history, or the order rows arrived in would show up in the next row's score.
        bucket_count = histogram.counts[bucket]
        samples = histogram.samples

        share = (bucket_count + 1) / (samples + self._buckets)

        result = CyclicHistogramResult(
            bucket_count=bucket_count,
            samples=samples,
            smoothed_share=quantize_value(share, decimals=self._decimals),
            surprise_bits=quantize_value(-math.log2(share), decimals=self._decimals),
            bucket_unseen=bucket_count == 0,
            mature=samples >= self._min_samples,
            out_of_order=out_of_order,
        )

        if (out_of_order):
            return result

        histogram.counts[bucket] += 1
        histogram.samples += 1
        histogram.last_seen_ns = event_time_ns
        histogram.started = True

        return result

    def _evict(self) -> None:
        """Forget the least recently seen entities until the cap holds."""
        while (len(self._histograms) > self._max_entities):
            self._histograms.popitem(last=False)
