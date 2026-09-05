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
Time-bounded binding tables for soft joins between OSI layers.

Layers 1 through 3 are tied together by facts that are only true for an interval: an IP address belonged to a MAC
address during a DHCP lease, a MAC address was reachable through a switch port during a forwarding-table entry's
lifetime. Resolving an observation against such a fact is a *soft join* -- correct only when both ends of the interval
are known, and only when the resolution rule is fixed in advance.

This module provides that fixed rule. A `BindingTable` answers "which binding covered this key at this instant" with a
documented tie-break, so the same inputs resolve the same way on replay, and it can flatten itself into the
discretized, one-row-per-bucket form that a SIEM lookup requires.
"""

import bisect
import dataclasses
import logging
import numbers
import typing

import pandas as pd

from morpheus.utils.lineage import event_uid

logger = logging.getLogger(__name__)

NS_PER_SECOND = 10**9

DEFAULT_MAX_BUCKETS_PER_BINDING = 10_000
"""Bucket count above which `BindingTable.to_bucketed_records` refuses to expand a single binding."""

BUCKET_START_COLUMN = "bucket_start"
"""Field carrying each bucket's start time, so a SIEM can anchor `_time` on the row rather than on ingest."""

TABLE_NAME_COLUMN = "binding_table"
"""Field naming which binding source a bucketed row came from, when `to_bucketed_records` is told."""

DEFAULT_BUCKET_SECONDS = 300
"""
Canonical bucket width for the discretized binding lookups.

This is a contract, not a preference: the pipeline expands bindings across buckets of this width and the SIEM
rediscretizes event times with the same divisor. The two sides must agree exactly, or every lookup silently misses.
The shipped Splunk app uses this value, and `tests/morpheus/utils/test_splunk_app_contracts.py` fails if they
diverge.
"""


@dataclasses.dataclass(frozen=True)
class Binding:
    """
    One time-bounded fact.

    Attributes
    ----------
    key : str
        The value being resolved from, for example an IP address.
    start_ns : int
        Start of the validity interval, in nanoseconds since the Unix epoch. Inclusive.
    end_ns : int
        End of the validity interval, in nanoseconds since the Unix epoch. Exclusive.
    values : tuple
        The resolved attributes, in the order given by `BindingTable.value_columns`.
    uid : str
        Content-addressed identifier for this binding, so a resolution can be traced back to the exact record that
        produced it.
    """

    key: str
    start_ns: int
    end_ns: int
    values: tuple
    uid: str

    @property
    def duration_ns(self) -> int:
        """Length of the validity interval in nanoseconds."""
        return self.end_ns - self.start_ns

    def covers(self, event_time_ns: int) -> bool:
        """Whether the half-open interval `[start_ns, end_ns)` contains `event_time_ns`."""
        return self.start_ns <= event_time_ns < self.end_ns


def to_epoch_ns(value: typing.Any, time_unit: str = "ns") -> typing.Optional[int]:
    """
    Coerce a timestamp to an integer count of nanoseconds since the Unix epoch.

    Parameters
    ----------
    value : any
        A `datetime`, a `pandas.Timestamp`, a parsable date string, or a number. Numbers are interpreted in
        `time_unit`, which is why that parameter exists: an integer timestamp carries no indication of its own scale
        and guessing produces bindings that are wrong by three orders of magnitude.
    time_unit : str, default = "ns"
        Unit for numeric input. One of `s`, `ms`, `us`, `ns`, or `cs` for hundredths of a second, which is what SNMP
        reports `sysUpTime` and `ifLastChange` in (`TimeTicks`). Passing those through as `s` inflates them a
        hundredfold and silently defeats every check that compares an uptime against a sampling gap.

    Returns
    -------
    int or None
        Nanoseconds since the epoch, or `None` if the value is null.

    Raises
    ------
    ValueError
        If the value cannot be interpreted as a timestamp.
    """
    if (value is None):
        return None

    try:
        if (pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass

    try:
        # `numbers.Number` rather than `int` so that NumPy scalars, which a DataFrame column yields and which do not
        # subclass the builtin types, still honor `time_unit` instead of silently defaulting to nanoseconds.
        if (isinstance(value, numbers.Number) and not isinstance(value, bool)):
            if (time_unit == "cs"):
                # pandas has no name for hundredths of a second, so scale to the unit it does have.
                stamp = pd.Timestamp(value * 10, unit="ms")
            else:
                stamp = pd.Timestamp(value, unit=time_unit)
        else:
            stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unable to interpret {value!r} as a timestamp") from exc

    if (pd.isna(stamp)):
        return None

    if (stamp.tz is not None):
        stamp = stamp.tz_convert("UTC").tz_localize(None)

    return int(stamp.value)


def _sort_key(binding: Binding) -> tuple:
    """
    Total order used to pick a winner when several bindings cover the same instant.

    Most recent start wins, then the longer interval, then the greater attribute tuple compared as strings. The last
    component only breaks a tie between records that are identical in every other respect, and exists so the choice is
    a function of the data rather than of input order. Callers take the maximum of this key.
    """
    return (binding.start_ns, binding.end_ns, tuple(str(value) for value in binding.values))


class BindingTable:
    """
    An interval index over time-bounded facts, resolved with a fixed tie-break.

    Overlapping intervals for one key are a fact of life -- a DHCP server hands out a lease before the previous
    client's release is observed, a MAC moves between ports mid-poll. Rather than rejecting them, the table resolves
    them deterministically: the binding with the most recent start wins, then the longer interval, then the greater
    attribute tuple compared as strings. Overlaps are counted at construction and logged, because a high overlap
    rate means the upstream collector is losing expiry records and every resolution it produces is suspect.

    Parameters
    ----------
    name : str
        Identifier for this table, used to label the resolution method that the resolver stage records on each row.
    value_columns : list of str
        Names of the resolved attributes, in the order they appear in each `Binding.values` tuple.
    bindings : iterable of `Binding`
        The facts. Order is irrelevant; the table sorts internally.
    """

    def __init__(self, name: str, value_columns: typing.Sequence[str], bindings: typing.Iterable[Binding]):
        if (not name):
            raise ValueError("name is required")

        if (len(value_columns) == 0):
            raise ValueError("At least one value column is required")

        self._name = name
        self._value_columns = list(value_columns)
        self._by_key: dict[str, list[Binding]] = {}
        self._starts: dict[str, list[int]] = {}
        self._max_duration: dict[str, int] = {}
        self._size = 0

        for binding in bindings:
            if (len(binding.values) != len(self._value_columns)):
                raise ValueError(f"Binding for key {binding.key!r} has {len(binding.values)} values, "
                                 f"expected {len(self._value_columns)}")

            if (binding.end_ns <= binding.start_ns):
                raise ValueError(f"Binding for key {binding.key!r} ends at or before it starts: "
                                 f"[{binding.start_ns}, {binding.end_ns})")

            self._by_key.setdefault(binding.key, []).append(binding)
            self._size += 1

        overlapping_keys = 0

        for (key, key_bindings) in self._by_key.items():
            key_bindings.sort(key=_sort_key)
            self._starts[key] = [binding.start_ns for binding in key_bindings]
            self._max_duration[key] = max(binding.duration_ns for binding in key_bindings)

            if (self._has_overlap(key_bindings)):
                overlapping_keys += 1

        self._overlapping_keys = overlapping_keys

        if (overlapping_keys > 0):
            logger.warning(
                "Binding table %r has overlapping intervals on %d of %d keys. Resolution is still deterministic, but "
                "overlaps usually mean expiry records are being lost upstream.",
                name,
                overlapping_keys,
                len(self._by_key))

    @staticmethod
    def _has_overlap(key_bindings: list[Binding]) -> bool:
        """Whether any two intervals in a start-sorted list of bindings for one key intersect."""
        highest_end = None

        for binding in key_bindings:
            if (highest_end is not None and binding.start_ns < highest_end):
                return True

            highest_end = binding.end_ns if highest_end is None else max(highest_end, binding.end_ns)

        return False

    @property
    def name(self) -> str:
        """Identifier for this table."""
        return self._name

    @property
    def value_columns(self) -> list[str]:
        """Names of the resolved attributes."""
        return list(self._value_columns)

    @property
    def size(self) -> int:
        """Number of bindings held."""
        return self._size

    @property
    def key_count(self) -> int:
        """Number of distinct keys held."""
        return len(self._by_key)

    @property
    def overlapping_key_count(self) -> int:
        """Number of keys with at least two intersecting intervals."""
        return self._overlapping_keys

    @classmethod
    def from_dataframe(cls,
                       df: "pd.DataFrame",
                       name: str,
                       key_column: str,
                       value_columns: typing.Sequence[str],
                       start_column: str,
                       end_column: str,
                       time_unit: str = "ns",
                       open_end_duration_ns: typing.Optional[int] = None) -> "BindingTable":
        """
        Build a table from a DataFrame of binding records.

        Parameters
        ----------
        df : `pandas.DataFrame` or `cudf.DataFrame`
            One row per binding. A cuDF frame is copied to the host, since resolution runs on the host.
        name : str
            Identifier for the table.
        key_column : str
            Column holding the value being resolved from.
        value_columns : list of str
            Columns holding the resolved attributes.
        start_column : str
            Column holding the inclusive start of the validity interval.
        end_column : str
            Column holding the exclusive end of the validity interval.
        time_unit : str, default = "ns"
            Unit for numeric timestamps. Ignored for datetime columns.
        open_end_duration_ns : int, optional
            How long a binding with no observed end is assumed to last. When `None`, a null end raises, which is the
            safe default: a lease with no release record and no assumed duration cannot be joined against without
            inventing an answer. Set this to the lease time when the collector cannot see releases.

        Returns
        -------
        `BindingTable`

        Raises
        ------
        KeyError
            If a required column is absent.
        ValueError
            If a row has a null key or start, a null end with no `open_end_duration_ns`, or an end at or before its
            start.
        """
        if (hasattr(df, "to_pandas")):
            df = df.to_pandas()

        required = [key_column, start_column, end_column, *value_columns]
        missing = [column for column in required if column not in df.columns]

        if (len(missing) > 0):
            raise KeyError(f"Binding frame is missing columns {missing}. Available columns: {sorted(df.columns)}")

        if (open_end_duration_ns is not None and open_end_duration_ns <= 0):
            raise ValueError(f"open_end_duration_ns must be positive, received {open_end_duration_ns}")

        bindings = []

        for row in df[required].itertuples(index=False, name=None):
            (key, raw_start, raw_end) = (row[0], row[1], row[2])
            values = tuple(row[3:])

            if (key is None or pd.isna(key)):
                raise ValueError("Binding rows must have a non-null key")

            start_ns = to_epoch_ns(raw_start, time_unit=time_unit)

            if (start_ns is None):
                raise ValueError(f"Binding for key {key!r} has a null start")

            end_ns = to_epoch_ns(raw_end, time_unit=time_unit)

            if (end_ns is None):
                if (open_end_duration_ns is None):
                    raise ValueError(
                        f"Binding for key {key!r} has no end. A soft join against an unbounded interval is a guess, "
                        "not a join. Supply open_end_duration_ns to cap it explicitly.")

                end_ns = start_ns + open_end_duration_ns

            key = str(key)
            uid = event_uid(name, key, start_ns, end_ns, *values)

            bindings.append(Binding(key=key, start_ns=start_ns, end_ns=end_ns, values=values, uid=uid))

        return cls(name=name, value_columns=value_columns, bindings=bindings)

    def resolve(self, key: typing.Any, event_time_ns: typing.Optional[int]) -> typing.Optional[Binding]:
        """
        Find the binding that covered `key` at `event_time_ns`.

        Parameters
        ----------
        key : any
            Value to resolve. Coerced to `str` to match the table's keys.
        event_time_ns : int, optional
            Instant to resolve at, in nanoseconds since the epoch. `None` never resolves.

        Returns
        -------
        `Binding` or None
            The winning binding, or `None` if no interval covered the key at that instant.
        """
        if (event_time_ns is None or key is None):
            return None

        lookup_key = str(key)
        key_bindings = self._by_key.get(lookup_key)

        if (key_bindings is None):
            return None

        # Bindings are sorted by start, so every candidate lies at or before the insertion point. Walking backwards
        # from there stops as soon as the gap exceeds the longest interval on this key, since nothing starting earlier
        # can still be open.
        limit = event_time_ns - self._max_duration[lookup_key]
        index = bisect.bisect_right(self._starts[lookup_key], event_time_ns)

        best = None

        for position in range(index - 1, -1, -1):
            binding = key_bindings[position]

            if (binding.start_ns <= limit):
                break

            if (binding.covers(event_time_ns) and (best is None or _sort_key(binding) > _sort_key(best))):
                best = binding

        return best

    def resolve_many(self, keys: typing.Sequence, event_times_ns: typing.Sequence) -> list[typing.Optional[Binding]]:
        """
        Resolve a batch of observations.

        Results are memoized on the `(key, event_time)` pair, which matters because telemetry rolled up into time bins
        repeats both heavily.

        Parameters
        ----------
        keys : sequence
            Per-row values to resolve.
        event_times_ns : sequence
            Per-row instants, in nanoseconds since the epoch.

        Returns
        -------
        list
            One `Binding` or `None` per input row.

        Raises
        ------
        ValueError
            If the two sequences differ in length.
        """
        if (len(keys) != len(event_times_ns)):
            raise ValueError(f"keys and event_times_ns must be the same length, received {len(keys)} and "
                             f"{len(event_times_ns)}")

        cache: dict[tuple, typing.Optional[Binding]] = {}
        results: list[typing.Optional[Binding]] = []
        missing = object()

        for pair in zip(keys, event_times_ns):
            result = cache.get(pair, missing)

            if (result is missing):
                result = self.resolve(pair[0], pair[1])
                cache[pair] = result

            results.append(result)

        return results

    def to_bucketed_records(self,
                            bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
                            key_name: str = "key",
                            bucket_name: str = "bucket",
                            include_uid: bool = True,
                            max_buckets_per_binding: int = DEFAULT_MAX_BUCKETS_PER_BINDING,
                            table_name: typing.Optional[str] = None) -> list[dict]:
        """
        Flatten the table into one row per key and time bucket.

        This is the form a SIEM lookup needs. Splunk's `lookup` command matches on equality, not on interval
        containment, so the containment has to be precomputed: every binding is expanded across the buckets its
        interval touches, and where several bindings land in one bucket the table's own tie-break picks the winner, so
        that exactly one row exists per key and bucket and the lookup is single-valued.

        The cost of the approximation is bounded by the bucket width and should be documented alongside any rule that
        depends on it. A binding shorter than one bucket still produces a row for the bucket it falls in, which means a
        lookup can attribute an event to a binding that had already expired by up to `bucket_seconds`.

        Parameters
        ----------
        bucket_seconds : int, default = 300
            Bucket width. Must match the discretization used by the query side; see `DEFAULT_BUCKET_SECONDS`.
        key_name : str, default = "key"
            Field name for the key in the emitted records.
        bucket_name : str, default = "bucket"
            Field name for the bucket ordinal.
        include_uid : bool, default = True
            Whether to emit the winning binding's `uid`, which lets an analyst recover the exact record behind an
            attribution.
        max_buckets_per_binding : int, default = 10000
            Guard against a single long-lived binding expanding into an unusable number of rows.
        table_name : str, optional
            What this table is, for example `dhcp_lease` or `cam_table`. Emitted as `binding_table` on every record.
            A SIEM that indexes several binding sources under one sourcetype has no other way to tell them apart, and
            a lookup refresh that cannot tell them apart builds the wrong lookup. Omitted when there is nothing to
            disambiguate.

        Returns
        -------
        list of dict
            Records sorted by key then bucket, so the output is byte-stable across runs. Every record carries
            `bucket_start`, the bucket's own start time rendered the way the SIEM reads timestamps: a bucketed row
            has no other time of its own, and a row a SIEM cannot timestamp is silently indexed at ingest time.

        Raises
        ------
        ValueError
            If `bucket_seconds` is not positive, or if a binding would expand past `max_buckets_per_binding`.
        """
        if (bucket_seconds <= 0):
            raise ValueError(f"bucket_seconds must be positive, received {bucket_seconds}")

        # Imported here because `siem_wire` reads `to_epoch_ns` from this module; at module scope the two would
        # import each other.
        from morpheus.utils.siem_wire import render_event_time

        bucket_ns = bucket_seconds * NS_PER_SECOND
        winners: dict[tuple, Binding] = {}

        for key_bindings in self._by_key.values():
            for binding in key_bindings:
                first_bucket = binding.start_ns // bucket_ns
                last_bucket = (binding.end_ns - 1) // bucket_ns
                span = last_bucket - first_bucket + 1

                if (span > max_buckets_per_binding):
                    raise ValueError(
                        f"Binding for key {binding.key!r} spans {span} buckets of {bucket_seconds}s, over the limit of "
                        f"{max_buckets_per_binding}. Widen the bucket, shorten open_end_duration_ns, or raise the "
                        "limit deliberately.")

                for bucket in range(first_bucket, last_bucket + 1):
                    slot = (binding.key, bucket)
                    incumbent = winners.get(slot)

                    if (incumbent is None or _sort_key(binding) > _sort_key(incumbent)):
                        winners[slot] = binding

        records = []

        for ((key, bucket), binding) in sorted(winners.items()):
            record = {key_name: key, bucket_name: bucket, BUCKET_START_COLUMN: render_event_time(bucket * bucket_ns)}
            record.update(dict(zip(self._value_columns, binding.values)))

            if (table_name is not None):
                record[TABLE_NAME_COLUMN] = table_name

            if (include_uid):
                record["binding_uid"] = binding.uid

            records.append(record)

        return records

    def to_bucketed_frame(self, bucket_seconds: int = DEFAULT_BUCKET_SECONDS, **kwargs) -> "pd.DataFrame":
        """
        Flatten the table into a pandas DataFrame, ready to be written as a SIEM lookup.

        Parameters
        ----------
        bucket_seconds : int, default = 300
            Bucket width in seconds; see `DEFAULT_BUCKET_SECONDS`.
        **kwargs
            Forwarded to `to_bucketed_records`.

        Returns
        -------
        `pandas.DataFrame`
        """
        return pd.DataFrame(self.to_bucketed_records(bucket_seconds, **kwargs))
