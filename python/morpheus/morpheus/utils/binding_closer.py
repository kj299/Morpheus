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
Turning a stream of binding observations into closed intervals.

Layer 2 is where the identifier ladder crosses from a physical port to a MAC address, and it can only carry that
weight if its bindings are time-bounded. `morpheus.utils.binding_table` resolves against half-open intervals
`[bind_start, bind_end)`, so a binding with no end resolves nothing and layer 2 stops being usable as a lineage hop
at all. The TC-2 telemetry class states this as a requirement: emit an explicit end on expiry rather than relying on
the next binding's start.

Nothing upstream provides it. A switch MAC table is a snapshot of what is bound now, RADIUS accounting stops go
missing, and DHCP releases are advisory. Ends have to be inferred, and this module is where that inference lives,
with the reason recorded on every record so a consumer can tell an observed end from a deduced one.

Five things end a binding, and only the first is a fact:

- **Explicit.** The source said so: an accounting stop, a release, a cleared entry. The end time is known.
- **Displacement.** The same key was observed somewhere else. The binding ended somewhere between its last
  observation and the new one, and this module closes it at the earlier bound.
- **Conflict.** The same key was observed somewhere else *at the same instant*, so neither sighting can be said to
  precede the other. The closed binding and the new one overlap by one tick, which is the honest record of a key in
  two places at once; at layer 2 that is what a spoofed or duplicated MAC looks like, and the reason is recorded
  separately from displacement so a rule can find it rather than having it read as a data quality warning.
- **Snapshot absence.** A reconciliation snapshot of a scope no longer lists the key, so it aged out of the table
  at some point since the last one.
- **Idle timeout.** Nothing has been heard for longer than the source's own aging interval, the safety net for the
  stop record that never arrived.

Where the end is inferred, it is placed at the earliest time consistent with what was observed rather than the
latest. That choice is what makes a resolution trustworthy: closing late leaves the old binding covering a period
the entity may have already left, so a join returns a confident wrong answer, while closing early leaves a gap, and
a gap resolves to nothing and tells the analyst the truth, which is that nobody knows.

Because the interval is half-open, an end placed exactly at the last observation would exclude that observation
from its own binding, and a key seen only once would produce a zero-width interval covering nothing at all. Every
inferred end therefore sits one tick past the last observation, which is the shortest interval that actually
contains what was seen.
"""

import collections
import dataclasses
import typing

NS_PER_SECOND = 10**9

MINIMUM_TICK_NS = 1
"""Gap between the last observation and an inferred end, so the half-open interval contains that observation."""

DEFAULT_IDLE_TIMEOUT_NS = 30 * 60 * NS_PER_SECOND
"""Silence after which an open binding is presumed aged out. Switch MAC tables commonly age at five minutes and
802.1X sessions live far longer, so this is a backstop rather than the primary mechanism; set it from the source's
own aging interval where that is known.
"""

DEFAULT_MAX_OPEN = 500_000
"""Open bindings held before the least recently seen is forced closed. Sized for the TC-2 telemetry class, whose
cardinality runs to low hundreds of thousands of MAC addresses, rather than for layer 1's port counts.
"""

EXPLICIT = "explicit"
DISPLACED = "displaced"
CONFLICT = "conflict"
SNAPSHOT_ABSENT = "snapshot_absent"
IDLE_TIMEOUT = "idle_timeout"
EVICTED = "evicted"
DRAINED = "drained"

INFERRED_REASONS = frozenset({DISPLACED, CONFLICT, SNAPSHOT_ABSENT, IDLE_TIMEOUT, EVICTED, DRAINED})
"""Every reason but `EXPLICIT`. A consumer that only trusts observed ends filters on this."""


@dataclasses.dataclass(frozen=True)
class ClosedBinding:
    """
    One binding, complete and ready to resolve against.

    Attributes
    ----------
    key : str
        What was bound, typically a MAC address.
    attributes : dict
        What it was bound to, typically the switch, port and VLAN.
    bind_start_ns : int
        First observation of this binding. Inclusive.
    bind_end_ns : int
        End of the interval. Exclusive, so it always sits at least one tick past the last observation.
    end_reason : str
        How the end was arrived at. Only `EXPLICIT` means the source stated it; the rest are inferred, and
        `INFERRED_REASONS` collects them.
    observations : int
        Samples that fell inside this binding, so a consumer can weigh a well-attested binding against one seen
        once.
    last_seen_ns : int
        The last observation, which is where an inferred end was derived from.
    """

    key: str
    attributes: dict
    bind_start_ns: int
    bind_end_ns: int
    end_reason: str
    observations: int
    last_seen_ns: int

    @property
    def duration_ns(self) -> int:
        """Length of the interval."""
        return self.bind_end_ns - self.bind_start_ns

    @property
    def end_observed(self) -> bool:
        """Whether the source stated the end, rather than it being deduced."""
        return self.end_reason == EXPLICIT


@dataclasses.dataclass(frozen=True)
class OpenBinding:
    """
    A binding that is still open: what is known so far, offered provisionally.

    A closed binding is the honest unit of lineage, but a device plugged in now is not resolvable until its binding
    closes, which by default is when it moves, vanishes from a snapshot, or goes quiet for the idle timeout. For a
    forensic replay that is correct. For an analyst asking where an address is right now, half an hour of "unknown"
    is not. An open binding carries no end; the consumer caps it with an explicit assumed duration, which is what
    `BindingTable.from_dataframe(open_end_duration_ns=...)` exists for, and the closed record that follows, with the
    same key and start, supersedes it.

    Attributes
    ----------
    key : str
        What is bound.
    attributes : dict
        What it is bound to.
    bind_start_ns : int
        First observation. Inclusive.
    last_seen_ns : int
        Most recent observation. There is no end yet.
    observations : int
        Samples so far.
    """

    key: str
    attributes: dict
    bind_start_ns: int
    last_seen_ns: int
    observations: int


@dataclasses.dataclass(frozen=True)
class ObserveResult:
    """
    The outcome of observing one binding sample.

    Attributes
    ----------
    closed : list of `ClosedBinding`
        Bindings this observation ended. Displacing a key closes one; evicting to stay within `max_open` may close
        another, which is emitted rather than dropped so that no binding is lost silently.
    out_of_order : bool
        The sample's event time was not after the open binding's last observation. State is left untouched.
    opened : bool
        This observation opened a new binding, either the key's first or one that replaced a displaced or
        conflicting binding. An observation that extends an open binding does not set this, so a consumer emitting
        provisional records emits one per binding rather than one per sample.
    """

    closed: list
    out_of_order: bool
    opened: bool = False


@dataclasses.dataclass
class _OpenBinding:
    key: str
    attributes: dict
    start_ns: int
    last_seen_ns: int
    observations: int


class BindingCloser:
    """
    Per-key open binding state, emitting intervals as they close.

    The closer holds at most one open binding per key, so its memory is bounded by the number of simultaneously
    bound keys rather than by the stream. Results depend only on the sequence of calls it has been shown, so
    replaying a stream reproduces the same intervals.

    Parameters
    ----------
    attribute_names : list of str
        Attributes that constitute the binding target, typically `["switch_id", "port_id", "vlan_id"]`. A sample
        whose attributes differ from the open binding's displaces it; one whose attributes match extends it.
    idle_timeout_ns : int, default = 30 minutes
        Silence after which `expire` closes a binding.
    max_open : int, default = 500000
        Open bindings retained before the least recently seen is forced closed and emitted.

    Notes
    -----
    An inferred end is the earliest one consistent with the observations, so bindings can leave gaps between them.
    That is deliberate: a gap resolves to nothing, which tells an analyst that the answer is unknown, whereas
    stretching a binding to meet the next one would have it cover a period the key may have already left and hand
    back a confident wrong answer.

    The closer never discards an open binding. Eviction under `max_open` emits it with reason `EVICTED`, so a
    binding that outlived the closer's memory still reaches the table as a short interval rather than vanishing.
    """

    def __init__(self,
                 attribute_names: typing.Sequence[str],
                 idle_timeout_ns: int = DEFAULT_IDLE_TIMEOUT_NS,
                 max_open: int = DEFAULT_MAX_OPEN):
        if (len(attribute_names) == 0):
            raise ValueError("At least one attribute name is required; a binding with no target binds nothing")

        if (idle_timeout_ns <= 0):
            raise ValueError(f"idle_timeout_ns must be positive, received {idle_timeout_ns}")

        if (max_open <= 0):
            raise ValueError(f"max_open must be positive, received {max_open}")

        self._attribute_names = list(attribute_names)
        self._idle_timeout_ns = idle_timeout_ns
        self._max_open = max_open

        self._open: collections.OrderedDict[str, _OpenBinding] = collections.OrderedDict()

    @property
    def attribute_names(self) -> list[str]:
        """Attributes that make up the binding target."""
        return list(self._attribute_names)

    @property
    def open_count(self) -> int:
        """Bindings currently open."""
        return len(self._open)

    def open_binding(self, key: str) -> typing.Optional[OpenBinding]:
        """
        What is currently known about a key's open binding, or `None` if nothing is open for it.

        Parameters
        ----------
        key : str
            The bound key.

        Returns
        -------
        `OpenBinding` or None
        """
        state = self._open.get(key)

        if (state is None):
            return None

        return OpenBinding(key=state.key,
                           attributes=dict(state.attributes),
                           bind_start_ns=state.start_ns,
                           last_seen_ns=state.last_seen_ns,
                           observations=state.observations)

    def _target(self, attributes: dict) -> dict:
        return {name: attributes.get(name) for name in self._attribute_names}

    @staticmethod
    def _close(state: _OpenBinding, end_ns: int, reason: str) -> ClosedBinding:
        # The interval is half-open, so an end at the last observation would exclude it, and a key seen once would
        # produce a zero-width interval covering nothing. One tick past is the shortest honest interval.
        floor_ns = state.last_seen_ns + MINIMUM_TICK_NS

        return ClosedBinding(key=state.key,
                             attributes=dict(state.attributes),
                             bind_start_ns=state.start_ns,
                             bind_end_ns=max(end_ns, floor_ns),
                             end_reason=reason,
                             observations=state.observations,
                             last_seen_ns=state.last_seen_ns)

    def observe(self, key: str, event_time_ns: int, attributes: dict) -> ObserveResult:
        """
        Record one binding observation.

        Parameters
        ----------
        key : str
            What is bound, typically a MAC address.
        event_time_ns : int
            When it was observed, in nanoseconds since the epoch. Event time, never ingest time.
        attributes : dict
            Where it was observed. Compared against the open binding's target to decide whether this extends the
            binding or displaces it.

        Returns
        -------
        `ObserveResult`
        """
        target = self._target(attributes)
        state = self._open.get(key)
        closed = []

        if (state is not None):
            if (event_time_ns < state.last_seen_ns):
                # Admitting this would move a binding's end backwards, or reopen one already closed against a
                # later observation, making the emitted intervals depend on delivery order. An equal timestamp is
                # admitted: two switches polled in the same second both reporting one MAC is the spoofing signal,
                # not a late sample, and turning it away would discard the strongest evidence layer 2 produces.
                return ObserveResult(closed=[], out_of_order=True)

            if (state.attributes == target):
                state.last_seen_ns = event_time_ns
                state.observations += 1
                self._open.move_to_end(key)

                return ObserveResult(closed=[], out_of_order=False)

            # Seen somewhere else. When the new sighting is later, the move happened between the two observations
            # and closing at the earlier bound leaves the interval between them uncovered rather than claiming the
            # key was still here. When it is simultaneous, nothing can be said about order, the two intervals
            # overlap by one tick, and the reason says so.
            reason = CONFLICT if event_time_ns == state.last_seen_ns else DISPLACED
            closed.append(self._close(state, state.last_seen_ns, reason))

        self._open[key] = _OpenBinding(key=key,
                                       attributes=target,
                                       start_ns=event_time_ns,
                                       last_seen_ns=event_time_ns,
                                       observations=1)
        self._open.move_to_end(key)
        closed.extend(self._evict())

        return ObserveResult(closed=closed, out_of_order=False, opened=True)

    def close(self, key: str, event_time_ns: int) -> typing.Optional[ClosedBinding]:
        """
        End a binding because the source said it ended.

        This is the only end that is a fact rather than an inference, so the given time is used directly, subject
        only to the half-open floor that keeps the last observation inside its own interval.

        Parameters
        ----------
        key : str
            The binding to end.
        event_time_ns : int
            When the source says it ended.

        Returns
        -------
        `ClosedBinding` or `None`
            `None` when no binding is open for the key, which is the case for a duplicate or unmatched stop record.
        """
        state = self._open.pop(key, None)

        if (state is None):
            return None

        return self._close(state, event_time_ns, EXPLICIT)

    def reconcile(self,
                  event_time_ns: int,
                  present_keys: typing.Iterable[str],
                  scope: typing.Optional[dict] = None) -> list[ClosedBinding]:
        """
        Close bindings a full-table snapshot no longer lists.

        Parameters
        ----------
        event_time_ns : int
            When the snapshot was taken. A binding observed since then is left alone: the snapshot's silence about
            it is stale news, and closing on it would end a binding that is demonstrably still live.
        present_keys : iterable of str
            Every key the snapshot listed.
        scope : dict, optional
            Attribute values the snapshot covers, for example `{"switch_id": "sw1"}`. Only open bindings matching
            every entry are candidates for closure. Without it the snapshot is treated as covering everything, so
            a per-device snapshot passed with no scope would close every other device's bindings.

        Returns
        -------
        list of `ClosedBinding`
        """
        present = set(present_keys)
        scope = scope or {}

        stale = [
            key for (key, state) in self._open.items()
            if key not in present and state.last_seen_ns < event_time_ns and all(
                state.attributes.get(name) == value for (name, value) in scope.items())
        ]

        # Zero as the end time lets the half-open floor place it one tick past the last observation, which is the
        # earliest the key can have left given that the snapshot no longer lists it.
        return [self._close(self._open.pop(key), 0, SNAPSHOT_ABSENT) for key in stale]

    def expire(self, now_ns: int) -> list[ClosedBinding]:
        """
        Close bindings that have gone quiet for longer than `idle_timeout_ns`.

        Parameters
        ----------
        now_ns : int
            Current event time.

        Returns
        -------
        list of `ClosedBinding`
        """
        horizon = now_ns - self._idle_timeout_ns
        stale = [key for (key, state) in self._open.items() if state.last_seen_ns < horizon]

        return [self._close(self._open.pop(key), 0, IDLE_TIMEOUT) for key in stale]

    def drain(self) -> list[ClosedBinding]:
        """
        Close every open binding, for a shutdown or the end of a replay.

        Returns
        -------
        list of `ClosedBinding`
            Emitted in the order the bindings were last seen, oldest first, so the output is reproducible.
        """
        drained = [self._close(state, 0, DRAINED) for state in self._open.values()]
        self._open.clear()

        return drained

    def _evict(self) -> list[ClosedBinding]:
        evicted = []

        while (len(self._open) > self._max_open):
            (_, state) = self._open.popitem(last=False)
            # Emitted rather than dropped: a binding that outlived the closer's memory still happened.
            evicted.append(self._close(state, 0, EVICTED))

        return evicted
