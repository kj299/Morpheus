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
Bitemporal facts: what was true about a person or an asset, and when this system learned it.

The TC-0 context store holds facts that are true for an interval -- a principal belonged to Finance from one date to
another, a server held confidential data from the day it was reclassified -- and it holds them twice over. **Valid
time** is when the fact was true in the world. **Transaction time** is when the system recorded it. The guide makes
both a hard requirement, because an investigation asks two different questions and a store with one axis can answer
only one of them: "was this user in Finance on March 3rd, as we know it now" and "... as we knew it on March 3rd".
The first is what an analyst needs to judge the event; the second is what a detection that fired on March 3rd
actually saw, and the only way to explain why it fired.

**The store is an append-only log of versions.** A version is one statement about one fact: this key had these
values over this valid interval, as recorded at this instant. A correction is a new version recorded later, never an
edit of an old one, so every answer the store has ever given can be given again by asking with the same knowledge
horizon. A **retraction** is a version saying the fact did not hold over its interval -- a leaver's group
membership, a decommissioned host -- because a later version covering only part of an earlier one's interval would
otherwise leave the rest of the earlier one standing.

**Transaction time is carried as the instant each version was recorded; its end is derived rather than stored.** A
version stops being current, for a given valid instant, when a later-recorded version covers that instant. Storing
that end on the record would mean rewriting a record every time something later superseded part of it, which is
exactly the mutation an append-only log exists to avoid, and it would still be per-instant rather than per-record,
because a correction usually covers only part of what it corrects. Every as-known-at question is answered by filtering
on the recorded instant, so nothing is lost by deriving it.

**The recorded instant comes from the source, never from the pipeline.** Stamping the moment a row reached this
process would make the answer to every as-known-at question depend on when the pipeline happened to run, and a replay
would disagree with the original. A row without one is refused.

**Resolution is a fixed rule.** Among the versions covering an instant and recorded no later than the horizon, the
most recently recorded wins; then the later valid start; then the later valid end, an open end counting as latest;
then the change and the values compared as text, which only separates records that are contradictory in every other
respect. The last component exists so the winner is a function of the data rather than of insertion order. Two
versions of one key recorded at the same instant, over overlapping intervals, with different content, are a genuine
contradiction in the source and are counted so it shows.
"""

import dataclasses
import logging
import typing

from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.entity_key import compose_key
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.lineage import event_uid

logger = logging.getLogger(__name__)

ASSERT = "assert"
"""The fact held over the version's valid interval."""

RETRACT = "retract"
"""The fact did not hold over the version's valid interval, whatever was recorded before."""

CHANGES = (ASSERT, RETRACT)

MEMBERSHIP = "membership"
"""The kind of a group or role membership: one fact per principal and group, each with its own dates."""

# The columns a context producer writes and `BitemporalStore.from_records` reads. Stated once so the two agree.
CONTEXT_KIND = "context_kind"
CONTEXT_ENTITY = "context_entity"
CONTEXT_KEY = "context_key"
CONTEXT_ATTRIBUTES = "context_attributes"
CONTEXT_UID = "context_uid"
CONTEXT_REFUSED = "context_refused"
VALID_FROM = "valid_from"
VALID_TO = "valid_to"
RECORDED_AT = "recorded_at"
CHANGE = "change"

ATTRIBUTE_SEPARATOR = ","
"""Joins the attribute names a record carries in `context_attributes`."""

# Refusal reasons. A refused row stays in the output, marked, so the count is visible rather than silently smaller.
NO_ENTITY = "no_entity"
NO_VALID_FROM = "no_valid_from"
NO_RECORDED_AT = "no_recorded_at"
INVERTED_INTERVAL = "inverted_interval"
UNKNOWN_CHANGE = "unknown_change"

_OPEN_END = 2**63
"""Sorts an open valid end after every closed one."""


@dataclasses.dataclass(frozen=True)
class Version:
    """
    One recorded statement about one fact.

    Attributes
    ----------
    kind : str
        What sort of fact this is, for example `profile`, `membership` or `asset`.
    entity : str
        The principal or asset the fact is about. Several keys can share one entity; a principal has one profile
        and a membership per group.
    key : str
        The fact's identity. Versions of one key supersede each other; versions of different keys never do.
    valid_from_ns : int
        Start of the valid interval, in nanoseconds since the epoch. Inclusive.
    valid_to_ns : int or None
        End of the valid interval. Exclusive. `None` for a fact that holds until something says otherwise.
    recorded_ns : int
        When the source recorded the version. The start of its transaction time.
    change : str
        `assert` or `retract`.
    values : tuple
        `(name, value)` pairs, sorted by name, with values as normalized text.
    uid : str
        Content-addressed identifier, so an answer can be traced to the exact version behind it.
    """

    kind: str
    entity: str
    key: str
    valid_from_ns: int
    valid_to_ns: typing.Optional[int]
    recorded_ns: int
    change: str
    values: tuple
    uid: str

    @property
    def asserted(self) -> bool:
        """Whether this version says the fact held."""
        return self.change == ASSERT

    @property
    def attributes(self) -> dict:
        """The values as a mapping."""
        return dict(self.values)

    def covers(self, valid_ns: int) -> bool:
        """Whether the half-open valid interval contains `valid_ns`."""
        return self.valid_from_ns <= valid_ns and (self.valid_to_ns is None or valid_ns < self.valid_to_ns)

    def overlaps(self, other: "Version") -> bool:
        """Whether the two valid intervals share an instant."""
        mine = _OPEN_END if self.valid_to_ns is None else self.valid_to_ns
        theirs = _OPEN_END if other.valid_to_ns is None else other.valid_to_ns

        return self.valid_from_ns < theirs and other.valid_from_ns < mine


def refusal_reason(entity: typing.Any,
                   valid_from_ns: typing.Optional[int],
                   valid_to_ns: typing.Optional[int],
                   recorded_ns: typing.Optional[int],
                   change: typing.Any) -> typing.Optional[str]:
    """
    Why a source record cannot become a version, or `None` if it can.

    Parameters
    ----------
    entity : any
        The principal or asset. Missing is a refusal.
    valid_from_ns : int or None
        Start of the valid interval, already converted.
    valid_to_ns : int or None
        End of the valid interval. The only instant that may be missing.
    recorded_ns : int or None
        When the source recorded the record.
    change : any
        `assert` or `retract`.

    Returns
    -------
    str or None
        One of the refusal reasons defined in this module.
    """
    if (normalize_text(entity) is None):
        return NO_ENTITY

    if (valid_from_ns is None):
        return NO_VALID_FROM

    if (recorded_ns is None):
        return NO_RECORDED_AT

    if (valid_to_ns is not None and valid_to_ns <= valid_from_ns):
        return INVERTED_INTERVAL

    if (change not in CHANGES):
        return UNKNOWN_CHANGE

    return None


def make_version(kind: str,
                 entity: typing.Any,
                 valid_from_ns: int,
                 valid_to_ns: typing.Optional[int],
                 recorded_ns: int,
                 change: str = ASSERT,
                 values: typing.Optional[dict] = None,
                 key_parts: typing.Sequence[typing.Any] = ()) -> Version:
    """
    Build a version, validating it and deriving its key and identifier.

    Parameters
    ----------
    kind : str
        The fact's kind.
    entity : any
        The principal or asset.
    valid_from_ns : int
        Start of the valid interval.
    valid_to_ns : int or None
        End of the valid interval, or `None` for an open one.
    recorded_ns : int
        When the source recorded the version.
    change : str, default = "assert"
        `assert` or `retract`.
    values : dict, optional
        Attribute name to value. Values are normalized to text.
    key_parts : sequence, optional
        Parts identifying the fact beyond its entity, for example the group of a membership. The key is the entity
        and these parts composed with the shared entity-key rule.

    Returns
    -------
    `Version`

    Raises
    ------
    ValueError
        If the record would be refused; the message names the reason.
    """
    reason = refusal_reason(entity, valid_from_ns, valid_to_ns, recorded_ns, change)

    if (reason is not None):
        raise ValueError(f"cannot record a {kind} version: {reason}")

    entity_text = normalize_text(entity)
    key = compose_key([entity_text, *key_parts])

    if (key is None):
        raise ValueError(f"cannot record a {kind} version: a key part is missing")

    normalized = tuple(sorted((str(name), normalize_text(value)) for (name, value) in (values or {}).items()))
    flattened = [part for pair in normalized for part in (pair[0], "" if pair[1] is None else pair[1])]
    uid = event_uid(kind,
                    key,
                    int(valid_from_ns),
                    "" if valid_to_ns is None else int(valid_to_ns),
                    int(recorded_ns),
                    change,
                    *flattened)

    return Version(kind=str(kind),
                   entity=entity_text,
                   key=key,
                   valid_from_ns=int(valid_from_ns),
                   valid_to_ns=None if valid_to_ns is None else int(valid_to_ns),
                   recorded_ns=int(recorded_ns),
                   change=change,
                   values=normalized,
                   uid=uid)


def _precedence(version: Version) -> tuple:
    """The fixed rule, as a sort key. The winner is the maximum. See the module docstring."""
    return (version.recorded_ns,
            version.valid_from_ns,
            _OPEN_END if version.valid_to_ns is None else version.valid_to_ns,
            version.change,
            tuple((name, "" if value is None else value) for (name, value) in version.values))


class BitemporalStore:
    """
    Versions of facts about principals and assets, answering what held at an instant as known at another.

    Parameters
    ----------
    name : str
        Identifier for this store, for log messages.
    versions : iterable of `Version`, optional
        The initial versions. Order is irrelevant.
    """

    def __init__(self, name: str, versions: typing.Iterable[Version] = ()):
        if (not name):
            raise ValueError("name is required")

        self._name = name
        self._by_key: dict[str, list[Version]] = {}
        self._keys_by_entity: dict[str, set] = {}
        self._kind_by_key: dict[str, str] = {}
        self._attributes_by_kind: dict[str, set] = {}
        self._sorted: set = set()
        self._size = 0

        for version in versions:
            self.add(version)

    @property
    def name(self) -> str:
        """Identifier for this store."""
        return self._name

    @property
    def size(self) -> int:
        """Number of versions held."""
        return self._size

    @property
    def key_count(self) -> int:
        """Number of distinct facts."""
        return len(self._by_key)

    def kinds(self) -> list[str]:
        """Every kind the store holds, sorted."""
        return sorted(self._attributes_by_kind)

    def attribute_names(self, kind: str) -> list[str]:
        """The attribute names recorded for a kind, sorted."""
        return sorted(self._attributes_by_kind.get(kind, ()))

    def add(self, version: Version):
        """
        Record a version.

        Raises
        ------
        ValueError
            If the key has already been recorded under a different kind, which would let a membership supersede a
            profile.
        """
        existing = self._kind_by_key.get(version.key)

        if (existing is not None and existing != version.kind):
            raise ValueError(f"key {version.key!r} is recorded as {existing!r} and cannot also be {version.kind!r}")

        self._kind_by_key[version.key] = version.kind
        self._by_key.setdefault(version.key, []).append(version)
        self._keys_by_entity.setdefault(version.entity, set()).add(version.key)
        self._attributes_by_kind.setdefault(version.kind, set()).update(name for (name, _) in version.values)
        self._sorted.discard(version.key)
        self._size += 1

    def history(self, key: str) -> list[Version]:
        """Every version of a fact, in precedence order, lowest first."""
        versions = self._by_key.get(key, [])

        if (key not in self._sorted):
            versions.sort(key=_precedence)
            self._sorted.add(key)

        return list(versions)

    def winner(self, key: str, valid_ns: int, known_ns: typing.Optional[int] = None) -> typing.Optional[Version]:
        """
        The version that decides a fact at an instant, retraction or not.

        Parameters
        ----------
        key : str
            The fact.
        valid_ns : int
            The instant the question is about.
        known_ns : int, optional
            The knowledge horizon: only versions recorded at or before it are considered. `None` considers
            everything the store holds.

        Returns
        -------
        `Version` or None
            `None` when nothing recorded by the horizon covers the instant.
        """
        best = None

        for version in reversed(self.history(key)):
            if (known_ns is not None and version.recorded_ns > known_ns):
                continue

            if (version.covers(valid_ns)):
                best = version
                break

        return best

    def resolve(self, key: str, valid_ns: int, known_ns: typing.Optional[int] = None) -> typing.Optional[Version]:
        """
        What held for a fact at an instant, as known at the horizon, or `None` if nothing did.

        A retraction answers `None`, the same as no record at all; `winner` distinguishes the two.
        """
        version = self.winner(key, valid_ns, known_ns)

        return version if version is not None and version.asserted else None

    def facts_about(self, entity: typing.Any, valid_ns: int, known_ns: typing.Optional[int] = None) -> list[Version]:
        """
        Every fact that held about a principal or asset at an instant, as known at the horizon.

        Returns
        -------
        list of `Version`
            Sorted by kind and key, so the order is a function of the data.
        """
        entity_text = normalize_text(entity)

        if (entity_text is None):
            return []

        held = []

        for key in sorted(self._keys_by_entity.get(entity_text, ())):
            version = self.resolve(key, valid_ns, known_ns)

            if (version is not None):
                held.append(version)

        return sorted(held, key=lambda version: (version.kind, version.key))

    def contradictions(self) -> int:
        """
        Facts with two versions recorded at the same instant, over overlapping intervals, saying different things.

        Resolution is still deterministic for them; the count exists because a source that contradicts itself in a
        single transaction is a source whose every answer is suspect.
        """
        count = 0

        for key in sorted(self._by_key):
            versions = self.history(key)
            found = False

            for (position, version) in enumerate(versions):
                for other in versions[position + 1:]:
                    if (other.recorded_ns != version.recorded_ns):
                        break

                    if (version.overlaps(other) and (version.change, version.values) != (other.change, other.values)):
                        found = True
                        break

                if (found):
                    break

            count += int(found)

        return count

    def snapshot_changes(self, kind: str, facts: typing.Iterable[dict], recorded_ns: int) -> list[Version]:
        """
        The versions a full snapshot adds to what the store already knows.

        A snapshot restates every fact of a kind as the source holds it at `recorded_ns`. Most of it restates what
        is already known, and recording it again would bury every real change under a daily copy of everything.
        This returns only the difference: an assertion for each fact that is new or whose values or valid interval
        differ from what the store knew at that instant, and a retraction, valid from the snapshot onward, for each
        fact of the kind that the store holds as current and the snapshot omits. The store itself is not changed.

        Parameters
        ----------
        kind : str
            The kind the snapshot covers. Facts of other kinds are never retracted by it.
        facts : iterable of dict
            Each with `entity`, `valid_from_ns`, and optionally `valid_to_ns`, `values` and `key_parts`.
        recorded_ns : int
            When the source took the snapshot.

        Returns
        -------
        list of `Version`
            Sorted by key.
        """
        changes = []
        present = set()

        for fact in facts:
            candidate = make_version(kind,
                                     fact["entity"],
                                     fact["valid_from_ns"],
                                     fact.get("valid_to_ns"),
                                     recorded_ns,
                                     ASSERT,
                                     fact.get("values"),
                                     fact.get("key_parts", ()))
            present.add(candidate.key)
            current = self.winner(candidate.key, candidate.valid_from_ns, known_ns=recorded_ns)

            unchanged = (current is not None and current.asserted and current.values == candidate.values
                         and current.valid_from_ns == candidate.valid_from_ns
                         and current.valid_to_ns == candidate.valid_to_ns)

            if (not unchanged):
                changes.append(candidate)

        for key in sorted(self._by_key):
            if (key in present or self._kind_by_key[key] != kind):
                continue

            current = self.resolve(key, recorded_ns, known_ns=recorded_ns)

            if (current is None):
                continue

            changes.append(
                make_version(kind,
                             current.entity,
                             recorded_ns,
                             None,
                             recorded_ns,
                             RETRACT,
                             dict(current.values),
                             key_parts_of(current)))

        return sorted(changes, key=lambda version: version.key)

    @classmethod
    def from_records(cls, name: str, records: typing.Iterable[dict]) -> "BitemporalStore":
        """
        Rebuild a store from the rows a context producer emitted.

        Refused rows are skipped. Each row's own `context_attributes` names which of its columns are its values, so
        rows of different kinds can share one frame.

        Parameters
        ----------
        name : str
            Identifier for the store.
        records : iterable of dict
            Rows carrying the columns this module names, as `DataFrame.to_dict("records")` produces them.
        """
        versions = []

        for record in records:
            if (normalize_text(record.get(CONTEXT_REFUSED)) is not None
                    or normalize_text(record.get(CONTEXT_UID)) is None):
                continue

            names = normalize_text(record.get(CONTEXT_ATTRIBUTES))
            names = [] if names is None else names.split(ATTRIBUTE_SEPARATOR)
            valid_to = record.get(VALID_TO)
            valid_to = None if normalize_text(valid_to) is None else int(valid_to)
            key = normalize_text(record[CONTEXT_KEY])
            entity = normalize_text(record[CONTEXT_ENTITY])
            key_parts = _parts_after(entity, key)

            version = make_version(normalize_text(record[CONTEXT_KIND]),
                                   entity,
                                   int(record[VALID_FROM]),
                                   valid_to,
                                   int(record[RECORDED_AT]),
                                   normalize_text(record[CHANGE]), {name: record.get(name)
                                                                    for name in names},
                                   key_parts)

            if (version.uid != normalize_text(record[CONTEXT_UID])):
                raise ValueError(f"row {record[CONTEXT_UID]!r} does not reproduce its own identifier; it was altered "
                                 "after it was produced")

            versions.append(version)

        store = cls(name, versions)
        contradictions = store.contradictions()

        if (contradictions > 0):
            logger.warning(
                "Context store %r has %d facts with contradictory versions recorded at the same instant. Resolution "
                "is still deterministic, but the source is contradicting itself.",
                name,
                contradictions)

        return store


def context_columns(kind: typing.Union[str, typing.Sequence[str]],
                    entities: typing.Sequence,
                    key_parts: typing.Sequence[typing.Sequence],
                    values: typing.Sequence[dict],
                    valid_from: typing.Sequence,
                    valid_to: typing.Sequence,
                    recorded: typing.Sequence,
                    changes: typing.Sequence,
                    time_unit: str = "ns") -> tuple[dict, dict]:
    """
    Turn a batch of source records into the columns a context producer writes.

    Shared by the TC-0 producer stages so that both write exactly what `BitemporalStore.from_records` reads. Every
    input row yields an output row; a refused row keeps its entity and carries its reason instead of an identifier.

    Parameters
    ----------
    kind : str or sequence of str
        The kind of every row, or of each row.
    entities : sequence
        Per row, the principal or asset.
    key_parts : sequence of sequences
        Per row, the key parts beyond the entity.
    values : sequence of dict
        Per row, the attribute mapping.
    valid_from : sequence
        Per row, the raw start of the valid interval, in any form `to_epoch_ns` accepts.
    valid_to : sequence
        Per row, the raw end of the valid interval, or a missing value for an open one.
    recorded : sequence
        Per row, the raw instant the source recorded it.
    changes : sequence
        Per row, `assert` or `retract`; a missing value means `assert`.
    time_unit : str, default = "ns"
        Unit for numeric instants.

    Returns
    -------
    tuple of (dict, dict)
        The output columns by name, and the count of refused rows by reason.
    """

    def instant(value):
        try:
            return to_epoch_ns(value, time_unit=time_unit)
        except ValueError:
            return None

    kinds = [kind] * len(entities) if isinstance(kind, str) else list(kind)
    columns: dict = {
        name: []
        for name in (CONTEXT_KIND, CONTEXT_ENTITY, CONTEXT_KEY, CONTEXT_ATTRIBUTES, CONTEXT_UID, CONTEXT_REFUSED,
                     VALID_FROM, VALID_TO, RECORDED_AT, CHANGE)
    }
    refused: dict = {}

    for (position, entity) in enumerate(entities):
        start = instant(valid_from[position])
        end = instant(valid_to[position])
        at = instant(recorded[position])
        change = normalize_text(changes[position])
        change = ASSERT if change is None else change.lower()
        reason = refusal_reason(entity, start, end, at, change)
        version = None

        if (reason is None):
            try:
                version = make_version(kinds[position],
                                       entity,
                                       start,
                                       end,
                                       at,
                                       change,
                                       values[position],
                                       key_parts[position])
            except ValueError:
                reason = NO_ENTITY

        if (reason is not None):
            refused[reason] = refused.get(reason, 0) + 1

        columns[CONTEXT_KIND].append(kinds[position])
        columns[CONTEXT_ENTITY].append(normalize_text(entity))
        columns[CONTEXT_KEY].append(None if version is None else version.key)
        columns[CONTEXT_ATTRIBUTES].append(ATTRIBUTE_SEPARATOR.join(sorted(values[position])) or None)
        columns[CONTEXT_UID].append(None if version is None else version.uid)
        columns[CONTEXT_REFUSED].append(reason)
        columns[VALID_FROM].append(start)
        columns[VALID_TO].append(end)
        columns[RECORDED_AT].append(at)
        columns[CHANGE].append(change)

    return (columns, refused)


def _parts_after(entity: str, key: str) -> list[str]:
    """The key parts beyond the entity, recovered from a composed key."""
    if (key == entity):
        return []

    prefix = compose_key([entity, "x"])[:-1]

    if (not key.startswith(prefix)):
        raise ValueError(f"key {key!r} does not begin with its entity {entity!r}")

    return [key[len(prefix):]]


def key_parts_of(version: Version) -> list[str]:
    """The parts of a version's key beyond its entity: the group of a membership, nothing for a profile."""
    return _parts_after(version.entity, version.key)
