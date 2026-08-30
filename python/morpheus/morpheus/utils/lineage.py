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
Deterministic lineage identifiers.

Every identifier produced here is a pure function of values already present in the data, which is what allows a replay
of the same input to reconstruct the same lineage. Nothing in this module reads a clock, a random source, or any
process-local state.

See the `Predictive Behavioral Analytics Across OSI Layers 1-7
<../developer_guide/guides/11_predictive_behavioral_analytics_osi.md>`_ guide for how these identifiers are consumed by
a SIEM.
"""

import hashlib
import typing

UNIT_SEPARATOR = "\x1f"
"""
Delimiter placed between fields before hashing.

A delimiter is required, not cosmetic: without it the field pairs `("ab", "c")` and `("a", "bc")` concatenate to the
same string and therefore collide. The ASCII unit separator is used because it does not occur in identifier values.
"""

DEFAULT_DIGEST_LENGTH = 32
"""Number of hexadecimal characters retained from a SHA-256 digest. 32 characters is 128 bits."""

_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"


def _digest(parts: typing.Iterable[typing.Any], digest_length: int = DEFAULT_DIGEST_LENGTH) -> str:
    """
    Hash a field sequence into a truncated hexadecimal SHA-256 digest.

    Parameters
    ----------
    parts : iterable
        Fields to hash. Each is converted with `str` and joined with `UNIT_SEPARATOR`.
    digest_length : int, default = 32
        Number of hexadecimal characters to retain. Must be between 1 and 64.

    Returns
    -------
    str
        The truncated digest.
    """
    if (not 1 <= digest_length <= 64):
        raise ValueError(f"digest_length must be between 1 and 64, received {digest_length}")

    payload = UNIT_SEPARATOR.join(str(part) for part in parts)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:digest_length]


def event_uid(*parts: typing.Any, digest_length: int = DEFAULT_DIGEST_LENGTH) -> str:
    """
    Compute a content addressed identifier for a single telemetry record.

    The conventional field set is `(collector_id, schema_version, origin_hash, collector_seq)`. Deriving the identifier
    from the collector sequence rather than from a row index makes it independent of how the collector batched the
    record, so the same record produces the same identifier no matter how it arrives.

    Parameters
    ----------
    *parts : any
        Fields identifying the record.
    digest_length : int, default = 32
        Number of hexadecimal characters to retain.

    Returns
    -------
    str
        The event identifier.
    """
    if (len(parts) == 0):
        raise ValueError("event_uid requires at least one field")

    return _digest(parts, digest_length=digest_length)


def link_uid(parent_uid: str,
             child_uid: str,
             relation: str,
             join_method: str,
             digest_length: int = DEFAULT_DIGEST_LENGTH) -> str:
    """
    Compute an identifier for a parent-child lineage edge.

    Parameters
    ----------
    parent_uid : str
        `event_uid` of the parent record, typically the lower OSI layer.
    child_uid : str
        `event_uid` of the child record.
    relation : str
        Semantic relationship, for example `carried_by` or `authenticated_via`.
    join_method : str
        How the edge was established, for example `hard:flow_id` or `soft:dhcp_lease`. Recording this on the edge is
        what later distinguishes an exact attribution from an inferred one.
    digest_length : int, default = 32
        Number of hexadecimal characters to retain.

    Returns
    -------
    str
        The edge identifier.
    """
    return _digest((parent_uid, child_uid, relation, join_method), digest_length=digest_length)


def merkle_root(uids: typing.Iterable[str]) -> str:
    """
    Compute a Merkle root over a set of event identifiers.

    The inputs are deduplicated and sorted before the tree is built, which makes the root a function of chain
    membership alone. That is the property that survives out-of-order delivery and replay: the same set of events
    always produces the same root, regardless of the order in which they arrived.

    Leaf and interior nodes are domain separated with distinct prefixes so that an interior digest can never be
    substituted for a leaf.

    Parameters
    ----------
    uids : iterable of str
        Event identifiers forming the chain.

    Returns
    -------
    str
        Hexadecimal SHA-256 root. An empty input yields the digest of the empty leaf, which is a well defined constant.
    """
    leaves = [hashlib.sha256(_LEAF_PREFIX + uid.encode("utf-8")).digest() for uid in sorted(set(uids))]

    if (len(leaves) == 0):
        return hashlib.sha256(_LEAF_PREFIX).hexdigest()

    while (len(leaves) > 1):
        pairs = []
        for i in range(0, len(leaves) - 1, 2):
            pairs.append(hashlib.sha256(_NODE_PREFIX + leaves[i] + leaves[i + 1]).digest())

        if (len(leaves) % 2 == 1):
            # Promote the odd node unchanged rather than duplicating it, which would make a tree of N nodes and one of
            # N+1 nodes with a repeated tail indistinguishable.
            pairs.append(leaves[-1])

        leaves = pairs

    return leaves[0].hex()


def lineage_id(root_entity_key: str,
               window_id: typing.Any,
               chain_root: str,
               digest_length: int = DEFAULT_DIGEST_LENGTH) -> str:
    """
    Compute the identifier for a correlation chain.

    Parameters
    ----------
    root_entity_key : str
        Entity the chain is anchored on, for example the user principal or source address.
    window_id : any
        Deterministic window identifier, see `window_id_from_timestamp`.
    chain_root : str
        Merkle root of the chain's event identifiers, see `merkle_root`.
    digest_length : int, default = 32
        Number of hexadecimal characters to retain.

    Returns
    -------
    str
        The chain identifier.
    """
    return _digest((root_entity_key, window_id, chain_root), digest_length=digest_length)


def window_id_from_timestamp(event_time_ns: int, period_ns: int, epoch_ns: int = 0) -> int:
    """
    Assign an event to a half-open window `[epoch + k*period, epoch + (k+1)*period)`.

    Anchoring to a fixed absolute epoch rather than to the first observed event is what makes window identity
    independent of where a replay starts.

    Parameters
    ----------
    event_time_ns : int
        Event time in nanoseconds since the Unix epoch.
    period_ns : int
        Window length in nanoseconds. Must be positive.
    epoch_ns : int, default = 0
        Absolute anchor in nanoseconds since the Unix epoch.

    Returns
    -------
    int
        The window index. Floor division is used, so windows before the anchor receive negative indices rather than
        being folded onto index zero.
    """
    if (period_ns <= 0):
        raise ValueError(f"period_ns must be positive, received {period_ns}")

    return (event_time_ns - epoch_ns) // period_ns


def event_uid_series(columns: typing.Sequence[typing.Sequence],
                     digest_length: int = DEFAULT_DIGEST_LENGTH) -> list[str]:
    """
    Compute `event_uid` values for a batch of records.

    Parameters
    ----------
    columns : sequence of sequences
        One sequence per identifying field, all of equal length. Field order is significant and must match across
        every producer that is expected to agree.
    digest_length : int, default = 32
        Number of hexadecimal characters to retain.

    Returns
    -------
    list of str
        One identifier per row.
    """
    if (len(columns) == 0):
        raise ValueError("event_uid_series requires at least one column")

    row_counts = {len(column) for column in columns}
    if (len(row_counts) > 1):
        raise ValueError(f"All columns must have the same length, received lengths {sorted(row_counts)}")

    return [_digest(row, digest_length=digest_length) for row in zip(*columns)]


def link_uid_series(parent_uids: typing.Sequence,
                    child_uids: typing.Sequence,
                    relation: str,
                    join_method: str,
                    digest_length: int = DEFAULT_DIGEST_LENGTH) -> list[typing.Optional[str]]:
    """
    Compute `link_uid` values for a batch of edges.

    Rows whose parent identifier is missing yield `None`, since a record with no parent is a chain root rather than an
    error.

    Parameters
    ----------
    parent_uids : sequence
        Per-row parent identifiers. Null or empty values mark chain roots.
    child_uids : sequence
        Per-row child identifiers.
    relation : str
        Semantic relationship applied to every edge in the batch.
    join_method : str
        Join method applied to every edge in the batch.
    digest_length : int, default = 32
        Number of hexadecimal characters to retain.

    Returns
    -------
    list
        One edge identifier per row, or `None` where there is no parent.
    """
    if (len(parent_uids) != len(child_uids)):
        raise ValueError(f"parent_uids and child_uids must have the same length, received {len(parent_uids)} and "
                         f"{len(child_uids)}")

    results: list[typing.Optional[str]] = []
    for (parent, child) in zip(parent_uids, child_uids):
        if (parent is None or parent != parent or parent == ""):  # pylint: disable=comparison-with-itself
            results.append(None)
        else:
            results.append(link_uid(parent, child, relation, join_method, digest_length=digest_length))

    return results
