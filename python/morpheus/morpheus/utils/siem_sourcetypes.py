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
What produces each sourcetype the shipped Splunk app is configured to parse, or what is missing where nothing does.

The app's `props.conf` declares fourteen stanzas. Each names the field its `TIME_PREFIX` anchors `_time` on, and
each of those fields has to reach the wire as a quoted string in the app's `TIME_FORMAT` or Splunk stamps the
event at index time instead -- silently, with no error anywhere, turning every windowed detection into a rule
about when the pipeline happened to be busy.

That is a contract with two halves, and only one of them lived in this repository. `props.conf` said what the
SIEM would look for; nothing said what this fork actually emits, so a stanza could anchor on a field no producer
writes and nothing would notice. This module is the other half. Every stanza is listed here exactly once, either
as a `Sourcetype` naming its producer and the columns it must carry, or as an `Unproduced` naming what does not
exist yet. `tests/morpheus/utils/test_siem_sourcetypes.py` asserts the two halves agree: every stanza in the
configuration file appears here, every entry here corresponds to a stanza, and each declared time column is the
one that stanza's own `TIME_PREFIX` regex anchors on.

Being unproduced is a fact worth recording rather than a gap worth hiding. Eight of the fourteen stanzas are
configuration for producers this fork has not built, and saying so in one place is what keeps "the app supports
seven layers" from reading as "seven layers are implemented".

On nanoseconds. Wire rendering is microsecond precision, because that is what Splunk's `%6N` reads. Columns whose
names end in `_ns` -- `window_start_ns`, `window_end_ns`, `bind_gap_ns` -- are deliberately left as numbers: they
are the exact values a consumer computes with, they are not what `_time` is taken from, and rounding them to fit
a timestamp format would quietly change arithmetic that depends on them.
"""

import dataclasses
import typing


@dataclasses.dataclass(frozen=True)
class Sourcetype:
    """A sourcetype something in this fork actually emits."""

    name: str
    """The stanza name in `props.conf`."""

    time_column: str
    """The column that stanza's `TIME_PREFIX` anchors `_time` on. Must be present and must be rendered."""

    time_columns: tuple
    """
    Every column rendered into wire format for this sourcetype, `time_column` among them.

    A record often carries more than one timestamp, and the anchor is only the one Splunk reads. Rendering the
    rest is not decoration: a consumer reading `bind_start` off a `binding:l2` record should not have to know that
    one field on the record is a string and its sibling is a nineteen-digit integer.
    """

    producer: str
    """What emits it, named concretely enough to go and read."""

    required_columns: tuple
    """
    Columns a record must carry beyond its timestamps, because the app's own searches read them.

    Deliberately the fields the shipped searches name, not every field a producer happens to emit. A producer is
    free to add columns; it is not free to drop these.
    """


@dataclasses.dataclass(frozen=True)
class Unproduced:
    """A sourcetype the app is configured to parse and nothing in this fork emits."""

    name: str
    """The stanza name in `props.conf`."""

    time_column: str
    """The column that stanza's `TIME_PREFIX` anchors on. Recorded so the contract test can check it regardless."""

    missing: str
    """What would have to exist. Specific, so it reads as a work item rather than an apology."""


PRODUCED: dict = {
    "morpheus:score:l1":
        Sourcetype(
            name="morpheus:score:l1",
            time_column="event_time",
            time_columns=("event_time", ),
            producer="The TC-1 stages (normalize, optical, flap, change) behind WindowSealStage; the `tc1` class of "
            "`tests/morpheus/determinism/telemetry_pipeline.py`.",
            required_columns=("event_uid", "entity_key", "site_id", "device_id", "port_id"),
        ),
    "morpheus:score:l2":
        Sourcetype(
            name="morpheus:score:l2",
            time_column="event_time",
            time_columns=("event_time", ),
            producer="The TC-2 stages (cardinality, ARP, auth) behind WindowSealStage; the `tc2_mac`, `tc2_arp` and "
            "`tc2_auth` classes of `tests/morpheus/determinism/telemetry_pipeline.py`.",
            # R-D-L2-001, R-D-L2-003 and R-D-L2-005 read these off this sourcetype.
            required_columns=("event_uid",
                              "port_key",
                              "macs_per_port_first_in_window",
                              "macs_claiming_sender_ip",
                              "arp_sender_ip_excluded",
                              "auth_unpaired",
                              "auth_port_key"),
        ),
    "morpheus:edge":
        Sourcetype(
            name="morpheus:edge",
            time_column="event_time",
            time_columns=("event_time", ),
            producer="`morpheus.stages.lineage.community_id_stage.CommunityIdStage` behind WindowSealStage; the "
            "reference pipeline in `tests/morpheus/determinism/lineage_pipeline.py`.",
            required_columns=("event_uid", "community_id", "src_ip", "dest_ip"),
        ),
    "binding:bucketed":
        Sourcetype(
            name="binding:bucketed",
            time_column="bucket_start",
            time_columns=("bucket_start", ),
            producer="`morpheus.utils.binding_table.BindingTable.to_bucketed_records`, which renders `bucket_start` "
            "itself rather than relying on a sink to do it.",
            required_columns=("binding_table", ),
        ),
    "binding:l2":
        Sourcetype(
            name="binding:l2",
            time_column="bind_end",
            time_columns=("bind_end", "bind_start", "event_time"),
            producer="`morpheus.stages.telemetry.tc2_binding_stage.TC2BindingStage`, one record per closed interval.",
            # R-D-L2-004 reads these.
            required_columns=("mac_address", "port_key", "bind_end_reason", "bind_gap_ns", "bind_observations"),
        ),
    "binding:l2:open":
        Sourcetype(
            name="binding:l2:open",
            time_column="bind_start",
            time_columns=("bind_start", "event_time"),
            producer="`morpheus.stages.telemetry.tc2_binding_stage.TC2BindingStage` with `emit_open_bindings`, one "
            "record the moment a binding opens.",
            required_columns=("mac_address", "port_key", "bind_provisional"),
        ),
}
"""Sourcetypes something in this fork emits, keyed by stanza name."""

_LAYER_ABOVE_2 = ("A telemetry class for this layer. Layers 3 through 7 are design in the guide; no collector, no "
                  "stage and no scoring path for them exists here.")

UNPRODUCED: dict = {
    "morpheus:score:l3":
        Unproduced("morpheus:score:l3", "event_time", _LAYER_ABOVE_2),
    "morpheus:score:l4":
        Unproduced("morpheus:score:l4", "event_time", _LAYER_ABOVE_2),
    "morpheus:score:l5":
        Unproduced("morpheus:score:l5", "event_time", _LAYER_ABOVE_2),
    "morpheus:score:l6":
        Unproduced("morpheus:score:l6", "event_time", _LAYER_ABOVE_2),
    "morpheus:score:l7":
        Unproduced("morpheus:score:l7", "event_time", _LAYER_ABOVE_2),
    "binding:l1":
        Unproduced(
            "binding:l1",
            "bind_start",
            "A layer 1 inventory producer: switch port to site, transceiver and LLDP neighbor, with the expiry that "
            "keeps it current. The app consumes this sourcetype into the `binding_l1` lookup and the TC-1 stages read "
            "the same facts from their input, but nothing here emits the binding records themselves.",
        ),
    "context:identity":
        Unproduced(
            "context:identity",
            "valid_from",
            "The TC-0 context store. Identity attribution is design in the guide and has no producer here.",
        ),
    "context:asset":
        Unproduced(
            "context:asset",
            "valid_from",
            "The TC-0 context store. Asset attribution is design in the guide and has no producer here.",
        ),
}
"""Sourcetypes the app parses and nothing here emits, with what is missing."""


def stanza_names() -> tuple:
    """
    Every sourcetype this module accounts for, produced or not, in sorted order.

    Returns
    -------
    tuple
        Stanza names.
    """
    return tuple(sorted(set(PRODUCED) | set(UNPRODUCED)))


def describe(name: str) -> typing.Union[Sourcetype, Unproduced]:
    """
    Look up a sourcetype without caring whether it has a producer.

    Parameters
    ----------
    name : str
        Stanza name, for example `morpheus:score:l2`.

    Returns
    -------
    `Sourcetype` or `Unproduced`
        The entry for `name`.

    Raises
    ------
    KeyError
        If `name` is not a stanza this module knows.
    """
    if (name in PRODUCED):
        return PRODUCED[name]

    if (name in UNPRODUCED):
        return UNPRODUCED[name]

    raise KeyError(f"Unknown sourcetype {name!r}. Known sourcetypes: {', '.join(stanza_names())}")


def sourcetype(name: str) -> Sourcetype:
    """
    Look up a sourcetype that must have a producer.

    Parameters
    ----------
    name : str
        Stanza name, for example `morpheus:score:l2`.

    Returns
    -------
    `Sourcetype`
        The entry for `name`.

    Raises
    ------
    KeyError
        If `name` is not a stanza this module knows.
    ValueError
        If `name` is a stanza nothing in this fork produces. The message names what is missing, because the useful
        answer to "why can I not emit this?" is what would have to be built, not that the lookup failed.
    """
    entry = describe(name)

    if (isinstance(entry, Unproduced)):
        raise ValueError(f"Nothing in this fork produces sourcetype {name!r}. Missing: {entry.missing}")

    return entry
