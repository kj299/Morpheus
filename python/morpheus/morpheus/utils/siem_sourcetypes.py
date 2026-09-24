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

Being unproduced is a fact worth recording rather than a gap worth hiding. Eight of the fourteen stanzas were
once configuration for producers this fork had not built, and saying so in one place is what kept "the app supports
seven layers" from reading as "seven layers are implemented". None is any longer -- the TC-0 context store was the
last -- and `UNPRODUCED` stays, empty, so that the next stanza shipped ahead of its producer is recorded the same way.

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
    "morpheus:score:l3":
        Sourcetype(
            name="morpheus:score:l3",
            time_column="event_time",
            time_columns=("event_time", ),
            producer="The TC-3 stages (cardinality, reach, beacon, TTL) behind WindowSealStage; the `tc3` class "
            "of `tests/morpheus/determinism/network_pipeline.py`.",
            # The five layer 3 detections read these off this sourcetype. `flow_pair_key` is here because
            # R-B-L3-002 is about a conversation rather than about a host, and a search grouping by `src_ip`
            # would average a beacon in with everything else that host does.
            required_columns=("event_uid",
                              "src_ip",
                              "dst_ip",
                              "flow_pair_key",
                              "dsts_per_src",
                              "internal_dst_ratio",
                              "flow_interval_cv",
                              "flow_size_cv",
                              "flow_regularity_mature",
                              "dst_is_reserved",
                              "dst_is_multicast",
                              "ip_ttl_shift",
                              "ip_ttl_shifted"),
        ),
    "morpheus:score:l4":
        Sourcetype(
            name="morpheus:score:l4",
            time_column="event_time",
            time_columns=("event_time", ),
            producer="The TC-4 stages (flow rollup, transfer envelope) behind WindowSealStage; the `tc4` class "
            "of `tests/morpheus/determinism/transport_pipeline.py`.",
            # The three fireable layer 4 detections read these off this sourcetype. The counts are here and the
            # ratio columns are not, because a running ratio is not monotone and a search that summarizes a bin
            # has to divide the counts' maxima rather than aggregate the ratio -- see `tc4_flow_stage`.
            required_columns=("event_uid",
                              "src_ip",
                              "dst_ip",
                              "dst_port",
                              "flow_id",
                              "rollup_time_ns",
                              "flow_syn",
                              "flow_ack",
                              "flow_rst",
                              "flow_all",
                              "transfer_triple",
                              "flow_data_len",
                              "flow_data_len_envelope",
                              "flow_data_len_envelope_ratio",
                              "flow_data_len_envelope_breached",
                              "flow_data_len_envelope_mature",
                              "flow_bpp",
                              "flow_bpp_envelope_ratio",
                              "flow_bpp_envelope_breached"),
        ),
    "morpheus:score:l6":
        Sourcetype(
            name="morpheus:score:l6",
            time_column="event_time",
            time_columns=("event_time", ),
            producer="The TC-6 stages (fingerprint, certificate, cipher, content) behind WindowSealStage; the "
            "`tc6` class of `tests/morpheus/determinism/presentation_pipeline.py`.",
            # The five layer 6 detections read these. `ja4_client_observations` and `cert_issuer_distinct` are
            # here because two of the rules are unusable without them: novelty alone fires on every host the
            # estate has just started seeing, and an issuer difference alone fires on every delivery host behind
            # more than one authority.
            required_columns=("event_uid",
                              "src_ip",
                              "dst_ip",
                              "ja4_client",
                              "ja4_client_first_seen",
                              "ja4_client_observations",
                              "certificate_issuer",
                              "cert_issuer_established",
                              "cert_issuer_differs",
                              "cert_issuer_distinct",
                              "cert_issuer_mature",
                              "cert_self_signed",
                              "cert_self_signed_external",
                              "cert_validity_days",
                              "cipher_suite",
                              "cipher_tier",
                              "cipher_floor_tier",
                              "cipher_downgraded",
                              "cipher_mature",
                              "content_type_declared",
                              "content_type_detected",
                              "content_category_declared",
                              "content_category_detected",
                              "content_category_crossed"),
        ),
    "morpheus:score:l7":
        Sourcetype(
            name="morpheus:score:l7",
            time_column="event_time",
            time_columns=("event_time", ),
            producer="The TC-7 DNS and HTTP stages behind WindowSealStage; the `tc7_dns` and `tc7_http` classes of "
            "`tests/morpheus/determinism/application_pipeline.py`. The SaaS and endpoint sub-classes will share this "
            "sourcetype when they land.",
            # R-B-L7-001 and R-D-L7-005 read these. The two count columns are here as well as the ratio because the
            # ratio is undefined for a client with no successes, and the search reads the counts for that reason.
            required_columns=("event_uid",
                              "src_ip",
                              "query_name",
                              "dns_registered_domain",
                              "dns_subdomain",
                              "dns_subdomain_entropy",
                              "dns_mean_label_length",
                              "dns_subdomains_per_domain",
                              "url_path",
                              "status_code",
                              "http_4xx_in_window",
                              "http_2xx_in_window",
                              "http_4xx_to_2xx_ratio",
                              "http_distinct_paths"),
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
    "morpheus:score:l5":
        Sourcetype(
            name="morpheus:score:l5",
            time_column="event_time",
            time_columns=("event_time", ),
            producer="The TC-5 stages (session, novelty, cadence, travel, risk) behind WindowSealStage; the "
            "`tc5_auth` and `tc5_session` classes of `tests/morpheus/determinism/session_pipeline.py`.",
            # R-D-L5-003 and R-D-L5-004 filter on the first eight; R-P-L5-006 on the rest, which TC5DriftStage
            # stamps over the daily windows a second WindowSealStage seals behind the hourly one.
            required_columns=("event_uid",
                              "user_principal",
                              "travel_status",
                              "travel_kmh",
                              "travel_elapsed_ns",
                              "mfa_denied_then_approved",
                              "mfa_attempts_in_window",
                              "mfa_denials_in_window",
                              "mean_abs_z",
                              "max_abs_z",
                              "day_window_id",
                              "drift_mature",
                              "drift_rising_windows",
                              "drift_rise_sigmas"),
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
    "binding:l1":
        Sourcetype(
            name="binding:l1",
            time_column="bind_start",
            time_columns=("bind_start", "bind_end"),
            producer="`morpheus.stages.telemetry.tc1_binding_stage.TC1BindingStage`, one record per port interval.",
            # The shipped `Binding lookup - L1 refresh` search builds its key from the first two and returns the
            # rest. `switch_id` rather than `device_id` because that is the name the search uses; the stage emits
            # the identifier under both.
            required_columns=("port_id",
                              "switch_id",
                              "site_id",
                              "transceiver_serial",
                              "lldp_neighbor_chassis_id",
                              "binding_uid"),
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
    "context:identity":
        Sourcetype(
            name="context:identity",
            time_column="valid_from",
            time_columns=("valid_from", "valid_to", "recorded_at"),
            producer="`morpheus.stages.telemetry.tc0_identity_stage.TC0IdentityStage`, one record per version of a "
            "profile or a group membership; the `tc0_identity` class of "
            "`tests/morpheus/determinism/context_pipeline.py`.",
            # What a consumer rebuilds a `BitemporalStore` from. `recorded_at` is the transaction time and `change`
            # distinguishes a retraction from an assertion; without either the store cannot answer as-known-at
            # questions, which is the whole of what it is for.
            required_columns=("context_uid",
                              "context_kind",
                              "context_entity",
                              "context_key",
                              "context_attributes",
                              "change",
                              "user_principal"),
        ),
    "context:asset":
        Sourcetype(
            name="context:asset",
            time_column="valid_from",
            time_columns=("valid_from", "valid_to", "recorded_at"),
            producer="`morpheus.stages.telemetry.tc0_asset_stage.TC0AssetStage`, one record per version of an asset; "
            "the `tc0_asset` class of `tests/morpheus/determinism/context_pipeline.py`.",
            required_columns=("context_uid",
                              "context_kind",
                              "context_entity",
                              "context_key",
                              "context_attributes",
                              "change",
                              "hostname",
                              "data_classification",
                              "peer_group"),
        ),
}
"""Sourcetypes something in this fork emits, keyed by stanza name."""

UNPRODUCED: dict = {}
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
