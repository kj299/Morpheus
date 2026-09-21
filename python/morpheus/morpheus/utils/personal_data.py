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
What each column this fork emits says about a person, and what may usefully be done to it.

The guide defers retention, lawful basis and minimization to the deploying organization and its counsel, and
that deferral is right: which fields are personal data in a given jurisdiction, on what basis they may be held
and for how long, are not questions a pipeline can answer. What was wrong was that the design also made those
questions *unanswerable*. Nobody could say what personal data the pipeline produced without reading every stage,
there was no mechanism to minimize anything, and so "minimize" could only ever be written down as an intention.

This module is the half an engineer owes counsel: an inventory, derived from the code and checked against it.
`tests/morpheus/utils/test_personal_data.py` asserts that every column the reference pipelines emit appears here
exactly once, so a new feature column fails a test until somebody has decided what it says about a person.

**The classification is of columns, not of rows.** A column's category is what that column on its own contributes
to identifying, locating, or describing a person. `event_time` alone says nothing about who, so it is
operational; `source_ip` alone resolves to a device through the rest of this design, so it is not. A row is
personal data if any column in it is, and dropping `user_principal` while keeping `source_ip` and `event_time`
minimizes nothing.

The six categories:

- `IDENTIFIES` -- names the person or the account they logged in with.
- `ADDRESSES` -- a network or hardware address, which this design exists to resolve to a person's device.
- `LOCATES` -- says where the person was, whether by coordinate, by network, or by which desk they sat at.
- `PROFILES` -- the behavioural history accumulated about them: what they use, when they work, how they deviate
  from their own past. This is the category the design manufactures rather than collects, and it is the one an
  estate is least likely to have thought about.
- `PSEUDONYMS` -- stable identifiers derived from the above. They carry no name and re-identify completely
  against the binding tables, which is the whole point of them.
- `OPERATIONAL` -- about a network element, a flow, or the pipeline, and not about a person.

**Three column names mean different things in different telemetry classes, and a minimization policy that missed
that would be wrong in both directions.** `device_id` is a switch at layers 1 and 2 and a person's laptop at
layer 5; `entity_key` and `chain_anchor` are a switch port in one class and the principal themselves in another.
They are listed in `AMBIGUOUS` rather than given a category, and `classify` refuses to answer for them without
being told the class. Guessing would either drop the switch identifier and break the identifier ladder, or keep
the laptop and call the result minimized.

**The layer 3 additions follow the same rule and land in two places.** A flow's counts, ratios, rhythms and TTL
reference are `profiles`: they are what this design derives about a host's behaviour, and a host is a person's
device. The raw protocol fields beside them are `operational`, because a port number or a hop count on its own
says nothing about who, and so are the `_saturated` and `_first_in_window` flags, which describe the tracker's
window rather than the person. `dst_asn_first_seen` is the exception that proves the split: it is permanent
rather than window-scoped, so it records that this source has never reached that network before, which is a fact
about a history rather than about a buffer.

**On pseudonymization.** `pseudonymize` is a keyed HMAC, not a bare digest, because an unkeyed hash of a
username is reversed with a dictionary in the time it takes to write one. Even keyed, it is pseudonymization and
never anonymization: the mapping is stable by design, because unstable pseudonyms would destroy the per-entity
state every stateful stage in this design depends on, and a stable mapping is a mapping. It is a control against
casual access to the SIEM, not a reason to treat the output as no longer personal data. The binding tables that
sit beside it exist to re-identify, and they still will.

**A keyed hash over a bounded domain is not minimization at all**, which `BOUNDED_DOMAIN` names and
`morpheus.stages.lineage.minimization_stage.MinimizationStage` refuses. There are about two hundred and fifty
country codes and exactly twenty-four hours; whoever holds the output can count how often each digest appears
and read the mapping off the frequencies, and no key prevents that. Those columns have to be dropped or kept,
and the choice is between the feature and the field. Fields bounded by the estate rather than by their own
definition -- `site_id`, `switch_id`, `app` -- are usually just as weak in practice, and the stage cannot know
that from the schema, so it does not refuse them. It is the caller's job to notice that an estate with nine
sites has a nine-value domain.
"""

import hashlib
import hmac
import typing

IDENTIFIES = "identifies"
ADDRESSES = "addresses"
LOCATES = "locates"
PROFILES = "profiles"
PSEUDONYMS = "pseudonyms"
OPERATIONAL = "operational"

CATEGORIES = (IDENTIFIES, ADDRESSES, LOCATES, PROFILES, PSEUDONYMS, OPERATIONAL)

PERSONAL_CATEGORIES = (IDENTIFIES, ADDRESSES, LOCATES, PROFILES, PSEUDONYMS)
"""Every category except `OPERATIONAL`. What this fork emits that is about a person at all."""

_IDENTIFIES = (
    "desk_identity",
    "dot1x_identity",
    "session_id",
    "session_key",
    "user_principal",
)

_ADDRESSES = (
    "arp_sender_ip",
    "arp_sender_mac",
    "arp_target_ip",
    "dest_ip",
    "dst_ip",
    "flow_id",
    "flow_pair_key",
    "mac",
    "mac_address",
    "source_ip",
    "src_ip",
    "transfer_triple",
)

_LOCATES = (
    "auth_port_key",
    "desk_port_key",
    "port_id",
    "port_key",
    "resolved_port_key",
    "site_id",
    "source_asn",
    "source_city",
    "source_country",
    "source_latitude",
    "source_longitude",
    "source_region",
    "switch_id",
    "user_location",
)

_PROFILES = (
    "app",
    "app_first_seen",
    "appincrement",
    "appincrement_z_loss",
    "asns_in_window",
    "asns_in_window_saturated",
    "asns_in_window_z_loss",
    "auth_attempts",
    "auth_attempts_in_window",
    "auth_attempts_in_window_z_loss",
    "auth_elapsed_seconds",
    "auth_failed_then_succeeded",
    "auth_failures_in_window",
    "auth_failures_in_window_z_loss",
    "auth_result",
    "auth_unpaired",
    "bgp_as_dst",
    "byte_asymmetry",
    "bytes_in",
    "bytes_out",
    "cadence_mature",
    "cadence_samples",
    "consecutive_auth_failures",
    "consecutive_mfa_denials",
    "device_first_seen",
    "deviceincrement",
    "deviceincrement_z_loss",
    "dot1x_result",
    "drift_acceleration",
    "drift_baseline_sigma",
    "drift_mature",
    "drift_rise_sigmas",
    "drift_rising_windows",
    "drift_run_restarted",
    "drift_total_rise",
    "drift_velocity",
    "data_len",
    "dst_asn_first_seen",
    "dst_is_multicast",
    "dst_is_private",
    "dst_is_reserved",
    "dst_port",
    "dst_ports_per_src",
    "dsts_per_src",
    "flow_ack",
    "flow_ackpush_ratio",
    "flow_all",
    "flow_bpp",
    "flow_bpp_envelope",
    "flow_bpp_envelope_breached",
    "flow_bpp_envelope_ratio",
    "flow_data_len",
    "flow_data_len_envelope",
    "flow_data_len_envelope_breached",
    "flow_data_len_envelope_ratio",
    "flow_fin",
    "flow_fin_ratio",
    "flow_interval_cv",
    "flow_intervals",
    "flow_mean_interval_ns",
    "flow_ppm",
    "flow_psh",
    "flow_rst",
    "flow_rst_ratio",
    "flow_size_cv",
    "flow_syn",
    "flow_syn_ratio",
    "hour_share",
    "hour_surprise_bits",
    "hour_surprise_bits_z_loss",
    "hour_unseen",
    "internal_dst_ratio",
    "internal_dsts_in_window",
    "ip_ttl_distinct",
    "ip_ttl_established",
    "ip_ttl_shift",
    "ip_ttl_shifted",
    "local_hour",
    "local_weekday",
    "location_first_seen",
    "locincrement",
    "locincrement_z_loss",
    "logcount",
    "logcount_saturated",
    "logcount_z_loss",
    "max_abs_z",
    "mean_abs_z",
    "mfa_attempts_in_window",
    "mfa_challenge",
    "mfa_denials_in_window",
    "mfa_denied_then_approved",
    "mfa_ratio",
    "mfa_ratio_z_loss",
    "mfa_result",
    "mfa_used",
    "risk_counts_saturated",
    "session_action",
    "session_duration_ns",
    "session_duration_s",
    "session_out_of_order",
    "session_starts",
    "session_unpaired",
    "srcs_per_dst",
    "token_type",
    "travel_distance_km",
    "travel_elapsed_floored",
    "travel_elapsed_ns",
    "travel_kmh",
    "travel_status",
    "weekday_share",
    "weekday_surprise_bits",
    "weekday_surprise_bits_z_loss",
    "weekday_unseen",
)

_PSEUDONYMS = (
    "binding_uid",
    "community_id",
    "event_uid",
    "lineage_id",
    "origin_hash",
    "row_key",
)

_OPERATIONAL = (
    "arp_count_in_window",
    "arp_is_gratuitous",
    "arp_operation",
    "arp_sender_ip_excluded",
    "bind_end",
    "bind_end_observed",
    "bind_end_reason",
    "bind_gap_ns",
    "bind_observations",
    "bind_provisional",
    "bind_start",
    "chain_anchor_source",
    "collector_id",
    "collector_seq",
    "counter_reset",
    "counter_wrapped",
    "crc_errors",
    "crc_errors_delta",
    "day_is_late",
    "day_revision",
    "day_sealed_by",
    "day_window_complete",
    "day_window_end_ns",
    "day_window_id",
    "day_window_start_ns",
    "dest_port",
    "directory_resolution",
    "dst_ports_per_src_first_in_window",
    "dst_ports_per_src_saturated",
    "dsts_per_src_first_in_window",
    "dsts_per_src_saturated",
    "event_time",
    "flow_bpp_envelope_mature",
    "flow_data_len_envelope_mature",
    "flow_regularity_mature",
    "flow_regularity_saturated",
    "gratuitous_arp_count",
    "gratuitous_arp_ratio",
    "gratuitous_arp_ratio_saturated",
    "if_last_change",
    "input_discards",
    "input_discards_delta",
    "internal_dst_ratio_saturated",
    "interval_seconds",
    "ip_ttl",
    "ip_ttl_mature",
    "ip_ttl_saturated",
    "is_late",
    "link_flap_device_reset",
    "link_flap_last_change_inconsistent",
    "link_flap_unpolled",
    "link_flaps",
    "link_flaps_in_window",
    "lldp_neighbor_chassis_id",
    "lldp_neighbor_chassis_id_changed",
    "lldp_neighbor_chassis_id_distinct_count",
    "lldp_neighbor_chassis_id_first_seen",
    "macs_claiming_sender_ip",
    "macs_claiming_sender_ip_saturated",
    "macs_per_port",
    "macs_per_port_first_in_window",
    "macs_per_port_saturated",
    "oper_status",
    "optical_rx_dbm",
    "optical_rx_dbm_baseline",
    "optical_rx_dbm_baseline_samples",
    "optical_rx_dbm_deviation",
    "optical_tx_dbm",
    "optical_tx_dbm_baseline",
    "optical_tx_dbm_baseline_samples",
    "optical_tx_dbm_deviation",
    "osi_layer",
    "ouis_per_vlan",
    "ouis_per_vlan_first_in_window",
    "ouis_per_vlan_saturated",
    "output_discards",
    "output_discards_delta",
    "ports_per_mac",
    "ports_per_mac_first_in_window",
    "ports_per_mac_saturated",
    "protocol",
    "resolution_method",
    "rollup_time_ns",
    "resolved_vlan_id",
    "revision",
    "sample_out_of_order",
    "schema_version",
    "sealed_by",
    "src_port",
    "srcs_per_dst_first_in_window",
    "srcs_per_dst_saturated",
    "supplicant_resolution",
    "symbol_errors",
    "symbol_errors_delta",
    "tcp_flags",
    "telemetry_class",
    "transceiver_serial",
    "transceiver_serial_changed",
    "transceiver_serial_distinct_count",
    "transceiver_serial_first_seen",
    "uptime",
    "vlan_id",
    "window_complete",
    "window_end_ns",
    "window_id",
    "window_seq",
    "window_start_ns",
)

COLUMNS: dict[str, str] = {}

for (_category, _names) in ((IDENTIFIES, _IDENTIFIES), (ADDRESSES, _ADDRESSES), (LOCATES, _LOCATES),
                            (PROFILES, _PROFILES), (PSEUDONYMS, _PSEUDONYMS), (OPERATIONAL, _OPERATIONAL)):
    for _name in _names:
        COLUMNS[_name] = _category

AMBIGUOUS: dict[str, dict[str, str]] = {
    "device_id": {
        "tc1": OPERATIONAL, "tc1_binding": OPERATIONAL, "tc5_auth": ADDRESSES, "tc5_session": ADDRESSES
    },
    "entity_key": {
        "tc1": LOCATES,
        "tc1_binding": LOCATES,
        "tc2_arp": ADDRESSES,
        "tc2_auth": LOCATES,
        "tc2_binding": ADDRESSES,
        "tc2_mac": LOCATES,
        "tc5_auth": IDENTIFIES,
        "tc5_session": IDENTIFIES,
    },
}
"""Column names whose meaning depends on which telemetry class the row belongs to.

`device_id` is the switch that reported a port at layers 1 and 2 and the laptop a person authenticated from at
layer 5. `entity_key` holds whatever that class's subject is, which Part 2 defines separately per layer: a port,
a MAC, or the principal. A policy applied by column name alone would either strip the switch identifier and
break the identifier ladder, or leave a person's laptop in the output and call it minimized.
"""

PER_ROW: dict[str, str] = {
    "chain_anchor": "chain_anchor_source",
}
"""Columns whose category is decided per row rather than per class, and the column that records the decision.

`ChainAnchorStage` copies the first candidate that matched, so one frame's `chain_anchor` holds a port on the
rows the ladder resolved and the principal on the rows it did not, and `chain_anchor_source` says which for each.
There is no per-class answer to give, so `classify` refuses rather than picking the one that is right more often,
and a policy has to name the column explicitly alongside the candidates it was copied from. Pseudonymizing an
anchor and leaving the column it was copied from is the same mistake as pseudonymizing one name for a person, and
it is caught the same way.
"""

BOUNDED_DOMAIN = frozenset({
    "arp_operation",
    "auth_result",
    "dot1x_result",
    "local_hour",
    "local_weekday",
    "mfa_result",
    "oper_status",
    "osi_layer",
    "protocol",
    "resolved_vlan_id",
    "session_action",
    "source_country",
    "source_region",
    "tcp_flags",
    "telemetry_class",
    "token_type",
    "travel_status",
    "vlan_id",
})
"""Columns whose set of possible values is fixed by the field's own definition rather than by the estate's size.

Hashing one of these is not minimization however good the key is. Count how often each digest appears and the
mapping falls out of the frequencies: one value in four for a weekday, one in twenty-four for an hour, and a
country distribution that in most estates is a single value with a long tail. The choice for these columns is to
keep them or to drop them.

Not a complete list of the weak ones, and cannot be. `site_id`, `switch_id` and `app` are bounded by the estate
instead, which is usually small too, and no schema can say how small.
"""

DEFAULT_DIGEST_LENGTH = 32
"""Hex characters kept from each pseudonym, matching `morpheus.utils.lineage`'s own identifiers."""

MINIMUM_KEY_BYTES = 16
"""Shortest key accepted. Below this the HMAC's own strength stops being the thing that protects the mapping."""


def classify(column: str, telemetry_class: typing.Optional[str] = None) -> str:
    """
    What a column says about a person.

    Parameters
    ----------
    column : str
        The column name.
    telemetry_class : str, optional
        Which class the row belongs to. Required for the names in `AMBIGUOUS` and ignored for the rest.

    Returns
    -------
    str
        One of `CATEGORIES`.

    Raises
    ------
    KeyError
        If the column is not classified, which for a column this fork emits means nobody has decided yet.
    ValueError
        If the column is ambiguous and no class was given, the class is one this column has no entry for, or
        the column is one of `PER_ROW` and has no single category to give.
    """
    if (column in PER_ROW):
        raise ValueError(f"{column!r} holds whichever candidate matched, so its category is a property of the "
                         f"row rather than of the class; {PER_ROW[column]!r} records which. Name it explicitly "
                         f"in a policy, together with the columns it is copied from.")

    if (column in AMBIGUOUS):
        meanings = AMBIGUOUS[column]

        if (telemetry_class is None):
            raise ValueError(f"{column!r} means different things in different telemetry classes "
                             f"({sorted(set(meanings.values()))}); name the class rather than guessing")

        if (telemetry_class not in meanings):
            raise ValueError(f"{column!r} is ambiguous and has no recorded meaning for telemetry class "
                             f"{telemetry_class!r}. Add one rather than letting it fall through to a default.")

        return meanings[telemetry_class]

    if (column not in COLUMNS):
        raise KeyError(f"{column!r} is not classified. A column this fork emits has to say what it tells "
                       f"somebody about a person before it can be minimized or deliberately kept.")

    return COLUMNS[column]


def columns_in(category: str, telemetry_class: typing.Optional[str] = None) -> list[str]:
    """
    Every classified column in one category, sorted.

    Parameters
    ----------
    category : str
        One of `CATEGORIES`.
    telemetry_class : str, optional
        When given, the ambiguous columns are resolved for that class and included where they match.

    Returns
    -------
    list of str
    """
    if (category not in CATEGORIES):
        raise ValueError(f"{category!r} is not one of {list(CATEGORIES)}")

    found = [column for (column, assigned) in COLUMNS.items() if assigned == category]

    if (telemetry_class is not None):
        found.extend(column for (column, meanings) in AMBIGUOUS.items() if meanings.get(telemetry_class) == category)

    return sorted(found)


def personal_columns(present: typing.Iterable[str], telemetry_class: typing.Optional[str] = None) -> list[str]:
    """
    Which of the columns actually present say something about a person.

    Parameters
    ----------
    present : iterable of str
        The frame's columns. Unclassified names raise, because silently treating an unknown column as
        operational is how a new feature reaches a SIEM unconsidered. The `PER_ROW` columns count as personal
        without a category, since every candidate they can hold is one.
    telemetry_class : str, optional
        Passed to `classify` for the ambiguous names.

    Returns
    -------
    list of str
        Sorted, and holding only columns in `PERSONAL_CATEGORIES`.
    """
    found = []

    for column in present:
        # A per-row column holds a personal value whichever candidate won -- a port or a principal, never
        # something operational -- so it counts without having to say which.
        if (column in PER_ROW or classify(column, telemetry_class) in PERSONAL_CATEGORIES):
            found.append(column)

    return sorted(found)


def _key_bytes(key: typing.Union[str, bytes]) -> bytes:
    if (isinstance(key, str)):
        key = key.encode("utf-8")

    if (not isinstance(key, bytes)):
        raise TypeError(f"key must be str or bytes, received {type(key).__name__}")

    if (len(key) < MINIMUM_KEY_BYTES):
        raise ValueError(f"key must be at least {MINIMUM_KEY_BYTES} bytes; a short key is the weakest part of "
                         f"a construction whose whole job is to be hard to invert")

    return key


def pseudonymize(values: typing.Sequence,
                 key: typing.Union[str, bytes],
                 digest_length: int = DEFAULT_DIGEST_LENGTH) -> list:
    """
    Replace a column's values with keyed digests, stably.

    A keyed HMAC rather than a bare hash: `sha256("alice@example.com")` is recovered from a wordlist, and an
    estate's directory is a wordlist. The key is required and has no default for the same reason.

    Stable by construction, because the alternative does not work here. Every stateful stage in this design keys
    per-entity history on the entity's identifier, and a pseudonym that changed between runs would split one
    person's history into as many people as there were runs. Stability is also exactly what keeps this
    pseudonymization rather than anonymization, and nothing downstream should be described as anonymous.

    Nulls are preserved as nulls. Hashing the absence of a value would manufacture a value, and an entity whose
    identifier is missing would acquire a stable identity it never had.

    Parameters
    ----------
    values : sequence
        The column's values.
    key : str or bytes
        The HMAC key, at least `MINIMUM_KEY_BYTES` long. Hold it outside the pipeline's output and rotate it
        knowing that rotation breaks every join against digests produced under the old one.
    digest_length : int, default = 32
        Hex characters kept.

    Returns
    -------
    list
        One digest per input value, `None` where the input was null.
    """
    if (digest_length <= 0):
        raise ValueError(f"digest_length must be positive, received {digest_length}")

    secret = _key_bytes(key)
    digests = []

    for value in values:
        if (value is None or value != value):  # pylint: disable=comparison-with-itself
            digests.append(None)
            continue

        rendered = str(value).encode("utf-8")
        digests.append(hmac.new(secret, rendered, hashlib.sha256).hexdigest()[:digest_length])

    return digests
