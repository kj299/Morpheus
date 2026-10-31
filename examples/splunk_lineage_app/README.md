<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Splunk Lineage App for Morpheus Behavioral Analytics

`TA-morpheus-lineage` is an installable Splunk app implementing the SIEM half of the
[Predictive Behavioral Analytics Across OSI Layers 1-7](../../docs/source/developer_guide/guides/11_predictive_behavioral_analytics_osi.md)
design guide. The guide's Part 4 explains every decision this app encodes; this page covers only
installation and the knobs that must match the Morpheus pipeline.

The app is configuration, not code: indexes, sourcetypes, KV Store binding lookups, and scheduled
searches. It assumes a Morpheus pipeline is publishing scored events, lineage edges, and bucketed
binding rows as described in the guide, through the Kafka connector or the forwarder inputs under `transport/`.

## Contents

| File | Deploy to | What it defines |
| --- | --- | --- |
| `default/indexes.conf` | Indexers | `behavior_events`, `behavior_lineage`, `behavior_bindings`, `behavior_context`, `behavior_summary`, `behavior_risk`, and the pipeline's own `behavior_health`, `behavior_quarantine` and `behavior_late`, with deliberately asymmetric retention, and a statement of what each one holds about a person beside the period it holds it for |
| `default/props.conf` | Indexers or heavy forwarders | One JSON sourcetype per OSI layer plus edges, bindings, context, health, quarantine and late rows, each with `_time` anchored on a field the record carries -- `event_time` for scores, edges, late rows and health (a window's end), an interval bound for bindings, `valid_from` for context, and `ingest_time` for a quarantined record, whose own event time is the thing in doubt |
| `default/collections.conf` | Search heads | KV Store collections for the L2/L3, L2 and L5/L2 bucketed bindings, the unbucketed L1 bindings, the bucketed L1 history beside them, the LLDP adjacency the topology search writes, and the principal and host watchlists the predictive rules write, with accelerated fields |
| `default/transforms.conf` | Search heads | The `binding_l2_l3`, `binding_l2`, `binding_l5_l2`, `binding_l1`, `binding_l1_history`, `lldp_adjacency`, `principal_watchlist` and `host_watchlist` lookups, and the `rule_metadata`, `principal_directory`, `port_designations`, `vlan_designations` and `scanner_allowlist` files |
| `default/savedsearches.conf` | Search heads | Eighty-six searches: lookup refresh and expiry jobs for the address, MAC and 802.1X identity rungs and the layer 1 bindings, including the principal and host watchlists' and the LLDP adjacency refresh, the 5-minute summary rollup, chain assembly, the connection evidence report joining layers 3, 4 and 6 on `community_id`, the host evidence report joining layers 3, 4, 6 and 7 on `host_key`, the identity walk from a sign-in to its switch port through the binding lookups, the chained detections R-C-001, R-C-002, R-C-003, R-C-004 and R-C-005, the ten layer 1 detections R-D-L1-001, 002, 003, 006, 008 and 010, R-B-L1-005 and 007 and the predictive R-P-L1-004 and R-P-L1-009, the device restart report, the eight layer 2 detections R-B-L2-002, 006, 007 and 008 and R-D-L2-001, 003, 004 and 005, the nine layer 3 detections R-B-L3-001, R-B-L3-002, R-D-L3-003, R-B-L3-004, the predictive R-P-L3-005, the host-baseline rules R-B-L3-006, R-B-L3-007 and R-D-L3-008, and the predictive host watchlist R-P-L3-009 over the host segment's scores on `morpheus:score:host`, the four layer 4 detections R-D-L4-002, R-D-L4-003, R-B-L4-005 and R-B-L4-004, the last joined to layer 7 on `host_key`, the five layer 6 detections R-B-L6-001, R-D-L6-002, R-D-L6-003, R-B-L6-004 and R-D-L6-005, the ten layer 7 detections R-B-L7-001, R-D-L7-005, R-B-L7-007, R-B-L7-008, R-B-L7-002, R-B-L7-004 and the predictive R-P-L7-006, R-P-L7-009, R-P-L7-010 and R-P-L7-011, the five deterministic layer 5 detections R-D-L5-003, R-D-L5-004, R-D-L5-007, R-D-L5-008 and R-D-L5-009, the two layer 5 model rules R-B-L5-001 and R-B-L5-002, gated on a principal's own model, which the models committed under `examples/layer5_model/models/` pin, and returning 42 and 28 rows on the validation corpus, the layer 5 session duration rule R-B-L5-005, the layer 5 predictive watchlist R-P-L5-006, a binding health alert, a TLS table coverage metric, and five reports on the pipeline itself -- the refusal rate, collector health, the stages' own counts, late arrivals by class and the platform baseline step. Every detection reads only rows whose window was complete (`window_complete=true`), ends by collecting its rows into `behavior_risk`, which Chain assembly sums, and is a per-result alert suppressed on its stated deduplication key for its dispatch window |
| `lookups/port_designations.csv` | Search heads | The port designation list R-D-L2-001 reads, and R-B-L2-002 reads to leave `trunk` and `lag-member` ports out: `port_key,designation,max_macs`. Ships header-only; populate it from the inventory |
| `lookups/vlan_designations.csv` | Search heads | The estate's VLANs by the prefix each one routes, `site_id,vlan_id,role,subnet`, matched as a CIDR on `subnet`; R-C-003 reads it to name the management VLAN a flow reached. Ships header-only; populate it from the IP address plan |
| `lookups/principal_directory.csv` | Search heads | The estate's directory: which 802.1X identity each principal presents, `user_principal,dot1x_identity,valid_from,valid_until`. Ships header-only; it is the first rung from a person to a port and no telemetry this app reads supplies it |
| `lookups/rule_metadata.csv` | Search heads | One row per detection: `rule_id,saved_search,suppress_fields,suppress_period,hysteresis`. The stanzas' `alert.suppress.*` keys are held to it by a test; `hysteresis` is `none` for every rule until a model scores near a threshold |
| `transport/kafka-connect-splunk.json` | The Kafka Connect worker | A Splunk Connect for Kafka sink over one topic per produced sourcetype, each mapped to its sourcetype and index, posted to the HTTP Event Collector's raw endpoint so the stanzas above set `_time`. Rendered from `morpheus.utils.siem_sourcetypes` by `transport/make_transport.py`; `transport/README.md` explains each setting |
| `transport/inputs.conf` | Indexers (the token) or the universal forwarder on the runners' host (the monitors) | The HEC token stanza the connector posts to, allowing exactly the seven indexes, and one monitor per sourcetype over the files the segment runners write |
| `lookups/scanner_allowlist.csv` | Search heads | The estate's own scanners, which R-B-L3-001 excludes: `src_ip,allowed,owner,note`. Ships header-only; until it is populated the rule fires on every scanner, authorized ones included |

## Installation

On a single instance, copy the app and restart:

```bash
cp -r TA-morpheus-lineage $SPLUNK_HOME/etc/apps/
$SPLUNK_HOME/bin/splunk restart
```

In a distributed deployment, split the app along the "Deploy to" column above: the index
definitions go to the indexer tier (through the cluster manager where one exists), the parsing
stanzas go to whichever tier parses (indexers, or heavy forwarders in front of them), and the
collections, lookups, and scheduled searches go to the search head tier. Shipping the whole app
everywhere is harmless; the split is only about which stanzas take effect where.

Verify the configuration parsed:

```bash
$SPLUNK_HOME/bin/splunk btool indexes list behavior_events --debug
$SPLUNK_HOME/bin/splunk btool props list morpheus:score:l3 --debug
$SPLUNK_HOME/bin/splunk btool savedsearches list "Chain assembly - cross-layer risk" --debug
```

## Getting the records in

The pipeline side ends in `SiemWireSinkStage`, which writes each sourcetype to a Kafka topic named after it
(`morpheus.score.l3` for `morpheus:score:l3`) or to a file (`morpheus_score_l3.jsonlines`). `transport/` holds
the configuration for the rest of the way, and its [README](transport/README.md) says when to use which:

- **Kafka**, the recommended path: apply `transport/kafka-connect-splunk.json` to a Kafka Connect worker running
  Splunk Connect for Kafka, and enable the token in `transport/inputs.conf` on the indexers it posts to. The
  connector uses the raw endpoint, so the `props.conf` stanzas above are what set `_time`; the event endpoint
  would stamp the Kafka record's time instead.
- **Files and a universal forwarder**: run the segments with `--output` under `/var/spool/morpheus/<segment>/`
  and deploy the monitor stanzas in `transport/inputs.conf` to the forwarder.

Both are rendered from the sourcetype inventory the pipeline writes from, and
`tests/morpheus/utils/test_splunk_transport.py` holds them to it, to the stanzas and indexes in this app and to
the index each saved search reads a sourcetype from. Neither has been run against a live connector or
forwarder; the search-head run under `validate/` indexes the sample events directly.

## Settings that must match the pipeline

These values are shared contracts between this app and the Morpheus pipeline. Changing either
side alone breaks the joins silently.

1. **Bucket width: 300 seconds at layers 2 and 3, 86400 at layer 1.** The pipeline expands bindings
   with `BindingTable.to_bucketed_frame(bucket_seconds=...)`, and every query in the guide rediscretizes
   event times with the same divisor. The expiry saved searches assume it too. The two widths are not a
   preference; they follow the intervals. A DHCP lease lasts hours and five minutes divides it usefully,
   while a transceiver sits in a port for months and expanding one of those at five minutes runs past
   `max_buckets_per_binding` before the optic is five weeks old. Each bucketed row carries `bucket_start`,
   the bucket's own start time, which is what `[binding:bucketed]` anchors `_time` on. A bucketed row has
   no other time of its own: it stands for a key in a bucket, not for a single binding record, so it
   carries neither `bind_start` nor `bind_end`.
2. **Binding retention, 400 days.** `frozenTimePeriodInSecs` on `behavior_bindings` and the cutoffs
   in the `Binding lookup - L2/L3 expiry` and `Binding lookup - L1 history expiry` searches must move
   together. A lookup that expires before its index produces unattributable events.
   It must also stay longer than the 395 days `behavior_events` and every index derived from a scored
   event keep, the thirteen months the guide's Part 2 asks for: an event that outlives its binding is
   unattributable too.
3. **The Community ID seed, zero.** Not a Splunk setting, but the reason the `community_id` field
   joins against Zeek and Suricata data in the same estate. If any producer changes the seed, they
   all must.
4. **The `event_time` rendering.** `props.conf` anchors `_time` on `event_time` arriving as an RFC 3339
   UTC string, for example `2026-08-30T18:25:00.123456UTC`. Produce it with
   `morpheus.utils.siem_wire.render_event_time_series` before the sink. Sending the pipeline's raw
   nanosecond integer instead is silent and severe: verified on a live instance, such an event is
   indexed at ingest time rather than its own, and where it follows a parsable event from the same
   source it inherits *that* event's timestamp, which looks plausible and is wrong.
   `tests/morpheus/utils/test_siem_wire.py` reads this app's `props.conf` directly and fails if the
   two sides drift.

5. **The two lists the layer 2 rules depend on.** R-D-L2-001 reads `lookups/port_designations.csv`,
   one row per port as `site_id:switch_id:port_id` with a `designation` (`single-host` is what the rule
   matches) and `max_macs`. R-D-L2-003 honours the HSRP and VRRP exclusion list, but that list lives in
   the pipeline, as `TC2ArpStage(excluded_sender_ips=[...])`; the stage marks excluded rows and the
   search reads the mark. The two fail in opposite directions until their lists exist: R-D-L2-001 matches
   nothing, and R-D-L2-003 matches every redundancy gateway once per window. Both are unusable
   without the list, which is the guide's own statement, but only one of them is quiet about it.
   R-B-L2-002 reads the same designation list the other way, to leave out the ports marked `trunk` or
   `lag-member`, whose MAC counts move with whatever is downstream; without the list it measures every port.
   Two more are the estate's to declare, and live in the pipeline: the management prefixes
   `TC3ReachStage(management_prefixes=[...])` classifies destinations by, which R-C-003's last step reads
   and `lookups/vlan_designations.csv` should list with the same subnets, and the change window
   `TC1DeviceStage(change_window=["HH:MM", "HH:MM"])` places configuration changes against, without which
   R-D-L1-008 matches nothing.
6. **The name of the binding source, on bucketed rows.** `Binding lookup - L2/L3 refresh` selects
   `binding_table=dhcp_lease`, because several binding sources land on the one `binding:bucketed`
   sourcetype and a refresh that cannot tell them apart builds the wrong lookup. The producer supplies it:
   `BindingTable.to_bucketed_records(table_name="dhcp_lease")`. Leave it unset and the refresh matches
   nothing and the KV Store stays empty, which is the same posture as the two lists above. There are two
   such sources now: `Binding lookup - L1 history refresh` selects `binding_table=port_inventory` on the
   same sourcetype, so the discriminator is what keeps a port history out of an IP-to-MAC lookup and an
   IP-to-MAC lease out of a port history.
7. **Which layer 1 lookup a search reaches for.** `binding_l1` answers what is in a port *now*;
   `binding_l1_history` answers what was in it on a given day, and holds a row only for a port whose
   binding has been superseded -- its optic replaced or its LLDP neighbour changed, which includes the poll a
   restarted switch reports no neighbour. A walk that needs a historical answer consults the history first and falls
   back to the current row on a miss, which is correct because a port that never changed is described
   for all time by the row the current lookup holds. Reaching for `binding_l1` alone is the silent
   failure: an investigation into last Tuesday resolves that port to the optic installed on Wednesday,
   with nothing to indicate the answer is from the wrong interval. `transforms.conf` carries the full
   two-lookup walk in its header comment.
8. **What each index holds about a person, beside how long it holds it.** `indexes.conf` states this per
   index, in the categories `morpheus.utils.personal_data` classifies columns into. It is not a legal
   position and sets no lawful basis; it is the half of a retention decision that was missing, since a
   period is not a decision until somebody can say what it applies to. `behavior_events` carries all five
   personal categories for 395 days, and most of what it carries is behavioural profile the pipeline derived
   rather than anything a collector sent. `behavior_bindings` carries addresses and locations for 400 days
   and is, by design, the thing that re-identifies the rest. Minimization is upstream of all of it --
   `morpheus.stages.lineage.minimization_stage.MinimizationStage` before the sink -- and is not a substitute:
   what reaches an index is a pipeline decision, how long it stays is this file's.
9. **Provisional bindings, if enabled.** `TC2BindingStage(emit_open_bindings=True)` emits a record on
   sourcetype `binding:l2:open` the moment a binding opens, with a null `bind_end`. Whatever builds the
   live lookup from those must cap the open interval with an explicit assumed duration
   (`BindingTable.from_dataframe(open_end_duration_ns=...)`, the source's own aging interval is the
   right value) and let the closed record on `binding:l2`, same `mac_address` and `bind_start`,
   supersede it.
10. **Where late rows go, and how to backfill them.** `WindowSealStage` marks a row that arrives after its window
    sealed with `is_late=true` and `window_complete=false`. Route those rows to sourcetype `morpheus:late` in
    `behavior_late`, never to `behavior_events`: every detection's base search filters on
    `window_complete=true` as the second guard, but the first is that a late row never reaches the index the
    detections read. `Late arrivals by class` counts them. The backfill procedure, when a class has delivered
    late rows that matter:

    1. Find the windows: `index=behavior_late sourcetype=morpheus:late | stats values(window_id) BY telemetry_class`.
    2. Replay the class's source records for those windows, late rows included, through the same pipeline
       configuration -- the same `pipeline_fingerprint` and `config_hash` -- with a sealer whose lateness bound
       covers them. Determinism control 13 is what makes the replay's rows for every row that was on time
       identical to the ones already indexed, so `event_uid` matches.
    3. Index only the rows whose `event_uid` is not already in `behavior_events` for those windows, onto the
       class's score sourcetype. Nothing already indexed is rewritten.
    4. Re-dispatch each detection over the replayed windows with `dispatch.earliest_time` and
       `dispatch.latest_time` set to their bounds. Each writes to `behavior_risk` again; Chain assembly counts a
       record once per `event_uid`, and suppression keeps the notables from repeating.
    5. Delete nothing from `behavior_late`: it is the record that the window was reported on before it was whole.

    Every score stanza's `MAX_DAYS_AGO` has to exceed the oldest window replayed, or the backfill is silently
    re-timed at index time.
11. **The envelope's clock fields, and the quarantine.** `EnvelopeValidateStage` holds a record out of the
    stream when its collector's `clock_source` is `unsynchronized` or its `|clock_offset_ms|` exceeds the stage's
    `max_offset_ms` (1000 by default), and the pipeline sends it to `morpheus:quarantine` in
    `behavior_quarantine`. A quarantined record is timed on `ingest_time`, so it is filed when it arrived rather
    than wherever the broken clock said. Fix the collector and replay the quarantined records by the backfill
    procedure above.
12. **Thresholds, and tuning them from a baseline period.** Most thresholds in these searches are the guide's
    placeholders: the right shape, not a value for any estate. `lookups/rule_thresholds.csv` lists all seventy-two,
    each with its kind, and for a placeholder that a scored event carries, the field and quantile to derive it
    from. Export a period the estate believes was quiet, one `morpheus_score_<layer>.jsonlines` file per score
    sourcetype, and run

        python validate/derive_thresholds.py --events <export> --start YYYY-MM-DD --end YYYY-MM-DD

    for a proposal beside every shipped value. Accepting one means editing the stanza, its row in
    `rule_thresholds.csv` and the rule's `thresholds_hash` in `rule_metadata.csv` together
    (`morpheus.utils.rule_thresholds.thresholds_hash` computes it); `tests/morpheus/utils/test_rule_thresholds.py`
    fails until the three agree. Every detection looks `thresholds_hash` up into its risk record, so notables from
    before and after a retune are told apart by a field rather than by memory.

## What to expect once data flows

- The two layer 1 refresh searches populate the KV Store within their first scheduled cycle: the L1 refresh reads
  `binding:l1`, and the L1 history refresh reads `binding:bucketed` with `binding_table=port_inventory`.
  `| inputlookup binding_l1 | head 5` confirms rows are landing. `binding_l2_l3` fills from the DHCP leases
  `TC2BindingStage` closes from a DHCP server's acknowledgements and releases with a `grant_seconds_column`,
  expanded with `to_bucketed_records(key_name="ip", table_name="dhcp_lease")`; `binding_l2` from the MAC table
  bindings, `binding_table=mac_table`; and `binding_l5_l2` from the 802.1X supplicant bindings closed with
  `key_column="dot1x_identity"`, expanded with `table_name="dot1x_supplicant"`. The directory that says which
  identity a principal presents is the estate's own export, `lookups/principal_directory.csv`, and ships
  header-only. `Identity walk - sign-in to switch port` walks a sign-in to a port through the three and shows the
  port the pipeline resolved beside the one the walk found; the two disagreeing is a lookup refresh behind.
- `binding_l1_history` is the one lookup that is *expected* to be empty in a healthy estate, and stays
  empty until somebody replaces an optic or moves a fibre. Do not read an empty result there as a broken
  refresh; read it against `| inputlookup binding_l1 | stats count`, which should hold one row per port
  from the first cycle onward.
- `behavior_summary` starts filling on the 5-minute cadence, lagged by the 15-minute lateness
  horizon. R-P-L3-005 reads from it; the five chained rules read the scored events directly. Detections
  trail real time by design; the guide's Part 5 explains why that trade is correct.
- `behavior_risk` fills as the detections fire: one record per row a detection returns, carrying the rule, its
  risk score, the layer, the accused entity, the lineage it was accused on and the `thresholds_hash` of the
  numbers it fired at. Overlapping windows write a row
  more than once; suppression throttles the notable, not the record, and Chain assembly counts each record once.
  `index=behavior_risk | stats count BY rule_id` is the quickest way to see which rules are firing at all.
- `behavior_health` fills one record per sealed window, collector, class and tenant. `Collector health` returns
  a row for any collector that skipped or repeated sequence numbers, reported an offset beyond a second, or
  delivered later than five minutes after the sensor saw a record; an empty result there is a healthy estate.
  `Pipeline health - stage counts` returns a row for any collector, class and tenant where a stage could not
  do its job: rows with no entity key, unpaired or abandoned exchanges, unclassified actions, samples out of
  order, saturated counts, or rows a fallback model scored. A count none of a class's stages writes is null and
  stays out of the sum, so a class appears only for something it can answer.
  `Platform baseline step` needs a thirty-day history before it can say anything: each collector, class and
  tenant's baseline is mature after twenty-four windows, and a step is two consecutive windows at twice its peak.
- The `Binding health - unresolved rate` alert is the canary for the soft-join substrate. If it
  fires, the collector is losing lease or expiry records, and attributions are degrading into
  guesses; fix collection before trusting anything downstream.

## Features without a rule

The feature stages of every composed corpus, layers 1 to 7, emit columns no search here reads and no committed model
under `examples/layer5_model/models/` takes as an input, and each is listed with the reason, so a reader can tell a
feature kept for the analyst or for a model from one nobody uses. A read is a column named in a `search =` body; a
comment that names a column to say why no search reads it is not one. The design guide's Part 6 carries the same
table, and `tests/morpheus/determinism/test_feature_coverage.py` keeps both honest against the stages, the models and
these searches.

| Feature | Stage | Why no rule reads it |
| --- | --- | --- |
| `counter_wrapped`, `sample_out_of_order` | `TC1NormalizeStage` | How a delta was arrived at: a wrapped counter still yields one, an out-of-order sample yields none. `PipelineHealthStage` counts the second per collector; the `Device restart` report reads `counter_reset`. |
| `identity_renamed`, `reported_entity_key` | `TC1NormalizeStage` | With an anchor configured, whether the port's reported names differ from the anchored ones, and the key as reported. Not a detection: a rename is configuration. `reported_entity_key` is what a renamed port's layer 2 `port_key` matches (gap G39). |
| `crc_errors_delta`, `symbol_errors_delta`, `input_discards_delta`, `output_discards_delta`, `if_hc_in_octets_delta`, `if_hc_out_octets_delta` | `TC1NormalizeStage` | The counter changes `TC1RateStage` divides by the interval. R-B-L1-005 reads `error_rate` and `discard_rate`; R-B-L1-007 reads `bits_per_second` and `utilization`. |
| `bits_in_per_second`, `bits_out_per_second` | `TC1RateStage` | The two directions behind `bits_per_second` and `utilization`, which R-B-L1-007 reads. |
| `error_rate_baseline_buckets`, `bits_per_second_baseline_buckets` | `TC1RateStage` | How deep each history is. R-B-L1-005 and R-B-L1-007 read `_baseline_mature`, which is this against the stage's floor. |
| `bits_per_second_step` | `TC1RateStage` | The departure above the port's record. R-B-L1-007 fires in both directions, so it reads `bits_per_second` against `_baseline_max` and `_baseline_min` rather than the one-sided step. |
| `optical_tx_dbm_baseline`, `optical_tx_dbm_baseline_samples` | `TC1OpticalStage` | The transmit history behind `optical_tx_dbm_deviation`, which R-D-L1-002 carries as context. The rule fires on the receive side and reads that side's baseline. |
| `optical_rx_dbm_trend_residual_db`, `optical_rx_dbm_trend_significance` | `TC1ForecastStage` | How well the line fits and how far the slope is from flat: the figures for tuning the stage's two refusals. R-P-L1-004 reads `optical_rx_dbm_days_to_floor`. |
| `link_flap_unpolled`, `link_flap_last_change_inconsistent` | `TC1FlapStage` | How a count was arrived at: transitions the device recorded between two polls that saw the same status, and a status change the device's own record did not show. R-D-L1-003 reads the count and `link_flap_device_reset`. |
| `lldp_neighbor_chassis_id_distinct_count` | `TC1ChangeStage` | How many neighbours the port has ever reported: context for `_changed` and `_first_seen`, which R-D-L1-006 reads. |
| `bind_end_observed`, `bind_provisional` | `TC1BindingStage`, `TC2BindingStage` | Whether the source reported the end itself, and whether the record is an open binding emitted before its close. The lookup refreshes load every binding; R-D-L2-004 reads `bind_end_reason` and `bind_gap_ns`. |
| `ports_per_mac`, `ports_per_mac_first_in_window`, `ports_per_mac_saturated` | `TC2CardinalityStage` | How many ports one address was seen on inside the window, with its event flag and sample cap. R-D-L2-004 reads the same condition from the closed bindings on `binding:l2`, where `bind_gap_ns` says how close the two sightings were, and no shipped search reads the count. |
| `ouis_per_vlan_saturated` | `TC2CardinalityStage` | The sample cap on one VLAN's vendors. R-B-L2-006 carries `ouis_per_vlan_first_in_window` and fires on the step. |
| `macs_per_port_baseline_buckets`, `ouis_per_vlan_baseline_buckets` | `TC2BaselineStage` | How deep the history is. R-B-L2-002 and R-B-L2-006 read the step, which is null until the baseline is mature. |
| `macs_per_port_baseline_mature` | `TC2BaselineStage` | Folded into `macs_per_port_step`, which R-B-L2-002 fires on and which is null until the baseline is mature. |
| `arp_is_gratuitous`, `gratuitous_arp_count`, `arp_count_in_window`, `gratuitous_arp_ratio`, `gratuitous_arp_ratio_saturated` | `TC2ArpStage` | The share of one sender's ARP that was unprompted, with its per-row flag, its two counts and its sample cap. The telemetry corpus's flood raises it and the harness holds it to its counts; no shipped search reads it. R-D-L2-003 reads `macs_claiming_sender_ip`, the other shape poisoning takes. |
| `macs_claiming_sender_ip_saturated` | `TC2ArpStage` | The sample cap on one address's claimants, which makes the count a lower bound. R-D-L2-003 fires on more than one either way. |
| `auth_abandoned` | `TC2AuthStage` | The pending exchanges this row's arrival timed out. `PipelineHealthStage` sums it per collector into `morpheus:health`; R-D-L2-005 reads `auth_unpaired`. |
| `dsts_per_src_first_in_window`, `srcs_per_dst_first_in_window`, `dst_ports_per_src_first_in_window` | `TC3CardinalityStage` | Says which flow raised the count. The rules read the count; this is for the analyst reading the row. |
| `internal_dsts_in_window` | `TC3ReachStage` | The numerator of `internal_dst_ratio`, which R-B-L3-001 reads instead. |
| `internal_dst_ratio_saturated` | `TC3ReachStage` | The ratio's sample cap. R-B-L3-001 carries `dsts_per_src_saturated`, the cap on the count it fires on. |
| `dsts_per_src_baseline_buckets`, `srcs_per_dst_baseline_buckets`, `byte_asymmetry_baseline_buckets` | `TC3BaselineStage` | How deep the history is. The rules read `_baseline_mature`, which is this against the stage's floor. |
| `flow_regularity_saturated` | `TC3BeaconStage` | The interval cap on one pair. A beacon is regular long before it binds. |
| `ip_ttl_saturated` | `TC3TtlStage` | The sample cap on one source's TTL samples, which a stable path never reaches. |
| `flow_bpp`, `flow_ppm`, `flow_psh`, `flow_fin`, `flow_ackpush_ratio`, `flow_fin_ratio`, `flow_rst_ratio`, `flow_syn_ratio` | `TC4FlowStage` | Per-flow model inputs. The searches recover the counts from their maxima rather than aggregate a running ratio -- an ordinary handshake opens with a bare SYN, so `max(flow_syn_ratio)` over it is 1.0, the scan's figure -- and no layer 4 model is pinned yet (#59). |
| `flow_bpp_envelope` | `TC4EnvelopeStage` | The envelope R-B-L4-005 measures against; it fires on `flow_bpp_envelope_breached`. |
| `flow_bpp_envelope_mature`, `flow_data_len_envelope_mature` | `TC4EnvelopeStage` | Folded into the `_breached` flags, which are null until the envelope is mature. |
| `logcount_saturated`, `asns_in_window_saturated` | `TC5NoveltyStage` | The sample caps on the two windowed features, which make them lower bounds. `PipelineHealthStage` counts saturated rows per collector. |
| `app_first_seen` | `TC5NoveltyStage` | Whether the principal has used this application before. R-D-L5-008 reads `location_first_seen` and `device_first_seen`; the application enters the score as `appincrement`. |
| `local_weekday`, `hour_share`, `weekday_share`, `weekday_unseen` | `TC5CadenceStage` | The weekday side of the histogram, and the smoothed shares behind both surprises. R-D-L5-007 reads `hour_unseen`; the weekday enters the score as `weekday_surprise_bits`. |
| `mfa_challenge` | `TC5RiskStage` | Whether this row was a multi-factor challenge. R-D-L5-004 reads the window counts, `mfa_attempts_in_window` and `mfa_denials_in_window`; the proportion enters the score as `mfa_ratio`. |
| `logcount_z_loss`, `appincrement_z_loss`, `deviceincrement_z_loss`, `asns_in_window_z_loss`, `hour_surprise_bits_z_loss`, `weekday_surprise_bits_z_loss`, `mfa_ratio_z_loss`, `auth_attempts_in_window_z_loss`, `auth_failures_in_window_z_loss` | `TC5ScoreStage` | The per-feature losses behind `mean_abs_z` and `max_abs_z`, which R-B-L5-001 reads. R-B-L5-002 reads `locincrement_z_loss` alone, the feature its rule names; the rest say which feature drove a score. |
| `drift_run_restarted` | `TC5DriftStage` | Whether a gap in the entity's windows restarted the run. The trajectory rules read `drift_rising_windows`, which the restart has already reset. |
| `session_duration_ns` | `TC5SessionStage` | The duration in nanoseconds. R-B-L5-005 reads `session_duration_s`, the TC-5 field. |
| `session_starts`, `session_unpaired`, `session_out_of_order`, `session_abandoned`, `session_unclassified` | `TC5SessionStage` | How the pairing went: a repeated start, a stop with no start, a stop before its start, the sessions this row's arrival timed out, an action that was neither. `PipelineHealthStage` counts them per collector into `morpheus:health`. |
| `ja4_client_changed` | `TC6FingerprintStage` | Measured, not reasoned about: in the layer 6 corpus every change on a settled host is either a first sighting, which R-B-L6-001 reads, or the host alternating between two stacks it has always had, which is that rule's control. A rule on the change alone fires only on the control. |
| `tls_client_key` | `TC6FingerprintStage` | The key the stage keeps a host's stacks on. Searches group by `src_ip`. |
| `cert_self_signed`, `cert_destination_is_global` | `TC6CertificateStage` | The two halves of `cert_self_signed_external`, which R-D-L6-003 reads. The first alone returns every self-signed management interface in the estate. |
| `cert_issuer_saturated` | `TC6CertificateStage` | The issuer cap on one destination, which a destination presenting more is already past. |
| `cipher_rank`, `cipher_floor` | `TC6CipherStage` | The numbers behind `cipher_tier` and `cipher_floor_tier`, which R-B-L6-004 reports, and `cipher_downgraded`, which it fires on. |
| `cipher_saturated` | `TC6CipherStage` | The suite cap on one pair. |
| `dns_client_key` | `TC7DnsStage` | The key the stage keeps a client's windows on. Searches group by `src_ip`. |
| `dns_subdomains_saturated`, `dns_domains_saturated` | `TC7DnsStage` | The sample caps on one domain's subdomains and one client's domains, which make the counts lower bounds. R-B-L7-001 and R-B-L7-008 fire on the counts either way. |
| `dns_answered_in_window` | `TC7DnsStage` | The denominator of `dns_nxdomain_ratio`, which R-B-L7-008 reads with `dns_nxdomain_in_window`. |
| `dns_subdomains_per_domain_baseline`, `dns_subdomains_per_domain_baseline_days`, `dns_domains_per_client_baseline`, `dns_domains_per_client_baseline_days` | `TC7DnsStage` | The client's busiest day and how many days it is over. R-B-L7-001 and R-B-L7-008 read the ratio against it and `_baseline_mature`. |
| `http_client_key` | `TC7HttpStage` | The key the stage keeps a client's windows on. Searches group by `src_ip`. |
| `http_status_class` | `TC7HttpStage` | The status reduced to its class. R-D-L7-005 reads the two counts the classes are summed into. |
| `http_4xx_to_2xx_ratio` | `TC7HttpStage` | Null for a client with no 2xx at all, the purest enumerator there is, so R-D-L7-005 tests `http_4xx_in_window > 0.7 * http_2xx_in_window`, the same inequality with the division removed. |
| `http_window_saturated` | `TC7HttpStage` | The sample cap on one client's requests, which makes the counts lower bounds. |
| `http_distinct_paths_baseline`, `http_distinct_paths_baseline_days` | `TC7HttpStage` | The client's busiest day and how many days it is over. R-D-L7-005 reads the ratio against it and `_baseline_mature`. |
| `http_ua_family_first_seen` | `TC7HttpStage` | Whether this client has sent this user agent family before. R-B-L4-004 reads `http_ua_os`; the novelty is per client rather than per principal, and no rule reads it until it is. |
| `service_connections_in_window` | `TC7ServiceStage` | The service's connections, beside its bytes and clients. R-B-L7-007 fires on the clients and carries the bytes. |
| `service_clients_saturated` | `TC7ServiceStage` | The sample cap on one service's connections, which makes the client count a lower bound. |
| `service_client_fingerprints` | `TC7ServiceStage` | How many client fingerprints the service has ever seen: context for `service_client_fingerprint_first_seen`, which R-B-L7-007 reads. |
| `service_clients_baseline_days` | `TC7ServiceStage` | How many days the busiest day is over. R-B-L7-007 reads `service_clients_baseline_mature`. |
| `saas_baseline_key` | `TC7SaasStage` | The principal and operation the baseline is kept on. R-B-L7-002 groups by `user_principal` and `operation`. |
| `saas_baseline_samples`, `saas_baseline_saturated` | `TC7SaasStage` | How many priors the quantile is over, and whether the cap bound. R-B-L7-002 reads `saas_baseline_mature`. |
| `saas_operation_failed` | `TC7SaasStage` | Whether the operation reached nothing and so entered neither measurement; its ratio and breadth are null. Refusals are worth a rule of their own, which is not written. |
| `saas_week_id` | `TC7SaasStage` | The week the breadth count is in, from the stage's own epoch. R-P-L7-006 reads the weekly seal's window beside it. |
| `endpoint_host_seen`, `endpoint_peer_seen`, `endpoint_mature` | `TC7EndpointStage` | The halves of `endpoint_pair_novel`, which R-B-L7-004 reads. Carried so an analyst can see which half decided. |

## Packaging

To produce an installable package for Splunkbase-style distribution:

```bash
COPYFILE_DISABLE=1 tar -czf TA-morpheus-lineage.spl TA-morpheus-lineage
```

## How this app was validated

Three levels, strongest last:

1. **AppInspect.** `splunk-appinspect inspect TA-morpheus-lineage --mode precert` passes with zero
   failures. The remaining warning is informational (the app contains `collections.conf`, which is
   expected).
2. **Live load.** The app was installed into a fresh Splunk Enterprise 10.2 instance: `btool check`
   reports no errors, all five indexes are created, all seven scheduled searches that existed at the
   time register, and every one of them executes without a parse error against empty indexes. The app
   ships eighty-six searches now; the seventy-nine added since have not been through this step.
3. **Functional.** With synthetic JSON telemetry seeded into the indexes and bindings written to the
   KV Store: timestamps anchor to `event_time` as the props intend, the identifier ladder resolves an
   IP through both lookups to a physical port and site, the chain assembly search emits the seeded
   cross-layer chain with the expected span and risk, and R-C-002 detects its ordered sequence with
   the expected gap. That seed carried edge and risk fields the pipeline does not yet emit; against pipeline
   output the chain assembly returns no rows and `binding_l2_l3` is empty (see `validate/VALIDATION.md`). That R-C-002 correlated two detections' notables; it has since been rewritten to read the
   scored events, and the rewrite returned its one expected row in the search-head run of 2026-10-09.

Several things were added after that validation and have **not** been run against a live instance: the
`binding:l2` and `binding:l2:open` sourcetypes, the `port_designations` lookup, the layer 2
detections `R-D-L2-001`, `R-D-L2-003`, `R-D-L2-004`, `R-D-L2-005` and, later, `R-B-L2-002`, and -- added later still -- the
`morpheus:score:l5` sourcetype with the two layer 5 detections `R-D-L5-003` and `R-D-L5-004` and the
predictive watchlist `R-P-L5-006`, then the `morpheus:score:l3` sourcetype with five layer 3 detections and,
later, three more (`R-B-L3-006`, `R-B-L3-007` and `R-D-L3-008`),
then `morpheus:score:l4` with three more, `morpheus:score:l6` with five, `morpheus:score:l7` with five, the chained `R-C-001`, `R-C-004`
and `R-C-005`, with `R-C-002` as rewritten, the layer 1 detections `R-D-L1-001` and `R-P-L1-004`, and the layer 5 baseline searches `R-D-L5-007`, `R-D-L5-008` and `R-D-L5-009` with the gated `R-B-L5-001` and `R-B-L5-002` and the session duration rule `R-B-L5-005`, together with the `principal_watchlist` lookup and its expiry job, and then the network-object detections: `R-D-L1-002`, `R-D-L1-003`, `R-B-L1-005`, `R-D-L1-006` and `R-B-L1-007` at layer 1, `R-B-L2-006`, `R-B-L2-007` and `R-B-L2-008` at layer 2, with the device restart report and the `lldp_adjacency` lookup, and then the service and stack detections: `R-B-L7-007`, `R-B-L7-008` and `R-B-L4-004`, and the device and physical-chain detections: `R-D-L1-008`, `R-P-L1-009` and `R-C-003`, and the host drift watchlist `R-P-L3-009` over the `morpheus:score:host` sourcetype, with the `host_watchlist` lookup and its expiry job, and the stack and subdomain trajectories `R-P-L6-006` and `R-P-L7-009`, which write the same list, and `R-D-L1-010`, which reads the commissioning level the network inventory on `context:network` gave each layer 1 row, and the path count and process breadth trajectories `R-P-L7-010` and `R-P-L7-011`, which write the hosts' list too. That is all sixty-one detection searches this app ships, so the live
pass above covers the app's oldest part and none of its detections as they now stand; the search-head run
described below has since run all of them. Their SPL follows
the same scheduling discipline as the validated searches, and the predicates they encode are asserted in
Python over the determinism harnesses' planted corpora (`tests/morpheus/determinism/test_first_detections.py` for
the first layer 1, 2 and 5 rules, `test_telemetry_harness.py` for the network-object rules, and the network, transport, presentation, application, SaaS, endpoint and campaign
harnesses beside it for the rest),
where each fires on exactly the planted cases and nothing else -- twice for `R-D-L2-004`, which the corpus
plants both a simultaneous and a cross-switch spoof for, alongside a legitimate move it must not fire on. That is evidence the columns and conditions are right; it is not evidence
the stanzas parse on a search head. Run `btool savedsearches list` after installing.

The run that would settle that is packaged: `validate/run_search_head.sh` starts the validation container, indexes
the sample events, dispatches all eighty-six searches in the order `validate/VALIDATION.md` prescribes, and writes
`validate/search_head_results.json`, which `tests/morpheus/determinism/test_search_head_run.py` compares with
`expected_results.json` while it is committed; when a change supersedes a run, the run moves to
`validate/search_head_runs/` with a test of exactly how it differs from what is now written. It needs Docker and nothing else. It was first run on 2026-10-05, on Splunk 10.2.8:
every one of the forty-eight searches ran without error over the pipeline's own events, and forty-six returned
what was written. The two that did not were nulls on the wire -- a field sent as `null` is not absent to Splunk, and
`macs_per_port_step>0` and `resolution_method=*` both let null values through. The serializer
(`morpheus.utils.siem_wire.to_wire_lines`) now leaves null fields out. That run is kept in
`validate/search_head_runs/`, as is the run over the regenerated events, on 2026-10-09 on the same version, which
indexed all 8,400 events and returned what was then written for all forty-eight. A third run the same day, with
every detection collecting into `behavior_risk`, made on Splunk installed from its tarball and kept in the same
directory, found the index held the 94
records the detections returned, Chain assembly found the one three-layer chain a detection accuses at 55 against
a threshold of 60, and R-P-L3-005 fired fifteen times once it took two passes over the summary -- its own SPL had
read a field its `streamstats` was still creating, which is why the second run's zero agreed with an expectation
that blamed the summary. A fourth, on the Docker image this package ships with, agreed with it on every search and
every check. A fifth, that night on the tarball install, ran over the events regenerated when each layer 5
principal gained a learned model of their own: it indexed all 8,680 events, returned what was then written for
all forty-eight -- R-B-L5-001 and R-B-L5-002 their first rows, 42 and 28 -- and held the 166 risk records the
detections returned. A sixth, on 2026-10-10 on the same install, ran over the events step 9 regenerated when layer 3
gained a fortnight of history and each host a baseline: it indexed all 9,559 events, returned what is written for
all fifty-two -- the three new layer 3 detections and the connection evidence report one row each -- and held the 198
risk records the detections returned. A seventh, later the same day on the same install, ran over the events step
10 regenerated when the estate gained its network-object plants: it indexed all 10,285 events, returned what was then
written for all sixty-two -- each of the ten new searches what its plant leaves for it -- and held the 209 risk
records the detections returned. An eighth, that night on the same install, ran over the events step 11 regenerated
when the pipeline began reporting on itself: it indexed all 11,352 events across sixteen sourcetypes, returned what
was then written for all sixty-six -- each of the four health reports one row -- and held the same 209 risk records,
every detection now reading only complete windows. A ninth, on 2026-10-15 on the same install, ran over the events
step 12 regenerated when hosts gained an identity across layers: it indexed all 16,423 events, returned what is
then written for all seventy-two -- the L2/L3 refresh its first rows, the host evidence report seventeen machines and
the identity walk all twenty-five sign-ins -- and held the same 209 risk records. Its first attempt found the walk empty, because the refreshes keyed rows on the
producer's bucket ordinal while the walk computed its bucket from a `_time` the run had moved; every refresh now keys
its rows on the bucket its own time falls in. A tenth, on 2026-10-16 on the same install, ran over the events step
13 regenerated when a service and an application became entities: it indexed all 20,127 events, returned what is
written for all seventy-five -- R-B-L7-007, R-B-L7-008 and R-B-L4-004 one row each, and R-D-L6-002 three -- held the
213 risk records the detections returned. It counted nineteen machines in the host evidence report where seventeen
were written, which nothing had derived from the events; the package derives it now. An eleventh, on 2026-10-17 on the
same install, ran over the events step 15 regenerated when the switch and the router became entities and R-C-003 got
a chain to find: it indexed all 22,410 events, returned what is written for all seventy-eight -- R-C-003, R-D-L1-008
and R-P-L1-009 one row each, and the restart report four -- and held the 221 risk records the detections returned. A
twelfth, on 2026-10-22 on the same install, ran over the events regenerated when the host segment gained a sourcetype
of its own: it indexed all 24,146 events, returned what was then written for all eighty -- R-P-L3-009 three host-days,
the host watchlist's expiry the same three, and the summary 8801 rows with the host frames among them -- and held the
224 risk records the detections returned. A thirteenth, on 2026-10-23 on the same install, ran over the events
regenerated when the daily drift reached the TLS and DNS columns: it indexed all 24,736 events, returned what is written
for all eighty-two -- R-P-L6-006 the creeping host's last three days, R-P-L7-009 the slow tunnel's last four, R-B-L6-001
seven rows where it returned five, and the host watchlist holding all ten entries until its expiry kept R-P-L3-009's
three -- and held the 233 risk records the detections returned. It is kept in `validate/search_head_runs/` with
the twelve before it, because on 2026-10-24 the estate corpus gained the desk spoof, the actor Chain assembly exists
for, and the chain search's expectation went from no rows to one. A fourteenth, on 2026-10-26 on the same install, ran
over the events regenerated for the desk spoof and the network inventory on `context:network`: it indexed all 24,766
events, returned what was then written for all eighty-three -- R-D-L1-010 two ports, and Chain assembly the tapped
desk's chain at 135 -- and held the 235 risk records the detections returned. It is kept in `validate/search_head_runs/`
too, because on 2026-10-28 plan section 3.3 gave the HTTP and endpoint classes a daily trajectory each, and a fifteenth,
on the same install, ran over the events that regenerated: it indexed all 25,632 events, returned what was then
written for all eighty-five -- R-P-L7-010 the slow enumerator's last four days, R-P-L7-011 the creeping build server's
last three, and R-B-L7-004 the build server's `curl` at 75 for the watchlist entry -- and held the 242 risk records the
detections returned. It is kept in `validate/search_head_runs/` as well, because plan section 3.7 added `Pipeline
health - stage counts`, and a sixteenth, on 2026-10-31 on Splunk 10.2.8 unpacked from the official tarball and run as
its own user rather than in the container, indexed the same 25,632 events, returned what is written for all
eighty-six -- the new search its five classes -- and held the same 242 risk records, every one of them carrying the
`thresholds_hash` its rule's `rule_metadata` row holds.

One wrinkle from that validation worth knowing when testing by hand: the sourcetypes declare
`KV_MODE = json`, so events seeded with `| collect` in its default stash rendering extract no fields
at search time and every query silently matches nothing. Seed test events with a JSON `_raw`
(`| eval _raw=json_object(...)`) including an `event_time` field, exactly as the Morpheus pipeline
emits them.

## Relationship to the design guide

The stanzas here are the normative copies of the fragments quoted in the guide's Part 4. When the
two disagree, this app is what was meant to run. The searches follow the guide's scheduling
discipline throughout: no search whose output a rule consumes ever ends its window at `now`, every
window is snapped to the minute, and every job uses continuous scheduling so skipped runs are
caught up rather than abandoned.
