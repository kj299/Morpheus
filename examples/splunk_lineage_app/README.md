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
binding rows as described in the guide, typically through Splunk Connect for Kafka.

## Contents

| File | Deploy to | What it defines |
| --- | --- | --- |
| `default/indexes.conf` | Indexers | `behavior_events`, `behavior_lineage`, `behavior_bindings`, `behavior_context`, `behavior_summary`, with deliberately asymmetric retention |
| `default/props.conf` | Indexers or heavy forwarders | One JSON sourcetype per OSI layer plus edges, bindings, and context, each with `_time` anchored to `event_time` |
| `default/collections.conf` | Search heads | KV Store collections for the L2/L3 bucketed bindings and the unbucketed L1 bindings, with accelerated fields |
| `default/transforms.conf` | Search heads | The `binding_l2_l3` and `binding_l1` lookups |
| `default/savedsearches.conf` | Search heads | Lookup refresh and expiry jobs, the 5-minute summary rollup, chain assembly, the R-C-002 sequence detection, the four layer 2 detections R-D-L2-001, 003, 004 and 005, and a binding health alert |
| `lookups/port_designations.csv` | Search heads | The port designation list R-D-L2-001 reads: `port_key,designation,max_macs`. Ships header-only; populate it from the inventory |

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

## Settings that must match the pipeline

These values are shared contracts between this app and the Morpheus pipeline. Changing either
side alone breaks the joins silently.

1. **Bucket width, 300 seconds.** The pipeline expands bindings with
   `BindingTable.to_bucketed_frame(bucket_seconds=300)`, and every query in the guide computes
   `bucket=floor(_time/300)`. The expiry saved search also assumes it. Each bucketed row carries
   `bucket_start`, the bucket's own start time, which is what `[binding:bucketed]` anchors `_time` on. A
   bucketed row has no other time of its own: it stands for a key in a bucket, not for a single binding
   record, so it carries neither `bind_start` nor `bind_end`.
2. **Binding retention, 400 days.** `frozenTimePeriodInSecs` on `behavior_bindings` and the cutoff
   in the `Binding lookup - L2/L3 expiry` search must move together. A lookup that expires before
   its index produces unattributable events.
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
6. **The name of the binding source, on bucketed rows.** `Binding lookup - L2/L3 refresh` selects
   `binding_table=dhcp_lease`, because several binding sources land on the one `binding:bucketed`
   sourcetype and a refresh that cannot tell them apart builds the wrong lookup. The producer supplies it:
   `BindingTable.to_bucketed_records(table_name="dhcp_lease")`. Leave it unset and the refresh matches
   nothing and the KV Store stays empty, which is the same posture as the two lists above.
7. **Provisional bindings, if enabled.** `TC2BindingStage(emit_open_bindings=True)` emits a record on
   sourcetype `binding:l2:open` the moment a binding opens, with a null `bind_end`. Whatever builds the
   live lookup from those must cap the open interval with an explicit assumed duration
   (`BindingTable.from_dataframe(open_end_duration_ns=...)`, the source's own aging interval is the
   right value) and let the closed record on `binding:l2`, same `mac_address` and `bind_start`,
   supersede it.

## What to expect once data flows

- The two refresh searches populate the KV Store within their first scheduled cycle; `| inputlookup
  binding_l2_l3 | head 5` confirms rows are landing.
- `behavior_summary` starts filling on the 5-minute cadence, lagged by the 15-minute lateness
  horizon. Chained rules read from it, so detections trail real time by design; the guide's Part 5
  explains why that trade is correct.
- The `Binding health - unresolved rate` alert is the canary for the soft-join substrate. If it
  fires, the collector is losing lease or expiry records, and attributions are degrading into
  guesses; fix collection before trusting anything downstream.

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
   reports no errors, all five indexes are created, all seven scheduled searches register, and every
   one of them executes without a parse error against empty indexes.
3. **Functional.** With synthetic JSON telemetry seeded into the indexes and bindings written to the
   KV Store: timestamps anchor to `event_time` as the props intend, the identifier ladder resolves an
   IP through both lookups to a physical port and site, the chain assembly search emits the seeded
   cross-layer chain with the expected span and risk, and R-C-002 detects its ordered sequence with
   the expected gap.

Several things were added after that validation and have **not** been run against a live instance: the
`binding:l2` and `binding:l2:open` sourcetypes, the `port_designations` lookup, and the four layer 2
detections `R-D-L2-001`, `R-D-L2-003`, `R-D-L2-004` and `R-D-L2-005`. Their SPL follows
the same scheduling discipline as the validated searches, and the predicates they encode are asserted in
Python over the determinism harness's planted corpus (`tests/morpheus/determinism/test_first_detections.py`),
where each fires exactly once. That is evidence the columns and conditions are right; it is not evidence
the stanzas parse on a search head. Run `btool savedsearches list` after installing.

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
