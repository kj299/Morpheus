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
| `default/indexes.conf` | Indexers | `behavior_events`, `behavior_lineage`, `behavior_bindings`, `behavior_context`, `behavior_summary`, `behavior_risk`, with deliberately asymmetric retention, and a statement of what each one holds about a person beside the period it holds it for |
| `default/props.conf` | Indexers or heavy forwarders | One JSON sourcetype per OSI layer plus edges, bindings, and context, each with `_time` anchored on a field the record carries -- `event_time` for scores and edges, an interval bound for bindings, `valid_from` for context |
| `default/collections.conf` | Search heads | KV Store collections for the L2/L3 bucketed bindings, the unbucketed L1 bindings, the bucketed L1 history beside them, and the principal watchlist the predictive rules write, with accelerated fields |
| `default/transforms.conf` | Search heads | The `binding_l2_l3`, `binding_l1`, `binding_l1_history` and `principal_watchlist` lookups, and `rule_metadata` |
| `default/savedsearches.conf` | Search heads | Forty-eight searches: lookup refresh and expiry jobs, including the principal watchlist's, the 5-minute summary rollup, chain assembly, the chained detections R-C-001, R-C-002, R-C-004 and R-C-005, the two layer 1 detections R-D-L1-001 and the predictive R-P-L1-004, the five layer 2 detections R-B-L2-002 and R-D-L2-001, 003, 004 and 005, the five layer 3 detections R-B-L3-001, R-B-L3-002, R-D-L3-003, R-B-L3-004 and the predictive R-P-L3-005, the three layer 4 detections R-D-L4-002, R-D-L4-003 and R-B-L4-005, the five layer 6 detections R-B-L6-001, R-D-L6-002, R-D-L6-003, R-B-L6-004 and R-D-L6-005, the five layer 7 detections R-B-L7-001, R-D-L7-005, R-B-L7-002, R-B-L7-004 and the predictive R-P-L7-006, the five deterministic layer 5 detections R-D-L5-003, R-D-L5-004, R-D-L5-007, R-D-L5-008 and R-D-L5-009, the two layer 5 model rules R-B-L5-001 and R-B-L5-002, gated on a principal's own model and empty until one is pinned, the layer 5 session duration rule R-B-L5-005, the layer 5 predictive watchlist R-P-L5-006, a binding health alert, and a TLS table coverage metric. Every detection ends by collecting its rows into `behavior_risk`, which Chain assembly sums, and is a per-result alert suppressed on its stated deduplication key for its dispatch window |
| `lookups/port_designations.csv` | Search heads | The port designation list R-D-L2-001 reads: `port_key,designation,max_macs`. Ships header-only; populate it from the inventory |
| `lookups/rule_metadata.csv` | Search heads | One row per detection: `rule_id,saved_search,suppress_fields,suppress_period,hysteresis`. The stanzas' `alert.suppress.*` keys are held to it by a test; `hysteresis` is `none` for every rule until a model scores near a threshold |
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
   nothing and the KV Store stays empty, which is the same posture as the two lists above. There are two
   such sources now: `Binding lookup - L1 history refresh` selects `binding_table=port_inventory` on the
   same sourcetype, so the discriminator is what keeps a port history out of an IP-to-MAC lookup and an
   IP-to-MAC lease out of a port history.
7. **Which layer 1 lookup a search reaches for.** `binding_l1` answers what is in a port *now*;
   `binding_l1_history` answers what was in it on a given day, and holds a row only for a port whose
   optic has been replaced. A walk that needs a historical answer consults the history first and falls
   back to the current row on a miss, which is correct because a port that never changed is described
   for all time by the row the current lookup holds. Reaching for `binding_l1` alone is the silent
   failure: an investigation into last Tuesday resolves that port to the optic installed on Wednesday,
   with nothing to indicate the answer is from the wrong interval. `transforms.conf` carries the full
   two-lookup walk in its header comment.
8. **What each index holds about a person, beside how long it holds it.** `indexes.conf` states this per
   index, in the categories `morpheus.utils.personal_data` classifies columns into. It is not a legal
   position and sets no lawful basis; it is the half of a retention decision that was missing, since a
   period is not a decision until somebody can say what it applies to. `behavior_events` carries all five
   personal categories for 90 days, and most of what it carries is behavioural profile the pipeline derived
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

## What to expect once data flows

- The two layer 1 refresh searches populate the KV Store from `binding:l1` within their first scheduled cycle;
  `| inputlookup binding_l1 | head 5` confirms rows are landing. `binding_l2_l3` stays empty until a DHCP lease
  feed is expanded with `to_bucketed_records(table_name="dhcp_lease")`, which nothing in the fork produces yet
  (issue #63).
- `binding_l1_history` is the one lookup that is *expected* to be empty in a healthy estate, and stays
  empty until somebody replaces an optic or moves a fibre. Do not read an empty result there as a broken
  refresh; read it against `| inputlookup binding_l1 | stats count`, which should hold one row per port
  from the first cycle onward.
- `behavior_summary` starts filling on the 5-minute cadence, lagged by the 15-minute lateness
  horizon. R-P-L3-005 reads from it; the four chained rules read the scored events directly. Detections
  trail real time by design; the guide's Part 5 explains why that trade is correct.
- `behavior_risk` fills as the detections fire: one record per row a detection returns, carrying the rule, its
  risk score, the layer, the accused entity and the lineage it was accused on. Overlapping windows write a row
  more than once; suppression throttles the notable, not the record, and Chain assembly counts each record once.
  `index=behavior_risk | stats count BY rule_id` is the quickest way to see which rules are firing at all.
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
   reports no errors, all five indexes are created, all seven scheduled searches that existed at the
   time register, and every one of them executes without a parse error against empty indexes. The app
   ships forty-eight searches now; the forty-one added since have not been through this step.
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
predictive watchlist `R-P-L5-006`, then the `morpheus:score:l3` sourcetype with five layer 3 detections,
then `morpheus:score:l4` with three more, `morpheus:score:l6` with five, `morpheus:score:l7` with five, the chained `R-C-001`, `R-C-004`
and `R-C-005`, with `R-C-002` as rewritten, the layer 1 detections `R-D-L1-001` and `R-P-L1-004`, and the layer 5 baseline searches `R-D-L5-007`, `R-D-L5-008` and `R-D-L5-009` with the gated `R-B-L5-001` and `R-B-L5-002` and the session duration rule `R-B-L5-005`, together with the `principal_watchlist` lookup and its expiry job. That is all thirty-eight detection searches this app ships, so the live
pass above covers the app's oldest part and none of its detections as they now stand; the search-head run
described below has since run all of them. Their SPL follows
the same scheduling discipline as the validated searches, and the predicates they encode are asserted in
Python over the determinism harnesses' planted corpora (`tests/morpheus/determinism/test_first_detections.py` for
the layer 1, 2 and 5 rules, and the network, transport, presentation, application, SaaS, endpoint and campaign
harnesses beside it for the rest),
where each fires on exactly the planted cases and nothing else -- twice for `R-D-L2-004`, which the corpus
plants both a simultaneous and a cross-switch spoof for, alongside a legitimate move it must not fire on. That is evidence the columns and conditions are right; it is not evidence
the stanzas parse on a search head. Run `btool savedsearches list` after installing.

The run that would settle that is packaged: `validate/run_search_head.sh` starts the validation container, indexes
the sample events, dispatches all forty-eight searches in the order `validate/VALIDATION.md` prescribes, and writes
`validate/search_head_results.json`, which `tests/morpheus/determinism/test_search_head_run.py` compares with
`expected_results.json`. It needs Docker and nothing else. It was first run on 2026-10-05, on Splunk 10.2.8:
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
principal gained a learned model of their own: it indexed all 8,680 events, returned what is written for all
forty-eight -- R-B-L5-001 and R-B-L5-002 their first rows, 42 and 28 -- held the 166 risk records the detections
returned, and is `validate/search_head_results.json`; the earlier four are kept beside it.

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
