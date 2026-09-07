<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

![NVIDIA Morpheus](./docs/source/img/morpheus-banner.png "Morpheus banner image")

# NVIDIA Morpheus

NVIDIA Morpheus is an open AI application framework that provides cybersecurity developers with a highly optimized AI framework and pre-trained AI capabilities that allow them to instantaneously inspect all IP traffic across their data center fabric. The Morpheus developer framework allows teams to build their own optimized pipelines that address cybersecurity and information security use cases. Bringing a new level of security to data centers, Morpheus provides development capabilities around dynamic protection, real-time telemetry, adaptive policies, and cyber defenses for detecting and remediating cybersecurity threats.

## What this fork is for

This fork exists to build one thing: **behavioral analytics that spans all seven OSI layers, feeds a
SIEM, and produces output a detection engineer can reproduce and defend six months later in front of
an auditor.**

Upstream Morpheus supplies the streaming runtime, the per-entity autoencoder, and the feature DSL.
It does not supply the substrate that makes seven layers of telemetry into one story about one entity:
stable identifiers that survive a replay, a way to attribute an IP at layer 3 back to a physical port
at layer 1, windows that close on event time rather than on when the data happened to arrive, and the
per-layer features the detection rules actually need. That substrate is what this fork adds.

The design was written down in full before any of it was built.
[**Predictive Behavioral Analytics Across OSI Layers 1-7**](./docs/source/developer_guide/guides/11_predictive_behavioral_analytics_osi.md)
analyzes the Morpheus codebase in three passes, then specifies the telemetry each layer must produce,
the detection rules worth writing, how to chain layer 1-7 lineage in Splunk down to the configuration
stanzas, and thirteen controls for keeping the output reproducible. Every claim about Morpheus in it
is anchored to a file path, and where a capability does not exist the guide says so rather than
implying the SDK already covers it. The code here implements that guide incrementally; the guide's
[Part 6](./docs/source/developer_guide/guides/11_predictive_behavioral_analytics_osi.md#part-6-gaps-and-build-list)
is the running ledger of what is built and what is not.

### What is built so far

| Area | What ships |
| --- | --- |
| **Lineage substrate** | Deterministic `event_uid` / `link_uid` provenance identifiers, the Community ID flow hash (matching the six reference vectors published by the specification and its normative implementation, plus the bidirectional-collapse property on four flow pairs), time-bounded binding resolution with a fixed tie-break, and event-time window sealing with a lateness horizon and a separate late-arrival stream |
| **Layer 1 (TC-1)** | Interface counter normalization that tells a counter wrap from a device reboot, transceiver and neighbor novelty, optical power scored against each port's own rolling baseline, link flap counting that catches flaps between two polls, identifier change detection with no period boundary, and port bindings closed into the resolvable intervals the `binding_l1` lookup is built from, which completes the identifier ladder: a chain query now reaches a site, an optic and an LLDP neighbour rather than stopping at a port name |
| **Layer 2 (TC-2)** | Binding closure into the half-open intervals the resolver consumes, optionally emitting a provisional record the moment a binding opens so live attribution has an answer inside the idle window, the three cardinality features, the gratuitous ARP proportion, and 802.1X authorization timing with unpaired authorization flagged |
| **Layer 5 (TC-5), deterministic half** | Session assembly from the separate start and stop records most identity providers emit, the digital fingerprinting features this layer is the sweet spot for (`logcount`, `locincrement`, `appincrement`, a device increment, distinct source ASNs per window), hour-of-day and day-of-week deviation scored against each principal's own histogram rather than a population's, implied travel speed between consecutive successful authentications, the failure and multi-factor denial runs that end in a success, and the drift trajectory R-P-L5-006 reads -- velocity, acceleration, the length of the rising run and its total rise in the principal's own standard deviations. The composed pipeline runs under control 13's six checks and the two deterministic rules ship as saved searches. What is still missing is the model that would give the trajectory a score to track -- see below |
| **Scoring determinism** | The envelope every scored event carries and the two hashes inside it, a model manifest resolved once per window that refuses to answer for any other, and entity sharding on a hash that does not change with `PYTHONHASHSEED`. Controls 1, 2, 4 and 12 of the guide's thirteen, built before the model that consumes them because retrofitting determinism onto a running pipeline means re-tuning every threshold |
| **Determinism** | A total row order imposed before any stateful stage, frame canonicalization and digesting, score quantization, and a CI harness running control 13's six checks against all three composed pipelines -- lineage, layer 1 and 2 telemetry, and layer 5 sessions -- over seeded corpora with planted anomalies and the negative controls beside them |
| **SIEM side** | `TA-morpheus-lineage`, an installable Splunk app (indexes, sourcetypes, KV Store binding lookups, and scheduled searches), validated by AppInspect, a live load into Splunk Enterprise 10.2, and a functional pass against seeded telemetry ([README](./examples/splunk_lineage_app/README.md)) |
| **First detections** | Six deterministic rules as saved searches in the app. Two at layer 5: a principal authenticated from two places faster than the journey can be made, and a run of multi-factor denials ended by an approval. The first excludes token refreshes and VPN egress ranges from the measurement *and* from becoming the location the next one is measured against, and one intrusion produces a pair of alerts rather than one. And four at layer 2: a MAC in two places at once, 802.1X authorization with no authentication in front of it, more MACs than permitted on a single-host port, and an address claimed by more than one MAC. The first fires on the interval between the two sightings rather than on their end reason, because an estate polls its switches in sequence and a cross-switch spoof is therefore seconds apart rather than simultaneous. The last two depend on a list the estate owns and ship with the hook for it. R-D-L2-001 fires on nothing until its port designation lookup is populated; R-D-L2-003 is the opposite, and fires on every redundancy gateway until its exclusion list is supplied. All four predicates asserted in Python over the planted corpus. Not yet run on a live search head |

Twenty-four stages and twenty-seven supporting modules, covered by 1,891 tests.

### What this fork is not

Being clear about the boundary is the point of writing it down:

- **The collectors are out of scope.** The SNMP, LLDP, DHCP, and 802.1X polling that produces layer 1
  and 2 telemetry is not Morpheus and is not here. What ships is everything downstream of it.
- **Layers 3, 4, 6 and 7 are designed, not built.** The telemetry classes, detection rules, and Splunk
  queries for those layers are specified in the guide and nothing runs for them.
- **The layer 5 scoring path runs end to end, and what occupies the model's slot is arithmetic.** Seven
  feature stages run under control 13's six checks against a week-long corpus, the two deterministic
  rules ship as saved searches, and the determinism controls the scoring path needs -- the envelope, the
  pinned manifest, the stable sharding -- were built ahead of it. `TC5ScoreStage` now gives `mean_abs_z`
  and `max_abs_z` a producer, resolving each entity to the model its manifest pins and refusing a row
  from another window. **It takes a scorer rather than training one**, which is what lets the path be
  tested where there is no Torch and no card, and what keeps training a scheduled job over history
  rather than something a stream does to the data it is scoring.
- **The scores in the composed pipeline are not model output and no detection claim attaches to them.**
  The corpus has no trained model in it, so the slot is filled by a reference scorer: each feature's
  distance from a frozen mean in units of a frozen deviation, with no learned parameters and no notion
  of normal beyond ten constants. Every scored row reports `model_fallback_used`. What the golden proves
  is that the path from features to scores is deterministic, batch-invariant and permutation-stable --
  a statement about plumbing, and a precondition for a real model rather than a substitute for one.
  R-B-L5-001 does not fire on this corpus and nothing was tuned to arrange that.
- **The autoencoder is not built here because it cannot be built here, and that is worth saying rather
  than implying.** `morpheus.models.dfencoder` is in the tree, but it is a Torch model on a CUDA device
  and the environment this work is developed and tested in has neither. Everything above was written and
  verified without them; a model path written here would be a model path that never ran. The controls
  landed first instead, which is the order the guide argues for anyway, since retrofitting determinism
  onto a running pipeline means re-tuning every threshold.
- **The model's scoring path is reproducible, and that is now measured rather than assumed.** On
  2026-09-07 at 12:01 UTC, on the same RTX 5000 Ada card with `torch==2.4.0+cu124`,
  `examples/layer5_model/run_model.py` trained a per-principal autoencoder on the layer 5 corpus for
  five principals and answered the only question a week of data can: **the double run was identical and
  the scores were invariant across batch sizes 1, 8 and 64.** That is control 3 and control 5, measured.
  Had either failed, R-P-L5-006 -- a rule about a score rising by fractions of a standard deviation --
  would have been measuring the model's own jitter, and every threshold tuned against it would have been
  tuned against noise.
- **What that run does not establish is that the model detects anything.** Five principals over one week
  is far too little data to train an autoencoder that detects anything, no claim is made that it does,
  and the artifact carries that caveat in a field of its own so it travels with the numbers rather than
  living in a document beside them. The pipeline still produces no `mean_abs_z`, so R-B-L5-001,
  R-B-L5-002, R-B-L5-005 and R-P-L5-006 continue to fire on nothing. What is now true is narrower and
  worth having: the path from features to scores gives the same answer twice.
- **The rule thresholds are placeholders** unless a rule says otherwise. They are starting points for
  tuning against an estate's own data, not calibrated values.
- **"Predictive" is a claim the guide qualifies rather than asserts.** Three of its four mechanisms are
  forward-looking in a defensible sense. The fourth, the premise that autoencoder reconstruction error
  rises during reconnaissance and staging, is a hypothesis this work does not establish, and a deployment
  should validate the lead time against its own incident history before promising prediction to
  anyone.
- **GPU execution mode has been measured on one machine and nowhere else.** On 2026-09-05 the 203
  `gpu_mode` variants were run for the first time, on an NVIDIA RTX 5000 Ada Generation Laptop GPU
  (compute capability 8.9, driver 596.58) under WSL2: **226 passed, 2 failed, 55 skipped**, and both
  failures are in upstream Morpheus files (`test_deserialize_stage_pipe`, `test_write_to_file_stage_pipe`)
  rather than in anything this fork adds. Every stage and utility added here passes in GPU mode. The suite
  was re-run on 2026-09-06 with the two parity repairs below in place -- **231 passed, 2 failed, 55
  skipped**, the same two upstream failures and nothing else.
- **Every `gpu_mode` variant this fork has passes on a GPU, and so does everything else it adds.** On
  2026-09-07 at 20:09 UTC, `ci/scripts/gpu_conformance.sh` ran on that same card over tiers that are
  total: the marked tier **417 collected, 417 passed**; the tier carrying no mode marker -- where the
  default execution mode on a machine with a card is the GPU -- **913 collected, 907 passed, 6
  skipped**. Nothing failed in either, both exited cleanly, and both counts reconcile exactly against
  what pytest collected. Every remaining skip is a field contract for a sourcetype nothing produces yet;
  there were seven until `binding:l1` gained a producer, and that test now runs instead of skipping.
  These runs also carry `torch==2.4.0+cu124` alongside the RAPIDS stack, so they say something the
  earlier ones could not: adding Torch to this environment does not disturb cuDF, and the two coexist in
  one process. That covers all twenty-two stages, all three composed pipelines,
  control 13's six checks, the stage parameter liveness registry and the first-detection corpus, in GPU
  mode. The wider upstream tier **skipped itself**, because that checkout's `tests/tests_data` fixtures
  were unfetched Git LFS pointers, so nothing here is a claim about the upstream suite. One card, no CI.
  This verdict supersedes an earlier 353 collected and 353 passed, which was narrower than it read -- see
  the three bullets below, one per defect, each of which produced an artifact saying `passed` while
  measuring less than it claimed.
- **Two earlier verdicts were narrower than they read, and both defects were in the runner.** The first
  selected from a list that was not total: it omitted `test_community_id_stage.py` and
  `test_column_assign.py`, and later the five TC-5 stage files, so "227 of 227" was 227 of the variants
  the list happened to name. The second counted from streamed output with a pattern that stopped at the
  first space, so fifteen tests whose parametrized identifiers contain one -- a saved search called
  `R-D-L5-004 - Multi-factor fatigue`, an entity-key case that is three spaces -- ran, passed, and were
  not counted, while the artifact reported a pass and named the last of them as where the run died. The
  tiers are now kept total by a test that identifies this fork's files by their copyright header, and
  the counts are **reconciled against what pytest collected**, so a total that does not add up is a
  failed verdict rather than a quiet one. A better pattern was not the repair; the reconciliation is.
- **A third narrowing, and the same shape as the first two.** The marked tier named the whole
  `tests/morpheus/determinism` directory, which reads as complete. It is not: `-m gpu_mode` selects the
  29 tests in the five files that carry a mode marker, and the other six files -- the stage parameter
  liveness registry, the first-detection corpus, the representation invariance suite, the end-to-end MAC
  spoof, the Splunk validation package -- carry none, so **320 of that directory's 349 tests were
  deselected on every GPU run this repository has rendered a verdict from**, including the 353 above.
  The test that keeps the tiers total could not have caught it, because it exempted the directory from
  the marker check on the grounds that a directory has no markers to check, which excused the one entry that
  most needed checking. Directory entries are gone; every entry is a file,
  nothing is exempt, and that change alone grew the unmarked tier from 508 tests to 890. What the 353 covered it still
  covers, control 13 included; what it never covered now runs.
- **A fourth defect, in the counting rather than the selection, and it was two errors that nearly
  cancelled.** The first run over total tiers reported 378 of 379 in the artifact while pytest's own
  summary said 379 passed. Three of those tests spawn subprocesses, and a subprocess's pytest output
  lands in the parent's log, so two names appeared twice and were counted twice while the three tests
  they displaced went unattributed. An off-by-one is what it looked like; two independent errors of
  opposite sign is what it was. Adding up streamed lines had been the default for three revisions and
  was the wrong one: a run that finishes has already been counted, by pytest, in a line it prints for
  the purpose. That tally is now the authority, and the line-by-line reading is the fallback for a run
  that crashed before writing one -- which is the only case it was ever needed for. The tally is
  authoritative about what pytest ran, not about what it was asked to run, so it is still reconciled
  against the collected list. Before that, the same run had parsed **zero** of 379: pytest puts an
  outcome on the next line when the identifier does not fit the terminal, and every identifier in that
  tier carries a `[gpu_mode]` suffix. The artifact now names what went unaccounted rather than only
  counting it.
- **The two modes did not agree, and the per-stage runs could not have told us.** Every one of those 203
  variants passes, and the composed telemetry pipeline still produced `arp_count_in_window = 3.0` on a GPU
  where the CPU golden holds `3`. Nothing raised. cuDF's `to_pandas` cannot put a null inside an integer
  column, so it widens the column to float64 and writes NaN -- and every windowed count here is null on
  the rows belonging to other telemetry classes. The conversion is not where it happens: collecting the
  classes fills that column with gaps for every other class's rows, and a plain integer column cannot
  hold one, so the fill widens it. Integer columns are now carried in a type that admits a gap, in both
  modes, before anything is joined, which also cost the golden fourteen columns' worth of trailing `.0`
  and is the better rendering. The lineage pipeline agreed across modes throughout.
- **Fixing that exposed a second place with the same cause.** Seven columns stayed wrong --
  `auth_attempts`, `link_flaps`, `link_flaps_in_window` and the four interface counter deltas -- and the
  fill was not what widened them: they were float64 before the fill saw them. `WindowSealStage` buffers
  on the host, and a device integer column holding a gap cannot cross as an integer, so it came back to
  the device as a float. What identified it was a column that was fine: `arp_count_in_window` is written
  by the same helper on the same line shape, and differs only in having a value on every row. The
  confirming detail was the one telemetry class that skips sealing, whose own gap-bearing integer columns
  were the only ones never flagged.
- **Control 13 is now verified in both execution modes.** On 2026-09-06, on the same laptop GPU, both
  composed pipelines ran in GPU mode against the same corpus and matched the same golden. The rest of
  control 13 followed: check 1 is a property of the corpus builder and has no execution mode, and checks
  2 through 6 -- the double run, the cross-restart, the golden, the batch-split sweep, and the
  permutation check with its negative control -- now run against both pipelines in either mode.
  **Fifteen GPU variants where there had been two, all passing.** One card, and not CI.
- **The obvious repair was tried first and was worse.** Asking the conversion for types that can hold a
  gap fixes integer columns and breaks every other kind: object columns start yielding `pandas.NA` where
  they yielded `None`, and stage code testing `value is None` stops recognising a missing value. Measured
  on the same GPU, that turned three failures into nine, all of them null-handling tests across the ARP,
  auth, binding-resolver and lineage-stamp stages. It is recorded here because the reasoning for it was
  sound and the result was not.
- **A GPU run under WSL2 requires `NUMBA_CUDA_USE_NVIDIA_BINDING=1`.** Without it, Numba's default
  driver bindings read back an invalid CUDA context through the WSL driver shim: `cuCtxGetDevice` yields
  a garbage device number and the process crashes partway through the suite. Setting the variable
  switches Numba to NVIDIA's own bindings and the failures disappear. This is an environment defect
  rather than a code one, but it costs a day to rediscover.

## What you would have to collect

The collectors are out of scope for this fork, so the question a deployment asks first is what its own
data mesh or lake has to land before any of this runs. That is answered in full in
[Part 2 of the guide](./docs/source/developer_guide/guides/11_predictive_behavioral_analytics_osi.md#part-2-telemetry-class-requirements),
including the complete field list per class. What follows is the short version, and the constraints that
are easy to get wrong.

### The envelope comes first, and it is not optional

**Every record from every layer carries the same fourteen fields before it reaches Morpheus.** They are
what make ordering total, lineage traceable and features comparable across sources. The ones a
lake design usually gets wrong:

| Field | Why it matters |
| --- | --- |
| `collector_seq` | Strictly monotonic per `collector_id`. This is the tiebreaker that makes the total order total. Without it, two events at the same nanosecond order differently on every run. |
| `event_time` vs `observed_time` vs `ingest_time` | Three distinct times. `event_time` is when it happened, `observed_time` when the sensor saw it, `ingest_time` when Morpheus received it. |
| `clock_source`, `clock_offset_ms` | Records beyond a configured bound are quarantined rather than dropped. A device whose clock is wrong by years is a real thing, and one such record can expire every open binding at once. |
| `sampling_policy` | `full`, `1:N` or `adaptive:<params>`. Any rate-based feature is uninterpretable without it. |
| `entity_key`, `osi_layer` | The behavioral subject for that class and the layer it came from. The subject differs per layer -- see the table below. Both are stamped by `EnvelopeStampStage` at the tail of each class's segment; searches that group by layer and entity drop any row missing either. |
| `origin_hash`, `event_uid` | Deterministic identity, so a score can be traced back to the bytes that produced it. |

Two rules govern it, and both are the kind that only hurt later:

1. **`ingest_time` never appears in a detection rule or a feature.** It is when your pipeline happened to
   be running, not when anything happened.
2. **An unavailable field is explicitly null with a reason code, never defaulted to a plausible value.**
   Three months into an investigation a defaulted value is indistinguishable from an observed one.

**Fix these at the collector, not in the pipeline.** Patching the envelope downstream is how lineage
silently breaks.

### What each layer needs

Status is what this fork does with it today, not what the guide specifies.

| Class | Entity key | Where it comes from | Status here |
| --- | --- | --- | --- |
| **TC-0** Identity and asset context | user, asset | HR system, IdP groups, CMDB, asset inventory | Schema only. **Bitemporal** is a hard requirement: valid time *and* transaction time, or you cannot answer "was this user in Finance on March 3rd" |
| **TC-1** Physical | `site_id:device_id:port_id` | SNMP interface tables, optical transceiver diagnostics, LLDP/CDP neighbors, patch-panel inventory | Six stages ship, the newest closing port observations into the `binding:l1` intervals that complete the identifier ladder |
| **TC-2** Data link | `mac_address`, and `site_id:switch_id:port_id:vlan_id` | Switch MAC tables, 802.1X/RADIUS accounting, ARP tables, DHCP leases, wireless controller associations | Four stages ship |
| **TC-3** Network | `src_ip`, and the directed pair | NetFlow v9, IPFIX, sFlow, VPC and cloud flow logs, firewall session logs | Schema only |
| **TC-4** Transport | `flow_id`, plus `community_id` for cross-tool joins | The TC-3 sources with per-packet detail, Zeek `conn.log` | Schema only; `community_id` ships and is verified against the reference vectors |
| **TC-5** Session | `user_principal`, plus `session_id` | Identity providers, Kerberos KDC, RADIUS accounting, VPN concentrators, RDP and SSH session logs, MFA providers, Windows security events | Seven stages ship, the newest scoring rows against the model the manifest pins for their entity |
| **TC-6** Presentation | `ja4_client`, plus `certificate_fingerprint_sha256` | TLS inspection points, Zeek `ssl.log` and `x509.log`, load balancer logs, certificate transparency | Schema only |
| **TC-7** Application | Varies: `user_principal` for SaaS, `hostname` for DNS, `process_guid` for endpoint | HTTP proxies and WAFs, DNS resolvers, SaaS audit APIs, database audit logs, API gateways, EDR | Schema only |

### Four constraints that decide whether the features mean anything

- **Counters must arrive as deltas with an explicit interval, not raw values.** A raw counter cannot be
  told from a wrap or a reboot without the uptime beside it, and a feature computed from the difference
  of two raw readings across a device restart is a large negative number that looks like an event.
  `TC1NormalizeStage` does this correctly given the inputs; it cannot do it without them.
- **TC-0 must be bitemporal.** It is the smallest and most valuable dataset in the architecture, and a
  non-bitemporal CMDB can only answer questions about the present.
- **Ordering is the pipeline's job, but the inputs have to make it possible.** Sort before any cumulative
  feature; `IncrementColumn`-style features are order-dependent and an unsorted batch silently produces
  different features and different scores. This is the most easily missed defect in the whole design,
  because the output stays plausible.
- **Retention has to match across the lake and the SIEM.** The binding lookups here assume 400 days, and
  the index retention and the expiry job that prunes against it must agree; a test compares the two, because
  they fail silently when they drift apart.

### Clock drift, which is three problems wearing one name

Every join in this architecture is a join on time, so time is a shared coordinate across sources that do
not share a clock. Three distinct things go wrong and only the first is what people mean by drift:

- **Drift.** Two clocks disagree. An NTP-disciplined host is typically within a few milliseconds; an
  undisciplined one is anywhere from seconds to hours out, and a device with an unset real-time clock is
  out by years.
- **Granularity.** Sources have different native precision. Packet capture is sub-microsecond, identity
  provider logs are milliseconds, NetFlow exports are commonly whole seconds, and an SNMP poll is a
  sample every 30 to 300 seconds. Joining a coarse layer to a fine one has a precision floor that no
  amount of clock discipline removes, and a rule written as though both sides were precise is measuring
  the polling interval.
- **Export lag.** A NetFlow active timeout can hold a flow open for a minute before exporting it. The
  `event_time` is right; the arrival is late. That is a watermark problem, not a clock problem, and it is
  what `WindowSealStage`'s lateness horizon and separate late-arrival stream exist for.

**What is most sensitive here, in order.** The single most fragile predicate this repository ships is
R-C-002's `gap > 0`: the *sign* of a difference between a layer 6 event and a layer 3 event from two
independent sensors. Sub-millisecond disagreement inverts it, and an inverted sequence is not a missed
detection that looks missed -- the rule simply does not fire. Below that:

| Feature | How drift moves it |
| --- | --- |
| Impossible travel (R-D-L5-003) | Speed is distance over elapsed time. The one-second floor in [`geo_velocity`](./python/morpheus/morpheus/utils/geo_velocity.py) bounds the arithmetic, but drift that makes the second authentication appear *before* the first has the row treated as out of order and not measured at all -- a real journey that silently carries no score |
| MAC in two places (R-D-L2-004) | The gap is measured between sightings on two switches polled in sequence. Each switch's own offset adds directly to that interval and can carry it across the threshold in either direction |
| Session duration | The start and the stop routinely come from different systems. Their disagreement is the duration's error bar |
| 802.1X exchange timing | The switch stamps one end and RADIUS the other |

**What the pipeline does about it, and what it does not.** [`event_clock`](./python/morpheus/morpheus/utils/event_clock.py) protects
stateful stages from a single catastrophic timestamp -- a clock wrong by years driving the expiry horizon
past every open binding at once -- by refusing a time further ahead of the stream's own progress than
`max_clock_skew_seconds` allows. It is deliberately **not** a drift correction, and its week-long default
will not notice a device an hour out. Nothing here silently corrects a timestamp, because a corrected
value is a defaulted value and three months later it cannot be told from an observed one.

**The burden therefore sits on collection.** Four things worth requiring:

1. **`clock_source` and `clock_offset_ms` on every record**, not per host per day. They are what let an
   investigation say whether a five-second gap was real. Records beyond a configured bound are
   quarantined for review, never dropped.
2. **PTP rather than NTP wherever ordering across sources decides an outcome** -- the same capture point
   feeding layers 3, 4 and 6 is the case that matters.
3. **Prefer a hard join to a time-window join.** The guide distinguishes the two: a hard join is exact
   equality on a shared identifier (`flow_id` between layers 4 and 6, `session_id` between 5 and 7), and
   it is immune to all three problems above. Reach for a time window only when no identifier is shared.
4. **Set `max_clock_skew_seconds` to something your estate justifies.** The default is a week because it
   is aimed at the catastrophic case; an estate that knows its clocks are disciplined to milliseconds
   should say so, and one that cannot make that claim should know it cannot.

A rule whose outcome turns on a sub-second difference between two independently clocked sources is not a
detection, it is a coin weighted by your NTP configuration. Write it as a hard join, widen it past the
disagreement you can actually bound, or do not ship it.

### What comes back out

The pipeline emits seven sourcetypes a SIEM consumes, and declares seven more that nothing here produces
yet -- see [`siem_sourcetypes`](./python/morpheus/morpheus/utils/siem_sourcetypes.py), which says for each
unproduced one exactly what would have to be built. `TA-morpheus-lineage` is an installable Splunk app
for the consuming side ([README](./examples/splunk_lineage_app/README.md)).

**Start with layer 5.** The guide's sequencing argues for it and so does the collection cost: identity
provider logs are already centralized in most estates, the entity is unambiguous, and layers 5 and 7
carry most of the standalone detection value. Layers 1 and 2 have the highest collection effort and the
lowest standalone value: they earn their place by completing the ladder, not on their own.

## Documentation
### Using Morpheus
* [Getting Started with Morpheus](./docs/source/getting_started.md) - Using pre-built Docker containers, building Docker containers from source, and fetching models and datasets
* [Morpheus CLI Overview](./docs/source/basics/overview.rst) - Brief overview of the `morpheus` command line interface
* [Building a Pipeline](./docs/source/basics/building_a_pipeline.md) - Introduction to building a pipeline using the command line interface
* [Morpheus Examples](./docs/source/examples.md) - Example pipelines using both the Python API and command line interface
* [Pre-built Models and Datasets](./models/README.md) - Pretrained models with corresponding training, validation scripts, and datasets
* [Developer Guides](./docs/source/developer_guide/guides.md) - Covers extending Morpheus with custom stages


### The behavioral analytics work in this fork
* [Predictive Behavioral Analytics Across OSI Layers 1-7](./docs/source/developer_guide/guides/11_predictive_behavioral_analytics_osi.md) - The design guide this fork implements: codebase analysis, per-layer telemetry requirements, detection rules, Splunk lineage chaining, and the determinism controls
* [List of available Morpheus stages](./docs/source/stages/morpheus_stages.md) - The `Lineage` and `Telemetry` sections cover the stages added here, along with the SIEM wire stage under `Output`
* [Splunk Lineage App](./examples/splunk_lineage_app/README.md) - The SIEM half, as an installable Splunk app

### Modifying Morpheus
* [Contributing to Morpheus](./docs/source/developer_guide/contributing.md) - Covers building from source, making changes and contributing to Morpheus

Full documentation for the latest official release is available at [https://docs.nvidia.com/morpheus/](https://docs.nvidia.com/morpheus/).
