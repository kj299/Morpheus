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

# How much clock skew each detection tolerates

Every join in this design is a join on time across sources that do not share a clock. The guide argued
qualitatively that some features are far more sensitive than others and said plainly that none of it had been
measured. This measures it.

```bash
./examples/clock_skew/run_experiment.py /tmp/clock_skew.json
```

It runs on any machine. No card, no Torch, about six minutes.

## What it does

Spread the collectors' clocks across a window of a given width, re-run the composed pipelines over the same
seeded corpora, and compare. Two choices make the answer mean something:

**The magnitude is the width of the spread, not a shift.** Moving every record together barely disturbs a
pipeline keyed on event time; what breaks a join is two clocks disagreeing. At magnitude `M` the offsets are
spread evenly across `[-M/2, +M/2]` by sorted name, so the worst pair disagrees by exactly `M` and no source is
privileged. `M` then reads as the sentence an estate needs: this rule wants the fleet synchronized to better
than `M`.

**A decision is keyed on what it accuses, never on when.** A detection identified by timestamp would differ
under every non-zero offset, which measures the injection rather than the damage. Each rule reduces to the set
of entities it flags, so a change means it accused something different, which is the only change an analyst
would ever see.

Row identity is what makes a column-by-column answer possible: `event_uid` derives from the collector sequence
rather than the timestamp, so a record keeps its identifier however its clock is set and the two runs align row
for row.

## The result

All forty-one shipped detections are swept, each over the pipeline that feeds it, through
[`rules.py`](./rules.py), which reduces every rule to what it accuses and reads every threshold, join tolerance
and window from `savedsearches.conf`. The ladder is one millisecond, ten, a hundred, one second, ten, thirty and
sixty; the chained rules continue to two minutes, four, ten, thirty, an hour, two and three, because their join
tolerance is two minutes and a ladder that stopped at one could not reach it.

It was run again after step 6 (#57) added a risk record and suppression to every detection, rewrote Chain
assembly and gave R-P-L3-005 its second `streamstats` pass, on 2026-10-09, beside a run of the tree before those
changes: the two artifacts are identical apart from their timestamps. None of it moved a decision, and R-P-L3-005's
model in `rules.py` already took two passes; it is the search that now agrees with it.

It was run once more the same night after step 8 (#59) gave each layer 5 principal a learned model of their own
and the session corpus the fortnight those models are trained on. Every rule's answer is the same as before but
one: R-P-L5-006, which changed at a millisecond under the reference arithmetic, now changes at no spread swept,
on the hour marks or off them. Why is below.

It was run again on 2026-10-10 after step 9 (#60) gave layer 3 a fortnight of history, measured each host's
counts against it, added three layer 3 rules, required R-C-001's last hop to land on a server and keyed the
endpoint rules on the normalized host name. The three new rules join the network corpus's other five, which
arrive through one exporter, so they are reported as not measured. The endpoint corpus's renamed host reports
under two names, and the sweep gives each reported name a clock of its own, so it spreads eight clocks where there
are seven hosts; R-B-L7-004 is unchanged to a minute all the same. Every chain breaks at the second it broke at
before, by accusing a control, and loses its attacker at the second it did before.

What a sweep can say depends on the clocks in the corpus, and the forty-one fall into three groups that must
not be added together:

| Group | Rules | What the sweep measured |
| --- | --- | --- |
| Inputs on clocks that disagree | 12 | A tolerance, or the spread at which the rule changes |
| Inputs on one clock, in a corpus with several | 11 | Boundary sensitivity under a uniform shift, not a tolerance |
| Corpus with one clock | 18 | Nothing; reported as not measured |

**Measured against clocks that disagree** -- the estate's five collectors and three switches for layers 1 and 2,
the campaign's nine collectors for the chains, and the endpoint corpus's seven hosts, each of whose agents stamps
its own process starts:

| Rule | Breaking spread |
| --- | --- |
| R-D-L1-001, R-P-L1-004, R-D-L2-001, R-B-L2-002, R-D-L2-003, R-D-L2-005 | unchanged to 60s on collectors and on switches |
| **R-D-L2-004, MAC in two places** | unchanged to 60s on collectors; **changes at 60s on switches** |
| R-B-L7-004, process ancestry novelty | unchanged to 60s on host clocks |
| **R-C-001, lateral movement chain** | **960s**, gaining a control |
| **R-C-002, TLS anomaly precedes beaconing** | **1120s**, gaining a control |
| **R-C-004, staged exfiltration** | **3200s**, gaining a control |
| **R-C-005, credential replay** | **7201s**, gaining a control |

**On one clock in a corpus with several.** The session corpus has three collectors, but every authentication
comes through the identity provider, so for the eight sign-in rules -- R-D-L5-003, 004, 007, 008 and 009,
R-B-L5-001 and 002, and R-P-L5-006 -- the sweep moves all their inputs together; R-B-L5-005's session starts and
ends each come through one collector too. The application corpus has two, one per class, and each of its two rules
reads one class. For these eleven the sweep is a uniform shift, which measures how close the events sit to a
window edge and nothing about disagreement. All eleven are unchanged to a minute; R-P-L5-006 used to change at a
millisecond, for the reason below.

**On a corpus with one clock.** The network, transport, presentation and SaaS corpora each arrive through a single
collector, so a collector sweep gives that one clock no offset and perturbs nothing. Their eighteen rules -- eight at
layer 3, three at layer 4, five at layer 6, R-B-L7-002 and R-P-L7-006 -- are reported as `"one clock"` in the
artifact rather than as unchanged, because the absence of a measurement is not a tolerance. Measuring them needs
the clocks a deployment really has, which the corpora do not carry: an exporter per flow at layers 3 and 4, an
inspection point per egress at layer 6. For SaaS the provider's audit log really is one clock, and the second
clock that matters is the context store's record time, which the sweep does not reach.

### The spoof rule fails against switches and not against collectors

The guide said the MAC-in-two-places interval absorbs each switch's offset directly. It does, and that is
precisely why a collector's offset does nothing: both sightings of a displaced MAC arrive through one MAC table
feed, so that feed's error moves them together and cancels out of the interval entirely. Only the switches'
own clocks pull the pair apart.

The cross-switch spoof is two seconds wide and the rule fires below sixty. Spread the switches across a minute
and the pair reads as sixty-two seconds apart, which is an ordinary move rather than a spoof, and the detection
is lost. The simultaneous spoof beside it is within one switch and keeps firing, so this is a missed detection
rather than a broken run -- the worst shape for an operator, because nothing looks wrong.

**The number an estate can act on: this rule needs its switches synchronized to better than the gap it is
tuned to.** At a sixty-second threshold that is sixty seconds of headroom, consumed by two seconds of real
sweep and fifty-eight of slack. Tighten the threshold to the sweep and the tolerance tightens with it.

### The drift rule is sensitive to boundaries, not to magnitude

Under the reference arithmetic, a millisecond of spread made the drift trajectory stop flagging three of the six
principal-days it flagged. That read like a rule needing millisecond synchronization, and it was not.

A hundred and twenty-seven of the layer 5 corpus's three hundred and eighty-five authentications sit exactly on
an hour mark, because the corpus builds its times from whole hours. An event on a boundary changes hour under an
offset of one nanosecond, and with it the hour's surprise and the row's score. Move the same events into the
middle of their windows -- a uniform shift, which is not a skew at all, since no two clocks disagree any more
than before -- and a full minute of spread changes nothing the rule accuses.

With the learned models a millisecond still moves those inputs: every boundary-aligned sign-in changes hour, and
forty-five principal-days' rise in standard deviations moves with it. What the rule accuses does not move. The
climbs it reports are steady enough under the models that a changed hour on some of their rows leaves them
climbing, where under frozen arithmetic it broke three runs.

**The exposure is therefore not the size of the clock error. It is the fraction of events sitting near a
window edge, and whether the scorer turns a changed input into a changed decision.** An estate cannot read a
tolerance off this rule. What it can read is that window-boundary proximity is the variable to think about, that
a corpus built on round numbers will overstate the fragility of anything measured against it, and that the answer
belongs to the scorer as much as to the rule: `test_clock_skew_experiment.py` asserts both the moved inputs and
the unmoved accusations, so a scorer that makes the rule fragile again fails there.

### The ladder's rungs do not fail together

Chains keep their span at every width swept on both axes: fifteen chains still hold all three layers, and all
fifteen desk sign-ins still resolve to the port they were made at. A principal reaches a port through an 802.1X
session lasting most of the hour, and a minute never takes it outside.

The rung below is not so comfortable. An ARP observation reaches a port through a MAC binding that lasts one
poll cadence, and a minute is a large fraction of that. At sixty seconds, of the 1040 ARP observations rooted
on a port, fifteen fall back to their own address and six acquire a port they did not have -- a net nine, and
twenty-one rows whose attribution changed.

**Attribution and reach are different properties and do not fail together.** A chain can span three layers
while an observation inside it has quietly been re-attributed. Counting spans alone would have reported
everything fine.

### The chains hold far past their tolerance, and fail by accusing a control

No chain changes at any spread up to ten minutes, five times its two-minute join tolerance. That is a property of
the corpus rather than of the tolerance: the closest any chain's steps come to the edge of what it allows is four
minutes, so a spread wide enough to close that gap is wide enough to say nothing about the two minutes itself.
A corpus that tests the tolerance has to put two steps within a minute or so of each other on different clocks.

Where the chains do break, each breaks the same way, and it is the worse of the two ways: a second before its edge
nothing has changed, and at it the chain adds a control the corpus planted for it to stay quiet on, while still
accusing its attacker. Each edge is arithmetic about which clock stamps which step, because the collectors are
spread by sorted name, so each sits at a fixed fraction of the spread `M`:

- **R-C-001 at 960s.** The login comes through `dc-01` at `-M/2` and the process through `edr-01` at `-M/4`, so
  they move apart by `M/4`. A control whose novel process ran 360 seconds before its login is admitted once that
  reaches the 120-second tolerance: `-360 + M/4 >= -120` at `M = 960`.
- **R-C-002 at 1120s.** The beacon's clock is the middle one and the fingerprint's sits at `+3M/8`, so they close
  by `3M/8`. A control whose beacon came 4020 seconds after its new fingerprint falls inside the hour at
  `4020 - 3M/8 <= 3600`, `M = 1120`.
- **R-C-004 at 3200s.** A control's session ends on `vpn-01` at `+M/2` 1200 seconds before its breach on
  `pcap-01` at `+M/8`; the end overtakes the breach at `3M/8 >= 1200`. What decided it was the session interval,
  not a join tolerance.
- **R-C-005 at 7201s.** It has no join tolerance; both sign-ins come through one identity provider. A control
  signed in an hour after her lease ended, and the lease table is not a collector the sweep moves, so her sign-in
  falls back inside the lease once its clock moves by more than an hour, at `M > 7200`.

The attackers are lost later, each measured to the second: R-C-001's at 1081 seconds, R-C-002's at 2561,
R-C-005's at 9601, and R-C-004's not until 12,961, past the three-hour ladder.

## What this does not do

It does not correct anything. `clock_source` and `clock_offset_ms` are in the universal envelope and no stage
produces or consumes either; they are schema, not a control. This measures the damage a deployment takes today,
not the damage after a correction that has not been built.

It also measures these corpora. The magnitudes at which things break are properties of an hour of one estate
and three weeks of seven principals, with these thresholds. The mechanisms generalize; the numbers are a worked
example, and the reason each number is what it is, is written down beside it so an estate can redo the
arithmetic against its own sweep times and thresholds.

## The artifact

```json
{
  "at": "…",
  "spread_widths": ["1ms", "10ms", "100ms", "1s", "10s", "30s", "60s"],
  "chain_spread_widths": ["1ms", "…", "60s", "120s", "240s", "600s", "1800s", "3600s", "7200s", "10800s"],
  "join_tolerance_ns": {"R-C-001": 120000000000, "…": "…"},
  "breaking_point": {
    "R-D-L2-004": {"estate_collectors": null, "estate_switches": {"spread": "60s"}},
    "R-P-L5-006": {"session": {"spread": "1ms"}, "session_off_boundary": null},
    "R-B-L3-001": {"network": "one clock"},
    "R-C-001": {"campaign": {"spread": "1800s"}}
  },
  "chain_breaking_point_to_the_second": {
    "R-C-001": {"held_at": "959s", "changed_at": "960s", "no_longer_flagged": [], "newly_flagged": ["…"]}
  },
  "sweeps": {"estate_collectors": {"clocks": ["…"], "runs": ["…"]}, "…": "…"}
}
```

`null` is a rule that held across the whole ladder; `"one clock"` is a sweep that could not move anything.
Each run records the offsets applied, whether the output was identical, every column that moved with a row
count and a maximum delta, what each rule stopped and started accusing, and how far the ladder reached.

The findings above are asserted in
[`tests/morpheus/determinism/test_clock_skew_experiment.py`](../../tests/morpheus/determinism/test_clock_skew_experiment.py)
against the pipelines rather than quoted from a saved artifact, so a change that invalidates one of them fails
a test rather than leaving a document describing a system that stopped behaving that way.
