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

All sixty-two shipped detections are swept, each over the pipeline that feeds it, through
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

It was run again later on 2026-10-10 after step 10 (#61) gave the port, the link, the VLAN and the 802.1X exchange
behaviour of their own and added eight rules over them: five at layer 1 and three at layer 2. Their inputs come
through the estate's collectors and switches like the other layer 1 and 2 rules', so they join the measured group,
and each decides on the estate with no skew exactly what the harness plants for it: one port each for the tap, the
flapping link and the error climb, the re-patch and the insertion, the surge and the silence, the two VLANs, and
one bench exchange each for the two timing rules. All eight are unchanged to a minute on collector clocks and on
switch clocks, and nothing else moved.

It was run again that night after step 11 (#62) put the envelope on every record, quarantined the estate's DMZ
poller and named the exporter on every flow. It gained two things. The network corpus is swept a second time on
its routers' clocks, which measures its eight rules for the first time. And every sweep over collector clocks runs
twice: uncorrected, as before, and with quarantine on, which is the corrected curve below. The uncorrected numbers
are unchanged: every rule that held still holds, and every chain breaks at the second it broke at before. The DMZ
poller is a sixth clock in the estate's spread, which moves the ladder's lower rung by a few observations and
nothing else.

It was run again on 2026-10-15 after step 12 (#63) resolved every layer 3, 4, 6 and 7 row to its machine through
the DHCP server's leases and an EDR agent's inventory reports. Those are two more collectors in each corpus that
resolves, so the network, transport and presentation corpora have three clocks where they had one, the application
corpus four, and the campaign eleven where it had nine. Every rule that held still holds: the sweep moves each lease
and report against the rows it resolves, which changes the binding identifier a row cites, and no row's machine
changed at any spread swept. The transport and presentation rules are measured now, as uniform shifts, and
R-B-L6-001, whose managed-host gate reads the machine, against clocks that disagree. The chains moved, and one moved
the other way: eleven clocks put each collector at a different fraction of the spread, so every edge is new, and
R-C-005 now first breaks by losing its attacker rather than by accusing a control.

It was run again on 2026-10-16 after step 13 (#64) made a service and an application entities, measured every DNS
and HTTP client against its own busiest day, keyed TLS destinations on the server name and carried the TCP stack on
each SYN packet. The application corpus gains a fifth clock, the load balancer in front of its services, and its
week of per-client history. Every rule that held still holds, and every chain breaks at the second it broke at
before. The three new rules are unchanged to a minute: R-B-L7-007 and R-B-L7-008 each read one collector, and
R-B-L4-004 joins the transport corpus's stacks to the operating systems the application corpus's requests claim,
on the machine both resolve to, so it reads the DHCP server's and the inventory feed's disagreement as R-B-L6-001
does. The claims come from the application corpus as recorded, and the transport sweep moves the stacks against
them.

It was run again on 2026-10-17 after step 15 (#65) made the switch and the router entities and gave R-C-003 a
chain to find. The device polls come from the poller that polls the ports, and the campaign's headquarters port
polls from the poller that walks the same switch's MAC table, so no corpus gained a clock: the estate keeps its six
and the campaign its eleven, and every edge measured before is the edge it was. R-D-L1-008 and R-P-L1-009 are
unchanged to a minute on collector clocks; the switch-clock sweep moves the port and MAC records by the switch that
made them and leaves the devices' own polls on the poller's clock, so it says nothing about them. R-C-003 is
unchanged at every spread swept, to three hours. Its first two steps, the port's link event and the vendor new to it,
come through one poller and cannot be pulled apart by a collector spread; the poller sits at the middle of the
eleven clocks; and the flow that completes the chain lands thirteen minutes after the first step, inside a four-hour
window and a lease whose clock moves the other way by less than the lease has left to run.

It was run again on 2026-10-22 after step 8's host segment (#59) gained its sourcetype and a rule, R-P-L3-009. The
host score corpus joins the sweep as a tenth pipeline, with five collector clocks: the flow exporter, the inspection
point and the EDR, each the only collector of its class, and the DHCP server and inventory feed its hosts resolve
through. The host frame is the one row in which three collectors' records meet, so the rule is measured against
clocks that disagree rather than under a shift. It reads a day's mean of hourly frames, and a minute of spread
moves at most an hour's edge of records into the neighbouring frame: the three host-days it accuses are the same
at every spread swept, and nothing else moved. With quarantine on, ten seconds puts every feed beyond the bound and
holds every record, so the segment has nothing to frame and the rule loses its three host-days and accuses nothing
new; the pipeline now returns the quarantine alone where it raised on an empty concatenation.

It was run again on 2026-10-23 after the daily drift reached the TLS and DNS columns and two rules read it, R-P-L6-006
and R-P-L7-009. Each reads rows that come through one collector, the handshakes the inspection point sends and the
queries the resolver sends, so both join the uniform-shift group, twenty-two now. The trajectory is a day's largest count under a daily
seal, and a minute of spread moves at most an hour's edge of rows across a day boundary without changing which day
holds the largest: the creeping host's three days and the slow tunnel's four are the same at every spread swept, and
nothing else moved. The sweeps move only the lease and inventory binding identifiers the hosts resolve through, as
they did before.

It was run again on 2026-10-28, for the application and endpoint corpora, after plan section 3.3 gave the HTTP and
endpoint classes a daily trajectory each and a rule to read it, R-P-L7-010 and R-P-L7-011. R-P-L7-010 reads the
requests the proxy sends, so it joins the uniform-shift group, twenty-three now, and the slow enumerator's four days are
the same at every spread swept. R-P-L7-011 reads process starts each host's agent stamps on its own clock, so it is
measured beside R-B-L7-004 against clocks that disagree, thirty-six rules now: the creeping build server's three days
are the same to a minute of spread across the seven hosts. With quarantine on, ten seconds holds every application
record, and R-P-L7-010 loses its four days with the other application rules, as R-P-L7-009 does.

What a sweep can say depends on the clocks in the corpus, and the sixty-two fall into three groups that must
not be added together:

| Group | Rules | What the sweep measured |
| --- | --- | --- |
| Inputs on clocks that disagree | 36 | A tolerance, or the spread at which the rule changes |
| Inputs on one clock, in a corpus with several | 24 | Boundary sensitivity under a uniform shift, not a tolerance |
| Corpus with one clock | 2 | Nothing; reported as not measured |

**Measured against clocks that disagree** -- the estate's six collectors and three switches for layers 1 and 2,
the campaign's eleven collectors for the chains, the network corpus's three routers, each of which stamps its own
flows, the endpoint corpus's seven hosts, each of whose agents stamps its own process starts, the presentation
corpus's inspection point, DHCP server and inventory feed, whose disagreement R-B-L6-001 reads through the machine
it gates on, as R-B-L4-004 reads the transport corpus's through the machine it joins on, and the host score corpus's
flow exporter, inspection point and EDR, whose records meet in one row of the host frame R-P-L3-009 reads:

| Rule | Breaking spread |
| --- | --- |
| R-D-L1-001, 002, 003, 006 and 010, R-B-L1-005 and 007, R-P-L1-004, R-D-L2-001, 003 and 005, R-B-L2-002, 006, 007 and 008 | unchanged to 60s on collectors and on switches |
| R-D-L1-008, configuration change outside the window, and R-P-L1-009, reboot-rate trajectory | unchanged to 60s on collectors; the devices' polls are on no switch clock |
| **R-D-L2-004, MAC in two places** | unchanged to 60s on collectors; **changes at 60s on switches** |
| R-B-L7-004, process ancestry novelty, and R-P-L7-011, process breadth trajectory | unchanged to 60s on host clocks |
| R-P-L3-009, host drift trajectory | unchanged to 60s on the five collector clocks that feed and resolve the host frame |
| R-B-L3-001, 002, 004, 006 and 007, R-D-L3-003 and 008 | unchanged to 60s on router clocks |
| R-B-L6-001, new TLS client fingerprint, and R-B-L4-004, stack fingerprint mismatch | unchanged to 60s on collector clocks |
| **R-P-L3-005, fan-out trajectory** | **changes at 30s on router clocks**, losing one of the scanner's twenty rises |
| **R-C-001, lateral movement chain** | **800s**, gaining a control |
| **R-C-002, TLS anomaly precedes beaconing** | **1400s**, gaining a control |
| **R-C-004, staged exfiltration** | **4000s**, gaining a control |
| **R-C-005, credential replay** | **7201s**, losing its attacker |
| R-C-003, physical compromise chain | unchanged to 10800s, the widest swept |

**On one clock in a corpus with several.** The session corpus has three collectors, but every authentication
comes through the identity provider, so for the eight sign-in rules -- R-D-L5-003, 004, 007, 008 and 009,
R-B-L5-001 and 002, and R-P-L5-006 -- the sweep moves all their inputs together; R-B-L5-005's session starts and
ends each come through one collector too. The application corpus has five, one per class -- the resolver, the
proxy and the load balancer -- and the DHCP server and inventory feed its hosts resolve through, and each of its
six rules reads one class, the subdomain and path count trajectories a daily seal of what the resolver and the proxy
sent. The transport and
presentation corpora have three each since step 12, but every packet comes through the packet sensor and every
handshake through the inspection point, so three of their four layer 4 rules and five of their six layer 6 rules read
one clock, the client stack trajectory among them: it is keyed on the address the handshake carries, and only the
watchlist entry it writes names the machine. For these twenty-four the sweep is a uniform shift, which measures how
close the events sit to a window edge and nothing about disagreement. All twenty-four are unchanged to a minute, R-B-L7-003 among them since plan section 4.1; R-P-L5-006 used to change at a
millisecond, for the reason below.

**On a corpus with one clock.** The SaaS corpus arrives through the provider's audit log alone, so a collector
sweep gives that one clock no offset and perturbs nothing. Its two rules, R-B-L7-002 and R-P-L7-006, are reported
as `"one clock"` in the artifact rather than as unchanged, because the absence of a measurement is not a
tolerance. The audit log really is one clock, and the second clock that matters is the context store's record
time, which the sweep does not reach. Layers 4 and 6 were in this group until step 12 gave their corpora a lease
and an inventory report per host; what still needs a corpus that carries it is the clock a deployment really has
at each, an exporter per packet capture and an inspection point per egress. Layer 3 left it when step 11 named the
exporter on every flow, and its router sweep is the measurement.

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
poll cadence, and a minute is a large fraction of that. At sixty seconds, of the 1045 ARP observations rooted
on a port, ten fall back to their own address and five acquire a port they did not have -- a net five, and
fifteen rows whose attribution changed. It was fifteen and six before step 11 added the DMZ poller: a sixth clock in
the spread places the other five a little differently at the same width.

**Attribution and reach are different properties and do not fail together.** A chain can span three layers
while an observation inside it has quietly been re-attributed. Counting spans alone would have reported
everything fine.

### The chains hold far past their tolerance, and three fail by accusing a control

No chain changes at any spread up to ten minutes, five times its two-minute join tolerance. That is a property of
the corpus rather than of the tolerance: the closest any chain's steps come to the edge of what it allows is four
minutes, so a spread wide enough to close that gap is wide enough to say nothing about the two minutes itself.
A corpus that tests the tolerance has to put two steps within a minute or so of each other on different clocks.

Where three of the chains break, each breaks the same way, and it is the worse of the two ways: a second before
its edge nothing has changed, and at it the chain adds a control the corpus planted for it to stay quiet on, while
still accusing its attacker. Each edge is arithmetic about which clock stamps which step, because the eleven
collectors are spread by sorted name, so the one at index `i` sits at `-M/2 + i*M/10` of the spread `M`:

- **R-C-001 at 800s.** The login comes through `dc-01` at `-M/2` and the process through `edr-01` at `-M/5`, so
  they move apart by `3M/10`. A control whose novel process ran 360 seconds before its login is admitted once that
  reaches the 120-second tolerance: `-360 + 3M/10 >= -120` at `M = 800`.
- **R-C-002 at 1400s.** The beacon comes through `netflow-01` at `+M/10` and the fingerprint through
  `tls-inspect-01` at `+2M/5`, so they close by `3M/10`. A control whose beacon came 4020 seconds after its new
  fingerprint falls inside the hour at `4020 - 3M/10 <= 3600`, `M = 1400`.
- **R-C-004 at 4000s.** A control's session ends on `vpn-01` at `+M/2` 1200 seconds before its breach on
  `pcap-01` at `+M/5`; the end overtakes the breach at `3M/10 >= 1200`. What decided it was the session interval,
  not a join tolerance.

R-C-005 breaks the other way, at 7201 seconds, and the change is step 12's. It has no join tolerance; both
sign-ins come through one identity provider, and what the sweep moves is each sign-in against the bindings that
place it. Its control signed in an hour after the lease ended, and while the leases were a table no sweep moved,
that sign-in fell back inside the lease once its clock moved by more than an hour. The leases now come through the
DHCP server at `-2M/5`, so the sign-in closes on its lease by only `M/10` and cannot be admitted below
`M = 36000`. What breaks first instead is the attacker: the first sign-in, at `-M/2`, came an hour after the HQ MAC
table, the middle clock at `0`, first saw that desk, so past `M = 7200` it precedes the sighting, resolves to no
port, and the replay has one site rather than two. At its edge the rule stops accusing its attacker and accuses
nothing new.

The other three attackers are lost later, each measured to the second: R-C-001's at 901 seconds, R-C-002's at
3201, and R-C-004's not until 16,201, past the three-hour ladder.

### Quarantine trades a wrong answer for no answer

Every sweep over collector clocks runs a second time with quarantine on: each skewed collector reports the offset
it was given in `clock_offset_ms`, as a disciplined collector reports its last measured offset, and
`EnvelopeValidateStage` holds every record more than its default second out. The two curves, for the corpora with
more than one collector:

| Corpus | Uncorrected | Quarantine on |
| --- | --- | --- |
| Estate, layers 1 and 2 | No rule changes to 60s | No change to 1s; at 10s four of the six collectors are held whole -- 2345 records, all but the MAC table's and the RADIUS server's -- and the ten rules reading them lose all thirteen accusations; at 30s all six are held and all seventeen rules lose all twenty-eight |
| Session, layer 5 | No rule changes to 60s, on the hour marks or off | No change to 1s; at 10s the identity provider and the RADIUS accounting feed are held and all nine rules lose all twenty-two accusations |
| Network, layer 3 | No rule changes to 60s | No change to 1s; at 10s the flow collector and the DHCP server are held -- 1335 records, all but the inventory feed's at the middle of the spread -- and all eight rules lose all twenty-seven accusations |
| Transport, layer 4 | No rule changes to 60s | No change to 1s; at 10s the packet sensor and the DHCP server are held, 483 records, and all four rules lose all five accusations |
| Presentation, layer 6 | No rule changes to 60s | No change to 1s; at 10s the inspection point and the DHCP server are held, 525 records, and all six rules lose all twelve accusations, the client stack trajectory's three days among them |
| Application | No rule changes to 60s | No change to 1s; at 10s every collector but the load balancer, at the middle of the spread, is held -- 5067 records -- and the four rules reading the resolver and the proxy lose all eight accusations, the subdomain count trajectory's four days among them, while R-B-L7-007 keeps its one |
| Campaign, the chains | No change below 800s; past it three chains **gain** a control and R-C-005 loses its attacker | No change to 1s; from 10s every collector but the HQ MAC table is held and all five chains lose their attackers |

Two things are worth reading off it. Inside the bound, quarantine costs nothing: at a second's spread every clock
is within half a second of true and nothing but the DMZ poller is held, in any corpus. And beyond it, quarantine
never makes a rule accuse something it should not -- across every run, no rule gained an accusation -- but it takes
away everything the held collectors carried, at spreads where the uncorrected pipeline still decided correctly. A
one-second bound is far tighter than any of these rules needs: the single-layer rules held to a minute uncorrected,
and the chains to thirteen minutes. The bound is a statement about which wrong answer an estate prefers, and the
measurement says it can be set per class from the tolerance above, not globally from the clock source.

The exporter sweep has no quarantine curve. A router stamps its own flows and reports no offset, so nothing in the
envelope could hold them; a router's clock is exactly the case the collector's `clock_offset_ms` cannot see.

## What this does not do

It does not correct an event time. Quarantine holds a record whose collector reports its clock out of bound; it
does not subtract the reported offset, because a correction is only as good as the offset measurement, and an
experiment that injected an offset and subtracted the same number would measure its own arithmetic. Nor can it see
a clock the envelope does not describe: a switch's, a router's or a host's.

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
    "R-P-L5-006": {"session": null, "session_off_boundary": null},
    "R-P-L7-006": {"saas": "one clock"},
    "R-C-001": {"campaign": {"spread": "1800s"}}
  },
  "chain_breaking_point_to_the_second": {
    "R-C-001": {"held_at": "799s", "changed_at": "800s", "no_longer_flagged": [], "newly_flagged": ["…"]}
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
