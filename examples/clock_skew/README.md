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

It runs on any machine. No card, no Torch, about forty seconds.

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

Swept at one millisecond, ten, a hundred, one second, ten, thirty and sixty, on three axes.

| Rule | Collector clocks | Switch clocks | Layer 5 clocks |
| --- | --- | --- | --- |
| R-D-L2-001, MAC count on an access port | unchanged to 60s | unchanged to 60s | -- |
| R-D-L2-003, ARP anomaly | unchanged to 60s | unchanged to 60s | -- |
| **R-D-L2-004, MAC in two places** | unchanged to 60s | **changes at 60s** | -- |
| R-D-L2-005, authorization without authentication | unchanged to 60s | unchanged to 60s | -- |
| R-D-L5-003, impossible travel | -- | -- | unchanged to 60s |
| R-D-L5-004, multi-factor fatigue | -- | -- | unchanged to 60s |
| **R-P-L5-006, drift trajectory** | -- | -- | **changes at 1ms, and at nothing once moved off the hour marks** |

Six of the seven are untouched by a full minute of disagreement. The two that move are the interesting ones,
and neither moves for the reason the guide predicted.

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

At a millisecond of spread the drift trajectory stops flagging three of the seven principal-days it flags on
the reference corpus. That reads like a rule needing millisecond synchronization, and it is not.

Forty-five of the layer 5 corpus's hundred and five authentications sit exactly on an hour mark, because the
corpus builds its times from whole hours. An event on a boundary changes window under an offset of one
nanosecond. Move the same events into the middle of their windows -- a uniform shift, which is not a skew at
all, since no two clocks disagree any more than before -- and a full minute of spread changes nothing the rule
accuses.

**The exposure is therefore not the size of the clock error. It is the fraction of events sitting near a
window edge.** An estate cannot read a tolerance off this rule. What it can read is that window-boundary proximity is
the variable to think about, and that a corpus built on round numbers will overstate the fragility of anything
measured against it.

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

## What this does not do

It does not correct anything. `clock_source` and `clock_offset_ms` are in the universal envelope and no stage
produces or consumes either; they are schema, not a control. This measures the damage a deployment takes today,
not the damage after a correction that has not been built.

It also measures these corpora. The magnitudes at which things break are properties of an hour of one estate
and a week of five principals, with these thresholds. The mechanisms generalize; the numbers are a worked
example, and the reason each number is what it is, is written down beside it so an estate can redo the
arithmetic against its own sweep times and thresholds.

## The artifact

```json
{
  "at": "…",
  "spread_widths": ["1ms", "10ms", "100ms", "1s", "10s", "30s", "60s"],
  "breaking_point": {
    "R-D-L2-004": {"estate_collectors": null, "estate_switches": {"spread": "60s"}},
    "R-P-L5-006": {"session": {"spread": "1ms"}, "session_off_boundary": null}
  },
  "sweeps": {"estate_collectors": {"runs": ["…"]}, "…": "…"}
}
```

Each run records the offsets applied, whether the output was identical, every column that moved with a row
count and a maximum delta, what each rule stopped and started accusing, and how far the ladder reached.

The findings above are asserted in
[`tests/morpheus/determinism/test_clock_skew_experiment.py`](../../tests/morpheus/determinism/test_clock_skew_experiment.py)
against the pipelines rather than quoted from a saved artifact, so a change that invalidates one of them fails
a test rather than leaving a document describing a system that stopped behaving that way.
