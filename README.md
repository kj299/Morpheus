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
| **Lineage substrate** | Deterministic `event_uid` / `link_uid` provenance identifiers, the Community ID flow hash (verified against the reference implementation over 46,448 flow tuples), time-bounded binding resolution with a fixed tie-break, and event-time window sealing with a lateness horizon and a separate late-arrival stream |
| **Layer 1 (TC-1)** | Interface counter normalization that tells a counter wrap from a device reboot, transceiver and neighbor novelty, optical power scored against each port's own rolling baseline, link flap counting that catches flaps between two polls, and identifier change detection with no period boundary |
| **Layer 2 (TC-2)** | Binding closure into the half-open intervals the resolver consumes, optionally emitting a provisional record the moment a binding opens so live attribution has an answer inside the idle window, the three cardinality features, the gratuitous ARP proportion, and 802.1X authorization timing with unpaired authorization flagged |
| **Determinism** | A total row order imposed before any stateful stage, frame canonicalization and digesting, score quantization, and a CI harness running control 13's six checks against both the lineage pipeline and the composed layer 1 and 2 telemetry pipeline over seeded, snapshot-shaped corpora with planted anomalies |
| **SIEM side** | `TA-morpheus-lineage`, an installable Splunk app (indexes, sourcetypes, KV Store binding lookups, and scheduled searches), validated by AppInspect, a live load into Splunk Enterprise 10.2, and a functional pass against seeded telemetry ([README](./examples/splunk_lineage_app/README.md)) |
| **First detections** | Four deterministic layer 2 rules as saved searches in the app: a MAC in two places at once, 802.1X authorization with no authentication in front of it, more MACs than permitted on a single-host port, and an address claimed by more than one MAC. The first fires on the interval between the two sightings rather than on their end reason, because an estate polls its switches in sequence and a cross-switch spoof is therefore seconds apart rather than simultaneous. The last two depend on a list the estate owns and ship with the hook for it. R-D-L2-001 fires on nothing until its port designation lookup is populated; R-D-L2-003 is the opposite, and fires on every redundancy gateway until its exclusion list is supplied. All four predicates asserted in Python over the planted corpus. Not yet run on a live search head |

Fourteen stages and nineteen supporting modules, covered by 914 tests.

### What this fork is not

Being clear about the boundary is the point of writing it down:

- **The collectors are out of scope.** The SNMP, LLDP, DHCP, and 802.1X polling that produces layer 1
  and 2 telemetry is not Morpheus and is not here. What ships is everything downstream of it.
- **Layers 3-7 are designed, not built.** The telemetry classes, detection rules, and Splunk queries
  for those layers are specified in the guide. Only layers 1 and 2 have running feature stages.
- **The rule thresholds are placeholders** unless a rule says otherwise. They are starting points for
  tuning against an estate's own data, not calibrated values.
- **"Predictive" is a claim the guide qualifies rather than asserts.** Three of its four mechanisms are
  forward-looking in a defensible sense. The fourth, the premise that autoencoder reconstruction error
  rises during reconnaissance and staging, is a hypothesis this work does not establish, and a deployment
  should validate the lead time against its own incident history before promising prediction to
  anyone.
- **GPU execution mode now has one measured result, on one machine.** On 2026-09-05 the 203 `gpu_mode`
  variants were run for the first time, on an NVIDIA RTX 5000 Ada Generation Laptop GPU (compute
  capability 8.9, driver 596.58) under WSL2: **226 passed, 2 failed, 55 skipped**, and both failures are
  in upstream Morpheus files (`test_deserialize_stage_pipe`, `test_write_to_file_stage_pipe`) rather than
  in anything this fork adds. Every stage and utility added here passes in GPU mode.
  That is one run on one card, not a support claim.
- **The two modes did not agree, and the per-stage runs could not have told us.** Every one of those 203
  variants passes, and the composed telemetry pipeline still produced `arp_count_in_window = 3.0` on a GPU
  where the CPU golden holds `3`. Nothing raised. cuDF's `to_pandas` cannot put a null inside an integer
  column, so it widens the column to float64 and writes NaN -- and every windowed count here is null on
  the rows belonging to other telemetry classes. The conversion is not where it happens: collecting the
  classes fills that column with gaps for every other class's rows, and a plain integer column cannot
  hold one, so the fill widens it. Integer columns are now carried in a type that admits a gap, in both
  modes,
  before anything is joined, which also cost the golden fourteen columns' worth of trailing `.0` and is
  the better rendering. The lineage pipeline agreed across modes throughout.
  **The repair has not itself been re-run on a GPU**, so control 13 is still verified in CPU mode alone.
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
* [List of available Morpheus stages](./docs/source/stages/morpheus_stages.md) - The `Lineage` and `Telemetry` sections cover the stages added here
* [Splunk Lineage App](./examples/splunk_lineage_app/README.md) - The SIEM half, as an installable Splunk app

### Modifying Morpheus
* [Contributing to Morpheus](./docs/source/developer_guide/contributing.md) - Covers building from source, making changes and contributing to Morpheus

Full documentation for the latest official release is available at [https://docs.nvidia.com/morpheus/](https://docs.nvidia.com/morpheus/).
