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

# Which upstream pieces the fork reuses, measured

The design guide names upstream building blocks for the per-layer pattern: `morpheus_dfp`'s split, rolling window,
training, MLflow writer, inference and postprocessing stages, the identity-provider source stages,
`MLFlowDriftStage` and `TimeSeriesStage`. The fork built its own paths instead and never wrote down why. This runs
each of those pieces and records what happened, so each reuse-or-reject decision rests on a measurement rather than
a reading.

```bash
./scripts/fetch_data.py fetch datasets   # the Azure and CloudTrail samples, from Git LFS
./examples/upstream_reuse/evaluate.py /tmp/upstream_reuse.json --lfs-data .
```

It needs Torch and the `morpheus_dfp` package importable, and took 45 seconds on a CPU-only container. Without
`--lfs-data` the source-stage section records the samples as not fetched and measures nothing about them.

The decisions and the reason for each are in
[Pass 4 of the design guide's Part 0](../../docs/source/developer_guide/guides/11_predictive_behavioral_analytics_osi.md#pass-4-what-to-reuse-measured).
This page says how the run was made and how far its numbers can be trusted.

## What it runs

- **The DFP model path** over the session corpus's fortnight and week, the same rows the fork's own models train
  and score on: the pipeline's acceptance of the stages in CPU mode, the writer as shipped and with each defect it
  meets repaired in turn, training twice with and without `manual_seed`, a second version registered under a warm
  and a cold inference stage, the scored week delivered whole, a day at a time and an event at a time, one late
  sign-in, and the postprocessing stage's `event_time` across two runs.
- **The source stages** over the upstream samples, mapped onto the fifteen columns the fork's layer 5 stages read
  and the four envelope fields a row needs for a lineage identity: the Azure sign-in sample through the stage's
  renaming and derivation, as shipped and with the renaming repaired, and the CloudTrail samples. The tree holds no
  Duo sample, so Duo's mapping is recorded from its declared schema with `measured` false.
- **`MLFlowDriftStage`** over the scored rows, then over the same scores given as the classifier tensor it reads,
  at two pipeline batch sizes.
- **`TimeSeriesStage`** over the failing optic's hour from the telemetry corpus's layer 1: at its defaults, scaled down
  to the hour, with a burst of polls added, and with the receive levels flattened and reversed.

## How it runs without a card

The DFP stages import cuDF when their module loads and declare the GPU as their only execution mode; the drift and
time-series stages compute with CuPy. On a machine with a card they run as shipped. Without one the runner stands
pandas in for cuDF and NumPy for CuPy, gives NumPy CuPy's `choose` and `.get()`, and calls each stage's own methods
rather than a pipeline that would refuse them. `environment.shims` in the artifact lists every substitution made.

The MLflow changes are different in kind. They are repairs to defects the run found, each applied only to measure
what comes after it and each recorded beside the defect in the artifact: logging the model in pickle format, because
MLflow 3 refuses its default `pt2` format without an input example; registering the version at the path the model
was logged to; and naming the model the same way on both sides. The runner writes its MLflow store and the rolling
window caches to a temporary directory, never to the tree.

## How far the numbers repeat

Rerun twice more on the same machine, every value in the artifact repeated except these, and each is a finding:

| Value | Why it differs between runs |
| --- | --- |
| `at` | When the run was made. |
| `dfp.event_time` | The postprocessing stage writes the clock time of detection; that is the finding. |
| `dfp.training_double_run.unseeded_digests` | Upstream training is not seeded; that is the finding. The seeded pair repeats. |
| `dfp.latest_version_and_cache.before_retraining` and `after_retraining_same_stage` scores | Version 1 is trained unseeded, as shipped: 1.6076, 1.5406 and 1.3501 on the first row across the three runs. The version each stage resolves, and version 2's scores, repeat. |
| `dfp.batch_split.events_whose_last_score_differs_across_batchings` | 1, 1 and 2: every principal but one is scored by an unseeded version 1, so which events' last scores differ by more than the rounding depends on the weights. The row counts and the largest difference repeat. |

`PYTHONHASHSEED=0` was set for the committed run. Versions are recorded under `environment`; the run used the
MLflow and pandas the fork's CPU lock resolves.

## What tests it

[`tests/morpheus/determinism/test_upstream_reuse.py`](../../tests/morpheus/determinism/test_upstream_reuse.py)
holds every number Pass 4 quotes to the latest committed artifact under [`artifacts/`](./artifacts/), and checks
the run's findings are what Pass 4 says they are. It also re-checks, against the code in the tree, the reasons that
need no card, Torch or LFS sample: the Azure and Duo stages still rename nothing, the writer still escapes a dot
the reader does not, `TimeSeriesStage`'s defaults still cannot reach its own threshold, and the DFP stages still
declare the GPU alone. If upstream repairs one of those, the test fails and the decision it supports is due another
look.
