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

# Layer 5 model determinism

`run_model.py` trains a per-principal autoencoder on the layer 5 corpus and asks whether it produces the same
numbers twice.

It is the one script in this fork that cannot run where the fork is developed. `morpheus.models.dfencoder` is in
the tree, but it is a Torch model on a CUDA device, and the container everything else here is tested in has
neither. Everything the model half depends on was therefore built and tested first -- the trajectory feature R-P-L5-006
reads, the determinism envelope, the pinned model manifest, the sharding -- and this arrives last, on a machine
with a card.

## What it measures, and what it does not

**Reproducibility, not detection quality.** The corpus is a week of five principals' authentications. That is
nowhere near enough data to train an autoencoder that detects anything, and no claim is made that it does. The
artifact says so in a field of its own, so the caveat travels with the numbers rather than living in a document
beside them.

What a week *is* enough for is the question the determinism controls exist to answer: run the same training
twice, with the same seed and the same data, and find out whether the scores come back identical. If they do
not, every threshold tuned against them is tuned against noise -- and R-P-L5-006, a rule about a score rising by
fractions of a standard deviation, is measuring the model's own jitter rather than a principal's behaviour.

Three checks, each a control from Part 5 of the guide:

| Check | Control | What a failure means |
| --- | --- | --- |
| Double run | The determinism envelope's premise | The same seed and the same data gave different scores. Nothing downstream can be tiered above D3. |
| Batch invariance | Control 5 | The model has a batch-dependent operation -- batch normalization left in training mode is the usual one. A row's score then depends on what it was batched with, and no amount of seeding fixes it. |
| Deterministic algorithms | Control 3 | `torch.use_deterministic_algorithms(True, warn_only=False)` raises at startup rather than an operation silently picking a non-deterministic kernel. |

`CUBLAS_WORKSPACE_CONFIG` is set before Torch is imported, because it is read when the CUDA context is created
and setting it later has no effect while looking exactly like setting it correctly. The script refuses to
continue if Torch is already imported rather than pretending.

## Features

The ten columns the model trains on are all derived by the five TC-5 stages, not raw columns a collector sent:

```
logcount  locincrement  appincrement  deviceincrement  asns_in_window
hour_surprise_bits  weekday_surprise_bits  mfa_ratio
auth_attempts_in_window  auth_failures_in_window
```

That distinction is the layer's whole argument, and it is asserted rather than described: a model trained on
`source_country` learns which countries the estate has, which is a fact about the estate; one trained on
`locincrement` learns how often this principal goes somewhere new, which is a fact about the principal.
`tests/morpheus/determinism/test_layer5_model_runner.py` compares the feature list against the raw corpus
frames and fails if a collected column appears in it.

Rows with a null in any feature are dropped rather than imputed. These columns are counts and surprise scores,
and a null means the row carried no such measurement -- not a measurement of zero. How many rows survive is
printed per principal.

## Running it

On a machine with a CUDA device and the Morpheus environment, plus Torch:

```bash
conda env create -f conda/environments/all_cuda-128_arch-x86_64.yaml   # ships torch==2.4.0+cu124
conda activate morpheus
./examples/layer5_model/run_model.py build/layer5_model.json
```

Options: `--epochs` (default 20) and `--seed` (default 42). The positional argument is where the artifact is
written and defaults to `layer5_model.json` at the repository root.

Exit status is zero only when both the double run and the batch sweep came back identical. On a machine without
Torch or without a device it exits non-zero and writes a `"verdict": "failed"` artifact saying which piece was
missing -- an artifact that is simply absent reads as not yet run, and a run that quietly skipped the model would
be worse than one that refuses. That refusal is itself tested, in the container that has neither.

## The result

Run on 2026-09-07 at 12:01 UTC, on an NVIDIA RTX 5000 Ada Generation Laptop GPU with `torch==2.4.0+cu124`,
over five principals carrying 26, 18, 26, 28 and 7 usable rows: **the double run was identical and the
scores were invariant across batch sizes 1, 8 and 64.** Verdict `passed`.

That is controls 3 and 5, measured. It is not a statement that the model detects anything, and the
paragraph above is not softened by the result: seven rows is not a training set, and neither is
twenty-eight.

## The artifact

Same shape as `gpu_conformance.json`: what ran, on what card, with what result.

```json
{
  "verdict": "passed",
  "at": "…",
  "device": "…",
  "torch": "2.4.0+cu124",
  "seed": 42,
  "epochs": 20,
  "principals": {"alice@example.com": 21, "…": 0},
  "double_run_reproducible": true,
  "batch_invariant": true,
  "batch_sizes": [1, 8, 64],
  "measures": "reproducibility of the scoring path, not detection quality: …"
}
```

Paste it back and the numbers in `README.md` and in the guide can be traced to a run rather than to a memory of
one -- which is the same standard the GPU conformance verdict is held to, and the reason neither claim in this
repository was written before its artifact existed.
