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

# Layer 5 models

Two scripts, two questions.

`train_models.py` trains the per-principal autoencoders the composed layer 5 pipeline scores with, on the CPU, and
commits them as numbers under [`models/`](./models/). CI scores those numbers without Torch. That is how a learned
model reaches the golden file, the sample events and a search head.

`run_model.py` trains the same models on a card and asks whether they produce the same numbers twice. It is the
one script in this fork that cannot run where the fork is developed: it measures the device's own determinism,
and the container everything else here is tested in has no CUDA device.

## The committed models

The session corpus runs a fortnight of ordinary habits before its scored week. `train_models.py` takes each
principal's rows from that fortnight -- six principals, forty rows apiece for the office workers, fifty-six for
the VPN user, fourteen for the service account -- and fits one `morpheus.models.dfencoder` autoencoder per
principal, on the host, from seed 42, for 20 epochs, with the runner's `[8, 4]` and `[4, 8]` layers. Each fitted
model is exported by `morpheus.utils.dfencoder_scorer.export_model`: the weights, the mean and deviation each input
was standardized with, and the mean and deviation of the training losses each feature's error is standardized
against. Those last two are as much the model as the weights are, so all three are covered by the digest the
manifest pins, and all three are written to `models/session_models.json` as the exact decimal of every value.

The pipeline reads that file through `load_models`, which recomputes every digest from the numbers and refuses
an entry whose recorded version no longer matches -- a file edited by hand, or a model moved to another
principal's key. `NumpyAutoEncoder` evaluates each one: standardize, encoder, decoder, squared error,
standardize the error. It does so in float64 from the float32 weights, so a score is a property of the row and
the file rather than of the machine's vector unit; Torch's own float32 pass agrees with it to float32's
precision, and `tests/morpheus/utils/test_dfencoder_scorer.py` checks that wherever Torch imports by putting the
committed numbers back into the upstream class with `load_torch_autoencoder`.

The manifest pins each principal to `dfencoder/<principal>:<digest>` and declares `scores_from_ns`, the end of
the fortnight. `TC5ScoreStage` scores no row before it and `DeterminismStampStage` names no model on
them, because a model asked about the rows it was fitted on reports how well it memorized them. A principal
with no fortnight -- the corpus's joiner -- is scored against the declared fallback, frozen population
statistics fitted on the same fortnight, and her rows say `model_fallback_used` is true.

Rerunning `train_models.py` says whether it reproduces the committed versions. On the machine that wrote them it
does, twice; another processor may sum in a different order and land on different last bits, which is why the
file is committed rather than regenerated, and why the script compares before it writes:

```bash
./examples/layer5_model/train_models.py            # train, compare with the committed file
./examples/layer5_model/train_models.py --write    # train and replace it
```

What the models found is recorded in the session harness and the validation package rather than here, and the
short version is three findings and no detection claim. The scores are not calibrated: where the fortnight never
varied a feature the training losses' spread is a rounding residue, and any change scores in the thousands. The
`*increment` features are cumulative, so a principal whose account reached somewhere new keeps firing until a
model is trained on a window that includes it. And the deterministic R-D-L5-008 misses the takeover the models
find, because its first sign-in failed. A fortnight of six principals is not a training set; what the committed
models establish is that a learned score reaches the SIEM pinned, out of sample and reproducible.

## The card run

`run_model.py` trains on a machine with Torch and a CUDA device. Everything the model half depends on was built
and tested first -- the trajectory feature R-P-L5-006 reads, the determinism envelope, the pinned model manifest,
the sharding -- and this arrives last, on a machine with a card.

## What it measures, and what it does not

**Reproducibility, not detection quality.** The 2026-10-03 run's corpus was a week of five principals'
authentications; the runner now trains on the fortnight of six. That is
nowhere near enough data to train an autoencoder that detects anything, and no claim is made that it does. The
artifact says so in a field of its own, so the caveat travels with the numbers rather than living in a document
beside them.

What a week *is* enough for is the question the determinism controls exist to answer: run the same training
twice, with the same seed and the same data, and find out whether the scores come back identical. If they do
not, every threshold tuned against them is tuned against noise -- and R-P-L5-006, a rule about a score rising by
fractions of a standard deviation, is measuring the model's own jitter rather than a principal's behaviour.

Four checks, each a control from Part 5 of the guide:

| Check | Control | What a failure means |
| --- | --- | --- |
| Double run | The determinism envelope's premise | The same seed and the same data gave different scores. Nothing downstream can be tiered above D3. |
| Batch invariance | Control 5 | The model has a batch-dependent operation -- batch normalization left in training mode is the usual one. A row's score then depends on what it was batched with, and no amount of seeding fixes it. |
| Deterministic algorithms | Control 3 | `torch.use_deterministic_algorithms(True, warn_only=False)` raises at startup rather than an operation silently picking a non-deterministic kernel. |
| The wired path | Controls 1, 3 and 5 together | The composed layer 5 pipeline, with the trained models behind `TC5ScoreStage` through `DfencoderScorer` and a manifest pinning each principal to a digest of its own weights, gave different scores on a second run or under the batch-split sweep. A failure here with the three above passing means the wiring, not the model, is where determinism was lost -- as happened on the first run, and `pipeline_differences` in the artifact names the column and the two values. |

The fourth is the one that puts the model in the slot the pipeline has held open for it. The adapter,
[`morpheus.utils.dfencoder_scorer`](../../python/morpheus/morpheus/utils/dfencoder_scorer.py), is inference only:
it holds fitted models by pinned version and never fits one, because training inside the scoring path would fit
on the rows being scored. The runner trains first and pins each principal to a digest of the resulting
weights, which makes two versions equal exactly when two sets of weights are; that is what control 1 has to
mean for a model that was trained rather than downloaded. It declares no fallback, and a principal without a
model is refused rather than scored against another principal's.

**A row goes to the model on its own.** `TC5ScoreStage` hands the scorer the rows one principal has in the
message it happens to be holding, so the size of that group is a fact about how the stream was chunked rather
than about the data. A network is not shape-invariant to the last decimal -- the same row in a batch of twenty
and in a batch of one takes different kernels and can come back differing in the seventh place -- and that
difference does not stay small: `drift_rise_sigmas` divides a rise by the spread of a few nearly equal scores,
so a seventh-place wobble upstream arrives as tenths downstream, far past what rounding absorbs. The adapter
therefore fixes the shape itself and asks for one row at a time, which makes a score a function of its row.
The first run of this check on a card failed for exactly this reason, which is what the check was for.

**The 2026-10-03 run's models scored the rows they were trained on.** That was a leak, made on purpose and
written into the artifact: the question that run answered was whether the wired path gives the same numbers twice
with a real model in the slot. The runner now trains on the fortnight and scores the week after it, through the
same `PinnedScorer` and fallback CI uses, and records whether the card trained the same numbers the CPU
committed in a field of its own, `committed_models_match`, outside the verdict. It has not been run on a card
since that change. The runner's own verdict requires all four checks.

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

The development container (`./docker/run_container_dev.sh`) works too, and is what the October run used; it
needs `./scripts/compile.sh` on every start and two additions the image lacks, which the
[README](../../README.md#what-this-fork-is-not) lists beside the GPU verdict.

Exit status is zero only when both the double run and the batch sweep came back identical. On a machine without
Torch or without a device it exits non-zero and writes a `"verdict": "failed"` artifact saying which piece was
missing -- an artifact that is simply absent reads as not yet run, and a run that quietly skipped the model would
be worse than one that refuses. That refusal is itself tested, in the container that has neither.

## The result

Run on 2026-10-03 at 23:53 UTC, on an NVIDIA RTX 5000 Ada Generation Laptop GPU with `torch==2.4.0+cu124`,
seed 42 and 20 epochs, over five principals carrying 26, 18, 26, 28 and 7 usable rows: **all four checks
passed.** The model's own double run was identical and its scores were invariant across batch sizes 1, 8 and
64; the composed pipeline, with those models behind `TC5ScoreStage`, gave the same 105 scores twice and again
under the batch-split sweep, with `pipeline_differences` empty, five principals pinned and none skipped.
Verdict `passed`, `pipeline_mean_abs_z_max` 2.3823, and these weight digests:

| Principal | `model_version` |
| --- | --- |
| `alice@example.com` | `dfencoder/alice@example.com:ebf33ed32d52626c` |
| `bob@example.com` | `dfencoder/bob@example.com:a3355010ab77c564` |
| `carol@example.com` | `dfencoder/carol@example.com:47e623d362182a67` |
| `dave@example.com` | `dfencoder/dave@example.com:aa2262f43228bb20` |
| `svc-batch@example.com` | `dfencoder/svc-batch@example.com:cd5ddb14349473d3` |

That is the first passing run, 2026-09-19 at 23:17 UTC, repeated on the tree after the layer 5 rules and
context landed: the same row counts, the same 105 rows and the same `pipeline_mean_abs_z_max`. The September
digests were not written down anywhere this repository keeps, so whether these five match them is not
something this file can say; recording them here is what makes the next comparison possible.

The fourth check failed on its first attempt, on 2026-09-13, and the failure was real: the adapter was
letting the size of a message's row group reach the model, so a score depended on how the stream had been
chunked. The adapter now asks for one row at a time. Two things are worth reading off the two artifacts
together. The five weight digests are identical across the two runs, so training is reproducible across days
and reboots and the repair changed how rows were handed over rather than what was fitted. And
`pipeline_mean_abs_z_max` is 2.3823 in both, so the scores themselves did not move -- what moved was whether
they stayed put under re-batching.

That is controls 1, 3 and 5, measured. It is not a statement that the model detects anything, and the
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
  "principals": {"alice@example.com": 26, "…": 0},
  "double_run_reproducible": true,
  "batch_invariant": true,
  "batch_sizes": [1, 8, 64],
  "pipeline_double_run_reproducible": true,
  "pipeline_batch_invariant": true,
  "pipeline_differences": {},
  "pipeline_scored_rows": 105,
  "pipeline_principals_pinned": 5,
  "pipeline_principals_skipped": {},
  "model_versions": {"alice@example.com": "dfencoder/alice@example.com:…", "…": "…"},
  "pipeline_mean_abs_z_max": 2.3823,
  "measures": "reproducibility of the scoring path, not detection quality: …"
}
```

The `pipeline_*` fields and `model_versions` are absent from artifacts written before 2026-09-13, which
predate the wired path; a run that reports them is one that scored the composed pipeline with the model.
`pipeline_differences` is empty on a passing run and otherwise carries one entry per disagreeing run, each
naming the column, the canonical row and the two values. A failure can then be read off the artifact rather
than reproduced on the card it happened on.

Commit it as `examples/layer5_model/artifacts/<date>/layer5_model.json`, the date the run's `at` field gives, and
the numbers in `README.md`, in the guide and in this file are tested against it by
`tests/morpheus/determinism/test_verdict_artifacts.py`: every date, count, digest and threshold quoted about the
newest run is built from its fields, so prose that drifts from the artifact fails without a card. The GPU
conformance artifact is held to the same standard under `ci/artifacts/`. Until 2026-10-04 both were ignored by
git, and the September figures outlived the tree they described by two weeks.
