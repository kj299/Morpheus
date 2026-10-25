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

# Model inventory

The two models this fork's rules read that are not per-entity baselines, each scored on the CPU and each pinned
to the bytes it was committed with. [`inventory.json`](./inventory.json) is the record: every file with its
sha256, the settings the model was trained under, and where it came from in the upstream repository at
`c6a3b56d`. `morpheus.utils.model_inventory.load_entry` hashes every file before a scorer may use it and refuses
a mismatch, so the version a stage writes beside a score names the bytes that produced it.

| Entry | Rule | Stage | Version |
| --- | --- | --- | --- |
| `abp-pcap-xgb` | R-B-L4-001 | `TC4AbpScoreStage` | `abp-pcap-xgb:c3e05a33fdc02a04` |
| `sid-minibert-20230424` | R-B-L7-003 | `TC7SidScoreStage` | `sid-minibert-20230424:d12f31d6dcebd03d` |

Upstream served both from a Triton model repository, which resolves a model by its name and nothing else, needs a
GPU, and is something CI here cannot run. Ruling D2 of the [repository audit](../../docs/source/developer_guide/guides/13_repository_audit.md)
removed that path and kept these two models.

## `abp-pcap-xgb`

The upstream booster over the thirteen flow features `TC4FlowStage` emits, in the order the upstream preprocessing
declared them. The upstream file is XGBoost's legacy binary format, which XGBoost 3 no longer reads. It was
loaded with XGBoost 1.7.6 and saved as [`abp-pcap-xgb.json`](./abp-pcap-xgb/abp-pcap-xgb.json); the original's
sha256 is in the inventory. [`conversion_check.json`](./abp-pcap-xgb/conversion_check.json) holds 220 rows 1.7.6
scored from the original file: the 20 feature rows the upstream test recorded for its packet capture sample,
and 200 rows drawn from seed 7 across the features' ranges so both classes appear, since the upstream sample is
all one class. XGBoost 3.4.2 reproduces every margin and probability from the JSON exactly, and a tree walk over
the JSON that uses no XGBoost agrees with it; `tests/morpheus/utils/test_abp_scorer.py` checks both.

A flow is anomalous above a probability of 0.5, the threshold the upstream Triton configuration set.

## `sid-minibert-20230424`

The upstream sensitive-information model, ten labels over text, as ONNX. It reads 256 token ids from the
`bert-base-uncased` vocabulary, lower-cased with no special tokens, and a long text is read in windows starting
every 64 tokens, each label taking its highest probability over the windows; those are the settings the upstream
model information recorded. `morpheus.utils.wordpiece` splits text into tokens on the CPU and gives the same ids as Hugging
Face's `tokenizers` on the validation set's 2,000 payloads and on strings built to reach every step of the
algorithm.

[`validation.csv`](./sid-minibert/validation.csv) is the labelled set the model shipped with. The CPU scorer
decides 19,982 of its 20,000 labels correctly and gets 1,982 of the 2,000 payloads entirely right; reading only
the first 256 tokens instead gets 1,975, the figure upstream recorded for this model family's small test set.
`bert-base-uncased-hash.txt` is the vocabulary compiled for cuDF's GPU tokenizer, which upstream read and nothing
here does; it is kept beside the vocabulary it was compiled from as the record of what the GPU path used.

## Fetching the files

The SID model and its validation set are stored in Git LFS. A checkout without them holds pointer files, and the
scorer says so rather than reporting a digest mismatch. To fetch only these:

```bash
git lfs install
git lfs pull --exclude="" --include="models/inventory/**"
```

The `--exclude=""` is needed: the repository's `.lfsconfig` excludes every LFS path from a fetch, and without it
the pull downloads nothing and still reports success.

The CPU workflow does the same before the tiers, caching the objects by their pointers.
