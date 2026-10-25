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

# Models

What this directory holds for the fork, and where each model a stage scores with is recorded.

## The model inventory

[`inventory`](./inventory/README.md) holds the two shipped models the design rules read: the `abp-pcap-xgb` flow
classifier behind R-B-L4-001 and the `sid-minibert-20230424` sensitive-information model behind R-B-L7-003. Each
file is recorded with its sha256, the settings the model was trained under and its upstream provenance, and both
score on the CPU inside the pipeline (`TC4AbpScoreStage`, `TC7SidScoreStage`). Upstream served them from a Triton
model repository; ruling D2 of the [repository audit](../docs/source/developer_guide/guides/13_repository_audit.md)
removed that path and moved these two models here.

The fork's own learned models are not in this directory. The layer 5 per-principal and per-peer-group models are
committed under [`examples/layer5_model/models`](../examples/layer5_model/README.md) with the manifests that pin them.

## Data the fork reads

| Path | Reader |
| --- | --- |
| `data/public_suffix_list.dat` | `morpheus.utils.query_entropy`, which splits a DNS name into its registered domain and subdomain for the layer 7 DNS features. The file is the [Public Suffix List](https://publicsuffix.org/list/) as upstream committed it on 2023-07-25, sha256 `3d1e812e00ca5c140b66ab5351f5cf990f167fad6558a8549eb5cd5922a79e2d`. |
| `datasets/validation-data/dfp_training_feature_samples.csv`, `datasets/validation-data/dfp-cloudtrail-role-g-validation-data-input.csv`, `datasets/validation-data/dfp_dataloader_test/` | The `dfencoder` tests under `tests/morpheus/dfencoder`; `dfencoder` is the autoencoder the fork's layer 5 scorer trains. |
| `datasets/training-data/azure/azure-ad-logs-sample-training-data.json`, `datasets/training-data/cloudtrail/hammah-role-g-training-part1.json`, `datasets/training-data/cloudtrail/hammah-user123-training-part2.json` | `examples/upstream_reuse`, the recorded measurement behind the decision not to reuse the upstream digital fingerprinting stages (ruling D4). |

## Everything else

The other directories here (`abp-models`, `phishing-models`, `training-tuning-scripts`,
`validation-inference-scripts`, the rest of `datasets` and `data`) are upstream model material for use cases the
audit removes. Each file's verdict and the ruling behind it are in
[`ci/artifacts/repository_audit_manifest.csv`](../ci/artifacts/repository_audit_manifest.csv); until they go, the
upstream description of each is at the fork point,
[`models/README.md` at `c6a3b56d`](https://github.com/nv-morpheus/Morpheus/blob/c6a3b56db1c77c2f375ae0833a65a7b6d2272269/models/README.md).
