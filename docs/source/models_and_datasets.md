<!--
SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

Every model a stage of this fork scores with is a committed file, and every score it writes carries a version that
names the bytes that produced it: `<name>:<first 16 hex digits of a sha256>`. A scorer hashes its files before it
loads them and refuses a mismatch, so a score that reaches the SIEM can be traced to one file and reproduced from it.
No model is resolved by name alone, and none is fitted inside the scoring path.

| Model | Rule | Stage | Where it is recorded |
| --- | --- | --- | --- |
| `abp-pcap-xgb`, an XGBoost flow classifier over the thirteen layer 4 features | R-B-L4-001 | `TC4AbpScoreStage` | [`models/inventory`](../../models/inventory/README.md) |
| `sid-minibert-20230424`, a BERT-mini sensitive-information model over ten labels | R-B-L7-003 | `TC7SidScoreStage` | [`models/inventory`](../../models/inventory/README.md) |
| Per-principal `dfencoder` autoencoders over the layer 5 session features | R-B-L5-001, R-B-L5-002 | `TC5ScoreStage` | [`examples/layer5_model`](../../examples/layer5_model/README.md), `models/session_models.json` |
| Per-peer-group `dfencoder` autoencoders over the host feature frame | R-P-L3-009 | `TC5ScoreStage` | [`examples/layer5_model`](../../examples/layer5_model/README.md), `models/host_models.json` |

The two inventory models come from the upstream repository at commit
[`c6a3b56d`](https://github.com/nv-morpheus/Morpheus/tree/c6a3b56db1c77c2f375ae0833a65a7b6d2272269), where a Triton
model repository served them on the GPU. `models/inventory/inventory.json` records each file's sha256, the upstream
path and digest it came from, and the settings the model was trained under; both now score on the CPU. The layer 5
models are trained by this fork, and a manifest pins each principal or host to the digest of its model's weights.
