<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Examples
Each example runs on the CPU from a clone, with the environment described in
[Getting Started with This Fork on a CPU](../docs/source/fork_getting_started.md). Only the simple Python stage
reads a file stored in Git LFS; fetch it first with `./scripts/fetch_data.py fetch examples`.

* [Behavioral analytics segment runners](./behavioral_analytics/README.md) - One runner per segment, from collector files to the records a SIEM receives
* [Splunk Lineage App for Morpheus Behavioral Analytics](./splunk_lineage_app/README.md) - The installable Splunk app that implements the SIEM half of the design
* [Layer 5 models](./layer5_model/README.md) - Training and scoring the per-principal and per-peer-group models the layer 5 rules read
* [Clock skew tolerance](./clock_skew/README.md) - How much clock skew each detection tolerates, measured rule by rule
* [Upstream reuse](./upstream_reuse/README.md) - Which upstream digital fingerprinting pieces the fork reuses, measured
* [Simple Python stage](./developer_guide/1_simple_python_stage/README.md) - The example for developer guide 1, writing a stage
