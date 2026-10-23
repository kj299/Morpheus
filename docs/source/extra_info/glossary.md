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

# Glossary of Common Morpheus Terms

<!-- Please keep these sorted alphabetically -->
## determinism tier
The reproducibility guarantee a pipeline declares and records on every event in the `determinism_tier` column. **D0** (bit-exact) means identical input bytes produce byte-identical output, floats included. **D1** (decision-stable) means identical input produces identical alert decisions and rule firings while scores may differ in low-order bits. **D2** (lineage-stable) means identical input produces identical [`event_uid`](#event_uid), `link_uid` and [`lineage_id`](#lineage_id) values and identical chain membership. **D3** (explanation-stable) is D1 plus an identical top-k ranking of the per-feature `z_loss` columns. Refer to [Four tiers](../developer_guide/guides/11_predictive_behavioral_analytics_osi.md#four-tiers).

## `entity_key`
The primary behavioral subject of a [telemetry class](#telemetry-class): a port at layer 1, a MAC address at layer 2, a principal at layer 5. Composite keys are joined with `:` by {py:mod}`~morpheus.utils.entity_key`, so every layer composes the same string, and a key with a missing part is null rather than a string containing `None`. `EnvelopeStampStage` stamps `entity_key` and `osi_layer` at the tail of each class's segment.

## `event_uid`
The deterministic identity of one record: the first 32 hexadecimal characters of a SHA-256 over `collector_id`, `schema_version`, `origin_hash` and `collector_seq`, separated by a unit-separator byte. It depends on the collector's sequence number rather than a row index, so it does not change with how the collector batched the record. Computed by {py:mod}`~morpheus.utils.lineage` and applied by `LineageStampStage`. Refer to [Deterministic lineage identifiers](../developer_guide/guides/11_predictive_behavioral_analytics_osi.md#deterministic-lineage-identifiers).

## `lineage_id`
The identity of a correlation chain: the first 32 hexadecimal characters of a SHA-256 over the root [`entity_key`](#entity_key), the `window_id` and the Merkle root of the chain's sorted, deduplicated [`event_uid`](#event_uid) values. Because sorting and deduplication come before hashing, it depends on chain membership alone and survives out-of-order delivery, redelivery and replay.

## MLflow Triton Plugin
[MLflow](https://mlflow.org/) plugin for deploying your models from MLflow to [Triton Inference Server](#triton-inference-server). Refer to [`mlflow-triton-plugin`](https://github.com/triton-inference-server/server/tree/main/deploy/mlflow-triton-plugin) for more information.

## module
A Morpheus module is a type of work unit that can be utilized in the Morpheus stage and can be registered to a MRC segment module registry. Modules are beneficial when there is a possibility for the work-unit to be reused.

## MRC
[Morpheus Runtime Core (MRC)](https://github.com/nv-morpheus/MRC). Pipelines in MRC are low level representations of Morpheus [pipelines](#pipeline).

## NGC
[NVIDIA GPU Cloud](https://catalog.ngc.nvidia.com/) is the official location for Morpheus and many other NVIDIA Docker containers.

## node
An individual node in the MRC pipeline. In Morpheus, MRC nodes are constructed by [stages](#stage).

## operator
Refers to small-reusable MRC nodes contained in the `mrc.core.operators` Python module which perform common tasks such as:
* `filter`
* `flatten`
* `map`
* `on_completed`
* `pairwise`
* `to_list`

## pipeline
Represents all work to be performed end-to-end in Morpheus. A Morpheus pipeline consists of one or more [segments](#segment), and each segment consists of one or more [stages](#stage). At build time, a Morpheus pipeline is transformed into a [MRC](#mrc) pipeline which is then executed.

## RxCpp
MRC is built on top of [RxCpp](https://github.com/ReactiveX/RxCpp) which is an open source C++ implementation of the [ReactiveX API](https://reactivex.io/). In general, Morpheus users are only exposed to this when they wish to write a [stage](#stage) in C++.

## Scorer
The protocol in `morpheus.stages.telemetry.tc5_score_stage` through which `TC5ScoreStage` asks a model for scores. Its one method, `score(model_version, features)`, takes the `name:version` the model manifest pinned for an entity and that entity's rows in event order, and returns each row's absolute z-score per feature; the stage derives `mean_abs_z` and `max_abs_z` from those itself. `ReferenceScorer` and `DfencoderScorer` ({py:mod}`~morpheus.utils.dfencoder_scorer`) implement it.

## segment
A subgraph of a [pipeline](#pipeline). Segments allow for both logical grouping, and distribution across multiple processes and execution hosts.

## sourcetype contract
The record in {py:mod}`~morpheus.utils.siem_sourcetypes` of what this fork emits for each stanza in the Splunk app's `props.conf`: the column `_time` is anchored on, every column rendered as a timestamp, the producer, and the columns the app's searches read, which a producer may add to but not drop. A stanza with no producer yet is listed as unproduced. `tests/morpheus/utils/test_siem_sourcetypes.py` holds the contract and `props.conf` to each other.

## stage
Fundamental building block in Morpheus representing a unit of work. Stages may consist of a single MRC node, a small collection of nodes, or an entire MRC subgraph. A stage can encapsulate any piece of functionality and is capable of integrating with any service or external library. Refer to [Simple Python Stage](../developer_guide/guides/1_simple_python_stage.md).

## telemetry class
A family of records that share a schema and a behavioral subject, named in the `telemetry_class` envelope field: `TC-1` through `TC-7` for OSI layers 1 to 7, and `TC-0` for identity and asset context. Records of classes `TC-1` to `TC-7` carry the same fourteen-field universal envelope. The guide also treats the platform that delivers the telemetry as a class, `TC-P`, whose records are the pipeline's own health per collector, class and tenant. Refer to [Part 2: Telemetry Class Requirements](../developer_guide/guides/11_predictive_behavioral_analytics_osi.md#part-2-telemetry-class-requirements).

## Triton Inference Server
Triton Inference Server, part of the NVIDIA AI platform, streamlines and standardizes AI inference by enabling teams to deploy, run, and scale trained AI models from any framework on any GPU- or CPU-based infrastructure. Most Morpheus pipelines utilize Triton for inferencing via the `TritonInferenceStage`. Refer to [https://developer.nvidia.com/nvidia-triton-inference-server](https://developer.nvidia.com/nvidia-triton-inference-server)

## window seal
The point at which a time window is closed for scoring: when the event-time watermark, the largest event time seen so far, reaches the window's end plus a lateness horizon. Windows are half-open intervals anchored to a fixed epoch, and only sealed windows are scored. A row that arrives for a sealed window goes to a late-arrival stream and never changes the published window. Because the watermark rather than the wall clock drives the seal, a replay seals the same windows at the same points. Implemented by `WindowSealStage` over {py:mod}`~morpheus.utils.window_seal`. Refer to [Control 7: Deterministic windows](../developer_guide/guides/11_predictive_behavioral_analytics_osi.md#control-7-deterministic-windows).
