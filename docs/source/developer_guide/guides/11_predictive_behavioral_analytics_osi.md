<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Predictive Behavioral Analytics Across OSI Layers 1-7

This document has two halves.

The first half is a structured analysis of the Morpheus codebase, conducted in three passes: the execution
model, the behavioral analytics substrate, and the determinism and I/O boundaries.

The second half applies that analysis to a concrete enterprise problem: building predictive behavioral
analytics that spans all seven layers of the OSI stack, feeds a SIEM, and produces output a detection
engineer can reproduce and defend.

Every claim about Morpheus below is anchored to a file path in this repository. Where a capability does not
exist and must be built, the document says so explicitly rather than implying the SDK already covers it.

## Summary

**The four decisions that matter.** If you read nothing else:

1. **Emit `community_id` on layers 3 and 4.** One field converts most cross-layer correlation from
   time-bounded approximate joins into equality joins, and makes Morpheus output joinable against Zeek,
   Suricata, and Elastic with no coordination. Implemented as
   {py:class}`~morpheus.stages.lineage.community_id_stage.CommunityIdStage`.
2. **Shard by entity hash instead of raising `pe_count`.** Intra-stage threading destroys output
   ordering; routing each entity to a fixed single-engine branch keeps N-way parallelism and gives
   total per-entity order. This is control 4 in Part 5 and the cheapest determinism win available.
3. **Sort before computing any cumulative feature.** `IncrementColumn` and `DistinctIncrementColumn`
   are order-dependent, so an unsorted batch silently produces different features and different scores.
   This is the most easily missed defect in the whole design, because the output stays plausible.
4. **Build layers 5 and 7 first, then the lineage substrate, then everything else.** The sequencing in
   Part 6 exists because the standalone detection value is concentrated at the top of the stack while
   the collection cost is concentrated at the bottom.

**The SIEM is Splunk.** Part 4 targets Splunk specifically and goes as far as the configuration stanzas,
rather than covering several SIEM products shallowly. The parts that are SIEM-independent (the
identifier ladder, the deterministic identifiers, `community_id`, the binding tables) are marked as such,
and
[If the SIEM is not Splunk](#if-the-siem-is-not-splunk) says what a port would have to change.

**What is verified versus designed.** This document was written before any of it was built, and the
boundary has moved since. What now runs: the lineage substrate (identifiers, Community ID, binding
resolution, window sealing), the TC-1 and TC-2 feature stages, control 8's total order, and control
13's CI harness. That is fourteen stages and eighteen supporting modules under 831 tests, itemized in
[Part 6](#provided). The Community ID implementation was checked against the reference implementation
over 46,448 flow tuples, and the Splunk app was validated three ways, the strongest being a functional
pass against seeded telemetry on a live Splunk Enterprise 10.2 instance
([how](../../../../examples/splunk_lineage_app/README.md#how-this-app-was-validated)).

What remains design rather than a running system: the telemetry classes for layers 3 through 7, every
detection rule in Part 3, the chained rule engine, and the determinism controls other than 7, 8, and
13. Control 9 is a partial exception, since `determinism.quantize_value` ships but the hysteresis half
of it does not. Thresholds are placeholders unless marked otherwise.

One caveat cuts across everything shipped: GPU execution mode is unexercised. Every stage declares
support for it and 198 `gpu_mode` test variants exist, but no GPU has been available to run them, so
CPU mode is the tested path.

**On the word "predictive."** Three of the four mechanisms in Part 1 are forward-looking in a defensible
sense: trajectory features over reconstruction error, forecast residual from `TimeSeriesStage`, and
ordered cross-layer precursor chains all fire before an objective is reached rather than after. The
fourth, the premise that autoencoder reconstruction error rises during reconnaissance and staging, is
a **hypothesis, not a measured result**. It is plausible and widely assumed, but this document does not
establish it, and a deployment should validate the lead time against its own historical incidents before
anyone promises prediction to a stakeholder. Absent that validation, what is described here is anomaly
detection with a forward-looking read on its trend.

**Contents**

- [Summary](#summary)
- [Part 0: Codebase Analysis](#part-0-codebase-analysis)
  - [Pass 1: Structure, Execution Model, and Type System](#pass-1-structure-execution-model-and-type-system)
  - [Pass 2: The Behavioral Analytics Substrate](#pass-2-the-behavioral-analytics-substrate)
  - [Pass 3: Determinism, Concurrency, and I/O Boundaries](#pass-3-determinism-concurrency-and-io-boundaries)
- [Part 1: Reference Architecture for Layers 1-7](#part-1-reference-architecture-for-layers-1-7)
- [Part 2: Telemetry Class Requirements](#part-2-telemetry-class-requirements)
- [Part 3: Detection Rule Recommendations](#part-3-detection-rule-recommendations)
- [Part 4: Chaining Layer 1-7 Lineage in Splunk](#part-4-chaining-layer-1-7-lineage-in-splunk)
- [Part 5: Preserving Determinism in the Output](#part-5-preserving-determinism-in-the-output)
- [Part 6: Gaps and Build List](#part-6-gaps-and-build-list)

---

## Part 0: Codebase Analysis

### Pass 1: Structure, Execution Model, and Type System

#### Package topology

The repository ships three Python packages plus a compiled core:

| Path | Contents |
| --- | --- |
| `python/morpheus/morpheus` | Core SDK: configuration, pipeline, stages, messages, modules, parsers, controllers, I/O |
| `python/morpheus/morpheus/_lib` | C++/CUDA core exposed through pybind11: messages, stages, tensors, cuDF interoperability |
| `python/morpheus/morpheus/_lib/doca` | Optional NVIDIA DOCA GPUNetIO path for line-rate packet capture directly into GPU memory |
| `python/morpheus_dfp/morpheus_dfp` | Digital fingerprinting: per-entity behavioral modeling, the piece most relevant here |
| `python/morpheus_llm/morpheus_llm` | LLM engine nodes, vector database services, retrieval-augmented enrichment |

`docs/source/developer_guide/architecture.md` describes five conceptual layers: orchestration, pipeline,
stage, module, and node. The practical consequence is that all nodes inside a stage are guaranteed to run
in the same process on the same machine, which is what permits raw device pointers to move between nodes
without serialization.

#### Configuration

`python/morpheus/morpheus/config.py` defines `Config`, a dataclass that becomes immutable once
`freeze()` is called (which happens the first time it is handed to a pipeline or stage). `__setattr__`
raises `dataclasses.FrozenInstanceError` after freezing. The fields that matter for this design:

- `execution_mode`: `ExecutionMode.GPU` or `ExecutionMode.CPU`. Freezing reconciles this with the global
  `CppConfig` toggle, raising if they disagree.
- `pipeline_batch_size` (default 256) and `model_max_batch_size` (default 8). A warning is emitted when
  the former is smaller than the latter.
- `feature_length`: second-axis dimension of the tensor handed to the model. Sequence length for NLP,
  feature count for forest inference.
- `num_threads`, `edge_buffer_size` (default 128, must be a power of two greater than one).
- `class_labels`: index-to-label mapping used by `AddClassificationsStage`.
- `ae: ConfigAutoEncoder`, which carries `userid_column_name`, `timestamp_column_name`,
  `feature_columns`, `feature_scaler`, `fallback_username` (default `generic_user`), and `use_generic_model`.
- `Config.save(filename)` and `Config.to_string()` serialize the whole object to JSON. Part 5 uses this
  as the basis for a configuration hash.

#### Pipeline construction and the MRC executor

`python/morpheus/morpheus/pipeline/pipeline.py` holds the build logic. Three lines determine the entire
runtime concurrency profile:

```python
mrc.Config.default_channel_size = self.edge_buffer_size
exec_options.topology.user_cpuset = f"0-{self._num_threads - 1}"
exec_options.engine_factories.default_engine_type = mrc.core.options.EngineType.Thread
```

Stages are built in `networkx.topological_sort` order per segment, with a fallback pass under reduced
constraints if the graph is cyclic. Segments are the unit of distribution: `add_segment_edge` connects
them through ingress and egress port pairs, which is the seam to use when splitting a seven-layer
deployment across processes or machines.

#### The stage contract

`python/morpheus/morpheus/pipeline/stage_base.py` defines what any custom stage must implement:

- `name` - the identifier used in logs and the CLI.
- `accepted_types()` - input type tuple.
- `compute_schema(schema: StageSchema)` - type inference. Output types are resolved at build time and
  the architecture document is explicit that returning a different type at runtime is undefined behavior.
- `supports_cpp_node()` and `supported_execution_modes()`.
- `_build(builder, input_nodes)` - node creation.
- `_needed_columns` - a dict of column name to `TypeId` consumed by `PreallocatorMixin`, which
  preallocates the column in the DataFrame so downstream writes do not trigger a reallocation. Any custom
  stage that adds lineage columns should declare them here.

For lighter work, `python/morpheus/morpheus/pipeline/stage_decorator.py` provides `@source` and `@stage`
decorators that wrap a plain generator or function into a full stage with inferred types.

#### Message model

`python/morpheus/morpheus/messages/control_message.py` defines `ControlMessage`, the primary in-flight
container. It carries five things:

1. `_payload: MessageMeta` - the DataFrame batch.
2. `_tensors: TensorMemory` - GPU tensors for inference input and output.
3. `_config["metadata"]` - a free-form dict, accessed through `set_metadata`/`get_metadata`/`list_metadata`.
4. `_tasks: dict[str, deque]` - work items such as `load`, `training`, `inference`, driving the
   control-message pipelines in `morpheus_dfp`.
5. `_timestamps: dict[str, datetime]` - accessed through `set_timestamp`, `get_timestamp`,
   `get_timestamps`, and `filter_timestamp(regex)`.

That fifth field is important and underused. It is a per-message, per-key timestamp map that survives the
whole pipeline, and it is the natural place to record stage-entry and stage-exit times for lineage without
polluting the DataFrame.

`python/morpheus/morpheus/messages/message_meta.py` wraps the DataFrame behind a mutex.
`mutable_dataframe()` is a context manager that acquires and releases that lock; `copy_ranges` produces a
new `MessageMeta` from selected row ranges while preserving order.

`python/morpheus/morpheus/_lib/include/morpheus/messages/raw_packet.hpp` defines `RawPacketMessage`, the
GPU-resident packet buffer produced by the DOCA source.

#### Stage inventory

Sources (`stages/input/`): `KafkaSourceStage`, `HttpServerSourceStage`, `HttpClientSourceStage`,
`FileSourceStage`, `RSSSourceStage`, `DatabricksDeltaLakeSourceStage`, `ControlMessageKafkaSourceStage`,
`ControlMessageFileSourceStage`, `AppShieldSourceStage`, `InMemorySourceStage`,
`InMemoryDataGenStage`, `ArxivSource`, and the autoencoder sources for CloudTrail, Duo, and Azure.

Preprocess (`stages/preprocess/`): `DeserializeStage`, `PreprocessNLPStage` (subword tokenization),
`PreprocessFILStage`, `DropNullStage`, `GroupByColumnStage`.

Inference (`stages/inference/`): `TritonInferenceStage`, `PyTorchInferenceStage`,
`IdentityInferenceStage`, over the shared `InferenceStage` base.

Postprocess (`stages/postprocess/`): `AddClassificationsStage`, `AddScoresStage`,
`FilterDetectionsStage`, `SerializeStage`, `ValidationStage`, `MLFlowDriftStage`, `TimeSeriesStage`,
`GenerateVizFramesStage`.

Output (`stages/output/`): `WriteToKafkaStage`, `WriteToElasticsearchStage`, `WriteToFileStage`,
`WriteToDatabricksDeltaLakeStage`, `HttpClientSinkStage`, `HttpServerSinkStage`,
`CompareDataFrameStage`, `InMemorySinkStage`.

General (`stages/general/`): `MonitorStage`, `TriggerStage`, `BufferStage`, `DelayStage`,
`RouterStage`, `MultiProcessingStage`, `LinearModulesStage`, `MultiPortModulesStage`.

Lineage (`stages/lineage/`): `LineageStampStage`, `CommunityIdStage`. These were added to support the
design in Part 4 and are covered in detail there.

There is no Splunk sink. Delivery to Splunk goes through Kafka, HTTP Event Collector via
`HttpClientSinkStage`, or a file drop consumed by a forwarder. Part 4 covers the tradeoffs.

#### Modules

Modules are the composable sub-stage unit introduced in 23.03. They are registered with
`@register_module(NAME, NAMESPACE)` from `python/morpheus/morpheus/utils/module_utils.py` and loaded with
`builder.load_module(...)`. The DFP deployment in `python/morpheus_dfp/morpheus_dfp/modules/` is the
reference example of nesting: `dfp_deployment` contains `dfp_training_pipe` and `dfp_inference_pipe`, each
of which contains `dfp_preproc`, `dfp_rolling_window`, `dfp_data_prep`, among others. Module identifiers live
in `python/morpheus_dfp/morpheus_dfp/utils/module_ids.py`.

---

### Pass 2: The Behavioral Analytics Substrate

#### The feature engineering DSL

`python/morpheus/morpheus/utils/column_info.py` is the most reusable asset in the repository for this
problem. It defines a declarative schema language that compiles to DataFrame operations:

| Class | Behavior |
| --- | --- |
| `ColumnInfo` | Pass-through with dtype enforcement |
| `RenameColumn` | Map an input column name to a canonical output name |
| `CustomColumn` | Arbitrary callable over the DataFrame |
| `BoolColumn` | Map string value sets to boolean (`true_values`, `false_values`) |
| `DateTimeColumn` | Parse to `datetime64[ns]` |
| `StringCatColumn` | Concatenate several columns with a separator |
| `StringJoinColumn` | Join a list-valued column |
| `IncrementColumn` | Per-group running count within a time period |
| `DistinctIncrementColumn` | Per-group count of **distinct** values seen so far |

`DataFrameInputSchema` bundles these with `json_columns` (nested JSON to flatten, handled by
`_json_flatten`) and `preserve_columns` (a regex of columns to carry through untouched). `process_dataframe`
applies the schema.

`IncrementColumn` and `DistinctIncrementColumn` are the behavioral primitives. `DistinctIncrementColumn`
in particular encodes "how many distinct values of this attribute has this entity used so far," which is
the novelty signal that most behavioral detections reduce to. `python/morpheus_dfp/morpheus_dfp/utils/schema_utils.py`
uses exactly this pattern for Azure and Duo:

```python
IncrementColumn(name="logcount", dtype=int,
                input_name=self._config.ae.timestamp_column_name,
                groupby_column=self._config.ae.userid_column_name),
DistinctIncrementColumn(name="locincrement", dtype=int, input_name="location"),
DistinctIncrementColumn(name="appincrement", dtype=int, input_name="appDisplayName"),
```

Note the ordering dependency: both are cumulative and therefore only well-defined given a total order on
rows. Part 5 treats this as a first-class determinism requirement.

#### Digital fingerprinting: The per-entity model pattern

DFP is the architectural template to generalize. The flow is:

```text
source -> dfp_preproc -> dfp_split_users -> dfp_rolling_window -> dfp_data_prep
       -> {dfp_training | dfp_inference} -> dfp_postprocessing -> filter -> serialize -> sink
```

`DFPSplitUsersStage` fans a mixed batch into one `ControlMessage` per entity, tagging
`message.set_metadata('user_id', ...)`.

`DFPRollingWindowStage` (`python/morpheus_dfp/morpheus_dfp/stages/dfp_rolling_window_stage.py`) holds a
`CachedUserWindow` per entity and gates emission on three parameters:

- `min_history` - suppress entities with too little data.
- `min_increment` - suppress entities that have not accumulated enough new records since the last emission.
- `max_history` - either an integer row count or a `pandas.Timedelta` string, in which case only rows
  within `[latest_timestamp - max_history, latest_timestamp]` are retained.

`python/morpheus_dfp/morpheus_dfp/utils/cached_user_window.py` adds two provenance columns on append:

```python
filtered_df["_row_hash"] = pd.util.hash_pandas_object(filtered_df, index=False)
filtered_df["_batch_id"] = self.batch_count
```

`_row_hash` is used to locate the boundary between already-seen and new rows, and `_batch_id` to select
rows newer than the last training batch. Both are excluded from the serialized output by
`dfp_inference_pipe.py` (`"exclude": ['batch_count', 'origin_hash', '_row_hash', '_batch_id']`). For a
lineage-bearing pipeline, that exclusion list should be revisited: `origin_hash` and `_row_hash` are
exactly what a SIEM needs to tie a score back to a source record.

`origin_hash` itself is set in `python/morpheus/morpheus/controllers/file_to_df_controller.py` and is a
hash of the source object set, which makes it a usable batch-level provenance token.

`DFPInferenceStage` resolves a per-entity model from MLflow through `ModelManager`/`ModelCache`
(`python/morpheus_dfp/morpheus_dfp/utils/model_cache.py`), using the format string `dfp-{user_id}` and
falling back to `fallback_user_ids=[self._fallback_user]`. It writes `model_version` into the payload as
`f"{model_cache.reg_model_name}:{model_cache.reg_model_version}"`. That single line is the seed of a
reproducibility envelope; Part 5 expands it.

#### The autoencoder and its output schema

`python/morpheus/morpheus/models/dfencoder/autoencoder.py` implements the tabular autoencoder. What
matters downstream is `get_results`, which returns a DataFrame with, for every feature `ft`:

- `ft` - the original value
- `ft_pred` - the reconstruction
- `ft_loss` - the raw reconstruction loss
- `ft_z_loss` - the loss scaled by a fitted loss scaler, optionally absolute

plus three aggregate columns:

- `max_abs_z` - maximum absolute scaled loss across features
- `mean_abs_z` - mean absolute scaled loss across features
- `z_loss_scaler_type` - `z`, `modz`, or a custom scaler label

This is a per-feature attribution vector for every scored row, not just a scalar anomaly score. It is the
difference between an alert that says "user is anomalous" and one that says "user is anomalous because
`locincrement` is 9.4 standard deviations out while every other feature is nominal." Detection rules in
Part 3 are written against these columns.

Scalers live in `python/morpheus/morpheus/models/dfencoder/scalers.py`: `StandardScaler`,
`GaussRankScaler`, `ModifiedScaler`, `NullScaler`. `preset_numerical_scaler_params` allows a fitted
scaler's attributes to be restored exactly rather than refitted, which matters for reproducibility.

#### Seasonality and the predictive angle

`python/morpheus/morpheus/stages/postprocess/timeseries_stage.py` is the one stage in core that is
genuinely predictive rather than reactive. It bins events per entity at a fixed `resolution`, computes a
periodogram over the binned signal (`to_periodogram`), filters by percentile, inverse-transforms, and
z-scores the residual (`fftAD`). Events whose residual exceeds `zscore_threshold` are flagged.

Two design details are worth copying. First, bin assignment is anchored to an absolute epoch:

```python
def calc_bin(obj, time0, resolution_sec):
    return round((round_seconds(obj) - time0).total_seconds()) // resolution_sec
```

Anchoring to a fixed `time0` rather than to the first observed event makes bin identity independent of
where a replay starts. Second, `hot_start` explicitly controls whether the stage will emit before its
window is satisfied, which is a deliberate choice between latency and stability.

#### Layer coverage already present in the repository

| Layer | What exists | Where |
| --- | --- | --- |
| 1-2 | GPU packet capture off a ConnectX NIC into device memory | `stages/doca/doca_source_stage.py`, `_lib/doca/` |
| 3-4 | Flow assembly and TCP flag features | `examples/abp_pcap_detection/abp_pcap_preprocessing.py` |
| 3 | IP arithmetic and classification | `parsers/ip.py` |
| 3-7 | Zeek log ingestion with typed headers | `parsers/zeek.py` |
| 5-7 | Authentication behavior modeling | `morpheus_dfp` with Duo and Azure schemas |
| 7 | Subword tokenization to transformer inference | `stages/preprocess/preprocess_nlp_stage.py` |
| 7 | Named entity recognition over logs | `examples/log_parsing/` |
| 7 | Sensitive information detection, phishing | `models/sid-models`, `models/phishing-models` |
| Host | Process, DLL, VAD, and handle snapshots | `stages/input/appshield_source_stage.py` |
| SIEM | Splunk notable event regex parsing | `parsers/splunk_notable_parser.py` |

The PCAP example is the clearest template for layer 3-4 feature engineering. It derives thirteen features
from raw packet fields:

```python
self.features = ["ack", "psh", "rst", "syn", "fin", "ppm", "data_len", "bpp",
                 "all", "ackpush/all", "rst/all", "syn/all", "fin/all"]
```

It builds `flow_id` as `"src_ip:src_port=dst_ip:dst_port"`, rolls timestamps into 60-second bins
(`rollup_time`), aggregates per `(rollup_time, flow_id)`, then merges back to the original row count so
that every input message retains an output. The merge-back-and-sort-by-index pattern preserves input
order, which is the right instinct.

`AppShieldSourceStage` is the host-telemetry analogue: it reads per-snapshot plugin output and stamps
`snapshot_id`, `timestamp`, `source`, and `plugin` onto every row, giving a natural composite key.

---

### Pass 3: Determinism, Concurrency, and I/O Boundaries

This pass asks a single question: if the same bytes enter the pipeline twice, what can make the output
differ?

#### Sources of nondeterminism

**Intra-stage parallelism.** `launch_options.pe_count` sets the number of processing engines for a node.
Grepping the repository:

| File | Setting |
| --- | --- |
| `stages/inference/inference_stage.py:271` | `pe_count = self._thread_count` (defaults to `config.num_threads`) |
| `stages/preprocess/preprocess_base_stage.py:67` | `pe_count = self._config.num_threads` |
| `stages/input/kafka_source_stage.py:246` | `pe_count = self._max_concurrent` |
| `stages/general/multi_processing_stage.py:162` | `pe_count = self._max_in_flight_messages` |
| `stages/general/router_stage.py:102` | `engines_per_pe = self._processing_engines` |
| `stages/doca/doca_source_stage.py:98` | `pe_count = 1` |
| `morpheus_llm/stages/llm/llm_engine_stage.py:136` | `pe_count = 1` |

Any node with `pe_count > 1` processes messages concurrently and emits them in completion order, not
arrival order. The DFP inference and preprocessing stages have their `pe_count` lines commented out,
which means they currently run single-engine. That is a fortunate default and should be treated as a
requirement, not an accident.

**Buffered edges.** `edge_buffer_size` sets `mrc.Config.default_channel_size`. Buffers do not reorder
within a single edge, but combined with multiple engines they make interleaving across parallel paths
unpredictable.

**Batch composition.** `InferenceStage._build_single` splits an incoming message into batches of
`model_max_batch_size`, dispatches each with a future, and reassembles:

```python
batches = self._split_batches(message, self._max_batch_size)
...
for f in fut_list:
    f.result()
return output_message
```

Reassembly into a single output message preserves order within a message. But batch **boundaries** depend
on how many rows arrived together, which depends on upstream timing. If the model is not batch-invariant,
identical rows can score differently depending on who they were batched with. Triton's dynamic batching
amplifies this.

**Wall-clock reads.** `DFPPostprocessingStage._process_events` writes
`df['event_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')`. This field varies between runs by
construction. `ElasticsearchController.refresh_client` uses a wall-clock refresh period, and
`DFPInferenceStage` caches models with `self._cache_timeout_sec = 600` and
`self._model_cache_size_max = 10`.

**Model resolution.** `ModelManager.load_user_model` resolves the latest registered MLflow version for
`dfp-{user_id}`, with fallback to `generic_user`. Two runs separated by a retraining event resolve
different models. The 600-second cache means even a single long run can switch models mid-stream.

**Training stochasticity.** `preprocess_training_data` defaults to `shuffle_rows_in_batch=True`, and the
training path sets `dataset.shuffle_batch_indices = True`. `GaussRankScaler` fitting depends on the
sample it sees. Torch and cuDNN kernel selection is nondeterministic unless constrained.

**Input enumeration order.** `fsspec.open_files(files)` in `modules/file_batcher.py` returns objects in
filesystem order, which is not guaranteed stable across object stores or listings.

**Sink concurrency.** `ElasticsearchController.parallel_bulk_write` uses
`elasticsearch.helpers.parallel_bulk`; `WriteToKafkaStage` produces asynchronously with a delivery
callback. Neither guarantees output ordering at the destination.

**Floating point.** GPU reductions in float32 are not associative. `AddScoresStage` and the autoencoder
loss aggregation can differ in the last bits between runs with different block scheduling. This is
irrelevant until a score sits within a few ULPs of a threshold, at which point it flips the decision.

#### What the codebase already gives you

- `python/morpheus/morpheus/utils/seed.py` provides `manual_seed(seed, cpu_only=False)`, seeding
  `random`, NumPy, Torch, CuPy, and all CUDA devices, and setting
  `torch.backends.cudnn.deterministic = True` with `benchmark = False`.
- `_row_hash` (`pd.util.hash_pandas_object`) and `origin_hash` give row-level and batch-level provenance.
- `_batch_id` gives a monotonic per-entity batch ordinal.
- `ControlMessage.set_timestamp`/`filter_timestamp` gives per-message stage timing without touching the payload.
- `Config.freeze()` guarantees configuration immutability once the pipeline is built; `Config.save()`
  serializes it for hashing.
- `FilterDetectionsController.filter_copy` uses `meta.copy_ranges(true_pairs)`, preserving relative order
  of surviving rows. `filter_slice` splits into multiple messages instead, which changes downstream
  batching; prefer `copy`.
- `compute_schema` fixes output types at build time, so column presence and dtype are static.
- `CompareDataFrameStage` and `ValidationStage` exist specifically for golden-file regression, and are
  the foundation of the replay harness in Part 5.

#### The honest summary

Morpheus is a high-throughput streaming inference framework with a strong per-entity behavioral modeling
subsystem and good provenance primitives that are currently used only internally. It is not, out of the
box, a deterministic system. Determinism is achievable, but it is a property you construct through
configuration discipline and a small number of custom stages, not one you inherit.

---

## Part 1: Reference Architecture for Layers 1-7

### Shape of the deployment

Three tiers:

```text
Tier 1  Collectors            Tier 2  Morpheus                  Tier 3  SIEM
------------------            ---------------------------       ---------------
L1/L2 SNMP, LLDP, NAC   -->   segment_l12  ---\
L3/L4 NetFlow, DOCA     -->   segment_l34  ----\
L5   auth, VPN, RDP     -->   segment_l5   -----+--> segment_fuse --> Kafka --> Splunk
L6   TLS, JA4, certs    -->   segment_l6   ----/                       |
L7   HTTP, DNS, SaaS    -->   segment_l7   ---/                        +--> graph store
Identity/asset context  -->   (enrichment lookups)                     +--> object store
```

Tier 1 is not Morpheus. Collectors normalize into a common envelope (Part 2) and publish to one Kafka
topic per telemetry class. Morpheus consumes with `KafkaSourceStage`.

Tier 2 is a Morpheus pipeline with one segment per layer group plus a fusion segment. Use
`add_segment_edge` rather than a single flat linear pipeline: segments are the natural isolation boundary,
they let each layer run at its own parallelism, and they let you restart one layer's processing without
disturbing the others.

Tier 3 receives two distinct streams: scored events, and lineage edges. Keeping them separate matters
because they have different volumes, different retention needs, and different acceleration strategies.

### Per-layer pipeline pattern

Every layer segment follows the same seven-step shape, which is the DFP pattern generalized beyond users:

1. **Ingest** - `KafkaSourceStage` or `HttpServerSourceStage`.
2. **Normalize** - a `DataFrameInputSchema` built from `column_info` primitives, applied through
   `process_dataframe`. This is where `json_columns` flattening and dtype enforcement happen.
3. **Key and stamp** - a custom stage that computes the entity key, the deterministic `event_uid`, and
   the lineage identifiers (Part 4). Declare these in `_needed_columns`.
4. **Window** - `DFPRollingWindowStage` or an analogue, keyed on the layer's entity, with `max_history`
   expressed as a duration.
5. **Feature derivation** - a second `DataFrameInputSchema` using `IncrementColumn` and
   `DistinctIncrementColumn` for novelty, plus `CustomColumn` for layer-specific ratios.
6. **Score** - `DFPInferenceStage` against a per-entity autoencoder, `TritonInferenceStage` for a shared
   model, or `TimeSeriesStage` for periodicity. Layers with high entity cardinality and low per-entity
   volume should use a shared model with the entity as a categorical feature rather than one model per entity.
7. **Emit** - `FilterDetectionsStage` (copy mode), `SerializeStage`, then `WriteToKafkaStage`.

### Where "predictive" actually comes from

Behavioral analytics as usually deployed is retrospective: it scores an event that already happened.
Four mechanisms in this architecture make it forward-looking, in increasing order of ambition:

1. **Reconstruction error as a leading indicator.** An autoencoder trained on an entity's normal behavior
   starts producing elevated `mean_abs_z` during the reconnaissance and staging phases of an intrusion,
   before the action-on-objective. The signal exists earlier than any signature fires. Capturing it
   requires alerting on the *trajectory* of `mean_abs_z`, not a single crossing.
2. **Velocity and acceleration features.** For each entity, maintain the first and second differences of
   `mean_abs_z` across consecutive windows. A large positive second difference is drift becoming a spike.
   These are `CustomColumn` computations over the rolling window.
3. **Forecast residual.** `TimeSeriesStage` already computes the difference between an entity's observed
   binned activity and the reconstruction of its dominant periodic components. Feeding the forecast
   itself, not just the residual flag, into the SIEM lets you alert on "this entity is projected to exceed
   its envelope in the next window."
4. **Cross-layer precursor chains.** The highest-value predictive signal is ordinal: an entity that shows
   a layer 3 scanning pattern, then a layer 5 authentication anomaly, then a layer 7 data access anomaly,
   within a bounded interval. No single layer's score need cross a threshold for the chain to be alarming.
   This requires the lineage substrate in Part 4 and is where most of the detection value sits.

---

## Part 2: Telemetry Class Requirements

### The universal envelope

Every record from every layer must carry the same fourteen fields before it reaches Morpheus. Collectors
that cannot produce them must be fixed at the collector, not patched in the pipeline; patching in the
pipeline is how lineage silently breaks.

| Field | Type | Requirement |
| --- | --- | --- |
| `schema_version` | string | Semantic version of the telemetry class schema. Changing a field's meaning requires a major bump. |
| `telemetry_class` | string | `TC-1` through `TC-7`, or `TC-0` for identity and asset context. |
| `collector_id` | string | Stable identifier for the collecting agent or sensor. |
| `collector_seq` | uint64 | Strictly monotonic sequence per `collector_id`. This is the tiebreaker that makes ordering total. |
| `observed_time` | timestamp, UTC, nanoseconds | When the sensor saw the event. |
| `event_time` | RFC 3339 UTC string on the wire, nanoseconds in the pipeline | When the event occurred, if the source reports it. Defaults to `observed_time`. The two representations are one field at different points; see [The event time wire format](#the-event-time-wire-format). |
| `ingest_time` | timestamp, UTC | Set at the Morpheus source. Never used in a detection rule. |
| `clock_source` | enum | `ptp`, `ntp_disciplined`, `ntp_undisciplined`, `unsynchronized`. |
| `clock_offset_ms` | int | Last measured offset. Records above a configured bound are quarantined, not dropped. |
| `sampling_policy` | string | `full`, `1:N`, `adaptive:<params>`. Required for any rate-based feature to be interpretable. |
| `origin_hash` | hex string | Hash of the source object or batch. Already produced by `file_to_df_controller.py`. |
| `event_uid` | hex string | Deterministic per-event identity. Construction in Part 4. |
| `entity_key` | string | The primary behavioral subject for this telemetry class. |
| `site_id` / `tenant_id` | string | Physical and logical scoping. |

Two rules govern the envelope. First, `ingest_time` must never appear in a detection rule or a feature.
Second, when a field is unavailable it must be explicitly null with a reason code, never defaulted to a
plausible value; a defaulted value is indistinguishable from an observed one three months later during an
investigation.

### TC-1: Physical

**Entity key:** `site_id:device_id:port_id`

**Sources:** SNMP interface tables, optical transceiver diagnostics, LLDP and CDP neighbor tables,
structured cabling and patch-panel inventory, out-of-band management controllers, and for physical
security, badge readers and rack door sensors.

**Required fields:** `port_id`, `admin_status`, `oper_status`, `link_speed_bps`, `duplex`,
`transceiver_serial`, `transceiver_type`, `optical_tx_dbm`, `optical_rx_dbm`, `crc_error_delta`,
`symbol_error_delta`, `input_discards_delta`, `output_discards_delta`, `last_change_time`,
`lldp_neighbor_chassis_id`, `lldp_neighbor_port_id`, `poe_draw_watts`.

Counters must be reported as **deltas with an explicit interval**, not raw values, because raw counters
wrap and reset silently on device reboot. A `counter_reset` flag on each record removes ambiguity.

{py:class}`~morpheus.stages.telemetry.tc1_normalize_stage.TC1NormalizeStage` performs that conversion in
the pipeline, so the collector stays a stateless poller. Differencing needs the previous sample, which
makes it stateful, and state at the collector is state to be replicated, aged, and lost on restart.
Placed here it also inherits the pipeline's determinism controls: the stage is single-engine and shards
by device (control 4), and it processes rows in the order given, flagging a sample that arrives out of
order rather than reporting the negative delta a naive subtraction would produce. A wrap and a reboot are
indistinguishable in the counter alone, so `sysUpTime` is what separates them; without it the stage emits
no delta rather than guessing, because either guess would read like a measurement.

**Behavioral features:** `DistinctIncrementColumn` over `transceiver_serial` per port catches hardware
substitution. `DistinctIncrementColumn` over `lldp_neighbor_chassis_id` catches topology change. Optical
power deviation from a per-port rolling baseline catches both degradation and physical tapping. Link
flap count per interval catches instability that often precedes a layer 2 loop.

The two novelty features ship as
{py:class}`~morpheus.stages.telemetry.tc1_feature_stage.TC1FeatureStage`, over the schema in
{py:mod}`~morpheus.utils.tc1_features`. Both are cumulative and therefore order-dependent, so the stage
imposes control 8's total order before counting rather than trusting the caller to have sorted; that is
the whole reason the derivation is a stage and not a bare schema.

A period-bucketed count resets at each boundary, so a change from the last poll before one to the first
poll after it reads as one distinct value on each side and is not detected. What decides whether that
matters is the period against the span of a frame, not anything about the estate: a boundary can only
hide a change when it falls inside the frame being counted. Set `period` longer than one frame and there
is no boundary for a change to hide behind. The default is monthly, which clears a daily window with
room to spare, where the daily bucketing the primitive defaults to would put a boundary inside every
window. Lengthening the period makes boundaries rarer but never removes them, so the stage warns when it
sees a frame that straddles one rather than leaving the residual exposure to be assumed away. The cost of
going longer still is drift: the count never decays within its period, so under `period="Q"` a port that
legitimately changed optics reads above 1 for the rest of the quarter and a rule with a threshold of
"greater than 1" fires for all of it. Closing the boundary entirely, rather than narrowing it, needs the last
value carried across periods per entity, the way {py:mod}`~morpheus.utils.counter_delta` carries counter
state, which is a different primitive from `DistinctIncrementColumn`.

Optical power deviation ships as {py:class}`~morpheus.stages.telemetry.tc1_optical_stage.TC1OpticalStage`
over {py:mod}`~morpheus.utils.optical_baseline`. The baseline is the median of the port's own prior
readings inside a trailing window, which is the only reference that generalizes: an absolute threshold
loose enough not to alarm on a long span is loose enough to miss a tap on a short one. The median rather
than the mean because optical diagnostics report an occasional wild value, and prior readings rather
than all of them because a sample included in its own baseline damps the very step it should expose.

{py:class}`~morpheus.stages.telemetry.tc1_change_stage.TC1ChangeStage`, over
{py:mod}`~morpheus.utils.value_novelty`, is the answer to the period boundary above, and it is the one
to reach for when the question is whether an identifier changed. It holds the previous value per port
and compares, so there is no bucket for a change to fall across: a substitution is detected the moment
the new value arrives, whether the samples are a minute or a year apart. It reports two things
separately, because they are separately actionable. `<name>_changed` is whether the value differs from
the previous sample, which is the alerting signal. `<name>_first_seen` is whether the port has ever
reported that value before, which distinguishes an optic that has never been in this cage from one
rotated back in after maintenance. Both are null on a port's first sample, which establishes what
normal looks like rather than being an event, so a rule matching `== True` never fires on a port's
first appearance. Returning to a previous value counts as a change, deliberately: A to B to A is two
substitutions, and swapped out and swapped back is a more interesting sequence than simply changed.

The period-bucketed counts remain useful as a measure of how much churn a port saw inside a period, but
for change detection they are superseded. `changed` has no blind spot at all; the only bound left is on
`first_seen`, whose recall set is finite, so a value evicted after many others reads as first seen
again. That errs toward over-reporting novelty, which is the safe direction for a signal meant to
surface the unexplained.
Note that the baseline follows the link, so a step is a transient signal: once the window has rolled
past the last pre-step reading the deviation returns to zero, and a degradation slower than the window
is invisible because the baseline drifts down with it. Catching that needs a commissioning value to
compare against, which is asset context and belongs in TC-0.

Link flap counting ships as {py:class}`~morpheus.stages.telemetry.tc1_flap_stage.TC1FlapStage` over
{py:mod}`~morpheus.utils.link_flap`, and the reason it is not a status comparison is worth stating: a
port that drops and recovers inside one sixty-second polling gap shows the same `oper_status` at both
polls, so a diff calls the flapping port stable. `last_change_time` is what closes that, since the
device records the transition even though no poll saw it. Every count is consequently a lower bound,
because the device retains only the most recent transition and a port that flapped nine times between
polls still reports two. A floor is the right shape for this signal: an under-counted flapping port is
still flagged, whereas an interpolated estimate would put a number nobody measured in front of an
analyst. Devices report the field relative to their own uptime, so a Tier 1 collector has to normalize
it to an absolute time; a value that goes backwards is read as a device restart, which is counted as a
transition and labelled so a planned reboot can be excluded by rule rather than silently inflating the
count.

**Cadence:** 30 to 60 seconds for counters, event-driven for state transitions.
**Cardinality:** thousands to low tens of thousands of ports. Low enough for per-port models.
**Retention:** 13 months. Physical changes are investigated long after the fact.

**Honest note:** layer 1 rarely produces a standalone detection worth paging on. Its value is almost
entirely as a lineage anchor: it is the only layer that ties a MAC address to a physical location. Build
it for that, and treat any detection value as a bonus.

### TC-2: Data Link

**Entity key:** `mac_address` and, separately, `site_id:switch_id:port_id:vlan_id`

**Sources:** switch MAC address tables, 802.1X and RADIUS accounting, ARP tables, DHCP server leases,
wireless controller association logs, spanning-tree topology change notifications, NAC posture decisions.

**Required fields:** `mac_address`, `oui`, `vlan_id`, `switch_id`, `port_id`, `bind_start`, `bind_end`,
`dot1x_identity`, `dot1x_result`, `eap_method`, `auth_vlan_assigned`, `wireless_ssid`, `wireless_bssid`,
`wireless_rssi`, `stp_topology_change_count`, `arp_sender_ip`, `arp_sender_mac`, `arp_target_ip`,
`arp_operation`.

**Behavioral features:** count of distinct MAC addresses per port over a window catches unauthorized
hubs and switches. Count of distinct ports per MAC catches spoofing or a device physically moving.
Ratio of gratuitous ARP replies to total ARP catches poisoning. Time-to-authorize distribution per port
catches 802.1X bypass attempts. OUI novelty per VLAN catches unmanaged device introduction.

The three cardinality features among those ship as
{py:class}`~morpheus.stages.telemetry.tc2_cardinality_stage.TC2CardinalityStage` over
{py:mod}`~morpheus.utils.distinct_window`, which is one primitive asked with the entity and the value
swapped around. Two details are worth knowing before writing a rule against them. The current sample is
counted inside its own window, so a threshold trips on the row that crosses it rather than the row
after. And each entity has a sample cap, because a MAC flood is simultaneously the condition the
per-port count exists to notice and the condition that would exhaust memory; when the cap binds, the
count becomes a lower bound and the row is marked saturated, so a floor is never mistaken for the
figure. The per-port counts key on `site_id:switch_id:port_id`, since an interface name alone repeats on
every switch in the estate.

Note that these three do not shard alike. Counts per port and per VLAN shard cleanly by switch, but
distinct ports per MAC needs every sighting of a MAC to reach one instance, which sharding by switch
breaks; run it unsharded or shard it by MAC.

The remaining two ship as {py:class}`~morpheus.stages.telemetry.tc2_arp_stage.TC2ArpStage` and
{py:class}`~morpheus.stages.telemetry.tc2_auth_stage.TC2AuthStage`.

ARP is scored as a *proportion* rather than a count ({py:mod}`~morpheus.utils.ratio_window`), because a
count of gratuitous replies mostly measures how chatty a host is. The share of one sender's ARP that is
gratuitous does not, and it is what separates a device announcing itself after a failover from one
flooding announcements to overwrite a neighbor's cache. No ratio is published until the window holds
`min_denominator` events, since one gratuitous packet out of one reads as 1.0 and means nothing. The
same stage computes the distinct MACs claiming each sender address, which is the condition R-D-L2-003
names; it marks rows whose sender is in the HSRP and VRRP exclusion list rather than dropping them, so
the exclusion is visible to the rule rather than silently applied.

**A correction to the field list above:** it previously omitted `arp_target_ip`. A gratuitous ARP is
defined by the sender and target protocol addresses being equal, so the feature this section asks for
was not computable from the fields it required. The field has been added.

Authorization timing ({py:mod}`~morpheus.utils.session_timer`) pairs each exchange's start with its
outcome per port. Both tails of the distribution mean something: slow is a supplicant retrying, a RADIUS
server under load, or credentials being guessed, and very fast can be a replayed success. The most
useful case is neither tail, though. An outcome arriving with *no exchange in front of it* is what a
bypass looks like from the switch, since MAC authentication bypass and a device bridged behind an
already authorized supplicant both produce authorization without anybody authenticating; that row is
flagged `auth_unpaired` rather than left as a null elapsed time, which would read as missing data
instead of as an event. The attempt count travels with the timing, because a success after three
retries is not a first-time one and timing from the last attempt alone would hide the two before it.

**Cadence:** event-driven, with a periodic full table snapshot every 5 minutes for reconciliation.
**Cardinality:** tens of thousands to low hundreds of thousands of MAC addresses.
**Retention:** 13 months for bindings, 90 days for raw ARP.

**Critical requirement:** every binding record must carry both `bind_start` and `bind_end`. A binding
without an end time cannot be used in a time-bounded join, which makes layer 2 unusable as a lineage hop.
Emit an explicit end record on expiry rather than relying on the next binding's start.

{py:class}`~morpheus.stages.telemetry.tc2_binding_stage.TC2BindingStage`, over
{py:mod}`~morpheus.utils.binding_closer`, is where that end comes from, because nothing upstream supplies
it: a switch MAC table reports what is bound now, accounting stops go missing, and releases are advisory.
Five things end a binding and only the first is a fact. An **explicit** stop states the end. A
**displacement** means the key was seen elsewhere later, so the binding ended somewhere between the two
sightings. A **conflict** means the key was seen elsewhere *at the same instant*, so neither sighting
precedes the other and the two bindings overlap by one tick; that is what a spoofed or duplicated MAC
looks like from the switches, and it gets its own reason so R-D-L2 rules can find it instead of reading
it as a data quality warning. A **snapshot absence** means a reconciliation pass over a scope no longer
lists the key. An **idle timeout** is the backstop for the stop record that never arrived. Every emitted
record carries `bind_end_reason`, and `bind_end_observed` is true only for the first, so a rule that
will act on a binding can insist on an end somebody actually reported.

The binding target defaults to this class's own entity key, `site_id`, `switch_id`, `port_id` and
`vlan_id`, and every closed binding also carries `port_key` as `site_id:switch_id:port_id`. That string
is byte for byte the layer 1 `entity_key` that `TC1NormalizeStage` writes for the same port, which is
the join the ladder's first arrow depends on; `switch_id` here and `device_id` at layer 1 are one
identifier under two names, and nothing renames anything. Every telemetry stage composes its keys with
{py:mod}`~morpheus.utils.entity_key`, and a key with a missing part is null rather than a string with
`None` in it, so a collector that omits the site does not pool every port with no site under one
fabricated site. Rows with a null key pass through with their per-entity features null, and the stage logs how many.

Inferred ends are placed at the earliest time consistent with the observations rather than the latest,
which leaves gaps between consecutive bindings. That is the intended behavior: a gap resolves to nothing
and tells an analyst the answer is unknown, whereas stretching a binding to meet the next one has it
cover a period the device may already have left and returns a confident wrong answer. Because the
interval is half-open, an inferred end sits one tick past the last observation, which is the shortest
interval that actually contains what was seen; without that a key seen once would produce a zero-width
binding covering nothing at all.

Closed bindings are the honest unit for replay, and they have a cost for live work: a device plugged in
now is not resolvable until its binding closes, which by default is up to the idle timeout of "unknown"
during an incident. `emit_open_bindings` addresses that without compromising the record. The moment a
binding opens the stage emits one provisional record, `bind_provisional = true`, `bind_end_reason =
open`, with a null end; the consumer building a live table caps that open interval with an explicit
assumed duration (`open_end_duration_ns`, the source's own aging interval is the right value), and the
closed record that follows, carrying the same key and `bind_start`, supersedes it. One record per
binding rather than per sample, so a stable estate emits one row per device and then silence. The
stage never invents the end itself: a null end fails to load without a stated duration, which is the
guide's own rule that a soft join against an unbounded interval is a guess.

### TC-3: Network

**Entity key:** `src_ip`, and separately the directed pair `src_ip:dst_ip`

**Sources:** NetFlow v9, IPFIX, sFlow, VPC and cloud flow logs, firewall session logs, router ACL logs,
and for full fidelity, the DOCA GPUNetIO path.

**Required fields:** `src_ip`, `dst_ip`, `ip_version`, `protocol`, `ip_ttl`, `ip_id`, `tos_dscp`,
`packet_count`, `byte_count`, `flow_start`, `flow_end`, `tcp_flags_union`, `direction`,
`ingress_interface`, `egress_interface`, `next_hop`, `bgp_as_src`, `bgp_as_dst`, `vrf`,
`fragmentation_flags`, `icmp_type`, `icmp_code`.

**Behavioral features:** `parsers/ip.py` provides the classification primitives directly. Derive:
fan-out (distinct `dst_ip` per `src_ip` per window), fan-in, distinct destination port count, ratio of
`is_private` to `is_global` destinations, first-time-seen destination ASN via `DistinctIncrementColumn`,
byte-count asymmetry (`bytes_out / (bytes_in + 1)`), TTL variance per source (fingerprints an
interposed device), and inter-flow arrival regularity (a low coefficient of variation is the beaconing
signal).

**Cadence:** continuous. Flow records on the shorter of active timeout (60s) or natural flow end.
**Cardinality:** millions of addresses, billions of flows per day at enterprise scale. Model at the
subnet or asset-group level, not per IP, except for servers and named assets.
**Retention:** 90 days hot, 13 months in an object store.

**Sampling is the trap.** A 1:1000 sampled flow feed cannot support fan-out counting; the count is a
sample statistic with enormous variance at low true counts. Either collect unsampled at the aggregation
layer, or restrict layer 3 behavioral features to volume ratios that survive sampling, and say so
explicitly in the rule documentation.

### TC-4: Transport

**Entity key:** `flow_id` = `src_ip:src_port=dst_ip:dst_port` (the format already used in
`abp_pcap_preprocessing.py`), plus `community_id` for cross-tool joins.

**Sources:** the same as TC-3 with per-packet detail, plus Zeek `conn.log`, plus the DOCA path for
line-rate capture.

**Required fields:** `src_port`, `dst_port`, `tcp_seq_initial`, `tcp_window_initial`, `tcp_options_order`,
`tcp_mss`, `retransmit_count`, `out_of_order_count`, `zero_window_count`, `rtt_estimate_ms`,
`handshake_duration_ms`, `connection_state`, `close_reason`, and the flag counters
`ack`, `psh`, `rst`, `syn`, `fin`.

**Behavioral features:** the thirteen features in `abp_pcap_preprocessing.py` are a validated starting
set and should be adopted as-is: the five flag sums, `ppm`, `data_len`, `bpp`, `all`, and the four ratios
`ackpush/all`, `rst/all`, `syn/all`, `fin/all`. Add: `syn` without matching `ack` per source (scanning),
`rst` ratio per destination (closed-port probing), handshake-duration distribution shift, and TCP options
ordering as a stack fingerprint, since an options order inconsistent with the claimed layer 7 user agent
is a strong proxy or tunneling indicator.

**Cadence:** continuous, aggregated into fixed bins. The PCAP example uses 60-second bins anchored with
an explicit rounding kernel; reuse that anchoring approach.
**Cardinality:** very high. Aggregate to `(rollup_time, flow_id)` before modeling, exactly as the example does.
**Retention:** 30 days for per-flow, 13 months for aggregates.

### TC-5: Session

**Entity key:** `user_principal`, plus `session_id`

**Sources:** identity providers (Entra ID, Okta), Kerberos KDC logs, RADIUS accounting, VPN
concentrators, RDP and SSH session logs, privileged access management, Duo and other MFA providers,
Windows security event logs.

**Required fields:** `session_id`, `user_principal`, `user_sid`, `auth_method`, `auth_result`,
`failure_reason`, `mfa_used`, `mfa_factor`, `mfa_result`, `source_ip`, `device_id`, `device_compliant`,
`session_start`, `session_end`, `session_duration_s`, `privilege_level`, `impersonated_principal`,
`conditional_access_result`, `token_lifetime_s`, `token_type`, `refresh_count`.

**Behavioral features:** this is where `morpheus_dfp` applies directly, and the Duo and Azure schemas in
`schema_utils.py` are the template. The proven set is `logcount` (`IncrementColumn` over timestamp grouped
by user), `locincrement` (`DistinctIncrementColumn` over a concatenated location), and `appincrement`.
Extend with: distinct source ASN per user per window, distinct device per user, hour-of-day and
day-of-week deviation from the user's own histogram, MFA-to-total-auth ratio, failure-then-success
sequences within a short window, and privilege escalation events per session.

**Cadence:** event-driven.
**Cardinality:** tens of thousands of users. This is the sweet spot for per-entity models; the DFP
`dfp-{user_id}` pattern works directly.
**Retention:** 13 months minimum, often longer for compliance.

**Requirement that is routinely missed:** `session_id` must be propagated into layer 7 application logs.
Without it, the session-to-request lineage hop degrades from an exact join to a time-and-IP heuristic,
which is where most cross-layer correlation projects quietly fail.

### TC-6: Presentation

**Entity key:** `ja4_client` fingerprint, plus `certificate_fingerprint_sha256`

**Sources:** TLS inspection points, Zeek `ssl.log` and `x509.log`, load balancer and reverse proxy logs,
certificate transparency feeds, and any decryption proxy.

**Required fields:** `tls_version`, `cipher_suite`, `ja4_client`, `ja4s_server`, `ja3_client` (legacy,
for compatibility with existing content), `sni`, `alpn`, `certificate_subject`, `certificate_issuer`,
`certificate_fingerprint_sha256`, `certificate_not_before`, `certificate_not_after`,
`certificate_chain_depth`, `validation_result`, `ocsp_status`, `session_resumed`, `early_data_used`,
`content_encoding`, `content_type_declared`, `content_type_detected`.

**Behavioral features:** `DistinctIncrementColumn` over `ja4_client` per source IP catches a new TLS
stack appearing on a known host, which is one of the highest-signal, lowest-volume detections available.
Certificate issuer novelty per destination catches interception. The mismatch between
`content_type_declared` and `content_type_detected` catches tunneling. Cipher suite downgrade relative to
the pair's own history catches active attack. Self-signed certificate rate per source, and
`certificate_not_after - certificate_not_before` distribution, catch attacker-generated infrastructure.

**Cadence:** per-connection.
**Cardinality:** thousands of distinct JA4 fingerprints in a typical enterprise, which is very tractable.
**Retention:** 13 months for fingerprints, 90 days for full handshake detail.

**Note:** layer 6 is where the OSI model fits real networks worst, and it is worth being honest about
that in the design rather than forcing it. What is being modeled here is the encoding and cryptographic
negotiation surface. That is genuinely useful and genuinely distinct from both layer 5 and layer 7, so
the class earns its place even though the mapping is loose.

### TC-7: Application

**Entity key:** varies by sub-class. `user_principal` for SaaS, `service_account` for API, `hostname`
for DNS, `process_guid` for endpoint.

**Sources:** HTTP proxies and WAFs, DNS resolvers, SaaS audit APIs (Microsoft 365, Salesforce,
Workday), database audit logs, API gateways, email security gateways, EDR process telemetry, and the
AppShield-style host snapshots supported by `AppShieldSourceStage`.

**Required fields:** for HTTP: `http_method`, `url_path`, `url_query`, `user_agent`, `referrer`,
`status_code`, `request_bytes`, `response_bytes`, `duration_ms`, `trace_id`, `span_id`. For DNS:
`query_name`, `query_type`, `response_code`, `answer_count`, `answer_ttl`, `resolved_ips`. For SaaS:
`operation`, `target_object`, `target_object_type`, `record_count`, `result`, `client_app`. For
endpoint: `process_guid`, `parent_process_guid`, `image_path`, `command_line_hash`, `integrity_level`,
`signature_status`.

**Behavioral features:** DNS query name entropy and label-length distribution for tunneling and domain
generation algorithms. `DistinctIncrementColumn` over `target_object_type` per principal for data access
novelty. Record-count-per-operation deviation from the principal's baseline for bulk extraction. Ratio of
`4xx`/`5xx` to `2xx` per client for enumeration. User agent novelty per principal. Process ancestry
novelty using the parent-child pair as a categorical. The `models/` directory ships pretrained models
usable here: `sid-models` for sensitive information detection, `phishing-models`, `log-parsing-models`
for NER over unstructured logs, and `ransomware-models` for AppShield-style host telemetry.

**Cadence:** continuous, very high volume.
**Cardinality:** highest of any layer.
**Retention:** 90 days hot, 13 months cold, with sensitive-data findings retained longer.

### TC-0: Identity and Asset Context

Not an OSI layer, but without it the layers cannot be joined into anything meaningful.

**Contents:** user-to-manager and user-to-department mappings, employment status and start and end dates,
group and role membership with effective dates, asset inventory with owner and criticality and data
classification, service account to owning-team mapping, and the CMDB application-to-server mapping.

**Requirement:** every record must be **bitemporal**, carrying both the valid-time interval (when the
fact was true in the world) and the transaction-time interval (when the system learned it). Point-in-time
correctness in an investigation depends on it: "was this user in the Finance group on March 3rd" is a
different question from "is this user in the Finance group," and a non-bitemporal CMDB can only answer the
second.

**Cadence:** daily full snapshot plus event-driven deltas.
**Retention:** indefinite. This is the smallest and most valuable dataset in the entire architecture.

---

## Part 3: Detection Rule Recommendations

### Rule taxonomy

Organize rules into four families and hold them to different standards:

| Family | Basis | Precision expectation | Response |
| --- | --- | --- | --- |
| **R-D** Deterministic | Exact conditions on observed fields | Very high | Alert directly |
| **R-B** Behavioral | Autoencoder `z_loss` columns | Moderate | Risk contribution, alert only on strong signal |
| **R-P** Predictive | Trajectory and forecast residual | Low individually | Risk contribution and watchlist placement |
| **R-C** Chained | Ordered multi-layer sequence | High when the chain is specific | Alert directly, high priority |

The mistake to avoid is treating R-B output as R-D output. An autoencoder score is a statement about
how unusual something is, not about whether it is malicious. Rules should say so in their metadata,
and the SOC's runbook should reflect it.

### On the numbers in this section

The thresholds below are of two kinds, and the difference matters.

A few are **conventional**: the 900 km/h impossible-travel speed and the `4xx`-to-`2xx` enumeration
ratio are widely used starting points that transfer between environments. Most of the rest are
**placeholders**. Values such as `max_abs_z >= 6.0`, DNS entropy above 4.0 bits per character, or a
beaconing coefficient of variation below 0.15 are stated to make each rule concrete and testable, not
because they are correct for any particular estate. They are the right *shape*; the values must be
derived from a baseline period in your own data before the rule goes live.

Treat a threshold copied from this document straight into production as an untuned rule, and expect it
to behave accordingly. The `hysteresis` field exists partly so that the first tuning pass does not
have to be perfect.

### Rule specification format

Every rule carries the same metadata block, regardless of which SIEM executes it:

```yaml
rule_id: R-B-L5-001
family: behavioral
telemetry_class: TC-5
osi_layer: 5
entity_key: user_principal
inputs: [mean_abs_z, max_abs_z, locincrement_z_loss, model_version, config_hash]
window: PT1H
threshold: {metric: max_abs_z, operator: ">=", value: 6.0, hysteresis: 0.5}
suppression: {key: user_principal, duration: PT6H}
determinism_tier: D1
attack_mapping: [T1078, T1078.004]
false_positive_notes: >
  Fires on first day back from extended leave and on legitimate relocation.
  Suppress with an HR leave-status lookup, not with a threshold increase.
```

`determinism_tier` and `hysteresis` are the unusual entries. Both are explained in Part 5.

### Layer 1 and 2

**R-D-L1-001 - Transceiver substitution.** `transceiver_serial` changes on a port whose `oper_status`
did not transition to down. Near-zero false positive rate outside of maintenance windows. Suppress by
change ticket, not by threshold.

**R-D-L2-001 - MAC address count exceeded on an access port.** More than one non-voice MAC observed on
a port designated as single-host. Classic unauthorized-switch detection. Ships as a saved search over
`macs_per_port` from `TC2CardinalityStage`, firing once per MAC new to the window, against a
`port_designations` lookup the app ships header-only: a designation list is a fact about one estate, so
until the inventory populates it the rule fires on nothing.

**R-B-L2-002 - Port-to-MAC binding novelty.** `DistinctIncrementColumn` over `mac_address` grouped by
`port_id` produces a step change relative to the port's 30-day baseline. Catches the same condition as
R-D-L2-001 without requiring an accurate port designation database, at the cost of precision.

**R-D-L2-003 - ARP anomaly.** A `arp_sender_ip` maps to more than one `arp_sender_mac` within a
5-minute window, excluding known HSRP and VRRP virtual addresses. Explicitly maintain the exclusion list;
this rule is unusable without it. `TC2ArpStage` emits the count as `macs_claiming_sender_ip` and marks
excluded senders rather than dropping them, so the exclusion list lives once, in pipeline configuration,
and the saved search that ships for this rule reads the mark rather than holding a copy. The harness
corpus carries a VRRP pair that legitimately shares an address, on the list, so the exclusion path is
exercised rather than assumed.

**R-D-L2-004 - MAC in two places at once.** A closed binding with `bind_end_reason = conflict`: the
same MAC was reported on two ports at the same instant, so `TC2BindingStage` closed the earlier binding
one tick past the sighting and the two intervals overlap. This is the direct form of the spoofing signal
that R-B-L2-002 approximates statistically, and it needs no designation list and no baseline. Tier D1.
Ships as a saved search in the Splunk app, reading the raw closed-binding records.

**R-D-L2-005 - Authorization without authentication.** `auth_unpaired = true` from `TC2AuthStage`: an
802.1X outcome arrived on a port with no exchange in front of it. From the switch this is what MAC
authentication bypass looks like, and also what a device bridged behind an already authorized
supplicant looks like. Near-zero false positives where every port runs 802.1X; on ports where MAB is
configured deliberately, suppress by port designation rather than by loosening the rule. Tier D1. Ships
as a saved search in the Splunk app.

These four, R-D-L2-001, 003, 004 and 005, are the rules in this part that exist as code rather than as
specification. 004 and 005 read columns the shipped stages produce and depend on nothing outside the
pipeline; 001 and 003 depend on a list the estate owns, and each ships with the hook for that list and
fires on nothing until it is populated. All four predicates are asserted in Python over the determinism
harness's planted corpus: 004 and 005 fire exactly once, 001 once per offending MAC, and 003 on the
flooded gateway and not on the redundancy pair.

**R-P-L1-004 - Optical degradation forecast.** Linear extrapolation of `optical_rx_dbm` per port
projects a crossing of the transceiver's minimum receive threshold within 14 days. This is an operations
rule, not a security rule, but it costs nothing once the telemetry class exists and it earns the layer 1
pipeline its budget.

### Layer 3

**R-B-L3-001 - Fan-out expansion.** `DistinctIncrementColumn` over `dst_ip` grouped by `src_ip`,
compared against the source's own 14-day distribution. Alert when the current window exceeds the 99.5th
percentile **and** the destinations are predominantly internal **and** the source is not on the scanner
allowlist. All three conditions are required; the first alone produces unusable volume.

**R-B-L3-002 - Beaconing.** Coefficient of variation of inter-flow arrival time for a
`(src_ip, dst_ip)` pair below 0.15 over at least 12 intervals, with per-flow byte counts in a narrow band.
Feed the binned series to `TimeSeriesStage`; the periodogram approach in `fftAD` detects this directly
and more robustly than a variance threshold, because it survives jitter.

**R-D-L3-003 - Reserved-range egress.** Traffic to `is_reserved` or `is_multicast` destinations
crossing an internet egress point, using `parsers/ip.py` classification. Low volume, high signal.

**R-B-L3-004 - TTL fingerprint shift.** Distribution of `ip_ttl` for a given `src_ip` shifts by more
than one hop-equivalent. Indicates an interposed device or spoofing.

**R-P-L3-005 - Fan-out trajectory.** Second difference of the per-source distinct-destination count
positive across three consecutive windows while the first difference is also positive. This fires during
the expansion phase of a scan rather than at its peak. It is a watchlist rule, not an alert rule.

### Layer 4

**R-B-L4-001 - Anomalous flow profile.** Direct application of the shipped `abp-pcap-xgb` model over
the thirteen features in `abp_pcap_preprocessing.py`, with `FilterDetectionsStage` at a tuned threshold.

**R-D-L4-002 - SYN without completion.** `syn/all` above 0.9 with `ack` near zero across more than 50
distinct destination ports from one source in a 60-second bin. Deterministic port scan.

**R-D-L4-003 - RST ratio.** `rst/all` above 0.8 from a single destination across many sources indicates
a service outage; from a single source across many destinations indicates closed-port enumeration.
Distinguish by the fan-out direction; the two conditions need different responses.

**R-B-L4-004 - Stack fingerprint mismatch.** `tcp_options_order` for a flow is inconsistent with the
operating system implied by the layer 7 `user_agent` on the same `flow_id`. Requires the layer 4 to layer 7
lineage hop from Part 4. High signal for proxying, tunneling, and user agent spoofing.

**R-B-L4-005 - Transfer envelope breach.** `bpp` and total `data_len` for an
`(src_ip, dst_ip, dst_port)` triple exceeding the triple's own 30-day 99th percentile by more than 3x.
Baseline per triple, not globally; global thresholds on transfer volume are useless in a heterogeneous
environment.

### Layer 5

**R-B-L5-001 - Composite authentication anomaly.** `max_abs_z` at or above 6.0 from the per-user DFP
model, with `mean_abs_z` at or above 2.0. Requiring both suppresses the common case where a single
feature spikes for a benign reason.

**R-B-L5-002 - Location novelty.** `locincrement_z_loss` at or above 4.0. Note that `locincrement` is
cumulative-distinct, so it rises permanently after a legitimate relocation; the z-score handles this
correctly because the loss scaler is fit per user, but the rule should still carry a 7-day suppression
after a confirmed benign relocation.

**R-D-L5-003 - Impossible travel.** Two successful authentications for one principal from locations
whose great-circle distance divided by the elapsed time exceeds 900 km/h. Exclude authentications from
known VPN egress ranges, and exclude token refreshes, which carry the original location.

**R-D-L5-004 - MFA fatigue.** More than 5 MFA challenges for one principal within 10 minutes with at
least 4 denials followed by an approval. The trailing approval is what makes this actionable rather than
merely noisy.

**R-B-L5-005 - Session duration anomaly.** `session_duration_s` beyond the principal's own 99th
percentile, weighted by `privilege_level`. Only meaningful for interactive sessions; exclude service
accounts, whose duration distribution is bimodal and uninformative.

**R-P-L5-006 - Drift trajectory.** `mean_abs_z` increasing monotonically across four consecutive daily
windows with a total increase above 1.5 standard deviations, without any single window crossing the
R-B-L5-001 threshold. This is the flagship predictive rule for insider risk. It should never page; it
should place the principal on a watchlist and raise the sensitivity of layer 7 rules for that principal.

### Layer 6

**R-B-L6-001 - New TLS client fingerprint.** A `ja4_client` value not previously observed for a
`src_ip` in 30 days, where the host is a managed endpoint. Low volume, high signal, and one of the best
value-for-effort rules in the whole set.

**R-D-L6-002 - Certificate issuer anomaly.** A `certificate_issuer` for a known destination that
differs from the destination's established issuer. Detects interception, including well-intentioned
interception that broke a policy.

**R-D-L6-003 - Self-signed to external destination.** `validation_result` is self-signed and the
destination is `is_global`. Almost always either a misconfiguration or attacker infrastructure.

**R-B-L6-004 - Cipher downgrade.** Negotiated `cipher_suite` strength for a `(src_ip, dst_ip)` pair is
below the pair's historical minimum. Requires an explicit strength ordering table maintained alongside
the rule.

**R-D-L6-005 - Content type mismatch.** `content_type_declared` differs from `content_type_detected`
in a way that crosses a category boundary, for example declared `image/png` and detected as an archive.
Strong tunneling indicator.

### Layer 7

**R-B-L7-001 - DNS tunneling.** Query name entropy above 4.0 bits per character with mean label length
above 30 and more than 100 distinct subdomains under one registered domain in an hour. All three
conditions; entropy alone flags every content delivery network.

**R-B-L7-002 - Bulk data access.** `record_count` for a principal's SaaS operation exceeding the
principal's own 30-day 99th percentile by more than 5x, weighted by the target object's data
classification from TC-0.

**R-B-L7-003 - Sensitive data in transit.** The shipped SID model over request and response bodies at
an egress point, using `TritonInferenceStage` with `AddClassificationsStage` mapped to
`config.class_labels`.

**R-B-L7-004 - Process ancestry novelty.** A `(parent_image_path, image_path)` pair not seen on the
host or in its peer group in 30 days, weighted by `integrity_level`. Peer-group comparison matters:
per-host novelty alone is dominated by long-tail legitimate software.

**R-D-L7-005 - Enumeration.** Ratio of 4xx to 2xx responses above 0.7 with more than 200 distinct
`url_path` values from one client in 10 minutes.

**R-P-L7-006 - Access breadth trajectory.** Distinct `target_object_type` count per principal
increasing across consecutive weekly windows while the principal's role assignment in TC-0 is unchanged.
Classic slow-burn insider indicator, invisible to any single-window rule.

### Cross-layer chained rules

These are the reason to build the lineage substrate at all. Each is expressed as an ordered sequence with
a bounded interval, joined on the lineage identifiers from Part 4.

**R-C-001 - Lateral movement chain.** Within 30 minutes, on one lineage chain: layer 3 fan-out
expansion (R-P-L3-005) on a source, then a layer 5 successful authentication to a new destination host
for the same `user_principal` bound to that source, then a layer 7 process creation with a novel
ancestry on that destination. No individual step needs to breach its own threshold.

**R-C-002 - Command and control establishment.** Layer 6 new client fingerprint (R-B-L6-001) on a host,
followed within 60 minutes by layer 3 beaconing (R-B-L3-002) from the same host to the destination that
the new fingerprint was used against.

**R-C-003 - Physical compromise chain.** Layer 1 transceiver substitution or a new port link event,
then a layer 2 binding of a novel OUI to that port, then a layer 3 flow from the resulting IP to a
management VLAN, all within 4 hours. This chain is the only reason the layer 1 pipeline pays for itself
and it justifies the whole TC-1 collection effort.

**R-C-004 - Staged exfiltration.** Layer 7 bulk data access (R-B-L7-002) by a principal, then within
2 hours a layer 4 transfer envelope breach (R-B-L4-005) from a host bound to that principal's active
session, then a layer 6 connection to a destination whose certificate issuer is novel for the
environment.

**R-C-005 - Credential replay across the stack.** One `user_principal` authenticating successfully at
layer 5 from two `src_ip` values that resolve, through the layer 2 binding table, to two different
physical `port_id` values in different `site_id` values, within a window shorter than physical travel
time. This is impossible travel with physical-layer corroboration, and it is far stronger than the
geolocation version because it does not depend on IP geolocation accuracy.

### Rule governance

Three requirements that are usually skipped and always regretted:

1. **Every rule carries its inverse.** For each rule, define what evidence would prove a firing benign.
   Without it, tuning becomes threshold inflation.
2. **Behavioral rules are versioned with their model.** A rule that references `max_abs_z` is meaningless
   without the model version that produced it. The rule's `inputs` list includes `model_version` for this
   reason, and the SIEM must store it.
3. **Chained rules declare their join tolerance.** `R-C-001` joins across four telemetry classes with
   different clock disciplines. The rule must state its tolerance window explicitly, and the tolerance
   must be at least twice the worst `clock_offset_ms` among the classes it joins.

---

## Part 4: Chaining Layer 1-7 Lineage in Splunk

This is the hardest part of the design and the part that determines whether the rest of it produces
anything usable.

This part targets Splunk specifically, down to the configuration stanzas. The identifier ladder, the
hard-versus-soft join distinction, and the deterministic identifier construction that open the part are
SIEM-independent and port to any SIEM product; everything from [Splunk implementation](#splunk-implementation)
onward is Splunk-specific and is meant to be read as something to deploy rather than as an illustration.

### The identifier ladder

Cross-layer lineage works because each layer shares at least one identifier with the layer above and the
layer below. Written out, the ladder is:

```text
L1  site_id : device_id : port_id
              |
              |  switch MAC table + LLDP, time-bounded
              v
L2  mac_address  ------------------- vlan_id
              |
              |  DHCP lease or ARP binding, time-bounded
              v
L3  ip_address
              |
              |  5-tuple containment, time-bounded
              v
L4  flow_id = src_ip:src_port=dst_ip:dst_port     (also: community_id)
              |
              |  session-to-flow: source IP + time containment, or explicit session_id
              v
L5  session_id : user_principal
              |
              |  TLS handshake within the flow
              v
L6  ja4_client : certificate_fingerprint
              |
              |  request within the session; trace_id if instrumented
              v
L7  trace_id : request_id : operation
```

One naming note before the joins. Layer 1 calls the device `device_id` and layer 2 calls it `switch_id`.
They are the same identifier under two names: both layers compose the port as `site_id:<device>:port_id`
through {py:mod}`~morpheus.utils.entity_key`, so the strings are identical, and the Splunk `binding_l1`
lookup below keys the layer 1 side on `switch_id` for the same reason. The join is on the composed key,
and nothing renames a column to make it.

Every arrow is a join, and every join is one of two kinds.

**Hard joins** are exact equality on a shared field: `flow_id` between layers 4 and 6, `session_id`
between layers 5 and 7 when the application propagates it, `trace_id` within layer 7. These are cheap,
exact, and should be used wherever the telemetry supports them. Most of the requirements in Part 2 exist
specifically to turn soft joins into hard ones.

**Soft joins** are time-bounded interval containments: an IP address belonged to a MAC only during a
lease interval; a MAC was on a port only during a binding interval; a session was active only during an
interval. These are correct only if both endpoints of every interval are known, which is why Part 2
insists on `bind_end` and explicit expiry records.

A soft join without an interval end is not a join. It is a guess, and it will produce confidently wrong
attribution during an incident.

### Deterministic lineage identifiers

Add a custom Morpheus stage, early in every layer segment, that computes three identifiers. All three are
pure functions of data already present, which is what makes the lineage reproducible on replay.

**Event identity.** A content-addressed identifier for the record:

```python
event_uid = sha256(
    collector_id || "\x1f" ||
    schema_version || "\x1f" ||
    origin_hash || "\x1f" ||
    str(collector_seq)
).hexdigest()[:32]
```

Using a unit-separator byte between fields prevents the concatenation ambiguity where
`("ab", "c")` and `("a", "bc")` hash identically. `collector_seq` rather than a row index makes the
identifier independent of how the collector batched the record.

**Link identity.** For each parent-child relationship discovered by a join:

```python
link_uid = sha256(
    parent_event_uid || "\x1f" ||
    child_event_uid  || "\x1f" ||
    relation_type    || "\x1f" ||
    join_method                        # "hard:flow_id" or "soft:dhcp_lease"
).hexdigest()[:32]
```

Recording `join_method` on the edge is what lets an analyst distinguish an exact attribution from an
inferred one, months later, without re-deriving the join.

**Chain identity.** For a correlation window, compute a Merkle root over the deduplicated, sorted
`event_uid` values in the chain:

```python
chain_root = merkle_root(event_uids)          # sorts and deduplicates internally
lineage_id = sha256(root_entity_key || window_id || chain_root).hexdigest()[:32]
```

Sorting and deduplicating before hashing makes `lineage_id` a function of chain membership alone, which
is the property that survives out-of-order delivery, redelivery, and replay. `window_id` is the
deterministic window identifier from Part 5.

These four functions ship as {py:mod}`~morpheus.utils.lineage`, and
{py:class}`~morpheus.stages.lineage.lineage_stamp_stage.LineageStampStage` applies the first two to a
message, declaring its outputs so the columns are preallocated:

```python
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage

pipe.add_stage(
    LineageStampStage(config,
                      id_columns=["collector_id", "schema_version", "origin_hash", "collector_seq"],
                      parent_uid_column="parent_event_uid",
                      relation="carried_by",
                      join_method="hard:flow_id"))
```

The stage hashes on the host in both GPU and CPU execution modes. That is a deliberate trade: an
identifier whose value depended on which execution mode produced it would defeat the purpose of having
it. The cost is measurable, so budget for it rather than assuming it is free. Measured single-core on
the reference implementation:

| Path | Throughput |
| --- | --- |
| Community ID, realistic flow data (5k distinct tuples over 200k rows) | ~2.9M rows/s |
| Community ID, every row a distinct tuple | ~180k rows/s |
| `event_uid`, four fields, all rows unique | ~590k rows/s |

The 16x spread on Community ID is the tuple memoization, and it is why the stage belongs *after* the
flow rollup rather than on raw packets: rolled-up telemetry repeats tuples heavily, unaggregated
telemetry does not. `event_uid` cannot benefit from memoization at all, because the identifiers are
unique by construction, so its ~590k rows/s is the harder ceiling and the one to plan against. At
layer 5 and layer 7 event volumes that is ample. At raw layer 3 and 4 packet rates it is not, and the
lineage stamp has to sit downstream of aggregation.

Where the host ceiling still binds, `LineageStampStage` offers an opt-in device path
(`use_gpu_hashing=True`) that computes the same SHA-256 digests through cuDF's `hash_values`. The
identifiers are still required to be byte-identical to the host path, and that requirement is enforced
rather than assumed: {py:mod}`~morpheus.utils.lineage_cudf` ships a digest-equivalence gate over probe
vectors, covering the separator boundary case, non-ASCII text, and negative integers, and the stage
runs it before the first device-hashed batch. If this cuDF's digests ever disagree with `hashlib`, the
pipeline fails closed instead of minting identifiers nothing else can reproduce. The device path
accepts only string and integer identifier columns and refuses nulls, because those are the cases
whose rendering is provably identical between the two implementations; anything else must be
pre-rendered as a string or stay on the host path.

Measured over two million rows on an RTX 5000 Ada laptop GPU, warmed and best of three: the device
path hashed ~57M rows/s against ~1.5M rows/s for the host path on the same machine's CPU, a 38x
speedup on the hashing alone. The comparison deliberately excludes the device-to-host copy that the
host path additionally pays inside a GPU pipeline, so the in-pipeline gain is larger. As with the
other figures in this section, treat these as one machine's measurement, not a specification.

Be precise about which constraint this lifts. The device path covers `event_uid` and `link_uid`
only, so a GPU pipeline that opts in can stamp lineage at raw layer 3 and 4 event rates. Community ID
remains host-only, and its worst-case ~180k rows/s still argues for computing it downstream of the
flow rollup, where memoization does the work; the placement advice earlier in this section stands for
that stage unchanged.

Emit the edges as a **separate stream** from the scored events, on its own Kafka topic and into its own
SIEM index. Edges are small, numerous, and queried with entirely different access patterns than events.
Denormalizing edges into event records seems simpler and makes the multi-hop walk in the next section
impossible to express efficiently.

### Community ID

For layers 3 and 4, emit `community_id` alongside `flow_id`. It is the widely adopted flow-hashing
convention: it orders the endpoint pair canonically before hashing, so both directions of a
bidirectional flow produce the same value, and it is computed identically by Zeek, Suricata, Elastic,
and most modern network tooling.

This one field turns most layer 3 and 4 joins from soft to hard, and it makes Morpheus output joinable
against network telemetry the enterprise already collects without any coordination. It is the single
highest-leverage decision in this entire section.

{py:class}`~morpheus.stages.lineage.community_id_stage.CommunityIdStage` implements version 1 of the
specification, including the ICMP and ICMPv6 message-type mapping:

```python
from morpheus.stages.lineage.community_id_stage import CommunityIdStage

pipe.add_stage(CommunityIdStage(config, src_ip_column="src_ip", dst_ip_column="dest_ip",
                                protocol_column="protocol", src_port_column="src_port",
                                dst_port_column="dest_port"))
```

Leave `seed` at its default of zero. The seed is part of the hash input, so a non-default value produces
identifiers that no other tool in the estate will agree with, which forfeits the entire benefit.

### Splunk implementation

Everything in this section ships as an installable app,
[`examples/splunk_lineage_app`](../../../../examples/splunk_lineage_app/README.md). The stanzas
quoted below are the explanation; the app is the normative copy, and when the two disagree the app
is what was meant to run. The app's README lists the three values that must match the Morpheus
pipeline configuration.

#### Getting data in

Four options, in descending order of preference:

1. **Kafka to Splunk Connect for Kafka.** `WriteToKafkaStage` to a topic, consumed by the connector.
   Backpressure is handled by Kafka, replay is possible by resetting consumer offsets, and Morpheus is
   decoupled from Splunk availability. Use this.
2. **HTTP Event Collector via `HttpClientSinkStage`.** Lower latency, no broker. But HEC failures become
   Morpheus backpressure, and there is no replay. Acceptable for low-volume high-priority streams only.
3. **File drop plus universal forwarder.** Highest durability, worst latency. Reasonable for the
   lineage edge stream, which is not latency-sensitive.
4. **Elasticsearch as primary with Splunk federated search.** Viable if Elastic is already deployed;
   `WriteToElasticsearchStage` exists and works.

#### The event time wire format

`event_time` is an integer count of nanoseconds inside the pipeline, which is what
`window_id_from_timestamp` and the binding tables consume. It must not reach the SIEM that way. Splunk
anchors timestamp extraction on a prefix and applies a `strptime` format; a nineteen-digit integer
matches neither, and the failure is silent.

Render it before the sink:

```python
from morpheus.utils.siem_wire import render_event_time_series

df["event_time"] = render_event_time_series(df["event_time"])   # 2026-08-30T18:25:00.123456UTC
```

Measured on a live Splunk instance, ingesting the same event both ways through Morpheus's own Kafka
serializer: the rendered form lands at its true event time, three hours in the past. The unrendered
form lands at **index time**, dated to the moment of ingest. Nothing in the pipeline, the sink, or
Splunk reports an error.

The fallback is worse than that experiment first suggested. Splunk only reaches for index time when it
has nothing else; where an unparsable event follows a parsable one from the same source, it inherits
**the previous event's timestamp**. A partially broken stream therefore produces timestamps that are
wrong but entirely plausible, clustered near real events, and no threshold on `_indextime` will find
them. This is the failure mode the four-timestamp discipline in [Handling time
correctly](#handling-time-correctly) exists to prevent, and it starts here, at the rendering.

Two consequences worth stating. The rendering is microsecond precision, matching Splunk's `%6N`, and
truncates rather than rounds, so an event never crosses a window boundary it did not cross; carry the
exact integer in a separate field where nanosecond fidelity has to survive the hop. And the
`event_time` rendering is a fourth shared contract between the pipeline and the SIEM, alongside the
bucket width, the binding retention, and the Community ID seed. `tests/morpheus/utils/test_siem_wire.py`
enforces it by reading the shipped `props.conf` directly, so the two sides cannot drift apart.

#### Index and sourcetype layout

Four indexes, split by volume and retention rather than by layer:

```ini
# indexes.conf
[behavior_events]
homePath   = $SPLUNK_DB/behavior_events/db
coldPath   = $SPLUNK_DB/behavior_events/colddb
thawedPath = $SPLUNK_DB/behavior_events/thaweddb
frozenTimePeriodInSecs = 7776000

[behavior_lineage]
homePath   = $SPLUNK_DB/behavior_lineage/db
coldPath   = $SPLUNK_DB/behavior_lineage/colddb
thawedPath = $SPLUNK_DB/behavior_lineage/thaweddb
frozenTimePeriodInSecs = 2592000

[behavior_bindings]
homePath   = $SPLUNK_DB/behavior_bindings/db
coldPath   = $SPLUNK_DB/behavior_bindings/colddb
thawedPath = $SPLUNK_DB/behavior_bindings/thaweddb
frozenTimePeriodInSecs = 34560000

[behavior_context]
homePath   = $SPLUNK_DB/behavior_context/db
coldPath   = $SPLUNK_DB/behavior_context/colddb
thawedPath = $SPLUNK_DB/behavior_context/thaweddb
frozenTimePeriodInSecs = 34560000

[behavior_summary]
homePath   = $SPLUNK_DB/behavior_summary/db
coldPath   = $SPLUNK_DB/behavior_summary/colddb
thawedPath = $SPLUNK_DB/behavior_summary/thaweddb
frozenTimePeriodInSecs = 7776000
```

Sourcetypes within them:

```text
index=behavior_events     sourcetype=morpheus:score:l<N>     # one per layer
index=behavior_lineage    sourcetype=morpheus:edge
index=behavior_bindings   sourcetype=binding:l1 | binding:l2 | binding:l3 | binding:bucketed
index=behavior_summary    sourcetype=stash               # written by collect
index=behavior_context    sourcetype=context:identity | context:asset
```

`behavior_summary` holds the 5-minute per-layer score rollups that the chained rules read; it is
populated by a scheduled search rather than by Morpheus directly, and its retention matches
`behavior_events` because a chained rule can look back as far as any event it correlates. Its
sourcetype stays at `collect`'s default (`stash`) deliberately: that is what keeps summary volume
exempt from license metering, so consumers filter on the index alone.

Four deliberate choices here.

**Bindings and context are retained for 400 days while events are retained for 90.** Bindings are the
only thing that makes a historical attribution reconstructible: an incident investigated in month eleven
needs to know which host held an address in month one, and if the binding has aged out, the layer 3 event
that survived is unattributable. Bindings are also two to three orders of magnitude smaller than events,
so the long retention is close to free. Getting this backwards, with long event retention and short
binding retention, is a common and expensive mistake.

**Edges are retained for 30 days**, shorter than events. An edge is derivable from the events it links, so
a chain older than the edge retention can be rebuilt if it ever matters. Edge volume is high enough that
matching event retention roughly doubles the total storage bill for something rarely queried past a month.

**Separating bindings from events is what makes the time-bounded joins tractable**, because bindings are
low-volume enough to be replicated into a KV Store collection for lookup-speed access.

#### Parsing and timestamps

Every stream is JSON. The setting that matters is the timestamp: Splunk's default is to find a timestamp
anywhere in the payload, and with four timestamps in the envelope it will regularly pick the wrong one.

```ini
# props.conf
[morpheus:score:l3]
KV_MODE            = json
SHOULD_LINEMERGE   = false
TRUNCATE           = 0
TIME_PREFIX        = "event_time":"
TIME_FORMAT        = %Y-%m-%dT%H:%M:%S.%6N%Z
MAX_TIMESTAMP_LOOKAHEAD = 32
MAX_DAYS_AGO       = 30
category           = Custom
disabled           = false
```

`TIME_PREFIX` anchored to `event_time` is the single most important line in the Splunk configuration.
Without it, `_time` drifts toward `ingest_time` and every windowed rule silently becomes a rule about when
the pipeline was busy. `MAX_DAYS_AGO` must exceed the longest backfill you intend to run, or replayed
history is quietly rejected.

Repeat the stanza per layer, or use a single `morpheus:score` sourcetype with `osi_layer` as a field. One
sourcetype per layer is worth the duplication because it lets you set different `MAX_DAYS_AGO` values,
since SaaS audit APIs at layer 7 backfill far later than NetFlow at layer 3, and because it makes index-time
routing per layer possible later without reindexing.

`KV_MODE = json` extracts at search time, which keeps the index small. `tstats` cannot see search-time
fields, so the acceleration path is data model acceleration rather than raw `tstats`, which is what the
[acceleration section](#acceleration-and-determinism-in-splunk) assumes. The alternative,
`INDEXED_EXTRACTIONS = json` with `KV_MODE = none`, makes `tstats` work directly against the raw index at
the cost of a substantially larger index. Choose one and record which, because the queries differ.

#### CIM mapping

Map to the Common Information Model so existing Splunk content works against this data without
modification:

| Telemetry class | CIM data model | Key field mappings |
| --- | --- | --- |
| TC-2 | Network Sessions | `mac`, `vlan`, `dest_ip`, `duration` |
| TC-3, TC-4 | Network Traffic | `src`, `dest`, `src_port`, `dest_port`, `bytes_in`, `bytes_out`, `action` |
| TC-5 | Authentication | `user`, `src`, `action`, `app`, `authentication_method` |
| TC-6 | Certificates | `ssl_issuer`, `ssl_subject`, `ssl_hash`, `ssl_version` |
| TC-7 | Web, Endpoint, Change | `url`, `http_method`, `status`, `process`, `parent_process` |

Add the Morpheus-specific fields (`mean_abs_z`, `max_abs_z`, per-feature `*_z_loss`, `model_version`,
`config_hash`, `event_uid`, `lineage_id`) as an extension rather than trying to force them into existing
CIM fields. `risk_score` maps cleanly onto the Risk data model if Enterprise Security is in play.

#### Time-bounded binding lookups

Splunk's `lookup` command matches on equality, not on interval containment. An exact interval join is
expressible in SPL and does not perform at enterprise volume, so the containment has to be precomputed:
every binding is expanded across the time buckets its interval touches, and the lookup matches on
`(key, bucket)`.

Joining against a discretized bucket is an approximation, and it is the right one: it is deterministic,
it is fast, and its error is bounded by the bucket width. State the bucket width in every rule that
depends on it. At a 300-second bucket, an event can be attributed to a binding that had already expired by
up to 300 seconds, which matters for a DHCP lease that churned within the bucket and does not matter for a
switch port binding that lasts for months.

**Compute the expansion in Morpheus, not in Splunk.**
{py:class}`~morpheus.utils.binding_table.BindingTable` holds the intervals and
{py:meth}`~morpheus.utils.binding_table.BindingTable.to_bucketed_frame` performs the expansion, applying
the same tie-break that the in-pipeline resolver applies:

```python
from morpheus.utils.binding_table import BindingTable

leases = BindingTable.from_dataframe(lease_df,
                                     name="dhcp_lease",
                                     key_column="ip",
                                     value_columns=["mac", "port_id", "switch_id"],
                                     start_column="bind_start",
                                     end_column="bind_end",
                                     open_end_duration_ns=8 * 3600 * 10**9)   # cap at the lease time

lookup_rows = leases.to_bucketed_frame(bucket_seconds=300, key_name="ip")
```

Doing the expansion once, upstream, is what keeps the two sides consistent. If Splunk discretizes
independently of the pipeline, a chain assembled in SPL can disagree with the columns Morpheus already
resolved on the same event, and reconciling that during an incident is miserable.

Two properties of the expansion are worth stating explicitly, because both are load-bearing:

- **Exactly one row per key and bucket.** Where several bindings land in one bucket, which happens
  whenever a lease churns faster than the bucket width, the most recent start wins. A multi-valued
  lookup would make `lookup` return an unpredictable row.
- **`open_end_duration_ns` is mandatory for open intervals.** A binding with no observed end is rejected
  unless you say how long to assume it lasted. This is the guide's own rule, enforced in code: a soft
  join against an unbounded interval is a guess.

**Resolve in the pipeline as well.** Splunk-side lookups serve ad-hoc investigation; the scored event
should already carry its resolution so that a rule at layer 3 can reference a physical port without a
join at all. {py:class}`~morpheus.stages.lineage.binding_resolver_stage.BindingResolverStage` does this at
full interval precision, with no bucketing error:

```python
from morpheus.stages.lineage.binding_resolver_stage import BindingResolverStage

pipe.add_stage(BindingResolverStage(config,
                                    binding_table=leases,
                                    key_column="src_ip",
                                    time_column="event_time",
                                    uid_column="binding_uid"))
```

Every row comes out with a `resolution_method` of either `soft:dhcp_lease` or `unresolved`, and resolved
rows carry `binding_uid`, the content-addressed identifier of the exact lease used. That field is what
makes an attribution auditable eleven months later: the analyst does not have to re-derive which lease was
in force, because the event names it.

Rows that do not resolve stay in the stream marked `unresolved` rather than being dropped. A dropped row
looks like an absence of activity, which is the worst possible failure mode for a detection pipeline.
Alert on the unresolved rate. A rising rate means the collector is losing expiry records, and every
attribution it produces is suspect. `BindingTable` counts overlapping intervals at construction for the
same reason and logs a warning.

**Define the lookup.** The KV Store collection and its lookup definition:

```ini
# collections.conf
[binding_l2_l3_collection]
enforceTypes = true
field.ip          = string
field.bucket      = number
field.mac         = string
field.port_id     = string
field.switch_id   = string
field.binding_uid = string
accelerated_fields.by_ip_bucket = {"ip": 1, "bucket": 1}
```

```ini
# transforms.conf
[binding_l2_l3]
external_type = kvstore
collection    = binding_l2_l3_collection
fields_list   = ip, bucket, mac, port_id, switch_id, binding_uid

[binding_l1]
external_type = kvstore
collection    = binding_l1_collection
fields_list   = port_id, switch_id, site_id, transceiver_serial, lldp_neighbor_chassis_id
```

The accelerated field definition is not optional. Without it, every `lookup` against a collection holding
tens of millions of bucket rows is a collection scan, and the chain queries below become unusable at
exactly the moment they are needed.

`binding_l1` has no bucket because a transceiver in a switch port is stable for months. Bucket only what
actually churns; bucketing a stable binding multiplies its row count by the retention period for no gain.

#### Populating the lookups

Morpheus publishes the expanded rows to `index=behavior_bindings sourcetype=binding:bucketed`, and a
scheduled search materializes them into the KV Store:

```ini
# savedsearches.conf
[Binding lookup - L2/L3 refresh]
enableSched          = 1
cron_schedule        = */5 * * * *
dispatch.earliest_time = -20m@m
dispatch.latest_time   = -5m@m
realtime_schedule    = 0
search = index=behavior_bindings sourcetype=binding:bucketed binding_table=dhcp_lease \
| eval _key = ip . "|" . bucket \
| table _key ip bucket mac port_id switch_id binding_uid \
| outputlookup binding_l2_l3 append=t key_field=_key
```

Three details carry the weight:

- **`_key` is a deterministic composite of the lookup's own key fields.** That makes the write idempotent:
  a re-run, a replay, or an overlapping schedule window overwrites the same document instead of appending
  a duplicate. This is control 11 from Part 5 applied at the SIEM boundary, and without it a retry
  silently corrupts the lookup.
- **`realtime_schedule = 0`** puts the search on continuous scheduling, so a run skipped during a restart
  is caught up rather than abandoned. With the default, a skipped run leaves a permanent hole in the
  lookup and every event in that window becomes unattributable.
- **The window trails by five minutes and is twice the schedule interval.** The lag absorbs indexing
  latency; the overlap means a single late-arriving binding still lands, and the idempotent `_key` makes
  the overlap harmless.

Expiry is a separate job, because `outputlookup append=t` never removes anything:

```ini
[Binding lookup - L2/L3 expiry]
enableSched   = 1
cron_schedule = 17 3 * * *
search = | inputlookup binding_l2_l3 \
| where bucket >= floor((now() - 34560000) / 300) \
| outputlookup binding_l2_l3
```

This is the one place `now()` is acceptable, because the job is maintenance rather than detection and its
output is never compared across runs. Every search whose result a rule depends on must still use
`info_max_time`.

Run the expiry job at an hour when the refresh job is least likely to be mid-write, and keep its retention
aligned with `frozenTimePeriodInSecs` on `behavior_bindings`. A lookup that expires before the index does
produces the same unattributable-event failure as losing the index itself.

#### Walking the ladder

The multi-hop walk, expressed with `stats` rather than `transaction`. Use `transaction` nowhere in this
architecture: it is memory-bound, it silently truncates on large event counts, and its results depend on
event arrival order, which breaks determinism.

Single hop, layer 3 to layer 2 to layer 1, resolving an IP to a physical port:

```spl
index=behavior_events sourcetype=morpheus:score:l3 max_abs_z>=6.0
| eval bucket=floor(_time/300)
| lookup binding_l2_l3 ip AS src_ip bucket OUTPUT mac port_id switch_id
| lookup binding_l1 port_id switch_id OUTPUT site_id transceiver_serial lldp_neighbor_chassis_id
| table _time src_ip mac port_id switch_id site_id max_abs_z event_uid lineage_id
```

Full chain assembly from the edge index. This is the query that makes R-C-001 through R-C-005
expressible:

```spl
index=behavior_lineage sourcetype=morpheus:edge earliest=-30m
| stats values(child_event_uid)  AS children
        values(parent_event_uid) AS parents
        values(join_method)      AS methods
        values(osi_layer)        AS layers
        min(_time) AS chain_start
        max(_time) AS chain_end
        dc(osi_layer) AS layer_span
  by lineage_id
| where layer_span >= 3
| eval chain_duration = chain_end - chain_start
```

To attach the scores, do not reach for `join`. Its subsearch silently truncates at 50,000 rows by
default, which is the same failure mode this document rejects `transaction` for, and it fails exactly
when the environment is busy enough to matter. Union the two sources and let a single `stats` do the
correlation:

```spl
(index=behavior_lineage sourcetype=morpheus:edge) OR (index=behavior_events) earliest=-30m
| stats dc(osi_layer)   AS layer_span
        min(_time)      AS chain_start
        max(_time)      AS chain_end
        max(max_abs_z)  AS peak_z
        sum(risk_score) AS total_risk
        values(rule_id) AS rules
        values(join_method) AS methods
  by lineage_id
| where layer_span >= 3
| where total_risk >= 60 OR (layer_span >= 4 AND peak_z >= 4.0)
| eval chain_duration = chain_end - chain_start
| sort - total_risk
```

This works because both streams carry `lineage_id`, which is the entire point of minting it upstream.
The `layer_span >= 3` filter is what keeps it cheap: the overwhelming majority of chains are
single-layer and are discarded before anything expensive happens.

Ordered-sequence detection for a specific chained rule, R-C-002:

```spl
index=behavior_events (rule_id="R-B-L6-001" OR rule_id="R-B-L3-002") earliest=-2h
| stats min(eval(if(rule_id="R-B-L6-001", _time, null()))) AS t_tls
        min(eval(if(rule_id="R-B-L3-002", _time, null()))) AS t_beacon
        values(lineage_id) AS lineage_id
  by src_ip dest_ip
| where isnotnull(t_tls) AND isnotnull(t_beacon)
| eval gap = t_beacon - t_tls
| where gap > 0 AND gap <= 3600
| eval rule_id="R-C-002", risk_score=70
```

Requiring `gap > 0` enforces the ordering, which is the entire point of a chained rule. A rule that
matches the same two events in either order is a co-occurrence rule and should be labeled as one.

#### Scheduling the detections

The queries above are the logic. Turning one into a detection that fires reproducibly is a matter of the
stanza around it:

```ini
# savedsearches.conf
[R-C-002 - TLS anomaly precedes beaconing]
enableSched            = 1
cron_schedule          = */15 * * * *
dispatch.earliest_time = -2h@m
dispatch.latest_time   = -15m@m
realtime_schedule      = 0
schedule_window        = 5
allow_skew             = 5m
description = Layer 6 TLS fingerprint anomaly followed within one hour by layer 3 beaconing on the same \
              endpoint pair. Determinism tier D2. Bucket width 300s on binding lookups.
search = index=behavior_events (rule_id="R-B-L6-001" OR rule_id="R-B-L3-002") \
| stats min(eval(if(rule_id="R-B-L6-001", _time, null()))) AS t_tls \
        min(eval(if(rule_id="R-B-L3-002", _time, null()))) AS t_beacon \
        values(lineage_id) AS lineage_id \
  by src_ip dest_ip \
| where isnotnull(t_tls) AND isnotnull(t_beacon) \
| eval gap = t_beacon - t_tls \
| where gap > 0 AND gap <= 3600 \
| eval rule_id = "R-C-002", risk_score = 70
action.correlationsearch.enabled = 1
action.correlationsearch.label   = R-C-002
```

Point by point, because each line is there to prevent a specific failure:

- **`dispatch.latest_time = -15m@m`, never `now`.** The 15-minute trailing edge is the lateness horizon
  from Part 5. A search ending at `now` includes whatever happened to be indexed at the moment it ran, so
  two runs over the same nominal window return different results and the rule is not reproducible. The
  `@m` snap matters as much as the offset: without it the window boundary moves with the scheduler's
  jitter.
- **The window is 2 hours for a rule with a 1-hour `maxspan`.** A sequence rule needs a search window of
  at least the span plus the schedule interval plus the lateness horizon, or a chain straddling a window
  boundary is never seen by either run. Getting this wrong produces a rule that works in testing, where
  events are dense, and misses in production, where they are not.
- **`schedule_window = 5` and `allow_skew = 5m`** let the scheduler move the run to reduce contention.
  Both are safe *only because* the time range is absolute rather than relative to the run: the search
  returns the same rows whenever it actually executes. With `latest_time = now` they would be a source of
  nondeterminism rather than a scheduling convenience.
- **`realtime_schedule = 0`** again, for the same reason as the lookup jobs: catch up rather than skip.

Overlapping windows mean a chain can match on consecutive runs. Deduplicate downstream on
`(rule_id, lineage_id)` rather than by narrowing the window. The alternative trades duplicate alerts for
missed ones, which is the wrong trade. The `lineage_id` is stable across runs by construction, which is
what makes this deduplication reliable rather than best-effort.

For rules driven off the summary index rather than the raw index, the same stanza applies with
`index=behavior_summary` and a longer `dispatch.earliest_time`, since summary rows are written on a
5-minute cadence and are themselves subject to the lateness horizon.

#### Acceleration and determinism in Splunk

- Use `tstats` against accelerated data models for the volume layers. Accept that acceleration introduces
  a summarization lag and document it in each rule's expected detection latency.
- Summary-index the per-layer scores at 5-minute granularity. Chained rules run against the summary, not
  the raw index, which makes their runtime independent of raw volume and their results independent of
  index-time variability.
- Pin every scheduled search to a fixed relative time range with a lag offset that exceeds the lateness
  horizon from Part 5. A search with `earliest=-30m latest=now` produces different results depending on
  when it runs. `earliest=-45m latest=-15m` does not.
- Never use `now()` inside a search that must be reproducible. Use `info_max_time`.

### A graph store alongside Splunk

Splunk is the detection and alerting surface. It is a poor investigation surface for open-ended multi-hop
questions, because every additional hop is another `stats` over another index and the query cost grows
with the estate rather than with the answer.

For investigation, fan the edge stream out to a property graph as well (Neo4j, or the RAPIDS `cugraph`
stack that is already in the Morpheus dependency set). Questions such as "every layer 7 operation
reachable from this physical port in the last 24 hours" are one Cypher clause and a page of SPL. This is
an addition, not an alternative: the same Kafka topic feeds both, the graph carries no detection logic,
and nothing in Part 3 depends on it.

`examples/gnn_fraud_detection_pipeline` shows the pattern for scoring a graph with a GNN once it is
built. The same technique applies to scoring lineage chains directly rather than scoring their
constituent events, which is the natural next step once the edge stream has been running long enough to
have a labeled history.

### If the SIEM is not Splunk

This part is deliberately Splunk-only. The pieces that transfer without change are the ones that live
upstream of the SIEM: the identifier ladder, `event_uid` / `link_uid` / `lineage_id`, `community_id`, the
binding tables, and the four-timestamp discipline. Those are properties of the data Morpheus emits.

What does not transfer is everything from the index layout down. Two things are worth knowing before
porting:

- **Ordered-sequence detection is where backends differ most.** The `stats`-based sequence idiom below
  exists because SPL has no native ordered-sequence operator that is safe at volume. Backends that do
  have one, such as Microsoft Sentinel's `scan` and Elastic's EQL `sequence`, express the chained rules in
  Part 3 far more directly, and a port should use them rather than transliterating the SPL.
- **The bucketed binding lookup is a workaround for equality-only lookups.** A SIEM product with interval
  joins should use them and skip the discretization, which removes the bounded attribution error
  described below.

### Handling time correctly

Four distinct timestamps must be kept separate throughout, and conflating any two of them is the most
common way this architecture fails in production:

| Timestamp | Meaning | Used for |
| --- | --- | --- |
| `event_time` | When it happened | All windowing, all joins, all rules |
| `observed_time` | When the sensor saw it | Sensor health monitoring, join tolerance calculation |
| `ingest_time` | When Morpheus received it | Pipeline latency monitoring only |
| `_indextime` | When the SIEM stored it | Operational monitoring only |

Splunk's `_time` must be set from `event_time`, never from ingest. Set a **lateness horizon** per
telemetry class, typically 15 minutes for network and 60 minutes for SaaS audit APIs, which are
notoriously delayed. A window is sealed only after the horizon has elapsed. Records arriving after the
seal go to a late-arrival index and trigger a documented backfill procedure, rather than silently
mutating an already-published result.

---

## Part 5: Preserving Determinism in the Output

### Why it is worth the cost

Determinism is not an aesthetic preference. Three concrete consequences:

- **Investigation.** An analyst who reruns a detection over the same data and gets a different answer
  cannot build a timeline anyone will defend.
- **Tuning.** If output varies run to run, the effect of a threshold change is unmeasurable, because the
  change is confounded with run-to-run variance.
- **Evidence.** In regulated environments or litigation, "the system produced this alert and here is the
  exact configuration and model that produced it" is a requirement, not a nicety.

Determinism costs throughput. The controls below are ordered so that the cheap ones come first and the
expensive ones are clearly marked, and so a deployment can choose its tier per pipeline rather than
globally.

### Four tiers

Declare a tier per pipeline and record it on every event.

| Tier | Guarantee | Typical cost |
| --- | --- | --- |
| **D0** Bit-exact | Identical input bytes produce byte-identical output, including all float fields | 40-70% throughput reduction |
| **D1** Decision-stable | Identical input produces identical alert or no-alert decisions and identical rule firings; scores may differ in low-order bits | 10-20% reduction |
| **D2** Lineage-stable | Identical input produces identical `event_uid`, `link_uid`, and `lineage_id` values and identical chain membership; scores and ordering may vary | Near zero |
| **D3** Explanation-stable | D1 plus identical top-k feature attribution ranking from the per-feature `z_loss` columns | 15-25% reduction |

**D1 plus D2 is the right default for enterprise detection.** D0 is worth paying for only on the specific
pipelines whose output may become evidence. D3 matters when analysts act on the attribution, which in
practice they do, so it is worth enabling for the layer 5 and layer 7 pipelines specifically.

### Control 1: Pin the model

Never resolve a model by "latest." `DFPInferenceStage` calls
`ModelManager.load_user_model(client, user_id, fallback_user_ids)`, which resolves whatever version is
current and caches it for `self._cache_timeout_sec = 600` with `self._model_cache_size_max = 10`. Two
consequences: a retraining event mid-run changes scores, and a cache eviction can change them back.

Resolve models once at window start into an explicit `{entity: "name:version"}` manifest, and hold that
manifest for the whole window. Emit `model_version`, which `DFPInferenceStage` already writes as
`f"{model_cache.reg_model_name}:{model_cache.reg_model_version}"`, on every scored event.

Also emit `model_fallback_used` as a boolean. When a per-entity model does not exist, the pipeline
silently substitutes `generic_user`. An event scored against the generic model is a different kind of
claim than one scored against the entity's own model, and the difference must be visible in the SIEM.

### Control 2: Freeze and hash the configuration

`Config.freeze()` already makes the object immutable at build time, and `Config.save()` serializes it to
JSON with `sort_keys=True`. Hash that serialization and emit it as `config_hash`:

```python
config_hash = hashlib.sha256(config.to_string().encode()).hexdigest()[:16]
```

Extend the hash to cover what `Config` does not: the `DataFrameInputSchema` definitions, the threshold
values, the code commit, and the container image digest. Emit the combination as `pipeline_fingerprint`.
Two events with the same `pipeline_fingerprint` were produced by the same system; two events with
different values are not directly comparable, and any drift analysis that ignores this will attribute a
configuration change to a behavioral change.

### Control 3: Seed everything, and go further than `manual_seed`

`python/morpheus/morpheus/utils/seed.py` covers `random`, NumPy, Torch, CuPy, all CUDA devices, and sets
`torch.backends.cudnn.deterministic = True` with `benchmark = False`. That is a good start and not
sufficient for D0. Add:

```python
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # must be set before CUDA init
import torch
torch.use_deterministic_algorithms(True, warn_only=False)
```

`CUBLAS_WORKSPACE_CONFIG` must be set before the CUDA context is created, which means before any Torch
or CuPy import. Set it in the container entrypoint, not in Python. `use_deterministic_algorithms` will
raise on operations with no deterministic implementation, which is the desired behavior: it converts a
silent reproducibility hole into a startup failure.

Emit the seed as `rng_seed` on every event.

### Control 4: Shard instead of thread

This is the most important control and the one with the best cost-to-benefit ratio.

Any node with `pe_count > 1` emits in completion order. But parallelism does not require intra-stage
threading. Use `RouterStage` to shard messages by a stable hash of the entity key:

```python
pipeline.add_stage(RouterStage(config,
                               keys=[f"shard_{i}" for i in range(N)],
                               key_fn=lambda msg: f"shard_{stable_hash(msg.get_metadata('entity_key')) % N}"))
```

Then run one single-engine scoring branch per shard. Each entity deterministically lands on the same
shard on every run, so per-entity ordering is total and stable, while N-way parallelism is preserved.
This recovers most of the throughput lost to serialization.

Use `stable_hash` from `hashlib`, not Python's built-in `hash`, which is randomized per process by
`PYTHONHASHSEED` and will silently produce a different sharding on every restart.

Set `pe_count = 1` explicitly on every stage in the scoring path. The DFP inference and preprocessing
stages already have their `pe_count` assignments commented out; make that an explicit `= 1` with a
comment explaining why, so a future performance optimization does not silently re-enable it.

### Control 5: Make batching irrelevant

Batch composition varies with upstream timing. Three mitigations, in order of preference:

1. **Verify batch invariance.** Score a fixed corpus at `model_max_batch_size` of 1, 8, and 64 and assert
   the outputs match within tolerance. If they do not, the model has a batch-dependent operation, most
   commonly batch normalization in training mode. Fix the model.
2. **Disable Triton dynamic batching** for scoring models. Set `max_queue_delay_microseconds` to 0 or
   remove the `dynamic_batching` block from the model configuration entirely.
3. **Fix the batch boundary deterministically.** Batch by window and entity rather than by arrival, so
   batch membership is a function of the data rather than of timing.

Note that `pipeline_batch_size` and `model_max_batch_size` both appear in `config_hash`, so a change to
either is at least visible even before it is eliminated.

### Control 6: Event time only

Replace the wall-clock read in `DFPPostprocessingStage`:

```python
df['event_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')   # nondeterministic
```

with a value derived from the data, typically the window's closing boundary. Keep the wall-clock value,
but under a distinct name (`processing_time`) that is excluded from every hash, every feature, and every
rule. A field named `event_time` that actually holds a processing timestamp is worse than no field at
all, because downstream consumers will trust it.

Audit for the same pattern anywhere else in a custom pipeline. `datetime.now()`, `time.time()`, and
`uuid4()` in a scoring path are all determinism defects.

### Control 7: Deterministic windows

Define windows as half-open intervals `[t0 + k*P, t0 + (k+1)*P)` anchored to a fixed absolute epoch, not
to the first observed event. `TimeSeriesStage.calc_bin` already does exactly this, and it is the pattern
to copy:

```python
window_id = (event_time_ns - EPOCH_NS) // period_ns
```

Then:

- A window is **sealed** at `window_end + lateness_horizon`, and only sealed windows are scored.
- Events arriving after the seal are written to a late-arrival stream. They never mutate a published
  window.
- Backfill is an explicit operation that produces a **new** result with an incremented `revision` field,
  never an in-place overwrite. The SIEM keeps both, and rules filter to the maximum revision per
  `window_id`.
- `window_id` is a component of `lineage_id`, so an event's chain membership is a function of its event
  time alone.

This control ships as {py:class}`~morpheus.stages.lineage.window_seal_stage.WindowSealStage`, backed by
{py:mod}`~morpheus.utils.window_seal`. The seal is driven by the event-time watermark rather than the
wall clock, which is the property that makes it replayable: the same rows in the same order seal the
same windows at the same points in the stream, however fast the replay runs. Window membership is a pure
function of each row's event time; only the on-time versus late split depends on stream order, and that
split is exactly what the late-arrival stream records:

```python
from morpheus.stages.lineage.window_seal_stage import WindowSealStage

pipe.add_stage(
    WindowSealStage(config,
                    period_seconds=300,
                    lateness_seconds=900,
                    order_columns=["event_time", "collector_id", "collector_seq"]))
```

`order_columns` applies control 8's stable total order to each emitted window, so cumulative features
computed downstream see rows in a reproducible order. The stage is stateful and runs single-engine;
for parallelism, shard by entity upstream (control 4) and give each shard its own instance.

### Control 8: A total order on rows

`IncrementColumn` and `DistinctIncrementColumn` are cumulative and therefore order-dependent. Two runs
that process the same rows in different orders produce different `logcount` and `locincrement` values,
which produce different features, which produce different scores. This is the most easily overlooked
determinism defect in the entire pipeline, because it produces plausible output.

Define and enforce a total order before any cumulative feature is computed:

```python
df = df.sort_values(["event_time", "collector_id", "collector_seq"], kind="mergesort")
```

`event_time` alone is insufficient because ties are common at second or millisecond resolution.
`collector_id` plus `collector_seq` breaks every tie deterministically, which is why Part 2 requires
`collector_seq` to be strictly monotonic per collector. `kind="mergesort"` selects a stable sort, so rows
that compare equal retain their relative order rather than being permuted arbitrarily.

`morpheus.utils.determinism.sort_for_cumulative_features` is that sort, with one addition: it rejects
ties by default rather than falling back on stability. A stable sort makes the output a function of the
input arrangement, which is the property determinism is supposed to remove, so leftover ties mean the
order columns do not identify a row and the caller is told rather than handed plausible output.

Apply the same discipline to the rolling window. `CachedUserWindow.append_dataframe` already computes
`_row_hash` via `pd.util.hash_pandas_object` and uses it to find the boundary between seen and unseen
rows; combined with a `max_history` expressed as a duration rather than a row count, window membership
becomes a pure function of `(entity, window_id, lateness_horizon)`. A row-count `max_history` is
order-dependent and should not be used in a deterministic pipeline.

### Control 9: Quantize scores and use hysteresis

Float32 GPU reductions are not associative, so `mean_abs_z` can differ in the last bits between runs.
This is invisible until a score sits within a few ULPs of a threshold, at which point the decision flips
and D1 is violated even though D0 was nearly satisfied.

Two mitigations, and use both:

**Quantize before comparing.** Round the decision variable to a fixed number of decimal places, chosen so
that the quantum is larger than the observed float noise but far smaller than any meaningful difference:

```python
score_q = float(decimal.Decimal(score).quantize(decimal.Decimal("0.0001"),
                                                rounding=decimal.ROUND_HALF_EVEN))
```

`ROUND_HALF_EVEN` is specified explicitly because the default rounding mode differs between platforms
and libraries. Emit both `score` and `score_q`; rules compare `score_q`.

**Add hysteresis.** An entity enters the alerting state at `threshold` and leaves it only at
`threshold - delta`. This eliminates the flapping that occurs when an entity's score oscillates around
the boundary, which is a distinct problem from float noise and much more common in practice. Both
`threshold` and `hysteresis` appear in the rule metadata block in Part 3 and in `config_hash`.

### Control 10: Keep LLM stages out of the decision path

`morpheus_llm` is genuinely useful for enrichment: summarizing an alert, retrieving related context from
a vector database, drafting an investigation note. It is not deterministic. Temperature 0 reduces
variance but does not eliminate it, provider-side model updates change behavior without notice, and
retrieval results shift as the vector store is updated.

The rule is simple: **no LLM output may be an input to a detection decision.** LLM stages consume the
decision and add narrative. When one runs, record `llm_model`, `llm_prompt_hash`, and
`llm_response_hash` so the narrative is at least auditable even though it is not reproducible, and mark
the enriched fields with a `nondeterministic: true` flag so no downstream rule accidentally keys on them.

Note that `LLMEngineStage` already sets `pe_count = 1`, which helps with ordering but does nothing for
output stability.

### Control 11: Idempotent sinks

Retries and replays must not duplicate. Use `event_uid` as the primary key at every sink:

- Elasticsearch: set `_id = event_uid` so a retry overwrites rather than appends.
  `ElasticsearchController.df_to_parallel_bulk_write` builds actions from the DataFrame; include `_id`
  in the action dict.
- Splunk: HEC has no native deduplication. Deduplicate at search time on `event_uid`, or run the
  Kafka connector with exactly-once semantics enabled.
- Kafka: enable the idempotent producer and set a message key of `event_uid` so compaction and retry
  behave correctly.

`ElasticsearchController.parallel_bulk_write` uses `elasticsearch.helpers.parallel_bulk`, which does not
preserve order. That is acceptable precisely because `event_uid` makes writes idempotent and order-
independent; without it, the parallel bulk write would be a determinism defect.

### Control 12: The determinism envelope

Every emitted event carries a fixed block of fields that make it reproducible. This is what turns
determinism from a claim into a verifiable property:

```json
{
  "determinism": {
    "tier": "D1",
    "pipeline_fingerprint": "a3f9c2e1b8d47506",
    "config_hash": "7d2e4a1f9c3b5e80",
    "code_commit": "c6a3b56",
    "image_digest": "sha256:1f0c...",
    "model_version": "dfp-jdoe:14",
    "model_fallback_used": false,
    "feature_schema_version": "TC-5/2.1.0",
    "rng_seed": 42,
    "window_id": 484512,
    "window_start": "2025-07-29T14:00:00Z",
    "window_end": "2025-07-29T15:00:00Z",
    "revision": 1,
    "lateness_horizon_s": 900,
    "batch_policy": "entity_window",
    "score_quantum": "0.0001"
  }
}
```

Twelve fields of overhead per event is not free at billions of events per day. Two mitigations: put the
block on the scored and alerted events only, not on raw telemetry, and intern the static portion into a
`pipeline_fingerprint` lookup so only the fingerprint and the varying fields travel with each event.

### Control 13: Prove it in CI

Determinism claims decay silently. Enforce them:

1. **Golden corpus.** A fixed input set per layer, checked into the repository or an artifact store,
   covering normal traffic, each rule's positive case, and the edge cases (empty windows, single-row
   entities, clock-skewed records, late arrivals).
2. **Double-run diff.** Run the pipeline twice over the corpus in the same container and diff the
   outputs. Any difference at the declared tier fails the build. This catches the majority of defects,
   including every wall-clock read and every unseeded RNG.
3. **Cross-restart diff.** Run twice in separate container instances. Catches `PYTHONHASHSEED`
   dependence and anything captured from the environment.
4. **Assertion via existing stages.** `CompareDataFrameStage` and `ValidationStage` exist for exactly
   this and need no new code:

   ```python
   pipeline.add_stage(CompareDataFrameStage(config, compare_df="golden/l5_expected.csv"))
   ```

5. **Batch-size sweep.** Score the corpus at three `model_max_batch_size` values and assert equality.
   This is control 5's verification step, automated.
6. **Permutation test.** Shuffle the input order within a window and assert identical output. This is
   the direct test for control 8, and it is the one that catches an accidentally-removed sort long after
   the fact.

Run 1 and 2 on every commit; run the rest nightly.

All six checks ship, implemented against the reference lineage pipeline in
`tests/morpheus/determinism/`, with the comparison half factored into
{py:mod}`~morpheus.utils.determinism` for reuse against any pipeline: `canonicalize` reduces output to
a normal form in which two deterministic runs compare equal, `diff_frames` explains the first
disagreement in build-log terms, `frame_digest` gives the one-line D0 verdict, and
`permute_within_contiguous_groups` produces the legal input shuffles for check 6. The corpus is seeded
code rather than checked-in data, which keeps it out of LFS while remaining exactly as fixed, and the
golden output is a checked-in CSV regenerated deliberately, never silently, via
`tests/morpheus/determinism/run_lineage_pipeline.py`. The cross-restart check runs that same driver in
fresh interpreters with two different `PYTHONHASHSEED` values, which is the variation an in-process
double run cannot see.

One trap in check 6 deserves calling out, because the shipped harness initially fell into it.
Canonicalization sorts output rows before comparing, so a permutation of the input can only be detected
through a *value* that depends on row order. A pipeline with no cumulative features passes the
permutation check unconditionally, sort or no sort, and the check proves nothing. The reference
pipeline therefore derives an `IncrementColumn`-style ordinal (`window_seq`) inside each sealed
window, and the harness carries a negative control that reintroduces the removed-sort defect and
asserts the check catches it. Apply the same discipline to any pipeline this harness is pointed at: if
none of the compared columns is order-derived, check 6 is decoration, and a negative control is the
only way to know.

### What cannot be made deterministic

Being straightforward about the boundaries is part of the design:

- **Wall-clock latency and throughput.** Inherently variable. Never build a rule on them.
- **External enrichment.** Threat intelligence lookups, DNS resolution, and geolocation return
  different answers over time. Snapshot the enrichment source, version it, and record the version, so the
  result is reproducible against a stated snapshot even though it is not reproducible against the live
  service.
- **LLM output.** Covered in control 10.
- **Anything derived from `ingest_time`.** By construction.
- **Cross-version model comparison.** Two model versions produce different scores. That is correct
  behavior, not a defect, but it means score trends must be segmented by `model_version` or the
  retraining event will appear as a behavioral shift across the entire population simultaneously. This is
  a common and embarrassing failure mode, and the fix is to emit `model_version` and require every trend
  visualization to break out by it.

---

## Part 6: Gaps and Build List

What Morpheus provides versus what has to be built, stated plainly.

### Provided

- Streaming pipeline runtime with GPU acceleration and segment-level distribution.
- Declarative feature engineering DSL with behavioral novelty primitives (`column_info.py`).
- Per-entity model training, registry integration, caching, and inference (`morpheus_dfp`).
- Tabular autoencoder with per-feature attribution output (`dfencoder`).
- Periodicity-based anomaly detection (`TimeSeriesStage`).
- Line-rate packet capture into GPU memory (DOCA).
- Protocol and log parsers for IP, Zeek, Windows events, URLs, and Splunk notables.
- Kafka, Elasticsearch, HTTP, file, and Delta Lake sinks.
- Pretrained models for sensitive information detection, phishing, log parsing, ransomware, fraud, and
  anomalous behavior profiling.
- Seeding utility and golden-file comparison stages.
- Deterministic lineage identifiers ({py:mod}`~morpheus.utils.lineage` and
  {py:class}`~morpheus.stages.lineage.lineage_stamp_stage.LineageStampStage`) and the Community ID flow
  hash ({py:mod}`~morpheus.utils.community_id` and
  {py:class}`~morpheus.stages.lineage.community_id_stage.CommunityIdStage`).
- Time-bounded soft joins with a fixed tie-break, in-pipeline resolution, and bucketed lookup generation
  ({py:mod}`~morpheus.utils.binding_table` and
  {py:class}`~morpheus.stages.lineage.binding_resolver_stage.BindingResolverStage`).
- The Splunk side of Part 4 as an installable app: indexes, sourcetypes, KV Store binding lookups,
  and the scheduled searches ([`examples/splunk_lineage_app`](../../../../examples/splunk_lineage_app/README.md)).
- Deterministic window sealing with a lateness horizon, revision numbering, and a late-arrival stream
  ({py:mod}`~morpheus.utils.window_seal` and
  {py:class}`~morpheus.stages.lineage.window_seal_stage.WindowSealStage`).
- The determinism CI harness: control 13's six checks running against the reference lineage pipeline
  over a seeded golden corpus ({py:mod}`~morpheus.utils.determinism` and
  `tests/morpheus/determinism/`).
- TC-1 counter normalization: monotonic interface counters turned into per-interval deltas, with a
  counter wrap distinguished from a device reboot, and the `site_id:device_id:port_id` entity key
  ({py:mod}`~morpheus.utils.counter_delta` and
  {py:class}`~morpheus.stages.telemetry.tc1_normalize_stage.TC1NormalizeStage`).
- The TC-1 novelty features, transceiver substitution and neighbor change, with control 8's total order
  imposed before the cumulative primitives run ({py:mod}`~morpheus.utils.tc1_features`,
  {py:class}`~morpheus.stages.telemetry.tc1_feature_stage.TC1FeatureStage`, and
  `morpheus.utils.determinism.sort_for_cumulative_features`).
- TC-1 optical power deviation against a per-port rolling baseline
  ({py:mod}`~morpheus.utils.optical_baseline` and
  {py:class}`~morpheus.stages.telemetry.tc1_optical_stage.TC1OpticalStage`).
- TC-1 link flap counting, including the flaps that begin and end between two polls
  ({py:mod}`~morpheus.utils.link_flap` and
  {py:class}`~morpheus.stages.telemetry.tc1_flap_stage.TC1FlapStage`). This completes the four
  behavioral features the TC-1 section names.
- Identifier change detection with no period boundary ({py:mod}`~morpheus.utils.value_novelty` and
  {py:class}`~morpheus.stages.telemetry.tc1_change_stage.TC1ChangeStage`), which closes the blind spot
  a period-bucketed distinct count can only narrow.
- TC-2 binding closure: layer 2 observations turned into the closed, half-open intervals
  `BindingTable` resolves against, with the reason for every inferred end recorded
  ({py:mod}`~morpheus.utils.binding_closer` and
  {py:class}`~morpheus.stages.telemetry.tc2_binding_stage.TC2BindingStage`).
- The three TC-2 cardinality features, distinct MACs per port, ports per MAC, and OUIs per VLAN, over a
  trailing window with saturation reported rather than hidden
  ({py:mod}`~morpheus.utils.distinct_window` and
  {py:class}`~morpheus.stages.telemetry.tc2_cardinality_stage.TC2CardinalityStage`).
- The remaining two TC-2 behavioral features: the gratuitous ARP proportion with the multi-claimant
  count R-D-L2-003 needs ({py:mod}`~morpheus.utils.ratio_window` and
  {py:class}`~morpheus.stages.telemetry.tc2_arp_stage.TC2ArpStage`), and 802.1X authorization timing
  with unpaired authorization flagged ({py:mod}`~morpheus.utils.session_timer` and
  {py:class}`~morpheus.stages.telemetry.tc2_auth_stage.TC2AuthStage`). This completes the five
  behavioral features the TC-2 section names.
- Control 8 as a stage ({py:class}`~morpheus.stages.lineage.total_order_stage.TotalOrderStage`), placed
  once ahead of the first stateful stage. The telemetry stages flag out-of-order arrival rather than
  repairing it, and this is what imposes the order they depend on.
- The composed telemetry pipeline under control 13's six checks
  (`tests/morpheus/determinism/telemetry_pipeline.py`): a snapshot-shaped layer 1 and layer 2 corpus with a
  hub, a spoof, an ARP flood, a reboot, a tap, an unpolled flap and an 802.1X bypass planted in it, run
  through every TC-1 and TC-2 stage, with the layer 2 bindings resolving the ARP stream onto the layer 1
  `entity_key`. Each planted anomaly is asserted as the column a rule would read, and nothing else fires.

- The four layer 2 detections, R-D-L2-001, 003, 004 and 005, as saved searches in the Splunk app, with
  their predicates asserted in Python over the planted corpus. 001 and 003 ship with the hook for the
  list each depends on and fire on nothing until it is populated.
- Provisional open bindings (`TC2BindingStage(emit_open_bindings=True)`), so live attribution has an
  answer inside the idle window, capped by a duration the consumer states rather than one the stage
  invents.
### Must be built

| Component | Effort | Notes |
| --- | --- | --- |
| Entity sharding router configuration | Small | `RouterStage` with a stable hash; replaces intra-stage threading |
| TC-1 and TC-2 collectors | Medium | The SNMP, LLDP, DHCP, and 802.1X polling itself. Tier 1 is not Morpheus; the counter normalization those collectors feed does ship, as `TC1NormalizeStage` |
| Binding table ingestion | Small | Refreshing `BindingTable` on a schedule and loading it into the SIEM. The resolution and expansion logic ships, and so does the closing of open bindings into resolvable intervals (`TC2BindingStage`) |
| Splunk sink or connector configuration | Small | Kafka Connect is the recommended path |
| Chained rule engine | Medium | Runs in Splunk, not in Morpheus. The `examples/splunk_lineage_app` searches are the starting set |
| Bitemporal TC-0 context store | Medium | Valid-time and transaction-time intervals |

### Sequencing recommendation

Do not build all seven layers at once. The dependency structure and the value distribution both argue for
this order:

1. **Layers 5 and 7 first.** `morpheus_dfp` applies directly, the entity (user) is unambiguous, the
   cardinality is tractable, and these two layers produce most of the standalone detection value. Ship
   R-B-L5-001, R-B-L5-006, and R-B-L7-002.
2. **The lineage substrate second**, connecting those two layers only. Prove the `event_uid` and
   `lineage_id` construction, the Splunk edge index, and the chain query on a two-layer chain before
   scaling it to seven. R-C-004 is the target.
3. **Layers 3 and 4 third.** Highest volume, so it benefits most from the operational lessons of steps 1
   and 2. `community_id` from day one. The `abp_pcap_detection` example is the template.
4. **Layer 6 fourth.** Small, high-signal, and cheap once layers 3 and 4 exist, since it rides the same
   collection points. R-B-L6-001 is one of the best rules in the set relative to its cost.
5. **Layers 1 and 2 last.** Highest collection effort, lowest standalone detection value, but they
   complete the ladder and enable R-C-003 and R-C-005, which are not expressible any other way.

Apply the D1 and D2 determinism controls from the start. Retrofitting determinism onto a running
detection pipeline means re-tuning every threshold, because the scores will move.
