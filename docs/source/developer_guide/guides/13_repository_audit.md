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

# Repository Audit: Every File Against the Objective

This document records an audit of every tracked file in the repository, taken on 2026-10-23 at the commit that
regenerated the real-format fixtures (7cd4bb7e). Each file was asked two questions: **what is its purpose**, and
**why is it here**. Each then got one verdict on whether it serves the future of the objective the
[README](../../../../README.md) states: behavioral analytics across all seven OSI layers, feeding a SIEM, with output
a detection engineer can reproduce and defend months later.

The answer for most of the repository is that it was never there for this objective. The fork diverged from upstream
Morpheus at commit c6a3b56d (2026-05-06) and inherited every upstream use case with it: language models and
retrieval, DOCA packet capture, ransomware, phishing, fraud graphs, log parsing, the Triton inference path and the
organization automation that builds and ships them. The fork's own work sits on a thin, well-defined slice of that
inheritance. This audit names the slice, marks the rest for removal, and lists the edits the removal forces on what
stays.

Nothing is deleted by this change. The removals are marked, in a manifest a test holds against the tree, and carried
out in later changes the repository owner approves one ruling at a time.

## Method

### Inventory and classification

The inventory is `git ls-files`: 2,407 files. Each was classified against the fork point:

- **`fork_added`**: 436 files the fork added.
- **`fork_modified`**: 18 upstream files the fork changed.
- **`upstream`**: the rest, untouched since c6a3b56d.

A static import closure over the fork's Python code found the 94 upstream modules the fork reaches directly or
transitively. That closure is a hint, not a proof: it misses dynamic imports, command-line registration and C++, so
every reader was told to look for a dependent by name before calling anything unused.

### Two passes, sixty readers

**Pass 1** split the inventory into 36 partitions. One reader per partition read every file in it: text files under
200 KB whole, larger text files by their head, tail and line count, binary files by type, size and every reference
to their name. Each reader wrote one record per file with its purpose, why it is here, a verdict, the reason, the
change it calls for, what depends on it, and anything the objective needs that the file's area does not provide.

Where readers in different partitions answered the same question differently, the conflict was settled once, as a
ruling, before pass 2 began. The rulings are below.

**Pass 2** split the inventory differently, into 24 partitions, so that no file had the same reader twice and every
cross-partition dependency had a second chance to be seen. Each reader re-read every file against its pass-1 record
and the rulings, and either agreed or recorded the disagreement with its evidence. Pass 2 changed 165 verdicts. The
final verdict is pass 2's, with one reconciliation: ruling D10 makes all thirteen corpus pipeline modules `update`,
and six of them still read `keep`.

### Verdicts

- **`keep`**: serves the objective and is current.
- **`update`**: serves the objective but is stale, wrong, duplicated or misplaced; the record says exactly what to
  change.
- **`keep-dependency`**: upstream code, build, packaging or test scaffolding that the fork's pipelines, command line,
  tests, documentation build or CI need, directly or transitively, on the CPU or the GPU. The record names the
  dependent.
- **`remove`**: not suitable for the future of the objective, and nothing on the objective's path uses it, or a
  fork file that something newer superseded.

A kept file that a removal forces to change is `update`, never `keep-dependency`, so the removal and its edits land
together.

## Result

The manifest is `ci/artifacts/repository_audit_manifest.csv`: one row per tracked file with its class, both passes'
verdicts, the final verdict, purpose, why it is here, reason, action, dependents and needs. The files marked for
removal are listed one per line in `ci/artifacts/repository_audit_removals.txt`. Both cover all 2,412 tracked files,
the 2,407 the passes read and the five this audit adds: the two manifests, this guide, its test and the shared
corpus pipeline helper.

| Verdict | Files |
| --- | --- |
| `remove` | 1476 |
| `keep-dependency` | 413 |
| `keep` | 375 |
| `update` | 148 |

Most of the removal is upstream code the fork never touched: 1,467 of the 1,476 files. Six are upstream
organization workflows the fork modified, and three are its own: `TC1FeatureStage`, its utility module and
its test, which `TC1ChangeStage` superseded.

Between the passes, the second readers moved files in both directions:

| Pass 1 | Pass 2 | Files |
| --- | --- | --- |
| `keep` | `update` | 49 |
| `keep-dependency` | `remove` | 40 |
| `remove` | `keep-dependency` | 28 |
| `keep-dependency` | `update` | 24 |
| `keep` | `keep-dependency` | 7 |
| `update` | `remove` | 6 |
| `remove` | `update` | 5 |
| `update` | `keep-dependency` | 4 |
| `keep-dependency` | `keep` | 1 |
| `keep` | `remove` | 1 |

The largest single movement, `keep` to `update`, is the second readers finding stale counts, stale docstrings and
duplicated helpers in files the first readers had accepted. The movement out of `remove` is upstream test scaffolding
and fixtures whose dependent the first reader had not found: the HTTP server test behind the kept HTTP stages, the
fixture `test_file_source_stage_pipe` reads, the example data guide 1 runs on.

## The Rulings

Each ruling settles a question at least two readers answered differently, and each follows from the objective and
from evidence a reader recorded.

**D1. Upstream use cases go, with every edit their removal forces.** Language models and retrieval
(`python/morpheus_llm`, `examples/llm`, `tests/morpheus_llm`, the Milvus and `kinetica` fixtures), DOCA, ransomware and
AppShield, phishing, SID visualization, graph fraud detection, data loss prevention, log parsing, root cause, ABP
`nvsmi`, the upstream digital fingerprinting production example, the RabbitMQ and C++ developer guide examples, and
the `external/morpheus-visualizations` submodule. The edits that go with them are recorded on the kept files they
touch: CMake options, the documentation build, `conf.py`, `py_api.rst`, the stages page, the examples indexes,
`dependencies.yaml`, `.gitattributes`, `.gitmodules`, `tests/conftest.py` fixtures and the `_lib` bindings.

**D2. The Triton inference path goes; the two rules that need a model get a new approach.** R-B-L4-001 (the ABP
packet-capture booster) and R-B-L7-003 (the SID model) are the only design rules that need inference. The upstream
path cannot meet the objective: `TritonInferenceStage` resolves a model by name with no digest, dynamic batching
makes a score depend on what else was in the batch, `InferenceStage` imports cuDF so there is no CPU path, and CI can
run none of it. The inference stages, preprocessing, inference and response memories, add-scores,
add-classifications, filter-detections, their C++ and bindings, the mock Triton server, the Triton model repository
and the model containers all go. The two model artifacts the rules name are kept and moved, as `update`, into a fork
model inventory with a recorded digest and provenance: the ABP booster and its configuration, and the SID model with
its vocabulary, hash and labels.

**D3. The generic runtime stays, with its upstream tests.** The pipeline runtime, messages, configuration, the
command-line framework, and the source, sink and transform stages a deployment composes: file, in-memory, Kafka and
HTTP sources; monitor, serialize, deserialize, router, trigger, validation, write-to-file, write-to-Kafka and HTTP
sinks; linear and multi-processing stages. Elasticsearch stays as the design guide's fourth sink option. Stages that
exist only for a removed use case go with it.

**D4. Upstream DFP pieces stay as evidence, for now.** `examples/upstream_reuse` and its test import six
`morpheus_dfp` stages, three of its utilities, `MLFlowDriftStage`, `TimeSeriesStage`, the Azure, Duo and CloudTrail
sources and the MLflow writer controller. They are the measured evidence behind the guide's decision not to reuse
them, so they and the package files that build them are `keep-dependency`. The rest of `morpheus_dfp` and
`tests/morpheus_dfp` go.

**D5. The fork's superseded stage goes.** `TC1FeatureStage` and its utility module, with their test, command-line
entry, liveness entry, the comparison in the `TC1ChangeStage` test, the stages page line, the guide's citations and the
CPU tier line.

**D6. Upstream CI and organization automation go; the fork's CI stays.** CODEOWNERS, the bots, project automation,
`ci_pipe.yml`, `pr.yaml`, `ci/scripts/github`, `ci/release`, `ci/runner`, `ci/docker`, the Conda recipes other than
`morpheus-core`, and the upstream development container (`.devcontainer`), whose Triton and Kafka helpers serve the
removed paths. The C++ lint configuration stays because the core library stays.

**D7. Upstream documentation stays where it documents the kept framework.** The architecture and CPU-only pages,
developer guide 1 and its example, guide 9, the stages page, the API reference and the Sphinx build. Guides 2 to 8
and 10, the module and loader pages and the stubs of removed examples go, after guide 6's input schema narrative and
guide 2's constructor section move into the fork's documentation.

**D8. A stale fork file is `update`, never `remove`, unless something superseded it.** That covers stale counts and
docstrings, duplicated helpers, hand-kept lists a test could derive, and deployable modules that live in the test
tree.

**D9. Citations of removed upstream files become permalinks** at upstream commit c6a3b56d, so the provenance outlives
the file.

**D10. Settled in pass 2.** Filter-detections goes with D2, and guide 11's emit step points at `SiemWireStage`
instead. The data-loader stack goes. The SID labels move with the model. All thirteen corpus pipeline modules are
`update`, for one shared helper.

## Folder Structure

The tree today is the upstream tree, with the fork's work threaded through it. After the removals it becomes a
small framework with one application on top, and the structure should say so.

| Area | Files | Today | After the removals |
| --- | --- | --- | --- |
| `python/morpheus/morpheus/stages/telemetry`, `stages/lineage`, `utils` | 133 | The fork's stages and their trackers, mixed into the upstream package. | Unchanged in place: they are the product. The trackers that live inside stage files move to `utils`, so every tracker is one module with one test. |
| `python/morpheus/morpheus/_lib` | 196 | The upstream C++ core: messages, sources, the inference stack, DOCA, loaders. | The messages, I/O and the sources and sinks D3 keeps. 89 files go. |
| `python/morpheus_llm`, `python/morpheus_dfp` | 134 | Two upstream packages. | `morpheus_llm` goes. `morpheus_dfp` shrinks to the D4 evidence and goes entirely once its source-text checks become citations. |
| `examples` | 641 | Fourteen upstream examples, five fork examples and the upstream shared data. | The five fork examples, guide 1's example and the data it reads. 507 files go. |
| `tests/morpheus/determinism` | 78 | Corpus builders, composed pipelines, harnesses, golden files, and three deployable modules. | Corpora and harnesses only. The compositions, sharding, resumable pipeline and host-frame helper move into the package. |
| `tests` (rest) | 585 | The upstream tests for every use case, the mock Triton server and its payloads. | The fork's stage and utility tests and the D3 runtime tests, every one in a CI tier. 368 files go. |
| `models` | 169 | The upstream model zoo, training scripts and datasets. | The Public Suffix List and what `upstream_reuse` reads. The two D2 models move to a fork model inventory. |
| `docs/source` | 108 | The upstream site with guides 11, 12 and this one added. | The kept framework pages and the fork's guides, built and checked in CI. |
| `ci`, `.github`, `conda`, `docker`, `.devcontainer`, `scripts` | 135 | The upstream release and organization automation beside the fork's CPU tiers. | The fork's CI, its lock file and overlay, the development image and the core recipe. |

Two structural needs come out of the folder view rather than any single file. Deployable code lives in the test
tree, so a deployment imports from `tests/`; and the examples import corpus modules from `tests/` through a path
insertion. Both are plan items below.

## Carrying Out the Removals

The removals land as one change per ruling, in this order, each with the `update` edits its records name:

1. **D5**, the smallest and entirely the fork's own.
2. **D1**, use case by use case if the change is too large to review at once.
3. **D2**, after the fork model inventory exists to receive the two models.
4. **D6**, then **D7**.
5. **D4**'s remainder, after the evidence is cited by permalink.

For each, the paths come from the removal list, selected by the ruling's directories, and the edits from the
`action` column of the rows that name the ruling. For the language-model part of D1, for example:

```bash
grep -E '^(python/morpheus_llm|examples/llm|tests/morpheus_llm)/' ci/artifacts/repository_audit_removals.txt \
    | git rm --pathspec-from-file=-
```

The audit test then fails until the manifest rows of the removed files are dropped and the guide's counts restated,
which is the point: the manifest describes the tree it sits in, not the tree it used to.

## Defects the Audit Found

Reading every file for purpose turned up defects that no test had caught. These three were reproduced, fixed and
given tests in this change:

- **`SiemWireStage` let a record with no event time through when composed in a pipeline.** The stage declared its
  time columns as needed columns, so the segment's source preallocated an absent anchor as an empty string, and the
  stage's presence check passed it. The SIEM then stamps such a record at index time, the quietest failure the stage
  exists to prevent. Calling the stage directly raised, which is what every test did. The stage no longer declares
  them, and a test now composes it behind a source.
- **`pseudonymize` raised on the missing marker of a pandas extension column, and gave one identifier two pseudonyms.**
  `pandas.NA` compared with itself raises, and a `string` or `Int64` column hands it out. An integer
  identifier widened to float by one missing row rendered as `5.0` instead of `5`, and hashed differently. Nulls are
  now tested with one helper shared by the minimization stage, the stack fingerprint and the user-agent parser, and a
  whole number renders as an integer whatever type carries it.
- **`TC6CertificateStage` and the fork's Zeek normalizer disagreed on what self-signed means.** The normalizer emits
  `self-signed-chain` for a self-signed root in the chain, which the stage did not count, and the stage counted
  OpenSSL's `unable to get local issuer certificate`, which is an issuer the verifier does not hold. A test now feeds
  every value the normalizer emits through the stage.

These were reported by a reader and are recorded on their files for the change that touches them:

- The C++ deserializer treats a first `INT64` column named `id` as the index, which the Python reader does not.
- The Kafka sink's delivery callback reads the value of a message that may be `None`, and a failed produce is
  logged, not retried; the Kafka source commits a batch it has not yet parsed.
- `TC7ServiceStage`'s window sums stop counting at their sample limit without saying so.
- The layer 3 corpus's `origin_hash` covers the source and destination only, while the collector contract says
  the flow tuple and its start.

## Updates Applied in This Change

Of the 148 `update` records, 61 are applied in full in this change, 8 in part, and one was checked and needs no
change; the manifest's `status` column says which. They do not wait for any removal:

- **Defects:** the three above, and two more found while applying the records. `TC4FlowStage` evicted the flow
  opened first at its bound rather than the least recently seen one its docstring promises, and stamped a record for
  a bin it had passed with the bin's ordinal where every other row carries nanoseconds.
- **Duplication:** the thirteen copies of the corpus collector and pipeline configuration are one module,
  `tests/morpheus/determinism/corpus_pipeline.py`. A host-score copy that dropped empty batches, and the lineage
  copy that sets the C++ switch, are parameters of it rather than silent differences. The Splunk field contract test
  finds every golden file by glob, and with it the host-score golden it had been missing.
- **Stale text:** docstrings in a dozen trackers and stages, counts in the corpus modules, harnesses, examples and
  Splunk app comments, the Splunk validation container pinned to the 10.2.8 image every recorded run used, and
  upstream citations turned into permalinks.
- **Fork metadata:** the issue forms and pull request template point at the fork, the repository helper derives
  owner and name from the remote, and the lock file's header names the channels its packages come from.
- **Documentation:** the glossary, known issues and troubleshooting pages carry the fork's entries, and two
  registered stages that had no command-line entry have one.

The other 78 wait for the removal they belong to: their record's `action` names the ruling.

## Plan: New Approaches for What the Objective Needs

Every reader recorded what the objective needs that the file's area does not provide. Collected across both passes
and stripped of duplicates, these are the needs the current approach cannot meet by being edited, and the approach
that replaces it. Each item says how to know it is done.

1. **Score the two model rules on a digest-pinned committed model, on the CPU and the GPU alike.** R-B-L4-001 and
   R-B-L7-003 today depend on the Triton path D2 removes. The new approach is layer 5's: a model file committed with
   its manifest, loaded by digest, scored through the `Scorer` protocol in the stage that owns the features. The
   booster scores through a CPU XGBoost runtime; the SID model needs a CPU tokenizer over the kept vocabulary and an
   ONNX runtime. Done when both rules fire on a corpus plant in the CPU tier, with the digest in the determinism
   envelope.
2. **A fork model inventory.** One directory and one manifest for every model the fork runs: the layer 5 session and
   host models, the ABP booster and the SID model, each with its digest, provenance (the upstream path and commit for
   the moved ones), input features and threshold. Done when a test refuses a model file the manifest does not name or
   whose digest differs.
3. **Pin every reference dataset in the determinism envelope.** The Public Suffix List and the time zone database
   change under a running pipeline and change its output. Record each one's digest in the envelope as the models'
   already are. Done when a changed list changes the envelope and a test says so.
4. **Ingest from Kafka and HTTP with provenance and without silent loss.** The kept sources drop the topic, partition
   and offset, the sink sends no message key, and a failed produce is logged and lost. Carry offset and partition
   into the lineage columns, key messages on `entity_key`, retry or route a failed produce to the quarantine stream,
   and put `TotalOrderStage` directly behind a multi-partition source. Done when the transport harness replays a
   two-partition topic in two orders and gets one output.
5. **Move deployable code out of the test tree.** The compositions, the shard split stage, the resumable pipeline
   and the host-frame helper are imported by deployments from `tests/`. Move them into the package, with the corpora
   and harnesses staying behind, and drop the path insertions. Done when nothing outside `tests/` imports from
   `tests/`.
6. **One copy of each helper.** The corpus collector and pipeline configuration are one module as of this change;
   three readers of `savedsearches.conf` and thirteen identical runner scripts remain. Done when each exists once
   and a test finds no second definition.
7. **Derive what is counted by hand.** Stage counts, rule counts, golden file lists and number words in prose drift
   every time something is added; this audit found them in tests, docstrings, README files and the Splunk app's comments. Tests should read the count from the source of
   truth, and prose should say "every" rather than a number where it can. Done when no test keeps a hand list of
   files a glob could find.
8. **A base class for the telemetry stages, with an event-clock guard.** The stateful stages of TC-3 to TC-7 repeat
   the same column checks, state hooks and null-preserving assignments, and only `TC5SessionStage` refuses an event time far
   in the future. One base class carries the guard for all of them. Done when one far-future row cannot expire every
   window in any stage.
9. **Exercise the GPU path end to end in CI, or say on every run that it was not.** The runners force the CPU, and
   the C++ file source ignores the parser arguments the Python one honors. Add a GPU job when a runner exists, and
   until then make the conformance artifact's date part of the CI summary. Done when the GPU verdict is never older
   than the code it describes without saying so.
10. **Build and test what stays in C++.** No CI tier builds the C++ tests, and the CUDA utilities carry an
    architecture workaround. Done when the kept C++ tests run in a tier.
11. **One environment definition.** `dependencies.yaml` and the CPU lock disagree on channels and pins. Generate the
    lock from the dependency file, or make the lock the only definition. Done when one regeneration produces both.
12. **Real-format fixtures for layers 1, 2 and 5, and one input-sample convention.** Layers 6 and 7 have them; the
    rest are synthetic only, and the input samples spell `schema_version` and `origin_hash` differently by class.
    Done when every class has a real-format fixture and the input contract test enforces one convention.
13. **Compose `EdgeEmitStage` and `PipelineHealthStage` in a runner.** Both are built and tested, and no runner uses
    them. Done when a segment runner emits both streams.
14. **Documentation the fork owns.** A contributor guide and a getting-started page for the fork's CPU path, the
    model inventory page, and the docs build, Vale and lint in the fork's CI. Done when a pull request that breaks the
    docs build fails CI.
15. **A throughput baseline.** No benchmark measures the telemetry and lineage stages. Done when a benchmark tier
    records rows per second per stage and a change that halves it is visible.
16. **Restore the fork's CI.** No fork CI run has been recorded since 2026-10-12, and the pull requests merged since
    were not checked. The cause is a setting, not the workflow: a manual start on 2026-10-24 was refused because
    Actions is disabled for the repository. Re-enable Actions under the repository's Settings, Actions,
    General. Done when every pull request shows a CPU tier result.

## Keeping the Manifest Honest

`tests/morpheus/determinism/test_repository_audit.py` holds the manifest to the tree. Every tracked file must have
exactly one verdict with its purpose, why it is here and reason filled in; the removal list must be the manifest's
`remove` rows; no file marked `keep` or `keep-dependency` may import a file marked `remove`, so carrying out the
removals cannot break what stays; every `update` row that imports a file marked `remove` must say what to change;
and the counts in this guide must be the manifest's. A file added later fails the first check until it is given a
verdict, which is how the audit stays an audit rather than a snapshot.
