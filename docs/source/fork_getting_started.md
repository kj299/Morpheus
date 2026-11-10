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

# Getting Started with This Fork on a CPU

This page takes you from a fresh clone to a running pipeline and a passing test run on a machine with no GPU. It
covers this fork's own work: the behavioral analytics pipelines for the seven network layers and their output for a
Security Information and Event Management (SIEM) system. For upstream Morpheus on a GPU, with its containers and
Conda packages, see [Getting Started with Morpheus](./getting_started.md).

## How the CPU path works

Morpheus has a compiled library that this fork does not change: everything the fork adds is Python, so there is
nothing to build. You install the released `morpheus-core` package for its compiled library, then copy this
repository's Python sources over the installed package. That copy is called the *overlay*, and you repeat it after
every edit to the package's sources.

Every pipeline here runs in CPU execution mode, on pandas. The same pipelines also run on a GPU, on cuDF; that path
is checked separately, on a machine with a card, and is not needed to follow this page.

## What you need

- Linux on x86-64. The environment is pinned for that platform only.
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), to create the
  environment.
- Git, and Git LFS if you want the two shared models (see step 2).
- About 6 GB of disk for the environment.

## 1. Clone the repository

```bash
git clone https://github.com/kj299/Morpheus.git
cd Morpheus
export MORPHEUS_ROOT=$(pwd)
```

## 2. Fetch the models, if you need them

Two of the detection stages score with models stored in Git LFS under `models/inventory`. Fetch them if you will
run the transport or application pipelines or the whole test suite; skip this step otherwise.

```bash
git lfs install --local
git lfs pull --exclude="" --include="models/inventory/**"
```

The `--exclude=""` is needed because the repository's `.lfsconfig` excludes every LFS file by default.
[Models](./models_and_datasets.md) describes what these files are and how each score is tied to them.

## 3. Create the environment

```bash
micromamba create -n morpheus -f ci/scripts/cpu_env/linux-64-cpu.lock
micromamba activate morpheus
```

The lock file lists every package with its exact build, so you get the environment the tests were run in, with no
dependency solving on your machine.

## 4. Overlay the fork's sources

```bash
python ci/scripts/cpu_env/overlay_fork_sources.py "${MORPHEUS_ROOT}"
```

Run this again whenever you change a file under `python/morpheus/morpheus`. It leaves the compiled library alone.

## 5. Run a pipeline

Each group of layers has a runner under `examples/behavioral_analytics`. With no arguments a runner reads the
sample input committed beside it. This runs the layer 1 and 2 pipeline over an hour of sample switch telemetry:

```bash
python examples/behavioral_analytics/run_telemetry.py --output out/telemetry
```

It prints how many records it wrote for each SIEM sourcetype, and `out/telemetry` holds one JSON lines file per
sourcetype: scored events for layers 1 and 2, the bindings between them, the network inventory, pipeline health and
any quarantined records. Each line is one event as the SIEM receives it.

The other runners work the same way:

| Runner | Layers | Sample input |
|---|---|---|
| `run_telemetry.py` | 1 and 2 | `samples/telemetry` |
| `run_network.py` | 3 | `samples/network` |
| `run_transport.py` | 4 | `samples/transport` |
| `run_session.py` | 5 | `samples/session` |
| `run_presentation.py` | 6 | `samples/presentation` |
| `run_application.py`, `run_saas.py`, `run_endpoint.py` | 7 | `samples/application`, `samples/saas`, `samples/endpoint` |
| `run_context.py` | Identity, asset and application context | `samples/context` |
| `run_host_score.py` | A per-host score across layers | `samples/host_score` |

To run a pipeline over your own data, pass `--input` with a directory holding one JSON lines file per input class.
The [behavioral analytics README](https://github.com/kj299/Morpheus/blob/branch-25.10/examples/behavioral_analytics/README.md) describes the options, and
[the collector contract](https://github.com/kj299/Morpheus/blob/branch-25.10/examples/behavioral_analytics/collector_contract.md) describes every input class's
columns. If your data is still in its source format, such as Zeek logs, an IPFIX capture or SNMP polls, the same
README lists the reader for each format.

## 6. Run the tests

```bash
export PYTHONPATH="${MORPHEUS_ROOT}/ci/scripts/cpu_env${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/morpheus/determinism -p gpuless -m "not gpu_mode" -q -p no:cacheprovider
```

Two options make this work without a GPU:

- `-p gpuless` loads a small pytest plugin from `ci/scripts/cpu_env`, which is why that directory is on
  `PYTHONPATH`. Importing Morpheus normally loads a GPU helper that needs a driver; the plugin skips that load.
- `-m "not gpu_mode"` selects the CPU variant of every test that runs in both modes.

Add `--run_slow` to include the slow tests. To run everything the fork's CI runs, see
[Contributing to This Fork](./developer_guide/fork_contributing.md).

## Where to go next

- [Predictive Behavioral Analytics Across OSI Layers 1-7](./developer_guide/guides/11_predictive_behavioral_analytics_osi.md)
  explains what the pipelines do and why, starting in plain terms.
- [The Splunk app's README](https://github.com/kj299/Morpheus/blob/branch-25.10/examples/splunk_lineage_app/README.md) covers the SIEM side: indexes, lookups
  and the saved searches that are the detection rules.
- [Contributing to This Fork](./developer_guide/fork_contributing.md) covers making a change and getting it
  through CI.
- [Troubleshooting](./extra_info/troubleshooting.md) has more on running without a GPU.
