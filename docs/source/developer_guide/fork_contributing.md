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

# Contributing to This Fork

This page covers how to make a change to this fork and get it through the fork's CI. It applies to the fork's own
work: the lineage and telemetry stages, the pipelines built from them, the Splunk app and their tests and
documentation. For upstream Morpheus's contribution process, building from source and licensing, see
[Contributing to Morpheus](./contributing.md); the licensing and attribution rules there apply here too.

## Before you start

Set up the CPU environment as [Getting Started with This Fork on a CPU](../fork_getting_started.md) describes. You
do not need a GPU or a compiler: the fork changes Python only.

Remember the overlay. The tests import the installed package, not the repository's sources, so after every edit
under `python/morpheus/morpheus` run:

```bash
python ci/scripts/cpu_env/overlay_fork_sources.py "${MORPHEUS_ROOT}"
```

A test that seems not to see your change usually means this step was skipped.

## The workflow

1. Create a branch from `branch-25.10`.
2. Make the change, with its tests and documentation in the same pull request.
3. Run the checks below.
4. Open a pull request against `branch-25.10`. CI runs on every push to it.

Keep a pull request to one subject. A change that regenerates recorded outputs is easier to review when the source
change and the regenerated files are separate commits.

## Checks to run before you push

### Formatting and lint

```bash
FILES="path/to/changed_file.py path/to/another.py"
python -m yapf -i ${FILES}
python -m isort --settings-path pyproject.toml ${FILES}
python -m pylint ${FILES}
```

Lines are at most 120 characters. Every new file starts with the SPDX copyright header; copy it from a neighbouring
file.

### Tests

Run the test files for what you changed first:

```bash
export PYTHONPATH="${MORPHEUS_ROOT}/ci/scripts/cpu_env${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/morpheus/stages/test_my_stage.py -p gpuless -m "not gpu_mode" -q -p no:cacheprovider
```

Then run everything CI runs. CI takes its list of test files from `ci/scripts/gpu_conformance.sh`, and this is the
same command:

```bash
mapfile -t TIERS < <(sed -n '/^TARGETS=(/,/^)/p;/^UNMARKED=(/,/^)/p' ci/scripts/gpu_conformance.sh \
  | grep -E '^\s+tests/' | tr -d ' ')
python -m pytest "${TIERS[@]}" -p gpuless -m "not gpu_mode" --run_slow -q -p no:cacheprovider
```

The full run takes about 45 minutes.

## Tests that keep the repository consistent

Several tests check the repository against itself and not any one feature. They fail when a change is incomplete,
and each failure names what to update. These are the common ones:

| If you | Then also | The test that checks |
|---|---|---|
| Add, rename or remove any tracked file | Add or remove its row in `ci/artifacts/repository_audit_manifest.csv`, and update the file counts in the [repository audit](./guides/13_repository_audit.md) | `test_repository_audit.py` |
| Add a test file | List it in the right tier of `ci/scripts/gpu_conformance.sh` | `test_gpu_conformance_targets.py` |
| Add or remove tests, stages or utility modules | Update the counts in the root `README.md` and the design guide's verification history | `test_readme_inventory.py` |
| Write a helper that already exists elsewhere | Import the existing one | `test_one_definition.py` |
| List by hand, in a test, something a directory or file already lists | Read it from the source | `test_derived_counts.py` |
| Change what a pipeline writes | Regenerate the recorded outputs (next section) | The harness for each pipeline |
| Change the CPU or docs environment in `dependencies.yaml` | Run `ci/scripts/cpu_env/make_lock.py` and re-run the tests in the new environment | `test_cpu_lock.py` |
| Change an input class's columns | Update `examples/behavioral_analytics/input_contracts.json`; `collector_contract.md` is generated from it | `test_input_contracts.py` |

## Regenerating recorded outputs

The pipelines are deterministic, and the tests hold them to recorded outputs byte for byte. When a change is
*meant* to alter what a pipeline writes, regenerate the recordings and review the difference: an unexpected changed
column is a defect, not noise.

In order, since each step reads what the one before it wrote:

```bash
# 1. The golden file of each pipeline you changed; CORPUS is the pipeline's name, such as telemetry
python tests/morpheus/determinism/run_golden.py CORPUS tests/morpheus/determinism/golden_CORPUS_expected.csv cpu

# 2. The sample events the Splunk app's validation reads
(cd examples/splunk_lineage_app/validate && python make_sample_events.py)

# 3. The input samples, then the same samples in their real feed formats
(cd examples/behavioral_analytics && python make_input_samples.py && python fixtures/make_fixtures.py)
```

Set `PYTHONHASHSEED=0` for these commands, as the tests do. If the sample events changed, the Splunk app's recorded
search-head run describes the old events; [the validation notes](../../../examples/splunk_lineage_app/validate/VALIDATION.md)
explain how to run it again.

## What CI runs

The workflow is `.github/workflows/fork-cpu.yaml`. On every push and pull request it:

- states how old the GPU verdict is against your code, since CI itself has no GPU;
- installs the pinned CPU environment, overlays the sources and runs the test tiers in CPU mode;
- measures every stage's throughput against a recorded baseline and warns when a stage falls below half of it;
- builds this documentation and checks its prose, so a broken build fails the pull request.

The GPU path is checked by `ci/scripts/gpu_conformance.sh` on a machine with a card. You are not expected to run
it; the run summary says when the GPU verdict no longer describes the code.

## Documentation

Documentation lives under `docs/source` and in the `README.md` files beside the code. To build it as CI does:

```bash
micromamba create -n morpheus-docs -f ci/scripts/cpu_env/linux-64-docs.lock
micromamba activate morpheus-docs
python -m pip install --no-deps -r ci/scripts/cpu_env/linux-64-docs.lock.pip.txt
python ci/scripts/cpu_env/overlay_fork_sources.py "${MORPHEUS_ROOT}"
ci/scripts/fork_docs.sh
```

Write for a reader who has not seen the code: say what something is for before how it works, define a term where
it first appears, and prefer a table to a long paragraph when comparing things.
