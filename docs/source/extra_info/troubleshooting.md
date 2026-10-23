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

# Troubleshooting

**Out of Date Build Cache**

The Morpheus build system, by default, stores all build artifacts and cached output in the `${MORPHEUS_ROOT}/.cache` directory. This cache directory is designed to speed up successive builds but occasionally can get out of date and cause unexpected build errors. In this situation, it's best to completely delete the build and cache directories and restart the build:

```bash
# Delete the build and cache folders
rm -rf ${MORPHEUS_ROOT}/.cache
rm -rf ${MORPHEUS_ROOT}/build

# Clean out documentation builds:
rm -rf docs/source/_modules docs/source/_lib

# Clean out shared-libs if compiled with `MORPHEUS_PYTHON_INPLACE_BUILD=ON`:
find ./morpheus -name "*.so" -delete

# Clean out shared libs if examples have been built:
find ./examples -name "*.so" -delete

# Restart the build
./scripts/compile.sh
```

**Incompatible MLflow Models**

Models trained with a previous version of Morpheus and stored into MLflow may be incompatible with the current version. This error can be identified by the following error message occurring in an MLflow based pipeline such as DFP.

```
Error trying to get model

Traceback (most recent call last):

File "/workspace/python/morpheus_dfp/morpheus_dfp/stages/dfp_inference_stage.py", line 101, in on_data

loaded_model = model_cache.load_model(self._client)
```
```
ModuleNotFoundError: No module named 'dfencoder'
```

The workarounds available for this issue are:

* Revert to the previous version of Morpheus until the models can be re-trained.
* Re-train the model using the current version of Morpheus

In the case of models trained by the DFP example, the existing models can be deleted by running the following command:

```bash
docker volume ls # list current docker volumes
docker volume rm production_db_data production_mlflow_data

# Re-build the MLflow container for DFP
cd ${MORPHEUS_ROOT}/examples/digital_fingerprinting/production/
docker compose build
docker compose up mlflow
```


**Debugging Python Code**

To debug issues in python code, several Visual Studio Code launch configurations have been included in the repo. These launch configurations can be found in `${MORPHEUS_ROOT}/morpheus.code-workspace`. To launch the debugging environment, ensure Visual Studio Code has opened the Morpheus workspace file (File->Open Workspace from File...). Once the workspace has been loaded, the launch configurations should be available in the debugging tab.

**Debugging C++ Code**

Similar to the Python launch configurations, several C++ launch configurations can be found in the Visual Studio Code workspace file. However, unlike the Python configuration, it's necessary to ensure Morpheus was compiled in Debug mode in order for breakpoints to work correctly. To build Morpheus in Debug mode, use the following:

```bash
CMAKE_CONFIGURE_EXTRA_ARGS="-DCMAKE_BUILD_TYPE=Debug" ./scripts/compile.sh
```

**Running the Fork's Tests Without a GPU**

Importing Morpheus loads the compiled cuDF helper, which needs a CUDA driver even to import. To run the CPU tiers on a machine with no GPU, install the pinned environment from `ci/scripts/cpu_env/linux-64-cpu.lock`, overlay the repository's Python sources onto the installed `morpheus-core` package, and load the `gpuless` pytest plugin from `ci/scripts/cpu_env`, as `.github/workflows/fork-cpu.yaml` does:

```bash
micromamba create -n morpheus -f ci/scripts/cpu_env/linux-64-cpu.lock
micromamba activate morpheus

# Repeat after every source edit; it leaves the installed _lib and _version.py alone
python ci/scripts/cpu_env/overlay_fork_sources.py "${MORPHEUS_ROOT}"

export PYTHONPATH="${MORPHEUS_ROOT}/ci/scripts/cpu_env${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/morpheus/determinism -p gpuless -m "not gpu_mode" --run_slow -q -p no:cacheprovider
```

Do not load `gpuless` for a GPU run: `ci/scripts/gpu_conformance.sh` exists to exercise the real helper.

**Git LFS Pointers in `tests/tests_data`**

Every file under `tests/tests_data` is stored in Git LFS (refer to `.gitattributes`). In a clone where the LFS objects were not fetched, each file holds a three-line pointer beginning `version https://git-lfs.github.com/spec/v1` instead of its data, and the tests that read those fixtures fail on parse errors or skip. The fork's own determinism corpora are seeded code rather than checked-in data and are not affected. Fetch the fixtures with:

```bash
git lfs install
scripts/fetch_data.py fetch tests validation
```

**Regenerating Golden Files After a cuDF Data Type Change**

The determinism harnesses compare each composed pipeline's output with a checked-in golden file, `tests/morpheus/determinism/golden_<corpus>_expected.csv`. A change in how cuDF or pandas returns a column's data type, such as an integer column arriving as float, shows up as a golden mismatch like `404.0` against `404`. First confirm the output is collected through `tests/morpheus/determinism/host_frame.py`, which keeps integer columns as pandas `Int64`, a type that can hold a gap, in both execution modes; a golden that disagrees only on such a widening usually means a collector skipped it. If the change in output is intended, regenerate the golden deliberately with that corpus's driver in CPU mode and review the diff before committing it:

```bash
python tests/morpheus/determinism/run_<corpus>_pipeline.py tests/morpheus/determinism/golden_<corpus>_expected.csv cpu
git diff tests/morpheus/determinism/golden_<corpus>_expected.csv
```

Every changed cell should be explained by the change you made; after a data type repair, each one should be a numerically equal value.
