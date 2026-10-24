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

# Known Issues

- Numba triggering IndexError on systems with driver version 580+ ([#2305](https://github.com/nv-morpheus/Morpheus/issues/2305))
- `ransomware_detection` example pipeline occasionally logs a `distributed.comm.core.CommClosedError` error during shutdown ([#2026](https://github.com/nv-morpheus/Morpheus/issues/2026)).
- Arm64 users need to install CUDA enabled PyTorch by hand ([#2095](https://github.com/nv-morpheus/Morpheus/issues/2095))
- Performance issues were observed running the `abp_pcap_detection` and `ransomware_detection` pipelines on AArch64 ([#2120](https://github.com/nv-morpheus/Morpheus/issues/2120)) & ([#2124](https://github.com/nv-morpheus/Morpheus/issues/2124)) on Ubuntu 22.04. Arm64 users should consider upgrading to Ubuntu 24.04.
- `gnn_fraud_detection_pipeline` not working on AArch64 ([#2123](https://github.com/nv-morpheus/Morpheus/issues/2123))
- DFP visualization fails to install on AArch64 ([#2125](https://github.com/nv-morpheus/Morpheus/issues/2125))
- Build errors after a CMake reconfigure, work-around is to delete the `build` directory and re-run CMake ([#2279](https://github.com/nv-morpheus/Morpheus/issues/2279)).
- Python `stubgen` failing for C++ examples inside devcontainer ([#2287](https://github.com/nv-morpheus/Morpheus/issues/2287)), the actual build succeeds, and code is functional, but the Python stub files are not generated.


Refer to [open issues in the Morpheus project](https://github.com/nv-morpheus/Morpheus/issues)

## Issues Specific to This Fork

- A GPU run under WSL2 requires `NUMBA_CUDA_USE_NVIDIA_BINDING=1`. Without it, Numba's default driver bindings read back an invalid CUDA context through the WSL driver shim, `cuCtxGetDevice` returns a garbage device number, and the run fails with `CUDA_ERROR_INVALID_CONTEXT` or a segmentation fault inside `libcuda` partway through the suite. `ci/scripts/gpu_conformance.sh` sets the variable to `1` unless it is already set.
- The development container image does not supply `libgflags.so.2.2`, so `pyarrow` fails to import partway through `./scripts/compile.sh`. Run `conda install -c conda-forge "gflags=2.2.*"` in every new container until the image carries it. The CPU lock file `ci/scripts/cpu_env/linux-64-cpu.lock` pins `gflags` 2.2.2; a fresh solve once chose a `libarrow` build without it, which is why CI installs from the lock.
- Integer columns can widen to float when per-class output frames are concatenated. A plain NumPy `int64` column, which cuDF's `to_pandas` returns for a frame with no gaps and which a pandas frame holds on the CPU path, cannot hold the gaps the concatenation fills in for rows of other classes, so it widens to `float64` and `404` renders as `404.0`. On 2026-10-04 this widened sixteen columns across three golden files on the CPU path. The determinism harnesses collect output through `tests/morpheus/determinism/host_frame.py`, which carries integer columns as pandas `Int64`, a type that can hold a gap, in both execution modes; a new collector must do the same.
- From 2026-10-12 to 2026-10-24 no GitHub Actions workflow ran in this repository, because Actions was disabled for it. Merges in that window were checked by local CPU tier runs, not by a recorded CI run. Actions was re-enabled on 2026-10-24; if runs stop again, check the repository's Settings, Actions, General first.
