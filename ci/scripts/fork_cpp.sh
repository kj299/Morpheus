#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Build the kept C++ core and its tests from the fork's sources, and run the tests (guide 13, plan item 10).
#
# One command for both places it runs. The C++ job of .github/workflows/fork-cpu.yaml runs it on a machine with no
# GPU: everything compiles, the CUDA sources included, and the tests that need no device run. gpu_conformance.sh
# runs it on a machine with a card, where every test runs. ci/scripts/cpp_tests.py decides which is which and says
# what did not run; arguments after the script's name go to it.
#
# The environment is the `cpu-build` file key of dependencies.yaml, locked as ci/scripts/cpu_env/linux-64-build.lock;
# the dev container's environment has the same tools. Two things the October GPU builds needed by hand are settled
# here and there rather than remembered:
#
#   - the CUDA architecture. On a machine with a card the build targets that card's own architecture, which is what
#     compiled matx_util.cu when the default list did not. With no card it targets the RAPIDS list, which the locked
#     toolchain compiles. MORPHEUS_CUDA_ARCHITECTURES overrides either.
#   - gflags 2.2, which the Arrow the build links needs and the solver no longer picks: pinned in the environment.
#
#   BUILD_DIR                      the build tree; build/fork_cpp by default
#   MORPHEUS_CUDA_ARCHITECTURES    the CUDA architectures to compile for
#   CMAKE_CONFIGURE_EXTRA_ARGS     more arguments for the configure step
#   PARALLEL_LEVEL                 build jobs; every processor by default
#
#   ./ci/scripts/fork_cpp.sh [--device auto|yes|no] [--result result.json]

set -eo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/../.." &> /dev/null && pwd )"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build/fork_cpp}"

cd "${REPO_ROOT}"

echo "=== toolchain ==="
for tool in cmake ninja python; do
    if ! command -v "${tool}" > /dev/null 2>&1; then
        echo "FAILED: ${tool} is not on PATH. Activate the build environment, ci/scripts/cpu_env/linux-64-build.lock."
        exit 1
    fi
done
cmake --version | head -1

# The build's CMake modules are a submodule, and a plain clone does not have them.
if [[ ! -f external/utilities/cmake/morpheus_utils/load.cmake ]]; then
    git submodule update --init external/utilities
fi

if [[ -z "${MORPHEUS_CUDA_ARCHITECTURES}" ]]; then
    COMPUTE_CAP=""
    if command -v nvidia-smi > /dev/null 2>&1; then
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
    fi
    MORPHEUS_CUDA_ARCHITECTURES="${COMPUTE_CAP:-RAPIDS}"
fi
echo "CUDA architectures: ${MORPHEUS_CUDA_ARCHITECTURES}"

echo ""
echo "=== configure ==="
# In place, so the tests that start an interpreter can import the package from the source tree. No stubs, which
# need a driver to generate, no install, and no DFP package, which has no C++.
# shellcheck disable=SC2086
cmake -S . -B "${BUILD_DIR}" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MESSAGE_CONTEXT_SHOW=ON \
    -DMORPHEUS_BUILD_TESTS=ON \
    -DMORPHEUS_BUILD_MORPHEUS_DFP=OFF \
    -DMORPHEUS_PYTHON_BUILD_STUBS=OFF \
    -DMORPHEUS_PYTHON_INPLACE_BUILD=ON \
    -DMORPHEUS_PYTHON_PERFORM_INSTALL=OFF \
    -DMORPHEUS_USE_CCACHE=OFF \
    -DMORPHEUS_USE_CLANG_TIDY=OFF \
    -DMORPHEUS_CUDA_ARCHITECTURES="${MORPHEUS_CUDA_ARCHITECTURES}" \
    ${CMAKE_ARGS} \
    ${CMAKE_CONFIGURE_EXTRA_ARGS}

echo ""
echo "=== build ==="
cmake --build "${BUILD_DIR}" -j "${PARALLEL_LEVEL:-$(nproc)}"

echo ""
echo "=== test ==="
python "${SCRIPT_DIR}/cpp_tests.py" "${BUILD_DIR}" --root "${REPO_ROOT}" "$@"
