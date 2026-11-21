#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -x
set -e

# Optionally can set INSTALL_PREFIX to build and install to a specific directory. Also causes cmake install to run
BUILD_DIR=${BUILD_DIR:-build}

# The CUDA architectures to compile for. The build's default list failed to compile matx_util.cu on the fork's GPU
# machine, and building for the card's own architecture compiled, so on a machine with a card that is what is
# targeted: compute capability 8.9 becomes 89. With no card visible, as in an image build, nothing is passed and the
# project's default stands. MORPHEUS_CUDA_ARCHITECTURES overrides the detection, and so does naming
# CMAKE_CUDA_ARCHITECTURES or MORPHEUS_CUDA_ARCHITECTURES in CMAKE_ARGS or CMAKE_CONFIGURE_EXTRA_ARGS.
# ci/scripts/fork_cpp.sh, which builds the C++ tests, does the same.
if [[ -z "${MORPHEUS_CUDA_ARCHITECTURES}" && "${CMAKE_ARGS} ${CMAKE_CONFIGURE_EXTRA_ARGS}" != *CUDA_ARCHITECTURES* ]]; then
    if command -v nvidia-smi > /dev/null 2>&1; then
        MORPHEUS_CUDA_ARCHITECTURES=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)
    fi
fi

echo "Runing CMake configure..."

# Use some standard default values plus CMAKE_ARGS and CMAKE_CONFIGURE_EXTRA_ARGS
# CMAKE_ARGS is supplied by the conda environment
# CMAKE_CONFIGURE_EXTRA_ARGS is supplied by the user
cmake -S . -B ${BUILD_DIR} -GNinja \
   -DCMAKE_MESSAGE_CONTEXT_SHOW=ON \
   -DMORPHEUS_USE_CLANG_TIDY=OFF \
   -DMORPHEUS_PYTHON_INPLACE_BUILD=ON \
   -DMORPHEUS_PYTHON_PERFORM_INSTALL=${MORPHEUS_PYTHON_PERFORM_INSTALL:-ON} \
   -DMORPHEUS_USE_CCACHE=ON \
   -DMORPHEUS_BUILD_MORPHEUS_CORE=${MORPHEUS_BUILD_MORPHEUS_CORE:-ON} \
   ${INSTALL_PREFIX:+-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}} \
   ${MORPHEUS_CUDA_ARCHITECTURES:+-DMORPHEUS_CUDA_ARCHITECTURES=${MORPHEUS_CUDA_ARCHITECTURES}} \
   ${CMAKE_ARGS:+${CMAKE_ARGS}} \
   ${CMAKE_CONFIGURE_EXTRA_ARGS:+${CMAKE_CONFIGURE_EXTRA_ARGS}}

echo "Running CMake build..."
cmake --build ${BUILD_DIR} -j ${INSTALL_PREFIX:+--target install} "$@"
