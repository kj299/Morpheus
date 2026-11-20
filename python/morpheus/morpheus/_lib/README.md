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

# The C++ library

This directory builds two libraries and three Python extension modules.

- `libmorpheus` (`cmake/libmorpheus.cmake`): the C++ messages, objects, stages, controllers and utilities. Sources
  are under `src/`, headers under `include/morpheus/`.
- `cudf_helpers` (`cudf_helpers.pyx`): a small Cython bridge that passes cuDF tables and columns between Python and
  C++.
- The pybind11 modules `common`, `messages` and `stages`, each defined in `<name>/module.cpp` with a typed stub in
  `<name>/__init__.pyi`, and declared with `morpheus_add_pybind11_module` in `CMakeLists.txt`.

The C++ unit tests are under `tests/` and build when `MORPHEUS_BUILD_TESTS=ON`. `ci/scripts/fork_cpp.sh` builds
and runs them.
