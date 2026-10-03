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
"""
A pytest plugin for a machine with no CUDA driver.

`morpheus.common.load_cudf_helper` loads the compiled cuDF helper, which needs a driver even to import, and nothing
a CPU-mode test touches needs it. Loaded with `pytest -p gpuless` (with this directory on `PYTHONPATH`), it turns
that load into a no-op so the fork's CPU tiers run on an ordinary runner. A GPU run must not load it: the point of
`ci/scripts/gpu_conformance.sh` is that the helper is real there.
"""


def pytest_configure(config):  # pylint: disable=unused-argument
    import morpheus.common  # pylint: disable=import-outside-toplevel

    morpheus.common.load_cudf_helper = lambda: None
