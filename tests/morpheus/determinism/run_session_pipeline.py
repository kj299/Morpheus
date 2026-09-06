#!/usr/bin/env python
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
Standalone driver for the composed layer 5 session pipeline's cross-restart check, and for regenerating its golden file.

Runs both layer 5 telemetry classes over the week-long corpus in a fresh interpreter and writes the canonicalized
output as CSV to the path given as the only argument. The harness invokes this twice with different
`PYTHONHASHSEED` values and compares the files byte for byte. A legitimate behavior change regenerates
`golden_session_expected.csv` with it and reviews the diff.
"""

import sys


def main() -> int:
    if (len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] not in ("cpu", "gpu"))):
        print(f"usage: {sys.argv[0]} OUTPUT_CSV [cpu|gpu]", file=sys.stderr)
        return 2

    # Deferred so the usage error above does not require a Morpheus installation.
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import session_pipeline

    # Deferred with the rest, so a usage error and a CPU run never need a GPU present.
    from morpheus.config import ExecutionMode

    requested = sys.argv[2] if len(sys.argv) == 3 else "cpu"
    config = session_pipeline.build_pipeline_config(
        execution_mode=ExecutionMode.GPU if requested == "gpu" else ExecutionMode.CPU)
    result = session_pipeline.run_pipeline(config, session_pipeline.build_corpus())

    with open(sys.argv[1], "w", encoding="utf-8", newline="") as handle:
        handle.write(session_pipeline.render(result))

    return 0


if (__name__ == "__main__"):
    sys.exit(main())
