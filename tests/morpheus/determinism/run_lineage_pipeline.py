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
Standalone driver for the cross-restart determinism check.

Runs the reference lineage pipeline over the golden corpus in a fresh interpreter and writes the canonicalized
output as CSV to the path given as the only argument. The harness invokes this twice, in separate processes with
different `PYTHONHASHSEED` values, and compares the files byte for byte: any dependence on hash randomization or on
state captured from the parent process shows up as a diff that the in-process double run cannot see.
"""

import sys


def main() -> int:
    if (len(sys.argv) != 2):
        print(f"usage: {sys.argv[0]} OUTPUT_CSV", file=sys.stderr)
        return 2

    # Deferred so the usage error above does not require a Morpheus installation.
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import lineage_pipeline

    config = lineage_pipeline.build_pipeline_config()
    result = lineage_pipeline.run_pipeline(config, [lineage_pipeline.build_corpus()])

    result.to_csv(sys.argv[1], index=False, lineterminator="\n")

    return 0


if (__name__ == "__main__"):
    sys.exit(main())
