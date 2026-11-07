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
"""
Standalone driver for every composed corpus's cross-restart check, and for regenerating its golden file.

    python tests/morpheus/determinism/run_golden.py CORPUS OUTPUT_CSV [cpu|gpu]

Runs the named corpus -- one of `CORPORA`, the corpora that have a `golden_<corpus>_expected.csv` beside this file --
through its composition in a fresh interpreter and writes the canonicalized output as CSV to `OUTPUT_CSV`. Each
harness invokes this twice with different `PYTHONHASHSEED` values and compares the files byte for byte. A legitimate
behavior change regenerates the golden with it and reviews the diff.
"""

import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CORPORA = tuple(
    sorted(
        os.path.basename(path)[len("golden_"):-len("_expected.csv")]
        for path in glob.glob(os.path.join(HERE, "golden_*_expected.csv"))))
"""Every corpus with a golden file, read from the files rather than listed."""


def golden_path(corpus: str) -> str:
    """The golden file a corpus is held to."""
    return os.path.join(HERE, f"golden_{corpus}_expected.csv")


def render(corpus: str, execution_mode: str = "cpu") -> str:
    """The corpus's composed output as its golden file holds it."""
    import importlib  # pylint: disable=import-outside-toplevel

    import pandas as pd  # pylint: disable=import-outside-toplevel

    # Deferred with the rest, so a usage error and a CPU run never need a GPU present.
    from morpheus.config import ExecutionMode  # pylint: disable=import-outside-toplevel

    sys.path.insert(0, HERE)
    module = importlib.import_module(f"{corpus}_pipeline")
    config = module.build_pipeline_config(
        execution_mode=ExecutionMode.GPU if execution_mode == "gpu" else ExecutionMode.CPU)
    built = module.build_corpus()

    # The lineage composition reads a list of batches of one class; every other reads a frame per class.
    result = module.run_pipeline(config, [built] if isinstance(built, pd.DataFrame) else built)

    return module.render(result)


def main(argv: list) -> int:
    if (len(argv) not in (3, 4) or argv[1] not in CORPORA or (len(argv) == 4 and argv[3] not in ("cpu", "gpu"))):
        print(f"usage: {argv[0]} CORPUS OUTPUT_CSV [cpu|gpu]; CORPUS is one of {', '.join(CORPORA)}", file=sys.stderr)
        return 2

    text = render(argv[1], argv[3] if len(argv) == 4 else "cpu")

    with open(argv[2], "w", encoding="utf-8", newline="") as handle:
        handle.write(text)

    return 0


if (__name__ == "__main__"):
    sys.exit(main(sys.argv))
