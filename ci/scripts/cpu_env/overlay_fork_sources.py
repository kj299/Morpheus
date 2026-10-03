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
Overlay this repository's Python sources onto an installed `morpheus-core` package.

The fork adds stages and utilities in pure Python and changes no compiled code, so a CPU test environment does not
need to build Morpheus: it installs the released `morpheus-core` conda package for its `_lib` and copies the
repository's `python/morpheus/morpheus` tree over the installed package, leaving `_lib` and `_version.py` alone. The
fork workflow (`.github/workflows/fork-cpu.yaml`) runs this before pytest; a developer with the same environment
runs it after every source edit.

Usage: `python ci/scripts/cpu_env/overlay_fork_sources.py [repository root]`
"""

import os
import shutil
import sys
import sysconfig

EXTENSIONS = (".py", ".json", ".yaml", ".yml", ".typed", ".txt", ".csv")


def overlay(repository: str) -> int:
    source = os.path.join(repository, "python", "morpheus", "morpheus")
    destination = os.path.join(sysconfig.get_paths()["purelib"], "morpheus")

    if (not os.path.isdir(destination)):
        raise SystemExit(f"no installed morpheus package at {destination}; install morpheus-core first")

    copied = 0

    for (root, _, names) in os.walk(source):
        relative = os.path.relpath(root, source)

        if (relative == "_lib" or relative.startswith("_lib" + os.sep) or "__pycache__" in relative):
            continue

        for name in names:
            if ((name == "_version.py" and relative == ".") or not name.endswith(EXTENSIONS)):
                continue

            target_dir = os.path.join(destination, relative)
            os.makedirs(target_dir, exist_ok=True)
            (src, dst) = (os.path.join(root, name), os.path.join(target_dir, name))

            with open(src, "rb") as handle:
                content = handle.read()

            if (os.path.exists(dst)):
                with open(dst, "rb") as handle:
                    if (handle.read() == content):
                        continue

            shutil.copyfile(src, dst)
            copied += 1

    print(f"overlaid {copied} files onto {destination}")

    return copied


if (__name__ == "__main__"):
    overlay(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
