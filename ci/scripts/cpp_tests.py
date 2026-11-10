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
Run the kept C++ tests from a build tree and say what ran, what did not and why (guide 13, plan item 10).

The C++ core has its own tests, under `python/morpheus/morpheus/_lib/tests`, and no tier built or ran them: the
fork's CI installs the upstream package and lays the fork's Python over it, so the fork's C++ was never compiled
there at all. `ci/scripts/fork_cpp.sh` builds the core and these tests from the fork's sources, and this runs them.

**Not every test can run everywhere, and a test that did not run is named.** Two things decide:

- *A device.* Most of the tests allocate device memory, and without a CUDA driver they fail in the allocator. The
  ones that do not are listed in `DEVICE_FREE`; on a machine with no GPU those run and the rest are reported as built
  and not run. The list is by name, and a name on it that the build no longer has fails the run: a renamed test
  must not leave the tier quietly.
- *The fixtures.* Some suites read files under `tests/tests_data`, which Git LFS stores. On a checkout that has only
  the pointers, those suites (`READS_FIXTURES`) are reported as not run rather than as failures of the code.

On a machine with a GPU and the fixtures, every test the build has runs. The result is a pass only when at least
one test ran, every one that ran passed, and every one that was selected is accounted for in what CTest reported.

    python ci/scripts/cpp_tests.py BUILD_DIR [--device auto|yes|no] [--result result.json]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import typing
from xml.etree import ElementTree

DEVICE_FREE = (
    "TestControlMessage.GetAllMetadata",
    "TestControlMessage.GetNonexistentMetadata",
    "TestControlMessage.GetTensorMemoryWhenNoneSet",
    "TestControlMessage.GetTimestampNonExistentKey",
    "TestControlMessage.GetTimestampWithRegex",
    "TestControlMessage.InitializationTest",
    "TestControlMessage.OverwriteMetadata",
    "TestControlMessage.SetAndGetMetadata",
    "TestControlMessage.SetAndGetTensorMemory",
    "TestControlMessage.SetAndGetTimestamp",
    "TestControlMessage.SetMessageTest",
    "TestControlMessage.SetTensorMemoryToNull",
    "TestControlMessage.TaskTest",
    "TestControlMessage.UpdateTimestamp",
    "TestDType.FromCudfNotSupported",
    "TestDType.FromCudfSupported",
    "TestDType.FromNumpyInvalidStr",
    "TestDType.FromNumpyValidStr",
    "TestDType.FromTritonInvalidStr",
    "TestDType.FromTritonValidStr",
    "TestDType.IsFullySupported",
    "TestTensor.GetElementStride",
    "TestTensor.UtilsShapeString",
    "TestTypeUtils.DTypeCopy",
)
"""The tests that need no CUDA device: every one that passed on a machine without a driver, by name."""

READS_FIXTURES = (
    "TestDeserializers",
    "TestFileInOut",
    "TestMatxUtil",
    "TestMessageMeta",
    "TestMultiSlices",
    "TestSlicedMessageMeta",
    "TestTableInfo",
    "TestTableUtil",
)
"""The suites whose source reads `tests/tests_data`. `tests/morpheus/utils/test_cpp_tests.py` holds this to the
sources: a test file that names the fixtures has every suite it defines here."""

TESTS = os.path.join("python", "morpheus", "morpheus", "_lib", "tests")
FIXTURES = os.path.join("tests", "tests_data")
FIXTURE_MARKERS = ("get_morpheus_root", "tests_data")
"""What a C++ test file says when it reads a fixture."""

NO_DEVICE = "built and not run: needs a CUDA device"
NO_FIXTURES = "built and not run: reads tests/tests_data, which this checkout has only as Git LFS pointers"
TEST_MACRO = re.compile(r"^\s*TEST(?:_F|_P)?\(\s*(\w+)\s*,\s*(\w+)\s*\)", re.MULTILINE)


def suites_by_file(root: str) -> dict[str, set[str]]:
    """The suites each C++ test file defines, by the file's path under the tests directory."""
    found = {}

    for (directory, _, files) in os.walk(os.path.join(root, TESTS)):
        for name in sorted(files):
            if (name.endswith((".cpp", ".cu"))):
                path = os.path.join(directory, name)

                with open(path, encoding="utf-8") as handle:
                    suites = {match[1] for match in TEST_MACRO.finditer(handle.read())}

                if (suites):
                    found[os.path.relpath(path, os.path.join(root, TESTS))] = suites

    return found


def reads_fixtures(root: str) -> set[str]:
    """The suites defined in a test file that names the fixtures, read from the sources."""
    suites = set()

    for (relative, defined) in suites_by_file(root).items():
        with open(os.path.join(root, TESTS, relative), encoding="utf-8") as handle:
            text = handle.read()

        if (any(marker in text for marker in FIXTURE_MARKERS)):
            suites |= defined

    return suites


def has_device() -> bool:
    """Whether `nvidia-smi` is here and names a GPU."""
    if (shutil.which("nvidia-smi") is None):
        return False

    result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                            capture_output=True,
                            text=True,
                            check=False)

    return result.returncode == 0 and bool(result.stdout.strip())


def has_fixtures(root: str) -> bool:
    """Whether the files under `tests/tests_data` are the files, not the pointers Git LFS leaves in their place."""
    probe = os.path.join(root, FIXTURES, "filter_probs.csv")

    try:
        with open(probe, "rb") as handle:
            return b"git-lfs" not in handle.read(64)
    except OSError:
        return False


def discover(build_dir: str) -> list[str]:
    """Every test the build tree has, as CTest names them."""
    listed = subprocess.run(["ctest", "--test-dir", build_dir, "--show-only=json-v1"],
                            capture_output=True,
                            text=True,
                            check=True)

    return sorted(test["name"] for test in json.loads(listed.stdout)["tests"])


def select(names: typing.Iterable[str], device: bool, fixtures: bool) -> tuple[list[str], dict[str, str]]:
    """
    Which tests to run here, and why each of the others is not run.

    Raises
    ------
    ValueError
        If a `DEVICE_FREE` name is not among `names`: the test was renamed or removed, and the list was not.
    """
    names = sorted(names)
    missing = sorted(set(DEVICE_FREE) - set(names))

    if (missing):
        raise ValueError(f"DEVICE_FREE names tests the build does not have: {missing}")

    (selected, not_run) = ([], {})

    for name in names:
        if (not device and name not in DEVICE_FREE):
            not_run[name] = NO_DEVICE
        elif (not fixtures and name.split(".")[0] in READS_FIXTURES):
            not_run[name] = NO_FIXTURES
        else:
            selected.append(name)

    return (selected, not_run)


def read_junit(path: str) -> dict[str, str]:
    """Each test's outcome, `passed` or `failed`, from CTest's JUnit report."""
    outcomes = {}

    for case in ElementTree.parse(path).getroot().iter("testcase"):
        outcomes[case.get("name")] = "passed" if case.get("status") == "run" else "failed"

    return outcomes


def summarize(selected: list[str], not_run: dict[str, str], outcomes: dict[str, str], status: int,
              device: bool) -> dict:
    """The run as the conformance artifact records it."""
    reasons: dict[str, list[str]] = {}

    for (name, reason) in sorted(not_run.items()):
        reasons.setdefault(reason, []).append(name)

    return {
        "exit_status": status,
        "device": device,
        "discovered": len(selected) + len(not_run),
        "selected": len(selected),
        "counts": {
            outcome: sum(1 for name in selected if outcomes.get(name) == outcome)
            for outcome in ("passed", "failed")
        },
        "failures": sorted(name for name in selected if outcomes.get(name) == "failed"),
        "unaccounted": sorted(name for name in selected if name not in outcomes),
        "not_run": reasons,
    }


def is_clean(result: dict) -> bool:
    """Whether a run is a pass: something ran, all of it passed, and nothing selected went missing."""
    counts = result.get("counts") or {}

    return (result.get("exit_status") == 0 and counts.get("passed", 0) > 0 and counts.get("failed", 0) == 0
            and not result.get("failures") and not result.get("unaccounted"))


def render(result: dict) -> str:
    """The run summary's markdown."""
    counts = result["counts"]
    lines = [
        "## C++ tests",
        "",
        f"The build has {result['discovered']} tests. {result['selected']} ran "
        f"{'on the device' if result['device'] else 'with no CUDA device'}: "
        f"{counts['passed']} passed, {counts['failed']} failed.",
    ]

    for (label, names) in (("Failed", result["failures"]), ("Selected and not reported by CTest",
                                                            result["unaccounted"])):
        if (names):
            lines += ["", f"**{label}:** " + ", ".join(f"`{name}`" for name in names)]

    for (reason, names) in result["not_run"].items():
        lines += ["", f"<details><summary>{len(names)} {reason}</summary>", ""]
        lines += [f"- `{name}`" for name in names]
        lines += ["", "</details>"]

    return "\n".join(lines) + "\n"


def run(build_dir: str, root: str, device: bool, fixtures: bool) -> dict:
    """Select, run under CTest, and summarize."""
    (selected, not_run) = select(discover(build_dir), device, fixtures)
    junit = os.path.join(build_dir, "fork_cpp_tests.xml")

    if (os.path.exists(junit)):
        os.remove(junit)

    # The tests that start an interpreter import the package from the source tree, where the in-place build put
    # the compiled modules, and find the fixtures from the root.
    environment = dict(os.environ, MORPHEUS_ROOT=root)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [os.path.join(root, "python", "morpheus"), os.environ.get("PYTHONPATH")]))

    status = 0
    outcomes = {}

    if (selected):
        pattern = "^(" + "|".join(re.escape(name) for name in selected) + ")$"
        status = subprocess.run(
            ["ctest", "--test-dir", build_dir, "--output-on-failure", "--output-junit", junit, "-R", pattern],
            env=environment,
            check=False).returncode

        if (os.path.exists(junit)):
            outcomes = read_junit(junit)

    return summarize(selected, not_run, outcomes, status, device)


def main(argv: typing.Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("build_dir", help="The CMake build tree, configured with MORPHEUS_BUILD_TESTS=ON.")
    parser.add_argument("--root", default=os.getcwd(), help="The repository's root.")
    parser.add_argument("--device",
                        choices=("auto", "yes", "no"),
                        default="auto",
                        help="Whether to run the tests that need a CUDA device; by default, if nvidia-smi names one.")
    parser.add_argument("--result", help="Write the run's result here, as JSON.")
    arguments = parser.parse_args(argv)

    root = os.path.abspath(arguments.root)
    device = has_device() if arguments.device == "auto" else arguments.device == "yes"
    result = run(arguments.build_dir, root, device, has_fixtures(root))

    if (arguments.result):
        with open(arguments.result, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")

    summary = render(result)
    print(summary)

    if (os.environ.get("GITHUB_STEP_SUMMARY")):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(summary + "\n")

    return 0 if is_clean(result) else 1


if (__name__ == "__main__"):
    sys.exit(main())
