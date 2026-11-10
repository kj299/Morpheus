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
Whether the C++ tier runs what it says and says what it did not run (guide 13, plan item 10).

`ci/scripts/cpp_tests.py` chooses which of the kept C++ tests run on a machine, from two hand lists: the tests that
need no CUDA device and the suites that read Git LFS fixtures. A hand list goes stale, so each is held to the C++
sources here, on the CPU, where no compiler is needed: the names are read from the test macros. The selection and
the verdict are tested with made-up runs, because the failure this guards is the quiet one, a test that left the
tier and a run that still passed.
"""

import importlib.util
import os
import re

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
SCRIPTS = os.path.join(REPO_ROOT, "ci", "scripts")
TESTS = os.path.join(REPO_ROOT, "python", "morpheus", "morpheus", "_lib", "tests")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture(name="cpp", scope="module")
def cpp_fixture():
    yield _load("cpp_tests")


@pytest.fixture(name="report", scope="module")
def report_fixture():
    yield _load("gpu_conformance_report")


def _defined(cpp) -> set[str]:
    """Every `Suite.Name` the C++ test sources define."""
    names = set()

    for relative in cpp.suites_by_file(REPO_ROOT):
        with open(os.path.join(TESTS, relative), encoding="utf-8") as handle:
            names |= {f"{match[1]}.{match[2]}" for match in cpp.TEST_MACRO.finditer(handle.read())}

    return names


def _read(*parts) -> str:
    with open(os.path.join(REPO_ROOT, *parts), encoding="utf-8") as handle:
        return handle.read()


# ---------------------------------------------------------------------------------------------------------------
# The hand lists, against the sources
# ---------------------------------------------------------------------------------------------------------------


def test_every_device_free_test_is_one_the_sources_define(cpp):
    assert set(cpp.DEVICE_FREE) <= _defined(cpp)
    assert len(set(cpp.DEVICE_FREE)) == len(cpp.DEVICE_FREE)


def test_the_suites_that_read_fixtures_are_the_ones_whose_sources_name_them(cpp):
    assert set(cpp.READS_FIXTURES) == cpp.reads_fixtures(REPO_ROOT)


def test_no_device_free_test_reads_a_fixture(cpp):
    # CI has no device and only the fixtures' pointers, so what runs there must need neither.
    assert not {name for name in cpp.DEVICE_FREE if name.split(".")[0] in cpp.READS_FIXTURES}


def test_every_cpp_test_file_is_one_the_build_compiles(cpp):
    # A test file CMake does not list is a test nobody runs, whatever tier exists.
    listed = set(re.findall(r"^\s+([\w/]+\.(?:cpp|cu))\s*$", _read(TESTS, "CMakeLists.txt"), re.MULTILINE))
    sources = {
        os.path.relpath(os.path.join(directory, name), TESTS)
        for (directory, _, files) in os.walk(TESTS)
        for name in files if name.endswith((".cpp", ".cu"))
    }

    assert listed == sources
    assert set(cpp.suites_by_file(REPO_ROOT)) <= listed


# ---------------------------------------------------------------------------------------------------------------
# The selection
# ---------------------------------------------------------------------------------------------------------------

DISCOVERED = [*sorted(_load("cpp_tests").DEVICE_FREE), "TestCuda.LargeShape", "TestTableUtil.GetColumnNames"]


def test_without_a_device_only_the_device_free_tests_run(cpp):
    (run, not_run) = cpp.select(DISCOVERED, device=False, fixtures=False)

    assert run == sorted(cpp.DEVICE_FREE)
    assert not_run == {"TestCuda.LargeShape": cpp.NO_DEVICE, "TestTableUtil.GetColumnNames": cpp.NO_DEVICE}


def test_on_a_device_without_the_fixtures_the_suites_that_read_them_are_named(cpp):
    (run, not_run) = cpp.select(DISCOVERED, device=True, fixtures=False)

    assert "TestCuda.LargeShape" in run
    assert not_run == {"TestTableUtil.GetColumnNames": cpp.NO_FIXTURES}


def test_on_a_device_with_the_fixtures_every_test_runs(cpp):
    assert cpp.select(DISCOVERED, device=True, fixtures=True) == (sorted(DISCOVERED), {})


def test_a_device_free_test_the_build_lost_fails_the_selection(cpp):
    with pytest.raises(ValueError, match="TestDType.IsFullySupported"):
        cpp.select([name for name in DISCOVERED if name != "TestDType.IsFullySupported"], device=False, fixtures=False)


# ---------------------------------------------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------------------------------------------

JUNIT = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="Linux" tests="3" failures="1">
  <testcase name="TestDType.IsFullySupported" classname="TestDType.IsFullySupported" time="0.03" status="run"/>
  <testcase name="TestTensor.UtilsShapeString" classname="TestTensor.UtilsShapeString" time="0.03" status="run"/>
  <testcase name="TestCuda.LargeShape" classname="TestCuda.LargeShape" time="0.1" status="fail">
    <failure message=""/>
  </testcase>
</testsuite>
"""


def _outcomes(cpp, tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(JUNIT, encoding="utf-8")

    return cpp.read_junit(str(path))


def test_the_junit_report_is_read_by_name(cpp, tmp_path):
    assert _outcomes(cpp, tmp_path) == {
        "TestDType.IsFullySupported": "passed",
        "TestTensor.UtilsShapeString": "passed",
        "TestCuda.LargeShape": "failed"
    }


def test_a_run_whose_tests_all_passed_is_clean(cpp, tmp_path):
    selected = ["TestDType.IsFullySupported", "TestTensor.UtilsShapeString"]
    result = cpp.summarize(selected, {"TestCuda.LargeShape": cpp.NO_DEVICE}, _outcomes(cpp, tmp_path), 0, False)

    assert result["counts"] == {"passed": 2, "failed": 0}
    assert result["discovered"] == 3
    assert result["not_run"] == {cpp.NO_DEVICE: ["TestCuda.LargeShape"]}
    assert cpp.is_clean(result)


def test_a_failure_is_named_and_is_not_clean(cpp, tmp_path):
    result = cpp.summarize(["TestCuda.LargeShape", "TestDType.IsFullySupported"], {}, _outcomes(cpp, tmp_path), 8, True)

    assert result["failures"] == ["TestCuda.LargeShape"]
    assert not cpp.is_clean(result)


def test_a_selected_test_ctest_did_not_report_is_not_clean(cpp, tmp_path):
    result = cpp.summarize(["TestDType.IsFullySupported", "TestDType.Vanished"], {}, _outcomes(cpp, tmp_path), 0, True)

    assert result["unaccounted"] == ["TestDType.Vanished"]
    assert not cpp.is_clean(result)


def test_a_run_in_which_nothing_ran_is_not_clean(cpp):
    assert not cpp.is_clean(cpp.summarize([], {"TestCuda.LargeShape": cpp.NO_DEVICE}, {}, 0, False))


def test_the_summary_names_what_did_not_run_and_why(cpp, tmp_path):
    summary = cpp.render(
        cpp.summarize(["TestDType.IsFullySupported"], {"TestCuda.LargeShape": cpp.NO_DEVICE},
                      _outcomes(cpp, tmp_path),
                      0,
                      False))

    assert "1 passed, 0 failed" in summary
    assert cpp.NO_DEVICE in summary
    assert "`TestCuda.LargeShape`" in summary


# ---------------------------------------------------------------------------------------------------------------
# The conformance artifact's section, and where the tier runs
# ---------------------------------------------------------------------------------------------------------------


def test_the_artifact_reads_the_result_the_tier_wrote(cpp, report, tmp_path):
    result = cpp.summarize(["TestDType.IsFullySupported"], {}, _outcomes(cpp, tmp_path), 0, True)
    path = tmp_path / "cpp.json"
    path.write_text(__import__("json").dumps(result), encoding="utf-8")

    section = report.cpp_tier(str(path))

    assert section == result
    assert report.cpp_is_clean(section)


def test_a_failing_cpp_tier_fails_the_verdict(cpp, report, tmp_path):
    result = cpp.summarize(["TestCuda.LargeShape"], {}, _outcomes(cpp, tmp_path), 8, True)

    assert not report.cpp_is_clean(result)


def test_a_build_that_wrote_no_result_is_not_a_pass(report):
    section = report.cpp_tier("/nonexistent/gpu_conformance_cpp.json")

    assert "note" in section
    assert not report.cpp_is_clean(section)


def test_a_tier_left_out_on_purpose_says_so(report):
    section = report.cpp_tier(report.CPP_SKIPPED)

    assert "SKIP_CPP_TESTS" in section["skipped"]
    assert report.cpp_is_clean(section)


def test_the_tier_runs_in_ci_without_a_device_and_in_the_conformance_run_with_one():
    assert os.access(os.path.join(SCRIPTS, "fork_cpp.sh"), os.X_OK)
    assert "ci/scripts/fork_cpp.sh --device no" in _read(".github", "workflows", "fork-cpu.yaml")
    assert "fork_cpp.sh\" --device yes" in _read("ci", "scripts", "gpu_conformance.sh")
    assert "cpp_tests.py" in _read("ci", "scripts", "fork_cpp.sh")


def test_the_package_does_not_import_a_binding_the_build_no_longer_makes():
    # `morpheus._lib` imported `modules` after the audit removed its sources. The installed upstream package still
    # had the compiled module, so CI passed; a build from the fork's sources could not import the package at all.
    bindings = set(
        re.findall(r"morpheus_add_pybind11_module\((\w+)",
                   _read("python", "morpheus", "morpheus", "_lib", "CMakeLists.txt")))
    imported = set(
        re.findall(r"^from \. import (\w+)",
                   _read("python", "morpheus", "morpheus", "_lib", "__init__.py"),
                   re.MULTILINE))

    assert imported == bindings
