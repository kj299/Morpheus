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
Whether the GPU conformance artifact says what actually happened.

This parser has been wrong twice, and both times the artifact reported a pass. The first read `-q` output and
came back `"counts": {}, "failures": []` for a run that had crashed. The second read verbose output with a regex
that stopped at the first space, so fifteen tests whose parametrized identifiers contain one -- a saved search
named `R-D-L5-004 - Multi-factor fatigue`, an entity-key case that is three spaces -- ran, passed, and were not
counted; the artifact recorded `"verdict": "passed"` and named the last of them as where the run died.

The lines below are the real shapes those runs produced, and the reconciliation is asserted alongside the
parsing, because a better regex is not the repair. The repair is that a count which does not add up cannot be a
pass.
"""

import importlib.util
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
MODULE_PATH = os.path.join(REPO_ROOT, "ci", "scripts", "gpu_conformance_report.py")


def _load():
    # Loaded by path: `ci/scripts` is not a package and putting one there to make an import work would be the
    # tail wagging the dog.
    spec = importlib.util.spec_from_file_location("gpu_conformance_report", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture(name="report", scope="module")
def report_fixture():
    yield _load()


PLAIN = "tests/morpheus/stages/test_tc5_travel_stage.py::test_an_ordinary_flight_is_measured PASSED [  1%]"

SPACED = ("tests/morpheus/utils/test_splunk_field_contracts.py::"
          "test_every_field_a_search_reads_is_a_field_something_writes[R-D-L5-004 - Multi-factor fatigue] "
          "PASSED [ 42%]")
"""The line that broke it. Everything after `::` up to the outcome is the name, spaces and all."""

WHITESPACE_PARAM = ("tests/morpheus/utils/test_entity_key.py::"
                    "test_any_missing_part_makes_the_whole_key_null[   ] PASSED [ 38%]")
"""A parameter that is three spaces, which the same regex also could not read."""

MODE_PARAM = ("tests/morpheus/stages/test_tc5_risk_stage.py::"
              "test_the_mfa_fatigue_shape_is_readable_off_the_approving_row[gpu_mode] PASSED [ 55%]")

FAILED = "tests/morpheus/utils/test_lineage_cudf.py::test_verify_digest_equivalence FAILED [ 77%]"

SUMMARY = ("=========================== short test summary info ============================\n"
           "FAILED tests/morpheus/utils/test_lineage_cudf.py::test_verify_digest_equivalence - AssertionError\n"
           "1 failed, 4 passed in 12.34s")
"""pytest's end-of-run summary, which names failures a second time. Counting those again would inflate the
total and hide a drop, which is the direction that matters."""


def test_an_identifier_containing_spaces_is_counted(report):
    # The whole reason this file exists.
    summary = report.summarize(SPACED)

    assert summary["counts"] == {"passed": 1}
    assert summary["died_in"] is None
    assert summary["failures"] == []


def test_the_name_survives_the_spaces_intact(report):
    summary = report.summarize(SPACED.replace("PASSED", "FAILED"))

    assert summary["failures"] == [
        "tests/morpheus/utils/test_splunk_field_contracts.py::"
        "test_every_field_a_search_reads_is_a_field_something_writes[R-D-L5-004 - Multi-factor fatigue]"
    ]


@pytest.mark.parametrize("line", [PLAIN, SPACED, WHITESPACE_PARAM, MODE_PARAM],
                         ids=["plain", "spaced", "whitespace_param", "mode_param"])
def test_every_shape_of_streamed_line_is_read(report, line):
    assert report.summarize(line)["counts"] == {"passed": 1}


WRAPPED = ("tests/morpheus/determinism/test_determinism_harness.py::test_double_run_diff[gpu_mode]\n"
           "PASSED [  1%]")
"""What pytest writes when the identifier does not fit the terminal: the outcome goes on the next line.

The third shape this parser could not read, and the most expensive so far. Every identifier in the marked tier
carries a `[gpu_mode]` suffix, so on a narrow terminal all of them wrapped: 379 passing tests parsed as zero
counted and 379 unaccounted, on a run that had genuinely passed."""


def test_an_outcome_that_wrapped_onto_the_next_line_is_counted(report):
    summary = report.summarize(WRAPPED, collected=1)

    assert summary["counts"] == {"passed": 1}
    assert summary["died_in"] is None
    assert summary["unaccounted"] == 0


def test_a_wrapped_failure_keeps_the_name_from_the_line_above(report):
    summary = report.summarize(WRAPPED.replace("PASSED", "FAILED"))

    assert summary["failures"] == [
        "tests/morpheus/determinism/test_determinism_harness.py::test_double_run_diff[gpu_mode]"
    ]


def test_a_whole_tier_of_wrapped_lines_reconciles(report):
    # The shape of the run that produced the defect, rather than one line of it.
    text = "\n".join([WRAPPED] * 3)
    summary = report.summarize(text, collected=3)

    assert summary["counts"] == {"passed": 3}
    assert summary["unaccounted"] == 0


def test_an_outcome_with_nothing_above_it_is_not_counted(report):
    # The other half of the discriminator. A wrapped outcome resolves an identifier that is waiting for one; an
    # outcome word with no such identifier belongs to something else entirely, and counting it would invent a
    # test. Inventing one hides a real drop just as well as missing one does, because the total still adds up.
    assert report.summarize("PASSED [ 10%]")["counts"] == {}
    assert report.summarize("PASSED [ 10%]", collected=1)["unaccounted"] == 1


def test_a_summary_line_is_not_mistaken_for_a_wrapped_outcome(report):
    # The discriminator. pytest's summary also starts with an outcome word, and counting those against a name
    # still waiting for one would inflate the total in the direction that hides a drop -- so a line naming a
    # test is never a wrap. Here the died_in name is genuinely unresolved and must stay that way.
    started = "tests/morpheus/determinism/test_gpu_parity.py::test_the_pipelines_agree[gpu_mode]"
    summary = report.summarize("\n".join([started, SUMMARY]))

    assert summary["counts"] == {}
    assert summary["died_in"] == started


def test_the_summary_section_does_not_count_a_failure_twice(report):
    summary = report.summarize("\n".join([PLAIN, FAILED, SUMMARY]))

    assert summary["counts"] == {"passed": 1, "failed": 1}
    assert len(summary["failures"]) == 1


def test_an_unaccounted_test_is_named_not_just_counted(report):
    # A count says a run does not add up. The name says which test it does not add up by, which is the whole
    # difference between a verdict and an investigation: the first run whose tiers were total came back one
    # short, with nothing failed, nothing crashed, and no name to go looking for.
    names = ["tests/a.py::test_one", "tests/a.py::test_two", "tests/a.py::test_three"]
    summary = report.summarize("tests/a.py::test_one PASSED [ 33%]", collected_names=names)

    assert summary["collected"] == 3
    assert summary["unaccounted"] == 2
    assert summary["unaccounted_names"] == ["tests/a.py::test_two", "tests/a.py::test_three"]


def test_a_run_that_accounts_for_everything_names_nothing(report):
    names = ["tests/a.py::test_one"]
    summary = report.summarize("tests/a.py::test_one PASSED [100%]", collected_names=names)

    assert summary["unaccounted"] == 0
    assert "unaccounted_names" not in summary


def test_a_wholesale_loss_is_counted_in_full_but_not_listed_in_full(report):
    # The list is capped; the count never is. An artifact that answers a lost tier with a thousand lines of JSON
    # is not more informative than one that answers with the number and the first twenty.
    names = [f"tests/a.py::test_{index}" for index in range(report.MAX_NAMED + 5)]
    summary = report.summarize("", collected_names=names)

    assert summary["unaccounted"] == report.MAX_NAMED + 5
    assert len(summary["unaccounted_names"]) == report.MAX_NAMED + 1
    assert summary["unaccounted_names"][-1] == "... and 5 more"


TALLY_LINE = ("=========================== 379 passed, 522 deselected, 1 warning in 456.25s "
              "===========================")
"""pytest's own final line, which is the authority whenever the run wrote one."""

SUBPROCESS_NOISE = "\n".join([
    "tests/a.py::test_one[gpu_mode] PASSED                 [  0%]",
    "tests/a.py::test_two[gpu_mode] PASSED [  7%]",
    "tests/a.py::test_two[gpu_mode] PASSED                 [  7%]",
])
"""The shape that broke the count on a real run.

Three of these tests spawn a subprocess, and the subprocess's own pytest output lands in the log -- so a name
appears twice while the tests it displaced appear not at all. Adding up streamed lines counted two tests that
never ran and missed three that did, and the two errors nearly cancelled: 379 became 378."""


def test_pytests_own_tally_is_preferred_to_adding_up_the_lines(report):
    summary = report.summarize("\n".join([SUBPROCESS_NOISE, TALLY_LINE]),
                               collected_names=[f"tests/a.py::test_{index}" for index in range(379)])

    assert summary["counts"] == {"passed": 379}
    assert summary["counted_from"] == "pytest's summary"
    assert summary["unaccounted"] == 0
    # And no names, because a name-level mismatch under a reconciled tally is this parser's noise, not a gap.
    assert "unaccounted_names" not in summary


def test_deselected_tests_are_not_counted_as_run(report):
    # They were never going to run, and the collected count they reconcile against already excludes them.
    summary = report.summarize(TALLY_LINE, collected_names=[f"tests/a.py::test_{index}" for index in range(379)])

    assert "deselected" not in summary["counts"]
    assert summary["unaccounted"] == 0


def test_a_tally_that_does_not_match_what_was_collected_still_fails(report):
    # The reconciliation is not weakened by trusting pytest: the tally is authoritative about what pytest ran,
    # not about what it was asked to run, and the gap between those two is the thing being watched for.
    summary = report.summarize(TALLY_LINE, collected_names=[f"tests/a.py::test_{index}" for index in range(400)])

    assert summary["unaccounted"] == 21
    assert not report.is_clean({"outcome": "exited cleanly", **summary})


def test_a_run_that_died_before_its_tally_is_still_read_line_by_line(report):
    # The streamed lines are why they are read at all. A run that crashes never writes a summary, and that is
    # exactly the run whose story matters most.
    summary = report.summarize("tests/a.py::test_one PASSED [ 50%]\ntests/a.py::test_two",
                               collected_names=["tests/a.py::test_one", "tests/a.py::test_two"])

    assert summary["counts"] == {"passed": 1}
    assert "counted_from" not in summary
    assert summary["died_in"] == "tests/a.py::test_two"
    assert summary["unaccounted_names"] == ["tests/a.py::test_two"]


def test_a_count_that_does_not_add_up_is_not_a_pass(report):
    # The general repair. A regex can be wrong again; a reconciliation says so out loud when it is.
    summary = report.summarize(PLAIN, collected=16)

    assert summary["unaccounted"] == 15
    assert report.is_clean({"outcome": "exited cleanly", **summary}) is False


def test_a_count_that_adds_up_is(report):
    summary = report.summarize("\n".join([PLAIN, SPACED]), collected=2)

    assert summary["unaccounted"] == 0
    assert report.is_clean({"outcome": "exited cleanly", **summary}) is True


def test_a_run_that_died_names_where(report):
    # A test line with no outcome, and nothing after it: the process went away mid-test.
    died = "tests/morpheus/stages/test_tc2_binding_stage.py::test_a_device_frame_seals"
    summary = report.summarize("\n".join([PLAIN, died]))

    assert summary["died_in"] == died
    assert report.is_clean({"outcome": "exited cleanly", **summary}) is False


def test_a_line_the_parser_could_not_read_does_not_read_as_a_death_if_the_run_continued(report):
    # The distinction the previous version got wrong in the other direction: it called the last unreadable line a
    # death even though the run went on to finish. Anything reporting an outcome afterwards settles it.
    summary = report.summarize("\n".join([SPACED, PLAIN]))

    assert summary["died_in"] is None


def test_a_failing_tier_is_not_a_pass_however_tidy_its_counts(report):
    summary = report.summarize("\n".join([PLAIN, FAILED]), collected=2)

    assert summary["unaccounted"] == 0
    assert report.is_clean({"outcome": "exited cleanly", **summary}) is False


def test_a_crash_is_not_a_pass(report):
    summary = report.summarize(PLAIN, collected=1)

    assert report.is_clean({"outcome": "killed by signal 6 (SIGABRT: a crash, not a test failure)", **summary}) is False


@pytest.mark.parametrize(
    ("status", "expected"),
    [(0, "exited cleanly"), (1, "exited 1"), (134, "killed by signal 6 (SIGABRT: a crash, not a test failure)"),
     (139, "killed by signal 11")],
    ids=["clean", "failed", "sigabrt", "sigsegv"])
def test_an_exit_code_is_described_in_words(report, status, expected):
    assert report.describe(status) == expected


def test_a_missing_log_is_reported_rather_than_treated_as_empty(report):
    section = report.tier("/nonexistent/gpu_conformance.log", 0, 10)

    assert "note" in section
    assert report.is_clean(section) is False


def test_the_module_is_where_the_runner_looks_for_it():
    # The runner calls this by path. A rename that misses one of them would leave the runner writing no artifact
    # at all, which on a machine with a GPU costs another full run to discover.
    with open(os.path.join(REPO_ROOT, "ci", "scripts", "gpu_conformance.sh"), encoding="utf-8") as handle:
        runner = handle.read()

    assert os.path.exists(MODULE_PATH)
    assert "gpu_conformance_report.py" in runner
