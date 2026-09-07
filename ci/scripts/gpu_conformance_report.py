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
Turns `gpu_conformance.sh`'s streamed pytest output into the JSON artifact that run reports.

A module rather than a heredoc inside the shell script, because this parser has now been wrong twice and a
parser nobody can test is a parser that will be wrong a third time. The first version read `-q` output and
reported `"counts": {}, "failures": []` for a run that had crashed. The second read verbose output with a regex
that stopped at the first space, so every parametrized test whose identifier contains one -- a saved search
called `R-D-L5-004 - Multi-factor fatigue`, an entity key case called `[   ]` -- ran, passed, and was not
counted. Fifteen of them. The artifact said `"verdict": "passed"` while silently dropping tests, which is the
same failure the whole script exists to prevent, wearing its third hat.

The third time was the terminal width. pytest prints the outcome on the line after the identifier when the
identifier does not fit, and the marked tier's identifiers all carry a `[gpu_mode]` suffix -- so on a narrow
terminal every single line wrapped, nothing matched, and a tier of 379 passing tests parsed as zero. The runner
now pins `COLUMNS`, and this reads a wrapped outcome as well, because pinning an environment variable is a thing
that can be overridden and the parser should not depend on it.

**The counts are reconciled against what was collected.** That is the general repair rather than a better regex:
a counter that can quietly drop a test is untrustworthy however carefully its pattern is written, so the number
it produces is checked against the number pytest said it would run, and a mismatch is a failed verdict with the
difference named. A regex can be wrong again -- it has been, three times now -- and each time the reconciliation
is what said so out loud.
"""

import datetime
import json
import re
import sys
import typing

OUTCOMES = ("PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS")

MAX_NAMED = 20
"""Unaccounted tests named in the artifact before it stops listing them.

A run that lost one test needs the name; a run that lost four hundred needs the number and a look at the log,
not four hundred lines of JSON. The count is never truncated -- only the list is."""

TEST_LINE = re.compile(r"^(?P<name>\S+::.*?)\s+(?P<outcome>" + "|".join(OUTCOMES) + r")\b")
"""One streamed verbose line.

`.*?` rather than `\\S+` after the `::`, because a parametrized identifier can contain spaces and the whole point
of reading the streamed lines is that they survive a process that dies before writing a summary. The outcome word
is what terminates the name, and it is anchored on a word boundary so a test whose parameter happens to contain
the text of an outcome does not truncate its own name.
"""

STARTED_LINE = re.compile(r"^\S+::")
"""A line that begins with a test identifier.

pytest's end-of-run summary writes `FAILED tests/x::test_y`, outcome first, which this deliberately does not
match: those are a second report of tests already counted, and counting them twice would break the
reconciliation in the direction that hides a drop.
"""

WRAPPED_OUTCOME = re.compile(r"^(?P<outcome>" + "|".join(OUTCOMES) + r")\b(?P<rest>.*)$")
"""An outcome standing alone on its own line, which is where pytest puts it when the identifier did not fit.

Only consumed when a name is waiting for one and the rest of the line names no test, which is what separates it
from the end-of-run summary's `FAILED tests/x::test_y`. Counting a summary line would inflate the total in the
direction that hides a drop, so the discriminator matters more than the convenience.
"""


def summarize(text: str, collected: typing.Optional[int] = None, collected_names: typing.Optional[list] = None) -> dict:
    """
    Read the streamed per-test lines rather than the summary, so an aborted run still says what happened.

    A summary is written once, at the end. A run that dies partway through has no end.

    Parameters
    ----------
    text : str
        The captured pytest output.
    collected : int, optional
        How many tests pytest said it would run. When given, the parsed total is checked against it and any
        difference is reported as `unaccounted`, which fails the verdict.
    collected_names : list, optional
        What pytest said it would run, by name. A count says a run does not add up; the names say which test it
        does not add up by, which is the difference between a verdict and an investigation. Reported as
        `unaccounted_names`, capped so a wholesale loss does not write a thousand-line artifact.

    Returns
    -------
    dict
        `counts` per outcome, `failures`, `died_in`, and the reconciliation.
    """
    counts: dict = {}
    failures = set()
    reported = set()
    died_in = None

    def record(name: str, outcome: str) -> None:
        counts[outcome.lower()] = counts.get(outcome.lower(), 0) + 1
        reported.add(name)

        if (outcome in ("FAILED", "ERROR")):
            failures.add(name)

    for line in text.splitlines():
        if (not STARTED_LINE.match(line)):
            wrapped = WRAPPED_OUTCOME.match(line)

            # An outcome alone on a line belongs to the identifier above it, which pytest put on its own line
            # because it did not fit the terminal. `died_in` is that identifier: it is set for a name with no
            # outcome yet, which is precisely the state a wrapped line resolves. `::` in the rest of the line
            # means this is the end-of-run summary re-reporting a test, not a wrap.
            if (wrapped is not None and died_in is not None and "::" not in wrapped.group("rest")):
                record(died_in, wrapped.group("outcome"))
                died_in = None

            continue

        match = TEST_LINE.match(line)

        if (match is None):
            # A test line with no outcome on it. Either the process died here, the outcome wrapped onto the next
            # line, or the parser cannot read it -- and they are told apart by what follows: an outcome resolves
            # the wrap, and anything reporting an outcome afterwards means the run continued past this point.
            died_in = line.strip()
            continue

        record(match.group("name"), match.group("outcome"))
        died_in = None

    result = {"counts": counts, "failures": sorted(failures), "died_in": died_in}

    if (collected_names is not None):
        missing = [name for name in collected_names if name not in reported]
        result["collected"] = len(collected_names)
        result["unaccounted"] = len(missing)
        result["unaccounted_names"] = missing[:MAX_NAMED] + ([f"... and {len(missing) - MAX_NAMED} more"]
                                                             if len(missing) > MAX_NAMED else [])
    elif (collected is not None):
        result["collected"] = collected
        result["unaccounted"] = collected - sum(counts.values())

    return result


def describe(status: int) -> str:
    """A shell exit code, in words. 134 is SIGABRT, which is what a device stage crashing looks like from here."""
    if (status == 0):
        return "exited cleanly"

    if (status > 128):
        return f"killed by signal {status - 128}" + (" (SIGABRT: a crash, not a test failure)" if status == 134 else "")

    return f"exited {status}"


def read_collected(path: str) -> typing.Optional[list]:
    """The names pytest said it would run, from a `--collect-only` capture. `None` when there is no such file."""
    try:
        with open(path, encoding="utf-8") as handle:
            return [line.strip() for line in handle if "::" in line]
    except OSError:
        return None


def tier(log: str,
         status: int,
         collected: typing.Optional[int] = None,
         collected_path: typing.Optional[str] = None) -> dict:
    """One tier's section of the artifact, read from its log."""
    try:
        with open(log, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return {"outcome": describe(status), "note": f"no log at {log}"}

    names = read_collected(collected_path) if collected_path else None

    return {"outcome": describe(status), **summarize(text, collected, names), "log": log}


def is_clean(section: dict) -> bool:
    """
    Whether a tier reported a result worth calling a pass.

    Five separate things, because each has been the whole story at least once: there is a log to read at all,
    it reported some tests, the process exited zero, nothing it reported was a failure, and every test it
    collected is accounted for. A run that exits zero having lost fifteen tests is not a pass, and neither is one
    whose log went missing -- an absent section has no failures in it, which is not the same as having none.
    """
    if ("note" in section or not section.get("counts")):
        return False

    return (section.get("outcome") == "exited cleanly" and not section.get("failures")
            and section.get("died_in") is None and section.get("unaccounted", 0) == 0)


def main() -> int:
    if (len(sys.argv) != 9):
        print(
            f"usage: {sys.argv[0]} ARTIFACT DEVICE MARKED_SELECTED MARKED_STATUS UNMARKED_SELECTED "
            f"UNMARKED_STATUS RUN_WIDER WIDER_STATUS",
            file=sys.stderr)
        return 2

    (path, device, marked_selected, marked_status, unmarked_selected, unmarked_status, run_wider,
     wider_status) = sys.argv[1:9]

    marked = tier("/tmp/gpu_conformance_marked.log",
                  int(marked_status),
                  int(marked_selected),
                  "/tmp/gpu_conformance_marked_collected.txt")
    unmarked = tier("/tmp/gpu_conformance_unmarked.log",
                    int(unmarked_status),
                    int(unmarked_selected),
                    "/tmp/gpu_conformance_unmarked_collected.txt")

    report = {
        "verdict":
            "passed" if (is_clean(marked) and is_clean(unmarked)) else "failed",
        "at":
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device":
            device,
        "gpu_mode":
            marked,
        "unmarked_gpu_coverage":
            unmarked,
        # Context, not verdict. Upstream stages reading upstream fixtures; a failure here is a report to make
        # upstream, not a reason to hold this fork.
        "wider_upstream_suite": (tier("/tmp/gpu_conformance_wider.log", int(wider_status)) if run_wider == "1" else {
            "skipped": "test fixtures are unfetched Git LFS pointers"
        }),
    }

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(json.dumps(report, indent=2))

    return 0 if report["verdict"] == "passed" else 1


if (__name__ == "__main__"):
    sys.exit(main())
