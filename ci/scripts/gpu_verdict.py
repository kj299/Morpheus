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
How old the GPU verdict is against the code a CI run tests (guide 13, plan item 9).

The fork's CI runs on a machine with no GPU, so every run tests the CPU path only; the GPU path is verified by
`gpu_conformance.sh` on a machine with a card, which writes `ci/artifacts/gpu_conformance-<time>.json`. A green CI
run said nothing about the GPU, and nothing said how long it had been since anything had. This says it on every run:
the newest artifact's verdict, when and on what device it was made, and how many commits since then have changed the
GPU path -- the package and every test file the conformance tiers run. Zero is a GPU verdict that describes this
code; anything else is code no GPU has run, and the run says so in its summary and as a warning.

An artifact made since this module carries the commit it ran at, and the count is the commits after it. An older
one carries only its time, and the count is the commits after that time. Without the history to count in (a shallow
checkout), the run says that it cannot tell rather than that nothing changed.
"""

import argparse
import dataclasses
import glob
import json
import os
import re
import subprocess
import sys
import typing

ARTIFACTS = os.path.join("ci", "artifacts")
CONFORMANCE = os.path.join("ci", "scripts", "gpu_conformance.sh")
PACKAGE = os.path.join("python", "morpheus", "morpheus")
TIER_FILE = re.compile(r"^\s+(tests/\S+)\s*$")


@dataclasses.dataclass
class Verdict:
    """The newest GPU verdict and what has changed since."""

    artifact: typing.Optional[str]
    verdict: typing.Optional[str] = None
    at: typing.Optional[str] = None
    device: typing.Optional[str] = None
    commit: typing.Optional[str] = None
    changed: typing.Optional[list[str]] = None
    """The commits since the verdict that touched the GPU path, newest first; `None` when they cannot be counted."""

    @property
    def current(self) -> bool:
        return self.verdict == "passed" and self.changed == []


def newest_artifact(root: str) -> typing.Optional[str]:
    """The conformance artifact with the latest time in its name, or `None`."""
    found = sorted(glob.glob(os.path.join(root, ARTIFACTS, "gpu_conformance-*.json")))

    return found[-1] if found else None


def gpu_path(root: str) -> list[str]:
    """What a GPU verdict covers: the package, and every test file `gpu_conformance.sh` names in its tiers."""
    with open(os.path.join(root, CONFORMANCE), encoding="utf-8") as handle:
        files = [match[1] for match in (TIER_FILE.match(line) for line in handle) if match]

    return [PACKAGE, CONFORMANCE] + sorted(set(files))


def _git(root: str, *arguments: str) -> typing.Optional[str]:
    result = subprocess.run(["git", "-C", root, *arguments], capture_output=True, text=True, check=False)

    return result.stdout if result.returncode == 0 else None


def changes_since(root: str, paths: list[str], commit: typing.Optional[str], at: str) -> typing.Optional[list[str]]:
    """The commits on `HEAD` after the verdict that touch `paths`, or `None` if the history does not reach it."""
    if (commit is not None):
        if (_git(root, "cat-file", "-e", f"{commit}^{{commit}}") is None):
            return None

        listed = _git(root, "log", "--format=%h %s", f"{commit}..HEAD", "--", *paths)
    else:
        # A shallow checkout cannot say what came after a time it does not reach back to.
        oldest = _git(root, "log", "--reverse", "--format=%cI", "-1", "--max-parents=0", "HEAD")
        shallow = (_git(root, "rev-parse", "--is-shallow-repository") or "").strip() == "true"

        if (oldest is None or (shallow and oldest.strip() > at)):
            return None

        listed = _git(root, "log", "--format=%h %s", f"--since={at}", "HEAD", "--", *paths)

    return None if listed is None else [line for line in listed.splitlines() if line]


def read(root: str) -> Verdict:
    """The newest artifact's verdict, and the GPU-path commits since it."""
    path = newest_artifact(root)

    if (path is None):
        return Verdict(artifact=None)

    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)

    verdict = Verdict(artifact=os.path.relpath(path, root),
                      verdict=report.get("verdict"),
                      at=report.get("at"),
                      device=report.get("device"),
                      commit=report.get("commit"))
    verdict.changed = changes_since(root, gpu_path(root), verdict.commit, verdict.at or "")

    return verdict


def render(verdict: Verdict) -> tuple[str, typing.Optional[str]]:
    """The run summary's markdown, and the warning to raise, if any."""
    if (verdict.artifact is None):
        return ("## GPU path\n\nNo GPU conformance artifact is committed: no GPU has run this code.\n",
                "No GPU conformance artifact: this run verifies the CPU path only.")

    lines = [
        "## GPU path",
        "",
        "This run has no GPU and tests the CPU path only. The newest GPU verdict:",
        "",
        f"- **{verdict.verdict}** at {verdict.at} on {verdict.device}",
        f"- artifact `{verdict.artifact}`" + (f", commit `{verdict.commit[:12]}`" if verdict.commit else ""),
    ]

    if (verdict.changed is None):
        lines.append("- the checkout's history does not reach the verdict, so whether the GPU path changed since "
                     "cannot be told")
        warning = f"The GPU verdict of {verdict.at} cannot be compared with this code: the history does not reach it."
    elif (verdict.changed):
        lines.append(f"- **{len(verdict.changed)} commits since have changed the GPU path**, so the verdict does not "
                     "describe this code; run `ci/scripts/gpu_conformance.sh` on a machine with a GPU")
        lines += ["", "<details><summary>The commits</summary>", ""]
        lines += [f"- {line}" for line in verdict.changed]
        lines += ["", "</details>"]
        warning = (f"The GPU path is unverified for this code: {len(verdict.changed)} commits have changed it since "
                   f"the GPU verdict of {verdict.at}.")
    else:
        lines.append("- no commit since has changed the GPU path, so the verdict describes this code")
        warning = None if verdict.verdict == "passed" else f"The GPU verdict of {verdict.at} is {verdict.verdict}."

    return ("\n".join(lines) + "\n", warning)


def main(argv: typing.Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--root", default=os.getcwd(), help="The repository's root.")
    arguments = parser.parse_args(argv)

    (summary, warning) = render(read(arguments.root))
    print(summary)

    if (os.environ.get("GITHUB_STEP_SUMMARY")):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(summary + "\n")

    if (warning is not None):
        print(f"::warning title=GPU path::{warning}")

    return 0


if (__name__ == "__main__"):
    sys.exit(main())
