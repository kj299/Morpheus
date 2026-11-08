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
Rows per second per stage, against a committed baseline (guide 13, plan item 15).

No benchmark measured the telemetry and lineage stages, so a change that made one ten times slower would have
passed every test. This runs every segment runner over its committed samples, times each telemetry and lineage
stage's `on_data` -- where every one of them does its work -- and records the rows it read and the seconds it took.

**A runner's speed is not a stage's.** The fork's CI runs on whatever machine GitHub gives it, so seconds alone
would move with the machine. Each run also times a fixed pandas workload, and a stage's throughput is reported in
rows per second times that workload's seconds: rows per unit of the machine's own speed, which stays put when the
machine changes and halves when the stage does. A stage is compared only when it spent long enough in both runs to
be measured rather than sampled (`MEASURABLE_SECONDS`), and each segment runs `--repeats` times with the fastest
kept, since a run is only ever slowed by noise, never sped up.

`--write` records the baseline, `ci/artifacts/stage_throughput_baseline.json`. Without it the run compares against
the baseline: every stage below half its baseline is a warning on the run and a row in its summary, and every stage
that is new or gone is named. It never fails the run; it makes the change visible.
"""

import argparse
import contextlib
import functools
import importlib
import json
import os
import pathlib
import pkgutil
import sys
import tempfile
import time
import typing

import numpy as np
import pandas as pd

BASELINE = os.path.join("ci", "artifacts", "stage_throughput_baseline.json")
MEASURED_PACKAGES = ("morpheus.stages.telemetry", "morpheus.stages.lineage")
"""The stages measured: every telemetry and lineage stage, the fork's own."""

MEASURABLE_SECONDS = 0.01
"""A stage that spent less than this in a run is reported but not compared: its time is mostly the clock's."""

HALVED = 0.5
"""A stage below this fraction of its baseline throughput is flagged."""

CALIBRATION_ROWS = 200_000


def calibrate(repeats: int = 5) -> float:
    """The fastest of `repeats` runs of a fixed pandas workload, in seconds: this machine's unit of speed."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "key": rng.integers(0, 1000, CALIBRATION_ROWS),
        "value": rng.random(CALIBRATION_ROWS),
        "text": rng.integers(0, 50, CALIBRATION_ROWS).astype(str),
    })
    best = float("inf")

    for _ in range(repeats):
        start = time.perf_counter()
        grouped = frame.groupby("key")["value"].agg(["sum", "mean", "max"])
        frame.sort_values(["text", "value"]).merge(grouped, left_on="key", right_index=True)
        frame["text"].str.upper().value_counts()
        best = min(best, time.perf_counter() - start)

    return best


def stage_classes() -> list[type]:
    """Every class the measured packages define that has an `on_data`, its own or inherited."""
    classes = []

    for package_name in MEASURED_PACKAGES:
        package = importlib.import_module(package_name)

        for info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package_name}.{info.name}")

            for value in vars(module).values():
                if (isinstance(value, type) and value.__module__ == module.__name__ and hasattr(value, "on_data")):
                    classes.append(value)

    return sorted(classes, key=lambda cls: cls.__name__)


def measured_classes() -> list[type]:
    """The stage classes that define `on_data` themselves: the ones whose method is wrapped."""
    return [cls for cls in stage_classes() if "on_data" in vars(cls)]


def rows_in(message: typing.Any) -> int:
    """How many rows a message carries: a `MessageMeta`'s, or a `ControlMessage` payload's."""
    if (hasattr(message, "payload")):
        meta = message.payload()

        return 0 if meta is None else meta.count

    return getattr(message, "count", 0)


@contextlib.contextmanager
def instrumented(classes: typing.Iterable[type], totals: dict):
    """
    Time every call of each class's `on_data`, adding seconds, rows and calls to `totals[stage]`.

    A stage is named by the class of the instance doing the work, so the layer 3 to 7 stages, which inherit
    `on_data` from `TelemetryStage`, are each measured as themselves. A call made inside another on the same
    instance -- a subclass's `on_data` calling its base's -- is part of the outer call and is not counted twice.
    """
    originals = {}
    running: set = set()

    for cls in classes:
        original = vars(cls)["on_data"]
        originals[cls] = original

        @functools.wraps(original)
        def timed(self, message, *args, __original=original, **kwargs):
            if (id(self) in running):
                return __original(self, message, *args, **kwargs)

            running.add(id(self))
            start = time.perf_counter()

            try:
                return __original(self, message, *args, **kwargs)
            finally:
                running.discard(id(self))
                entry = totals.setdefault(type(self).__name__, {"seconds": 0.0, "rows": 0, "calls": 0})
                entry["seconds"] += time.perf_counter() - start
                entry["rows"] += rows_in(message)
                entry["calls"] += 1

        setattr(cls, "on_data", timed)

    try:
        yield totals
    finally:
        for (cls, original) in originals.items():
            setattr(cls, "on_data", original)


def measure(example: pathlib.Path, segments: typing.Optional[list[str]], repeats: int) -> dict:
    """Each measured stage's fastest run over every segment's samples: its rows and its seconds, summed by stage."""
    sys.path.insert(0, str(example))
    segment_runner = importlib.import_module("segment_runner")
    names = sorted(segments or segment_runner.SEGMENTS)
    classes = measured_classes()
    best: dict = {}

    for segment in names:
        runs = []

        for _ in range(repeats):
            totals: dict = {}

            with tempfile.TemporaryDirectory() as output, instrumented(classes, totals):
                segment_runner.run(segment, example / "samples" / segment, output_dir=pathlib.Path(output))

            runs.append(totals)

        for stage in {name for run in runs for name in run}:
            fastest = min((run[stage] for run in runs if stage in run), key=lambda entry: entry["seconds"])
            entry = best.setdefault(stage, {"seconds": 0.0, "rows": 0, "calls": 0, "segments": []})
            entry["seconds"] += fastest["seconds"]
            entry["rows"] += fastest["rows"]
            entry["calls"] += fastest["calls"]
            entry["segments"].append(segment)

    return best


def throughput(stages: dict, unit_seconds: float) -> dict:
    """Each stage's rows per second, and rows per unit of the machine's speed."""
    return {
        name: {
            **entry,
            "rows_per_second": entry["rows"] / entry["seconds"] if entry["seconds"] > 0 else None,
            "rows_per_unit": entry["rows"] / entry["seconds"] * unit_seconds if entry["seconds"] > 0 else None,
        }
        for (name, entry) in sorted(stages.items())
    }


def compare(current: dict, baseline: dict) -> dict:
    """
    Each stage's throughput as a fraction of its baseline's, and which stages fall below `HALVED`.

    Returns
    -------
    dict
        `ratios` (stage to fraction, for every stage measurable in both runs), `halved` (those below `HALVED`),
        `new` and `gone` (stages only one run measured at all).
    """
    (now, then) = (current["stages"], baseline["stages"])
    ratios = {}

    for name in sorted(set(now) & set(then)):
        if (min(now[name]["seconds"], then[name]["seconds"]) >= MEASURABLE_SECONDS and then[name]["rows_per_unit"]):
            ratios[name] = now[name]["rows_per_unit"] / then[name]["rows_per_unit"]

    return {
        "ratios": ratios,
        "halved": sorted(name for (name, ratio) in ratios.items() if ratio < HALVED),
        "new": sorted(set(now) - set(then)),
        "gone": sorted(set(then) - set(now)),
    }


def render(current: dict, comparison: typing.Optional[dict]) -> str:
    """The run summary's markdown table."""
    lines = [
        "## Stage throughput",
        "",
        f"Calibration workload: {current['unit_seconds'] * 1000:.1f} ms. Throughput is rows per second times that, "
        "so it does not move with the runner.",
        "",
        "| Stage | Rows | Seconds | Rows/s | Rows/unit | Of baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for (name, entry) in current["stages"].items():
        ratio = None if comparison is None else comparison["ratios"].get(name)
        flag = "" if ratio is None or ratio >= HALVED else " **halved**"
        lines.append(f"| `{name}` | {entry['rows']:,} | {entry['seconds']:.3f} | "
                     f"{(entry['rows_per_second'] or 0):,.0f} | {(entry['rows_per_unit'] or 0):,.0f} | "
                     f"{'-' if ratio is None else f'{ratio:.2f}'}{flag} |")

    if (comparison is not None):
        for (label, names) in (("New since the baseline", comparison["new"]), ("Gone since the baseline",
                                                                               comparison["gone"])):
            if (names):
                lines += ["", f"{label}: " + ", ".join(f"`{name}`" for name in names)]

    return "\n".join(lines) + "\n"


def main(argv: typing.Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=os.getcwd(), help="The repository's root.")
    parser.add_argument("--segments", nargs="*", help="The segments to run; every one by default.")
    parser.add_argument("--repeats", type=int, default=3, help="Runs per segment; the fastest is kept.")
    parser.add_argument("--write", action="store_true", help="Record the run as the baseline.")
    parser.add_argument("--output", help="Also write the run's measurements here.")
    arguments = parser.parse_args(argv)

    root = pathlib.Path(arguments.root)
    unit = calibrate()
    current = {
        "unit_seconds":
            unit,
        "stages":
            throughput(measure(root / "examples" / "behavioral_analytics", arguments.segments, arguments.repeats), unit)
    }

    for path in [arguments.output] + ([str(root / BASELINE)] if arguments.write else []):
        if (path):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(current, handle, indent=1, sort_keys=True)
                handle.write("\n")

    comparison = None

    if (not arguments.write and (root / BASELINE).exists()):
        with open(root / BASELINE, encoding="utf-8") as handle:
            comparison = compare(current, json.load(handle))

    summary = render(current, comparison)
    print(summary)

    if (os.environ.get("GITHUB_STEP_SUMMARY")):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(summary + "\n")

    for name in (comparison or {}).get("halved", []):
        print(f"::warning title=Stage throughput::{name} runs at {comparison['ratios'][name]:.2f} of its baseline "
              "throughput")

    return 0


if (__name__ == "__main__"):
    sys.exit(main())
