#!/usr/bin/env python3
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
Index the sample events into the validation container and dispatch every saved search, recording what each returns.

Runs inside the container, under Splunk's own interpreter (`splunk cmd python3`), so the machine driving it needs
Docker and nothing else; `run_search_head.sh` is the host side. Standard library only, and every call to Splunk
goes through its command line, which already knows how to reach its own management port.

Two things stand between the checked-in events and a faithful run, and both are handled here rather than left to
whoever runs it:

**The events are dated 1970.** The corpora count time from the epoch, so every timestamp in `sample_events/` falls
in January 1970, and no search head accepts that: `MAX_DAYS_AGO` cannot exceed about thirty years, and an event
outside it is quietly given some other time, which is the failure the procedure's own first check exists to catch.
So every timestamp is moved forward by one whole number of weeks, chosen so the newest event lands a few days
before the run. A whole number of weeks keeps the hour of day, the weekday and every five-minute, hourly and daily
boundary where it was, and every search compares `_time` only with another `_time`, so what each rule decides is
unchanged. Only the timestamp strings move; the window identifiers and buckets in the events are left as they are,
and they are what the searches join on. The span of the events is still fifty-seven days, which is wider than
layer 1's thirty-day `MAX_DAYS_AGO`, so `run_search_head.sh` installs a validation-only app whose `props.conf` raises it
for this run alone.

**The searches look back from now.** Every stanza dispatches over a window like `-2h@m` to `-5m@m`. Each is run
here as `| savedsearch` with an explicit time range covering every event, which overrides the stanza's own window,
so a search is asked the same question its schedule would ask, over all the data at once rather than a slice.
That is the same simplification the Python recomputation in `expected_results.json` makes.

The order follows VALIDATION.md: the binding refreshes first, so the lookups exist; the two predictive searches,
so R-B-L7-002 reads a populated watchlist; every other detection, each writing its rows into `behavior_risk`; the
summary search, so R-P-L3-005 has something to read; R-P-L3-005; Chain assembly, which sums what the detections
wrote; and the expiry jobs last, because the events are historical and expiry drops what they wrote. A search
that reads an index another search wrote is dispatched only once that index holds every row that was written to
it: `collect` hands its rows to the indexer and returns, and they become searchable a few seconds later.
"""

import argparse
import csv
import datetime
import glob
import io
import json
import os
import re
import subprocess
import sys
import time

SPLUNK = "/opt/splunk/bin/splunk"
APP = "TA-morpheus-lineage"

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_EVENTS = os.path.join(HERE, "sample_events")
SAVED_SEARCHES = os.path.join(os.path.dirname(HERE), APP, "default", "savedsearches.conf")
RESULTS = "/tmp/search_head_results.json"
"""Written inside the container; `run_search_head.sh` copies it to `validate/search_head_results.json`."""

WEEK_SECONDS = 7 * 86400
LANDING_SECONDS = 3 * 86400
"""How long before the run the newest shifted event lands: inside every window the stanzas use, clear of now."""

TIMESTAMP = re.compile(r'"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(\d{6})UTC"')
"""Every timestamp the wire format renders, in the one form `props.conf` parses."""

INDEX_BY_PREFIX = (("morpheus:edge", "behavior_lineage"), ("morpheus:score:", "behavior_events"),
                   ("binding:", "behavior_bindings"), ("context:", "behavior_context"))
"""Where each sourcetype is indexed, as VALIDATION.md's indexing loop has it."""

CALL_TIMEOUT_SECONDS = 900
"""How long one call to the splunk client may take; a search that has not finished by then is recorded as an error
rather than left to hold the whole run."""

PREDICTIVE = ("R-P-L5-006 - Drift trajectory", "R-P-L7-006 - Access breadth trajectory")
SUMMARY = "Behavior summary - per-layer scores"
TRAJECTORY = "R-P-L3-005 - Fan-out trajectory"
CHAIN = "Chain assembly - cross-layer risk"


def sourcetype_of(path: str) -> str:
    return os.path.basename(path)[:-len(".jsonlines")].replace("_", ":")


def index_of(sourcetype: str) -> str:
    for (prefix, index) in INDEX_BY_PREFIX:
        if (sourcetype.startswith(prefix)):
            return index

    return "behavior_events"


def _epoch(match: re.Match) -> float:
    whole = datetime.datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc)

    return whole.timestamp() + int(match.group(2)) / 1e6


def newest_and_oldest(lines: list[str]) -> tuple[float, float]:
    stamps = [_epoch(match) for line in lines for match in TIMESTAMP.finditer(line)]

    return (max(stamps), min(stamps))


def week_offset(newest: float, now: float) -> int:
    """The whole number of weeks that moves the newest event to just before `now - LANDING_SECONDS`."""
    return int((now - LANDING_SECONDS - newest) // WEEK_SECONDS) * WEEK_SECONDS


def shift_line(line: str, offset_seconds: int) -> str:
    """Move every timestamp in one rendered event by `offset_seconds`, leaving every other byte where it was."""

    def moved(match: re.Match) -> str:
        whole = datetime.datetime.strptime(match.group(1),
                                           "%Y-%m-%dT%H:%M:%S") + datetime.timedelta(seconds=offset_seconds)

        return f'"{whole.strftime("%Y-%m-%dT%H:%M:%S")}.{match.group(2)}UTC"'

    return TIMESTAMP.sub(moved, line)


def stanzas(path: str = SAVED_SEARCHES) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return re.findall(r"^\[(.+)\]\s*$", handle.read(), re.MULTILINE)


def ordered(names: list[str]) -> list[tuple[str, str]]:
    """Every stanza, with the phase it runs in, in the order VALIDATION.md prescribes."""
    refresh = [name for name in names if name.startswith("Binding lookup") and "refresh" in name]
    expiry = [name for name in names if "expiry" in name]
    placed = set(refresh) | set(expiry) | set(PREDICTIVE) | {SUMMARY, TRAJECTORY, CHAIN}
    rest = [name for name in names if name not in placed]

    return ([(name, "refresh") for name in refresh] + [(name, "predictive") for name in PREDICTIVE] +
            [(name, "search") for name in rest] + [(SUMMARY, "summary"), (TRAJECTORY, "trajectory"),
                                                   (CHAIN, "assembly")] + [(name, "expiry") for name in expiry])


def search_text(name: str, path: str = SAVED_SEARCHES) -> str:
    """One stanza's SPL with its continuation lines folded."""
    with open(path, encoding="utf-8") as handle:
        folded = re.sub(r"\\\s*\r?\n\s*", " ", handle.read())

    body = folded.split(f"[{name}]", 1)[1].split("\n[", 1)[0]

    return re.search(r"^search\s*=\s*(.*)$", body, re.MULTILINE).group(1).strip()


def chain_risk_query(path: str = SAVED_SEARCHES) -> str:
    """
    Chain assembly as shipped, up to its risk threshold: the risk on every chain that spans three layers, counted by
    total. An empty Chain assembly says no chain cleared the threshold; this says how close each came.
    """
    head = search_text(CHAIN, path).split("| where total_risk", 1)[0]

    return head + "| eval total_risk = coalesce(total_risk, 0) | stats count BY total_risk"


def writes_risk(name: str) -> bool:
    """Every detection, and only the detections, ends in `| collect index=behavior_risk`."""
    return name.startswith("R-")


def splunk(*arguments: str, password: str, check: bool = True) -> subprocess.CompletedProcess:
    try:
        completed = subprocess.run([SPLUNK, *arguments, "-auth", f"admin:{password}"],
                                   capture_output=True,
                                   text=True,
                                   stdin=subprocess.DEVNULL,
                                   timeout=CALL_TIMEOUT_SECONDS,
                                   check=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"splunk {arguments[0]} did not return in {CALL_TIMEOUT_SECONDS}s") from error

    if (check and completed.returncode != 0):
        raise RuntimeError(f"splunk {arguments[0]} failed: {completed.stderr.strip() or completed.stdout.strip()}")

    return completed


def search(query: str, earliest: int, latest: int, password: str) -> list[dict]:
    completed = splunk("search",
                       query,
                       "-app",
                       APP,
                       "-earliest_time",
                       str(earliest),
                       "-latest_time",
                       str(latest),
                       "-output",
                       "csv",
                       "-maxout",
                       "0",
                       password=password)

    return list(csv.DictReader(io.StringIO(completed.stdout)))


def counts_by_sourcetype(earliest: int, latest: int, password: str) -> dict:
    rows = search("| tstats count WHERE index=behavior_* BY sourcetype", earliest, latest, password)

    return {row["sourcetype"]: int(row["count"]) for row in rows}


def searchable_by_sourcetype(earliest: int, latest: int, password: str) -> dict:
    """
    The same count as `counts_by_sourcetype`, by an ordinary search. The two disagree for a while after indexing:
    on 2026-10-09 the index-time count reached every sourcetype's total while a search still saw ten of layer 1's
    sixty-five summary groups, and the summary search dispatched in that interval returned 3,649 rows instead of
    4,335. What the searches see is what has to be complete.
    """
    rows = search("index=behavior_* | stats count BY sourcetype", earliest, latest, password)

    return {row["sourcetype"]: int(row["count"]) for row in rows}


def wait_for(expected: dict, earliest: int, latest: int, password: str, limit_seconds: int = 900) -> dict:
    deadline = time.time() + limit_seconds
    seen: dict = {}

    while (time.time() < deadline):
        seen = counts_by_sourcetype(earliest, latest, password)

        if (all(seen.get(sourcetype, 0) >= count for (sourcetype, count) in expected.items())):
            seen = searchable_by_sourcetype(earliest, latest, password)

            if (all(seen.get(sourcetype, 0) >= count for (sourcetype, count) in expected.items())):
                return seen

        time.sleep(5)

    return seen


def roll(indexes: set, password: str):
    """
    Roll each index's hot buckets to warm, so every event already indexed is committed before a search reads it.

    A count is not enough. On 2026-10-09 every count, index-time and search-time, reported layer 1's 305 events
    while R-D-L1-001 -- which needs one of them, the poll whose serial changed -- returned nothing, and the same
    search returned its row a minute later and on every run since. Events become searchable in a hot bucket in no
    order a count can see; a warm bucket is complete.
    """
    for index in sorted(indexes):
        splunk("_internal", "call", f"/data/indexes/{index}/roll-hot-buckets", "-method", "POST", password=password)


def index_count(index: str, earliest: int, latest: int, password: str) -> int:
    rows = search(f"index={index} | stats count", earliest, latest, password)

    return int(rows[0]["count"]) if rows else 0


def wait_for_index(index: str, count: int, earliest: int, latest: int, password: str, limit_seconds: int = 300) -> int:
    """Wait until an index another search writes holds `count` events, and say how many it holds."""
    deadline = time.time() + limit_seconds
    seen = index_count(index, earliest, latest, password)

    while (seen < count and time.time() < deadline):
        time.sleep(3)
        seen = index_count(index, earliest, latest, password)

    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--password", default=os.environ.get("SPLUNK_PASSWORD"))
    parser.add_argument("--out", default=RESULTS)
    arguments = parser.parse_args()

    if (not arguments.password):
        print("SPLUNK_PASSWORD is not set; it is the admin password the container was started with.")
        return 2

    password = arguments.password
    now = time.time()
    files = sorted(glob.glob(os.path.join(SAMPLE_EVENTS, "*.jsonlines")))
    contents = {}

    for path in files:
        with open(path, encoding="utf-8") as handle:
            contents[path] = handle.read().splitlines()

    (newest, oldest) = newest_and_oldest([line for lines in contents.values() for line in lines])
    offset = week_offset(newest, now)
    earliest = int(oldest + offset - 86400)
    latest = int(now + 3600)

    print(f"shifting every timestamp by {offset // WEEK_SECONDS} weeks; events now span "
          f"{datetime.datetime.utcfromtimestamp(oldest + offset):%Y-%m-%d} to "
          f"{datetime.datetime.utcfromtimestamp(newest + offset):%Y-%m-%d}")

    staged = "/tmp/morpheus_sample_events"
    os.makedirs(staged, exist_ok=True)
    expected_counts = {}

    for (path, lines) in contents.items():
        sourcetype = sourcetype_of(path)
        target = os.path.join(staged, os.path.basename(path))

        with open(target, "w", encoding="utf-8") as handle:
            handle.write("".join(shift_line(line, offset) + "\n" for line in lines))

        expected_counts[sourcetype] = len(lines)
        splunk("add", "oneshot", target, "-index", index_of(sourcetype), "-sourcetype", sourcetype, password=password)

    indexed = wait_for(expected_counts, earliest, latest, password)
    roll({index_of(sourcetype) for sourcetype in expected_counts}, password)
    indexed = wait_for(expected_counts, earliest, latest, password)
    print(f"indexed: {indexed}")

    checks = {
        "time_drift_seconds":
            search(
                "index=behavior_events sourcetype=morpheus:score:l2 | eval drift = _time - _indextime "
                "| stats max(drift) AS max_drift min(drift) AS min_drift",
                earliest,
                latest,
                password),
        "binding_tables":
            search("index=behavior_bindings sourcetype=binding:bucketed | stats count BY binding_table",
                   earliest,
                   latest,
                   password),
    }

    results: dict = {}
    order = ordered(stanzas())
    written = {"behavior_risk": 0, "behavior_summary": 0}

    for (name, phase) in order:
        # A reader of an index another search writes waits for that index to hold everything written to it.
        if (phase in ("trajectory", "assembly")):
            wait_for_index("behavior_risk", written["behavior_risk"], earliest, latest, password)
            roll({"behavior_risk"}, password)

        if (phase == "trajectory"):
            wait_for_index("behavior_summary", written["behavior_summary"], earliest, latest, password)
            roll({"behavior_summary"}, password)

        query = f'| savedsearch "{name}" | stats count AS rows'

        try:
            rows = search(query, earliest, latest, password)
            results[name] = {"phase": phase, "rows": int(rows[0]["rows"]) if rows else 0}
        except RuntimeError as error:
            results[name] = {"phase": phase, "error": str(error)[-2000:]}

        print(f"  {phase:>10}  {results[name].get('rows', 'ERROR')!s:>6}  {name}")

        if (writes_risk(name)):
            written["behavior_risk"] += results[name].get("rows", 0)

        if (name == SUMMARY):
            written["behavior_summary"] += results[name].get("rows", 0)

        if (name == CHAIN):
            # Every detection has run, so the risk index should hold exactly what they returned between them.
            held = wait_for_index("behavior_risk", written["behavior_risk"], earliest, latest, password)
            checks["three_layer_chain_risk"] = search(chain_risk_query(), earliest, latest, password)
            checks["risk_records"] = {
                "written": written["behavior_risk"],
                "indexed": held,
                "by_rule": search("index=behavior_risk | stats count BY rule_id", earliest, latest, password),
            }

        if (name == PREDICTIVE[-1]):
            checks["principal_watchlist"] = search("| inputlookup principal_watchlist | stats count BY rule_id reason",
                                                   earliest,
                                                   latest,
                                                   password)

    version = splunk("version", password=password, check=False).stdout.strip()
    btool = subprocess.run([SPLUNK, "btool", "check", f"--app={APP}"],
                           capture_output=True,
                           text=True,
                           stdin=subprocess.DEVNULL,
                           timeout=CALL_TIMEOUT_SECONDS,
                           check=False)

    report = {
        "splunk_version":
            version,
        "at":
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "timestamp_shift_weeks":
            offset // WEEK_SECONDS,
        "dispatch_window": {
            "earliest": earliest, "latest": latest
        },
        "indexed":
            dict(sorted(indexed.items())),
        "expected_indexed":
            dict(sorted(expected_counts.items())),
        "btool_check": (btool.stdout + btool.stderr).strip(),
        "checks":
            checks,
        "order": [name for (name, _) in order],
        "searches":
            results,
        "measures": ("what each shipped saved search returns on one Splunk instance over the checked-in sample "
                     "events, with every timestamp moved by a whole number of weeks and each search dispatched "
                     "over all of them at once. A row count per search, not the rows themselves."),
    }

    with open(arguments.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    errors = sorted(name for (name, entry) in results.items() if "error" in entry)
    print(f"\n{len(results)} searches dispatched, {len(errors)} errored. Results written to {arguments.out}.")

    return 1 if errors else 0


if (__name__ == "__main__"):
    sys.exit(main())
