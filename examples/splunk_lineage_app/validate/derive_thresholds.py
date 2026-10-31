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
Propose each placeholder threshold from a baseline period of the estate's own scored events.

The guide states most of the app's thresholds to make each rule concrete, and says the values "must be derived from
a baseline period in your own data before the rule goes live". This is the tool for that step. It reads the
registry the app ships, `TA-morpheus-lineage/lookups/rule_thresholds.csv`, and for every placeholder whose row names
the event field it judges, takes that field's values from the scored events of the row's sourcetype and class
inside the baseline period, and proposes the value at the row's quantile by nearest rank
(`morpheus.utils.rule_thresholds.derive_threshold`).

    python examples/splunk_lineage_app/validate/derive_thresholds.py \\
        --events /path/to/exported/jsonlines --start 2026-01-01 --end 2026-02-01 --out proposals.json

The events are what `make_sample_events.py` writes and what a search head exports: one directory, one
`morpheus_score_<layer>.jsonlines` file per score sourcetype, one JSON object per line. With no `--events` the
package's own sample events are read, which shows the tool working and proposes nothing worth deploying, because
the sample corpora plant their attacks inside the period a baseline would be taken over.

**It proposes and never edits.** Each proposal is reported beside the value the search ships with, and a reviewer
who accepts one edits the stanza, its row in the registry and the rule's `thresholds_hash` in `rule_metadata.csv`
together; `tests/morpheus/utils/test_rule_thresholds.py` fails until all three agree. A threshold that moved because
a script said so, without anyone reading why, is the defect the registry exists to prevent.

**The baseline period should be one the estate believes was quiet.** A quantile of a period with an incident in it
places the threshold above the incident. Nothing here can tell a quiet period from a busy one; the period is the
reviewer's statement, which is why it is an argument and why the report records it.

Rows the tool does not propose are reported with the reason: a conventional value, a gate, a time window or a
definitional number is not a measurement a baseline should move, and a placeholder computed inside its search
(`qualifying_subdomains`, the SYN ratio) has no event field to read until the search's own aggregation is exported.
"""

import argparse
import csv
import datetime
import json
import os
import sys

from morpheus.utils.rule_thresholds import derive_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(os.path.dirname(HERE), "TA-morpheus-lineage", "lookups", "rule_thresholds.csv")
SAMPLE_EVENTS = os.path.join(HERE, "sample_events")

TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fUTC"
"""How `morpheus.utils.siem_wire.render_event_time` renders `event_time`."""


def _event_time(event: dict) -> datetime.datetime:
    return datetime.datetime.strptime(event["event_time"], TIME_FORMAT).replace(tzinfo=datetime.timezone.utc)


def _day(text: str) -> datetime.datetime:
    return datetime.datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)


def read_registry(path: str = REGISTRY) -> list[dict]:
    """The registry's rows, as the app ships them."""
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_events(directory: str, sourcetype: str) -> list[dict]:
    """Every event of one score sourcetype in an export directory; none when the file is absent."""
    path = os.path.join(directory, sourcetype.replace("morpheus:score:", "morpheus_score_") + ".jsonlines")

    if (not os.path.exists(path)):
        return []

    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def propose(rows: list[dict],
            events_directory: str,
            start: datetime.datetime = None,
            end: datetime.datetime = None) -> list[dict]:
    """
    One entry per registry row: the shipped value, and the proposal or the reason there is none.

    Parameters
    ----------
    rows : list of dict
        Registry rows, as `read_registry` returns them.
    events_directory : str
        The directory of exported score events.
    start, end : datetime, optional
        The baseline period, start inclusive and end exclusive; unbounded where omitted.

    Returns
    -------
    list of dict
        In registry order, each with `rule_id`, `threshold`, `operator`, `kind`, `shipped`, `status` and, when
        `status` is `proposed`, `proposed` and `samples`.
    """
    cache: dict = {}
    report = []

    for row in rows:
        entry = {
            "rule_id": row["rule_id"],
            "threshold": row["threshold"],
            "operator": row["operator"],
            "kind": row["kind"],
            "shipped": float(row["value"]),
        }

        if (row["kind"] != "placeholder"):
            entry["status"] = f"not proposed: a {row['kind']} value is not a measurement a baseline moves"
        elif (not row["field"]):
            entry["status"] = "not proposed: computed inside the search, so no event field carries it"
        else:
            if (row["sourcetype"] not in cache):
                cache[row["sourcetype"]] = read_events(events_directory, row["sourcetype"])

            values = [
                event.get(row["field"]) for event in cache[row["sourcetype"]]
                if event.get("telemetry_class") == row["telemetry_class"] and (
                    start is None or _event_time(event) >= start) and (end is None or _event_time(event) < end)
            ]
            proposed = derive_threshold(values, row["fires"], float(row["quantile"]))

            if (proposed is None):
                entry["status"] = "not proposed: the baseline period holds no measurement of the field"
            else:
                entry["status"] = "proposed"
                entry["proposed"] = proposed
                entry["samples"] = sum(1 for value in values if value is not None and not isinstance(value, bool))

        report.append(entry)

    return report


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--events", default=SAMPLE_EVENTS, help="directory of morpheus_score_*.jsonlines exports")
    parser.add_argument("--registry", default=REGISTRY, help="the app's rule_thresholds.csv")
    parser.add_argument("--start", type=_day, help="first day of the baseline period, YYYY-MM-DD, inclusive")
    parser.add_argument("--end", type=_day, help="day after the baseline period, YYYY-MM-DD, exclusive")
    parser.add_argument("--out", help="write the report here as JSON rather than to standard output")
    arguments = parser.parse_args(argv)

    report = {
        "events": os.path.abspath(arguments.events),
        "start": arguments.start.date().isoformat() if arguments.start else None,
        "end": arguments.end.date().isoformat() if arguments.end else None,
        "thresholds": propose(read_registry(arguments.registry), arguments.events, arguments.start, arguments.end),
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"

    if (arguments.out):
        with open(arguments.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)

    return 0


if (__name__ == "__main__"):
    sys.exit(main())
