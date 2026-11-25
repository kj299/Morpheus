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
Hunt agent telemetry for malicious agentic AI behavior from the command line.

    ./hunt.py run fixtures/sessions/*.jsonlines fixtures/mcp/*.json*      # every hunt, findings as JSON
    ./hunt.py run --hunt AAH-003 --min-severity high --table telemetry.jsonl
    ./hunt.py audit-config ~/.cursor/mcp.json                              # MCP configuration and tool checks
    ./hunt.py scan-text --file suspicious_issue.md                         # injection indicators in one text
    ./hunt.py baseline approved_tools.json > tool_baseline.json            # for AAH-007 later
    ./hunt.py catalog --markdown

Needs only the Python standard library. Exit status is 0 with no findings at or above `--fail-on` (default
`critical`), 1 with some, and 2 when an input cannot be read.
"""

import argparse
import json
import os
import sys

# The engine is `morpheus.utils.agentic_hunt`, which needs only the standard library; reading it from this checkout
# lets the script run where Morpheus is not installed.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python", "morpheus"))

from morpheus.utils.agentic_hunt import catalog  # noqa: E402  pylint: disable=wrong-import-position
from morpheus.utils.agentic_hunt import events  # noqa: E402  pylint: disable=wrong-import-position
from morpheus.utils.agentic_hunt import inputs  # noqa: E402  pylint: disable=wrong-import-position
from morpheus.utils.agentic_hunt import service  # noqa: E402  pylint: disable=wrong-import-position


def table(result: dict) -> str:
    """Findings as aligned text for a terminal."""
    rows = [("SEVERITY", "HUNT", "SESSION", "TIME", "SUMMARY")]

    for finding in result["findings"]:
        rows.append((finding["severity"],
                     finding["hunt_id"],
                     finding["session_id"] or "-",
                     finding["ts"] or "-",
                     finding["summary"]))

    widths = [max(len(row[column]) for row in rows) for column in range(4)]
    lines = ["  ".join(cell.ljust(width) for (cell, width) in zip(row[:4], widths)) + "  " + row[4] for row in rows]
    coverage = result["coverage"]
    lines.append(f"\n{len(result['findings'])} finding(s) over {coverage['events']} event(s), "
                 f"{coverage['sessions']} session(s), {coverage['configs']} configuration(s)")

    if (coverage["missing_types"]):
        lines.append(f"no {', '.join(coverage['missing_types'])} events: hunts that need them saw nothing")

    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run hunts over telemetry, transcripts, manifests and configurations")
    run.add_argument("paths", nargs="+")
    run.add_argument("--hunt", action="append", dest="hunt_ids", choices=sorted(catalog.BY_ID))
    run.add_argument("--window", type=int, dest="window_seconds", help="seconds a sequence may span (600)")
    run.add_argument("--allow-host",
                     action="append",
                     dest="egress_allowlist",
                     help="a destination that is ordinary here; replaces the default list when given")
    run.add_argument("--baseline", dest="baseline_path", help="tool baseline from the baseline command, for AAH-007")
    run.add_argument("--min-severity", default="info", choices=catalog.SEVERITIES)
    run.add_argument("--fail-on", default="critical", choices=catalog.SEVERITIES)
    run.add_argument("--table", action="store_true", help="print a table rather than JSON")

    audit = commands.add_parser("audit-config", help="audit an MCP client configuration or tool manifest")
    audit.add_argument("path")
    audit.add_argument("--baseline", dest="baseline_path")
    audit.add_argument("--fail-on", default="high", choices=catalog.SEVERITIES)
    audit.add_argument("--table", action="store_true")

    scan = commands.add_parser("scan-text", help="score one text for injection indicators and credentials")
    source = scan.add_mutually_exclusive_group(required=True)
    source.add_argument("--file")
    source.add_argument("--text")

    commands.add_parser("summarize",
                        help="what event types, sessions and times the files cover").add_argument("paths", nargs="+")
    commands.add_parser("baseline", help="digest every tool definition, for AAH-007").add_argument("paths", nargs="+")

    listing = commands.add_parser("catalog", help="the hunts, as JSON or as the skill's reference page")
    listing.add_argument("--markdown", action="store_true")
    listing.add_argument("--hunt", dest="hunt_id", choices=sorted(catalog.BY_ID))

    return parser.parse_args(argv)


def _failed(result: dict, fail_on: str) -> bool:
    floor = catalog.SEVERITIES.index(fail_on)

    return any(catalog.SEVERITIES.index(finding["severity"]) >= floor for finding in result["findings"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        if (args.command in ("run", "audit-config")):
            if (args.command == "run"):
                result = service.run(args.paths,
                                     args.hunt_ids,
                                     window_seconds=args.window_seconds,
                                     egress_allowlist=args.egress_allowlist,
                                     baseline_path=args.baseline_path,
                                     min_severity=args.min_severity)
            else:
                result = service.audit_config(args.path, args.baseline_path)

            print(table(result) if args.table else json.dumps(result, indent=2))

            return 1 if _failed(result, args.fail_on) else 0

        if (args.command == "scan-text"):
            if (args.file):
                with open(args.file, encoding="utf-8") as handle:
                    text = handle.read()
            else:
                text = args.text

            print(json.dumps(service.scan_text(text), indent=2))
        elif (args.command == "summarize"):
            print(json.dumps(service.summarize(args.paths), indent=2))
        elif (args.command == "baseline"):
            print(json.dumps(service.baseline(args.paths), indent=2, sort_keys=True))
        elif (args.markdown):
            sys.stdout.write(catalog.markdown())
        elif (args.hunt_id):
            print(json.dumps(service.explain_hunt(args.hunt_id), indent=2))
        else:
            print(json.dumps(service.list_hunts(), indent=2))
    except (OSError, inputs.InputError, events.TelemetryError) as error:
        print(f"hunt.py: {error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
