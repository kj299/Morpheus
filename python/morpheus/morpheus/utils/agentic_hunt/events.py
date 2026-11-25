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
Agent telemetry, read into one event shape the hunts share.

Two inputs are read, one JSON object per line:

- **Agent events**, the schema in `.claude/skills/agentic-ai-threat-hunt/references/telemetry-schema.md`: what an
  agent was asked, what it called, what came back, and what its processes did on the host.
- **MCP JSON-RPC transcripts**: one line per message between an MCP client and server, as a proxy or a client log
  records them, `{"ts", "session_id", "server", "message": {"jsonrpc": "2.0", ...}}`. `tools/call` requests become
  tool calls, their responses tool results, and `tools/list` responses the manifest a server offered.

A file whose top level is a JSON object holding `mcpServers` is an MCP client configuration; `mcp_audit` reads those.
"""

import datetime
import json
import os

EVENT_TYPES = ("prompt",
               "model_output",
               "tool_call",
               "tool_result",
               "document",
               "tool_manifest",
               "process",
               "file",
               "network",
               "auth")
"""Every event type the hunts read. `document` is content retrieved into a context other than by a tool call."""


class TelemetryError(ValueError):
    """A line that is not an event either reader understands."""


def parse_time(value) -> datetime.datetime | None:
    """An ISO 8601 time, or epoch seconds, as an aware UTC time; `None` when the event carries none."""
    if (value is None or value == ""):
        return None

    if (isinstance(value, (int, float))):
        return datetime.datetime.fromtimestamp(value, tz=datetime.timezone.utc)

    parsed = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    return parsed if parsed.tzinfo else parsed.replace(tzinfo=datetime.timezone.utc)


def _section(record: dict, name: str) -> dict:
    value = record.get(name)

    return dict(value) if isinstance(value, dict) else {}


def normalize(record: dict, ref: str) -> dict:
    """One agent event in the shared shape. Unknown fields are dropped; a missing section is an empty one."""
    kind = record.get("type")

    if (kind not in EVENT_TYPES):
        raise TelemetryError(f"{ref}: event type {kind!r} is not one of {', '.join(EVENT_TYPES)}")

    args = record.get("args")

    return {
        "ref": ref,
        "ts": parse_time(record.get("ts")),
        "session_id": str(record.get("session_id") or ""),
        "agent": str(record.get("agent") or ""),
        "host": str(record.get("host") or ""),
        "principal": str(record.get("principal") or ""),
        "type": kind,
        "tool": str(record.get("tool") or ""),
        "server": str(record.get("server") or ""),
        "args": dict(args) if isinstance(args, dict) else {},
        "content": record.get("content") if isinstance(record.get("content"), str) else "",
        "tools": list(record.get("tools") or []),
        "process": _section(record, "process"),
        "file": _section(record, "file"),
        "network": _section(record, "network"),
        "auth": _section(record, "auth"),
    }


def _result_text(result: dict) -> str:
    parts = []

    for item in result.get("content") or []:
        if (isinstance(item, dict) and item.get("type") == "text"):
            parts.append(str(item.get("text", "")))
        elif (isinstance(item, dict) and item.get("type") == "resource"):
            parts.append(str((item.get("resource") or {}).get("text", "")))

    if (isinstance(result.get("structuredContent"), (dict, list))):
        parts.append(json.dumps(result["structuredContent"], sort_keys=True))

    return "\n".join(parts)


class _TranscriptReader:
    """Turns MCP JSON-RPC messages into events, pairing each response with the request it answers."""

    def __init__(self):
        self.pending: dict = {}

    def read(self, record: dict, ref: str) -> dict | None:
        message = record["message"]
        session = str(record.get("session_id") or "")
        server = str(record.get("server") or "")
        base = {"ts": record.get("ts"), "session_id": session, "server": server, "agent": record.get("agent")}
        key = (session, server, json.dumps(message.get("id"), sort_keys=True))

        if ("method" in message):
            if ("id" in message):
                self.pending[key] = message

            if (message["method"] == "tools/call"):
                params = message.get("params") or {}
                return normalize(
                    {
                        **base, "type": "tool_call", "tool": params.get("name"), "args": params.get("arguments")
                    }, ref)

            return None

        request = self.pending.pop(key, None)

        if (request is None or not isinstance(message.get("result"), dict)):
            return None

        result = message["result"]

        if (request["method"] == "tools/list"):
            return normalize({**base, "type": "tool_manifest", "tools": result.get("tools")}, ref)

        if (request["method"] == "tools/call"):
            return normalize(
                {
                    **base,
                    "type": "tool_result",
                    "tool": (request.get("params") or {}).get("name"),
                    "content": _result_text(result),
                    "args": {
                        "is_error": bool(result.get("isError"))
                    }
                },
                ref)

        return None


def is_mcp_config(document) -> bool:
    return isinstance(document, dict) and isinstance(document.get("mcpServers"), dict)


def read_lines(path: str) -> list[dict]:
    """The events in one JSON-lines file, of either kind, in file order."""
    events = []
    reader = _TranscriptReader()

    with open(path, encoding="utf-8") as handle:
        for (number, line) in enumerate(handle, start=1):
            if (not line.strip()):
                continue

            ref = f"{os.path.basename(path)}:{number}"

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TelemetryError(f"{ref}: not JSON ({error.msg})") from error

            if (not isinstance(record, dict)):
                raise TelemetryError(f"{ref}: a line must be a JSON object")

            if (isinstance(record.get("message"), dict) and record["message"].get("jsonrpc") == "2.0"):
                event = reader.read(record, ref)
            else:
                event = normalize(record, ref)

            if (event is not None):
                events.append(event)

    return events


def order(events: list[dict]) -> list[dict]:
    """Events by session, then time, keeping file order where times tie or are missing."""
    indexed = list(enumerate(events))
    earliest = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    indexed.sort(key=lambda pair: (pair[1]["session_id"], pair[1]["ts"] or earliest, pair[0]))

    return [event for (_, event) in indexed]


def summarize(events: list[dict]) -> dict:
    """What a set of events covers: how many of each type, how many sessions, and the time they span."""
    times = [event["ts"] for event in events if event["ts"] is not None]
    by_type = {}

    for event in events:
        by_type[event["type"]] = by_type.get(event["type"], 0) + 1

    return {
        "events": len(events),
        "sessions": len({event["session_id"]
                         for event in events}),
        "by_type": dict(sorted(by_type.items())),
        "missing_types": [kind for kind in EVENT_TYPES if kind not in by_type],
        "first": min(times).isoformat() if times else None,
        "last": max(times).isoformat() if times else None,
    }
