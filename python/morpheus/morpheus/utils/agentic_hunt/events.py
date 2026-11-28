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

Real telemetry is untidy, and the reader is built for that. A value of the wrong type is coerced or dropped, a time
in any common form is read, and a line that still cannot be read is **skipped and counted**, never allowed to end
the run: the first version lost every finding of every file to one truncated line. What was skipped is returned
beside the events so that a caller can say so, loudly; a skipped line is a blind spot, and an attacker who can
write telemetry can make one.
"""

import datetime
import json
import math
import os
import re

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

MAX_LINE = 8 * 1024 * 1024
"""The longest line read as one event. A longer one is skipped and counted."""

MAX_NESTING = 64


class TelemetryError(ValueError):
    """A record that is not an event either reader understands."""


MAX_TIME_TEXT = 64
_ISO = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(?:[.,](\d{1,9}))?)?[ ]{0,2}"
    r"(Z|UTC|GMT|[+-]\d{2}(?::?\d{2})?)?$",
    re.I)


def parse_time(value) -> datetime.datetime | None:  # pylint: disable=too-many-return-statements
    """
    A time in any form telemetry commonly carries it, as an aware UTC time; `None` when there is none to read.

    ISO 8601 with `T` or a space, any number of fractional digits, and `Z`, `UTC` or a numeric offset with or
    without its colon; and epoch numbers, as numbers or digit strings, in seconds, milliseconds, microseconds or
    nanoseconds, told apart by magnitude. Anything else is no time, not an error.
    """
    if (value is None or isinstance(value, bool)):
        return None

    if (isinstance(value, datetime.datetime)):
        return value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)

    if (isinstance(value, str)):
        text = value.strip()

        # No time is written in more than a few dozen characters, and a long string of spaces must not be searched.
        if (not text or len(text) > MAX_TIME_TEXT):
            return None

        match = _ISO.match(text)

        if (match is not None):
            (year, month, day, hour, minute, second, fraction, zone) = match.groups()
            micro = int((fraction or "0")[:6].ljust(6, "0"))
            offset = datetime.timezone.utc

            if (zone and zone.upper() not in ("Z", "UTC", "GMT")):
                digits = zone[1:].replace(":", "").ljust(4, "0")
                delta = datetime.timedelta(hours=int(digits[:2]), minutes=int(digits[2:4]))
                offset = datetime.timezone(delta if zone[0] == "+" else -delta)

            try:
                return datetime.datetime(int(year),
                                         int(month),
                                         int(day),
                                         int(hour),
                                         int(minute),
                                         int(second or 0),
                                         micro,
                                         tzinfo=offset).astimezone(datetime.timezone.utc)
            except (ValueError, OverflowError):
                return None

        try:
            value = float(text)
        except ValueError:
            return None

    try:
        usable = isinstance(value, (int, float)) and math.isfinite(value) and 0 < value < 1e20
    except OverflowError:  # an integer too large to be a float
        usable = False

    if (not usable):
        return None

    # Seconds until the year 5138 are below 1e11; each larger unit is a thousand times that.
    for (limit, divisor) in ((1e11, 1), (1e14, 1e3), (1e17, 1e6), (1e20, 1e9)):
        if (value < limit):
            try:
                return datetime.datetime.fromtimestamp(value / divisor, tz=datetime.timezone.utc)
            except (ValueError, OverflowError, OSError):
                return None

    return None


def _mapping(value) -> dict:
    """A section as a mapping: given as one, or as JSON text of one; anything else is an empty section."""
    if (isinstance(value, dict)):
        return dict(value)

    if (isinstance(value, str) and value.lstrip()[:1] == "{"):
        try:
            parsed = json.loads(value)
        except (ValueError, RecursionError):
            return {}

        return parsed if isinstance(parsed, dict) else {}

    return {}


def _sequence(value) -> list:
    if (isinstance(value, (list, tuple))):
        return list(value)

    if (isinstance(value, str) and value.lstrip()[:1] == "["):
        try:
            parsed = json.loads(value)
        except (ValueError, RecursionError):
            return []

        return parsed if isinstance(parsed, list) else []

    return []


def _text(value) -> str:
    """
    Content as text: a string as it is, a list of content blocks as their text joined, any other value as JSON that
    keeps non-ASCII characters as characters, so that a hunt reads what a model read.
    """
    if (value is None):
        return ""

    if (isinstance(value, str)):
        return value

    if (isinstance(value, (list, tuple))):
        parts = []

        for item in value:
            if (isinstance(item, dict) and isinstance(item.get("text"), str)):
                parts.append(item["text"])
            elif (isinstance(item, dict) and isinstance(item.get("content"), (str, list))):
                parts.append(_text(item["content"]))
            elif (isinstance(item, dict) and isinstance(item.get("resource"), dict)):
                parts.append(_text(item["resource"].get("text")))
            elif (isinstance(item, str)):
                parts.append(item)
            elif (item is not None):
                parts.append(_json(item))

        return "\n".join(part for part in parts if part)

    if (isinstance(value, (int, float, bool))):
        return str(value)

    return _json(value)


def _json(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (ValueError, TypeError, RecursionError):
        return ""


def _name(value) -> str:
    return value if isinstance(value, str) else ("" if value is None or isinstance(value, (dict, list)) else str(value))


def normalize(record: dict, ref: str, session: str = "") -> dict:
    """
    One agent event in the shared shape.

    Unknown fields are dropped and a missing or mistyped section is an empty one. `args` may be a mapping, JSON text
    of one, or a list (kept under `argv`); `content` may be text or content blocks. An event without a session id
    is given one from its host, or failing that from `session`, so that events of unrelated machines are never
    hunted as one session.

    Raises
    ------
    TelemetryError
        If the record is not a mapping or its `type` is not one of `EVENT_TYPES`.
    """
    if (not isinstance(record, dict)):
        raise TelemetryError(f"{ref}: an event must be a JSON object")

    kind = record.get("type")

    if (kind not in EVENT_TYPES):
        raise TelemetryError(f"{ref}: event type {str(kind)[:40]!r} is not one of {', '.join(EVENT_TYPES)}")

    args = record.get("args")

    if (isinstance(args, (list, tuple))):
        args = {"argv": list(args)}

    host = _name(record.get("host"))
    session_id = _name(record.get("session_id"))
    inferred = not session_id

    if (inferred):
        session_id = f"(no session id; host {host})" if host else (session or "(no session id)")

    process = _mapping(record.get("process"))

    if ("ancestors" in process):
        process["ancestors"] = [_name(item) for item in _sequence(process["ancestors"])]

    auth = _mapping(record.get("auth"))

    for key in ("scopes", "declared_scopes"):
        if (key in auth):
            auth[key] = [_name(item) for item in _sequence(auth[key]) if _name(item)]

    return {
        "ref": ref,
        "ts": parse_time(record.get("ts")),
        "session_id": session_id,
        "session_inferred": inferred,
        "agent": _name(record.get("agent")),
        "host": host,
        "principal": _name(record.get("principal")),
        "type": kind,
        "tool": _name(record.get("tool")),
        "server": _name(record.get("server")),
        "args": _mapping(args),
        "content": _text(record.get("content")),
        "tools": [tool for tool in _sequence(record.get("tools")) if isinstance(tool, dict)],
        "process": process,
        "file": _mapping(record.get("file")),
        "network": _mapping(record.get("network")),
        "auth": auth,
    }


def _result_text(result: dict) -> str:
    parts = [_text(result.get("content"))]

    if (isinstance(result.get("structuredContent"), (dict, list))):
        parts.append(_json(result["structuredContent"]))

    return "\n".join(part for part in parts if part)


CLIENT_REQUESTS = ("initialize", "ping", "tools/", "prompts/", "resources/", "completion/", "logging/")
"""Methods a client asks of a server. Anything else carrying an id is the server asking the client."""

MAX_PENDING_PER_ID = 32


class _TranscriptReader:
    """
    Turns MCP JSON-RPC messages into events, pairing each response with the request it answers.

    A server may send requests of its own, and may choose their ids, so requests are kept by who asked: a server
    request that reuses a client's id cannot displace the client's request, which is how a hostile server once hid
    its tool list from every hunt. A response is matched to the client's request when its shape fits one.
    """

    def __init__(self):
        self.pending: dict = {}

    def read(self, record: dict, ref: str, session: str) -> dict | None:
        message = record["message"]
        base = {
            "ts": record.get("ts"),
            "session_id": record.get("session_id"),
            "server": record.get("server"),
            "agent": record.get("agent"),
            "host": record.get("host")
        }

        try:
            key = (str(record.get("session_id")), str(record.get("server")), json.dumps(message.get("id")))
        except (TypeError, ValueError):
            key = (str(record.get("session_id")), str(record.get("server")), "?")

        method = message.get("method")

        if (isinstance(method, str)):
            if ("id" in message and method.startswith(CLIENT_REQUESTS)):
                waiting = self.pending.setdefault(key, [])
                waiting.append(message)

                # A client does not leave many requests open under one id; a log that does is not paired at all.
                if (len(waiting) > MAX_PENDING_PER_ID):
                    del waiting[0]

            if (method == "tools/call"):
                params = _mapping(message.get("params"))

                return normalize(
                    {
                        **base, "type": "tool_call", "tool": params.get("name"), "args": params.get("arguments")
                    },
                    ref,
                    session)

            return None

        result = message.get("result")
        waiting = self.pending.get(key) or []

        if (not isinstance(result, dict) or not waiting):
            return None

        # The oldest request this result's shape can answer; a tool list answers `tools/list`, and so on.
        wanted = "tools/list" if isinstance(result.get("tools"), list) else (
            "tools/call" if "content" in result or "structuredContent" in result or "isError" in result else None)
        request = next((item for item in waiting if item.get("method") == wanted), None) if wanted else None

        if (request is None):
            request = waiting[0]

        waiting.remove(request)

        if (not waiting):
            del self.pending[key]

        if (request.get("method") == "tools/list" and isinstance(result.get("tools"), list)):
            return normalize({**base, "type": "tool_manifest", "tools": result.get("tools")}, ref, session)

        if (request.get("method") == "tools/call"):
            return normalize(
                {
                    **base,
                    "type": "tool_result",
                    "tool": _mapping(request.get("params")).get("name"),
                    "content": _result_text(result),
                    "args": {
                        "is_error": bool(result.get("isError"))
                    }
                },
                ref,
                session)

        return None


def _too_deep(text: str) -> bool:
    """Whether a line nests brackets past `MAX_NESTING`; such a line is refused before the parser recurses on it."""
    depth = 0
    quoted = False
    escaped = False

    for character in text:
        if (escaped):
            escaped = False
        elif (character == "\\"):
            escaped = quoted
        elif (character == '"'):
            quoted = not quoted
        elif (not quoted and character in "[{"):
            depth += 1

            if (depth > MAX_NESTING):
                return True
        elif (not quoted and character in "]}"):
            depth -= 1

    return False


def read_lines(path: str, skipped: list | None = None) -> list[dict]:
    """
    The events in one JSON-lines file, of either kind, in file order.

    Parameters
    ----------
    path : str
        The file. A byte-order mark is ignored, and bytes that are not UTF-8 are replaced, not refused.
    skipped : list, optional
        Receives `(ref, reason)` for every line that could not be read as an event. Without it, the first such line
        raises `TelemetryError`, which is the strict reading.
    """
    events = []
    reader = _TranscriptReader()
    name = os.path.basename(path)

    def refuse(ref: str, reason: str):
        if (skipped is None):
            raise TelemetryError(f"{ref}: {reason}")

        skipped.append((ref, reason))

    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        for (number, line) in enumerate(handle, start=1):
            if (not line.strip()):
                continue

            ref = f"{name}:{number}"

            if (len(line) > MAX_LINE):
                refuse(ref, f"line of {len(line)} characters is over the {MAX_LINE} limit")
                continue

            if (line.count("[") + line.count("{") > MAX_NESTING and _too_deep(line)):
                refuse(ref, f"nested more than {MAX_NESTING} deep")
                continue

            try:
                record = json.loads(line)

                if (not isinstance(record, dict)):
                    raise TelemetryError(f"{ref}: a line must be a JSON object")

                message = record.get("message")

                if (isinstance(message, dict) and message.get("jsonrpc") == "2.0"):
                    event = reader.read(record, ref, f"(no session id; file {name})")
                else:
                    event = normalize(record, ref, f"(no session id; file {name})")
            except TelemetryError as error:
                refuse(ref, str(error).split(": ", 1)[-1])
                continue
            except (ValueError, TypeError, AttributeError, OverflowError, RecursionError) as error:
                refuse(ref, f"not a readable event ({type(error).__name__})")
                continue

            if (event is not None):
                events.append(event)

    return events


def order(events: list[dict]) -> list[dict]:
    """
    Events by session and then time, keeping the order they were given in where times tie.

    An event without a time keeps its place: it is ordered as if it happened when the event before it in the same
    session did. Moving such events to the start of their session, as the first version did, put a later action
    before the content that caused it.
    """
    earliest = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    last_seen: dict = {}
    keyed = []

    for (index, event) in enumerate(events):
        session = event["session_id"]
        moment = event["ts"] or last_seen.get(session, earliest)
        last_seen[session] = moment
        keyed.append((session, moment, index, event))

    keyed.sort(key=lambda item: item[:3])

    return [item[3] for item in keyed]


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
        "untimed": len(events) - len(times),
        "without_session_id": sum(1 for event in events if event.get("session_inferred")),
        "first": min(times).isoformat() if times else None,
        "last": max(times).isoformat() if times else None,
    }
