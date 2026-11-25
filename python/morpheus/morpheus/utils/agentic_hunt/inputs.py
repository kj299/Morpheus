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
The files a hunt is pointed at, sorted by what they are.

- `.jsonl` / `.jsonlines` / `.ndjson`: agent events or MCP JSON-RPC transcripts (`events.read_lines`).
- `.json` holding `mcpServers`: an MCP client configuration (`mcp_audit.audit_config`).
- `.json` holding `tools`: a captured `tools/list` result, read as one tool manifest event. The server is the
  document's `server` field, or the file name without its extension.
"""

import json
import os

from . import events

LINE_SUFFIXES = (".jsonl", ".jsonlines", ".ndjson")
MAX_BYTES = 64 * 1024 * 1024
"""The largest file read. A hunt over more belongs in a SIEM; the queries under `queries/` are for that."""


class InputError(ValueError):
    """A file a hunt cannot read."""


def _manifest_event(document: dict, path: str) -> dict:
    server = str(document.get("server") or os.path.splitext(os.path.basename(path))[0])

    return events.normalize(
        {
            "type": "tool_manifest",
            "server": server,
            "session_id": document.get("session_id"),
            "ts": document.get("ts"),
            "tools": document.get("tools")
        },
        os.path.basename(path))


def load(paths: list[str]) -> tuple:
    """`(events, configs)`: the events of every events file and manifest, ordered, and each configuration."""
    found = []
    configs = []

    for path in paths:
        if (os.path.getsize(path) > MAX_BYTES):
            raise InputError(f"{path} is over {MAX_BYTES // (1024 * 1024)} MiB")

        if (path.endswith(LINE_SUFFIXES)):
            found.extend(events.read_lines(path))
            continue

        try:
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise InputError(f"{path} is neither JSON lines nor a JSON document") from error

        if (events.is_mcp_config(document)):
            configs.append((os.path.basename(path), document))
        elif (isinstance(document, dict) and isinstance(document.get("tools"), list)):
            found.append(_manifest_event(document, path))
        else:
            raise InputError(f"{path} is JSON but neither an MCP configuration nor a tool manifest")

    return (events.order(found), configs)


def load_baseline(path: str | None) -> dict:
    """A tool baseline, `{server: {tool: digest}}`, as `hunt.py baseline` writes it; empty without a path."""
    if (not path):
        return {}

    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)

    if (not isinstance(document, dict) or not all(isinstance(tools, dict) for tools in document.values())):
        raise InputError(f"{path} is not a tool baseline")

    return document
