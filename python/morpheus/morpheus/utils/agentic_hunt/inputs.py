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

- `.jsonl` / `.jsonlines` / `.ndjson`, in any letter case: agent events or MCP JSON-RPC transcripts
  (`events.read_lines`).
- `.json` holding MCP servers in any of the shapes clients use (`mcp_audit.servers_in`): an MCP client
  configuration.
- `.json` holding `tools`: a captured `tools/list` result, read as one tool manifest event. The server is the
  document's `server` field, or the file name without its extension.
- A directory: every such file under it.

A file that cannot be read is skipped and reported, with every other file still hunted; `strict` makes it an
error instead.
"""

import json
import os

from . import events
from . import mcp_audit

LINE_SUFFIXES = (".jsonl", ".jsonlines", ".ndjson")
MAX_BYTES = 64 * 1024 * 1024
"""The largest file read. A hunt over more belongs in a SIEM; the queries under `queries/` are for that."""

MAX_FILES = 5000


class InputError(ValueError):
    """A file a hunt cannot read."""


def _manifest_event(document: dict, path: str) -> dict:
    name = os.path.basename(path)
    server = str(document.get("server") or os.path.splitext(name)[0])

    return events.normalize({"type": "tool_manifest", "server": server, "session_id": document.get("session_id"),
                             "ts": document.get("ts"), "tools": document.get("tools")}, name,
                            f"(no session id; file {name})")


def expand(paths: list[str]) -> list[str]:
    """The files named, with each directory replaced by the readable files under it, in a stable order."""
    found = []

    for path in paths:
        if (os.path.isdir(path)):
            for (root, directories, names) in os.walk(path):
                directories.sort()

                for name in sorted(names):
                    if (name.lower().endswith((*LINE_SUFFIXES, ".json"))):
                        found.append(os.path.join(root, name))

                if (len(found) > MAX_FILES):
                    raise InputError(f"{path} holds more than {MAX_FILES} files; name a narrower directory")
        else:
            found.append(path)

    return found


def load(paths: list[str], strict: bool = False) -> tuple:
    """
    `(events, configs, skipped)`: the ordered events of every events file and manifest, each configuration as
    `(name, document)`, and `(ref, reason)` for every line or file that could not be read.

    Raises
    ------
    InputError
        Under `strict`, for the first thing that could not be read. Otherwise only when nothing could be.
    """
    found = []
    configs = []
    skipped = []
    files = expand(paths)

    def refuse(ref: str, reason: str):
        if (strict):
            raise InputError(f"{ref}: {reason}")

        skipped.append((ref, reason))

    for path in files:
        name = os.path.basename(path)

        try:
            if (os.path.getsize(path) > MAX_BYTES):
                refuse(name, f"over {MAX_BYTES // (1024 * 1024)} MiB")
                continue

            if (path.lower().endswith(LINE_SUFFIXES)):
                problems = []
                found.extend(events.read_lines(path, problems))

                for (ref, reason) in problems:
                    refuse(ref, reason)

                continue

            with open(path, encoding="utf-8-sig", errors="replace") as handle:
                document = json.load(handle)
        except InputError:
            raise
        except OSError as error:
            refuse(name, f"cannot be opened ({error.strerror or type(error).__name__})")
            continue
        except (ValueError, RecursionError):
            refuse(name, "neither JSON lines nor a JSON document")
            continue

        if (mcp_audit.servers_in(document)):
            configs.append((name, document))
        elif (isinstance(document, dict) and isinstance(document.get("tools"), list)):
            found.append(_manifest_event(document, path))
        else:
            refuse(name, "JSON, but neither an MCP configuration nor a tool manifest")

    if (files and not found and not configs and skipped and len(skipped) >= len(files)):
        raise InputError(f"nothing could be read: {skipped[0][0]}: {skipped[0][1]}")

    return (events.order(found), configs, skipped)


def load_baseline(path: str | None) -> dict:
    """A tool baseline, `{server: {tool: digest}}`, as `hunt.py baseline` writes it; empty without a path."""
    if (not path):
        return {}

    try:
        with open(path, encoding="utf-8-sig") as handle:
            document = json.load(handle)
    except (ValueError, RecursionError) as error:
        raise InputError(f"{os.path.basename(path)} is not a tool baseline") from error

    if (not isinstance(document, dict) or not all(isinstance(tools, dict) for tools in document.values())):
        raise InputError(f"{os.path.basename(path)} is not a tool baseline")

    return document
