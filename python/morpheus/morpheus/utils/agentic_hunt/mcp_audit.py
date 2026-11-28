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
MCP servers, audited from what a client is configured to launch and from the tools each server offers.

A client configuration is the `{"mcpServers": {name: {"command", "args", "env"} | {"url"}}}` object that Claude Code
(`.mcp.json`), Claude Desktop (`claude_desktop_config.json`), Cursor (`.cursor/mcp.json`) and others read. A tool
manifest is the `tools` list of a `tools/list` response. Nothing here starts a server or fetches a package: the audit
reads only what it is given.
"""

import dataclasses
import hashlib
import json
import re
import shlex
import urllib.parse

from . import commands
from . import text_indicators

KNOWN_PACKAGES = ("@modelcontextprotocol/server-filesystem",
                  "@modelcontextprotocol/server-github",
                  "@modelcontextprotocol/server-gitlab",
                  "@modelcontextprotocol/server-postgres",
                  "@modelcontextprotocol/server-slack",
                  "@modelcontextprotocol/server-memory",
                  "@modelcontextprotocol/server-puppeteer",
                  "@modelcontextprotocol/server-brave-search",
                  "@modelcontextprotocol/server-google-maps",
                  "@modelcontextprotocol/server-everything",
                  "@modelcontextprotocol/server-sequential-thinking",
                  "@playwright/mcp",
                  "mcp-server-fetch",
                  "mcp-server-git",
                  "mcp-server-time",
                  "mcp-server-sqlite")
"""Widely used server packages a typosquat would imitate. Extend it with the servers your organization approves."""
_KNOWN_LOWER = frozenset(name.lower() for name in KNOWN_PACKAGES)

PACKAGE_RUNNERS = {"npx": "npm", "bunx": "npm", "pnpx": "npm", "uvx": "pypi", "pipx": "pypi"}
SECRET_ENV_NAME = re.compile(r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|CREDENTIAL)", re.I)
ENV_REFERENCE = re.compile(r"^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?$|^\$\{env:[A-Za-z_][A-Za-z0-9_]*\}$")
AUTO_APPROVE_KEYS = ("alwaysAllow", "autoApprove", "trust", "alwaysAllowTools")
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "[::1]")
DOWNLOAD_TO_SHELL = re.compile(
    r"(?:curl|wget)\b[^|;&\n]{0,2000}\|\s{0,8}(?:ba|z|da)?sh\b|iex\s{0,8}\(|invoke-expression", re.I)
"""A download piped to a shell, for a launch line the command reader could not follow."""
POISON_THRESHOLD = 3
"""The indicator score at which a tool description is reported as poisoned."""


def edit_distance(left: str, right: str) -> int:
    """Levenshtein distance, the number of single-character edits between two names."""
    previous = list(range(len(right) + 1))

    for (i, a) in enumerate(left, start=1):
        current = [i]

        for (j, b) in enumerate(right, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))

        previous = current

    return previous[-1]


def _package_spec(args: list) -> str | None:
    """The package argument of an `npx`-style launch: the first argument that is not an option."""
    skip_next = False

    for arg in args:
        if (skip_next):
            skip_next = False
            continue

        if (arg in ("-p", "--package", "--from", "--with")):
            skip_next = True
            continue

        if (not arg.startswith("-")):
            return arg

    return None


def _is_pinned(spec: str, ecosystem: str) -> bool:
    if (ecosystem == "pypi"):
        return "==" in spec or spec.startswith(("git+", "/", "./")) or re.search(r"@\d", spec) is not None

    name = spec[1:] if spec.startswith("@") else spec

    if ("@" not in name):
        return spec.startswith(("/", "./", "file:", "git+"))

    version = name.rsplit("@", 1)[1]

    return re.match(r"^v?\d+(?:\.\d+){0,2}(?:[-+][\w.]+)?$", version) is not None


def _package_name(spec: str) -> str:
    if (spec.startswith("@")):
        return "@" + spec[1:].split("@", 1)[0]

    return re.split(r"==|@|\[", spec, maxsplit=1)[0]


def _finding(server: str, check: str, severity: str, detail: str) -> dict:
    return {"hunt_id": "AAH-008", "server": server, "check": check, "severity": severity, "detail": detail}


def audit_server(name: str, entry: dict) -> list[dict]:
    """What is risky about launching one configured server, one finding per check that fails."""
    found = []
    command = str(entry.get("command") or "")
    args = [str(arg) for arg in entry.get("args") or []]
    program = command.replace("\\", "/").rsplit("/", 1)[-1].lower().removesuffix(".exe").removesuffix(".cmd")
    line = " ".join([command, *args])
    # Quoted as the client will run it, so that `bash -c "curl ... | sh"` is read as the shell reads it.
    launched = commands.facts(shlex.join([command, *args]))

    if (program in PACKAGE_RUNNERS):
        spec = _package_spec(args)
        ecosystem = PACKAGE_RUNNERS[program]

        if (spec and not _is_pinned(spec, ecosystem)):
            found.append(
                _finding(name,
                         "unpinned_package",
                         "medium",
                         f"{program} runs {text_indicators.defang(spec)} at whatever version is latest"))

        if (spec):
            package = _package_name(spec)

            for known in (KNOWN_PACKAGES if package.lower() not in _KNOWN_LOWER else ()):
                distance = edit_distance(package.lower(), known.lower())

                if (0 < distance <= 2):
                    found.append(
                        _finding(name,
                                 "typosquat_candidate",
                                 "high",
                                 f"package {text_indicators.defang(package)} is {distance} edit(s) from "
                                 f"{known}"))

    if (program == "docker" and "run" in args):
        images = [arg for arg in args[args.index("run") + 1:] if not arg.startswith("-") and ("/" in arg or ":" in arg)]

        for image in images[:1]:
            if ("@sha256:" not in image and (":" not in image.rsplit("/", 1)[-1] or image.endswith(":latest"))):
                found.append(
                    _finding(name,
                             "unpinned_image",
                             "medium",
                             f"image {text_indicators.defang(image)} is not pinned to a tag or digest"))

    if (any(risk in ("download_and_execute", "decode_and_execute") for (risk, _, _) in launched.risky)
            or DOWNLOAD_TO_SHELL.search(line)):
        found.append(
            _finding(name,
                     "download_to_shell",
                     "high",
                     f"launch pipes a download into a shell: {text_indicators.defang(line)}"))

    url = str(entry.get("url") or entry.get("serverUrl") or "")

    if (url):
        parsed = urllib.parse.urlparse(url)

        if (parsed.scheme == "http" and parsed.hostname not in LOCAL_HOSTS):
            found.append(
                _finding(name,
                         "plaintext_remote",
                         "high",
                         f"remote server {text_indicators.defang(url)} is reached without TLS"))

    for (key, value) in (entry.get("env") or {}).items():
        literal = isinstance(value, str) and value and not ENV_REFERENCE.match(value)

        if (literal and (SECRET_ENV_NAME.search(key) or text_indicators.find_secrets(value))):
            found.append(
                _finding(name,
                         "inline_secret",
                         "high",
                         f"env {key} holds a literal credential ({len(value)} characters, redacted)"))

    for (key, value) in (entry.get("headers") or {}).items():
        if (isinstance(value, str) and key.lower() in ("authorization", "x-api-key") and "${" not in value):
            found.append(_finding(name, "inline_secret", "high", f"header {key} holds a literal credential"))

    if ("server-filesystem" in line):
        roots = [arg for arg in args if arg in ("/", "~", "~/", "C:\\", "C:/", "$HOME", "${HOME}")]

        if (roots):
            found.append(
                _finding(name,
                         "broad_filesystem",
                         "medium",
                         f"filesystem server is given {', '.join(roots)}, the whole disk or home directory"))

    for key in AUTO_APPROVE_KEYS:
        value = entry.get(key)

        if (value is True or (isinstance(value, list) and value)):
            found.append(
                _finding(name,
                         "auto_approve",
                         "low",
                         f"{key} lets the client call {value if value is not True else 'every tool'} "
                         "without asking"))

    return found


SERVER_KEYS = ("mcpServers", "servers", "mcp_servers", "context_servers")
"""The keys clients keep their MCP servers under: Claude Code, Claude Desktop, Cursor and Windsurf use `mcpServers`,
VS Code's `mcp.json` uses `servers`, Codex uses `mcp_servers` and Zed `context_servers`."""


def servers_in(document) -> dict:
    """
    Every MCP server a client configuration defines, by name, whichever shape the client uses.

    Besides the top-level keys in `SERVER_KEYS`, this reads VS Code settings (`mcp.servers`) and the Claude Code
    user configuration, which keeps a project's servers under `projects.<path>.mcpServers`; a server defined for a
    project is named `<path>:<name>`. The first version read only top-level `mcpServers` and so reported a user
    configuration with a dozen project servers as clean.
    """
    found = {}

    if (not isinstance(document, dict)):
        return found

    def take(servers, prefix: str = ""):
        if (isinstance(servers, dict)):
            for (name, entry) in servers.items():
                found.setdefault(f"{prefix}{name}", entry)

    for key in SERVER_KEYS:
        take(document.get(key))

    if (isinstance(document.get("mcp"), dict)):
        take(document["mcp"].get("servers"))

    take(document.get("mcp.servers"))

    if (isinstance(document.get("projects"), dict)):
        for (path, project) in document["projects"].items():
            if (isinstance(project, dict)):
                for key in SERVER_KEYS:
                    take(project.get(key), f"{path}:")

    return found


def audit_config(document: dict) -> list[dict]:
    """Every server in an MCP client configuration, audited. One unreadable entry does not hide the others."""
    found = []

    for (name, entry) in sorted(servers_in(document).items()):
        if (not isinstance(entry, dict)):
            found.append(_finding(str(name), "unreadable_entry", "medium",
                                  f"this server's entry is a {type(entry).__name__}, not a mapping, and was not audited"))
            continue

        try:
            found.extend(audit_server(str(name), entry))
        except (TypeError, ValueError, AttributeError, RecursionError) as error:
            found.append(_finding(str(name), "unreadable_entry", "medium",
                                  f"this server's entry could not be audited ({type(error).__name__})"))

    return found


def _schema_texts(schema, path: str = "") -> list[tuple]:
    """Every description and title inside a JSON Schema, with where it sits."""
    found = []

    if (isinstance(schema, dict)):
        for (key, value) in schema.items():
            where = f"{path}.{key}" if path else key

            if (key in ("description", "title") and isinstance(value, str)):
                found.append((where, value))
            elif (isinstance(value, (dict, list))):
                found.extend(_schema_texts(value, where))
    elif (isinstance(schema, list)):
        for (index, value) in enumerate(schema):
            found.extend(_schema_texts(value, f"{path}[{index}]"))

    return found


def tool_texts(tool: dict) -> list[tuple]:
    """The text a model reads about one tool: its description and every description in its input schema."""
    texts = [("description", str(tool.get("description") or ""))]
    texts.extend(_schema_texts(tool.get("inputSchema") or {}, "inputSchema"))

    return texts


MODEL_DIRECTED = frozenset(["instruction_override", "role_reassignment", "new_instructions", "tool_directive"])
"""Addressing the model is what a tool description is for, so of the indicators that address it only these count
towards poisoning: the ones that try to replace its instructions, and the ones that tell it what to do with other
tools or with data. The second kind is ordinary by itself ("call `resolve` first") and is one kind of evidence."""


@dataclasses.dataclass(frozen=True)
class ToolAssessment:
    """What the text a model reads about one tool adds up to."""

    score: int
    categories: tuple
    indicators: tuple
    places: tuple
    hidden: bool

    @property
    def poisoned(self) -> bool:
        """Two kinds of evidence that the text works against the user, or anything hidden in it."""
        return self.score >= POISON_THRESHOLD and (self.hidden or len(self.categories) >= 2)


def tool_assessment(tool: dict) -> ToolAssessment:
    """
    Grade one tool's descriptions as a hunter would: by what they ask for beyond using the tool. "Call this before
    that" is ordinary and scores nothing; naming a credential, asking for secrecy, naming a way out or hiding text
    each count, and two of them together, or anything hidden, make the tool poisoned.
    """
    indicators = []
    places = []

    for (where, text) in tool_texts(tool):
        found = [indicator for indicator in text_indicators.assess(text).indicators
                 if indicator.category != text_indicators.ADDRESS or indicator.name in MODEL_DIRECTED
                 or indicator.kind in text_indicators.STRONG_KINDS]

        if (found):
            indicators.extend(found)
            places.append(where)

    # A passing mention weighs 1 and agrees with nothing, as in `text_indicators.assess`.
    categories = tuple(sorted({indicator.category for indicator in indicators if indicator.weight >= 2}))
    hidden = any(indicator.kind in text_indicators.STRONG_KINDS for indicator in indicators)

    return ToolAssessment(text_indicators.score(indicators), categories, tuple(indicators), tuple(places), hidden)


def poisoned_tool(tool: dict) -> dict | None:
    """The indicators in one tool's descriptions, when they add up to an instruction that works against the user."""
    found = tool_assessment(tool)

    if (not found.poisoned):
        return None

    return {
        "tool": str(tool.get("name") or ""),
        "score": found.score,
        "places": list(found.places),
        "categories": list(found.categories),
        "indicators": list(found.indicators),
    }


def definition_digest(tool: dict) -> str:
    """A digest of what a model is told about a tool, which changes when the definition does."""
    canonical = json.dumps({
        "description": tool.get("description") or "", "inputSchema": tool.get("inputSchema") or {}
    },
                           sort_keys=True,
                           separators=(",", ":"))

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def baseline_of(manifests: dict) -> dict:
    """`{server: {tool: digest}}` for manifests given as `{server: [tool, ...]}`: what a later audit compares to."""
    return {
        server: {
            str(tool.get("name")): definition_digest(tool)
            for tool in tools if isinstance(tool, dict)
        }
        for (server, tools) in manifests.items()
    }


def changed_definitions(baseline: dict, server: str, tools: list) -> list[dict]:
    """The tools whose definitions differ from the baseline's, and the tools the baseline did not have."""
    known = baseline.get(server)

    if (known is None):
        return []

    changes = []

    for tool in tools:
        name = str(tool.get("name") or "")
        digest = definition_digest(tool)

        if (name not in known):
            changes.append({"server": server, "tool": name, "change": "added", "digest": digest})
        elif (known[name] != digest):
            changes.append({
                "server": server, "tool": name, "change": "redefined", "was": known[name], "digest": digest
            })

    return changes


def cross_references(server: str, tool: dict, others: dict) -> list[str]:
    """The names of other servers' tools that one tool's description mentions: a description steering another."""
    text = " ".join(text for (_, text) in tool_texts(tool))
    named = []

    for (other_server, names) in others.items():
        if (other_server == server):
            continue

        for name in names:
            # A bare word such as `search` would match ordinary prose; a compound or long name would not.
            distinctive = "_" in name or "-" in name or len(name) >= 8

            if (distinctive and re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text)):
                named.append(f"{other_server}.{name}")

    return sorted(named)
