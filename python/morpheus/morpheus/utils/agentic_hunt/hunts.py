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
The hunts the catalog describes, run over events from `events` and configurations for `mcp_audit`.

Each hunt is one function from the events (ordered by session and time) and a `HuntContext` to findings. Three
things hold for every finding, whichever hunt made it:

- **It is safe to show.** `Finding.to_dict` is the one way out, and it passes every string of a finding, its
  summary, session, evidence and details alike, through `text_indicators.clean`: credentials redacted, invisible
  and control characters shown, links broken. What a finding quotes is still evidence from a possibly hostile
  source: data to analyse, never an instruction to follow.
- **It is about something that happened.** Commands are judged by `commands.facts`, which reads a command line as a
  shell does, so text an agent only wrote, searched for or quoted is not reported as something it ran.
- **Its severity says how sure the hunt is.** A sequence hunt reports at the top only when the events are tied by
  more than time: the same command reads and sends, the action reaches what the injected text named, the payload
  carries a credential.
"""

import collections
import dataclasses
import datetime
import itertools
import re
import urllib.parse

from . import catalog
from . import commands
from . import mcp_audit
from . import text_indicators

MODEL_API_HOSTS = ("api.anthropic.com", "api.openai.com", "generativelanguage.googleapis.com", "api.githubcopilot.com",
                   "aiplatform.googleapis.com", "api.mistral.ai", "api.cohere.com", "openrouter.ai", "api.x.ai",
                   "bedrock-runtime.", "bedrock.", "statsig.anthropic.com", "api2.cursor.sh")
"""Where an agent sends its context on every turn. Egress there is the agent working, not a hunt's egress. An entry
that starts with a dot matches every host under it; one that ends with a dot, every host that starts with it."""

DEFAULT_ALLOWLIST = ("github.com", "api.github.com", "raw.githubusercontent.com", "registry.npmjs.org", "pypi.org",
                     "files.pythonhosted.org")
"""Destinations that are ordinary for a coding agent. Sending there still counts, at a lower severity. An entry that
starts with a dot allows every host under it."""

AGENT_PROCESS = re.compile(
    r"(?:^|[/\\\s])(?:claude|cursor|cursor-agent|codex|copilot|windsurf|aider|gemini|cline|goose|amp|opencode)"
    r"(?:\.exe|\.js|\.mjs)?(?=$|\s)|@anthropic-ai[/\\]claude-code|@openai[/\\]codex|@google[/\\]gemini-cli|"
    r"[/\\]\.cursor-server[/\\]|claude-code[/\\]cli", re.I)
"""An agent's own process, by image or command line. An editor is not one: `code` hosts people as well as agents."""

PERMISSION_WIDENING = re.compile(
    r"\"Bash\(\*\)\"|\"allow\"\s{0,3}:\s{0,3}\[\s{0,3}\"\*\"|enableAllProjectMcpServers\"\s{0,3}:\s{0,3}true|"
    r"\"autoApprove\"|\"alwaysAllow\"|bypassPermissions|dangerously-skip-permissions|"
    r"chat\.tools\.autoApprove\"\s{0,3}:\s{0,3}true|approval_policy\s{0,3}=\s{0,3}\"never\"|"
    r"sandbox_mode\s{0,3}=\s{0,3}\"danger-full-access\"|\"mcpServers\"\s{0,3}:|\"hooks\"\s{0,3}:",
    re.I)
"""Edits that turn an agent's approvals off, add servers it will load, or add commands it will run by itself."""

COMMAND_KEYS = ("command", "cmd", "script", "shell", "commandLine", "command_line", "cmdline", "bash_command",
                "shell_command")
PATH_KEYS = ("path", "file", "file_path", "filepath", "filename", "target", "destination", "source", "uri",
             "notebook_path", "target_file", "absolute_path", "paths", "files")
CONTENT_KEYS = ("content", "contents", "text", "new_string", "old_string", "body", "data", "new_str", "old_str",
                "diff", "patch", "edits", "code_edit", "file_text")
URL_KEYS = ("url", "uri", "endpoint", "href", "link", "webhook", "webhook_url", "target_url", "base_url", "address",
            "urls")
NOT_SENT_KEYS = frozenset([*CONTENT_KEYS, "pattern", "query", "prompt", "description", "regex", "glob", "old_string"])
"""Arguments whose text stays on the machine or goes only to the model: a URL in a file being written is not egress."""

EXECUTE = "execute"
WRITE = "write"
READ = "read"
SEND = "send"
FETCH = "fetch"

KNOWN_TOOLS = {
    **dict.fromkeys(("bash", "shell", "exec_command", "local_shell", "run_terminal_cmd", "run_shell_command",
                     "execute_command", "runinterminal", "run_in_terminal", "terminal", "run_command"), {EXECUTE}),
    **dict.fromkeys(("write", "edit", "multiedit", "notebookedit", "apply_patch", "edit_file", "search_replace",
                     "write_file", "replace", "write_to_file", "replace_in_file", "create_file",
                     "str_replace_editor", "str_replace_based_edit_tool", "insert_edit_into_file", "editfiles"),
                    {WRITE}),
    **dict.fromkeys(("read", "read_file", "view", "open_file", "read_many_files", "readfile", "cat"), {READ}),
    **dict.fromkeys(("webfetch", "web_fetch", "fetch", "browser_action", "browser_navigate", "puppeteer_navigate",
                     "navigate", "goto", "download_file", "http_get", "fetch_url"), {FETCH}),
    **dict.fromkeys(("todowrite", "todoread", "task", "agent", "websearch", "web_search", "google_web_search",
                     "exitplanmode", "askuserquestion", "bashoutput", "killshell", "killbash", "skill", "toolsearch",
                     "grep", "glob", "ls", "list_dir", "search_file_content", "codebase_search", "update_plan",
                     "memory", "sequentialthinking"), set()),
}
"""What the tools of the common coding agents do, by name. Names not listed are judged by the words in them."""

CAPABILITY_WORDS = {
    EXECUTE: {"bash", "shell", "exec", "execute", "terminal", "powershell", "cmd", "subprocess", "spawn", "eval"},
    WRITE: {"write", "edit", "patch", "apply", "delete", "remove", "move", "rename", "append", "replace", "save"},
    READ: {"read", "view", "cat", "load"},
    SEND: {"post", "upload", "send", "email", "mail", "webhook", "publish", "share", "message", "gist", "paste",
           "comment", "tweet", "notify", "push", "submit"},
    FETCH: {"fetch", "http", "request", "browse", "navigate", "goto", "download", "crawl", "scrape", "curl"},
}


@dataclasses.dataclass
class HuntContext:
    """What the hunts are tuned with. Every value has a default a first hunt can run on."""

    window_seconds: int = 600
    injection_threshold: int = text_indicators.LIKELY_SCORE
    egress_allowlist: tuple = DEFAULT_ALLOWLIST
    model_api_hosts: tuple = MODEL_API_HOSTS
    tool_baseline: dict = dataclasses.field(default_factory=dict)
    auth_baseline: dict = dataclasses.field(default_factory=dict)
    calls_per_minute: int = 30
    distinct_files: int = 50
    untimed_gap: int = 200
    """How many events apart two events may be and still be one sequence, when either has no time to go by."""


def _clean(value, limit: int = 400, depth: int = 0):
    """A finding's value made safe to show: strings cleaned and cut, containers cleaned through, and capped."""
    if (isinstance(value, str)):
        return text_indicators.clean(value, limit)

    if (isinstance(value, dict) and depth < 4):
        return {text_indicators.clean(str(key), 80): _clean(item, limit, depth + 1)
                for (key, item) in itertools.islice(value.items(), 50)}

    if (isinstance(value, (list, tuple, set, frozenset)) and depth < 4):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else list(value)

        return [_clean(item, limit, depth + 1) for item in items[:50]]

    if (value is None or isinstance(value, (bool, int, float))):
        return value

    return text_indicators.clean(str(value), limit)


@dataclasses.dataclass
class Finding:
    hunt_id: str
    severity: str
    summary: str
    session_id: str = ""
    ts: str | None = None
    evidence: list = dataclasses.field(default_factory=list)
    refs: list = dataclasses.field(default_factory=list)
    details: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        """The finding as it leaves the engine: every string cleaned, whatever telemetry field it came from."""
        hunt = catalog.BY_ID[self.hunt_id]

        return {
            "hunt_id": self.hunt_id,
            "severity": self.severity,
            "summary": _clean(self.summary, 300),
            "session_id": _clean(self.session_id, 200),
            "ts": self.ts,
            "evidence": _clean(self.evidence[:12]),
            "refs": _clean(self.refs[:40], 200),
            "details": _clean(self.details, 300),
            "title": hunt.title,
            "atlas": list(hunt.atlas),
            "attack": list(hunt.attack),
            "owasp": list(hunt.owasp),
        }


def _iso(value: datetime.datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def sessions(events: list[dict]):
    """Events grouped by session, each group in time order, as `events.order` leaves them."""
    for (session, group) in itertools.groupby(events, key=lambda event: event["session_id"]):
        yield (session, list(group))


def _later(group: list, index: int, ctx: HuntContext):
    """The events after `group[index]` that can belong to one sequence with it: inside the window, by time when
    both have one and by position when either does not."""
    first = group[index]

    for (distance, later) in enumerate(group[index + 1:], start=1):
        if (first["ts"] is None or later["ts"] is None):
            if (distance > ctx.untimed_gap):
                return
        elif ((later["ts"] - first["ts"]).total_seconds() > ctx.window_seconds):
            return

        yield later


def _strings(value, depth: int = 0) -> list[str]:
    if (isinstance(value, str)):
        return [value]

    if (depth > 6):
        return []

    if (isinstance(value, dict)):
        return [text for item in value.values() for text in _strings(item, depth + 1)]

    if (isinstance(value, (list, tuple))):
        return [text for item in value for text in _strings(item, depth + 1)]

    return []


def _arg(event: dict, keys: tuple) -> list[str]:
    found = []

    for key in keys:
        value = event["args"].get(key)

        if (isinstance(value, (str, int, float)) and not isinstance(value, bool)):
            found.append(str(value))
        elif (isinstance(value, (list, tuple))):
            found.extend(str(item) for item in value if isinstance(item, (str, int, float)))

    return found


def _tokens(name: str) -> set:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)

    return {token.lower() for token in re.split(r"[^A-Za-z0-9]+", spaced) if token}


def tool_capabilities(name: str) -> set:
    """What a tool does, from its name: a known tool of a common agent, or the words in the name."""
    short = name.rsplit("__", 1)[-1].rsplit(".", 1)[-1].lower()

    if (short in KNOWN_TOOLS):
        return set(KNOWN_TOOLS[short])

    tokens = _tokens(short)

    if ("todo" in tokens or "plan" in tokens):
        return set()

    return {capability for (capability, words) in CAPABILITY_WORDS.items() if tokens & words}


def command_of(event: dict) -> str:
    """The command line an event runs: a process's, or the command argument of a tool call."""
    if (event["type"] == "process"):
        return str(event["process"].get("cmdline") or "")

    if (event["type"] != "tool_call"):
        return ""

    for key in COMMAND_KEYS:
        value = event["args"].get(key)

        if (isinstance(value, str) and value.strip()):
            return value

        if (isinstance(value, (list, tuple)) and value and all(isinstance(item, str) for item in value)):
            # `["bash", "-lc", "the command"]`, as some agents record a command.
            if (len(value) >= 3 and value[1].startswith("-") and value[1].endswith("c")):
                return value[2]

            return " ".join(value)

    if (EXECUTE in tool_capabilities(event["tool"])):
        value = event["args"].get("input") or event["args"].get("argv")

        if (isinstance(value, str)):
            return value

        if (isinstance(value, (list, tuple))):
            return " ".join(str(item) for item in value)

    return ""


def paths_of(event: dict) -> list[str]:
    if (event["type"] == "file"):
        return [str(event["file"].get("path") or "")]

    if (event["type"] == "tool_call"):
        return _arg(event, PATH_KEYS)

    return []


def from_agent(event: dict) -> bool:
    """Whether a process event belongs to an agent: attributed to one, or started under one's process tree."""
    process = event["process"]

    if (event["agent"] or process.get("agent_lineage") is True):
        return True

    lineage = [process.get("parent_image"), process.get("parent_cmdline"), *(process.get("ancestors") or [])]

    return any(AGENT_PROCESS.search(str(item)) for item in lineage if item)


def _allowed(host: str, allowlist: tuple) -> bool:
    return any(host == entry or (entry.startswith(".") and host.endswith(entry)) for entry in allowlist)


def _ignored(host: str, ctx: HuntContext) -> bool:
    return (not host or host in commands.LOOPBACK or host in ctx.model_api_hosts
            or any((entry.startswith(".") and host.endswith(entry)) or (entry.endswith(".") and host.startswith(entry))
                   for entry in ctx.model_api_hosts))


def egress_of(event: dict, ctx: HuntContext) -> list:
    """
    The hosts an event reaches, as `commands.Egress`, leaving out the local machine and the model provider.

    A tool call's addresses are taken from the arguments that are addresses, and from any argument of a tool that
    sends or fetches; never from the content of a file being written or a pattern being searched for.
    """
    found = []

    if (event["type"] == "network"):
        host = str(event["network"].get("dest_host") or event["network"].get("dest_ip") or "").lower()
        found.append(commands.Egress(host, False, f"connection to {host}"))
    elif (event["type"] == "tool_call"):
        capabilities = tool_capabilities(event["tool"])
        sends = SEND in capabilities

        for (key, value) in event["args"].items():
            if (key in NOT_SENT_KEYS or not (key in URL_KEYS or capabilities & {SEND, FETCH})):
                continue

            for text in _strings(value):
                for url in commands.URL.findall(text)[:20] if key not in URL_KEYS else [text]:
                    host = commands._host_of(url)  # pylint: disable=protected-access

                    if (host):
                        # pylint: disable-next=protected-access
                        carries = sends or commands._data_in_address(url) or bool(_secrets_in([url]))
                        found.append(commands.Egress(host, carries, f"{event['tool']} {url}"))

        if (sends and isinstance(event["args"].get("to"), str) and "@" in event["args"]["to"]):
            found.append(commands.Egress(event["args"]["to"].rsplit("@", 1)[-1].lower(), True,
                                         f"{event['tool']} to {event['args']['to']}"))

    command = command_of(event)

    if (command and (event["type"] != "process" or from_agent(event))):
        found.extend(commands.facts(command).egress)

    return list(dict.fromkeys(item for item in found if not _ignored(item.host, ctx)))


ENVIRONMENT = "(environment)"
"""`env` with nothing after it prints every variable. It starts no sequence: agents print the environment to debug,
and a later request would make each one a finding. Carried to the network in the same line, it still counts."""


def secret_reads(event: dict) -> list[str]:
    """The credential files an event reads: a file read, a reading tool's path, or a command that reads one."""
    if (event["type"] == "file"):
        if (str(event["file"].get("op") or "read") not in ("read", "open", "copy")):
            return []

        return commands._secret_operands(paths_of(event))  # pylint: disable=protected-access

    if (event["type"] == "tool_call"):
        capabilities = tool_capabilities(event["tool"])
        found = []

        if (capabilities & {READ, SEND} and WRITE not in capabilities):
            found.extend(commands._secret_operands(paths_of(event)))  # pylint: disable=protected-access

        found.extend(what for (what, _) in commands.facts(command_of(event)).secret_reads if what != ENVIRONMENT)

        return sorted(set(found))

    if (event["type"] == "process" and from_agent(event)):
        return sorted({what for (what, _) in commands.facts(command_of(event)).secret_reads if what != ENVIRONMENT})

    return []


def _tail(path: str) -> str:
    """A credential path without the home it sits in, so `~/.aws/credentials` and `/home/bob/.aws/credentials` agree."""
    normal = path.replace("\\", "/")
    marker = normal.rfind("/.")

    return normal[marker + 1:] if marker != -1 else normal.rsplit("/", 1)[-1]


def _assessment(event: dict):
    return text_indicators.assess(event["content"])


def _injected(event: dict, ctx: HuntContext) -> bool:
    """Whether an event's content is likely an injection: the score, and two kinds of evidence or something hidden."""
    if (event["type"] not in ("tool_result", "document")):
        return False

    found = _assessment(event)

    return found.likely and found.score >= ctx.injection_threshold


def _indicator_lines(indicators, limit: int = 8) -> list[str]:
    """One line per kind and name of indicator, the heaviest first, so filler cannot crowd out the directive."""
    seen = {}

    for indicator in indicators:
        seen.setdefault((indicator.kind, indicator.name), indicator)

    ranked = sorted(seen.values(), key=lambda indicator: (-indicator.weight, indicator.offset))

    return [f"{indicator.kind}/{indicator.name}" + (f" x{indicator.count}" if indicator.count > 1 else "") +
            f": {indicator.evidence}" for indicator in ranked[:limit]]


def _source(event: dict) -> str:
    if (event["type"] == "tool_result"):
        return f"result of {event['server'] + '.' if event['server'] else ''}{event['tool'] or 'a tool'}"

    return "a retrieved document"


def hunt_injected_content(events: list[dict], ctx: HuntContext) -> list[Finding]:
    found = []

    for event in events:
        if (event["type"] not in ("tool_result", "document")):
            continue

        result = _assessment(event)

        if (result.level not in ("suspicious", "likely") or result.score < ctx.injection_threshold):
            continue

        severity = "low" if not result.likely else ("high" if result.strong else "medium")
        how = "addresses the model" if result.likely else "carries one kind of injection indicator"
        found.append(
            Finding("AAH-001", severity, f"{_source(event)} {how} (score {result.score})", event["session_id"],
                    _iso(event["ts"]), _indicator_lines(result.indicators), [event["ref"]],
                    {"score": result.score, "level": result.level, "categories": list(result.categories),
                     "partial_scan": result.partial}))

    return found


def _named_in(text: str) -> dict:
    """What a piece of injected text names that an action could then touch: hosts, credential paths and tools."""
    hosts = {commands._host_of(url) for url in commands.URL.findall(text)}  # pylint: disable=protected-access
    hosts |= {match.group(0).lower() for match in text_indicators.COLLECTION_HOSTS.finditer(text)}
    paths = {_tail(match.group(0)) for match in commands.SECRET_PATH.finditer(text)}
    tools = {match.group(1).lower() for match in re.finditer(
        r"(?i)\b(?:call|invoke|use|run|execute)\s{1,3}(?:the\s{1,3})?[`'\"]?([a-z][a-z0-9_.-]{2,40})[`'\"]?\s{1,3}tool",
        text)}
    tools |= {match.group(1).lower() for match in re.finditer(r"`([A-Za-z][A-Za-z0-9_.-]{2,40})`", text)}

    return {"hosts": hosts - {""}, "paths": paths, "tools": tools}


def _risk_of(event: dict) -> list[str]:
    """Why an event would be reported by another hunt: the names of the risky things its command does."""
    if (event["type"] == "process" and not from_agent(event)):
        return []

    facts = commands.facts(command_of(event))
    reasons = [name for (name, _, _) in facts.risky]
    reasons += [name for (name, severity, _) in facts.destructive if severity == "high"]
    reasons += ["credential sent"] if facts.secret_flow else []
    reasons += ["agent configuration written"] if facts.config_writes else []
    reasons += [flag for (flag, _) in facts.guardrail_flags]

    return list(dict.fromkeys(reasons))


def _excerpt(event: dict) -> str:
    """One event in a few words, for an evidence line."""
    command = command_of(event)

    if (command):
        return text_indicators.defang(command, 200)

    if (event["type"] == "file"):
        return f"file {event['file'].get('op') or 'access'}: {text_indicators.defang(', '.join(paths_of(event)), 160)}"

    if (event["type"] == "network"):
        network = event["network"]

        return "connection to " + text_indicators.defang(str(network.get("dest_host") or network.get("dest_ip")), 120)

    described = ", ".join(f"{key}={text_indicators.defang(str(value), 80)}" for (key, value) in
                          itertools.islice(event["args"].items(), 3) if key not in CONTENT_KEYS)

    return f"{event['tool'] or event['type']}({described})"


def hunt_injection_then_action(events: list[dict], ctx: HuntContext) -> list[Finding]:
    """
    An action tied to injected content by more than time: it reaches a host, reads a credential or calls a tool the
    text named, or it is itself something another hunt reports. A coding agent runs a command after nearly every
    tool result, so an action that merely follows is not reported.
    """
    found = []

    for (session, group) in sessions(events):
        for (index, event) in enumerate(group):
            if (not _injected(event, ctx)):
                continue

            named = _named_in(text_indicators.matching_view(event["content"]))
            linked = []
            worst = False

            for later in _later(group, index, ctx):
                if (later["type"] not in ("tool_call", "process", "file", "network")):
                    continue

                reasons = []
                hosts = {item.host for item in egress_of(later, ctx)}
                read = {_tail(path) for path in secret_reads(later)}

                if (hosts & named["hosts"]):
                    reasons.append(f"reaches {', '.join(sorted(hosts & named['hosts']))}, which the text names")
                    worst = True

                if (read & named["paths"]):
                    reasons.append(f"reads {', '.join(sorted(read & named['paths']))}, which the text names")
                    worst = True

                if (later["type"] == "tool_call" and later["tool"].rsplit("__", 1)[-1].lower() in named["tools"]):
                    reasons.append(f"calls {later['tool']}, which the text names")

                risks = _risk_of(later)

                if (risks):
                    reasons.append(f"does what another hunt reports ({', '.join(risks)})")
                    worst = worst or "credential sent" in risks or "reverse_shell" in risks

                if (reasons):
                    linked.append((later, reasons))

            if (linked):
                found.append(
                    Finding("AAH-002", "critical" if worst else "high",
                            f"{len(linked)} action(s) after injected content in {_source(event)} do what it asked "
                            "or are risky in themselves", session, _iso(event["ts"]),
                            [f"{_excerpt(later)}: {'; '.join(reasons)}" for (later, reasons) in linked[:6]],
                            [event["ref"], *(later["ref"] for (later, _) in linked)],
                            {"linked_actions": len(linked), "named_hosts": sorted(named["hosts"]),
                             "named_paths": sorted(named["paths"])}))

    return found


def _carried(event: dict) -> list:
    """Credentials an event visibly carries out: in the arguments of a call, as given and URL-decoded."""
    if (event["type"] != "tool_call"):
        return []

    found = []

    for text in _strings(event["args"]):
        found.extend(text_indicators.find_secrets(text) or text_indicators.find_secrets(urllib.parse.unquote(text)))

    # A published example value counts here: what matters after a credential read is that a key-shaped value went
    # out, and AAH-012 is where the value's own standing is weighed.
    return found


def _grade(item, event: dict, ctx: HuntContext) -> str | None:
    """How much a contact after a credential read says, from nothing (a page fetched) to a payload that carries it."""
    listed = _allowed(item.host, ctx.egress_allowlist)
    collects = text_indicators.COLLECTION_HOSTS.search(item.host) is not None

    if (item.sends and _carried(event)):
        return "critical"

    if (item.sends):
        return "medium" if listed else "high"

    if (collects):
        return "high"

    if (event["type"] == "network" and not listed):
        return "medium"

    return None


def hunt_secret_then_egress(events: list[dict], ctx: HuntContext) -> list[Finding]:
    """
    A credential leaving: read and sent by one command, which is certain; or read, and then data sent somewhere
    within the window, which is graded by what was sent and where. A page fetched after a credential was read is
    not reported: it is what ordinary work looks like.
    """
    found = []
    order = {"medium": 0, "high": 1, "critical": 2}

    for (session, group) in sessions(events):
        flows = {}

        # One command that reads a credential and carries it out. A tool call and the process it started are one.
        for event in group:
            if (event["type"] == "process" and not from_agent(event)):
                continue

            command = command_of(event)
            facts = commands.facts(command) if command else None
            uploads = []

            if (event["type"] == "tool_call" and SEND in tool_capabilities(event["tool"])):
                paths = commands._secret_operands(paths_of(event))  # pylint: disable=protected-access
                uploads = [(path, item.host, _excerpt(event)) for path in paths for item in egress_of(event, ctx)]

            for (what, host, evidence) in [*(facts.secret_flow if facts else ()), *uploads]:
                if (_ignored(host, ctx)):
                    continue

                key = (command or event["ref"], host)

                if (key in flows):
                    flows[key].refs.append(event["ref"])
                    continue

                flows[key] = Finding("AAH-003", "critical", f"a credential is read and sent to {host} in one command",
                                     session, _iso(event["ts"]),
                                     [f"reads {what}", text_indicators.defang(evidence, 240)], [event["ref"]],
                                     {"secret_paths": [what], "hosts": [host], "unlisted_hosts":
                                      [] if _allowed(host, ctx.egress_allowlist) else [host], "same_command": True})

        found.extend(flows.values())
        flowed = {ref for finding in flows.values() for ref in finding.refs}
        reported = {}

        for (index, event) in enumerate(group):
            read = secret_reads(event)

            if (not read or event["ref"] in flowed):
                continue

            graded = {}
            refs = []
            evidence = []

            for later in _later(group, index, ctx):
                for item in egress_of(later, ctx):
                    grade = _grade(item, later, ctx)

                    if (grade is None):
                        continue

                    if (order[grade] > order.get(graded.get(item.host), -1)):
                        graded[item.host] = grade

                    if (later["ref"] not in refs):
                        refs.append(later["ref"])
                        evidence.append(f"{item.host}: {text_indicators.defang(item.evidence, 160)}")

            if (not graded):
                continue

            # The same file read twice in a moment (the tool call, then the file event it caused) is one read.
            key = (tuple(sorted(_tail(path) for path in read)), tuple(refs))

            if (key in reported):
                reported[key].refs.insert(1, event["ref"])
                continue

            severity = max(graded.values(), key=order.get)
            hosts = sorted(graded)
            finding = Finding("AAH-003", severity, f"credential file(s) read, then data sent to {', '.join(hosts)}",
                              session, _iso(event["ts"]), [*read, *evidence[:6]], [event["ref"], *refs],
                              {"secret_paths": read, "hosts": hosts, "grades": graded, "unlisted_hosts":
                               [host for host in hosts if not _allowed(host, ctx.egress_allowlist)],
                               "same_command": False})
            reported[key] = finding
            found.append(finding)

    return found


def _commands_run(events: list[dict]):
    """Each distinct command an agent ran in a session, once: `(event, facts, refs)`. A tool call and the process
    event for the same command line are one command."""
    for (_, group) in sessions(events):
        seen = {}

        for event in group:
            if (event["type"] not in ("tool_call", "process") or (event["type"] == "process"
                                                                  and not from_agent(event))):
                continue

            command = command_of(event)

            if (not command):
                continue

            if (command in seen):
                seen[command][2].append(event["ref"])
            else:
                seen[command] = (event, commands.facts(command), [event["ref"]])

        yield from seen.values()


RISK_SEVERITY = {"reverse_shell": "critical", "service_enable": "low", "shell_profile_change": "low"}
"""What each risky pattern is worth by itself; `high` where not named. Enabling a unit that exists and adding a
`PATH` entry to a shell profile are worth a line in a timeline and not an alert."""


def _worst(levels) -> str:
    return max(levels, key=catalog.SEVERITIES.index)


def _written_persistence(event: dict) -> list:
    """`(name, technique)` where a write made with a tool, not a command, lands on a place that runs again."""
    if (event["type"] == "file"):
        if (str(event["file"].get("op") or "") not in ("write", "create", "modify", "append")):
            return []
    elif (event["type"] != "tool_call" or WRITE not in tool_capabilities(event["tool"])):
        return []

    found = []
    content = _written_content(event)

    for path in paths_of(event):
        if (path.replace("\\", "/").rsplit("/", 1)[-1] in commands.PROFILE_FILES):
            routine = bool(content.strip()) and all(commands.PROFILE_ROUTINE.match(line)
                                                    for line in content.splitlines() if line.strip())
            found.append(("shell_profile_change" if routine else "shell_profile_persistence", "T1546.004"))

        found.extend((name, technique) for (name, technique, pattern) in commands.PERSISTENCE_PATHS
                     if pattern.search(path))

    return found


def hunt_risky_commands(events: list[dict], ctx: HuntContext) -> list[Finding]:
    del ctx
    found = []

    def report(event: dict, risky: list, refs: list, uncertain: bool):
        names = list(dict.fromkeys(name for (name, _, _) in risky))
        found.append(
            Finding("AAH-004", _worst(RISK_SEVERITY.get(name, "high") for name in names),
                    f"agent ran {', '.join(names)}", event["session_id"], _iso(event["ts"]),
                    list(dict.fromkeys(text_indicators.defang(evidence, 240) for (_, _, evidence) in risky)), refs,
                    {"patterns": names, "techniques": sorted({technique for (_, technique, _) in risky}),
                     "uncertain_parse": uncertain}))

    for (event, facts, refs) in _commands_run(events):
        if (facts.risky):
            report(event, list(facts.risky), refs, facts.uncertain)

    for event in events:
        if (event["type"] == "file" and not from_agent(event)):
            continue

        written = _written_persistence(event)

        if (written):
            report(event, [(name, technique, f"write to {', '.join(paths_of(event))}")
                           for (name, technique) in written], [event["ref"]], False)

    return found


def _manifests(events: list[dict]) -> list[dict]:
    return [event for event in events if event["type"] == "tool_manifest"]


def hunt_poisoned_tools(events: list[dict], ctx: HuntContext) -> list[Finding]:
    del ctx
    found = []
    seen = set()

    for event in _manifests(events):
        for tool in event["tools"]:
            key = (event["server"], str(tool.get("name")), mcp_audit.definition_digest(tool))
            poisoned = mcp_audit.poisoned_tool(tool) if key not in seen else None

            if (poisoned is None):
                continue

            seen.add(key)
            found.append(
                Finding("AAH-005", "critical" if poisoned["score"] >= 8 else "high",
                        f"tool {event['server']}.{poisoned['tool']} carries instructions for the model "
                        f"(score {poisoned['score']}, in {', '.join(poisoned['places'][:4])})", event["session_id"],
                        _iso(event["ts"]), _indicator_lines(poisoned["indicators"]), [event["ref"]],
                        {"server": event["server"], "tool": poisoned["tool"], "score": poisoned["score"],
                         "categories": poisoned["categories"]}))

    return found


def hunt_shadowed_tools(events: list[dict], ctx: HuntContext) -> list[Finding]:
    """
    One tool name from two servers, or a description that steers another server's tool. Reported once per set of
    servers, not once per session, and at high only when the shadowing definition also carries an indicator: two
    servers both offering `search` is common and innocent.
    """
    del ctx
    found = []
    seen = set()

    for (session, group) in sessions(events):
        offered = collections.defaultdict(dict)
        refs = collections.defaultdict(list)

        for event in _manifests(group):
            for tool in event["tools"]:
                if (tool.get("name")):
                    offered[event["server"]][str(tool["name"])] = tool
                    refs[event["server"]].append(event["ref"])

        servers_of = collections.defaultdict(set)

        for (server, tools) in offered.items():
            for name in tools:
                servers_of[name].add(server)

        for (name, servers) in sorted(servers_of.items()):
            key = ("name", name, tuple(sorted(servers)))

            if (len(servers) < 2 or key in seen):
                continue

            seen.add(key)
            marked = [server for server in sorted(servers) if mcp_audit.tool_assessment(offered[server][name]).score > 0]
            found.append(
                Finding("AAH-006", "high" if marked else "low", f"tool name {name!r} is offered by {len(servers)} servers",
                        session, None, sorted(servers) + [f"{server}'s definition carries an injection indicator"
                                                          for server in marked],
                        sorted({ref for server in servers for ref in refs[server]}),
                        {"tool": name, "servers": sorted(servers), "marked": marked}))

        names = {server: set(tools) for (server, tools) in offered.items()}

        for (server, tools) in sorted(offered.items()):
            for (name, tool) in sorted(tools.items()):
                mentioned = mcp_audit.cross_references(server, tool, names)
                key = ("mention", server, name, tuple(mentioned))

                if (not mentioned or key in seen):
                    continue

                seen.add(key)
                marked = mcp_audit.tool_assessment(tool).score > 0
                found.append(
                    Finding("AAH-006", "high" if marked else "low",
                            f"{server}.{name}'s description names other servers' tools", session, None,
                            [text_indicators.defang(str(tool.get("description") or ""))], refs[server][:5],
                            {"tool": f"{server}.{name}", "mentions": mentioned, "marked": marked}))

    return found


def hunt_redefined_tools(events: list[dict], ctx: HuntContext) -> list[Finding]:
    found = []
    known = {server: dict(tools) for (server, tools) in ctx.tool_baseline.items()}
    earliest = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

    for event in sorted(_manifests(events), key=lambda event: event["ts"] or earliest):
        tools = event["tools"]
        server = event["server"]

        for change in mcp_audit.changed_definitions(known, server, tools):
            if (change["change"] != "redefined"):
                continue

            tool = next(tool for tool in tools if str(tool.get("name")) == change["tool"])
            found.append(
                Finding("AAH-007", "high" if mcp_audit.poisoned_tool(tool) else "medium",
                        f"{server}.{change['tool']} was redefined after it was first seen", event["session_id"],
                        _iso(event["ts"]), [text_indicators.defang(str(tool.get("description") or ""))],
                        [event["ref"]], change))

        known.setdefault(server, {}).update(mcp_audit.baseline_of({server: tools})[server])

    return found


def audit_configs(configs: list[tuple]) -> list[Finding]:
    """AAH-008 over MCP client configurations given as `(name, document)` pairs."""
    found = []

    for (name, document) in configs:
        for item in mcp_audit.audit_config(document):
            found.append(
                Finding("AAH-008", item["severity"], f"{item['server']}: {item['check']}", "", None, [item["detail"]],
                        [name], {"server": item["server"], "check": item["check"]}))

    return found


def hunt_agent_credentials(events: list[dict], ctx: HuntContext) -> list[Finding]:
    found = []
    known = collections.defaultdict(set, {key: set(value) for (key, value) in ctx.auth_baseline.items()})
    earliest = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

    for event in sorted((event for event in events if event["type"] == "auth"), key=lambda e: e["ts"] or earliest):
        auth = event["auth"]
        identity = str(auth.get("identity") or auth.get("token_id") or "")

        if (not identity or not (event["agent"] or auth.get("identity_type") in ("agent", "service", "workload"))):
            continue

        source = str(auth.get("source_asn") or auth.get("source_ip") or "")

        if (source and known[identity] and source not in known[identity]):
            found.append(
                Finding("AAH-009", "high", f"agent identity {identity} used from new source {source}",
                        event["session_id"], _iso(event["ts"]),
                        [f"known sources: {', '.join(sorted(known[identity])[:10])}"], [event["ref"]],
                        {"identity": identity, "source": source}))

        declared = set(auth.get("declared_scopes") or [])
        extra = sorted(set(auth.get("scopes") or []) - declared) if declared else []

        if (extra):
            found.append(
                Finding("AAH-009", "high", f"agent identity {identity} used scopes beyond its declaration",
                        event["session_id"], _iso(event["ts"]), extra, [event["ref"]],
                        {"identity": identity, "extra_scopes": extra}))

        if (source):
            known[identity].add(source)

    return found


def hunt_destructive_activity(events: list[dict], ctx: HuntContext) -> list[Finding]:
    found = []

    for (event, facts, refs) in _commands_run(events):
        if (not facts.destructive):
            continue

        names = list(dict.fromkeys(name for (name, _, _) in facts.destructive))
        severity = _worst(level for (_, level, _) in facts.destructive)
        found.append(
            Finding("AAH-010", severity, f"agent issued {', '.join(names)}", event["session_id"], _iso(event["ts"]),
                    list(dict.fromkeys(text_indicators.defang(evidence, 240)
                                       for (_, _, evidence) in facts.destructive)), refs,
                    {"patterns": names, "uncertain_parse": facts.uncertain}))

    for (session, group) in sessions(events):
        calls = [event for event in group if event["type"] == "tool_call" and event["ts"] is not None]
        (start, busiest) = (0, (0, 0, 0))

        for (end, call) in enumerate(calls):
            while (call["ts"] - calls[start]["ts"]).total_seconds() > 60:
                start += 1

            busiest = max(busiest, (end - start + 1, start, end))

        if (busiest[0] > ctx.calls_per_minute):
            (count, first, last) = busiest
            found.append(
                Finding("AAH-010", "medium", f"{count} tool calls within one minute", session,
                        _iso(calls[first]["ts"]), [], [calls[first]["ref"], calls[last]["ref"]], {"calls": count}))

        read = set()

        for event in group:
            if ((event["type"] == "file" and str(event["file"].get("op") or "read") == "read")
                    or (event["type"] == "tool_call" and READ in tool_capabilities(event["tool"]))):
                read.update(paths_of(event))

        if (len(read) > ctx.distinct_files):
            found.append(
                Finding("AAH-010", "medium", f"{len(read)} distinct files read in one session", session, None, [],
                        [group[0]["ref"]], {"files": len(read)}))

    return found


def _written_content(event: dict) -> str:
    if (event["type"] == "file"):
        return str(event["file"].get("content") or "")

    return "\n".join(_strings({key: event["args"].get(key) for key in CONTENT_KEYS}))


INSTRUCTION_WIDENING = re.compile(
    r"\b(?:without\s{1,3}(?:asking|confirm\w{0,8}|approval|permission|prompting)|"
    r"(?:never|do\s{1,3}not|don't)\s{1,3}(?:ask|prompt|confirm|request\s{1,3}(?:approval|permission))|"
    r"(?:always|automatically)\s{1,3}(?:approve|allow|accept|run\s{1,3}(?:any|all|every|commands?))|"
    r"no\s{1,3}(?:confirmation|approval)\s{1,3}(?:is\s{1,3})?(?:needed|required)|"
    r"skip\s{1,3}(?:the\s{1,3})?(?:approval|confirmation|permission)s?|"
    r"(?:do\s{1,3}not|don't|never)\s{1,3}(?:tell|mention|inform|show)\s{1,3}(?:the\s{1,3})?(?:user|human|developer))",
    re.I)
"""Standing instructions that tell an agent to stop asking, or to keep something from the person it works for."""

INSTRUCTION_FILE = re.compile(r"(?i)\.(?:md|mdc|txt)$|rules(?:[/\\]|$)|\.cursorrules$|\.windsurfrules$|\.goosehints$")


def hunt_guardrail_tampering(events: list[dict], ctx: HuntContext) -> list[Finding]:
    """
    An agent changing what governs agents: a write to configuration or standing instructions, an MCP server
    registered from the command line, an agent started with its approvals off. Reading those files is not a write.

    What was written sets the severity. High where it turns approvals off, adds servers or hooks, tells the agent to
    stop asking or to keep something from the user, or carries injection indicators. Medium for any other change to
    a file that holds permissions, and for a write whose content the telemetry does not show. Low for ordinary text
    added to an instructions file, which is what `/init` and most documentation edits are.
    """
    del ctx
    found = []

    def report(event: dict, paths: list, content: str | None, refs: list):
        text = content or ""
        widening = sorted({match.lower() for match in PERMISSION_WIDENING.findall(text)}
                          | {" ".join(match.lower().split()) for match in INSTRUCTION_WIDENING.findall(text)})
        injected = text_indicators.assess(text).likely
        prose = all(INSTRUCTION_FILE.search(path) for path in paths)

        if (widening or injected):
            severity = "high"
        elif (content is None or not content.strip() or not prose):
            severity = "medium"
        else:
            severity = "low"

        evidence = [*sorted(set(paths)), *(f"widens: {item}" for item in widening)]

        if (injected):
            evidence.append("the written text carries injection indicators")

        found.append(
            Finding("AAH-011", severity, "an agent wrote to agent configuration or standing instructions",
                    event["session_id"], _iso(event["ts"]), evidence, refs,
                    {"paths": sorted(set(paths)), "widening": widening, "injection": injected}))

    for event in events:
        if (event["type"] == "file" and str(event["file"].get("op") or "") in ("write", "create", "modify", "append",
                                                                              "delete", "rename")):
            written = True
        else:
            written = event["type"] == "tool_call" and WRITE in tool_capabilities(event["tool"])

        if (not written):
            continue

        content = _written_content(event)
        targets = [path for path in paths_of(event) if commands.AGENT_CONFIG_PATH.search(path.strip())]
        # An editor's settings file governs agents only in the keys that do.
        targets.extend(path for path in paths_of(event) if commands.EDITOR_SETTINGS_PATH.search(path.strip())
                       and commands.EDITOR_AGENT_SETTING.search(content))

        if (targets):
            report(event, targets, content if event["type"] == "tool_call" or content else None, [event["ref"]])

    for (event, facts, refs) in _commands_run(events):
        if (facts.config_writes):
            # A command shows what it writes only when the text is in the line: a heredoc or an `echo`.
            report(event, [path for (path, _) in facts.config_writes], command_of(event) + "\n", refs)

        for evidence in facts.mcp_registrations:
            found.append(
                Finding("AAH-011", "medium", "an agent registered an MCP server from the command line",
                        event["session_id"], _iso(event["ts"]), [text_indicators.defang(evidence, 240)], refs,
                        {"registration": True}))

        for (flag, evidence) in facts.guardrail_flags:
            found.append(
                Finding("AAH-011", "medium", "an agent was started with its approvals turned off",
                        event["session_id"], _iso(event["ts"]), [text_indicators.defang(evidence, 240)], refs,
                        {"flag": flag}))

    # A person starting an agent with its approvals off is the same exposure, whoever typed it.
    for event in events:
        if (event["type"] == "process" and not from_agent(event)):
            for (flag, evidence) in commands.facts(command_of(event)).guardrail_flags:
                found.append(
                    Finding("AAH-011", "medium", "an agent was started with its approvals turned off",
                            event["session_id"], _iso(event["ts"]), [text_indicators.defang(evidence, 240)],
                            [event["ref"]], {"flag": flag}))

    return found


def _secrets_in(texts) -> list:
    """Credentials in each text: as given, URL-decoded, and inside base64 runs."""
    found = []

    for text in texts:
        found.extend(text_indicators.find_secrets(text) or text_indicators.find_secrets(urllib.parse.unquote(text)))
        found.extend(text_indicators.encoded_secrets(text))

    return found


def hunt_leaked_credentials(events: list[dict], ctx: HuntContext) -> list[Finding]:
    """
    A credential where it should not be: in what the model said, in the arguments or command line of a call, or in
    what a tool returned, which is the model's context and the provider's logs from then on.

    Critical where the call that carries it also sends to the network; high in model output; medium in arguments
    that stay on the machine and in tool output. A value a vendor publishes as an example is reported at low.
    """
    found = []

    for event in events:
        if (event["type"] == "model_output"):
            (secrets, where, severity) = (_secrets_in([event["content"]]), "model output", "high")
        elif (event["type"] == "tool_result"):
            (secrets, where, severity) = (text_indicators.find_secrets(event["content"]),
                                          f"output of {event['tool'] or 'a tool'}", "medium")
        elif (event["type"] == "tool_call" or (event["type"] == "process" and from_agent(event))):
            texts = _strings(event["args"]) if event["type"] == "tool_call" else [command_of(event)]
            secrets = _secrets_in(texts)
            where = f"arguments of {event['tool']}" if event["type"] == "tool_call" else "an agent's command line"
            severity = "critical" if any(item.sends for item in egress_of(event, ctx)) else "medium"
        else:
            continue

        if (not secrets):
            continue

        if (all(indicator.weight == 0 for indicator in secrets)):
            severity = "low"

        found.append(
            Finding("AAH-012", severity, f"credential(s) in {where}", event["session_id"], _iso(event["ts"]),
                    list(dict.fromkeys(indicator.evidence for indicator in secrets)), [event["ref"]],
                    {"kinds": sorted({indicator.name for indicator in secrets}), "where": where.split(" of ")[0]}))

    return found


HUNTS = {
    "AAH-001": hunt_injected_content,
    "AAH-002": hunt_injection_then_action,
    "AAH-003": hunt_secret_then_egress,
    "AAH-004": hunt_risky_commands,
    "AAH-005": hunt_poisoned_tools,
    "AAH-006": hunt_shadowed_tools,
    "AAH-007": hunt_redefined_tools,
    "AAH-009": hunt_agent_credentials,
    "AAH-010": hunt_destructive_activity,
    "AAH-011": hunt_guardrail_tampering,
    "AAH-012": hunt_leaked_credentials,
}
"""The event hunts. AAH-008 reads configurations rather than events and runs through `audit_configs`."""


def run_hunts(events: list[dict],
              configs: list[tuple] | None = None,
              ctx: HuntContext | None = None,
              hunt_ids: list[str] | None = None,
              min_severity: str = "info",
              failures: list | None = None) -> list[Finding]:
    """
    The findings of the chosen hunts (all of them by default), most severe first.

    A hunt that raises stops the run, unless `failures` is given: then its id and the error's type are appended
    there and the other hunts still report, so that one event a hunt cannot digest does not hide every other
    finding. The caller owns saying so; a hunt that did not run is a blind spot, not a clean result.
    """
    ctx = ctx or HuntContext()
    chosen = list(hunt_ids or catalog.BY_ID)
    unknown = [hunt_id for hunt_id in chosen if hunt_id not in catalog.BY_ID]

    if (unknown):
        raise ValueError(f"no such hunt: {', '.join(text_indicators.clean(str(item), 40) for item in unknown)}")

    if (min_severity not in catalog.SEVERITIES):
        raise ValueError(f"severity must be one of {', '.join(catalog.SEVERITIES)}")

    found = []

    for hunt_id in chosen:
        try:
            if (hunt_id == "AAH-008"):
                found.extend(audit_configs(configs or []))
            else:
                found.extend(HUNTS[hunt_id](events, ctx))
        except Exception as error:  # pylint: disable=broad-exception-caught
            if (failures is None):
                raise

            failures.append((hunt_id, type(error).__name__))

    floor = catalog.SEVERITIES.index(min_severity)
    kept = [finding for finding in found if catalog.SEVERITIES.index(finding.severity) >= floor]

    return sorted(kept, key=lambda f: (-catalog.SEVERITIES.index(f.severity), f.hunt_id, f.session_id, f.ts or ""))
