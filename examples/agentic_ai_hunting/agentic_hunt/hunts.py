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

Each hunt is one function from the events (ordered by session and time) and a `HuntContext` to findings. A finding
quotes what it saw only through `text_indicators.defang`, and never quotes a credential, so a finding is safe to show
to a person or hand to the model that asked for it. What a finding quotes is still evidence from a possibly hostile
source: it is data to analyse, never an instruction to follow.
"""

import collections
import dataclasses
import datetime
import itertools
import json
import re
import urllib.parse

from . import catalog
from . import mcp_audit
from . import text_indicators

LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
MODEL_API_HOSTS = ("api.anthropic.com",
                   "api.openai.com",
                   "generativelanguage.googleapis.com",
                   "bedrock-runtime.us-east-1.amazonaws.com",
                   "api.githubcopilot.com")
"""Where an agent sends its context on every turn. Egress there is the agent working, not a hunt's egress."""
DEFAULT_ALLOWLIST = ("github.com", "api.github.com", "registry.npmjs.org", "pypi.org", "files.pythonhosted.org")
"""Destinations that are ordinary for a coding agent. Egress there still counts, at a lower severity."""

AGENT_IMAGES = re.compile(
    r"(?:^|[/\\])(?:claude|cursor|codex|copilot|windsurf|aider|gemini|cline|goose|"
    r"code-insiders|code)(?:\.exe)?$",
    re.I)
"""Process images of agents and of the editors that host them; a process under one of these is an agent's."""

SECRET_PATH = re.compile(
    r"(?<![\w.-])(?:~|\$HOME|/home/[^/\s]{1,64}|/Users/[^/\s]{1,64}|/root)?/?"
    r"(?:\.ssh/(?:id_[a-z0-9]+|[^\s/]{0,128}\.pem|config)|"
    r"\.aws/credentials|\.config/gcloud/(?:credentials\.db|application_default_credentials\.json)|\.azure/"
    r"(?:accessTokens\.json|msal_token_cache\.json)|\.kube/config|\.docker/config\.json|\.npmrc|\.pypirc|"
    r"\.git-credentials|\.netrc|\.config/gh/hosts\.yml|\.claude/\.credentials\.json|\.env(?:\.[a-z]+)?|"
    r"Library/Keychains/[^\s]{1,256}|Login Data|Cookies|/etc/shadow)(?![\w.-])",
    re.I)
"""Files that hold credentials: keys, cloud and registry tokens, environment files, keychains, browser stores."""

AGENT_CONFIG_PATH = re.compile(
    r"(?:^|[/\\])(?:\.claude[/\\]settings(?:\.local)?\.json|\.claude\.json|\.mcp\.json|\.cursor[/\\]mcp\.json|"
    r"\.cursor[/\\]rules(?:[/\\][^\s]{0,256})?|\.cursorrules|\.vscode[/\\](?:mcp|settings)\.json|"
    r"claude_desktop_config\.json|\.windsurfrules|\.codeium[/\\]windsurf[/\\]mcp_config\.json|"
    r"\.github[/\\]copilot-instructions\.md|CLAUDE\.md|AGENTS\.md|GEMINI\.md|\.gemini[/\\]settings\.json|"
    r"\.codex[/\\]config\.toml|\.aider\.conf\.yml)$")
"""Files agents read their permissions, MCP servers and standing instructions from."""

PERMISSION_WIDENING = re.compile(
    r"\"Bash\(\*\)\"|\"allow\"\s*:\s*\[\s*\"\*\"|enableAllProjectMcpServers\"\s*:\s*true|\"autoApprove\"|"
    r"\"alwaysAllow\"|bypassPermissions|dangerously-skip-permissions|chat\.tools\.autoApprove\"\s*:\s*true|"
    r"approval_policy\s*=\s*\"never\"|sandbox_mode\s*=\s*\"danger-full-access\"|\"mcpServers\"\s*:",
    re.I)
"""Edits that turn an agent's approvals off or add servers it will load."""

GUARDRAIL_FLAGS = re.compile(r"--dangerously-skip-permissions|--dangerously-bypass-approvals-and-sandbox|--yolo\b")

RISKY_COMMANDS = (
    ("download_and_execute",
     "T1105",
     re.compile(
         r"(?:curl|wget)\b[^|;&\n]{0,512}\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b|(?:curl|wget)\b[^|;&\n]{0,512}\|\s*python|"
         r"\biex\s*\(\s*(?:new-object\s+net\.webclient|iwr|invoke-webrequest)",
         re.I)),
    ("decode_and_execute",
     "T1027",
     re.compile(
         r"base64\s+(?:-d|--decode)\b[^|;\n]{0,512}\|\s*(?:ba|z)?sh\b|"
         r"powershell(?:\.exe)?\s+[^\n]{0,256}-e(?:nc|ncodedcommand)?\s+"
         r"[A-Za-z0-9+/=]{20,}",
         re.I)),
    ("reverse_shell",
     "T1059.004",
     re.compile(
         r"/dev/tcp/[\w.-]{1,255}/\d{1,5}|\bnc(?:at)?\b[^\n]{0,256}\s-[ec]\s|\bsocat\b[^\n]{0,256}exec:|bash\s+-i\s+>&",
         re.I)),
    ("shell_profile_persistence",
     "T1546.004",
     re.compile(r">>?\s{0,8}(?:~|\$HOME)?/?[^\s]{0,256}\.(?:bashrc|zshrc|profile|bash_profile|zprofile)\b", re.I)),
    ("scheduled_persistence",
     "T1053.003",
     re.compile(
         r"\bcrontab\s+(?:-[a-z]*\s+)*-\s*$|\|\s*crontab\b|launchctl\s+(?:load|bootstrap)\b|systemctl\s+--user\s+"
         r"enable\b",
         re.I | re.M)),
    ("history_tampering",
     "T1070.003",
     re.compile(r"unset\s+HISTFILE|HISTFILE=/dev/null|set\s+\+o\s+history|\bhistory\s+-c\b", re.I)),
)
"""Command shapes an agent has no ordinary reason to produce, each with its ATT&CK technique."""

DESTRUCTIVE_COMMANDS = (
    ("recursive_delete", re.compile(r"\brm\s+-[a-z]*r[a-z]*f?[a-z]*\s+(?:/|~|\$HOME|\*|\.\s|\.$|/\*)", re.I | re.M)),
    ("force_push", re.compile(r"\bgit\s+push\b[^\n]{0,256}(?:\s--force\b|\s-f\b|\s\+[\w/-]+)")),
    ("drop_database_object", re.compile(r"\b(?:DROP\s+(?:TABLE|DATABASE|SCHEMA)|TRUNCATE\s+TABLE)\b", re.I)),
    ("cloud_storage_delete",
     re.compile(
         r"\baws\s+s3\s+(?:rb\b|rm\b[^\n]{0,256}--recursive)|\bgsutil\s+(?:-m\s+)?rm\s+-r|"
         r"\baz\s+storage\s+(?:container|blob)\s+delete",
         re.I)),
    ("cluster_delete", re.compile(r"\bkubectl\s+delete\s+(?:ns|namespace|deployment|pvc|--all)\b", re.I)),
    ("infrastructure_destroy", re.compile(r"\bterraform\s+destroy\b|\bpulumi\s+destroy\b|\bgh\s+repo\s+delete\b",
                                          re.I)),
)

COMMAND_KEYS = ("command", "cmd", "script", "code", "shell", "commandLine")
PATH_KEYS = ("path", "file", "file_path", "filepath", "filename", "target", "destination", "source", "uri")
CONTENT_KEYS = ("content", "contents", "text", "new_string", "body", "data")
URL = re.compile(r"\bhttps?://[^\s\"'<>)]+", re.I)
EGRESS_COMMAND = re.compile(
    r"\b(?:curl|wget|nc|ncat|scp|rsync|ftp|sftp|ssh)\b[^\n]{0,256}?(?:https?://|\s|@)"
    r"(?P<host>[a-z0-9-]{1,63}(?:\.[a-z0-9-]{1,63}){1,8})(?![\w.-])",
    re.I)

CAPABILITY_WORDS = {
    "execute": {
        "bash",
        "shell",
        "exec",
        "execute",
        "run",
        "terminal",
        "command",
        "powershell",
        "cmd",
        "subprocess",
        "spawn",
        "eval"
    },
    "write": {
        "write",
        "edit",
        "create",
        "save",
        "patch",
        "apply",
        "delete",
        "remove",
        "move",
        "rename",
        "append",
        "replace",
        "upload"
    },
    "egress": {
        "fetch",
        "http",
        "request",
        "post",
        "upload",
        "send",
        "email",
        "mail",
        "webhook",
        "slack",
        "browse",
        "navigate",
        "curl",
        "publish",
        "share",
        "message"
    },
}


@dataclasses.dataclass
class HuntContext:
    """What the hunts are tuned with. Every value has a default a first hunt can run on."""

    window_seconds: int = 600
    injection_threshold: int = 3
    egress_allowlist: tuple = DEFAULT_ALLOWLIST
    model_api_hosts: tuple = MODEL_API_HOSTS
    tool_baseline: dict = dataclasses.field(default_factory=dict)
    auth_baseline: dict = dataclasses.field(default_factory=dict)
    calls_per_minute: int = 30
    distinct_files: int = 50


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
        hunt = catalog.BY_ID[self.hunt_id]

        return {
            **dataclasses.asdict(self),
            "title": hunt.title,
            "atlas": list(hunt.atlas),
            "attack": list(hunt.attack),
            "owasp": list(hunt.owasp)
        }


def _iso(value: datetime.datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _within(earlier: dict, later: dict, ctx: HuntContext) -> bool:
    if (earlier["ts"] is None or later["ts"] is None):
        return True

    gap = (later["ts"] - earlier["ts"]).total_seconds()

    return 0 <= gap <= ctx.window_seconds


def sessions(events: list[dict]):
    """Events grouped by session, each group in time order, as `events.order` leaves them."""
    for (session, group) in itertools.groupby(events, key=lambda event: event["session_id"]):
        yield (session, list(group))


def _strings(value) -> list[str]:
    if (isinstance(value, str)):
        return [value]
    if (isinstance(value, dict)):
        return [text for item in value.values() for text in _strings(item)]
    if (isinstance(value, list)):
        return [text for item in value for text in _strings(item)]
    return []


def _arg(event: dict, keys: tuple) -> list[str]:
    return [str(event["args"][key]) for key in keys if isinstance(event["args"].get(key), (str, int, float))]


def command_of(event: dict) -> str:
    """The command line an event runs: a process's, or the command argument of a tool call."""
    if (event["type"] == "process"):
        return str(event["process"].get("cmdline") or "")

    if (event["type"] == "tool_call"):
        return " ".join(_arg(event, COMMAND_KEYS))

    return ""


def paths_of(event: dict) -> list[str]:
    if (event["type"] == "file"):
        return [str(event["file"].get("path") or "")]

    if (event["type"] == "tool_call"):
        return _arg(event, PATH_KEYS)

    return []


def _tokens(name: str) -> set:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)

    return {token.lower() for token in re.split(r"[^A-Za-z0-9]+", spaced) if token}


def from_agent(event: dict) -> bool:
    """Whether a process event belongs to an agent: attributed to one, or started under one's process tree."""
    if (event["agent"] or event["process"].get("agent_lineage")):
        return True

    ancestors = [event["process"].get("parent_image") or "", *(event["process"].get("ancestors") or [])]

    return any(AGENT_IMAGES.search(str(image)) for image in ancestors)


def egress_hosts(event: dict, ctx: HuntContext) -> list[str]:
    """The hosts an event sends to, leaving out the local machine and the model provider."""
    hosts = []

    if (event["type"] == "network"):
        hosts.append(str(event["network"].get("dest_host") or event["network"].get("dest_ip") or ""))
    elif (event["type"] == "tool_call"):
        for text in _strings(event["args"]):
            hosts.extend(urllib.parse.urlparse(url).hostname or "" for url in URL.findall(text))

        if (_tokens(event["tool"]) & CAPABILITY_WORDS["egress"] and event["args"].get("to")):
            hosts.append(str(event["args"]["to"]).rsplit("@", 1)[-1])

    command = command_of(event)

    if (command):
        hosts.extend(match.group("host") for match in EGRESS_COMMAND.finditer(command))

    ignored = (*LOCAL_HOSTS, *ctx.model_api_hosts)

    return sorted({host.lower() for host in hosts if host and host.lower() not in ignored})


def secret_paths(event: dict) -> list[str]:
    """The credential files an event reads: a file read, a path argument, or a path in a command line."""
    texts = list(paths_of(event))

    if (event["type"] == "file" and str(event["file"].get("op") or "read") not in ("read", "open", "copy")):
        texts = []

    # Writing a credential file is not reading one.
    if (event["type"] == "tool_call" and _tokens(event["tool"]) & CAPABILITY_WORDS["write"]
            and "read" not in _tokens(event["tool"])):
        texts = []

    command = command_of(event)

    if (command):
        texts.append(command)

    return sorted({match.group(0) for text in texts for match in SECRET_PATH.finditer(text)})


def capabilities(event: dict, ctx: HuntContext) -> set:
    """What an event does that matters after an injection: execute, write, egress, read_secret."""
    found = set()

    if (event["type"] == "tool_call"):
        tokens = _tokens(event["tool"])
        found |= {capability for (capability, words) in CAPABILITY_WORDS.items() if tokens & words}

        if (command_of(event)):
            found.add("execute")
    elif (event["type"] == "process" and from_agent(event)):
        found.add("execute")
    elif (event["type"] == "file"
          and str(event["file"].get("op") or "") in ("write", "create", "modify", "append", "delete", "rename")):
        found.add("write")

    if (egress_hosts(event, ctx)):
        found.add("egress")
    else:
        found.discard("egress")

    if (secret_paths(event)):
        found.add("read_secret")

    return found


def _injection(event: dict, ctx: HuntContext) -> tuple:
    indicators = text_indicators.scan_text(event["content"])
    total = text_indicators.score(indicators)

    return (total >= ctx.injection_threshold, total, indicators)


def _evidence(indicators: list, limit: int = 5) -> list[str]:
    return [f"{indicator.kind}/{indicator.name}: {indicator.evidence}" for indicator in indicators[:limit]]


def _source(event: dict) -> str:
    if (event["type"] == "tool_result"):
        return f"result of {event['server'] + '.' if event['server'] else ''}{event['tool'] or 'a tool'}"

    return "a retrieved document"


def hunt_injected_content(events: list[dict], ctx: HuntContext) -> list[Finding]:
    found = []

    for event in events:
        if (event["type"] not in ("tool_result", "document")):
            continue

        (injected, total, indicators) = _injection(event, ctx)

        if (injected):
            found.append(
                Finding("AAH-001",
                        "high" if total >= 6 else "medium",
                        f"{_source(event)} addresses the model (indicator score {total})",
                        event["session_id"],
                        _iso(event["ts"]),
                        _evidence(indicators), [event["ref"]], {"score": total}))

    return found


def hunt_injection_then_action(events: list[dict], ctx: HuntContext) -> list[Finding]:
    found = []

    for (session, group) in sessions(events):
        for (index, event) in enumerate(group):
            if (event["type"] not in ("tool_result", "document") or not _injection(event, ctx)[0]):
                continue

            actions = []

            for later in group[index + 1:]:
                if (not _within(event, later, ctx)):
                    break

                done = capabilities(later, ctx) & {"execute", "write", "egress", "read_secret"}

                if (done):
                    actions.append((later, sorted(done)))

            if (actions):
                worst = {capability for (_, done) in actions for capability in done}
                severity = "critical" if worst & {"egress", "read_secret"} else "high"
                evidence = [
                    f"{later['type']} {later['tool'] or command_of(later)[:60] or ''}: {', '.join(done)}"
                    for (later, done) in actions[:5]
                ]
                found.append(
                    Finding("AAH-002",
                            severity,
                            f"{len(actions)} privileged action(s) followed injected content in {_source(event)}",
                            session,
                            _iso(event["ts"]), [text_indicators.defang(line) for line in evidence],
                            [event["ref"], *(later["ref"] for (later, _) in actions)], {"capabilities": sorted(worst)}))

    return found


def hunt_secret_then_egress(events: list[dict], ctx: HuntContext) -> list[Finding]:
    found = []

    for (session, group) in sessions(events):
        reported = {}

        for (index, event) in enumerate(group):
            read = secret_paths(event)

            if (not read):
                continue

            hosts = set(egress_hosts(event, ctx))
            refs = [event["ref"]]

            for later in group[index + 1:]:
                if (not _within(event, later, ctx)):
                    break

                sent = egress_hosts(later, ctx)

                if (sent):
                    hosts.update(sent)
                    refs.append(later["ref"])

            if (not hosts):
                continue

            # A second read of the same file (the tool call, then the file event it caused) joins the first finding.
            earlier = next((finding for (sent, finding) in reported.items() if set(refs[1:]) <= set(sent)), None)

            if (earlier is not None):
                earlier.refs.append(event["ref"])
                earlier.details["secret_paths"] = sorted(set(earlier.details["secret_paths"]) | set(read))
                continue

            unlisted = sorted(host for host in hosts if host not in ctx.egress_allowlist)
            finding = Finding("AAH-003",
                              "critical" if unlisted else "high",
                              f"credential file(s) read, then data sent to {', '.join(sorted(hosts))}",
                              session,
                              _iso(event["ts"]),
                              [text_indicators.defang(path)
                               for path in read] + [text_indicators.defang(host) for host in sorted(hosts)],
                              refs, {
                                  "secret_paths": read, "hosts": sorted(hosts), "unlisted_hosts": unlisted
                              })
            reported[tuple(refs[1:])] = finding
            found.append(finding)

    return found


def hunt_risky_commands(events: list[dict], ctx: HuntContext) -> list[Finding]:
    del ctx
    found = []

    for event in events:
        if (event["type"] == "process" and not from_agent(event)):
            continue

        command = command_of(event)
        matched = [(name, technique) for (name, technique, pattern) in RISKY_COMMANDS if pattern.search(command)]

        if (matched):
            severity = "critical" if any(name in ("reverse_shell", "download_and_execute") for (name, _) in matched) \
                else "high"
            found.append(
                Finding(
                    "AAH-004",
                    severity,
                    f"agent ran {', '.join(name for (name, _) in matched)}",
                    event["session_id"],
                    _iso(event["ts"]), [text_indicators.defang(command, 240)], [event["ref"]],
                    {
                        "patterns": [name for (name, _) in matched],
                        "techniques": sorted({technique
                                              for (_, technique) in matched})
                    }))

    return found


def _manifests(events: list[dict]) -> list[dict]:
    return [event for event in events if event["type"] == "tool_manifest"]


def hunt_poisoned_tools(events: list[dict], ctx: HuntContext) -> list[Finding]:
    del ctx
    found = []
    seen = set()

    for event in _manifests(events):
        for tool in event["tools"]:
            if (not isinstance(tool, dict)):
                continue

            key = (event["server"], str(tool.get("name")), mcp_audit.definition_digest(tool))
            poisoned = mcp_audit.poisoned_tool(tool)

            if (poisoned is None or key in seen):
                continue

            seen.add(key)
            indicators = [text_indicators.Indicator(**item) for item in poisoned["indicators"]]
            found.append(
                Finding(
                    "AAH-005",
                    "critical" if poisoned["score"] >= 8 else "high",
                    f"tool {event['server']}.{poisoned['tool']} carries instructions for the model "
                    f"(score {poisoned['score']}, in {', '.join(poisoned['places'])})",
                    event["session_id"],
                    _iso(event["ts"]),
                    _evidence(indicators), [event["ref"]], {
                        "server": event["server"], "tool": poisoned["tool"], "score": poisoned["score"]
                    }))

    return found


def hunt_shadowed_tools(events: list[dict], ctx: HuntContext) -> list[Finding]:
    del ctx
    found = []

    for (session, group) in sessions(events):
        offered = collections.defaultdict(dict)
        refs = collections.defaultdict(list)

        for event in _manifests(group):
            for tool in event["tools"]:
                if (isinstance(tool, dict) and tool.get("name")):
                    offered[event["server"]][str(tool["name"])] = tool
                    refs[event["server"]].append(event["ref"])

        servers_of = collections.defaultdict(set)

        for (server, tools) in offered.items():
            for name in tools:
                servers_of[name].add(server)

        for (name, servers) in sorted(servers_of.items()):
            if (len(servers) > 1):
                found.append(
                    Finding("AAH-006",
                            "high",
                            f"tool name {name!r} is offered by {len(servers)} servers",
                            session,
                            None,
                            sorted(servers),
                            sorted({ref
                                    for server in servers
                                    for ref in refs[server]}), {
                                        "tool": name, "servers": sorted(servers)
                                    }))

        names = {server: set(tools) for (server, tools) in offered.items()}

        for (server, tools) in sorted(offered.items()):
            for (name, tool) in sorted(tools.items()):
                mentioned = mcp_audit.cross_references(server, tool, names)

                if (mentioned):
                    found.append(
                        Finding("AAH-006",
                                "high" if mcp_audit.poisoned_tool(tool) else "medium",
                                f"{server}.{name}'s description names other servers' tools",
                                session,
                                None, [text_indicators.defang(str(tool.get("description") or ""))],
                                refs[server], {
                                    "tool": f"{server}.{name}", "mentions": mentioned
                                }))

    return found


def hunt_redefined_tools(events: list[dict], ctx: HuntContext) -> list[Finding]:
    found = []
    known = {server: dict(tools) for (server, tools) in ctx.tool_baseline.items()}
    earliest = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

    for event in sorted(_manifests(events), key=lambda event: event["ts"] or earliest):
        tools = [tool for tool in event["tools"] if isinstance(tool, dict)]
        server = event["server"]

        for change in mcp_audit.changed_definitions(known, server, tools):
            if (change["change"] != "redefined"):
                continue

            tool = next(tool for tool in tools if str(tool.get("name")) == change["tool"])
            found.append(
                Finding("AAH-007",
                        "high" if mcp_audit.poisoned_tool(tool) else "medium",
                        f"{server}.{change['tool']} was redefined after it was first seen",
                        event["session_id"],
                        _iso(event["ts"]), [text_indicators.defang(str(tool.get("description") or ""))], [event["ref"]],
                        change))

        known.setdefault(server, {}).update(mcp_audit.baseline_of({server: tools})[server])

    return found


def audit_configs(configs: list[tuple]) -> list[Finding]:
    """AAH-008 over MCP client configurations given as `(name, document)` pairs."""
    found = []

    for (name, document) in configs:
        for item in mcp_audit.audit_config(document):
            found.append(
                Finding("AAH-008",
                        item["severity"],
                        f"{item['server']}: {item['check']}",
                        "",
                        None, [item["detail"]], [name], {
                            "server": item["server"], "check": item["check"]
                        }))

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
                Finding("AAH-009",
                        "high",
                        f"agent identity {identity} used from new source {source}",
                        event["session_id"],
                        _iso(event["ts"]), [f"known sources: {', '.join(sorted(known[identity]))}"], [event["ref"]], {
                            "identity": identity, "source": source
                        }))

        declared = set(auth.get("declared_scopes") or [])
        extra = sorted(set(auth.get("scopes") or []) - declared) if declared else []

        if (extra):
            found.append(
                Finding("AAH-009",
                        "high",
                        f"agent identity {identity} used scopes beyond its declaration",
                        event["session_id"],
                        _iso(event["ts"]),
                        extra, [event["ref"]], {
                            "identity": identity, "extra_scopes": extra
                        }))

        if (source):
            known[identity].add(source)

    return found


def hunt_destructive_activity(events: list[dict], ctx: HuntContext) -> list[Finding]:
    found = []

    for event in events:
        if (event["type"] == "process" and not from_agent(event)):
            continue

        command = command_of(event)
        matched = [name for (name, pattern) in DESTRUCTIVE_COMMANDS if pattern.search(command)]

        if (matched):
            found.append(
                Finding("AAH-010",
                        "high",
                        f"agent issued {', '.join(matched)}",
                        event["session_id"],
                        _iso(event["ts"]), [text_indicators.defang(command, 240)], [event["ref"]],
                        {"patterns": matched}))

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
                Finding("AAH-010",
                        "medium",
                        f"{count} tool calls within one minute",
                        session,
                        _iso(calls[first]["ts"]), [], [calls[first]["ref"], calls[last]["ref"]], {"calls": count}))

        read = set()

        for event in group:
            if ((event["type"] == "file" and str(event["file"].get("op") or "read") == "read")
                    or (event["type"] == "tool_call" and "read" in _tokens(event["tool"]))):
                read.update(paths_of(event))

        if (len(read) > ctx.distinct_files):
            found.append(
                Finding("AAH-010",
                        "medium",
                        f"{len(read)} distinct files read in one session",
                        session,
                        None, [], [group[0]["ref"]], {"files": len(read)}))

    return found


def _written_content(event: dict) -> str:
    if (event["type"] == "file"):
        return str(event["file"].get("content") or "")

    return "\n".join(_arg(event, CONTENT_KEYS))


def hunt_guardrail_tampering(events: list[dict], ctx: HuntContext) -> list[Finding]:
    found = []

    for event in events:
        if (event["type"] == "process" and not from_agent(event) and not GUARDRAIL_FLAGS.search(command_of(event))):
            continue

        targets = [path for path in paths_of(event) if AGENT_CONFIG_PATH.search(path.strip())]
        command = command_of(event)
        writes = "write" in capabilities(event, ctx) or event["type"] == "tool_call"

        for match in re.finditer(
                r"(?:>>?|\btee\s+(?:-a\s+)?|\bsed\s+-i\S*\s+(?:'[^']{0,512}'|\"[^\"]{0,512}\"|\S{1,512})\s+)"
                r"(\S{1,512})",
                command):
            if (AGENT_CONFIG_PATH.search(match.group(1))):
                targets.append(match.group(1))
                writes = True

        if (GUARDRAIL_FLAGS.search(command)):
            found.append(
                Finding("AAH-011",
                        "medium",
                        "an agent was started with its approvals turned off",
                        event["session_id"],
                        _iso(event["ts"]), [text_indicators.defang(command, 240)], [event["ref"]],
                        {"flag": GUARDRAIL_FLAGS.search(command).group(0)}))

        if (not targets or not writes):
            continue

        content = _written_content(event) or command
        widening = PERMISSION_WIDENING.findall(content)
        injected = text_indicators.score(text_indicators.scan_text(content)) >= ctx.injection_threshold
        severity = "high" if widening or injected else "medium"
        evidence = [text_indicators.defang(target) for target in sorted(set(targets))]
        evidence += [f"widens: {text_indicators.defang(item)}" for item in sorted(set(widening))]
        found.append(
            Finding("AAH-011",
                    severity,
                    "an agent wrote to agent configuration or standing instructions",
                    event["session_id"],
                    _iso(event["ts"]),
                    evidence, [event["ref"]], {
                        "paths": sorted(set(targets)), "widening": sorted(set(widening)), "injection": injected
                    }))

    return found


def hunt_leaked_credentials(events: list[dict], ctx: HuntContext) -> list[Finding]:
    found = []

    for event in events:
        if (event["type"] == "model_output"):
            secrets = text_indicators.find_secrets(event["content"])
            severity = "high"
        elif (event["type"] == "tool_call"):
            # Arguments are searched as given and URL-decoded, since a key in a query string arrives encoded.
            encoded = json.dumps(event["args"])
            secrets = text_indicators.find_secrets(encoded) or text_indicators.find_secrets(
                urllib.parse.unquote(encoded))
            severity = "critical" if "egress" in capabilities(event, ctx) else "medium"
        else:
            continue

        if (secrets):
            where = "model output" if event["type"] == "model_output" else f"arguments of {event['tool']}"
            found.append(
                Finding("AAH-012",
                        severity,
                        f"credential(s) in {where}",
                        event["session_id"],
                        _iso(event["ts"]), [indicator.evidence for indicator in secrets], [event["ref"]],
                        {"kinds": sorted({indicator.name
                                          for indicator in secrets})}))

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
              min_severity: str = "info") -> list[Finding]:
    """The findings of the chosen hunts (all of them by default), most severe first."""
    ctx = ctx or HuntContext()
    chosen = list(hunt_ids or catalog.BY_ID)
    unknown = [hunt_id for hunt_id in chosen if hunt_id not in catalog.BY_ID]

    if (unknown):
        raise ValueError(f"no such hunt: {', '.join(unknown)}")

    if (min_severity not in catalog.SEVERITIES):
        raise ValueError(f"severity must be one of {', '.join(catalog.SEVERITIES)}")

    found = []

    for hunt_id in chosen:
        if (hunt_id == "AAH-008"):
            found.extend(audit_configs(configs or []))
        else:
            found.extend(HUNTS[hunt_id](events, ctx))

    floor = catalog.SEVERITIES.index(min_severity)
    kept = [finding for finding in found if catalog.SEVERITIES.index(finding.severity) >= floor]

    return sorted(kept, key=lambda f: (-catalog.SEVERITIES.index(f.severity), f.hunt_id, f.session_id, f.ts or ""))
