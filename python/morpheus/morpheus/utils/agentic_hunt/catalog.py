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
The hunts, described: the hypothesis each tests, the telemetry it needs, how it maps to MITRE ATLAS, MITRE ATT&CK and
the OWASP Top 10 for LLM Applications (2025), and how to triage what it finds.

The mappings name technique IDs as those frameworks published them; ATLAS in particular renames techniques as agent
attacks are catalogued, so check an ID against the current matrix before quoting it in a report.
"""

import dataclasses

SEVERITIES = ("info", "low", "medium", "high", "critical")


@dataclasses.dataclass(frozen=True)
class Hunt:
    id: str
    title: str
    hypothesis: str
    data: tuple
    atlas: tuple
    attack: tuple
    owasp: tuple
    severity: str
    triage: tuple
    false_positives: tuple

    def to_dict(self) -> dict:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for (name, value) in dataclasses.asdict(self).items()
        }


CATALOG = (
    Hunt(
        "AAH-001",
        "Indirect prompt injection in content an agent read",
        "Content that came back from a tool, a fetched page or a retrieved document addresses the model: it tells it "
        "to ignore its instructions, hide what it does, call a tool, or send data somewhere.",
        ("tool_result", "document"), ("AML.T0051.001 LLM Prompt Injection: Indirect", ), ("T1204 User Execution", ),
        ("LLM01 Prompt Injection", ),
        "medium",
        ("Find where the content came from: the tool, its arguments, the URL or document behind it.",
         "Read what the agent did next in the same session; AAH-002 joins the two for you.",
         "Check whether other sessions read the same source, and whether the source is one an attacker can write to "
         "(an issue, a web page, a shared document, an email)."),
        ("Security research, red-team notes and documentation about prompt injection quote these phrases.",
         "Templates and chat transcripts can carry role markers.")),
    Hunt(
        "AAH-002",
        "A privileged action follows injected content",
        "Within one session, after a tool result carrying an injection, the agent runs a command, writes a file, "
        "sends data out or reads a credential: the confused deputy acting on the injection.",
        ("tool_result", "document", "tool_call", "process"),
        ("AML.T0051.001 LLM Prompt Injection: Indirect", "AML.T0053 LLM Plugin Compromise"),
        ("T1059 Command and Scripting Interpreter", "T1567 Exfiltration Over Web Service"),
        ("LLM01 Prompt Injection", "LLM06 Excessive Agency"),
        "high",
        ("Compare the action with the injected text: does it do what the text asked?",
         "Establish whether a person approved the action, and from which prompt.",
         "Contain the session's credentials if the action reached a network or a secret."),
        ("A user who pasted injection-like text and then asked for a command legitimately.", )),
    Hunt(
        "AAH-003",
        "A credential read followed by egress",
        "In one session an agent reads a credential store (SSH keys, cloud credentials, .env files, token caches, "
        "browser login data) and then sends something over the network.", ("file", "tool_call", "process", "network"),
        ("AML.T0057 LLM Data Leakage", ),
        ("T1552.001 Unsecured Credentials: Credentials In Files",
         "T1567 Exfiltration Over Web Service",
         "T1041 Exfiltration Over C2 Channel"), ("LLM02 Sensitive Information Disclosure", "LLM06 Excessive Agency"),
        "critical",
        ("Identify the destination host and whether it belongs to the organization.",
         "Rotate the credential that was read, whatever the destination turns out to be.",
         "Look back for the content that led the agent to the file (AAH-001)."),
        ("Deploy tooling that reads a cloud profile and then calls that cloud's API.", )),
    Hunt(
        "AAH-004",
        "Risky commands from an agent's process tree",
        "An agent, or a shell it started, downloads and runs code, decodes and runs a payload, opens a reverse "
        "shell, adds persistence, or erases its history.", ("process", "tool_call"),
        ("AML.T0053 LLM Plugin Compromise", ),
        ("T1059.004 Command and Scripting Interpreter: Unix Shell",
         "T1105 Ingress Tool Transfer",
         "T1027 Obfuscated Files or Information",
         "T1546.004 Unix Shell Configuration Modification",
         "T1053.003 Scheduled Task/Job: Cron",
         "T1070.003 Clear Command History"), ("LLM06 Excessive Agency", ),
        "high",
        ("Pull the full command line and the agent prompt or tool result that preceded it.",
         "Fetch whatever was downloaded and analyse it in isolation.",
         "Check the host for the persistence the command would have written."),
        ("Installers documented as `curl ... | sh` that a developer asked the agent to run.", )),
    Hunt(
        "AAH-005",
        "Poisoned MCP tool descriptions",
        "A tool an MCP server offers carries instructions for the model in its description or parameter "
        "descriptions: read a file, call another tool, hide something from the user, put data in a parameter.",
        ("tool_manifest", ), ("AML.T0010 ML Supply Chain Compromise", "AML.T0051.001 LLM Prompt Injection: Indirect"),
        ("T1195.002 Compromise Software Supply Chain", ), ("LLM03 Supply Chain", "LLM01 Prompt Injection"),
        "high",
        ("Record the server's package, version and source, and who added it to the client configuration.",
         "Search other sessions and hosts for the same server and tool name.",
         "Disable the server before reading further; the description acts on every session that loads it."),
        ("Descriptions that tell the model how to use the tool itself correctly, which score low.", )),
    Hunt(
        "AAH-006",
        "MCP tool shadowing",
        "Two servers in one session offer a tool of the same name, or one server's description steers how the model "
        "uses another server's tool, so a malicious server intercepts or rewrites a trusted one.", ("tool_manifest", ),
        ("AML.T0053 LLM Plugin Compromise", ), ("T1195.002 Compromise Software Supply Chain", ),
        ("LLM03 Supply Chain", "LLM06 Excessive Agency"),
        "high",
        ("List which server answered each call to the shadowed name.",
         "Diff the two descriptions; the shadowing one usually adds an instruction."),
        ("Two versions of the same server configured side by side during an upgrade.", )),
    Hunt(
        "AAH-007",
        "MCP tool definitions that changed after approval",
        "A server's tool description or input schema differs from the one seen before: the rug pull, where a tool "
        "is approved in a benign form and later redefined.", ("tool_manifest", ),
        ("AML.T0010 ML Supply Chain Compromise", ), ("T1195.002 Compromise Software Supply Chain", ),
        ("LLM03 Supply Chain", ),
        "medium",
        ("Diff the old and new definitions and scan the new text (AAH-005 does).",
         "Match the change to a release of the server; an unannounced change is the finding."),
        ("A legitimate release that reworded documentation.", )),
    Hunt(
        "AAH-008",
        "Risky MCP server configuration",
        "An MCP client configuration launches servers from unpinned packages, pipes downloads into a shell, talks "
        "to a remote server in plaintext, writes credentials into the file, grants a filesystem server the "
        "whole disk, or names a package one edit away from a well-known one.", ("config", ),
        ("AML.T0010 ML Supply Chain Compromise", ),
        ("T1195.002 Compromise Software Supply Chain", "T1552.001 Unsecured Credentials: Credentials In Files"),
        ("LLM03 Supply Chain", "LLM06 Excessive Agency"),
        "medium",
        ("Pin each server to a version or digest and record who owns it.",
         "Move credentials written into the file to the client's secret store, and rotate them."),
        ("Local development servers run from a checkout.", )),
    Hunt(
        "AAH-009",
        "Agent credentials used from somewhere new",
        "A token or service identity that an agent uses appears from a network it has not been seen from, or with "
        "a scope wider than the one declared for it.", ("auth", ), (),
        ("T1078.004 Valid Accounts: Cloud Accounts", "T1528 Steal Application Access Token"),
        ("LLM06 Excessive Agency", ),
        "high",
        ("Confirm whether the agent's host moved networks (a VPN, a new runner pool).",
         "Revoke the token if the source is not the agent's own infrastructure."),
        ("Agents moved between CI runner pools or cloud regions.", )),
    Hunt(
        "AAH-010",
        "Destructive or runaway agent activity",
        "An agent issues destructive operations (recursive deletes, force pushes, dropped tables, deleted buckets "
        "or namespaces) or acts at a rate or breadth no task needs.", ("tool_call", "process", "file"), (),
        ("T1485 Data Destruction", "T1119 Automated Collection"),
        ("LLM06 Excessive Agency", "LLM10 Unbounded Consumption"),
        "high",
        ("Stop the session, then establish what the operation reached.",
         "Read the prompt that started the run for scope the agent exceeded."),
        ("Cleanup jobs whose purpose is deletion.", "Large refactors that read many files.")),
    Hunt(
        "AAH-011",
        "Agents tampering with their own guardrails",
        "An agent writes to its own or another agent's configuration or standing instructions: widening "
        "permissions, approving tools automatically, adding an MCP server, or planting instructions in a file "
        "every future session reads.", ("file", "tool_call", "process"),
        ("AML.T0051.001 LLM Prompt Injection: Indirect", ), ("T1562.001 Impair Defenses: Disable or Modify Tools", ),
        ("LLM06 Excessive Agency", "LLM01 Prompt Injection"),
        "high",
        ("Diff the configuration or instruction file against its last reviewed version.",
         "Revert the change and look for the content that asked for it."),
        ("A developer asking the agent to update its own settings.", )),
    Hunt("AAH-012",
         "Credentials in what an agent says or sends",
         "A model output, or the arguments of a tool call that sends data out, contains a credential.",
         ("model_output", "tool_call"), ("AML.T0057 LLM Data Leakage", ),
         ("T1552 Unsecured Credentials", "T1567 Exfiltration Over Web Service"),
         ("LLM02 Sensitive Information Disclosure", "LLM07 System Prompt Leakage"),
         "high",
         ("Rotate the credential.", "Find how it entered the context: a file read, a tool result, the system prompt."),
         ("Documented example keys, which still deserve a look at why they were printed.", )),
)
"""Every hunt, in order. Each id is implemented once in `hunts.HUNTS`."""

BY_ID = {hunt.id: hunt for hunt in CATALOG}


def markdown() -> str:
    """The catalog as the reference page the threat-hunt skill links to."""
    lines = [
        "<!-- Generated by `examples/agentic_ai_hunting/hunt.py catalog --markdown`; do not edit by hand. -->",
        "",
        "# Agentic AI hunt catalog",
        "",
        "| ID | Hunt | Default severity | Telemetry | ATLAS | ATT&CK | OWASP LLM |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for hunt in CATALOG:
        cells = (hunt.id,
                 hunt.title,
                 hunt.severity,
                 ", ".join(f"`{kind}`" for kind in hunt.data),
                 "; ".join(hunt.atlas) or "-",
                 "; ".join(hunt.attack) or "-",
                 "; ".join(hunt.owasp))
        lines.append("| " + " | ".join(cells) + " |")

    for hunt in CATALOG:
        lines += ["", f"## {hunt.id}: {hunt.title}", "", hunt.hypothesis, "", "Triage:", ""]
        lines += [f"1. {step}" for step in hunt.triage]
        lines += ["", "Known benign causes:", ""]
        lines += [f"- {cause}" for cause in hunt.false_positives]

    return "\n".join(lines) + "\n"
