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

Triage steps are written for the analyst who owns the response. Where a step changes something (rotating a
credential, disabling a server, stopping a session) it is worded as a recommendation to make, because the reader may
be an agent working a hunt, and an agent that revokes and disables on the strength of text it read in telemetry is
the confused deputy these hunts exist to find.
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
    grading: str = ""
    """How a finding's severity is set, where it is not always the hunt's default."""

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
         "Templates and chat transcripts can carry role markers.",
         "Agent instruction files (CLAUDE.md, AGENTS.md) address a model on purpose."),
        "High where two kinds of evidence agree and the score is 6 or more, or text was hidden from a reader; medium "
        "where two kinds agree; low where there is one kind only (an override phrase and nothing else, a link with a "
        "slot for data and nothing else). The kinds are: addressing or redirecting the model, asking for secrecy, "
        "naming a credential, naming a way out, and hiding text."),
    Hunt(
        "AAH-002",
        "A privileged action follows injected content",
        "Within one session, after a tool result carrying an injection, the agent does what the text asked (reaches "
        "a host it named, reads a credential it named, calls a tool it named) or does something another hunt "
        "reports. An action that merely follows is not reported: a coding agent runs a command after nearly every "
        "tool result.", ("tool_result", "document", "tool_call", "process"),
        ("AML.T0051.001 LLM Prompt Injection: Indirect", "AML.T0053 LLM Plugin Compromise"),
        ("T1059 Command and Scripting Interpreter", "T1567 Exfiltration Over Web Service"),
        ("LLM01 Prompt Injection", "LLM06 Excessive Agency"),
        "high",
        ("Compare the action with the injected text: does it do what the text asked?",
         "Establish whether a person approved the action, and from which prompt.",
         "If the action reached a network or a secret, recommend that the owner rotate the session's credentials."),
        ("A user who pasted injection-like text and then asked for a command legitimately.", ),
        "Critical where an action reads a credential or reaches a host the text named, sends a credential, or opens "
        "a reverse shell; high otherwise. One finding per injected text, listing the actions tied to it."),
    Hunt(
        "AAH-003",
        "A credential read followed by egress",
        "In one session an agent reads a credential store (SSH keys, cloud credentials, .env files, token caches, "
        "browser login data, a command that prints a token) and then sends data over the network; or one command "
        "does both.", ("file", "tool_call", "process", "network"), ("AML.T0057 LLM Data Leakage", ),
        ("T1552.001 Unsecured Credentials: Credentials In Files",
         "T1567 Exfiltration Over Web Service",
         "T1041 Exfiltration Over C2 Channel"), ("LLM02 Sensitive Information Disclosure", "LLM06 Excessive Agency"),
        "critical",
        ("Identify the destination host and whether it belongs to the organization.",
         "Recommend rotating the credential that was read, whatever the destination turns out to be.",
         "Look back for the content that led the agent to the file (AAH-001)."),
        ("Deploy tooling that reads a cloud profile and then calls that cloud's API.",
         "A deliberate upload of a configuration file to the organization's own bucket."),
        "Critical where one command reads the credential and carries it out, or a later request visibly carries a "
        "credential; high where data is later sent to a host that is not on the allowlist, or anything goes to a "
        "known collection host; medium where data is sent to an allowlisted host, or a connection is seen to an "
        "unlisted one. A page fetched after a credential was read is not reported."),
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
         "Record the address of whatever was downloaded and hand it to malware analysis; do not fetch it from the "
         "machine the hunt runs on.",
         "Check the host for the persistence the command would have written."),
        ("Installers documented as `curl ... | sh` that a developer asked the agent to run.",
         "A tool's own initialiser added to a shell profile, which is reported at low."),
        "Critical for a reverse shell; high for download and run, decode and run, persistence and erased history; "
        "low for a unit that exists being enabled and for a `PATH` entry, alias or well-known initialiser added to "
        "a shell profile. A command is judged on what it runs: text that is only written, searched for or "
        "committed is not a command."),
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
         "Recommend that the server be disabled wherever it is configured: its description acts on every session "
         "that loads it, a hunter's own agent included."),
        ("Descriptions that tell the model how to use the tool itself correctly: \"call `resolve` first\" is one "
         "kind of evidence and is not reported alone.", ),
        "Reported where two kinds of evidence agree (a directive about other tools or data, a request for secrecy, a "
        "credential named, a way out named) or anything is hidden. Critical at a score of 6 or more, high below."),
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
        ("Two versions of the same server configured side by side during an upgrade.",
         "Unrelated servers that both offer `search` or `get`."),
        "High where one of the definitions carries an injection indicator or names another server's tools; low for "
        "a shared name with nothing else, which is common and is reported once per set of servers."),
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
        ("Recommend pinning each server to a version or digest, and record who owns it.",
         "Recommend moving credentials written into the file to the client's secret store and rotating them."),
        ("Local development servers run from a checkout.", ),
        "Each check carries its own severity: high for a download piped to a shell, a plaintext remote, a "
        "credential in the file and a likely typosquat; medium for unpinned packages and images and whole-disk "
        "filesystem access; low for tools approved automatically."),
    Hunt(
        "AAH-009",
        "Agent credentials used from somewhere new",
        "A token or service identity that an agent uses appears from a network it has not been seen from, or with "
        "a scope wider than the one declared for it.", ("auth", ), (),
        ("T1078.004 Valid Accounts: Cloud Accounts", "T1528 Steal Application Access Token"),
        ("LLM06 Excessive Agency", ),
        "high",
        ("Confirm whether the agent's host moved networks (a VPN, a new runner pool).",
         "If the source is not the agent's own infrastructure, recommend that the token's owner revoke it."),
        ("Agents moved between CI runner pools or cloud regions.", )),
    Hunt(
        "AAH-010",
        "Destructive or runaway agent activity",
        "An agent issues destructive operations (recursive deletes of a root or home, force pushes and deleted "
        "branches, dropped tables and unfiltered deletes, deleted buckets, clusters and cloud resources) or acts at "
        "a rate or breadth no task needs.", ("tool_call", "process", "file"), (),
        ("T1485 Data Destruction", "T1119 Automated Collection"),
        ("LLM06 Excessive Agency", "LLM10 Unbounded Consumption"),
        "high",
        ("Establish what the operation reached, and recommend stopping the session if it is still running.",
         "Read the prompt that started the run for scope the agent exceeded."),
        ("Cleanup jobs whose purpose is deletion.",
         "Large refactors that read many files.",
         "A test's teardown of its own database or namespace, which is reported at low."),
        "By what is destroyed. High where the command names production or a protected branch, and for a root, a "
        "home or a system directory; low where it names a test, development or scratch target; otherwise the "
        "operation's own weight (high for a database, a namespace, infrastructure; medium for one workload or an "
        "unprotected branch). A build directory removed, `--force-with-lease`, `terraform plan -destroy` and a "
        "`find` with a name filter are not reported. Bursts and sweeps are medium."),
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
         "Recommend reverting the change, and look for the content that asked for it."),
        ("A developer asking the agent to update its own settings.",
         "`/init` and documentation edits to an instructions file, which are reported at low."),
        "By what was written. High where it turns approvals off, adds servers or hooks, tells the agent to stop "
        "asking or to keep something from the user, or carries injection indicators; medium for another change to a "
        "file that holds permissions, for content the telemetry does not show, for an MCP server registered from "
        "the command line and for an agent started with approvals off; low for ordinary text in an instructions "
        "file. Reading these files is not reported, and an editor's settings file counts only in the keys that "
        "govern agents."),
    Hunt(
        "AAH-012",
        "Credentials in what an agent says, sends or is shown",
        "A credential appears in a model output, in the arguments or command line of a call, or in what a tool "
        "returned, which is the model's context and the provider's logs from then on.",
        ("model_output", "tool_call", "tool_result", "process"), ("AML.T0057 LLM Data Leakage", ),
        ("T1552 Unsecured Credentials", "T1567 Exfiltration Over Web Service"),
        ("LLM02 Sensitive Information Disclosure", "LLM07 System Prompt Leakage"),
        "high",
        ("Recommend that the credential's owner rotate it.",
         "Find how it entered the context: a file read, a tool result, the system prompt."),
        ("Documented example keys, which are reported at low and still deserve a look at why they were printed.", ),
        "Critical where the call that carries the credential also sends to the network; high in model output; "
        "medium in arguments that stay on the machine and in tool output; low for a value a vendor publishes as "
        "an example. Values are looked for as given, URL-decoded and inside base64."),
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
        lines += ["", f"## {hunt.id}: {hunt.title}", "", hunt.hypothesis]

        if (hunt.grading):
            lines += ["", f"Severity: {hunt.grading}"]

        lines += ["", "Triage:", ""]
        lines += [f"1. {step}" for step in hunt.triage]
        lines += ["", "Known benign causes:", ""]
        lines += [f"- {cause}" for cause in hunt.false_positives]

    return "\n".join(lines) + "\n"
