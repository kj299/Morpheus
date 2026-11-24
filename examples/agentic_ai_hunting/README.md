<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Hunting malicious agentic AI behavior

Twelve hunts for what goes wrong when AI agents read untrusted content, call tools and run commands: indirect
prompt injection and the actions that follow it, credential theft, risky and destructive commands, poisoned,
shadowed and redefined MCP tools, risky MCP configurations, agent tokens used from new places, and agents disabling
their own guardrails. The hunts are written once, in `agentic_hunt/`, and reached three ways:

- **Threat hunters** run `hunt.py` from a shell, or ask Claude Code, which has four skills for this work in
  [`.claude/skills/`](../../.claude/skills/): `agentic-ai-threat-hunt`, `mcp-server-audit`,
  `prompt-injection-triage` and `agentic-detection-engineering`.
- **Agents** call the hunts as MCP tools through `mcp_server.py`, which the repository's `.mcp.json` registers for
  Claude Code sessions opened at the repository root.
- **SIEMs and EDRs** run the ports in [`queries/`](./queries/): Splunk SPL, KQL for Defender XDR and Sentinel, and
  Sigma rules.

Everything here needs only the Python standard library (3.10 or later), so it runs wherever the telemetry is.

```bash
cd examples/agentic_ai_hunting
./hunt.py summarize fixtures/sessions/*.jsonlines                     # what the data covers
./hunt.py run fixtures/sessions/*.jsonlines fixtures/mcp/transcript.jsonlines fixtures/mcp/config_risky.json --table
./hunt.py audit-config fixtures/mcp/config_risky.json --table
./hunt.py scan-text --text "$(cat some_tool_output.txt)"
./hunt.py baseline fixtures/mcp/approved_tools.json > /tmp/tool_baseline.json
./hunt.py run fixtures/mcp/transcript.jsonlines --baseline /tmp/tool_baseline.json --hunt AAH-007
./hunt.py catalog --markdown
```

`run` exits 1 when a finding reaches `--fail-on` (default `critical`), so it can gate a pipeline.

## The hunts

| ID | Looks for | Telemetry |
| --- | --- | --- |
| AAH-001 | Tool results and documents that address the model | `tool_result`, `document` |
| AAH-002 | A command, write, egress or credential read soon after injected content | the above, `tool_call`, `process` |
| AAH-003 | A credential file read followed by egress | `file`, `tool_call`, `process`, `network` |
| AAH-004 | Download-and-run, decode-and-run, reverse shells, persistence, history erasure | `process`, `tool_call` |
| AAH-005 | MCP tool descriptions carrying instructions for the model | `tool_manifest` |
| AAH-006 | One tool name from two servers, or a description steering another server's tool | `tool_manifest` |
| AAH-007 | MCP tool definitions that changed after they were first seen or approved | `tool_manifest` |
| AAH-008 | Unpinned or typosquatted packages, pipes to a shell, plaintext remotes, inline secrets | MCP client configuration |
| AAH-009 | Agent identities used from a new network or with undeclared scopes | `auth` |
| AAH-010 | Force pushes, destroyed infrastructure, dropped tables, recursive deletes, runaway call rates | `tool_call`, `process` |
| AAH-011 | Agents writing their own permissions, MCP servers or instructions; approvals turned off | `file`, `tool_call`, `process` |
| AAH-012 | Credentials in model output or in arguments sent out | `model_output`, `tool_call` |

Each hunt's hypothesis, its MITRE ATLAS, ATT&CK and OWASP LLM Top 10 (2025) mapping, its triage steps and its
known benign causes are in [the catalog](../../.claude/skills/agentic-ai-threat-hunt/references/hunt-catalog.md),
which `hunt.py catalog --markdown` writes from `agentic_hunt/catalog.py`. The input format is in
[the telemetry schema](../../.claude/skills/agentic-ai-threat-hunt/references/telemetry-schema.md): agent events
as JSON lines, MCP JSON-RPC transcripts as a proxy records them, MCP client configurations, and captured
`tools/list` results.

## The MCP server

```bash
claude mcp add agentic-hunt -- python3 examples/agentic_ai_hunting/mcp_server.py   # or use the repo's .mcp.json
```

It offers `list_hunts`, `explain_hunt`, `summarize_telemetry`, `run_hunts`, `audit_mcp_config` and `scan_text`,
over stdio, to any MCP client. Two properties matter because the server's caller is itself an agent that may be
under attack:

- It reads only files under `AGENTIC_HUNT_ROOTS` (directories separated by `:`; the working directory by default),
  resolved through symbolic links, and never returns a file's content, only findings.
- Every quote in a finding is defanged (hidden characters shown as `<U+200B>`, links broken as
  `hxxps://example[.]com`, cut short) and credentials are reported by kind and length only. Results say, in a
  `note`, that the evidence is data to analyse and never instructions.

## How far to trust it

The fixtures are synthetic. `fixtures/sessions/malicious.jsonlines` holds one session per attack the hunts look for
and `benign.jsonlines` an ordinary coding session that must produce nothing; the MCP fixtures hold a transcript in
which a server's tool is redefined with hidden instructions and another server shadows a tool, plus a risky and a
clean client configuration. They use documentation addresses and AWS's published example key only.
`tests/morpheus/determinism/test_agentic_hunt.py` holds every hunt to them, and
`test_agentic_hunt_mcp_server.py` drives the MCP server over stdio.

The hunts are indicators, not verdicts. Text about prompt injection quotes the phrases AAH-001 looks for; a
developer can ask an agent to run an installer piped to a shell. The catalog lists each hunt's known benign causes.
The thresholds (an indicator score of 3, a 600-second window, 30 calls a minute, 50 distinct files) are starting
points to tune against an environment's own baseline.

The SPL, KQL and Sigma ports are written against the same patterns but have not been run on a live Splunk, Sentinel
or Defender tenant here. Test each on the fixtures first, as the `agentic-detection-engineering` skill describes;
where a query and `hunt.py` disagree, the query is wrong. The ATLAS technique names follow the matrix as published
when this was written; ATLAS renames techniques as agent attacks are catalogued, so check an ID before quoting it.
