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
their own guardrails. The hunts are written once, in `morpheus.utils.agentic_hunt`
([`python/morpheus/morpheus/utils/agentic_hunt/`](../../python/morpheus/morpheus/utils/agentic_hunt/)), and reached
four ways:

- **Threat hunters** run `hunt.py` from a shell, or ask Claude Code, which has four skills for this work in
  [`.claude/skills/`](../../.claude/skills/): `agentic-ai-threat-hunt`, `mcp-server-audit`,
  `prompt-injection-triage` and `agentic-detection-engineering`.
- **Agents** call the hunts as MCP tools through `mcp_server.py`, which the repository's `.mcp.json` registers for
  Claude Code sessions opened at the repository root.
- **SIEMs and EDRs** run the ports in [`queries/`](./queries/): Splunk SPL, KQL for Defender XDR and Sentinel, and
  Sigma rules.
- **Morpheus pipelines** run them on a stream with `AgenticHuntStage` (`agentic-hunt` on the command line); see
  [the stage](#the-pipeline-stage) below.

The engine, `hunt.py` and `mcp_server.py` need only the Python standard library (3.10 or later), so they run
wherever the telemetry is; the scripts read the engine from this checkout, with or without Morpheus installed.

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
which `hunt.py catalog --markdown` writes from `morpheus.utils.agentic_hunt.catalog`. The input format is in
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

## The pipeline stage

`morpheus.stages.telemetry.agentic_hunt_stage.AgenticHuntStage` runs the event hunts (all but AAH-008, which reads
configurations) on a stream whose rows are agent events in the telemetry schema. Sections such as `process` may be
mappings, JSON text, or dotted columns (`process.cmdline`). The stage passes each frame on with three columns:
`agentic_severity` (the worst finding the row completes, or empty), `agentic_hunts`, and `agentic_findings` (the
findings as JSON, defanged as the engine reports them).

```python
pipeline.set_source(KafkaSourceStage(config, bootstrap_servers=servers, input_topic=["agent-telemetry"]))
pipeline.add_stage(DeserializeStage(config))
pipeline.add_stage(AgenticHuntStage(config, window_seconds=600, egress_allowlist=["github.com"],
                                    min_severity="medium"))
pipeline.add_stage(SerializeStage(config))
```

A stream is not a file read whole, and three things follow from that:

- **Chains across batches.** Each session keeps its events for `window_seconds` (at least a minute), so an
  injection in one batch and the command it caused in the next are joined. What the hunts compare against
  history, MCP tool definitions (AAH-007) and the networks each agent identity has used (AAH-009), is kept for
  the life of the stage, and `tool_baseline` seeds the first with approved definitions.
- **Written once, when complete.** A finding is written on the newest row it rests on, when it first completes,
  and again only if it grows more severe: an injection followed by a command is written at high, and written at
  critical if egress follows. `tests/morpheus/stages/test_agentic_hunt_stage.py` holds the stream, in batches of
  one, five and the whole file, to what the engine finds reading each fixture whole.
- **Guarded event time.** A row timed too far ahead of the stream is refused and marked in `clock_refused`, as in
  every stateful telemetry stage, so a forged time cannot push a session's history out of its window. The stage's
  memory survives a restart through the same checkpoints as theirs.

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
