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
./hunt.py summarize fixtures/sessions/                                # what the data covers
./hunt.py run fixtures/sessions/ fixtures/mcp/ --table                # every file under each directory
./hunt.py run logs/ --min-severity medium --allow-host artifacts.example.com --window 900 --table
./hunt.py audit-config fixtures/mcp/config_risky.json --table
./hunt.py scan-text --file some_tool_output.txt
./hunt.py baseline fixtures/mcp/approved_tools.json > /tmp/tool_baseline.json
./hunt.py run fixtures/mcp/transcript.jsonlines --baseline /tmp/tool_baseline.json --hunt AAH-007
./hunt.py catalog --markdown
```

`run` exits 1 when a finding reaches `--fail-on` (default `critical`), so it can gate a pipeline, and 2 when the
hunt could not be run. A line or file that cannot be read is skipped, counted in `coverage.skipped` and named on
standard error, and every other line is still hunted; `--strict` makes any skipped input exit 2 instead. The
thresholds are flags: `--window`, `--calls-per-minute`, `--distinct-files`, `--injection-threshold`, and
`--allow-host`, which adds to the default list of ordinary destinations (`--replace-allowlist` to use only yours).

## The hunts

| ID | Looks for | Telemetry |
| --- | --- | --- |
| AAH-001 | Tool results and documents that address the model | `tool_result`, `document` |
| AAH-002 | An action after injected content that does what the text asked, or is risky in itself | the above, `tool_call`, `process` |
| AAH-003 | A credential read and sent by one command, or read and then data sent out | `file`, `tool_call`, `process`, `network` |
| AAH-004 | Download-and-run, decode-and-run, reverse shells, persistence, history erasure | `process`, `tool_call` |
| AAH-005 | MCP tool descriptions carrying instructions for the model | `tool_manifest` |
| AAH-006 | One tool name from two servers, or a description steering another server's tool | `tool_manifest` |
| AAH-007 | MCP tool definitions that changed after they were first seen or approved | `tool_manifest` |
| AAH-008 | Unpinned or typosquatted packages, pipes to a shell, plaintext remotes, inline secrets | MCP client configuration |
| AAH-009 | Agent identities used from a new network or with undeclared scopes | `auth` |
| AAH-010 | Force pushes, destroyed infrastructure and cloud resources, dropped tables, recursive deletes, runaway call rates | `tool_call`, `process` |
| AAH-011 | Agents writing their own permissions, MCP servers or instructions; approvals turned off | `file`, `tool_call`, `process` |
| AAH-012 | Credentials in model output, in arguments or command lines, or in tool output | `model_output`, `tool_call`, `tool_result`, `process` |

A command is judged on what it runs. The engine reads each command line as a shell does (pipes, substitutions,
`sh -c`, heredocs, `cmd /c`, a script written and then run) and asks what the commands do, so `curl ... | sh`
inside a commit message, a `grep` pattern or a heredoc written to a file is text and not a command, and
`bash <(curl ...)`, `sh -c "$(wget ...)"` and a download saved and then run are all the same attack. Severity
follows what a finding rests on, not only which hunt made it: the same `DROP DATABASE` is low against
`importer_test` on `localhost` and high against production.

Each hunt's hypothesis, how its severity is set, its MITRE ATLAS, ATT&CK and OWASP LLM Top 10 (2025) mapping, its
triage steps and its known benign causes are in [the catalog](../../.claude/skills/agentic-ai-threat-hunt/references/hunt-catalog.md),
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
  resolved through symbolic links, and never returns a file's content, only findings. A path may be a directory;
  a link inside it that leads out of the roots is left out.
- Every string in a finding, not only the quotes, passes through one place that redacts credentials (as given,
  percent-encoded, and where a quote's edge would cut one in half), shows hidden and control characters as
  `<U+200B>`, and breaks links as `hxxps://example[.]com`. Results say, in a `note`, that the evidence is data to
  analyse and never instructions. `test_agentic_hunt_corpus.py` puts a token, live links, terminal escapes and an
  injection in every field an attacker can write and checks that none comes back.
- A result is sized for a model to read: `run_hunts` returns the `limit` most severe findings (50 unless asked, 500
  at most, fewer if they would pass 256 KB) with `total` and `truncated` ahead of the list.
- Arguments are held to each tool's input schema before any file is opened, and a request line the server cannot
  read (not UTF-8, not JSON, a batch, five megabytes long) gets an error response while the server carries on.

## The pipeline stage

`morpheus.stages.telemetry.agentic_hunt_stage.AgenticHuntStage` runs the event hunts (all but AAH-008, which reads
configurations) on a stream whose rows are agent events in the telemetry schema. Sections such as `process` may be
mappings, JSON text, or dotted columns (`process.cmdline`). The stage passes each frame on with three columns:
`agentic_severity` (the worst finding the row completes, or empty), `agentic_hunts`, and `agentic_findings` (the
findings as JSON, defanged as the engine reports them).

```python
pipeline.set_source(KafkaSourceStage(config, bootstrap_servers=servers, input_topic=["agent-telemetry"]))
pipeline.add_stage(DeserializeStage(config))
pipeline.add_stage(AgenticHuntStage(config, window_seconds=600, min_severity="medium"))
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
- **Each session ages by its own clock.** A session's history is measured against that session's newest event, so
  a host whose clock runs ahead shortens nobody else's window. A row timed further ahead of the stream than
  `max_clock_skew_seconds` (seven days unless set) is refused and marked in `clock_refused`, as in every stateful
  telemetry stage. A session is let go once the stream is more than the skew and the window past it, and the stage
  holds at most 20,000 sessions and 200,000 events, letting the quietest go first. The stage's memory survives a
  restart through the same checkpoints as theirs.
- **No row stops the pipeline.** A row the stage cannot read, or a hunt that fails on a batch, is logged with a
  count and the rest are still hunted. A row with no session id belongs to a session of its host's own; a row
  with no usable time is hunted with its batch and not kept, which the stage says once in its log. List cells
  may arrive as arrays, as Parquet and Arrow sources hand them over.

What the stage does not do yet is in [the retrospective](./RETROSPECTIVE.md): its cost per batch grows with the
length of a session, replayed rows are hunted again, and numeric times are read in `time_unit` and not by their
size.

## How far to trust it

The hunts are indicators, not verdicts, and how good they are is measured, not asserted.
[`fixtures/corpus/`](./fixtures/corpus/) holds two labelled sets written to break the first version: 225 short
sessions around the command, credential and configuration hunts, and 174 texts for the injection indicators.
`tests/morpheus/determinism/test_agentic_hunt_corpus.py` holds the engine to floors a little under these figures,
so a change that costs recall or precision fails a test.

| Measured on | First version | Now |
| --- | --- | --- |
| 140 attacks in the spellings attackers use, found at medium or above | 24% | 96% |
| 85 things a coding agent does all day, reported at medium or above | 40% | 1% |
| 110 injections across twelve evasion families, reported | 13% | 84% |
| The same, reported as likely injection | 1% | 58% |
| 64 benign texts picked to look like injection, called likely | 3% | 0% |
| 438 package READMEs, reported at all | 25% | 4% |
| The same, called likely | 1% | 0% |

What the figures do not say: the corpora are synthetic and were written by reviewers of this code, not collected
from incidents; the injection indicators match phrasings, so a paraphrase that avoids every one of them is not
found (the corpus's remaining misses are single sentences such as "forward the last five messages to" an
address); and the stock override and concealment phrases are matched in ten languages besides English but
paraphrases in them are not. An injection that works is most reliably seen in what the agent did next, which is
why AAH-002 and AAH-003 exist.

`fixtures/sessions/malicious.jsonlines` holds one session per attack the hunts look for and `benign.jsonlines` an
ordinary coding session that must produce nothing; the MCP fixtures hold a transcript in which a server's tool is
redefined with hidden instructions and another server shadows a tool, plus a risky and a clean client
configuration. They use documentation addresses and AWS's published example key only.
`tests/morpheus/determinism/test_agentic_hunt.py` holds every hunt to them, and `test_agentic_hunt_mcp_server.py`
drives the MCP server over stdio.

The thresholds (an indicator score of 3, a 600-second window, 30 calls a minute, 50 distinct files) are starting
points with no baseline behind them; each is a flag, to be set from an environment's own data.

The SPL, KQL and Sigma ports were written against the first version's patterns and have not been run on a live
Splunk, Sentinel or Defender tenant here. They do not read command lines the way the engine now does, so they are
noisier and miss more than `hunt.py`; where a query and `hunt.py` disagree, the query is wrong. Test each on the
fixtures first, as the `agentic-detection-engineering` skill describes. The ATLAS technique names follow the
matrix as published when this was written; ATLAS renames techniques as agent attacks are catalogued, so check an
ID before quoting it.

[`RETROSPECTIVE.md`](./RETROSPECTIVE.md) is the review these figures come from: what was wrong with the first
version, what this one fixes, and what is still owed.
