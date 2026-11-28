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

# Retrospective on the agentic AI hunts

A review of the first version of this example (the hunt engine, the command line, the MCP server, the four skills,
the SIEM ports and `AgenticHuntStage`), what it found, what this revision fixes, and what is still owed.

## Summary

The first version passed its tests and was wrong in ways its tests could not see. Its fixtures were written by
the same hand as its patterns, so each hunt found the one spelling of an attack it was written against. Measured on
inputs written to break it, it found 24% of the attacks, reported 40% of ordinary activity, and reported 13% of the
injections. It also broke two promises it made in writing: that no finding shows a credential, and that no input
can stall it.

This revision rewrites how commands and text are read, makes the two promises true and tests them on hostile
input, and stops one bad line or row from ending a hunt. On the same inputs it finds 96% of the attacks, reports 1%
of ordinary activity at medium or above, and reports 84% of the injections with no benign text called likely
injection. The inputs are now in the repository as a labelled corpus with floors under them.

It does not fix everything. The plan at the end lists what is left, in the order it should be done.

## How the review was done

Ten reviewers each took one part (the injection indicators, the command hunts, the engine's logic, the MCP audit,
the MCP server, the stage, the SIEM ports, the skills and documentation, the tests, and a run of the hunts over
real agent sessions), with the instruction to break it and to show the input that did. Every finding was then
given to a second reviewer told to refute it by running it. Of 120 findings, 89 were confirmed as stated, 31
confirmed in part with the claim or severity corrected, and none refuted. A last reviewer looked for what the ten
had not covered and added eight gaps.

The probes the reviewers wrote are the corpus in [`fixtures/corpus/`](./fixtures/corpus/). Everything in it is
synthetic: documentation addresses, AWS's published example key, and token-shaped values that are stored as
placeholders and built when the tests run.

## What held

- **One engine, four surfaces.** The command line, the MCP server, the stage and the skills all call the same
  functions, so a fix made once reached all of them. No surface had to be changed for a hunt's logic.
- **The MCP server's file boundary.** Paths outside the allowed roots were refused, through symbolic links too,
  and no reviewer got a file's content out of it.
- **The stage against the engine.** The test that streams the fixtures in batches of one, five and all, and
  requires what the engine finds reading them whole, caught regressions during this rewrite before anything else
  did.
- **The consistency checks.** The generated catalog page, the skills' links and the hunt tables are tested against
  the engine, and none had drifted.

## What was wrong

### The two written promises were false

- **Credentials reached the output.** Redaction ran on quoted evidence only. About thirty other fields (session
  ids, tool and server names, paths, hosts, details) were passed through as written, so a token in a tool name or
  a path came back whole. A quote cut from the middle of a text could end halfway through a key, which no pattern
  recognised. A key written with URL escapes (`%3DAKIA...`) was not recognised either, and only the first fifty credentials
  in a text were looked for.
- **Links and control characters reached the output** the same way, and only `http` links were broken.
- **Two patterns could be stalled.** One was quadratic and one cubic on input an attacker chooses. They were found
  by timing every pattern, which the first version did only for the two patterns it had already been bitten by.
  The same timing, run again during this revision, found a third that the review had missed, in the MCP audit,
  and a fourth in this revision's own draft.

### Detection matched one spelling

- **Risky commands (AAH-004)** found 11 of 40. `bash <(curl ...)`, `sh -c "$(wget ...)"`, a download saved and then
  run, PowerShell and `cmd /c` forms were all missed, while `curl ... | python -c 'import json...'` and a port
  check with `nc -z` were reported as critical.
- **Commands were matched as text**, so a commit message, a `grep` pattern or a here-document that mentioned an attack
  was reported as one.
- **Credential theft (AAH-003)** treated file names as hosts, any later network contact as data sent, and
  `.env.example` as a credential, and merged unrelated reads into one finding.
- **Destruction (AAH-010)** reported any recursive delete of an absolute path, including a build directory, and
  `--force-with-lease`, and missed deleted branches, cloud resources and unfiltered `DELETE`.
- **Guardrail tampering (AAH-011)** reported reads of a configuration file as writes and missed a write whose
  redirection had a space in it.
- **Action after injection (AAH-002)** reported every command that followed an injection, which for a coding agent
  is every injection, each at critical.
- **Injection indicators (AAH-001)** matched the stock English phrases. Of 110 realistic variants, 96 scored below
  the threshold, while a quarter of real package READMEs were reported because of their status badges.

### One bad input ended the run

- A line that was not JSON, an unknown event type or a field of the wrong shape raised out of the reader and
  ended the hunt with exit status 1, which is the status that means critical findings.
- Times were read in one format; anything else was an error.
- Events without a session id were all put in one session, so unrelated hosts were hunted as one conversation.
  Events without a time were moved to the start of their session, ahead of the content that caused them.
- A request line the MCP server could not decode ended the server. Its results had no size limit.
- In the stage, one row with an unexpected cell raised out of `on_data` and stopped the pipeline; list cells that
  arrived as arrays, which is how Parquet and Arrow hand them over, were dropped without a word; and one believed
  row a few minutes ahead, in any session, emptied every other session's history.

### The parts around the engine

- **MCP audit.** Only top-level `mcpServers` was read, so a user configuration with a dozen project servers was
  reported clean. Two well-known packages were reported as typosquats of each other. Any two servers that both
  offered `search` were a high finding.
- **Skills.** Triage steps told the reader to fetch a sample, disable a server and revoke a token. The reader of a
  skill is an agent, and an agent that does those things because of text in telemetry is the attack.
- **Tests.** Of 85 small changes made to detection logic, 68 left every test passing.
- **SIEM ports.** One SPL pattern was broken and the ports had drifted from the engine, with no test comparing
  them.

## What this revision changes

- **Commands are parsed.** [`shell.py`](../../python/morpheus/morpheus/utils/agentic_hunt/shell.py) reads a line
  as a shell does (quotes, pipes, lists, redirections, here-documents, substitutions, `sh -c`, `eval`, `ssh`, `cmd /c`,
  a script written and then run, Windows paths) and never raises.
  [`commands.py`](../../python/morpheus/morpheus/utils/agentic_hunt/commands.py) asks what the parsed commands do:
  what is risky, what is destroyed and how much it matters, which hosts are reached and whether data is sent,
  which credentials are read and whether the same line carries one out.
- **Severity follows the evidence.** A finding is graded by what it rests on (what was sent and where, what was
  destroyed, what was written), and the catalog says how for each hunt. Low findings are lines in a timeline.
- **AAH-002 needs a tie.** An action is reported only where it does what the injected text asked or is risky in
  itself.
- **Injection is judged on agreement.** Indicators carry a category, and text is called likely injection where two
  kinds of evidence agree or something was hidden. More phrasings, hidden markup, ten more languages for the stock
  phrases, more encodings, and more of what a renderer loads by itself are recognised.
- **One place makes output safe.** Every string in every finding passes through `Finding.to_dict`, which redacts,
  shows hidden and control characters, and breaks links of every scheme.
- **Input is tolerated and counted.** A line, file, row or request that cannot be read is skipped, counted and
  named, and never fatal. A hunt that fails is named and the others still report. Not being able to hunt exits 2.
- **The MCP server is bounded.** Results carry a limit, a total and a truncation flag; arguments are held to the
  schema; no request line stops it.
- **The stage** ages each session against its own clock, bounds its state, survives a bad row, reads array cells,
  and keeps rows without a session id apart by host.
- **The MCP audit** reads the layouts clients use, reports an entry it cannot read, and reports a shared tool name
  at low unless something else is wrong with it.
- **The skills** say to recommend and not to act, and the catalog's triage steps are worded that way.

## The rewrite was reviewed too

Before this revision was offered for merge, one more reviewer was given it cold, with the same instruction: break
it and show the input. It reproduced 29 defects, eight of them serious, and all are fixed here with a test each:

- **Redaction still depended on what stood around a credential.** A token straight after a terminal colour code
  or an escaped newline, a key inside another match, a key with its last line missing, `DB_PASSWORD=` and other
  prefixed names, and a token that a length limit cut in half all came through. Redaction now uses looser
  patterns than reporting does, merges overlapping matches, and never cuts a text inside a credential.
- **The stalls had moved from the patterns to the command reader.** A line of repeated substitutions cost memory by
  the square of its length, a long pipeline time by the cube, and a time field of spaces minutes. The test that
  times every pattern could not see these; a second test now times whole command lines at two sizes.
- **Ordinary shell shapes silenced the command hunts:** a newline after a pipe, a function body, a `case` arm,
  `$((1<<2))` read as a here-document, options written as one cluster, and options before a subcommand
  (`git -C dir push --force`). Any host whose name began with `bedrock.` was treated as a model endpoint.
- **Some benign commands were reported:** a dry run, `mkfs` on an image file, a script that imports both `socket`
  and `subprocess`, and any variable used after `source .env`.

The lesson is the first one again. The rewrite was measured on the corpus its author had been looking at for the
whole of the work, and a reader who had not seen that corpus found what it did not hold.

## Measured

`fixtures/corpus/commands.json` holds 140 attack sessions and 85 benign ones; `fixtures/corpus/texts.json` holds
110 injections in twelve evasion families and 64 benign texts picked to look like them. "First version" is the
engine as merged before this revision, run on the same files by the same script.

| Measured on | First version | Now |
| --- | --- | --- |
| Attacks found at medium or above | 33 of 140 (24%) | 134 of 140 (96%) |
| Credential read then egress (AAH-003) | 7 of 29 | 26 of 29 |
| Risky commands (AAH-004) | 11 of 40 | 39 of 40 |
| Destruction (AAH-010) | 7 of 29 | 28 of 29 |
| Guardrail tampering (AAH-011) | 5 of 27 | 26 of 27 |
| Credentials in output or arguments (AAH-012) | 3 of 15 | 15 of 15 |
| Benign sessions reported at medium or above | 34 of 85 (40%) | 1 of 85 (1%) |
| Injections reported | 14 of 110 (13%) | 92 of 110 (84%) |
| Injections called likely | 1 of 110 | 64 of 110 |
| Benign texts called likely | 2 of 64 | 0 of 64 |
| Benign texts reported at all | 27 of 64 | 18 of 64 |
| Package READMEs reported at all | 110 of 438 (25%) | 16 of 438 (4%) |
| Package READMEs called likely | 4 of 438 | 0 of 438 |

The one benign session still reported is an agent adding an allow rule to its own settings file, which is the
hypothesis of AAH-011 and is reported at medium on purpose.

The review also ran the hunts over ten real coding-agent sessions (1,076 events, not in the repository). The first
version raised 122 findings at medium or above, 76 of them critical, and the review found no attack among them.
This revision raises 39 and none critical: text about prompt injection that the sessions read, synthetic tokens
printed by tests, and one MCP server the agent was asked to register.

What these figures are not: a measure on real incidents. The corpus was written by reviewers of this code. It is
far harder than the fixtures and it is not the world.

## What is still owed

In the order it should be done. Each is one pull request.

1. **Injection recall, stated honestly.** The remaining misses are single sentences with one weak signal (a
   request to forward mail to an address, a URL with a slot in its path) and paraphrases that avoid every pattern.
   Patterns will not close that gap. The next step is to score a text together with what the agent did after
   reading it, and to say in every result that a clean scan is not a clean text.
2. **The MCP surfaces the audit does not read.** A server's `instructions` from `initialize`, its prompts and
   resources, and sampling requests reach the model as tool descriptions do and are not scanned. A launch line
   wrapped in another program is not checked for pinning. The inline-secret check is noisy.
3. **The stage as a stream.** Its cost per batch grows with the length of a session, because the session is hunted
   again with every batch. Replayed rows are hunted again and counted twice. A chain that grows without growing
   more severe is not written again. The learned tool baseline is global, so two releases of one server in a fleet
   produce a finding on almost every session start. Numeric times are read in `time_unit` and not by their size.
   The event clock cannot recover from a first row with a very old time; that is shared code and needs its own
   review.
4. **Converters and usability.** There is no converter from any agent product's transcript format or from an EDR
   export, so every hunter writes one. Findings have no stable identifier and there is no way to suppress a known
   benign one.
5. **SIEM ports generated from the engine.** The SPL, KQL and Sigma ports match command lines as text and have
   drifted. They should be generated from the same tables the engine uses, and tested against the corpus.
6. **Catalog names.** The ATLAS and ATT&CK technique names were written from the matrices as they stood and have
   not been checked against the current releases.
7. **Thresholds with data behind them.** The window, the call rate, the file count and the indicator score are
   flags now, and none has a baseline behind its default.

Smaller gaps the review found that are not yet closed: `git push` of a credential file to a new remote and a
`git reset --hard` followed by `git clean -fdx` are not reported, and the repository's copyright checker does not
accept the front matter of the skill files.

## What the work taught

- **A fixture written with the pattern proves the pattern compiles.** Every hunt passed on the fixtures and four
  of five found under a third of the labelled attacks. Detection needs inputs written by someone trying to get
  past it, kept in the repository, with floors.
- **A promise needs a place and a test.** "Evidence is defanged" was true of the fields that were remembered.
  It became true of all of them when one function was made the only way out, and a test put a token in every
  field an attacker can write.
- **Time every pattern, not the one that bit you.** All four stalls were found by a loop over every pattern and a
  list of hostile shapes, and none by reading the patterns. That loop is now a test.
- **Match the tool versions that gate the merge.** A documentation lint failed in CI after passing locally because
  the local linter was older and did not read front matter.
- **Review the review.** A third of the findings were corrected by the reviewer told to refute them, mostly in
  severity. Without that pass the plan would have been ordered wrongly.

## Reproducing the figures

```bash
python -m pytest tests/morpheus/determinism/test_agentic_hunt_corpus.py -q
```

runs the corpus and fails if recall or precision falls below the floors. The first version's figures come from the
same corpus run against the engine as it stood at the merge before this revision.
