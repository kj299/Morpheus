---
name: agentic-detection-engineering
description: Turn the agentic AI hunts into scheduled detections for a SIEM or EDR (Splunk SPL, Microsoft Sentinel or Defender KQL, Sigma), tune them to an environment, and test them against the repository's malicious and benign fixtures. Use it when asked to write, port, tune or test detections for malicious AI agent behavior, MCP abuse or prompt injection.
---

# Agentic detection engineering

The hunts in `examples/agentic_ai_hunting` come with queries for three platforms in
`examples/agentic_ai_hunting/queries/`:

- `splunk.spl`: one search per hunt over the normalized agent event schema indexed as JSON;
- `kql.kql`: Defender XDR `Device*` tables for what an endpoint sensor sees (AAH-003, 004, 010, 011), and a custom
  `AgentTelemetry_CL` table in the normalized schema for the rest;
- `sigma/`: process and file rules for an EDR (AAH-003, 004, 010, 011).

Where the telemetry already flows through a Morpheus pipeline, `AgenticHuntStage` runs the engine itself on the
stream, with no port to keep in step; its output columns can feed the SIEM directly. Prefer it to a port when
the pipeline exists.

The Python engine is the reference implementation. A query that disagrees with `hunt.py` on the fixtures is the one
that is wrong.

## Workflow

1. **Pick the hunt and the platform.** Read its catalog entry (`hunt.py catalog --hunt AAH-004` or `explain_hunt`)
   for the hypothesis, the data it needs and its benign causes.
2. **Check the data exists on the platform.** Name the index or table and fields the query reads; if the
   environment's agent telemetry is in another shape, write the field mapping first (the schema is in the
   threat-hunt skill's `references/telemetry-schema.md`).
3. **Adapt the query** from `queries/`, keeping the hunt ID in its title or comment. Tune what the catalog marks as
   tunable: the agent process names, the egress allow list, the time window, the call-rate threshold.
4. **Test it on the fixtures.** Load `examples/agentic_ai_hunting/fixtures/sessions/malicious.jsonlines` and
   `benign.jsonlines` into a test index or table, run the query, and compare with
   `hunt.py run <the same files> --hunt <id>`: every engine finding should appear, and the benign file should give
   nothing. Add a fixture line for each new behavior the query covers, and the matching assertion in
   `tests/morpheus/determinism/test_agentic_hunt.py`, so the engine and the query keep meaning the same thing.
5. **Set severity and response** from the hunt's default severity and triage steps; record the known benign causes
   as the detection's documented false positives.
6. **Write it down**: the query, the data it needs, the tuning applied, the fixture results, and the owner.

## Rules

- Detection content quotes attacker phrasing (injection text, reverse-shell shapes). Keep it in query files and
  fixtures; do not paste live attacker payloads collected from production into the repository; write a synthetic
  equivalent.
- Fixtures use documentation addresses (RFC 5737, `example.com`) and documented example keys only, never a real
  credential, even a revoked one.
- Sigma rules keep `status: experimental` until they have run against production data for a while.
