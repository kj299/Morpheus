# Agent telemetry schema

The hunts in `examples/agentic_ai_hunting` read JSON lines, one event per line, in this shape. Every field is
optional except `type`; a hunt uses the fields its catalog entry names and ignores the rest. Map other sources onto
it with a small script.

A line the reader cannot use (not JSON, not an object, no known `type`, nested past 64 levels, longer than 8 MB) is
skipped and counted, and the rest of the file is still read. `coverage.skipped` in every result is that count and
`coverage.skipped_examples` names the first few by file and line; `--strict` turns any of them into an error. A
field of the wrong shape is read as empty and not as an error. Files may be UTF-8 with or without a byte-order mark,
with either line ending; bytes that are not UTF-8 are replaced.

## Common fields

| Field | Meaning |
| --- | --- |
| `ts` | ISO 8601 time (`2026-11-19T14:00:05Z`, with or without a zone or fraction, `T` or a space) or an epoch number in seconds, milliseconds, microseconds or nanoseconds, told apart by size. A time that cannot be read leaves the event untimed: it keeps the place it was given in, and sequences that include it are bounded by position (200 events) and not by the window |
| `session_id` | One agent session or conversation; sequence hunts never join across sessions. An event without one is put in a session of its `host` (or of its file, for a transcript), never in one shared by every host; `coverage.without_session_id` counts them |
| `type` | `prompt`, `model_output`, `tool_call`, `tool_result`, `document`, `tool_manifest`, `process`, `file`, `network` or `auth` |
| `agent` | The agent product (`claude-code`, `cursor`, ...). A `process` event with `agent` set counts as the agent's |
| `host`, `principal` | The machine and the person or identity the agent runs as |
| `tool`, `server` | For tool events: the tool name and the MCP server offering it (empty for built-in tools) |
| `args` | For `tool_call`: the call's arguments as an object (`command`, `file_path`, `url`, `content`, ...), as JSON text of one, or as a list (read as `argv`) |
| `content` | For `prompt`, `model_output`, `tool_result`, `document`: the text, or content blocks (`[{"type": "text", "text": ...}]`) as agent frameworks log them |
| `tools` | For `tool_manifest`: the `tools` array of a `tools/list` result (`name`, `description`, `inputSchema`) |

## Sections

| Section | Fields | Used by |
| --- | --- | --- |
| `process` | `image`, `cmdline`, `parent_image`, `ancestors` (list of images), `agent_lineage` (true or false) | AAH-002, 003, 004, 010, 011, 012 |
| `file` | `path`, `op` (`read`, `write`, `create`, `modify`, `append`, `delete`, `rename`), `content` (written text, optional) | AAH-003, 010, 011 |
| `network` | `dest_host`, `dest_ip`, `dest_port`, `bytes_out` | AAH-002, 003 |
| `auth` | `identity` or `token_id`, `identity_type` (`agent`, `service`, `workload`), `source_ip`, `source_asn`, `scopes`, `declared_scopes` | AAH-009 |

A process belongs to an agent when the event carries `agent`, `process.agent_lineage` is true, or `parent_image` or
`ancestors` names an agent (`claude`, `cursor`, `codex`, `copilot`, `windsurf`, `aider`, `gemini`, `cline`, `goose`,
`amp`, `opencode`). An editor is not one: `code` hosts people as well as agents, so a process under it needs
`agent` or `agent_lineage` set by the collector.

## MCP JSON-RPC transcripts

A proxy or client log of MCP traffic is read directly, one message per line:

```json
{"ts": "2026-11-20T10:00:06Z", "session_id": "s-1", "server": "world-time", "message": {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}}
```

`tools/call` requests become `tool_call` events, their responses `tool_result` events (text and structured content
joined), and `tools/list` responses `tool_manifest` events. Responses are paired with requests by session, server
and id; two requests that reuse an id are answered in the order they were sent. What a server says outside those
three (the `instructions` of its `initialize` result, prompts, resources, sampling requests) is not read yet.

## Other files

- An MCP client configuration is audited by AAH-008, in any of the layouts clients use: top-level `mcpServers`
  (`.mcp.json`, `claude_desktop_config.json`, `.cursor/mcp.json`), `servers` (`.vscode/mcp.json`), `mcp.servers`
  (VS Code settings), `context_servers` (Zed), and `projects.<path>.mcpServers` (the Claude Code user
  configuration, whose servers are named `<path>:<name>`).
- A captured `tools/list` result, `{"server": "name", "tools": [...]}`, is read as one manifest event.
- A directory is read whole: every `.jsonl`, `.jsonlines`, `.ndjson` and `.json` file under it, in any letter case.

## Mapping hints

- **Agent session logs** (transcripts an agent product writes): user turns are `prompt`, assistant text is
  `model_output`, each tool invocation is a `tool_call` with its input as `args`, and its output a `tool_result`.
- **EDR process events**: `process` with `image`, `cmdline` and `parent_image`; keep the session id if the agent
  sets one in the environment, otherwise use the agent process id plus host.
- **Proxy or firewall logs** from developer machines: `network`, attributed by process where the source allows.
- **Identity provider or cloud audit logs** for agent tokens: `auth`, with `declared_scopes` from the token's
  registration.
