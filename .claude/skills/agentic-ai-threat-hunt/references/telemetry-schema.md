# Agent telemetry schema

The hunts in `examples/agentic_ai_hunting` read JSON lines, one event per line, in this shape. Every field is
optional except `type`; a hunt uses the fields its catalog entry names and ignores the rest. Map other sources onto
it with a small script.

## Common fields

| Field | Meaning |
| --- | --- |
| `ts` | ISO 8601 time (`2026-11-19T14:00:05Z`) or epoch seconds |
| `session_id` | One agent session or conversation; sequence hunts never join across sessions |
| `type` | `prompt`, `model_output`, `tool_call`, `tool_result`, `document`, `tool_manifest`, `process`, `file`, `network` or `auth` |
| `agent` | The agent product (`claude-code`, `cursor`, ...). A `process` event with `agent` set counts as the agent's |
| `host`, `principal` | The machine and the person or identity the agent runs as |
| `tool`, `server` | For tool events: the tool name and the MCP server offering it (empty for built-in tools) |
| `args` | For `tool_call`: the call's arguments as an object (`command`, `file_path`, `url`, `content`, ...) |
| `content` | For `prompt`, `model_output`, `tool_result`, `document`: the text |
| `tools` | For `tool_manifest`: the `tools` array of a `tools/list` result (`name`, `description`, `inputSchema`) |

## Sections

| Section | Fields | Used by |
| --- | --- | --- |
| `process` | `image`, `cmdline`, `parent_image`, `ancestors` (list of images), `agent_lineage` (true or false) | AAH-002, 003, 004, 010, 011 |
| `file` | `path`, `op` (`read`, `write`, `create`, `modify`, `append`, `delete`, `rename`), `content` (written text, optional) | AAH-003, 010, 011 |
| `network` | `dest_host`, `dest_ip`, `dest_port`, `bytes_out` | AAH-002, 003 |
| `auth` | `identity` or `token_id`, `identity_type` (`agent`, `service`, `workload`), `source_ip`, `source_asn`, `scopes`, `declared_scopes` | AAH-009 |

A process belongs to an agent when the event carries `agent`, `process.agent_lineage` is true, or `parent_image` or
`ancestors` names an agent or the editor hosting one (`claude`, `cursor`, `codex`, `copilot`, `windsurf`, `aider`,
`gemini`, `cline`, `code`).

## MCP JSON-RPC transcripts

A proxy or client log of MCP traffic is read directly, one message per line:

```json
{"ts": "2026-11-20T10:00:06Z", "session_id": "s-1", "server": "world-time", "message": {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}}
```

`tools/call` requests become `tool_call` events, their responses `tool_result` events (text and structured content
joined), and `tools/list` responses `tool_manifest` events. Responses are paired with requests by session, server
and id.

## Other files

- An MCP client configuration (`.mcp.json`, `claude_desktop_config.json`, `.cursor/mcp.json`, any JSON object with
  `mcpServers`) is audited by AAH-008.
- A captured `tools/list` result, `{"server": "name", "tools": [...]}`, is read as one manifest event.

## Mapping hints

- **Agent session logs** (transcripts an agent product writes): user turns are `prompt`, assistant text is
  `model_output`, each tool invocation is a `tool_call` with its input as `args`, and its output a `tool_result`.
- **EDR process events**: `process` with `image`, `cmdline` and `parent_image`; keep the session id if the agent
  sets one in the environment, otherwise use the agent process id plus host.
- **Proxy or firewall logs** from developer machines: `network`, attributed by process where the source allows.
- **Identity provider or cloud audit logs** for agent tokens: `auth`, with `declared_scopes` from the token's
  registration.
