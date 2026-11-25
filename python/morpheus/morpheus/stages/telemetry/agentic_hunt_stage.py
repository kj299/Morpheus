# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The agentic AI hunts of `morpheus.utils.agentic_hunt`, run on a stream of agent telemetry.

Each row is one agent event in the schema the hunts read (`.claude/skills/agentic-ai-threat-hunt/references/
telemetry-schema.md`): a `type`, a `session_id`, a time, and the sections an event carries (`args`, `process`,
`file`, `network`, `auth`, `tools`), each a mapping, a JSON string, or flattened into dotted columns such as
`process.cmdline`. The stage writes three columns and passes the frame on:

- `agentic_severity`: the worst finding the row completes, or empty;
- `agentic_hunts`: the hunts it completes, comma-separated;
- `agentic_findings`: those findings as a JSON list, with evidence defanged and credentials redacted as the
  engine reports them.

A finding is written on the newest row of the batch that it rests on, and only once, unless it later grows more
severe. A chain whose first event
came in an earlier batch is still found: the stage keeps each session's events for the hunts' window (at least a
minute, the call-rate hunt's), and feeds what earlier batches taught it, the MCP tool definitions seen and the
networks each agent identity was seen from, to the hunts that compare against history (AAH-007, AAH-009). Hunt
AAH-008 reads MCP client configurations, not a stream, and is refused here; `examples/agentic_ai_hunting/hunt.py
audit-config` runs it.

Event time is guarded like every stateful telemetry stage's: a row from too far ahead of the stream is refused,
marked in `clock_refused`, hunted on its own and never kept, so that one forged time cannot empty a session's
history. All the stage's memory is in `_checkpoint_attributes` and survives a restart.

The quoted evidence comes from telemetry an attacker may have written. It is data for an analyst or a SIEM, never
an instruction to a model that reads the column.
"""

import datetime
import json
import logging
import typing

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.pipeline.checkpoint_mixin import CheckpointMixin
from morpheus.stages.telemetry.telemetry_base import TelemetryStage
from morpheus.utils.agentic_hunt import catalog
from morpheus.utils.agentic_hunt import events
from morpheus.utils.agentic_hunt import hunts
from morpheus.utils.agentic_hunt import mcp_audit
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_frame
from morpheus.utils.event_clock import DEFAULT_MAX_SKEW_SECONDS

logger = logging.getLogger(__name__)

SEVERITY_COLUMN = "agentic_severity"
HUNTS_COLUMN = "agentic_hunts"
FINDINGS_COLUMN = "agentic_findings"

SCALAR_FIELDS = ("session_id", "agent", "host", "principal", "type", "tool", "server", "content")
SECTION_FIELDS = ("args", "process", "file", "network", "auth")
CONFIGURATION_HUNTS = ("AAH-008", )
"""Hunts that read MCP client configurations rather than events, which a stream does not carry."""

MAX_RETAINED_PER_SESSION = 5000
"""The most events one session keeps between batches, so a runaway session cannot grow the stage without bound."""

MIN_RETENTION_SECONDS = 60
"""The call-rate hunt counts calls within a minute, so a session keeps at least a minute of events."""

NS_PER_SECOND = 1_000_000_000


def _blank(value) -> bool:
    if (value is None):
        return True

    try:
        # NaN, NaT and pandas' NA are the only values not equal to themselves.
        return value != value  # pylint: disable=comparison-with-itself
    except (TypeError, ValueError):
        return False


def _section(value) -> typing.Optional[typing.Any]:
    """A section as a mapping or list: given as one, or as JSON text."""
    if (_blank(value)):
        return None

    if (isinstance(value, (dict, list))):
        return value

    if (isinstance(value, str) and value.strip()[:1] in ("{", "[")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    return None


def _plain(value):
    """A frame cell as a plain Python value, for a record that has to survive a JSON round trip."""
    if (_blank(value)):
        return None

    if (hasattr(value, "item") and not isinstance(value, (str, bytes))):
        return value.item()

    return value


def _iso(ns: typing.Optional[int]) -> typing.Optional[str]:
    if (ns is None):
        return None

    seconds, remainder = divmod(int(ns), NS_PER_SECOND)

    try:
        moment = datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc)
    except (OverflowError, OSError, ValueError):
        # A time no calendar holds, such as nanoseconds read as microseconds, is a row without a time.
        return None

    return moment.replace(microsecond=remainder // 1000).isoformat()


def finding_key(finding: dict) -> str:
    """
    What makes two findings the same finding, across the batches that rediscover it.

    The hunt, the session, the event it starts from, which of the hunt's kinds it is, and its severity. A rate or
    breadth finding (calls a minute, files a session) is one per session, since its first event moves as the window
    slides. Severity is part of it so that a chain written when it first completes is written again when it grows
    worse: an injection followed by a command, and later by egress, is reported at high and then at critical.
    """
    details = finding.get("details") or {}
    aggregate = "calls" in details or "files" in details
    first = "" if aggregate else (finding.get("refs") or [""])[0]
    kind = ",".join(sorted(details)) + "|" + str(details.get("tool", ""))

    return "|".join((finding["hunt_id"], finding.get("session_id") or "", first, kind, finding["severity"]))


@register_stage("agentic-hunt", ignore_args=["tool_baseline"])
class AgenticHuntStage(CheckpointMixin, TelemetryStage):
    """
    Run the agentic AI hunts on agent telemetry and write each row's findings.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    hunt_ids : list of str, default = ()
        The hunts to run; every event hunt when empty. AAH-008 reads configurations and is refused.
    window_seconds : int, default = 600
        How long a sequence (injected content then an action, a credential read then egress) may span, and so how
        long each session's events are kept between batches.
    egress_allowlist : list of str, default = `morpheus.utils.agentic_hunt.hunts.DEFAULT_ALLOWLIST`
        Destinations that are ordinary here. Egress to them still counts, at a lower severity.
    min_severity : str, default = "info"
        The least severe finding written.
    tool_baseline : dict, optional
        Approved MCP tool definitions, `{server: {tool: digest}}` as `hunt.py baseline` writes it, for AAH-007.
        Definitions the stream shows are added to it as they arrive.
    time_column : str, default = "ts"
        The event-time column: ISO 8601 text, a timestamp, or numbers in `time_unit`.
    time_unit : str, default = "ns"
        Unit for a numeric `time_column`.
    max_clock_skew_seconds : int, default = 7 days
        How far ahead of the stream's progress a row's time may be before it is refused.
    """

    _checkpoint_attributes = ("_clock", "_retained", "_reported", "_tool_baseline", "_auth_sources", "_batches")

    def __init__(self,
                 c: Config,
                 hunt_ids: typing.Sequence[str] = (),
                 window_seconds: int = 600,
                 egress_allowlist: typing.Sequence[str] = hunts.DEFAULT_ALLOWLIST,
                 min_severity: str = "info",
                 tool_baseline: typing.Optional[dict] = None,
                 time_column: str = "ts",
                 time_unit: str = "ns",
                 max_clock_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS):
        super().__init__(c, max_clock_skew_seconds=max_clock_skew_seconds)

        chosen = list(hunt_ids) if hunt_ids else list(hunts.HUNTS)
        unknown = sorted(set(chosen) - set(catalog.BY_ID))
        configuration = sorted(set(chosen) & set(CONFIGURATION_HUNTS))

        if (unknown):
            raise ValueError(f"no such hunt: {', '.join(unknown)}")

        if (configuration):
            raise ValueError(f"{', '.join(configuration)} reads MCP client configurations, not a stream; run it with "
                             "examples/agentic_ai_hunting/hunt.py audit-config")

        if (min_severity not in catalog.SEVERITIES):
            raise ValueError(f"min_severity must be one of {', '.join(catalog.SEVERITIES)}")

        if (window_seconds <= 0):
            raise ValueError("window_seconds must be positive")

        self._hunt_ids = chosen
        self._window_seconds = int(window_seconds)
        self._egress_allowlist = tuple(host.lower() for host in egress_allowlist)
        self._min_severity = min_severity
        self._time_column = time_column
        self._time_unit = time_unit

        self._retained: dict = {}
        """Each session's recent events, as the plain records they were read into, oldest first."""
        self._reported: dict = {}
        """Each session's findings already written, by `finding_key`."""
        self._tool_baseline: dict = {str(server): dict(tools) for (server, tools) in (tool_baseline or {}).items()}
        self._auth_sources: dict = {}
        """The networks each agent identity has been seen from."""
        self._batches = 0

        for column in (SEVERITY_COLUMN, HUNTS_COLUMN, FINDINGS_COLUMN):
            self._needed_columns[column] = TypeId.STRING

    @property
    def name(self) -> str:
        """Stage name."""
        return "agentic-hunt"

    def _required_columns(self, df) -> typing.Iterable[str]:
        del df

        return ("type", "session_id")

    def _records(self, host, times: list, refused: list) -> list:
        """
        One plain record per row, in the engine's input shape, each naming its batch and its place in it.

        Believed rows are numbered among themselves and refused rows apart, so that a refused row inserted into a
        batch renumbers nothing a finding about the other rows refers to.
        """
        columns = list(host.columns)
        dotted = [column for column in columns if "." in column and column.split(".", 1)[0] in SECTION_FIELDS]
        records = []
        counts = {False: 0, True: 0}

        for (position, row) in enumerate(host.itertuples(index=False, name=None)):
            cells = dict(zip(columns, row))
            kind = bool(refused[position])
            ref = f"{self._batches}:{'refused:' if kind else ''}{counts[kind]}"
            counts[kind] += 1
            record = {"ref": ref, "ts": _iso(times[position])}

            for field in SCALAR_FIELDS:
                value = _plain(cells.get(field))
                record[field] = None if value is None else str(value)

            for field in (*SECTION_FIELDS, "tools"):
                value = _section(cells.get(field))
                record[field] = json.loads(json.dumps(value, default=str)) if value is not None else None

            for column in dotted:
                (section, key) = column.split(".", 1)
                value = _plain(cells[column])

                if (value is not None):
                    record[section] = {**(record[section] or {}), key: value}

            records.append(record)

        return records

    def _context(self) -> hunts.HuntContext:
        return hunts.HuntContext(
            window_seconds=self._window_seconds,
            egress_allowlist=self._egress_allowlist,
            tool_baseline={server: dict(tools)
                           for (server, tools) in self._tool_baseline.items()},
            auth_baseline={identity: set(sources)
                           for (identity, sources) in self._auth_sources.items()})

    def _learn(self, normalized: list) -> None:
        """Add this batch's tool definitions and agent identities' sources to what later batches compare against."""
        for event in normalized:
            if (event["type"] == "tool_manifest"):
                tools = [tool for tool in event["tools"] if isinstance(tool, dict)]
                self._tool_baseline.setdefault(event["server"], {}).update(
                    mcp_audit.baseline_of({event["server"]: tools})[event["server"]])
            elif (event["type"] == "auth"):
                auth = event["auth"]
                identity = str(auth.get("identity") or auth.get("token_id") or "")
                source = str(auth.get("source_asn") or auth.get("source_ip") or "")

                if (identity and source
                        and (event["agent"] or auth.get("identity_type") in ("agent", "service", "workload"))):
                    known = self._auth_sources.setdefault(identity, [])

                    if (source not in known):
                        known.append(source)
                        known.sort()

    def _retain(self, records: list) -> None:
        """Keep each session's events for the window; let a session go once nothing of it is left."""
        for record in records:
            if (record["ts"] is not None):
                self._retained.setdefault(record["session_id"] or "", []).append(record)

        horizon_s = max(self._window_seconds, MIN_RETENTION_SECONDS)
        latest = max((events.parse_time(record["ts"]) for kept in self._retained.values() for record in kept),
                     default=None)

        for session in list(self._retained):
            kept = [
                record for record in self._retained[session]
                if (latest - events.parse_time(record["ts"])).total_seconds() <= horizon_s
            ]
            self._retained[session] = kept[-MAX_RETAINED_PER_SESSION:]

            if (not self._retained[session]):
                del self._retained[session]
                self._reported.pop(session, None)

    def _process(self, df) -> None:
        host = to_host_frame(df)
        (times, refused) = (self._guarded_times(df, self._time_column, self._time_unit)
                            if self._time_column in host.columns else ([None] * len(host), [False] * len(host)))
        self._refuse(df, refused, logger=logger, what="agent events", outcome="hunted alone and not kept")

        records = self._records(host, times, refused)
        current = {}
        believed = []
        refused_events = []
        skipped = 0

        for (position, record) in enumerate(records):
            try:
                event = events.normalize(record, record["ref"])
            except events.TelemetryError:
                skipped += 1
                continue

            current[record["ref"]] = position
            (refused_events if refused[position] else believed).append(event)

        if (skipped):
            logger.warning("AgenticHuntStage skipped %d row(s) whose type is not one of %s",
                           skipped,
                           ", ".join(events.EVENT_TYPES))

        sessions = {event["session_id"] for event in believed}
        earlier = [
            events.normalize(record, record["ref"]) for session in sorted(sessions)
            for record in self._retained.get(session, [])
        ]
        found = hunts.run_hunts(events.order(earlier + believed), [],
                                self._context(),
                                self._hunt_ids,
                                self._min_severity)

        # A refused row has no time the stream believes, so it may not join any session's sequence: it is hunted by
        # itself, and nothing it carries is learnt.
        for event in refused_events:
            found.extend(hunts.run_hunts([event], [], self._context(), self._hunt_ids, self._min_severity))

        per_row = [[] for _ in range(len(host))]

        for finding in (item.to_dict() for item in found):
            positions = [current[ref] for ref in finding["refs"] if ref in current]

            if (not positions):
                continue

            key = finding_key(finding)
            reported = self._reported.setdefault(finding["session_id"] or "", [])

            if (key in reported):
                continue

            reported.append(key)
            per_row[max(positions)].append(finding)

        severities = []
        hunt_lists = []
        payloads = []

        for findings in per_row:
            if (not findings):
                (severity, hunt_list, payload) = ("", "", "")
            else:
                severity = max((finding["severity"] for finding in findings), key=catalog.SEVERITIES.index)
                hunt_list = ",".join(sorted({finding["hunt_id"] for finding in findings}))
                payload = json.dumps(findings, sort_keys=True)

            severities.append(severity)
            hunt_lists.append(hunt_list)
            payloads.append(payload)

        assign_str_column(df, SEVERITY_COLUMN, severities)
        assign_str_column(df, HUNTS_COLUMN, hunt_lists)
        assign_str_column(df, FINDINGS_COLUMN, payloads)

        self._learn(believed)
        self._retain([records[current[event["ref"]]] for event in believed])
        self._batches += 1
