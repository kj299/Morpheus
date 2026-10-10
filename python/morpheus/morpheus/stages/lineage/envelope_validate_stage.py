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
Checks the universal envelope a collector owes on every record, stamps the one field the pipeline owes, and keeps a
record whose clock cannot be trusted out of every stateful stage.

Part 2 of the design guide lists fourteen fields every record carries before it reaches Morpheus, and two rules: a
field a collector cannot supply is null with a reason code, never a plausible default, and a record whose clock
offset is above a configured bound is quarantined, not dropped. For as long as the guide has said so, nothing read
ten of the fourteen. `clock_source` and `clock_offset_ms` appeared only in prose, so the clock-skew experiment could
measure what a wrong clock does to a detection and nothing in the pipeline could act on a clock the collector had
itself reported wrong.

**What the collector owes and what the pipeline owes.** Of the fourteen, three are stamped here or downstream:
`ingest_time` by this stage, at the Morpheus source; `event_uid` by `LineageStampStage`; `entity_key` by
`EnvelopeStampStage` at the tail of the class. `telemetry_class` names the segment the record is in rather than
anything the collector measured. The other ten are the collector's, and they are what this stage checks by
default: each present, or null with a reason in `envelope_null_reasons` (`field=reason`, separated by `;`). A
record that falls short is marked rather than refused -- `envelope_valid` false and the shortfall named in
`envelope_defects` -- because the guide's own rule is that the collector is what gets fixed, and a record nobody
sees is a defect nobody fixes. `clock_source` must be one of the guide's four values and `sampling_policy` one of
its three shapes; a value outside them is a defect of the same kind.

**`event_time` defaults to `observed_time`**, as the envelope table says it does: a source that reports no time of
its own is the sensor's time. This is not a fabricated value; it is the field's definition.

**`ingest_time` is never used in a detection.** It is stamped here because this is where the record enters Morpheus,
from `ingest_clock`, which reads the wall clock unless the composition says otherwise. A replay or a test corpus
passes a clock derived from the records themselves, because a field that changed on every run would make every
golden file a record of when it was generated. A record that arrives already carrying `ingest_time` -- from a
source stage that stamped it -- keeps it.

**Quarantine is for the clock.** A record whose collector says its clock is `unsynchronized`, or whose last measured
offset is further from true than `max_offset_ms`, has an `event_time` nobody can place, and every stateful stage
downstream would trust it: open an interval at it, expire a window on it, measure a step against it. It is marked
`clock_quarantined`, with the reason in `quarantine_reason`. With `on_quarantine` set it is also taken out of the
stream and handed to that callable -- the delivery a deployment routes to `morpheus:quarantine` and
`behavior_quarantine` -- so it reaches no feature stage and is still kept, which is what "quarantined, not dropped"
requires. Without it the record stays in the stream, marked, and the stage says how many; a pipeline that has no
route for them yet loses nothing and hides nothing.

A defective envelope is not quarantined on that account alone. An unknown offset with a stated reason is a clock
nobody measured, not a clock known to be wrong, and the record is judged on its other fields.
"""

import logging
import re
import time
import typing

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_frame
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import normalize_text

logger = logging.getLogger(__name__)

CLOCK_SOURCES = ("ptp", "ntp_disciplined", "ntp_undisciplined", "unsynchronized")
"""The guide's four clock sources, best first."""

UNSYNCHRONIZED = "unsynchronized"

DEFAULT_REQUIRED_COLUMNS = ("schema_version",
                            "collector_id",
                            "collector_seq",
                            "observed_time",
                            "event_time",
                            "clock_source",
                            "clock_offset_ms",
                            "sampling_policy",
                            "origin_hash",
                            "tenant_id")
"""The envelope fields a collector owes. The pipeline stamps `ingest_time`, `event_uid` and `entity_key`, and the
segment names `telemetry_class`; `site_id` is required only by the classes that have a site, and they say so."""

DEFAULT_MAX_OFFSET_MS = 1000
"""A second. Well inside what any disciplined source holds and well outside what an operational rule tolerates:
the spoof rule's sixty-second gap is consumed by a minute of switch skew, so a clock a second out is the earliest
sign rather than the damage."""

NULL_REASONS_COLUMN = "envelope_null_reasons"
INGEST_TIME_COLUMN = "ingest_time"
VALID_COLUMN = "envelope_valid"
DEFECTS_COLUMN = "envelope_defects"
QUARANTINED_COLUMN = "clock_quarantined"
QUARANTINE_REASON_COLUMN = "quarantine_reason"

CLOCK_UNSYNCHRONIZED = "clock_unsynchronized"
CLOCK_OFFSET_EXCEEDED = "clock_offset_exceeded"

_SAMPLING = re.compile(r"^(full|1:[1-9][0-9]*|adaptive:\S+)$")
_REASON = re.compile(r"^[a-z][a-z0-9_]*$")

IngestClock = typing.Callable[[typing.Sequence[typing.Optional[int]]], typing.Sequence[int]]


def wall_clock(observed: typing.Sequence[typing.Optional[int]]) -> list[int]:
    """The ingest clock a deployment uses: now, the same for every row in the batch."""
    now = time.time_ns()

    return [now] * len(observed)


def parse_null_reasons(value: typing.Any) -> dict[str, str]:
    """
    Read `field=reason;field=reason` into a mapping.

    A pair without a field, or with a reason that is not a lowercase code, is ignored: a reason is meant to be a code
    an operator can count, and free text would not count.
    """
    text = normalize_text(value)

    if (text is None):
        return {}

    reasons = {}

    for pair in text.split(";"):
        (field, _, reason) = pair.partition("=")
        (field, reason) = (field.strip(), reason.strip())

        if (field and _REASON.match(reason)):
            reasons[field] = reason

    return reasons


def _present(value: typing.Any) -> bool:
    return normalize_text(value) is not None


def _offset_ms(value: typing.Any) -> typing.Optional[float]:
    if (not _present(value)):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@register_stage("envelope-validate", ignore_args=["required_columns", "ingest_clock", "on_quarantine"])
class EnvelopeValidateStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Validate the collector's envelope, stamp `ingest_time`, and quarantine records whose clock cannot be trusted.

    Place it first in a class's segment, after `LineageStampStage` so a quarantined record still has an identity, and
    before any stateful stage, since the point is that none of them sees a quarantined record.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    required_columns : list of str, optional
        The envelope fields checked, each present or null with a reason. Defaults to the ten a collector owes.
    max_offset_ms : int, default = 1000
        Largest `clock_offset_ms`, either side of true, a record's clock may report and still be trusted.
    time_unit : str, default = "ns"
        Unit of numeric `event_time` and `observed_time` values.
    ingest_clock : callable, optional
        Takes each row's `observed_time` (or its `event_time` where that is null, or `None` where both are) and
        returns its `ingest_time` in nanoseconds. Defaults to the wall clock. A replay passes a clock derived from
        the records.
    on_quarantine : callable, optional
        Receives each batch's quarantined records as a host `pandas.DataFrame`. Given, they are removed from the
        stream; not given, they stay in it, marked.
    """

    def __init__(self,
                 c: Config,
                 required_columns: list[str] = None,
                 max_offset_ms: int = DEFAULT_MAX_OFFSET_MS,
                 time_unit: str = "ns",
                 ingest_clock: IngestClock = None,
                 on_quarantine: typing.Callable[[typing.Any], None] = None):
        super().__init__(c)

        required = list(DEFAULT_REQUIRED_COLUMNS) if required_columns is None else list(required_columns)

        if (len(required) == 0):
            raise ValueError("required_columns must name at least one field; an envelope nobody checks is the "
                             "state this stage exists to end")

        if (max_offset_ms < 0):
            raise ValueError(f"max_offset_ms must not be negative, received {max_offset_ms}")

        self._required = required
        self._max_offset_ms = max_offset_ms
        self._time_unit = time_unit
        self._ingest_clock = wall_clock if ingest_clock is None else ingest_clock
        self._on_quarantine = on_quarantine

        # `ingest_time` is deliberately not declared. A declared column is created before the stage runs, filled with
        # zeros, and a zero is indistinguishable from a source having stamped the epoch -- so every record would keep
        # it and none would be stamped. It is written here, and kept where a source wrote it first.
        self._needed_columns[VALID_COLUMN] = TypeId.BOOL8
        self._needed_columns[DEFECTS_COLUMN] = TypeId.STRING
        self._needed_columns[QUARANTINED_COLUMN] = TypeId.BOOL8
        self._needed_columns[QUARANTINE_REASON_COLUMN] = TypeId.STRING

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "envelope-validate"

    def accepted_types(self) -> tuple:
        """
        Accepted input types for this stage.

        Returns
        -------
        tuple
            Accepted input types.
        """
        return (ControlMessage, MessageMeta)

    def supports_cpp_node(self) -> bool:
        """Whether this stage supports a C++ node."""
        return False

    def _time_ns(self, value: typing.Any) -> typing.Optional[int]:
        if (not _present(value)):
            return None

        try:
            return to_epoch_ns(value, time_unit=self._time_unit)
        except ValueError:
            return None

    def _defects(self, row: dict, reasons: dict[str, str]) -> list[str]:
        defects = []

        for field in self._required:
            if (not _present(row.get(field)) and field not in reasons):
                defects.append(field)

        source = normalize_text(row.get("clock_source"))

        if (source is not None and source not in CLOCK_SOURCES):
            defects.append("clock_source:invalid")

        policy = normalize_text(row.get("sampling_policy"))

        if (policy is not None and not _SAMPLING.match(policy)):
            defects.append("sampling_policy:invalid")

        if (_present(row.get("clock_offset_ms")) and _offset_ms(row.get("clock_offset_ms")) is None):
            defects.append("clock_offset_ms:invalid")

        return defects

    def _quarantine_reason(self, row: dict) -> typing.Optional[str]:
        if (normalize_text(row.get("clock_source")) == UNSYNCHRONIZED):
            return CLOCK_UNSYNCHRONIZED

        offset = _offset_ms(row.get("clock_offset_ms"))

        if (offset is not None and abs(offset) > self._max_offset_ms):
            return CLOCK_OFFSET_EXCEEDED

        return None

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Validate every row, stamp `ingest_time`, and route quarantined rows.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming records.

        Returns
        -------
        The message with the validation columns attached and, when `on_quarantine` is set, the quarantined rows
        removed.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        quarantine_mask = None

        with meta.mutable_dataframe() as df:
            row_count = len(df)
            fields = set(self._required) | {
                "clock_source",
                "clock_offset_ms",
                "sampling_policy",
                "observed_time",
                "event_time",
                NULL_REASONS_COLUMN,
                INGEST_TIME_COLUMN
            }
            columns = {name: to_host_list(df, name) if name in df.columns else [None] * row_count for name in fields}
            rows = [{name: values[position] for (name, values) in columns.items()} for position in range(row_count)]

            observed = [self._time_ns(row["observed_time"]) for row in rows]
            events = [self._time_ns(row["event_time"]) for row in rows]

            # The envelope's own default, and only where the source gave no time of its own.
            filled_events = [event if event is not None else seen for (event, seen) in zip(events, observed)]

            if (any(event is None for event in events) and "event_time" in df.columns):
                assign_nullable_int_column(df, "event_time", filled_events)

                for (row, event) in zip(rows, filled_events):
                    row["event_time"] = event

            # Stamped where it is missing and kept where a source stamped it already.
            held = [self._time_ns(row[INGEST_TIME_COLUMN]) for row in rows]
            reference = [seen if seen is not None else event for (seen, event) in zip(observed, filled_events)]
            clocked = list(self._ingest_clock(reference)) if any(value is None for value in held) else held
            ingest = [value if value is not None else clocked[position] for (position, value) in enumerate(held)]

            valid = []
            defects = []
            quarantined = []
            reasons_out = []

            for row in rows:
                reasons = parse_null_reasons(row[NULL_REASONS_COLUMN])
                found = self._defects(row, reasons)
                reason = self._quarantine_reason(row)

                valid.append(len(found) == 0)
                defects.append(",".join(found) if found else None)
                quarantined.append(reason is not None)
                reasons_out.append(reason)

            assign_nullable_int_column(df, INGEST_TIME_COLUMN, ingest)
            assign_nullable_bool_column(df, VALID_COLUMN, valid)
            assign_str_column(df, DEFECTS_COLUMN, defects)
            assign_nullable_bool_column(df, QUARANTINED_COLUMN, quarantined)
            assign_str_column(df, QUARANTINE_REASON_COLUMN, reasons_out)

            if (any(quarantined)):
                quarantine_mask = quarantined

        invalid = valid.count(False)

        if (invalid > 0):
            logger.warning(
                "EnvelopeValidateStage found %d of %d records whose envelope is incomplete or malformed; they are "
                "marked envelope_valid=false with the shortfall in envelope_defects. The collector is what gets "
                "fixed.",
                invalid,
                row_count)

        if (quarantine_mask is None):
            return message

        held_back = sum(quarantine_mask)

        if (self._on_quarantine is None):
            logger.warning(
                "EnvelopeValidateStage quarantined %d of %d records whose clock cannot be trusted; with no "
                "on_quarantine route they stay in the stream, marked clock_quarantined.",
                held_back,
                row_count)

            return message

        # Imported here so that this module remains importable in CPU-only environments where cuDF is absent.
        from morpheus.utils.type_utils import get_df_class

        frame = to_host_frame(meta.copy_dataframe())
        keep = [not flag for flag in quarantine_mask]

        self._on_quarantine(frame[list(quarantine_mask)].reset_index(drop=True))

        accepted = MessageMeta(get_df_class(self._config.execution_mode)(frame[keep].reset_index(drop=True)))

        if (isinstance(message, ControlMessage)):
            message.payload(accepted)
            return message

        return accepted

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
