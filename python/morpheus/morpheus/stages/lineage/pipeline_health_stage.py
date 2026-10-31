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
One health record per sealed window per collector, telemetry class and tenant: what the pipeline itself did.

Every stage in this fork counts what it could not do -- a row it refused, a session it abandoned, an exchange that
never paired -- and until now said so in a log line. A log line answers nothing an operator asks of a pipeline that
has run for a month: which collector went quiet on Tuesday, which tenant's audit feed doubled, whether last night's
spike in unpaired authorizations was the estate or one RADIUS server. The stages now write those counts on the rows
they concern, and this stage gathers them.

Each record is keyed on the window the rows were sealed in and on who sent them -- `collector_id`, the telemetry
class and `tenant_id`, the platform the guide's envelope names for logical scoping -- and carries, for that window:

- `rows`, and how many of them had an invalid envelope, a clock a stage refused, no entity key, an unpaired outcome,
  an unclassified action, a sample out of order, a late arrival, an unresolved binding, a saturated count, or a
  fallback model; and how many earlier exchanges or sessions their arrival abandoned;
- the collector's sequence gaps and regressions, from `CollectorSequenceStage`;
- the largest lag from the sensor to the pipeline (`ingest_time` less `observed_time`) and from the event to the
  sensor (`observed_time` less `event_time`), in seconds;
- the collector's reported clock offset, smallest, median and largest;
- and the record count measured against the same collector, class and tenant's own history: the most rows any
  earlier window held, how many windows that history spans, whether it is mature, and this window's step above it.
  That last is the first baseline any platform has in this fork: a tenant whose feed doubles reads as a step the
  way a port's MAC count does.

A count whose column is absent from the rows is null rather than zero, so "nothing was unpaired" and "nothing here
can be unpaired" stay different answers.

The record's `event_time` is the window's end, the moment the record is about, so the SIEM indexes it there.

**Placement.** Behind `WindowSealStage`, reading what it seals. It replaces each batch with that batch's health
records, so in a deployment it sits on its own branch of the sealed stream, beside the sink the scored rows go to.
A late row for a window already reported produces a second record for that window, carrying its late count, which
is the honest account of when the pipeline learned of it.

The stage is stateful in its baselines and must run single-engine -- determinism control 4.
"""

import logging
import statistics
import typing

import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.checkpoint_mixin import CheckpointMixin
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.bucket_peak import DEFAULT_MAX_BUCKETS
from morpheus.utils.bucket_peak import DEFAULT_MIN_BUCKETS
from morpheus.utils.bucket_peak import NS_PER_SECOND
from morpheus.utils.bucket_peak import BucketPeakTracker
from morpheus.utils.bucket_peak import measure_rows
from morpheus.utils.column_assign import to_host_frame
from morpheus.utils.entity_key import KEY_SEPARATOR
from morpheus.utils.entity_key import normalize_text

logger = logging.getLogger(__name__)

DEFAULT_UNPAIRED_COLUMNS = ("auth_unpaired", "session_unpaired")
DEFAULT_ABANDONED_COLUMNS = ("auth_abandoned", "session_abandoned")
DEFAULT_UNCLASSIFIED_COLUMNS = ("session_unclassified", )
DEFAULT_OUT_OF_ORDER_COLUMNS = ("sample_out_of_order", "session_out_of_order")
DEFAULT_FALLBACK_COLUMNS = ("model_fallback_used", )
SATURATED_SUFFIX = "_saturated"
UNRESOLVED = "unresolved"

DEFAULT_WINDOW_SECONDS = 30 * 24 * 3600
"""How far back a collector's record counts reach. Thirty days."""

HEALTH_COLUMNS = ("event_time",
                  "window_id",
                  "window_start_ns",
                  "window_end_ns",
                  "collector_id",
                  "telemetry_class",
                  "tenant_id",
                  "rows",
                  "envelope_invalid",
                  "clock_refused",
                  "keyless",
                  "unpaired",
                  "abandoned",
                  "unclassified",
                  "out_of_order",
                  "late",
                  "unresolved",
                  "saturated",
                  "fallback",
                  "seq_gap",
                  "seq_regressed",
                  "max_ingest_lag_seconds",
                  "max_observe_lag_seconds",
                  "clock_offset_ms_min",
                  "clock_offset_ms_median",
                  "clock_offset_ms_max",
                  "rows_baseline_max",
                  "rows_baseline_buckets",
                  "rows_baseline_mature",
                  "rows_step")
"""Every field a health record carries, in order."""


def _true(value: typing.Any) -> bool:
    try:
        # A nullable boolean, compared by value rather than identity.
        return value is not None and bool(value == True)  # noqa: E712 pylint: disable=singleton-comparison
    except (TypeError, ValueError):
        return False


def _number(value: typing.Any) -> typing.Optional[float]:
    if (normalize_text(value) is None):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@register_stage("pipeline-health",
                ignore_args=[
                    "unpaired_columns",
                    "abandoned_columns",
                    "unclassified_columns",
                    "out_of_order_columns",
                    "fallback_columns"
                ])
class PipelineHealthStage(CheckpointMixin, GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Gather the stages' own counts into one health record per window, collector, telemetry class and tenant.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    period_seconds : int, default = 300
        The sealing period, which is the history's bucket: one window, one committed period.
    window_seconds : int, default = 2592000
        How far back the record-count history reaches.
    min_buckets : int, default = 24
        Windows a history needs before its baseline is published.
    max_buckets : int, default = 1024
        Windows retained per history.
    telemetry_class : str, optional
        The class name, for a segment that carries one class and no column naming it. A `telemetry_class` column,
        where present, wins.
    unpaired_columns, abandoned_columns, unclassified_columns, out_of_order_columns, fallback_columns : list of str
        The flag and count columns each total reads. The defaults are what the shipped stages write.
    """

    _checkpoint_attributes = ("_history", )

    def __init__(self,
                 c: Config,
                 period_seconds: int = 300,
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 min_buckets: int = DEFAULT_MIN_BUCKETS,
                 max_buckets: int = DEFAULT_MAX_BUCKETS,
                 telemetry_class: str = None,
                 unpaired_columns: list[str] = None,
                 abandoned_columns: list[str] = None,
                 unclassified_columns: list[str] = None,
                 out_of_order_columns: list[str] = None,
                 fallback_columns: list[str] = None):
        super().__init__(c)

        if (period_seconds <= 0):
            raise ValueError(f"period_seconds must be positive, received {period_seconds}")

        self._period_ns = period_seconds * NS_PER_SECOND
        self._telemetry_class = normalize_text(telemetry_class)
        self._unpaired = tuple(DEFAULT_UNPAIRED_COLUMNS if unpaired_columns is None else unpaired_columns)
        self._abandoned = tuple(DEFAULT_ABANDONED_COLUMNS if abandoned_columns is None else abandoned_columns)
        self._unclassified = tuple(DEFAULT_UNCLASSIFIED_COLUMNS if unclassified_columns is
                                   None else unclassified_columns)
        self._out_of_order = tuple(DEFAULT_OUT_OF_ORDER_COLUMNS if out_of_order_columns is
                                   None else out_of_order_columns)
        self._fallback = tuple(DEFAULT_FALLBACK_COLUMNS if fallback_columns is None else fallback_columns)

        self._history = BucketPeakTracker(bucket_ns=self._period_ns,
                                          window_ns=window_seconds * NS_PER_SECOND,
                                          min_buckets=min_buckets,
                                          max_buckets=max_buckets)

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "pipeline-health"

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

    @staticmethod
    def _count(group: pd.DataFrame, columns: typing.Iterable[str]) -> typing.Optional[int]:
        """Rows on which any of the columns is true; null when none of them is on these rows at all."""
        present = [name for name in columns if name in group.columns]

        if (len(present) == 0):
            return None

        flags = zip(*[[_true(value) for value in group[name]] for name in present])

        return int(sum(1 for row in flags if any(row)))

    @staticmethod
    def _answered_count(group: pd.DataFrame, columns: typing.Iterable[str]) -> typing.Optional[int]:
        """
        `_count`, but null when the columns are null on every row of the window as well as when they are absent.

        For a flag a stage writes only on the rows it acted on. `model_fallback_used` is stamped on a scored row and
        null on one the model never saw, so a window with no scored row cannot say how many fell back -- and whether
        the column survives into the frame depends on what else the run held, a whole replay or a run stopped before
        the first scored week. Answering from the window's own rows makes the count the same either way.
        """
        present = [name for name in columns if name in group.columns]

        if (not any(normalize_text(value) is not None for name in present for value in group[name])):
            return None

        return PipelineHealthStage._count(group, present)

    @staticmethod
    def _sum(group: pd.DataFrame, columns: typing.Iterable[str]) -> typing.Optional[int]:
        present = [name for name in columns if name in group.columns]

        if (len(present) == 0):
            return None

        return int(sum(_number(value) or 0 for name in present for value in group[name]))

    @staticmethod
    def _lag(group: pd.DataFrame, later: str, earlier: str) -> typing.Optional[float]:
        if (later not in group.columns or earlier not in group.columns):
            return None

        lags = [(b - a) / NS_PER_SECOND for (a, b) in zip(map(_number, group[earlier]), map(_number, group[later]))
                if a is not None and b is not None]

        return round(max(lags), 4) if lags else None

    def _record(self, key: tuple, group: pd.DataFrame) -> dict:
        (window_id, collector, telemetry_class, tenant) = key
        starts = [_number(value) for value in group["window_start_ns"]] if "window_start_ns" in group else []
        ends = [_number(value) for value in group["window_end_ns"]] if "window_end_ns" in group else []
        start = min((value for value in starts if value is not None), default=None)
        end = max((value for value in ends if value is not None), default=None)

        offsets = sorted(value for value in map(_number, group["clock_offset_ms"])
                         if value is not None) if "clock_offset_ms" in group else []
        saturated = [name for name in group.columns if name.endswith(SATURATED_SUFFIX)]
        regressions = self._count(group, ("collector_seq_regressed", ))

        return {
            "event_time": None if end is None else int(end),
            "window_id": int(window_id),
            "window_start_ns": None if start is None else int(start),
            "window_end_ns": None if end is None else int(end),
            "collector_id": collector,
            "telemetry_class": telemetry_class,
            "tenant_id": tenant,
            "rows": len(group),
            "envelope_invalid": (None if "envelope_valid" not in group else int(
                sum(1 for value in group["envelope_valid"] if normalize_text(value) is not None and not _true(value)))),
            "clock_refused": self._count(group, ("clock_refused", )),
            "keyless": (None if "entity_key" not in group else int(
                sum(1 for value in group["entity_key"] if normalize_text(value) is None))),
            "unpaired": self._count(group, self._unpaired),
            "abandoned": self._sum(group, self._abandoned),
            "unclassified": self._count(group, self._unclassified),
            "out_of_order": self._count(group, self._out_of_order),
            "late": self._count(group, ("is_late", )),
            "unresolved": (None if "resolution_method" not in group else int(
                sum(1 for value in group["resolution_method"] if normalize_text(value) == UNRESOLVED))),
            "saturated": self._count(group, saturated),
            "fallback": self._answered_count(group, self._fallback),
            "seq_gap": self._sum(group, ("collector_seq_gap", )),
            "seq_regressed": regressions,
            "max_ingest_lag_seconds": self._lag(group, "ingest_time", "observed_time"),
            "max_observe_lag_seconds": self._lag(group, "observed_time", "event_time"),
            "clock_offset_ms_min": offsets[0] if offsets else None,
            "clock_offset_ms_median": float(statistics.median(offsets)) if offsets else None,
            "clock_offset_ms_max": offsets[-1] if offsets else None,
        }

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Replace the batch with its health records.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Sealed rows.

        Returns
        -------
        A message whose payload is the batch's health records, ordered by window, collector, class and tenant.

        Raises
        ------
        KeyError
            If the rows carry no `window_id`, which means they were not sealed.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        frame = to_host_frame(meta.copy_dataframe())

        if ("window_id" not in frame.columns):
            raise KeyError("PipelineHealthStage reads sealed rows and these carry no window_id; place it behind "
                           f"WindowSealStage. Available columns: {sorted(frame.columns)}")

        def column(name: str, fallback: typing.Any = None) -> list:
            return [normalize_text(value)
                    for value in frame[name]] if name in frame.columns else [fallback] * len(frame)

        classes = column("telemetry_class", self._telemetry_class)
        keys = list(
            zip([None if normalize_text(value) is None else int(value) for value in frame["window_id"]],
                column("collector_id"), [value if value is not None else self._telemetry_class for value in classes],
                column("tenant_id")))

        frame = frame.assign(_health_key=keys)
        groups = {}

        for (key, group) in frame.groupby("_health_key", sort=False):
            if (key[0] is None):
                continue

            groups[key] = group.drop(columns=["_health_key"])

        def order(key: tuple) -> tuple:
            return tuple("" if part is None else str(part).zfill(20) if isinstance(part, int) else part for part in key)

        records = [self._record(key, groups[key]) for key in sorted(groups, key=order)]

        history_keys = [
            None if record["collector_id"] is None else KEY_SEPARATOR.join(
                str(part) if part is not None else ""
                for part in (record["collector_id"], record["telemetry_class"], record["tenant_id"]))
            for record in records
        ]
        measured = measure_rows(self._history,
                                history_keys, [record["rows"] for record in records],
                                [record["window_start_ns"] for record in records])

        for (record, reference, buckets, mature, step) in zip(records,
                                                              measured.references,
                                                              measured.buckets,
                                                              measured.mature,
                                                              measured.steps):
            record["rows_baseline_max"] = reference
            record["rows_baseline_buckets"] = buckets
            record["rows_baseline_mature"] = mature
            record["rows_step"] = step

        # Imported here so that this module remains importable in CPU-only environments where cuDF is absent.
        from morpheus.utils.type_utils import get_df_class

        health = pd.DataFrame(records, columns=list(HEALTH_COLUMNS))

        for name in ("rows_baseline_max",
                     "rows_baseline_buckets",
                     "rows_step",
                     "event_time",
                     "window_start_ns",
                     "window_end_ns",
                     "envelope_invalid",
                     "clock_refused",
                     "keyless",
                     "unpaired",
                     "abandoned",
                     "unclassified",
                     "out_of_order",
                     "late",
                     "unresolved",
                     "saturated",
                     "fallback",
                     "seq_gap",
                     "seq_regressed"):
            health[name] = health[name].astype("Int64")

        health["rows_baseline_mature"] = health["rows_baseline_mature"].astype("boolean")

        result = MessageMeta(get_df_class(self._config.execution_mode)(health))

        if (isinstance(message, ControlMessage)):
            message.payload(result)
            return message

        return result

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
