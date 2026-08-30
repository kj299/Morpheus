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
"""Groups rows into deterministically sealed event-time windows."""

import logging
import typing

import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.stages.lineage._column_utils import to_host_list
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.window_seal import SEALED_BY_FLUSH
from morpheus.utils.window_seal import SEALED_BY_WATERMARK
from morpheus.utils.window_seal import WindowSealer

logger = logging.getLogger(__name__)


@register_stage("window-seal")
class WindowSealStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Group rows into fixed event-time windows and emit each window once it is deterministically sealed.

    This stage implements determinism control 7 of the OSI behavioral analytics guide. Windows are half-open
    intervals anchored to a fixed absolute epoch, and a window is emitted only once the event-time watermark, the
    largest event time seen so far, has passed the window's end by the lateness horizon. Sealing is driven by the
    data rather than the wall clock, so a replay of the same input produces the same windows with the same
    membership, sealed at the same points in the stream.

    Rows that arrive for a window that has already been sealed are emitted separately, marked late and incomplete.
    They never mutate a window that has already been published; turning them into an authoritative result is an
    explicit backfill, which republishes the whole window under a higher revision.

    Each emitted message carries one window, with these columns added to the buffered rows:

    - `window_id`, `window_start_ns`, `window_end_ns`: the window ordinal and its half-open bounds. `window_end_ns`
      is the value to use wherever a scoring timestamp is needed, in place of any wall-clock read.
    - `revision`: 0 for the sealed publication, incrementing for each subsequent publication of the same window.
      Consumers keep every revision and filter to the maximum per window.
    - `sealed_by`: `watermark` normally; `flush` when the stream ended while the window was still open; `invalid`
      on the message carrying rows whose event time could not be interpreted.
    - `is_late`, `window_complete`: the late-arrival flags. Only `window_complete` rows are scoring inputs.

    Rows whose event time cannot be interpreted are emitted in their own message with `window_id = -1` and
    `sealed_by = "invalid"`, rather than being dropped: a dropped row looks like an absence of activity, which is
    the worst failure mode for a detection pipeline. Set `raise_on_invalid` to make them fatal instead.

    This stage is stateful and must run single-engine, which is the Morpheus default. For parallelism, shard the
    stream by entity upstream and give each shard its own instance, which is determinism control 4.

    Incoming `ControlMessage` tasks and metadata are not propagated: windowing regroups rows across message
    boundaries, so per-message metadata has no well-defined destination. Emitted `ControlMessage` objects are fresh
    and carry only the window payload.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    period_seconds : int, default = 300
        Window length.
    lateness_seconds : int, default = 900
        Lateness horizon: how far the watermark must pass a window's end before the window seals. Set it per
        telemetry class; 15 minutes suits network telemetry, SaaS audit feeds need more.
    epoch : str, default = "1970-01-01"
        Absolute anchor for window boundaries, as any timestamp string `pandas.Timestamp` accepts. An integer
        nanosecond value is also accepted when constructing programmatically.
    time_column : str, default = "event_time"
        Column holding the event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    order_columns : list of str, optional
        When set, each emitted window's rows are sorted by these columns with a stable sort before emission,
        which is the total row order that cumulative features require (determinism control 8). The guide's
        recommendation is `["event_time", "collector_id", "collector_seq"]`.
    seal_on_complete : bool, default = True
        Whether windows still open when the stream ends are emitted with `sealed_by = "flush"`. When False they are
        discarded, which is only appropriate when a resumed pipeline will observe the same rows again.
    raise_on_invalid : bool, default = False
        When True a row with an uninterpretable event time fails the batch instead of being emitted flagged.
    """

    def __init__(self,
                 c: Config,
                 period_seconds: int = 300,
                 lateness_seconds: int = 900,
                 epoch: str = "1970-01-01",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 order_columns: list[str] = None,
                 seal_on_complete: bool = True,
                 raise_on_invalid: bool = False):
        super().__init__(c)

        if (period_seconds <= 0):
            raise ValueError(f"period_seconds must be positive, received {period_seconds}")

        if (lateness_seconds < 0):
            raise ValueError(f"lateness_seconds must not be negative, received {lateness_seconds}")

        if (not time_column):
            raise ValueError("time_column is required")

        epoch_ns = to_epoch_ns(epoch) if not isinstance(epoch, int) else epoch

        self._sealer = WindowSealer(period_ns=period_seconds * NS_PER_SECOND,
                                    lateness_ns=lateness_seconds * NS_PER_SECOND,
                                    epoch_ns=epoch_ns)
        self._time_column = time_column
        self._time_unit = time_unit
        self._order_columns = list(order_columns) if order_columns is not None else None
        self._seal_on_complete = seal_on_complete
        self._raise_on_invalid = raise_on_invalid

        # Buffered on-time rows per open window, as pandas fragments in arrival order.
        self._buffers: dict[int, list[pd.DataFrame]] = {}
        self._emit_control_messages: typing.Optional[bool] = None

        self._needed_columns.update({
            "window_id": TypeId.INT64,
            "window_start_ns": TypeId.INT64,
            "window_end_ns": TypeId.INT64,
            "revision": TypeId.INT64,
            "sealed_by": TypeId.STRING,
            "is_late": TypeId.BOOL8,
            "window_complete": TypeId.BOOL8,
        })

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "window-seal"

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

    def _to_message(self, df: pd.DataFrame) -> typing.Union[ControlMessage, MessageMeta]:
        # Imported here so that this module remains importable in CPU-only environments where cuDF is absent.
        from morpheus.utils.type_utils import get_df_class

        out_df = get_df_class(self._config.execution_mode)(df.reset_index(drop=True))
        meta = MessageMeta(out_df)

        if (self._emit_control_messages):
            message = ControlMessage()
            message.payload(meta)

            return message

        return meta

    def _build_window(self,
                      window_id: typing.Optional[int],
                      fragments: list[pd.DataFrame],
                      sealed_by: str,
                      late: bool,
                      complete: bool) -> pd.DataFrame:
        df = pd.concat(fragments, ignore_index=True) if len(fragments) > 1 else fragments[0].reset_index(drop=True)

        if (self._order_columns is not None and window_id is not None):
            df = df.sort_values(self._order_columns, kind="mergesort").reset_index(drop=True)

        if (window_id is not None):
            (start_ns, end_ns) = self._sealer.window_bounds(window_id)
            revision = self._sealer.next_revision(window_id)
        else:
            (window_id, start_ns, end_ns) = (-1, -1, -1)
            revision = 0

        df["window_id"] = window_id
        df["window_start_ns"] = start_ns
        df["window_end_ns"] = end_ns
        df["revision"] = revision
        df["sealed_by"] = sealed_by
        df["is_late"] = late
        df["window_complete"] = complete

        return df

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]) -> list:
        """
        Buffer a batch of rows into their windows and emit every window this batch sealed.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming message.

        Returns
        -------
        list
            Zero or more messages: one per window sealed by this batch, plus one per window that received late
            rows, plus at most one carrying rows with uninterpretable event times.

        Raises
        ------
        KeyError
            If the time column is absent.
        ValueError
            If `raise_on_invalid` is set and a row's event time cannot be interpreted.
        """
        if (self._emit_control_messages is None):
            self._emit_control_messages = isinstance(message, ControlMessage)

        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return []

        with meta.mutable_dataframe() as df:
            if (self._time_column not in df.columns):
                raise KeyError(f"WindowSealStage requires column {self._time_column!r} which is not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            pdf = df.to_pandas() if hasattr(df, "to_pandas") else df.copy(deep=False)
            raw_times = to_host_list(df, self._time_column)

        times = []
        for value in raw_times:
            try:
                times.append(to_epoch_ns(value, time_unit=self._time_unit))
            except ValueError:
                times.append(None)

        invalid_count = sum(1 for t in times if t is None)

        if (invalid_count > 0 and self._raise_on_invalid):
            raise ValueError(f"{invalid_count} of {len(times)} rows have event times that cannot be interpreted "
                             f"from column {self._time_column!r}")

        result = self._sealer.observe(times)

        on_time_rows: dict[int, list[int]] = {}
        late_rows: dict[int, list[int]] = {}
        invalid_rows: list[int] = []

        for (position, assignment) in enumerate(result.assignments):
            if (assignment.window_id is None):
                invalid_rows.append(position)
            elif (assignment.late):
                late_rows.setdefault(assignment.window_id, []).append(position)
            else:
                on_time_rows.setdefault(assignment.window_id, []).append(position)

        # One fragment per window per batch; iloc with a position list copies, so the buffers do not alias pdf.
        for (window_id, positions) in on_time_rows.items():
            self._buffers.setdefault(window_id, []).append(pdf.iloc[positions])

        emissions = []

        for window_id in result.sealed:
            fragments = self._buffers.pop(window_id)
            emissions.append(
                self._to_message(
                    self._build_window(window_id, fragments, SEALED_BY_WATERMARK, late=False, complete=True)))

        for (window_id, positions) in sorted(late_rows.items()):
            emissions.append(
                self._to_message(
                    self._build_window(window_id, [pdf.iloc[positions]], SEALED_BY_WATERMARK, late=True,
                                       complete=False)))

        if (len(invalid_rows) > 0):
            logger.warning("WindowSealStage could not interpret event times for %d of %d rows on column %r",
                           len(invalid_rows),
                           len(times),
                           self._time_column)
            emissions.append(
                self._to_message(
                    self._build_window(None, [pdf.iloc[invalid_rows]], "invalid", late=False, complete=False)))

        return emissions

    def on_completed(self) -> typing.Optional[list]:
        """Seal and emit every window still open when the stream ends."""
        if (not self._seal_on_complete):
            self._buffers.clear()
            return None

        emissions = []

        for window_id in self._sealer.flush():
            fragments = self._buffers.pop(window_id, None)

            if (fragments is None):
                continue

            emissions.append(
                self._to_message(self._build_window(window_id, fragments, SEALED_BY_FLUSH, late=False, complete=True)))

        return emissions if len(emissions) > 0 else None

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name,
                                 ops.map(self.on_data),
                                 ops.filter(lambda x: len(x) > 0),
                                 ops.on_completed(self.on_completed),
                                 ops.flatten())
        builder.make_edge(input_node, node)

        return node
