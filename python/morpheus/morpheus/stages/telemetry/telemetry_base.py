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
The one base class of the TC-3 to TC-7 telemetry stages, and the event-clock guard it gives every one of them.

Every telemetry stage of layers 3 to 7 was written as the same five pieces around its own arithmetic: accept a
`ControlMessage` or a `MessageMeta`, return an empty message untouched, refuse a frame missing a column it reads
with the same `KeyError`, run on the host in either execution mode, and build one MRC map node. Each stage that keeps
state on event time also built its own `EventClock` from its own `max_clock_skew_seconds`, declared
`clock_refused`, and passed its times through `guard_times` -- and four layer 5 stages, which keep windows,
histories and an entity's last sighting on event time, did none of that (guide 13, plan item 8). A single row from a
clock wrong by years emptied every principal's ten-minute authentication window in `TC5RiskStage` at once.

`TelemetryStage` holds those pieces once. A subclass states what it reads in `_required_columns`, does its work in
`_process`, and, when its state moves with event time, passes `max_clock_skew_seconds` to the constructor and reads
its times through `_guarded_times`. **The guard is not optional for a stage that keeps state on event time.**
`tests/morpheus/stages/test_telemetry_base.py` holds every TC-3 to TC-7 stage to the base class, every one that
takes an event-time column to the guard unless it is listed with the reason it keeps no such state, and every guarded
stage to the property the guard exists for: one row from years ahead, inserted among the stage's own corpus, changes
no other row's output.

The base class decides nothing about what a stage computes. A refused row is given to the stage's own code as a row
without a time, which every tracker in `morpheus.utils` already treats as a row with nothing to measure, so the
guard adds no branch to any stage's arithmetic.
"""

import logging
import typing

import mrc
from mrc.core import operators as ops

from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.event_clock import CLOCK_REFUSED_COLUMN
from morpheus.utils.event_clock import clock_for_skew_seconds
from morpheus.utils.event_clock import event_times_ns
from morpheus.utils.event_clock import guard_times
from morpheus.utils.event_clock import refuse_rows


class TelemetryStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    A telemetry stage of layers 3 to 7: one frame in, the same frame out with this stage's columns written.

    A stage with memory also takes `morpheus.pipeline.checkpoint_mixin.CheckpointMixin`, first, and lists its memory
    in `_checkpoint_attributes`; a stage with a clock lists `_clock` there, since the clock is memory too.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    max_clock_skew_seconds : int, optional
        Passed by a stage whose state moves with event time: how far ahead of the stream's own progress a row's event
        time may be before it is refused. Building the clock declares `clock_refused`. A stage that keeps no state on
        event time passes nothing and has no clock. See `morpheus.utils.event_clock`.
    """

    def __init__(self, c: Config, max_clock_skew_seconds: typing.Optional[int] = None):
        super().__init__(c)

        self._clock = None

        if (max_clock_skew_seconds is not None):
            self._clock = clock_for_skew_seconds(max_clock_skew_seconds)
            self._needed_columns[CLOCK_REFUSED_COLUMN] = TypeId.BOOL8

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

    def _required_columns(self, df) -> typing.Iterable[str]:
        """The columns this stage cannot run without, given the frame it was handed. None, unless overridden."""
        del df

        return ()

    def _process(self, df) -> None:
        """Write this stage's columns into `df`, in place."""
        raise NotImplementedError

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write this stage's columns.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming message.

        Returns
        -------
        The input message, with this stage's columns populated. An empty message is returned untouched.

        Raises
        ------
        KeyError
            If a column `_required_columns` names is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            missing = sorted({column for column in self._required_columns(df) if column not in df.columns})

            if (len(missing) > 0):
                raise KeyError(f"{type(self).__name__} requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            self._process(df)

        return message

    def _guarded_times(self, df, time_column: str, time_unit: str = "ns") -> tuple[list, list]:
        """
        The frame's event times in nanoseconds, with every time the clock refuses replaced by `None`.

        Returns
        -------
        tuple of (list, list)
            The times, and one boolean per row saying which were refused, for `_refuse`.
        """
        return guard_times(self._clock, event_times_ns(df, time_column, time_unit))

    def _refuse(self, df, refused: typing.Sequence[bool], *, logger: logging.Logger, what: str, outcome: str) -> int:
        """Mark this stage's refusals in `clock_refused` and log them; see `morpheus.utils.event_clock.refuse_rows`."""
        return refuse_rows(df, refused, logger=logger, stage=type(self).__name__, what=what, outcome=outcome)

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
