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
Checks the one promise a collector makes about its own stream: `collector_seq` counts up by one.

The envelope requires `collector_seq` to be strictly monotonic per `collector_id`, and the whole design leans on it --
it is the tiebreaker that makes control 8's order total. Until now that was an assumption written in
`morpheus.utils.determinism` and checked by nothing, so a collector that dropped records, restarted from zero or
replayed a buffer it had already sent produced output indistinguishable from a quiet one.

Each record is compared with the last its collector sent:

- **`collector_seq_gap`** is how many sequence numbers were skipped before it: zero on the next record, the count of
  missing records otherwise. Records a collector numbered and never delivered are the plainest sign that it is
  losing data, and nothing downstream can tell an absent record from an event that did not happen.
- **`collector_seq_regressed`** is true when it is not ahead of the last: the same number again, or an earlier one. A
  collector that restarted its counter, or a buffer replayed into the stream, reads this way.

A collector's first record answers neither question and both are null on it, rather than claiming a stream that
was never established was contiguous. A regressed record does not move the collector's position, so one replayed
record does not turn every honest record after it into a gap.

**Where it goes.** Behind `TotalOrderStage`, which puts every batch into control 8's order -- event time, then
collector, then sequence -- because that order is what makes the check a property of the records rather than of how
they happened to arrive. A collector whose numbers do not follow its own event times is itself reporting something,
and reads as regressed.

The stage is stateful and must run single-engine, or sharded so every record from one collector reaches one
instance -- determinism control 4.
"""

import logging
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
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import normalize_text

logger = logging.getLogger(__name__)

GAP_COLUMN = "collector_seq_gap"
REGRESSED_COLUMN = "collector_seq_regressed"


def _sequence(value: typing.Any) -> typing.Optional[int]:
    if (normalize_text(value) is None):
        return None

    try:
        number = int(value)
    except (TypeError, ValueError):
        return None

    return number if number == value or str(number) == str(value).strip() else None


@register_stage("collector-sequence")
class CollectorSequenceStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Flag records a collector skipped past or sent again.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    collector_column : str, default = "collector_id"
        Column naming the collector whose sequence is checked.
    seq_column : str, default = "collector_seq"
        Column holding the collector's sequence number.
    """

    def __init__(self, c: Config, collector_column: str = "collector_id", seq_column: str = "collector_seq"):
        super().__init__(c)

        self._collector_column = collector_column
        self._seq_column = seq_column
        self._last: dict[str, int] = {}

        self._needed_columns[GAP_COLUMN] = TypeId.INT64
        self._needed_columns[REGRESSED_COLUMN] = TypeId.BOOL8

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "collector-sequence"

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

    @property
    def tracked_collectors(self) -> int:
        """How many collectors the stage holds a position for."""
        return len(self._last)

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the gap and the regression flag on every record.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming records, in control 8's order.

        Returns
        -------
        The same message, with `collector_seq_gap` and `collector_seq_regressed` written.

        Raises
        ------
        KeyError
            If the collector or sequence column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            missing = [name for name in (self._collector_column, self._seq_column) if name not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"CollectorSequenceStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            collectors = [normalize_text(value) for value in to_host_list(df, self._collector_column)]
            sequences = [_sequence(value) for value in to_host_list(df, self._seq_column)]

            gaps: list = []
            regressed: list = []

            for (collector, sequence) in zip(collectors, sequences):
                # No collector, or no number: there is no stream to place the record in.
                if (collector is None or sequence is None):
                    gaps.append(None)
                    regressed.append(None)
                    continue

                last = self._last.get(collector)

                if (last is None):
                    gaps.append(None)
                    regressed.append(None)
                    self._last[collector] = sequence
                    continue

                if (sequence <= last):
                    gaps.append(0)
                    regressed.append(True)
                    continue

                gaps.append(sequence - last - 1)
                regressed.append(False)
                self._last[collector] = sequence

            assign_nullable_int_column(df, GAP_COLUMN, gaps)
            assign_nullable_bool_column(df, REGRESSED_COLUMN, regressed)

        skipped = sum(gap for gap in gaps if gap is not None)
        replayed = sum(1 for flag in regressed if flag)

        if (skipped > 0 or replayed > 0):
            logger.warning(
                "CollectorSequenceStage saw %d sequence numbers skipped and %d records not ahead of their collector's "
                "last in a batch of %d. A gap is a record a collector numbered and never delivered.",
                skipped,
                replayed,
                len(collectors))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
