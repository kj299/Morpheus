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
Each transfer against the envelope its own source, destination and port have kept, which is R-B-L4-005.

{py:mod}`~morpheus.utils.transfer_envelope` holds the arithmetic and the decisions inside it, including why a
quantile over too few samples is the maximum and why the stage will not publish one.

Two things belong at the pipeline level.

**The key is the triple and not the flow.** A flow identifier carries the ephemeral source port, so every
conversation is a new entity and no entity ever has a history -- the baseline would be empty forever while
looking configured. `(src_ip, dst_ip, dst_port)` is the unit a transfer size is a property of, which is what the
guide names, and the same pair on a different port is a different conversation with a different normal.

**More than one magnitude is tracked, because the rule names two.** `bpp` and total `data_len` answer different
questions -- one is the shape of the traffic and the other is its volume -- and a transfer can breach either
without the other. Each gets its own tracker and its own columns, so a search can say which.
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
from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import compose_key
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.transfer_envelope import DEFAULT_MIN_SAMPLES
from morpheus.utils.transfer_envelope import DEFAULT_MULTIPLIER
from morpheus.utils.transfer_envelope import DEFAULT_QUANTILE
from morpheus.utils.transfer_envelope import NS_PER_SECOND
from morpheus.utils.transfer_envelope import TransferEnvelopeTracker

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 30 * 24 * 3600
"""Trailing window the envelope is taken over. Thirty days, which is the period R-B-L4-005 names."""

DEFAULT_MAGNITUDES = ("flow_bpp", "flow_data_len")
"""What the envelope is kept for: the traffic's shape and its volume, which the rule names together."""

TRIPLE_KEY = "transfer_triple"


@register_stage("tc4-envelope", ignore_args=["magnitude_columns", "key_columns"])
class TC4EnvelopeStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Compare each record's transfer magnitudes against the envelope its triple has kept.

    For each magnitude column `m` the stage writes `m_envelope`, `m_envelope_ratio`, `m_envelope_breached` and
    `m_envelope_mature`.

    The stage is stateful across messages and must run single-engine, or sharded by the triple.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    key_columns : list of str, optional
        Columns composing the entity the envelope is kept for. Defaults to `["src_ip", "dst_ip", "dst_port"]`,
        which is the triple R-B-L4-005 names. Keying on a flow identifier instead would put the ephemeral source
        port in the key, so every conversation would be a new entity and no baseline would ever mature.
    magnitude_columns : list of str, optional
        Columns whose envelopes are tracked. Defaults to `["flow_bpp", "flow_data_len"]`.
    time_column : str, default = "event_time"
        Column holding the record's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    window_seconds : int, default = 2592000
        Trailing window the envelope is taken over.
    quantile : float, default = 0.99
        Quantile the envelope is taken at.
    multiplier : float, default = 3.0
        How many times the envelope counts as a breach.
    min_samples : int, default = 100
        Prior transfers required before an envelope is published. A hundred because below it a 99th percentile
        by nearest rank is simply the maximum, so the figure would not be what its name says.
    max_samples : int, default = 4096
        Transfers retained per entity regardless of the window.
    """

    def __init__(self,
                 c: Config,
                 key_columns: list[str] = None,
                 magnitude_columns: list[str] = None,
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 quantile: float = DEFAULT_QUANTILE,
                 multiplier: float = DEFAULT_MULTIPLIER,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 max_samples: int = 4096):
        super().__init__(c)

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        key_columns = ["src_ip", "dst_ip", "dst_port"] if key_columns is None else list(key_columns)
        magnitude_columns = list(DEFAULT_MAGNITUDES) if magnitude_columns is None else list(magnitude_columns)

        if (len(key_columns) == 0):
            raise ValueError("key_columns must name at least one column")

        if (len(magnitude_columns) == 0):
            raise ValueError("magnitude_columns must name at least one column; an envelope around nothing is "
                             "a tracker that answers every question with a null")

        self._key_columns = key_columns
        self._magnitude_columns = magnitude_columns
        self._time_column = time_column
        self._time_unit = time_unit

        self._trackers = {
            column:
                TransferEnvelopeTracker(window_ns=window_seconds * NS_PER_SECOND,
                                        quantile=quantile,
                                        multiplier=multiplier,
                                        min_samples=min_samples,
                                        max_samples=max_samples)
            for column in magnitude_columns
        }

        self._needed_columns[TRIPLE_KEY] = TypeId.STRING

        for column in magnitude_columns:
            self._needed_columns[f"{column}_envelope"] = TypeId.FLOAT64
            self._needed_columns[f"{column}_envelope_ratio"] = TypeId.FLOAT64
            self._needed_columns[f"{column}_envelope_breached"] = TypeId.BOOL8
            self._needed_columns[f"{column}_envelope_mature"] = TypeId.BOOL8

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc4-envelope"

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
    def _magnitude(value: typing.Any) -> typing.Optional[float]:
        """The magnitude as a finite number, or `None` where the record does not carry one."""
        try:
            if (value is None):
                return None

            number = float(value)
        except (TypeError, ValueError):
            return None

        # A record whose magnitude is missing, infinite or not a number contributes nothing rather than putting
        # a value the entity never transferred into the reference every later transfer is measured against.
        # pylint: disable=comparison-with-itself
        return None if (number != number or number in (float("inf"), float("-inf"))) else number

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the triple key and each magnitude's envelope, ratio and breach flag.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming transport records.

        Returns
        -------
        The input message, with the envelope columns populated.

        Raises
        ------
        KeyError
            If a key column or the time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = list(self._key_columns) + [self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC4EnvelopeStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            key_parts = {name: to_host_list(df, name) for name in self._key_columns}
            raw_times = to_host_list(df, self._time_column)
            rows = len(raw_times)

            magnitudes = {
                column: (to_host_list(df, column) if column in df.columns else [None] * rows)
                for column in self._magnitude_columns
            }

            keys: list = []
            envelopes: dict[str, list] = {column: [] for column in self._magnitude_columns}
            ratios: dict[str, list] = {column: [] for column in self._magnitude_columns}
            breached: dict[str, list] = {column: [] for column in self._magnitude_columns}
            mature: dict[str, list] = {column: [] for column in self._magnitude_columns}
            unusable = 0
            unordered = 0

            for position in range(rows):
                key = compose_key([normalize_text(key_parts[name][position]) for name in self._key_columns])

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                keys.append(key)

                if (key is None or event_time_ns is None):
                    unusable += 1

                for column in self._magnitude_columns:
                    value = self._magnitude(magnitudes[column][position])

                    if (key is None or event_time_ns is None or value is None):
                        envelopes[column].append(None)
                        ratios[column].append(None)
                        breached[column].append(None)
                        mature[column].append(False)
                        continue

                    result = self._trackers[column].observe(key, event_time_ns, value)

                    envelopes[column].append(result.baseline)
                    ratios[column].append(result.ratio)
                    breached[column].append(result.breached)
                    mature[column].append(result.mature)
                    unordered += int(result.out_of_order)

            assign_str_column(df, TRIPLE_KEY, keys)

            for column in self._magnitude_columns:
                df[f"{column}_envelope"] = envelopes[column]
                df[f"{column}_envelope_ratio"] = ratios[column]
                assign_nullable_bool_column(df, f"{column}_envelope_breached", breached[column])
                df[f"{column}_envelope_mature"] = mature[column]

            if (unusable > 0):
                logger.info(
                    "TC4EnvelopeStage left %d of %d records unmeasured for want of a triple or a usable event "
                    "time.",
                    unusable,
                    rows)

            if (unordered > 0):
                logger.warning(
                    "TC4EnvelopeStage saw %d magnitude observations arrive out of order; they did not join any "
                    "envelope. The baseline is over prior transfers, and which are prior is what an "
                    "out-of-order arrival disagrees about.",
                    unordered)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
