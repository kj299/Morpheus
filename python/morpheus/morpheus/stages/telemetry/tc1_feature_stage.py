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
"""Derives the TC-1 behavioral novelty features."""

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
from morpheus.utils.column_info import process_dataframe
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import sort_for_cumulative_features
from morpheus.utils.tc1_features import DEFAULT_NEIGHBOR_COLUMN
from morpheus.utils.tc1_features import DEFAULT_PERIOD
from morpheus.utils.tc1_features import DEFAULT_TRANSCEIVER_COLUMN
from morpheus.utils.tc1_features import NEIGHBOR_INCREMENT_COLUMN
from morpheus.utils.tc1_features import TRANSCEIVER_INCREMENT_COLUMN
from morpheus.utils.tc1_features import build_tc1_feature_schema

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [TRANSCEIVER_INCREMENT_COLUMN, NEIGHBOR_INCREMENT_COLUMN]
"""Columns this stage adds to the frame."""

_PERIOD_TIME_COLUMN = "_tc1_period_time"
"""Private datetime column the period is derived from, dropped before the frame is emitted."""


@register_stage("tc1-features")
class TC1FeatureStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Add the TC-1 novelty features to normalized layer 1 telemetry.

    A port reports the same transceiver serial and the same LLDP neighbor on every poll for months at a time, so the
    value of either identifier carries almost no information while a *change* in it carries a great deal. A new
    `transceiver_serial` is hardware substitution, which is what a physical tap looks like from the switch. A new
    `lldp_neighbor_chassis_id` is topology change. This stage counts distinct values per port per period and writes
    each count as an integer, so the steady state is 1 and anything higher is the event.

    The counting is cumulative, which makes it order-dependent: the value written for a row is a function of which
    rows preceded it. The stage therefore imposes determinism control 8's total order before computing anything, and
    emits the frame in that order. This is not a detail that can be left to the caller. An unsorted batch does not
    fail, it quietly answers a different question, and the answer looks entirely reasonable.

    Ordering the whole frame means the emitted rows are permuted relative to the input. Everything else on the row is
    carried through untouched, `event_time` included: the period is derived into a private column rather than by
    converting the envelope's own timestamp, so downstream stages still see the nanoseconds the envelope specifies.

    Place this after `morpheus.stages.telemetry.tc1_normalize_stage.TC1NormalizeStage`, which supplies `entity_key`.
    The stage is stateless across messages, so novelty is counted within each batch; feed it whole windows rather
    than arbitrary batches, or a swap that spans a batch boundary is split across two counts and seen by neither.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    entity_key_column : str, default = "entity_key"
        Column to group by. Grouping any coarser than the port makes the feature meaningless, because a switch
        legitimately reports one transceiver serial per port.
    timestamp_column : str, default = "event_time"
        Column the period is derived from.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `timestamp_column`. Ignored when the column is already a datetime.
    transceiver_column : str, default = "transceiver_serial"
        Identifier for the installed optic.
    neighbor_column : str, default = "lldp_neighbor_chassis_id"
        Identifier for the neighbor the port sees.
    period : str, default = "D"
        Period over which distinct values are counted, as a pandas offset alias. The count resets at each boundary,
        so a change that straddles one is not seen; see `morpheus.utils.tc1_features`.
    order_columns : list of str, optional
        Total order imposed before counting. Defaults to `["event_time", "collector_id", "collector_seq"]`.
    require_total_order : bool, default = True
        Fail when `order_columns` leave ties, rather than letting the features depend on how the batch arrived.
    preserve_columns : list of str, optional
        Additional input columns to carry into the intermediate feature frame, as regular expressions. Rarely
        needed, since the stage merges its features back onto the full input frame.
    """

    def __init__(self,
                 c: Config,
                 entity_key_column: str = "entity_key",
                 timestamp_column: str = "event_time",
                 time_unit: str = "ns",
                 transceiver_column: str = DEFAULT_TRANSCEIVER_COLUMN,
                 neighbor_column: str = DEFAULT_NEIGHBOR_COLUMN,
                 period: str = DEFAULT_PERIOD,
                 order_columns: list[str] = None,
                 require_total_order: bool = True,
                 preserve_columns: list[str] = None):
        super().__init__(c)

        order_columns = list(DEFAULT_ORDER_COLUMNS) if order_columns is None else list(order_columns)

        if (len(order_columns) == 0):
            raise ValueError("order_columns must contain at least one column; a cumulative feature without a total "
                             "order is not reproducible")

        self._entity_key_column = entity_key_column
        self._timestamp_column = timestamp_column
        self._time_unit = time_unit
        self._transceiver_column = transceiver_column
        self._neighbor_column = neighbor_column
        self._order_columns = order_columns
        self._require_total_order = require_total_order

        self._feature_schema = build_tc1_feature_schema(
            entity_key_column=entity_key_column,
            timestamp_column=_PERIOD_TIME_COLUMN,
            transceiver_column=transceiver_column,
            neighbor_column=neighbor_column,
            period=period,
            preserve_columns=() if preserve_columns is None else preserve_columns)

        for column in FEATURE_COLUMNS:
            self._needed_columns[column] = TypeId.INT64

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc1-features"

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

    def _period_time(self, series: pd.Series) -> pd.Series:
        """Return `series` as a datetime, which is what the period is derived from."""
        if (pd.api.types.is_datetime64_any_dtype(series)):
            return series

        return pd.to_datetime(series, unit=self._time_unit)

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Order the payload and add the novelty features.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming message.

        Returns
        -------
        The message, its payload replaced by one holding the same rows in total order with the feature columns added.

        Raises
        ------
        KeyError
            If an order column, the entity key, the timestamp, or an identifier column is absent.
        ValueError
            If `require_total_order` is set and the order columns leave ties.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        source = meta.copy_dataframe()
        is_cudf = hasattr(source, "to_pandas")

        required = [self._entity_key_column, self._timestamp_column, self._transceiver_column, self._neighbor_column]
        missing = [column for column in required if column not in source.columns]

        if (len(missing) > 0):
            raise KeyError(f"TC1FeatureStage requires columns {missing} which are not present in the DataFrame. "
                           f"Available columns: {sorted(source.columns)}")

        ordered = sort_for_cumulative_features(source,
                                               order_columns=self._order_columns,
                                               require_total_order=self._require_total_order)

        ordered[_PERIOD_TIME_COLUMN] = self._period_time(ordered[self._timestamp_column])

        features = process_dataframe(ordered, self._feature_schema)
        ordered[FEATURE_COLUMNS] = features[FEATURE_COLUMNS]

        ordered = ordered.drop(columns=[_PERIOD_TIME_COLUMN])

        if (is_cudf):
            import cudf
            ordered = cudf.from_pandas(ordered)

        payload = MessageMeta(ordered)

        if (isinstance(message, ControlMessage)):
            message.payload(payload)
            return message

        return payload

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
