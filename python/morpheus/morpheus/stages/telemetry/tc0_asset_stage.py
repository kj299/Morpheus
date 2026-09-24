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
The asset half of the TC-0 context store: who owned a host, how much it mattered, and what data it held, over time.

Each input row is one record from an asset inventory or CMDB, carrying the interval it holds for and the instant the
source recorded it, and the stage writes it as a version of a fact in `morpheus.utils.bitemporal`'s sense -- the
columns `context:asset` carries and a `BitemporalStore` is rebuilt from.

**The peer group is read from the source, not computed.** R-B-L7-004 compares a host's process ancestry with its peer
group's, and the peer group is only useful to that rule if it is stable and explainable: the build servers, the
finance workstations. A group derived by clustering hosts on their own behaviour would move when the behaviour it is
supposed to judge moves, and would be one more model to pin. The inventory already knows which hosts are meant to look
alike, so that is what is recorded, and a host with no peer group gets none rather than a guessed one.

**Data classification is a fact with a history like any other.** R-B-L7-002 weights bulk access by the
classification of what was accessed, and a reclassification recorded a week after it took effect is exactly the case
where "as known then" and "as known now" disagree -- the alert that fired said internal, and the investigation reads
restricted. Both answers are kept.

A row without the instant its source recorded it is refused, as it is for identity.
"""

import typing

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.stages.telemetry.tc0_identity_stage import declare_context_columns
from morpheus.stages.telemetry.tc0_identity_stage import log_refusals
from morpheus.stages.telemetry.tc0_identity_stage import write_context_columns
from morpheus.utils import bitemporal
from morpheus.utils.column_assign import to_host_list

ASSET = "asset"
"""The kind of an asset's attributes."""

DEFAULT_ASSET_COLUMNS = ("owner", "owning_team", "criticality", "data_classification", "peer_group")


@register_stage("tc0-asset", ignore_args=["asset_columns"])
class TC0AssetStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write each asset record as a bitemporal version.

    The stage is stateless: each row's output depends on that row alone.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    asset_column : str, default = "hostname"
        Column holding the asset the record is about.
    asset_columns : list of str, optional
        The attributes an asset carries. Defaults to owner, owning team, criticality, data classification and peer
        group.
    valid_from_column : str, default = "valid_from"
        Column holding when the fact started to hold.
    valid_to_column : str, default = "valid_to"
        Column holding when it stopped. Null for a fact that still holds. The column may be absent.
    recorded_column : str, default = "recorded_at"
        Column holding when the source recorded the record. Required on every row.
    change_column : str, default = "change"
        Column holding `assert` or `retract`. The column may be absent, in which case every row asserts.
    time_unit : str, default = "ns"
        Unit for numeric instants. Ignored for datetime and string columns.
    """

    def __init__(self,
                 c: Config,
                 asset_column: str = "hostname",
                 asset_columns: list[str] = None,
                 valid_from_column: str = bitemporal.VALID_FROM,
                 valid_to_column: str = bitemporal.VALID_TO,
                 recorded_column: str = bitemporal.RECORDED_AT,
                 change_column: str = bitemporal.CHANGE,
                 time_unit: str = "ns"):
        super().__init__(c)

        if (not asset_column):
            raise ValueError("asset_column is required")

        asset_columns = list(DEFAULT_ASSET_COLUMNS) if asset_columns is None else list(asset_columns)

        if (len(asset_columns) == 0):
            raise ValueError("asset_columns must name at least one attribute")

        self._asset_column = asset_column
        self._asset_columns = asset_columns
        self._valid_from_column = valid_from_column
        self._valid_to_column = valid_to_column
        self._recorded_column = recorded_column
        self._change_column = change_column
        self._time_unit = time_unit

        declare_context_columns(self._needed_columns)
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc0-asset"

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

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the context columns for every asset record.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming asset records.

        Returns
        -------
        The input message, with the context columns populated.

        Raises
        ------
        KeyError
            If the asset column, the valid-from column, the recorded column, or an attribute column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._asset_column, self._valid_from_column, self._recorded_column]
            required += list(self._asset_columns)
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC0AssetStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            count = len(df)
            attributes = {name: to_host_list(df, name) for name in self._asset_columns}

            (columns, refused) = bitemporal.context_columns(
                ASSET,
                to_host_list(df, self._asset_column), [()] * count, [{
                    name: attributes[name][position]
                    for name in self._asset_columns
                } for position in range(count)],
                to_host_list(df, self._valid_from_column),
                (to_host_list(df, self._valid_to_column) if self._valid_to_column in df.columns else [None] * count),
                to_host_list(df, self._recorded_column),
                (to_host_list(df, self._change_column) if self._change_column in df.columns else [None] * count),
                time_unit=self._time_unit)

            write_context_columns(df, columns)
            log_refusals("TC0AssetStage", refused, count)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
