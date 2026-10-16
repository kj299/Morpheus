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
The application half of the TC-0 context store: which application a server's port belongs to, who owns it, and which
service account it runs as, over time.

Each input row is one record from a CMDB's application-to-server mapping, carrying the interval it holds for and the
instant the source recorded it, and the stage writes it as a version of a fact in `morpheus.utils.bitemporal`'s sense
-- the columns `context:application` carries and a `BitemporalStore` is rebuilt from.

**The entity is the server's address and port, not the application.** What an event carries is where a connection
went, `dst_ip` and `dst_port`, and what an analyst needs on that event is which application answered. Keying the
fact on `address:port`, composed as {py:func}`~morpheus.utils.entity_key.compose_key` composes every key, makes the
point-in-time join a lookup on two columns the event already has. An application on three servers is three facts;
one server running two applications on two ports is two.

**A server reassigned from one application to another is the case the bitemporal record exists for.** The CMDB
learns of the move when someone updates it, not when the move happens, and an event in between was a connection to
the old application as far as anyone could have known and to the new one as is now understood. Both answers are
kept, and `TC0EnrichStage`'s `knowledge` decides which one a row carries.

**The service account is an attribute of the application, not a principal.** The guide's TC-0 contents name the
service-account-to-owning-team mapping; the application record carries both, so a connection to a server's port
resolves to the team that owns it and the identity it runs as in one lookup.

A row without the instant its source recorded it is refused, as it is for identity and assets.
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
from morpheus.utils.entity_key import compose_key

APPLICATION = "application"
"""The kind of an application's attributes."""

DEFAULT_APPLICATION_COLUMNS = ("name", "owning_team", "service_account", "criticality", "data_classification")


@register_stage("tc0-application", ignore_args=["application_columns"])
class TC0ApplicationStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write each application-to-server record as a bitemporal version keyed on the server's address and port.

    The stage is stateless: each row's output depends on that row alone.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    address_column : str, default = "server_address"
        Column holding the address the application is served from.
    port_column : str, default = "server_port"
        Column holding the port it is served on.
    application_columns : list of str, optional
        The attributes an application carries. Defaults to its name, owning team, service account, criticality and
        data classification.
    valid_from_column : str, default = "valid_from"
        Column holding when the mapping started to hold.
    valid_to_column : str, default = "valid_to"
        Column holding when it stopped. Null for a mapping that still holds. The column may be absent.
    recorded_column : str, default = "recorded_at"
        Column holding when the source recorded the record. Required on every row.
    change_column : str, default = "change"
        Column holding `assert` or `retract`. The column may be absent, in which case every row asserts.
    time_unit : str, default = "ns"
        Unit for numeric instants. Ignored for datetime and string columns.
    """

    def __init__(self,
                 c: Config,
                 address_column: str = "server_address",
                 port_column: str = "server_port",
                 application_columns: list[str] = None,
                 valid_from_column: str = bitemporal.VALID_FROM,
                 valid_to_column: str = bitemporal.VALID_TO,
                 recorded_column: str = bitemporal.RECORDED_AT,
                 change_column: str = bitemporal.CHANGE,
                 time_unit: str = "ns"):
        super().__init__(c)

        if (not address_column or not port_column):
            raise ValueError("address_column and port_column are required")

        application_columns = (list(DEFAULT_APPLICATION_COLUMNS)
                               if application_columns is None else list(application_columns))

        if (len(application_columns) == 0):
            raise ValueError("application_columns must name at least one attribute")

        self._address_column = address_column
        self._port_column = port_column
        self._application_columns = application_columns
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
        return "tc0-application"

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
        Write the context columns for every application record.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming application-to-server records.

        Returns
        -------
        The input message, with the context columns populated.

        Raises
        ------
        KeyError
            If the address, port, valid-from or recorded column, or an attribute column, is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._address_column, self._port_column, self._valid_from_column, self._recorded_column]
            required += list(self._application_columns)
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC0ApplicationStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            count = len(df)
            attributes = {name: to_host_list(df, name) for name in self._application_columns}
            servers = [
                compose_key([address, port])
                for (address, port) in zip(to_host_list(df, self._address_column), to_host_list(df, self._port_column))
            ]

            (columns, refused) = bitemporal.context_columns(
                APPLICATION,
                servers, [()] * count, [{
                    name: attributes[name][position]
                    for name in self._application_columns
                } for position in range(count)],
                to_host_list(df, self._valid_from_column),
                (to_host_list(df, self._valid_to_column) if self._valid_to_column in df.columns else [None] * count),
                to_host_list(df, self._recorded_column),
                (to_host_list(df, self._change_column) if self._change_column in df.columns else [None] * count),
                time_unit=self._time_unit)

            write_context_columns(df, columns)
            log_refusals("TC0ApplicationStage", refused, count)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
