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
The identity half of the TC-0 context store: who a principal was, and which groups they were in, over time.

Each input row is one record from an identity source -- an HR system, a directory -- carrying the interval it holds
for and the instant the source recorded it. The stage turns each into a version of a fact in
`morpheus.utils.bitemporal`'s sense and writes the columns `context:identity` carries and a `BitemporalStore` is
rebuilt from.

**Two kinds of fact share the sourcetype.** A *profile* is the principal's department, manager and employment status,
one fact per principal. A *membership* is one group or role the principal belongs to, one fact per principal and
group, each with its own dates, because that is how the sources hold them and because "the principal's role
assignment is unchanged" -- which R-P-L7-006 depends on -- is then a question about intervals rather than about
whether two lists happen to be spelled the same. A row naming a group is a membership; any other row is a profile.

**A row without the instant its source recorded it is refused**, not stamped with the moment it arrived here. See
the `bitemporal` module for why that would make every as-known-at answer depend on when the pipeline ran. Refused
rows stay in the output with their reason, so a source that stops sending the field shows up as a count rather than as
a store that quietly knows less.

This is the most personal data this fork holds -- employment history and reporting lines about identifiable people.
The stage keeps no state and sets no retention; how long the records are kept is a decision for the estate and its
counsel, not for a default here.
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
from morpheus.utils import bitemporal
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import normalize_text

logger = logging.getLogger(__name__)

PROFILE = "profile"
"""The kind of a principal's own attributes."""

DEFAULT_PROFILE_COLUMNS = ("department", "manager", "employment_status")

GROUP_ATTRIBUTE = "group_name"
"""The attribute a membership records its group under, whatever the source column is called, so a consumer of the
store never has to know how the source spelled it."""


def write_context_columns(df, columns: dict):
    """Assign the columns `bitemporal.context_columns` produced, with the types the sourcetype expects."""
    for name in (bitemporal.VALID_FROM, bitemporal.VALID_TO, bitemporal.RECORDED_AT):
        assign_nullable_int_column(df, name, columns[name])

    for name in (bitemporal.CONTEXT_KIND,
                 bitemporal.CONTEXT_ENTITY,
                 bitemporal.CONTEXT_KEY,
                 bitemporal.CONTEXT_ATTRIBUTES,
                 bitemporal.CONTEXT_UID,
                 bitemporal.CONTEXT_REFUSED,
                 bitemporal.CHANGE):
        assign_str_column(df, name, columns[name])


def declare_context_columns(needed: dict):
    """Declare the columns `write_context_columns` writes."""
    for name in (bitemporal.VALID_FROM, bitemporal.VALID_TO, bitemporal.RECORDED_AT):
        needed[name] = TypeId.INT64

    for name in (bitemporal.CONTEXT_KIND,
                 bitemporal.CONTEXT_ENTITY,
                 bitemporal.CONTEXT_KEY,
                 bitemporal.CONTEXT_ATTRIBUTES,
                 bitemporal.CONTEXT_UID,
                 bitemporal.CONTEXT_REFUSED,
                 bitemporal.CHANGE):
        needed[name] = TypeId.STRING


def log_refusals(stage_name: str, refused: dict, total: int):
    """Say how many records were refused and why."""
    if (len(refused) > 0):
        logger.warning(
            "%s refused %d of %d context records: %s. A refused record is kept in the output, marked, "
            "and never enters a store.",
            stage_name,
            sum(refused.values()),
            total,
            ", ".join(f"{count} {reason}" for (reason, count) in sorted(refused.items())))


@register_stage("tc0-identity", ignore_args=["profile_columns"])
class TC0IdentityStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write each identity record as a bitemporal version: a profile, or one group membership.

    The stage is stateless: each row's output depends on that row alone, so it needs no ordering and no sharding.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    principal_column : str, default = "user_principal"
        Column holding the principal the record is about.
    group_column : str, default = "group_name"
        Column naming the group of a membership record. A row with a value here is a membership and its profile
        columns are ignored. The column may be absent, in which case every row is a profile. The group is recorded
        under `group_name` whatever this column is called.
    profile_columns : list of str, optional
        The attributes a profile carries. Defaults to department, manager and employment status.
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
                 principal_column: str = "user_principal",
                 group_column: str = "group_name",
                 profile_columns: list[str] = None,
                 valid_from_column: str = bitemporal.VALID_FROM,
                 valid_to_column: str = bitemporal.VALID_TO,
                 recorded_column: str = bitemporal.RECORDED_AT,
                 change_column: str = bitemporal.CHANGE,
                 time_unit: str = "ns"):
        super().__init__(c)

        if (not principal_column):
            raise ValueError("principal_column is required")

        profile_columns = list(DEFAULT_PROFILE_COLUMNS) if profile_columns is None else list(profile_columns)

        if (len(profile_columns) == 0):
            raise ValueError("profile_columns must name at least one attribute")

        self._principal_column = principal_column
        self._group_column = group_column
        self._profile_columns = profile_columns
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
        return "tc0-identity"

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
        Write the context columns for every identity record.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming identity records.

        Returns
        -------
        The input message, with the context columns populated.

        Raises
        ------
        KeyError
            If the principal column, the valid-from column, the recorded column, or a profile column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._principal_column, self._valid_from_column, self._recorded_column]
            required += list(self._profile_columns)
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC0IdentityStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            count = len(df)
            principals = to_host_list(df, self._principal_column)
            groups = (to_host_list(df, self._group_column) if self._group_column in df.columns else [None] * count)
            profile = {name: to_host_list(df, name) for name in self._profile_columns}

            kinds = []
            key_parts = []
            values = []

            for position in range(count):
                group = normalize_text(groups[position])

                if (group is None):
                    kinds.append(PROFILE)
                    key_parts.append(())
                    values.append({name: profile[name][position] for name in self._profile_columns})
                else:
                    kinds.append(bitemporal.MEMBERSHIP)
                    key_parts.append((group, ))
                    values.append({GROUP_ATTRIBUTE: group})

            (columns, refused) = bitemporal.context_columns(
                kinds,
                principals,
                key_parts,
                values,
                to_host_list(df, self._valid_from_column),
                (to_host_list(df, self._valid_to_column) if self._valid_to_column in df.columns else [None] * count),
                to_host_list(df, self._recorded_column),
                (to_host_list(df, self._change_column) if self._change_column in df.columns else [None] * count),
                time_unit=self._time_unit)

            write_context_columns(df, columns)
            log_refusals("TC0IdentityStage", refused, count)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
