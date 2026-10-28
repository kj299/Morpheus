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
The network half of the TC-0 context store: what a port, a device and a VLAN were commissioned as, what they are for
and who owns them, over time.

Each input row is one record from a network inventory, naming the kind of object it is about in `object_kind` --
`port`, `device` or `vlan` -- and carrying the interval it holds for and the instant the inventory recorded it. The
stage writes it as a version of a fact in `morpheus.utils.bitemporal`'s sense, the columns `context:network` carries
and a `BitemporalStore` is rebuilt from. Gap G70 in the retrospective.

**Each object is keyed as the telemetry keys it.** A port is `site_id:device_id:port_id`, the `entity_key` layer 1
composes; a device is `site_id:device_id`, the `device_key` of the device class; a VLAN is `site_id:vlan_id`, the
`vlan_key` layer 2 counts on. All three are composed with {py:func}`~morpheus.utils.entity_key.compose_key`, so
`TC0EnrichStage` finds a port's record from the key the event already carries. A port renamed after it was
recorded is still found: layer 1 keys a renamed port on the name its `ifIndex` was first seen with
(`morpheus.utils.identity_anchor`), which is the name the inventory recorded it under.

**The commissioning level is the reference a baseline cannot be.** `TC1OpticalStage` compares a port's receive light
with its own recent median, which follows a slow decline down and so never sees one. The level the link was
commissioned at does not move, and a reading a decibel and a half below it is a link that has lost light however
slowly it lost it; R-D-L1-010 reads that. An optic replaced on purpose is recommissioned, which is a new version of
the port's record from the moment of the replacement.

**A record of an unknown kind, or without every part of its key, is refused** and kept in the output, marked, as a
record without the instant its source recorded it is. It enters no store.
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
from morpheus.utils.entity_key import normalize_text

PORT = "port"
DEVICE = "device"
VLAN = "vlan"

DEFAULT_KEY_COLUMNS = {
    PORT: ("site_id", "device_id", "port_id"),
    DEVICE: ("site_id", "device_id"),
    VLAN: ("site_id", "vlan_id"),
}
"""The columns composing each kind's key, in the order the telemetry composes it."""

DEFAULT_KIND_ATTRIBUTES = {
    PORT: ("commissioning_rx_dbm", "designation", "owner", "criticality"),
    DEVICE: ("device_role", "owner", "owning_team", "criticality"),
    VLAN: ("designation", "owner", "criticality"),
}
"""What an inventory records about each kind of object."""


@register_stage("tc0-network", ignore_args=["key_columns", "kind_attributes"])
class TC0NetworkStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write each port, device and VLAN record as a bitemporal version keyed as the telemetry keys the object.

    The stage is stateless: each row's output depends on that row alone.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    kind_column : str, default = "object_kind"
        Column naming the kind of object each record is about: `port`, `device` or `vlan`.
    key_columns : dict, optional
        Per kind, the columns composing its key. Defaults to `DEFAULT_KEY_COLUMNS`.
    kind_attributes : dict, optional
        Per kind, the attributes a record of that kind carries. Defaults to `DEFAULT_KIND_ATTRIBUTES`. A column one
        kind names may be absent from rows of another, and is read only for the kind that names it.
    valid_from_column : str, default = "valid_from"
        Column holding when the fact started to hold.
    valid_to_column : str, default = "valid_to"
        Column holding when it stopped. Null for a fact that still holds. The column may be absent.
    recorded_column : str, default = "recorded_at"
        Column holding when the inventory recorded the record. Required on every row.
    change_column : str, default = "change"
        Column holding `assert` or `retract`. The column may be absent, in which case every row asserts.
    time_unit : str, default = "ns"
        Unit for numeric instants. Ignored for datetime and string columns.
    """

    def __init__(self,
                 c: Config,
                 kind_column: str = "object_kind",
                 key_columns: dict = None,
                 kind_attributes: dict = None,
                 valid_from_column: str = bitemporal.VALID_FROM,
                 valid_to_column: str = bitemporal.VALID_TO,
                 recorded_column: str = bitemporal.RECORDED_AT,
                 change_column: str = bitemporal.CHANGE,
                 time_unit: str = "ns"):
        super().__init__(c)

        if (not kind_column):
            raise ValueError("kind_column is required")

        key_columns = dict(DEFAULT_KEY_COLUMNS if key_columns is None else key_columns)
        kind_attributes = dict(DEFAULT_KIND_ATTRIBUTES if kind_attributes is None else kind_attributes)

        if (set(key_columns) != set(kind_attributes)):
            raise ValueError(f"key_columns names kinds {sorted(key_columns)} and kind_attributes names "
                             f"{sorted(kind_attributes)}; every kind needs both")

        for (kind, columns) in key_columns.items():
            if (len(columns) == 0):
                raise ValueError(f"key_columns[{kind!r}] must name at least one column")

        for (kind, names) in kind_attributes.items():
            if (len(names) == 0):
                raise ValueError(f"kind_attributes[{kind!r}] must name at least one attribute")

        self._kind_column = kind_column
        self._key_columns = {kind: tuple(columns) for (kind, columns) in key_columns.items()}
        self._kind_attributes = {kind: tuple(names) for (kind, names) in kind_attributes.items()}
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
        return "tc0-network"

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
        Write the context columns for every network object record.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming inventory records.

        Returns
        -------
        The input message, with the context columns populated.

        Raises
        ------
        KeyError
            If the kind, valid-from or recorded column is absent, or a key or attribute column a kind present in the
            batch needs.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._kind_column, self._valid_from_column, self._recorded_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC0NetworkStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            count = len(df)
            kinds = [normalize_text(kind) for kind in to_host_list(df, self._kind_column)]
            kinds = [None if kind is None else kind.lower() for kind in kinds]
            present = sorted({kind for kind in kinds if kind in self._key_columns})
            needed = {column for kind in present for column in self._key_columns[kind] + self._kind_attributes[kind]}
            missing = sorted(column for column in needed if column not in df.columns)

            if (len(missing) > 0):
                raise KeyError(f"TC0NetworkStage requires columns {missing} for the kinds {present} in this batch, "
                               f"which are not present in the DataFrame. Available columns: {sorted(df.columns)}")

            values = {column: to_host_list(df, column) for column in sorted(needed)}
            entities = []
            attributes = []

            for (position, kind) in enumerate(kinds):
                if (kind not in self._key_columns):
                    # No kind this stage knows, so no key: refused as a record without an entity.
                    entities.append(None)
                    attributes.append({})
                    continue

                entities.append(compose_key([values[column][position] for column in self._key_columns[kind]]))
                attributes.append({name: values[name][position] for name in self._kind_attributes[kind]})

            (columns, refused) = bitemporal.context_columns(
                [kind if kind is not None else "unknown" for kind in kinds],
                entities, [()] * count,
                attributes,
                to_host_list(df, self._valid_from_column),
                (to_host_list(df, self._valid_to_column) if self._valid_to_column in df.columns else [None] * count),
                to_host_list(df, self._recorded_column),
                (to_host_list(df, self._change_column) if self._change_column in df.columns else [None] * count),
                time_unit=self._time_unit)

            write_context_columns(df, columns)
            log_refusals("TC0NetworkStage", refused, count)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
