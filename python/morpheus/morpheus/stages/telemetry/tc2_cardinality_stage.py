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
"""Counts distinct layer 2 identifiers per entity over a trailing window."""

import logging
import math
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
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.distinct_window import NS_PER_SECOND
from morpheus.utils.distinct_window import DistinctWindowTracker
from morpheus.utils.entity_key import KEY_SEPARATOR
from morpheus.utils.entity_key import compose_key

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 3600
"""Trailing window distinct values are counted over, in seconds of event time."""

MACS_PER_PORT = "macs_per_port"
PORTS_PER_MAC = "ports_per_mac"
OUIS_PER_VLAN = "ouis_per_vlan"

COUNTS = (MACS_PER_PORT, PORTS_PER_MAC, OUIS_PER_VLAN)
"""The three cardinality questions the TC-2 telemetry class asks."""

PORT_KEY_SEPARATOR = KEY_SEPARATOR
"""Separator for the `site_id:switch_id:port_id` key the per-port counts group on."""


@register_stage("tc2-cardinality")
class TC2CardinalityStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Count distinct layer 2 identifiers per entity over a trailing window.

    Three of the TC-2 behavioral features are the same question with the entity and the value swapped, and each is
    diagnostic in a different direction:

    - **`macs_per_port`** — an access port serves one device, or a handful behind an IP phone. A count that climbs
      is an unauthorized hub or switch, or a MAC flood against the forwarding table.
    - **`ports_per_mac`** — one MAC belongs to one interface. On several ports at once it is spoofing; moving
      between them over time it is a device being carried around, benign in an office and much less so in a
      datacentre.
    - **`ouis_per_vlan`** — an OUI a VLAN has not carried before is an unmanaged device class appearing on a
      segment that was meant to be homogeneous.

    Each count comes with a `<name>_first_in_window` flag saying whether this particular value was absent from the
    window before this sample, which is what turns a count into an event, and a `<name>_saturated` flag saying
    whether the per-entity sample cap is binding. Saturation matters because a MAC flood is both the condition this
    feature exists to notice and the condition that would exhaust memory: when the cap binds, the count is a lower
    bound and says so rather than quietly under-reporting.

    The current sample is counted inside its own window, so a threshold trips on the row that crosses it rather
    than on the row after.

    The stage is stateful across messages and must run single-engine. Sharding is awkward here and worth thinking
    about before configuring: `macs_per_port` and `ouis_per_vlan` shard cleanly by switch, but `ports_per_mac`
    needs every sighting of a MAC to reach one instance, which sharding by switch breaks. Run that count on an
    unsharded instance, or shard it by MAC, which is determinism control 4 applied to a different key.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    mac_column : str, default = "mac_address"
        Column holding the MAC address.
    site_column : str, default = "site_id"
        Column holding the site identifier, used to build the port key.
    switch_column : str, default = "switch_id"
        Column holding the switch identifier.
    port_column : str, default = "port_id"
        Column holding the port identifier.
    vlan_column : str, default = "vlan_id"
        Column holding the VLAN identifier.
    oui_column : str, default = "oui"
        Column holding the organizationally unique identifier. Where a collector does not supply it, it is derived
        from the first three octets of the MAC address rather than being left null.
    time_column : str, default = "event_time"
        Column holding the observation's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    window_seconds : int, default = 3600
        Trailing window the counts cover.
    max_samples : int, default = 4096
        Observations retained per entity regardless of the window. When this binds the count is a lower bound and
        the row is marked saturated.
    """

    def __init__(self,
                 c: Config,
                 mac_column: str = "mac_address",
                 site_column: str = "site_id",
                 switch_column: str = "switch_id",
                 port_column: str = "port_id",
                 vlan_column: str = "vlan_id",
                 oui_column: str = "oui",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 max_samples: int = 4096):
        super().__init__(c)

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        self._mac_column = mac_column
        self._site_column = site_column
        self._switch_column = switch_column
        self._port_column = port_column
        self._vlan_column = vlan_column
        self._oui_column = oui_column
        self._time_column = time_column
        self._time_unit = time_unit

        self._trackers = {
            name: DistinctWindowTracker(window_ns=window_seconds * NS_PER_SECOND, max_samples=max_samples)
            for name in COUNTS
        }

        for name in COUNTS:
            self._needed_columns[name] = TypeId.INT64
            self._needed_columns[f"{name}_first_in_window"] = TypeId.BOOL8
            self._needed_columns[f"{name}_saturated"] = TypeId.BOOL8

        self._needed_columns["port_key"] = TypeId.STRING

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc2-cardinality"

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
    def _text(value: typing.Any) -> typing.Optional[str]:
        """Normalize a host value, collapsing every flavour of missing to `None`."""
        if (value is None):
            return None

        if (isinstance(value, float) and math.isnan(value)):
            return None

        return str(value)

    @classmethod
    def _oui(cls, supplied: typing.Any, mac: typing.Optional[str]) -> typing.Optional[str]:
        """
        Return the OUI, deriving it from the MAC where the collector did not supply one.

        Deriving rather than leaving it null keeps the VLAN count meaningful on estates whose collectors omit the
        field, which would otherwise report one distinct OUI, forever, on every segment.
        """
        value = cls._text(supplied)

        if (value is not None):
            return value.lower()

        if (mac is None):
            return None

        octets = mac.lower().replace("-", ":").split(":")

        return ":".join(octets[0:3]) if len(octets) >= 3 else None

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the three distinct counts and their flags.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming message.

        Returns
        -------
        The input message, with the cardinality columns populated.

        Raises
        ------
        KeyError
            If the MAC, port, VLAN, or time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._mac_column, self._switch_column, self._port_column, self._vlan_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC2CardinalityStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            macs = to_host_list(df, self._mac_column)
            switches = to_host_list(df, self._switch_column)
            ports = to_host_list(df, self._port_column)
            vlans = to_host_list(df, self._vlan_column)
            raw_times = to_host_list(df, self._time_column)

            sites = to_host_list(df, self._site_column) if self._site_column in df.columns else [None] * len(macs)
            ouis = to_host_list(df, self._oui_column) if self._oui_column in df.columns else [None] * len(macs)

            port_keys: list = []
            counts: dict[str, list] = {name: [] for name in COUNTS}
            first: dict[str, list] = {name: [] for name in COUNTS}
            saturated: dict[str, list] = {name: [] for name in COUNTS}
            unordered = 0
            keyless = 0

            has_site = self._site_column in df.columns

            for (position, raw_mac) in enumerate(macs):
                mac = self._text(raw_mac)
                # The same composition the layer 1 stages use for `entity_key`, so a MAC resolved to this port lands
                # on the identical string. Without a site column the key is `switch:port`, which still counts per
                # port but cannot join to layer 1. A null part yields a null key, never the string "None".
                location = (sites[position], switches[position], ports[position]) if has_site else (switches[position],
                                                                                                    ports[position])
                port_key = compose_key(location)
                port_keys.append(port_key)

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                if (event_time_ns is None):
                    for name in COUNTS:
                        counts[name].append(None)
                        first[name].append(None)
                        saturated[name].append(False)

                    unordered += 1
                    continue

                observations = {
                    MACS_PER_PORT: (port_key, mac),
                    PORTS_PER_MAC: (mac, port_key),
                    OUIS_PER_VLAN: (self._text(vlans[position]), self._oui(ouis[position], mac)),
                }

                out_of_order = False
                row_keyless = False

                for (name, (entity, value)) in observations.items():
                    if (entity is None or value is None):
                        # Nothing to count per, or nothing to count. Pooling these under a fabricated key would make
                        # every keyless row in the estate look like one very busy port, and counting an unknown port
                        # as a distinct port would make a MAC with one bad sample look like it moved.
                        counts[name].append(None)
                        first[name].append(None)
                        saturated[name].append(False)
                        row_keyless = True
                        continue

                    result = self._trackers[name].observe(entity, event_time_ns, value)
                    counts[name].append(result.distinct)
                    first[name].append(result.first_in_window)
                    saturated[name].append(result.saturated)
                    out_of_order = out_of_order or result.out_of_order

                unordered += int(out_of_order)
                keyless += int(row_keyless)

            assign_str_column(df, "port_key", port_keys)

            for name in COUNTS:
                assign_nullable_int_column(df, name, counts[name])
                assign_nullable_bool_column(df, f"{name}_first_in_window", first[name])
                df[f"{name}_saturated"] = saturated[name]

        if (keyless > 0):
            logger.warning(
                "TC2CardinalityStage saw %d of %d observations with a null MAC, port, or VLAN; the affected counts "
                "are null rather than pooled under a fabricated key.",
                keyless,
                len(macs))

        if (unordered > 0):
            logger.warning(
                "TC2CardinalityStage saw %d of %d observations out of order or without a usable event time; they "
                "did not enter any window. Preserve per-entity ordering upstream.",
                unordered,
                len(macs))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
