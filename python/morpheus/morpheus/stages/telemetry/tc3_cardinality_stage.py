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
How many distinct places a layer 3 source reached, how many reached it, and across how many ports.

Three counts, the same question with the entity and the value swapped, exactly as the TC-2 cardinality features
are. Each is diagnostic in a different direction:

- **`dsts_per_src`** is fan-out, and it is the scan signal R-B-L3-001 and R-P-L3-005 both read. A workstation
  talks to a handful of servers; one talking to four hundred addresses in an hour is enumerating them.
- **`srcs_per_dst`** is fan-in, and it is the same arithmetic pointed the other way. A rise on a server is
  ordinary and a rise on a workstation is not, which is why the count is published rather than thresholded here.
- **`dst_ports_per_src`** separates the two shapes a scan comes in. Many addresses on one port is a hunt for a
  service; many ports on one address is a hunt for a way in, and an operator answers them differently.

**Sampling is the trap, and this stage cannot detect it.** A 1:1000 sampled flow feed turns every count here into
a sample statistic with enormous variance at exactly the low true counts that matter, and nothing in the record
says whether sampling happened. The guide is explicit that fan-out counting needs an unsampled feed at the
aggregation layer; a deployment that cannot supply one should read the ratios from
{py:class}`~morpheus.stages.telemetry.tc3_reach_stage.TC3ReachStage` instead, which survive sampling, and should
not run R-B-L3-001 at all.

Each count carries `<name>_first_in_window`, saying this value was absent from the window before this flow, and
`<name>_saturated`, saying the per-entity sample cap is binding so the count is a floor. Saturation is not a
footnote at this layer: a scan is both the thing the feature exists to see and the thing that would exhaust the
window, so a count that stopped rising has to say whether the source stopped or the tracker did.
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
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.distinct_window import NS_PER_SECOND
from morpheus.utils.distinct_window import DistinctWindowTracker
from morpheus.utils.entity_key import normalize_text

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 3600
"""Trailing window the counts cover. An hour, which is the unit R-B-L3-001 compares against its own history."""

DSTS_PER_SRC = "dsts_per_src"
SRCS_PER_DST = "srcs_per_dst"
DST_PORTS_PER_SRC = "dst_ports_per_src"

COUNTS = (DSTS_PER_SRC, SRCS_PER_DST, DST_PORTS_PER_SRC)
"""The three cardinality questions the TC-3 telemetry class asks."""


@register_stage("tc3-cardinality")
class TC3CardinalityStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Count distinct layer 3 counterparties and ports per entity over a trailing window.

    The current flow is counted inside its own window, so a threshold trips on the flow that crosses it rather
    than on the one after.

    The stage is stateful across messages and must run single-engine. Sharding splits cleanly for the two counts
    keyed on the source and breaks for the one keyed on the destination: every flow reaching an address has to
    arrive at one instance for `srcs_per_dst` to mean anything, and sharding by source scatters exactly those.
    Run fan-in unsharded, or shard it by destination, which is determinism control 4 applied to the other key.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    src_column : str, default = "src_ip"
        Column holding the source address.
    dst_column : str, default = "dst_ip"
        Column holding the destination address.
    dst_port_column : str, default = "dst_port"
        Column holding the destination port.
    time_column : str, default = "event_time"
        Column holding the flow's event time. Event time, never ingest time.
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
                 src_column: str = "src_ip",
                 dst_column: str = "dst_ip",
                 dst_port_column: str = "dst_port",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 max_samples: int = 4096):
        super().__init__(c)

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        self._src_column = src_column
        self._dst_column = dst_column
        self._dst_port_column = dst_port_column
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

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc3-cardinality"

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
        Write the three distinct counts and their flags.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming flow records.

        Returns
        -------
        The input message, with the cardinality columns populated.

        Raises
        ------
        KeyError
            If the source, destination, port or time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._src_column, self._dst_column, self._dst_port_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC3CardinalityStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            sources = to_host_list(df, self._src_column)
            destinations = to_host_list(df, self._dst_column)
            dst_ports = to_host_list(df, self._dst_port_column)
            raw_times = to_host_list(df, self._time_column)

            counts: dict[str, list] = {name: [] for name in COUNTS}
            first: dict[str, list] = {name: [] for name in COUNTS}
            saturated: dict[str, list] = {name: [] for name in COUNTS}
            unordered = 0
            keyless = 0

            for (position, raw_source) in enumerate(sources):
                source = normalize_text(raw_source)
                destination = normalize_text(destinations[position])
                # The port is normalized through the shared rule rather than through `str`, so a column widened to
                # float by one null row does not fork port 443 into `443` and `443.0` and restart its count.
                dst_port = normalize_text(dst_ports[position])

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                # Each count needs its own entity and its own value, and a flow missing either contributes to
                # neither rather than to a fabricated key.
                pairs = {
                    DSTS_PER_SRC: (source, destination),
                    SRCS_PER_DST: (destination, source),
                    DST_PORTS_PER_SRC: (source, dst_port),
                }

                missing_here = event_time_ns is None or any(part is None for pair in pairs.values() for part in pair)

                if (missing_here):
                    keyless += 1

                out_of_order = False

                for (name, (entity, value)) in pairs.items():
                    if (event_time_ns is None or entity is None or value is None):
                        counts[name].append(None)
                        first[name].append(False)
                        saturated[name].append(False)
                        continue

                    result = self._trackers[name].observe(entity, event_time_ns, value)
                    counts[name].append(result.distinct)
                    first[name].append(result.first_in_window)
                    saturated[name].append(result.saturated)
                    out_of_order = out_of_order or result.out_of_order

                unordered += int(out_of_order)

            for name in COUNTS:
                assign_nullable_int_column(df, name, counts[name])
                assign_nullable_bool_column(df, f"{name}_first_in_window", first[name])
                df[f"{name}_saturated"] = saturated[name]

            if (keyless > 0):
                logger.warning(
                    "TC3CardinalityStage left %d of %d flows uncounted for want of an address, a port or a "
                    "usable event time. A flow with no destination is not a flow to nowhere; it is a record the "
                    "collector did not finish.",
                    keyless,
                    len(sources))

            if (unordered > 0):
                logger.warning(
                    "TC3CardinalityStage saw %d of %d flows out of order; they did not advance any window. "
                    "Shard by the counted entity and preserve per-entity ordering upstream.",
                    unordered,
                    len(sources))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
