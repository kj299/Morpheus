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
The thirteen transport features the shipped model was trained on, rolled up per flow per time bin.

`examples/abp_pcap_detection/abp_pcap_preprocessing.py` is the reference the guide names, and its feature set is
adopted here unchanged: the five flag sums, `ppm`, `data_len`, `bpp`, `all`, and the four ratios. Three things
about how it computes them are changed deliberately, and each is a correctness point rather than a preference.

**The rollup is a running total, not a per-batch group.** The reference groups by `(rollup_time, flow_id)` inside
the message it was handed. A flow's sixty-second bin that spans two messages is therefore aggregated twice, and
the model sees two partial bins as though each were complete -- a figure that depends on how the stream happened
to be chunked, which is the defect determinism control 5 exists to catch and the same one the layer 5 scoring
adapter was found to have. This stage keeps a running total per flow per bin instead, so a bin split across
messages still ends on the right number, and the row that closes a bin carries the complete figure. A search
reads `max(...) by flow_id, rollup_time_ns` for the counts and the sums, exactly as the layer 2 and 3 detections
already do, and divides two of those maxima to recover a ratio. It must not aggregate a ratio column: the counts
rise monotonically through a bin and their maxima are therefore the bin's totals, but a running ratio is not
monotone, and `max(flow_syn_ratio)` over an ordinary handshake is 1.0 -- the figure from its first packet, when
the only flag seen so far was the SYN. The ratio columns are on the row for the model, which consumes one row at
a time; a search that summarizes a bin divides.

**The bin is half-open and labelled by its start.** The reference's rounding kernel computes `time + (secs - time
% secs)`, which labels a bin by its *end* and puts a packet landing exactly on a boundary into the *following*
bin. Every window in this fork is `[start, end)` anchored on a fixed epoch
({py:func}`~morpheus.utils.lineage.window_id_from_timestamp`), and a layer 4 bin that closed the other side of
the boundary would disagree with every other layer about which side an event falls on -- for exactly the events
most likely to be on a boundary, which the clock-skew experiment found are not rare. Joining layers is what this
design is for, so the bins are floored and half-open like everything else.

**The ratios are named with underscores.** `ackpush/all` is the reference's name and is awkward to select in SPL
and in most stores without quoting. The mapping to the model's feature list is one line and lives in
`MODEL_FEATURES`, so a deployment feeding the shipped model renames on the way in rather than carrying a field
name it has to escape everywhere else.

**`ppm` and `bpp` carry the reference's assumption with them.** It treats each record as one packet, so "packets
per minute" is really records per bin and "bytes per packet" is bytes per record. That is sound for a capture
that emits one record per packet and wrong for a flow exporter that emits one record per flow, where `ppm` would
count flows. The assumption is the reference's and is kept, because the model was trained under it; a deployment
feeding flow records rather than packet records is feeding the model something other than what it learned.
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
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.lineage import window_id_from_timestamp
from morpheus.utils.tcp_flags import FLAG_NAMES
from morpheus.utils.tcp_flags import flags_from_byte

logger = logging.getLogger(__name__)

DEFAULT_BIN_SECONDS = 60
"""Bin width. Sixty seconds, which is what the reference uses and what R-D-L4-002 names."""

FLOW_ID = "flow_id"
ROLLUP_TIME = "rollup_time_ns"

COUNTS = tuple(f"flow_{name}" for name in FLAG_NAMES)
PPM = "flow_ppm"
DATA_LEN = "flow_data_len"
BPP = "flow_bpp"
ALL_FLAGS = "flow_all"
ACKPUSH_RATIO = "flow_ackpush_ratio"
RST_RATIO = "flow_rst_ratio"
SYN_RATIO = "flow_syn_ratio"
FIN_RATIO = "flow_fin_ratio"

RATIOS = (ACKPUSH_RATIO, RST_RATIO, SYN_RATIO, FIN_RATIO)

MODEL_FEATURES = {
    "ack": "flow_ack",
    "psh": "flow_psh",
    "rst": "flow_rst",
    "syn": "flow_syn",
    "fin": "flow_fin",
    "ppm": PPM,
    "data_len": DATA_LEN,
    "bpp": BPP,
    "all": ALL_FLAGS,
    "ackpush/all": ACKPUSH_RATIO,
    "rst/all": RST_RATIO,
    "syn/all": SYN_RATIO,
    "fin/all": FIN_RATIO,
}
"""The shipped `abp-pcap-xgb` feature list, mapped to the column each one is emitted as.

Thirteen entries in the order `abp_pcap_preprocessing.py` declares them, so a deployment feeding the model can
build its input frame from this rather than from a second copy of the list that will drift.
"""


@register_stage("tc4-flow")
class TC4FlowStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Roll each record into its flow's time bin and write the running transport features.

    The stage is stateful across messages and must run single-engine, or sharded by `flow_id`, which is the one
    sharding that keeps a bin whole.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    src_ip_column : str, default = "src_ip"
        Column holding the source address.
    src_port_column : str, default = "src_port"
        Column holding the source port.
    dst_ip_column : str, default = "dst_ip"
        Column holding the destination address.
    dst_port_column : str, default = "dst_port"
        Column holding the destination port.
    flags_column : str, default = "tcp_flags"
        Column holding the TCP flags byte. When it is absent the five flag columns are read individually, which
        is what a Zeek or flow-exporter feed supplies instead of the raw byte.
    data_len_column : str, default = "data_len"
        Column holding the record's payload length.
    time_column : str, default = "event_time"
        Column holding the record's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    bin_seconds : int, default = 60
        Bin width, anchored on the Unix epoch and half-open.
    max_flows : int, default = 500000
        Flows holding a partial bin before the least recently seen is dropped.
    """

    def __init__(self,
                 c: Config,
                 src_ip_column: str = "src_ip",
                 src_port_column: str = "src_port",
                 dst_ip_column: str = "dst_ip",
                 dst_port_column: str = "dst_port",
                 flags_column: str = "tcp_flags",
                 data_len_column: str = "data_len",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 bin_seconds: int = DEFAULT_BIN_SECONDS,
                 max_flows: int = 500_000):
        super().__init__(c)

        if (bin_seconds <= 0):
            raise ValueError(f"bin_seconds must be positive, received {bin_seconds}")

        if (max_flows <= 0):
            raise ValueError(f"max_flows must be positive, received {max_flows}")

        self._src_ip_column = src_ip_column
        self._src_port_column = src_port_column
        self._dst_ip_column = dst_ip_column
        self._dst_port_column = dst_port_column
        self._flags_column = flags_column
        self._data_len_column = data_len_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._bin_ns = bin_seconds * NS_PER_SECOND
        self._max_flows = max_flows

        # One partial bin per flow. Bounded by the flow count rather than by the stream, and reset rather than
        # accumulated when a flow's next record lands in a later bin.
        self._open: dict = {}

        self._needed_columns[FLOW_ID] = TypeId.STRING
        self._needed_columns[ROLLUP_TIME] = TypeId.INT64

        for column in COUNTS + (PPM, DATA_LEN, ALL_FLAGS):
            self._needed_columns[column] = TypeId.INT64

        for column in (BPP, ) + RATIOS:
            self._needed_columns[column] = TypeId.FLOAT64

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc4-flow"

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
    def _whole(value: typing.Any) -> typing.Optional[int]:
        """A whole number, or `None` for anything else. A fractional byte count is a parsing fault."""
        try:
            if (value is None):
                return None

            number = float(value)
        except (TypeError, ValueError):
            return None

        if (number != number or number != int(number)):  # pylint: disable=comparison-with-itself
            return None

        return int(number)

    def _flags_for(self, position: int, byte_values: list, columns: dict) -> typing.Optional[dict]:
        """The five bits for one record, from the flags byte where there is one and from the columns otherwise."""
        if (byte_values is not None):
            return flags_from_byte(byte_values[position])

        bits = {}

        for name in FLAG_NAMES:
            value = self._whole(columns[name][position]) if name in columns else None

            if (value is None):
                return None

            bits[name] = 1 if value else 0

        return bits

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the flow identifier, its bin, and the running features for that bin.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming transport records.

        Returns
        -------
        The input message, with the transport columns populated.

        Raises
        ------
        KeyError
            If an address, a port, the time column, or any source of flags is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [
                self._src_ip_column,
                self._src_port_column,
                self._dst_ip_column,
                self._dst_port_column,
                self._time_column
            ]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC4FlowStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            has_byte = self._flags_column in df.columns
            flag_columns = {name: to_host_list(df, name) for name in FLAG_NAMES if name in df.columns}

            if (not has_byte and len(flag_columns) != len(FLAG_NAMES)):
                raise KeyError(f"TC4FlowStage needs either a {self._flags_column!r} byte or all of "
                               f"{list(FLAG_NAMES)}; the frame carries neither. A record whose flags cannot be "
                               f"read contributes nothing to any of the thirteen features.")

            byte_values = to_host_list(df, self._flags_column) if has_byte else None

            src_ips = to_host_list(df, self._src_ip_column)
            src_ports = to_host_list(df, self._src_port_column)
            dst_ips = to_host_list(df, self._dst_ip_column)
            dst_ports = to_host_list(df, self._dst_port_column)
            raw_times = to_host_list(df, self._time_column)

            rows = len(src_ips)
            payloads = (to_host_list(df, self._data_len_column) if self._data_len_column in df.columns else [None] *
                        rows)

            flow_ids: list = []
            bins: list = []
            counts: dict[str, list] = {column: [] for column in COUNTS}
            ppm: list = []
            data_len: list = []
            all_flags: list = []
            bpp: list = []
            ratios: dict[str, list] = {column: [] for column in RATIOS}
            unusable = 0
            unordered = 0

            for position in range(rows):
                flow_id = self._flow_id(src_ips[position], src_ports[position], dst_ips[position], dst_ports[position])

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                bits = self._flags_for(position, byte_values, flag_columns)
                flow_ids.append(flow_id)

                if (flow_id is None or event_time_ns is None or bits is None):
                    unusable += 1
                    bins.append(None)

                    for column in COUNTS:
                        counts[column].append(None)

                    ppm.append(None)
                    data_len.append(None)
                    all_flags.append(None)
                    bpp.append(None)

                    for column in RATIOS:
                        ratios[column].append(None)

                    continue

                # Floored and half-open, anchored on the epoch, exactly as every other window in this fork is.
                ordinal = window_id_from_timestamp(event_time_ns, self._bin_ns)
                state = self._open.get(flow_id)

                if (state is None or state["bin"] != ordinal):
                    if (state is not None and state["bin"] > ordinal):
                        # A record for a bin this flow has already moved past. Reopening it would produce a
                        # second partial aggregate for a bin that was already reported complete.
                        unordered += 1
                        bins.append(ordinal)

                        for column in COUNTS:
                            counts[column].append(None)

                        ppm.append(None)
                        data_len.append(None)
                        all_flags.append(None)
                        bpp.append(None)

                        for column in RATIOS:
                            ratios[column].append(None)

                        continue

                    state = {"bin": ordinal, "ppm": 0, "data_len": 0}
                    state.update({name: 0 for name in FLAG_NAMES})
                    self._open[flow_id] = state

                    while (len(self._open) > self._max_flows):
                        self._open.pop(next(iter(self._open)))

                for name in FLAG_NAMES:
                    state[name] += bits[name]

                state["ppm"] += 1
                payload = self._whole(payloads[position])
                state["data_len"] += payload if payload is not None else 0

                total = sum(state[name] for name in FLAG_NAMES)

                bins.append(ordinal * self._bin_ns)

                for (column, name) in zip(COUNTS, FLAG_NAMES):
                    counts[column].append(state[name])

                ppm.append(state["ppm"])
                data_len.append(state["data_len"])
                all_flags.append(total)
                bpp.append(state["data_len"] / state["ppm"])

                # A bin with no flags set at all has no denominator, and a ratio of zero would claim the flow
                # had flags and none of them were RST rather than that it had none.
                ratios[ACKPUSH_RATIO].append(None if total == 0 else (state["ack"] + state["psh"]) / total)
                ratios[RST_RATIO].append(None if total == 0 else state["rst"] / total)
                ratios[SYN_RATIO].append(None if total == 0 else state["syn"] / total)
                ratios[FIN_RATIO].append(None if total == 0 else state["fin"] / total)

            assign_str_column(df, FLOW_ID, flow_ids)
            assign_nullable_int_column(df, ROLLUP_TIME, bins)

            for column in COUNTS:
                assign_nullable_int_column(df, column, counts[column])

            assign_nullable_int_column(df, PPM, ppm)
            assign_nullable_int_column(df, DATA_LEN, data_len)
            assign_nullable_int_column(df, ALL_FLAGS, all_flags)
            df[BPP] = bpp

            for column in RATIOS:
                df[column] = ratios[column]

            if (unusable > 0):
                logger.warning(
                    "TC4FlowStage left %d of %d records out of their flow's bin for want of an identifier, a "
                    "usable event time, or readable flags.",
                    unusable,
                    rows)

            if (unordered > 0):
                logger.warning(
                    "TC4FlowStage saw %d of %d records arrive for a bin their flow had already passed; they "
                    "were not aggregated. Reopening a closed bin produces a second partial figure for a bin "
                    "already reported complete, which is a number that depends on how the stream was chunked.",
                    unordered,
                    rows)

        return message

    @staticmethod
    def _flow_id(src_ip: typing.Any, src_port: typing.Any, dst_ip: typing.Any, dst_port: typing.Any):
        """`src_ip:src_port=dst_ip:dst_port`, the format the reference and Part 2 both name.

        Directed, and composed from normalized parts so a port column widened to float by one null row does not
        render 443 as `443.0` and fork a flow into two.
        """
        parts = [normalize_text(value) for value in (src_ip, src_port, dst_ip, dst_port)]

        if (any(part is None for part in parts)):
            return None

        return f"{parts[0]}:{parts[1]}={parts[2]}:{parts[3]}"

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
