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
What kind of places a layer 3 source reaches, and how lopsided the traffic to them is.

The counts in {py:class}`~morpheus.stages.telemetry.tc3_cardinality_stage.TC3CardinalityStage` say how many. This
says what kind, which is what turns a number into a rule. Four hundred destinations is a scan when they are
inside the estate and a busy web browser when they are not, and R-B-L3-001 requires both conditions for exactly
that reason.

**The classification comes from {py:mod}`~morpheus.parsers.ip`, not from a private copy.** It is already in
Morpheus, it is vectorized over both frame libraries, and it follows Python's `ipaddress` semantics rather than
an intuition about what looks internal. Those semantics are worth knowing before thresholding on them: loopback
and the reserved ranges are `is_private`, so "internal" here means "not routable on the public internet" rather
than "belongs to this estate". An estate whose internal space is a routable allocation it owns will find this
feature says the opposite of what it means, and should classify against its own prefix list instead.

`dst_is_reserved` and `dst_is_multicast` are kept as their own columns rather than folded into the ratio because
R-D-L3-003 is a different kind of rule: low volume, high signal, and about a single flow rather than about a
proportion. Traffic to a reserved range leaving an egress point is not a matter of degree.

**The ratio survives sampling and the counts do not.** A 1:1000 feed destroys a distinct-destination count at
the low true counts that matter and leaves a proportion roughly intact, so an estate that cannot collect
unsampled has this feature and not the other one. That is the whole of the guide's advice on the subject made
into two stages, so the choice is visible in a pipeline definition rather than buried in a rule comment.

**Byte asymmetry is `bytes_out / (bytes_in + 1)`**, the guide's own formula, and the plus one is load-bearing
rather than cosmetic: an exfiltration flow's defining shape is that nothing came back, so the denominator is
zero exactly when the feature matters most.

The destination ASN's novelty is per source and permanent, not windowed --
{py:mod}`~morpheus.utils.value_novelty` answers "has this source ever reached this network", which is the
question worth asking about a first contact.
"""

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
from morpheus.parsers import ip
from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.ratio_window import NS_PER_SECOND
from morpheus.utils.ratio_window import DEFAULT_MIN_DENOMINATOR
from morpheus.utils.ratio_window import RatioWindowTracker
from morpheus.utils.value_novelty import ValueNoveltyTracker

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 3600
"""Trailing window the internal-destination proportion is taken over."""

INTERNAL_RATIO = "internal_dst_ratio"
INTERNAL_COUNT = "internal_dsts_in_window"
RATIO_SATURATED = "internal_dst_ratio_saturated"
DST_PRIVATE = "dst_is_private"
DST_RESERVED = "dst_is_reserved"
DST_MULTICAST = "dst_is_multicast"
BYTE_ASYMMETRY = "byte_asymmetry"
ASN_FIRST_SEEN = "dst_asn_first_seen"

DEFAULT_ASN_COLUMN = "bgp_as_dst"
"""Where the destination's autonomous system number is expected, as the TC-3 required field list names it."""


@register_stage("tc3-reach")
class TC3ReachStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Classify each flow's destination and describe the mix of destinations a source has been reaching.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    src_column : str, default = "src_ip"
        Column holding the source address.
    dst_column : str, default = "dst_ip"
        Column holding the destination address.
    asn_column : str, default = "bgp_as_dst"
        Column holding the destination's autonomous system number. Absent, the novelty flag is left null rather
        than reported as novel, because a missing network is not a new one. It is also null on a source's first
        flow, where the question has no answer yet.
    bytes_out_column : str, default = "bytes_out"
        Column holding bytes sent by the source.
    bytes_in_column : str, default = "bytes_in"
        Column holding bytes returned to it.
    time_column : str, default = "event_time"
        Column holding the flow's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    window_seconds : int, default = 3600
        Trailing window the internal-destination proportion covers.
    min_denominator : int, default = 10
        Flows required in the window before a proportion is published. Below it a ratio is one flow's opinion:
        a source's first destination makes it either wholly internal or wholly external, and a rule reading that
        fires on every host's first minute.
    max_samples : int, default = 4096
        Flows retained per source for the proportion.
    """

    def __init__(self,
                 c: Config,
                 src_column: str = "src_ip",
                 dst_column: str = "dst_ip",
                 asn_column: str = DEFAULT_ASN_COLUMN,
                 bytes_out_column: str = "bytes_out",
                 bytes_in_column: str = "bytes_in",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 min_denominator: int = DEFAULT_MIN_DENOMINATOR,
                 max_samples: int = 4096):
        super().__init__(c)

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        self._src_column = src_column
        self._dst_column = dst_column
        self._asn_column = asn_column
        self._bytes_out_column = bytes_out_column
        self._bytes_in_column = bytes_in_column
        self._time_column = time_column
        self._time_unit = time_unit

        self._ratios = RatioWindowTracker(window_ns=window_seconds * NS_PER_SECOND,
                                          min_denominator=min_denominator,
                                          max_samples=max_samples)
        self._novelty = ValueNoveltyTracker(field_names=[asn_column])

        self._needed_columns[INTERNAL_RATIO] = TypeId.FLOAT64
        self._needed_columns[INTERNAL_COUNT] = TypeId.INT64
        self._needed_columns[RATIO_SATURATED] = TypeId.BOOL8
        self._needed_columns[DST_PRIVATE] = TypeId.BOOL8
        self._needed_columns[DST_RESERVED] = TypeId.BOOL8
        self._needed_columns[DST_MULTICAST] = TypeId.BOOL8
        self._needed_columns[BYTE_ASYMMETRY] = TypeId.FLOAT64
        self._needed_columns[ASN_FIRST_SEEN] = TypeId.BOOL8

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc3-reach"

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
    def _classify(df, column: str) -> dict[str, list]:
        """Run the shared IP classifiers over the destination column, as series rather than row by row."""
        series = df[column]

        return {
            DST_PRIVATE: list(ip.is_private(series)),
            DST_RESERVED: list(ip.is_reserved(series)),
            DST_MULTICAST: list(ip.is_multicast(series)),
        }

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Classify destinations, track the internal proportion, and write the asymmetry and novelty columns.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming flow records.

        Returns
        -------
        The input message, with the reach columns populated.

        Raises
        ------
        KeyError
            If the source, destination or time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._src_column, self._dst_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC3ReachStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            classified = self._classify(df, self._dst_column)
            sources = to_host_list(df, self._src_column)
            raw_times = to_host_list(df, self._time_column)

            rows = len(sources)
            asns = to_host_list(df, self._asn_column) if self._asn_column in df.columns else [None] * rows
            out_bytes = (to_host_list(df, self._bytes_out_column) if self._bytes_out_column in df.columns else [None] *
                         rows)
            in_bytes = (to_host_list(df, self._bytes_in_column) if self._bytes_in_column in df.columns else [None] *
                        rows)

            ratios: list = []
            internal_counts: list = []
            ratio_saturated: list = []
            asymmetry: list = []
            asn_novel: list = []

            for position in range(rows):
                source = normalize_text(sources[position])

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                private = classified[DST_PRIVATE][position]

                if (source is None or event_time_ns is None or private is None):
                    ratios.append(None)
                    internal_counts.append(None)
                    ratio_saturated.append(False)
                else:
                    result = self._ratios.observe(source, event_time_ns, bool(private))
                    ratios.append(result.ratio)
                    internal_counts.append(result.numerator)
                    ratio_saturated.append(result.saturated)

                asn = normalize_text(asns[position])

                if (source is None or event_time_ns is None or asn is None):
                    # A flow with no ASN is not a flow to a new network. Reporting novelty here would make a
                    # collector that stopped populating the field look like a host that started roaming.
                    asn_novel.append(None)
                else:
                    seen = self._novelty.observe(source, event_time_ns, {self._asn_column: asn})
                    # Left as the tracker reports it, nulls and all. A source's very first flow has no answer to
                    # "has it reached this network before", and coercing that to False would say it had.
                    asn_novel.append(seen.first_seen.get(self._asn_column))

                asymmetry.append(self._asymmetry(out_bytes[position], in_bytes[position]))

            df[INTERNAL_RATIO] = ratios
            assign_nullable_int_column(df, INTERNAL_COUNT, internal_counts)
            df[RATIO_SATURATED] = ratio_saturated
            df[BYTE_ASYMMETRY] = asymmetry
            assign_nullable_bool_column(df, ASN_FIRST_SEEN, asn_novel)

            for (column, values) in classified.items():
                assign_nullable_bool_column(df, column, values)

        return message

    @staticmethod
    def _asymmetry(out_value: typing.Any, in_value: typing.Any) -> typing.Optional[float]:
        """`bytes_out / (bytes_in + 1)`, or `None` where either side is missing.

        Null rather than zero for a missing byte count, because zero is the value for a flow that sent nothing,
        and a feature that reported the same figure for "sent nothing" and "the collector omitted the field"
        would make an incomplete record look like a quiet one.
        """
        if (out_value is None or in_value is None):
            return None

        try:
            (sent, received) = (float(out_value), float(in_value))
        except (TypeError, ValueError):
            return None

        # `math.isnan` rather than a self-comparison: a widened column's missing value arrives as a float NaN,
        # which `is None` does not catch and which would otherwise propagate into the ratio.
        if (math.isnan(sent) or math.isnan(received)):
            return None

        return sent / (received + 1.0)

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
