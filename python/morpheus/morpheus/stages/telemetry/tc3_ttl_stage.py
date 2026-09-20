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
Each flow's IP time-to-live against what its source usually sends, which is R-B-L3-004's whole input.

{py:mod}`~morpheus.utils.ttl_profile` holds the arithmetic and the four decisions inside it, including why a
shift of exactly one hop counts when the guide's own wording would have excluded it.

Two things belong at the pipeline level rather than in the primitive.

**A flow record's TTL is one packet's TTL.** NetFlow and IPFIX report the value from the first packet of the
flow, or from the last, or the minimum, depending on the exporter, and none of them says which. The feature is
still sound -- whatever the exporter picks, it picks it consistently, so a change in the reported value is a
change in the path -- but an estate mixing exporters will see shifts that are exporter differences rather than
interpositions. Key the profile per collector as well as per source if that is the case, which the `key_columns`
parameter allows without a code change.

**A missing TTL is left alone rather than defaulted.** Cloud flow logs mostly do not carry the field at all, and
a stage that substituted 64 would build a reference out of its own assumption and then report every real packet
as a shift away from it.
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
from morpheus.utils.entity_key import compose_key
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.ttl_profile import DEFAULT_MIN_SAMPLES
from morpheus.utils.ttl_profile import DEFAULT_MIN_SHIFT
from morpheus.utils.ttl_profile import MAX_TTL
from morpheus.utils.ttl_profile import NS_PER_SECOND
from morpheus.utils.ttl_profile import TtlProfileTracker

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 24 * 3600
"""Trailing window the reference is taken over."""

TTL_ESTABLISHED = "ip_ttl_established"
TTL_SHIFT = "ip_ttl_shift"
TTL_DISTINCT = "ip_ttl_distinct"
TTL_SHIFTED = "ip_ttl_shifted"
TTL_MATURE = "ip_ttl_mature"
TTL_SATURATED = "ip_ttl_saturated"


@register_stage("tc3-ttl", ignore_args=["key_columns"])
class TC3TtlStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Compare each flow's TTL against its source's established value.

    The stage is stateful across messages and must run single-engine, or sharded by the same key the profile is
    kept on.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    key_columns : list of str, optional
        Columns composing the entity the reference is kept for. Defaults to `["src_ip"]`. An estate whose
        exporters disagree about which packet's TTL a flow reports should add the collector, so a profile is per
        source per exporter and an exporter difference stops reading as a path change.
    ttl_column : str, default = "ip_ttl"
        Column holding the observed time-to-live.
    time_column : str, default = "event_time"
        Column holding the flow's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    window_seconds : int, default = 86400
        Trailing window the reference is taken over.
    min_samples : int, default = 5
        Prior flows required before a reference is published.
    min_shift : int, default = 1
        Hops of difference that count as a shift. One, because one is what an interposed device adds.
    max_samples : int, default = 512
        Flows retained per entity regardless of the window.
    """

    def __init__(self,
                 c: Config,
                 key_columns: list[str] = None,
                 ttl_column: str = "ip_ttl",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 min_shift: int = DEFAULT_MIN_SHIFT,
                 max_samples: int = 512):
        super().__init__(c)

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        key_columns = ["src_ip"] if key_columns is None else list(key_columns)

        if (len(key_columns) == 0):
            raise ValueError("key_columns must name at least one column")

        self._key_columns = key_columns
        self._ttl_column = ttl_column
        self._time_column = time_column
        self._time_unit = time_unit

        self._tracker = TtlProfileTracker(window_ns=window_seconds * NS_PER_SECOND,
                                          min_samples=min_samples,
                                          min_shift=min_shift,
                                          max_samples=max_samples)

        self._needed_columns[TTL_ESTABLISHED] = TypeId.INT64
        self._needed_columns[TTL_SHIFT] = TypeId.INT64
        self._needed_columns[TTL_DISTINCT] = TypeId.INT64
        self._needed_columns[TTL_SHIFTED] = TypeId.BOOL8
        self._needed_columns[TTL_MATURE] = TypeId.BOOL8
        self._needed_columns[TTL_SATURATED] = TypeId.BOOL8

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc3-ttl"

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
    def _ttl_of(value: typing.Any) -> typing.Optional[int]:
        """The TTL as an integer in the field's range, or `None` for anything else.

        A float that is not whole is a parsing fault rather than a hop count, and is refused rather than
        truncated: the reference is a mode, and a truncated value would be a different mode from the real one.
        """
        try:
            # `value != value` is the null check for a float NaN, which `is None` does not catch.
            if (value is None or value != value):  # pylint: disable=comparison-with-itself
                return None

            number = float(value)

            if (number != int(number)):
                return None

            number = int(number)

            return number if 0 <= number <= MAX_TTL else None
        except (TypeError, ValueError):
            return None

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the established TTL, the per-flow shift, and the flags around them.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming flow records.

        Returns
        -------
        The input message, with the TTL columns populated.

        Raises
        ------
        KeyError
            If a key column, the TTL column, or the time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = list(self._key_columns) + [self._ttl_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC3TtlStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            key_parts = {name: to_host_list(df, name) for name in self._key_columns}
            raw_ttls = to_host_list(df, self._ttl_column)
            raw_times = to_host_list(df, self._time_column)

            established: list = []
            shifts: list = []
            distinct: list = []
            shifted: list = []
            mature: list = []
            saturated: list = []
            unusable = 0
            unordered = 0

            for (position, raw_ttl) in enumerate(raw_ttls):
                key = compose_key([normalize_text(key_parts[name][position]) for name in self._key_columns])
                ttl = self._ttl_of(raw_ttl)

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                if (key is None or ttl is None or event_time_ns is None):
                    unusable += 1
                    established.append(None)
                    shifts.append(None)
                    distinct.append(None)
                    shifted.append(None)
                    mature.append(False)
                    saturated.append(False)
                    continue

                result = self._tracker.observe(key, event_time_ns, ttl)

                established.append(result.established)
                shifts.append(result.shift)
                distinct.append(result.distinct)
                shifted.append(result.shifted)
                mature.append(result.mature)
                saturated.append(result.saturated)
                unordered += int(result.out_of_order)

            assign_nullable_int_column(df, TTL_ESTABLISHED, established)
            assign_nullable_int_column(df, TTL_SHIFT, shifts)
            assign_nullable_int_column(df, TTL_DISTINCT, distinct)
            assign_nullable_bool_column(df, TTL_SHIFTED, shifted)
            df[TTL_MATURE] = mature
            df[TTL_SATURATED] = saturated

            if (unusable > 0):
                logger.info(
                    "TC3TtlStage left %d of %d flows unprofiled for want of a key, a usable event time, or a "
                    "time-to-live in the field's range. Cloud flow logs mostly omit the field, and substituting "
                    "a default would build a reference out of an assumption.",
                    unusable,
                    len(raw_ttls))

            if (unordered > 0):
                logger.warning(
                    "TC3TtlStage saw %d of %d flows out of order; they did not join any reference. The "
                    "reference is over prior flows, and which flows are prior is what an out-of-order arrival "
                    "disagrees about.",
                    unordered,
                    len(raw_ttls))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
