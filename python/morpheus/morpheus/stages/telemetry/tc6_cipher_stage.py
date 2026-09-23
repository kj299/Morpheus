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
Whether a pair's negotiated cipher suite is weaker than the weakest that pair has previously settled on.

R-B-L6-004 catches an active downgrade: something sitting between two endpoints that strips the strong suites
from the client's offer so the negotiation lands somewhere the attacker can work with. What makes it tractable
is that the comparison is against the pair's own floor rather than against a policy. An estate running a fleet
that spans two decades of operating systems has no single acceptable cipher, and a global threshold either
alerts on every legacy appliance or sits low enough to miss a downgrade to it.

**The reference is a minimum, not a mode**, which is the one design decision here worth arguing about. A pair
that negotiates a modern suite nine times in ten has a mode of that suite, and the tenth, weaker, entirely
routine negotiation sits below it -- so a mode would report the estate's own variation as an attack. The floor
reports only what the pair has never done before.

**Strength is an ordinal from an explicit table.** {py:mod}`~morpheus.utils.cipher_strength` holds it, because
there is no field in the handshake that says how strong a suite is and the ordering is a judgement that has to
be written down and maintained. A suite the table does not recognize gets no rank, contributes nothing to the
floor, and is counted on the row instead: `cipher_unrecognized` is how an estate discovers that its feed needs
an entry, rather than discovering it from a rule that quietly never fires.

**An unrecognized suite is kept out of the floor as well as out of the comparison.** Letting it in under any
default would set the floor to that default, and every subsequent negotiation would be measured against a
number nobody chose.
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
from morpheus.utils.cipher_strength import rank as cipher_rank
from morpheus.utils.cipher_strength import tier_name
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import compose_key
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.established_value import DEFAULT_MIN_SAMPLES
from morpheus.utils.established_value import ValueHistoryTracker
from morpheus.utils.established_value import minimum_of

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 30 * 24 * 3600
"""Trailing window the floor is taken over. Thirty days, which is what the layer 6 rules name."""

PAIR_KEY = "tls_pair_key"
CIPHER_RANK = "cipher_rank"
CIPHER_TIER = "cipher_tier"
CIPHER_FLOOR = "cipher_floor"
CIPHER_FLOOR_TIER = "cipher_floor_tier"
CIPHER_DOWNGRADED = "cipher_downgraded"
CIPHER_MATURE = "cipher_mature"
CIPHER_SATURATED = "cipher_saturated"
CIPHER_UNRECOGNIZED = "cipher_unrecognized"


@register_stage("tc6-cipher", ignore_args=["key_columns"])
class TC6CipherStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Rank each negotiated cipher suite and compare it against its pair's own historical floor.

    The stage is stateful across messages and must run single-engine, or sharded by the same key the floor is
    kept on.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    key_columns : list of str, optional
        Columns composing the pair the floor is kept for. Defaults to `["src_ip", "dst_ip"]`, which is the pair
        the rule names. Directed, because a downgrade is something done to one direction of a conversation.
    cipher_column : str, default = "cipher_suite"
        Column holding the negotiated suite.
    time_column : str, default = "event_time"
        Column holding the handshake's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    window_seconds : int, default = 2592000
        Trailing window the floor is taken over.
    min_samples : int, default = 5
        Prior handshakes required before a floor is published. Below it a floor is whatever arrived first, and
        every stronger negotiation afterwards would read as though the pair had improved.
    max_samples : int, default = 512
        Handshakes retained per pair regardless of the window.
    """

    def __init__(self,
                 c: Config,
                 key_columns: list[str] = None,
                 cipher_column: str = "cipher_suite",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 max_samples: int = 512):
        super().__init__(c)

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        key_columns = ["src_ip", "dst_ip"] if key_columns is None else list(key_columns)

        if (len(key_columns) == 0):
            raise ValueError("key_columns must name at least one column")

        self._key_columns = key_columns
        self._cipher_column = cipher_column
        self._time_column = time_column
        self._time_unit = time_unit

        self._tracker = ValueHistoryTracker(reduction=minimum_of,
                                            window_ns=window_seconds * NS_PER_SECOND,
                                            min_samples=min_samples,
                                            max_samples=max_samples)

        self._needed_columns[PAIR_KEY] = TypeId.STRING
        self._needed_columns[CIPHER_TIER] = TypeId.STRING
        self._needed_columns[CIPHER_FLOOR_TIER] = TypeId.STRING
        self._needed_columns[CIPHER_RANK] = TypeId.INT64
        self._needed_columns[CIPHER_FLOOR] = TypeId.INT64

        for column in (CIPHER_DOWNGRADED, CIPHER_MATURE, CIPHER_SATURATED, CIPHER_UNRECOGNIZED):
            self._needed_columns[column] = TypeId.BOOL8

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc6-cipher"

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
        Write each handshake's strength tier, its pair's floor, and whether this negotiation fell below it.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming handshake records.

        Returns
        -------
        The input message, with the cipher columns populated.

        Raises
        ------
        KeyError
            If a key column, the cipher column, or the time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = list(self._key_columns) + [self._cipher_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC6CipherStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            key_parts = {name: to_host_list(df, name) for name in self._key_columns}
            suites = to_host_list(df, self._cipher_column)
            raw_times = to_host_list(df, self._time_column)

            keys: list = []
            ranks: list = []
            tiers: list = []
            floors: list = []
            floor_tiers: list = []
            downgraded: list = []
            mature: list = []
            saturated: list = []
            unrecognized: list = []
            unknown_names: set = set()
            unordered = 0

            for (position, raw_suite) in enumerate(suites):
                key = compose_key([normalize_text(key_parts[name][position]) for name in self._key_columns])
                suite = normalize_text(raw_suite)
                rank = cipher_rank(suite)

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                keys.append(key)
                ranks.append(rank)
                tiers.append(tier_name(rank))
                unrecognized.append(suite is not None and rank is None)

                if (suite is not None and rank is None):
                    unknown_names.add(suite)

                # An unranked suite is kept out of the floor as well as out of the comparison. Admitting it
                # under any default would set the floor to that default and measure every later negotiation
                # against a number nobody chose.
                if (key is None or rank is None or event_time_ns is None):
                    floors.append(None)
                    floor_tiers.append(None)
                    downgraded.append(None)
                    mature.append(False)
                    saturated.append(False)
                    continue

                result = self._tracker.observe(key, event_time_ns, rank)

                floors.append(result.reference)
                floor_tiers.append(tier_name(result.reference))
                downgraded.append(None if result.reference is None else rank < result.reference)
                mature.append(result.mature)
                saturated.append(result.saturated)
                unordered += int(result.out_of_order)

            assign_str_column(df, PAIR_KEY, keys)
            assign_nullable_int_column(df, CIPHER_RANK, ranks)
            assign_str_column(df, CIPHER_TIER, tiers)
            assign_nullable_int_column(df, CIPHER_FLOOR, floors)
            assign_str_column(df, CIPHER_FLOOR_TIER, floor_tiers)
            assign_nullable_bool_column(df, CIPHER_DOWNGRADED, downgraded)
            assign_nullable_bool_column(df, CIPHER_MATURE, mature)
            assign_nullable_bool_column(df, CIPHER_SATURATED, saturated)
            assign_nullable_bool_column(df, CIPHER_UNRECOGNIZED, unrecognized)

            if (len(unknown_names) > 0):
                logger.warning(
                    "TC6CipherStage did not recognize %d cipher suite name(s), for example %r. They were ranked "
                    "as nothing rather than guessed at, so R-B-L6-004 cannot fire on them. Add them to "
                    "morpheus.utils.cipher_strength, or translate the feed's naming scheme at the collector.",
                    len(unknown_names),
                    sorted(unknown_names)[0])

            if (unordered > 0):
                logger.warning(
                    "TC6CipherStage saw %d of %d handshakes arrive no later than their pair's previous one; "
                    "they did not enter the floor.",
                    unordered,
                    len(suites))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
