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
Whether a host's TLS client fingerprint is one that host has presented before.

R-B-L6-001 is, in the guide's words, one of the best value-for-effort rules in the whole set, and the reason is
the cardinality. An enterprise estate has thousands of distinct JA4 fingerprints in total and a handful per
host, because a fingerprint is a property of the TLS stack rather than of the traffic: one per browser, one per
runtime, one per agent. A host that has presented three for a year and presents a fourth this morning has
acquired a new TLS stack, and there are not many innocent ways for that to happen on a managed endpoint.

The tracker is {py:class}`~morpheus.utils.value_novelty.ValueNoveltyTracker`, which layer 1 already uses for
transceiver and neighbor substitution. The question is the same shape -- has this entity ever reported this
value -- and the module answers it with no period boundary, which matters here for the same reason it mattered
there: a fingerprint first seen at 23:59 and again at 00:01 is not two first sightings, and a distinct count
per calendar window would call it one.

**A host's first few handshakes are all novel, which is why the prior-handshake count is on the row.** Every
fingerprint is new to a host that has just appeared, so a rule reading novelty alone fires on every host the
estate has recently started seeing -- a laptop back from repair, a new starter, anything behind a fresh DHCP
lease. The corpus found this: a host alternating between two stacks it has always had was reported the first
time the second one appeared, identically to a host that had presented one stack for two hours and then acquired
another. What separates them is not the novelty but the history behind it, so `ja4_client_observations` carries
how many handshakes the host had made before this one and the rule requires a settled history.

**The rule says thirty days and this is permanent recall, bounded by a value cap rather than by a clock.** The
difference is deliberate and is the same one `TC3ReachStage` makes for destination-network novelty. A host whose
fourth fingerprint appeared thirty-one days ago is not a host with three fingerprints, and re-alerting on it
because a window rolled is the behaviour the window was supposed to prevent. The cap bounds memory instead: a
host that has genuinely presented more distinct stacks than `max_values` is already the anomaly this feature
exists to surface, and the saturation flag says so on the row.
"""

import collections
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
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import compose_key
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.value_novelty import DEFAULT_MAX_VALUES
from morpheus.utils.value_novelty import ValueNoveltyTracker

logger = logging.getLogger(__name__)

CLIENT_KEY = "tls_client_key"
JA4_FIRST_SEEN = "ja4_client_first_seen"
JA4_CHANGED = "ja4_client_changed"
JA4_DISTINCT = "ja4_client_distinct"
JA4_OBSERVATIONS = "ja4_client_observations"
JA4_SATURATED = "ja4_client_saturated"


@register_stage("tc6-fingerprint", ignore_args=["key_columns"])
class TC6FingerprintStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write whether each handshake's client fingerprint is new to the host that offered it.

    The stage is stateful across messages and must run single-engine, or sharded by the same key the history is
    kept on.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    key_columns : list of str, optional
        Columns composing the entity the fingerprint history is kept for. Defaults to `["src_ip"]`, which is
        what the rule names. An estate with long DHCP leases and a reliable layer 2 binding would do better
        keying on the resolved device, and that is what the ladder is for.
    fingerprint_column : str, default = "ja4_client"
        Column holding the client fingerprint.
    time_column : str, default = "event_time"
        Column holding the handshake's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    max_values : int, default = 64
        Distinct fingerprints recalled per host before the least recently seen is forgotten.
    max_entities : int, default = 100000
        Hosts tracked before the least recently seen is forgotten.
    """

    def __init__(self,
                 c: Config,
                 key_columns: list[str] = None,
                 fingerprint_column: str = "ja4_client",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 max_values: int = DEFAULT_MAX_VALUES,
                 max_entities: int = 100_000):
        super().__init__(c)

        key_columns = ["src_ip"] if key_columns is None else list(key_columns)

        if (len(key_columns) == 0):
            raise ValueError("key_columns must name at least one column")

        self._key_columns = key_columns
        self._fingerprint_column = fingerprint_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._max_values = max_values

        self._tracker = ValueNoveltyTracker(field_names=[fingerprint_column],
                                            max_values=max_values,
                                            max_entities=max_entities)

        # Handshakes seen per host before the current one. Kept here rather than inside the novelty tracker,
        # which answers a different question and is shared with layer 1. Bounded and evicted the same way, so
        # the two cannot disagree about which hosts they remember.
        self._observations: collections.OrderedDict = collections.OrderedDict()
        self._max_entities = max_entities

        self._needed_columns[CLIENT_KEY] = TypeId.STRING
        self._needed_columns[JA4_FIRST_SEEN] = TypeId.BOOL8
        self._needed_columns[JA4_CHANGED] = TypeId.BOOL8
        self._needed_columns[JA4_DISTINCT] = TypeId.INT64
        self._needed_columns[JA4_OBSERVATIONS] = TypeId.INT64
        self._needed_columns[JA4_SATURATED] = TypeId.BOOL8

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc6-fingerprint"

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
        Write the host key and whether its fingerprint is novel.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming handshake records.

        Returns
        -------
        The input message, with the fingerprint columns populated.

        Raises
        ------
        KeyError
            If a key column, the fingerprint column, or the time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = list(self._key_columns) + [self._fingerprint_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC6FingerprintStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            key_parts = {name: to_host_list(df, name) for name in self._key_columns}
            fingerprints = to_host_list(df, self._fingerprint_column)
            raw_times = to_host_list(df, self._time_column)

            keys: list = []
            first_seen: list = []
            changed: list = []
            distinct: list = []
            observations: list = []
            saturated: list = []
            unusable = 0
            unordered = 0

            for (position, raw_fingerprint) in enumerate(fingerprints):
                key = compose_key([normalize_text(key_parts[name][position]) for name in self._key_columns])
                fingerprint = normalize_text(raw_fingerprint)

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                keys.append(key)

                # A handshake with no fingerprint is not a host with a new one. The collector either did not
                # compute it or the record is not a handshake, and either way the history must not learn a null
                # as though it were a stack.
                if (key is None or fingerprint is None or event_time_ns is None):
                    unusable += 1
                    first_seen.append(None)
                    changed.append(None)
                    distinct.append(None)
                    observations.append(None)
                    saturated.append(False)
                    continue

                # Read before this handshake is counted, so the figure is the history behind the row rather
                # than one this row has already joined.
                observations.append(self._observations.get(key, 0))
                self._seen(key)

                result = self._tracker.observe(key, event_time_ns, {self._fingerprint_column: fingerprint})
                count = result.distinct_counts.get(self._fingerprint_column, 0)

                first_seen.append(result.first_seen.get(self._fingerprint_column))
                changed.append(result.changed.get(self._fingerprint_column))
                distinct.append(count)
                saturated.append(count >= self._max_values)
                unordered += int(result.out_of_order)

            assign_str_column(df, CLIENT_KEY, keys)
            assign_nullable_bool_column(df, JA4_FIRST_SEEN, first_seen)
            assign_nullable_bool_column(df, JA4_CHANGED, changed)
            assign_nullable_int_column(df, JA4_DISTINCT, distinct)
            assign_nullable_int_column(df, JA4_OBSERVATIONS, observations)
            assign_nullable_bool_column(df, JA4_SATURATED, saturated)

            if (unusable > 0):
                logger.warning(
                    "TC6FingerprintStage left %d of %d records out of their host's history for want of a key, a "
                    "fingerprint, or a usable event time.",
                    unusable,
                    len(fingerprints))

            if (unordered > 0):
                logger.warning(
                    "TC6FingerprintStage saw %d of %d handshakes arrive no later than their host's previous "
                    "one; they did not enter the history. Whether a fingerprint is new is a question about what "
                    "came before it, which is what an out-of-order arrival disagrees about.",
                    unordered,
                    len(fingerprints))

        return message

    def _seen(self, key: str) -> None:
        """Count one handshake for a host, evicting the least recently seen when the cap is reached."""
        count = self._observations.pop(key, 0)
        self._observations[key] = count + 1

        while (len(self._observations) > self._max_entities):
            self._observations.pop(next(iter(self._observations)))

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
