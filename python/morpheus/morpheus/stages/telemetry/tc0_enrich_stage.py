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
Attaches what the TC-0 context store says about an event's principal or asset, at the event's own time.

This is the join the SaaS and endpoint rules at layer 7 need and nothing else in the fork can make: a bulk read
weighted by the classification of what was read, a principal's access breadth set against a role assignment, a
process pair compared with its host's peer group. Each is a question about the context *at the event's time*, which
is a point-in-time join against a `BitemporalStore`, and making it here rather than in a SIEM search keeps it
reproducible: the answer is a pure function of the row and the store.

**Which knowledge the answer uses is a parameter, and the default is what was known at the event's time.** A store
grows: a correction recorded next week changes what "now" says about last Tuesday. Enriching with everything the store
holds would make a replay of last Tuesday's events disagree with the original run as soon as anything about last
Tuesday was corrected, and would attribute to a detection knowledge it could not have had. With `knowledge="event"`,
only versions recorded at or before the event are considered, so the answer is fixed once the event has happened.
`knowledge="latest"` is the investigator's view, the best current understanding of what was true then; it is for
reports, and a detection that uses it is a detection that can change its mind about the past.

Every row gets `ctx_found`, and the identifiers of the versions its answer rests on, so an analyst can tell a
principal the store has never heard of from one with nothing to say, and can trace any attached value to the record
behind it.
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
from morpheus.stages.telemetry.tc0_identity_stage import GROUP_ATTRIBUTE
from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.bitemporal import MEMBERSHIP
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list

logger = logging.getLogger(__name__)

KNOWLEDGE_MODES = ("event", "latest")

SET_SEPARATOR = "|"
"""Joins the members of a set-valued attribute, sorted, so equal sets render equally."""

FOUND = "found"
RECORDED_AT = "recorded_at"
VERSION_UIDS = "version_uids"
KNOWLEDGE = "knowledge"


@register_stage("tc0-enrich", ignore_args=["store", "set_kinds"])
class TC0EnrichStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Attach the context that held for each row's principal or asset at the row's event time.

    Single-valued attributes -- a profile's department, an asset's classification -- are written one column each.
    Facts of a set-valued kind -- a principal's group memberships -- are collected into one column holding the sorted
    members, so two rows with the same memberships carry the same string.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    store : `morpheus.utils.bitemporal.BitemporalStore`
        The context to enrich from. Required; the default of `None` exists only so the CLI can register the stage,
        and construction rejects it.
    entity_column : str, default = "user_principal"
        Column holding the principal or asset to look up.
    time_column : str, default = "event_time"
        Column holding the event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    knowledge : str, default = "event"
        `event` considers only versions recorded at or before each row's event time; `latest` considers everything
        the store holds. See the module docstring for why the default is the first.
    prefix : str, default = "ctx_"
        Prepended to every column this stage writes.
    set_kinds : dict, optional
        Kinds whose facts are collected into a set, mapped to the attribute that names each member and the output
        column. Defaults to collecting memberships by `group_name` into `groups`, when the store holds any.
    """

    def __init__(self,
                 c: Config,
                 store: BitemporalStore = None,
                 entity_column: str = "user_principal",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 knowledge: str = "event",
                 prefix: str = "ctx_",
                 set_kinds: dict = None):
        super().__init__(c)

        if (store is None):
            raise ValueError("store is required")

        if (not entity_column):
            raise ValueError("entity_column is required")

        if (knowledge not in KNOWLEDGE_MODES):
            raise ValueError(f"knowledge must be one of {KNOWLEDGE_MODES}, received {knowledge!r}")

        if (set_kinds is None):
            set_kinds = {MEMBERSHIP: (GROUP_ATTRIBUTE, "groups")} if MEMBERSHIP in store.kinds() else {}

        for (kind, target) in set_kinds.items():
            if (not isinstance(target, (tuple, list)) or len(target) != 2):
                raise ValueError(f"set_kinds[{kind!r}] must be (member attribute, output column), received {target!r}")

        self._store = store
        self._entity_column = entity_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._knowledge = knowledge
        self._prefix = prefix
        self._set_kinds = {kind: tuple(target) for (kind, target) in set_kinds.items()}

        # Single-valued attributes, from every kind that is not collected as a set. The first kind in sorted order to
        # name an attribute supplies it, so two kinds sharing a name resolve the same way on every run.
        self._attributes: dict[str, str] = {}

        for kind in store.kinds():
            if (kind in self._set_kinds):
                continue

            for name in store.attribute_names(kind):
                self._attributes.setdefault(name, kind)

        for name in self._attributes:
            self._needed_columns[prefix + name] = TypeId.STRING

        for (_, output) in self._set_kinds.values():
            self._needed_columns[prefix + output] = TypeId.STRING

        self._needed_columns[prefix + FOUND] = TypeId.BOOL8
        self._needed_columns[prefix + RECORDED_AT] = TypeId.INT64
        self._needed_columns[prefix + VERSION_UIDS] = TypeId.STRING
        self._needed_columns[prefix + KNOWLEDGE] = TypeId.STRING

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc0-enrich"

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
        Attach the context columns to every row.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming events.

        Returns
        -------
        The input message, with the context columns populated.

        Raises
        ------
        KeyError
            If the entity column or the time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            missing = [column for column in (self._entity_column, self._time_column) if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC0EnrichStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            entities = to_host_list(df, self._entity_column)
            raw_times = to_host_list(df, self._time_column)

            attributes = {name: [] for name in self._attributes}
            sets = {kind: [] for kind in self._set_kinds}
            found = []
            recorded = []
            uids = []
            untimed = 0

            for (position, entity) in enumerate(entities):
                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                if (event_time_ns is None):
                    untimed += 1
                    facts = []
                else:
                    horizon = event_time_ns if self._knowledge == "event" else None
                    facts = self._store.facts_about(entity, event_time_ns, known_ns=horizon)

                for (name, kind) in self._attributes.items():
                    value = None

                    for fact in facts:
                        if (fact.kind == kind):
                            value = fact.attributes.get(name)
                            break

                    attributes[name].append(value)

                for (kind, (member, _)) in self._set_kinds.items():
                    members = sorted(
                        fact.attributes.get(member) for fact in facts
                        if fact.kind == kind and fact.attributes.get(member) is not None)
                    sets[kind].append(SET_SEPARATOR.join(members) if len(members) > 0 else None)

                found.append(len(facts) > 0)
                recorded.append(max((fact.recorded_ns for fact in facts), default=None))
                uids.append(SET_SEPARATOR.join(sorted(fact.uid for fact in facts)) or None)

            for (name, values) in attributes.items():
                assign_str_column(df, self._prefix + name, values)

            for (kind, (_, output)) in self._set_kinds.items():
                assign_str_column(df, self._prefix + output, sets[kind])

            assign_nullable_bool_column(df, self._prefix + FOUND, found)
            assign_nullable_int_column(df, self._prefix + RECORDED_AT, recorded)
            assign_str_column(df, self._prefix + VERSION_UIDS, uids)
            assign_str_column(df, self._prefix + KNOWLEDGE, [self._knowledge] * len(entities))

            if (untimed > 0):
                logger.warning("TC0EnrichStage attached no context to %d of %d rows for want of a usable event time.",
                               untimed,
                               len(entities))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
