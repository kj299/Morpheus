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
What R-B-L7-004 reads: whether a process's parent and image are a pairing its host, and its host's peers, have seen.

**Novel means new to the host *and* to its peer group, in thirty days.** Per-host novelty alone is dominated by
long-tail legitimate software: the first time a developer runs a new build of their toolchain, every pair it spawns
is new to that laptop and unremarkable on the twenty laptops beside it. A pair is novel here only if neither the
host nor any host in its peer group has produced it in the window. The peer group is the one the TC-0 asset
inventory gives the host at the event's time, attached upstream by
{py:class}`~morpheus.stages.telemetry.tc0_enrich_stage.TC0EnrichStage`; a pairing is remembered under the group the
host belonged to when it ran, so a host that changes group brings no history with it.

**A host with no peer group is judged on its own history and says so.** Dropping it would hide exactly the hosts
nobody has classified; judging it as though it had peers would claim a comparison that was not made. The row carries
`endpoint_host_only`, so a search can weigh or route those alerts separately.

**No answer during the warm-up.** Whatever the comparison is made against -- the peer group, or the host alone --
needs seven days of history before anything it has not seen counts as novel. A new laptop in an established group is
judged against the group from its first process, which is the point of having a group.

**Paths are compared as the same software would be.** Case is folded, because Windows paths are case-insensitive and
EDR products disagree about how they report them, and the folder under a per-user profile -- `C:\\Users\\<name>\\`,
`/home/<name>/`, `/Users/<name>/` -- is collapsed, so software installed per user is one pair across the estate
rather than one per person.

**The weight is the process's integrity level**, normalized here to `system`, `high`, `medium` and `low`, and turned
into a severity by the saved search. The trigger is novelty alone.
"""

import logging
import re
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
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.pair_history import DAY_NS
from morpheus.utils.pair_history import DEFAULT_MAX_ENTITIES
from morpheus.utils.pair_history import DEFAULT_MAX_PAIRS
from morpheus.utils.pair_history import PairHistoryTracker

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 30
"""How recently a pair must have been seen to count as seen. Thirty days, which is what R-B-L7-004 names."""

DEFAULT_WARMUP_DAYS = 7
"""History the comparison needs before a pair can be called novel."""

PAIR_SEPARATOR = " -> "

INTEGRITY_LEVELS = {
    "system": "system",
    "high": "high",
    "medium": "medium",
    "medium plus": "medium",
    "mediumplus": "medium",
    "low": "low",
    "untrusted": "low",
    "appcontainer": "low",
}
"""Integrity levels as EDR products report them, lower-cased, and the level each is weighted as."""

_PROFILE_FOLDERS = (
    re.compile(r"^([a-z]:\\users\\)[^\\]+(\\)"),
    re.compile(r"^(/home/)[^/]+(/)"),
    re.compile(r"^(/users/)[^/]+(/)"),
)

PAIR = "endpoint_pair"
PEER_GROUP = "endpoint_peer_group"
HOST_SEEN = "endpoint_host_seen"
PEER_SEEN = "endpoint_peer_seen"
HOST_ONLY = "endpoint_host_only"
MATURE = "endpoint_mature"
NOVEL = "endpoint_pair_novel"
INTEGRITY = "endpoint_integrity"


def normalize_image_path(path: typing.Any) -> typing.Optional[str]:
    """
    An image path as the comparison sees it: case folded, quotes stripped, a per-user profile folder collapsed.

    Parameters
    ----------
    path : any
        A path as the EDR reported it.

    Returns
    -------
    str or None
        The normalized path, or `None` if there was none.
    """
    text = normalize_text(path)

    if (text is None):
        return None

    text = text.strip('"').strip().lower()

    for pattern in _PROFILE_FOLDERS:
        text = pattern.sub(r"\1*\2", text, count=1)

    return text if len(text) > 0 else None


def normalize_integrity(level: typing.Any) -> typing.Optional[str]:
    """
    An integrity level as one of `system`, `high`, `medium` or `low`, or `None` if it is missing or unrecognized.

    Parameters
    ----------
    level : any
        The level as the EDR reported it.

    Returns
    -------
    str or None
    """
    text = normalize_text(level)

    return None if text is None else INTEGRITY_LEVELS.get(text.lower())


@register_stage("tc7-endpoint")
class TC7EndpointStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write whether each process's parent-image pairing is new to its host and to its host's peer group.

    The stage is stateful across messages and must run single-engine, since a peer group's history is shared by
    every host in it.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    host_column : str, default = "hostname"
        Column holding the host the process ran on.
    parent_image_column : str, default = "parent_image_path"
        Column holding the parent process's image path.
    image_column : str, default = "image_path"
        Column holding the process's image path.
    integrity_column : str, default = "integrity_level"
        Column holding the process's integrity level. May be absent, in which case every row's weight is unknown.
    peer_group_column : str, default = "ctx_peer_group"
        Column holding the host's peer group as TC-0 knew it at the event's time. May be absent, in which case every
        host is judged on its own history.
    time_column : str, default = "event_time"
        Column holding the event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    window_days : int, default = 30
        How recently a pair must have been seen to count as seen.
    warmup_days : int, default = 7
        History the comparison needs before a pair can be called novel.
    max_pairs : int, default = 16384
        Pairs recalled per host and per peer group.
    max_entities : int, default = 200000
        Hosts and peer groups recalled before the least recently seen is forgotten.
    """

    def __init__(self,
                 c: Config,
                 host_column: str = "hostname",
                 parent_image_column: str = "parent_image_path",
                 image_column: str = "image_path",
                 integrity_column: str = "integrity_level",
                 peer_group_column: str = "ctx_peer_group",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 window_days: int = DEFAULT_WINDOW_DAYS,
                 warmup_days: int = DEFAULT_WARMUP_DAYS,
                 max_pairs: int = DEFAULT_MAX_PAIRS,
                 max_entities: int = DEFAULT_MAX_ENTITIES):
        super().__init__(c)

        if (window_days <= 0):
            raise ValueError(f"window_days must be positive, received {window_days}")

        if (warmup_days < 0):
            raise ValueError(f"warmup_days must not be negative, received {warmup_days}")

        self._host_column = host_column
        self._parent_image_column = parent_image_column
        self._image_column = image_column
        self._integrity_column = integrity_column
        self._peer_group_column = peer_group_column
        self._time_column = time_column
        self._time_unit = time_unit

        # Hosts and groups are kept apart, so a host and a group that happen to share a name share nothing else.
        self._hosts = PairHistoryTracker(window_ns=window_days * DAY_NS,
                                         warmup_ns=warmup_days * DAY_NS,
                                         max_pairs=max_pairs,
                                         max_entities=max_entities)
        self._groups = PairHistoryTracker(window_ns=window_days * DAY_NS,
                                          warmup_ns=warmup_days * DAY_NS,
                                          max_pairs=max_pairs,
                                          max_entities=max_entities)

        self._needed_columns[PAIR] = TypeId.STRING
        self._needed_columns[PEER_GROUP] = TypeId.STRING
        self._needed_columns[HOST_SEEN] = TypeId.BOOL8
        self._needed_columns[PEER_SEEN] = TypeId.BOOL8
        self._needed_columns[HOST_ONLY] = TypeId.BOOL8
        self._needed_columns[MATURE] = TypeId.BOOL8
        self._needed_columns[NOVEL] = TypeId.BOOL8
        self._needed_columns[INTEGRITY] = TypeId.STRING

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc7-endpoint"

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
        Write the pair, what its host and peer group had seen, and whether it is novel, for every process.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming EDR process records.

        Returns
        -------
        The input message, with the endpoint columns populated.

        Raises
        ------
        KeyError
            If the host, parent image, image or time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._host_column, self._parent_image_column, self._image_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC7EndpointStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            hosts = to_host_list(df, self._host_column)
            parents = to_host_list(df, self._parent_image_column)
            images = to_host_list(df, self._image_column)
            raw_times = to_host_list(df, self._time_column)
            levels = (to_host_list(df, self._integrity_column) if self._integrity_column in df.columns else [None] *
                      len(hosts))
            groups = (to_host_list(df, self._peer_group_column) if self._peer_group_column in df.columns else [None] *
                      len(hosts))

            pairs: list = []
            peer_groups: list = []
            host_seen: list = []
            peer_seen: list = []
            host_only: list = []
            mature: list = []
            novel: list = []
            integrity: list = []
            unusable = 0
            unordered = 0

            for (position, raw_host) in enumerate(hosts):
                host = normalize_text(raw_host)
                parent = normalize_image_path(parents[position])
                image = normalize_image_path(images[position])
                group = normalize_text(groups[position])
                pair = None if (parent is None or image is None) else f"{parent}{PAIR_SEPARATOR}{image}"

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                pairs.append(pair)
                peer_groups.append(group)
                integrity.append(normalize_integrity(levels[position]))
                host_only.append(host is not None and group is None)

                if (host is None or pair is None or event_time_ns is None):
                    unusable += 1
                    host_seen.append(None)
                    peer_seen.append(None)
                    mature.append(False)
                    novel.append(None)
                    continue

                on_host = self._hosts.observe(host, event_time_ns, pair)
                in_group = None if group is None else self._groups.observe(group, event_time_ns, pair)
                basis = on_host if in_group is None else in_group

                host_seen.append(on_host.seen)
                peer_seen.append(None if in_group is None else in_group.seen)
                mature.append(basis.mature)

                if (on_host.out_of_order or (in_group is not None and in_group.out_of_order)):
                    unordered += 1
                    novel.append(None)
                elif (not basis.mature):
                    novel.append(None)
                else:
                    novel.append(not on_host.seen and (in_group is None or not in_group.seen))

            assign_str_column(df, PAIR, pairs)
            assign_str_column(df, PEER_GROUP, peer_groups)
            assign_nullable_bool_column(df, HOST_SEEN, host_seen)
            assign_nullable_bool_column(df, PEER_SEEN, peer_seen)
            assign_nullable_bool_column(df, HOST_ONLY, host_only)
            assign_nullable_bool_column(df, MATURE, mature)
            assign_nullable_bool_column(df, NOVEL, novel)
            assign_str_column(df, INTEGRITY, integrity)

            if (unusable > 0):
                logger.warning(
                    "TC7EndpointStage judged nothing for %d of %d processes for want of a host, a parent and "
                    "image path, or a usable event time.",
                    unusable,
                    len(hosts))

            if (unordered > 0):
                logger.warning(
                    "TC7EndpointStage saw %d processes arrive earlier than their host's or peer group's previous "
                    "one; they were not judged, and did not join the history that refused them.",
                    unordered)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
