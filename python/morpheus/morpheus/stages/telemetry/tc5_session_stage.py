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
"""Assembles layer 5 sessions from their start and end records."""

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
from morpheus.utils.event_clock import DEFAULT_MAX_SKEW_SECONDS
from morpheus.utils.event_clock import EventClock
from morpheus.utils.session_timer import NS_PER_SECOND
from morpheus.utils.session_timer import SessionTimer

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 36 * 3600
"""Silence after which a session that never ended is abandoned.

Thirty-six hours rather than the five minutes an 802.1X exchange gets. A workstation left logged in over a night
and most of the next day is ordinary; one still open two days later has almost certainly ended in a way its
collector never reported, and holding it open indefinitely would leak a slot per lost stop record.
"""

START_ACTIONS = ("start", "session_start", "logon", "login", "begin")
"""Values of the lifecycle column that open a session."""

END_ACTIONS = ("end", "session_end", "logoff", "logout", "stop", "terminate")
"""Values of the lifecycle column that close one."""


@register_stage("tc5-session")
class TC5SessionStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Pair a session's start record with its end record and write how long it lasted.

    The TC-5 telemetry class names `session_duration_s` among its required fields, and on estates whose identity
    provider emits one record per session carrying both `session_start` and `session_end` this stage has nothing to
    do: the duration is a subtraction and needs no state. It exists for the other shape, which is the common one --
    RADIUS accounting, VPN concentrators, Windows 4624 and 4634, and most SSH and RDP session logging emit a start
    record and, later, a separate stop record, with nothing in either one saying how long the session ran.

    Three things beyond the duration come out of the pairing, and each is a defect report rather than a feature:

    - **`session_unpaired`** -- a stop with no start in front of it. Ordinary at the beginning of a stream, where
      the starts predate the pipeline's view, and a lost record or a collector restart everywhere else. It is
      reported rather than dropped so that a null duration reads as an explained absence instead of missing data.
    - **`session_starts`** -- how many starts this session identifier collected. More than one is not a retry.
      `morpheus.utils.session_timer` counts attempts because an 802.1X supplicant genuinely does try repeatedly,
      but a session identifier is meant to be unique to a session, so a second start on one is either a collector
      duplicating records or an identifier being reused, and the two are worth telling apart from a clean pairing.
    - **`session_out_of_order`** -- a stop that predates the start it would pair with. Nothing is timed.

    Sessions that never end are abandoned once `timeout_seconds` of event time has passed with no stop record.
    Expiry runs off the event stream rather than a wall clock, so a replay abandons the same sessions the live run
    did. Abandoned sessions are counted and logged; no row is synthesized for them, because this stage annotates
    the records it is given and does not invent any.

    The session key it writes, `session_key`, is the principal and the session identifier composed together. It is
    the TC-5 entity key the guide specifies, and it is what a downstream `WindowSealStage` should anchor
    `lineage_id` on for this telemetry class.

    The stage is stateful across messages and must run single-engine, or sharded so that every record for one
    session identifier reaches one instance -- determinism control 4.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    session_column : str, default = "session_id"
        Column holding the session identifier that a start and its stop share.
    principal_column : str, default = "user_principal"
        Column holding the authenticated principal.
    action_column : str, default = "session_action"
        Column naming the record's place in the lifecycle. Compared case-insensitively against `start_actions` and
        `end_actions`.
    time_column : str, default = "event_time"
        Column holding the record's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    start_actions : tuple, default = see `START_ACTIONS`
        Lifecycle values that open a session.
    end_actions : tuple, default = see `END_ACTIONS`
        Lifecycle values that close one.
    timeout_seconds : int, default = 129600
        Silence after which a session that never ended is abandoned.
    max_clock_skew_seconds : int, default = 604800
        Event time this far beyond the highest seen so far is refused rather than allowed to drive expiry. A source
        whose clock is wrong by years would otherwise abandon every open session in the estate at once. A refused
        row carries no timing and is counted. See `morpheus.utils.event_clock`.
    max_open_sessions : int, default = 500000
        Sessions held open before the least recently started is dropped. A dropped session's stop arrives unpaired,
        which over-reports the defect rather than hiding it.
    """

    def __init__(self,
                 c: Config,
                 session_column: str = "session_id",
                 principal_column: str = "user_principal",
                 action_column: str = "session_action",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 start_actions: typing.Sequence[str] = START_ACTIONS,
                 end_actions: typing.Sequence[str] = END_ACTIONS,
                 timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
                 max_clock_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS,
                 max_open_sessions: int = 500_000):
        super().__init__(c)

        if (timeout_seconds <= 0):
            raise ValueError(f"timeout_seconds must be positive, received {timeout_seconds}")

        if (max_clock_skew_seconds <= 0):
            raise ValueError(f"max_clock_skew_seconds must be positive, received {max_clock_skew_seconds}")

        start_set = {value.lower() for value in start_actions}
        end_set = {value.lower() for value in end_actions}
        overlap = sorted(start_set & end_set)

        if (len(overlap) > 0):
            # A value in both sets would open and close the same session, and which it did would depend on the
            # order the sets were tested in rather than on anything about the record.
            raise ValueError(f"start_actions and end_actions must not overlap, received {overlap} in both")

        self._session_column = session_column
        self._principal_column = principal_column
        self._action_column = action_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._start_actions = start_set
        self._end_actions = end_set

        self._timer = SessionTimer(timeout_ns=timeout_seconds * NS_PER_SECOND, max_pending=max_open_sessions)
        self._clock = EventClock(max_skew_ns=max_clock_skew_seconds * NS_PER_SECOND)

        self._needed_columns["session_key"] = TypeId.STRING
        self._needed_columns["session_duration_ns"] = TypeId.INT64
        self._needed_columns["session_duration_s"] = TypeId.INT64
        self._needed_columns["session_starts"] = TypeId.INT64
        self._needed_columns["session_unpaired"] = TypeId.BOOL8
        self._needed_columns["session_out_of_order"] = TypeId.BOOL8

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc5-session"

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

    @property
    def open_sessions(self) -> int:
        """Sessions currently started and awaiting a stop record."""
        return self._timer.pending_count

    def _lifecycle(self, value: typing.Any) -> typing.Optional[str]:
        """
        Classify a record as opening a session, closing one, or neither.

        Unlike `TC2AuthStage`, a missing value is not treated as an opening. There, a null result genuinely means an
        exchange that has not resolved; here a record whose lifecycle column is empty says nothing about which end
        of a session it is, and guessing would pair a start against a record that was never a stop.
        """
        action = normalize_text(value)

        if (action is None):
            return None

        action = action.lower()

        if (action in self._start_actions):
            return "start"

        return "end" if action in self._end_actions else None

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the session key, the duration, and the pairing flags.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming session lifecycle records.

        Returns
        -------
        The input message, with the session columns populated.

        Raises
        ------
        KeyError
            If the session, action, or time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._session_column, self._action_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC5SessionStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            sessions = to_host_list(df, self._session_column)
            actions = to_host_list(df, self._action_column)
            raw_times = to_host_list(df, self._time_column)

            has_principal = self._principal_column in df.columns
            principals = to_host_list(df, self._principal_column) if has_principal else [None] * len(sessions)

            session_keys: list = []
            duration_ns: list = []
            duration_s: list = []
            starts: list = []
            unpaired: list = []
            out_of_order: list = []
            keyless = 0
            unclassified = 0
            abandoned = 0
            implausible = 0

            for (position, raw_session) in enumerate(sessions):
                session_id = normalize_text(raw_session)
                principal = normalize_text(principals[position])
                session_key = compose_key((principal, session_id))
                session_keys.append(session_key)

                lifecycle = self._lifecycle(actions[position])

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                if (session_id is None or lifecycle is None or event_time_ns is None):
                    # Nothing to pair, or nothing to pair it on. Pooling records under a fabricated identifier would
                    # let one principal's logoff close another's session.
                    duration_ns.append(None)
                    duration_s.append(None)
                    starts.append(None)
                    unpaired.append(None)
                    out_of_order.append(False)

                    keyless += int(session_id is None or event_time_ns is None)
                    unclassified += int(session_id is not None and lifecycle is None)
                    continue

                # Expiry is driven by the stream's own clock, so a replay abandons the same sessions a live run
                # did. A source whose clock is wrong by years would otherwise abandon every open session at once.
                if (not self._clock.accept(event_time_ns)):
                    duration_ns.append(None)
                    duration_s.append(None)
                    starts.append(None)
                    unpaired.append(None)
                    out_of_order.append(False)
                    implausible += 1
                    continue

                abandoned += len(self._timer.expire(event_time_ns))

                # The identifier alone is what a start and its stop share. Keying on the principal too would fail to
                # pair the two whenever one of the records carries the principal and the other does not, which is
                # the normal shape of a Windows logoff record.
                if (lifecycle == "start"):
                    self._timer.begin(session_id, event_time_ns)
                    duration_ns.append(None)
                    duration_s.append(None)
                    starts.append(None)
                    unpaired.append(None)
                    out_of_order.append(False)
                    continue

                timing = self._timer.complete(session_id, event_time_ns)

                duration_ns.append(timing.elapsed_ns)
                # A floor rather than a rounding. The exact figure is the nanosecond column beside it, which is what
                # anything doing arithmetic should read; the seconds are for a rule's threshold to be legible.
                duration_s.append(None if timing.elapsed_ns is None else timing.elapsed_ns // NS_PER_SECOND)
                starts.append(timing.attempts if timing.attempts > 0 else None)
                unpaired.append(timing.unpaired)
                out_of_order.append(timing.out_of_order)

            assign_str_column(df, "session_key", session_keys)
            assign_nullable_int_column(df, "session_duration_ns", duration_ns)
            assign_nullable_int_column(df, "session_duration_s", duration_s)
            assign_nullable_int_column(df, "session_starts", starts)
            assign_nullable_bool_column(df, "session_unpaired", unpaired)
            df["session_out_of_order"] = out_of_order

        if (keyless > 0):
            logger.warning(
                "TC5SessionStage saw %d of %d records with no session identifier or no usable event time; they were "
                "not paired. Pooling them under a fabricated identifier would let one principal's logoff close "
                "another principal's session.",
                keyless,
                len(sessions))

        if (unclassified > 0):
            logger.warning(
                "TC5SessionStage saw %d of %d records whose %s column named neither a start nor an end; they were "
                "not paired. Extend start_actions or end_actions to cover what this collector emits.",
                unclassified,
                len(sessions),
                self._action_column)

        if (implausible > 0):
            logger.warning(
                "TC5SessionStage refused %d of %d records whose event time was further ahead than "
                "max_clock_skew_seconds allows; they carry no timing and did not abandon any session.",
                implausible,
                len(sessions))

        if (abandoned > 0):
            logger.warning(
                "TC5SessionStage abandoned %d sessions that ran past the timeout with no end record. A session that "
                "never closes is a lost stop record rather than a very long session, and its stop, if it ever "
                "arrives, will read as unpaired.",
                abandoned)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
