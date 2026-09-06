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
"""Counts a principal's authentication failures, its multi-factor denials, and the successes that end a run."""

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
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.outcome_run import NS_PER_SECOND
from morpheus.utils.outcome_run import OutcomeRunTracker
from morpheus.utils.ratio_window import DEFAULT_MIN_DENOMINATOR
from morpheus.utils.ratio_window import RatioWindowTracker

logger = logging.getLogger(__name__)

DEFAULT_RUN_WINDOW_SECONDS = 600
"""Trailing window the run counts cover, ten minutes: the interval R-D-L5-004 names."""

DEFAULT_RATIO_WINDOW_SECONDS = 24 * 3600
"""Trailing window the multi-factor proportion covers.

A day rather than the rule's ten minutes. The proportion is a statement about how a principal normally
authenticates, and ten minutes of it is a statement about one sitting.
"""

DEFAULT_SUCCESS_VALUES = ("success", "succeeded", "allow", "allowed", "0")
"""Values of the authentication result column that count as success."""

DEFAULT_MFA_SUCCESS_VALUES = ("success", "succeeded", "approved", "approve", "allow", "allowed", "accept")
"""Values of the multi-factor result column that count as an approval."""

TRUE_VALUES = ("true", "yes", "y", "1", "t")
"""Textual renderings of a true flag, for collectors that do not send a boolean."""


@register_stage("tc5-risk")
class TC5RiskStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write the TC-5 failure, multi-factor and run features for each authenticating principal.

    Three of the behavioral features the guide names for this layer live here -- the multi-factor to total
    authentication ratio, failure-then-success sequences within a short window, and the denial pattern
    R-D-L5-004 reads -- because all three are the same primitive counted over different subsets of the stream.

    **The trailing success is what separates a signal from the workforce's Monday morning.** Failed
    authentications are the most ordinary event in an estate. A run of them that then stops being a run is a
    different claim, and in the multi-factor case the guide is explicit that the approval at the end is the part
    worth paging on: it means the victim eventually pressed accept.

    **Two counts of denials, because the two rules ask different questions.** R-D-L5-004 wants at least four
    denials among the challenges in a ten-minute window, and they need not be contiguous, because a fatigue attack
    interleaves with the victim's own traffic; that is `mfa_denials_in_window`. The plain failure-then-success
    feature wants the unbroken run immediately before the approval; that is `consecutive_mfa_denials`. The same
    pair is written for ordinary authentication outcomes.

    **Two windows, because the two features are about different spans.** The run counts use a ten-minute window,
    which is the interval the rule names. The ratio uses a day, because a proportion over ten minutes describes
    one sitting rather than how a principal normally authenticates.

    A record counts as a multi-factor challenge when `mfa_used` is true or when `mfa_result` carries a value.
    Either alone is enough: sources differ on which they populate, and requiring both would leave the feature
    silent on half of them.

    The stage is stateful across messages and must run single-engine, or sharded by principal -- determinism
    control 4.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    principal_column : str, default = "user_principal"
        Column holding the authenticated principal, which every count is grouped by.
    time_column : str, default = "event_time"
        Column holding the record's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    result_column : str, default = "auth_result"
        Column holding the authentication result.
    success_values : tuple, default = see `DEFAULT_SUCCESS_VALUES`
        Result values that count as success. Compared case-insensitively. A result that is neither absent nor in
        this set is a failure; an absent one carries no outcome and is not counted either way, because an unknown
        outcome counted as a failure inflates a run and counted as a success ends one that is still going.
    mfa_column : str, default = "mfa_used"
        Column flagging that a multi-factor challenge was involved.
    mfa_result_column : str, default = "mfa_result"
        Column holding the multi-factor outcome. A record carrying one is a challenge whatever `mfa_column` says.
    mfa_success_values : tuple, default = see `DEFAULT_MFA_SUCCESS_VALUES`
        Multi-factor result values that count as an approval. Compared case-insensitively.
    run_window_seconds : int, default = 600
        Trailing window the run and denial counts cover.
    ratio_window_seconds : int, default = 86400
        Trailing window the multi-factor proportion covers.
    min_denominator : int, default = 10
        Authentications a principal must have in the ratio window before a proportion is reported. Below it the
        figure would be noise rather than a measurement, and the column is null.
    max_samples : int, default = 4096
        Records retained per principal per window regardless of the window. When this binds the counts are lower
        bounds and the row is marked saturated. It must be at least `min_denominator`, or the ratio window could
        never hold enough to publish a proportion; the shared window refuses that combination when the pipeline is
        built rather than leaving a column silently null.
    max_entities : int, default = 100000
        Principals tracked before the least recently seen is forgotten.
    """

    def __init__(self,
                 c: Config,
                 principal_column: str = "user_principal",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 result_column: str = "auth_result",
                 success_values: typing.Sequence[str] = DEFAULT_SUCCESS_VALUES,
                 mfa_column: str = "mfa_used",
                 mfa_result_column: str = "mfa_result",
                 mfa_success_values: typing.Sequence[str] = DEFAULT_MFA_SUCCESS_VALUES,
                 run_window_seconds: int = DEFAULT_RUN_WINDOW_SECONDS,
                 ratio_window_seconds: int = DEFAULT_RATIO_WINDOW_SECONDS,
                 min_denominator: int = DEFAULT_MIN_DENOMINATOR,
                 max_samples: int = 4096,
                 max_entities: int = 100_000):
        super().__init__(c)

        if (run_window_seconds <= 0):
            raise ValueError(f"run_window_seconds must be positive, received {run_window_seconds}")

        if (ratio_window_seconds <= 0):
            raise ValueError(f"ratio_window_seconds must be positive, received {ratio_window_seconds}")

        self._principal_column = principal_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._result_column = result_column
        self._success_values = {value.lower() for value in success_values}
        self._mfa_column = mfa_column
        self._mfa_result_column = mfa_result_column
        self._mfa_success_values = {value.lower() for value in mfa_success_values}

        self._auth_runs = OutcomeRunTracker(window_ns=run_window_seconds * NS_PER_SECOND,
                                            max_samples=max_samples,
                                            max_entities=max_entities)
        self._mfa_runs = OutcomeRunTracker(window_ns=run_window_seconds * NS_PER_SECOND,
                                           max_samples=max_samples,
                                           max_entities=max_entities)
        self._mfa_ratio = RatioWindowTracker(window_ns=ratio_window_seconds * NS_PER_SECOND,
                                             min_denominator=min_denominator,
                                             max_samples=max_samples,
                                             max_entities=max_entities)

        self._needed_columns["mfa_ratio"] = TypeId.FLOAT64
        self._needed_columns["mfa_challenge"] = TypeId.BOOL8
        self._needed_columns["mfa_attempts_in_window"] = TypeId.INT64
        self._needed_columns["mfa_denials_in_window"] = TypeId.INT64
        self._needed_columns["consecutive_mfa_denials"] = TypeId.INT64
        self._needed_columns["mfa_denied_then_approved"] = TypeId.BOOL8
        self._needed_columns["auth_attempts_in_window"] = TypeId.INT64
        self._needed_columns["auth_failures_in_window"] = TypeId.INT64
        self._needed_columns["consecutive_auth_failures"] = TypeId.INT64
        self._needed_columns["auth_failed_then_succeeded"] = TypeId.BOOL8
        self._needed_columns["risk_counts_saturated"] = TypeId.BOOL8

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc5-risk"

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
    def tracked_principals(self) -> int:
        """Principals currently holding an authentication window."""
        return self._auth_runs.tracked_entities

    @staticmethod
    def _is_true(value: typing.Any) -> bool:
        """Whether a flag column's value is true, for collectors that send a string where a boolean belongs."""
        if (isinstance(value, bool)):
            return value

        text = normalize_text(value)

        return text is not None and text.lower() in TRUE_VALUES

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the multi-factor proportion and the failure and denial counts.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming authentication records.

        Returns
        -------
        The input message, with the TC-5 risk columns populated.

        Raises
        ------
        KeyError
            If the principal or time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._principal_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC5RiskStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            principals = to_host_list(df, self._principal_column)
            raw_times = to_host_list(df, self._time_column)
            row_count = len(principals)

            def optional(column: str) -> list:
                return to_host_list(df, column) if column in df.columns else [None] * row_count

            results = optional(self._result_column)
            mfa_flags = optional(self._mfa_column)
            mfa_results = optional(self._mfa_result_column)

            ratio: list = []
            challenge: list = []
            mfa_attempts: list = []
            mfa_denials: list = []
            mfa_run: list = []
            mfa_pattern: list = []
            auth_attempts: list = []
            auth_failures: list = []
            auth_run: list = []
            auth_pattern: list = []
            saturated: list = []
            keyless = 0
            outcomeless = 0
            unordered = 0

            for (position, raw_principal) in enumerate(principals):
                principal = normalize_text(raw_principal)

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                mfa_result = normalize_text(mfa_results[position])
                is_challenge = self._is_true(mfa_flags[position]) or mfa_result is not None
                challenge.append(is_challenge)

                if (principal is None or event_time_ns is None):
                    # Pooling records under a fabricated principal would make every unattributed failure in the
                    # estate look like one account under sustained attack.
                    ratio.append(None)
                    mfa_attempts.append(None)
                    mfa_denials.append(None)
                    mfa_run.append(None)
                    mfa_pattern.append(None)
                    auth_attempts.append(None)
                    auth_failures.append(None)
                    auth_run.append(None)
                    auth_pattern.append(None)
                    saturated.append(False)
                    keyless += 1
                    continue

                proportion = self._mfa_ratio.observe(principal, event_time_ns, is_challenge)
                ratio.append(proportion.ratio)

                result = normalize_text(results[position])
                row_saturated = proportion.saturated

                if (result is None):
                    # An unknown outcome counted as a failure inflates a run, and counted as a success ends one
                    # that is still going. Neither is better than declining to guess.
                    auth_attempts.append(None)
                    auth_failures.append(None)
                    auth_run.append(None)
                    auth_pattern.append(None)
                    outcomeless += 1
                else:
                    outcome = self._auth_runs.observe(principal, event_time_ns, result.lower() in self._success_values)
                    auth_attempts.append(outcome.attempts)
                    auth_failures.append(outcome.failures)
                    auth_run.append(outcome.consecutive_failures)
                    auth_pattern.append(outcome.failure_then_success)
                    row_saturated = row_saturated or outcome.saturated
                    unordered += int(outcome.out_of_order)

                if (not is_challenge or mfa_result is None):
                    # A record flagged as a challenge but carrying no outcome says a factor was involved, not how
                    # it resolved, so it cannot join a run of denials or end one.
                    mfa_attempts.append(None)
                    mfa_denials.append(None)
                    mfa_run.append(None)
                    mfa_pattern.append(None)
                else:
                    factor = self._mfa_runs.observe(principal,
                                                    event_time_ns,
                                                    mfa_result.lower() in self._mfa_success_values)
                    mfa_attempts.append(factor.attempts)
                    mfa_denials.append(factor.failures)
                    mfa_run.append(factor.consecutive_failures)
                    mfa_pattern.append(factor.failure_then_success)
                    row_saturated = row_saturated or factor.saturated

                saturated.append(row_saturated)

            assign_nullable_float_column(df, "mfa_ratio", ratio)
            df["mfa_challenge"] = challenge
            assign_nullable_int_column(df, "mfa_attempts_in_window", mfa_attempts)
            assign_nullable_int_column(df, "mfa_denials_in_window", mfa_denials)
            assign_nullable_int_column(df, "consecutive_mfa_denials", mfa_run)
            assign_nullable_bool_column(df, "mfa_denied_then_approved", mfa_pattern)
            assign_nullable_int_column(df, "auth_attempts_in_window", auth_attempts)
            assign_nullable_int_column(df, "auth_failures_in_window", auth_failures)
            assign_nullable_int_column(df, "consecutive_auth_failures", auth_run)
            assign_nullable_bool_column(df, "auth_failed_then_succeeded", auth_pattern)
            df["risk_counts_saturated"] = saturated

        if (keyless > 0):
            logger.warning(
                "TC5RiskStage saw %d of %d records with no principal or no usable event time; they carry no "
                "counts. Pooling them under a fabricated principal would make every unattributed failure in the "
                "estate look like one account under sustained attack.",
                keyless,
                len(principals))

        if (outcomeless > 0):
            logger.warning(
                "TC5RiskStage saw %d of %d records whose %s column was empty; they carry no authentication "
                "counts. An unknown outcome counted as a failure inflates a run and counted as a success ends "
                "one that is still going.",
                outcomeless,
                len(principals),
                self._result_column)

        if (unordered > 0):
            logger.warning(
                "TC5RiskStage saw %d of %d records out of order; they did not join a run. Preserve per-principal "
                "ordering upstream with TotalOrderStage.",
                unordered,
                len(principals))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
