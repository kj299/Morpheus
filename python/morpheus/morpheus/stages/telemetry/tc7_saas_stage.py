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
What the two SaaS rules read: how much a principal just took, and how widely they have been reaching.

**R-B-L7-002 is a principal against their own history, per operation.** `record_count` for a SaaS operation above
the principal's own 30-day 99th percentile by more than five times. The baseline is keyed on the principal *and*
the operation, because an export and a read are different scales of the same person's normal: an analyst reads a
few records at a time all day and exports thousands once a month, and a baseline pooling the two would call every
export a breach or excuse every bulk read. It is the layer 4 transfer envelope --
{py:mod}`~morpheus.utils.transfer_envelope` -- over a different magnitude, with its conventions: a nearest-rank
quantile of *prior* events only, so one enormous export cannot partly excuse itself, and no baseline at all below a
hundred priors, where a 99th percentile by nearest rank is simply the largest value ever seen.

**R-P-L7-006 reads breadth per calendar week.** The stage writes a running count of the distinct
`target_object_type` values the principal has touched so far in the week the event falls in, weeks starting Monday
00:00 UTC. The week's figure is the count's last value, which is why the trajectory stage downstream reduces a week
by its maximum; the trajectory itself -- whether breadth has risen across consecutive weeks -- is
{py:class}`~morpheus.stages.telemetry.tc5_drift_stage.TC5DriftStage`'s, over weekly windows sealed behind the hourly
ones, exactly as layer 5's daily trajectory is.

**A failed operation reached nothing.** Neither measurement counts an operation whose `result` says it failed: a
denied export read no records, and a type the principal was refused is not a type they reached. Refusals are worth a
rule of their own; folding them into these would make a principal probing for access look like one who has it.

Neither rule reads `target_object` itself -- the weighting by data classification is attached upstream, from the
TC-0 context store, as known at the event's time -- so the object's name can be pseudonymized or dropped at the wire
without changing either answer.
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
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.determinism import DEFAULT_FLOAT_DECIMALS
from morpheus.utils.determinism import quantize_value
from morpheus.utils.entity_key import compose_key
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.lineage import window_id_from_timestamp
from morpheus.utils.transfer_envelope import DEFAULT_MIN_SAMPLES
from morpheus.utils.transfer_envelope import DEFAULT_QUANTILE
from morpheus.utils.transfer_envelope import TransferEnvelopeTracker

logger = logging.getLogger(__name__)

DEFAULT_BASELINE_DAYS = 30
"""Trailing window the record-count baseline is taken over. Thirty days, which is what R-B-L7-002 names."""

DEFAULT_WEEK_EPOCH = "1970-01-05"
"""A Monday. Weeks are counted from it, so every week starts Monday 00:00 UTC."""

WEEK_SECONDS = 7 * 24 * 3600

FAILED_RESULTS = frozenset({"fail", "failed", "failure", "denied", "error", "blocked"})
"""`result` values, lower-cased, that mean the operation reached nothing."""

BASELINE_KEY = "saas_baseline_key"
RECORD_BASELINE = "saas_record_baseline"
RECORD_RATIO = "saas_record_ratio"
BASELINE_SAMPLES = "saas_baseline_samples"
BASELINE_MATURE = "saas_baseline_mature"
BASELINE_SATURATED = "saas_baseline_saturated"
OPERATION_FAILED = "saas_operation_failed"
WEEK_ID = "saas_week_id"
OBJECT_TYPES_IN_WEEK = "saas_object_types_in_week"


@register_stage("tc7-saas")
class TC7SaasStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write each operation's record count against the principal's own baseline for it, and the week's breadth so far.

    The stage is stateful across messages and must run single-engine, or sharded by principal.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    principal_column : str, default = "user_principal"
        Column holding the principal.
    operation_column : str, default = "operation"
        Column holding the SaaS operation. Part of the baseline's key.
    object_type_column : str, default = "target_object_type"
        Column holding the kind of object the operation reached.
    record_count_column : str, default = "record_count"
        Column holding how many records the operation returned or touched.
    result_column : str, default = "result"
        Column holding the operation's outcome. A failed operation enters neither measurement. The column may be
        absent, in which case every operation is taken to have succeeded.
    time_column : str, default = "event_time"
        Column holding the event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    baseline_days : int, default = 30
        Trailing window the record-count baseline is taken over.
    quantile : float, default = 0.99
        Quantile the baseline is taken at.
    min_samples : int, default = 100
        Prior operations, per principal and operation, before a baseline is published.
    week_epoch : str, default = "1970-01-05"
        A Monday; weeks are counted from it. Must match the epoch of the weekly `WindowSealStage` downstream, or
        this stage's week and the trajectory's week will disagree about where a week ends.
    max_samples : int, default = 4096
        Operations retained per baseline regardless of the window.
    max_entities : int, default = 500000
        Baselines and weekly tallies retained before the least recently seen is forgotten.
    decimals : int, default = 4
        Decimal places the baseline and ratio are rounded to, under determinism control 9.
    """

    def __init__(self,
                 c: Config,
                 principal_column: str = "user_principal",
                 operation_column: str = "operation",
                 object_type_column: str = "target_object_type",
                 record_count_column: str = "record_count",
                 result_column: str = "result",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 baseline_days: int = DEFAULT_BASELINE_DAYS,
                 quantile: float = DEFAULT_QUANTILE,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 week_epoch: str = DEFAULT_WEEK_EPOCH,
                 max_samples: int = 4096,
                 max_entities: int = 500_000,
                 decimals: int = DEFAULT_FLOAT_DECIMALS):
        super().__init__(c)

        if (baseline_days <= 0):
            raise ValueError(f"baseline_days must be positive, received {baseline_days}")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        week_epoch_ns = to_epoch_ns(week_epoch)

        if (week_epoch_ns is None):
            raise ValueError("week_epoch is required")

        self._principal_column = principal_column
        self._operation_column = operation_column
        self._object_type_column = object_type_column
        self._record_count_column = record_count_column
        self._result_column = result_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._week_epoch_ns = week_epoch_ns
        self._max_entities = max_entities
        self._decimals = decimals

        # The multiplier is the rule's, and the rule reads the ratio, so the tracker's own breach flag is not used;
        # it is given the smallest value it accepts rather than a second copy of the rule's threshold.
        self._baselines = TransferEnvelopeTracker(window_ns=baseline_days * 24 * 3600 * NS_PER_SECOND,
                                                  quantile=quantile,
                                                  multiplier=1.0 + 1e-9,
                                                  min_samples=min_samples,
                                                  max_samples=max_samples,
                                                  max_entities=max_entities)

        # Per principal: the week being tallied and the types touched in it.
        self._weeks: dict[str, tuple[int, set]] = {}

        self._needed_columns[BASELINE_KEY] = TypeId.STRING
        self._needed_columns[RECORD_BASELINE] = TypeId.FLOAT64
        self._needed_columns[RECORD_RATIO] = TypeId.FLOAT64
        self._needed_columns[BASELINE_SAMPLES] = TypeId.INT64
        self._needed_columns[BASELINE_MATURE] = TypeId.BOOL8
        self._needed_columns[BASELINE_SATURATED] = TypeId.BOOL8
        self._needed_columns[OPERATION_FAILED] = TypeId.BOOL8
        self._needed_columns[WEEK_ID] = TypeId.INT64
        self._needed_columns[OBJECT_TYPES_IN_WEEK] = TypeId.INT64

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc7-saas"

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

    def _round(self, value: typing.Optional[float]) -> typing.Optional[float]:
        return None if value is None else quantize_value(value, self._decimals)

    def _tally(self, principal: str, week_id: int, object_type: typing.Optional[str]) -> typing.Optional[int]:
        """The distinct types this principal has reached so far in this week, or `None` for an earlier week."""
        current = self._weeks.get(principal)

        if (current is not None and week_id < current[0]):
            return None

        if (current is None or week_id > current[0]):
            current = (week_id, set())
            self._weeks.pop(principal, None)
            self._weeks[principal] = current

            while (len(self._weeks) > self._max_entities):
                self._weeks.pop(next(iter(self._weeks)))

        if (object_type is not None):
            current[1].add(object_type)

        return len(current[1])

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the baseline, the ratio and the week's breadth for every operation.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming SaaS audit records.

        Returns
        -------
        The input message, with the SaaS columns populated.

        Raises
        ------
        KeyError
            If the principal, operation, object type, record count or time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [
                self._principal_column,
                self._operation_column,
                self._object_type_column,
                self._record_count_column,
                self._time_column
            ]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC7SaasStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            principals = to_host_list(df, self._principal_column)
            operations = to_host_list(df, self._operation_column)
            object_types = to_host_list(df, self._object_type_column)
            counts = to_host_list(df, self._record_count_column)
            raw_times = to_host_list(df, self._time_column)
            outcomes = (to_host_list(df, self._result_column) if self._result_column in df.columns else [None] *
                        len(principals))

            keys: list = []
            baselines: list = []
            ratios: list = []
            samples: list = []
            mature: list = []
            saturated: list = []
            failed: list = []
            weeks: list = []
            breadth: list = []
            unusable = 0
            unordered = 0

            for (position, raw_principal) in enumerate(principals):
                principal = normalize_text(raw_principal)
                key = compose_key([principal, operations[position]])
                outcome = normalize_text(outcomes[position])
                did_fail = outcome is not None and outcome.lower() in FAILED_RESULTS

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                keys.append(key)
                failed.append(did_fail)
                week_id = (None if event_time_ns is None else window_id_from_timestamp(
                    event_time_ns, WEEK_SECONDS * NS_PER_SECOND, epoch_ns=self._week_epoch_ns))
                weeks.append(week_id)

                if (principal is None or event_time_ns is None):
                    unusable += 1
                    baselines.append(None)
                    ratios.append(None)
                    samples.append(None)
                    mature.append(False)
                    saturated.append(False)
                    breadth.append(None)
                    continue

                breadth.append(
                    None if did_fail else self._tally(principal, week_id, normalize_text(object_types[position])))

                count = counts[position]

                if (did_fail or key is None or normalize_text(count) is None):
                    baselines.append(None)
                    ratios.append(None)
                    samples.append(None)
                    mature.append(False)
                    saturated.append(False)
                    continue

                result = self._baselines.observe(key, event_time_ns, float(count))
                unordered += int(result.out_of_order)

                baselines.append(self._round(result.baseline))
                ratios.append(self._round(result.ratio))
                samples.append(result.samples)
                mature.append(result.mature)
                saturated.append(result.saturated)

            assign_str_column(df, BASELINE_KEY, keys)
            assign_nullable_float_column(df, RECORD_BASELINE, baselines)
            assign_nullable_float_column(df, RECORD_RATIO, ratios)
            assign_nullable_int_column(df, BASELINE_SAMPLES, samples)
            assign_nullable_bool_column(df, BASELINE_MATURE, mature)
            assign_nullable_bool_column(df, BASELINE_SATURATED, saturated)
            assign_nullable_bool_column(df, OPERATION_FAILED, failed)
            assign_nullable_int_column(df, WEEK_ID, weeks)
            assign_nullable_int_column(df, OBJECT_TYPES_IN_WEEK, breadth)

            if (unusable > 0):
                logger.warning(
                    "TC7SaasStage measured nothing for %d of %d operations for want of a principal or a usable "
                    "event time.",
                    unusable,
                    len(principals))

            if (unordered > 0):
                logger.warning(
                    "TC7SaasStage saw %d operations arrive no later than the previous one for the same principal "
                    "and operation; they were compared with nothing and did not join the baseline.",
                    unordered)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
