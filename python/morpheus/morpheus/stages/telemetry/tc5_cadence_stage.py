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
"""Scores when a principal authenticated against the hours and days that principal has used before."""

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
from morpheus.utils.cyclic_histogram import DEFAULT_MIN_SAMPLES
from morpheus.utils.cyclic_histogram import CyclicHistogramTracker
from morpheus.utils.determinism import DEFAULT_FLOAT_DECIMALS
from morpheus.utils.entity_key import normalize_text

logger = logging.getLogger(__name__)

NS_PER_SECOND = 10**9
NS_PER_HOUR = 3600 * NS_PER_SECOND
NS_PER_DAY = 24 * NS_PER_HOUR

HOURS_IN_DAY = 24
DAYS_IN_WEEK = 7

EPOCH_WEEKDAY = 3
"""1970-01-01 was a Thursday, which is 3 on a Monday-is-zero week."""


@register_stage("tc5-cadence")
class TC5CadenceStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write how unusual an authentication's hour of day and day of week are for the principal making it.

    The TC-5 telemetry class asks for "hour-of-day and day-of-week deviation from the user's own histogram", and
    the comparison against the principal's own history is the whole point: 03:00 is unremarkable for the batch
    account that has run at 03:00 nightly for a year and is the entire signal for the analyst who has never once
    authenticated outside office hours. Both are scored by `morpheus.utils.cyclic_histogram`, which excludes the
    current sample from the history it is judged against and smooths unseen buckets so that a first-ever 03:00
    login has a finite surprise that grows with how much history it contradicts.

    **The local time is a fixed offset, and that is a determinism decision rather than a convenience.** Resolving a
    named zone through the IANA database would make every feature this stage writes depend on a data file that is
    revised several times a year, so a re-run after a `tzdata` update could move scores with nothing in the
    pipeline having changed -- and determinism control 2 would then have to freeze and hash that file along with
    the configuration. A fixed offset has no such dependency. It also costs something real, and it is worth being
    plain about it: an estate spanning several zones either runs one instance per zone or accepts that the
    histogram is in one zone's hours. Because each principal is scored against its own histogram, a principal who
    keeps regular hours anywhere keeps them just as regularly in UTC, so the cost falls almost entirely on
    principals who travel.

    Daylight saving is the visible edge of the same trade. A fixed offset means a principal in a zone that observes
    it shifts by one bucket twice a year, and the shift shows for as long as it takes the histogram to absorb the
    new hour. That is a known, bounded, twice-yearly effect on one feature, and it is preferable to a feature that
    can change silently whenever a package is upgraded.

    The stage is stateful across messages and must run single-engine, or sharded by principal -- determinism
    control 4.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    principal_column : str, default = "user_principal"
        Column holding the principal whose own histogram each sample is judged against.
    time_column : str, default = "event_time"
        Column holding the record's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    utc_offset_minutes : int, default = 0
        Fixed offset applied before the hour and weekday are taken. Minutes rather than hours because several
        populated zones are not on an hour boundary.
    min_samples : int, default = 32
        Prior observations before a principal's histogram is reported as mature. Maturity is reported rather than
        used to withhold a score: a brand new account authenticating at 03:00 is not obviously the less
        interesting case, and that is a rule's judgement to make.
    max_entities : int, default = 100000
        Principals tracked before the least recently seen is forgotten.
    decimals : int, default = 4
        Decimal places the two quantized scores are rounded to, under determinism control 9.
    """

    def __init__(self,
                 c: Config,
                 principal_column: str = "user_principal",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 utc_offset_minutes: int = 0,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 max_entities: int = 100_000,
                 decimals: int = DEFAULT_FLOAT_DECIMALS):
        super().__init__(c)

        if (abs(utc_offset_minutes) >= 24 * 60):
            raise ValueError(f"utc_offset_minutes must be inside a day, received {utc_offset_minutes}")

        self._principal_column = principal_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._offset_ns = utc_offset_minutes * 60 * NS_PER_SECOND

        self._hours = CyclicHistogramTracker(buckets=HOURS_IN_DAY,
                                             min_samples=min_samples,
                                             max_entities=max_entities,
                                             decimals=decimals)
        self._weekdays = CyclicHistogramTracker(buckets=DAYS_IN_WEEK,
                                                min_samples=min_samples,
                                                max_entities=max_entities,
                                                decimals=decimals)

        self._needed_columns["local_hour"] = TypeId.INT64
        self._needed_columns["local_weekday"] = TypeId.INT64
        self._needed_columns["hour_share"] = TypeId.FLOAT64
        self._needed_columns["hour_surprise_bits"] = TypeId.FLOAT64
        self._needed_columns["hour_unseen"] = TypeId.BOOL8
        self._needed_columns["weekday_share"] = TypeId.FLOAT64
        self._needed_columns["weekday_surprise_bits"] = TypeId.FLOAT64
        self._needed_columns["weekday_unseen"] = TypeId.BOOL8
        self._needed_columns["cadence_samples"] = TypeId.INT64
        self._needed_columns["cadence_mature"] = TypeId.BOOL8

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc5-cadence"

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
        """Principals currently holding an hour histogram."""
        return self._hours.tracked_entities

    def _buckets(self, event_time_ns: int) -> tuple:
        """
        The local hour of day and day of week, as integer arithmetic on the epoch.

        No calendar library and no timezone database, so the answer cannot change when a package is upgraded.
        Python's floor division handles times before the epoch correctly, which a truncating division would not.
        """
        local_ns = event_time_ns + self._offset_ns
        days = local_ns // NS_PER_DAY

        return ((local_ns // NS_PER_HOUR) % HOURS_IN_DAY, (days + EPOCH_WEEKDAY) % DAYS_IN_WEEK)

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the local hour and weekday and how unusual each is for this principal.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming authentication records.

        Returns
        -------
        The input message, with the cadence columns populated.

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
                raise KeyError(f"TC5CadenceStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            principals = to_host_list(df, self._principal_column)
            raw_times = to_host_list(df, self._time_column)

            hours: list = []
            weekdays: list = []
            hour_share: list = []
            hour_bits: list = []
            hour_unseen: list = []
            weekday_share: list = []
            weekday_bits: list = []
            weekday_unseen: list = []
            samples: list = []
            mature: list = []
            keyless = 0
            unordered = 0

            for (position, raw_principal) in enumerate(principals):
                principal = normalize_text(raw_principal)

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                if (event_time_ns is None):
                    hours.append(None)
                    weekdays.append(None)
                else:
                    (hour, weekday) = self._buckets(event_time_ns)
                    hours.append(hour)
                    weekdays.append(weekday)

                if (principal is None or event_time_ns is None):
                    # The hour and weekday are still written where the time allows: they are facts about the record
                    # rather than about the principal, and a rule may want them even where no history exists.
                    hour_share.append(None)
                    hour_bits.append(None)
                    hour_unseen.append(None)
                    weekday_share.append(None)
                    weekday_bits.append(None)
                    weekday_unseen.append(None)
                    samples.append(None)
                    mature.append(False)
                    keyless += 1
                    continue

                hour_result = self._hours.observe(principal, event_time_ns, hours[position])
                weekday_result = self._weekdays.observe(principal, event_time_ns, weekdays[position])

                hour_share.append(hour_result.smoothed_share)
                hour_bits.append(hour_result.surprise_bits)
                hour_unseen.append(hour_result.bucket_unseen)
                weekday_share.append(weekday_result.smoothed_share)
                weekday_bits.append(weekday_result.surprise_bits)
                weekday_unseen.append(weekday_result.bucket_unseen)
                samples.append(hour_result.samples)
                mature.append(hour_result.mature)
                unordered += int(hour_result.out_of_order)

            assign_nullable_int_column(df, "local_hour", hours)
            assign_nullable_int_column(df, "local_weekday", weekdays)
            assign_nullable_float_column(df, "hour_share", hour_share)
            assign_nullable_float_column(df, "hour_surprise_bits", hour_bits)
            assign_nullable_bool_column(df, "hour_unseen", hour_unseen)
            assign_nullable_float_column(df, "weekday_share", weekday_share)
            assign_nullable_float_column(df, "weekday_surprise_bits", weekday_bits)
            assign_nullable_bool_column(df, "weekday_unseen", weekday_unseen)
            assign_nullable_int_column(df, "cadence_samples", samples)
            df["cadence_mature"] = mature

        if (keyless > 0):
            logger.warning(
                "TC5CadenceStage saw %d of %d records with no principal or no usable event time; they carry no "
                "histogram score. Pooling them would build one histogram out of every unattributed authentication "
                "in the estate, which no principal's hours resemble.",
                keyless,
                len(principals))

        if (unordered > 0):
            logger.warning(
                "TC5CadenceStage saw %d of %d records out of order; they were scored against the history as it "
                "stood and did not join it. Preserve per-principal ordering upstream with TotalOrderStage.",
                unordered,
                len(principals))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
