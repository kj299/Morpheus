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
"""Detects identifier changes per port, with no period boundary to hide behind."""

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
from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.value_novelty import DEFAULT_MAX_VALUES
from morpheus.utils.value_novelty import ValueNoveltyTracker

logger = logging.getLogger(__name__)

DEFAULT_NOVELTY_COLUMNS = ["transceiver_serial", "lldp_neighbor_chassis_id"]
"""Identifiers whose change is the TC-1 signal."""

CHANGED_SUFFIX = "_changed"
FIRST_SEEN_SUFFIX = "_first_seen"
DISTINCT_SUFFIX = "_distinct_count"


@register_stage("tc1-change")
class TC1ChangeStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Report when a port's identifiers change, whenever that happens.

    `morpheus.stages.telemetry.tc1_feature_stage.TC1FeatureStage` answers the same question by counting distinct
    values per period, and the period is its limit: the count resets at each boundary, so a change from the last
    sample before one to the first after it is a single distinct value on each side and reads as no change.
    Lengthening the period makes boundaries rarer without removing them. This stage removes them, by holding the
    previous value per port and comparing rather than bucketing. A substitution is detected the moment the new
    value arrives, whether the samples are a minute or a year apart.

    Two columns per identifier, because two separately actionable questions:

    - `<name>_changed` is whether the value differs from the previous sample. This is the alerting signal: a
      transceiver serial that differs from the one seen a minute ago is a substitution that just happened.
    - `<name>_first_seen` is whether this port has ever reported this value before. An optic that has never been in
      this cage is a different situation from one rotated back in after maintenance. Both are changes; only the
      first is unexplained by the estate's own history.

    A third, `<name>_distinct_count`, is how many distinct values the port has reported, which is context for the
    other two rather than a signal of its own.

    A null is a value. A port with an empty cage reports no serial, and moving into or out of that state is a
    change worth seeing, so it is tracked like any other value rather than skipped.

    Both columns are null on a port's first sample, which establishes what normal looks like and is not itself an
    event. A rule matching on `== True` therefore never fires on a port's first appearance.

    The stage is stateful across messages and must run single-engine. For parallelism, shard by device upstream and
    give each shard its own instance, which is determinism control 4. Place it after
    `morpheus.stages.telemetry.tc1_normalize_stage.TC1NormalizeStage`, which supplies `entity_key`.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    entity_key_column : str, default = "entity_key"
        Column holding the port identity changes are tracked per. Anything coarser than the port is meaningless
        here, since a switch legitimately reports one serial per port.
    time_column : str, default = "event_time"
        Column holding the sample's event time. Used only to reject out-of-order samples, never to bucket.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    novelty_columns : list of str, optional
        Identifiers to watch. Defaults to `["transceiver_serial", "lldp_neighbor_chassis_id"]`.
    max_values : int, default = 64
        Distinct values recalled per identifier per port. Beyond this the least recently seen is forgotten, so its
        return reads as first seen again; the bound errs toward over-reporting novelty rather than hiding it.
    """

    def __init__(self,
                 c: Config,
                 entity_key_column: str = "entity_key",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 novelty_columns: list[str] = None,
                 max_values: int = DEFAULT_MAX_VALUES):
        super().__init__(c)

        novelty_columns = list(DEFAULT_NOVELTY_COLUMNS) if novelty_columns is None else list(novelty_columns)

        if (len(novelty_columns) == 0):
            raise ValueError("novelty_columns must name at least one identifier")

        self._entity_key_column = entity_key_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._novelty_columns = novelty_columns

        self._tracker = ValueNoveltyTracker(field_names=novelty_columns, max_values=max_values)

        for name in novelty_columns:
            self._needed_columns[f"{name}{CHANGED_SUFFIX}"] = TypeId.BOOL8
            self._needed_columns[f"{name}{FIRST_SEEN_SUFFIX}"] = TypeId.BOOL8
            self._needed_columns[f"{name}{DISTINCT_SUFFIX}"] = TypeId.INT64

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc1-change"

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
    def _value(value: typing.Any) -> typing.Optional[str]:
        """Normalize a host value, collapsing every flavour of missing to `None`."""
        if (value is None):
            return None

        # A null in a string column can arrive as a float NaN, which compares unequal to itself and would look
        # like a change on every single sample.
        if (isinstance(value, float) and math.isnan(value)):
            return None

        return str(value)

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the change, novelty, and distinct-count columns for every watched identifier.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming message.

        Returns
        -------
        The input message, with the change columns populated.

        Raises
        ------
        KeyError
            If the entity key, the time column, or a watched identifier column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._entity_key_column, self._time_column] + self._novelty_columns
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC1ChangeStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            entity_keys = to_host_list(df, self._entity_key_column)
            raw_times = to_host_list(df, self._time_column)
            identifiers = {name: to_host_list(df, name) for name in self._novelty_columns}

            changed: dict[str, list] = {name: [] for name in self._novelty_columns}
            first_seen: dict[str, list] = {name: [] for name in self._novelty_columns}
            distinct: dict[str, list] = {name: [] for name in self._novelty_columns}
            unordered = 0

            for (position, entity_key) in enumerate(entity_keys):
                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                if (event_time_ns is None):
                    for name in self._novelty_columns:
                        changed[name].append(None)
                        first_seen[name].append(None)
                        distinct[name].append(None)

                    unordered += 1
                    continue

                result = self._tracker.observe(
                    str(entity_key),
                    event_time_ns, {name: self._value(identifiers[name][position])
                                    for name in self._novelty_columns})

                for name in self._novelty_columns:
                    changed[name].append(result.changed[name])
                    first_seen[name].append(result.first_seen[name])
                    distinct[name].append(result.distinct_counts[name] if not result.out_of_order else None)

                unordered += int(result.out_of_order)

            for name in self._novelty_columns:
                assign_nullable_bool_column(df, f"{name}{CHANGED_SUFFIX}", changed[name])
                assign_nullable_bool_column(df, f"{name}{FIRST_SEEN_SUFFIX}", first_seen[name])
                assign_nullable_int_column(df, f"{name}{DISTINCT_SUFFIX}", distinct[name])

        if (unordered > 0):
            logger.warning(
                "TC1ChangeStage saw %d of %d samples out of order or without a usable event time; they carry no "
                "verdict and did not advance any port's state. Shard by device and preserve per-port ordering "
                "upstream.",
                unordered,
                len(entity_keys))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
