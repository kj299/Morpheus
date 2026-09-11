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
"""Tracks whether an entity's reconstruction error is climbing across consecutive windows."""

import logging
import typing

import pandas as pd
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
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.determinism import DEFAULT_FLOAT_DECIMALS
from morpheus.utils.drift_trajectory import DEFAULT_MAX_WINDOWS
from morpheus.utils.drift_trajectory import DEFAULT_MIN_WINDOWS
from morpheus.utils.drift_trajectory import DriftTrajectoryTracker
from morpheus.utils.entity_key import normalize_text

logger = logging.getLogger(__name__)


@register_stage("tc5-drift")
class TC5DriftStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write the trajectory of an entity's score across consecutive windows.

    This is what R-P-L5-006 reads, and R-P-L5-006 is the rule the word "predictive" rests on: a score rising
    monotonically across four consecutive windows, by more than one and a half of the principal's own standard
    deviations in total, without any single window crossing the alerting threshold. The guide is explicit that
    it should never page -- it puts a principal on a watchlist and raises the sensitivity of layer 7 rules for
    them -- and equally explicit that the premise behind it is a hypothesis this work does not establish.
    Nothing in this stage establishes it either. What it does is measure the trajectory exactly, so a deployment
    can test the premise against its own incident history before promising prediction to anybody.

    **The stage does not care what produced the score.** It reads a column. That the column is usually
    `mean_abs_z` from a per-user autoencoder is a fact about the pipeline upstream, and keeping it out of here is
    what lets the rule, its feature and its corpus be built and tested where no model can run. On a deployment
    with no model the column is absent, every row carries nulls, and the stage says so once rather than per row.

    **One row per entity per window is what this expects.** The trajectory is a sequence of windows, so a stage
    fed several rows for one entity in one window would count each as a separate window and report a run four
    long inside a single afternoon. Place it after the aggregation that produces one score per entity per
    window; a frame carrying more is counted and warned about rather than silently misread.

    The stage is stateful across messages and must run single-engine, or sharded by entity --
    {py:mod}`~morpheus.utils.sharding` is determinism control 4, and this is exactly the kind of stage it exists
    for.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    entity_column : str, default = "user_principal"
        Column holding the entity whose own trajectory is tracked.
    score_column : str, default = "mean_abs_z"
        Column holding the per-window score.
    window_column : str, default = "window_id"
        Column holding the window. Consecutive identifiers are what make a run a run, and a gap restarts it.
    max_windows : int, default = 64
        Windows retained per entity, which the standard deviation is computed over.
    min_windows : int, default = 4
        Prior windows before a trajectory is reported as mature.
    max_entities : int, default = 100000
        Entities tracked before the least recently seen is forgotten.
    decimals : int, default = 4
        Decimal places every reported figure is rounded to, under determinism control 9.
    """

    def __init__(self,
                 c: Config,
                 entity_column: str = "user_principal",
                 score_column: str = "mean_abs_z",
                 window_column: str = "window_id",
                 max_windows: int = DEFAULT_MAX_WINDOWS,
                 min_windows: int = DEFAULT_MIN_WINDOWS,
                 max_entities: int = 100_000,
                 decimals: int = DEFAULT_FLOAT_DECIMALS):
        super().__init__(c)

        self._entity_column = entity_column
        self._score_column = score_column
        self._window_column = window_column
        self._warned_scoreless = False

        self._tracker = DriftTrajectoryTracker(max_windows=max_windows,
                                               min_windows=min_windows,
                                               max_entities=max_entities,
                                               decimals=decimals)

        self._needed_columns["drift_velocity"] = TypeId.FLOAT64
        self._needed_columns["drift_acceleration"] = TypeId.FLOAT64
        self._needed_columns["drift_rising_windows"] = TypeId.INT64
        self._needed_columns["drift_total_rise"] = TypeId.FLOAT64
        self._needed_columns["drift_baseline_sigma"] = TypeId.FLOAT64
        self._needed_columns["drift_rise_sigmas"] = TypeId.FLOAT64
        self._needed_columns["drift_mature"] = TypeId.BOOL8
        self._needed_columns["drift_run_restarted"] = TypeId.BOOL8

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc5-drift"

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
    def tracked_entities(self) -> int:
        """Entities currently holding a trajectory."""
        return self._tracker.tracked_entities

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the velocity, acceleration and rise of each entity's score.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming scored events, one row per entity per window.

        Returns
        -------
        The input message, with the drift columns populated.

        Raises
        ------
        KeyError
            If the entity or window column is absent. The score column may be absent -- that is a pipeline with
            no model, which is a state this fork ships in -- and its absence yields nulls rather than an error.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._entity_column, self._window_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC5DriftStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            entities = to_host_list(df, self._entity_column)
            windows = to_host_list(df, self._window_column)
            row_count = len(entities)
            has_scores = self._score_column in df.columns
            scores = to_host_list(df, self._score_column) if has_scores else [None] * row_count

            velocity: list = []
            acceleration: list = []
            rising: list = []
            total_rise: list = []
            sigma: list = []
            rise_sigmas: list = []
            mature: list = []
            restarted: list = []
            unscored = 0
            unordered = 0
            seen: dict = {}
            repeated = 0

            for position in range(row_count):
                entity = normalize_text(entities[position])
                window = windows[position]
                score = scores[position]

                if (entity is None or window is None or pd.isna(score)):
                    velocity.append(None)
                    acceleration.append(None)
                    rising.append(None)
                    total_rise.append(None)
                    sigma.append(None)
                    rise_sigmas.append(None)
                    mature.append(False)
                    restarted.append(False)
                    unscored += 1
                    continue

                window_id = int(window)
                repeated += int(seen.get(entity) == window_id)
                seen[entity] = window_id

                result = self._tracker.observe(entity, window_id, float(score))

                velocity.append(result.velocity)
                acceleration.append(result.acceleration)
                rising.append(result.rising_windows)
                total_rise.append(result.total_rise)
                sigma.append(result.baseline_sigma)
                rise_sigmas.append(result.rise_sigmas)
                mature.append(result.mature)
                restarted.append(result.run_restarted)
                unordered += int(result.out_of_order)

            assign_nullable_float_column(df, "drift_velocity", velocity)
            assign_nullable_float_column(df, "drift_acceleration", acceleration)
            assign_nullable_int_column(df, "drift_rising_windows", rising)
            assign_nullable_float_column(df, "drift_total_rise", total_rise)
            assign_nullable_float_column(df, "drift_baseline_sigma", sigma)
            assign_nullable_float_column(df, "drift_rise_sigmas", rise_sigmas)
            df["drift_mature"] = mature
            df["drift_run_restarted"] = restarted

        if (not has_scores and not self._warned_scoreless):
            self._warned_scoreless = True
            logger.warning(
                "TC5DriftStage found no %s column, so no trajectory is tracked and every drift column is null. "
                "That is what a pipeline with no per-entity model looks like, and R-P-L5-006 fires on nothing "
                "until one exists. Said once rather than per batch.",
                self._score_column)

        if (unscored > 0 and has_scores):
            logger.warning(
                "TC5DriftStage saw %d of %d rows with no entity, window or score; they carry no trajectory. A "
                "rise measured across a gap in the scores would be a claim about windows nothing scored.",
                unscored,
                row_count)

        if (repeated > 0):
            logger.warning(
                "TC5DriftStage saw %d rows repeating an entity's window inside one batch. The trajectory counts "
                "windows, so several rows for one entity in one window read as several windows and can report a "
                "run four long inside a single afternoon. Aggregate to one score per entity per window first.",
                repeated)

        if (unordered > 0):
            logger.warning(
                "TC5DriftStage saw %d of %d rows whose window was not after the previous one for that entity; "
                "they did not join the trajectory.",
                unordered,
                row_count)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
