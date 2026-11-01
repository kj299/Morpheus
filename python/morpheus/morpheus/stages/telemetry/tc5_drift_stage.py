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
import math
import statistics
import typing

import pandas as pd

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.pipeline.checkpoint_mixin import CheckpointMixin
from morpheus.stages.telemetry.telemetry_base import TelemetryStage
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.determinism import DEFAULT_FLOAT_DECIMALS
from morpheus.utils.drift_trajectory import DEFAULT_MAX_WINDOWS
from morpheus.utils.drift_trajectory import DEFAULT_MIN_WINDOWS
from morpheus.utils.drift_trajectory import DriftTrajectoryTracker
from morpheus.utils.entity_key import normalize_text

logger = logging.getLogger(__name__)

AGGREGATE_NONE = "none"
AGGREGATE_MEAN = "mean"
AGGREGATE_MAX = "max"
AGGREGATE_DISTINCT = "distinct"
AGGREGATES = (AGGREGATE_NONE, AGGREGATE_MEAN, AGGREGATE_MAX, AGGREGATE_DISTINCT)


@register_stage("tc5-drift")
class TC5DriftStage(CheckpointMixin, TelemetryStage):
    """
    Write the trajectory of an entity's score across consecutive windows.

    This is what R-P-L5-006 reads, and R-P-L5-006 is the rule the word "predictive" rests on: a score rising
    monotonically across four consecutive windows, by more than one and a half of the principal's own standard
    deviations in total, without any single window crossing the alerting threshold. The guide is explicit that
    it should never page -- it puts a principal on a watchlist and raises the sensitivity of layer 7 rules for
    them, which in the Splunk app is the `principal_watchlist` lookup R-P-L5-006 writes and R-B-L7-002 reads --
    and equally explicit that the premise behind it is a hypothesis this work does not establish.
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

    **A rising run is also projected.** `drift_trend_per_window` is the least-squares slope through the current
    run's windows and `drift_forecast_next` the value it gives the next one, from a run of three windows on; see
    `morpheus.utils.drift_trajectory`. Given a `ceiling` -- the level a rule fires at, in the score's own units --
    `drift_windows_to_ceiling` says how many more windows the line takes to reach it: zero for a window already at
    or above it, and null for a run not rising. This is the forecast the guide names as its third predictive
    mechanism, "this entity is projected to exceed its envelope", carried for every trajectory rather than only
    for a port's receive light.

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
    aggregate : str, default = "none"
        How rows sharing an entity and a window inside one message are treated. `"none"` observes each row as
        its own window, which is right when the input already carries one row per entity per window and wrong
        otherwise -- the stage warns when it sees repeats. `"mean"` reduces them to one observation, the mean
        of their scores, and stamps that observation's trajectory on every one of them. That is the reduction
        the trajectory needs when the score is per event and the window is a day, and it relies on the message
        holding the whole window, which is what `WindowSealStage` emits: a fragment would give the mean of a
        fragment, and nothing here can tell the difference. `"max"` reduces them to the largest score instead,
        which is the reduction for a running count -- the distinct object types a principal has touched so far
        this week -- whose last value is the window's total and whose mean is not. `"distinct"` reduces them to
        the number of distinct values the score column takes, which is the reduction for breadth -- the distinct
        process pairs a host started in a day -- and the one aggregate whose score column need not be a number.
    ceiling : float, optional
        The level `drift_windows_to_ceiling` counts windows to, in the score's own units. Left unset, that column is
        null on every row and the trend and the forecast are still written.
    """

    _checkpoint_attributes = ("_tracker", )

    def __init__(self,
                 c: Config,
                 entity_column: str = "user_principal",
                 score_column: str = "mean_abs_z",
                 window_column: str = "window_id",
                 max_windows: int = DEFAULT_MAX_WINDOWS,
                 min_windows: int = DEFAULT_MIN_WINDOWS,
                 max_entities: int = 100_000,
                 decimals: int = DEFAULT_FLOAT_DECIMALS,
                 aggregate: str = AGGREGATE_NONE,
                 ceiling: float = None):
        super().__init__(c)

        if (aggregate not in AGGREGATES):
            raise ValueError(f"aggregate must be one of {AGGREGATES}, received {aggregate!r}")

        self._entity_column = entity_column
        self._score_column = score_column
        self._window_column = window_column
        self._aggregate = aggregate
        self._ceiling = None if ceiling is None else float(ceiling)
        self._decimals = decimals
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
        self._needed_columns["drift_trend_per_window"] = TypeId.FLOAT64
        self._needed_columns["drift_forecast_next"] = TypeId.FLOAT64
        self._needed_columns["drift_windows_to_ceiling"] = TypeId.INT64

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc5-drift"

    @property
    def tracked_entities(self) -> int:
        """Entities currently holding a trajectory."""
        return self._tracker.tracked_entities

    def _required_columns(self, df) -> list:
        """The columns this stage reads and cannot run without."""
        del df

        required = [self._entity_column, self._window_column]

        return required

    def _process(self, df):
        """
        Write the velocity, acceleration and rise of each entity's score.

        Parameters
        ----------
        df : `pandas.DataFrame` or `cudf.DataFrame`
            The incoming frame, written in place.
        """
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
        trend: list = []
        forecast: list = []
        to_ceiling: list = []
        unscored = 0
        unordered = 0
        seen: dict = {}
        repeated = 0

        # One observation per entity per window, when asked for it. The members are summed in sorted order
        # so the mean depends on membership alone, the same property the chain root has.
        reduced: dict = {}

        if (self._aggregate != AGGREGATE_NONE):
            members: dict = {}

            for position in range(row_count):
                entity = normalize_text(entities[position])
                window = windows[position]
                score = scores[position]

                if (entity is None or window is None or pd.isna(score)):
                    continue

                member = str(score) if self._aggregate == AGGREGATE_DISTINCT else float(score)
                members.setdefault((entity, int(window)), []).append(member)

            reduce = {
                AGGREGATE_MEAN: statistics.fmean,
                AGGREGATE_MAX: max,
                AGGREGATE_DISTINCT: lambda values: float(len(set(values))),
            }[self._aggregate]
            reduced = {key: reduce(sorted(values)) for (key, values) in members.items()}

        observed: dict = {}

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
                trend.append(None)
                forecast.append(None)
                to_ceiling.append(None)
                unscored += 1
                continue

            window_id = int(window)

            if (self._aggregate != AGGREGATE_NONE):
                key = (entity, window_id)
                value = reduced[key]

                if (key not in observed):
                    observed[key] = self._tracker.observe(entity, window_id, value)

                result = observed[key]
            else:
                repeated += int(seen.get(entity) == window_id)
                seen[entity] = window_id
                value = float(score)
                result = self._tracker.observe(entity, window_id, value)

            velocity.append(result.velocity)
            acceleration.append(result.acceleration)
            rising.append(result.rising_windows)
            total_rise.append(result.total_rise)
            sigma.append(result.baseline_sigma)
            rise_sigmas.append(result.rise_sigmas)
            mature.append(result.mature)
            restarted.append(result.run_restarted)
            trend.append(result.trend_per_window)
            forecast.append(result.forecast_next)
            to_ceiling.append(self._windows_to_ceiling(value, result.trend_per_window))
            unordered += int(result.out_of_order)

        assign_nullable_float_column(df, "drift_velocity", velocity)
        assign_nullable_float_column(df, "drift_acceleration", acceleration)
        assign_nullable_int_column(df, "drift_rising_windows", rising)
        assign_nullable_float_column(df, "drift_total_rise", total_rise)
        assign_nullable_float_column(df, "drift_baseline_sigma", sigma)
        assign_nullable_float_column(df, "drift_rise_sigmas", rise_sigmas)
        df["drift_mature"] = mature
        df["drift_run_restarted"] = restarted
        assign_nullable_float_column(df, "drift_trend_per_window", trend)
        assign_nullable_float_column(df, "drift_forecast_next", forecast)
        assign_nullable_int_column(df, "drift_windows_to_ceiling", to_ceiling)

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

    def _windows_to_ceiling(self, value: float, trend: typing.Optional[float]) -> typing.Optional[int]:
        """Windows the run's line takes from this one to the ceiling: zero at or above it, null for no rising line."""
        if (self._ceiling is None or trend is None):
            return None

        if (value >= self._ceiling):
            return 0

        if (trend <= 0):
            return None

        # Rounded before the ceiling so a quotient that is whole in decimal is not lifted by its binary tail.
        return math.ceil(round((self._ceiling - value) / trend, self._decimals))
