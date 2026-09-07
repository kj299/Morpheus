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
Scores layer 5 rows against the model pinned for their entity, and emits the columns four rules read.

`mean_abs_z` and `max_abs_z` have had no producer in this fork, so R-B-L5-001, R-B-L5-002, R-B-L5-005 and
R-P-L5-006 have fired on nothing and the drift trajectory has read a column of nulls. This is the stage that
changes that.

**It takes a scorer; it does not train one.** The model is injected, which is the decision that makes the stage
testable at all: a stage that trained inside itself could only be tested where Torch and a card are, which is
not where this fork is developed, and the determinism checks that matter most here -- the double run, the batch
split, the permutation -- would have been the checks that could never run. Injecting the scorer also puts the
training lifecycle where it belongs. Training is a scheduled job against a window of history; scoring is a
per-row question asked in a stream. Conflating them is how a pipeline ends up retraining on the data it is
scoring, which is a leak no determinism control would catch because every run would leak identically.

**The manifest decides which model, and this stage never chooses.** {py:mod}`~morpheus.utils.model_manifest`
resolves an entity to a pinned `name:version` for exactly one window and refuses to answer for any other. A
scorer is asked for that identifier and nothing else, so the stage cannot silently score against "latest" --
the resolution this control exists to forbid.
{py:class}`~morpheus.stages.lineage.determinism_stamp_stage.DeterminismStampStage` stamps the same resolution
onto the row, from the same manifest, which is what makes `model_version` on a scored event a statement about
the model that actually produced its score rather than a label applied beside it.

**Entities are scored in a fixed order.** A scorer holding any shared state -- a Torch global generator is the
obvious one -- would otherwise produce output depending on which entity happened to be scored first, which
varies with how the stream is batched. Sorting the groups costs nothing and removes the failure mode; the
per-principal reseeding in `examples/layer5_model/run_model.py` exists for the same reason.

A row with no entity, or with a gap in any feature, carries a null score rather than a zero. A zero is a
confident statement that the entity looked exactly average, which is the opposite of what a missing measurement
says, and the drift trajectory downstream treats a null as a break in the run rather than bridging it.
"""

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
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.determinism import DEFAULT_FLOAT_DECIMALS
from morpheus.utils.determinism import quantize_value
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.model_manifest import ModelManifest

logger = logging.getLogger(__name__)

MEAN_COLUMN = "mean_abs_z"
MAX_COLUMN = "max_abs_z"
LOSS_SUFFIX = "_z_loss"


class Scorer(typing.Protocol):
    """
    What this stage needs from a model, and nothing more.

    One method, taking the pinned model identifier and the rows for a single entity, returning a per-row absolute
    z-score for each feature. The stage derives `mean_abs_z` and `max_abs_z` from those rather than asking for
    them, so a scorer cannot report a mean that disagrees with the losses beside it.
    """

    def score(self, model_version: str, features: list) -> list:
        """
        Score one entity's rows.

        Parameters
        ----------
        model_version : str
            The pinned `name:version` the manifest resolved for this entity.
        features : list of dict
            One dict per row, keyed by feature name. Rows arrive in event order.

        Returns
        -------
        list of dict
            One dict per input row, keyed by feature name, holding that feature's absolute z-score. Must be the
            same length as `features`.
        """
        raise NotImplementedError


@register_stage("tc5-score", modes=[])
class TC5ScoreStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Attach per-entity model scores to layer 5 rows.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    scorer : `Scorer`
        The model, injected. Asked for one entity's rows at a time, against the identifier the manifest pinned.
    manifest : `morpheus.utils.model_manifest.ModelManifest`
        The models pinned for the window being scored. Resolving a row from a different window raises, which is
        the control's whole point.
    feature_columns : list of str
        The columns handed to the scorer, in this order. A row with a gap in any of them is not scored.
    entity_column : str, default = "user_principal"
        The behavioral subject. Rows are grouped by it and each group is scored against its own model.
    window_column : str, default = "window_id"
        The window each row belongs to, checked against the manifest's. Absent, every row is taken to be the
        manifest's own window, which is the single-window case.
    decimals : int, default = 4
        Decimal places every emitted score is rounded to, under determinism control 9.
    """

    def __init__(self,
                 c: Config,
                 scorer: Scorer,
                 manifest: ModelManifest,
                 feature_columns: list[str],
                 entity_column: str = "user_principal",
                 window_column: str = "window_id",
                 decimals: int = DEFAULT_FLOAT_DECIMALS):
        super().__init__(c)

        if (scorer is None):
            raise ValueError("scorer is required; this stage does not train one, it is given one")

        if (manifest is None):
            raise ValueError("manifest is required; a stage that chose its own model would defeat control 1")

        if (not feature_columns):
            raise ValueError("feature_columns must name at least one column")

        self._scorer = scorer
        self._manifest = manifest
        self._feature_columns = list(feature_columns)
        self._entity_column = entity_column
        self._window_column = window_column
        self._decimals = decimals

        self._needed_columns[MEAN_COLUMN] = TypeId.FLOAT64
        self._needed_columns[MAX_COLUMN] = TypeId.FLOAT64

        for name in self._feature_columns:
            self._needed_columns[f"{name}{LOSS_SUFFIX}"] = TypeId.FLOAT64

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc5-score"

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
        Score every row whose entity has a model and whose features are complete.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming layer 5 rows.

        Returns
        -------
        The same message, with the score columns attached.

        Raises
        ------
        KeyError
            If the entity column or any feature column is absent.
        ValueError
            If a row belongs to a window this manifest was not resolved for.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            missing = [name for name in [self._entity_column] + self._feature_columns if name not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC5ScoreStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            entities = to_host_list(df, self._entity_column)
            row_count = len(entities)
            features = {name: to_host_list(df, name) for name in self._feature_columns}
            windows = to_host_list(df, self._window_column) if self._window_column in df.columns else None

            # Grouped by entity, and the groups visited in sorted order. A scorer with shared state would
            # otherwise depend on which entity came first, which depends on batching.
            groups: dict = {}
            unscorable = 0

            for position in range(row_count):
                entity = normalize_text(entities[position])
                row = {name: features[name][position] for name in self._feature_columns}

                if (entity is None or any(pd.isna(value) for value in row.values())):
                    unscorable += 1
                    continue

                if (windows is not None and windows[position] is not None):
                    # Raises when the row is not this manifest's window. Deliberately not caught: a window scored
                    # against models pinned for another is the defect control 1 exists to prevent, and continuing
                    # would produce scores that look ordinary.
                    self._manifest.resolve(entity, int(windows[position]))

                groups.setdefault(entity, []).append((position, row))

            means: list = [None] * row_count
            maxima: list = [None] * row_count
            losses: dict = {name: [None] * row_count for name in self._feature_columns}

            for entity in sorted(groups):
                positions = [position for (position, _) in groups[entity]]
                rows = [row for (_, row) in groups[entity]]
                resolution = self._manifest.resolve(entity, self._manifest.window_id)
                scored = self._scorer.score(resolution.model_version, rows)

                if (len(scored) != len(rows)):
                    raise ValueError(f"the scorer returned {len(scored)} rows for {entity}, which was given "
                                     f"{len(rows)}. A scorer that drops or invents rows cannot be aligned back "
                                     f"onto the frame, and silently misaligned scores are worse than none.")

                for (index, position) in enumerate(positions):
                    # Quantized first, then summarized, so the mean is the mean of the numbers actually
                    # published rather than of the ones behind them. Averaging the raw losses and rounding
                    # afterwards is marginally more accurate and leaves `mean_abs_z` disagreeing with the ten
                    # loss columns beside it by up to half a quantum -- an analyst who adds up what is on the
                    # row gets a different answer than the row gives, which is a worse failure than a
                    # ten-thousandth of a deviation.
                    per_feature = [
                        quantize_value(abs(float(scored[index][name])), decimals=self._decimals)
                        for name in self._feature_columns
                    ]

                    for (name, value) in zip(self._feature_columns, per_feature):
                        losses[name][position] = value

                    means[position] = quantize_value(sum(per_feature) / len(per_feature), decimals=self._decimals)
                    maxima[position] = max(per_feature)

            assign_nullable_float_column(df, MEAN_COLUMN, means)
            assign_nullable_float_column(df, MAX_COLUMN, maxima)

            for name in self._feature_columns:
                assign_nullable_float_column(df, f"{name}{LOSS_SUFFIX}", losses[name])

        if (unscorable > 0):
            logger.warning(
                "TC5ScoreStage left %d of %d rows unscored: no entity, or a gap in a feature. They carry null "
                "scores rather than zeros, because a zero is a confident claim that the entity looked exactly "
                "average and a gap is the absence of any claim at all.",
                unscorable,
                row_count)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
