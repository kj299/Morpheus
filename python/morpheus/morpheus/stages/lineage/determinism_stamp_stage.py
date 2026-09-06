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
"""Stamps the determinism envelope, and the model each entity was scored against, onto every event."""

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
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.determinism_envelope import DeterminismEnvelope
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.model_manifest import MODEL_FALLBACK_COLUMN
from morpheus.utils.model_manifest import MODEL_VERSION_COLUMN
from morpheus.utils.model_manifest import ModelManifest

logger = logging.getLogger(__name__)


@register_stage("determinism-stamp")
class DeterminismStampStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write the determinism envelope onto every row, and the model version onto every scored one.

    This is determinism control 12 as a stage, with control 1's manifest carried alongside it because the two
    answer one question between them: what produced this number. The envelope's fields are constant for a
    pipeline run and the model version is per entity, so they arrive together and land on the same row.

    **The envelope is stamped whether or not a model scored the row.** A telemetry event that no model touched
    still came out of a particular configuration, commit and image, and a consumer comparing it with an event
    from another run needs to know whether the two are comparable. Only `model_version` and
    `model_fallback_used` are specific to scoring, and they are null where no manifest is supplied.

    **A manifest is refused for the wrong window.** `ModelManifest.resolve` raises if the window it is asked
    about is not the one it was pinned for, and this stage does not catch that: a run that scored a window
    against models pinned for another is a defect worth stopping on rather than annotating. What it does catch
    is an entity with no model and no declared fallback, which is a fact about the estate rather than about the
    pipeline -- those rows carry a null model version, and are counted and logged.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    envelope : `morpheus.utils.determinism_envelope.DeterminismEnvelope`
        The fields every event carries.
    manifest : `morpheus.utils.model_manifest.ModelManifest`, optional
        The models pinned for the window being scored. Omitted on a pipeline that scores nothing, where the
        model columns are null rather than absent -- a consumer reading them should see "no model" rather than
        a missing field it cannot distinguish from a dropped one.
    entity_column : str, default = "user_principal"
        Column holding the entity the manifest is keyed by.
    window_column : str, default = "window_id"
        Column holding the window each row belongs to, which the manifest checks itself against. Absent from
        the frame means the manifest's own window is assumed, which is correct for a pipeline that seals before
        scoring and is why `WindowSealStage` belongs upstream of this.
    """

    def __init__(self,
                 c: Config,
                 envelope: DeterminismEnvelope,
                 manifest: typing.Optional[ModelManifest] = None,
                 entity_column: str = "user_principal",
                 window_column: str = "window_id"):
        super().__init__(c)

        self._envelope = envelope
        self._manifest = manifest
        self._entity_column = entity_column
        self._window_column = window_column
        self._columns = envelope.to_columns()

        for (name, value) in self._columns.items():
            self._needed_columns[name] = TypeId.INT64 if isinstance(value, int) else TypeId.STRING

        self._needed_columns[MODEL_VERSION_COLUMN] = TypeId.STRING
        self._needed_columns[MODEL_FALLBACK_COLUMN] = TypeId.BOOL8

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "determinism-stamp"

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
        Write the envelope columns and, where a manifest is supplied, the model each entity was scored against.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming events.

        Returns
        -------
        The input message, with the determinism columns populated.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            row_count = len(df)

            for (name, value) in self._columns.items():
                df[name] = [value] * row_count

            versions: list = [None] * row_count
            fallbacks: list = [None] * row_count
            unpinned = 0

            if (self._manifest is not None):
                entities = (to_host_list(df, self._entity_column) if self._entity_column in df.columns else [None] *
                            row_count)
                windows = (to_host_list(df, self._window_column)
                           if self._window_column in df.columns else [self._manifest.window_id] * row_count)

                for position in range(row_count):
                    window = windows[position]
                    window_id = self._manifest.window_id if window is None else int(window)

                    try:
                        resolution = self._manifest.resolve(normalize_text(entities[position]), window_id)
                    except ValueError:
                        # An entity with no model and no declared fallback. A fact about the estate rather than
                        # about the pipeline, so the row carries no model rather than stopping the run -- and a
                        # manifest asked about the wrong window raises out of this stage rather than being
                        # annotated, which is the distinction the except clause has to preserve.
                        if (window_id != self._manifest.window_id):
                            raise

                        unpinned += 1
                        continue

                    versions[position] = resolution.model_version
                    fallbacks[position] = resolution.fallback_used

            assign_str_column(df, MODEL_VERSION_COLUMN, versions)
            assign_nullable_bool_column(df, MODEL_FALLBACK_COLUMN, fallbacks)

        if (unpinned > 0):
            logger.warning(
                "DeterminismStampStage found no pinned model for %d of %d rows and no fallback is declared; "
                "they carry a null %s. Scoring them against nothing in particular and reporting a number would "
                "be worse, but a rule reading a score on these rows is reading one nothing produced.",
                unpinned,
                row_count,
                MODEL_VERSION_COLUMN)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
