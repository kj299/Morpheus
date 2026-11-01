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
R-B-L7-003's model, the shipped SID model, scored over a text column on the CPU.

The rule is the upstream sensitive-information model over request and response bodies at an egress point. This stage
scores whatever column holds that text with {py:class}`~morpheus.utils.sid_scorer.SidScorer` and writes one
probability per label, the labels above the threshold, and the model's inventory version. No feed this fork composes
carries a body yet, which is why the column is a parameter: the HTTP records layer 7 reads hold the request line and
headers, not the content, and a deployment pointing this at its own body field is the step that makes the rule fire.

**The labels column lists what was found, sorted and comma-separated.** Ten probability columns are what the model
says; the labels column is what a search reads. A row the scorer found nothing on has an empty string there, and a
row whose text was empty has a null, because "looked and found nothing" and "had nothing to look at" are different
claims.

**The stage is stateless.** A body's score depends on its own text, so the stage may run at any concurrency and has
nothing to checkpoint.
"""

import logging

import pandas as pd

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.stages.telemetry.telemetry_base import TelemetryStage
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.model_inventory import load_entry
from morpheus.utils.sid_scorer import SidScorer

logger = logging.getLogger(__name__)

SID_LABELS = "sid_labels"
SID_MODEL_VERSION = "sid_model_version"


def score_column(label: str) -> str:
    """The column a label's probability is written to."""
    return f"sid_{label}_score"


@register_stage("tc7-sid-score")
class TC7SidScoreStage(TelemetryStage):
    """
    Score a text column with the inventory's SID model.

    Stateless, so it may run at any concurrency.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    inventory_path : str
        The model inventory document, `models/inventory/inventory.json` in a checkout of this repository.
    text_column : str, default = "body"
        Column holding the text to score.
    model_name : str, default = "sid-minibert-20230424"
        The inventory entry to score with.
    """

    def __init__(self,
                 c: Config,
                 inventory_path: str,
                 text_column: str = "body",
                 model_name: str = "sid-minibert-20230424"):
        super().__init__(c)

        self._scorer = SidScorer(load_entry(inventory_path, model_name))
        self._text_column = text_column

        for label in self._scorer.labels:
            self._needed_columns[score_column(label)] = TypeId.FLOAT64

        self._needed_columns[SID_LABELS] = TypeId.STRING
        self._needed_columns[SID_MODEL_VERSION] = TypeId.STRING

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc7-sid-score"

    def _process(self, df):
        """
        Write each row's per-label probabilities, the labels found, and the model version.

        Parameters
        ----------
        df : `pandas.DataFrame` or `cudf.DataFrame`
            The incoming frame, written in place.

        Raises
        ------
        KeyError
            If the text column is absent.
        """
        if (self._text_column not in df.columns):
            raise KeyError(f"TC7SidScoreStage scores {self._text_column!r}, which the DataFrame does not carry. "
                           f"Available columns: {sorted(df.columns)}")

        # A body read from a file arrives as NaN where it was null, and `str` would make it the text "nan" and
        # score it; a missing body is no text, whatever the reader spelled it as.
        texts = [None if pd.isna(value) else str(value) for value in to_host_list(df, self._text_column)]
        probabilities = self._scorer.probabilities(texts)
        labels = self._scorer.labels

        for (index, label) in enumerate(labels):
            assign_nullable_float_column(df,
                                         score_column(label),
                                         [None if row is None else float(row[index]) for row in probabilities])

        assign_str_column(
            df,
            SID_LABELS,
            [
                None if row is None else ",".join(
                    sorted(label for (index, label) in enumerate(labels) if row[index] > self._scorer.threshold))
                for row in probabilities
            ])
        assign_str_column(df, SID_MODEL_VERSION, [self._scorer.version] * len(df))
