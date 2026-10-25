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
R-B-L4-001's model, the shipped `abp-pcap-xgb` booster, scored over the flow features layer 4 already emits.

{py:class}`~morpheus.stages.telemetry.tc4_flow_stage.TC4FlowStage` writes the thirteen features the booster was
trained on, and `MODEL_FEATURES` names the column carrying each. This stage reads those columns, scores each row with
{py:class}`~morpheus.utils.abp_scorer.AbpScorer`, and writes the probability, the verdict at the threshold the
upstream configuration set, and the model's inventory version, so a row says which bytes scored it.

**The model is loaded from the inventory and verified there.** `inventory_path` is the inventory document, and the
stage refuses to start if the booster's file is not the one the inventory records; see
{py:mod}`~morpheus.utils.model_inventory`.

**The stage is stateless.** A flow's score depends on its own thirteen features and nothing else the stage has seen,
so it may run at any concurrency and has nothing to checkpoint.
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
from morpheus.stages.telemetry.tc4_flow_stage import MODEL_FEATURES
from morpheus.utils.abp_scorer import AbpScorer
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.model_inventory import load_entry

logger = logging.getLogger(__name__)

ABP_SCORE = "abp_score"
ABP_ANOMALOUS = "abp_anomalous"
ABP_MODEL_VERSION = "abp_model_version"


@register_stage("tc4-abp-score")
class TC4AbpScoreStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Score each flow record with the inventory's `abp-pcap-xgb` booster.

    Stateless, so it may run at any concurrency.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    inventory_path : str
        The model inventory document, `models/inventory/inventory.json` in a checkout of this repository.
    model_name : str, default = "abp-pcap-xgb"
        The inventory entry to score with.
    """

    def __init__(self, c: Config, inventory_path: str, model_name: str = "abp-pcap-xgb"):
        super().__init__(c)

        entry = load_entry(inventory_path, model_name)
        self._scorer = AbpScorer(entry)

        if (self._scorer.features != list(MODEL_FEATURES)):
            raise ValueError(f"{model_name} was trained on {self._scorer.features}, and TC4FlowStage emits "
                             f"{list(MODEL_FEATURES)}; the two orders must agree")

        self._needed_columns[ABP_SCORE] = TypeId.FLOAT64
        self._needed_columns[ABP_ANOMALOUS] = TypeId.BOOL8
        self._needed_columns[ABP_MODEL_VERSION] = TypeId.STRING

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc4-abp-score"

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
        Write each row's probability, verdict and model version.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Flow records carrying `TC4FlowStage`'s feature columns.

        Returns
        -------
        The input message, with the score columns populated.

        Raises
        ------
        KeyError
            If a feature column is absent. A frame without them has not been through `TC4FlowStage`, and scoring it
            would mean scoring nulls.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            absent = [column for column in MODEL_FEATURES.values() if column not in df.columns]

            if (absent):
                raise KeyError(f"TC4AbpScoreStage reads TC4FlowStage's feature columns, and {absent} are absent. "
                               f"Available columns: {sorted(df.columns)}")

            matrix = self._scorer.matrix(
                {feature: to_host_list(df, column)
                 for (feature, column) in MODEL_FEATURES.items()})
            probabilities = [float(value) for value in self._scorer.probabilities(matrix)]

            assign_nullable_float_column(df, ABP_SCORE, probabilities)
            assign_nullable_bool_column(df, ABP_ANOMALOUS, [value > self._scorer.threshold for value in probabilities])
            assign_str_column(df, ABP_MODEL_VERSION, [self._scorer.version] * meta.count)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
