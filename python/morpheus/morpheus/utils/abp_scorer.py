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
The shipped `abp-pcap-xgb` booster, scored on the CPU from the inventory rather than behind a Triton endpoint.

R-B-L4-001 is the upstream booster applied to the thirteen flow features
{py:class}`~morpheus.stages.telemetry.tc4_flow_stage.TC4FlowStage` emits. Upstream served it from Triton's FIL
backend: a GPU, a model repository resolved by name, and nothing CI could run. This reads the same trees with
XGBoost on the CPU. The upstream file is XGBoost's legacy binary format, which XGBoost 3 no longer reads, so the
inventory holds it converted to JSON by XGBoost 1.7.6, with 220 rows that release scored from the original file;
`tests/morpheus/utils/test_abp_scorer.py` holds this scorer to every one of them exactly, and to a tree walk written
from the JSON without XGBoost.

**The features go in by name and in the order the model was trained on.** The inventory records the thirteen names
in training order, and the caller says which column carries each, so a renamed column is a `KeyError` rather than a
silently shifted feature.

**A missing feature value is missing, not zero.** A null in a feature column reaches XGBoost as NaN, which its trees
route down each split's default branch, as FIL does. Filling it with zero would score a different flow.

**The score is a probability; the verdict is the configuration's.** Upstream's Triton configuration set
`output_class` and a threshold of 0.5, so the endpoint returned a class. This returns the probability and leaves the
threshold, read from the inventory, to the caller, because a SIEM tuning the rule needs the number and not only the
side of 0.5 it fell on.
"""

import typing

import numpy as np

from morpheus.utils.model_inventory import InventoryEntry


class AbpScorer:
    """
    Score flow feature rows with the inventory's `abp-pcap-xgb` booster.

    Parameters
    ----------
    entry : InventoryEntry
        The inventory entry, already verified by `load_entry`.

    Raises
    ------
    ValueError
        If the entry is not an XGBoost JSON model.
    """

    def __init__(self, entry: InventoryEntry):
        import xgboost  # pylint: disable=import-outside-toplevel

        if (entry.settings.get("format") != "xgboost-json"):
            raise ValueError(f"{entry.name} is {entry.settings.get('format')!r}, and this scores 'xgboost-json'")

        self._booster = xgboost.Booster(model_file=entry.model_path)
        # One thread: the trees' outputs are summed in a fixed order regardless, and a scorer inside a pipeline
        # stage should not take every core of the host it runs on.
        self._booster.set_param({"nthread": 1})
        self._xgboost = xgboost
        self._features = list(entry.settings["features"])
        self._threshold = float(entry.settings["threshold"])
        self._version = entry.version

        if (self._booster.num_features() != len(self._features)):
            raise ValueError(f"{entry.name} reads {self._booster.num_features()} features, and the inventory names "
                             f"{len(self._features)}")

    @property
    def version(self) -> str:
        """The scored model, as `name:<digest>`."""
        return self._version

    @property
    def features(self) -> list:
        """The feature names, in the order the model was trained on."""
        return list(self._features)

    @property
    def threshold(self) -> float:
        """The probability above which a row is anomalous, as the upstream configuration set it."""
        return self._threshold

    def margins(self, matrix: np.ndarray) -> np.ndarray:
        """
        The booster's raw output, the log-odds before the logistic.

        Parameters
        ----------
        matrix : numpy.ndarray
            Rows of the features in `features` order, float32; NaN is a missing value.
        """
        return self._booster.predict(self._xgboost.DMatrix(self._checked(matrix)), output_margin=True)

    def probabilities(self, matrix: np.ndarray) -> np.ndarray:
        """
        The probability each row is anomalous.

        Parameters
        ----------
        matrix : numpy.ndarray
            Rows of the features in `features` order, float32; NaN is a missing value.
        """
        return self._booster.predict(self._xgboost.DMatrix(self._checked(matrix)))

    def matrix(self, columns: typing.Mapping[str, list]) -> np.ndarray:
        """
        Assemble the model's input from per-feature column values.

        Parameters
        ----------
        columns : mapping
            Each feature name to that feature's values, one per row; `None` is a missing value.

        Raises
        ------
        KeyError
            If a feature has no column.
        """
        missing = [name for name in self._features if name not in columns]

        if (missing):
            raise KeyError(f"the abp-pcap-xgb features {missing} have no column")

        return np.column_stack([
            np.array([np.nan if value is None else value for value in columns[name]], dtype=np.float32)
            for name in self._features
        ]).astype(np.float32)

    def _checked(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)

        if (matrix.ndim != 2 or matrix.shape[1] != len(self._features)):
            raise ValueError(f"rows of {len(self._features)} features are needed, and the matrix is {matrix.shape}")

        return matrix
