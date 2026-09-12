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
The per-principal autoencoder behind the `Scorer` protocol `TC5ScoreStage` takes.

`TC5ScoreStage` asks one thing of a model: given the pinned version the manifest resolved for an entity and that
entity's rows, return a per-row absolute z-score per feature. Until now the only thing answering was
`ReferenceScorer`, frozen arithmetic that its own docstring calls not a model. This is the adapter that puts
`morpheus.models.dfencoder` in the same slot, so the composed layer 5 pipeline can be run with the model the
guide actually names -- on a machine that has Torch and a card, which the one this fork is developed in does not.

**Inference only, and the models arrive trained.** The adapter holds fitted models keyed by pinned version and
never fits one. Training inside the scoring path would fit on the rows being scored, a leak no determinism
control would catch because every run would leak identically; `ReferenceScorer` freezes its parameters for the
same reason. `train_dfencoder_models` below trains, on a caller-supplied frame per principal, and returns models
whose pinned version is a digest of their weights -- so two versions are equal exactly when two models are, which
is what control 1's "pin the model" has to mean for a model that was trained rather than downloaded.

**Nothing here imports Torch at module import.** The adapter is usable wherever a fitted model can be handed in,
including a stub in a test; only training reaches for Torch, and refuses in plain words when it is absent.

**What the runner measures with this is reproducibility of the wired path, not detection.** The models are
trained on the corpus and score the corpus. That is the leak stated above, made deliberately, because the
question the run answers is whether the pipeline gives the same scores twice with a real model in the slot --
and a week of five principals cannot answer any other question about a model.
"""

import dataclasses
import hashlib
import logging
import typing

import pandas as pd

from morpheus.utils.model_manifest import ModelManifest

logger = logging.getLogger(__name__)

MODEL_NAME_PREFIX = "dfencoder"
DIGEST_LENGTH = 16
DEFAULT_MIN_ROWS = 4


def _is_pinned(version: typing.Any) -> bool:
    return isinstance(version, str) and ":" in version and version.rsplit(":", 1)[1] != ""


class DfencoderScorer:
    """
    Score an entity's rows with the fitted autoencoder its pinned version names.

    Parameters
    ----------
    models : dict
        Pinned version (`name:version`) to a fitted model exposing `get_results(df, return_abs)` the way
        `morpheus.models.dfencoder.AutoEncoder` does. Anything with that method will do, which is what lets the
        alignment be tested without Torch.
    feature_columns : list of str
        The features, in the order the models were trained on. Every row handed to `score` must carry exactly
        these keys; a row with more or fewer is refused rather than reordered around.

    Raises
    ------
    ValueError
        If no models are given, a version is not pinned, a model has no `get_results`, or no features are named.
    """

    def __init__(self, models: dict, feature_columns: list[str]):
        if (not models):
            raise ValueError("models must hold at least one fitted model; a scorer with nothing behind it would "
                             "answer for every version the manifest could resolve")

        if (not feature_columns):
            raise ValueError("feature_columns must name at least one column")

        for (version, model) in models.items():
            if (not _is_pinned(version)):
                raise ValueError(f"model version {version!r} is not pinned as name:version; a bare name resolves "
                                 f"to whatever is current, which is the defect control 1 exists to prevent")

            if (not callable(getattr(model, "get_results", None))):
                raise ValueError(f"the model behind {version!r} has no get_results method; this adapter speaks "
                                 f"to morpheus.models.dfencoder.AutoEncoder or anything shaped like it")

        self._models = dict(models)
        self._feature_columns = list(feature_columns)

    @property
    def versions(self) -> list[str]:
        """The pinned versions this scorer can answer for, sorted."""
        return sorted(self._models)

    def score(self, model_version: str, features: list) -> list:
        """
        Score one entity's rows against the model pinned for it.

        Parameters
        ----------
        model_version : str
            The version the manifest resolved. Must be one this scorer holds.
        features : list of dict
            One dict per row, keyed by feature name.

        Returns
        -------
        list of dict
            One dict per input row, each feature's absolute z-score.

        Raises
        ------
        KeyError
            If the version is not held. A manifest pinning a version the scorer does not have is a wiring
            mistake, not a row to skip.
        ValueError
            If a row does not carry exactly the feature columns, or the model returns a frame of the wrong
            length or without a loss column per feature.
        """
        model = self._models.get(model_version)

        if (model is None):
            raise KeyError(f"no model is held for {model_version!r}; this scorer holds {self.versions}. The "
                           f"manifest pinned a version the scorer was not given, and scoring against a different "
                           f"one would make model_version on the row a lie.")

        if (len(features) == 0):
            return []

        expected = set(self._feature_columns)

        for (index, row) in enumerate(features):
            if (set(row) != expected):
                raise ValueError(f"row {index} carries {sorted(row)}, and the model was trained on "
                                 f"{self._feature_columns}; a row reordered or padded to fit would be scored on "
                                 f"the wrong features without anything saying so")

        frame = pd.DataFrame([[row[name] for name in self._feature_columns] for row in features],
                             columns=self._feature_columns).astype("float64")

        results = model.get_results(frame, return_abs=True)

        if (len(results) != len(features)):
            raise ValueError(f"the model returned {len(results)} rows for {len(features)}; scores that cannot be "
                             f"aligned back onto the rows are worse than none")

        results = results.reset_index(drop=True)
        loss_columns = {name: f"{name}_z_loss" for name in self._feature_columns}
        absent = [column for column in loss_columns.values() if column not in results.columns]

        if (absent):
            raise ValueError(f"the model's results carry no {absent}; a feature it was trained on has no loss, "
                             f"which means the model and the feature list disagree about what was trained")

        scored = []

        for position in range(len(features)):
            scored.append({name: abs(float(results[column].iloc[position])) for (name, column) in loss_columns.items()})

        return scored


@dataclasses.dataclass(frozen=True)
class TrainedModels:
    """What training produced: fitted models by pinned version, and which version each principal is pinned to."""

    models: dict
    versions: dict
    skipped: dict


def weight_digest(model) -> str:
    """
    A digest of a fitted model's parameters, so its pinned version says which weights it has.

    Each parameter tensor's bytes are hashed in sorted name order, on the host, independent of the device it
    was trained on and of how Torch would serialize it. Two models trained from the same seed on the same rows
    to the same weights get the same digest, which is the property a version has to have for "same version" to
    mean "same model".
    """
    state = model.model.state_dict() if hasattr(model, "model") else model.state_dict()
    digest = hashlib.sha256()

    for name in sorted(state):
        tensor = state[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(b"\x1e")

    return digest.hexdigest()[:DIGEST_LENGTH]


def train_dfencoder_models(frames: dict,
                           feature_columns: list[str],
                           seed: int,
                           epochs: int,
                           eval_batch_size: int,
                           encoder_layers: typing.Optional[list[int]] = None,
                           decoder_layers: typing.Optional[list[int]] = None,
                           min_rows: int = DEFAULT_MIN_ROWS) -> TrainedModels:
    """
    Train one autoencoder per principal and pin each to a digest of its weights.

    Parameters
    ----------
    frames : dict
        Principal to a frame of `feature_columns`, float, no nulls -- the shape `run_model.build_features`
        produces.
    feature_columns : list of str
        The features, in order.
    seed : int
        Re-seeded before each principal's training, so one principal's training cannot move another's weights
        through the shared generator.
    epochs : int
        Training epochs.
    eval_batch_size : int
        The batch size scoring uses. Control 5 says it must not matter; the runner measures that it does not.
    encoder_layers, decoder_layers : list of int, optional
        Layer sizes. The runner's defaults, `[8, 4]` and `[4, 8]`, are used when unset.
    min_rows : int, default = 4
        A principal with fewer usable rows is skipped rather than fitted on nothing, and reported.

    Returns
    -------
    `TrainedModels`

    Raises
    ------
    ImportError
        If Torch is absent. This is the one function here that needs it, and it says so rather than pretending.
    """
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise ImportError(f"training a dfencoder model needs Torch, which is not importable here: {error}. The "
                          f"adapter itself does not need it -- hand it fitted models -- but there is nothing to "
                          f"hand it without a machine that has Torch.") from error

    from morpheus.models.dfencoder import AutoEncoder  # pylint: disable=import-outside-toplevel
    from morpheus.utils.seed import manual_seed  # pylint: disable=import-outside-toplevel

    models: dict = {}
    versions: dict = {}
    skipped: dict = {}

    for (principal, frame) in sorted(frames.items()):
        usable = frame[list(feature_columns)].dropna().astype("float64").reset_index(drop=True)

        if (len(usable) < min_rows):
            skipped[principal] = len(usable)
            continue

        manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=False)

        model = AutoEncoder(encoder_layers=list(encoder_layers or [8, 4]),
                            decoder_layers=list(decoder_layers or [4, 8]),
                            batch_size=min(8, len(usable)),
                            eval_batch_size=eval_batch_size,
                            verbose=False,
                            progress_bar=False,
                            patience=-1)
        model.fit(usable, epochs=epochs)

        version = f"{MODEL_NAME_PREFIX}/{principal}:{weight_digest(model)}"
        models[version] = model
        versions[principal] = version

    if (skipped):
        logger.warning("train_dfencoder_models skipped %d principals with fewer than %d usable rows: %s",
                       len(skipped),
                       min_rows,
                       skipped)

    return TrainedModels(models=models, versions=versions, skipped=skipped)


def build_manifest(versions: dict, window_id: int) -> ModelManifest:
    """
    A manifest pinning each principal to its own model, with no fallback.

    No fallback, because a principal without a model of their own should be refused rather than scored
    against somebody else's -- which is the manifest's own rule, applied here on purpose. The caller who wants
    a population fallback declares one explicitly.
    """
    if (not versions):
        raise ValueError("versions must pin at least one principal; a manifest that pins nobody scores nobody")

    return ModelManifest(window_id=window_id, models=dict(versions), fallback=None)
