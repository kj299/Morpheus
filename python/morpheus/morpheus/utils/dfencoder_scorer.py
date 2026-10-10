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
whose pinned version is a digest of every number they score with -- so two versions are equal exactly when two
models are, which is what control 1's "pin the model" has to mean for a model that was trained rather than
downloaded.

**A trained model is committed as numbers, and scored without Torch.** `export_model` writes what
`get_results` depends on -- the weights, the input scaling and the training losses' scaling -- as plain JSON,
and `NumpyAutoEncoder` evaluates it. That is how the composed layer 5 pipeline scores with a learned model in
CI, where there is no Torch: `examples/layer5_model/train_models.py` trains on the corpus's training window and
commits the result, and `load_models` reads it back and refuses any entry whose numbers no longer digest to the
version recorded beside them. `load_torch_autoencoder` puts the same numbers back into the upstream class, so
the numpy pass is checked against the thing it reimplements wherever Torch is installed.

**Nothing here imports Torch at module import.** The adapter is usable wherever a fitted model can be handed in,
including a stub in a test; only training and the reverse load reach for Torch, and training refuses in plain
words when it is absent.

**A row is scored on its own, because control 5 says batching must be irrelevant.** `TC5ScoreStage` hands over
the rows one entity has *in the message it is processing*, so the size of that group is a function of how the
stream was chunked, not of the data. A neural network is not shape-invariant to the last decimal: the same row
in a batch of twenty and in a batch of one goes through different kernels and can come back differing in the
seventh place. That difference does not stay small. `drift_rise_sigmas` divides a rise by the spread of a few
nearly-equal scores, so a seventh-place wobble upstream lands as a difference of tenths downstream, far too
large for quantization to absorb. So the adapter fixes the shape itself: `rows_per_call` rows go to the model
at a time, one by default, and a row's score is then a function of the row rather than of its company. Raising
it trades control 5 for throughput, because the last chunk of a group is only as big as the group leaves it.
"""

import dataclasses
import hashlib
import logging
import typing

import pandas as pd

from morpheus.utils.model_manifest import ModelManifest

logger = logging.getLogger(__name__)

MODEL_NAME_PREFIX = "dfencoder"
DEFAULT_ROWS_PER_CALL = 1
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
    rows_per_call : int, default = 1
        How many rows go to the model at a time. The default of one makes a row's score a function of the row:
        the model sees the same shape however the pipeline chunked the stream, which is what control 5 asks for.
        Raising it batches the model's work and gives that up, because the last chunk of an entity's rows is
        only as big as the group leaves it.

    Raises
    ------
    ValueError
        If no models are given, a version is not pinned, a model has no `get_results`, no features are named,
        or `rows_per_call` is not a positive integer.
    """

    def __init__(self, models: dict, feature_columns: list[str], rows_per_call: int = DEFAULT_ROWS_PER_CALL):
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

        if (not isinstance(rows_per_call, int) or isinstance(rows_per_call, bool) or rows_per_call < 1):
            raise ValueError(f"rows_per_call must be a positive integer, not {rows_per_call!r}; there is no "
                             f"batch size at which a model is asked for no rows")

        self._models = dict(models)
        self._feature_columns = list(feature_columns)
        self._rows_per_call = rows_per_call

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

        scored = []

        for start in range(0, len(features), self._rows_per_call):
            scored.extend(self._score_chunk(model, features[start:start + self._rows_per_call]))

        return scored

    def _score_chunk(self, model: typing.Any, rows: list) -> list:
        """Score one fixed-size chunk. The shape the model sees is `rows_per_call`, never the group's size."""
        frame = pd.DataFrame([[row[name] for name in self._feature_columns] for row in rows],
                             columns=self._feature_columns).astype("float64")

        results = model.get_results(frame, return_abs=True)

        if (len(results) != len(rows)):
            raise ValueError(f"the model returned {len(results)} rows for {len(rows)}; scores that cannot be "
                             f"aligned back onto the rows are worse than none")

        results = results.reset_index(drop=True)
        loss_columns = {name: f"{name}_z_loss" for name in self._feature_columns}
        absent = [column for column in loss_columns.values() if column not in results.columns]

        if (absent):
            raise ValueError(f"the model's results carry no {absent}; a feature it was trained on has no loss, "
                             f"which means the model and the feature list disagree about what was trained")

        return [{
            name: abs(float(results[column].iloc[position]))
            for (name, column) in loss_columns.items()
        } for position in range(len(rows))]


@dataclasses.dataclass(frozen=True)
class TrainedModels:
    """What training produced: fitted models by pinned version, and which version each principal is pinned to."""

    models: dict
    versions: dict
    skipped: dict


MODEL_FORMAT = "morpheus/dfencoder-numpy/1"
"""What `export_model` writes and `NumpyAutoEncoder` reads. Bumped if either side's arithmetic changes."""

SUPPORTED_ACTIVATIONS = ("relu", None)
"""The activations the numpy forward pass evaluates. The runner trains with `relu` throughout; anything else is
refused at export rather than approximated at inference."""


def export_model(model, feature_columns: list[str]) -> dict:
    """
    Everything a fitted autoencoder needs to score a row, as plain numbers.

    The weights alone are not the model. `get_results` standardizes each feature with the mean and deviation
    seen in training, runs the network, takes the per-feature squared error, and standardizes that against the
    errors seen in training. The two sets of statistics are as much a part of the answer as the weights, so
    they are exported beside them and covered by the same digest.

    Parameters
    ----------
    model : `morpheus.models.dfencoder.AutoEncoder`
        A fitted model whose features are all numeric, with the `standard` scaler on both the inputs and the
        losses and no categorical or binary features -- the shape `train_dfencoder_models` produces.
    feature_columns : list of str
        The features, in the order the model was trained on.

    Returns
    -------
    dict
        JSON-serializable. Weights are written as the exact decimal of their float32 value, so reading them back
        reproduces every bit.

    Raises
    ------
    ValueError
        If the model has a feature, scaler or activation the numpy forward pass does not evaluate.
    """
    if (list(model.numeric_fts) != list(feature_columns)):
        raise ValueError(f"the model's numeric features {list(model.numeric_fts)} are not {feature_columns}; an "
                         f"export in a different order would score each feature against another's weights")

    if (model.binary_fts or model.categorical_fts):
        raise ValueError("the model has binary or categorical features; the numpy forward pass evaluates numeric "
                         "features only")

    if (model.loss_scaler_str != "standard"):
        raise ValueError(f"the loss scaler is {model.loss_scaler_str!r}; only 'standard' is evaluated")

    layers = []

    for (stack, modules) in (("encoder", model.model.encoder), ("decoder", model.model.decoder)):
        for module in modules:
            if (module.activation not in SUPPORTED_ACTIVATIONS):
                raise ValueError(f"a {stack} layer uses {module.activation!r}; only {SUPPORTED_ACTIVATIONS} are "
                                 f"evaluated")

            layers.append({
                "stack": stack,
                "activation": module.activation,
                "weight": _floats(module.linear_layer.weight),
                "bias": _floats(module.linear_layer.bias),
            })

    numeric_scaler = {}

    for name in feature_columns:
        scaler = model.numeric_fts[name]["scaler"]

        if (type(scaler).__name__ != "StandardScaler"):
            raise ValueError(f"{name} is scaled by {type(scaler).__name__}; only StandardScaler is evaluated")

        numeric_scaler[name] = {"mean": float(scaler.mean), "std": float(scaler.std)}

    return {
        "format": MODEL_FORMAT,
        "features": list(feature_columns),
        "layers": layers,
        "output": {
            "weight": _floats(model.model.numeric_output.weight), "bias": _floats(model.model.numeric_output.bias)
        },
        "numeric_scaler": numeric_scaler,
        "loss_scaler": {
            name: {
                "mean": float(model.feature_loss_stats[name]["scaler"].mean),
                "std": float(model.feature_loss_stats[name]["scaler"].std),
            }
            for name in feature_columns
        },
    }


def _floats(tensor) -> list:
    """A parameter tensor as nested lists of the exact float32 values, independent of the device it lives on."""
    return tensor.detach().cpu().contiguous().numpy().astype("float32").tolist()


def model_digest(document: dict) -> str:
    """
    A digest of an exported model, so the version a manifest pins says which numbers score the row.

    Every weight is hashed as float32 and every statistic as float64, in a fixed order, from the document rather
    than from Torch: a model loaded from its committed file and the same model straight out of training give the
    same digest, which is what lets a machine with no Torch check that the file it holds is the version pinned.
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    digest = hashlib.sha256()

    def feed(name: str, values, dtype: str):
        array = np.asarray(values, dtype=dtype)
        digest.update(name.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(f"{dtype}{array.shape}".encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(b"\x1e")

    digest.update(document["format"].encode("utf-8"))
    feed("features", [len(name) for name in document["features"]], "int64")
    digest.update("\x1f".join(document["features"]).encode("utf-8"))

    for (index, layer) in enumerate(document["layers"]):
        prefix = f"{layer['stack']}.{index}.{layer['activation']}"
        feed(f"{prefix}.weight", layer["weight"], "float32")
        feed(f"{prefix}.bias", layer["bias"], "float32")

    feed("output.weight", document["output"]["weight"], "float32")
    feed("output.bias", document["output"]["bias"], "float32")

    for section in ("numeric_scaler", "loss_scaler"):
        for name in document["features"]:
            feed(f"{section}.{name}", [document[section][name]["mean"], document[section][name]["std"]], "float64")

    return digest.hexdigest()[:DIGEST_LENGTH]


class NumpyAutoEncoder:
    """
    An exported autoencoder, scored without Torch.

    The arithmetic is `AutoEncoder.get_results` for a numeric-only model, written out: standardize each feature,
    run the encoder and decoder, take each feature's squared reconstruction error, and standardize the error
    against the errors seen in training. It is evaluated in float64 from the float32 weights, which is the
    choice that makes the score a property of the row and the file rather than of the machine: a float32
    product summed in a different order by a different BLAS moves the last bits, and a loss whose training
    spread was small turns a last-bit difference into one in the fourth decimal that `TC5ScoreStage` publishes.
    Torch's own float32 pass agrees with this to within float32's precision, which
    `tests/morpheus/utils/test_dfencoder_scorer.py` checks wherever Torch is installed.

    Parameters
    ----------
    document : dict
        What `export_model` wrote.

    Raises
    ------
    ValueError
        If the document is another format, or its shapes do not chain from the features through the layers and
        back.
    """

    def __init__(self, document: dict):
        import numpy as np  # pylint: disable=import-outside-toplevel

        if (document.get("format") != MODEL_FORMAT):
            raise ValueError(f"the model file is {document.get('format')!r}, and this reads {MODEL_FORMAT!r}")

        self._features = list(document["features"])
        self._layers = []
        width = len(self._features)

        for layer in document["layers"]:
            weight = np.asarray(layer["weight"], dtype="float32").astype("float64")
            bias = np.asarray(layer["bias"], dtype="float32").astype("float64")

            if (weight.ndim != 2 or weight.shape[1] != width or bias.shape != (weight.shape[0], )):
                raise ValueError(f"a {layer['stack']} layer of shape {weight.shape} does not follow one of width "
                                 f"{width}; the file's layers do not chain")

            if (layer["activation"] not in SUPPORTED_ACTIVATIONS):
                raise ValueError(f"activation {layer['activation']!r} is not one this evaluates")

            self._layers.append((weight, bias, layer["activation"]))
            width = weight.shape[0]

        self._output_weight = np.asarray(document["output"]["weight"], dtype="float32").astype("float64")
        self._output_bias = np.asarray(document["output"]["bias"], dtype="float32").astype("float64")

        if (self._output_weight.shape != (len(self._features), width)):
            raise ValueError(f"the output layer is {self._output_weight.shape}, and reconstructing "
                             f"{len(self._features)} features from width {width} needs "
                             f"{(len(self._features), width)}")

        def stats(section: str) -> tuple:
            return (np.array([document[section][name]["mean"] for name in self._features], dtype="float64"),
                    np.array([document[section][name]["std"] for name in self._features], dtype="float64"))

        (self._input_mean, self._input_std) = stats("numeric_scaler")
        (self._loss_mean, self._loss_std) = stats("loss_scaler")
        self._version_digest = model_digest(document)

    @property
    def digest(self) -> str:
        """The digest of the document this was built from."""
        return self._version_digest

    @property
    def features(self) -> list[str]:
        """The features, in training order."""
        return list(self._features)

    def get_results(self, df: pd.DataFrame, return_abs: bool = False) -> pd.DataFrame:
        """
        Score rows the way `AutoEncoder.get_results` does, returning the columns `DfencoderScorer` reads.

        Parameters
        ----------
        df : `pandas.DataFrame`
            One column per feature. A null is refused rather than filled: `TC5ScoreStage` never sends one, and a
            filled value would be scored as a measurement nobody made.
        return_abs : bool, default = False
            Return the absolute z-scores.

        Returns
        -------
        `pandas.DataFrame`
            `<feature>_loss` and `<feature>_z_loss` per feature, then `max_abs_z` and `mean_abs_z`.
        """
        import numpy as np  # pylint: disable=import-outside-toplevel

        values = df[self._features].to_numpy(dtype="float64")

        if (not np.isfinite(values).all()):
            raise ValueError("a row carries a null or non-finite feature; this scores measurements, not gaps")

        scaled = (values - self._input_mean) / self._input_std
        hidden = scaled

        for (weight, bias, activation) in self._layers:
            hidden = hidden @ weight.T + bias

            if (activation == "relu"):
                hidden = np.maximum(hidden, 0.0)

        reconstructed = hidden @ self._output_weight.T + self._output_bias
        loss = (reconstructed - scaled)**2
        z_loss = (loss - self._loss_mean) / self._loss_std

        if (return_abs):
            z_loss = np.abs(z_loss)

        result = pd.DataFrame(index=df.index)

        for (position, name) in enumerate(self._features):
            result[f"{name}_loss"] = loss[:, position]
            result[f"{name}_z_loss"] = z_loss[:, position]

        absolute = np.abs(z_loss)
        result["max_abs_z"] = absolute.max(axis=1)
        result["mean_abs_z"] = absolute.mean(axis=1)

        return result


def load_models(path: str) -> dict:
    """
    Read a committed model file: principal to (pinned version, `NumpyAutoEncoder`).

    The file is what `examples/layer5_model/train_models.py` writes: a `models` object of exported documents
    keyed by principal, each beside the version it was pinned to when it was trained. That version is
    recomputed here from the numbers and must match, so a file edited by hand, or a document moved to another
    principal's key, is refused rather than scored under a version that no longer describes it.

    Raises
    ------
    ValueError
        If a recorded version is not the digest of the document beside it.
    """
    import json  # pylint: disable=import-outside-toplevel

    with open(path, encoding="utf-8") as handle:
        recorded = json.load(handle)

    loaded = {}

    for (principal, entry) in sorted(recorded["models"].items()):
        model = NumpyAutoEncoder(entry["model"])
        version = f"{MODEL_NAME_PREFIX}/{principal}:{model.digest}"

        if (entry["model_version"] != version):
            raise ValueError(f"{path} records {entry['model_version']!r} for {principal}, and the numbers beside "
                             f"it digest to {version!r}. The file was changed after it was trained, or the entry "
                             f"was moved; scoring under either version would make model_version on the row a lie.")

        loaded[principal] = (version, model)

    return loaded


def load_torch_autoencoder(document: dict):
    """
    Rebuild an exported model as a `morpheus.models.dfencoder.AutoEncoder` on the CPU.

    The reverse of `export_model`, for the machine that has Torch: the committed numbers go back into the
    upstream class, and its own `get_results` answers. That is how the numpy forward pass is checked against the
    thing it reimplements, rather than against a second reimplementation.

    Raises
    ------
    ImportError
        If Torch is absent.
    ValueError
        If the document is another format.
    """
    import torch  # pylint: disable=import-outside-toplevel

    from morpheus.models.dfencoder import AutoEncoder  # pylint: disable=import-outside-toplevel
    from morpheus.models.dfencoder.scalers import StandardScaler  # pylint: disable=import-outside-toplevel

    if (document.get("format") != MODEL_FORMAT):
        raise ValueError(f"the model file is {document.get('format')!r}, and this reads {MODEL_FORMAT!r}")

    features = list(document["features"])
    encoder = [layer for layer in document["layers"] if layer["stack"] == "encoder"]
    decoder = [layer for layer in document["layers"] if layer["stack"] == "decoder"]

    model = AutoEncoder(encoder_layers=[len(layer["bias"]) for layer in encoder],
                        decoder_layers=[len(layer["bias"]) for layer in decoder],
                        encoder_activations=[layer["activation"] for layer in encoder],
                        decoder_activations=[layer["activation"] for layer in decoder],
                        device=torch.device("cpu"),
                        preset_cats={},
                        binary_feature_list=[],
                        preset_numerical_scaler_params={
                            name: {
                                "scaler_type": "standard",
                                "scaler_attr_dict": dict(document["numeric_scaler"][name]),
                                "mean": document["numeric_scaler"][name]["mean"],
                                "std": document["numeric_scaler"][name]["std"],
                            }
                            for name in features
                        },
                        verbose=False,
                        progress_bar=False,
                        patience=-1)
    model._build_model()  # pylint: disable=protected-access

    with torch.no_grad():
        for (module, layer) in zip(list(model.model.encoder) + list(model.model.decoder), encoder + decoder):
            module.linear_layer.weight.copy_(torch.tensor(layer["weight"], dtype=torch.float32))
            module.linear_layer.bias.copy_(torch.tensor(layer["bias"], dtype=torch.float32))

        model.model.numeric_output.weight.copy_(torch.tensor(document["output"]["weight"], dtype=torch.float32))
        model.model.numeric_output.bias.copy_(torch.tensor(document["output"]["bias"], dtype=torch.float32))

    for name in features:
        scaler = StandardScaler()
        scaler.mean = document["loss_scaler"][name]["mean"]
        scaler.std = document["loss_scaler"][name]["std"]
        model.feature_loss_stats[name] = {"scaler": scaler}

    model.eval()

    return model


def train_dfencoder_models(frames: dict,
                           feature_columns: list[str],
                           seed: int,
                           epochs: int,
                           eval_batch_size: int,
                           encoder_layers: typing.Optional[list[int]] = None,
                           decoder_layers: typing.Optional[list[int]] = None,
                           min_rows: int = DEFAULT_MIN_ROWS,
                           device: typing.Optional[str] = None) -> TrainedModels:
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
        The batch size the fitted model evaluates in. This is the model's own knob, not the adapter's:
        `DfencoderScorer` fixes the shape it asks in separately, so that the score of a row does not depend on
        how the pipeline chunked the stream.
    encoder_layers : list of int, optional
        Encoder layer sizes. The runner's default, `[8, 4]`, is used when unset.
    decoder_layers : list of int, optional
        Decoder layer sizes. The runner's default, `[4, 8]`, is used when unset.
    min_rows : int, default = 4
        A principal with fewer usable rows is skipped rather than fitted on nothing, and reported.
    device : str, optional
        Where to train. Unset, the upstream default: the first CUDA device when there is one. `"cpu"` trains on
        the host and seeds only the host's generators, which is how the committed models were trained on a
        machine with no card.

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

        manual_seed(seed, cpu_only=device == "cpu")
        torch.use_deterministic_algorithms(True, warn_only=False)

        model = AutoEncoder(encoder_layers=list(encoder_layers or [8, 4]),
                            decoder_layers=list(decoder_layers or [4, 8]),
                            batch_size=min(8, len(usable)),
                            eval_batch_size=eval_batch_size,
                            verbose=False,
                            progress_bar=False,
                            patience=-1,
                            device=None if device is None else torch.device(device))
        model.fit(usable, epochs=epochs)

        version = f"{MODEL_NAME_PREFIX}/{principal}:{model_digest(export_model(model, feature_columns))}"
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
