#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Trains a per-principal autoencoder on the layer 5 corpus and asks whether it produces the same numbers twice.

This is the third verdict this repository cannot render itself. The models the composed pipeline scores with in CI
were trained on a CPU by `train_models.py` and are committed as numbers; what no CI here can say is whether
training and scoring on a CUDA device give the same numbers twice, which is what a deployment's scores come from.

**What this measures is reproducibility, not detection quality.** The training data is a fortnight of six
principals' authentications, which is nowhere near enough to train an autoencoder that detects anything in
general, and no claim is made that it does. What a fortnight is enough for is the question the determinism
controls exist to answer: run the same training twice with the same seed and the same data, and find out whether
the scores are identical. If they are not, every threshold tuned against them is tuned against noise, and
R-P-L5-006 -- a rule about a score rising by fractions of a standard deviation -- is measuring the model's own
jitter.

Four things are checked, each a control from Part 5:

- **Control 3, seeding.** `CUBLAS_WORKSPACE_CONFIG` is set before Torch is imported, because it has no effect
  once the CUDA context exists, and `torch.use_deterministic_algorithms(True)` is enabled so that an operation
  with no deterministic implementation raises at startup instead of silently producing a different answer.
- **Control 5, batching.** The same rows are scored at three batch sizes. A model with a batch-dependent
  operation -- batch normalization left in training mode is the usual one -- gives different scores for the same
  row depending on what it was batched with, and no amount of seeding fixes that.
- **The double run.** Two full train-and-score cycles, compared byte for byte after quantization.
- **The wired path.** The trained models are placed behind `TC5ScoreStage` through the session corpus's
  `PinnedScorer`, with a manifest pinning each principal to a digest of their model and the population fallback
  for the joiner who has none, and the composed layer 5 pipeline is run twice and then under the batch-split
  sweep. The models are trained on the corpus's fortnight and score the week after it, as the committed models CI
  scores with are: this is the same path with Torch's own numbers in it, on the device.

The artifact it writes is the evidence, in the same shape as `gpu_conformance.json`: what ran, on what card, with
what result, so a number in a document can be traced to a run rather than to a memory of one.
"""

import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

FEATURE_COLUMNS = [
    "logcount",
    "locincrement",
    "appincrement",
    "deviceincrement",
    "asns_in_window",
    "hour_surprise_bits",
    "weekday_surprise_bits",
    "mfa_ratio",
    "auth_attempts_in_window",
    "auth_failures_in_window",
]
"""The numeric features the model is trained on, all produced by the five TC-5 stages.

Deliberately not the raw columns a collector sent. A model trained on `source_country` learns which countries an
estate has, which is a fact about the estate; one trained on `locincrement` learns how often this principal goes
somewhere new, which is a fact about the principal. The guide's whole argument for this layer is the second one.
"""

DEFAULT_EPOCHS = 20
DEFAULT_SEED = 42
BATCH_SIZES = (1, 8, 64)


def fail(message: str, artifact: str) -> int:
    """Write a failed verdict and say why, so a run that could not happen is not mistaken for one that passed."""
    print(f"\nFAILED: {message}\n", file=sys.stderr)

    with open(artifact, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "verdict": "failed",
                "reason": message,
                "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            handle,
            indent=2)
        handle.write("\n")

    return 1


def prepare_environment(seed: int) -> None:
    """
    Control 3, in the only order that works.

    `CUBLAS_WORKSPACE_CONFIG` is read when the CUDA context is created, so setting it after anything has imported
    Torch is setting it too late -- and setting it too late looks exactly like setting it correctly, because
    nothing complains. The guide says to set it in the container entrypoint for that reason; this refuses instead
    of pretending, which is the same argument one step further.
    """
    if ("torch" in sys.modules):
        raise RuntimeError("torch was imported before CUBLAS_WORKSPACE_CONFIG could be set. The variable is read "
                           "when the CUDA context is created, so setting it now would have no effect and would "
                           "look identical to setting it correctly. Run this script directly rather than "
                           "importing it after Torch.")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def build_features():
    """The layer 5 corpus's training fortnight, through the five TC-5 stages, as one frame per principal."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "morpheus", "determinism"))

    import session_pipeline  # pylint: disable=import-outside-toplevel

    if (list(session_pipeline.SCORED_FEATURES) != FEATURE_COLUMNS):
        raise RuntimeError("the runner's features and the pipeline's scored features disagree")

    # The committed models score this run's pipeline by default; the features do not depend on any score, and
    # what is trained here is trained on the fortnight alone, as they were.
    result = session_pipeline.run_pipeline(session_pipeline.build_pipeline_config(), session_pipeline.build_corpus())

    return session_pipeline.training_frames(result)


def pipeline_checks(frames: dict, seed: int, epochs: int, eval_batch_size: int) -> dict:
    """
    The fourth check: the composed pipeline, with the trained models in the scoring slot.

    Trains once, pins each principal to the digest of its own weights, runs the pipeline twice, then runs the
    same batch-split sweep the harness runs against the reference scorer. Returns what the artifact records.
    """
    import pandas as pd  # pylint: disable=import-outside-toplevel

    import session_pipeline  # pylint: disable=import-outside-toplevel
    from morpheus.utils.determinism import diff_frames  # pylint: disable=import-outside-toplevel
    from morpheus.utils.dfencoder_scorer import load_models  # pylint: disable=import-outside-toplevel
    from morpheus.utils.dfencoder_scorer import train_dfencoder_models  # pylint: disable=import-outside-toplevel

    trained = train_dfencoder_models(frames, FEATURE_COLUMNS, seed, epochs, eval_batch_size)
    (scorer, manifest) = session_pipeline.build_scoring(
        {principal: (version, trained.models[version])
         for (principal, version) in trained.versions.items()})
    committed = {principal: version for (principal, (version, _)) in load_models(session_pipeline.MODELS_PATH).items()}

    config = session_pipeline.build_pipeline_config()
    corpus = session_pipeline.build_corpus()

    first = session_pipeline.run_pipeline(config, corpus, scorer=scorer, manifest=manifest)
    second = session_pipeline.run_pipeline(config, corpus, scorer=scorer, manifest=manifest)
    double_run = diff_frames(first, second) is None

    def split(frame: pd.DataFrame, parts: int) -> list:
        size = max(1, len(frame) // parts)
        return [frame.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(frame), size)]

    thirds = {name: split(frame, 3) for (name, frame) in corpus.items()}
    by_row = {name: split(frame, len(frame)) for (name, frame) in corpus.items()}

    # The disagreement itself is recorded, not just that there was one. A sweep that says only "DIFFERENT"
    # sends whoever reads the artifact back to the machine to find out what moved, and the machine with the
    # card is usually not the one asking.
    sweep_differences = {}

    for (label, batches) in (("thirds", thirds), ("by_row", by_row)):
        difference = diff_frames(
            first, session_pipeline.run_pipeline(config, corpus, batches=batches, scorer=scorer, manifest=manifest))

        if (difference is not None):
            sweep_differences[label] = difference

    differences = dict(sweep_differences)

    if (not double_run):
        differences["second_run"] = diff_frames(first, second)

    scored = first[first["telemetry_class"] == "tc5_auth"]

    return {
        # Whether the device trained the same numbers the CPU committed. Not part of the verdict: a different
        # processor sums in a different order, and the committed file is what CI pins, not a recipe for it.
        "committed_models_match": dict(trained.versions) == committed,
        "pipeline_double_run_reproducible": double_run,
        "pipeline_batch_invariant": not sweep_differences,
        "pipeline_differences": differences,
        "pipeline_scored_rows": int(scored["mean_abs_z"].notna().sum()),
        "pipeline_principals_pinned": len(trained.versions),
        "pipeline_principals_skipped": dict(trained.skipped),
        "model_versions": dict(trained.versions),
        "pipeline_mean_abs_z_max": float(scored["mean_abs_z"].astype(float).max()),
    }


def train_and_score(frames: dict, seed: int, epochs: int, eval_batch_size: int) -> dict:
    """One full cycle: seed, train a model per principal, score its own rows, return the scores."""
    from morpheus.models.dfencoder import AutoEncoder  # pylint: disable=import-outside-toplevel
    from morpheus.utils.determinism import quantize_value  # pylint: disable=import-outside-toplevel
    from morpheus.utils.seed import manual_seed  # pylint: disable=import-outside-toplevel

    import torch  # pylint: disable=import-outside-toplevel

    scores = {}

    for (principal, features) in sorted(frames.items()):
        if (len(features) < 4):
            continue

        # Re-seeded per principal, so that one principal's training cannot move another's scores through the
        # shared generator -- which would make the output depend on how many principals happened to precede it.
        manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=False)

        model = AutoEncoder(encoder_layers=[8, 4],
                            decoder_layers=[4, 8],
                            batch_size=min(8, len(features)),
                            eval_batch_size=eval_batch_size,
                            verbose=False,
                            progress_bar=False,
                            patience=-1)
        model.fit(features, epochs=epochs)

        results = model.get_results(features)
        scores[principal] = [quantize_value(float(value)) for value in results["mean_abs_z"].tolist()]

    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", default=os.path.join(REPO_ROOT, "layer5_model.json"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()

    try:
        prepare_environment(arguments.seed)
    except RuntimeError as error:
        return fail(str(error), arguments.artifact)

    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        return fail(
            f"no Torch: {error}. The layer 5 model is a Torch model, and this verdict needs a machine with "
            f"Torch and a CUDA device. It cannot be rendered in CI or in a container without them, and a run "
            f"that quietly skipped the model would be worse than one that refuses.",
            arguments.artifact)

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    if (device is None):
        return fail(
            "Torch reports no CUDA device. The scores this measures are the ones a deployment would act on, "
            "and a CPU run would answer a different question than the one asked.",
            arguments.artifact)

    print(f"=== device ===\n{device}\ntorch {torch.__version__}\n")
    print("=== features ===")
    frames = build_features()

    for (principal, features) in sorted(frames.items()):
        print(f"{principal:<28} {len(features):>4} rows, {len(features.columns)} features")

    print("\n=== double run ===")
    first = train_and_score(frames, arguments.seed, arguments.epochs, BATCH_SIZES[-1])
    second = train_and_score(frames, arguments.seed, arguments.epochs, BATCH_SIZES[-1])
    reproducible = first == second
    print("identical" if reproducible else "DIFFERENT: the same seed and the same data gave different scores")

    print("\n=== batch invariance ===")
    by_batch = {size: train_and_score(frames, arguments.seed, arguments.epochs, size) for size in BATCH_SIZES}
    batch_invariant = all(by_batch[size] == by_batch[BATCH_SIZES[0]] for size in BATCH_SIZES)
    print("identical across " + ", ".join(
        str(size)
        for size in BATCH_SIZES) if batch_invariant else "DIFFERENT: the model has a batch-dependent operation")

    print("\n=== the wired path ===")
    wired = pipeline_checks(frames, arguments.seed, arguments.epochs, BATCH_SIZES[-1])
    print("pipeline double run: " + ("identical" if wired["pipeline_double_run_reproducible"] else "DIFFERENT"))
    print("pipeline batch sweep: " + ("identical" if wired["pipeline_batch_invariant"] else "DIFFERENT"))

    for (label, difference) in sorted(wired["pipeline_differences"].items()):
        print(f"  {label}: {difference}")

    print(f"{wired['pipeline_scored_rows']} rows scored against {wired['pipeline_principals_pinned']} pinned models")

    verdict_passed = (reproducible and batch_invariant and wired["pipeline_double_run_reproducible"]
                      and wired["pipeline_batch_invariant"])

    report = {
        "verdict": "passed" if verdict_passed else "failed",
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
        "torch": torch.__version__,
        "seed": arguments.seed,
        "epochs": arguments.epochs,
        "principals": {
            principal: len(features)
            for (principal, features) in sorted(frames.items())
        },
        "features": list(FEATURE_COLUMNS),
        "double_run_reproducible": reproducible,
        "batch_invariant": batch_invariant,
        "batch_sizes": list(BATCH_SIZES),
        **wired,
        "measures": "reproducibility of the scoring path, not detection quality: a fortnight of six principals "
                    "is far too little data to train an autoencoder that detects anything in general, and no claim "
                    "is made that it does. The models are trained on the corpus's fortnight and the wired-path "
                    "checks score the week after it.",
    }

    with open(arguments.artifact, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(f"\n{json.dumps(report, indent=2)}")
    print(f"\n{'PASSED' if report['verdict'] == 'passed' else 'FAILED'}. Artifact written to {arguments.artifact}.")

    return 0 if report["verdict"] == "passed" else 1


if (__name__ == "__main__"):
    sys.exit(main())
