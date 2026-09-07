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

This is the third verdict this repository cannot render itself. `morpheus.models.dfencoder` is in the tree, but
it is a Torch model, and the environment this fork is developed and tested in has no Torch -- so the model half
of layer 5 is written here and run on a machine that has one. Everything it depends on that is not the model was
built and tested first, which is why the trajectory feature R-P-L5-006 reads, the determinism envelope, the
pinned manifest and the sharding are all merged already and this is the only piece that arrives unexercised.

**What this measures is reproducibility, not detection quality.** The corpus is a week of five principals'
authentications, which is nowhere near enough to train an autoencoder that detects anything, and no claim is
made that it does. What a week is enough for is the question the determinism controls exist to answer: run the
same training twice with the same seed and the same data, and find out whether the scores are identical. If they
are not, every threshold tuned against them is tuned against noise, and R-P-L5-006 -- a rule about a score
rising by fractions of a standard deviation -- is measuring the model's own jitter.

Three things are checked, each a control from Part 5:

- **Control 3, seeding.** `CUBLAS_WORKSPACE_CONFIG` is set before Torch is imported, because it has no effect
  once the CUDA context exists, and `torch.use_deterministic_algorithms(True)` is enabled so that an operation
  with no deterministic implementation raises at startup instead of silently producing a different answer.
- **Control 5, batching.** The same rows are scored at three batch sizes. A model with a batch-dependent
  operation -- batch normalization left in training mode is the usual one -- gives different scores for the same
  row depending on what it was batched with, and no amount of seeding fixes that.
- **The double run.** Two full train-and-score cycles, compared byte for byte after quantization.

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
    """The layer 5 corpus, through the five TC-5 stages, as one frame per principal."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "morpheus", "determinism"))

    import session_pipeline  # pylint: disable=import-outside-toplevel

    result = session_pipeline.run_pipeline(session_pipeline.build_pipeline_config(), session_pipeline.build_corpus())
    scored = result[result["telemetry_class"] == "tc5_auth"]

    frames = {}

    for principal in sorted(set(scored["user_principal"].dropna())):
        rows = scored[scored["user_principal"] == principal]
        features = rows[[column for column in FEATURE_COLUMNS if column in rows.columns]].copy()
        # A gap is not a value the model should learn a distribution for: these columns are counts and scores,
        # and a null means the row carried no such measurement rather than a measurement of zero. Dropping the
        # row is the honest reduction, and how many were dropped is reported.
        frames[principal] = features.dropna().astype("float64").reset_index(drop=True)

    return frames


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

    report = {
        "verdict": "passed" if (reproducible and batch_invariant) else "failed",
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
        "measures": "reproducibility of the scoring path, not detection quality: a week of five principals is "
                    "far too little data to train an autoencoder that detects anything, and no claim is made "
                    "that it does.",
    }

    with open(arguments.artifact, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(f"\n{json.dumps(report, indent=2)}")
    print(f"\n{'PASSED' if report['verdict'] == 'passed' else 'FAILED'}. Artifact written to {arguments.artifact}.")

    return 0 if report["verdict"] == "passed" else 1


if (__name__ == "__main__"):
    sys.exit(main())
