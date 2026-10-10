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
Trains the per-principal models the layer 5 pipeline scores with, and commits them as numbers.

Each principal with enough rows in the session corpus's training fortnight gets an autoencoder fitted to that
fortnight alone, on the host, from a fixed seed. The fitted models are exported -- weights, input scaling and the
training losses' scaling -- and written to `models/session_models.json`, each beside a version that is the digest
of its numbers. `tests/morpheus/determinism/session_pipeline.py` reads that file, refuses any entry whose numbers
no longer digest to its version, and scores the week after the fortnight with them through a numpy forward pass,
which is how a learned model scores in CI where there is no Torch.

**Trained on the CPU, on purpose.** These are the numbers CI scores with, and the machine that trains them only
has to be one with Torch. The card's own run, `run_model.py`, measures something else: whether training and
scoring on a device give the same numbers twice. Retraining here on another processor may land on different last
bits -- a different vector unit sums in a different order -- and that is why the file is committed rather than
regenerated: the pinned version names these numbers, not a recipe for them. The script says whether its result
matches the committed file, so a difference is visible rather than silently replacing it.

Run it with the Morpheus environment and Torch:

    ./examples/layer5_model/train_models.py              # train and compare with the committed file
    ./examples/layer5_model/train_models.py --write      # train and replace it
"""

import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODELS_PATH = os.path.join(HERE, "models", "session_models.json")

DEFAULT_EPOCHS = 20
DEFAULT_SEED = 42
EVAL_BATCH_SIZE = 64


def training_frames() -> dict:
    """The fortnight's features per principal, from a pipeline run that needs no model to produce them."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "morpheus", "determinism"))

    import session_pipeline  # pylint: disable=import-outside-toplevel
    from morpheus.utils.model_manifest import ModelManifest  # pylint: disable=import-outside-toplevel

    # The features are what the stages emit and do not depend on any score, so the frozen arithmetic stands in
    # for the scorer here; it is the only scorer that exists before the models do.
    placeholder = ModelManifest(window_id=session_pipeline.SCORING_WINDOW,
                                models={},
                                fallback=session_pipeline.FALLBACK_VERSION)
    result = session_pipeline.run_pipeline(session_pipeline.build_pipeline_config(),
                                           session_pipeline.build_corpus(),
                                           scorer=session_pipeline.ReferenceScorer(),
                                           manifest=placeholder)

    return session_pipeline.training_frames(result), session_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="replace the committed file with this run's models")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()

    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        print(f"no Torch: {error}. Training needs it; scoring the committed models does not.", file=sys.stderr)
        return 1

    from morpheus.utils.dfencoder_scorer import export_model  # pylint: disable=import-outside-toplevel
    from morpheus.utils.dfencoder_scorer import train_dfencoder_models  # pylint: disable=import-outside-toplevel

    (frames, session_pipeline) = training_frames()
    features = list(session_pipeline.SCORED_FEATURES)

    for (principal, frame) in sorted(frames.items()):
        print(f"{principal:<28} {len(frame):>4} training rows")

    trained = train_dfencoder_models(frames,
                                     features,
                                     seed=arguments.seed,
                                     epochs=arguments.epochs,
                                     eval_batch_size=EVAL_BATCH_SIZE,
                                     device="cpu")

    document = {
        "trained": {
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "torch": torch.__version__,
            "device": "cpu",
            "seed": arguments.seed,
            "epochs": arguments.epochs,
            "training_days": session_pipeline.TRAINING_DAYS,
            "scores_from_ns": session_pipeline.SCORES_FROM_NS,
            "rows": {
                principal: len(frame)
                for (principal, frame) in sorted(frames.items())
            },
            "skipped": dict(trained.skipped),
        },
        "models": {
            principal: {
                "model_version": version, "model": export_model(trained.models[version], features)
            }
            for (principal, version) in sorted(trained.versions.items())
        },
    }

    print()

    for (principal, version) in sorted(trained.versions.items()):
        print(f"{principal:<28} {version}")

    committed = None

    if (os.path.exists(MODELS_PATH)):
        with open(MODELS_PATH, encoding="utf-8") as handle:
            recorded = json.load(handle)["models"]

        committed = {principal: entry["model_version"] for (principal, entry) in recorded.items()}

    if (committed == dict(trained.versions)):
        print(f"\nIdentical to the committed models in {MODELS_PATH}.")
    elif (committed is not None):
        print(f"\nDIFFERENT from the committed models in {MODELS_PATH}: {committed}")

    if (arguments.write):
        with open(MODELS_PATH, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=1, sort_keys=True)
            handle.write("\n")

        print(f"Written to {MODELS_PATH}.")

    return 0


if (__name__ == "__main__"):
    sys.exit(main())
