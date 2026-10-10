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
Measures the upstream pieces the design guide names and the fork never used, before deciding whether to reuse them.

The guide names `morpheus_dfp`'s rolling window, training and inference stages, the identity-provider source stages,
`MLFlowDriftStage` and `TimeSeriesStage` as the building blocks of the per-layer pattern, and the fork built its
own paths instead without writing down why. This runs each of them over the fork's own corpora, or over the
upstream sample data where the question is what the source emits, and records what happened in an artifact. The
decisions in the guide's Part 0 are quoted from that artifact and tested against it.

**What runs where.** The DFP stages import cuDF when their module is imported and declare the GPU as their only
execution mode, and the drift and time-series stages compute with CuPy. On a machine with a card all of it runs as
shipped. On a machine without one, the runner says so in the artifact and substitutes pandas for cuDF and NumPy for
CuPy, calling the stages' own methods directly rather than through a pipeline that would refuse them; every
substitution is listed under `environment.shims`, so a number measured through one is never mistaken for one
measured without it. The mlflow shims are different in kind: they are repairs to defects the run found, applied
only so that the measurement can continue past them, and each is recorded with the defect it works around.

    ./examples/upstream_reuse/evaluate.py OUTPUT.json --lfs-data DIR

`--lfs-data` is a directory holding `models/datasets/...` fetched from Git LFS; without it the source-stage section
records the sample data as unavailable rather than guessing at its columns.
"""

import argparse
import contextlib
import datetime
import io
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

TRAINING_KWARGS = {"device": "cpu", "encoder_layers": [8, 4], "decoder_layers": [4, 8]}
"""DFPTraining's model arguments, overridden only where the upstream defaults cannot apply here: the CPU device, and
the runner's small layers in place of `[512, 500]`, which on forty rows would describe the network rather than the
data. Everything else -- the optimizer, swap noise, learning rate and its decay, the scaler -- is upstream's."""

TRAINING_EPOCHS = 5

TRAINING_WINDOW = {"min_history": 20, "min_increment": 20, "max_history": "60d"}
"""Upstream trains on 300 rows and retrains every 300 more; a principal in this corpus has 40 over the fortnight, so
both are scaled to 20. The window's length is upstream's."""

INFERENCE_WINDOW = {"min_history": 1, "min_increment": 0, "max_history": "1d"}
"""Upstream's inference window, unchanged: every batch is scored with up to a day of the principal's history."""

# --- Environment ------------------------------------------------------------------------------------------------


def _probe(statement: str) -> dict:
    """Run one import in a fresh interpreter, so a failure cannot leave half-initialized native state here."""
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONPATH":
                os.pathsep.join([os.path.join(REPO_ROOT, "python", "morpheus_dfp"), os.environ.get("PYTHONPATH", "")])
        })
    lines = [line for line in completed.stderr.strip().splitlines() if line.strip()]

    return {"ok": completed.returncode == 0, "error": lines[-1][:240] if (completed.returncode and lines) else None}


def install_shims(shims: list) -> None:
    """Stand pandas in for cuDF and NumPy for CuPy where the real ones cannot load, and say so."""
    if (not _probe("import cudf")["ok"]):
        import pandas as pd  # pylint: disable=import-outside-toplevel

        pd.DataFrame.to_pandas = lambda self: self
        pd.DataFrame.from_pandas = staticmethod(lambda frame: frame)
        pd.Series.to_pandas = lambda self: self
        pd.Index.to_pandas = lambda self: self
        cudf = types.ModuleType("cudf")
        cudf.DataFrame = pd.DataFrame
        sys.modules["cudf"] = cudf
        shims.append("cudf -> pandas: no CUDA driver; the DFP stages' own methods are called off-pipeline")

    if (not _probe("import cupy; cupy.zeros(1)")["ok"]):
        import numpy  # pylint: disable=import-outside-toplevel

        class HostArray(numpy.ndarray):
            """A NumPy array answering CuPy's `.get()`, which copies a device array to the host."""

            def get(self):
                return numpy.asarray(self)

        cupy = types.ModuleType("cupy")
        cupy.__getattr__ = lambda name: getattr(numpy, name)
        # CuPy indexes a one-dimensional `choices` array; NumPy treats each element as its own choice array.
        cupy.choose = lambda a, choices: numpy.asarray(choices)[numpy.asarray(a)].view(HostArray)
        sys.modules["cupy"] = cupy
        shims.append("cupy -> numpy: no CUDA device; the arithmetic is the stages' own, with `choose` and `.get()` "
                     "given CuPy's semantics")

    import morpheus.common  # pylint: disable=import-outside-toplevel

    morpheus.common.load_cudf_helper = lambda: None

    from morpheus.config import CppConfig  # pylint: disable=import-outside-toplevel

    CppConfig.set_should_use_cpp(False)


@contextlib.contextmanager
def quiet():
    """The upstream stages log every exception they swallow; the counts are recorded, the tracebacks are not."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    sink = io.StringIO()

    try:
        with contextlib.redirect_stderr(sink):
            yield
    finally:
        logging.disable(previous)


# --- The corpus -------------------------------------------------------------------------------------------------


def session_features():
    """The session corpus's authentications, as the ten features the fork's models are trained on."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "morpheus", "determinism"))

    import pandas as pd  # pylint: disable=import-outside-toplevel

    import session_pipeline as sp  # pylint: disable=import-outside-toplevel

    result = sp.run_pipeline(sp.build_pipeline_config(), sp.build_corpus())
    auth = result[result["telemetry_class"] == "tc5_auth"]
    auth = auth[["event_uid", "event_time", "user_principal"] + sp.SCORED_FEATURES].dropna().copy()
    # The fork's stages emit nullable integers; dfencoder's scaler refuses them, and upstream's preprocessing stage
    # would have cast them. Casting here is that cast and nothing more.
    auth[sp.SCORED_FEATURES] = auth[sp.SCORED_FEATURES].astype("float64")
    # The rolling window compares against a naive numpy datetime, so the column is naive.
    auth["timestamp"] = pd.to_datetime(auth["event_time"].astype("int64"), unit="ns")
    auth = auth.sort_values(["timestamp", "user_principal"], kind="mergesort").reset_index(drop=True)
    fortnight = auth[auth["event_time"].astype("int64") < sp.SCORES_FROM_NS].reset_index(drop=True)
    week = auth[auth["event_time"].astype("int64") >= sp.SCORES_FROM_NS].reset_index(drop=True)

    return (sp, fortnight, week)


# --- The DFP model path -----------------------------------------------------------------------------------------


class MlflowRepairs:
    """
    Repairs to the upstream writer and reader, each applied only when asked for and each recorded.

    `serialization`: the writer calls `mlflow.pytorch.log_model` without an input example, which mlflow 3 refuses
    for its default `pt2` format; the writer logs the exception and registers nothing. The repair asks for `pickle`.

    `source`: under mlflow 3 the writer registers the model version's source as a run-artifact path where mlflow 3
    does not put the model, so every load fails. The repair registers the logged model's own URI.

    `naming`: the writer names a user's model with `.` replaced by `_dot_` and the inference side looks it up
    without that replacement. The repair makes the reader apply it too.
    """

    def __init__(self, enabled: set):
        self.enabled = set(enabled)
        self._saved = []

    def __enter__(self):
        import mlflow  # pylint: disable=import-outside-toplevel
        import mlflow.pytorch  # pylint: disable=import-outside-toplevel

        from morpheus_dfp.utils import model_cache  # pylint: disable=import-outside-toplevel

        last = {}
        original_log = mlflow.pytorch.log_model
        original_uri = mlflow.get_artifact_uri
        original_name = model_cache.user_to_model_name

        def log_model(*args, **kwargs):
            if ("serialization" in self.enabled):
                kwargs.setdefault("serialization_format", "pickle")

            info = original_log(*args, **kwargs)
            last["uri"] = info.model_uri

            return info

        def artifact_uri(path=None):
            if ("source" in self.enabled and path is not None and "uri" in last):
                return last["uri"]

            return original_uri(path)

        def user_to_model_name(user_id: str, model_name_formatter: str):
            if ("naming" in self.enabled):
                user_id = user_id.replace(".", "_dot_")

            return original_name(user_id=user_id, model_name_formatter=model_name_formatter)

        for (owner, name, value) in ((mlflow.pytorch, "log_model",
                                      log_model), (mlflow, "get_artifact_uri", artifact_uri),
                                     (model_cache, "user_to_model_name", user_to_model_name)):
            self._saved.append((owner, name, getattr(owner, name)))
            setattr(owner, name, value)

        return self

    def __exit__(self, *exc):
        for (owner, name, value) in reversed(self._saved):
            setattr(owner, name, value)


class DfpPath:
    """The upstream training and inference pipelines' stages, driven message by message over the session corpus."""

    def __init__(self, sp, workdir: str):
        from morpheus.config import Config  # pylint: disable=import-outside-toplevel
        from morpheus.config import ConfigAutoEncoder  # pylint: disable=import-outside-toplevel
        from morpheus.config import ExecutionMode  # pylint: disable=import-outside-toplevel

        self.sp = sp
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.join(workdir, 'mlflow.db')}"
        # mlflow writes artifacts relative to the working directory unless told otherwise.
        os.chdir(workdir)

        self.config = Config()
        self.config.execution_mode = ExecutionMode.CPU
        self.config.ae = ConfigAutoEncoder()
        self.config.ae.feature_columns = list(sp.SCORED_FEATURES)
        self.config.ae.userid_column_name = "user_principal"
        self.config.ae.timestamp_column_name = "timestamp"

    def _split(self, frame, generic: bool):
        from morpheus_dfp.stages.dfp_split_users_stage import DFPSplitUsersStage  # pylint: disable=import-outside-toplevel

        return DFPSplitUsersStage(self.config, include_generic=generic,
                                  include_individual=True).extract_users(frame.drop(columns=["event_time"]))

    def window(self, name: str, settings: dict):
        from morpheus_dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage  # pylint: disable=import-outside-toplevel

        return DFPRollingWindowStage(self.config, cache_dir=os.path.join(self.workdir, name), **settings)

    def trainer(self):
        from morpheus_dfp.stages.dfp_training import DFPTraining  # pylint: disable=import-outside-toplevel

        return DFPTraining(self.config, model_kwargs=dict(TRAINING_KWARGS), epochs=TRAINING_EPOCHS)

    def writer(self):
        from morpheus_dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage  # pylint: disable=import-outside-toplevel

        return DFPMLFlowModelWriterStage(self.config)

    def inference(self):
        from morpheus_dfp.stages.dfp_inference_stage import DFPInferenceStage  # pylint: disable=import-outside-toplevel

        return DFPInferenceStage(self.config)

    def train(self, fortnight) -> int:
        """Upstream's training pipeline over the fortnight: split, window, train, write. Returns windows trained."""
        window = self.window("train", TRAINING_WINDOW)
        trainer = self.trainer()
        writer = self.writer()
        trained = 0

        for message in self._split(fortnight, generic=True):
            emitted = window.on_data(message)

            if (emitted is not None):
                writer._controller.on_data(trainer.on_data(emitted))  # pylint: disable=protected-access
                trained += 1

        return trained

    def registered(self) -> dict:
        """Registered model name to its latest version, as the registry holds it."""
        from mlflow.tracking import MlflowClient  # pylint: disable=import-outside-toplevel

        client = MlflowClient()
        found = {}

        for model in client.search_registered_models():
            versions = [int(version.version) for version in client.search_model_versions(f"name='{model.name}'")]
            found[model.name] = max(versions) if versions else 0

        return found

    def score(self, batches: list, stage=None, window=None, postprocess: bool = False):
        """
        Upstream's inference pipeline over a sequence of batches: split, window, infer, and optionally postprocess.

        Returns the scored frames, one per emitted window, each tagged with the batch it came from.
        """
        import pandas as pd  # pylint: disable=import-outside-toplevel

        from morpheus_dfp.stages.dfp_postprocessing_stage import DFPPostprocessingStage  # pylint: disable=import-outside-toplevel

        stage = stage if stage is not None else self.inference()
        window = window if window is not None else self.window(f"infer-{id(stage)}-{len(batches)}", INFERENCE_WINDOW)
        post = DFPPostprocessingStage(self.config) if postprocess else None
        frames = []

        for (index, batch) in enumerate(batches):
            for message in self._split(batch, generic=False):
                emitted = window.on_data(message)

                if (emitted is None):
                    continue

                scored = stage.on_data(emitted)

                if (scored is None):
                    continue

                if (post is not None):
                    scored = post.on_data(scored)

                frame = pd.DataFrame(scored.payload().copy_dataframe())
                frame["_batch"] = index
                frames.append(frame)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _digest(model, features: list) -> str:
    from morpheus.utils.dfencoder_scorer import export_model  # pylint: disable=import-outside-toplevel
    from morpheus.utils.dfencoder_scorer import model_digest  # pylint: disable=import-outside-toplevel

    return model_digest(export_model(model, features))


def measure_dfp(sp, fortnight, week, workroot: str) -> dict:
    """Every measurement of the DFP model path. Each sub-dictionary is one question and what was found."""
    found = {}
    features = list(sp.SCORED_FEATURES)
    principals = sorted(set(week["user_principal"]))

    # 1. Can it run where the fork's CI runs? A pipeline, not a method call: the stage's declared modes decide.
    found["imports"] = {
        name: _probe(f"import morpheus_dfp.stages.{name}")
        for name in ("dfp_split_users_stage", "dfp_rolling_window_stage", "dfp_training", "dfp_inference_stage")
    }
    found["cpu_pipeline"] = _cpu_pipeline(fortnight)

    # 2. The writer as shipped, under the mlflow the fork's CPU lock resolves.
    with quiet(), MlflowRepairs(set()):
        shipped = DfpPath(sp, os.path.join(workroot, "shipped"))
        windows = shipped.train(fortnight)
        found["writer_as_shipped"] = {"windows_trained": windows, "models_registered": len(shipped.registered())}

    with quiet(), MlflowRepairs({"serialization"}):
        partial = DfpPath(sp, os.path.join(workroot, "serialization"))
        partial.train(fortnight)
        scored = partial.score([week])
        found["with_serialization_repair"] = {
            "models_registered": len(partial.registered()),
            "rows_scored": int(len(scored)),
        }

    # 3. The naming mismatch: source repaired, naming not.
    with quiet(), MlflowRepairs({"serialization", "source"}):
        named = DfpPath(sp, os.path.join(workroot, "naming"))
        named.train(fortnight)
        scored = named.score([week])
        from morpheus_dfp.utils.model_cache import user_to_model_name  # pylint: disable=import-outside-toplevel

        found["naming"] = {
            "registered":
                sorted(named.registered()),
            "reader_looks_up":
                user_to_model_name("alice@example.com", "dfp-{user_id}"),
            "versions_scored_with":
                dict(sorted(scored["model_version"].value_counts().items())),
            "principals_scored_by_own_model":
                sum(1 for principal in principals if any(
                    version.startswith(f"dfp-{principal.replace('.', '_dot_')}:")
                    for version in scored[scored["user_principal"] == principal]["model_version"])),
            "principals":
                len(principals),
            "principals_with_a_registered_model":
                sum(1 for principal in principals if f"dfp-{principal.replace('.', '_dot_')}" in named.registered()),
            "output_columns_naming_a_fallback":
                sorted(column for column in scored.columns if "fallback" in column),
        }

    # 4. Everything repaired: the remaining properties belong to the design rather than to a version skew.
    with quiet(), MlflowRepairs({"serialization", "source", "naming"}):
        path = DfpPath(sp, os.path.join(workroot, "repaired"))
        path.train(fortnight)
        found["registered_when_repaired"] = path.registered()

        found["training_double_run"] = _training_double_run(path, fortnight, features)
        found["latest_version_and_cache"] = _latest_and_cache(path, fortnight, week)
        found["batch_split"] = _batch_split(path, week)
        found["late_row"] = _late_row(path, week)
        found["event_time"] = _event_time(path, week)

    return found


def _cpu_pipeline(fortnight) -> dict:
    """Build a CPU-mode pipeline with the split stage in it, as a deployment without a card would."""
    from morpheus.config import Config  # pylint: disable=import-outside-toplevel
    from morpheus.config import ConfigAutoEncoder  # pylint: disable=import-outside-toplevel
    from morpheus.config import ExecutionMode  # pylint: disable=import-outside-toplevel
    from morpheus.pipeline import LinearPipeline  # pylint: disable=import-outside-toplevel
    from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage  # pylint: disable=import-outside-toplevel
    from morpheus_dfp.stages.dfp_split_users_stage import DFPSplitUsersStage  # pylint: disable=import-outside-toplevel

    config = Config()
    config.execution_mode = ExecutionMode.CPU
    config.ae = ConfigAutoEncoder()
    config.ae.userid_column_name = "user_principal"
    config.ae.timestamp_column_name = "timestamp"
    stage = DFPSplitUsersStage(config, include_generic=False, include_individual=True)

    try:
        pipe = LinearPipeline(config)
        pipe.set_source(InMemorySourceStage(config, dataframes=[fortnight.head(4)]))
        pipe.add_stage(stage)
        pipe.build()
    except Exception as error:  # pylint: disable=broad-exception-caught
        return {
            "refused": True,
            "error": f"{type(error).__name__}: {str(error)[:200]}",
            "declared_modes": [mode.value for mode in stage.supported_execution_modes()]
        }

    return {"refused": False, "declared_modes": [mode.value for mode in stage.supported_execution_modes()]}


def _first_window(path: DfpPath, fortnight, principal: str):
    window = path.window(f"once-{principal}-{len(os.listdir(path.workdir))}", TRAINING_WINDOW)

    for message in path._split(fortnight[fortnight["user_principal"] == principal], generic=False):  # pylint: disable=protected-access
        emitted = window.on_data(message)

        if (emitted is not None):
            return emitted

    raise RuntimeError(f"no training window for {principal}")


def _training_double_run(path: DfpPath, fortnight, features: list) -> dict:
    """Train the same principal's window twice, as shipped and then after seeding, and compare the weights."""
    from morpheus.utils.seed import manual_seed  # pylint: disable=import-outside-toplevel

    principal = "alice@example.com"
    window = _first_window(path, fortnight, principal)
    trainer = path.trainer()
    shipped = [_digest(trainer.on_data(window).get_metadata("model"), features) for _ in range(2)]
    seeded = []

    for _ in range(2):
        manual_seed(42, cpu_only=True)
        seeded.append(_digest(trainer.on_data(window).get_metadata("model"), features))

    return {
        "principal": principal,
        "rows": int(window.payload().count),
        "unseeded_digests": shipped,
        "unseeded_identical": shipped[0] == shipped[1],
        "seeded_digests": seeded,
        "seeded_identical": seeded[0] == seeded[1],
    }


def _latest_and_cache(path: DfpPath, fortnight, week) -> dict:
    """Score one batch, retrain the principal, and score the same batch again: once warm, once from cold."""
    principal = "alice@example.com"
    batch = week[week["user_principal"] == principal].head(4)
    warm = path.inference()
    before = path.score([batch], stage=warm)

    window = _first_window(path, fortnight, principal)
    path.writer()._controller.on_data(path.trainer().on_data(window))  # pylint: disable=protected-access

    still = path.score([batch], stage=warm)
    cold = path.score([batch])

    def summary(frame):
        return {
            "model_version": sorted(set(frame["model_version"])),
            "mean_abs_z": [round(float(value), 4) for value in frame["mean_abs_z"]],
        }

    return {
        "principal": principal,
        "rows": int(len(batch)),
        "cache_timeout_sec": warm._model_manager.cache_timeout_sec,  # pylint: disable=protected-access
        "before_retraining": summary(before),
        "after_retraining_same_stage": summary(still),
        "after_retraining_new_stage": summary(cold),
        "manifest_or_pin_parameter": False,
    }


def _batch_split(path: DfpPath, week) -> dict:
    """The scored week through the inference pipeline whole, a day at a time, and an event at a time."""
    days = week["timestamp"].dt.floor("D")
    batchings = {
        "whole": [week],
        "by_day": [frame for (_, frame) in week.groupby(days, sort=True)],
        "by_event": [week.iloc[[position]] for position in range(len(week))],
    }
    found = {}
    last = {}

    for (name, batches) in batchings.items():
        scored = path.score(batches)
        emitted = scored.groupby("event_uid").size() if len(scored) else []
        found[name] = {
            "batches": len(batches),
            "rows_emitted": int(len(scored)),
            "events_scored": int(len(emitted)),
            "most_emissions_of_one_event": int(max(emitted)) if len(emitted) else 0,
        }
        last[name] = scored.drop_duplicates("event_uid", keep="last").set_index("event_uid")["mean_abs_z"]

    common = sorted(set(last["whole"].index) & set(last["by_day"].index) & set(last["by_event"].index))
    found["events_in_scored_week"] = int(len(week))
    found["events_scored_by_every_batching"] = len(common)
    differences = [
        max(abs(float(last["whole"][key]) - float(last[name][key])) for name in ("by_day", "by_event"))
        for key in common
    ]
    found["events_whose_last_score_differs_across_batchings"] = sum(1 for value in differences if value >= 5e-5)
    found["largest_last_score_difference"] = round(max(differences), 4) if differences else 0.0

    return found


def _late_row(path: DfpPath, week) -> dict:
    """One event delivered after a later one by the same principal, an event at a time."""
    principal = "erin@example.com"
    rows = week[week["user_principal"] == principal].reset_index(drop=True)
    order = list(range(len(rows)))
    (order[3], order[4]) = (order[4], order[3])
    late = rows.iloc[3]["event_uid"]

    try:
        scored = path.score([rows.iloc[[position]] for position in order])
    except Exception as error:  # pylint: disable=broad-exception-caught
        # In a pipeline an exception out of a stage's map function ends the pipeline.
        return {
            "principal": principal,
            "events": int(len(rows)),
            "raised": f"{type(error).__name__}: {str(error)[:160]}",
        }

    return {
        "principal": principal,
        "events": int(len(rows)),
        "raised": None,
        "late_event_scored": bool(late in set(scored["event_uid"])) if len(scored) else False,
        "events_scored": int(scored["event_uid"].nunique()) if len(scored) else 0,
    }


def _event_time(path: DfpPath, week) -> dict:
    """What DFPPostprocessingStage leaves in `event_time`, twice over the same rows."""
    import time  # pylint: disable=import-outside-toplevel

    batch = week[week["user_principal"] == "dave@example.com"].head(2)
    first = path.score([batch], postprocess=True)
    time.sleep(1.1)
    second = path.score([batch], postprocess=True)

    return {
        "input_event_time": [str(value) for value in batch["timestamp"]],
        "first_run_event_time": sorted(set(first["event_time"])),
        "second_run_event_time": sorted(set(second["event_time"])),
        "identical_across_runs": sorted(set(first["event_time"])) == sorted(set(second["event_time"])),
    }


# --- The identity-provider source stages ------------------------------------------------------------------------

FORK_TC5_INPUTS = [
    "event_time",
    "user_principal",
    "source_country",
    "source_region",
    "source_city",
    "source_latitude",
    "source_longitude",
    "source_asn",
    "source_ip",
    "app",
    "device_id",
    "auth_result",
    "mfa_used",
    "mfa_result",
    "token_type",
]
"""The columns the fork's layer 5 stages read from an authentication, as `session_pipeline` gives them."""

ENVELOPE = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
"""What `LineageStampStage` needs on every row to give it an identity that survives a replay."""

SOURCE_MAPPINGS = {
    "azure": {
        "event_time": "createdDateTime",
        "user_principal": "userPrincipalName",
        "source_country": "locationcountryOrRegion",
        "source_region": "locationstate",
        "source_city": "locationcity",
        "source_latitude": "locationgeoCoordinateslatitude",
        "source_longitude": "locationgeoCoordinateslongitude",
        "source_asn": "autonomousSystemNumber",
        "source_ip": "ipAddress",
        "app": "appDisplayName",
        "device_id": "deviceDetaildeviceId",
        "auth_result": "statuserrorCode",
        "mfa_used": "authenticationRequirement",
        "mfa_result": "authenticationDetails",
        "token_type": "incomingTokenType",
    },
    "cloudtrail": {
        "event_time": "eventTime",
        "user_principal": "userIdentitysessionContextsessionIssueruserName",
        "source_country": None,
        "source_region": None,
        "source_city": None,
        "source_latitude": None,
        "source_longitude": None,
        "source_asn": None,
        "source_ip": "sourceIPAddress",
        "app": "eventSource",
        "device_id": None,
        "auth_result": "errorCode",
        "mfa_used": "additionalEventDataMFAUsed",
        "mfa_result": None,
        "token_type": None,
    },
    "duo": {
        "event_time": "isotimestamp",
        "user_principal": "username",
        "source_country": "accessdevicelocationcountry",
        "source_region": "accessdevicelocationstate",
        "source_city": "accessdevicelocationcity",
        "source_latitude": None,
        "source_longitude": None,
        "source_asn": None,
        "source_ip": "accessdeviceip",
        "app": "applicationname",
        "device_id": "authdevicename",
        "auth_result": "result",
        "mfa_used": "factor",
        "mfa_result": "result",
        "token_type": None,
    },
}
"""The source column a deployment would map each fork input from, after the stage's own column renaming.

A candidate, not a claim: for Azure and CloudTrail the measurement is whether the sample the stage reads carries
it, and how often it is filled. For Duo the tree holds no sample, so the candidates are the fields the upstream
Duo pipeline's own input schema and `DuoSourceStage.derive_features` read, and are recorded as declared rather
than measured. `None` is a field the source does not carry at all.
"""

SAMPLES = {
    "azure": ["models/datasets/training-data/azure/azure-ad-logs-sample-training-data.json"],
    "cloudtrail": [
        "models/datasets/training-data/cloudtrail/hammah-user123-training-part2.json",
        "models/datasets/training-data/cloudtrail/hammah-role-g-training-part1.json",
    ],
}


def _falls(series) -> bool:
    """Whether a per-row counter ever drops below a value it already reached: a reset."""
    values = [float(value) for value in series if not math.isnan(float(value))]

    return any(value < max(values[:index]) for (index, value) in enumerate(values) if index)


def measure_sources(lfs_dir: str) -> dict:
    """Read each sample through its stage's own methods and map the result onto the fork's TC-5 inputs."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    from morpheus.stages.input.azure_source_stage import AzureSourceStage  # pylint: disable=import-outside-toplevel
    from morpheus.stages.input.cloud_trail_source_stage import CloudTrailSourceStage  # pylint: disable=import-outside-toplevel
    from morpheus.common import FileTypes  # pylint: disable=import-outside-toplevel
    from morpheus.stages.input.duo_source_stage import DuoSourceStage  # pylint: disable=import-outside-toplevel

    found = {}
    stages = {"azure": AzureSourceStage, "cloudtrail": CloudTrailSourceStage, "duo": DuoSourceStage}
    user_columns = {"azure": "userPrincipalName", "cloudtrail": "userIdentitysessionContextsessionIssueruserName"}

    for (name, stage) in stages.items():
        entry = {
            "execution_modes":
                "GpuAndCpuMixin" if any(base.__name__ == "GpuAndCpuMixin" for base in stage.__mro__) else "GPU only",
        }
        files = [os.path.join(lfs_dir, path) for path in SAMPLES.get(name, [])] if lfs_dir else []
        available = bool(files) and all(os.path.exists(path) for path in files)

        if (not available):
            entry["sample"] = "none in the tree" if name not in SAMPLES else "not fetched from Git LFS"
            entry["mapping"] = {
                field: {
                    "source": column, "measured": False
                }
                for (field, column) in SOURCE_MAPPINGS[name].items()
            }
            entry["envelope_present"] = []
            found[name] = entry
            continue

        if (name == "azure"):
            # What `AzureSourceStage.files_to_dfs_per_user` reads, before the per-user split.
            raw = pd.concat([pd.json_normalize(pd.read_json(path, orient="records")["properties"]) for path in files],
                            ignore_index=True)
            renamed = stage.change_columns(raw.copy())
            entry["renaming_as_shipped"] = {
                "columns_renamed": int(sum(1 for (old, new) in zip(raw.columns, renamed.columns) if old != new)),
                "columns_with_separators": int(sum(1 for column in raw.columns if "." in column)),
            }

            try:
                stage.derive_features(raw.copy(), None)
                entry["derive_as_shipped"] = None
            except Exception as error:  # pylint: disable=broad-exception-caught
                entry["derive_as_shipped"] = f"{type(error).__name__}: {str(error)[:160]}"

            # The repair: the pattern the stage meant, applied as a pattern. pandas 2 reads `str.replace`'s pattern
            # as a literal unless told otherwise, so the shipped renaming changes nothing.
            frame = raw.copy()
            frame.columns = frame.columns.str.replace("[_,.,{,},:]", "", regex=True).str.strip()
            stage.change_columns = staticmethod(lambda df: df)
        else:
            frame = pd.concat([stage.cleanup_df(stage.read_file(path, FileTypes.Auto), None) for path in files],
                              ignore_index=True)
            frame.columns = frame.columns.str.replace(".", "", regex=False)

        derived = []

        try:
            for (_, user_frame) in frame.groupby(user_columns[name], sort=True):
                derived.append(stage.derive_features(user_frame.copy(), None))
        finally:
            if (name == "azure"):
                del stage.change_columns

        derived = pd.concat(derived, ignore_index=True)
        mapping = {}

        for (field, column) in SOURCE_MAPPINGS[name].items():
            present = column is not None and column in frame.columns
            mapping[field] = {
                "source": column,
                "measured": True,
                "present": present,
                "filled": round(float(frame[column].notna().mean()), 3) if present else 0.0,
            }

        counters = {}

        for counter in ("locincrement", "appincrement", "logcount"):
            if (counter in derived.columns):
                key = user_columns[name] if user_columns[name] in derived.columns else None
                groups = derived.groupby(key, sort=True) if key else [("all", derived)]
                counters[counter] = {
                    "users": len(groups) if key else 1,
                    "users_whose_count_resets": sum(1 for (_, group) in groups if _falls(group[counter])),
                }

        entry.update({
            "sample": [os.path.relpath(path, lfs_dir) for path in files],
            "rows": int(len(frame)),
            "users": int(frame[user_columns[name]].nunique()),
            "columns_after_stage": int(len(frame.columns)),
            "columns_derived": sorted(set(derived.columns) - set(frame.columns)),
            "mapping": mapping,
            "fork_inputs_present": sum(1 for value in mapping.values() if value["present"]),
            "fork_inputs": len(FORK_TC5_INPUTS),
            "envelope_present": sorted(set(ENVELOPE) & set(frame.columns)),
            "counters": counters,
        })
        found[name] = entry

    return found


# --- MLFlowDriftStage and TimeSeriesStage -------------------------------------------------------------------------


def _modes(stage_class) -> str:
    return "GpuAndCpuMixin" if any(base.__name__ == "GpuAndCpuMixin" for base in stage_class.__mro__) else "GPU only"


def measure_drift(sp, workdir: str) -> dict:
    """What MLFlowDriftStage reads, what it writes, and whether the batching decides it."""
    import numpy as np  # pylint: disable=import-outside-toplevel
    from mlflow.tracking import MlflowClient  # pylint: disable=import-outside-toplevel

    from morpheus.config import Config  # pylint: disable=import-outside-toplevel
    from morpheus.config import ExecutionMode  # pylint: disable=import-outside-toplevel
    from morpheus.messages import ControlMessage  # pylint: disable=import-outside-toplevel
    from morpheus.messages import MessageMeta  # pylint: disable=import-outside-toplevel
    from morpheus.messages.memory.tensor_memory import TensorMemory  # pylint: disable=import-outside-toplevel
    from morpheus.stages.postprocess.ml_flow_drift_stage import MLFlowDriftStage  # pylint: disable=import-outside-toplevel

    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    uri = f"sqlite:///{os.path.join(workdir, 'drift.db')}"
    scored = sp.run_pipeline(sp.build_pipeline_config(), sp.build_corpus())
    scored = scored[(scored["telemetry_class"] == "tc5_auth") & scored["mean_abs_z"].notna()].reset_index(drop=True)
    found = {"execution_modes": _modes(MLFlowDriftStage)}

    def stage(batch_size: int, name: str):
        config = Config()
        config.execution_mode = ExecutionMode.CPU
        config.pipeline_batch_size = batch_size
        config.class_labels = ["anomaly"]

        return MLFlowDriftStage(config, tracking_uri=uri, experiment_name=name, force_new_run=True)

    # The fork's scored rows as TC5ScoreStage leaves them: columns, no tensor.
    message = ControlMessage()
    message.payload(MessageMeta(scored.copy()))

    try:
        stage(256, "fork-rows")._calc_drift(message)  # pylint: disable=protected-access
        found["on_the_fork_rows"] = None
    except Exception as error:  # pylint: disable=broad-exception-caught
        found["on_the_fork_rows"] = f"{type(error).__name__}: {str(error)[:160]}"
    finally:
        import mlflow  # pylint: disable=import-outside-toplevel

        mlflow.end_run()

    # The same scores given to it as the classifier probability it expects, at two pipeline batch sizes.
    scores = scored["mean_abs_z"].astype(float).to_numpy()
    probs = (scores / (1.0 + scores)).reshape(-1, 1)
    histories = {}

    for batch_size in (32, 1024):
        drift = stage(batch_size, f"batch-{batch_size}")
        message = ControlMessage()
        message.payload(MessageMeta(scored.copy()))
        message.tensors(TensorMemory(count=len(probs), tensors={"probs": np.asarray(probs)}))
        drift._calc_drift(message)  # pylint: disable=protected-access
        run = mlflow.active_run().info.run_id
        mlflow.end_run()
        history = MlflowClient(tracking_uri=uri).get_metric_history(run, "total")
        histories[batch_size] = [round(point.value, 4) for point in history]

    found["rows"] = int(len(probs))
    found["points_logged"] = {str(size): len(values) for (size, values) in histories.items()}
    found["first_point"] = {str(size): values[0] for (size, values) in histories.items()}
    found["reports_model_version"] = False
    found["writes_to_the_row"] = False

    return found


def measure_timeseries(workdir: str) -> dict:
    """TimeSeriesStage over the failing optic's hour: as shipped, scaled to the hour, and with its values flattened."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "morpheus", "determinism"))

    import telemetry_pipeline as tp  # pylint: disable=import-outside-toplevel

    from morpheus.messages import ControlMessage  # pylint: disable=import-outside-toplevel
    from morpheus.messages import MessageMeta  # pylint: disable=import-outside-toplevel
    from morpheus.stages.postprocess import timeseries_stage  # pylint: disable=import-outside-toplevel

    del workdir
    layer_1 = tp.build_corpus()["tc1"]
    optic = layer_1[(layer_1["device_id"] == tp.SWITCH) & (layer_1["port_id"] == tp.MAINTENANCE_PORT)].copy()
    optic["timestamp"] = pd.to_datetime(optic["event_time"].astype("int64"), unit="ns")
    optic = optic.sort_values("timestamp").reset_index(drop=True)

    def run(frame, resolution: str, min_window: str, hot_start: bool) -> dict:
        series = timeseries_stage._UserTimeSeries(  # pylint: disable=protected-access
            user_id="optic",
            timestamp_col="timestamp",
            resolution=resolution,
            min_window=min_window,
            hot_start=hot_start,
            cold_end=False,
            filter_percent=90.0,
            zscore_threshold=8.0)
        released = []
        held_until = []
        calculations = []
        calculate = series._calc_outliers  # pylint: disable=protected-access

        def counted(action):
            # The stage returns the anomalous bins only when they fall in the message being released.
            found = calculate(action)
            calculations.append(found is not None)
            return found

        series._calc_outliers = counted  # pylint: disable=protected-access

        for position in range(len(frame)):
            message = ControlMessage()
            message.payload(MessageMeta(frame.iloc[[position]][["timestamp", "optical_rx_dbm"]]))
            out = series._calc_timeseries(message, False)  # pylint: disable=protected-access
            released.extend(out)
            held_until.extend([position] * len(out))

        at_end = series._calc_timeseries(None, True)  # pylint: disable=protected-access
        released.extend(at_end)
        flags = [bool(message.payload().get_data("ts_anomaly").iloc[0]) for message in released]
        waits = [arrived - index for (index, arrived) in enumerate(held_until)]

        bins = 2 * series._half_window_bins + 1  # pylint: disable=protected-access

        return {
            "polls": int(len(frame)),
            "window_bins": bins,
            "largest_possible_zscore": round(math.sqrt(bins - 1), 3),
            "calculations": len(calculations),
            "anomaly_found_in_the_released_message": int(sum(calculations)),
            "released_before_the_end": len(held_until),
            "released_at_the_end": len(at_end),
            "flagged": int(sum(flags)),
            "longest_hold_in_polls": max(waits) if waits else None,
            "flags": flags,
            "found": list(calculations),
        }

    shipped = run(optic, "1 h", "12 h", False)
    scaled = run(optic, "1 min", "12 min", True)

    # A burst: thirty extra polls in one minute, the shape a count-based detector exists to see. Then the same
    # timestamps with every receive level set to one constant, and with the receive levels reversed.
    extra = pd.concat([optic.iloc[[40]]] * 30, ignore_index=True)
    burst = pd.concat([optic, extra], ignore_index=True).sort_values("timestamp", kind="mergesort")
    burst = burst.reset_index(drop=True)
    flat = burst.copy()
    flat["optical_rx_dbm"] = -7.0
    reversed_levels = burst.copy()
    reversed_levels["optical_rx_dbm"] = list(reversed(burst["optical_rx_dbm"].tolist()))
    bursts = [run(frame, "1 min", "40 min", True) for frame in (burst, flat, reversed_levels)]

    return {
        "burst": {
            key: value
            for (key, value) in bursts[0].items() if key not in ("flags", "found")
        },
        "burst_detections_identical_whatever_the_values":
            bursts[0]["found"] == bursts[1]["found"] == bursts[2]["found"],
        "execution_modes":
            _modes(timeseries_stage.TimeSeriesStage),
        "series":
            f"{tp.SWITCH} {tp.MAINTENANCE_PORT} optical_rx_dbm, one poll a minute",
        "receive_level_falls_by_db":
            round(float(optic["optical_rx_dbm"].max() - optic["optical_rx_dbm"].min()), 3),
        "as_shipped": {
            key: value
            for (key, value) in shipped.items() if key not in ("flags", "found")
        },
        "scaled_to_the_hour": {
            key: value
            for (key, value) in scaled.items() if key not in ("flags", "found")
        },
        "zscore_threshold":
            8.0,
        "reads": ["the timestamp column"],
    }


# --- The run --------------------------------------------------------------------------------------------------


def _version(module: str) -> str:
    try:
        return __import__(module).__version__
    except Exception:  # pylint: disable=broad-exception-caught
        return "absent"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact")
    parser.add_argument("--lfs-data", default=None, help="directory holding models/datasets fetched from Git LFS")
    arguments = parser.parse_args()
    artifact = os.path.abspath(arguments.artifact)
    lfs_dir = os.path.abspath(arguments.lfs_data) if arguments.lfs_data else None

    sys.path.insert(0, os.path.join(REPO_ROOT, "python", "morpheus_dfp"))
    shims: list = []
    install_shims(shims)

    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        print("The DFP model path trains with Torch, which is not importable here.", file=sys.stderr)
        return 1

    (sp, fortnight, week) = session_features()
    workroot = tempfile.mkdtemp(prefix="upstream-reuse-")
    home = os.getcwd()

    try:
        report = {
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "environment": {
                "cuda_device": torch.cuda.is_available(),
                "torch": torch.__version__,
                "mlflow": _version("mlflow"),
                "pandas": _version("pandas"),
                "shims": shims,
            },
            "corpus": {
                "fortnight_rows": int(len(fortnight)),
                "week_rows": int(len(week)),
                "principals": sorted(set(week["user_principal"])),
            },
            "dfp": measure_dfp(sp, fortnight, week, os.path.join(workroot, "dfp")),
            "sources": measure_sources(lfs_dir),
            "drift": measure_drift(sp, os.path.join(workroot, "drift")),
            "timeseries": measure_timeseries(os.path.join(workroot, "timeseries")),
        }
    finally:
        os.chdir(home)

    with open(artifact, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")

    print(f"Written to {artifact}.")

    return 0


if (__name__ == "__main__"):
    sys.exit(main())
