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
The reuse decisions in the guide's Part 0, held to the run that made them.

`examples/upstream_reuse/evaluate.py` ran the upstream pieces the guide names and wrote what happened to an
artifact; Pass 4 quotes it. Two kinds of check follow. The first holds every number Pass 4 quotes to the committed
artifact, so the prose cannot drift from the run. The second re-checks, against the code in the tree, the reasons
that can be checked without a card, Torch or the Git LFS samples: if upstream repairs one, the check fails and the
decision it supports is due to be revisited rather than left standing on a reason that stopped being true.
"""

import ast
import glob
import inspect
import json
import math
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ARTIFACTS = os.path.join(REPO_ROOT, "examples", "upstream_reuse", "artifacts")
GUIDE = os.path.join(REPO_ROOT,
                     "docs",
                     "source",
                     "developer_guide",
                     "guides",
                     "11_predictive_behavioral_analytics_osi.md")


@pytest.fixture(name="run", scope="module")
def run_fixture() -> dict:
    paths = sorted(glob.glob(os.path.join(ARTIFACTS, "*", "upstream_reuse.json")))

    assert paths, "the reuse evaluation's artifact is not committed"

    with open(paths[-1], encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(name="pass_4", scope="module")
def pass_4_fixture() -> str:
    with open(GUIDE, encoding="utf-8") as handle:
        text = handle.read()

    start = text.index("### Pass 4: What to Reuse, Measured")
    end = text.index("## Part 1:", start)

    return " ".join(text[start:end].split())


# --- The guide quotes the run -----------------------------------------------------------------------------------


def test_pass_4_names_the_artifact_it_quotes(run: dict, pass_4: str):
    date = run["at"][:10]

    assert f"examples/upstream_reuse/artifacts/{date}/upstream_reuse.json" in pass_4


def test_the_dfp_numbers_are_the_run_s(run: dict, pass_4: str):
    dfp = run["dfp"]
    batches = dfp["batch_split"]
    naming = dfp["naming"]
    latest = dfp["latest_version_and_cache"]
    shipped = dfp["writer_as_shipped"]
    [generic] = naming["versions_scored_with"]

    for phrase in (
            f"Over the scored week's {batches['events_in_scored_week']} events",
            f"emits {batches['whole']['rows_emitted']} rows when the week arrives as one batch",
            f"{batches['by_day']['rows_emitted']} a day at a time and {batches['by_event']['rows_emitted']} an "
            f"event at a time, scoring one event up to {batches['by_event']['most_emissions_of_one_event']} times",
            f"two runs over one principal's {dfp['training_double_run']['rows']} rows",
            f"registers {shipped['models_registered']} of {shipped['windows_trained']} trained models",
            f"MLflow {run['environment']['mlflow']}",
            f"names a model `{naming['registered'][0]}` and the reader looks up `{naming['reader_looks_up']}`",
            f"so {naming['principals_scored_by_own_model']} of the {naming['principals_with_a_registered_model']} "
            f"principals with a registered model",
            f"all {naming['versions_scored_with'][generic]} rows are scored by `{generic}`",
            f"its {latest['cache_timeout_sec']}-second cache",
            f"moves from {latest['before_retraining']['mean_abs_z'][0]} to "
            f"{latest['after_retraining_new_stage']['mean_abs_z'][0]}",
    ):
        assert phrase in pass_4, phrase


def test_the_source_numbers_are_the_run_s(run: dict, pass_4: str):
    azure = run["sources"]["azure"]
    cloudtrail = run["sources"]["cloudtrail"]
    locations = azure["counters"]["locincrement"]

    for phrase in (
            f"holds {azure['rows']:,} sign-ins by {azure['users']} users",
            f"carries all {azure['fork_inputs']} fields the fork's layer 5 stages read",
            f"none of the {len(['collector_id', 'schema_version', 'origin_hash', 'collector_seq'])} envelope fields",
            f"changes {azure['renaming_as_shipped']['columns_renamed']} of the "
            f"{azure['renaming_as_shipped']['columns_with_separators']} dotted column names under pandas "
            f"{run['environment']['pandas']}",
            f"resets each day for {locations['users_whose_count_resets']} of the {locations['users']} users",
            f"carries {cloudtrail['fork_inputs_present']} of the {cloudtrail['fork_inputs']}",
    ):
        assert phrase in pass_4, phrase

    assert azure["fork_inputs_present"] == azure["fork_inputs"]
    assert azure["envelope_present"] == []
    assert azure["derive_as_shipped"].startswith("KeyError")
    assert run["sources"]["duo"]["sample"] == "none in the tree"


def test_the_drift_and_time_series_numbers_are_the_run_s(run: dict, pass_4: str):
    drift = run["drift"]
    series = run["timeseries"]

    for phrase in (
            f"logs {drift['points_logged']['32']} points at a pipeline batch size of 32 and "
            f"{drift['points_logged']['1024']} at 1024 over the same {drift['rows']} rows",
            f"{drift['first_point']['32']} against {drift['first_point']['1024']}",
            f"its window is {series['as_shipped']['window_bins']} bins",
            f"more than {series['as_shipped']['largest_possible_zscore']} standard deviations",
            f"under its threshold of {int(series['zscore_threshold'])}",
            f"it ran {series['as_shipped']['calculations']} calculations",
            f"Widened to {series['burst']['window_bins']} bins it found an anomalous bin in "
            f"{series['burst']['anomaly_found_in_the_released_message']} released messages and flagged "
            f"{series['burst']['flagged']}",
    ):
        assert phrase in pass_4, phrase


# --- What the run found ------------------------------------------------------------------------------------------


def test_every_dfp_finding_is_what_pass_4_says_it_is(run: dict):
    dfp = run["dfp"]

    assert dfp["cpu_pipeline"]["refused"] and dfp["cpu_pipeline"]["declared_modes"] == ["GPU"]
    assert dfp["writer_as_shipped"]["models_registered"] == 0 < dfp["writer_as_shipped"]["windows_trained"]
    assert dfp["with_serialization_repair"]["rows_scored"] == 0
    assert dfp["naming"]["principals_scored_by_own_model"] == 0
    assert dfp["naming"]["output_columns_naming_a_fallback"] == []
    assert not dfp["training_double_run"]["unseeded_identical"]
    assert dfp["training_double_run"]["seeded_identical"]

    latest = dfp["latest_version_and_cache"]
    assert latest["before_retraining"]["model_version"] == latest["after_retraining_same_stage"]["model_version"]
    assert latest["after_retraining_new_stage"]["model_version"] != latest["before_retraining"]["model_version"]

    batches = dfp["batch_split"]
    assert batches["whole"]["rows_emitted"] < batches["by_day"]["rows_emitted"] < batches["by_event"]["rows_emitted"]
    assert dfp["late_row"]["raised"].startswith("RuntimeError: Invalid rolling window")
    assert not dfp["event_time"]["identical_across_runs"]


def test_the_drift_and_time_series_findings_are_what_pass_4_says(run: dict):
    drift = run["drift"]
    series = run["timeseries"]

    assert drift["execution_modes"] == series["execution_modes"] == "GPU only"
    assert drift["on_the_fork_rows"].startswith("AttributeError")
    assert drift["points_logged"]["32"] != drift["points_logged"]["1024"]
    assert series["as_shipped"]["largest_possible_zscore"] < series["zscore_threshold"]
    assert series["burst"]["anomaly_found_in_the_released_message"] > series["burst"]["flagged"] == 0
    assert series["burst_detections_identical_whatever_the_values"]


# --- The reasons, re-checked against the tree ----------------------------------------------------------------


def test_the_azure_stage_still_renames_nothing():
    # The reason AzureSourceStage cannot derive its features from its own sample. If upstream passes the pattern as
    # a pattern, this fails and the source-stage decision is due another look.
    import pandas as pd

    from morpheus.stages.input.azure_source_stage import AzureSourceStage
    from morpheus.stages.input.duo_source_stage import DuoSourceStage

    for stage in (AzureSourceStage, DuoSourceStage):
        frame = stage.change_columns(pd.DataFrame(columns=["location.city", "status.errorCode"]))
        assert list(frame.columns) == ["location.city", "status.errorCode"], stage.__name__


def test_the_writer_and_the_reader_still_name_a_model_differently():
    # The writer escapes a dot; the reader's `user_to_model_name` does not. Read from source, because the reader's
    # module imports cuDF and so cannot be imported where this runs.
    from morpheus.controllers.mlflow_model_writer_controller import MLFlowModelWriterController

    writer = MLFlowModelWriterController(model_name_formatter="dfp-{user_id}",
                                         experiment_name_formatter="/dfp-models/{reg_model_name}",
                                         databricks_permissions=None,
                                         conda_env=None,
                                         timeout=1.0,
                                         timestamp_column_name="timestamp")

    assert writer.user_id_to_model("alice@example.com") == "dfp-alice@example_dot_com"

    path = os.path.join(REPO_ROOT, "python", "morpheus_dfp", "morpheus_dfp", "utils", "model_cache.py")

    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    [reader] = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "user_to_model_name"]
    calls = [
        node.func.attr for node in ast.walk(reader)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]

    assert "replace" not in calls


def test_the_time_series_stage_still_cannot_reach_its_own_threshold():
    # Arithmetic over the stage's own defaults: 2 * ceil(min_window / resolution) + 1 bins, and no value among n can
    # sit more than sqrt(n - 1) population standard deviations from their mean.
    import pandas as pd

    from morpheus.stages.postprocess.timeseries_stage import TimeSeriesStage

    defaults = {
        name: parameter.default
        for (name, parameter) in inspect.signature(TimeSeriesStage.__init__).parameters.items()
    }
    bins = 2 * math.ceil(pd.Timedelta(defaults["min_window"]) / pd.Timedelta(defaults["resolution"])) + 1

    assert bins == 25
    assert math.sqrt(bins - 1) < defaults["zscore_threshold"]


def test_the_dfp_stages_still_declare_the_gpu_only():
    # Read from source for the same reason as the reader above: the modules import cuDF.
    stages = os.path.join(REPO_ROOT, "python", "morpheus_dfp", "morpheus_dfp", "stages")

    for name in ("dfp_split_users_stage", "dfp_rolling_window_stage", "dfp_training", "dfp_inference_stage"):
        with open(os.path.join(stages, f"{name}.py"), encoding="utf-8") as handle:
            source = handle.read()

        assert "GpuAndCpuMixin" not in source and "CpuOnlyMixin" not in source, name
        assert "import cudf" in source or name == "dfp_inference_stage", name
