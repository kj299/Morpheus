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
Control 13's six checks against the host segment, and what its learned scores say about the corpus's hosts.

The host segment is the first composition in this fork whose rows are built rather than collected: one row per
machine per hour, reduced from three classes' rows by `EntityFrameStage`. So the checks ask two things the other
harnesses do not -- that the frame is exactly one row per host per working hour, made of exactly that host's rows
in that hour, and that every score on it came from the model its peer group was pinned to -- before they ask what
the scores show.
"""

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.utils.determinism import diff_frames
from morpheus.utils.determinism import frame_digest
from morpheus.utils.determinism import permute_within_contiguous_groups
from morpheus.utils.lineage import window_id_from_timestamp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import host_identity  # noqa: E402
import host_score_pipeline as hs_  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_host_score_expected.csv")
DRIVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_host_score_pipeline.py")

RISING_WINDOWS = 4
RISE_SIGMAS = 1.5
"""R-P-L3-009's trajectory, which is R-P-L5-006's asked of a host: four consecutive rising windows, by more than one and
a half of the entity's own deviations in all, and neither of the principals' ceilings. The validation package reads the
two numbers from the stanza; here they are stated, so a changed rule and a changed corpus fail differently."""


@pytest.fixture(name="corpus", scope="module")
def corpus_fixture() -> dict[str, pd.DataFrame]:
    yield hs_.build_corpus()


# One composed run per execution mode, reused by every check in that mode.
_RESULTS: dict = {}


@pytest.fixture(name="pipeline_config")
def pipeline_config_fixture(execution_mode) -> Config:
    yield hs_.build_pipeline_config(execution_mode)


@pytest.fixture(name="result")
def result_fixture(pipeline_config: Config, corpus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mode = pipeline_config.execution_mode

    if (mode not in _RESULTS):
        _RESULTS[mode] = hs_.run_pipeline(pipeline_config, corpus)

    yield _RESULTS[mode]


def _split(frame: pd.DataFrame, parts: int) -> list:
    size = max(1, len(frame) // parts)

    return [frame.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(frame), size)]


def _permuted(corpus: dict, seed: int) -> dict:
    period_ns = hs_.PERIOD_SECONDS * hs_.NS_PER_SECOND

    return {
        name:
            permute_within_contiguous_groups(frame,
                                             [window_id_from_timestamp(int(t), period_ns) for t in frame["event_time"]],
                                             seed=seed)
        for (name, frame) in corpus.items()
    }


def _day(result: pd.DataFrame) -> pd.Series:
    """The corpus day each row's hour falls on, counted from the first day of the fortnight."""
    return (result["event_time"].astype("int64") - hs_.at(0, 0)) // (hs_.DAY_S * hs_.NS_PER_SECOND)


def _daily(result: pd.DataFrame) -> pd.DataFrame:
    """One row per host per scored day: the day's mean score and the trajectory the drift stage wrote for it."""
    rows = result.assign(day=_day(result))
    rows = rows[rows["day"] >= hs_.TRAINING_DAYS]
    daily = rows.groupby(["host_key", "day"]).agg(mean_abs_z=("mean_abs_z", "mean"),
                                                  rising=("drift_rising_windows", "max"),
                                                  rise_sigmas=("drift_rise_sigmas", "max"))

    return daily.reset_index()


def _climbs(daily: pd.DataFrame) -> pd.DataFrame:
    """The host-days whose trajectory meets R-P-L3-009's two conditions, mature or not."""
    rising = daily["rising"].astype("Float64").fillna(0) >= RISING_WINDOWS
    far = daily["rise_sigmas"].astype("Float64").fillna(0) >= RISE_SIGMAS

    return daily[(rising & far).astype(bool)]


# --- Check 1: the corpus is fixed and shaped like the three collectors' records ------------------------------------


def test_corpus_is_fixed(corpus: dict[str, pd.DataFrame]):
    again = hs_.build_corpus()

    assert set(again) == set(corpus) == {*hs_.FEATURE_CLASSES, host_identity.LEASE_CLASS, host_identity.INVENTORY_CLASS}

    for name in corpus:
        pd.testing.assert_frame_equal(corpus[name], again[name])


def test_corpus_is_three_weeks_of_working_hours(corpus: dict[str, pd.DataFrame]):
    for name in hs_.FEATURE_CLASSES:
        frame = corpus[name]
        assert set(hs_.ID_COLUMNS) <= set(frame.columns)
        assert frame["event_time"].is_monotonic_increasing
        hours = (frame["event_time"].astype("int64") // (hs_.HOUR_S * hs_.NS_PER_SECOND)) % 24
        assert set(hours) == set(hs_.WORK_HOURS), name

    flows = corpus[hs_.FLOW_CLASS]
    joiner = flows[flows["src_ip"] == hs_.JOINER]

    # The joiner has no fortnight: its first flow is on the scored week's first day.
    assert int(joiner["event_time"].min()) >= hs_.SCORES_FROM_NS
    # Every workstation's address is leased, and every one but the kiosk's is in a peer group.
    assert set(corpus[host_identity.LEASE_CLASS]["ip"]) == set(hs_.HOSTS)
    assert [address for (address, group) in hs_.HOSTS.items() if group is None] == [hs_.KIOSK]


# --- Checks 2 through 6: determinism -------------------------------------------------------------------------------


@pytest.mark.gpu_and_cpu_mode
def test_double_run_diff(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    second = hs_.run_pipeline(pipeline_config, corpus)

    assert diff_frames(result, second) is None
    assert frame_digest(result) == frame_digest(second)


@pytest.mark.slow
@pytest.mark.gpu_and_cpu_mode
def test_cross_restart_diff(execution_mode, tmp_path):
    mode = "gpu" if execution_mode.value == "GPU" else "cpu"
    outputs = []

    for (label, hash_seed) in (("a", "0"), ("b", "4242")):
        out_path = tmp_path / f"restart_{label}.csv"
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hash_seed

        subprocess.run([sys.executable, DRIVER_PATH, str(out_path), mode], env=env, check=True, timeout=900)
        outputs.append(out_path.read_bytes())

    assert outputs[0] == outputs[1]


@pytest.mark.gpu_and_cpu_mode
def test_against_golden(result: pd.DataFrame):
    with open(GOLDEN_PATH, encoding="utf-8") as handle:
        golden_text = handle.read()

    rendered = hs_.render(result)

    if (rendered != golden_text):
        from io import StringIO  # pylint: disable=import-outside-toplevel
        as_text = {"dtype": str, "keep_default_na": False}
        difference = diff_frames(pd.read_csv(StringIO(rendered), **as_text),
                                 pd.read_csv(StringIO(golden_text), **as_text))

        pytest.fail(f"Output drifted from {os.path.basename(GOLDEN_PATH)}: {difference}. If the change is "
                    f"intended, regenerate the golden with {os.path.basename(DRIVER_PATH)} and review the diff.")


@pytest.mark.gpu_and_cpu_mode
def test_batch_split_sweep(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    thirds = {name: _split(frame, 3) for (name, frame) in corpus.items()}

    # The classes in thirds, and the merged stream into the host segment a day at a time.
    assert diff_frames(result, hs_.run_pipeline(pipeline_config, corpus, batches=thirds,
                                                host_batches=hs_.CORPUS_DAYS)) is None


@pytest.mark.gpu_and_cpu_mode
def test_permutation_within_windows(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    for seed in (1, 2):
        assert diff_frames(result, hs_.run_pipeline(pipeline_config, _permuted(corpus, seed))) is None, seed


@pytest.mark.gpu_and_cpu_mode
def test_permutation_check_has_teeth(pipeline_config: Config, corpus: dict):
    # The negative control for the check above. Without the imposed order, a flow from later in the hour can arrive
    # first, and the fan-out and reach the hour's frame is built from are counted in another order.
    ordered = hs_.run_pipeline(pipeline_config, corpus, impose_order=False)
    shuffled = hs_.run_pipeline(pipeline_config, _permuted(corpus, 5), impose_order=False)

    assert diff_frames(ordered, shuffled) is not None


# --- The frame -----------------------------------------------------------------------------------------------------


@pytest.mark.cpu_mode
def test_one_row_per_host_per_working_hour(result: pd.DataFrame):
    hosts = result[result["telemetry_class"] == hs_.HOST_CLASS]
    expected = sum(
        len(hs_.WORK_HOURS) for address in hs_.HOSTS for day in range(hs_.CORPUS_DAYS) if hs_.active(address, day))

    assert len(hosts) == len(result) == expected
    assert hosts.groupby(["host_key", "window_id"]).size().max() == 1
    assert set(hosts["host_key"]) == {hs_.host_key(address) for address in hs_.HOSTS}


@pytest.mark.cpu_mode
def test_each_frame_row_is_made_of_its_hosts_rows_in_its_hour(corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    hour_ns = hs_.HOUR_S * hs_.NS_PER_SECOND
    counts: dict = {}

    for name in hs_.FEATURE_CLASSES:
        frame = corpus[name]
        keys = ([hs_.host_key(address) for address in frame["src_ip"]]
                if "src_ip" in frame.columns else [host.lower() for host in frame["hostname"]])

        for (key, when) in zip(keys, frame["event_time"]):
            counts[(key, int(when) // hour_ns)] = counts.get((key, int(when) // hour_ns), 0) + 1

    built = {
        (key, int(window)): int(rows)
        for (key, window, rows) in zip(result["host_key"], result["window_id"], result["frame_rows"])
    }

    assert built == counts


@pytest.mark.cpu_mode
def test_every_frame_row_carries_its_peer_group_and_is_stamped_at_layer_three(result: pd.DataFrame):
    groups = {hs_.host_key(address): group for (address, group) in hs_.HOSTS.items()}

    assert [None if pd.isna(group) else group
            for group in result["ctx_peer_group"]] == [groups[key] for key in result["host_key"]]
    assert set(result["osi_layer"]) == {hs_.OSI_LAYER}
    assert list(result["entity_key"]) == list(result["host_key"])
    assert result["lineage_id"].notna().all()
    assert set(result["collector_id"]) == {"host-frame"}
    assert sorted(result["collector_seq"]) == list(range(len(result)))


# --- The models ----------------------------------------------------------------------------------------------------


def test_the_committed_models_are_the_fortnight_this_corpus_builds():
    with open(hs_.MODELS_PATH, encoding="utf-8") as handle:
        recorded = json.load(handle)

    trained = recorded["trained"]

    assert set(recorded["models"]) == {hs_.FINANCE, hs_.ENGINEERING, hs_.POPULATION}
    assert trained["scores_from_ns"] == hs_.SCORES_FROM_NS
    assert trained["training_days"] == hs_.TRAINING_DAYS
    assert trained["device"] == "cpu"

    for entry in recorded["models"].values():
        assert entry["model"]["features"] == hs_.SCORED_FEATURES

    # The versions are the digests of the numbers beside them; `load_models` refuses the file otherwise.
    (_, manifest) = hs_.committed_scoring()

    assert manifest.fallback == recorded["models"][hs_.POPULATION]["model_version"]


@pytest.mark.cpu_mode
def test_the_training_rows_are_what_the_models_were_fitted_on(result: pd.DataFrame):
    with open(hs_.MODELS_PATH, encoding="utf-8") as handle:
        recorded = json.load(handle)["trained"]["rows"]

    frames = hs_.training_frames(result)

    assert {key: len(frame) for (key, frame) in frames.items()} == recorded


@pytest.mark.cpu_mode
def test_no_row_of_the_fortnight_is_scored_or_names_a_model(result: pd.DataFrame):
    training = result[result["event_time"].astype("int64") < hs_.SCORES_FROM_NS]

    assert len(training) > 0
    assert training["mean_abs_z"].isna().all()
    assert training["model_version"].isna().all()


@pytest.mark.cpu_mode
def test_every_host_is_scored_against_its_peer_groups_model_and_the_kiosk_against_the_estate(result: pd.DataFrame):
    (_, manifest) = hs_.committed_scoring()
    with open(hs_.MODELS_PATH, encoding="utf-8") as handle:
        versions = {key: entry["model_version"] for (key, entry) in json.load(handle)["models"].items()}

    scored = result[result["event_time"].astype("int64") >= hs_.SCORES_FROM_NS]

    for (address, group) in hs_.HOSTS.items():
        rows = scored[scored["host_key"] == hs_.host_key(address)]
        expected = versions[group if group is not None else hs_.POPULATION]

        assert set(rows["model_version"]) == {expected}, address
        assert set(rows["model_fallback_used"].astype(bool)) == {group is None}, address

    # Two hosts of a group share one model, which is the point of a group's model.
    assert len(set(manifest.models.values())) == 2
    assert manifest.entities == len(hs_.HOSTS) - 1


@pytest.mark.cpu_mode
def test_the_joiner_is_scored_by_its_groups_model_once_its_features_are_whole(result: pd.DataFrame):
    rows = result[result["host_key"] == hs_.host_key(hs_.JOINER)].assign(day=_day(result))
    gaps = rows["dsts_per_src_step"].isna()

    # Its first day of flows has no hour of its own to step past; those frames carry a gap and no score, though
    # they name the model the host resolves to.
    assert gaps.any() and (~gaps).any()
    assert rows.loc[gaps, "mean_abs_z"].isna().all()
    assert rows.loc[~gaps, "mean_abs_z"].notna().all()
    assert set(rows.loc[~gaps, "model_version"]) == {hs_.committed_scoring()[1].models[hs_.host_key(hs_.JOINER)]}
    # And scored, it looks like the rest of its group.
    controls = _daily(result)
    controls = controls[~controls["host_key"].isin([hs_.host_key(hs_.DRIFTING), hs_.host_key(hs_.STEPPED)])]

    assert rows.loc[~gaps, "mean_abs_z"].mean() <= controls["mean_abs_z"].max()


# --- What the scores show ------------------------------------------------------------------------------------------


@pytest.mark.cpu_mode
def test_the_drifting_host_climbs_every_scored_day(result: pd.DataFrame):
    daily = _daily(result)
    drifting = daily[daily["host_key"] == hs_.host_key(hs_.DRIFTING)].sort_values("day")

    assert len(drifting) == hs_.SCORED_DAYS
    assert list(drifting["mean_abs_z"]) == sorted(drifting["mean_abs_z"])
    assert list(drifting["rising"]) == list(range(1, hs_.SCORED_DAYS + 1))


@pytest.mark.cpu_mode
def test_only_the_drifting_host_meets_the_trajectory(result: pd.DataFrame):
    climbs = _climbs(_daily(result))

    assert set(climbs["host_key"]) == {hs_.host_key(hs_.DRIFTING)}
    # From the fourth rising day on: the scored week's first day starts the run, the drift its second.
    assert sorted(climbs["day"]) == list(range(hs_.TRAINING_DAYS + RISING_WINDOWS - 1, hs_.CORPUS_DAYS))


@pytest.mark.cpu_mode
def test_the_stepped_host_scores_high_and_does_not_climb(result: pd.DataFrame):
    daily = _daily(result)
    stepped = daily[daily["host_key"] == hs_.host_key(hs_.STEPPED)]
    after = stepped[stepped["day"] >= hs_.TRAINING_DAYS + hs_.STEP_DAY]
    others = daily[~daily["host_key"].isin([hs_.host_key(hs_.STEPPED), hs_.host_key(hs_.DRIFTING)])]

    # Every day after the change scores above any day of any ordinary host...
    assert after["mean_abs_z"].min() > 10 * others["mean_abs_z"].max()
    # ...and the trajectory, which asks whether it keeps rising, says it does not.
    assert stepped["rising"].max() < RISING_WINDOWS


@pytest.mark.cpu_mode
def test_every_scored_row_names_the_features_its_score_is_made_of(result: pd.DataFrame):
    scored = result[result["mean_abs_z"].notna()]
    losses = scored[[f"{name}_z_loss" for name in hs_.SCORED_FEATURES]]

    assert len(scored) > 0
    assert losses.notna().all().all()
    assert ((losses.mean(axis=1).round(4) - scored["mean_abs_z"]).abs() <= 1e-4).all()


@pytest.mark.cpu_mode
def test_the_numbers_the_documents_quote(result: pd.DataFrame):
    # README.md and examples/layer5_model/README.md quote these; a retrained model or a changed corpus moves them.
    daily = _daily(result)
    drifting = daily[daily["host_key"] == hs_.host_key(hs_.DRIFTING)].sort_values("day")
    stepped = daily[(daily["host_key"] == hs_.host_key(hs_.STEPPED))
                    & (daily["day"] >= hs_.TRAINING_DAYS + hs_.STEP_DAY)]
    ordinary = daily[~daily["host_key"].isin([hs_.host_key(hs_.DRIFTING), hs_.host_key(hs_.STEPPED)])]

    assert round(drifting["mean_abs_z"].iloc[0], 1) == 0.7
    assert round(drifting["mean_abs_z"].iloc[-1]) == 379
    assert ordinary["mean_abs_z"].max() < 1.6
    assert 50 < stepped["mean_abs_z"].min() and stepped["mean_abs_z"].max() < 75

    with open(hs_.MODELS_PATH, encoding="utf-8") as handle:
        rows = json.load(handle)["trained"]["rows"]

    assert rows == {hs_.FINANCE: 224, hs_.ENGINEERING: 224, hs_.POPULATION: 504}
