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
Control 13's six checks against the composed layer 5 pipeline, and every planted case asserted.

The five TC-5 stages had per-stage tests and had never been run composed with each other. That is the gap this
file closes, in the same six checks the layer 1 and 2 harness runs: a fixed corpus, a double run, a cross-restart
under a different hash seed, a golden file, a batch-split sweep, and a permutation within windows with its own
negative control.

Beyond the six, every planted case is asserted as the column a rule would read, together with the case beside it
that must stay quiet. That second half is what the assertions are for: a rule that fires on a corpus built to
make it fire has been shown almost nothing.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.utils.determinism import diff_frames
from morpheus.utils.determinism import frame_digest
from morpheus.utils.determinism import permute_within_contiguous_groups
from morpheus.utils.determinism import quantize_value
from morpheus.utils.dfencoder_scorer import DfencoderScorer
from morpheus.utils.dfencoder_scorer import load_models
from morpheus.utils.model_manifest import ModelManifest
from morpheus.utils.lineage import window_id_from_timestamp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import session_pipeline as sp  # noqa: E402
import stamping  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_session_expected.csv")
DRIVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_session_pipeline.py")

NS = sp.NS_PER_SECOND

IMPOSSIBLE_KMH = 900
"""R-D-L5-003's threshold. Named here so the assertions read as the rule rather than as a number."""


@pytest.fixture(name="corpus", scope="module")
def corpus_fixture() -> dict[str, pd.DataFrame]:
    yield sp.build_corpus()


# One composed run per execution mode, reused by every check in that mode.
_RESULTS: dict = {}


@pytest.fixture(name="pipeline_config")
def pipeline_config_fixture(execution_mode) -> Config:
    yield sp.build_pipeline_config(execution_mode)


@pytest.fixture(name="result")
def result_fixture(pipeline_config: Config, corpus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mode = pipeline_config.execution_mode

    if (mode not in _RESULTS):
        _RESULTS[mode] = sp.run_pipeline(pipeline_config, corpus)

    yield _RESULTS[mode]


def _rows(result: pd.DataFrame, telemetry_class: str) -> pd.DataFrame:
    return result[result["telemetry_class"] == telemetry_class]


def _principal(result: pd.DataFrame, principal: str) -> pd.DataFrame:
    auth = _rows(result, "tc5_auth")

    return auth[auth["user_principal"] == principal].sort_values("event_time")


def _scored_week(auth: pd.DataFrame) -> pd.DataFrame:
    """The rows on or after the end of the training fortnight: the only ones a model may score."""
    return auth[auth["event_time"].astype("int64") >= sp.SCORES_FROM_NS]


def _fortnight(auth: pd.DataFrame) -> pd.DataFrame:
    """The rows the models were trained on."""
    return auth[auth["event_time"].astype("int64") < sp.SCORES_FROM_NS]


MODELLED = (sp.ALICE, sp.BOB, sp.CAROL, sp.DAVE, sp.BATCH, sp.ERIN)
"""Every principal with a fortnight to train on, and so a model of their own."""

DEPARTED = {sp.ERIN, sp.ALICE, sp.BOB, sp.CAROL}
"""The modelled principals whose scored week departs from their fortnight: the takeover, the fumbled password and
the off-hours sign-in and the journey, the flight and the move to New York, and the fatigue burst."""

UNCHANGED = {sp.DAVE, sp.BATCH}
"""The modelled principals whose scored week is their fortnight again: the controls."""


def _windows(frame: pd.DataFrame) -> list[int]:
    return [window_id_from_timestamp(int(t), sp.PERIOD_SECONDS * NS) for t in frame["event_time"]]


def _split(frame: pd.DataFrame, parts: int) -> list[pd.DataFrame]:
    size = max(1, len(frame) // parts)

    return [frame.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(frame), size)]


def _permuted(corpus: dict[str, pd.DataFrame], seed: int) -> dict[str, pd.DataFrame]:
    return {
        name: permute_within_contiguous_groups(frame, _windows(frame), seed=seed)
        for (name, frame) in corpus.items()
    }


# --- Check 1: the corpus ---------------------------------------------------------------------------------------


def test_corpus_is_fixed(corpus: dict[str, pd.DataFrame]):
    again = sp.build_corpus()

    assert set(again) == set(corpus)

    for name in corpus:
        pd.testing.assert_frame_equal(corpus[name], again[name])


def test_corpus_is_shaped_like_an_identity_provider(corpus: dict[str, pd.DataFrame]):
    auth = corpus["tc5_auth"]

    # A week, so a histogram of hours has something to be a histogram of.
    span_days = (auth["event_time"].max() - auth["event_time"].min()) / (NS * sp.DAY_S)
    assert span_days > 5

    # Several principals with different habits, which is what makes each one's own history the comparison.
    assert auth["user_principal"].nunique() >= 5

    # Sessions arrive mostly as separate start and stop records with no duration on either, which is the shape
    # TC5SessionStage exists to pair; the identity provider's own collector adds the other shape, one record per
    # session with both ends in it.
    sessions = corpus["tc5_session"]
    assert set(sessions["session_action"]) == {"start", "end", "session"}
    assert "session_duration_s" not in sessions.columns
    single = sessions[sessions["session_action"] == "session"]
    assert single["session_start"].notna().all() and single["session_end"].notna().all()
    assert sessions[sessions["session_action"] != "session"]["session_start"].isna().all()

    # Every class carries the envelope the identifiers derive from, each collector with a monotonic sequence.
    for frame in corpus.values():
        assert set(sp.ID_COLUMNS) <= set(frame.columns)

        for (_, rows) in frame.groupby("collector_id"):
            assert rows["collector_seq"].is_monotonic_increasing


# --- Checks 2 through 6: determinism ---------------------------------------------------------------------------


@pytest.mark.gpu_and_cpu_mode
def test_double_run_diff(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    second = sp.run_pipeline(pipeline_config, corpus)

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

    rendered = sp.render(result)

    if (rendered != golden_text):
        from io import StringIO
        as_text = {"dtype": str, "keep_default_na": False}
        difference = diff_frames(pd.read_csv(StringIO(rendered), **as_text),
                                 pd.read_csv(StringIO(golden_text), **as_text))

        pytest.fail(f"Output drifted from {os.path.basename(GOLDEN_PATH)}: {difference}. If the change is intended, "
                    f"regenerate the golden file with {os.path.basename(DRIVER_PATH)} and review the diff.")


@pytest.mark.gpu_and_cpu_mode
def test_batch_split_sweep(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    thirds = {name: _split(frame, 3) for (name, frame) in corpus.items()}
    by_row = {name: _split(frame, len(frame)) for (name, frame) in corpus.items()}

    assert diff_frames(result, sp.run_pipeline(pipeline_config, corpus, batches=thirds)) is None
    assert diff_frames(result, sp.run_pipeline(pipeline_config, corpus, batches=by_row)) is None


class _ShapeSensitiveModel:
    """
    Stands in for what a real network on a card is: not invariant to the shape of the batch it is handed.

    Two matrices multiplied on a GPU do not agree to the last bit across batch shapes, because the shape picks
    the kernel. The perturbation here is a seventh-place function of the row count, which is the order of what
    a card actually does, applied deterministically so the test is about shape and nothing else.
    """

    def get_results(self, df: pd.DataFrame, return_abs: bool = False) -> pd.DataFrame:
        del return_abs
        wobble = 1e-7 * len(df)

        return pd.DataFrame({f"{name}_z_loss": df[name].astype("float64") * 0.001 + wobble
                             for name in df.columns},
                            index=df.index)


def _shape_sensitive(rows_per_call: int):
    version = "dfencoder/probe:0001"
    principals = list(MODELLED) + [sp.GRACE]
    scorer = DfencoderScorer({version: _ShapeSensitiveModel()}, sp.SCORED_FEATURES, rows_per_call=rows_per_call)
    manifest = ModelManifest(window_id=sp.SCORING_WINDOW,
                             models={principal: version
                                     for principal in principals},
                             fallback=None)

    return (scorer, manifest)


@pytest.mark.gpu_and_cpu_mode
def test_a_row_is_scored_the_same_however_the_stream_was_chunked(pipeline_config: Config,
                                                                 corpus: dict[str, pd.DataFrame]):
    # Check 5 reaches the model. TC5ScoreStage hands the scorer the rows an entity has in the message it is
    # holding, so the size of that group is a fact about the batching. Handing that size through to a model
    # makes the score a function of how the stream arrived, which is exactly what check 5 forbids.
    (scorer, manifest) = _shape_sensitive(rows_per_call=1)

    whole = sp.run_pipeline(pipeline_config, corpus, scorer=scorer, manifest=manifest)
    thirds = {name: _split(frame, 3) for (name, frame) in corpus.items()}
    chunked = sp.run_pipeline(pipeline_config, corpus, batches=thirds, scorer=scorer, manifest=manifest)

    assert diff_frames(whole, chunked) is None


@pytest.mark.gpu_and_cpu_mode
def test_the_shape_check_has_teeth(pipeline_config: Config, corpus: dict[str, pd.DataFrame]):
    # The negative control for the check above. Let the group's size reach the model and the run must differ --
    # and differ by far more than the seventh place it started in, because drift_rise_sigmas divides a rise by
    # the spread of a few nearly equal scores. If this ever stops differing, the check above proves nothing.
    (scorer, manifest) = _shape_sensitive(rows_per_call=10_000)

    whole = sp.run_pipeline(pipeline_config, corpus, scorer=scorer, manifest=manifest)
    thirds = {name: _split(frame, 3) for (name, frame) in corpus.items()}
    chunked = sp.run_pipeline(pipeline_config, corpus, batches=thirds, scorer=scorer, manifest=manifest)

    assert diff_frames(whole, chunked) is not None


@pytest.mark.gpu_and_cpu_mode
def test_permutation_check_has_teeth(pipeline_config: Config, corpus: dict[str, pd.DataFrame]):
    # The negative control for check 6. Every stage at this layer is cumulative -- a run of denials, a distinct
    # location count, a histogram, a previous location -- so removing the total-order stage must make the output a
    # function of arrival order. If this ever passes with no diff, the permutation check has stopped proving
    # anything.
    unsorted_baseline = sp.run_pipeline(pipeline_config, corpus, impose_order=False)

    detected = False
    for seed in (1, 2, 3):
        detected = detected or (diff_frames(
            unsorted_baseline, sp.run_pipeline(pipeline_config, _permuted(corpus, seed), impose_order=False))
                                is not None)

    assert detected, "Removing the total-order stage did not change any output under permutation."


@pytest.mark.gpu_and_cpu_mode
def test_permutation_within_windows(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    for seed in (1, 2, 3):
        shuffled = _permuted(corpus, seed)

        assert any(not shuffled[name]["collector_seq"].equals(corpus[name]["collector_seq"])
                   for name in corpus), "permutation was a no-op"

        difference = diff_frames(result, sp.run_pipeline(pipeline_config, shuffled))
        assert difference is None, f"seed {seed}: {difference}"


# --- The composition -------------------------------------------------------------------------------------------


@pytest.mark.cpu_mode
def test_every_class_ran_and_sealed(result: pd.DataFrame):
    assert set(result["telemetry_class"]) == set(sp.TELEMETRY_CLASSES)

    for name in sp.TELEMETRY_CLASSES:
        rows = _rows(result, name)
        assert len(rows) > 0, name
        assert rows["window_id"].notna().all(), f"{name} left a row unsealed"
        assert rows["lineage_id"].notna().any(), f"{name} carries no lineage identifier"


@pytest.mark.cpu_mode
def test_a_chain_is_one_principal_inside_one_window(result: pd.DataFrame):
    auth = _rows(result, "tc5_auth")
    grouped = auth.groupby("lineage_id")[["user_principal", "window_id"]].nunique()

    assert (grouped["user_principal"] == 1).all(), "a chain spans two principals"
    assert (grouped["window_id"] == 1).all(), "a chain spans two windows"

    # And two principals in one window are two chains, or the identifier would be a window number with extra
    # characters.
    per_window = auth.groupby("window_id")["lineage_id"].nunique()
    per_window_principals = auth.groupby("window_id")["user_principal"].nunique()

    assert (per_window == per_window_principals).all()


# --- The planted cases -----------------------------------------------------------------------------------------


@pytest.mark.cpu_mode
def test_the_impossible_journey_is_impossible_and_the_flight_is_not(result: pd.DataFrame):
    auth = _rows(result, "tc5_auth")
    fast = auth[auth["travel_kmh"] > IMPOSSIBLE_KMH]

    # One interloper produces two crossings, not one: the jump to New York and the victim's next authentication
    # back in London. A detection engineer tuning this rule should expect a pair per intrusion rather than a
    # single alert, and that is a property of the rule rather than of this corpus.
    assert set(fast["user_principal"]) == {sp.ALICE}
    assert len(fast) == 2

    outbound = fast[fast["user_location"] == "us:ny:new-york"]
    assert len(outbound) == 1
    assert float(outbound["travel_distance_km"].iloc[0]) == pytest.approx(5570, abs=15)

    # Bob makes the same crossing in eight hours, which is what the aircraft takes, and stays quiet.
    bob = _principal(result, sp.BOB)
    crossing = bob[bob["travel_distance_km"] > 100]

    assert len(crossing) == 1
    assert float(crossing["travel_kmh"].iloc[0]) < IMPOSSIBLE_KMH
    assert float(crossing["travel_kmh"].iloc[0]) > 500, "the flight should still be a real journey"


@pytest.mark.cpu_mode
def test_the_token_refresh_did_not_become_the_anchor(result: pd.DataFrame):
    alice = _principal(result, sp.ALICE)
    refresh = alice[alice["travel_status"] == "token_refresh"]

    assert len(refresh) == 1
    assert refresh["travel_kmh"].isna().all()

    # The measurement that follows is timed from her office login, half an hour earlier, not from the refresh
    # fifteen minutes earlier. Were the refresh the anchor the speed would read twice as high, and a refresh from
    # somewhere else would have erased the journey entirely.
    outbound = alice[alice["user_location"] == "us:ny:new-york"]

    assert len(outbound) == 1
    assert int(outbound["travel_elapsed_ns"].iloc[0]) == sp.IMPOSSIBLE_GAP_S * NS


@pytest.mark.cpu_mode
def test_the_vpn_user_is_excluded_rather_than_alerted_on(result: pd.DataFrame):
    # The negative control for the exclusion path, and the reason it exists: without the egress range this
    # principal crosses six hundred kilometres twice a day, every day, and the rule fires on all of it.
    dave = _principal(result, sp.DAVE)
    excluded = dave[dave["travel_status"] == "vpn_egress"]

    assert len(excluded) > 10
    assert excluded["travel_kmh"].isna().all()
    assert float(dave["travel_kmh"].max()) == 0.0, "an excluded record moved the anchor"


@pytest.mark.cpu_mode
def test_the_fatigue_burst_is_readable_off_the_approving_row(result: pd.DataFrame):
    # R-D-L5-004: more than five challenges in ten minutes, at least four denials, and an approval at the end.
    auth = _rows(result, "tc5_auth")
    firing = auth[(auth["mfa_attempts_in_window"] > 5) & (auth["mfa_denials_in_window"] >= 4)
                  & (auth["mfa_denied_then_approved"])]

    assert len(firing) == 1
    assert firing["user_principal"].iloc[0] == sp.CAROL
    assert int(firing["mfa_denials_in_window"].iloc[0]) == sp.FATIGUE_DENIALS
    assert int(firing["consecutive_mfa_denials"].iloc[0]) == sp.FATIGUE_DENIALS


@pytest.mark.cpu_mode
def test_the_fumbled_password_is_not_a_fatigue_attack(result: pd.DataFrame):
    # Failure-then-success without any of the denial volume the rule requires, which is what most of an estate
    # does on a Monday. The feature fires; the rule must not.
    alice = _principal(result, sp.ALICE)
    recovered = alice[alice["auth_failed_then_succeeded"]]

    assert len(recovered) == 1
    assert int(recovered["consecutive_auth_failures"].iloc[0]) == 2
    assert recovered["mfa_denials_in_window"].isna().all(), "the factor was never challenged"


@pytest.mark.cpu_mode
def test_the_off_hours_login_is_unseen_and_the_nightly_batch_is_not(result: pd.DataFrame):
    # The negative control that makes the cadence feature a statement about a principal rather than about a clock.
    # The two rows are the same hour on the same night.
    alice = _principal(result, sp.ALICE)
    off_hours = alice[alice["local_hour"] == sp.OFF_HOURS_HOUR]

    assert len(off_hours) == 1
    assert bool(off_hours["hour_unseen"].iloc[0]) is True
    assert bool(off_hours["cadence_mature"].iloc[0]) is True

    batch = _principal(result, sp.BATCH)
    assert (batch["local_hour"] == sp.BATCH_HOUR).all()

    # Its first night is new, and every night after it is not.
    assert bool(batch["hour_unseen"].iloc[0]) is True
    assert not batch["hour_unseen"].iloc[1:].any()

    # By the last night it has enough history to be believed, and the same hour that alarms for her is ordinary.
    assert bool(batch["cadence_mature"].iloc[-1]) is True
    assert float(batch["hour_surprise_bits"].iloc[-1]) < float(off_hours["hour_surprise_bits"].iloc[0])


@pytest.mark.cpu_mode
def test_a_new_country_raises_the_cumulative_count_and_keeps_it_raised(result: pd.DataFrame):
    alice = _principal(result, sp.ALICE)
    increments = alice["locincrement"].dropna().tolist()

    assert increments[0] == 1
    assert increments[-1] == 2
    assert increments == sorted(increments), "a cumulative count went backwards"

    # And the row where it rose is the one that named the new place.
    rose = alice[alice["location_first_seen"] == True]  # noqa: E712  pylint: disable=singleton-comparison
    assert len(rose) == 1
    assert rose["user_location"].iloc[0] == "us:ny:new-york"


@pytest.mark.cpu_mode
def test_an_ordinary_session_carries_its_duration(result: pd.DataFrame):
    sessions = _rows(result, "tc5_session")
    closed = sessions[sessions["session_duration_s"].notna()]

    assert len(closed) > 10
    assert (closed[closed["session_id"].str.startswith("sess-alice")]["session_duration_s"] == 8 * 3600).all()


@pytest.mark.cpu_mode
def test_the_three_ways_a_session_goes_wrong_are_each_reported(result: pd.DataFrame):
    sessions = _rows(result, "tc5_session")

    # A stop with no start at all.
    orphan = sessions[sessions["session_id"] == sp.UNPAIRED_SESSION]
    assert len(orphan) == 1
    assert bool(orphan["session_unpaired"].iloc[0]) is True

    # A start abandoned past the timeout, whose stop arrives days later. It must read as unpaired rather than as a
    # five-day session: reporting one would put an absurd duration on an ordinary working day.
    abandoned = sessions[sessions["session_id"] == sp.ABANDONED_SESSION]
    stop = abandoned[abandoned["session_action"] == "end"]

    assert len(stop) == 1
    assert bool(stop["session_unpaired"].iloc[0]) is True
    assert stop["session_duration_s"].isna().all()

    # One identifier collecting two starts, which is a duplicating collector rather than a retry.
    duplicated = sessions[sessions["session_id"] == sp.DUPLICATED_SESSION]
    closed = duplicated[duplicated["session_action"] == "end"]

    assert len(closed) == 1
    assert int(closed["session_starts"].iloc[0]) == 2
    # Timed from the second start, which is what the timer holds.
    assert int(closed["session_duration_s"].iloc[0]) == 2 * 3600 - 30


@pytest.mark.cpu_mode
def test_nothing_else_fires(result: pd.DataFrame):
    # The half that matters. Each planted case above is asserted to fire; this says the rest of the week does not.
    auth = _rows(result, "tc5_auth")

    quiet = auth[~auth["user_principal"].isin([sp.ALICE, sp.CAROL])]

    assert (quiet["travel_kmh"].fillna(0) <= IMPOSSIBLE_KMH).all(), "a principal with no planted journey crossed"
    assert not quiet["mfa_denied_then_approved"].fillna(False).any(), "a principal with no planted burst fired"

    # And within the two principals that do fire, the firing is confined to the days it was planted on.
    days = ((auth["event_time"] // (NS * sp.DAY_S)) - (sp.CORPUS_EPOCH_S // sp.DAY_S))
    fired = auth[auth["travel_kmh"] > IMPOSSIBLE_KMH]

    assert set(days[fired.index]) == {sp.IMPOSSIBLE_DAY}


@pytest.mark.cpu_mode
def test_the_columns_four_rules_have_waited_for_now_exist(result: pd.DataFrame):
    # `mean_abs_z` and `max_abs_z` had no producer anywhere in this fork. R-B-L5-001, R-B-L5-002, R-B-L5-005 and
    # R-P-L5-006 read them, so all four fired on nothing and the drift trajectory read a column of nulls.
    #
    # Every row of the scored week carries them, and no row of the fortnight does: those rows were the models'
    # training data, and a score on them would say how well a model memorized its own inputs.
    auth = _rows(result, "tc5_auth")
    week = _scored_week(auth)
    fortnight = _fortnight(auth)

    assert len(week) > 0 and len(fortnight) > 0
    assert week["mean_abs_z"].notna().all()
    assert week["max_abs_z"].notna().all()
    assert fortnight["mean_abs_z"].isna().all()
    assert fortnight["max_abs_z"].isna().all()

    for name in sp.SCORED_FEATURES:
        assert f"{name}_z_loss" in auth.columns


@pytest.mark.cpu_mode
def test_the_summaries_agree_with_the_losses_beside_them(result: pd.DataFrame):
    # Derived here rather than asked of the scorer, so a model cannot report a mean that disagrees with the
    # per-feature losses printed next to it -- a discrepancy no rule would catch and every analyst would trust.
    # Summed in feature order and quantized the way the stage does it: at the tens of thousands a learned model
    # reaches here, a sum taken in another order can land the mean on the other side of a half-quantum.
    scored = _scored_week(_rows(result, "tc5_auth"))
    losses = [f"{name}_z_loss" for name in sp.SCORED_FEATURES]

    for (_, row) in scored.iterrows():
        values = [float(row[column]) for column in losses]

        assert max(values) == float(row["max_abs_z"])
        assert quantize_value(sum(values) / len(values)) == float(row["mean_abs_z"])


@pytest.mark.cpu_mode
def test_each_principal_is_scored_against_its_own_committed_model(result: pd.DataFrame):
    # Control 1 with a learned model in the slot. Every principal with a fortnight to train on is pinned to the
    # digest of the numbers committed for them, and every row of theirs in the scored week names that version
    # and says it was not a fallback. The joiner, who has no fortnight, is scored against the population and
    # says so on every row.
    manifest = sp.scoring_manifest()
    committed = load_models(sp.MODELS_PATH)

    assert sorted(manifest.models) == sorted(MODELLED)
    assert manifest.fallback == sp.FALLBACK_VERSION
    assert manifest.scores_from_ns == sp.SCORES_FROM_NS

    week = _scored_week(_rows(result, "tc5_auth"))

    for principal in MODELLED:
        rows = week[week["user_principal"] == principal]
        version = committed[principal][0]

        assert manifest.models[principal] == version
        assert version.startswith(f"dfencoder/{principal}:")
        assert len(rows) > 0
        assert (rows["model_version"] == version).all(), principal
        assert not rows["model_fallback_used"].astype("boolean").any(), principal

    joiner = week[week["user_principal"] == sp.GRACE]

    assert len(joiner) > 0
    assert (joiner["model_version"] == sp.FALLBACK_VERSION).all()
    assert joiner["model_fallback_used"].astype("boolean").all()
    assert set(week[week["model_fallback_used"].astype("boolean")]["user_principal"]) == {sp.GRACE}


@pytest.mark.cpu_mode
def test_the_fortnight_names_no_model_and_the_session_class_is_not_scored(result: pd.DataFrame):
    # A row the models were trained on was not scored, so naming a model beside it would claim a score nobody
    # produced. The session class has no scorer at all.
    fortnight = _fortnight(_rows(result, "tc5_auth"))

    assert len(fortnight) > 0
    assert fortnight["model_version"].isna().all()
    assert fortnight["model_fallback_used"].isna().all()
    assert sp.GRACE not in set(fortnight["user_principal"])

    unscored = result[result["telemetry_class"] == "tc5_session"]

    assert len(unscored) > 0
    assert unscored["model_version"].isna().all()
    assert unscored["model_fallback_used"].isna().all()


@pytest.mark.cpu_mode
def test_every_row_carries_the_determinism_envelope(result: pd.DataFrame):
    # Control 12 at the row: the tier the harness asserts, one configuration hash per corpus, and a fingerprint that
    # names the class, so two events from two runs can be told comparable or not without the runs' logs.
    assert (result["determinism_tier"] == "D1").all()
    assert result["config_hash"].nunique() == 1
    assert result["pipeline_fingerprint"].notna().all()
    assert set(result["feature_schema_version"]) == {"tc5_auth/1.0.0", "tc5_session/1.0.0"}
    assert (result["code_commit"] == "unknown").all()
    assert (result["rng_seed"] == 0).all()


def _model_rules(auth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """R-B-L5-001 and R-B-L5-002 as their searches state them, gate included."""
    limits = stamping.rule_thresholds(["R-B-L5-001", "R-B-L5-002"])
    own = auth[auth["mean_abs_z"].notna() & ~auth["model_fallback_used"].astype("boolean").fillna(True)]

    composite = own[(own["max_abs_z"].astype(float) >= limits["R-B-L5-001"]["max_threshold"])
                    & (own["mean_abs_z"].astype(float) >= limits["R-B-L5-001"]["mean_threshold"])]
    location = own[own["locincrement_z_loss"].astype(float) >= limits["R-B-L5-002"]["loss_threshold"]]

    return (composite, location)


@pytest.mark.cpu_mode
def test_the_model_rules_fire_on_the_takeover_and_not_on_the_controls(result: pd.DataFrame):
    # The case the models exist for. Erin's afternoon is nothing the deterministic rules call impossible -- four
    # hours is time enough to reach Amsterdam -- and it is unlike every row of her fortnight on half the features
    # at once. Both model rules fire on every sign-in of it.
    auth = _rows(result, "tc5_auth")
    (composite, location) = _model_rules(auth)
    erin = _principal(result, sp.ERIN)
    takeover = erin[erin["source_city"] == sp.AMSTERDAM_PLACE[2]]

    assert len(takeover) == len(sp.TAKEOVER_HOURS) * len(sp.TAKEOVER_APPS)
    assert set(takeover.index) <= set(composite.index)
    assert set(takeover.index) <= set(location.index)

    # Nothing of hers fires before the first takeover sign-in.
    before = erin[erin["event_time"] < takeover["event_time"].min()]
    assert not set(before.index) & (set(composite.index) | set(location.index))

    # The controls. Dave and the service account have models of their own and a week that repeats their
    # fortnight, and neither rule fires on them; the joiner has no model, and the gate keeps both rules off her.
    for principal in UNCHANGED | {sp.GRACE}:
        assert principal not in set(composite["user_principal"]), principal
        assert principal not in set(location["user_principal"]), principal

    assert set(composite["user_principal"]) == DEPARTED


@pytest.mark.cpu_mode
def test_every_departure_the_week_was_built_with_is_one_the_composite_rule_reads(result: pd.DataFrame):
    # The other planted cases are departures too, and the composite rule fires on each from the row it starts:
    # Alice's first failed password, Bob's sign-in at an hour he has never used before his flight, and the first
    # of Carol's refused prompts. These are the rows a per-principal model should find unlike a fortnight of
    # office hours; asserting them stops a later change from quietly losing one.
    auth = _rows(result, "tc5_auth")
    (composite, _) = _model_rules(auth)

    def first(principal: str) -> pd.Series:
        rows = composite[composite["user_principal"] == principal].sort_values("event_time")
        return rows.iloc[0]

    assert first(sp.ALICE)["auth_result"] == "failure"
    assert int(first(sp.ALICE)["local_hour"]) == sp.FUMBLE_HOUR
    assert int(first(sp.BOB)["local_hour"]) == sp.FLIGHT_DEPART_HOUR
    assert first(sp.CAROL)["mfa_result"] == "denied"


@pytest.mark.cpu_mode
def test_the_location_rule_fires_on_the_three_who_went_somewhere_new(result: pd.DataFrame):
    # R-B-L5-002 reads the reconstruction error on `locincrement`, which is that feature's error given all ten,
    # not a flag for a new place: Bob's sign-in before his flight moves it, from London, because the hour is one
    # his fortnight never used. Every principal it fires on reached a new place in the week, and the ones who did
    # not are quiet.
    auth = _rows(result, "tc5_auth")
    (_, location) = _model_rules(auth)

    assert set(location["user_principal"]) == {sp.ERIN, sp.ALICE, sp.BOB}
    assert not set(location["user_principal"]) & (UNCHANGED | {sp.CAROL, sp.GRACE})


@pytest.mark.cpu_mode
def test_a_cumulative_feature_keeps_the_score_raised_until_the_next_training(result: pd.DataFrame):
    # The `*increment` features never fall: once Erin's account has reached four applications, two devices and two
    # places, every later row of hers carries those counts, back in London on Thursday and Friday included. A
    # model trained before the afternoon has never seen them, so both rules keep firing on her until a model is
    # trained on a window that includes it. That is the property, recorded rather than tuned away.
    auth = _rows(result, "tc5_auth")
    (composite, location) = _model_rules(auth)
    erin = _scored_week(_principal(result, sp.ERIN))
    days = ((erin["event_time"].astype("int64") // NS - sp.CORPUS_EPOCH_S) // sp.DAY_S).astype(int)
    after = erin[days > sp.TAKEOVER_DAY]

    assert len(after) > 0
    assert (after["source_city"] == sp.LONDON_PLACE[2]).all()
    assert (after["appincrement"].astype(float) == 1 + len(sp.TAKEOVER_APPS)).all()
    assert set(after.index) <= set(composite.index)
    assert set(after.index) <= set(location.index)


@pytest.mark.cpu_mode
def test_the_scores_are_not_calibrated_where_the_fortnight_never_varied(result: pd.DataFrame):
    # The upstream loss scaler standardizes each feature's reconstruction error against the errors seen in
    # training. Where the fortnight never varied a feature -- every office worker used one application on one
    # device -- the model reconstructs it almost exactly and the spread of those errors is a rounding residue,
    # so any change at all is divided by next to nothing. Erin's afternoon scores in the tens of thousands. That
    # is not tens of thousands of deviations of anything: it means "departed from the fortnight", and it is why
    # the rules' thresholds of 6 and 2 read as that and nothing finer, and why scores are not comparable across
    # principals. A floor on the spread, or a longer and more varied training window, is what calibration needs.
    committed = load_models(sp.MODELS_PATH)
    document = None

    with open(sp.MODELS_PATH, encoding="utf-8") as handle:
        import json  # pylint: disable=import-outside-toplevel
        document = json.load(handle)["models"][sp.ERIN]["model"]

    assert committed[sp.ERIN][1].features == sp.SCORED_FEATURES
    assert document["loss_scaler"]["appincrement"]["std"] < 1e-3

    erin = _principal(result, sp.ERIN)
    takeover = erin[erin["source_city"] == sp.AMSTERDAM_PLACE[2]]

    assert takeover["max_abs_z"].astype(float).min() > 1_000
    assert _principal(result, sp.DAVE)["max_abs_z"].astype(float).max() < 1.0


# --- The trajectory --------------------------------------------------------------------------------------------


def _days(auth: pd.DataFrame) -> pd.DataFrame:
    """One row per principal per day: the trajectory as the daily sealer and the drift stage see it."""
    return (auth.sort_values(["user_principal", "day_window_id"]).drop_duplicates(["user_principal", "day_window_id"]))


@pytest.mark.cpu_mode
def test_the_trajectory_is_one_observation_per_principal_per_day(result: pd.DataFrame):
    # The scores are per event and the rule is per day, so the day has to be reduced before it is tracked. The
    # drift stage reduces each complete day to the mean of its events and stamps that on every row, which is
    # what makes the columns readable from any event of the day and identical across them.
    #
    # A day of the fortnight has no scores to reduce, so its trajectory columns are null; the trajectory starts
    # with the scored week.
    everything = _rows(result, "tc5_auth")
    auth = _scored_week(everything)

    assert everything["day_window_id"].notna().all()
    assert auth["drift_rising_windows"].notna().all()
    assert _fortnight(everything)["drift_rising_windows"].isna().all()

    within = auth.groupby(["user_principal", "day_window_id"
                           ])[["drift_velocity", "drift_rising_windows", "drift_rise_sigmas",
                               "drift_mature"]].nunique(dropna=False)

    assert (within <= 1).all().all(), "two events on one day disagreed about the day's trajectory"

    # The day's observation is the mean of its events, which is what the search's comment promises.
    days = _days(auth)
    means = auth.groupby(["user_principal", "day_window_id"])["mean_abs_z"].apply(lambda s: s.astype(float).mean())

    for (principal, day, velocity) in zip(days["user_principal"], days["day_window_id"], days["drift_velocity"]):
        prior = (principal, day - 1)

        if (prior in means.index and pd.notna(velocity)):
            assert float(velocity) == pytest.approx(round(means[(principal, day)] - means[prior], 4), abs=2e-4)


@pytest.mark.cpu_mode
def test_the_daily_windows_seal_the_way_the_hourly_ones_do(result: pd.DataFrame):
    # A second sealer behind the first, with its own prefix. Every day but the last is sealed by the watermark,
    # the last by the flush at end of stream, nothing arrives late, and the hourly columns are untouched.
    auth = _rows(result, "tc5_auth")

    assert set(auth["day_sealed_by"]) == {"watermark", "flush"}
    assert auth.loc[auth["day_sealed_by"] == "flush", "day_window_id"].nunique() == 1
    assert not auth["day_is_late"].astype(bool).any()
    assert (auth["day_window_id"].astype(int) * 24 <= auth["window_id"].astype(int)).all()
    assert (auth["window_id"].astype(int) < (auth["day_window_id"].astype(int) + 1) * 24).all()


@pytest.mark.cpu_mode
def test_the_drift_rule_fires_on_exactly_the_climbs_it_should(result: pd.DataFrame):
    # R-P-L5-006 as written: four consecutive rising days, a rise above 1.5 of the principal's own standard
    # deviations, no day crossing R-B-L5-001's mean of 2.0, and a steady rise rather than a spike. It fires on
    # the last three days of the week for the two principals whose week repeats their fortnight, and neither is
    # behaviour; each is explained here or the golden could grow another without anyone noticing.
    #
    # Their daily mean rises by hundredths a day, never as much as a twentieth. The cadence features keep moving
    # after the fortnight ends -- each ordinary sign-in makes its hour a little less surprising -- so every day
    # sits a little further from anything the model was trained on. The rule measures the rise in units of the
    # principal's own day-to-day spread, and for a principal who never changes that spread is tiny, so a rise of
    # hundredths reads as several of it. The days that do depart, Erin's, Alice's, Bob's and Carol's, all cross the mean
    # ceiling and leave, which is the ceiling doing its job: those are R-B-L5-001's to report.
    auth = _scored_week(_rows(result, "tc5_auth"))
    days = _days(auth)
    limits = stamping.rule_thresholds(["R-P-L5-006"])["R-P-L5-006"]

    trajectory = days[(days["drift_mature"] == True)  # noqa: E712  pylint: disable=singleton-comparison
                      & (days["drift_rising_windows"].astype(float) >= limits["rising_threshold"])
                      & (days["drift_rise_sigmas"].astype(float) > limits["sigma_threshold"])
                      & (days["mean_abs_z"].astype(float) < limits["mean_ceiling"])]
    fires = trajectory[trajectory["drift_acceleration"].astype(float).abs() < limits["acceleration_ceiling"]]

    last = sp.CORPUS_EPOCH_S // sp.DAY_S + sp.CORPUS_DAYS - 1
    climbs = {(principal, day) for principal in UNCHANGED for day in (last - 2, last - 1, last)}

    assert set(zip(trajectory["user_principal"], trajectory["day_window_id"].astype(int))) == climbs
    assert set(zip(fires["user_principal"], fires["day_window_id"].astype(int))) == climbs
    assert fires["drift_velocity"].astype(float).abs().max() < 0.05


def _true(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("boolean").fillna(False)


@pytest.mark.cpu_mode
def test_the_off_hours_rule_fires_on_the_planted_login_and_not_on_the_nightly_batch(result: pd.DataFrame):
    # R-D-L5-007 as the search states it. The planted 03:00 sign-in fires and the service account's 03:00 does
    # not, which is the cadence feature's own negative control turned into a rule. The traveller's sign-in before
    # his flight fires too: 08:00 is an hour his office week never used, and a real change of habit.
    auth = _rows(result, "tc5_auth")
    fires = auth[_true(auth, "hour_unseen") & _true(auth, "cadence_mature") & (auth["auth_result"] == "success")]

    # Erin's takeover afternoon fires twice, at the two of its hours her office week never used.
    assert set(zip(fires["user_principal"], fires["local_hour"].astype(int))) == {
        (sp.ALICE, sp.OFF_HOURS_HOUR),
        (sp.BOB, sp.FLIGHT_DEPART_HOUR),
        (sp.ERIN, 15),
        (sp.ERIN, 17),
    }
    assert sp.BATCH not in set(fires["user_principal"])

    # The fatigue burst is at an unseen hour too, and its rows are refusals: R-D-L5-004's and R-D-L5-009's.
    unseen = auth[_true(auth, "hour_unseen") & _true(auth, "cadence_mature")]
    assert set(unseen["user_principal"]) == {sp.ALICE, sp.BOB, sp.CAROL, sp.ERIN}


@pytest.mark.cpu_mode
def test_the_novelty_rule_fires_on_the_new_country_and_not_on_the_controls(result: pd.DataFrame):
    # R-D-L5-008. The new country fires for the principal whose journey was impossible and for the one who flew;
    # the VPN user's first concentrator sign-in is new too and is stopped by the maturity gate alone, which is
    # the control that makes the gate a condition rather than a decoration.
    auth = _rows(result, "tc5_auth")
    new = auth[(auth["auth_result"] == "success")
               & (_true(auth, "location_first_seen") | _true(auth, "device_first_seen"))]
    fires = new[_true(new, "cadence_mature")]

    assert set(zip(fires["user_principal"], fires["user_location"])) == {
        (sp.ALICE, "us:ny:new-york"),
        (sp.BOB, "us:ny:new-york"),
    }
    assert set(new["user_principal"]) - set(fires["user_principal"]) == {sp.DAVE}


@pytest.mark.cpu_mode
def test_the_novelty_rule_misses_a_new_place_whose_first_sign_in_failed(result: pd.DataFrame):
    # A gap in R-D-L5-008, found by the takeover and recorded rather than worked around. The first sign-in from
    # Amsterdam failed, and it is the row that carries `location_first_seen` and `device_first_seen`; every
    # success after it is the second sighting. The rule reads successes only, so the new place and the new device
    # never reach it -- an attacker who gets the password wrong once first is invisible to it. The model rules
    # fire on the whole afternoon.
    erin = _principal(result, sp.ERIN)
    takeover = erin[erin["source_city"] == sp.AMSTERDAM_PLACE[2]]
    first = takeover.iloc[0]

    assert first["auth_result"] == "failure"
    assert bool(first["location_first_seen"]) and bool(first["device_first_seen"])

    successes = takeover[takeover["auth_result"] == "success"]
    assert not _true(successes, "location_first_seen").any()
    assert not _true(successes, "device_first_seen").any()

    (composite, _) = _model_rules(_rows(result, "tc5_auth"))
    assert set(successes.index) <= set(composite.index)


@pytest.mark.cpu_mode
def test_the_failure_run_rule_fires_on_the_burst_and_not_on_the_fumbled_password(result: pd.DataFrame):
    # R-D-L5-009. Both are failure-then-success; only the threshold separates them.
    auth = _rows(result, "tc5_auth")
    threshold = stamping.rule_thresholds(["R-D-L5-009"])["R-D-L5-009"]["failure_threshold"]
    runs = auth[_true(auth, "auth_failed_then_succeeded")]
    fires = runs[runs["consecutive_auth_failures"].astype(float) >= threshold]

    assert list(fires["user_principal"]) == [sp.CAROL]
    assert set(runs["user_principal"]) == {sp.ALICE, sp.CAROL}
    assert int(runs[runs["user_principal"] == sp.ALICE]["consecutive_auth_failures"].iloc[0]) < threshold


@pytest.mark.cpu_mode
def test_the_model_rules_read_nothing_a_fallback_scored(result: pd.DataFrame):
    # R-B-L5-001 and R-B-L5-002 read only rows whose model was fitted to the principal. The joiner's rows are
    # scored by the population fallback and are never read -- and nothing was tuned to make her quiet: with the
    # gate removed, her population scores cross neither threshold either, which is asserted rather than assumed.
    auth = _rows(result, "tc5_auth")
    limits = stamping.rule_thresholds(["R-B-L5-001", "R-B-L5-002"])
    fallback = auth[auth["model_fallback_used"].astype("boolean").fillna(False)]

    assert set(fallback["user_principal"]) == {sp.GRACE}

    composite = fallback[(fallback["max_abs_z"].astype(float) >= limits["R-B-L5-001"]["max_threshold"])
                         & (fallback["mean_abs_z"].astype(float) >= limits["R-B-L5-001"]["mean_threshold"])]
    location = fallback[fallback["locincrement_z_loss"].astype(float) >= limits["R-B-L5-002"]["loss_threshold"]]

    assert composite.empty
    assert location.empty

    (gated_composite, gated_location) = _model_rules(auth)
    assert not set(gated_composite.index) & set(fallback.index)
    assert not set(gated_location.index) & set(fallback.index)


@pytest.mark.cpu_mode
def test_the_population_fallback_is_fitted_on_the_fortnight(result: pd.DataFrame):
    # The fallback is a model of the population and is held to the models' rule: fitted on the fortnight alone,
    # never on the week it scores. Recomputed here from the fortnight's features, so the frozen constants cannot
    # drift from what they claim to be.
    pooled = pd.concat(sp.training_frames(result).values(), ignore_index=True)

    for (name, (mean, deviation)) in sp.REFERENCE_PARAMETERS.items():
        assert round(float(pooled[name].mean()), 6) == mean, name
        assert round(float(pooled[name].std()), 6) == deviation, name


def _day(result: pd.DataFrame, principal: str, corpus_day: int) -> pd.DataFrame:
    rows = _principal(result, principal)
    days = ((rows["event_time"].astype("int64") // sp.NS_PER_SECOND - sp.CORPUS_EPOCH_S) // sp.DAY_S).astype(int)

    return rows[days == corpus_day]


@pytest.mark.cpu_mode
def test_the_leavers_sign_ins_carry_what_was_known_at_their_time(result: pd.DataFrame):
    # Dave leaves on the Saturday and the directory hears on the Sunday. A detection running on Saturday could not
    # have known, so his Saturday sign-ins say active; the enrichment answers as known at each event, and the
    # Sunday ones say terminated.
    assert set(_day(result, sp.DAVE, sp.LEAVER_DAY)["ctx_employment_status"]) == {"active"}
    assert set(_day(result, sp.DAVE, sp.LEAVER_RECORDED_DAY)["ctx_employment_status"]) == {"terminated"}
    assert set(_day(result, sp.DAVE, sp.LEAVER_DAY)["ctx_knowledge"]) == {"event"}


@pytest.mark.cpu_mode
def test_the_service_principal_says_so_on_both_layer_5_classes(result: pd.DataFrame):
    batch = result[result["user_principal"] == sp.BATCH]

    assert set(batch["telemetry_class"]) == {"tc5_auth", "tc5_session"}
    assert set(batch["ctx_account_type"]) == {"service"}
    assert set(result[result["user_principal"] == sp.ALICE]["ctx_account_type"]) == {"human"}


@pytest.mark.cpu_mode
def test_a_group_change_takes_effect_on_the_day_it_was_made(result: pd.DataFrame):
    (london, newyork) = sp.GROUP_CHANGE_GROUPS

    assert set(_day(result, sp.BOB, sp.FLIGHT_DAY - 1)["ctx_groups"]) == {london}
    assert set(_day(result, sp.BOB, sp.FLIGHT_DAY + 1)["ctx_groups"]) == {newyork}


@pytest.mark.cpu_mode
def test_the_lifecycle_is_start_or_end_on_every_paired_record(result: pd.DataFrame):
    sessions = _rows(result, "tc5_session")
    paired = sessions[sessions["session_action"].isin(["start", "end"])]
    single = sessions[sessions["session_action"] == "session"]

    assert (paired["session_lifecycle"] == paired["session_action"]).all()
    assert single["session_lifecycle"].isna().all()


@pytest.mark.cpu_mode
def test_a_single_record_session_is_timed_and_an_inverted_one_is_reported(result: pd.DataFrame):
    sessions = _rows(result, "tc5_session")
    single = sessions[sessions["session_id"] == sp.SINGLE_RECORD_SESSION]
    inverted = sessions[sessions["session_id"] == sp.INVERTED_SESSION]
    (start_hour, end_hour) = sp.SINGLE_RECORD_HOURS

    assert int(single["session_duration_s"].iloc[0]) == (end_hour - start_hour) * 3600
    assert bool(single["session_out_of_order"].iloc[0]) is False
    assert inverted["session_duration_s"].isna().all()
    assert bool(inverted["session_out_of_order"].iloc[0]) is True


def _duration_rule(result: pd.DataFrame, exclude_services: bool = True) -> pd.DataFrame:
    """R-B-L5-005 as its search states it; the account-type exclusion can be switched off."""
    limits = stamping.rule_thresholds(["R-B-L5-005"])["R-B-L5-005"]
    sessions = _rows(result, "tc5_session")
    measured = sessions[sessions["session_duration_mature"].astype("boolean").fillna(False)]
    fires = measured[measured["session_duration_ratio"].astype(float) > limits["ratio_threshold"]]

    if (exclude_services):
        fires = fires[fires["ctx_account_type"].fillna("unknown") != "service"]

    return fires


@pytest.mark.cpu_mode
def test_the_long_interactive_session_fires_and_the_bimodal_service_account_does_not(result: pd.DataFrame):
    # Carol's eleven-hour last day, against four of eight. The service account's last night is the longest it has
    # had too, and fires the moment the account-type exclusion is removed -- so the exclusion is what keeps it
    # quiet, not a percentile that happens to sit above it.
    fires = _duration_rule(result)

    assert list(fires["user_principal"]) == [sp.CAROL]
    assert float(fires["session_duration_ratio"].iloc[0]) == pytest.approx((sp.LONG_SESSION_END_HOUR - 9) / 8, abs=1e-4)
    assert set(_duration_rule(result, exclude_services=False)["user_principal"]) == {sp.CAROL, sp.BATCH}

    # The ordinary working days measure at exactly their own baseline, which is not beyond it.
    ordinary = _rows(result, "tc5_session")
    ordinary = ordinary[ordinary["user_principal"].isin([sp.ALICE, sp.BOB])
                        & ordinary["session_duration_mature"].astype("boolean").fillna(False)]
    assert (ordinary["session_duration_ratio"].astype(float) == 1.0).all()


@pytest.mark.cpu_mode
def test_the_reference_scorer_is_documented_as_not_a_model():
    # Stated in the class that produces them, because the golden file's numbers will outlive anyone's memory of
    # where they came from.
    assert "**Not a model.**" in sp.ReferenceScorer.__doc__
    assert "No detection claim attaches" in sp.ReferenceScorer.__doc__
