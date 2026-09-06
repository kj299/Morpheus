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
from morpheus.utils.lineage import window_id_from_timestamp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import session_pipeline as sp  # noqa: E402

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


def _windows(frame: pd.DataFrame) -> list[int]:
    return [window_id_from_timestamp(int(t), sp.PERIOD_SECONDS * NS) for t in frame["event_time"]]


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

    # Sessions arrive as separate start and stop records with no duration on either, which is the shape
    # TC5SessionStage exists for.
    sessions = corpus["tc5_session"]
    assert set(sessions["session_action"]) == {"start", "end"}
    assert "session_duration_s" not in sessions.columns

    # Every class carries the envelope the identifiers derive from, with a monotonic sequence.
    for frame in corpus.values():
        assert set(sp.ID_COLUMNS) <= set(frame.columns)
        assert frame["collector_seq"].is_monotonic_increasing


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

    def split(frame: pd.DataFrame, parts: int) -> list[pd.DataFrame]:
        size = max(1, len(frame) // parts)
        return [frame.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(frame), size)]

    thirds = {name: split(frame, 3) for (name, frame) in corpus.items()}
    by_row = {name: split(frame, len(frame)) for (name, frame) in corpus.items()}

    assert diff_frames(result, sp.run_pipeline(pipeline_config, corpus, batches=thirds)) is None
    assert diff_frames(result, sp.run_pipeline(pipeline_config, corpus, batches=by_row)) is None


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
