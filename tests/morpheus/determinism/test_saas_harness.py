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
Control 13's six checks against the layer 7 SaaS class, and both SaaS rules' predicates over its corpus.

Both rules read the TC-0 context store, and both are asserted the way the layer 7 DNS and HTTP rules are: for each
benign actor, the check names the one condition that keeps it quiet and shows the rule would have fired without it.
R-P-L7-006 adds a case no earlier rule had -- an answer that depends on *when the context was known* -- and asserts
both answers, the one the pipeline gives with what was known at the time and the one it would give with everything
recorded since.

The predicates are evaluated the way the shipped searches evaluate them. R-B-L7-002 reads one row. R-P-L7-006 reads
four whole weeks of a principal's rows at once, the way its search does when it runs on a Monday: the longest rising
run anywhere in them, and how many different role assignments they carry.
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
import saas_pipeline as sp_  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_saas_expected.csv")
DRIVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_saas_pipeline.py")
SAVEDSEARCHES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..",
                             "..",
                             "..",
                             "examples",
                             "splunk_lineage_app",
                             "TA-morpheus-lineage",
                             "default",
                             "savedsearches.conf")


@pytest.fixture(name="corpus", scope="module")
def corpus_fixture() -> dict[str, pd.DataFrame]:
    yield sp_.build_corpus()


# One composed run per execution mode, reused by every check in that mode.
_RESULTS: dict = {}


@pytest.fixture(name="pipeline_config")
def pipeline_config_fixture(execution_mode) -> Config:
    yield sp_.build_pipeline_config(execution_mode)


@pytest.fixture(name="result")
def result_fixture(pipeline_config: Config, corpus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mode = pipeline_config.execution_mode

    if (mode not in _RESULTS):
        _RESULTS[mode] = sp_.run_pipeline(pipeline_config, corpus)

    yield _RESULTS[mode]


def _bulk(result: pd.DataFrame, multiple: bool = True, history: bool = True) -> dict:
    """Principals R-B-L7-002 fires on, with the severity its search gives them.

    Either condition can be switched off, which is how the counterfactual checks ask what a control would have done
    without the one condition that stops it. Without the history condition, a principal with no baseline is read as
    having one of the largest export it has made so far -- the nearest-rank figure the stage refuses to publish.
    """
    rows = result[result["operation"] == sp_.EXPORT]
    rows = rows[~rows["saas_operation_failed"].astype(bool)]
    ratio = rows["saas_record_ratio"].astype("Float64")

    if (not history):
        prior_max = rows.sort_values("event_time").groupby("user_principal")["record_count"].transform(
            lambda counts: counts.shift(1).cummax())
        ratio = ratio.fillna(rows["record_count"] / prior_max)

    keep = rows["saas_baseline_mature"].astype(bool) if history else pd.Series(True, index=rows.index)

    if (multiple):
        keep &= (ratio > sp_.RECORD_MULTIPLE).fillna(False)

    fired = {}

    for (_, row) in rows[keep].iterrows():
        classification = row["ctx_object_data_classification"]
        classification = None if pd.isna(classification) else classification
        fired[row["user_principal"]] = sp_.SEVERITY.get(classification, sp_.UNCLASSIFIED_SEVERITY)

    return fired


def _creeping(result: pd.DataFrame, role: bool = True) -> dict:
    """Principals R-P-L7-006 watchlists, with the weeks on which its Monday run would have fired.

    For each week, the search reads that week and the three before it: the longest rising run anywhere in them, and
    the number of distinct role assignments. The role condition can be switched off.
    """
    breadth = result[result["saas_object_types_in_week"].notna()].copy()
    breadth["role"] = breadth["ctx_groups"].fillna("none")
    weeks = sorted(breadth["week_window_id"].astype(int).unique())
    fired: dict = {}

    for week in weeks:
        span = breadth[breadth["week_window_id"].astype(int).between(week - sp_.RISING_WEEKS + 1, week)]

        for (principal, rows) in span.groupby("user_principal"):
            rising = rows["drift_rising_windows"].astype("Int64").max()

            if (rising >= sp_.RISING_WEEKS and (not role or rows["role"].nunique() == 1)):
                fired.setdefault(principal, []).append(week)

    return fired


def _split(frame: pd.DataFrame, parts: int) -> list:
    size = max(1, len(frame) // parts)

    return [frame.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(frame), size)]


def _permuted(corpus: dict, seed: int) -> dict:
    period_ns = sp_.PERIOD_SECONDS * sp_.NS_PER_SECOND

    return {
        name:
            permute_within_contiguous_groups(frame,
                                             [window_id_from_timestamp(int(t), period_ns) for t in frame["event_time"]],
                                             seed=seed)
        for (name, frame) in corpus.items()
    }


# --- Check 1: the corpus is fixed and shaped like a SaaS audit log ---------------------------------------------


def test_corpus_is_fixed(corpus: dict[str, pd.DataFrame]):
    again = sp_.build_corpus()

    assert set(again) == set(corpus) == {sp_.SAAS_CLASS}
    pd.testing.assert_frame_equal(corpus[sp_.SAAS_CLASS], again[sp_.SAAS_CLASS])


def test_corpus_is_shaped_like_a_saas_audit_log(corpus: dict[str, pd.DataFrame]):
    frame = corpus[sp_.SAAS_CLASS]

    # The guide's required SaaS fields, all present.
    for column in ("operation", "target_object", "target_object_type", "record_count", "result", "client_app"):
        assert column in frame.columns, column

    assert set(sp_.ID_COLUMNS) <= set(frame.columns)
    assert frame["collector_seq"].is_monotonic_increasing
    assert frame["event_time"].is_monotonic_increasing
    # Nothing precedes the context store's first snapshot, so every operation could have seen it.
    assert frame["event_time"].min() > sp_.at(sp_.CONTEXT_RECORDED_DAY)


# --- Checks 2 through 6: determinism ---------------------------------------------------------------------------


@pytest.mark.gpu_and_cpu_mode
def test_double_run_diff(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    second = sp_.run_pipeline(pipeline_config, corpus)

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

    rendered = sp_.render(result)

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
    by_row = {name: _split(frame, len(frame)) for (name, frame) in corpus.items()}

    assert diff_frames(result, sp_.run_pipeline(pipeline_config, corpus, batches=thirds)) is None
    assert diff_frames(result, sp_.run_pipeline(pipeline_config, corpus, batches=by_row)) is None


@pytest.mark.gpu_and_cpu_mode
def test_permutation_within_windows(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    for seed in (1, 2, 3):
        assert diff_frames(result, sp_.run_pipeline(pipeline_config, _permuted(corpus, seed))) is None, seed


@pytest.mark.gpu_and_cpu_mode
def test_permutation_check_has_teeth(pipeline_config: Config, corpus: dict):
    # The negative control for the check above. The weekly tally is a running count, so without the imposed order
    # the count a row carries depends on which rows of its week arrived first.
    ordered = sp_.run_pipeline(pipeline_config, corpus, impose_order=False)
    shuffled = sp_.run_pipeline(pipeline_config, _permuted(corpus, 5), impose_order=False)

    assert diff_frames(ordered, shuffled) is not None


# --- R-B-L7-002: the bulk exports, and one control per condition -----------------------------------------------


@pytest.mark.cpu_mode
def test_the_bulk_rule_fires_on_exactly_the_three_it_should_at_the_severity_classification_gives(result: pd.DataFrame):
    # The classification is a weight on the notable, not a bar the export must clear: the same multiple from a
    # public object fires, at a quarter of a restricted object's severity, and an object the inventory has never
    # heard of fires at the default rather than being dropped.
    assert _bulk(result) == {
        sp_.EXFILTRATOR: sp_.SEVERITY["restricted"],
        sp_.PUBLIC_EXPORTER: sp_.SEVERITY["public"],
        sp_.UNKNOWN_EXPORTER: sp_.UNCLASSIFIED_SEVERITY,
    }


@pytest.mark.cpu_mode
def test_the_baseline_is_the_operations_and_not_the_principals(result: pd.DataFrame):
    # The exfiltrator queries two to three thousand records at a time all month. Measured against everything they
    # do, the five-thousand-record export is under twice their normal and the rule is quiet; measured against their
    # exports, it is eighty times.
    rows = result[result["user_principal"] == sp_.EXFILTRATOR].sort_values("event_time")
    export = rows[(rows["operation"] == sp_.EXPORT) & (rows["target_object"] == sp_.RESTRICTED_OBJECT)]
    prior = rows[rows["event_time"] < export["event_time"].iloc[0]]["record_count"].astype(float).sort_values()
    pooled = prior.iloc[-(-99 * len(prior) // 100) - 1]

    assert float(export["saas_record_ratio"].iloc[0]) > 50
    assert 5000 / pooled < 2


@pytest.mark.cpu_mode
def test_the_unclassified_object_is_flagged_rather_than_guessed(result: pd.DataFrame):
    rows = result[(result["user_principal"] == sp_.UNKNOWN_EXPORTER) & (result["target_object"] == sp_.UNKNOWN_OBJECT)]

    assert len(rows) == 1
    assert not bool(rows["ctx_object_found"].iloc[0])
    assert pd.isna(rows["ctx_object_data_classification"].iloc[0])


@pytest.mark.cpu_mode
def test_routine_and_near_exports_are_stopped_by_the_multiple_alone(result: pd.DataFrame):
    # Both have a baseline and both exported thousands of records; what they lack is five times their own normal.
    # Take the multiple away and both fire.
    assert sp_.ROUTINE_EXPORTER not in _bulk(result)
    assert sp_.NEAR_EXPORTER not in _bulk(result)
    assert {sp_.ROUTINE_EXPORTER, sp_.NEAR_EXPORTER} <= set(_bulk(result, multiple=False))

    near = result[(result["user_principal"] == sp_.NEAR_EXPORTER) & (result["target_object"] == sp_.RESTRICTED_OBJECT)]

    assert 4.5 < float(near["saas_record_ratio"].iloc[0]) < sp_.RECORD_MULTIPLE


@pytest.mark.cpu_mode
def test_the_newcomer_is_stopped_by_the_missing_history_alone(result: pd.DataFrame):
    # Twenty exports is not a baseline: by nearest rank their 99th percentile is simply the largest of them.
    # Read that way the newcomer's export is far past five times, which is why the stage publishes nothing.
    assert sp_.NEWCOMER not in _bulk(result)
    assert sp_.NEWCOMER in _bulk(result, history=False)

    newcomer = result[(result["user_principal"] == sp_.NEWCOMER) & (result["operation"] == sp_.EXPORT)]

    assert not newcomer["saas_baseline_mature"].astype(bool).any()


@pytest.mark.cpu_mode
def test_a_failed_export_read_nothing_and_is_not_measured(result: pd.DataFrame):
    denied = result[(result["user_principal"] == sp_.FAILED_EXPORTER) & (result["result"] == "denied")]

    assert len(denied) == 1
    assert bool(denied["saas_operation_failed"].iloc[0])
    assert pd.isna(denied["saas_record_ratio"].iloc[0])
    assert sp_.FAILED_EXPORTER not in _bulk(result)


# --- R-P-L7-006: the rising breadth, and one control per condition ---------------------------------------------


@pytest.mark.cpu_mode
def test_the_breadth_rule_watchlists_exactly_the_creeper_and_the_late_role_change(result: pd.DataFrame):
    fired = _creeping(result)

    assert set(fired) == {sp_.CREEPER, sp_.LATE_ROLE_CHANGER}
    # Four rising weeks is weeks 3 to 6; the rise continues into week 7.
    assert fired[sp_.CREEPER] == [6, 7]


@pytest.mark.cpu_mode
def test_a_week_is_measured_by_its_total_and_not_by_its_running_count(result: pd.DataFrame):
    # The stage writes a running count within the week, so the trajectory has to reduce a week to its last value,
    # the maximum. The creeper reached two types in week 3 and six in week 7; a mean of running counts would put
    # both lower by different amounts and report a different rise.
    creeper = result[(result["user_principal"] == sp_.CREEPER) & (result["week_window_id"].astype(int) == 7)]

    assert set(creeper["drift_total_rise"].astype(float)) == {4.0}
    assert set(creeper["drift_velocity"].astype(float)) == {1.0}


@pytest.mark.cpu_mode
def test_the_role_change_is_stopped_by_the_role_condition_alone(result: pd.DataFrame):
    # The same rise as the creeper's, week for week. New duties explain new objects, which is the reason the rule
    # asks about the role at all.
    assert sp_.ROLE_CHANGER not in _creeping(result)
    assert sp_.ROLE_CHANGER in _creeping(result, role=False)


@pytest.mark.cpu_mode
def test_a_short_rise_and_a_flat_broad_user_are_quiet_with_the_role_condition_removed(result: pd.DataFrame):
    # Neither is stopped by the role, so removing that condition does not change them: the short rise lacks a
    # fourth week and the broad user has breadth and no rise.
    for principal in (sp_.SHORT_RISE, sp_.EVERYTHING):
        assert principal not in _creeping(result)
        assert principal not in _creeping(result, role=False)


@pytest.mark.cpu_mode
def test_a_role_change_recorded_late_is_watchlisted_on_what_was_known_and_would_not_be_on_what_is_known_now(
        pipeline_config: Config, corpus: dict, result: pd.DataFrame):
    # Kate's role changed in week 5 and the directory heard in week 9. Every week of the rise, the store said the
    # role was unchanged, so the rule -- reading the event-time view the enrichment defaults to -- watchlists Kate.
    # With everything recorded since, the rise straddles a role change and the rule would have stayed quiet.
    assert sp_.LATE_ROLE_CHANGER in _creeping(result)

    latest = sp_.run_pipeline(pipeline_config, corpus, knowledge="latest")

    assert sp_.LATE_ROLE_CHANGER not in _creeping(latest)
    assert sp_.CREEPER in _creeping(latest)
    assert _bulk(latest) == _bulk(result)


# --- Minimization, and the searches as shipped -----------------------------------------------------------------


def _search(name: str) -> str:
    with open(SAVEDSEARCHES, encoding="utf-8") as handle:
        text = handle.read()

    return text.split(f"[{name}]", 1)[1].split("action.correlationsearch.label", 1)[0].split("search =", 1)[1]


def test_neither_saas_rule_reads_the_object_name():
    # `target_object` is the name of what was read -- a file, a site, a record -- and routinely personal. Neither
    # rule needs it: the classification it would be looked up by is attached upstream, so an estate can drop or
    # pseudonymize the name at the wire and keep both rules.
    for name in ("R-B-L7-002 - Bulk data access", "R-P-L7-006 - Access breadth trajectory"):
        search = _search(name)

        assert "target_object " not in search and not search.rstrip().endswith("target_object"), name
        assert "target_object," not in search, name


def test_the_searches_carry_the_thresholds_this_harness_asserts():
    bulk = _search("R-B-L7-002 - Bulk data access")
    breadth = _search("R-P-L7-006 - Access breadth trajectory")

    assert f"saas_record_ratio>{int(sp_.RECORD_MULTIPLE)}" in bulk
    assert "saas_baseline_mature=true" in bulk

    for (classification, severity) in sp_.SEVERITY.items():
        assert f'"{classification}", {severity}' in bulk

    assert f"true(), {sp_.UNCLASSIFIED_SEVERITY}" in bulk
    assert f"rising_weeks >= {sp_.RISING_WEEKS}" in breadth
    assert "role_versions = 1" in breadth


# --- Stamping --------------------------------------------------------------------------------------------------


@pytest.mark.cpu_mode
def test_every_row_is_stamped_at_layer_seven_and_keyed_on_its_principal(result: pd.DataFrame):
    assert set(result["osi_layer"]) == {sp_.OSI_LAYER}
    assert list(result["entity_key"]) == list(result["user_principal"])
    assert result["lineage_id"].notna().all()
    assert result["week_window_id"].notna().all()
    assert (result["week_window_id"].astype(int) == result["saas_week_id"].astype(int)).all()
