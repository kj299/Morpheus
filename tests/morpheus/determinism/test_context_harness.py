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
Control 13's six checks against the TC-0 context store, and the bitemporal answers the store exists to give.

Every case in the corpus is asserted as a pair: what the context said as known at the event's time, and what it says
with everything recorded since. A store with one time axis would give one answer for both, and for most of the corpus
that is the right answer -- which is why the checks name, for each case, whether the two views must agree or must
differ, rather than asserting only the cases where they differ.

The permutation check has a different shape here from the telemetry harnesses'. The producers are stateless, so
permuting their input cannot change their output; what can change is which version the store picks when several cover
an instant, if the choice depended on the order versions arrived in. The negative control is therefore a resolver that
lets the last version added win, which is the rule a store gets by default if nobody writes a different one, shown to
change its answer when the same versions arrive in a different order.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.utils import bitemporal
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.determinism import diff_frames
from morpheus.utils.determinism import frame_digest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import context_pipeline as cp  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_context_expected.csv")
DRIVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_context_pipeline.py")


@pytest.fixture(name="corpus", scope="module")
def corpus_fixture() -> dict[str, pd.DataFrame]:
    yield cp.build_corpus()


# One composed run per execution mode, reused by every check in that mode.
_RESULTS: dict = {}


@pytest.fixture(name="pipeline_config")
def pipeline_config_fixture(execution_mode) -> Config:
    yield cp.build_pipeline_config(execution_mode)


@pytest.fixture(name="result")
def result_fixture(pipeline_config: Config, corpus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mode = pipeline_config.execution_mode

    if (mode not in _RESULTS):
        _RESULTS[mode] = cp.run_pipeline(pipeline_config, corpus)

    yield _RESULTS[mode]


def _probe(result: pd.DataFrame, probe_id: str, knowledge: str) -> pd.Series:
    rows = result[result["row_key"] == f"{probe_id}@{knowledge}"]

    assert len(rows) == 1, f"{probe_id}@{knowledge}"

    return rows.iloc[0]


def _pair(result: pd.DataFrame, probe_id: str, column: str) -> tuple:
    """The value of one context column as known at the event, and as known now."""

    def value(knowledge):
        cell = _probe(result, probe_id, knowledge)[f"ctx_{column}"]

        return None if pd.isna(cell) else cell

    return (value("event"), value("latest"))


def _producer(result: pd.DataFrame, name: str) -> pd.DataFrame:
    return result[result["telemetry_class"] == name]


def _split(frame: pd.DataFrame, parts: int) -> list:
    size = max(1, len(frame) // parts)

    return [frame.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(frame), size)]


def _permuted(corpus: dict, seed: int) -> dict:
    return {name: frame.sample(frac=1, random_state=seed).reset_index(drop=True) for (name, frame) in corpus.items()}


# --- Check 1: the corpus is fixed and shaped like an HR export and an inventory ----------------------------------


def test_corpus_is_fixed(corpus: dict[str, pd.DataFrame]):
    again = cp.build_corpus()

    assert set(again) == set(corpus) == set(cp.CORPUS_CLASSES)

    for name in corpus:
        pd.testing.assert_frame_equal(corpus[name], again[name])


def test_corpus_is_shaped_like_an_hr_export_and_an_inventory(corpus: dict[str, pd.DataFrame]):
    for column in ("user_principal", "group_name", "department", "manager", "employment_status"):
        assert column in corpus[cp.IDENTITY_CLASS].columns, column

    for column in ("hostname", "owner", "owning_team", "criticality", "data_classification", "peer_group"):
        assert column in corpus[cp.ASSET_CLASS].columns, column

    for name in cp.PRODUCER_CLASSES:
        frame = corpus[name]

        for column in (bitemporal.VALID_FROM, bitemporal.VALID_TO, bitemporal.RECORDED_AT, bitemporal.CHANGE):
            assert column in frame.columns, (name, column)

        # The source's own log order: by when it recorded each record, the unrecorded one last.
        recorded = frame[bitemporal.RECORDED_AT].dropna()
        assert recorded.is_monotonic_increasing, name

    for name in cp.PROBE_CLASSES:
        assert corpus[name]["event_time"].is_monotonic_increasing, name


# --- Checks 2 through 6: determinism ---------------------------------------------------------------------------


@pytest.mark.gpu_and_cpu_mode
def test_double_run_diff(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    second = cp.run_pipeline(pipeline_config, corpus)

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

    rendered = cp.render(result)

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

    assert diff_frames(result, cp.run_pipeline(pipeline_config, corpus, batches=thirds)) is None
    assert diff_frames(result, cp.run_pipeline(pipeline_config, corpus, batches=by_row)) is None


@pytest.mark.gpu_and_cpu_mode
def test_permutation(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    # The whole source log shuffled, not only within windows: context has no windows, and a snapshot diff or a
    # correction arriving ahead of what it corrects is exactly what a replay from a lagging export produces.
    for seed in (1, 2, 3):
        assert diff_frames(result, cp.run_pipeline(pipeline_config, _permuted(corpus, seed))) is None, seed


def _last_added_wins(versions: list, key: str, valid_ns: int):
    """The rule a store gets by default: whatever covering version arrived last."""
    answer = None

    for version in versions:
        if (version.key == key and version.covers(valid_ns)):
            answer = version

    return answer


def test_permutation_check_has_teeth():
    # The negative control. Carol's correction and the record it corrects both cover day five. The store picks the
    # correction whichever arrives first; a last-added-wins resolver picks whichever arrived second, so the same
    # versions in a different order give a different department.
    log = cp.identity_log()
    day5 = cp.at(5, 12)

    assert _last_added_wins(log, cp.CAROL, day5).attributes["department"] == "Marketing"
    assert _last_added_wins(list(reversed(log)), cp.CAROL, day5).attributes["department"] == "Sales"

    for order in (log, list(reversed(log))):
        assert BitemporalStore("order", order).resolve(cp.CAROL, day5).attributes["department"] == "Marketing"


# --- The cases: where the two views must agree, and where they must not ------------------------------------------


@pytest.mark.cpu_mode
def test_a_move_recorded_on_time_reads_the_same_either_way(result: pd.DataFrame):
    # The control for every case below. The source heard the same morning, so there is nothing for the views to
    # disagree about, before or after -- and before and after differ from each other, so the agreement is not two
    # blanks.
    assert _pair(result, "bob-before-move", "department") == ("Engineering", "Engineering")
    assert _pair(result, "bob-after-move", "department") == ("Finance", "Finance")
    assert _pair(result, "bob-before-move", "groups") == ("eng-users", "eng-users")
    assert _pair(result, "bob-after-move", "groups") == ("finance-users", "finance-users")


@pytest.mark.cpu_mode
def test_a_retroactive_correction_splits_the_two_views(result: pd.DataFrame):
    # "Was Carol in Sales on day five" is yes as it was understood on day five and no as it is understood now. A
    # detection that fired on day five saw the first; an investigation reading it today needs both.
    assert _pair(result, "carol-before-correction", "department") == ("Sales", "Marketing")
    assert _pair(result, "carol-after-correction", "department") == ("Marketing", "Marketing")


@pytest.mark.cpu_mode
def test_a_leaver_recorded_late_was_active_as_far_as_anyone_knew(result: pd.DataFrame):
    # The six days between a termination and the source hearing of it. Anything Dave did then was done, as far as
    # any detection could have known, by an active employee in finance-users.
    assert _pair(result, "dave-unreported-leaver", "employment_status") == ("active", "terminated")
    assert _pair(result, "dave-unreported-leaver", "groups") == ("finance-users", None)
    assert _pair(result, "dave-reported-leaver", "employment_status") == ("terminated", "terminated")
    assert _pair(result, "dave-reported-leaver", "groups") == (None, None)


@pytest.mark.cpu_mode
def test_a_snapshot_absence_retracts_from_the_snapshot_onward(result: pd.DataFrame):
    # Erin is absent from the day-fifteen snapshot. The retraction is valid from the snapshot, not from some
    # guessed earlier date the export never stated, so the day before still reads as she was.
    assert _pair(result, "erin-before-snapshot", "employment_status") == ("contractor", "contractor")
    assert _pair(result, "erin-after-snapshot", "employment_status") == (None, None)
    assert not _probe(result, "erin-after-snapshot", "latest")["ctx_found"]


@pytest.mark.cpu_mode
def test_a_reclassification_recorded_late_splits_the_two_views(result: pd.DataFrame):
    # What R-B-L7-002 will weight by. An alert on day six weighted the read as confidential; the investigation
    # reads restricted.
    assert _pair(result, "ledger-before-reclassification-recorded",
                 "data_classification") == ("confidential", "restricted")
    assert _pair(result, "ledger-after-reclassification-recorded",
                 "data_classification") == ("restricted", "restricted")


@pytest.mark.cpu_mode
def test_a_peer_group_change_recorded_late_splits_the_two_views(result: pd.DataFrame):
    # What R-B-L7-004 will compare against. On the day of the move the inventory still had the workstation among
    # the engineering machines.
    assert _pair(result, "bob-workstation-unnoticed-move", "peer_group") == ("eng-workstations", "finance-workstations")


@pytest.mark.cpu_mode
def test_a_decommissioned_host_is_not_found(result: pd.DataFrame):
    assert _pair(result, "builder-after-decommission", "criticality") == (None, None)
    assert not _probe(result, "builder-after-decommission", "event")["ctx_found"]


@pytest.mark.cpu_mode
def test_an_unknown_principal_is_not_found_rather_than_blank(result: pd.DataFrame):
    # Mallory has never been recorded. `ctx_found` is what separates that from a principal who is known and has
    # nothing to say, which a column of nulls alone cannot.
    for knowledge in cp.KNOWLEDGE_MODES:
        assert not _probe(result, "mallory-unknown", knowledge)["ctx_found"]
        assert _probe(result, "alice-day3", knowledge)["ctx_found"]

    assert _pair(result, "alice-day3",
                 "groups") == ("expense-approvers|finance-users", "expense-approvers|finance-users")


# --- The snapshot diff -----------------------------------------------------------------------------------------


def _recorded_at(log: list, recorded_ns: int) -> list:
    return sorted((version.kind, version.key, version.change) for version in log if version.recorded_ns == recorded_ns)


def test_the_second_snapshot_records_only_the_absences():
    # A full export restates everything, and all but two principals' worth of it is unchanged. Recording the
    # restatement would bury every real change under a daily copy of the directory; recording nothing for the
    # absences would leave the departed contractor current forever.
    snapshot = cp.at(cp.SNAPSHOT_DAY, cp.SNAPSHOT_HOUR)

    assert _recorded_at(cp.identity_log(), snapshot) == [
        (bitemporal.MEMBERSHIP, f"{cp.ERIN}:eng-users", bitemporal.RETRACT),
        ("profile", cp.ERIN, bitemporal.RETRACT),
    ]
    assert _recorded_at(cp.asset_log(), snapshot) == [("asset", cp.BUILDER, bitemporal.RETRACT)]


def test_the_first_snapshot_asserts_everything_through_the_same_diff():
    day0 = cp.at(0, cp.SNAPSHOT_HOUR)

    assert len(_recorded_at(cp.identity_log(), day0)) == 11
    assert len(_recorded_at(cp.asset_log(), day0)) == 4
    assert all(change == bitemporal.ASSERT for (_, _, change) in _recorded_at(cp.identity_log(), day0))


# --- Refusal, knowledge horizons, and provenance ---------------------------------------------------------------


@pytest.mark.cpu_mode
def test_records_the_store_cannot_trust_are_refused_and_kept_visible(result: pd.DataFrame):
    identity = _producer(result, cp.IDENTITY_CLASS)
    refused = identity[identity[bitemporal.CONTEXT_REFUSED].notna()]

    assert sorted(refused[bitemporal.CONTEXT_REFUSED]) == [bitemporal.INVERTED_INTERVAL, bitemporal.NO_RECORDED_AT]
    assert refused[bitemporal.CONTEXT_UID].isna().all()

    store = cp.build_store("identity", identity)

    assert store.facts_about(cp.FRANK, cp.at(1)) == []
    assert not store.history(f"{cp.GRACE}:eng-users")


@pytest.mark.cpu_mode
def test_event_knowledge_never_rests_on_a_version_recorded_after_the_event(result: pd.DataFrame):
    probes = result[result["telemetry_class"].isin(cp.PROBE_CLASSES) & (result["ctx_knowledge"] == "event")]
    found = probes[probes["ctx_found"].astype(bool)]

    assert len(found) > 0
    assert (found["ctx_recorded_at"].astype("int64") <= found["event_time"].astype("int64")).all()


def test_event_knowledge_does_not_change_as_the_store_grows():
    # The property that makes the default reproducible. Asked with the event-time horizon, the full store gives
    # exactly what a store holding only what had been recorded by then would give -- so recording a correction
    # tomorrow cannot change what an enrichment of today's events said.
    full = BitemporalStore("full", cp.identity_log() + cp.asset_log())

    for (_, entity, event_time) in cp.IDENTITY_PROBE_EVENTS + cp.ASSET_PROBE_EVENTS:
        then = BitemporalStore(
            "then", [version for version in cp.identity_log() + cp.asset_log() if version.recorded_ns <= event_time])

        assert full.facts_about(entity, event_time, known_ns=event_time) == then.facts_about(entity, event_time), entity


@pytest.mark.cpu_mode
def test_the_store_rebuilt_from_the_wire_is_the_one_that_was_recorded(result: pd.DataFrame):
    # The producers' output is what a consumer of the sourcetype rebuilds a store from. Every version survives the
    # trip with its identifier intact, and nothing is added.
    produced = set()

    for name in cp.PRODUCER_CLASSES:
        produced |= set(_producer(result, name)[bitemporal.CONTEXT_UID].dropna())

    assert produced == {version.uid for version in cp.identity_log() + cp.asset_log()}


@pytest.mark.cpu_mode
def test_every_attached_value_traces_to_a_recorded_version(result: pd.DataFrame):
    produced = set()

    for name in cp.PRODUCER_CLASSES:
        produced |= set(_producer(result, name)[bitemporal.CONTEXT_UID].dropna())

    probes = result[result["telemetry_class"].isin(cp.PROBE_CLASSES)]
    cited = {uid for cell in probes["ctx_version_uids"].dropna() for uid in cell.split("|")}

    assert len(cited) > 0
    assert cited <= produced


def test_the_corpus_never_contradicts_itself():
    assert BitemporalStore("identity", cp.identity_log()).contradictions() == 0
    assert BitemporalStore("asset", cp.asset_log()).contradictions() == 0
