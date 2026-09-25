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
Control 13's six checks against the layer 7 endpoint class, and R-B-L7-004's predicate over its corpus.

The rule is asserted the way the other layer 7 rules are: for each quiet process on the decisive day, the check names
the one condition that keeps it quiet and shows the rule would have fired without it. The predicate is evaluated the
way the shipped search evaluates it, one row at a time, from the columns the stage writes.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.stages.telemetry.tc7_endpoint_stage import normalize_image_path
from morpheus.utils.determinism import diff_frames
from morpheus.utils.determinism import frame_digest
from morpheus.utils.determinism import permute_within_contiguous_groups
from morpheus.utils.lineage import window_id_from_timestamp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import endpoint_pipeline as ep_  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_endpoint_expected.csv")
DRIVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_endpoint_pipeline.py")
SAVEDSEARCHES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..",
                             "..",
                             "..",
                             "examples",
                             "splunk_lineage_app",
                             "TA-morpheus-lineage",
                             "default",
                             "savedsearches.conf")
RULE = "R-B-L7-004 - Process ancestry novelty"


@pytest.fixture(name="corpus", scope="module")
def corpus_fixture() -> dict[str, pd.DataFrame]:
    yield ep_.build_corpus()


# One composed run per execution mode, reused by every check in that mode.
_RESULTS: dict = {}


@pytest.fixture(name="pipeline_config")
def pipeline_config_fixture(execution_mode) -> Config:
    yield ep_.build_pipeline_config(execution_mode)


@pytest.fixture(name="result")
def result_fixture(pipeline_config: Config, corpus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mode = pipeline_config.execution_mode

    if (mode not in _RESULTS):
        _RESULTS[mode] = ep_.run_pipeline(pipeline_config, corpus)

    yield _RESULTS[mode]


def _flag(series: pd.Series) -> pd.Series:
    """A nullable flag column as plain booleans, with no answer reading as False."""
    return series.astype("boolean").fillna(False).astype(bool)


def _raw_peer_seen(result: pd.DataFrame) -> pd.Series:
    """Whether the pair, compared as the EDR reported it, was seen in the peer group in the window before each row."""
    rows = result.sort_values("event_time")
    seen = pd.Series(False, index=rows.index)
    last: dict = {}
    window_ns = ep_.WINDOW_DAYS * ep_.DAY_SECONDS * ep_.NS_PER_SECOND

    for (index, row) in rows.iterrows():
        group = row["ctx_peer_group"]

        if (pd.isna(group)):
            continue

        key = (group, row["parent_image_path"], row["image_path"])
        when = int(row["event_time"])
        previous = last.get(key)
        seen[index] = previous is not None and when - window_ns <= previous < when

        if (previous is None or previous < when):
            last[key] = when

    return seen.reindex(result.index)


def _conditions(result: pd.DataFrame,
                peer: bool = True,
                host_history: bool = True,
                warmup: bool = True,
                normalize: bool = True) -> pd.Series:
    """The rule's conditions taken apart, from the columns the stage writes, with any of them switched off."""
    rows = result[result["endpoint_pair"].notna()]
    host_seen = _flag(rows["endpoint_host_seen"])
    peer_seen = _flag(rows["endpoint_peer_seen"]) if normalize else _raw_peer_seen(result).loc[rows.index]

    keep = pd.Series(True, index=rows.index)

    if (warmup):
        keep &= _flag(rows["endpoint_mature"])

    if (host_history):
        keep &= ~host_seen

    if (peer):
        keep &= ~peer_seen

    return keep


def _ancestry(result: pd.DataFrame, **switches) -> dict:
    """(host, image) pairs R-B-L7-004 fires on, with the severity its search gives them.

    With every condition in place this reads `endpoint_pair_novel`, which is the column the search reads. Switching a
    condition off -- see `_conditions` -- is how the counterfactual checks ask what a control would have done without
    the one condition that stops it.
    """
    rows = result[result["endpoint_pair"].notna()]
    keep = _conditions(result, **switches) if switches else _flag(rows["endpoint_pair_novel"])
    fired = {}

    for (_, row) in rows[keep].iterrows():
        level = row["endpoint_integrity"]
        fired[(row["hostname"], row["image_path"])] = ep_.SEVERITY.get(None if pd.isna(level) else level,
                                                                       ep_.UNWEIGHTED_SEVERITY)

    return fired


def _decisive(result: pd.DataFrame, host: str, image: str) -> pd.Series:
    rows = result[(result["hostname"] == host) & (result["image_path"] == image)
                  & (result["event_time"].astype("int64") >= ep_.at(ep_.LAST_DAY, 10))]

    assert len(rows) == 1, (host, image)

    return rows.iloc[0]


def _split(frame: pd.DataFrame, parts: int) -> list:
    size = max(1, len(frame) // parts)

    return [frame.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(frame), size)]


def _permuted(corpus: dict, seed: int) -> dict:
    period_ns = ep_.PERIOD_SECONDS * ep_.NS_PER_SECOND

    return {
        name:
            permute_within_contiguous_groups(frame,
                                             [window_id_from_timestamp(int(t), period_ns) for t in frame["event_time"]],
                                             seed=seed)
        for (name, frame) in corpus.items()
    }


# --- Check 1: the corpus is fixed and shaped like EDR process telemetry --------------------------------------------


def test_corpus_is_fixed(corpus: dict[str, pd.DataFrame]):
    again = ep_.build_corpus()

    assert set(again) == set(corpus) == {ep_.ENDPOINT_CLASS}
    pd.testing.assert_frame_equal(corpus[ep_.ENDPOINT_CLASS], again[ep_.ENDPOINT_CLASS])


def test_corpus_is_shaped_like_edr_process_telemetry(corpus: dict[str, pd.DataFrame]):
    frame = corpus[ep_.ENDPOINT_CLASS]

    # The guide's required endpoint fields, and the two the rule itself names.
    for column in ("process_guid",
                   "parent_process_guid",
                   "image_path",
                   "command_line_hash",
                   "integrity_level",
                   "signature_status",
                   "parent_image_path",
                   "hostname"):
        assert column in frame.columns, column

    assert set(ep_.ID_COLUMNS) <= set(frame.columns)
    assert frame["collector_seq"].is_monotonic_increasing
    assert frame["event_time"].is_monotonic_increasing
    # Every host's day starts with processes that share a second, as an EDR log does at logon.
    assert frame.groupby(["hostname", "event_time"]).size().max() >= 3


# --- Checks 2 through 6: determinism -------------------------------------------------------------------------------


@pytest.mark.gpu_and_cpu_mode
def test_double_run_diff(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    second = ep_.run_pipeline(pipeline_config, corpus)

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

    rendered = ep_.render(result)

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

    assert diff_frames(result, ep_.run_pipeline(pipeline_config, corpus, batches=thirds)) is None
    assert diff_frames(result, ep_.run_pipeline(pipeline_config, corpus, batches=by_row)) is None


@pytest.mark.gpu_and_cpu_mode
def test_permutation_within_windows(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    for seed in (1, 2, 3):
        assert diff_frames(result, ep_.run_pipeline(pipeline_config, _permuted(corpus, seed))) is None, seed


@pytest.mark.gpu_and_cpu_mode
def test_permutation_check_has_teeth(pipeline_config: Config, corpus: dict):
    # The negative control for the check above. Without the imposed order, a process from later in the hour can
    # arrive first, and the one it overtook is refused as out of order.
    ordered = ep_.run_pipeline(pipeline_config, corpus, impose_order=False)
    shuffled = ep_.run_pipeline(pipeline_config, _permuted(corpus, 5), impose_order=False)

    assert diff_frames(ordered, shuffled) is not None


# --- R-B-L7-004: the decisive day, and one control per condition ---------------------------------------------------


@pytest.mark.cpu_mode
def test_the_rule_fires_on_exactly_the_four_it_should_at_the_severity_integrity_gives(result: pd.DataFrame):
    # Across all forty-one days, not only the last: nothing in the history is novel once its warm-up has passed.
    assert _ancestry(result) == {
        (ep_.NOVEL_HIGH, ep_.POWERSHELL): ep_.SEVERITY["high"],
        (ep_.STALE, ep_.TEAMVIEWER): ep_.SEVERITY["system"],
        (ep_.HOST_ONLY_NOVEL, ep_.MSHTA): ep_.SEVERITY["medium"],
        (ep_.UNWEIGHTED, ep_.CURL): ep_.UNWEIGHTED_SEVERITY,
    }


@pytest.mark.cpu_mode
def test_the_novelty_column_is_the_conjunction_the_counterfactuals_take_apart(result: pd.DataFrame):
    # The counterfactuals below switch conditions off in `_conditions`; they mean something only if the conditions,
    # all in place, are exactly what the stage decided.
    rows = result[result["endpoint_pair"].notna()]

    assert (_conditions(result) == _flag(rows["endpoint_pair_novel"])).all()


@pytest.mark.cpu_mode
def test_a_pairing_routine_in_another_group_is_no_excuse(result: pd.DataFrame):
    # The build servers render documentation through Word and PowerShell every day. The finance workstation doing
    # the same is still novel, because a peer group is the comparison and another group is not a peer.
    build = result[(result["hostname"] == "build-01") & (result["image_path"] == ep_.POWERSHELL)]
    novel = _decisive(result, ep_.NOVEL_HIGH, ep_.POWERSHELL)

    assert len(build) == ep_.LAST_DAY + 1
    assert novel["endpoint_peer_group"] == ep_.FINANCE_GROUP
    assert bool(novel["endpoint_pair_novel"])


@pytest.mark.cpu_mode
def test_the_compile_is_stopped_by_the_peer_group_alone(result: pd.DataFrame):
    # New to the host, run by its peer every day. Judged per host, as the guide warns against, it fires.
    compile_row = _decisive(result, ep_.PEER_SEEN, ep_.COMPILER)

    assert not bool(compile_row["endpoint_host_seen"])
    assert bool(compile_row["endpoint_peer_seen"])
    assert (ep_.PEER_SEEN, ep_.COMPILER) not in _ancestry(result)
    assert (ep_.PEER_SEEN, ep_.COMPILER) in _ancestry(result, peer=False)


@pytest.mark.cpu_mode
def test_the_editor_is_stopped_by_collapsing_the_profile_folder_alone(result: pd.DataFrame):
    # Bob runs the editor from his profile every day; Carol's copy is in hers. Compared as reported, the paths
    # differ and the pair is novel; compared as the same software, the peer group has seen it.
    editor = ep_.editor("Carol")

    assert normalize_image_path(editor) == normalize_image_path(ep_.editor(ep_.FINANCE_HOSTS["fin-02"]))
    assert (ep_.PER_USER, editor) not in _ancestry(result)
    assert (ep_.PER_USER, editor) in _ancestry(result, normalize=False)


@pytest.mark.cpu_mode
def test_notepad_on_the_kiosk_is_stopped_by_its_own_history_alone(result: pd.DataFrame):
    notepad = _decisive(result, ep_.HOST_SEEN, ep_.NOTEPAD)

    assert bool(notepad["endpoint_host_only"])
    assert bool(notepad["endpoint_host_seen"])
    assert (ep_.HOST_SEEN, ep_.NOTEPAD) not in _ancestry(result)
    assert (ep_.HOST_SEEN, ep_.NOTEPAD) in _ancestry(result, host_history=False)


@pytest.mark.cpu_mode
def test_the_lab_machine_is_stopped_by_the_warmup_alone(result: pd.DataFrame):
    # Four days of history, and not in the inventory, so there is no group to lend it one.
    nmap = _decisive(result, ep_.WARMING_UP, ep_.NMAP)

    assert not bool(nmap["ctx_found"])
    assert not bool(nmap["endpoint_mature"])
    assert pd.isna(nmap["endpoint_pair_novel"])
    assert (ep_.WARMING_UP, ep_.NMAP) not in _ancestry(result)
    assert (ep_.WARMING_UP, ep_.NMAP) in _ancestry(result, warmup=False)


@pytest.mark.cpu_mode
def test_a_pair_last_seen_more_than_thirty_days_ago_is_novel_again(result: pd.DataFrame):
    runs = result[(result["hostname"] == ep_.STALE) & (result["image_path"] == ep_.TEAMVIEWER)]
    gap_days = (int(runs["event_time"].max()) - int(runs["event_time"].min())) / (ep_.DAY_SECONDS * ep_.NS_PER_SECOND)

    assert len(runs) == 2
    assert gap_days > ep_.WINDOW_DAYS
    assert bool(_decisive(result, ep_.STALE, ep_.TEAMVIEWER)["endpoint_pair_novel"])


@pytest.mark.cpu_mode
def test_the_ungrouped_host_and_the_unweighted_process_are_flagged_rather_than_dropped(result: pd.DataFrame):
    kiosk = _decisive(result, ep_.HOST_ONLY_NOVEL, ep_.MSHTA)
    unweighted = _decisive(result, ep_.UNWEIGHTED, ep_.CURL)

    assert bool(kiosk["endpoint_host_only"])
    assert pd.isna(kiosk["endpoint_peer_seen"])
    assert bool(kiosk["ctx_found"])
    assert pd.isna(unweighted["endpoint_integrity"])
    assert not bool(unweighted["endpoint_host_only"])


@pytest.mark.cpu_mode
def test_processes_that_share_a_second_are_all_judged(result: pd.DataFrame):
    # Logon bursts put three processes in one second. None of them is refused as out of order, and none of them is
    # prior history for another.
    burst = result[result["event_time"].astype("int64") == ep_.at(ep_.LAST_DAY, 9)]

    assert len(burst) >= 3 * len(set(ep_.FINANCE_HOSTS) | set(ep_.BUILD_HOSTS))
    assert burst["endpoint_pair_novel"].notna().sum() == len(burst[burst["hostname"] != ep_.LAB])


# --- Minimization, and the search as shipped -----------------------------------------------------------------------


def _search() -> str:
    with open(SAVEDSEARCHES, encoding="utf-8") as handle:
        text = handle.read()

    return text.split(f"[{RULE}]", 1)[1].split("action.correlationsearch.label", 1)[0].split("search =", 1)[1]


def test_the_rule_reads_neither_the_command_line_nor_the_process_identifiers():
    # The command line routinely carries file names, user names and secrets; process identifiers are per process
    # and say nothing a pair does not. An estate can drop both at the wire and keep the rule.
    search = _search()

    for column in ("command_line_hash", "process_guid", "parent_process_guid"):
        assert column not in search, column


def test_the_search_carries_the_weights_this_harness_asserts():
    search = _search()

    assert "endpoint_pair_novel=true" in search

    for (level, severity) in ep_.SEVERITY.items():
        assert f'"{level}", {severity}' in search

    assert f"true(), {ep_.UNWEIGHTED_SEVERITY}" in search
    assert "endpoint_host_only" in search


# --- Stamping ------------------------------------------------------------------------------------------------------


@pytest.mark.cpu_mode
def test_every_row_is_stamped_at_layer_seven_and_keyed_on_its_host(result: pd.DataFrame):
    assert set(result["osi_layer"]) == {ep_.OSI_LAYER}
    assert list(result["entity_key"]) == list(result["hostname"])
    assert result["lineage_id"].notna().all()
