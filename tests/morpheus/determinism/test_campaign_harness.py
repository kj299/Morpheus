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
Control 13's six checks against the campaign corpus, and R-C-001's predicate over it.

The chained rule is asserted the way the single-layer rules are: for each control, the check names the one condition
that keeps it quiet and shows the rule would have fired without it. Beyond that, the steps are shown to be below
their own rules' thresholds, because a chain whose steps each fire alone adds nothing an analyst would not already
see.
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
import campaign_pipeline as cp_  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_campaign_expected.csv")
DRIVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_campaign_pipeline.py")
SAVEDSEARCHES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..",
                             "..",
                             "..",
                             "examples",
                             "splunk_lineage_app",
                             "TA-morpheus-lineage",
                             "default",
                             "savedsearches.conf")
RULE = "R-C-001 - Lateral movement chain"
EXFIL_RULE = "R-C-004 - Staged exfiltration"
C2_RULE = "R-C-002 - TLS anomaly precedes beaconing"


@pytest.fixture(name="corpus", scope="module")
def corpus_fixture() -> dict[str, pd.DataFrame]:
    yield cp_.build_corpus()


# One composed run per execution mode, reused by every check in that mode.
_RESULTS: dict = {}


@pytest.fixture(name="pipeline_config")
def pipeline_config_fixture(execution_mode) -> Config:
    yield cp_.build_pipeline_config(execution_mode)


@pytest.fixture(name="result")
def result_fixture(pipeline_config: Config, corpus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mode = pipeline_config.execution_mode

    if (mode not in _RESULTS):
        _RESULTS[mode] = cp_.run_pipeline(pipeline_config, corpus)

    yield _RESULTS[mode]


def _principals(fired: dict) -> set:
    return {principal for (_, principal, _) in fired}


def _split(frame: pd.DataFrame, parts: int) -> list:
    size = max(1, len(frame) // parts)

    return [frame.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(frame), size)]


def _permuted(corpus: dict, seed: int) -> dict:
    period_ns = cp_.PERIOD_SECONDS * cp_.NS_PER_SECOND

    return {
        name:
            permute_within_contiguous_groups(frame,
                                             [window_id_from_timestamp(int(t), period_ns) for t in frame["event_time"]],
                                             seed=seed)
        for (name, frame) in corpus.items()
    }


# --- Check 1: the corpus is fixed and is one estate seen by three collectors -----------------------------------------


def test_corpus_is_fixed(corpus: dict[str, pd.DataFrame]):
    again = cp_.build_corpus()

    assert set(again) == set(corpus) == set(cp_.CLASSES)

    for name in cp_.CLASSES:
        pd.testing.assert_frame_equal(corpus[name], again[name])


def test_the_three_collectors_share_their_entities(corpus: dict[str, pd.DataFrame]):
    # The point of this corpus: the source address a flow comes from is the one a login comes from, and the host a
    # login reaches is the one an EDR reports, give or take case.
    flows = corpus[cp_.FLOW_CLASS]
    logins = corpus[cp_.AUTH_CLASS]
    processes = corpus[cp_.PROCESS_CLASS]

    lateral = flows[flows["src_ip"].isin({actor.source for actor in cp_.ACTORS.values()})]

    assert set(lateral["src_ip"]) <= set(logins["source_ip"])
    assert {host.lower() for host in logins["target_host"]} == set(processes["hostname"])

    for frame in corpus.values():
        assert set(cp_.ID_COLUMNS) <= set(frame.columns)
        assert frame["event_time"].is_monotonic_increasing
        assert frame["collector_seq"].is_monotonic_increasing


# --- Checks 2 through 6: determinism -------------------------------------------------------------------------------


@pytest.mark.gpu_and_cpu_mode
def test_double_run_diff(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    second = cp_.run_pipeline(pipeline_config, corpus)

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

    rendered = cp_.render(result)

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

    assert diff_frames(result, cp_.run_pipeline(pipeline_config, corpus, batches=thirds)) is None
    assert diff_frames(result, cp_.run_pipeline(pipeline_config, corpus, batches=by_row)) is None


@pytest.mark.gpu_and_cpu_mode
def test_permutation_within_windows(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    for seed in (1, 2, 3):
        assert diff_frames(result, cp_.run_pipeline(pipeline_config, _permuted(corpus, seed))) is None, seed


@pytest.mark.gpu_and_cpu_mode
def test_permutation_check_has_teeth(pipeline_config: Config, corpus: dict):
    # The negative control for the check above. The fan-out burst is twenty-five flows in one hour; without the
    # imposed order, each flow's distinct count depends on which of the others arrived first.
    ordered = cp_.run_pipeline(pipeline_config, corpus, impose_order=False)
    shuffled = cp_.run_pipeline(pipeline_config, _permuted(corpus, 5), impose_order=False)

    assert diff_frames(ordered, shuffled) is not None


# --- R-C-001: the campaign, and one control per condition ----------------------------------------------------------


@pytest.mark.cpu_mode
def test_the_chain_fires_on_the_attacker_alone(result: pd.DataFrame):
    attacker = cp_.ACTORS[cp_.ATTACKER]

    assert set(cp_.lateral_movement(result)) == {(attacker.source, cp_.ATTACKER, attacker.target.lower())}


@pytest.mark.cpu_mode
def test_no_step_of_the_attackers_chain_breaches_a_threshold_of_its_own(result: pd.DataFrame):
    # The fan-out is half what R-B-L3-001 fires on, and neither a first login to a server nor one new process on it
    # pages anyone: the chain is the only thing that sees the campaign whole.
    attacker = cp_.ACTORS[cp_.ATTACKER]
    flows = result[(result["telemetry_class"] == cp_.FLOW_CLASS) & (result["src_ip"] == attacker.source)]

    assert flows["dsts_per_src"].astype(int).max() < cp_.FANOUT_THRESHOLD


@pytest.mark.cpu_mode
@pytest.mark.parametrize("principal, condition",
                         [(cp_.WRONG_ORDER, {
                             "ordered": False
                         }), (cp_.TOO_SLOW, {
                             "window_seconds": None
                         }), (cp_.KNOWN_HOST, {
                             "new_host": False
                         }), (cp_.OTHER_SOURCE, {
                             "same_source": False
                         }), (cp_.FLAT_FANOUT, {
                             "rising": False
                         }), (cp_.NO_NEW_PROCESS, {
                             "novel_process": False
                         })])
def test_each_control_is_stopped_by_its_one_condition(result: pd.DataFrame, principal: str, condition: dict):
    assert principal not in _principals(cp_.lateral_movement(result))
    assert principal in _principals(cp_.lateral_movement(result, **condition))


@pytest.mark.cpu_mode
def test_each_control_does_every_other_step(result: pd.DataFrame):
    # The counterfactual above could pass for a control that fails two conditions, if removing one happened to let
    # another through. Here each control's other steps are shown directly.
    rises = set(cp_.fanout_rises(result)["src_ip"])
    logins = result[result["telemetry_class"] == cp_.AUTH_CLASS]
    novel = result[(result["telemetry_class"] == cp_.PROCESS_CLASS)
                   & result["endpoint_pair_novel"].astype("boolean").fillna(False).astype(bool)]

    for (principal, actor) in cp_.ACTORS.items():
        login = logins[(logins["user_principal"] == principal) & (logins["target_host"] == actor.target)
                       & (logins["event_time"].astype("int64") >= cp_.at(cp_.LAST_DAY, cp_.CAMPAIGN_HOUR))]

        assert len(login) == 1, principal
        assert (actor.source in rises) == (principal != cp_.FLAT_FANOUT), principal
        assert bool(login["target_host_first_seen"].iloc[0]) == (principal != cp_.KNOWN_HOST), principal
        assert (actor.target.lower() in set(novel["hostname"])) == (principal != cp_.NO_NEW_PROCESS), principal


@pytest.mark.cpu_mode
def test_the_join_tolerance_admits_a_step_logged_slightly_early(result: pd.DataFrame):
    # Move the attacker's process to one minute before the login: inside the two-minute tolerance, it still chains;
    # three minutes before, it does not.
    attacker = cp_.ACTORS[cp_.ATTACKER]
    host = attacker.target.lower()
    process = (result["telemetry_class"] == cp_.PROCESS_CLASS) & (
        result["hostname"] == host) & result["endpoint_pair_novel"].astype("boolean").fillna(False).astype(bool)
    login_time = cp_.at(cp_.LAST_DAY, cp_.CAMPAIGN_HOUR, attacker.login_minute)

    for (offset_minutes, fires) in ((1, True), (3, False)):
        shifted = result.copy()
        shifted.loc[process, "event_time"] = login_time - offset_minutes * 60 * cp_.NS_PER_SECOND

        assert (cp_.ATTACKER in _principals(cp_.lateral_movement(shifted))) == fires, offset_minutes


# --- R-C-004: the staged exfiltration, and one control per condition ------------------------------------------------


@pytest.mark.cpu_mode
def test_the_exfiltration_chain_fires_on_the_attacker_alone(result: pd.DataFrame):
    assert set(cp_.staged_exfiltration(result)) == {(cp_.KIM, cp_.EXFILTRATORS[cp_.KIM].address)}


@pytest.mark.cpu_mode
@pytest.mark.parametrize("principal, condition",
                         [(cp_.LEE, {
                             "session_bound": False
                         }), (cp_.NIA, {
                             "in_session": False
                         }), (cp_.OTO, {
                             "new_issuer": False
                         }), (cp_.PIA, {
                             "ordered": False
                         })])
def test_each_exfiltration_control_is_stopped_by_its_one_condition(result: pd.DataFrame,
                                                                   principal: str,
                                                                   condition: dict):
    assert principal not in {who for (who, _) in cp_.staged_exfiltration(result)}
    assert principal in {who for (who, _) in cp_.staged_exfiltration(result, **condition)}


@pytest.mark.cpu_mode
def test_every_exfiltrator_takes_every_single_layer_step(result: pd.DataFrame):
    # Each actor's export is a bulk access and each breach is a breach on its own terms -- the controls differ only in
    # how the steps relate, which is what the chain adds. Only the known-issuer control's handshake is ordinary.
    exports = result[result["telemetry_class"] == cp_.SAAS_CLASS]
    bulk = exports[exports["saas_baseline_mature"].astype("boolean").fillna(False).astype(bool)
                   & (exports["saas_record_ratio"].astype(float) > cp_.RECORD_MULTIPLE)]
    transfers = result[result["telemetry_class"] == cp_.TRANSFER_CLASS]
    breached = transfers[transfers["flow_data_len_envelope_breached"].astype("boolean").fillna(False).astype(bool)]
    handshakes = result[result["telemetry_class"] == cp_.HANDSHAKE_CLASS]
    novel = handshakes[handshakes["cert_issuer_new_to_estate"].astype("boolean").fillna(False).astype(bool)]

    assert set(bulk["user_principal"]) == set(cp_.EXFILTRATORS)
    assert set(breached["src_ip"]) == {actor.breach_address for actor in cp_.EXFILTRATORS.values()}
    # Among the exfiltrators' addresses; R-C-002's attacker also meets an issuer the estate has not seen.
    addresses = {actor.breach_address for actor in cp_.EXFILTRATORS.values()}
    expected = {actor.breach_address for actor in cp_.EXFILTRATORS.values() if actor.principal != cp_.OTO}

    assert set(novel["src_ip"]) & addresses == expected


@pytest.mark.cpu_mode
def test_a_session_that_ended_before_the_breach_does_not_bind_it(result: pd.DataFrame):
    nia = cp_.EXFILTRATORS[cp_.NIA]
    sessions = cp_.session_intervals(result)
    last = sessions[(sessions["user_principal"] == cp_.NIA)].sort_values("opened").iloc[-1]
    breach_time = cp_.at(cp_.LAST_DAY, cp_.CAMPAIGN_HOUR, nia.breach_minute)

    assert last["source_ip"] == nia.address
    assert int(last["closed"]) < breach_time


@pytest.mark.cpu_mode
def test_the_estate_issuer_is_new_only_after_the_estate_has_a_week_of_history(result: pd.DataFrame):
    handshakes = result[result["telemetry_class"] == cp_.HANDSHAKE_CLASS]
    early = handshakes[handshakes["event_time"].astype("int64") < cp_.at(7)]

    assert early["cert_issuer_new_to_estate"].isna().all()


# --- R-C-002: command and control establishment, and one control per condition --------------------------------------


@pytest.mark.cpu_mode
def test_the_command_and_control_chain_fires_on_the_attacker_alone(result: pd.DataFrame):
    attacker = cp_.BEACONERS[cp_.C2_ATTACKER]

    assert set(cp_.tls_before_beaconing(result)) == {(attacker.host, attacker.beacon_destination)}


@pytest.mark.cpu_mode
@pytest.mark.parametrize("host, condition",
                         [(cp_.BEACON_FIRST, {
                             "ordered": False
                         }), (cp_.OTHER_DESTINATION, {
                             "same_destination": False
                         }), (cp_.C2_TOO_SLOW, {
                             "window_seconds": None
                         }), (cp_.UNSETTLED, {
                             "settled": False
                         })])
def test_each_command_and_control_control_is_stopped_by_its_one_condition(result: pd.DataFrame,
                                                                          host: str,
                                                                          condition: dict):
    assert host not in {source for (source, _) in cp_.tls_before_beaconing(result)}
    assert host in {source for (source, _) in cp_.tls_before_beaconing(result, **condition)}


@pytest.mark.cpu_mode
def test_every_beaconer_presents_a_new_fingerprint_and_beacons(result: pd.DataFrame):
    # Every one of the five does both halves; only the unsettled host's fingerprint is below R-B-L6-001's floor.
    handshakes = result[result["telemetry_class"] == cp_.HANDSHAKE_CLASS]
    first = handshakes[handshakes["ja4_client_first_seen"].astype("boolean").fillna(False).astype(bool)]
    flows = result[result["telemetry_class"] == cp_.FLOW_CLASS]
    mature = flows[flows["flow_regularity_mature"].astype("boolean").fillna(False).astype(bool)]

    assert set(first["src_ip"]) >= set(cp_.BEACONERS)
    assert set(mature["src_ip"]) >= set(cp_.BEACONERS)

    settled = first[first["ja4_client_observations"].astype(int) >= 20]

    assert set(settled["src_ip"]) & set(cp_.BEACONERS) == set(cp_.BEACONERS) - {cp_.UNSETTLED}


# --- The search as shipped -----------------------------------------------------------------------------------------


def _search() -> str:
    with open(SAVEDSEARCHES, encoding="utf-8") as handle:
        text = handle.read()

    return text.split(f"[{RULE}]", 1)[1].split("action.correlationsearch.label", 1)[0].split("search =", 1)[1]


def test_the_search_carries_the_conditions_this_harness_asserts():
    search = _search()
    tolerance = cp_.JOIN_TOLERANCE_SECONDS
    window = cp_.CHAIN_WINDOW_SECONDS

    assert "where dsts_per_src > previous_peak" in search
    assert "eval window_id = window_id + 1" in search
    assert 'auth_result="success" target_host_first_seen=true' in search
    assert "endpoint_pair_novel=true" in search
    assert "rename source_ip AS src_ip" in search
    assert "lower(target_host)" in search and "lower(hostname)" in search
    assert f"t_login >= t_fanout - {tolerance} AND t_login - t_fanout <= {window}" in search
    assert f"t_process >= t_login - {tolerance} AND t_process - t_fanout <= {window}" in search
    assert f"risk_score = {cp_.SEVERITY}" in search


def _exfil_search() -> str:
    with open(SAVEDSEARCHES, encoding="utf-8") as handle:
        text = handle.read()

    return text.split(f"[{EXFIL_RULE}]", 1)[1].split("action.correlationsearch.label", 1)[0].split("search =", 1)[1]


def test_the_exfiltration_search_carries_the_conditions_this_harness_asserts():
    search = _exfil_search()
    tolerance = cp_.JOIN_TOLERANCE_SECONDS
    window = cp_.EXFIL_WINDOW_SECONDS

    assert f"saas_baseline_mature=true saas_record_ratio>{int(cp_.RECORD_MULTIPLE)}" in search
    assert "(flow_data_len_envelope_breached=true OR flow_bpp_envelope_breached=true)" in search
    assert "cert_issuer_new_to_estate=true" in search
    assert "t_breach >= opened AND (isnull(closed) OR t_breach <= closed)" in search
    assert f"t_breach >= t_export - {tolerance} AND t_breach - t_export <= {window}" in search
    assert f"t_handshake >= t_breach - {tolerance} AND t_handshake - t_export <= {window}" in search
    assert f"risk_score = {cp_.EXFIL_SEVERITY}" in search
    assert "rule_id" not in search.split("| eval _time = t_handshake, rule_id", 1)[0]


def test_the_command_and_control_search_carries_the_conditions_this_harness_asserts():
    with open(SAVEDSEARCHES, encoding="utf-8") as handle:
        text = handle.read()

    search = text.split(f"[{C2_RULE}]", 1)[1].split("action.correlationsearch.label", 1)[0].split("search =", 1)[1]
    tolerance = cp_.JOIN_TOLERANCE_SECONDS
    window = cp_.C2_WINDOW_SECONDS

    assert "ja4_client_first_seen=true ja4_client_observations>=20" in search
    assert "flow_regularity_mature=true flow_interval_cv<0.15 flow_size_cv<0.15" in search
    assert "join type=inner max=0 src_ip dst_ip" in search
    assert f"t_beacon >= t_tls - {tolerance} AND t_beacon - t_tls <= {window}" in search
    assert f"risk_score = {cp_.C2_SEVERITY}" in search
    assert "rule_id" not in search.split("rule_id = ", 1)[0]


def test_the_search_reads_scored_events_rather_than_notables():
    # Notables exist only once the detection searches have run and written them; R-C-002 read them, and returned no
    # row anywhere this fork could test until it was rewritten to read events too. Everything before a rule stamps
    # its own rule_id is what it reads.
    reads = _search().split('| eval _time = t_process, rule_id', 1)[0]

    assert "rule_id" not in reads


# --- Stamping ------------------------------------------------------------------------------------------------------


@pytest.mark.cpu_mode
def test_each_layer_is_stamped_and_keyed_on_its_own_entity(result: pd.DataFrame):
    for (name, layer, entity) in ((cp_.FLOW_CLASS, 3, "src_ip"), (cp_.AUTH_CLASS, 5, "user_principal"),
                                  (cp_.PROCESS_CLASS, 7, "hostname")):
        rows = result[result["telemetry_class"] == name]

        assert set(rows["osi_layer"].astype(int)) == {layer}, name
        assert list(rows["entity_key"]) == list(rows[entity]), name
        assert rows["lineage_id"].notna().all(), name
