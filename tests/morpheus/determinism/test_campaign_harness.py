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
from morpheus.utils.entity_key import normalize_hostname
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
REPLAY_RULE = "R-C-005 - Credential replay across the stack"
PHYSICAL_RULE = "R-C-003 - Physical compromise chain"
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

    assert set(again) == set(corpus) == set(cp_.CORPUS_CLASSES)

    for name in cp_.CORPUS_CLASSES:
        pd.testing.assert_frame_equal(corpus[name], again[name])


def test_the_three_collectors_share_their_entities(corpus: dict[str, pd.DataFrame]):
    # The point of this corpus: the source address a flow comes from is the one a login comes from, and the host a
    # login reaches is the one an EDR reports, give or take case.
    flows = corpus[cp_.FLOW_CLASS]
    logins = corpus[cp_.AUTH_CLASS]
    processes = corpus[cp_.PROCESS_CLASS]

    lateral = flows[flows["src_ip"].isin({actor.source for actor in cp_.ACTORS.values()})]

    assert set(lateral["src_ip"]) <= set(logins["source_ip"])

    # R-C-001's actors' logins, that is: R-C-005's principals sign in to the intranet, which no EDR reports.
    actor_logins = logins[logins["user_principal"].isin({actor.principal for actor in cp_.ACTORS.values()})]
    assert {cp_.edr_name(host) for host in actor_logins["target_host"]} == set(processes["hostname"])
    # The directory and the EDR spell the attacker's target differently, which is the case the shared host rule is
    # for: the join below is on keys both pipelines render the same way, not on a case-fold in the test.
    assert cp_.ACTORS[cp_.ATTACKER].target != cp_.edr_name(cp_.ACTORS[cp_.ATTACKER].target)

    for frame in corpus.values():
        assert set(cp_.ID_COLUMNS) <= set(frame.columns)
        assert frame["event_time"].is_monotonic_increasing
        # Each collector numbers its own stream, so the sequence rises per collector rather than across them.
        assert frame.groupby("collector_id")["collector_seq"].apply(lambda seq: seq.is_monotonic_increasing).all()


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

    assert set(cp_.lateral_movement(result)) == {(attacker.source, cp_.ATTACKER, normalize_hostname(attacker.target))}


@pytest.mark.cpu_mode
def test_both_pipelines_render_the_target_as_one_host(result: pd.DataFrame):
    # The login names SRV-DB-02 and the EDR reports srv-db-02. Each pipeline renders its own spelling with the shared
    # rule, and the two keys are equal, which is all R-C-001 joins on.
    attacker = cp_.ACTORS[cp_.ATTACKER]
    logins = result[(result["telemetry_class"] == cp_.AUTH_CLASS) & (result["target_host"] == attacker.target)]
    processes = result[(result["telemetry_class"] == cp_.PROCESS_CLASS)
                       & (result["hostname"] == cp_.edr_name(attacker.target))]

    assert set(logins["target_host_key"]) == set(processes["endpoint_host"]) == {"srv-db-02"}
    assert set(processes["entity_key"]) == {"srv-db-02"}


@pytest.mark.cpu_mode
def test_the_inventory_says_which_targets_are_servers(result: pd.DataFrame):
    # R-C-001 is a login to a server. The process step reads the role the TC-0 enrichment attached, and the one
    # workstation in the inventory is the only process host that is not a server.
    processes = result[result["telemetry_class"] == cp_.PROCESS_CLASS]
    roles = processes.groupby("endpoint_host")["ctx_device_role"].agg(lambda values: set(values))

    assert {host for (host, role) in roles.items() if role != {"server"}} == set(cp_.WORKSTATIONS)
    assert all(role == {"workstation"} for (host, role) in roles.items() if host in cp_.WORKSTATIONS)


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
                         }), (cp_.WORKSTATION_TARGET, {
                             "server_only": False
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
        assert (normalize_hostname(actor.target) in set(novel["endpoint_host"])) == (principal
                                                                                     != cp_.NO_NEW_PROCESS), principal


@pytest.mark.cpu_mode
def test_the_join_tolerance_admits_a_step_logged_slightly_early(result: pd.DataFrame):
    # Move the attacker's process to one minute before the login: inside the two-minute tolerance, it still chains;
    # three minutes before, it does not.
    attacker = cp_.ACTORS[cp_.ATTACKER]
    host = normalize_hostname(attacker.target)
    process = (result["telemetry_class"] == cp_.PROCESS_CLASS) & (
        result["endpoint_host"] == host) & result["endpoint_pair_novel"].astype("boolean").fillna(False).astype(bool)
    login_time = cp_.at(cp_.LAST_DAY, cp_.CAMPAIGN_HOUR, attacker.login_minute)

    for (offset_minutes, fires) in ((1, True), (3, False)):
        shifted = result.copy()
        shifted.loc[process, "event_time"] = login_time - offset_minutes * 60 * cp_.NS_PER_SECOND

        assert (cp_.ATTACKER in _principals(cp_.lateral_movement(shifted))) == fires, offset_minutes


# --- Connection evidence: one connection, three collectors -----------------------------------------------------------


@pytest.mark.cpu_mode
def test_one_connection_is_joined_across_layers_3_4_and_6_by_its_community_id(result: pd.DataFrame):
    # The attacker's breach connection, as the flow exporter, the packet sensor and the inspection point each saw
    # it. The four other breaches were seen at layers 3 and 4 only, and are not returned.
    connections = cp_.connections_across_layers(result)
    attacker = cp_.EXFILTRATORS[cp_.KIM]

    assert len(connections) == 1

    [rows] = connections.values()

    assert set(rows["telemetry_class"]) == {cp_.FLOW_CLASS, cp_.TRANSFER_CLASS, cp_.HANDSHAKE_CLASS}
    assert set(rows["src_ip"]) == {attacker.breach_address}
    assert set(rows["dst_ip"]) == {cp_.SYNC_SERVER}
    assert set(rows["src_port"].astype(int)) == {cp_.BREACH_PORT}

    breaches = result[(result["telemetry_class"] == cp_.FLOW_CLASS) & (result["src_port"] == cp_.BREACH_PORT)]
    assert breaches["src_ip"].nunique() == len(cp_.EXFILTRATORS)


def test_the_connection_search_joins_on_community_id_and_wants_all_three_layers():
    with open(SAVEDSEARCHES, encoding="utf-8") as handle:
        text = handle.read()

    stanza = text.split("[Connection evidence - layers 3, 4 and 6]", 1)[1].split("\n[", 1)[0]
    search = stanza.split("search =", 1)[1]

    assert "BY community_id" in search
    assert "where layers = 3" in search
    # A report: it writes nothing to the risk index.
    assert "| collect" not in search


# --- R-C-004: the staged exfiltration, and one control per condition ------------------------------------------------


@pytest.mark.cpu_mode
def test_the_exfiltration_chain_fires_on_the_attacker_alone(result: pd.DataFrame):
    assert set(cp_.staged_exfiltration(result)) == {(cp_.KIM, cp_.EXFILTRATORS[cp_.KIM].address)}


@pytest.mark.cpu_mode
def test_the_chain_fires_through_a_collector_that_says_logon_and_logoff(result: pd.DataFrame):
    # Kim's sessions come from a domain controller, whose records say logon and logoff. The chain fires because
    # the search and this recomputation read session_lifecycle, which TC5SessionStage normalizes to start and end;
    # read with the literal words the other collector uses, Kim has no session and the one firing disappears.
    sessions = result[(result["telemetry_class"] == cp_.SESSION_CLASS) & (result["user_principal"] == cp_.KIM)]

    assert set(sessions["session_action"]) == {"logon", "logoff"}
    assert set(sessions["session_lifecycle"]) == {"start", "end"}

    literal = result.copy()
    literal["session_lifecycle"] = literal["session_action"].where(literal["session_action"].isin(["start", "end"]))

    assert cp_.KIM in {who for (who, _) in cp_.staged_exfiltration(result)}
    assert cp_.KIM not in {who for (who, _) in cp_.staged_exfiltration(literal)}


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


@pytest.mark.cpu_mode
def test_the_credential_replay_fires_on_the_attacker_alone(result: pd.DataFrame):
    fired = cp_.credential_replays(result)

    assert set(fired) == {cp_.QUIN}

    (previous_port, port, kmh) = fired[cp_.QUIN]
    replay = cp_.REPLAYERS[cp_.QUIN]

    assert previous_port == f"hq:hq-sw1:{replay.first.port}"
    assert port == f"edinburgh:edi-sw1:{replay.second.port}"
    assert kmh > 1500


def _second_sign_in(result: pd.DataFrame, principal: str) -> pd.Series:
    logins = result[(result["telemetry_class"] == cp_.AUTH_CLASS)
                    & (result["user_principal"] == principal)].sort_values("event_time")

    assert len(logins) == 2, f"{principal} should sign in exactly twice"

    return logins.iloc[1]


@pytest.mark.cpu_mode
def test_every_desk_address_resolves_to_its_own_port(result: pd.DataFrame):
    # The ladder itself, before any rule reads it: a leased address goes to its workstation and the workstation to
    # the port the switch saw it on. A ladder that resolved to the wrong port would make every journey wrong.
    logins = result[result["telemetry_class"] == cp_.AUTH_CLASS]
    checked = 0

    for (index, replay) in enumerate(cp_.REPLAYERS.values()):
        (first_time, _) = cp_._replay_times(index, replay)  # pylint: disable=protected-access
        first = logins[(logins["user_principal"] == replay.principal) & (logins["event_time"] == first_time)].iloc[0]

        assert first["login_mac"] == replay.first.mac
        assert first["login_port_key"] == f"{replay.first.site}:{cp_.SITE_SWITCHES[replay.first.site]}:"\
            f"{replay.first.port}"
        assert first["login_site_id"] == replay.first.site
        checked += 1

    assert checked == len(cp_.REPLAYERS)


@pytest.mark.cpu_mode
def test_the_same_site_is_no_journey(result: pd.DataFrame):
    second = _second_sign_in(result, cp_.RHEA)

    assert second["login_site_id"] == "hq"
    assert second["site_travel_status"] == "measured"
    assert float(second["site_travel_distance_km"]) == 0


@pytest.mark.cpu_mode
def test_a_journey_with_time_to_make_it_is_not_impossible(result: pd.DataFrame):
    second = _second_sign_in(result, cp_.SAM)

    assert second["login_site_id"] == "edinburgh"
    assert second["site_travel_status"] == "measured"
    assert 100 < float(second["site_travel_kmh"]) < cp_.REPLAY_KMH


@pytest.mark.cpu_mode
@pytest.mark.parametrize("principal", [cp_.TIA, cp_.UMA])
def test_an_address_no_current_lease_names_claims_no_port(result: pd.DataFrame, principal: str):
    # The geolocated rule would measure this sign-in; this one cannot, because no switch port is behind it. That is
    # the corroboration the rule exists for, not a gap in it.
    second = _second_sign_in(result, principal)

    assert second[cp_.LEASE_METHOD_COLUMN] == "unresolved"
    assert pd.isna(second["login_site_id"])
    assert second["site_travel_status"] == "no_coordinate"


@pytest.mark.cpu_mode
def test_the_expired_lease_did_name_the_address_until_an_hour_before(pipeline_config: Config,
                                                                     corpus: dict,
                                                                     result: pd.DataFrame):
    # What separates the expired lease from the remote address: the address was leased to an Edinburgh desk, and
    # resolving it against the sign-in's own time rather than against what the lease once said is what refuses it.
    # The lease is the DHCP server's: an acknowledgement whose term the server stated, closed at that term.
    replay = cp_.REPLAYERS[cp_.UMA]
    bindings = cp_.close_lease_bindings(pipeline_config, [corpus[cp_.LEASE_CLASS]])
    leases = cp_.build_lease_table(bindings)
    second = _second_sign_in(result, cp_.UMA)
    earlier = int(second["event_time"]) - 61 * 60 * 10**9

    assert leases.resolve(replay.second.address, earlier) is not None
    assert leases.resolve(replay.second.address, int(second["event_time"])) is None
    assert leases.resolve(cp_.REMOTE_ADDRESS, earlier) is None


@pytest.mark.cpu_mode
def test_a_refused_sign_in_proves_nobody_was_there(result: pd.DataFrame):
    second = _second_sign_in(result, cp_.VIC)

    assert second["auth_result"] == "failure"
    assert second["login_site_id"] == "edinburgh"
    assert second["site_travel_status"] == "not_successful"


@pytest.mark.cpu_mode
def test_the_physical_compromise_fires_on_the_attacker_alone(result: pd.DataFrame):
    intrusion = cp_.INTRUSIONS["rogue"]
    fired = cp_.physical_compromises(result)

    assert set(fired) == {(f"hq:hq-sw1:{intrusion.desk.port}", intrusion.device_mac)}

    (t_port, t_vendor, t_management) = fired[(f"hq:hq-sw1:{intrusion.desk.port}", intrusion.device_mac)]

    # In order, and inside the window: the poll after the link came back, the walk that first listed the device,
    # the first flow to the switches' own interfaces.
    assert t_port <= t_vendor <= t_management
    assert t_management - t_port <= cp_.PHYSICAL_WINDOW_SECONDS * 10**9


@pytest.mark.cpu_mode
@pytest.mark.parametrize("name, condition",
                         [("known-vendor", {
                             "new_vendor": False
                         }), ("no-management", {
                             "management": False
                         }), ("no-link-event", {
                             "port_event": False
                         })])
def test_each_physical_control_is_stopped_by_its_one_condition(result: pd.DataFrame, name: str, condition: dict):
    intrusion = cp_.INTRUSIONS[name]
    pair = (f"hq:hq-sw1:{intrusion.desk.port}", intrusion.device_mac)

    assert pair not in cp_.physical_compromises(result)
    assert pair in cp_.physical_compromises(result, **condition)


def _steps(result: pd.DataFrame) -> dict:
    """Per R-C-003 port, which of the three steps the pipeline's output holds."""
    polls = result[result["telemetry_class"] == cp_.PORT_CLASS]
    walks = result[result["telemetry_class"] == cp_.MAC_CLASS]
    flows = result[result["telemetry_class"] == cp_.FLOW_CLASS]
    steps = {}

    for (name, intrusion) in cp_.INTRUSIONS.items():
        port = f"hq:hq-sw1:{intrusion.desk.port}"
        own_polls = polls[polls["entity_key"] == port]
        own_walks = walks[(walks["port_key"] == port) & (walks["mac_address"] == intrusion.device_mac)]
        own_flows = flows[flows["mac_address"] == intrusion.device_mac]
        steps[name] = (bool((own_polls["link_flaps"].astype("Int64").fillna(0) > 0).any()),
                       bool(own_walks["oui_first_seen_on_port"].astype("boolean").fillna(False).any()),
                       bool(own_flows["dst_is_management"].astype("boolean").fillna(False).any()))

    return steps


@pytest.mark.cpu_mode
def test_each_intrusion_takes_the_steps_it_was_planted_with(result: pd.DataFrame):
    # Each control takes two of the three steps, and a different two, so each step is shown to be necessary.
    assert _steps(result) == {
        "rogue": (True, True, True),
        "known-vendor": (True, False, True),
        "no-management": (True, True, False),
        "no-link-event": (False, True, True),
    }


@pytest.mark.cpu_mode
def test_the_management_flow_reaches_the_port_through_the_lease(result: pd.DataFrame):
    # The chain's last join is on the MAC, which the flow carries only because the DHCP lease named its address:
    # the rogue device is in no inventory, so the flow knows no machine, only the interface.
    intrusion = cp_.INTRUSIONS["rogue"]
    flows = result[(result["telemetry_class"] == cp_.FLOW_CLASS) & (result["src_ip"] == intrusion.device_address)]

    assert set(flows["mac_address"]) == {intrusion.device_mac}
    assert flows["host_lease_uid"].notna().all()
    assert flows["host_key"].isna().all()


@pytest.mark.cpu_mode
def test_a_ports_first_walk_says_nothing_about_vendors(result: pd.DataFrame):
    # Null, not false and not true: on the first walk there is nothing for a vendor to be new against.
    walks = result[result["telemetry_class"] == cp_.MAC_CLASS].sort_values("event_time", kind="mergesort")
    first = walks.groupby("port_key", sort=False).head(1)

    assert first["oui_first_seen_on_port"].isna().all()


def _physical_search() -> str:
    with open(SAVEDSEARCHES, encoding="utf-8") as handle:
        text = handle.read()

    return text.split(f"[{PHYSICAL_RULE}]", 1)[1].split("action.correlationsearch.label", 1)[0].split("search =", 1)[1]


def test_the_physical_compromise_search_carries_the_conditions_this_harness_asserts():
    search = _physical_search()

    assert "(transceiver_serial_changed=true OR link_flaps>0)" in search
    assert "oui_first_seen_on_port=true" in search
    assert "dst_is_management=true mac_address=*" in search
    assert "join type=inner max=0 port_key" in search
    assert "join type=inner max=0 mac_address" in search
    assert f"t_vendor >= t_port - {cp_.JOIN_TOLERANCE_SECONDS}" in search
    assert f"t_management >= t_vendor - {cp_.JOIN_TOLERANCE_SECONDS}" in search
    assert f"t_management - t_port <= {cp_.PHYSICAL_WINDOW_SECONDS}" in search
    assert f"risk_score = {cp_.PHYSICAL_SEVERITY}" in search
    assert "notable" not in search


def _replay_search() -> str:
    with open(SAVEDSEARCHES, encoding="utf-8") as handle:
        text = handle.read()

    return text.split(f"[{REPLAY_RULE}]", 1)[1].split("action.correlationsearch.label", 1)[0].split("search =", 1)[1]


def test_the_credential_replay_search_carries_the_conditions_this_harness_asserts():
    search = _replay_search()

    assert "sourcetype=morpheus:score:l5" in search
    assert 'site_travel_status="measured" OR site_travel_status="first_for_principal"' in search
    assert "streamstats current=f window=1" in search
    assert f"site_travel_kmh >= {cp_.REPLAY_KMH}" in search
    assert "notable" not in search


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
    assert "endpoint_pair_novel=true ctx_device_role=server" in search
    assert "rename source_ip AS src_ip" in search
    # Joined on the keys both pipelines render with the shared host rule, so the search folds no case of its own.
    assert "eval target = target_host_key" in search and "eval target = endpoint_host" in search
    assert "lower(" not in search
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
    assert 'if(session_lifecycle="start", _time, null())' in search
    assert 'if(session_lifecycle="end", _time, null())' in search
    assert "session_action" not in search
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


@pytest.mark.cpu_mode
def test_every_lease_is_the_dhcp_servers_and_lasts_its_whole_term(pipeline_config: Config, corpus: dict):
    # The leases used to be a table written into the harness. They are the server's acknowledgements now, closed by
    # the stage that closes every binding, and each lasts the term the server granted rather than ending a tick after
    # its acknowledgement -- the difference between a lease that resolves the afternoon's sign-ins and one that
    # resolves nothing after eight in the morning.
    # The desks' acknowledgements are the ones the relay placed on a port; the rest are the workstations' daily
    # renewals, each of which extends its lease rather than starting one.
    every = corpus[cp_.LEASE_CLASS]
    acks = every[every["port_id"].notna()]
    bindings = cp_.close_lease_bindings(pipeline_config, [every]).set_index("ip")

    assert set(bindings.index) == set(every["ip"]) and cp_.REMOTE_ADDRESS not in set(bindings.index)
    assert len(acks) > 0 and set(acks["ip"]).isdisjoint(set(every.loc[every["port_id"].isna(), "ip"]))

    for (_, ack) in acks.iterrows():
        assert bindings.loc[ack["ip"], "bind_end"] == ack["event_time"] + ack["lease_seconds"] * 10**9
        assert bindings.loc[ack["ip"], "port_id"] == ack["port_id"]


@pytest.mark.cpu_mode
def test_each_end_of_the_replay_names_the_lease_and_the_mac_binding_it_rests_on(pipeline_config: Config,
                                                                                corpus: dict,
                                                                                result: pd.DataFrame):
    # The port a sign-in is placed at is an inference two bindings deep, and an analyst disputing it needs those two
    # records, not just their conclusion. Each end of the attacker's journey carries the lease its address resolved
    # through and the MAC binding the workstation resolved through, and they are the bindings the tables hold.
    leases = cp_.build_lease_table(cp_.close_lease_bindings(pipeline_config, [corpus[cp_.LEASE_CLASS]]))
    macs = cp_.build_mac_table(cp_.close_mac_bindings(pipeline_config, [corpus[cp_.MAC_CLASS]]))
    logins = result[(result["telemetry_class"] == cp_.AUTH_CLASS) & (result["user_principal"] == cp_.QUIN)
                    & result["login_port_key"].notna()]

    assert len(logins) >= 2

    for (_, row) in logins.iterrows():
        when = int(row["event_time"])

        assert row[cp_.LEASE_UID_COLUMN] == leases.resolve(row["source_ip"], when).uid
        assert row[cp_.PORT_UID_COLUMN] == macs.resolve(row["login_mac"], when).uid

    # And a sign-in no lease names carries neither.
    remote = result[(result["telemetry_class"] == cp_.AUTH_CLASS) & (result["source_ip"] == cp_.REMOTE_ADDRESS)]

    assert len(remote) > 0
    assert remote[cp_.LEASE_UID_COLUMN].isna().all() and remote[cp_.PORT_UID_COLUMN].isna().all()


# --- The edge stream ------------------------------------------------------------------------------------------------


@pytest.mark.cpu_mode
def test_the_breach_connection_is_carried_from_layer_3_to_layer_6_on_its_community_id(
        pipeline_config: Config, result: pd.DataFrame):
    # The connection the connection evidence report returns, as edges: the packet sensor's records joined to the
    # flow exporter's, and the inspection point's to the packet sensor's.
    edges = cp_.build_edges(pipeline_config, result)
    carried = edges[edges["relation_type"] == "carried_by"]
    [rows] = cp_.connections_across_layers(result).values()
    [community] = set(rows["community_id"])
    attacker = carried[carried["community_id"] == community]

    assert set(attacker["join_method"]) == {"hard:community_id"}
    assert set(zip(attacker["parent_osi_layer"], attacker["child_osi_layer"])) == {(3, 4), (4, 6)}

    # The other breaches were seen at layers 3 and 4 only: one edge each, and none reaching layer 6.
    others = carried[carried["community_id"] != community]

    assert len(others) == len(cp_.EXFILTRATORS) - 1
    assert set(zip(others["parent_osi_layer"], others["child_osi_layer"])) == {(3, 4)}


@pytest.mark.cpu_mode
def test_every_resolved_record_is_the_child_of_an_edge_from_each_binding(pipeline_config: Config, result: pd.DataFrame):
    edges = cp_.build_edges(pipeline_config, result)
    resolved = edges[edges["relation_type"] == "resolved_through"]
    lineage = dict(zip(result["event_uid"], result["lineage_id"]))

    for (method, uid) in zip(cp_.EDGE_HOPS["hop_method_columns"], cp_.EDGE_HOPS["hop_uid_columns"]):
        rows = result[result[uid].notna()]
        parents = resolved[resolved["parent_event_uid"].isin(set(rows[uid]))]

        assert len(rows) > 0, method
        assert set(zip(rows[uid], rows["event_uid"])) <= set(zip(parents["parent_event_uid"],
                                                                   parents["child_event_uid"])), method

    assert all(lineage[child] == chain for (child, chain) in zip(edges["child_event_uid"], edges["lineage_id"]))
    assert not edges["link_uid"].duplicated().any()


# --- Hard joins: R-C-004 on session_key, R-C-002 on community_id, each with the value join beneath it ---------------


def _without_hard_keys(result: pd.DataFrame) -> pd.DataFrame:
    """The same estate from collectors nothing resolved: no step placed in a session, no connection recorded twice."""
    stripped = result.copy()
    stepped = stripped["telemetry_class"] != cp_.SESSION_CLASS
    stripped.loc[stepped, "session_key"] = None
    stripped["community_id"] = None

    return stripped


@pytest.mark.cpu_mode
def test_each_exfiltration_step_is_placed_in_the_session_that_held_its_address(result: pd.DataFrame):
    # The export, the breach and the handshake each carry the session that held their source address when they
    # happened -- the attacker's own. Nia's breach, twenty minutes after she logged off, is in no session; Lee's,
    # from Max's address, is in Max's.
    kim = cp_.EXFILTRATORS[cp_.KIM]
    steps = result[result["telemetry_class"].isin([cp_.SAAS_CLASS, cp_.TRANSFER_CLASS, cp_.HANDSHAKE_CLASS])]
    attacker = steps[((steps["telemetry_class"] == cp_.SAAS_CLASS) & (steps["user_principal"] == cp_.KIM))
                     | (steps["src_ip"] == kim.breach_address)]
    campaign = attacker[attacker["event_time"].astype("int64") >= cp_.at(cp_.LAST_DAY, cp_.CAMPAIGN_HOUR)]

    assert set(campaign["session_key"]) == {f"{cp_.KIM}:kim-{cp_.LAST_DAY}"}
    assert set(campaign["session_resolution"]) == {"soft:session"}

    breaches = steps[(steps["telemetry_class"] == cp_.TRANSFER_CLASS) & (steps["src_port"] == cp_.BREACH_PORT)]
    by_address = dict(zip(breaches["src_ip"], breaches["session_key"]))

    assert pd.isna(by_address[cp_.EXFILTRATORS[cp_.NIA].breach_address])
    assert by_address[cp_.EXFILTRATORS[cp_.LEE].breach_address] == f"{cp_.MAX}:max-{cp_.LAST_DAY}"


@pytest.mark.cpu_mode
def test_the_exfiltration_chain_is_joined_on_the_session(result: pd.DataFrame):
    chains = cp_.exfiltration_chains(result)

    assert set(chains) == {(cp_.KIM, cp_.EXFILTRATORS[cp_.KIM].address)}
    assert {method for (*_, method) in chains.values()} == {cp_.HARD_SESSION}


@pytest.mark.cpu_mode
def test_without_session_keys_the_exfiltration_chain_falls_back_to_the_address(result: pd.DataFrame):
    # A deployment whose collectors nothing placed in a session gets the same answer, on the address and the
    # session's interval, and says so; the controls stay quiet on the fallback too.
    chains = cp_.exfiltration_chains(_without_hard_keys(result))

    assert set(chains) == {(cp_.KIM, cp_.EXFILTRATORS[cp_.KIM].address)}
    assert {method for (*_, method) in chains.values()} == {cp_.VALUE_ADDRESS}
    assert {key: times[:3] for (key, times) in chains.items()} == cp_.staged_exfiltration(result)


@pytest.mark.cpu_mode
def test_the_new_stack_and_the_beacon_are_one_connection(result: pd.DataFrame):
    # The attacker's new fingerprint was presented on the connection the beacon's first flow record closes: one
    # Community ID at layers 3 and 6. No control's fingerprint shares a connection with any flow.
    handshakes = result[(result["telemetry_class"] == cp_.HANDSHAKE_CLASS)
                        & result["ja4_client_first_seen"].map(lambda value: value is True)
                        & result["src_ip"].isin(set(cp_.BEACONERS))]
    flows = set(result.loc[result["telemetry_class"] == cp_.FLOW_CLASS, "community_id"])
    shared = handshakes[handshakes["community_id"].isin(flows)]

    assert list(shared["src_ip"]) == [cp_.C2_ATTACKER]


@pytest.mark.cpu_mode
def test_the_command_and_control_chain_is_joined_on_the_connection(result: pd.DataFrame):
    chains = cp_.c2_chains(result)

    assert set(chains) == {(cp_.C2_ATTACKER, cp_.BEACONERS[cp_.C2_ATTACKER].beacon_destination)}
    assert {method for (*_, method) in chains.values()} == {cp_.HARD_CONNECTION}


@pytest.mark.cpu_mode
def test_without_community_ids_the_command_and_control_chain_falls_back_to_the_pair(result: pd.DataFrame):
    chains = cp_.c2_chains(_without_hard_keys(result))

    assert set(chains) == {(cp_.C2_ATTACKER, cp_.BEACONERS[cp_.C2_ATTACKER].beacon_destination)}
    assert {method for (*_, method) in chains.values()} == {cp_.VALUE_PAIR}
    assert {key: times[:2] for (key, times) in chains.items()} == cp_.tls_before_beaconing(result)


def test_the_chained_searches_join_on_the_hard_keys_and_fall_back():
    with open(SAVEDSEARCHES, encoding="utf-8") as handle:
        text = handle.read()

    exfiltration = text.split(f"[{EXFIL_RULE}]", 1)[1].split("action.correlationsearch.label", 1)[0]
    c2 = text.split(f"[{C2_RULE}]", 1)[1].split("action.correlationsearch.label", 1)[0]

    assert "link = coalesce(session_key, src_ip)" in exfiltration
    assert "(isnotnull(breach_session) AND link == session_key) OR (isnull(breach_session) AND link == src_ip" in (
        exfiltration)
    assert f'"{cp_.HARD_SESSION}", "{cp_.VALUE_ADDRESS}"' in exfiltration
    assert "join type=left max=1 community_id" in c2
    assert f'"{cp_.HARD_CONNECTION}", "{cp_.VALUE_PAIR}"' in c2
