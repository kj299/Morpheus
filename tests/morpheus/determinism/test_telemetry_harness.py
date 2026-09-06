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
Control 13's six determinism checks over the composed telemetry pipeline, plus the planted-anomaly assertions.

The lineage harness asks "does the substrate reproduce". This one asks two more things: "do the telemetry stages
compose", and "do they see what they were built to see". A pipeline that is deterministic and blind is not worth
much, so every anomaly planted in the corpus is asserted here as the column a rule would read.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.utils.binding_closer import CONFLICT
from morpheus.utils.determinism import diff_frames
from morpheus.utils.determinism import frame_digest
from morpheus.utils.determinism import permute_within_contiguous_groups
from morpheus.utils.lineage import window_id_from_timestamp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import telemetry_pipeline as tp  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_telemetry_expected.csv")
DRIVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_telemetry_pipeline.py")

NS = tp.NS_PER_SECOND


@pytest.fixture(name="corpus", scope="module")
def corpus_fixture() -> dict[str, pd.DataFrame]:
    yield tp.build_corpus()


# One composed run per execution mode, reused by every check in that mode. The fixture cannot be module-scoped any
# more, because which mode a test wants is a function-scoped decision, so the caching moves here instead.
_RESULTS: dict = {}


@pytest.fixture(name="pipeline_config")
def pipeline_config_fixture(execution_mode) -> Config:
    # The harness's own configuration rather than the shared `config` fixture, so the only thing the mode markers
    # vary is the execution mode: everything else about the pipeline stays what the golden was produced under.
    yield tp.build_pipeline_config(execution_mode)


@pytest.fixture(name="result")
def result_fixture(pipeline_config: Config, corpus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mode = pipeline_config.execution_mode

    if (mode not in _RESULTS):
        _RESULTS[mode] = tp.run_pipeline(pipeline_config, corpus)

    yield _RESULTS[mode]


def _rows(result: pd.DataFrame, telemetry_class: str) -> pd.DataFrame:
    return result[result["telemetry_class"] == telemetry_class]


def _windows(frame: pd.DataFrame) -> list[int]:
    return [window_id_from_timestamp(int(t), tp.PERIOD_SECONDS * NS) for t in frame["event_time"]]


def _permuted(corpus: dict[str, pd.DataFrame], seed: int) -> dict[str, pd.DataFrame]:
    return {
        name: permute_within_contiguous_groups(frame, _windows(frame), seed=seed)
        for (name, frame) in corpus.items()
    }


# --- Check 1: the corpus -------------------------------------------------------------------------------------------


def test_corpus_is_fixed(corpus: dict[str, pd.DataFrame]):
    again = tp.build_corpus()

    assert set(again) == set(corpus)

    for name in corpus:
        pd.testing.assert_frame_equal(corpus[name], again[name])


def test_corpus_is_shaped_like_the_network(corpus: dict[str, pd.DataFrame]):
    # The shape the retrospective found the unit tests lacked. A snapshot stamps every address at one instant.
    snapshots = corpus["tc2_mac"]
    per_instant = snapshots.groupby("event_time").size()
    assert per_instant.max() >= 5

    # A flood lands on one second.
    arp = corpus["tc2_arp"]
    assert (arp["event_time"] == tp.FLOOD_AT_SECONDS * NS).sum() >= tp.FLOOD_PACKETS

    # SNMP time is in hundredths of a second, and the reboot makes uptime go backwards.
    layer_1 = corpus["tc1"]
    rebooting = layer_1[layer_1["device_id"] == tp.REBOOTING_SWITCH]["uptime"]
    assert (rebooting.diff().dropna() < 0).any()

    # Every 802.1X event names the device being authorized. Without it the stage falls back to timing exchanges
    # per port, which is the degraded mode this corpus ran in and the reason the keying went uncovered.
    assert corpus["tc2_auth"]["mac_address"].notna().all()
    assert corpus["tc2_auth"].groupby("port_id")["mac_address"].nunique().max() > 1, "no port carries two devices"

    # Every class carries the envelope the identifiers derive from, with a monotonic sequence.
    for frame in corpus.values():
        assert set(tp.ID_COLUMNS) <= set(frame.columns)
        assert frame["collector_seq"].is_monotonic_increasing


# --- Checks 2 through 6: determinism ----------------------------------------------------------------------------


@pytest.mark.gpu_and_cpu_mode
def test_double_run_diff(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    second = tp.run_pipeline(pipeline_config, corpus)

    assert diff_frames(result, second) is None
    assert frame_digest(result) == frame_digest(second)


@pytest.mark.slow
@pytest.mark.gpu_and_cpu_mode
def test_cross_restart_diff(execution_mode, tmp_path):
    # The driver takes the mode as an argument, because this check is the one that cannot share a process with the
    # test that asks for it: a fresh interpreter is the whole point, so the mode has to travel on the command line.
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
    # The rendering is compared byte for byte, the same way the cross-restart check compares it, so the nullable
    # integer and boolean columns never pass through a CSV-to-dtype round trip. On a mismatch, both sides are read
    # back as text so the first differing cell is named.
    with open(GOLDEN_PATH, encoding="utf-8") as handle:
        golden_text = handle.read()

    rendered = tp.render(result)

    if (rendered != golden_text):
        from io import StringIO
        as_text = {"dtype": str, "keep_default_na": False}
        difference = diff_frames(pd.read_csv(StringIO(rendered), **as_text),
                                 pd.read_csv(StringIO(golden_text), **as_text))

        pytest.fail(f"Output drifted from {os.path.basename(GOLDEN_PATH)}: {difference}. If the change is intended, "
                    f"regenerate the golden file with {os.path.basename(DRIVER_PATH)} and review the diff.")


@pytest.mark.gpu_and_cpu_mode
def test_batch_split_sweep(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    # Control 5, batching must be irrelevant: one frame, three frames, and one frame per row, per class.
    def split(frame: pd.DataFrame, parts: int) -> list[pd.DataFrame]:
        size = max(1, len(frame) // parts)
        return [frame.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(frame), size)]

    thirds = {name: split(frame, 3) for (name, frame) in corpus.items()}
    by_row = {name: split(frame, len(frame)) for (name, frame) in corpus.items()}

    assert diff_frames(result, tp.run_pipeline(pipeline_config, corpus, batches=thirds)) is None
    assert diff_frames(result, tp.run_pipeline(pipeline_config, corpus, batches=by_row)) is None


@pytest.mark.gpu_and_cpu_mode
def test_permutation_check_has_teeth(pipeline_config: Config, corpus: dict[str, pd.DataFrame]):
    # The negative control for check 6. Remove the total-order stage and the counter deltas, the binding intervals
    # and the distinct counts all become functions of arrival order. If this test ever passes without a diff, the
    # permutation check has stopped proving anything.
    unsorted_baseline = tp.run_pipeline(pipeline_config, corpus, impose_order=False)

    detected = False
    for seed in (1, 2, 3):
        detected = detected or (diff_frames(
            unsorted_baseline, tp.run_pipeline(pipeline_config, _permuted(corpus, seed), impose_order=False))
                                is not None)

    assert detected, "Removing the total-order stage did not change any output under permutation."


@pytest.mark.gpu_and_cpu_mode
def test_permutation_within_windows(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    # Check 6, and the direct test for control 8 on the telemetry stages: shuffling rows within a window must not
    # change the output once the total-order stage is in front of them.
    for seed in (1, 2, 3):
        shuffled = _permuted(corpus, seed)

        assert any(not shuffled[name]["collector_seq"].equals(corpus[name]["collector_seq"])
                   for name in corpus), "permutation was a no-op"

        difference = diff_frames(result, tp.run_pipeline(pipeline_config, shuffled))
        assert difference is None, f"seed {seed}: {difference}"


# --- The composition ---------------------------------------------------------------------------------------------


@pytest.mark.cpu_mode
def test_a_chain_is_one_entity_inside_one_window(result: pd.DataFrame):
    # `lineage_id` is the field all four shipped detections already select, and until now nothing populated it.
    # What it has to mean: the events that would be correlated together carry one identifier, and events that
    # would not, do not. Anything weaker makes the column decoration.
    for (name, anchor) in tp.CHAIN_ANCHORS.items():
        rows = _rows(result, name)
        chained = rows[rows["lineage_id"].notna()]

        assert len(chained) > 0, name

        per_window = chained.groupby([anchor, "window_id"])["lineage_id"].nunique()
        assert per_window.max() == 1, f"{name}: one entity in one window carried more than one chain"

        # And the identifier separates windows, or a rule reading it would correlate across a boundary the
        # windowing exists to draw.
        spanning = chained.groupby(anchor)["window_id"].nunique()

        if ((spanning > 1).any()):
            entity = spanning[spanning > 1].index[0]
            walked = chained[chained[anchor] == entity]

            assert walked["lineage_id"].nunique() == walked["window_id"].nunique(), (
                f"{name}: {entity} carried the same chain across two windows")


@pytest.mark.cpu_mode
def test_the_class_that_is_not_sealed_carries_no_chain(result: pd.DataFrame):
    # The bindings never enter a window, so there is no chain for them to belong to. A blank column is the honest
    # answer; a fabricated one would be worse than the gap this step set out to close.
    assert _rows(result, "tc2_binding")["lineage_id"].isna().all()


@pytest.mark.cpu_mode
def test_a_chain_survives_a_permutation_of_its_own_events(corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    # The chain root is a Merkle root over the members, which deduplicates and sorts, so membership rather than
    # arrival order decides it. Control 13's own permutation check covers the whole frame; this says the property
    # the identifier depends on is the reason.
    shuffled = tp.run_pipeline(tp.build_pipeline_config(), _permuted(corpus, 1))

    for name in tp.CHAIN_ANCHORS:
        before = set(_rows(result, name)["lineage_id"].dropna())
        after = set(_rows(shuffled, name)["lineage_id"].dropna())

        assert before == after, f"{name}: permuting the input changed which chains exist"


@pytest.mark.cpu_mode
def test_a_mac_resolves_to_the_port_layer_1_knows(result: pd.DataFrame):
    # The ladder's first arrow, end to end: layer 2 closed the binding, the table was built from it, the ARP stream
    # resolved through it, and the port it landed on is a string layer 1 actually emitted for that port.
    arp = _rows(result, "tc2_arp")
    flood = arp[(arp["arp_sender_mac"] == tp.MAC_A) & (arp["event_time"] == tp.FLOOD_AT_SECONDS * NS)]
    layer_1_ports = set(_rows(result, "tc1")["entity_key"].dropna())

    assert len(flood) == tp.FLOOD_PACKETS
    assert set(flood["resolved_port_key"]) == {f"{tp.SITE}:{tp.SWITCH}:Gi1/0/1"}
    assert set(flood["resolved_port_key"]) <= layer_1_ports
    assert set(flood["resolution_method"]) == {"soft:mac_table"}
    assert flood["binding_uid"].notna().all()


@pytest.mark.cpu_mode
def test_every_class_ran_and_sealed(result: pd.DataFrame):
    assert set(result["telemetry_class"]) == set(tp.TELEMETRY_CLASSES)

    for name in ("tc1", "tc2_mac", "tc2_arp", "tc2_auth"):
        sealed = _rows(result, name)
        assert sealed["window_id"].notna().all(), name
        assert (sealed["is_late"] == False).all(), name  # noqa: E712  pylint: disable=singleton-comparison


# --- The planted anomalies, as a rule would read them --------------------------------------------------------------


@pytest.mark.cpu_mode
def test_the_hub_is_visible(result: pd.DataFrame):
    port = f"{tp.SITE}:{tp.SWITCH}:{tp.HUB_PORT}"
    on_port = _rows(result, "tc2_mac")
    on_port = on_port[on_port["port_key"] == port]

    before = on_port[on_port["event_time"] < tp.HUB_FROM_SECONDS * NS]["macs_per_port"]
    after = on_port[on_port["event_time"] >= tp.HUB_FROM_SECONDS * NS]["macs_per_port"]

    assert before.max() == 1
    assert after.max() == 1 + len(tp.HUB_MACS)


@pytest.mark.cpu_mode
def test_the_spoof_is_a_conflict(result: pd.DataFrame):
    bindings = _rows(result, "tc2_binding")
    conflicts = bindings[bindings["bind_end_reason"] == CONFLICT]

    assert len(conflicts) == 1
    assert conflicts.iloc[0]["mac_address"] == tp.MAC_A
    assert conflicts.iloc[0]["port_key"] == f"{tp.SITE}:{tp.SWITCH}:Gi1/0/1"


@pytest.mark.cpu_mode
def test_the_flood_is_visible_in_full(result: pd.DataFrame):
    arp = _rows(result, "tc2_arp")
    flood = arp[(arp["arp_sender_mac"] == tp.MAC_A) & (arp["event_time"] == tp.FLOOD_AT_SECONDS * NS)]

    # All twenty counted. The window also holds the host's ordinary requests, so the proportion is the flood's
    # share of everything this host sent, not 1.0: that is the point of a proportion over a count. The gateway is
    # claimed by two MACs, the router and the host.
    last = flood.sort_values("collector_seq").iloc[-1]
    assert last["gratuitous_arp_count"] == tp.FLOOD_PACKETS
    assert last["arp_count_in_window"] > tp.FLOOD_PACKETS
    assert last["gratuitous_arp_ratio"] == pytest.approx(last["gratuitous_arp_count"] / last["arp_count_in_window"])
    assert last["gratuitous_arp_ratio"] > 0.3
    assert flood["macs_claiming_sender_ip"].max() == 2

    # The router announces itself once a minute, which is every packet it sends, but five in a window is below the
    # denominator floor, so no ratio is published for it: one-out-of-one is exactly the noise the floor removes.
    router = arp[arp["arp_sender_mac"] == tp.ROUTER_MAC]
    assert router["gratuitous_arp_ratio"].isna().all()


@pytest.mark.cpu_mode
def test_the_reboot_is_a_reset_not_a_wrap(result: pd.DataFrame):
    layer_1 = _rows(result, "tc1")
    rebooted = layer_1[layer_1["device_id"] == tp.REBOOTING_SWITCH]
    at_reboot = rebooted[rebooted["event_time"] == tp.REBOOT_AT_MINUTE * 60 * NS]

    assert (at_reboot["counter_reset"] == True).all()  # noqa: E712  pylint: disable=singleton-comparison
    assert (at_reboot["counter_wrapped"] == False).all()  # noqa: E712  pylint: disable=singleton-comparison
    # The interval is capped at the uptime, thirty seconds, not the sixty-second polling gap.
    assert at_reboot["interval_seconds"].iloc[0] == pytest.approx(30.0)
    assert (rebooted["counter_reset"] == True).sum() == 1  # noqa: E712  pylint: disable=singleton-comparison
    assert (at_reboot["link_flap_device_reset"] == True).all()  # noqa: E712  pylint: disable=singleton-comparison


@pytest.mark.cpu_mode
def test_the_tap_shows_as_lost_receive_power_only(result: pd.DataFrame):
    layer_1 = _rows(result, "tc1")
    port = f"{tp.SITE}:{tp.SWITCH}:{tp.HUB_PORT}"
    tapped = layer_1[(layer_1["entity_key"] == port) & (layer_1["event_time"] == tp.TAP_AT_MINUTE * 60 * NS)]

    assert tapped["optical_rx_dbm_deviation"].iloc[0] < -(tp.TAP_LOSS_DB - 0.5)
    assert abs(tapped["optical_tx_dbm_deviation"].iloc[0]) < 0.5


@pytest.mark.cpu_mode
def test_the_flap_between_polls_is_counted(result: pd.DataFrame):
    layer_1 = _rows(result, "tc1")
    port = f"{tp.SITE}:{tp.SWITCH}:Gi1/0/1"
    flapped = layer_1[(layer_1["entity_key"] == port) & (layer_1["event_time"] == tp.FLAP_AT_MINUTE * 60 * NS)]

    # The state read "up" on both sides of the gap. Only the device's own ifLastChange reveals the transition.
    assert flapped["link_flaps"].iloc[0] >= 1
    assert (flapped["link_flap_unpolled"] == True).all()  # noqa: E712  pylint: disable=singleton-comparison
    assert flapped["oper_status"].iloc[0] == "up"


@pytest.mark.cpu_mode
def test_the_bypass_is_an_unpaired_authorization(result: pd.DataFrame):
    auth = _rows(result, "tc2_auth")
    unpaired = auth[auth["auth_unpaired"] == True]  # noqa: E712  pylint: disable=singleton-comparison

    # Two authorizations with no exchange of their own: the one on a quiet port, and the one that arrived while a
    # legitimate exchange on its port was still open. The second is the harder case and the reason exchanges are
    # keyed by device: timed per port it would have paired with the innocent device's request and vanished.
    assert len(unpaired) == 2

    by_time = unpaired.sort_values("event_time")

    assert list(by_time["event_time"]) == [tp.BYPASS_AT_SECONDS * NS, (tp.CONCURRENT_BYPASS_AT_SECONDS + 10) * NS]
    assert list(by_time["auth_port_key"]) == [
        f"{tp.SITE}:{tp.SWITCH}:{tp.BYPASS_PORT}",
        f"{tp.SITE}:{tp.SWITCH}:{tp.CONCURRENT_BYPASS_PORT}",
    ]
    # The alert has to say which device, or an analyst cannot tell a bypass from the legitimate session beside it.
    assert list(by_time["mac_address"]) == [tp.BYPASS_MAC, tp.CONCURRENT_BYPASS_MAC]


@pytest.mark.cpu_mode
def test_the_legitimate_device_beside_a_bypass_is_timed_against_its_own_request(result: pd.DataFrame):
    # The other half of the same claim. The rogue's authorization must not become the legitimate device's outcome,
    # so the exchange that opened before it still closes correctly, twenty seconds later, on its own request.
    auth = _rows(result, "tc2_auth")
    legitimate = auth[(auth["mac_address"] == tp.LEGIT_SUPPLICANT_MAC) & (auth["dot1x_result"] == "success")]

    assert len(legitimate) == 1
    assert legitimate.iloc[0]["auth_unpaired"] == False  # noqa: E712  pylint: disable=singleton-comparison
    assert legitimate.iloc[0]["auth_elapsed_seconds"] == 20.0


@pytest.mark.cpu_mode
def test_a_shared_port_does_not_report_its_own_devices_as_bypasses(result: pd.DataFrame):
    # A phone with a workstation behind it is a standard access configuration, not an anomaly. Their outcomes
    # arrive in the other order, and each must be timed against its own request: the workstation's twelve-second
    # exchange is not the phone's, and neither one is unpaired. Keyed by port this fired once per reauthentication.
    auth = _rows(result, "tc2_auth")
    shared = auth[auth["port_id"] == tp.MULTI_DOMAIN_PORT]
    outcomes = shared[shared["dot1x_result"] == "success"].sort_values("mac_address")

    assert len(shared) == 4
    assert (shared["auth_unpaired"] != True).all()  # noqa: E712  pylint: disable=singleton-comparison
    assert list(outcomes["mac_address"]) == [tp.PHONE_MAC, tp.DESK_MAC]
    # The phone requested first and was authorized last: twelve seconds for it, ten for the workstation. Swapping
    # these is the defect, so they are asserted per device rather than as a set.
    assert list(outcomes["auth_elapsed_seconds"]) == [12.0, 10.0]
    assert list(outcomes["auth_attempts"]) == [1, 1]


@pytest.mark.cpu_mode
def test_nothing_else_fired(result: pd.DataFrame):
    # Precision on the planted corpus: the anomalies above are the only ones.
    layer_1 = _rows(result, "tc1")
    assert (layer_1["counter_reset"] == True).sum() == 1  # noqa: E712  pylint: disable=singleton-comparison

    # A tap is a persistent step, so the deviation persists until the baseline rolls over: every sample at or after
    # the tap, on the tapped port, and nowhere else.
    lost_light = layer_1[layer_1["optical_rx_dbm_deviation"] < -1.0]
    assert set(lost_light["entity_key"]) == {f"{tp.SITE}:{tp.SWITCH}:{tp.HUB_PORT}"}
    assert lost_light["event_time"].min() == tp.TAP_AT_MINUTE * 60 * NS
    assert len(lost_light) == tp.CORPUS_SECONDS // 60 - tp.TAP_AT_MINUTE + 1

    bindings = _rows(result, "tc2_binding")
    assert (bindings["bind_end_reason"] == CONFLICT).sum() == 1

    # The gateway is contested from the flood until the flood leaves the window, whoever is sending: the router's
    # own announcements in that span read two claimants too, which is the right input for R-D-L2-003. Nothing else
    # is ever contested.
    arp = _rows(result, "tc2_arp")
    contested = arp[arp["macs_claiming_sender_ip"].fillna(0) > 1]
    excluded = contested[contested["arp_sender_ip_excluded"] == True]  # noqa: E712  pylint: disable=singleton-comparison
    live = contested[contested["arp_sender_ip_excluded"] == False]  # noqa: E712  pylint: disable=singleton-comparison

    assert set(live["arp_sender_ip"]) == {tp.GATEWAY_IP}
    assert live["event_time"].min() == tp.FLOOD_AT_SECONDS * NS
    assert live["event_time"].max() < (tp.FLOOD_AT_SECONDS + tp.PERIOD_SECONDS) * NS
    # The redundancy pair is contested for the whole corpus and excluded for the whole corpus: visible, marked, and
    # not a detection.
    assert set(excluded["arp_sender_ip"]) == {tp.VRRP_IP}
    assert len(excluded) > 0
