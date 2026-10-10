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
Control 13's six checks against the composed layer 3 pipeline, and the five rules' predicates over its corpus.

Layer 3 is the first layer in this fork above 2, and the first whose entity is an address rather than a piece of
hardware or a person. Everything the other harnesses assert about determinism applies here unchanged; what is new
is what the corpus has to prove, which is that each rule fires on the thing it names and stays quiet on the thing
beside it. Both halves are asserted for all five, because a fan-out rule that also flagged the browser would be
worse than no rule -- volume nobody can triage is how a detection stops being read at all.
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
import network_pipeline as np_  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_network_expected.csv")
DRIVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_network_pipeline.py")

BEACON_CV_THRESHOLD = 0.15
"""R-B-L3-002's own figure, repeated so these tests evaluate the rule rather than something near it."""

INTERNAL_MAJORITY = 0.5
"""What "predominantly internal" is taken to mean. Named here because the guide does not give a number, and a
rule whose threshold lives only in prose is a rule two people implement differently."""

FAN_OUT_FLOOR = 50
"""The literal R-B-L3-001 shipped with, which it now keeps as a floor beneath the step."""

FAN_IN_FLOOR = 10
"""R-B-L3-006's floor: sources a workstation must be reached by before a step is a finding."""

UPLOAD_FLOOR = 10
"""R-B-L3-007's floor: bytes out for each byte back."""

SWEEP_PORTS = 25
"""R-D-L3-008's threshold."""


@pytest.fixture(name="corpus", scope="module")
def corpus_fixture() -> dict[str, pd.DataFrame]:
    yield np_.build_corpus()


# One composed run per execution mode, reused by every check in that mode.
_RESULTS: dict = {}


@pytest.fixture(name="pipeline_config")
def pipeline_config_fixture(execution_mode) -> Config:
    yield np_.build_pipeline_config(execution_mode)


@pytest.fixture(name="result")
def result_fixture(pipeline_config: Config, corpus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mode = pipeline_config.execution_mode

    if (mode not in _RESULTS):
        _RESULTS[mode] = np_.run_pipeline(pipeline_config, corpus)

    yield _RESULTS[mode]


def _for(result: pd.DataFrame, source: str) -> pd.DataFrame:
    return result[result["src_ip"] == source]


def _to(result: pd.DataFrame, destination: str) -> pd.DataFrame:
    return result[result["dst_ip"] == destination]


def _true(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("boolean").fillna(False).astype(bool)


def _scored(result: pd.DataFrame) -> pd.DataFrame:
    return result[result["event_time"].astype("int64") >= np_.at(0)]


def _history(result: pd.DataFrame) -> pd.DataFrame:
    return result[result["event_time"].astype("int64") < np_.at(0)]


def _scored_window(hour: int) -> int:
    return window_id_from_timestamp(np_.at(hour), np_.PERIOD_SECONDS * np_.NS_PER_SECOND)


def fan_out_expansion(frame: pd.DataFrame) -> pd.DataFrame:
    """R-B-L3-001's predicate, row by row, as the stanza states it."""
    return frame[_true(frame, "dsts_per_src_baseline_mature") & (frame["dsts_per_src_step"] > 0)
                 & (frame["dsts_per_src"] > FAN_OUT_FLOOR) & (frame["internal_dst_ratio"] > INTERNAL_MAJORITY)]


def fan_in_onto_a_workstation(frame: pd.DataFrame) -> pd.DataFrame:
    """R-B-L3-006's predicate."""
    return frame[(frame["dst_ctx_device_role"] == "workstation").fillna(False)
                 & _true(frame, "srcs_per_dst_baseline_mature") & (frame["srcs_per_dst_step"] > 0)
                 & (frame["srcs_per_dst"] >= FAN_IN_FLOOR)]


def first_contact_upload(frame: pd.DataFrame) -> pd.DataFrame:
    """R-B-L3-007's predicate."""
    return frame[_true(frame, "dst_asn_first_seen") & _true(frame, "byte_asymmetry_baseline_mature")
                 & (frame["byte_asymmetry_step"] > 0) & (frame["byte_asymmetry"] >= UPLOAD_FLOOR)]


def port_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    """R-D-L3-008's predicate."""
    return frame[(frame["dst_ports_per_src"] >= SWEEP_PORTS) & _true(frame, "dst_is_private")]


def _split(frame: pd.DataFrame, parts: int) -> list:
    size = max(1, len(frame) // parts)

    return [frame.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(frame), size)]


def _windows(frame: pd.DataFrame) -> list:
    period_ns = np_.PERIOD_SECONDS * np_.NS_PER_SECOND

    return [window_id_from_timestamp(int(stamp), period_ns) for stamp in frame["event_time"]]


def _permuted(corpus: dict, seed: int) -> dict:
    return {
        name: permute_within_contiguous_groups(frame, _windows(frame), seed=seed)
        for (name, frame) in corpus.items()
    }


# --- Check 1: the corpus is fixed and shaped like a flow exporter ----------------------------------------------


def test_corpus_is_fixed(corpus: dict[str, pd.DataFrame]):
    again = np_.build_corpus()

    assert set(again) == set(corpus)

    for name in corpus:
        pd.testing.assert_frame_equal(corpus[name], again[name])


def test_corpus_is_shaped_like_a_flow_exporter(corpus: dict[str, pd.DataFrame]):
    flows = corpus[np_.TELEMETRY_CLASS]

    # One record per flow, carrying both directions' byte counts and the fields Part 2 lists as required.
    for column in ("src_ip",
                   "src_port",
                   "dst_ip",
                   "dst_port",
                   "protocol",
                   "ip_ttl",
                   "bytes_out",
                   "bytes_in",
                   "bgp_as_dst"):
        assert column in flows.columns, column

    assert set(np_.ID_COLUMNS) <= set(flows.columns)
    assert flows["collector_seq"].is_monotonic_increasing
    assert flows["event_time"].is_monotonic_increasing

    # A fortnight of history, then four scored hours, so an hourly window has three predecessors, a trajectory has
    # somewhere to go and a host has a fortnight to be measured against.
    first_scored = flows[flows["event_time"] >= np_.at(0)]
    span_hours = (first_scored["event_time"].max() - first_scored["event_time"].min()) / (np_.NS_PER_SECOND * 3600)
    assert span_hours > 3
    assert flows["event_time"].min() < np_.history_at(1, 0)


def test_the_documentation_ranges_are_not_used_for_the_public_internet():
    # The trap this corpus fell into once. `192.0.2.0/24`, `198.51.100.0/24` and `203.0.113.0/24` are reserved
    # for documentation, so the classifier calls them private -- and a corpus built from them reports a browser
    # on the public internet as a host that never left the estate, which is the exact condition R-B-L3-001
    # distinguishes a scan by. The external hosts here are outside those ranges for that reason.
    from morpheus.parsers import ip  # pylint: disable=import-outside-toplevel

    external = pd.Series(list(np_.EXTERNAL_HOSTS) + [np_.BEACON_SERVER])

    assert all(ip.is_global(external))
    assert not any(ip.is_private(external))


# --- Checks 2 through 6: determinism ---------------------------------------------------------------------------


@pytest.mark.gpu_and_cpu_mode
def test_double_run_diff(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    second = np_.run_pipeline(pipeline_config, corpus)

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

    rendered = np_.render(result)

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

    assert diff_frames(result, np_.run_pipeline(pipeline_config, corpus, batches=thirds)) is None
    assert diff_frames(result, np_.run_pipeline(pipeline_config, corpus, batches=by_row)) is None


@pytest.mark.gpu_and_cpu_mode
def test_permutation_within_windows(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    for seed in (1, 2, 3):
        shuffled = np_.run_pipeline(pipeline_config, _permuted(corpus, seed))

        assert diff_frames(result, shuffled) is None, seed


@pytest.mark.gpu_and_cpu_mode
def test_permutation_check_has_teeth(pipeline_config: Config, corpus: dict):
    # The negative control for the check above. Every stage here is cumulative -- a count, a proportion, a
    # rhythm and a reference are all functions of what came before -- so removing the imposed order has to
    # change the answer, or the check is passing over a pipeline that never needed it.
    ordered = np_.run_pipeline(pipeline_config, corpus, impose_order=False)
    shuffled = np_.run_pipeline(pipeline_config, _permuted(corpus, 5), impose_order=False)

    assert diff_frames(ordered, shuffled) is not None


# --- The five rules, each with the case beside it that must stay quiet ------------------------------------------


@pytest.mark.cpu_mode
def test_the_scan_fans_out_and_the_browser_does_not(result: pd.DataFrame):
    # R-B-L3-001. Both hosts reach a comparable number of addresses; only one of them reaches them inside the
    # estate, and that is the whole of the difference between a scan and a lunch break.
    scanner = _for(result, np_.SCANNER)
    browser = _for(result, np_.BROWSER)

    assert scanner["dsts_per_src"].max() == np_.SCAN_PER_HOUR[-1]
    assert browser["dsts_per_src"].max() == np_.BROWSE_PER_HOUR[-1]

    assert scanner["internal_dst_ratio"].max() > INTERNAL_MAJORITY
    assert browser["internal_dst_ratio"].max() < INTERNAL_MAJORITY


@pytest.mark.cpu_mode
def test_the_scan_rises_across_windows_and_the_browser_holds_level(result: pd.DataFrame):
    # R-P-L3-005 reads the shape rather than the level, and fires during the expansion rather than at the peak.
    # A corpus whose scan arrived all at once would satisfy R-B-L3-001 and say nothing about this.
    peaks = _scored(result).groupby(["src_ip", "window_id"])["dsts_per_src"].max().unstack(fill_value=0)

    scan = list(peaks.loc[np_.SCANNER])
    browse = list(peaks.loc[np_.BROWSER])

    assert scan == list(np_.SCAN_PER_HOUR)
    assert all(later > earlier for (earlier, later) in zip(scan, scan[1:]))
    assert browse == list(np_.BROWSE_PER_HOUR)
    assert len(set(browse)) == 1


@pytest.mark.cpu_mode
def test_the_beacon_is_regular_and_the_worker_is_not(result: pd.DataFrame):
    # R-B-L3-002, both halves. The worker makes plenty of flows to one server, which is what a coefficient over
    # a count rather than over the intervals would have called a beacon.
    beacon = _for(result, np_.BEACONER)
    worker = _for(result, np_.WORKER)

    mature_beacon = beacon[beacon["flow_regularity_mature"].astype(bool)]
    mature_worker = worker[worker["flow_regularity_mature"].astype(bool)]

    assert len(mature_beacon) > 0
    assert len(mature_worker) > 0

    assert mature_beacon["flow_interval_cv"].max() < BEACON_CV_THRESHOLD
    assert mature_beacon["flow_size_cv"].max() < BEACON_CV_THRESHOLD
    assert mature_worker["flow_interval_cv"].min() > BEACON_CV_THRESHOLD


@pytest.mark.cpu_mode
def test_the_beacon_is_one_pair_rather_than_one_host(result: pd.DataFrame):
    # The reason the rhythm is keyed on the directed pair. Every flow the beaconing host makes is to one server,
    # and the key on the row says so rather than the row merely belonging to a host that happens to beacon.
    beacon = _for(result, np_.BEACONER)

    assert set(beacon["flow_pair_key"]) == {f"{np_.BEACONER}:{np_.BEACON_SERVER}"}


@pytest.mark.cpu_mode
def test_the_stray_flow_to_a_reserved_range_is_the_only_one(result: pd.DataFrame):
    # R-D-L3-003. Low volume and high signal by nature, which is only true if the classification is right about
    # everything else -- so the assertion is that exactly one flow in the corpus is reserved, not that one is.
    reserved = result[result["dst_is_reserved"].astype("boolean").fillna(False)]

    assert len(reserved) == 1
    assert list(reserved["src_ip"]) == [np_.STRAY]
    assert list(reserved["dst_ip"]) == [np_.RESERVED_DESTINATION]


@pytest.mark.cpu_mode
def test_the_tapped_host_loses_a_hop_and_the_quiet_one_does_not(result: pd.DataFrame):
    # R-B-L3-004. One hop is what an interposed device costs, which is why the threshold is one and not more.
    tapped = _for(result, np_.TAPPED)
    quiet = _for(result, np_.QUIET)

    shifted = tapped[tapped["ip_ttl_shifted"].astype("boolean").fillna(False)]

    assert len(shifted) > 0
    assert set(shifted["ip_ttl_shift"]) == {-1}
    assert not any(quiet["ip_ttl_shifted"].astype("boolean").fillna(False))


@pytest.mark.cpu_mode
def test_the_hop_is_lost_when_the_device_arrives_and_not_before(result: pd.DataFrame):
    # The reference is trailing, so a genuine change is an anomaly while it is new and a fact afterwards. What
    # must not happen is the shift being reported in the hours before the device existed.
    tapped = _for(result, np_.TAPPED)
    shifted = tapped[tapped["ip_ttl_shifted"].astype("boolean").fillna(False)]

    assert shifted["window_id"].min() >= _scored_window(np_.TAP_AT_HOUR)


@pytest.mark.cpu_mode
def test_every_row_is_stamped_at_layer_three_and_keyed_on_its_source(result: pd.DataFrame):
    # Layer 3's entity is the address, which is what Part 2 names and what a chain at this layer roots on.
    assert set(result["osi_layer"]) == {np_.OSI_LAYER}
    assert set(result["telemetry_class"]) == {np_.TELEMETRY_CLASS}
    assert list(result["entity_key"]) == list(result["src_ip"])
    assert result["lineage_id"].notna().all()
    assert result["window_id"].notna().all()


# --- Measured against the host's own fortnight -----------------------------------------------------------------


@pytest.mark.cpu_mode
def test_the_history_is_quiet(result: pd.DataFrame):
    # The fortnight is ordinary hosts on ordinary days, and the corpus is only a test of the rules if nothing in
    # it fires. Every predicate the harness evaluates is checked over it, the literal R-B-L3-001 used to read
    # included, which fired on every Monday the DHCP server had.
    history = _history(result)

    assert len(fan_out_expansion(history)) == 0
    assert len(fan_in_onto_a_workstation(history)) == 0
    assert len(first_contact_upload(history)) == 0
    assert len(port_sweep(history)) == 0
    assert not _true(history, "dst_is_reserved").any()
    assert not _true(history, "ip_ttl_shifted").any()

    literal = history[(history["dsts_per_src"] > FAN_OUT_FLOOR) & (history["internal_dst_ratio"] > INTERNAL_MAJORITY)]
    assert set(literal["src_ip"]) == {np_.DHCP_SERVER}


@pytest.mark.cpu_mode
def test_every_host_with_a_fortnight_has_a_mature_baseline(result: pd.DataFrame):
    first_hour = result[result["window_id"] == _scored_window(0)]

    for host in (np_.SCANNER, np_.DHCP_SERVER, np_.UPLOADER):
        rows = _for(first_hour, host)
        assert len(rows) > 0, host
        assert _true(rows, "dsts_per_src_baseline_mature").all(), host

    for host in (np_.DNS_SERVER, np_.REACHED_WORKSTATION, np_.APP_SERVER):
        rows = _to(result[result["window_id"] >= _scored_window(0)], host)
        assert _true(rows, "srcs_per_dst_baseline_mature").all(), host


@pytest.mark.cpu_mode
def test_the_scan_is_a_step_and_the_dhcp_server_is_not(result: pd.DataFrame):
    # R-B-L3-001. Both hosts reach more than fifty internal addresses in a scored hour; only the scanner has never
    # done so before. The DHCP server's Monday checks reach sixty, and its own Mondays reach sixty-four.
    scanner = _for(result, np_.SCANNER)
    dhcp = _scored(_for(result, np_.DHCP_SERVER))

    assert dhcp["dsts_per_src"].max() == np_.DHCP_CHECKS_TODAY
    assert set(dhcp["dsts_per_src_baseline_max"].dropna()) >= {np_.DHCP_CHECKS_MONDAY}
    assert (dhcp["dsts_per_src_step"].dropna() <= 0).all()

    # The scanner's fortnight never reached more than its three servers; each scored hour joins the history as the
    # next one opens, so the last hour is measured against the forty-five before it.
    first_hour = scanner[scanner["window_id"] == _scored_window(0)]
    assert set(first_hour["dsts_per_src_baseline_max"]) == {len(np_.SCANNER_SERVERS)}

    fired = fan_out_expansion(result)
    assert set(fired["src_ip"]) == {np_.SCANNER}
    assert set(fired["window_id"]) == {_scored_window(3)}


@pytest.mark.cpu_mode
def test_a_workstation_reached_by_many_and_the_servers_that_are_not_findings(result: pd.DataFrame):
    # R-B-L3-006, both controls. The DNS server is reached by eighteen clients this morning and by twenty on its
    # own Mondays, so it never steps; the application server steps when a new service is pointed at it, and is a
    # server. Only the workstation, reached by fourteen where its fortnight held one, fires.
    dns = _scored(_to(result, np_.DNS_SERVER))
    app = _scored(_to(result, np_.APP_SERVER))
    workstation = _scored(_to(result, np_.REACHED_WORKSTATION))

    assert (dns["srcs_per_dst_step"].dropna() <= 0).all()
    assert app["srcs_per_dst_step"].max() > 0
    assert set(app["dst_ctx_device_role"]) == {"server"}
    assert workstation["srcs_per_dst"].max() == np_.REACHING_SOURCES
    assert workstation["srcs_per_dst_baseline_max"].max() == 1

    fired = fan_in_onto_a_workstation(result)
    assert set(fired["dst_ip"]) == {np_.REACHED_WORKSTATION}
    assert set(fired["srcs_per_dst"]) == set(range(FAN_IN_FLOOR, np_.REACHING_SOURCES + 1))


@pytest.mark.cpu_mode
def test_a_first_contact_upload_and_each_half_alone(result: pd.DataFrame):
    # R-B-L3-007. The upload to a network the host has never reached fires; the download from another new network
    # is first contact without the shape, and the heavier nightly copy is the shape without first contact.
    uploader = _scored(_for(result, np_.UPLOADER))
    by_network = uploader.set_index("bgp_as_dst")

    assert _true(uploader, "dst_asn_first_seen").sum() == 2
    assert by_network.loc[np_.DOWNLOAD_ASN, "byte_asymmetry_step"] < 0
    assert not bool(by_network.loc[np_.BACKUP_ASN, "dst_asn_first_seen"])
    assert by_network.loc[np_.BACKUP_ASN, "byte_asymmetry_step"] > 0

    fired = first_contact_upload(result)
    assert list(fired["src_ip"]) == [np_.UPLOADER]
    assert list(fired["bgp_as_dst"]) == [np_.UPLOAD_ASN]


@pytest.mark.cpu_mode
def test_a_port_sweep_inside_the_estate_and_media_outside_it(result: pd.DataFrame):
    # R-D-L3-008. Both hosts reach more than twenty-five ports in the hour; the conferencing client's are on a
    # relay on the internet. The horizontal scan reaches eighty addresses on one port and is not this rule's.
    assert _for(result, np_.CONFERENCING)["dst_ports_per_src"].max() == np_.MEDIA_PORTS
    assert _for(result, np_.SCANNER)["dst_ports_per_src"].max() == 1

    fired = port_sweep(result)
    assert set(fired["src_ip"]) == {np_.SWEEPER}
    assert fired["dst_ports_per_src"].max() == np_.SWEEP_PORTS


@pytest.mark.cpu_mode
def test_every_flow_carries_its_community_id(result: pd.DataFrame):
    # The equality join key across layers 3, 4 and 6. Every flow here has the ports it needs, so none is null.
    assert result["community_id"].notna().all()
    assert result["community_id"].str.startswith("1:").all()
