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
    for column in ("src_ip", "dst_ip", "dst_port", "protocol", "ip_ttl", "bytes_out", "bytes_in", "bgp_as_dst"):
        assert column in flows.columns, column

    assert set(np_.ID_COLUMNS) <= set(flows.columns)
    assert flows["collector_seq"].is_monotonic_increasing
    assert flows["event_time"].is_monotonic_increasing

    # Four hours, so an hourly window has three predecessors and a trajectory has somewhere to go.
    span_hours = (flows["event_time"].max() - flows["event_time"].min()) / (np_.NS_PER_SECOND * 3600)
    assert span_hours > 3


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
    peaks = result.groupby(["src_ip", "window_id"])["dsts_per_src"].max().unstack(fill_value=0)

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

    assert shifted["window_id"].min() >= np_.TAP_AT_HOUR


@pytest.mark.cpu_mode
def test_every_row_is_stamped_at_layer_three_and_keyed_on_its_source(result: pd.DataFrame):
    # Layer 3's entity is the address, which is what Part 2 names and what a chain at this layer roots on.
    assert set(result["osi_layer"]) == {np_.OSI_LAYER}
    assert set(result["telemetry_class"]) == {np_.TELEMETRY_CLASS}
    assert list(result["entity_key"]) == list(result["src_ip"])
    assert result["lineage_id"].notna().all()
    assert result["window_id"].notna().all()
