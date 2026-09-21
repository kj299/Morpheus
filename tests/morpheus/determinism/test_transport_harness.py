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
Control 13's six checks against the composed layer 4 pipeline, and the rules' predicates over its corpus.

Layer 4 is the first layer in this fork whose entity is a conversation rather than a thing -- a flow lives as
long as two endpoints keep talking and then stops existing -- and the first whose features were fixed by a model
someone else trained. Everything the other harnesses assert about determinism applies unchanged. What is new is
that three of the five rules read a *direction*: which way the SYNs go, which way the refusals come back, which
side of a triple is sending. A corpus that carried only one direction would let a rule pass while reading the
wrong one, so the responses here travel the way a capture sees them, and each rule is asserted against the case
beside it that must stay quiet.

Two of the five rules cannot be asserted here and are not pretended to be. R-B-L4-001 is the shipped
`abp-pcap-xgb` model behind Triton, so what this fork owes it is the thirteen features under the names it was
trained on, which `MODEL_FEATURES` carries and `test_the_model_feature_list_is_complete` pins. R-B-L4-004 needs
a layer 7 `user_agent` on the same `flow_id`, and layer 7 does not exist yet.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.stages.telemetry.tc4_flow_stage import MODEL_FEATURES
from morpheus.utils.determinism import diff_frames
from morpheus.utils.determinism import frame_digest
from morpheus.utils.determinism import permute_within_contiguous_groups
from morpheus.utils.lineage import window_id_from_timestamp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import transport_pipeline as tp_  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_transport_expected.csv")
DRIVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_transport_pipeline.py")

SYN_RATIO_THRESHOLD = 0.9
"""R-D-L4-002's own figure, repeated so these tests evaluate the rule rather than something near it."""

SCAN_PORT_THRESHOLD = 50
"""R-D-L4-002's own figure, likewise. The corpus reaches sixty, so the assertion is not on the threshold."""

RST_RATIO_THRESHOLD = 0.5
"""R-D-L4-003 says 0.8, and this is a deliberate departure recorded in the same place the app's detection
records it. A connection refused with RST+ACK puts exactly two flags on the flow that carries the refusal, so
`rst/all` for it is 0.5 and nothing a sweep or an outage produces can reach 0.8. The rule as written would never
fire on either of the two conditions it names. Half is what a flow carrying refusals and nothing else scores,
and a flow that also carried a handshake or data scores well below it, so the separation the rule wants is the
one this threshold makes."""

ENVELOPE_MULTIPLIER = 3.0
"""R-B-L4-005's own figure."""


@pytest.fixture(name="corpus", scope="module")
def corpus_fixture() -> dict[str, pd.DataFrame]:
    yield tp_.build_corpus()


# One composed run per execution mode, reused by every check in that mode.
_RESULTS: dict = {}


@pytest.fixture(name="pipeline_config")
def pipeline_config_fixture(execution_mode) -> Config:
    yield tp_.build_pipeline_config(execution_mode)


@pytest.fixture(name="result")
def result_fixture(pipeline_config: Config, corpus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mode = pipeline_config.execution_mode

    if (mode not in _RESULTS):
        _RESULTS[mode] = tp_.run_pipeline(pipeline_config, corpus)

    yield _RESULTS[mode]


def _bins(result: pd.DataFrame) -> pd.DataFrame:
    """One row per flow per bin, aggregated the way a detection has to aggregate it.

    The counts are running totals and rise monotonically through a bin, so their maxima are the bin's totals.
    The ratios are not monotone and are therefore recomputed from those maxima rather than aggregated -- which
    is the recipe every layer 4 detection in the app uses, and `test_a_search_divides_the_maxima_rather_than
    _aggregating_the_ratio` is the reason it has to.
    """
    bins = result.groupby(["flow_id", "rollup_time_ns"]).agg(src_ip=("src_ip", "first"),
                                                             dst_ip=("dst_ip", "first"),
                                                             dst_port=("dst_port", "first"),
                                                             syn=("flow_syn", "max"),
                                                             ack=("flow_ack", "max"),
                                                             rst=("flow_rst", "max"),
                                                             total=("flow_all", "max")).reset_index()

    bins["syn_ratio"] = bins["syn"] / bins["total"]
    bins["rst_ratio"] = bins["rst"] / bins["total"]

    return bins


def _split(frame: pd.DataFrame, parts: int) -> list:
    size = max(1, len(frame) // parts)

    return [frame.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(frame), size)]


def _windows(frame: pd.DataFrame) -> list:
    period_ns = tp_.PERIOD_SECONDS * tp_.NS_PER_SECOND

    return [window_id_from_timestamp(int(stamp), period_ns) for stamp in frame["event_time"]]


def _permuted(corpus: dict, seed: int) -> dict:
    return {
        name: permute_within_contiguous_groups(frame, _windows(frame), seed=seed)
        for (name, frame) in corpus.items()
    }


def _truth(result: pd.DataFrame, column: str) -> pd.Series:
    return result[column].astype("boolean").fillna(False)


# --- Check 1: the corpus is fixed and shaped like a capture ----------------------------------------------------


def test_corpus_is_fixed(corpus: dict[str, pd.DataFrame]):
    again = tp_.build_corpus()

    assert set(again) == set(corpus)

    for name in corpus:
        pd.testing.assert_frame_equal(corpus[name], again[name])


def test_corpus_is_shaped_like_a_capture(corpus: dict[str, pd.DataFrame]):
    packets = corpus[tp_.TELEMETRY_CLASS]

    # One record per packet, carrying the five-tuple, the flags byte and the payload length. This is the shape
    # `abp_pcap_preprocessing.py` assumes, and the assumption is load-bearing: `ppm` counts records, so a flow
    # exporter's one-record-per-flow feed would have the model reading flows per minute as packets per minute.
    for column in ("src_ip", "src_port", "dst_ip", "dst_port", "protocol", "tcp_flags", "data_len"):
        assert column in packets.columns, column

    assert set(tp_.ID_COLUMNS) <= set(packets.columns)
    assert packets["collector_seq"].is_monotonic_increasing
    assert packets["event_time"].is_monotonic_increasing

    # Two hours, so the exfiltration lands in a window after the one its baseline was built in.
    span_hours = (packets["event_time"].max() - packets["event_time"].min()) / (tp_.NS_PER_SECOND * 3600)
    assert span_hours > 1


def test_the_answers_travel_the_other_way(corpus: dict[str, pd.DataFrame]):
    # What the corpus would be wrong about if `respond` were `add`. A capture sees a refusal with the server as
    # its source, so it lands on the reverse flow; put it on the forward flow instead and the SYN dilutes the
    # ratio to a third, the fan-out direction inverts, and R-D-L4-003 reads the opposite of what happened.
    packets = corpus[tp_.TELEMETRY_CLASS]
    refusals = packets[packets["tcp_flags"] == tp_.RST_ACK]

    assert len(refusals) == tp_.SWEEP_TARGETS + len(tp_.OUTAGE_CLIENTS)
    assert not any(refusals["src_ip"] == tp_.SWEEPER)
    assert set(refusals[refusals["dst_ip"] == tp_.SWEEPER]["src_ip"]) == {
        f"10.0.2.{index + 1}"
        for index in range(tp_.SWEEP_TARGETS)
    }
    assert set(refusals[refusals["dst_ip"].isin(tp_.OUTAGE_CLIENTS)]["src_ip"]) == {tp_.FAILING_SERVER}


# --- Checks 2 through 6: determinism ---------------------------------------------------------------------------


@pytest.mark.gpu_and_cpu_mode
def test_double_run_diff(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    second = tp_.run_pipeline(pipeline_config, corpus)

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

    rendered = tp_.render(result)

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

    assert diff_frames(result, tp_.run_pipeline(pipeline_config, corpus, batches=thirds)) is None
    assert diff_frames(result, tp_.run_pipeline(pipeline_config, corpus, batches=by_row)) is None


@pytest.mark.gpu_and_cpu_mode
def test_a_bin_split_across_messages_still_ends_on_the_right_number(pipeline_config: Config, corpus: dict):
    # Control 5 aimed at the departure this stage makes from the reference. `abp_pcap_preprocessing.py` groups
    # within the message it was handed, so a sixty-second bin arriving in two messages is aggregated twice and
    # the model sees two partial bins as though each were whole. The check above says the whole frame does not
    # drift; this one says the figure a detection reads is the same number no matter how the stream was chunked.
    by_row = {name: _split(frame, len(frame)) for (name, frame) in corpus.items()}

    whole = _bins(tp_.run_pipeline(pipeline_config, corpus))
    chunked = _bins(tp_.run_pipeline(pipeline_config, corpus, batches=by_row))

    scan = whole[(whole["src_ip"] == tp_.SCANNER)]
    assert len(scan) == tp_.SCAN_PORTS

    pd.testing.assert_frame_equal(whole, chunked)


@pytest.mark.gpu_and_cpu_mode
def test_permutation_within_windows(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    for seed in (1, 2, 3):
        shuffled = tp_.run_pipeline(pipeline_config, _permuted(corpus, seed))

        assert diff_frames(result, shuffled) is None, seed


@pytest.mark.gpu_and_cpu_mode
def test_permutation_check_has_teeth(pipeline_config: Config, corpus: dict):
    # The negative control for the check above. Both stages are cumulative -- a running bin total and a trailing
    # quantile are each functions of what came before -- so removing the imposed order has to change the answer,
    # or the check is passing over a pipeline that never needed it.
    ordered = tp_.run_pipeline(pipeline_config, corpus, impose_order=False)
    shuffled = tp_.run_pipeline(pipeline_config, _permuted(corpus, 5), impose_order=False)

    assert diff_frames(ordered, shuffled) is not None


# --- The rules, each with the case beside it that must stay quiet -----------------------------------------------


@pytest.mark.cpu_mode
def test_the_scan_is_syn_without_completion_and_the_workstation_completes(result: pd.DataFrame):
    # R-D-L4-002, both halves. The workstation opens connections to the same kind of ports in the same way; the
    # difference is that something answers, which is what `ack` near zero is asking about.
    bins = _bins(result)

    scan = bins[bins["src_ip"] == tp_.SCANNER]
    workstation = bins[bins["src_ip"] == tp_.WORKSTATION]

    assert len(scan) == tp_.SCAN_PORTS
    assert set(scan["syn_ratio"]) == {1.0}
    assert set(scan["ack"]) == {0}

    assert len(workstation) > 0
    assert workstation["syn_ratio"].max() < SYN_RATIO_THRESHOLD
    assert workstation["ack"].min() > 0


@pytest.mark.cpu_mode
def test_only_the_scan_reaches_fifty_ports_in_a_bin(result: pd.DataFrame):
    # The aggregate half of R-D-L4-002, which is where the sweep stops being a false positive. Every one of its
    # SYNs is unanswered and scores exactly like the scanner's per flow; it touches one port across thirty
    # hosts rather than sixty ports on one, and a vertical-scan rule is right not to fire on it.
    bins = _bins(result)
    unanswered = bins[(bins["syn_ratio"] >= SYN_RATIO_THRESHOLD) & (bins["ack"] == 0)]

    reach = unanswered.groupby(["src_ip", "rollup_time_ns"])["dst_port"].nunique()

    assert reach.max() == tp_.SCAN_PORTS
    assert list(reach[reach >= SCAN_PORT_THRESHOLD].index.get_level_values("src_ip")) == [tp_.SCANNER]
    assert tp_.SWEEPER in set(unanswered["src_ip"])
    assert reach[tp_.SWEEPER].max() == 1


@pytest.mark.cpu_mode
def test_the_sweep_and_the_outage_are_the_same_ratio_pointed_two_ways(result: pd.DataFrame):
    # R-D-L4-003, which is one predicate and two conclusions. Both cases sit at the same `rst/all`, so a rule
    # reading only the ratio would report them identically and send the wrong response to one of them. The
    # corpus contains both for that reason, and what separates them is which side of the refusal fans out.
    bins = _bins(result)
    refusing = bins[bins["rst_ratio"] >= RST_RATIO_THRESHOLD]

    assert len(refusing) == tp_.SWEEP_TARGETS + len(tp_.OUTAGE_CLIENTS)
    assert set(refusing["rst_ratio"]) == {0.5}

    # One server refusing many clients is an outage; many servers refusing one client is a sweep.
    fan_out = refusing.groupby(["src_ip", "rollup_time_ns"])["dst_ip"].nunique()
    fan_in = refusing.groupby(["dst_ip", "rollup_time_ns"])["src_ip"].nunique()

    assert fan_out.max() == len(tp_.OUTAGE_CLIENTS)
    assert list(fan_out[fan_out > 1].index.get_level_values("src_ip")) == [tp_.FAILING_SERVER]

    assert fan_in.max() == tp_.SWEEP_TARGETS
    assert list(fan_in[fan_in > 1].index.get_level_values("dst_ip")) == [tp_.SWEEPER]


@pytest.mark.cpu_mode
def test_the_refusal_ratio_does_not_fire_on_a_conversation_that_carried_traffic(result: pd.DataFrame):
    # Why the threshold can be half. A refusal is the only thing on the flow that carries it; anything that also
    # carried a handshake or data has flags the RST is a small share of, so the two are not near each other.
    bins = _bins(result)
    carried = bins[bins["total"] > 2]

    assert len(carried) > 0
    assert carried["rst_ratio"].max() == 0.0


@pytest.mark.cpu_mode
def test_the_exfiltration_breaches_the_triples_own_envelope_and_the_busy_host_does_not(result: pd.DataFrame):
    # R-B-L4-005, both halves. The busy triple moves more data in total than the backup triple and varies by a
    # factor of two doing it; a global threshold on transfer volume would have to be set above it and would then
    # be above the exfiltration too. Per triple, one of these is unremarkable and the other is fifty times the
    # envelope.
    breached = result[_truth(result, "flow_data_len_envelope_breached")]

    assert len(breached) == 1
    assert list(breached["src_ip"]) == [tp_.BACKUP_CLIENT]
    assert list(breached["dst_ip"]) == [tp_.BACKUP_SERVER]
    assert list(breached["flow_data_len"]) == [tp_.EXFIL_BYTES]
    assert breached["flow_data_len_envelope_ratio"].iloc[0] == tp_.EXFIL_BYTES / tp_.BACKUP_BYTES

    busy = result[result["src_ip"] == tp_.BUSY_CLIENT]
    mature_busy = busy[_truth(busy, "flow_data_len_envelope_mature")]

    assert len(mature_busy) > 0
    assert mature_busy["flow_data_len_envelope_ratio"].max() < ENVELOPE_MULTIPLIER
    assert not any(_truth(busy, "flow_data_len_envelope_breached"))


@pytest.mark.cpu_mode
def test_the_envelope_is_a_percentile_rather_than_the_largest_ever_seen(result: pd.DataFrame):
    # The reason the backup triple is given a hundred and twenty prior transfers. A 99th percentile by nearest
    # rank over fewer than a hundred samples is the maximum, so a corpus with twenty would be demonstrating
    # "three times the largest ever seen" while the rule says "three times the 99th percentile". Here the
    # envelope sits at a figure the triple transferred routinely, and the busy triple's sits below its own peak.
    backup = result[result["src_ip"] == tp_.BACKUP_CLIENT]
    mature = backup[_truth(backup, "flow_data_len_envelope_mature")]

    assert len(mature) > 0
    assert set(mature["flow_data_len_envelope"]) == {float(tp_.BACKUP_BYTES)}

    busy = result[result["src_ip"] == tp_.BUSY_CLIENT]
    mature_busy = busy[_truth(busy, "flow_data_len_envelope_mature")]

    assert mature_busy["flow_data_len_envelope"].max() < busy["flow_data_len"].max()


@pytest.mark.cpu_mode
def test_no_envelope_is_published_before_the_history_supports_one(result: pd.DataFrame):
    # A tracker that answered before it had samples would breach on the second transfer of everything. Maturity
    # is what stops that, and the count is the rule's hundred rather than a number chosen to make this pass.
    from morpheus.utils.transfer_envelope import DEFAULT_MIN_SAMPLES  # pylint: disable=import-outside-toplevel

    backup = result[result["src_ip"] == tp_.BACKUP_CLIENT].sort_values("event_time", kind="mergesort")
    mature = _truth(backup, "flow_data_len_envelope_mature")

    assert list(mature).count(False) == DEFAULT_MIN_SAMPLES
    assert list(mature) == sorted(mature)
    assert backup["flow_data_len_envelope"][~mature].isna().all()


@pytest.mark.cpu_mode
def test_a_search_divides_the_maxima_rather_than_aggregating_the_ratio(result: pd.DataFrame):
    # Why every detection at this layer is written the way it is. The counts are running totals, so their maxima
    # are the bin's totals; the ratios are running too, and a running ratio is not monotone. The workstation's
    # first packet is a bare SYN, so `max(flow_syn_ratio)` over its bin is 1.0 -- the scan's figure exactly --
    # while the bin it summarizes is a completed handshake. A detection that aggregated the ratio column would
    # report every ordinary connection in the estate as a port scan.
    workstation = result[result["src_ip"] == tp_.WORKSTATION]

    assert workstation.groupby(["flow_id", "rollup_time_ns"])["flow_syn_ratio"].max().max() == 1.0

    bins = _bins(result)
    assert bins[bins["src_ip"] == tp_.WORKSTATION]["syn_ratio"].max() < SYN_RATIO_THRESHOLD


@pytest.mark.cpu_mode
def test_the_bin_is_floored_and_half_open(result: pd.DataFrame):
    # Every window in this fork is `[start, end)` anchored on a fixed epoch. The reference's kernel labels a bin
    # by its end and puts a packet landing exactly on a boundary into the following one, which would disagree
    # with every other layer about which side such an event falls on -- for exactly the events most likely to be
    # on a boundary.
    bin_ns = tp_.BIN_SECONDS * tp_.NS_PER_SECOND
    rollup = result["rollup_time_ns"].astype("Int64")

    assert rollup.notna().all()
    assert ((rollup % bin_ns) == 0).all()
    assert (rollup <= result["event_time"]).all()
    assert (result["event_time"] < rollup + bin_ns).all()

    # The corpus has packets on a boundary on purpose; without one this says nothing about the floor.
    assert any(result["event_time"] == rollup)


@pytest.mark.cpu_mode
def test_the_model_feature_list_is_complete(result: pd.DataFrame):
    # What this fork owes R-B-L4-001, which is a model it does not train: the thirteen features under the names
    # `abp_pcap_preprocessing.py` declares them, mapped to the columns they are emitted as. A deployment builds
    # the model's input frame from this mapping rather than from a second copy of the list that will drift.
    assert len(MODEL_FEATURES) == 13
    assert set(MODEL_FEATURES) == {
        "ack",
        "psh",
        "rst",
        "syn",
        "fin",
        "ppm",
        "data_len",
        "bpp",
        "all",
        "ackpush/all",
        "rst/all",
        "syn/all",
        "fin/all"
    }

    for column in MODEL_FEATURES.values():
        assert column in result.columns, column


@pytest.mark.cpu_mode
def test_every_row_is_stamped_at_layer_four_and_keyed_on_its_flow(result: pd.DataFrame):
    # Layer 4's entity is the flow, which is what Part 2 names and what a chain at this layer roots on. The
    # `community_id` beside it is the join to whatever else saw the same conversation.
    assert set(result["osi_layer"]) == {tp_.OSI_LAYER}
    assert set(result["telemetry_class"]) == {tp_.TELEMETRY_CLASS}
    assert list(result["entity_key"]) == list(result["flow_id"])
    assert result["lineage_id"].notna().all()
    assert result["window_id"].notna().all()
    assert result["community_id"].str.startswith("1:").all()


@pytest.mark.cpu_mode
def test_the_envelope_is_kept_for_the_triple_and_not_for_the_flow(result: pd.DataFrame):
    # Why the two stages key on different things. The source port is ephemeral, so a baseline kept per flow
    # would start empty on every connection and never mature -- the rule would be inert rather than wrong,
    # which is harder to notice.
    backup = result[result["src_ip"] == tp_.BACKUP_CLIENT]

    assert backup["flow_id"].nunique() == len(backup)
    assert set(backup["transfer_triple"]) == {f"{tp_.BACKUP_CLIENT}:{tp_.BACKUP_SERVER}:445"}
