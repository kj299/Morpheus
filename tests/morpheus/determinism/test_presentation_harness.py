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
Control 13's six checks against the composed layer 6 pipeline, and the five rules' predicates over its corpus.

Layer 6 is where the OSI model fits real networks worst, which the guide says plainly, and the class earns its
place anyway because what it models -- the encoding and cryptographic negotiation surface -- is distinct from
both layer 5 and layer 7. It is also the cheapest layer in the set: it rides collection points layers 3 and 4
already established, and four of its five rules reuse primitives this fork had before layer 6 existed.

What the corpus has to prove here is unusual in one way. Three of these rules need a reference and two do not,
so "the rule is quiet" means two different things, and a harness that asserted only the firings would not tell
them apart. The checks below assert both: that the self-signed and content rules answer on a destination's first
handshake, and that the issuer and cipher rules decline until their entity has a history rather than guessing.
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
import presentation_pipeline as pp_  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_presentation_expected.csv")
DRIVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_presentation_pipeline.py")

SETTLED_HANDSHAKES = 20
"""Prior handshakes R-B-L6-001 requires before a host's new stack is a finding. The corpus separates its two
hosts at 34 and 2, so any floor between them serves and the assertion is not on the threshold."""

SINGLE_ISSUER = 1
"""What R-D-L6-002 means by a destination with a settled authority: one it has only ever presented."""


@pytest.fixture(name="corpus", scope="module")
def corpus_fixture() -> dict[str, pd.DataFrame]:
    yield pp_.build_corpus()


# One composed run per execution mode, reused by every check in that mode.
_RESULTS: dict = {}


@pytest.fixture(name="pipeline_config")
def pipeline_config_fixture(execution_mode) -> Config:
    yield pp_.build_pipeline_config(execution_mode)


@pytest.fixture(name="result")
def result_fixture(pipeline_config: Config, corpus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mode = pipeline_config.execution_mode

    if (mode not in _RESULTS):
        _RESULTS[mode] = pp_.run_pipeline(pipeline_config, corpus)

    yield _RESULTS[mode]


def _truth(frame: pd.DataFrame, column: str) -> pd.Series:
    """A nullable flag as a plain boolean, with "not answerable" reading as "did not fire"."""
    return frame[column].astype("boolean").fillna(False)


def _split(frame: pd.DataFrame, parts: int) -> list:
    size = max(1, len(frame) // parts)

    return [frame.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(frame), size)]


def _windows(frame: pd.DataFrame) -> list:
    period_ns = pp_.PERIOD_SECONDS * pp_.NS_PER_SECOND

    return [window_id_from_timestamp(int(stamp), period_ns) for stamp in frame["event_time"]]


def _permuted(corpus: dict, seed: int) -> dict:
    return {
        name: permute_within_contiguous_groups(frame, _windows(frame), seed=seed)
        for (name, frame) in corpus.items()
    }


# --- Check 1: the corpus is fixed and shaped like an inspection point -------------------------------------------


def test_corpus_is_fixed(corpus: dict[str, pd.DataFrame]):
    again = pp_.build_corpus()

    assert set(again) == set(corpus)

    for name in corpus:
        pd.testing.assert_frame_equal(corpus[name], again[name])


def test_corpus_is_shaped_like_an_inspection_point(corpus: dict[str, pd.DataFrame]):
    handshakes = corpus[pp_.TELEMETRY_CLASS]

    # One record per connection, carrying the fields Part 2 lists as required for the rules that ship.
    for column in ("src_ip",
                   "dst_ip",
                   "tls_version",
                   "ja4_client",
                   "cipher_suite",
                   "certificate_issuer",
                   "certificate_fingerprint_sha256",
                   "validation_result",
                   "certificate_not_before",
                   "certificate_not_after"):
        assert column in handshakes.columns, column

    assert set(pp_.ID_COLUMNS) <= set(handshakes.columns)
    assert handshakes["collector_seq"].is_monotonic_increasing
    assert handshakes["event_time"].is_monotonic_increasing

    # Three hours, so a reference has time to settle, be broken, and settle again.
    span_hours = (handshakes["event_time"].max() - handshakes["event_time"].min()) / (pp_.NS_PER_SECOND * 3600)
    assert span_hours > 2


def test_the_corpus_carries_no_value_a_restart_would_change(corpus: dict[str, pd.DataFrame]):
    # The trap this corpus fell into once. The certificate fingerprints were built from `hash()`, which is
    # salted by `PYTHONHASHSEED` for strings, so every value differed between interpreters -- which the
    # cross-restart check would have caught, for a defect the corpus had no reason to contain.
    handshakes = corpus[pp_.TELEMETRY_CLASS]
    again = pp_.build_corpus()[pp_.TELEMETRY_CLASS]

    assert list(handshakes["certificate_fingerprint_sha256"]) == list(again["certificate_fingerprint_sha256"])
    assert all(value.startswith("sha256:") for value in handshakes["certificate_fingerprint_sha256"])


def test_the_documentation_ranges_are_not_used_for_the_public_internet():
    # The same trap layer 3 fell into. `192.0.2.0/24`, `198.51.100.0/24` and `203.0.113.0/24` are reserved for
    # documentation, so the classifier calls them private -- and R-D-L6-003 is entirely a question about
    # whether the connection left the estate.
    from morpheus.parsers import ip  # pylint: disable=import-outside-toplevel

    external = pd.Series([pp_.PORTAL, pp_.CDN, pp_.ROTATED, pp_.ATTACKER])

    assert all(ip.is_global(external))
    assert not any(ip.is_private(external))
    assert bool(ip.is_private(pd.Series([pp_.APPLIANCE]))[0])


# --- Checks 2 through 6: determinism ---------------------------------------------------------------------------


@pytest.mark.gpu_and_cpu_mode
def test_double_run_diff(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    second = pp_.run_pipeline(pipeline_config, corpus)

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

    rendered = pp_.render(result)

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

    assert diff_frames(result, pp_.run_pipeline(pipeline_config, corpus, batches=thirds)) is None
    assert diff_frames(result, pp_.run_pipeline(pipeline_config, corpus, batches=by_row)) is None


@pytest.mark.gpu_and_cpu_mode
def test_permutation_within_windows(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    for seed in (1, 2, 3):
        shuffled = pp_.run_pipeline(pipeline_config, _permuted(corpus, seed))

        assert diff_frames(result, shuffled) is None, seed


@pytest.mark.gpu_and_cpu_mode
def test_permutation_check_has_teeth(pipeline_config: Config, corpus: dict):
    # The negative control for the check above. Three of the four stages are cumulative -- a novelty set, a mode
    # and a running minimum are each functions of what came before -- so removing the imposed order has to
    # change the answer, or the check is passing over a pipeline that never needed it.
    ordered = pp_.run_pipeline(pipeline_config, corpus, impose_order=False)
    shuffled = pp_.run_pipeline(pipeline_config, _permuted(corpus, 5), impose_order=False)

    assert diff_frames(ordered, shuffled) is not None


# --- The five rules, each with the case beside it that must stay quiet ------------------------------------------


@pytest.mark.cpu_mode
def test_a_settled_host_acquiring_a_stack_is_the_finding_and_a_new_host_is_not(result: pd.DataFrame):
    # R-B-L6-001, both halves, and the half the corpus had to be rebuilt to show. Both hosts below report a
    # fingerprint they have never presented; only one of them had a history to depart from. Without the
    # prior-handshake count the two rows are identical, which is why the count is on the row.
    novel = result[_truth(result, "ja4_client_first_seen")]

    assert set(novel["src_ip"]) == {pp_.MANAGED, pp_.VARIED}

    settled = novel[novel["ja4_client_observations"] >= SETTLED_HANDSHAKES]

    assert list(settled["src_ip"]) == [pp_.MANAGED]
    assert list(settled["ja4_client"]) == [pp_.NEW_STACK]
    assert novel[novel["src_ip"] == pp_.VARIED]["ja4_client_observations"].max() < SETTLED_HANDSHAKES


@pytest.mark.cpu_mode
def test_alternating_between_two_known_stacks_is_never_novel_again(result: pd.DataFrame):
    # The case a rule reading "the fingerprint changed" would flag on every handshake. The varied host swaps
    # between a browser and an updater all corpus long, and after each has been seen once neither is new.
    varied = result[result["src_ip"] == pp_.VARIED].sort_values("event_time", kind="mergesort")

    assert int(_truth(varied, "ja4_client_changed").sum()) > 10
    assert int(_truth(varied, "ja4_client_first_seen").sum()) == 1


@pytest.mark.cpu_mode
def test_an_interception_differs_and_a_delivery_host_behind_four_authorities_does_not(result: pd.DataFrame):
    # R-D-L6-002, both halves. The content delivery host rotates among four authorities legitimately and
    # differs from its own mode constantly; the distinct count is what separates it from a destination that
    # has only ever presented one and suddenly presents another.
    mature = result[_truth(result, "cert_issuer_mature")]
    differs = mature[_truth(mature, "cert_issuer_differs")]

    assert pp_.CDN in set(differs["dst_ip"]), "the negative control must reach the predicate to be a control"

    settled = differs[differs["cert_issuer_distinct"] == SINGLE_ISSUER]

    assert pp_.CDN not in set(settled["dst_ip"])
    assert pp_.PORTAL in set(settled["dst_ip"])

    interception = settled[settled["dst_ip"] == pp_.PORTAL]

    assert list(interception["certificate_issuer"]) == [pp_.PROXY_CA]
    assert list(interception["cert_issuer_established"]) == [pp_.CORP_CA]


@pytest.mark.cpu_mode
def test_an_authority_rotation_is_reported_once_and_then_stops(result: pd.DataFrame):
    # A certificate authority migration is a genuine change and then a fact, and a rule that went on reporting
    # it would be silenced by hand within a week. Two things stop it: the reference follows the destination, and
    # the single-issuer gate closes the moment the destination has demonstrably presented two.
    rotated = result[result["dst_ip"] == pp_.ROTATED].sort_values("event_time", kind="mergesort")
    differs = _truth(rotated, "cert_issuer_differs")

    assert int(differs.sum()) > 1, "the rotation must be visible at all for the gate to be doing anything"

    gated = rotated[differs & (rotated["cert_issuer_distinct"] == SINGLE_ISSUER)]

    assert len(gated) == 1
    assert list(gated["certificate_issuer"]) == [pp_.ROTATED_CA]

    # And it settles on its own: the last handshakes of the corpus no longer differ at all.
    assert not bool(differs.iloc[-1])


@pytest.mark.cpu_mode
def test_a_self_signed_certificate_leaving_the_estate_fires_and_an_internal_one_does_not(result: pd.DataFrame):
    # R-D-L6-003, both halves. The internal appliance's self-signed certificate is ordinary -- a management
    # interface nobody bought a certificate for -- and an estate has hundreds of them.
    self_signed = result[_truth(result, "cert_self_signed")]

    assert set(self_signed["dst_ip"]) == {pp_.ATTACKER, pp_.APPLIANCE}

    external = result[_truth(result, "cert_self_signed_external")]

    assert set(external["dst_ip"]) == {pp_.ATTACKER}
    assert set(external["cert_validity_days"]) == {float(pp_.SHORT_VALIDITY_DAYS)}


@pytest.mark.cpu_mode
def test_a_downgrade_fires_and_a_pair_s_own_variation_does_not(result: pd.DataFrame):
    # R-B-L6-004, both halves. The varied host's conversations move between two sound suites routinely, which a
    # reference taken as a mode would have called a downgrade every time the weaker one was negotiated.
    downgraded = result[_truth(result, "cipher_downgraded")]

    assert len(downgraded) == 1
    assert list(downgraded["src_ip"]) == [pp_.LEGACY_CLIENT]
    assert list(downgraded["cipher_tier"]) == ["broken"]
    assert list(downgraded["cipher_floor_tier"]) == ["modern"]

    varied = result[result["src_ip"] == pp_.VARIED]

    assert len(set(varied["cipher_suite"])) > 1
    assert not any(_truth(varied, "cipher_downgraded"))


@pytest.mark.cpu_mode
def test_a_legacy_appliance_keeps_its_own_low_floor_without_firing(result: pd.DataFrame):
    # The second negative control, and the reason the floor is per pair. An appliance that has only ever
    # negotiated a legacy suite is not being downgraded; a global threshold would either alert on it forever or
    # sit below the downgrade this rule exists to catch.
    appliance = result[result["dst_ip"] == pp_.APPLIANCE]

    assert set(appliance["cipher_tier"]) == {"legacy"}
    assert not any(_truth(appliance, "cipher_downgraded"))


@pytest.mark.cpu_mode
def test_a_file_behind_a_declared_image_crosses_and_a_re_encoding_does_not(result: pd.DataFrame):
    # R-D-L6-005, both halves. The re-encodings are the volume an estate actually has, and a rule reporting
    # every declared-versus-detected mismatch would bury the one row that matters under them.
    crossed = result[_truth(result, "content_category_crossed")]

    assert len(crossed) == 1
    assert list(crossed["content_type_declared"]) == [pp_.PNG]
    assert list(crossed["content_type_detected"]) == [pp_.ZIP]

    compared = result[result["content_category_crossed"].notna()]

    assert len(compared) > 40, "the re-encodings must reach the comparison to be a control"


@pytest.mark.cpu_mode
def test_nothing_in_the_corpus_falls_outside_the_two_tables(result: pd.DataFrame):
    # Both tables report what they could not place rather than guessing, so a corpus that quietly drifted out of
    # their coverage would make the rules above fire less and say nothing about why. Asserting zero here is what
    # keeps the counts above meaning what they claim.
    assert int(_truth(result, "cipher_unrecognized").sum()) == 0
    assert int(_truth(result, "content_category_unclassified").sum()) == 0


@pytest.mark.cpu_mode
def test_the_two_rules_that_need_no_history_answer_on_the_first_handshake(result: pd.DataFrame):
    # Three of these rules wait for a reference and two do not, and "the rule is quiet" means something
    # different in each case. The self-signed and content verdicts are present on every row that carries the
    # fields, including a destination's first.
    first_per_destination = (result.sort_values("event_time", kind="mergesort").groupby("dst_ip",
                                                                                        as_index=False).first())

    assert first_per_destination["cert_self_signed"].notna().all()

    with_content = result[result["content_type_declared"].notna()]

    assert with_content["content_category_declared"].notna().all()


@pytest.mark.cpu_mode
def test_the_two_rules_that_need_a_history_decline_until_they_have_one(result: pd.DataFrame):
    # The other half of the same statement. Before maturity the issuer and cipher rules publish no verdict at
    # all rather than a negative one, which is what stops a destination nobody has seen enough of from being
    # reported as consistent.
    immature = result[~_truth(result, "cert_issuer_mature")]

    assert len(immature) > 0
    assert immature["cert_issuer_differs"].isna().all()
    assert immature["cert_issuer_established"].isna().all()

    unfloored = result[~_truth(result, "cipher_mature")]

    assert len(unfloored) > 0
    assert unfloored["cipher_downgraded"].isna().all()


@pytest.mark.cpu_mode
def test_every_row_is_stamped_at_layer_six_and_keyed_on_its_host(result: pd.DataFrame):
    # The departure from Part 2, asserted so it cannot drift back silently. The class's own identifiers are
    # carried on every row and neither is the sealing entity: a JA4 fingerprint is a property of a TLS stack
    # shared by thousands of unrelated hosts, so a chain rooted on it would group them into one entity.
    assert set(result["osi_layer"]) == {pp_.OSI_LAYER}
    assert set(result["telemetry_class"]) == {pp_.TELEMETRY_CLASS}
    assert list(result["entity_key"]) == list(result["src_ip"])
    assert result["lineage_id"].notna().all()
    assert result["window_id"].notna().all()

    assert result["ja4_client"].notna().all()
    assert result["certificate_fingerprint_sha256"].notna().all()

    # The argument itself, rather than a restatement of it: at least one fingerprint in this corpus is presented
    # by more than one host, so sealing on the fingerprint would have merged two entities into one.
    hosts_per_fingerprint = result.groupby("ja4_client")["src_ip"].nunique()

    assert hosts_per_fingerprint.max() > 1
