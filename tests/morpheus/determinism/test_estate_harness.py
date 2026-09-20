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
Control 13's six checks against the estate pipeline, and the third rung of the ladder asserted.

The two-layer chain was proved by the telemetry harness. This file is about the rung above it: an authentication
at layer 5 carried down to the switch port its principal was sitting at, through a supplied directory and a
derived supplicant table, so that a chain holds events from layers 1, 2 and 5 at once.

Every assertion about the ladder has a case beside it that must stay quiet. A resolver that attributed everyone
to a port would pass a test that only counted three-layer chains, and would be worse than the two-layer chains it
replaced -- attribution that is confidently wrong is not an improvement on attribution that is visibly missing.
So the remote principal must reach no port, and the sign-ins that happen after the last 802.1X exchange must
reach no port either, because a binding is an interval rather than a lookup.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.stages.lineage.binding_resolver_stage import UNRESOLVED
from morpheus.utils.determinism import diff_frames
from morpheus.utils.determinism import frame_digest
from morpheus.utils.determinism import permute_within_contiguous_groups
from morpheus.utils.lineage import window_id_from_timestamp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import estate_pipeline as ep  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_estate_expected.csv")
DRIVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_estate_pipeline.py")

NS = ep.NS_PER_SECOND

THREE_LAYER_CHAINS = 15
"""Chains spanning all three layers in this corpus: one per port per window the desks were resolved in.

Named here rather than derived, because a count the test computes from the same run it is checking would pass
whatever the pipeline did.
"""

DESK_PORTS = sorted(f"{ep.tp.SITE}:{ep.tp.SWITCH}:{port}" for port in ep.DESKS.values())


@pytest.fixture(name="corpus", scope="module")
def corpus_fixture() -> dict[str, pd.DataFrame]:
    yield ep.build_corpus()


_RESULTS: dict = {}


@pytest.fixture(name="pipeline_config")
def pipeline_config_fixture(execution_mode) -> Config:
    yield ep.build_pipeline_config(execution_mode)


@pytest.fixture(name="result")
def result_fixture(pipeline_config: Config, corpus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mode = pipeline_config.execution_mode

    if (mode not in _RESULTS):
        _RESULTS[mode] = ep.run_pipeline(pipeline_config, corpus)

    yield _RESULTS[mode]


def _rows(result: pd.DataFrame, telemetry_class: str) -> pd.DataFrame:
    return result[result["telemetry_class"] == telemetry_class]


def _auth(result: pd.DataFrame) -> pd.DataFrame:
    return _rows(result, ep.AUTH_CLASS)


def _chained(result: pd.DataFrame) -> pd.DataFrame:
    return result[result["lineage_id"].notna() & (result["lineage_id"] != "")]


def _layers_per_chain(result: pd.DataFrame) -> pd.Series:
    return _chained(result).groupby("lineage_id")["osi_layer"].nunique()


def _split(frame: pd.DataFrame, parts: int) -> list[pd.DataFrame]:
    size = max(1, len(frame) // parts)

    return [frame.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(frame), size)]


def _windows(frame: pd.DataFrame) -> list[int]:
    return [window_id_from_timestamp(int(t), ep.PERIOD_SECONDS * NS) for t in frame["event_time"]]


def _permuted(corpus: dict[str, pd.DataFrame], seed: int) -> dict[str, pd.DataFrame]:
    return {
        name: permute_within_contiguous_groups(frame, _windows(frame), seed=seed)
        for (name, frame) in corpus.items()
    }


# --- Check 1: the corpus -------------------------------------------------------------------------------------------


def test_corpus_is_fixed(corpus: dict[str, pd.DataFrame]):
    again = ep.build_corpus()

    assert set(again) == set(corpus)

    for name in corpus:
        pd.testing.assert_frame_equal(corpus[name], again[name])


def test_the_estate_is_one_estate(corpus: dict[str, pd.DataFrame]):
    # The premise the whole harness rests on: layers 1, 2 and 5 describe the same hour. Two corpora that do not
    # overlap in time can never share a sealed window, so a chain across them is impossible before any stage runs.
    auth = corpus[ep.AUTH_CLASS]

    assert len(auth) > 0

    for name in ("tc1", "tc2_mac", "tc2_auth"):
        frame = corpus[name]
        assert int(frame["event_time"].min()) < int(auth["event_time"].max())
        assert int(auth["event_time"].min()) < int(frame["event_time"].max())

    assert int(auth["event_time"].max()) < ep.CORPUS_SECONDS * NS


def test_the_directory_describes_the_supplicants_the_estate_actually_reports(corpus: dict[str, pd.DataFrame]):
    # A directory naming identities no exchange presents would resolve nothing, and the ladder's failure would
    # look exactly like a pipeline defect. This is the premise stated as an assertion instead.
    presented = set(corpus["tc2_auth"]["dot1x_identity"])

    assert set(ep.DIRECTORY.values()) <= presented
    assert ep.REMOTE not in ep.DIRECTORY


# --- Checks 2 to 5: determinism --------------------------------------------------------------------------------


@pytest.mark.gpu_and_cpu_mode
def test_double_run_diff(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    second = ep.run_pipeline(pipeline_config, corpus)

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

    rendered = ep.render(result)

    if (rendered != golden_text):
        from io import StringIO
        as_text = {"dtype": str, "keep_default_na": False}
        difference = diff_frames(pd.read_csv(StringIO(rendered), **as_text),
                                 pd.read_csv(StringIO(golden_text), **as_text))

        pytest.fail(f"Output drifted from {os.path.basename(GOLDEN_PATH)}: {difference}. If the change is intended, "
                    f"regenerate the golden file with {os.path.basename(DRIVER_PATH)} and review the diff.")


@pytest.mark.gpu_and_cpu_mode
def test_batch_split_sweep(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    # Control 5 over a pipeline whose chains are decided by a sealer fed the union of five classes. Batching
    # changes how that union is chunked, so a chain that depended on the chunking would show up here.
    thirds = {name: _split(frame, 3) for (name, frame) in corpus.items()}

    assert diff_frames(result, ep.run_pipeline(pipeline_config, corpus, batches=thirds)) is None


@pytest.mark.slow
@pytest.mark.gpu_and_cpu_mode
def test_batch_split_sweep_by_row(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    # The same control at its extreme, one frame per row. Slow because the union then arrives as more than a
    # thousand single-row messages, which is the point: nothing about a chain may depend on that.
    by_row = {name: _split(frame, len(frame)) for (name, frame) in corpus.items()}

    assert diff_frames(result, ep.run_pipeline(pipeline_config, corpus, batches=by_row)) is None


# --- Check 6: permutation ----------------------------------------------------------------------------------------


@pytest.mark.gpu_and_cpu_mode
def test_permutation_within_windows(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    for seed in (1, 2, 3):
        shuffled = _permuted(corpus, seed)

        assert any(not shuffled[name]["collector_seq"].equals(corpus[name]["collector_seq"])
                   for name in corpus), "permutation was a no-op"

        difference = diff_frames(result, ep.run_pipeline(pipeline_config, shuffled))
        assert difference is None, f"seed {seed}: {difference}"


@pytest.mark.gpu_and_cpu_mode
def test_permutation_check_has_teeth(pipeline_config: Config, corpus: dict[str, pd.DataFrame]):
    unsorted_baseline = ep.run_pipeline(pipeline_config, corpus, impose_order=False)

    detected = False

    for seed in (1, 2, 3):
        detected = detected or (diff_frames(
            unsorted_baseline, ep.run_pipeline(pipeline_config, _permuted(corpus, seed), impose_order=False))
                                is not None)

    assert detected, "Removing the total-order stage did not change any output under permutation."


# --- The third rung ----------------------------------------------------------------------------------------------


@pytest.mark.cpu_mode
def test_a_chain_reaches_three_layers(result: pd.DataFrame):
    # What `Chain assembly - cross-layer risk` has been waiting for: `dc(osi_layer) >= 3` over one `lineage_id`.
    spans = _layers_per_chain(result)
    three = spans[spans >= 3]

    assert len(three) == THREE_LAYER_CHAINS

    reached = _chained(result)
    reached = reached[reached["lineage_id"].isin(set(three.index))]

    assert sorted(set(reached["chain_anchor"])) == DESK_PORTS
    assert set(reached["osi_layer"]) == {1, 2, 5}


@pytest.mark.cpu_mode
def test_an_authentication_joins_the_chain_of_the_port_its_principal_sat_at(result: pd.DataFrame):
    # The ladder end to end, stated as the join it is: for every resolved sign-in there is a layer 1 sample on the
    # same port in the same window, and the two carry one `lineage_id`.
    auth = _auth(result)
    resolved = auth[auth["desk_port_key"].notna()]

    assert len(resolved) > 0

    layer_1 = _rows(result, "tc1")
    ports = dict(zip(layer_1["lineage_id"], layer_1["chain_anchor"]))

    for (_, row) in resolved.iterrows():
        assert row["lineage_id"] in ports, "a resolved sign-in is in no chain layer 1 is in"
        assert ports[row["lineage_id"]] == row["desk_port_key"]
        assert row["chain_anchor"] == row["desk_port_key"]
        assert row["chain_anchor_source"] == ep.DESK_PORT_COLUMN


@pytest.mark.cpu_mode
def test_the_port_a_principal_resolves_to_is_the_port_layer_1_polls(result: pd.DataFrame):
    # The three spellings have to be one string. `auth_port_key`, `port_key` and layer 1's `entity_key` are
    # composed by different stages from different columns, and a chain rooted on a port only holds members from
    # every layer while they agree byte for byte.
    auth = _auth(result)
    resolved = auth[auth["desk_port_key"].notna()]

    assert sorted(set(resolved["desk_port_key"])) == DESK_PORTS
    assert set(resolved["desk_port_key"]) <= set(_rows(result, "tc1")["entity_key"])
    assert set(resolved["desk_port_key"]) <= set(_rows(result, "tc2_auth")["auth_port_key"])


@pytest.mark.cpu_mode
def test_the_principal_who_is_never_at_a_desk_reaches_no_port(result: pd.DataFrame):
    # The negative control, and the one that matters most. A ladder that resolved everybody would satisfy every
    # assertion above while attributing a remote worker to somebody else's desk.
    auth = _auth(result)
    remote = auth[auth["user_principal"] == ep.REMOTE]

    assert len(remote) > 0
    assert remote["desk_identity"].isna().all()
    assert remote["desk_port_key"].isna().all()
    assert (remote[ep.DIRECTORY_METHOD_COLUMN] == UNRESOLVED).all()
    assert (remote["chain_anchor"] == ep.REMOTE).all()

    ports = set(_rows(result, "tc1")["lineage_id"])

    assert set(remote["lineage_id"]).isdisjoint(ports)


@pytest.mark.cpu_mode
def test_a_binding_is_an_interval_and_not_a_lookup(result: pd.DataFrame):
    # The second negative control, and a free one: the 802.1X exchanges stop before the hour does, so the same
    # principal resolves to a port earlier in the hour and to nothing after the last exchange. A resolver that
    # treated the table as a dictionary would attribute those late sign-ins to a port nobody was observed at.
    auth = _auth(result)
    named = auth[auth["user_principal"].isin(ep.DIRECTORY)]

    assert named["desk_identity"].notna().all(), "the directory covers the whole hour"

    resolved = named[named["desk_port_key"].notna()]
    unresolved = named[named["desk_port_key"].isna()]

    assert len(resolved) > 0 and len(unresolved) > 0
    assert int(unresolved["event_time"].min()) > int(resolved["event_time"].max())
    assert (unresolved[ep.SUPPLICANT_METHOD_COLUMN] == UNRESOLVED).all()
    assert (unresolved["chain_anchor"] == unresolved["user_principal"]).all()


@pytest.mark.cpu_mode
def test_every_sign_in_says_how_far_down_the_ladder_it_got(result: pd.DataFrame):
    # Both rungs are recorded separately, because "resolved to a port" and "found in the directory" fail for
    # different reasons and an operator reading one column could not tell them apart.
    auth = _auth(result)

    for column in (ep.DIRECTORY_METHOD_COLUMN, ep.SUPPLICANT_METHOD_COLUMN):
        assert auth[column].notna().all()
        assert set(auth[column]) <= {UNRESOLVED, "soft:directory", "soft:dot1x"}

    assert set(auth["chain_anchor_source"]) == {ep.DESK_PORT_COLUMN, "user_principal"}


@pytest.mark.cpu_mode
def test_the_bypass_identity_binds_to_a_port_and_carries_nobody(result: pd.DataFrame):
    # The other end of the same guard. `unknown-supplicant` is a real identity on a real port, and it is in the
    # supplicant table -- but no principal presents it, so it takes nobody down the ladder.
    assert "unknown-supplicant" not in set(ep.DIRECTORY.values())

    auth = _auth(result)

    assert "unknown-supplicant" not in set(auth["desk_identity"].dropna())


@pytest.mark.cpu_mode
def test_every_class_ran_and_only_the_chained_ones_sealed(result: pd.DataFrame):
    # The binding classes ride along because the Splunk package is rendered from this one run, and they are what
    # the two binding sourcetypes are made of. They carry no chain, and saying so here keeps a future sealer that
    # swept them up from passing quietly.
    assert set(result["telemetry_class"]) == set(ep.CHAINED_CLASSES) | set(ep.UNCHAINED_CLASSES)

    for name in ep.CHAINED_CLASSES:
        rows = _rows(result, name)

        assert len(rows) > 0, name
        assert rows["lineage_id"].notna().all(), name
        assert rows["window_id"].notna().all(), name

    for name in ep.UNCHAINED_CLASSES:
        rows = _rows(result, name)

        assert len(rows) > 0, name
        assert rows["lineage_id"].isna().all(), name


@pytest.mark.cpu_mode
def test_the_layers_are_stamped_where_they_came_from(result: pd.DataFrame):
    expected = {"tc1": 1, "tc2_mac": 2, "tc2_arp": 2, "tc2_auth": 2, ep.AUTH_CLASS: 5}

    for (name, layer) in expected.items():
        assert set(_rows(result, name)["osi_layer"]) == {layer}, name


# --- What minimizing the top layer does and does not achieve ----------------------------------------------------
#
# The guide leaves retention and minimization to the deploying organization, and `MinimizationStage` is the
# mechanism that lets one act on the decision. What follows is that mechanism against a real estate rather than a
# constructed frame, and then its honest limit: an estate that digests every name and every place in its layer 5
# records has still not anonymized anybody, because the layers below carry both and the chain leads to them.

LAYER_FIVE_POLICY = ["user_principal", "entity_key", "desk_identity", "chain_anchor", "desk_port_key"]
"""Everything in a layer 5 row that names the principal or says where they were sitting.

Arrived at by running a smaller policy and reading the refusals. `chain_anchor` has to be here because it is
copied from whichever candidate matched, and `desk_port_key` because it is one of the candidates -- digesting
the anchor and shipping the column it came from would put the plaintext back beside the digest.
"""

MINIMIZATION_KEY = b"an-estate-key-of-sufficient-length"


def _minimized(config: Config, result: pd.DataFrame, policy: list = None) -> pd.DataFrame:
    """The layer 5 rows with a policy applied, exactly as a deployment would apply it per segment."""
    from morpheus.messages import MessageMeta  # pylint: disable=import-outside-toplevel
    from morpheus.stages.lineage.minimization_stage import MinimizationStage  # pylint: disable=import-outside-toplevel

    meta = MessageMeta(_rows(result, ep.AUTH_CLASS).reset_index(drop=True))
    stage = MinimizationStage(config,
                              pseudonymize=LAYER_FIVE_POLICY if policy is None else policy,
                              key=MINIMIZATION_KEY)
    stage.on_data(meta)

    return meta.copy_dataframe()


@pytest.mark.cpu_mode
def test_digesting_only_the_principal_is_refused_and_says_what_else_carries_them(pipeline_config: Config,
                                                                                 result: pd.DataFrame):
    # The policy an estate writes first. A layer 5 row carries the principal's own string three times, and two
    # of the three are places nobody thinks of as a name: the entity key the window was sealed on and the anchor
    # the chain was rooted on.
    with pytest.raises(ValueError) as caught:
        _minimized(pipeline_config, result, policy=["user_principal"])

    for column in ("entity_key", "chain_anchor"):
        assert column in str(caught.value), column


@pytest.mark.cpu_mode
def test_the_refusal_does_not_catch_a_second_identifier_for_the_same_person(pipeline_config: Config,
                                                                            result: pd.DataFrame):
    # The stated limit of that safety net, asserted so nobody mistakes it for completeness. The check compares
    # values, so it finds the principal's string wherever it was copied -- and `desk_identity` holds the 802.1X
    # identity the directory resolved, which is a different string for the same person. No value comparison can
    # find that. It is why a policy is written from the inventory rather than from whatever the refusals happen
    # to name.
    auth = _auth(result)
    identities = set(auth["desk_identity"].dropna())

    assert len(identities) > 0
    assert not (identities & set(auth["user_principal"]))

    with pytest.raises(ValueError) as caught:
        _minimized(pipeline_config, result, policy=["user_principal"])

    assert "desk_identity" not in str(caught.value)


@pytest.mark.cpu_mode
def test_the_whole_layer_five_policy_moves_every_name_together(pipeline_config: Config, result: pd.DataFrame):
    # The mechanism doing its job, so that the limit below is a limit rather than a defect. Every name moves,
    # and one principal is still one principal afterwards or the per-entity story would break in the SIEM
    # instead of in the pipeline.
    minimized = _minimized(pipeline_config, result)
    principals = set(_auth(result)["user_principal"])
    desks = set(DESK_PORTS)

    assert len(principals) > 1

    for column in ("user_principal", "entity_key", "desk_identity", "chain_anchor", "desk_port_key"):
        surviving = set(minimized[column].dropna())

        assert not (surviving & principals), column
        assert not (surviving & desks), column

    assert minimized["user_principal"].nunique() == len(principals)
    assert list(minimized["entity_key"]) == list(minimized["user_principal"])


@pytest.mark.cpu_mode
def test_a_fully_minimized_layer_five_is_still_not_anonymous(pipeline_config: Config, result: pd.DataFrame):
    # The sentence that matters, proved through the shipped artifacts rather than argued. Take a layer 5 row
    # that names nobody and locates nobody, follow its `lineage_id` into the chain it was sealed into, read the
    # plaintext 802.1X identity and port off the layer 2 row it reaches, and look the identity up in the
    # directory the estate supplies. The principal and their desk both come back.
    #
    # Nothing here is an attack. It is the identifier ladder being used for exactly what it was built for, by
    # somebody holding the same two indexes as everybody else. Minimizing one layer of a design whose purpose is
    # to join layers is a control against casual reading, and it is not anonymization.
    minimized = _minimized(pipeline_config, result)
    chained = minimized[minimized["lineage_id"].notna()]
    identity_to_principal = {identity: principal for (principal, identity) in ep.DIRECTORY.items()}

    assert len(chained) > 0

    neighbours = _rows(result, "tc2_auth")
    recovered = {}

    for lineage_id in set(chained["lineage_id"]):
        reached = neighbours[neighbours["lineage_id"] == lineage_id]

        for row in reached.itertuples():
            principal = identity_to_principal.get(row.dot1x_identity)

            if (principal is not None):
                recovered.setdefault(principal, set()).add(row.auth_port_key)

    # Every principal who sat at a desk is recovered, at the desk they sat at.
    assert set(recovered) == set(ep.DESKS)
    assert ep.REMOTE not in recovered

    for (principal, ports) in recovered.items():
        assert ports == {f"{ep.tp.SITE}:{ep.tp.SWITCH}:{ep.DESKS[principal]}"}, principal
