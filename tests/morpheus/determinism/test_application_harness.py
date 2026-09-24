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
Control 13's six checks against the first two layer 7 classes, and both rules' predicates over their corpus.

Both rules here are conjunctions, and a conjunction is only tested when each of its conditions is shown to be the one
that matters. A corpus that proved R-B-L7-001 only against ordinary browsing would pass with any of its three
conditions deleted, and the guide's own warning -- entropy alone flags every content delivery network -- would be a
sentence rather than an assertion. So the checks below are counterfactual: for each benign actor, they assert the
rule would have fired had the one condition that stops it been removed. That is what makes each condition
load-bearing rather than decorative.

The predicates are evaluated the way the shipped searches evaluate them, and the search for R-B-L7-001 does
something the stage's own count does not: it keeps only the queries that clear entropy and label length and counts
distinct subdomains among those. Counting every subdomain would let a SaaS domain with a hundred ordinary tenant
names and one random hostname satisfy all three conditions on different traffic.
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
import application_pipeline as ap_  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_application_expected.csv")
DRIVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_application_pipeline.py")


@pytest.fixture(name="corpus", scope="module")
def corpus_fixture() -> dict[str, pd.DataFrame]:
    yield ap_.build_corpus()


# One composed run per execution mode, reused by every check in that mode.
_RESULTS: dict = {}


@pytest.fixture(name="pipeline_config")
def pipeline_config_fixture(execution_mode) -> Config:
    yield ap_.build_pipeline_config(execution_mode)


@pytest.fixture(name="result")
def result_fixture(pipeline_config: Config, corpus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mode = pipeline_config.execution_mode

    if (mode not in _RESULTS):
        _RESULTS[mode] = ap_.run_pipeline(pipeline_config, corpus)

    yield _RESULTS[mode]


def _dns(result: pd.DataFrame) -> pd.DataFrame:
    return result[result["telemetry_class"] == ap_.DNS_CLASS]


def _http(result: pd.DataFrame) -> pd.DataFrame:
    return result[result["telemetry_class"] == ap_.HTTP_CLASS]


def _tunnelling(dns: pd.DataFrame, entropy: bool = True, length: bool = True) -> pd.Series:
    """Distinct qualifying subdomains per registered domain, the way R-B-L7-001's search counts them.

    Either per-query condition can be switched off, which is how the counterfactual checks ask what a control would
    have done without the condition that stops it.
    """
    keep = pd.Series(True, index=dns.index)

    if (entropy):
        keep &= dns["dns_subdomain_entropy"].astype("Float64").fillna(0) > ap_.ENTROPY_THRESHOLD

    if (length):
        keep &= dns["dns_mean_label_length"].astype("Float64").fillna(0) > ap_.LABEL_LENGTH_THRESHOLD

    return dns[keep].groupby("dns_registered_domain")["dns_subdomain"].nunique()


def _fires_dns(dns: pd.DataFrame, **conditions) -> set:
    counts = _tunnelling(dns, **conditions)

    return set(counts[counts > ap_.SUBDOMAIN_THRESHOLD].index)


def _enumerating(http: pd.DataFrame, paths: bool = True, ratio: bool = True) -> set:
    """Clients satisfying R-D-L7-005 on some row, the way its search reads them.

    Both conditions are read off the same row, so they held over the same ten-minute window. The ratio is tested as a
    multiplication, which is true for a client with refusals and no successes.
    """
    keep = pd.Series(True, index=http.index)

    if (paths):
        keep &= http["http_distinct_paths"].astype("Int64").fillna(0) > ap_.PATH_THRESHOLD

    if (ratio):
        errors = http["http_4xx_in_window"].astype("Float64")
        successes = http["http_2xx_in_window"].astype("Float64")
        keep &= (errors > ap_.ERROR_RATIO_THRESHOLD * successes).fillna(False)

    return set(http[keep]["src_ip"])


def _split(frame: pd.DataFrame, parts: int) -> list:
    size = max(1, len(frame) // parts)

    return [frame.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(frame), size)]


def _windows(frame: pd.DataFrame) -> list:
    period_ns = ap_.PERIOD_SECONDS * ap_.NS_PER_SECOND

    return [window_id_from_timestamp(int(stamp), period_ns) for stamp in frame["event_time"]]


def _permuted(corpus: dict, seed: int) -> dict:
    return {
        name: permute_within_contiguous_groups(frame, _windows(frame), seed=seed)
        for (name, frame) in corpus.items()
    }


# --- Check 1: the corpus is fixed and shaped like a resolver and a proxy ----------------------------------------


def test_corpus_is_fixed(corpus: dict[str, pd.DataFrame]):
    again = ap_.build_corpus()

    assert set(again) == set(corpus) == set(ap_.TELEMETRY_CLASSES)

    for name in corpus:
        pd.testing.assert_frame_equal(corpus[name], again[name])


def test_corpus_is_shaped_like_a_resolver_and_a_proxy(corpus: dict[str, pd.DataFrame]):
    for column in ("src_ip", "query_name", "query_type"):
        assert column in corpus[ap_.DNS_CLASS].columns, column

    for column in ("src_ip", "dst_ip", "http_method", "url_path", "status_code", "user_agent"):
        assert column in corpus[ap_.HTTP_CLASS].columns, column

    for frame in corpus.values():
        assert set(ap_.ID_COLUMNS) <= set(frame.columns)
        assert frame["collector_seq"].is_monotonic_increasing
        assert frame["event_time"].is_monotonic_increasing


# --- Checks 2 through 6: determinism ---------------------------------------------------------------------------


@pytest.mark.gpu_and_cpu_mode
def test_double_run_diff(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    second = ap_.run_pipeline(pipeline_config, corpus)

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

    rendered = ap_.render(result)

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

    assert diff_frames(result, ap_.run_pipeline(pipeline_config, corpus, batches=thirds)) is None
    assert diff_frames(result, ap_.run_pipeline(pipeline_config, corpus, batches=by_row)) is None


@pytest.mark.gpu_and_cpu_mode
def test_permutation_within_windows(pipeline_config: Config, corpus: dict[str, pd.DataFrame], result: pd.DataFrame):
    for seed in (1, 2, 3):
        shuffled = ap_.run_pipeline(pipeline_config, _permuted(corpus, seed))

        assert diff_frames(result, shuffled) is None, seed


@pytest.mark.gpu_and_cpu_mode
def test_permutation_check_has_teeth(pipeline_config: Config, corpus: dict):
    # The negative control for the check above. Both stages keep windowed state, so removing the imposed order has
    # to change the answer, or the check is passing over a pipeline that never needed it.
    ordered = ap_.run_pipeline(pipeline_config, corpus, impose_order=False)
    shuffled = ap_.run_pipeline(pipeline_config, _permuted(corpus, 5), impose_order=False)

    assert diff_frames(ordered, shuffled) is not None


# --- R-B-L7-001: the tunnel, and one control per condition ------------------------------------------------------


@pytest.mark.cpu_mode
def test_the_tunnel_is_the_only_domain_that_clears_all_three(result: pd.DataFrame):
    assert _fires_dns(_dns(result)) == {ap_.TUNNEL_DOMAIN}


@pytest.mark.cpu_mode
def test_the_content_delivery_network_is_stopped_by_label_length_alone(result: pd.DataFrame):
    # The guide's own warning, asserted: entropy alone flags every content delivery network. Take away the length
    # condition and this CDN fires -- with a margin, not by one name.
    dns = _dns(result)

    assert ap_.CDN_DOMAIN not in _fires_dns(dns)
    assert ap_.CDN_DOMAIN in _fires_dns(dns, length=False)
    assert _tunnelling(dns, length=False)[ap_.CDN_DOMAIN] > ap_.SUBDOMAIN_THRESHOLD + 10


@pytest.mark.cpu_mode
def test_the_tenant_hostname_is_stopped_by_the_count_alone(result: pd.DataFrame):
    # Every one of its queries clears entropy and label length. What it lacks is a second name.
    dns = _dns(result)
    counts = _tunnelling(dns)

    assert counts[ap_.TENANT_DOMAIN] == 1
    assert ap_.TENANT_DOMAIN not in _fires_dns(dns)

    tenant = dns[dns["dns_registered_domain"] == ap_.TENANT_DOMAIN]

    assert (tenant["dns_subdomain_entropy"] > ap_.ENTROPY_THRESHOLD).all()
    assert (tenant["dns_mean_label_length"] > ap_.LABEL_LENGTH_THRESHOLD).all()


@pytest.mark.cpu_mode
def test_the_saas_provider_clears_the_count_and_nothing_else(result: pd.DataFrame):
    # A hundred distinct subdomains alone flags any large SaaS provider, which is why the count is the third
    # condition rather than the only one.
    dns = _dns(result)
    saas = dns[dns["dns_registered_domain"] == ap_.SAAS_DOMAIN]

    assert saas["dns_subdomains_per_domain"].max() > ap_.SUBDOMAIN_THRESHOLD
    assert ap_.SAAS_DOMAIN in _fires_dns(dns, entropy=False, length=False)
    assert ap_.SAAS_DOMAIN not in _fires_dns(dns)


@pytest.mark.cpu_mode
def test_counting_every_subdomain_rather_than_qualifying_ones_would_merge_different_traffic(result: pd.DataFrame):
    # Why the search counts distinct subdomains among the queries that clear the other two conditions, rather than
    # reading the stage's own count of every subdomain. The two figures differ for every domain but the tunnel, and
    # the stage's figure is what a search would reach for if nothing said otherwise.
    dns = _dns(result)
    everything = dns.groupby("dns_registered_domain")["dns_subdomains_per_domain"].max()
    qualifying = _tunnelling(dns)

    assert everything[ap_.CDN_DOMAIN] > ap_.SUBDOMAIN_THRESHOLD
    assert qualifying.get(ap_.CDN_DOMAIN, 0) == 0
    assert everything[ap_.TUNNEL_DOMAIN] == qualifying[ap_.TUNNEL_DOMAIN]


@pytest.mark.cpu_mode
def test_ordinary_browsing_clears_nothing(result: pd.DataFrame):
    browsing = _dns(result)[_dns(result)["src_ip"] == ap_.BROWSER]

    assert (browsing["dns_subdomain_entropy"].astype("Float64").fillna(0) < ap_.ENTROPY_THRESHOLD).all()
    assert (browsing["dns_mean_label_length"].astype("Float64").fillna(0) < ap_.LABEL_LENGTH_THRESHOLD).all()


# --- R-D-L7-005: the enumerators, and one control per condition --------------------------------------------------


@pytest.mark.cpu_mode
def test_both_enumerators_fire_and_nothing_else_does(result: pd.DataFrame):
    assert _enumerating(_http(result)) == {ap_.ENUMERATOR, ap_.EMPTY_HANDED}


@pytest.mark.cpu_mode
def test_the_enumerator_that_found_nothing_fires_only_because_the_rule_is_a_multiplication(result: pd.DataFrame):
    # The client the rule is most for, and the one a ratio column would lose. It has had no success at all, so its
    # 4xx:2xx ratio is undefined on every row; a search thresholding that column would never see it.
    http = _http(result)
    empty = http[http["src_ip"] == ap_.EMPTY_HANDED]

    assert empty["http_4xx_to_2xx_ratio"].isna().all()
    assert (empty["http_2xx_in_window"].astype("Int64").fillna(0) == 0).all()

    thresholded_on_the_column = http[
        (http["http_distinct_paths"].astype("Int64").fillna(0) > ap_.PATH_THRESHOLD)
        & (http["http_4xx_to_2xx_ratio"].astype("Float64").fillna(0) > ap_.ERROR_RATIO_THRESHOLD)]

    assert ap_.EMPTY_HANDED not in set(thresholded_on_the_column["src_ip"])
    assert ap_.EMPTY_HANDED in _enumerating(http)


@pytest.mark.cpu_mode
def test_the_crawler_is_stopped_by_the_ratio_alone(result: pd.DataFrame):
    # As many distinct paths as the enumerator, nearly all of them there.
    http = _http(result)

    assert ap_.CRAWLER not in _enumerating(http)
    assert ap_.CRAWLER in _enumerating(http, ratio=False)


@pytest.mark.cpu_mode
def test_the_broken_client_is_stopped_by_the_path_count_alone(result: pd.DataFrame):
    # Refused on every request, on one path. The distinct count is what keeps a misconfiguration out of the rule.
    http = _http(result)

    assert ap_.BROKEN_CLIENT not in _enumerating(http)
    assert ap_.BROKEN_CLIENT in _enumerating(http, paths=False)
    assert http[http["src_ip"] == ap_.BROKEN_CLIENT]["http_distinct_paths"].max() == 1


@pytest.mark.cpu_mode
def test_redirects_stay_out_of_the_ratio(result: pd.DataFrame):
    reader = _http(result)[_http(result)["src_ip"] == ap_.READER]
    redirects = reader[reader["http_status_class"] == "3xx"]

    assert len(redirects) > 0
    assert redirects["http_4xx_in_window"].isna().all()


# --- Minimization: the sensitive columns neither rule needs at the wire -----------------------------------------

SAVEDSEARCHES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..",
                             "..",
                             "..",
                             "examples",
                             "splunk_lineage_app",
                             "TA-morpheus-lineage",
                             "default",
                             "savedsearches.conf")

MINIMIZATION_KEY = b"test-key-not-a-secret-0123456789"


def _search(name: str) -> str:
    with open(SAVEDSEARCHES, encoding="utf-8") as handle:
        text = handle.read()

    return text.split(f"[{name}]", 1)[1].split("action.correlationsearch.label", 1)[0]


@pytest.mark.cpu_mode
def test_the_tunnelling_rule_survives_pseudonymizing_the_names(result: pd.DataFrame):
    # `query_name` and `dns_subdomain` are browsing history, and the claim in `personal_data` is that they can be
    # replaced with keyed digests at the wire without changing R-B-L7-001's answer, because the rule counts
    # distinct values and a keyed digest preserves distinctness. Asserted, not left in prose.
    from morpheus.utils.personal_data import pseudonymize  # pylint: disable=import-outside-toplevel

    dns = _dns(result).copy()
    before = _tunnelling(dns)

    dns["dns_subdomain"] = pseudonymize(list(dns["dns_subdomain"]), MINIMIZATION_KEY)
    after = _tunnelling(dns)

    assert dict(before) == dict(after)
    assert _fires_dns(dns) == {ap_.TUNNEL_DOMAIN}


@pytest.mark.cpu_mode
def test_the_enumeration_rule_never_reads_the_path_itself():
    # The other half of the same claim: R-D-L7-005 reads the counts the stage computed from `url_path`, never the
    # path, so an estate can drop the column at the wire and keep the rule.
    search = _search("R-D-L7-005 - Enumeration")

    assert "url_path" not in search.split("search =", 1)[1]
    assert "http_distinct_paths" in search


# --- Stamping --------------------------------------------------------------------------------------------------


@pytest.mark.cpu_mode
def test_every_row_is_stamped_at_layer_seven_and_keyed_on_its_client(result: pd.DataFrame):
    # Both classes seal on the client. For DNS that is a choice the guide leaves open -- its `hostname` could be
    # the host that asked or the name it asked for -- and the host that asked is what behaves.
    assert set(result["osi_layer"]) == {ap_.OSI_LAYER}
    assert set(result["telemetry_class"]) == set(ap_.TELEMETRY_CLASSES)
    assert list(result["entity_key"]) == list(result["src_ip"])
    assert result["lineage_id"].notna().all()
    assert result["window_id"].notna().all()
