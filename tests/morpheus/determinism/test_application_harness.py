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
Control 13's six checks against the DNS, HTTP and service classes, and their four rules' predicates over the corpus.

Every rule here is a conjunction, and a conjunction is only tested when each of its conditions is shown to be the one
that matters. A corpus that proved R-B-L7-001 only against ordinary browsing would pass with any of its four
conditions deleted, and the guide's own warning -- entropy alone flags every content delivery network -- would be a
sentence rather than an assertion. So the checks below are counterfactual: for each benign actor, they assert the
rule would have fired had the one condition that stops it been removed. That is what makes each condition
load-bearing rather than decorative.

The predicates are evaluated the way the shipped searches evaluate them, and the search for R-B-L7-001 does
something the stage's own count does not: it keeps only the queries that clear entropy and label length and counts
distinct subdomains among those. Counting every subdomain would let a SaaS domain with a hundred ordinary tenant
names and one random hostname satisfy all three conditions on different traffic. The count is taken per hour, as the
search groups it, so a week of an agent's ordinary hours never adds up to one.

Three of the rules also measure a client or a service against its own days, through a baseline the stages publish.
Each has a control that clears every other condition every day of its life and that its own history alone keeps quiet,
and each has an actor too new to have a history, held to the rule's floor alone.
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
import host_identity  # noqa: E402

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


def _service(result: pd.DataFrame) -> pd.DataFrame:
    return result[result["telemetry_class"] == ap_.SERVICE_CLASS]


def _true(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("boolean").fillna(False).astype(bool)


def _measured(frame: pd.DataFrame, column: str) -> pd.Series:
    """Rows a baseline condition admits: a client with no week of days, or one at least twice its busiest day."""
    unmeasured = ~_true(frame, f"{column}_baseline_mature")
    far_beyond = frame[f"{column}_ratio"].astype("Float64").fillna(0) >= ap_.BASELINE_MULTIPLE

    return unmeasured | far_beyond


def _tunnelling(dns: pd.DataFrame, entropy: bool = True, length: bool = True, baseline: bool = True) -> pd.Series:
    """Distinct qualifying subdomains per registered domain, in its busiest hour, the way R-B-L7-001's search counts
    them.

    Any per-query condition can be switched off, which is how the counterfactual checks ask what a control would
    have done without the condition that stops it.
    """
    keep = pd.Series(True, index=dns.index)

    if (entropy):
        keep &= dns["dns_subdomain_entropy"].astype("Float64").fillna(0) > ap_.ENTROPY_THRESHOLD

    if (length):
        keep &= dns["dns_mean_label_length"].astype("Float64").fillna(0) > ap_.LABEL_LENGTH_THRESHOLD

    if (baseline):
        keep &= _measured(dns, "dns_subdomains_per_domain")

    hourly = dns[keep].groupby(["dns_registered_domain", "window_id"])["dns_subdomain"].nunique()

    return hourly.groupby(level=0).max()


def _fires_dns(dns: pd.DataFrame, **conditions) -> set:
    counts = _tunnelling(dns, **conditions)

    return set(counts[counts > ap_.SUBDOMAIN_THRESHOLD].index)


def _enumerating(http: pd.DataFrame, paths: bool = True, ratio: bool = True, baseline: bool = True) -> set:
    """Clients satisfying R-D-L7-005 on some row, the way its search reads them.

    Every condition is read off the same row, so they held over the same ten-minute window. The ratio is tested as a
    multiplication, which is true for a client with refusals and no successes.
    """
    keep = pd.Series(True, index=http.index)

    if (paths):
        keep &= http["http_distinct_paths"].astype("Int64").fillna(0) > ap_.PATH_THRESHOLD

    if (ratio):
        errors = http["http_4xx_in_window"].astype("Float64")
        successes = http["http_2xx_in_window"].astype("Float64")
        keep &= (errors > ap_.ERROR_RATIO_THRESHOLD * successes).fillna(False)

    if (baseline):
        keep &= _measured(http, "http_distinct_paths")

    return set(http[keep]["src_ip"])


def _generating(dns: pd.DataFrame, domains: bool = True, refused: bool = True, baseline: bool = True) -> set:
    """Clients satisfying R-B-L7-008 on some row, the way its search reads them."""
    keep = pd.Series(True, index=dns.index)

    if (domains):
        keep &= dns["dns_domains_per_client"].astype("Int64").fillna(0) > ap_.DGA_DOMAIN_THRESHOLD

    if (refused):
        keep &= dns["dns_nxdomain_ratio"].astype("Float64").fillna(0) > ap_.NXDOMAIN_THRESHOLD

    if (baseline):
        keep &= _measured(dns, "dns_domains_per_client")

    return set(dns[keep]["src_ip"])


def _service_hours(service: pd.DataFrame) -> pd.DataFrame:
    """Each service's hours, reduced the way R-B-L7-007's search reduces them."""
    frame = service.assign(clients=service["service_clients_in_window"].astype("Int64"),
                           ratio=service["service_clients_ratio"].astype("Float64"),
                           novel=_true(service, "service_client_fingerprint_first_seen").astype(int),
                           mature=_true(service, "service_clients_baseline_mature").astype(int))

    return frame.groupby(["service_key", "window_id"]).agg(clients=("clients", "max"),
                                                           ratio=("ratio", "max"),
                                                           novel=("novel", "sum"),
                                                           mature=("mature", "max"))


def _serving(service: pd.DataFrame,
             floor: bool = True,
             ratio: bool = True,
             novel: bool = True,
             mature: bool = True) -> set:
    """Services satisfying R-B-L7-007 in some hour."""
    hours = _service_hours(service)
    keep = pd.Series(True, index=hours.index)

    if (mature):
        keep &= hours["mature"] > 0

    if (floor):
        keep &= hours["clients"].fillna(0) >= ap_.SERVICE_CLIENT_FLOOR

    if (ratio):
        keep &= hours["ratio"].fillna(0) >= ap_.BASELINE_MULTIPLE

    if (novel):
        keep &= hours["novel"] > 0

    return set(hours[keep].index.get_level_values(0))


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

    assert set(again) == set(corpus) == set(ap_.TELEMETRY_CLASSES) | set(host_identity.HOST_CLASSES)

    for name in corpus:
        pd.testing.assert_frame_equal(corpus[name], again[name])


def test_corpus_is_shaped_like_a_resolver_and_a_proxy(corpus: dict[str, pd.DataFrame]):
    for column in ("src_ip", "query_name", "query_type"):
        assert column in corpus[ap_.DNS_CLASS].columns, column

    for column in ("src_ip", "dst_ip", "http_method", "url_path", "status_code", "user_agent"):
        assert column in corpus[ap_.HTTP_CLASS].columns, column

    for column in ("src_ip", "dst_ip", "dst_port", "sni", "bytes_total", "ja4_client"):
        assert column in corpus[ap_.SERVICE_CLASS].columns, column

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
def test_the_tunnel_is_the_only_domain_that_clears_all_four(result: pd.DataFrame):
    dns = _dns(result)
    tunnel = dns[dns["dns_registered_domain"] == ap_.TUNNEL_DOMAIN]

    assert _fires_dns(dns) == {ap_.TUNNEL_DOMAIN}
    # Through its baseline, not around it: the tunneller has a week of ordinary days, and the tunnel is a hundred
    # and thirty times the busiest of them.
    assert _true(tunnel, "dns_subdomains_per_domain_baseline_mature").all()
    assert tunnel["dns_subdomains_per_domain_ratio"].astype("Float64").max() > 100


@pytest.mark.cpu_mode
def test_the_reputation_agent_is_stopped_by_its_own_history_alone(result: pd.DataFrame):
    # The case the baseline exists for. Every one of its lookups clears entropy and label length, and on the day the
    # rules are read it clears the hundred too; it is no busier than any day of its life, and that is all that keeps
    # it out.
    dns = _dns(result)
    today = dns[(dns["src_ip"] == ap_.HASH_LOOKUP) & _true(dns, "dns_subdomains_per_domain_baseline_mature")]

    assert ap_.REPUTATION_DOMAIN not in _fires_dns(dns)
    assert ap_.REPUTATION_DOMAIN in _fires_dns(dns, baseline=False)
    assert _tunnelling(dns, baseline=False)[ap_.REPUTATION_DOMAIN] == ap_.REPUTATION_TODAY
    assert today["dns_subdomains_per_domain_ratio"].astype("Float64").max() < ap_.BASELINE_MULTIPLE


@pytest.mark.cpu_mode
def test_a_client_with_no_week_of_days_is_held_to_the_floor_alone(result: pd.DataFrame):
    # The content delivery network, the tenant and the SaaS provider are new to the estate on the day the rules are
    # read. They have no baseline, so the baseline admits them, and each is stopped by the condition it always was.
    dns = _dns(result)

    for client in (ap_.CDN_CLIENT, ap_.TENANT_CLIENT, ap_.SAAS_CLIENT):
        rows = dns[dns["src_ip"] == client]

        assert not _true(rows, "dns_subdomains_per_domain_baseline_mature").any(), client
        assert rows["dns_subdomains_per_domain_baseline"].isna().all(), client


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
    qualifying = _tunnelling(dns, baseline=False)

    assert everything[ap_.CDN_DOMAIN] > ap_.SUBDOMAIN_THRESHOLD
    assert qualifying.get(ap_.CDN_DOMAIN, 0) == 0
    assert everything[ap_.TUNNEL_DOMAIN] == qualifying[ap_.TUNNEL_DOMAIN]


@pytest.mark.cpu_mode
def test_ordinary_browsing_clears_nothing(result: pd.DataFrame):
    browsing = _dns(result)[_dns(result)["src_ip"] == ap_.BROWSER]

    assert (browsing["dns_subdomain_entropy"].astype("Float64").fillna(0) < ap_.ENTROPY_THRESHOLD).all()
    assert (browsing["dns_mean_label_length"].astype("Float64").fillna(0) < ap_.LABEL_LENGTH_THRESHOLD).all()


# --- R-P-L7-009: the slow tunnel, under every literal and over the trajectory ----------------------------------

RISING_WINDOWS = 4
RISE_SIGMAS = 1.5
"""R-P-L7-009's trajectory, which is R-P-L5-006's asked of a client's busiest domain: four consecutive rising days, by
more than one and a half of the client's own day-to-day deviations in all. The validation package reads the two
numbers from the stanza; here they are stated, so a changed rule and a changed corpus fail differently."""


@pytest.mark.cpu_mode
def test_the_slow_tunnel_is_watchlisted_from_its_fifth_day(result: pd.DataFrame):
    # A few more chunks every day than the day before: the busiest domain of each day rises by the same step, so
    # the run of rising days is the day's number, and the rule reports the client from the fifth day, the first the
    # drift stage calls mature, through the event day.
    dns = _dns(result)
    slow = dns[dns["src_ip"] == ap_.SLOW_TUNNELLER]
    days = slow["event_time"].astype("int64") // (ap_.DAY_SECONDS * ap_.NS_PER_SECOND)
    daily = slow.assign(day=days).groupby("day").agg(busiest=("dns_subdomains_per_domain", "max"),
                                                     rising=("drift_rising_windows", "max"),
                                                     rise_sigmas=("drift_rise_sigmas", "max"),
                                                     mature=("drift_mature", "max")).reset_index()

    assert list(daily["busiest"]) == [ap_.SLOW_START + ap_.SLOW_STEP * day for day in range(ap_.HISTORY_DAYS + 1)]
    assert list(daily["rising"]) == list(range(1, ap_.HISTORY_DAYS + 2))

    fired = daily[daily["mature"].astype("boolean").fillna(False) & (daily["rising"] >= RISING_WINDOWS)
                  & (daily["rise_sigmas"].astype("Float64").fillna(0) > RISE_SIGMAS)]

    assert sorted(fired["day"]) == list(range(RISING_WINDOWS, ap_.HISTORY_DAYS + 1))


@pytest.mark.cpu_mode
def test_the_slow_tunnel_clears_every_literal_but_the_count_and_is_never_twice_its_busiest_day(result: pd.DataFrame):
    # Why the trajectory exists. Each chunk is as random and as long-labelled as the tunnel's, and no day reaches
    # the hundred; by the event day the client has a week of days, and that day is a tenth above its busiest, not
    # twice. R-B-L7-001 is quiet on every day with every condition it has, and would be with the baseline removed.
    dns = _dns(result)
    slow = dns[dns["src_ip"] == ap_.SLOW_TUNNELLER]

    assert (slow["dns_subdomain_entropy"].astype("Float64") > ap_.ENTROPY_THRESHOLD).all()
    assert (slow["dns_mean_label_length"].astype("Float64") > ap_.LABEL_LENGTH_THRESHOLD).all()
    assert slow["dns_subdomains_per_domain"].astype("Int64").max() < ap_.SUBDOMAIN_THRESHOLD
    assert ap_.SLOW_DOMAIN not in _fires_dns(dns)
    assert ap_.SLOW_DOMAIN not in _fires_dns(dns, baseline=False)

    event_day = slow[slow["event_time"].astype("int64") >= ap_.event(0)]

    assert _true(event_day, "dns_subdomains_per_domain_baseline_mature").all()
    assert 1 < event_day["dns_subdomains_per_domain_ratio"].astype("Float64").max() < ap_.BASELINE_MULTIPLE

    # No other client's busiest domain rises for four days: the tunneller's week is flat and then a spike, and the
    # reputation agent's days are as busy as each other.
    others = dns[dns["src_ip"] != ap_.SLOW_TUNNELLER]

    assert others["drift_rising_windows"].astype("Float64").max() < RISING_WINDOWS


# --- R-D-L7-005: the enumerators, and one control per condition --------------------------------------------------


@pytest.mark.cpu_mode
def test_both_enumerators_fire_and_nothing_else_does(result: pd.DataFrame):
    assert _enumerating(_http(result)) == {ap_.ENUMERATOR, ap_.EMPTY_HANDED}


@pytest.mark.cpu_mode
def test_the_scanner_is_stopped_by_its_own_history_alone(result: pd.DataFrame):
    # An enumerator by every other measure, every morning of its life.
    http = _http(result)
    today = http[(http["src_ip"] == ap_.SCANNER) & _true(http, "http_distinct_paths_baseline_mature")]

    assert ap_.SCANNER not in _enumerating(http)
    assert ap_.SCANNER in _enumerating(http, baseline=False)
    assert today["http_distinct_paths_baseline"].astype("Int64").max() == ap_.SCANNER_HISTORY


@pytest.mark.cpu_mode
def test_the_enumerator_fires_through_its_baseline_and_the_new_one_on_the_floor(result: pd.DataFrame):
    # The enumerator read the news in its browser all week, so it is measured, and it is twenty times its busiest
    # day. The one that found nothing is new to the estate, and is held to the two hundred alone.
    http = _http(result)
    seasoned = http[(http["src_ip"] == ap_.ENUMERATOR) & (http["http_distinct_paths"].astype("Int64") > 200)]
    newcomer = http[http["src_ip"] == ap_.EMPTY_HANDED]

    assert _true(seasoned, "http_distinct_paths_baseline_mature").all()
    assert seasoned["http_distinct_paths_ratio"].astype("Float64").min() > 10 * ap_.BASELINE_MULTIPLE
    assert not _true(newcomer, "http_distinct_paths_baseline_mature").any()


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


@pytest.mark.cpu_mode
def test_only_the_enumerator_brings_client_software_it_has_never_run(result: pd.DataFrame):
    # User-agent novelty per client: the enumerator browsed in Chrome all week and enumerates with curl. Every other
    # client sends what it always has, and a client's first request is novel to nothing.
    http = _http(result)
    novel = http[_true(http, "http_ua_family_first_seen")]

    assert set(zip(novel["src_ip"], novel["http_ua_family"])) == {(ap_.ENUMERATOR, "curl")}

    first = http.sort_values("event_time", kind="mergesort").groupby("src_ip").head(1)

    assert first["http_ua_family_first_seen"].isna().all()


# --- R-B-L7-008: the generator, and one control per condition ----------------------------------------------------


@pytest.mark.cpu_mode
def test_the_generator_is_the_only_client_that_clears_all_three(result: pd.DataFrame):
    dns = _dns(result)
    generator = dns[(dns["src_ip"] == ap_.GENERATOR) & dns["dns_nxdomain_ratio"].notna()]

    assert _generating(dns) == {ap_.GENERATOR}
    assert generator["dns_nxdomain_ratio"].astype("Float64").max() > 0.8


@pytest.mark.cpu_mode
def test_the_stale_suffix_is_stopped_by_the_domain_count_alone(result: pd.DataFrame):
    # Told no about one domain all hour.
    dns = _dns(result)

    assert ap_.STALE_SUFFIX not in _generating(dns)
    assert ap_.STALE_SUFFIX in _generating(dns, domains=False)


@pytest.mark.cpu_mode
def test_the_busy_browser_is_stopped_by_the_nxdomain_share_alone(result: pd.DataFrame):
    dns = _dns(result)

    assert ap_.BUSY_BROWSER not in _generating(dns)
    assert ap_.BUSY_BROWSER in _generating(dns, refused=False)


@pytest.mark.cpu_mode
def test_the_link_checker_is_stopped_by_its_own_history_alone(result: pd.DataFrame):
    dns = _dns(result)

    assert ap_.LINK_CHECKER not in _generating(dns)
    assert ap_.LINK_CHECKER in _generating(dns, baseline=False)


# --- R-B-L7-007: the service, and one control per condition ------------------------------------------------------


@pytest.mark.cpu_mode
def test_the_payroll_api_is_the_only_service_that_clears_all_four(result: pd.DataFrame):
    assert _serving(_service(result)) == {f"{ap_.PAYROLL}:443"}


@pytest.mark.cpu_mode
def test_one_address_terminating_two_names_is_two_services(result: pd.DataFrame):
    # The payroll API and the wiki share an address and a port, which is why a service is the name where there is
    # one. Keyed on the address they would be one service whose readers and new client both doubled.
    service = _service(result)
    behind_the_proxy = service[service["dst_ip"] == ap_.EDGE_PROXY]

    assert set(behind_the_proxy["service_key"]) == {f"{ap_.PAYROLL}:443", f"{ap_.WIKI}:443"}
    assert set(behind_the_proxy["service_key_source"]) == {"sni"}
    assert set(service[service["dst_ip"] == ap_.METRICS_SERVER]["service_key"]) == {
        f"{ap_.METRICS_SERVER}:{ap_.METRICS_PORT}"
    }


@pytest.mark.cpu_mode
@pytest.mark.parametrize(("name", "condition"),
                         [(f"{ap_.WIKI}:443", "novel"), (f"{ap_.METRICS_SERVER}:{ap_.METRICS_PORT}", "ratio"),
                          (f"{ap_.CI}:443", "floor")])
def test_each_quiet_service_is_stopped_by_one_condition_alone(result: pd.DataFrame, name: str, condition: str):
    # The wiki's readers doubled in the browser it knows; one of the metrics collector's usual agents upgraded; a new
    # build tool reached the CI server from a handful of machines.
    service = _service(result)

    assert name not in _serving(service)
    assert name in _serving(service, **{condition: False})


@pytest.mark.cpu_mode
def test_a_service_deployed_this_morning_has_no_days_to_change_from(result: pd.DataFrame):
    service = _service(result)
    new = service[service["service_key"] == f"{ap_.NEW_APP}:443"]

    assert f"{ap_.NEW_APP}:443" not in _serving(service)
    assert new["service_clients_ratio"].isna().all()
    assert new["service_clients_in_window"].astype("Int64").max() == ap_.NEW_APP_CLIENTS
    assert _true(new, "service_client_fingerprint_first_seen").any()


@pytest.mark.cpu_mode
def test_a_request_and_a_connection_carry_the_application_that_answered(result: pd.DataFrame):
    # The CMDB maps a server's port to an application; the enrichment reads it on dst_ip:dst_port at the event's
    # time. A service the CMDB has never heard of is found by nothing, and says so.
    http = _http(result)
    service = _service(result)

    assert set(http["ctx_application_name"]) == {"intranet-web"}
    assert set(service[service["dst_ip"] == ap_.EDGE_PROXY]["ctx_application_name"]) == {"corp-edge-proxy"}
    assert set(service[service["dst_ip"] == ap_.EDGE_PROXY]["ctx_application_service_account"]) == {"svc-edge"}
    assert not _true(service[service["dst_ip"] == ap_.NEW_SERVER], "ctx_application_found").any()


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
def test_every_row_is_stamped_at_layer_seven_and_keyed_on_its_subject(result: pd.DataFrame):
    # DNS and HTTP seal on the client. For DNS that is a choice the guide leaves open -- its `hostname` could be the
    # host that asked or the name it asked for -- and the host that asked is what behaves. A service seals on itself.
    clients = result[result["telemetry_class"] != ap_.SERVICE_CLASS]
    service = _service(result)

    assert set(result["osi_layer"]) == {ap_.OSI_LAYER}
    assert set(result["telemetry_class"]) == set(ap_.TELEMETRY_CLASSES)
    assert list(clients["entity_key"]) == list(clients["src_ip"])
    assert list(service["entity_key"]) == list(service["service_key"])
    assert list(service["chain_anchor"]) == list(service["service_key"])
    assert result["lineage_id"].notna().all()
    assert result["window_id"].notna().all()
