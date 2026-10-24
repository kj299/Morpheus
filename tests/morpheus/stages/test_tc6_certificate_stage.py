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

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.stages.telemetry.tc6_certificate_stage import TC6CertificateStage
from morpheus.utils.collector_schemas import VALIDATION_RESULTS
from morpheus.utils.type_utils import get_df_class

SECOND = 10**9
START = 10**18

CORP = "CN=Corp Issuing CA, O=Example"
PROXY = "CN=Interception Proxy, O=Unknown"
EXTERNAL = "93.184.216.34"
INTERNAL = "10.0.1.20"
DAY = 86400 * SECOND


def handshakes(issuers, destinations=None, validations=None, spans=None) -> dict:
    count = len(issuers)

    return {
        "dst_ip": list(destinations) if destinations is not None else [EXTERNAL] * count,
        "certificate_issuer": list(issuers),
        "validation_result": list(validations) if validations is not None else ["ok"] * count,
        "certificate_not_before": [0] * count,
        "certificate_not_after": [(span if spans is not None else 90) * DAY for span in (spans or [90] * count)],
        "event_time": [START + index * SECOND for index in range(count)],
    }


def truthy(value) -> bool:
    """A column value as a plain bool, with a null reading as False.

    `bool(pd.NA)` raises rather than returning False, and every flag on these rows is nullable because "not yet
    answerable" is a distinct state from "no". Tests that mean "this did not fire" say so through this.
    """
    return False if pd.isna(value) else bool(value)


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC6CertificateStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_an_issuer_nobody_in_the_estate_has_seen_is_new_to_it(config: Config):
    # Two destinations, each presenting only the corporate issuer for a week; then a third, never seen before,
    # presenting an authority nobody in the estate has seen. The per-destination reference cannot say so -- the new
    # destination has no reference -- and the estate-wide history can.
    payload = handshakes([CORP, CORP, PROXY, CORP, PROXY],
                         destinations=[EXTERNAL, INTERNAL, "203.0.113.9", INTERNAL, EXTERNAL])
    payload["event_time"] = [START, START + DAY, START + 8 * DAY, START + 8 * DAY + SECOND, START + 9 * DAY]
    result = run(config, payload)

    assert result["cert_issuer_new_to_estate"].tolist()[2:] == [True, False, False]
    assert pd.isna(result["cert_issuer_established"].iloc[2])


@pytest.mark.gpu_and_cpu_mode
def test_nothing_is_new_to_the_estate_during_its_warmup(config: Config):
    result = run(config, handshakes([CORP, PROXY]))

    assert result["cert_issuer_new_to_estate"].isna().all()


@pytest.mark.gpu_and_cpu_mode
def test_an_issuer_last_seen_outside_the_window_is_new_again(config: Config):
    payload = handshakes([PROXY, CORP, PROXY])
    payload["event_time"] = [START, START + 8 * DAY, START + 40 * DAY]
    # The month's jump is the point of the test, and it is further than the clock guard believes by default.
    result = run(config, payload, max_clock_skew_seconds=60 * 86400)

    assert bool(result["cert_issuer_new_to_estate"].iloc[2])


@pytest.mark.gpu_and_cpu_mode
def test_an_issuer_a_destination_has_never_presented_differs_from_its_reference(config: Config):
    result = run(config, handshakes([CORP] * 6 + [PROXY]), min_samples=3)

    assert list(result["cert_issuer_established"])[-1] == CORP
    assert truthy(list(result["cert_issuer_differs"])[-1]) is True


@pytest.mark.gpu_and_cpu_mode
def test_a_destination_presenting_its_usual_issuer_is_quiet(config: Config):
    result = run(config, handshakes([CORP] * 7), min_samples=3)

    assert not any(truthy(value) for value in list(result["cert_issuer_differs"])[3:])


@pytest.mark.gpu_and_cpu_mode
def test_no_reference_is_published_before_the_history_supports_one(config: Config):
    # Until the floor is reached the rule has nothing to compare against, and a destination nobody has seen
    # enough of should produce silence rather than a verdict.
    result = run(config, handshakes([CORP] * 5), min_samples=3)

    assert [truthy(value) for value in result["cert_issuer_mature"]] == [False, False, False, True, True]


@pytest.mark.gpu_and_cpu_mode
def test_a_destination_behind_several_authorities_says_how_many(config: Config):
    # A difference from a destination that presents four issuers means far less than the same difference from
    # one that has only ever presented one, and the count is what lets a search tell them apart.
    result = run(config, handshakes(["a", "b", "c", "d", "a", "b"]), min_samples=3)

    assert list(result["cert_issuer_distinct"])[-1] == 4


@pytest.mark.gpu_and_cpu_mode
def test_a_self_signed_certificate_to_the_internet_is_flagged(config: Config):
    result = run(config, handshakes([PROXY], validations=["self-signed"]))

    assert truthy(list(result["cert_self_signed"])[0]) is True
    assert truthy(list(result["cert_destination_is_global"])[0]) is True
    assert truthy(list(result["cert_self_signed_external"])[0]) is True


@pytest.mark.gpu_and_cpu_mode
def test_a_self_signed_certificate_inside_the_estate_is_not(config: Config):
    # The negative half of R-D-L6-003, and the reason the rule names `is_global`. Internal self-signed
    # certificates are ordinary: management interfaces, appliances, anything nobody bought a certificate for.
    result = run(config, handshakes([PROXY], destinations=[INTERNAL], validations=["self-signed"]))

    assert truthy(list(result["cert_self_signed"])[0]) is True
    assert truthy(list(result["cert_destination_is_global"])[0]) is False
    assert truthy(list(result["cert_self_signed_external"])[0]) is False


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("spelling", ["self-signed", "self signed certificate", "SELF_SIGNED"])
def test_the_spellings_a_real_feed_emits_are_recognized(config: Config, spelling: str):
    # A rule that recognized only the tidy spelling would silently never fire against the feed most estates
    # actually have, which is the failure mode this whole app is careful about.
    result = run(config, handshakes([PROXY], validations=[spelling]))

    assert truthy(list(result["cert_self_signed"])[0]) is True


@pytest.mark.gpu_and_cpu_mode
def test_a_validated_certificate_is_not_self_signed(config: Config):
    result = run(config, handshakes([CORP], validations=["ok"]))

    assert truthy(list(result["cert_self_signed"])[0]) is False


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize(("raw", "self_signed"),
                         [
                             ("self signed certificate", True),
                             ("self signed certificate in certificate chain", True),
                             ("unable to get local issuer certificate", False),
                             ("certificate has expired", False),
                         ])
def test_the_normalizer_s_vocabulary_is_the_stage_s(config: Config, raw: str, self_signed: bool):
    # Held through the normalizer the fork ships for Zeek, so the two vocabularies cannot drift apart again: an
    # issuer the verifier does not hold is not a certificate signing itself.
    normalized = VALIDATION_RESULTS[raw]
    result = run(config, handshakes([PROXY], validations=[normalized]))

    assert truthy(list(result["cert_self_signed"])[0]) is self_signed


@pytest.mark.gpu_and_cpu_mode
def test_the_validity_window_is_carried_in_days(config: Config):
    result = run(config, handshakes([CORP, CORP], spans=[90, 3]))

    assert list(result["cert_validity_days"]) == [90.0, 3.0]


@pytest.mark.gpu_and_cpu_mode
def test_a_rotation_stops_differing_once_it_is_what_the_destination_does(config: Config):
    # A certificate authority migration is a change and then a fact. The reference follows it, or the rule
    # reports the estate's own new normal every day until somebody silences it.
    result = run(config, handshakes([CORP] * 6 + [PROXY] * 10), min_samples=3)
    differs = [truthy(value) for value in result["cert_issuer_differs"]]

    assert differs[6] is True
    assert differs[-1] is False


@pytest.mark.gpu_and_cpu_mode
def test_a_handshake_without_an_issuer_teaches_the_reference_nothing(config: Config):
    result = run(config, handshakes([CORP] * 4 + [None] + [CORP]), min_samples=3)

    assert pd.isna(list(result["cert_issuer_differs"])[4])
    assert list(result["cert_issuer_established"])[-1] == CORP


@pytest.mark.cpu_mode
def test_a_missing_column_is_refused(config: Config):
    with pytest.raises(KeyError, match="certificate_issuer"):
        run(config, {"dst_ip": [EXTERNAL], "event_time": [START]})


@pytest.mark.gpu_and_cpu_mode
def test_a_destination_is_the_name_at_the_address_where_the_handshake_has_one(config: Config):
    payload = handshakes(["CN=A", "CN=B", "CN=C"])
    payload["sni"] = ["shop.example.net", "Mail.Example.net", None]
    result = run(config, payload)

    assert list(
        result["tls_destination_key"]) == [f"{EXTERNAL}:shop.example.net", f"{EXTERNAL}:mail.example.net", EXTERNAL]


@pytest.mark.gpu_and_cpu_mode
def test_two_names_on_one_address_keep_two_references(config: Config):
    # Each name has presented one authority. Keyed on the address the two alternate, and every visit differs from
    # an established issuer half the time; keyed on the name, neither ever differs.
    issuers = ["CN=Shop CA", "CN=Mail CA"] * 6
    payload = handshakes(issuers)
    payload["sni"] = ["shop.example.net", "mail.example.net"] * 6

    by_name = run(config, payload, min_samples=2)
    by_address = run(config, payload, min_samples=2, key_columns=["dst_ip"])

    assert not by_name["cert_issuer_differs"].fillna(False).astype(bool).any()
    assert by_address["cert_issuer_differs"].fillna(False).astype(bool).any()
    assert set(by_address["tls_destination_key"]) == {EXTERNAL}


@pytest.mark.cpu_mode
def test_an_explicit_key_ignores_the_name(config: Config):
    payload = handshakes(["CN=A"])
    payload["sni"] = ["shop.example.net"]

    assert list(run(config, payload, key_columns=["dst_ip"])["tls_destination_key"]) == [EXTERNAL]


# --- The clock guard ------------------------------------------------------------------------------------------

YEARS_AHEAD_NS = 10 * 365 * 86400 * 10**9
"""A collector clock wrong by a decade: the failure an unset RTC produces, far beyond any gap a real feed has."""


def _without_row(payload: dict, position: int) -> dict:
    """The same frame with one row left out, which is what a refused row must be indistinguishable from."""
    return {
        name: [value for (index, value) in enumerate(values) if index != position]
        for (name, values) in payload.items()
    }


def _pandas(meta: MessageMeta) -> pd.DataFrame:
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _drive(config: Config, stage, payload: dict) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    stage.on_data(meta)

    return _pandas(meta)


@pytest.mark.gpu_and_cpu_mode
def test_a_time_years_ahead_of_the_stream_is_refused_and_moves_nothing(config: Config):
    # Row 2 is stamped a decade ahead. Admitted, it would drive every horizon in the stage past the rows before it
    # and refuse the real row after it as out of order; refused, the stream reads exactly as if it had never arrived.
    payload = handshakes([CORP] * 4)
    payload["event_time"] = [START, START + SECOND, START + YEARS_AHEAD_NS, START + 2 * SECOND]

    with_bad = _drive(config, TC6CertificateStage(config), payload)
    control = _drive(config, TC6CertificateStage(config), _without_row(payload, 2))

    assert [bool(value) for value in with_bad["clock_refused"]] == [False, False, True, False]
    assert pd.isna(with_bad["cert_issuer_distinct"][2])

    kept = with_bad.drop(index=2).reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, control, check_dtype=False)


@pytest.mark.gpu_and_cpu_mode
def test_a_frame_with_nothing_refused_carries_the_column_as_false(config: Config):
    # The mark is written on every frame, so whether a frame carries the column is a property of the stage and not of
    # which rows shared a batch; a believed row reads False, which the health record counts as 0 rather than null.
    payload = handshakes([CORP] * 4)

    frame = _drive(config, TC6CertificateStage(config), payload)

    assert [bool(value) for value in frame["clock_refused"]] == [False] * len(frame)


def test_a_clock_bound_that_admits_nothing_is_refused(config: Config):
    with pytest.raises(ValueError, match="max_clock_skew_seconds"):
        TC6CertificateStage(config, max_clock_skew_seconds=0)


@pytest.mark.gpu_and_cpu_mode
def test_a_stop_and_restore_round_trip_keeps_the_clock(config: Config):
    # A fresh instance believes the first time it is shown, so a decade-ahead row that opens a resumed process would
    # set its clock. The clock is part of the snapshot, and the restored instance refuses it as the stopped one would.
    payload = handshakes([CORP] * 4)
    payload["event_time"] = [START, START + SECOND, START + YEARS_AHEAD_NS, START + 2 * SECOND]
    (before, after) = ({
        name: values[:2]
        for (name, values) in payload.items()
    }, {
        name: values[2:]
        for (name, values) in payload.items()
    })

    stopped = TC6CertificateStage(config)
    _drive(config, stopped, before)

    resumed = TC6CertificateStage(config)
    resumed.restore_state(stopped.checkpoint_state())

    assert [bool(value) for value in _drive(config, resumed, after)["clock_refused"]] == [True, False]
