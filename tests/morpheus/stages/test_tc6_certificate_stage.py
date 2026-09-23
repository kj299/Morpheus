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
