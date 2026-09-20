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
from morpheus.stages.lineage.minimization_stage import MinimizationStage
from morpheus.utils.type_utils import get_df_class

KEY = b"a-key-long-enough-to-be-accepted"


def authentications() -> dict:
    """Three layer 5 authentications by two principals, with the principal under all three of its names."""
    return {
        "telemetry_class": ["tc5_auth", "tc5_auth", "tc5_auth"],
        "user_principal": ["alice@example.com", "bob@example.com", "alice@example.com"],
        "entity_key": ["alice@example.com", "bob@example.com", "alice@example.com"],
        "chain_anchor": ["alice@example.com", "bob@example.com", "alice@example.com"],
        "source_ip": ["203.0.113.5", "203.0.113.6", "198.51.100.7"],
        "source_country": ["US", "US", "FR"],
        "logcount": [4, 7, 9],
        "window_id": [0, 0, 1],
    }


ALL_NAMES = ["user_principal", "entity_key", "chain_anchor"]


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    MinimizationStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_dropping_removes_the_column_and_leaves_the_rest(config: Config):
    result = run(config, authentications(), drop=["source_country", "source_ip"])

    assert "source_country" not in result.columns
    assert "source_ip" not in result.columns
    assert list(result["user_principal"]) == ["alice@example.com", "bob@example.com", "alice@example.com"]
    assert list(result["logcount"]) == [4, 7, 9]


@pytest.mark.gpu_and_cpu_mode
def test_a_pseudonym_replaces_the_principal_and_keeps_the_join(config: Config):
    # The property the whole mechanism depends on: the same person is still the same person afterwards, in
    # every column that named them, or the per-entity story falls apart in the SIEM instead of in the pipeline.
    result = run(config, authentications(), pseudonymize=ALL_NAMES, key=KEY)

    principals = list(result["user_principal"])

    assert "alice@example.com" not in principals
    assert principals[0] == principals[2]
    assert principals[0] != principals[1]
    assert list(result["entity_key"]) == principals
    assert list(result["chain_anchor"]) == principals


@pytest.mark.gpu_and_cpu_mode
def test_a_column_the_batch_does_not_carry_is_not_an_error(config: Config):
    # A policy is written once for an estate and applied to segments carrying different classes, so naming a
    # layer 5 column in a policy that also runs over layer 1 has to be ordinary.
    payload = authentications()
    del payload["source_country"]

    result = run(config, payload, drop=["source_country", "source_ip"])

    assert "source_ip" not in result.columns


@pytest.mark.gpu_and_cpu_mode
def test_pseudonymizing_one_name_for_a_person_is_refused(config: Config):
    # The way this is normally got wrong. At layer 5 the principal is in the frame three times; digesting one
    # of them produces output that looks minimized and is not, and nothing downstream would ever say so.
    with pytest.raises(ValueError, match="pseudonymized elsewhere"):
        run(config, authentications(), pseudonymize=["user_principal"], key=KEY)


@pytest.mark.gpu_and_cpu_mode
def test_the_refusal_names_the_columns_that_still_carry_it(config: Config):
    with pytest.raises(ValueError) as caught:
        run(config, authentications(), pseudonymize=["user_principal"], key=KEY)

    assert "entity_key" in str(caught.value)
    assert "chain_anchor" in str(caught.value)


@pytest.mark.gpu_and_cpu_mode
def test_the_overlap_check_can_be_turned_off_deliberately(config: Config):
    result = run(config, authentications(), pseudonymize=["user_principal"], key=KEY, allow_unmasked_copies=True)

    assert list(result["entity_key"]) == ["alice@example.com", "bob@example.com", "alice@example.com"]
    assert "alice@example.com" not in list(result["user_principal"])


@pytest.mark.gpu_and_cpu_mode
def test_dropping_the_other_names_satisfies_the_check_too(config: Config):
    # Dropping is as good an answer as pseudonymizing, and the check has to accept it or it would force an
    # estate to keep columns it had decided to remove.
    result = run(config,
                 authentications(),
                 pseudonymize=["user_principal"],
                 drop=["entity_key", "chain_anchor"],
                 key=KEY)

    assert "entity_key" not in result.columns
    assert "alice@example.com" not in list(result["user_principal"])


def test_a_bounded_domain_cannot_be_pseudonymized(config: Config):
    with pytest.raises(ValueError, match="fixed by their own definition"):
        MinimizationStage(config, pseudonymize=["source_country"], key=KEY)


def test_pseudonymizing_without_a_key_is_refused(config: Config):
    with pytest.raises(ValueError, match="needs a key"):
        MinimizationStage(config, pseudonymize=["user_principal"])


def test_a_policy_that_names_nothing_is_refused(config: Config):
    with pytest.raises(ValueError, match="names nothing"):
        MinimizationStage(config)


def test_naming_a_column_for_both_is_refused(config: Config):
    with pytest.raises(ValueError, match="both"):
        MinimizationStage(config, drop=["user_principal"], pseudonymize=["user_principal"], key=KEY)


def test_an_unclassified_column_cannot_be_put_in_a_policy(config: Config):
    # A policy naming a column nobody has classified would act on something whose meaning is unrecorded, which
    # is the state the inventory exists to prevent.
    with pytest.raises(ValueError, match="neither a category nor a column"):
        MinimizationStage(config, drop=["some_new_feature"])


@pytest.mark.gpu_and_cpu_mode
def test_a_category_expands_to_its_columns_for_the_class_it_is_given(config: Config):
    result = run(config, authentications(), drop=["locates"], telemetry_class="tc5_auth", pseudonymize=None)

    # `source_country` is `locates`; `entity_key` is not, at layer 5, because there the subject is the person.
    assert "source_country" not in result.columns
    assert "entity_key" in result.columns


@pytest.mark.gpu_and_cpu_mode
def test_a_category_refuses_a_batch_that_does_not_say_which_class_it_is(config: Config):
    # `locates` covers `entity_key` at layer 1 and not at layer 5, so the expansion happens where the class is
    # known. A batch that says nothing gets a refusal rather than the half of the policy that is unambiguous,
    # which would look applied and quietly leave out the columns that most needed deciding.
    payload = authentications()
    del payload["telemetry_class"]

    with pytest.raises(ValueError, match="does not say"):
        run(config, payload, drop=["locates"])


@pytest.mark.gpu_and_cpu_mode
def test_one_policy_means_different_columns_in_different_segments(config: Config):
    # Why the expansion is deferred at all. The same three words are written once for an estate and applied to
    # every segment, and `locates` has to reach the port at layer 1 and not the principal at layer 5.
    ports = {
        "telemetry_class": ["tc1", "tc1"],
        "entity_key": ["hq:sw1:Gi1/0/1", "hq:sw1:Gi1/0/2"],
        "site_id": ["hq", "hq"],
        "uptime": [10, 20],
    }

    at_layer_one = run(config, ports, drop=["locates"])
    at_layer_five = run(config, authentications(), drop=["locates"])

    assert "entity_key" not in at_layer_one.columns
    assert "site_id" not in at_layer_one.columns
    assert "entity_key" in at_layer_five.columns
    assert "source_country" not in at_layer_five.columns


def test_a_category_with_no_ambiguous_members_needs_no_class(config: Config):
    stage = MinimizationStage(config, drop=["profiles"])

    assert stage is not None


@pytest.mark.gpu_and_cpu_mode
def test_the_class_is_read_off_the_frame_when_it_is_not_given(config: Config):
    # A segment carries one class, and the frame already says which. Making the caller repeat it would be one
    # more thing to get out of step with the pipeline it describes.
    result = run(config, authentications(), pseudonymize=["addresses"], key=KEY)

    assert "203.0.113.5" not in list(result["source_ip"])
    assert list(result["user_principal"]) == ["alice@example.com", "bob@example.com", "alice@example.com"]


@pytest.mark.gpu_and_cpu_mode
def test_a_frame_mixing_classes_is_refused(config: Config):
    payload = authentications()
    payload["telemetry_class"] = ["tc5_auth", "tc1", "tc5_auth"]

    with pytest.raises(ValueError, match="carries telemetry classes"):
        run(config, payload, drop=["user_principal"])


@pytest.mark.gpu_and_cpu_mode
def test_an_empty_batch_passes_through(config: Config):
    meta = MessageMeta(get_df_class(config.execution_mode)({name: [] for name in authentications()}))
    MinimizationStage(config, drop=["source_ip"]).on_data(meta)

    assert meta.count == 0
