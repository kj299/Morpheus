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
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.type_utils import get_df_class


def ports(sites=("hq", "hq"), devices=("sw1", "sw1"), port_ids=("Gi1/0/1", "Gi1/0/2")) -> dict:
    return {"site_id": list(sites), "device_id": list(devices), "port_id": list(port_ids)}


def run(config: Config, payload: dict, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("osi_layer", 1)
    kwargs.setdefault("entity_columns", ["site_id", "device_id", "port_id"])
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    EnvelopeStampStage(config, **kwargs).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


@pytest.mark.gpu_and_cpu_mode
def test_every_row_carries_the_layer_and_the_key(config: Config):
    # The gap this closes. Two shipped searches group by these, and `stats by` drops a row whose grouping field
    # is absent -- so both returned nothing regardless of what else was on the record.
    result = run(config, ports())

    assert list(result["osi_layer"]) == [1, 1]
    assert list(result["entity_key"]) == ["hq:sw1:Gi1/0/1", "hq:sw1:Gi1/0/2"]


@pytest.mark.gpu_and_cpu_mode
def test_a_single_column_subject_is_the_column_itself(config: Config):
    # Layer 5's subject is one column, not a composite. The key is the principal, unadorned.
    result = run(config, {"user_principal": ["alice@example.com", "bob@example.com"]},
                 osi_layer=5,
                 entity_columns=["user_principal"])

    assert list(result["entity_key"]) == ["alice@example.com", "bob@example.com"]
    assert list(result["osi_layer"]) == [5, 5]


@pytest.mark.gpu_and_cpu_mode
def test_a_row_missing_part_of_its_key_carries_none(config: Config):
    # `compose_key` refuses a partial identity, and this inherits that. A key of "None:sw1:Gi1/0/1" would pool
    # every siteless row in the estate under one invented site, which is worse than having no key.
    result = run(config, ports(sites=("hq", None)))

    assert result["entity_key"].iloc[0] == "hq:sw1:Gi1/0/1"
    assert pd.isna(result["entity_key"].iloc[1])
    # The layer is still stamped: it is a property of the stream, not of the row's completeness.
    assert list(result["osi_layer"]) == [1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_an_existing_key_is_left_alone(config: Config):
    # The TC-1 stages compose `entity_key` themselves and key their own state on it. A stage that overwrote it
    # could disagree with state those stages keep, which would be a far worse defect than a missing column.
    payload = ports()
    payload["entity_key"] = ["already:composed:1", "already:composed:2"]

    result = run(config, payload)

    assert list(result["entity_key"]) == ["already:composed:1", "already:composed:2"]
    assert list(result["osi_layer"]) == [1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_a_declared_but_empty_key_column_is_still_composed(config: Config):
    # The case that made this stage do nothing in the composed pipelines while passing every test above. A stage
    # declares the columns it writes through `_needed_columns`, and Morpheus creates them before the stage runs,
    # so `entity_key` is always present by then. Checking the column's existence declared every row already
    # keyed; only layer 1 looked right, because its own stages fill it. The check has to be on the value.
    payload = ports()
    payload["entity_key"] = ["", ""]

    result = run(config, payload)

    assert list(result["entity_key"]) == ["hq:sw1:Gi1/0/1", "hq:sw1:Gi1/0/2"]


@pytest.mark.gpu_and_cpu_mode
def test_a_key_is_kept_or_composed_per_row(config: Config):
    # Half a column of real keys and half empty is what a partly-keyed stream looks like. Each row is decided on
    # its own value rather than the frame being judged as a whole.
    payload = ports()
    payload["entity_key"] = ["already:composed", None]

    result = run(config, payload)

    assert list(result["entity_key"]) == ["already:composed", "hq:sw1:Gi1/0/2"]


@pytest.mark.gpu_and_cpu_mode
def test_an_existing_key_can_be_replaced_when_asked(config: Config):
    payload = ports()
    payload["entity_key"] = ["stale:1", "stale:2"]

    result = run(config, payload, overwrite=True)

    assert list(result["entity_key"]) == ["hq:sw1:Gi1/0/1", "hq:sw1:Gi1/0/2"]


@pytest.mark.gpu_and_cpu_mode
def test_the_key_matches_what_the_layer_1_stages_compose(config: Config):
    # The ladder joins on this string. Composing it differently here than `entity_key.compose_key` does would
    # turn every hop into a reconstruction.
    from morpheus.utils.entity_key import compose_key  # pylint: disable=import-outside-toplevel

    result = run(config, ports())

    assert result["entity_key"].iloc[0] == compose_key(["hq", "sw1", "Gi1/0/1"])


@pytest.mark.gpu_and_cpu_mode
def test_the_entity_columns_are_composed_in_the_order_given(config: Config):
    result = run(config, ports(), entity_columns=["port_id", "device_id", "site_id"])

    assert result["entity_key"].iloc[0] == "Gi1/0/1:sw1:hq"


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    stage = EnvelopeStampStage(config, osi_layer=2, entity_columns=["site_id"])
    message = ControlMessage()
    message.payload(MessageMeta(get_df_class(config.execution_mode)(ports())))

    assert stage.on_data(message) is message


@pytest.mark.gpu_and_cpu_mode
def test_envelope_stamp_stage_pipe(config: Config):
    payload = get_df_class(config.execution_mode)(ports())
    pipeline = LinearPipeline(config)
    pipeline.set_source(InMemorySourceStage(config, [payload]))
    pipeline.add_stage(EnvelopeStampStage(config, osi_layer=1, entity_columns=["site_id", "device_id", "port_id"]))
    sink = pipeline.add_stage(InMemorySinkStage(config))
    pipeline.run()

    assert len(sink.get_messages()) == 1


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = ports()
    del payload["port_id"]

    with pytest.raises(KeyError, match="EnvelopeStampStage requires columns.*port_id"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError, match="osi_layer must be between"):
        EnvelopeStampStage(config, osi_layer=8, entity_columns=["site_id"])

    with pytest.raises(ValueError, match="osi_layer must be between"):
        EnvelopeStampStage(config, osi_layer=-1, entity_columns=["site_id"])

    with pytest.raises(ValueError, match="osi_layer must be an integer"):
        EnvelopeStampStage(config, osi_layer=True, entity_columns=["site_id"])

    with pytest.raises(ValueError, match="entity_columns must name at least one column"):
        EnvelopeStampStage(config, osi_layer=1, entity_columns=[])


def test_layer_zero_is_allowed(config: Config):
    # TC-0 is the identity and asset context. Not an OSI layer, but it is a telemetry class and its records need
    # the same envelope as everything else.
    assert EnvelopeStampStage(config, osi_layer=0, entity_columns=["site_id"]) is not None
