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
from morpheus.stages.lineage.determinism_stamp_stage import DeterminismStampStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.determinism_envelope import DeterminismEnvelope
from morpheus.utils.model_manifest import ModelManifest
from morpheus.utils.type_utils import get_df_class

ALICE = "alice@example.com"
BOB = "bob@example.com"
WINDOW = 484512

ENVELOPE = DeterminismEnvelope(tier="D1",
                               fingerprint="a3f9c2e1b8d47506",
                               configuration="7d2e4a1f9c3b5e80",
                               code_commit="c6a3b56",
                               image_digest="sha256:1f0c",
                               feature_schema_version="TC-5/2.1.0",
                               rng_seed=42)


def manifest(**kwargs) -> ModelManifest:
    defaults = {"window_id": WINDOW, "models": {ALICE: "dfp-alice:14", BOB: "dfp-bob:3"}}
    defaults.update(kwargs)

    return ModelManifest(**defaults)


def frame(principals: list = None, windows: list = None) -> dict:
    principals = [ALICE, BOB] if principals is None else principals
    count = len(principals)

    return {
        "user_principal": principals,
        "window_id": [WINDOW] * count if windows is None else windows,
        "event_time": list(range(count)),
    }


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [None if pd.isna(value) else value for value in series.tolist()]


def run(config: Config, payload: dict, **kwargs) -> MessageMeta:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    DeterminismStampStage(config, envelope=ENVELOPE, **kwargs).on_data(meta)

    return meta


@pytest.mark.gpu_and_cpu_mode
def test_every_row_carries_the_envelope(config: Config):
    meta = run(config, frame())

    assert _as_list(meta, "determinism_tier") == ["D1", "D1"]
    assert _as_list(meta, "pipeline_fingerprint") == ["a3f9c2e1b8d47506"] * 2
    assert _as_list(meta, "config_hash") == ["7d2e4a1f9c3b5e80"] * 2
    assert _as_list(meta, "rng_seed") == [42, 42]


@pytest.mark.gpu_and_cpu_mode
def test_the_envelope_is_stamped_even_where_no_model_scored_the_row(config: Config):
    # A telemetry event no model touched still came out of a particular configuration, commit and image, and a
    # consumer comparing it with an event from another run needs to know whether the two are comparable.
    meta = run(config, frame())

    assert _as_list(meta, "code_commit") == ["c6a3b56"] * 2
    assert _as_list(meta, "model_version") == [None, None]
    assert _as_list(meta, "model_fallback_used") == [None, None]


@pytest.mark.gpu_and_cpu_mode
def test_each_entity_carries_the_model_it_was_scored_against(config: Config):
    meta = run(config, frame(), manifest=manifest())

    assert _as_list(meta, "model_version") == ["dfp-alice:14", "dfp-bob:3"]
    assert _as_list(meta, "model_fallback_used") == [False, False]


@pytest.mark.gpu_and_cpu_mode
def test_a_fallback_is_visible_on_the_row(config: Config):
    # An event scored against a population model is a different claim from one scored against the entity's own,
    # and the difference has to reach the SIEM rather than being inferable only from the model name.
    meta = run(config, frame(principals=[ALICE, "carol@example.com"]), manifest=manifest(fallback="dfp-generic:2"))

    assert _as_list(meta, "model_version") == ["dfp-alice:14", "dfp-generic:2"]
    assert _as_list(meta, "model_fallback_used") == [False, True]


@pytest.mark.gpu_and_cpu_mode
def test_an_entity_with_no_model_and_no_fallback_carries_no_model(config: Config):
    # A fact about the estate rather than about the pipeline, so the row is annotated rather than the run
    # stopped -- and a rule reading a score on such a row is reading one nothing produced.
    meta = run(config, frame(principals=[ALICE, "carol@example.com"]), manifest=manifest())

    assert _as_list(meta, "model_version") == ["dfp-alice:14", None]
    assert _as_list(meta, "model_fallback_used") == [False, None]


@pytest.mark.gpu_and_cpu_mode
def test_a_null_principal_carries_no_model(config: Config):
    meta = run(config, frame(principals=[None, ALICE]), manifest=manifest())

    assert _as_list(meta, "model_version") == [None, "dfp-alice:14"]


@pytest.mark.cpu_mode
def test_a_window_the_manifest_was_not_pinned_for_stops_the_run(config: Config):
    # Deliberately not annotated. A run that scored a window against models pinned for another is a defect worth
    # stopping on, and it is the distinction the stage's own except clause has to preserve.
    with pytest.raises(ValueError, match="was resolved for window"):
        run(config, frame(windows=[WINDOW, WINDOW + 1]), manifest=manifest())


@pytest.mark.gpu_and_cpu_mode
def test_a_frame_with_no_window_column_is_taken_to_be_the_manifests_own(config: Config):
    # Correct for a pipeline that seals before scoring, which is why WindowSealStage belongs upstream.
    payload = frame()
    del payload["window_id"]

    meta = run(config, payload, manifest=manifest())

    assert _as_list(meta, "model_version") == ["dfp-alice:14", "dfp-bob:3"]


@pytest.mark.gpu_and_cpu_mode
def test_a_differently_keyed_entity_column_is_honoured(config: Config):
    payload = {"service_account": [ALICE], "window_id": [WINDOW], "event_time": [0]}

    meta = run(config, payload, manifest=manifest(), entity_column="service_account")

    assert _as_list(meta, "model_version") == ["dfp-alice:14"]


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    message = ControlMessage()
    message.payload(MessageMeta(get_df_class(config.execution_mode)(frame())))

    DeterminismStampStage(config, envelope=ENVELOPE, manifest=manifest()).on_data(message)

    assert _as_list(message.payload(), "determinism_tier") == ["D1", "D1"]


@pytest.mark.gpu_and_cpu_mode
def test_determinism_stamp_stage_pipe(config: Config):
    source_df = get_df_class(config.execution_mode)(frame())

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(DeterminismStampStage(config, envelope=ENVELOPE, manifest=manifest()))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "model_version") == ["dfp-alice:14", "dfp-bob:3"]


def test_the_stage_refuses_an_uninterpretable_tier():
    # Caught when the envelope is built, which is before the pipeline is assembled.
    with pytest.raises(ValueError, match="is not one of"):
        DeterminismEnvelope(tier="D4",
                            fingerprint="a",
                            configuration="b",
                            code_commit="c",
                            image_digest="d",
                            feature_schema_version="e",
                            rng_seed=1)
