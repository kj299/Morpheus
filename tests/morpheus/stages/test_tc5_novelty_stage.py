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
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc5_novelty_stage import TC5NoveltyStage
from morpheus.utils.distinct_window import NS_PER_SECOND
from morpheus.utils.type_utils import get_df_class

MINUTE_NS = 60 * NS_PER_SECOND
ALICE = "alice@example.com"


def frame(count: int,
          principals: list = None,
          countries: list = None,
          cities: list = None,
          apps: list = None,
          devices: list = None,
          asns: list = None,
          times: list = None) -> dict:

    def column(supplied, default):
        return [default] * count if supplied is None else supplied

    return {
        "user_principal": column(principals, ALICE),
        "source_country": column(countries, "gb"),
        "source_region": ["england"] * count,
        "source_city": column(cities, "london"),
        "app": column(apps, "vpn"),
        "device_id": column(devices, "laptop-1"),
        "source_asn": column(asns, "as5089"),
        "event_time": column(times, None) if times is not None else [index * MINUTE_NS for index in range(count)],
    }


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    # `pd.isna` rather than the `value != value` NaN trick, which pylint reads as a redundant self-comparison. It
    # covers None, pandas.NA and NaN together, and which of the three arrives depends on the column's dtype.
    return [None if pd.isna(value) else value for value in series.tolist()]


def run(config: Config, payload: dict, **kwargs) -> MessageMeta:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC5NoveltyStage(config, **kwargs).on_data(meta)

    return meta


@pytest.mark.gpu_and_cpu_mode
def test_logcount_counts_every_record_in_the_window(config: Config):
    meta = run(config, frame(4))

    assert _as_list(meta, "logcount") == [1, 2, 3, 4]


@pytest.mark.gpu_and_cpu_mode
def test_logcount_decays_out_of_the_window(config: Config):
    # Volume is a question about now. A cumulative answer would rise monotonically for every account in the estate.
    times = [0, MINUTE_NS, 2 * 3600 * NS_PER_SECOND]
    meta = run(config, frame(3, times=times), window_seconds=3600)

    assert _as_list(meta, "logcount") == [1, 2, 1]


@pytest.mark.gpu_and_cpu_mode
def test_the_increments_are_cumulative_and_do_not_decay(config: Config):
    # The guide is explicit that this is intended: R-B-L5-002 is scored against a per-user loss scaler, and a
    # decaying count would re-fire on the same relocation every window.
    times = [0, 10 * 24 * 3600 * NS_PER_SECOND]
    meta = run(config, frame(2, countries=["gb", "fr"], cities=["london", "paris"], times=times), window_seconds=3600)

    assert _as_list(meta, "locincrement") == [1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_a_returning_location_does_not_raise_the_increment(config: Config):
    meta = run(config, frame(3, countries=["gb", "fr", "gb"], cities=["london", "paris", "london"]))

    assert _as_list(meta, "locincrement") == [1, 2, 2]
    assert _as_list(meta, "location_first_seen") == [None, True, False]


@pytest.mark.gpu_and_cpu_mode
def test_the_first_sample_answers_null_rather_than_true(config: Config):
    # The first sample establishes what normal looks like and is not itself an event. The increment beside it
    # already reads one, which carries the same fact without answering a question the history cannot yet answer.
    meta = run(config, frame(1))

    assert _as_list(meta, "location_first_seen") == [None]
    assert _as_list(meta, "locincrement") == [1]


@pytest.mark.gpu_and_cpu_mode
def test_applications_and_devices_are_counted_the_same_way(config: Config):
    meta = run(config, frame(3, apps=["vpn", "wiki", "vpn"], devices=["laptop-1", "laptop-1", "phone-1"]))

    assert _as_list(meta, "appincrement") == [1, 2, 2]
    assert _as_list(meta, "deviceincrement") == [1, 1, 2]


@pytest.mark.gpu_and_cpu_mode
@pytest.mark.parametrize("absent", [None, pd.NA, "", "   "], ids=["none", "pandas_na", "blank", "whitespace"])
def test_a_dropped_field_is_an_omission_rather_than_a_new_application(config: Config, absent):
    # The defect this separation exists to prevent. Were the omission observed as a value, `appincrement` would
    # rise the first time a collector dropped the field and stay raised, so a gap in collection would read as an
    # application the principal had never used before -- and R-B-L5-002 reads exactly that column.
    meta = run(config, frame(3, apps=["vpn", absent, "vpn"]))

    assert _as_list(meta, "appincrement") == [1, None, 1]
    assert _as_list(meta, "app_first_seen") == [None, None, False]


@pytest.mark.gpu_and_cpu_mode
def test_a_dropped_field_does_not_stop_the_others_counting(config: Config):
    # Each cumulative field is tracked on its own, so a row missing one still counts the rest.
    meta = run(config, frame(2, apps=["vpn", None], devices=["laptop-1", "phone-1"]))

    assert _as_list(meta, "appincrement") == [1, None]
    assert _as_list(meta, "deviceincrement") == [1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_the_location_is_the_concatenation_of_its_parts(config: Config):
    meta = run(config, frame(1))

    assert _as_list(meta, "user_location") == ["gb:england:london"]


@pytest.mark.gpu_and_cpu_mode
def test_a_null_part_yields_no_location_rather_than_a_second_one(config: Config):
    # Dropping the null part and composing what is left would turn one place into two whenever a city lookup
    # failed, and each would read as somewhere the principal had never authenticated from.
    meta = run(config, frame(3, cities=["london", None, "london"]))

    assert _as_list(meta, "user_location") == ["gb:england:london", None, "gb:england:london"]
    assert _as_list(meta, "locincrement") == [1, None, 1]


@pytest.mark.gpu_and_cpu_mode
def test_a_source_reporting_only_a_country_still_counts_locations(config: Config):
    payload = frame(2, countries=["gb", "fr"])
    del payload["source_region"]
    del payload["source_city"]

    meta = run(config, payload)

    assert _as_list(meta, "user_location") == ["gb", "fr"]
    assert _as_list(meta, "locincrement") == [1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_distinct_asns_are_windowed_rather_than_cumulative(config: Config):
    times = [0, MINUTE_NS, 2 * 3600 * NS_PER_SECOND]
    meta = run(config, frame(3, asns=["as1", "as2", "as3"], times=times), window_seconds=3600)

    assert _as_list(meta, "asns_in_window") == [1, 2, 1]


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_asn_carries_no_count(config: Config):
    meta = run(config, frame(2, asns=["as1", None]))

    assert _as_list(meta, "asns_in_window") == [1, None]


@pytest.mark.gpu_and_cpu_mode
def test_principals_do_not_see_each_others_history(config: Config):
    payload = frame(3, principals=[ALICE, "bob@example.com", ALICE], countries=["gb", "fr", "gb"])

    meta = run(config, payload)

    assert _as_list(meta, "logcount") == [1, 1, 2]
    assert _as_list(meta, "locincrement") == [1, 1, 1]


@pytest.mark.gpu_and_cpu_mode
def test_a_record_with_no_principal_carries_no_counts(config: Config):
    # Pooling these would make every unattributed authentication in the estate look like one very busy account.
    meta = run(config, frame(2, principals=[None, ALICE]))

    assert _as_list(meta, "logcount") == [None, 1]
    assert _as_list(meta, "locincrement") == [None, 1]


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    stage = TC5NoveltyStage(config)

    first = MessageMeta(get_df_class(config.execution_mode)(frame(1, countries=["gb"])))
    stage.on_data(first)

    second = MessageMeta(get_df_class(config.execution_mode)(frame(1, countries=["fr"], times=[MINUTE_NS])))
    stage.on_data(second)

    assert _as_list(second, "logcount") == [2]
    assert _as_list(second, "locincrement") == [2]
    assert stage.tracked_principals == 1


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    message = ControlMessage()
    message.payload(MessageMeta(get_df_class(config.execution_mode)(frame(2))))

    TC5NoveltyStage(config).on_data(message)

    assert _as_list(message.payload(), "logcount") == [1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_tc5_novelty_stage_pipe(config: Config):
    payload = frame(3, countries=["gb", "gb", "fr"], cities=["london", "london", "paris"])
    source_df = get_df_class(config.execution_mode)(payload)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC5NoveltyStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "locincrement") == [1, 1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = frame(1)
    del payload["user_principal"]

    with pytest.raises(KeyError, match="user_principal"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError, match="window_seconds must be positive"):
        TC5NoveltyStage(config, window_seconds=0)

    with pytest.raises(ValueError, match="At least one location column"):
        TC5NoveltyStage(config, location_columns=())


@pytest.mark.cpu_mode
def test_an_out_of_order_record_leaves_the_increments_untouched(config: Config):
    stage = TC5NoveltyStage(config)

    first = MessageMeta(get_df_class(config.execution_mode)(frame(1, countries=["gb"], times=[10 * MINUTE_NS])))
    stage.on_data(first)

    late = MessageMeta(get_df_class(config.execution_mode)(frame(1, countries=["fr"], times=[0])))
    stage.on_data(late)

    assert _as_list(late, "locincrement") == [None]

    # And the history it declined to join is unchanged: France is still new.
    after = MessageMeta(get_df_class(config.execution_mode)(frame(1, countries=["fr"], times=[20 * MINUTE_NS])))
    stage.on_data(after)

    assert _as_list(after, "locincrement") == [2]
