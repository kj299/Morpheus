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
from morpheus.stages.telemetry.tc5_travel_stage import TC5TravelStage
from morpheus.utils.geo_velocity import NS_PER_SECOND
from morpheus.utils.type_utils import get_df_class

HOUR_NS = 3600 * NS_PER_SECOND
ALICE = "alice@example.com"

LONDON = (51.5074, -0.1278)
NEW_YORK = (40.7128, -74.0060)


def frame(places: list,
          times: list = None,
          principals: list = None,
          results: list = None,
          tokens: list = None,
          addresses: list = None) -> dict:
    count = len(places)

    def column(supplied, default):
        return [default] * count if supplied is None else supplied

    return {
        "user_principal": column(principals, ALICE),
        "source_latitude": [None if place is None else place[0] for place in places],
        "source_longitude": [None if place is None else place[1] for place in places],
        "auth_result": column(results, "success"),
        "token_type": column(tokens, "bearer"),
        "source_ip": column(addresses, "203.0.113.10"),
        "event_time": [index * HOUR_NS for index in range(count)] if times is None else times,
    }


def _as_list(meta: MessageMeta, column: str) -> list:
    series = meta.get_data(column)

    if (hasattr(series, "to_pandas")):
        series = series.to_pandas()

    return [None if pd.isna(value) else value for value in series.tolist()]


def run(config: Config, payload: dict, **kwargs) -> MessageMeta:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    TC5TravelStage(config, **kwargs).on_data(meta)

    return meta


@pytest.mark.gpu_and_cpu_mode
def test_an_ordinary_flight_is_measured_and_is_an_ordinary_speed(config: Config):
    meta = run(config, frame([LONDON, NEW_YORK], times=[0, 7 * HOUR_NS]))

    assert _as_list(meta, "travel_status") == ["first_for_principal", "measured"]
    assert _as_list(meta, "travel_kmh")[1] == pytest.approx(796, abs=5)
    assert _as_list(meta, "travel_distance_km")[1] == pytest.approx(5570, abs=15)
    assert _as_list(meta, "travel_elapsed_ns")[1] == 7 * HOUR_NS


@pytest.mark.gpu_and_cpu_mode
def test_the_same_journey_in_one_hour_is_the_rules_firing_condition(config: Config):
    meta = run(config, frame([LONDON, NEW_YORK], times=[0, HOUR_NS]))

    assert _as_list(meta, "travel_kmh")[1] > 900


@pytest.mark.gpu_and_cpu_mode
def test_two_successes_at_one_instant_read_as_enormous_rather_than_null(config: Config):
    # A rule written as `travel_kmh >= 900` would otherwise miss the most impossible journey there is.
    meta = run(config, frame([LONDON, NEW_YORK], times=[0, 0]))

    assert _as_list(meta, "travel_status")[1] == "measured"
    assert _as_list(meta, "travel_kmh")[1] > 900
    assert _as_list(meta, "travel_elapsed_ns")[1] == 0
    assert _as_list(meta, "travel_elapsed_floored") == [False, True]


@pytest.mark.gpu_and_cpu_mode
def test_a_failed_authentication_is_not_measured(config: Config):
    # Evidence that somebody tried, not that anybody was there.
    meta = run(config, frame([LONDON, NEW_YORK], results=["success", "failure"]))

    assert _as_list(meta, "travel_status") == ["first_for_principal", "not_successful"]
    assert _as_list(meta, "travel_kmh") == [None, None]


@pytest.mark.gpu_and_cpu_mode
def test_a_failed_authentication_does_not_become_the_anchor(config: Config):
    # The half that matters. Were the failure recorded, the journey from London would be erased and the third
    # record would measure a distance of nothing.
    payload = frame([LONDON, NEW_YORK, NEW_YORK],
                    results=["success", "failure", "success"],
                    times=[0, HOUR_NS, 2 * HOUR_NS])

    meta = run(config, payload)

    assert _as_list(meta, "travel_status") == ["first_for_principal", "not_successful", "measured"]
    assert _as_list(meta, "travel_distance_km")[2] == pytest.approx(5570, abs=15)


@pytest.mark.gpu_and_cpu_mode
def test_a_token_refresh_is_excluded_and_does_not_move_the_anchor(config: Config):
    # A refresh carries the location the original authentication had, so it is evidence of nothing about now. One
    # that updated the anchor would erase the journey the next real authentication is supposed to reveal.
    payload = frame([LONDON, LONDON, NEW_YORK], tokens=["bearer", "refresh", "bearer"], times=[0, HOUR_NS, 2 * HOUR_NS])

    meta = run(config, payload)

    assert _as_list(meta, "travel_status") == ["first_for_principal", "token_refresh", "measured"]
    assert _as_list(meta, "travel_elapsed_ns")[2] == 2 * HOUR_NS


@pytest.mark.gpu_and_cpu_mode
def test_a_vpn_egress_address_is_excluded(config: Config):
    payload = frame([LONDON, NEW_YORK], addresses=["203.0.113.10", "198.51.100.7"])

    meta = run(config, payload, excluded_source_networks=("198.51.100.0/24", ))

    assert _as_list(meta, "travel_status") == ["first_for_principal", "vpn_egress"]


@pytest.mark.gpu_and_cpu_mode
def test_a_bare_address_in_the_exclusion_list_is_one_host(config: Config):
    payload = frame([LONDON, NEW_YORK, NEW_YORK], addresses=["203.0.113.10", "198.51.100.7", "198.51.100.8"])

    meta = run(config, payload, excluded_source_networks=("198.51.100.7", ))

    assert _as_list(meta, "travel_status") == ["first_for_principal", "vpn_egress", "measured"]


@pytest.mark.gpu_and_cpu_mode
def test_the_exclusion_list_ships_empty_so_nothing_is_excluded_by_default(config: Config):
    # The same shape as R-D-L2-003's exclusion list: this repository cannot know an estate's egress ranges, so the
    # rule fires on VPN users until the estate supplies them. Stated rather than hidden.
    meta = run(config, frame([LONDON, NEW_YORK], addresses=["198.51.100.7"] * 2))

    assert _as_list(meta, "travel_status") == ["first_for_principal", "measured"]


@pytest.mark.gpu_and_cpu_mode
def test_an_absent_coordinate_is_not_measured(config: Config):
    meta = run(config, frame([LONDON, None]))

    assert _as_list(meta, "travel_status") == ["first_for_principal", "no_coordinate"]


@pytest.mark.gpu_and_cpu_mode
def test_a_coordinate_outside_the_globe_is_not_measured(config: Config):
    # A nonsense distance would read as impossible travel, which is an alert about a geolocation database.
    meta = run(config, frame([LONDON, (91.0, 0.0)]))

    assert _as_list(meta, "travel_status") == ["first_for_principal", "no_coordinate"]
    assert _as_list(meta, "travel_kmh") == [None, None]


@pytest.mark.gpu_and_cpu_mode
def test_a_record_with_no_principal_is_not_measured(config: Config):
    meta = run(config, frame([LONDON, NEW_YORK], principals=[None, ALICE]))

    assert _as_list(meta, "travel_status") == ["no_principal", "first_for_principal"]


@pytest.mark.gpu_and_cpu_mode
def test_principals_do_not_share_a_location(config: Config):
    payload = frame([LONDON, NEW_YORK], principals=[ALICE, "bob@example.com"], times=[0, HOUR_NS])

    meta = run(config, payload)

    assert _as_list(meta, "travel_status") == ["first_for_principal", "first_for_principal"]


@pytest.mark.gpu_and_cpu_mode
def test_a_source_that_reports_no_result_is_taken_at_face_value(config: Config):
    # Over-measuring rather than measuring nothing: a rule that silently stops firing because a column is absent
    # is indistinguishable from a rule with nothing to fire on.
    payload = frame([LONDON, NEW_YORK], times=[0, HOUR_NS])
    del payload["auth_result"]

    meta = run(config, payload)

    assert _as_list(meta, "travel_status") == ["first_for_principal", "measured"]


@pytest.mark.gpu_and_cpu_mode
def test_the_result_vocabulary_is_case_insensitive(config: Config):
    meta = run(config, frame([LONDON, NEW_YORK], results=["SUCCESS", "Success"], times=[0, HOUR_NS]))

    assert _as_list(meta, "travel_status") == ["first_for_principal", "measured"]


@pytest.mark.gpu_and_cpu_mode
def test_a_collectors_own_result_vocabulary_can_be_supplied(config: Config):
    meta = run(config,
               frame([LONDON, NEW_YORK], results=["ACCEPT", "ACCEPT"], times=[0, HOUR_NS]),
               success_values=("accept", ))

    assert _as_list(meta, "travel_status") == ["first_for_principal", "measured"]


@pytest.mark.gpu_and_cpu_mode
def test_state_persists_across_messages(config: Config):
    stage = TC5TravelStage(config)

    first = MessageMeta(get_df_class(config.execution_mode)(frame([LONDON], times=[0])))
    stage.on_data(first)

    assert stage.tracked_principals == 1

    second = MessageMeta(get_df_class(config.execution_mode)(frame([NEW_YORK], times=[HOUR_NS])))
    stage.on_data(second)

    assert _as_list(second, "travel_kmh")[0] > 900


@pytest.mark.gpu_and_cpu_mode
def test_control_message_accepted(config: Config):
    message = ControlMessage()
    message.payload(MessageMeta(get_df_class(config.execution_mode)(frame([LONDON, NEW_YORK], times=[0, HOUR_NS]))))

    TC5TravelStage(config).on_data(message)

    assert _as_list(message.payload(), "travel_status") == ["first_for_principal", "measured"]


@pytest.mark.gpu_and_cpu_mode
def test_tc5_travel_stage_pipe(config: Config):
    source_df = get_df_class(config.execution_mode)(frame([LONDON, NEW_YORK], times=[0, HOUR_NS]))

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))
    pipe.add_stage(TC5TravelStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    assert len(messages) == 1
    assert _as_list(messages[0], "travel_kmh")[1] > 900


@pytest.mark.gpu_and_cpu_mode
def test_missing_column_raises(config: Config):
    payload = frame([LONDON])
    del payload["user_principal"]

    with pytest.raises(KeyError, match="user_principal"):
        run(config, payload)


def test_constructor_validation(config: Config):
    with pytest.raises(ValueError, match="min_elapsed_seconds must be positive"):
        TC5TravelStage(config, min_elapsed_seconds=0)


def test_a_malformed_exclusion_range_is_refused_when_the_pipeline_is_built(config: Config):
    # Finding it here beats finding it as a per-row exception that quietly excludes nothing.
    with pytest.raises(ValueError, match="is not an address or CIDR range"):
        TC5TravelStage(config, excluded_source_networks=("not-a-network", ))


@pytest.mark.cpu_mode
def test_an_unparseable_source_address_is_measured_rather_than_excluded(config: Config):
    # Excluding it would let a collector emitting a hostname where an address belongs silence the rule for the
    # whole estate.
    payload = frame([LONDON, NEW_YORK], addresses=["203.0.113.10", "vpn.example.com"], times=[0, HOUR_NS])

    meta = run(config, payload, excluded_source_networks=("198.51.100.0/24", ))

    assert _as_list(meta, "travel_status") == ["first_for_principal", "measured"]


@pytest.mark.cpu_mode
def test_an_out_of_order_record_measures_nothing_and_moves_nothing(config: Config):
    stage = TC5TravelStage(config)

    first = MessageMeta(get_df_class(config.execution_mode)(frame([LONDON], times=[10 * HOUR_NS])))
    stage.on_data(first)

    late = MessageMeta(get_df_class(config.execution_mode)(frame([NEW_YORK], times=[0])))
    stage.on_data(late)

    assert _as_list(late, "travel_status") == ["out_of_order"]

    # The anchor is still London.
    after = MessageMeta(get_df_class(config.execution_mode)(frame([LONDON], times=[11 * HOUR_NS])))
    stage.on_data(after)

    assert _as_list(after, "travel_distance_km")[0] == pytest.approx(0.0, abs=1e-3)
