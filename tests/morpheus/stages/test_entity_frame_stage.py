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

from morpheus.messages import MessageMeta
from morpheus.stages.telemetry.entity_frame_stage import EntityFrameStage
from morpheus.stages.telemetry.entity_frame_stage import reduce_values
from morpheus.utils.tracker_state import from_bytes
from morpheus.utils.tracker_state import to_bytes

HOUR_NS = 3600 * 10**9

FEATURES = {
    "fan_out": ["dsts_per_src", "max"],
    "ratio": ["internal_dst_ratio", "last"],
    "novel": ["endpoint_pair_novel", "count_true"],
}


def _window(window_id: int = 7, **overrides) -> pd.DataFrame:
    """One sealed hour: two rows of a flow class and two of an endpoint class for ws-1, one flow for ws-2.

    The first key is padded, as a value read from a feed can be; it is the same entity once stripped.
    """
    rows = {
        "host_key": [" ws-1 ", "ws-1", "ws-2", "ws-1", "ws-1"],
        "dsts_per_src": [3, 5, 9, None, None],
        "internal_dst_ratio": [1.0, 0.8, 0.5, None, None],
        "endpoint_pair_novel": [None, None, None, True, False],
        "ctx_peer_group": [None, "finance", "build", "finance", None],
        "lineage_id": ["chain-1", "chain-1", "chain-2", "chain-1", "chain-1"],
        "is_late": [False] * 5,
        "window_complete": [True] * 5,
    }
    rows.update(overrides)
    frame = pd.DataFrame(rows)
    frame["window_id"] = window_id
    frame["window_start_ns"] = window_id * HOUR_NS
    frame["window_end_ns"] = (window_id + 1) * HOUR_NS
    frame["revision"] = 0
    frame["sealed_by"] = "watermark"

    return frame


def _stage(config, **kwargs) -> EntityFrameStage:
    return EntityFrameStage(config, features=kwargs.pop("features", FEATURES), **kwargs)


@pytest.mark.cpu_mode
def test_a_window_becomes_one_row_per_entity(config):
    out = _stage(config, carry_columns=["ctx_peer_group"]).reduce(_window())

    assert list(out["host_key"]) == ["ws-1", "ws-2"]
    first = out.iloc[0]

    assert first["fan_out"] == 5.0
    assert first["ratio"] == 0.8
    assert first["novel"] == 1
    assert first["frame_rows"] == 4
    # Carried from the first row that has one, not from the first row.
    assert first["ctx_peer_group"] == "finance"
    assert first["lineage_id"] == "chain-1"
    assert first["event_time"] == 7 * HOUR_NS
    assert (first["window_id"], first["window_end_ns"], first["sealed_by"]) == (7, 8 * HOUR_NS, "watermark")
    assert bool(first["window_complete"]) and not bool(first["is_late"])


@pytest.mark.cpu_mode
def test_a_feature_no_row_carries_is_null_not_zero(config):
    out = _stage(config).reduce(_window())
    second = out.iloc[1]

    # ws-2 had a flow and no EDR record: its novel count is unknown, which is not none.
    assert second["fan_out"] == 9.0
    assert pd.isna(second["novel"])


@pytest.mark.cpu_mode
def test_late_and_incomplete_rows_are_left_out(config):
    out = _stage(config).reduce(
        _window(is_late=[False, True, False, False, False], window_complete=[True, True, False, True, True]))

    assert list(out["host_key"]) == ["ws-1"]
    assert out.iloc[0]["fan_out"] == 3.0
    assert out.iloc[0]["frame_rows"] == 3


@pytest.mark.cpu_mode
def test_the_envelope_identifies_each_row_and_the_sequence_survives_a_restart(config):
    stage = _stage(config, collector_id="host-frame", schema_version="host_frame.v1", first_seq=4)
    out = stage.reduce(_window())

    assert list(out["collector_seq"]) == [4, 5]
    assert list(out["origin_hash"]) == ["ws-1@7:0", "ws-2@7:0"]
    assert set(out["collector_id"]) == {"host-frame"}
    assert out["event_uid"].is_unique

    resumed = _stage(config, collector_id="host-frame", schema_version="host_frame.v1")
    resumed.restore_state(from_bytes(to_bytes(stage.checkpoint_state())))
    again = resumed.reduce(_window(8))

    assert list(again["collector_seq"]) == [6, 7]
    # The same entity in another window is another row, with another identity.
    assert set(again["event_uid"]).isdisjoint(out["event_uid"])


@pytest.mark.cpu_mode
def test_a_missing_column_and_a_bad_feature_are_refused(config):
    with pytest.raises(KeyError, match="internal_dst_ratio"):
        _stage(config).reduce(_window().drop(columns=["internal_dst_ratio"]))

    with pytest.raises(KeyError, match="window_id"):
        _stage(config).reduce(_window().drop(columns=["window_id"]))

    with pytest.raises(ValueError, match="reduction"):
        _stage(config, features={"x": ["dsts_per_src", "median"]})

    with pytest.raises(ValueError, match="at least one"):
        _stage(config, features={})

    # A carried column is checked as a feature's source is: misspelled, it would be null on every row.
    with pytest.raises(KeyError, match="ctx_peer_grp"):
        _stage(config, carry_columns=["ctx_peer_grp"]).reduce(_window())


@pytest.mark.cpu_mode
def test_a_feature_may_be_spelled_as_the_command_line_spells_it(config):
    spelled = _stage(config, features={"fan_out": "dsts_per_src:max", "novel": "endpoint_pair_novel:count_true"})
    listed = _stage(config,
                    features={
                        "fan_out": ["dsts_per_src", "max"], "novel": ["endpoint_pair_novel", "count_true"]
                    })

    pd.testing.assert_frame_equal(spelled.reduce(_window()), listed.reduce(_window()))

    with pytest.raises(ValueError, match="reduction"):
        _stage(config, features={"fan_out": "dsts_per_src,max"})


@pytest.mark.cpu_mode
def test_a_feature_or_carried_column_may_not_take_a_name_every_row_carries(config):
    with pytest.raises(ValueError, match="event_time"):
        _stage(config, features={"event_time": ["dsts_per_src", "max"]})

    with pytest.raises(ValueError, match="lineage_id"):
        _stage(config, carry_columns=["lineage_id"])

    with pytest.raises(ValueError, match="ctx_peer_group"):
        _stage(config, features={"ctx_peer_group": ["dsts_per_src", "max"]}, carry_columns=["ctx_peer_group"])

    # The entity column itself may be named among the carried columns; it is carried once.
    assert _stage(config, carry_columns=["host_key"]).output_columns.count("host_key") == 1


@pytest.mark.cpu_mode
def test_text_under_a_numeric_feature_is_a_mistake_not_a_zero(config):
    # A numeral in text is read; a word is refused and the column named.
    out = _stage(config).reduce(_window(dsts_per_src=["3", "5", "9", None, None]))

    assert list(out["fan_out"]) == [5.0, 9.0]

    with pytest.raises(TypeError, match="dsts_per_src"):
        _stage(config).reduce(_window(dsts_per_src=["three", "5", "9", None, None]))


@pytest.mark.cpu_mode
def test_a_flag_widened_to_float_or_spelled_out_still_counts(config):
    # A nullable boolean widened by a gap arrives as 1.0 and 0.0; a flag read back from text arrives as a word.
    widened = _stage(config).reduce(_window(endpoint_pair_novel=[None, None, None, 1.0, 0.0]))
    spelled = _stage(config).reduce(_window(endpoint_pair_novel=[None, None, None, "true", "false"]))

    assert widened.iloc[0]["novel"] == 1 and spelled.iloc[0]["novel"] == 1

    # And a late flag that arrived as 1.0 still keeps its row out of the frame.
    assert _stage(config).reduce(_window(is_late=[0.0, 1.0, 0.0, 0.0, 0.0])).iloc[0]["frame_rows"] == 3


@pytest.mark.parametrize("reduction, expected", [("max", 4.0), ("min", 1.0), ("mean", 2.5), ("sum", 5.0), ("last", 4.0),
                                                 ("count_true", 2)])
def test_each_reduction_ignores_nulls(reduction: str, expected):
    values = [True, None, 0, float("nan"), 1.0] if reduction == "count_true" else [1, None, float("nan"), 4]

    assert reduce_values(reduction, values) == expected
    assert reduce_values(reduction, [None, float("nan")]) is None


@pytest.mark.cpu_mode
def test_on_data_replaces_the_window_and_drops_one_with_nothing_to_frame(config):
    stage = _stage(config)
    out = stage.on_data(MessageMeta(_window()))

    assert list(out.copy_dataframe()["host_key"]) == ["ws-1", "ws-2"]
    assert stage.on_data(MessageMeta(_window(host_key=[None] * 5))) is None
