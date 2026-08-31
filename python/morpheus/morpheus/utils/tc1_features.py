# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The behavioral feature schema for the TC-1 telemetry class.

Layer 1 produces two kinds of signal. The counters are quantitative and are handled upstream by
`morpheus.utils.counter_delta`. The identifiers are qualitative, and what matters about them is not their value but
whether the value has changed: a port reports the same transceiver serial and the same LLDP neighbor every poll for
months, so a second distinct value on one port is the event. `DistinctIncrementColumn` is the primitive for exactly
that, counting distinct values seen per group per period rather than occurrences.

The two features this builds are the ones the guide's TC-1 section names. A new `transceiver_serial` on a port is
hardware substitution: someone pulled an optic and put another one in, which is what a physical tap looks like from
the switch's point of view. A new `lldp_neighbor_chassis_id` is topology change: the thing on the other end of the
cable is not the thing that was there before.

Both are cumulative, and therefore order-dependent, so the frame must be in determinism control 8's total order
before the schema is applied. `morpheus.stages.telemetry.tc1_feature_stage.TC1FeatureStage` is the supported way to
run this, because it establishes that order first; applying the schema through a stage that does not sort produces
plausible values that are wrong.

The counting is bucketed by period and resets at each boundary, so the period has to be chosen against the span of
the frames the schema is applied to rather than against anything about the estate: a boundary can only hide a change
if it falls inside a frame. The default is monthly on that reasoning. See `build_tc1_feature_schema`.
"""

import typing
from datetime import datetime

import pandas as pd

from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import DistinctIncrementColumn

DEFAULT_TRANSCEIVER_COLUMN = "transceiver_serial"
"""Identifier whose change means the optic in the cage was substituted."""

DEFAULT_NEIGHBOR_COLUMN = "lldp_neighbor_chassis_id"
"""Identifier whose change means the far end of the cable was substituted."""

DEFAULT_ENTITY_KEY_COLUMN = "entity_key"
"""Group for both features: novelty is per port, never per device or per site."""

DEFAULT_TIMESTAMP_COLUMN = "event_time"
"""Column the novelty period is derived from."""

DEFAULT_PERIOD = "M"
"""Period over which distinct values are counted, as a pandas period alias.

Monthly rather than daily because the count resets at each boundary and a change straddling one is not seen. Layer 1
identifiers change on the cadence of maintenance windows, not of days, so a daily bucket puts a boundary between
almost every pair of consecutive changes for no benefit. See `build_tc1_feature_schema` for what the choice costs
and for the rule that actually governs it.
"""

TRANSCEIVER_INCREMENT_COLUMN = "transceiver_increment"
"""Distinct `transceiver_serial` values seen on this port so far in the period."""

NEIGHBOR_INCREMENT_COLUMN = "lldp_neighbor_increment"
"""Distinct `lldp_neighbor_chassis_id` values seen on this port so far in the period."""


def build_tc1_feature_schema(
    entity_key_column: str = DEFAULT_ENTITY_KEY_COLUMN,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
    transceiver_column: str = DEFAULT_TRANSCEIVER_COLUMN,
    neighbor_column: str = DEFAULT_NEIGHBOR_COLUMN,
    period: str = DEFAULT_PERIOD,
    preserve_columns: typing.Sequence[str] = ()) -> DataFrameInputSchema:
    """
    Build the TC-1 novelty feature schema.

    The result carries the entity key and the timestamp through unchanged, so that the output frame is still keyed
    and still orderable, and adds one integer column per identifier. A value of 1 means the port has reported a
    single distinct value for that identifier so far in the period, which is the steady state. Anything above 1 is
    the signal.

    Nulls are counted as a distinct value rather than skipped. An empty cage really is a different physical state
    from a populated one, so a port that goes from no transceiver to a transceiver increments, which is the intended
    reading.

    Parameters
    ----------
    entity_key_column : str, default = "entity_key"
        Column to group by. This is the `site_id:device_id:port_id` key
        `morpheus.stages.telemetry.tc1_normalize_stage.TC1NormalizeStage` writes. Grouping any coarser than the port
        makes the feature meaningless: a device with forty ports legitimately sees forty transceiver serials.
    timestamp_column : str, default = "event_time"
        Column the period is derived from. Must already be a datetime column; `TC1FeatureStage` converts it.
    transceiver_column : str, default = "transceiver_serial"
        Identifier for the installed optic.
    neighbor_column : str, default = "lldp_neighbor_chassis_id"
        Identifier for the neighbor the port sees.
    period : str, default = "M"
        Period over which distinct values are counted, as a pandas period alias. Note that these are period aliases,
        not offset aliases: `"M"` is monthly and `"ME"` is rejected. See the note below on how to choose one.
    preserve_columns : list of str, default = ()
        Additional input columns to carry through to the output frame, as regular expressions.

    Returns
    -------
    `morpheus.utils.column_info.DataFrameInputSchema`

    Raises
    ------
    ValueError
        If `period` is empty or is not a period alias pandas accepts, or if the two identifier columns are the same.

    Notes
    -----
    The count resets at each period boundary, and a change straddling one is not seen: a transceiver swapped just
    before a boundary and first polled just after is one distinct value on each side, so both read 1. Lengthening
    the period makes boundaries rarer but never removes them, so it narrows the blind spot rather than closing it.

    What actually governs this is the relationship between the period and the frame the schema is applied to. A
    boundary can only hide a change if it falls *inside* a frame, so a period longer than the span of one frame
    leaves no boundary for a change to hide behind. With daily windows, a monthly period puts a live boundary in
    roughly one window in thirty instead of in every one.
    `morpheus.stages.telemetry.tc1_feature_stage.TC1FeatureStage` warns when it sees a frame that straddles a
    boundary, so the remaining exposure is reported rather than assumed absent.

    The cost of a long period is drift. The count is cumulative within the period and never decays, so under
    `period="Q"` a port that legitimately changes optics twice reads 3 for the remainder of the quarter, and a rule
    thresholding on "greater than 1" fires for all of it. Choose the period long enough to exceed the frame span and
    no longer.
    """
    if (not period):
        raise ValueError("period is required; it is the window over which distinct values are counted")

    # Validated here so that a bad alias fails when the pipeline is built rather than on the first batch, where it
    # would surface as an exception from inside a groupby.
    try:
        pd.Period("2026-01-01", freq=period)
    except ValueError as error:
        raise ValueError(f"period '{period}' is not a pandas period alias. Note that these are period aliases, not "
                         f"offset aliases: use 'M' for monthly, not 'ME'. Underlying error: {error}") from error

    if (transceiver_column == neighbor_column):
        raise ValueError(f"transceiver_column and neighbor_column are both '{transceiver_column}'; the two features "
                         "would be identical")

    column_info = [
        ColumnInfo(name=entity_key_column, dtype=str),
        ColumnInfo(name=timestamp_column, dtype=datetime),
        DistinctIncrementColumn(name=TRANSCEIVER_INCREMENT_COLUMN,
                                dtype=int,
                                input_name=transceiver_column,
                                groupby_column=entity_key_column,
                                timestamp_column=timestamp_column,
                                period=period),
        DistinctIncrementColumn(name=NEIGHBOR_INCREMENT_COLUMN,
                                dtype=int,
                                input_name=neighbor_column,
                                groupby_column=entity_key_column,
                                timestamp_column=timestamp_column,
                                period=period),
    ]

    return DataFrameInputSchema(column_info=column_info, preserve_columns=list(preserve_columns))
