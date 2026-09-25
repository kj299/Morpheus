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
from morpheus.stages.telemetry.tc7_saas_stage import TC7SaasStage
from morpheus.utils.type_utils import get_df_class

HOUR = 3600 * 10**9
DAY = 24 * HOUR
MONDAY = 4 * DAY


def truthy(value) -> bool:
    """A column value as a plain bool, with a null reading as False."""
    return False if pd.isna(value) else bool(value)


def cell(value):
    return None if pd.isna(value) else value


def run(config: Config, payload: dict, stage: TC7SaasStage = None, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    (stage if stage is not None else TC7SaasStage(config, **kwargs)).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


def operations(counts, principals=None, ops=None, types=None, results=None, times=None) -> dict:
    size = len(counts)

    return {
        "user_principal": list(principals) if principals is not None else ["ann"] * size,
        "operation": list(ops) if ops is not None else ["Report.Export"] * size,
        "target_object_type": list(types) if types is not None else ["Report"] * size,
        "record_count": list(counts),
        "result": list(results) if results is not None else ["success"] * size,
        "event_time": list(times) if times is not None else [MONDAY + index * HOUR for index in range(size)],
    }


@pytest.mark.gpu_and_cpu_mode
def test_a_baseline_is_published_only_after_min_samples_priors(config: Config):
    result = run(config, operations([10, 10, 10, 50]), min_samples=3)

    assert [truthy(value) for value in result["saas_baseline_mature"]] == [False, False, False, True]
    assert cell(result["saas_record_baseline"][3]) == 10.0
    assert cell(result["saas_record_ratio"][3]) == 5.0


@pytest.mark.gpu_and_cpu_mode
def test_the_default_needs_a_hundred_priors(config: Config):
    result = run(config, operations([10] * 100 + [50]))

    assert not truthy(result["saas_baseline_mature"][99])
    assert truthy(result["saas_baseline_mature"][100])


@pytest.mark.gpu_and_cpu_mode
def test_each_operation_has_its_own_baseline(config: Config):
    # Large queries do not excuse an export: each is measured against its own kind.
    counts = [3000, 10, 3000, 10, 3000, 10, 100]
    ops = ["Query", "Export"] * 3 + ["Export"]
    result = run(config, operations(counts, ops=ops), min_samples=3)

    assert list(result["saas_baseline_key"])[-1] == "ann:Export"
    assert cell(result["saas_record_ratio"].iloc[-1]) == 10.0


@pytest.mark.gpu_and_cpu_mode
def test_an_operation_is_not_part_of_its_own_baseline(config: Config):
    result = run(config, operations([10, 10, 10, 1000, 10]), min_samples=3)

    assert cell(result["saas_record_ratio"][3]) == 100.0
    # The enormous export is prior history for the next one, as it should be.
    assert cell(result["saas_record_baseline"][4]) == 1000.0


@pytest.mark.gpu_and_cpu_mode
def test_exports_logged_in_the_same_second_are_each_measured(config: Config):
    # Audit logs stamp to the second, so a burst of exports can share one. None of them is refused, and none is
    # part of the baseline another is measured against.
    times = [MONDAY, MONDAY + HOUR, MONDAY + 2 * HOUR, MONDAY + 3 * HOUR, MONDAY + 3 * HOUR]
    result = run(config, operations([10, 10, 10, 500, 20], times=times), min_samples=3)

    assert [cell(value) for value in result["saas_record_ratio"]][3:] == [50.0, 2.0]
    assert [cell(value) for value in result["saas_record_baseline"]][3:] == [10.0, 10.0]


@pytest.mark.gpu_and_cpu_mode
def test_a_failed_operation_is_measured_by_nothing(config: Config):
    result = run(config,
                 operations([10, 10, 10, 9000, 20],
                            results=["success"] * 3 + ["denied", "success"],
                            types=["A", "A", "A", "B", "A"]),
                 min_samples=3)

    assert truthy(result["saas_operation_failed"][3])
    assert cell(result["saas_record_ratio"][3]) is None
    assert cell(result["saas_object_types_in_week"][3]) is None
    # The denied row neither joined the baseline nor added its type to the week.
    assert cell(result["saas_record_baseline"][4]) == 10.0
    assert cell(result["saas_object_types_in_week"][4]) == 1


@pytest.mark.gpu_and_cpu_mode
def test_breadth_is_a_running_count_of_distinct_types_that_restarts_on_monday(config: Config):
    times = [MONDAY + HOUR, MONDAY + 2 * HOUR, MONDAY + 3 * HOUR, MONDAY + 6 * DAY, MONDAY + 7 * DAY]
    result = run(config, operations([1] * 5, types=["A", "B", "A", "C", "A"], times=times))

    assert list(result["saas_object_types_in_week"]) == [1, 2, 2, 3, 1]
    assert list(result["saas_week_id"]) == [0, 0, 0, 0, 1]


@pytest.mark.gpu_and_cpu_mode
def test_the_week_epoch_decides_where_a_week_ends(config: Config):
    # Sunday and Monday are one week when weeks start on Sunday.
    times = [MONDAY - HOUR, MONDAY + HOUR]
    monday = run(config, operations([1, 1], types=["A", "B"], times=times))
    sunday = run(config, operations([1, 1], types=["A", "B"], times=times), week_epoch="1970-01-04")

    assert list(monday["saas_object_types_in_week"]) == [1, 1]
    assert list(sunday["saas_object_types_in_week"]) == [1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_principals_are_kept_apart(config: Config):
    result = run(config, operations([1, 1, 1], principals=["ann", "bob", "ann"], types=["A", "B", "C"]))

    assert list(result["saas_object_types_in_week"]) == [1, 1, 2]


@pytest.mark.gpu_and_cpu_mode
def test_a_row_without_a_principal_or_time_is_measured_by_nothing(config: Config):
    result = run(config, operations([1, 1], principals=[None, "ann"], times=[None, MONDAY]))

    assert cell(result["saas_object_types_in_week"][0]) is None
    assert cell(result["saas_object_types_in_week"][1]) == 1


@pytest.mark.gpu_and_cpu_mode
def test_state_carries_across_messages(config: Config):
    stage = TC7SaasStage(config, min_samples=2)
    run(config, operations([10, 10], times=[MONDAY, MONDAY + HOUR]), stage=stage)
    later = run(config, operations([40], times=[MONDAY + 2 * HOUR], types=["B"]), stage=stage)

    assert cell(later["saas_record_ratio"][0]) == 4.0
    assert cell(later["saas_object_types_in_week"][0]) == 2


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_required_column_names_itself(config: Config):
    payload = operations([1])
    del payload["target_object_type"]

    with pytest.raises(KeyError, match="target_object_type"):
        run(config, payload)


def test_constructor_rejects_bad_configuration(config: Config):
    with pytest.raises(ValueError):
        TC7SaasStage(config, baseline_days=0)

    with pytest.raises(ValueError):
        TC7SaasStage(config, min_samples=0)
