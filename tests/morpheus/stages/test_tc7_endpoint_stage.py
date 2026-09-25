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
from morpheus.stages.telemetry.tc7_endpoint_stage import TC7EndpointStage
from morpheus.stages.telemetry.tc7_endpoint_stage import normalize_image_path
from morpheus.stages.telemetry.tc7_endpoint_stage import normalize_integrity
from morpheus.utils.type_utils import get_df_class

DAY = 24 * 3600 * 10**9
START = 10**18

EXPLORER = r"C:\Windows\explorer.exe"
CMD = r"C:\Windows\System32\cmd.exe"
WORD = r"C:\Program Files\Microsoft Office\WINWORD.EXE"
POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"


def cell(value):
    return None if pd.isna(value) else value


def truthy(value) -> bool:
    return False if pd.isna(value) else bool(value)


def answered_no(value) -> bool:
    """An answer of "no", as opposed to no answer at all."""
    return not pd.isna(value) and not bool(value)


def processes(rows: list) -> dict:
    """Rows of (host, parent, image, day, peer group, integrity), put in time order as a pipeline would."""
    rows = sorted(rows, key=lambda row: row[3])

    return {
        "hostname": [row[0] for row in rows],
        "parent_image_path": [row[1] for row in rows],
        "image_path": [row[2] for row in rows],
        "event_time": [START + int(row[3] * DAY) for row in rows],
        "ctx_peer_group": [row[4] for row in rows],
        "integrity_level": [row[5] for row in rows],
    }


def run(config: Config, payload: dict, stage: TC7EndpointStage = None, **kwargs) -> pd.DataFrame:
    meta = MessageMeta(get_df_class(config.execution_mode)(payload))
    (stage if stage is not None else TC7EndpointStage(config, **kwargs)).on_data(meta)
    df = meta.copy_dataframe()

    return df.to_pandas() if hasattr(df, "to_pandas") else df


def warm(host: str, group, days: int = 8) -> list:
    """A host doing the ordinary thing once a day for `days` days."""
    return [(host, EXPLORER, CMD, day, group, "Medium") for day in range(days)]


@pytest.mark.gpu_and_cpu_mode
def test_a_pair_new_to_the_host_and_its_group_is_novel(config: Config):
    result = run(config, processes(warm("ws-01", "finance") + [("ws-01", WORD, POWERSHELL, 8, "finance", "High")]))

    assert truthy(result["endpoint_pair_novel"].iloc[-1])
    assert cell(result["endpoint_integrity"].iloc[-1]) == "high"
    assert cell(result["endpoint_pair"].iloc[-1]) == (r"c:\program files\microsoft office\winword.exe -> "
                                                      r"c:\windows\system32\windowspowershell\v1.0\powershell.exe")


@pytest.mark.gpu_and_cpu_mode
def test_a_pair_new_to_the_host_but_common_in_its_group_is_not(config: Config):
    rows = warm("ws-01", "dev") + [("ws-02", WORD, POWERSHELL, 3, "dev", "Medium"),
                                   ("ws-01", WORD, POWERSHELL, 8, "dev", "Medium")]
    result = run(config, processes(rows))

    assert truthy(result["endpoint_peer_seen"].iloc[-1])
    assert not truthy(result["endpoint_host_seen"].iloc[-1])
    assert answered_no(result["endpoint_pair_novel"].iloc[-1])


@pytest.mark.gpu_and_cpu_mode
def test_the_same_pair_in_another_group_does_not_excuse_it(config: Config):
    rows = warm("ws-01", "finance") + [("ws-09", WORD, POWERSHELL, 3, "dev", "Medium"),
                                       ("ws-01", WORD, POWERSHELL, 8, "finance", "Medium")]
    result = run(config, processes(rows))

    assert truthy(result["endpoint_pair_novel"].iloc[-1])


@pytest.mark.gpu_and_cpu_mode
def test_a_host_without_a_peer_group_is_judged_alone_and_flagged(config: Config):
    result = run(config, processes(warm("srv-9", None) + [("srv-9", WORD, POWERSHELL, 8, None, "System")]))

    assert truthy(result["endpoint_pair_novel"].iloc[-1])
    assert truthy(result["endpoint_host_only"].iloc[-1])
    assert cell(result["endpoint_peer_seen"].iloc[-1]) is None


@pytest.mark.gpu_and_cpu_mode
def test_nothing_is_novel_during_the_warmup(config: Config):
    result = run(config,
                 processes(warm("ws-01", "finance", days=3) + [("ws-01", WORD, POWERSHELL, 3, "finance", "High")]))

    assert cell(result["endpoint_pair_novel"].iloc[-1]) is None
    assert not truthy(result["endpoint_mature"].iloc[-1])


@pytest.mark.gpu_and_cpu_mode
def test_a_new_host_in_an_established_group_is_judged_at_once(config: Config):
    rows = warm("ws-01", "finance") + [("ws-new", WORD, POWERSHELL, 8, "finance", "Medium")]
    result = run(config, processes(rows))

    assert truthy(result["endpoint_mature"].iloc[-1])
    assert truthy(result["endpoint_pair_novel"].iloc[-1])


@pytest.mark.gpu_and_cpu_mode
def test_a_pair_seen_more_than_thirty_days_ago_is_novel_again(config: Config):
    rows = warm("ws-01", "finance") + [("ws-01", WORD, POWERSHELL, 1, "finance", "Medium")]
    rows += [("ws-01", EXPLORER, CMD, day, "finance", "Medium") for day in range(8, 40)]
    rows += [("ws-01", WORD, POWERSHELL, 40, "finance", "Medium")]
    result = run(config, processes(rows))

    assert truthy(result["endpoint_pair_novel"].iloc[-1])


@pytest.mark.gpu_and_cpu_mode
def test_per_user_folders_and_case_do_not_make_a_pair_new(config: Config):
    rows = warm("ws-01", "dev") + [
        ("ws-02", EXPLORER, r"C:\Users\alice\AppData\Local\Programs\Code\Code.exe", 3, "dev", "Medium"),
        ("ws-01", EXPLORER.upper(), r"c:\users\BOB\appdata\local\programs\code\code.exe", 8, "dev", "Medium"),
    ]
    result = run(config, processes(rows))

    assert answered_no(result["endpoint_pair_novel"].iloc[-1])


@pytest.mark.gpu_and_cpu_mode
def test_processes_in_one_second_do_not_excuse_each_other(config: Config):
    rows = warm("ws-01", "finance") + [("ws-01", WORD, POWERSHELL, 8, "finance", "High"),
                                       ("ws-02", WORD, POWERSHELL, 8, "finance", "High")]
    result = run(config, processes(rows))

    assert [truthy(value) for value in result["endpoint_pair_novel"].iloc[-2:]] == [True, True]


@pytest.mark.gpu_and_cpu_mode
def test_state_carries_across_messages(config: Config):
    stage = TC7EndpointStage(config)
    run(config, processes(warm("ws-01", "finance") + [("ws-01", WORD, POWERSHELL, 8, "finance", "High")]), stage=stage)
    later = run(config, processes([("ws-01", WORD, POWERSHELL, 9, "finance", "High")]), stage=stage)

    assert answered_no(later["endpoint_pair_novel"].iloc[0])


@pytest.mark.gpu_and_cpu_mode
def test_a_row_without_a_path_is_judged_by_nothing(config: Config):
    result = run(config, processes(warm("ws-01", "finance") + [("ws-01", None, POWERSHELL, 8, "finance", "High")]))

    assert cell(result["endpoint_pair"].iloc[-1]) is None
    assert cell(result["endpoint_pair_novel"].iloc[-1]) is None


@pytest.mark.gpu_and_cpu_mode
def test_missing_peer_group_and_integrity_columns_are_tolerated(config: Config):
    payload = processes(warm("ws-01", None) + [("ws-01", WORD, POWERSHELL, 8, None, None)])
    del payload["ctx_peer_group"]
    del payload["integrity_level"]
    result = run(config, payload)

    assert truthy(result["endpoint_pair_novel"].iloc[-1])
    assert truthy(result["endpoint_host_only"].iloc[-1])
    assert cell(result["endpoint_integrity"].iloc[-1]) is None


@pytest.mark.gpu_and_cpu_mode
def test_a_missing_required_column_names_itself(config: Config):
    payload = processes(warm("ws-01", None))
    del payload["parent_image_path"]

    with pytest.raises(KeyError, match="parent_image_path"):
        run(config, payload)


def test_integrity_levels_are_normalized():
    assert [normalize_integrity(level) for level in ("System", "HIGH", "Medium Plus", "untrusted", "weird", None)
            ] == ["system", "high", "medium", "low", None, None]


def test_image_paths_are_normalized():
    assert normalize_image_path(' "C:\\Users\\Alice\\x.exe" ') == "c:\\users\\*\\x.exe"
    assert normalize_image_path("/home/alice/bin/tool") == "/home/*/bin/tool"
    assert normalize_image_path("/Users/alice/bin/tool") == "/users/*/bin/tool"
    assert normalize_image_path("/usr/bin/bash") == "/usr/bin/bash"
    assert normalize_image_path("") is None


def test_constructor_rejects_bad_configuration(config: Config):
    with pytest.raises(ValueError):
        TC7EndpointStage(config, window_days=0)

    with pytest.raises(ValueError):
        TC7EndpointStage(config, warmup_days=-1)
