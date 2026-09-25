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
"""
The layer 7 endpoint sub-class, `tc7_endpoint`, and its composed pipeline, for the determinism harness.

R-B-L7-004 asks whether a process's `(parent_image_path, image_path)` pair is new to its host and to the host's peer
group in thirty days, and the peer group is TC-0's: the pipeline enriches every process with its host's asset record
as known at the process's time, then judges the pair, stamps and seals hourly windows keyed on the host.

The corpus is forty-one days of EDR process starts from seven hosts: three finance workstations and two build
servers in two peer groups, a kiosk the inventory knows but has put in no group, and a lab machine the inventory has
never heard of, which appears on day 36. Every host starts its day with a burst of processes in the same second,
because that is what an EDR log looks like at logon. On day 40, eight processes decide the rule:

- a finance workstation's **word processor starts PowerShell**, at high integrity. Nothing in the finance group has
  done that, and fires -- although a build server runs the same pair every day, because a pairing that is routine
  in another group is no excuse in this one;
- a **remote-access tool** is started at system integrity on a workstation that last ran it thirty-eight days ago, and
  fires, because a pair not seen in thirty days is as unexplained as one never seen;
- the **kiosk starts `mshta.exe`**, and fires on its own history, flagged as judged without a peer group;
- a build server's **`git` starts `curl`** with no integrity level reported, and fires at the default weight, flagged
  as unweighted;
- a **build server compiles** with a pair it has never run and its peer runs every day, and is quiet on the peer
  group alone;
- a workstation starts an **editor installed under a different user's profile** than the peer who runs it daily, and
  is quiet because per-user folders are collapsed;
- the **kiosk starts Notepad**, which it last did on day 20, and is quiet on its own history;
- and the **lab machine**, four days old, starts `nmap`, and is quiet because no comparison is possible yet.
"""

import hashlib
import typing

import pandas as pd

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.stages.telemetry.tc7_endpoint_stage import TC7EndpointStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900

ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []

OSI_LAYER = 7
ENTITY_COLUMNS = ["hostname"]
ENDPOINT_CLASS = "tc7_endpoint"

# The rule's own weights, stated once so the corpus is built to exercise each deliberately.
SEVERITY = {"system": 70, "high": 55, "medium": 40, "low": 25}
UNWEIGHTED_SEVERITY = 40
WINDOW_DAYS = 30
WARMUP_DAYS = 7

DAY_SECONDS = 86400
LAST_DAY = 40
"""The day the eight processes that decide the rule run. Every day before it is history."""

CONTEXT_RECORDED_DAY = 0

FINANCE_GROUP = "finance-workstations"
BUILD_GROUP = "build-servers"

FINANCE_HOSTS = {"fin-01": "alice", "fin-02": "bob", "fin-03": "carol"}
BUILD_HOSTS = ("build-01", "build-02")
KIOSK = "kiosk-01"
LAB = "lab-01"
LAB_FIRST_DAY = 36

# Image paths, as an EDR reports them.
SERVICES = r"C:\Windows\System32\services.exe"
SVCHOST = r"C:\Windows\System32\svchost.exe"
EXPLORER = r"C:\Windows\explorer.exe"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUTLOOK = r"C:\Program Files\Microsoft Office\root\Office16\OUTLOOK.EXE"
WORD = r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"
POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
JAVA = r"C:\Program Files\Eclipse Adoptium\jdk-17\bin\java.exe"
MSBUILD = r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
COMPILER = r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.38\bin\Hostx64\x64\cl.exe"
GIT = r"C:\Program Files\Git\cmd\git.exe"
CURL = r"C:\Windows\System32\curl.exe"
TEAMVIEWER = r"C:\Program Files\TeamViewer\TeamViewer.exe"
NOTEPAD = r"C:\Windows\System32\notepad.exe"
MSHTA = r"C:\Windows\System32\mshta.exe"
NMAP = r"C:\Program Files (x86)\Nmap\nmap.exe"
KIOSK_APP = r"C:\Program Files\Kiosk\kiosk-shell.exe"


def editor(user: str) -> str:
    """The code editor, installed per user."""
    return rf"C:\Users\{user}\AppData\Local\Programs\Microsoft VS Code\Code.exe"


# The eight processes that decide the rule, by what each is meant to show.
NOVEL_HIGH = "fin-01"
"""Word starts PowerShell. Fires at high."""
STALE = "fin-02"
"""A remote-access tool last run thirty-eight days ago. Fires at system."""
HOST_ONLY_NOVEL = KIOSK
"""`mshta.exe` on the ungrouped kiosk. Fires on its own history, flagged."""
UNWEIGHTED = "build-02"
"""`git` starts `curl` with no integrity reported. Fires at the default weight, flagged."""
PEER_SEEN = "build-02"
"""A compile the host has never run and its peer runs daily. Quiet."""
PER_USER = "fin-03"
"""An editor under a profile the peer who runs it does not have. Quiet."""
HOST_SEEN = KIOSK
"""Notepad on the kiosk, which last ran it on day 20. Quiet."""
WARMING_UP = LAB
"""`nmap` on a machine four days old. Quiet."""

STALE_FIRST_DAY = 2
"""Inside the warm-up, so the tool's first run is history rather than an alert of its own."""
KIOSK_NOTEPAD_DAYS = (3, 20)
"""First inside the warm-up, then again: the second run is already seen, and the one on day 40 is seen again."""


def at(day: float, hour: float = 0, minute: float = 0, second: float = 0) -> int:
    """An instant in the corpus, in nanoseconds since the epoch."""
    return int((day * DAY_SECONDS + hour * 3600 + minute * 60 + second) * NS_PER_SECOND)


def asset_versions() -> list:
    """The inventory: the finance and build groups, and the kiosk with no group. The lab machine is not in it."""
    versions = []
    owners = dict(FINANCE_HOSTS)

    for host in sorted(set(FINANCE_HOSTS) | set(BUILD_HOSTS) | {KIOSK}):
        group = FINANCE_GROUP if host in FINANCE_HOSTS else (BUILD_GROUP if host in BUILD_HOSTS else None)
        versions.append(
            make_version("asset",
                         host,
                         0,
                         None,
                         at(CONTEXT_RECORDED_DAY),
                         values={
                             "owner": f"{owners[host]}@example.com" if host in owners else "it-ops@example.com",
                             "owning_team": "finance" if host in FINANCE_HOSTS else "platform",
                             "criticality": "high" if host in BUILD_HOSTS else "medium",
                             "data_classification": "confidential" if host in FINANCE_HOSTS else "internal",
                             "peer_group": group,
                         }))

    return versions


def build_store() -> BitemporalStore:
    """The asset store the enrichment reads."""
    return BitemporalStore("asset", asset_versions())


def _process(rows: list,
             host: str,
             parent: str,
             image: str,
             when: int,
             integrity: typing.Optional[str] = "Medium",
             signature: str = "Valid"):
    command_line = f'"{image}"'
    rows.append({
        "collector_id": "edr-01",
        "schema_version": "tc7_endpoint.v1",
        "origin_hash": f"{host}>{image}@{when}",
        "collector_seq": len(rows),
        "event_time": when,
        "hostname": host,
        "process_guid": f"{{{hashlib.sha256(f'{host}|{image}|{when}'.encode()).hexdigest()[:32]}}}",
        "parent_process_guid": f"{{{hashlib.sha256(f'{host}|{parent}|{when}'.encode()).hexdigest()[:32]}}}",
        "parent_image_path": parent,
        "image_path": image,
        "command_line_hash": hashlib.sha256(command_line.encode()).hexdigest(),
        "integrity_level": integrity,
        "signature_status": signature,
    })


def _logon(rows: list, host: str, day: int):
    """A host's first second of the day: three process starts at one instant, then two more in the hour."""
    burst = at(day, 9)

    if (host in BUILD_HOSTS):
        _process(rows, host, SERVICES, SVCHOST, burst, "System")
        _process(rows, host, SERVICES, JAVA, burst, "System")
        _process(rows, host, JAVA, MSBUILD, burst, "High")
        _process(rows, host, JAVA, GIT, at(day, 9, 10), "High")
    elif (host == KIOSK):
        _process(rows, host, SERVICES, SVCHOST, burst, "System")
        _process(rows, host, EXPLORER, KIOSK_APP, burst)
        _process(rows, host, EXPLORER, CHROME, burst)
        _process(rows, host, KIOSK_APP, CHROME, at(day, 9, 10))
    else:
        _process(rows, host, SERVICES, SVCHOST, burst, "System")
        _process(rows, host, EXPLORER, CHROME, burst)
        _process(rows, host, EXPLORER, OUTLOOK, burst)
        _process(rows, host, OUTLOOK, CHROME, at(day, 9, 10))

    if (host == "build-01"):
        # The compile the other build server has never run, and documentation rendered through Word and PowerShell,
        # which the finance group never does.
        _process(rows, host, MSBUILD, COMPILER, at(day, 9, 20), "High")
        _process(rows, host, WORD, POWERSHELL, at(day, 9, 30), "High")

    if (host == "fin-02"):
        # The editor, installed under Bob's profile.
        _process(rows, host, EXPLORER, editor(FINANCE_HOSTS[host]), at(day, 9, 20))


def build_corpus() -> dict[str, pd.DataFrame]:
    """The seeded corpus: one frame of EDR process starts."""
    rows: list = []

    for day in range(LAST_DAY + 1):
        for host in sorted(set(FINANCE_HOSTS) | set(BUILD_HOSTS) | {KIOSK}):
            _logon(rows, host, day)

        if (day >= LAB_FIRST_DAY):
            _logon(rows, LAB, day)

    _process(rows, STALE, EXPLORER, TEAMVIEWER, at(STALE_FIRST_DAY, 11), "High")

    for day in KIOSK_NOTEPAD_DAYS:
        _process(rows, KIOSK, EXPLORER, NOTEPAD, at(day, 11))

    decisive = at(LAST_DAY, 10)
    minute = 60 * NS_PER_SECOND

    _process(rows, NOVEL_HIGH, WORD, POWERSHELL, decisive + 1 * minute, "High")
    _process(rows, PEER_SEEN, MSBUILD, COMPILER, decisive + 2 * minute, "High")
    _process(rows, PER_USER, EXPLORER, editor("Carol"), decisive + 3 * minute)
    _process(rows, HOST_SEEN, EXPLORER, NOTEPAD, decisive + 4 * minute)
    _process(rows, HOST_ONLY_NOVEL, EXPLORER, MSHTA, decisive + 5 * minute)
    _process(rows, STALE, EXPLORER, TEAMVIEWER, decisive + 6 * minute, "System")
    _process(rows, WARMING_UP, EXPLORER, NMAP, decisive + 7 * minute, "High", "Unsigned")
    _process(rows, UNWEIGHTED, GIT, CURL, decisive + 8 * minute, None)

    frame = pd.DataFrame(rows).sort_values(["event_time", "collector_seq"], kind="mergesort").reset_index(drop=True)
    frame["collector_seq"] = range(len(frame))

    return {ENDPOINT_CLASS: frame}


def build_pipeline_config(execution_mode=None) -> Config:
    """
    A pipeline configuration, defaulting to CPU mode and importable without a GPU.

    Parameters
    ----------
    execution_mode : `morpheus.config.ExecutionMode`, optional
        Mode to build for. Defaults to CPU, which is what the golden file is an artifact of.
    """
    from morpheus.config import CppConfig  # pylint: disable=import-outside-toplevel
    from morpheus.config import ExecutionMode  # pylint: disable=import-outside-toplevel

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.execution_mode = ExecutionMode.CPU if execution_mode is None else execution_mode

    return config


def _collect(sink: InMemorySinkStage) -> pd.DataFrame:
    """Everything the sink received, as one host frame."""
    frames = []

    for message in sink.get_messages():
        meta = message.payload() if hasattr(message, "payload") else message
        frame = meta.copy_dataframe()
        frames.append(frame.to_pandas() if hasattr(frame, "to_pandas") else frame)

    if (len(frames) == 0):
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True) -> pd.DataFrame:
    """
    Source, stamp, total order, enrich with the host's asset record, judge, envelope, seal hourly, sink.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    corpus : dict
        The frames from `build_corpus`, possibly permuted.
    batches : dict, optional
        How the corpus is split across source frames. Defaults to one frame.
    impose_order : bool, default = True
        Place `TotalOrderStage` ahead of the stateful stages.

    Returns
    -------
    `pandas.DataFrame`
        The class's output, tagged with `telemetry_class`, keyed by `row_key`, canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=batches[ENDPOINT_CLASS]))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    pipe.add_stage(TC0EnrichStage(config, store=build_store(), entity_column="hostname"))
    pipe.add_stage(TC7EndpointStage(config, window_days=WINDOW_DAYS, warmup_days=WARMUP_DAYS))
    pipe.add_stage(EnvelopeStampStage(config, osi_layer=OSI_LAYER, entity_columns=ENTITY_COLUMNS))
    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=PERIOD_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        entity_key_column="entity_key"))

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    frame = _collect(sink)
    frame["telemetry_class"] = ENDPOINT_CLASS
    frame["row_key"] = frame["event_uid"]

    return canonicalize(frame, key_columns=KEY_COLUMNS, ignore_columns=IGNORE_COLUMNS)


def render(result: pd.DataFrame) -> str:
    """The canonical CSV rendering the golden file holds."""
    return result.to_csv(index=False, lineterminator="\n")
