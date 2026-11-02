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
The synthetic corpus `morpheus.compositions.endpoint` is tested over, and the facts about it the harnesses assert.

Moved out of `tests/morpheus/determinism/endpoint_pipeline.py` under plan section 4.5, which put the composition in the
package and left the data here, where the tests, the sample and fixture generators and the experiments read it. The
composition module's docstring describes what this corpus plants and why.
"""

import hashlib
import typing

import pandas as pd

from morpheus.compositions import envelope
from morpheus.compositions.endpoint import BUILD_HOSTS
from morpheus.compositions.endpoint import ENDPOINT_CLASS
from morpheus.compositions.endpoint import FINANCE_HOSTS
from morpheus.compositions.endpoint import KIOSK
from morpheus.compositions.endpoint import at
from morpheus.utils.binding_table import NS_PER_SECOND

# The rule's own weights, stated once so the corpus is built to exercise each deliberately.
SEVERITY = {"system": 70, "high": 55, "medium": 40, "low": 25}
UNWEIGHTED_SEVERITY = 40
CRITICALITY_WEIGHT = {"high": 1.5, "medium": 1.0, "low": 0.5}
UNKNOWN_CRITICALITY_WEIGHT = 1.0
LAST_DAY = 40
"""The day the eight processes that decide the rule run. Every day before it is history."""
LAB = "lab-01"
LAB_FIRST_DAY = 36
RENAMED_HOST = "fin-01"
RENAME_DAY = 20
"""From this day the EDR reports `fin-01` in upper case, as an agent upgrade does."""
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
GRADLE = r"C:\Tools\gradle\bin\gradle.exe"
NUGET = r"C:\Tools\nuget.exe"
DOTNET = r"C:\Program Files\dotnet\dotnet.exe"
PYTHON = r"C:\Program Files\Python312\python.exe"
CMAKE = r"C:\Program Files\CMake\bin\cmake.exe"
BUILD_TOOLS = ((JAVA, GRADLE), (MSBUILD, NUGET), (MSBUILD, DOTNET), (JAVA, PYTHON), (MSBUILD, CMAKE))
"""The tool pairs the first build server runs every morning, which the second takes up one a day at the end."""
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
CREEPING = "build-02"
"""A build server that takes up one more of its peer's tool pairs every day for five days. R-P-L7-011's host."""
CREEP_FIRST_DAY = 27
STALE_FIRST_DAY = 2
"""Inside the warm-up, so the tool's first run is history rather than an alert of its own."""
KIOSK_NOTEPAD_DAYS = (3, 20)
"""First inside the warm-up, then again: the second run is already seen, and the one on day 40 is seen again."""


def reported_name(host: str, when: int) -> str:
    """The host as the EDR spells it at an instant."""
    return host.upper() if (host == RENAMED_HOST and when >= at(RENAME_DAY)) else host


def _process(rows: list,
             host: str,
             parent: str,
             image: str,
             when: int,
             integrity: typing.Optional[str] = "Medium",
             signature: str = "Valid"):
    command_line = f'"{image}"'
    host = reported_name(host, when)
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

        for (index, (parent, image)) in enumerate(BUILD_TOOLS):
            _process(rows, host, parent, image, at(day, 9, 40 + index), "High")

    if (host == CREEPING and day >= CREEP_FIRST_DAY):
        # One more of its peer's tools each day: one on the first, all five from the fifth on.
        for (index, (parent, image)) in enumerate(BUILD_TOOLS[:day - CREEP_FIRST_DAY + 1]):
            _process(rows, host, parent, image, at(day, 9, 40 + index), "High")

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

    return envelope.attach_all({ENDPOINT_CLASS: frame})
