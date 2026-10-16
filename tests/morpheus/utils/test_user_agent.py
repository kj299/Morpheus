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

import pytest

from morpheus.utils.user_agent import stack_family_for
from morpheus.utils.user_agent import user_agent_family
from morpheus.utils.user_agent import user_agent_os

CHROME_WINDOWS = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/126.0.0.0 Safari/537.36")
EDGE = CHROME_WINDOWS + " Edg/126.0.0.0"
SAFARI_IPHONE = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
                 "Version/17.5 Mobile/15E148 Safari/604.1")
CHROME_ANDROID = ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/126.0.0.0 Mobile Safari/537.36")
FIREFOX_MAC = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:128.0) Gecko/20100101 Firefox/128.0"


@pytest.mark.parametrize(
    ("header", "family", "system"),
    [(CHROME_WINDOWS, "chrome", "windows"), (EDGE, "edge", "windows"), (SAFARI_IPHONE, "safari", "ios"),
     (CHROME_ANDROID, "chrome", "android"), (FIREFOX_MAC, "firefox", "macos"), ("curl/8.5.0", "curl", None),
     ("python-requests/2.32.3", "python-requests", None),
     ("Mozilla/5.0 (compatible; VulnScanner/10.4)", "mozilla", None)])
def test_a_header_names_its_family_and_the_system_it_claims(header: str, family: str, system: str):
    assert user_agent_family(header) == family
    assert user_agent_os(header) == system


def test_a_browser_is_named_by_its_most_specific_token():
    # Every Chromium browser says Chrome and Safari, and Edge says Edg as well.
    assert user_agent_family(EDGE) == "edge"
    assert user_agent_family(CHROME_WINDOWS) == "chrome"


def test_a_version_is_not_a_family():
    assert user_agent_family("curl/8.5.0") == user_agent_family("curl/7.88.1")
    assert user_agent_family(CHROME_WINDOWS) == user_agent_family(CHROME_WINDOWS.replace("126.0", "127.0"))


@pytest.mark.parametrize("empty", [None, "", "   ", float("nan")])
def test_an_empty_header_says_nothing(empty):
    assert user_agent_family(empty) is None
    assert user_agent_os(empty) is None


def test_each_system_runs_the_stack_its_kernel_does():
    assert stack_family_for("windows") == "windows"
    assert stack_family_for("macos") == stack_family_for("ios") == "darwin"
    assert stack_family_for("linux") == stack_family_for("android") == stack_family_for("chromeos") == "linux"
    assert stack_family_for(None) is None
    assert stack_family_for("plan9") is None
