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

from morpheus.utils.stack_fingerprint import options_order
from morpheus.utils.stack_fingerprint import stack_family
from morpheus.utils.stack_fingerprint import stack_fingerprint


def test_the_options_are_normalized_whatever_the_capture_calls_them():
    assert options_order("MSS,NOP,WScale,NOP,NOP,SACK_PERM") == "mss,nop,ws,nop,nop,sok"
    assert options_order("mss nop ws nop nop sackOK") == "mss,nop,ws,nop,nop,sok"
    assert options_order("mss;sok;timestamp;nop;window_scale") == "mss,sok,ts,nop,ws"


def test_an_option_the_table_does_not_know_is_kept():
    assert options_order("mss,tfo,sok") == "mss,tfo,sok"


@pytest.mark.parametrize(("options", "family"),
                         [("mss,nop,ws,nop,nop,sok", "windows"), ("mss,sok,ts,nop,ws", "linux"),
                          ("mss,nop,ws,nop,nop,ts,sok,eol", "darwin"), ("ws,nop,mss,ts,sok,eol", None)])
def test_the_family_is_read_off_the_order(options: str, family: str):
    assert stack_family(options) == family


def test_the_fingerprint_is_the_order_the_segment_size_and_the_window():
    assert stack_fingerprint("mss,sok,ts,nop,ws", 1460, 29200.0) == "mss,sok,ts,nop,ws|1460|29200"
    assert stack_fingerprint("mss,sok,ts,nop,ws", None, 29200) == "mss,sok,ts,nop,ws||29200"


@pytest.mark.parametrize("missing", [None, "", float("nan")])
def test_a_record_without_options_has_no_fingerprint(missing):
    assert stack_fingerprint(missing, 1460, 64240) is None
    assert stack_family(missing) is None
