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

from morpheus.utils.tcp_flags import ACK
from morpheus.utils.tcp_flags import FIN
from morpheus.utils.tcp_flags import FLAG_NAMES
from morpheus.utils.tcp_flags import PSH
from morpheus.utils.tcp_flags import RST
from morpheus.utils.tcp_flags import SYN
from morpheus.utils.tcp_flags import flags_from_byte


def reference(flags: int) -> dict:
    """The extraction `abp_pcap_preprocessing.py` performs, transcribed.

    It renders the byte as a five-character binary string and indexes into it. Reproduced here rather than
    described so the mask below is checked against the thing it replaces rather than against a reading of it.
    """
    flags = flags & 0xFF
    rendered = (str(flags // 16 % 2) + str(flags // 8 % 2) + str(flags // 4 % 2) + str(flags // 2 % 2) + str(flags % 2))

    return {FLAG_NAMES[index]: int(rendered[index]) for index in range(5)}


@pytest.mark.parametrize("value", list(range(256)))
def test_the_mask_agrees_with_the_example_for_every_byte(value: int):
    # The whole reason this module exists: a CPU pipeline and a GPU one have to agree bit for bit with the
    # reference, and "it is the same arithmetic" is a claim worth checking over the whole domain rather than
    # over the handful of values somebody thought of.
    assert flags_from_byte(value) == reference(value)


def test_the_named_bits_are_the_ones_the_rfc_assigns():
    assert flags_from_byte(SYN)["syn"] == 1
    assert flags_from_byte(ACK)["ack"] == 1
    assert flags_from_byte(RST)["rst"] == 1
    assert flags_from_byte(PSH)["psh"] == 1
    assert flags_from_byte(FIN)["fin"] == 1


def test_a_handshake_byte_reads_as_both_bits():
    assert flags_from_byte(SYN | ACK) == {"ack": 1, "psh": 0, "rst": 0, "syn": 1, "fin": 0}


def test_the_three_bits_the_model_never_saw_are_not_reported():
    # URG, ECE and CWR are in the byte and are deliberately absent. An estate wanting them is adding features
    # the shipped model has never seen, which is a decision rather than a default.
    assert set(flags_from_byte(0xFF)) == set(FLAG_NAMES)
    assert len(FLAG_NAMES) == 5


@pytest.mark.parametrize("value", [None, -1, 256, 7.5, "syn", object()])
def test_a_value_that_is_not_a_flags_byte_yields_nothing(value):
    # A flags byte of 300 is a parsing fault, and masking it would turn a wrong record into five plausible bits.
    assert flags_from_byte(value) is None


def test_a_whole_float_is_accepted():
    # A column widened to float by one null row still carries whole numbers, and refusing them would drop every
    # record in the batch rather than the one that was missing.
    assert flags_from_byte(18.0) == flags_from_byte(18)
