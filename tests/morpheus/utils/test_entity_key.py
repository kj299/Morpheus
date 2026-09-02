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

import numpy as np
import pandas as pd
import pytest

from morpheus.utils.entity_key import KEY_SEPARATOR
from morpheus.utils.entity_key import compose_key
from morpheus.utils.entity_key import normalize_text


def test_a_complete_key_is_joined_in_order():
    assert compose_key(("hq", "sw1", "Gi1/0/1")) == "hq:sw1:Gi1/0/1"
    assert KEY_SEPARATOR == ":"


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA, np.nan, "", "   "])
def test_any_missing_part_makes_the_whole_key_null(missing):
    # Half an identity is not an identity. "None:sw1:Gi1/0/1" would pool every siteless port under one fabricated
    # site, which is exactly the defaulted-value failure the envelope forbids.
    assert compose_key((missing, "sw1", "Gi1/0/1")) is None
    assert compose_key(("hq", "sw1", missing)) is None


def test_numbers_and_padding_are_normalized():
    # A VLAN read as an integer and one read as a string must produce the same key, or the same segment splits in two.
    assert compose_key(("hq", "sw1", 10)) == compose_key(("hq", "sw1", "10"))
    assert compose_key((" hq ", "sw1", "Gi1/0/1 ")) == "hq:sw1:Gi1/0/1"


def test_case_is_preserved():
    # Interface names are case-sensitive on some platforms and "Gi1/0/1" is how the operator reads it.
    assert compose_key(("HQ", "sw1", "Gi1/0/1")) == "HQ:sw1:Gi1/0/1"


def test_normalize_text_collapses_every_flavor_of_missing():
    for missing in (None, float("nan"), pd.NA, pd.NaT, np.nan, ""):
        assert normalize_text(missing) is None

    assert normalize_text(" x ") == "x"
    assert normalize_text(0) == "0"
