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

import math

import pandas as pd
import pytest

from morpheus.utils.identity_anchor import AnchoredNames
from morpheus.utils.tracker_state import from_bytes
from morpheus.utils.tracker_state import to_bytes


def test_the_first_name_an_anchor_is_seen_with_is_its_name():
    names = AnchoredNames()

    assert names.resolve(("hq", "sw1"), 10105, "Gi1/0/5") == "Gi1/0/5"
    assert names.resolve(("hq", "sw1"), 10105, "uplink-core-2") == "Gi1/0/5"


def test_an_anchor_widened_to_float_is_the_same_anchor():
    # A gap elsewhere in the column turns 10105 into 10105.0; that must not start a second port.
    names = AnchoredNames()
    names.resolve(("hq", "sw1"), 10105, "Gi1/0/5")

    assert names.resolve(("hq", "sw1"), 10105.0, "uplink-core-2") == "Gi1/0/5"


@pytest.mark.parametrize("missing", [None, math.nan, pd.NA, ""])
def test_a_missing_anchor_keeps_the_reported_name_and_is_not_remembered(missing):
    names = AnchoredNames()

    assert names.resolve(("hq", "sw1"), missing, "Gi1/0/5") == "Gi1/0/5"
    assert len(names) == 0


def test_a_missing_name_or_scope_is_not_remembered():
    names = AnchoredNames()

    assert names.resolve(("hq", "sw1"), 10105, None) is None
    assert names.resolve((None, "sw1"), 10105, "Gi1/0/5") == "Gi1/0/5"
    assert len(names) == 0

    # The first row with a name is the one that names the anchor.
    assert names.resolve(("hq", "sw1"), 10105, "Gi1/0/5") == "Gi1/0/5"


def test_scopes_do_not_share_anchors():
    names = AnchoredNames()
    names.resolve(("hq", "sw1"), 10105, "Gi1/0/5")

    assert names.resolve(("hq", "sw2"), 10105, "Gi1/0/9") == "Gi1/0/9"


def test_a_row_that_may_not_learn_is_named_but_teaches_nothing():
    names = AnchoredNames()

    assert names.resolve(("hq", "sw1"), 10105, "uplink-core-2", learn=False) == "uplink-core-2"
    assert len(names) == 0

    names.resolve(("hq", "sw1"), 10105, "Gi1/0/5")
    assert names.resolve(("hq", "sw1"), 10105, "uplink-core-2", learn=False) == "Gi1/0/5"


def test_the_least_recently_seen_anchor_is_forgotten_first():
    names = AnchoredNames(max_anchors=2)
    names.resolve(("hq", "sw1"), 1, "a")
    names.resolve(("hq", "sw1"), 2, "b")
    names.resolve(("hq", "sw1"), 1, "a")
    names.resolve(("hq", "sw1"), 3, "c")

    # Anchor 1 was seen again after 2, so 2 is the one forgotten: 1 keeps its name, and 2's next name becomes its name.
    assert names.resolve(("hq", "sw1"), 1, "a-renamed") == "a"
    assert names.resolve(("hq", "sw1"), 2, "b-renamed") == "b-renamed"


def test_the_bound_must_be_positive():
    with pytest.raises(ValueError, match="max_anchors"):
        AnchoredNames(max_anchors=0)


def test_the_map_round_trips_through_its_state():
    names = AnchoredNames()
    names.resolve(("hq", "sw1"), 10105, "Gi1/0/5")
    names.resolve(("dc2", "sw9"), 7, "Te1/1/4")

    restored = AnchoredNames.from_state(from_bytes(to_bytes(names.to_state())))

    assert to_bytes(restored.to_state()) == to_bytes(names.to_state())
    assert restored.resolve(("hq", "sw1"), 10105, "renamed") == "Gi1/0/5"
