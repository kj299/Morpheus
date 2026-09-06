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


def test_a_widened_column_does_not_rename_the_entities_in_it():
    # pandas widens an integer column to float as soon as one row in the batch is missing. Port 5 must compose to
    # the same key in both batches; otherwise a single unrelated null renames an entity, its baseline restarts
    # under the new name, and control 13's batch-split sweep disagrees with itself on where the corpus was cut.
    complete = pd.DataFrame({"site": ["hq", "hq"], "switch": ["sw1", "sw1"], "port": [5, 6]})
    widened = pd.DataFrame({"site": ["hq", "hq"], "switch": ["sw1", "sw1"], "port": [5, None]})

    assert complete["port"].dtype != widened["port"].dtype, "pandas no longer widens; this test proves nothing"

    def keys(frame):
        return [compose_key(row) for row in zip(frame["site"], frame["switch"], frame["port"])]

    assert keys(complete)[0] == keys(widened)[0] == "hq:sw1:5"
    assert keys(widened)[1] is None


@pytest.mark.parametrize("five", [5, np.int64(5), 5.0, np.float32(5.0), np.float64(5.0)])
def test_a_whole_number_renders_the_same_from_every_numeric_type(five):
    # Layer 1 composes `site_id:device_id:port_id` and layer 2 composes the same three values under different
    # column names. The two halves of the ladder join on the rendered string, so the storage type each pipeline
    # happened to read must not survive into it.
    assert normalize_text(five) == "5"


def test_only_numbers_are_renumbered():
    # The negative control for the rule above: it reads values, never text. Reinterpreting a string would strip
    # the leading zeros off a port named "007" and rewrite an identifier the estate chose.
    assert normalize_text(5.5) == "5.5"
    assert normalize_text("5.0") == "5.0"
    assert normalize_text("007") == "007"
    assert normalize_text(True) == "True"


def test_every_site_that_normalizes_a_value_uses_this_rule():
    """
    The rule is only worth having if nothing keeps a private copy of it.

    Each of these sites normalizes a value that identifies something: a VLAN that `ouis_per_vlan` counts by, an
    802.1X outcome that decides whether an exchange opens or closes, and the attributes that break a tie between
    two bindings covering the same instant. Each had, or could grow, its own `str(value).strip()`, and the copies
    do not fail loudly -- they fail by renaming an entity, or by turning a missing value into the string `"<NA>"`.
    """
    # Imported here rather than at module scope: this file is about the rule, and only this test is about who
    # uses it.
    from morpheus.stages.telemetry.tc2_auth_stage import TC2AuthStage
    from morpheus.stages.telemetry.tc2_cardinality_stage import TC2CardinalityStage

    for normalize in (TC2CardinalityStage._text, TC2AuthStage._text):  # pylint: disable=protected-access
        assert normalize(10) == normalize(10.0) == normalize("10") == "10"
        assert normalize(pd.NA) is None
        assert normalize("") is None
        assert normalize("  accept  ") == "accept"


def test_two_bindings_differing_only_in_dtype_break_ties_the_same_way():
    # The binding table's tie-break renders attributes with the same rule, so a column widened to float by some
    # other row cannot pick a different winner depending on where the batch was cut.
    from morpheus.utils.binding_table import Binding
    from morpheus.utils.binding_table import _sort_key  # pylint: disable=protected-access

    integral = Binding(key="10.0.0.1", start_ns=0, end_ns=100, values=(10, ), uid="a")
    widened = Binding(key="10.0.0.1", start_ns=0, end_ns=100, values=(10.0, ), uid="b")

    assert _sort_key(integral) == _sort_key(widened)
