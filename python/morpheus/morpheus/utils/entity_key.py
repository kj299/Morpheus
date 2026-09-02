# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Composite entity keys, built the same way by every telemetry stage.

The identifier ladder joins layers on a composed string: a port is `site_id:device_id:port_id` at layer 1 and the
same three values, with the middle one called `switch_id`, at layer 2. The join only works if both layers compose
the string identically, so the composition lives here rather than in each stage.

A key with a missing part is null, not a string with `None` in it. The universal envelope's rule is that an
unavailable field must be explicitly null rather than defaulted to something plausible, because a defaulted value
is indistinguishable from an observed one three months later during an investigation. A row with a null key gets
no per-entity features, and the stage says how many rows that happened to.
"""

import math
import typing

import pandas as pd

KEY_SEPARATOR = ":"
"""Joins the parts of a composite key. The same character at every layer, so keys compare across layers."""


def normalize_text(value: typing.Any) -> typing.Optional[str]:
    """
    Render a host value as stripped text, collapsing every flavor of missing to `None`.

    Parameters
    ----------
    value : any
        A value read from a DataFrame column: a string, a number, `None`, NaN, or `pandas.NA`.

    Returns
    -------
    str or None
        The value as text with surrounding whitespace removed, or `None` if it was missing or blank.
    """
    if (value is None):
        return None

    if (isinstance(value, float) and math.isnan(value)):
        return None

    try:
        if (pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    return text if len(text) > 0 else None


def compose_key(parts: typing.Sequence[typing.Any]) -> typing.Optional[str]:
    """
    Join the parts of a composite key, or return `None` if any part is missing.

    Parameters
    ----------
    parts : sequence
        The key's components in order, for example `(site_id, device_id, port_id)`.

    Returns
    -------
    str or None
        The parts joined with `KEY_SEPARATOR`, or `None` when any part is missing or blank. A key that is half
        an identity is not an identity, and `None:sw1:Gi1/0/1` would silently pool every siteless row under one
        fabricated site.
    """
    normalized = [normalize_text(part) for part in parts]

    if (any(part is None for part in normalized)):
        return None

    return KEY_SEPARATOR.join(normalized)
