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

A part's rendering depends on its value, never on the dtype of the column it arrived in. This matters because
pandas widens an integer column to float the moment one row in the batch is missing: without the rule, port `5`
composes to `hq:sw1:5` in a batch where every port is present and `hq:sw1:5.0` in the next batch, where some
unrelated row had no port. One entity would become two, its baseline would restart under the new name, and
control 13's batch-split sweep would disagree with itself purely on where the corpus was cut.
"""

import ipaddress
import math
import typing

import pandas as pd

KEY_SEPARATOR = ":"
"""Joins the parts of a composite key. The same character at every layer, so keys compare across layers."""


def render_integral(value: typing.Any) -> typing.Optional[str]:
    """
    Render a whole number as an integer, whatever numeric type is carrying it, or `None` for anything else.

    `bool` is excluded deliberately: it satisfies every test for a whole number, but rendering `True` as `1`
    would change an existing key for a type that is not part of the identifier ladder in the first place.
    """
    if (isinstance(value, bool)):
        return None

    is_integer = getattr(value, "is_integer", None)

    if (is_integer is None):
        return None

    try:
        if (not is_integer()):
            return None

        return str(int(value))
    except (TypeError, ValueError, OverflowError):
        return None


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
        The value as text with surrounding whitespace removed, or `None` if it was missing or blank. A whole
        number renders as an integer whatever numeric type is carrying it, so that a column widened to float
        by a missing sibling row does not rename the entities in it.
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

    text = render_integral(value)

    if (text is None):
        text = str(value).strip()

    return text if len(text) > 0 else None


def normalize_hostname(value: typing.Any, strip_domain: bool = False) -> typing.Optional[str]:
    """
    Render a host name as one host, however the source that reported it spelled it.

    DNS names are case-insensitive, and the sources this fork reads disagree about case and about the root: an EDR
    reports `FIN-01`, a directory logs a login to `fin-01`, a resolver answers `fin-01.corp.example.com.`. Compared
    as text they are three hosts, and a host whose pair history is split across them never has a history at all.
    The name is case-folded and a trailing root dot removed, which makes every spelling of one name compare equal
    and changes nothing a name means.

    Parameters
    ----------
    value : any
        A value read from a DataFrame column.
    strip_domain : bool, default = False
        Also drop everything after the first dot, so a short name and its fully qualified form compare equal. Off
        by default because it is a claim about the estate, not about DNS: two domains can each hold a `fin-01`, and
        stripping makes them one host. An address is never stripped, since `10.0.0.5` is not a host called `10`.

    Returns
    -------
    str or None
        The normalized name, or `None` if the value was missing or blank.
    """
    text = normalize_text(value)

    if (text is None):
        return None

    text = text.casefold().rstrip(".")

    if (strip_domain and not _is_address(text)):
        text = text.split(".", 1)[0]

    return text if len(text) > 0 else None


def _is_address(text: str) -> bool:
    try:
        ipaddress.ip_address(text)
    except ValueError:
        return False

    return True


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


LINK_SEPARATOR = "|"
"""Joins the two ends of a link key. Not `KEY_SEPARATOR`, which already joins the parts inside each end."""


def compose_link_key(near_key: typing.Any, far_chassis: typing.Any, far_port: typing.Any) -> typing.Optional[str]:
    """
    Name a link by its two ends: the port it was seen from, and the LLDP neighbour on the other end of it.

    The ends are sorted rather than kept near-then-far, so a link has one name whichever end describes it, wherever
    the far end's LLDP identity and the near end's key are the same strings; where they are not, which is the usual
    case, each end of a link carries a key of its own and the adjacency lookup holds both. A missing part on either
    end yields `None`, for the reason `compose_key` gives: a link to nobody in particular is not a link.

    Parameters
    ----------
    near_key : str
        The near port's own key, `site_id:device_id:port_id` as `compose_key` builds it.
    far_chassis : str
        The neighbour's LLDP chassis identifier.
    far_port : str
        The neighbour's LLDP port identifier.

    Returns
    -------
    str or None
    """
    near = normalize_text(near_key)
    far = compose_key([far_chassis, far_port])

    if (near is None or far is None):
        return None

    return LINK_SEPARATOR.join(sorted([near, far]))
