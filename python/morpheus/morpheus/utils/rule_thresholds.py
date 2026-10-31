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
The numbers a detection fires at: where they are written, how a baseline period proposes them, and their digest.

The guide is plain that most of its thresholds are placeholders -- the right shape, a value that "must be derived
from a baseline period in your own data before the rule goes live" -- and until plan section 3.6 nothing in the
tree did either half of that (gap G60). The thresholds lived as literals inside sixty-one searches, no tool read
an estate's own history to propose a value, and `config_hash` digested the pipeline's settings but not the numbers
the SIEM judged its output against, so two notables fired under different thresholds looked alike.

This module is the Python half of three pieces that close it:

**A registry.** The Splunk app ships `rule_thresholds.csv`, one row per threshold per rule, with the operator, the
value, what kind of number it is, and -- where it is a placeholder a scored event can inform -- the event field
and the quantile a baseline proposes it from. `extract_thresholds` reads a search the same way the registry is
written, so a test can hold the two together and a stanza retuned without its row fails rather than drifting.

**A derivation.** `derive_threshold` takes a field's values over a baseline period and returns the value at the
row's quantile by nearest rank -- always one of the values seen, never an interpolation -- so a proposed
threshold is something the estate actually did. The direction is the operator's: a rule firing above a value is
proposed from a high quantile, one firing below it from a low one. `examples/splunk_lineage_app/validate/
derive_thresholds.py` runs it over scored events and reports each proposal beside the shipped value. The tool
proposes; it never edits a search, because a threshold that moved without a reviewer reading why is the defect
the registry exists to prevent.

**A digest.** `thresholds_hash` renders one rule's thresholds through `settings_hash`, the same canonical form the
envelope's `config_hash` uses, so `4` and `4.0` do not fork it. The app's `rule_metadata.csv` carries it per rule
and every detection looks it up into the risk record it writes, so the record of a notable says which numbers
fired it, and a retune is visible as a change of digest on every notable after it.

**Zero is not a threshold.** `dsts_per_src_step > 0` says the count rose; `flags > 0` guards a division. A
comparison against zero is the definition of the condition rather than a number anyone would tune, and
`extract_thresholds` leaves it out so the registry holds only what a baseline could move.
"""

import re
import typing

from morpheus.utils.determinism_envelope import settings_hash
from morpheus.utils.transfer_envelope import nearest_rank

KINDS = ("conventional", "placeholder", "gate", "window", "definition")
"""What a threshold is, which decides whether a baseline period may move it.

- `conventional`: a value the guide calls a widely used starting point that transfers between estates, such as
  the 900 km/h impossible-travel speed.
- `placeholder`: a value stated to make the rule concrete; the one kind `derive_threshold` proposes.
- `gate`: a floor on how much evidence a judgement needs, such as twenty prior observations of a client. Moving it
  trades how early a rule speaks against how often it is wrong, which is a policy rather than a measurement.
- `window`: a time a chain's steps must fall within, or a gap two sightings must exceed.
- `definition`: a number the condition is made of, such as more than one MAC claiming an address.
"""

OPERATORS = (">=", "<=", ">", "<", "=")
"""`=` marks a named threshold the search assigns with `eval` and compares by name."""

DIRECTIONS = ("above", "below")
"""Which side of its threshold a rule fires on. Stated rather than read from the operator, because a named
threshold's operator is the `=` that assigns it and says nothing about the comparison that uses it."""

_OPERAND = r"(\([^()]*\)|[A-Za-z_][\w.]*(?:\([^()]*\))?)"
_COMPARISON = re.compile(_OPERAND + r"\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)")
_NAMED = re.compile(r"\b(\w+_(?:threshold|ceiling))\s*=\s*(-?\d+(?:\.\d+)?)")


def extract_thresholds(search: str) -> list[tuple[str, str, float]]:
    """
    Every threshold a search compares against, as the registry writes it.

    Parameters
    ----------
    search : str
        The search, with any line continuations already joined.

    Returns
    -------
    list of tuple
        `(threshold, operator, value)` in the order they first appear, each once. A named threshold -- an
        `eval` of a name ending in `_threshold` or `_ceiling` -- is `(name, "=", value)`; any other comparison of a
        field or expression with a number is `(operand, operator, value)`. Comparisons with zero are left out.
    """
    found = []

    for (name, value) in _NAMED.findall(search):
        found.append((name, "=", float(value)))

    for (operand, operator, value) in _COMPARISON.findall(search):
        if (float(value) != 0.0):
            found.append((operand, operator, float(value)))

    unique = []

    for item in found:
        if (item not in unique):
            unique.append(item)

    return unique


def thresholds_hash(thresholds: typing.Iterable[tuple[str, str, float]]) -> str:
    """
    The digest of one rule's thresholds, in the canonical form `config_hash` uses.

    Parameters
    ----------
    thresholds : iterable of tuple
        `(threshold, operator, value)`, as `extract_thresholds` returns them or the registry holds them. Order
        does not matter.

    Returns
    -------
    str
        Sixteen hex characters. A rule with no thresholds hashes a stated absence rather than an empty dictionary,
        so it has a digest of its own and gains a different one the day it gains a threshold.
    """
    settings = {f"{threshold} {operator}": value for (threshold, operator, value) in thresholds}

    return settings_hash(settings if settings else {"thresholds": "none"})


def derive_threshold(values: typing.Iterable[typing.Any], fires: str, quantile: float) -> typing.Optional[float]:
    """
    The threshold a baseline period proposes for one comparison.

    Parameters
    ----------
    values : iterable
        The field's values over the baseline period. Nulls and non-numeric values are dropped; a baseline is what
        the estate measured, and a gap is not a measurement.
    fires : str
        `above` for a rule that fires when the value exceeds its threshold, so the quantile is taken as given;
        `below` for one that fires under it, so the quantile is taken from the other end.
    quantile : float
        How much of the baseline the threshold should sit beyond, in (0, 1): 0.99 proposes the value 99 per cent
        of the baseline does not exceed.

    Returns
    -------
    float or None
        The value at that quantile by nearest rank, or None when the baseline holds no measurement at all.
    """
    if (fires not in DIRECTIONS):
        raise ValueError(f"unknown direction {fires!r}; expected one of {DIRECTIONS}")

    if (not 0.0 < quantile < 1.0):
        raise ValueError(f"quantile must lie strictly between 0 and 1, got {quantile}")

    measured = []

    for value in values:
        if (isinstance(value, bool)):
            continue

        try:
            number = float(value)
        except (TypeError, ValueError):
            continue

        if (number == number):
            measured.append(number)

    if (not measured):
        return None

    # Rounded, because 1 - 0.99 is 0.010000000000000009 in floating point, and nearest rank takes the ceiling: an
    # unrounded complement moves the proposal one rank up for any baseline whose size is a multiple of a hundred.
    return nearest_rank(measured, quantile if fires == "above" else round(1.0 - quantile, 12))
