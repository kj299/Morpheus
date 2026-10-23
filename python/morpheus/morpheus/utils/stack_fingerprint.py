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
Which TCP stack opened a connection, read from the options its SYN carried.

An operating system's TCP stack writes the options in its SYN in a fixed order of its own, and that order survives
everything the application above it does: a browser can claim to be anything in its `User-Agent`, but the kernel it
runs on writes the options the way it always does. R-B-L4-004 rests on that asymmetry, and the guide names the
options order as the stack fingerprint at layer 4.

**The fingerprint is the order, the maximum segment size and the initial window together**, rendered
`options|mss|window`, the three fields passive fingerprinting has used since p0f. Two hosts running one build of one
operating system on one network share it; a host that starts presenting a second one is running a second stack --
a virtual machine, a tunnel endpoint, a tool with a userspace stack -- which is what per-host novelty over it shows.

**The family is read from the order alone.** The segment size moves with the path's MTU and the window with tuning,
so a family keyed on all three would split one operating system across every link it sat behind. The three orders
recognized here are the stock ones of current Windows, Linux and the Darwin kernel under macOS and iOS; an order that
matches none of them is `None`, and a comparison with `None` is no comparison.

**Option names are normalized** to the short forms passive fingerprinting uses -- `mss`, `nop`, `ws`, `sok`, `ts`,
`eol` -- because capture tools spell them differently (`sackOK`, `SACK_PERM`, `wscale`, `timestamp`), and a
fingerprint that differed by spelling would be a new stack every time the collector changed.
"""

import typing

from morpheus.utils.entity_key import is_missing

OPTION_ALIASES = {
    "mss": "mss",
    "maxseg": "mss",
    "nop": "nop",
    "ws": "ws",
    "wscale": "ws",
    "window_scale": "ws",
    "sok": "sok",
    "sackok": "sok",
    "sack_perm": "sok",
    "sack_permitted": "sok",
    "ts": "ts",
    "timestamp": "ts",
    "timestamps": "ts",
    "eol": "eol",
}
"""Spellings capture tools use for each option, mapped to the short form."""

FAMILIES = {
    "mss,nop,ws,nop,nop,sok": "windows",
    "mss,sok,ts,nop,ws": "linux",
    "mss,nop,ws,nop,nop,ts,sok,eol": "darwin",
}
"""The stock SYN option orders of the stacks a workstation estate runs."""

SEPARATOR = "|"
"""Joins the three parts of a fingerprint. Not a comma, which already separates the options."""


def options_order(value: typing.Any) -> typing.Optional[str]:
    """
    A SYN's option order, normalized.

    Parameters
    ----------
    value : str
        The options as the capture recorded them, separated by commas, spaces or semicolons.

    Returns
    -------
    str or None
        The options in their original order, each in its short form, joined by commas. An option the alias table does
        not know is kept, lowercased, so an unfamiliar stack is still a fingerprint. `None` for no options.
    """
    if (is_missing(value)):
        return None

    parts = [part.strip().lower() for part in str(value).replace(";", ",").replace(" ", ",").split(",")]
    parts = [OPTION_ALIASES.get(part, part) for part in parts if len(part) > 0]

    return ",".join(parts) if len(parts) > 0 else None


def _integral(value: typing.Any) -> typing.Optional[int]:
    try:
        if (is_missing(value)):
            return None

        number = float(value)
    except (TypeError, ValueError):
        return None

    return int(number) if number == int(number) else None


def stack_fingerprint(options: typing.Any, mss: typing.Any, window: typing.Any) -> typing.Optional[str]:
    """
    The fingerprint a SYN's options, segment size and initial window make together.

    Parameters
    ----------
    options : str
        The SYN's options, in order.
    mss : int
        The maximum segment size it advertised.
    window : int
        Its initial receive window.

    Returns
    -------
    str or None
        `options|mss|window`, or `None` when the record carries no options -- which is every record but a SYN, so a
        flow's later packets say nothing about the stack rather than something empty.
    """
    order = options_order(options)

    if (order is None):
        return None

    rendered = [order] + ["" if part is None else str(part) for part in (_integral(mss), _integral(window))]

    return SEPARATOR.join(rendered)


def stack_family(options: typing.Any) -> typing.Optional[str]:
    """
    The operating system family whose stock stack writes these options in this order.

    Parameters
    ----------
    options : str
        The SYN's options, in order.

    Returns
    -------
    str or None
        `windows`, `linux` or `darwin`, or `None` for an order none of them writes.
    """
    return FAMILIES.get(options_order(options))
