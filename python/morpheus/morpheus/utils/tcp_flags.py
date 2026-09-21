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
The five TCP control bits the layer 4 features are built from, read out of the byte a capture reports.

`examples/abp_pcap_detection/abp_pcap_preprocessing.py` extracts these by rendering the flags byte as a binary
string and indexing into it, which is fast on a card and is the same arithmetic as a mask. This module is the
mask, so that a CPU pipeline and a GPU one agree bit for bit and so that the extraction is testable without a
frame library.

**Only five of the eight bits are read, and that is the example's choice rather than an oversight.** The byte
also carries URG, ECE and CWR. The thirteen features the shipped model was trained on use ACK, PSH, RST, SYN and
FIN, so those are what this returns; an estate wanting the other three is adding features the model has never
seen and should say so rather than discovering it in a score.
"""

import typing

FIN = 0x01
SYN = 0x02
RST = 0x04
PSH = 0x08
ACK = 0x10

FLAG_BITS = (("ack", ACK), ("psh", PSH), ("rst", RST), ("syn", SYN), ("fin", FIN))
"""Name and mask for each bit, in the order the model's feature list names them."""

FLAG_NAMES = tuple(name for (name, _) in FLAG_BITS)

MAX_FLAGS = 0xFF
"""The field is eight bits. A value outside it did not come off a wire."""


def flags_from_byte(value: typing.Any) -> typing.Optional[dict]:
    """
    Split a TCP flags byte into the five bits the layer 4 features read.

    Parameters
    ----------
    value : any
        The flags byte, as an integer or as something that converts to a whole one. `None`, a fractional value,
        or a value outside the field's range yields `None` rather than a guess: a flags byte of 300 is a parsing
        fault, and masking it would produce five plausible bits from a record that is wrong.

    Returns
    -------
    dict or None
        One entry per name in `FLAG_NAMES`, each 0 or 1.
    """
    if (value is None):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if (number != number or number != int(number)):  # pylint: disable=comparison-with-itself
        return None

    number = int(number)

    if (not 0 <= number <= MAX_FLAGS):
        return None

    return {name: int(bool(number & mask)) for (name, mask) in FLAG_BITS}
