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
"""
A count a document spells out, read back as the number it names (guide 13 plan item 7).

Tests that hold a README or a guide to the count it states read that count as words. Each kept its own reader, and
six of the eight were tables of the words the documents had used so far -- a list kept by hand beside the thing it
lists, which failed the first time a count reached a number nobody had written into it. This parses instead.
"""

import re

UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def from_words(text: str) -> int:
    """
    The integer an English count below a hundred names: "none", "eleven", "Fifty", "eighty-two".

    Raises
    ------
    ValueError
        If `text` is not such a word. A word this cannot read is a sentence that has been rewritten, and quietly
        comparing against zero would turn that into a passing test.
    """
    word = text.strip().lower()

    if (word == "none"):
        return 0

    if (word in UNITS):
        return UNITS.index(word)

    match = re.fullmatch(r"([a-z]+)(?:-([a-z]+))?", word)

    if (match is not None and match.group(1) in TENS):
        (tens, unit) = match.groups()

        if (unit is None):
            return TENS[tens]

        if (unit in UNITS[1:10]):
            return TENS[tens] + UNITS.index(unit)

    raise ValueError(f"cannot read {text!r} as a number")
