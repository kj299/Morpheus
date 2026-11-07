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
The one reader of a Splunk app's `.conf` files (guide 13 plan item 6).

The stamping that folds each rule's thresholds into the fingerprint, the clock-skew experiment that mirrors the
rules, the search-head runner and every test of the app read `savedsearches.conf`, `props.conf` and the rest. Each
used to parse them itself, and they disagreed: some folded Splunk's backslash continuations and some did not, which
for `savedsearches.conf` is the difference between a search and its first line. This module is the rule they share.

A Splunk `.conf` is close enough to an INI file that `configparser` reads it once two things are settled. Splunk
continues a value with a trailing backslash, and `configparser` continues one with indentation, so continuations
are folded into one line first. And a value holds `%` and `$` freely, so interpolation is off. Duplicate stanzas or
keys are refused, as `btool` would report them.
"""

import configparser
import re

CONTINUATION = re.compile(r"\\\s*\r?\n\s*")
"""A trailing backslash and the line break after it: Splunk's continuation, folded to one space."""


def fold(text: str) -> str:
    """`text` with every backslash continuation joined onto the line it continues."""
    return CONTINUATION.sub(" ", text)


def read(path: str, preserve_case: bool = False) -> configparser.ConfigParser:
    """
    One `.conf` file, its continuations folded.

    Parameters
    ----------
    path : str
        The file.
    preserve_case : bool, default = False
        Keep each key as written (`enableSched`, `MAX_DAYS_AGO`), for a reader that lists or compares the keys
        themselves. Otherwise keys are lowercased, as `configparser` does, and looked up without regard to case.

    Returns
    -------
    `configparser.ConfigParser`

    Raises
    ------
    configparser.DuplicateSectionError, configparser.DuplicateOptionError
        If a stanza, or a key within one, is stated twice.
    """
    with open(path, encoding="utf-8") as handle:
        text = fold(handle.read())

    parser = configparser.ConfigParser(interpolation=None)

    if (preserve_case):
        parser.optionxform = str

    parser.read_string(text, source=path)

    return parser


def stanzas(path: str) -> list[str]:
    """Every stanza's name, in the order the file states them."""
    return read(path).sections()


def searches(path: str) -> dict[str, str]:
    """Each stanza of a `savedsearches.conf` that states a search, to its SPL on one line."""
    parser = read(path)

    return {name: parser[name]["search"] for name in parser.sections() if parser.has_option(name, "search")}


def search_text(path: str, name: str) -> str:
    """
    One saved search's SPL on one line.

    Raises
    ------
    KeyError
        If the file has no such stanza, or the stanza states no search.
    """
    return searches(path)[name]
