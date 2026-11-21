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
"""
What a shipped saved search states, read from its stanza: one reader for every harness.

A harness evaluates a rule over its corpus to say what the rule would find, and to do that it needs the rule's
numbers: the ratio a scan has to reach, the floor beneath a step, the severity a privileged account carries. Written
into the test as constants, those numbers are a second copy of the rule, and a stanza edit leaves every test green
while the expectations go stale. Read from `savedsearches.conf`, the predicate a test evaluates is the app's.

`morpheus.compositions.stamping.rule_thresholds` reads a rule's numbers for the fingerprint an event carries and
names them itself; these read one number a test names with a pattern, and fail if the search no longer states it.
"""

import os
import re

from morpheus.utils import splunk_conf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

SAVEDSEARCHES = os.path.join(REPO_ROOT,
                             "examples",
                             "splunk_lineage_app",
                             "TA-morpheus-lineage",
                             "default",
                             "savedsearches.conf")
"""The app's saved searches."""


def search(name: str) -> str:
    """One saved search's SPL, with its continuation lines folded."""
    return splunk_conf.search_text(SAVEDSEARCHES, name)


def threshold(stanza: str, pattern: str) -> float:
    """
    A threshold as the shipped search states it, so the predicate a test evaluates is the app's and not a copy.

    `pattern` names the eval variable or the comparison, with one group around the number.
    """
    match = re.search(pattern, search(stanza))

    assert match is not None, f"{stanza} no longer states {pattern!r}"

    return float(match.group(1))


def compared(stanza: str, field: str, operator: str) -> float:
    """The number a shipped search compares `field` against with `operator`, as in `dsts_per_src>50`."""
    return threshold(stanza, r"(?<![\w.])" + re.escape(field) + r"\s*" + re.escape(operator) + r"\s*([\d.]+)")


def _plain(number: str):
    """A number as the search writes it: `70` an integer, `1.5` a float."""
    return float(number) if "." in number else int(number)


def case_weights(stanza: str, variable: str) -> tuple:
    """
    The `eval <variable> = case(<field> == "<value>", <number>, ..., true(), <default>)` a shipped search states.

    Returns
    -------
    tuple
        `({value: number}, default)`, so a severity map a test scores with is the app's and not a copy.
    """
    match = re.search(r"eval\s+" + re.escape(variable) + r"\s*=\s*case\((.*?),\s*true\(\),\s*([\d.]+)\)",
                      search(stanza))

    assert match is not None, f"{stanza} no longer derives {variable} with a case() that ends in a default"

    arms = re.findall(r'==\s*"([^"]+)",\s*([\d.]+)', match.group(1))

    assert len(arms) > 0, f"{stanza}: the case() for {variable} has no arm this reader understands"

    return ({value: _plain(number) for (value, number) in arms}, _plain(match.group(2)))
