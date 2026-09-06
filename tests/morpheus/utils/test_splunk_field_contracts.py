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
"""
Every field the shipped searches read must be a field something writes.

A search that names a field nothing emits does not fail. It returns no rows, or returns rows with a blank column,
and a detection that fires on nothing looks exactly like a detection with nothing to find. Three defects in this
app were that shape, and one of them -- `lineage_id`, selected by all four detections and populated by no stage --
survived every other check in the repository because nothing here compared the two artifacts against each other.

There is no search head in this repository and there will not be one in CI, so this is the largest share of
search-head risk the repo can carry on its own: parse the searches, resolve each field they reference against
what the stages declare, what the lookups define, and what the search itself creates, and require anything left
over to be listed as knowingly unproduced with a reason.

The registry starts non-empty, which is the point. Naming the void is what makes it shrink deliberately rather
than by accident.
"""

import configparser
import os
import re

import pytest

from morpheus.utils import siem_sourcetypes

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
APP = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "TA-morpheus-lineage", "default")
SAVEDSEARCHES = os.path.join(APP, "savedsearches.conf")
TRANSFORMS = os.path.join(APP, "transforms.conf")
PROPS = os.path.join(APP, "props.conf")

# Splunk's own, plus the ones every search may lean on.
SPLUNK_INTRINSICS = {"_time", "_key", "_raw", "_indextime", "index", "sourcetype", "source", "host", "count"}

KNOWN_UNPRODUCED: dict = {}
"""Fields the searches read that nothing in this repository writes, each with the reason it is still referenced.

Empty, which was not true when this file was written. It named two: `lineage_id`, selected by all four detections
and computed by no stage, and `binding_table`, the field the L2/L3 refresh search filters on and which
`to_bucketed_records` emitted only when a caller remembered to ask for it. Both are now produced, so the honest
registry is empty rather than decorative.

An entry here is a claim that a gap is known and deliberate, and the tests below make it an uncomfortable one: it
must be read by some search, it must carry a real reason, and it must stop being registered the moment something
produces it."""


def _load(path: str) -> configparser.ConfigParser:
    """
    Read a Splunk `.conf`, folding its line continuations first.

    Splunk continues a value with a trailing backslash; configparser continues one with indentation, and reading
    a search written the Splunk way as though it were written the Python way is a parse error rather than a
    misreading -- which is the better failure, but only if it is handled here rather than at the call site.
    """
    with open(path, encoding="utf-8") as handle:
        folded = re.sub(r"\\\s*\r?\n\s*", " ", handle.read())

    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read_string(folded)

    return parser


def searches() -> dict:
    """Every scheduled search in the app, by name."""
    parsed = _load(SAVEDSEARCHES)

    return {name: parsed[name]["search"] for name in parsed.sections() if parsed.has_option(name, "search")}


def stage_columns() -> set:
    """Every column the fork's stages declare they write, read from the stages themselves."""
    from morpheus.config import Config
    from morpheus.config import CppConfig
    from morpheus.config import ExecutionMode

    CppConfig.set_should_use_cpp(False)
    config = Config()
    config.execution_mode = ExecutionMode.CPU

    # Imported here so this module stays importable without a Morpheus runtime for the parsing tests above.
    from morpheus.stages.lineage.binding_resolver_stage import BindingResolverStage
    from morpheus.stages.lineage.community_id_stage import CommunityIdStage
    from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
    from morpheus.stages.lineage.window_seal_stage import WindowSealStage
    from morpheus.stages.telemetry.tc1_change_stage import TC1ChangeStage
    from morpheus.stages.telemetry.tc1_flap_stage import TC1FlapStage
    from morpheus.stages.telemetry.tc1_normalize_stage import TC1NormalizeStage
    from morpheus.stages.telemetry.tc1_optical_stage import TC1OpticalStage
    from morpheus.stages.telemetry.tc2_arp_stage import TC2ArpStage
    from morpheus.stages.telemetry.tc2_auth_stage import TC2AuthStage
    from morpheus.stages.telemetry.tc2_binding_stage import TC2BindingStage
    from morpheus.stages.telemetry.tc2_cardinality_stage import TC2CardinalityStage
    from morpheus.stages.telemetry.tc5_cadence_stage import TC5CadenceStage
    from morpheus.stages.telemetry.tc5_novelty_stage import TC5NoveltyStage
    from morpheus.stages.telemetry.tc5_risk_stage import TC5RiskStage
    from morpheus.stages.telemetry.tc5_session_stage import TC5SessionStage
    from morpheus.stages.telemetry.tc5_travel_stage import TC5TravelStage
    from morpheus.utils.binding_table import Binding
    from morpheus.utils.binding_table import BindingTable

    table = BindingTable(name="t",
                         value_columns=["port_key", "vlan_id"],
                         bindings=[Binding(key="k", start_ns=0, end_ns=1, values=("p", "v"), uid="u")])
    built = [
        TC1NormalizeStage(config),
        TC1OpticalStage(config),
        TC1FlapStage(config),
        TC1ChangeStage(config),
        TC2CardinalityStage(config),
        TC2ArpStage(config),
        TC2AuthStage(config),
        TC2BindingStage(config),
        TC5SessionStage(config),
        TC5NoveltyStage(config),
        TC5CadenceStage(config),
        TC5TravelStage(config),
        TC5RiskStage(config),
        CommunityIdStage(config),
        LineageStampStage(config, id_columns=["collector_id"]),
        BindingResolverStage(config, binding_table=table, key_column="k", uid_column="binding_uid"),
        WindowSealStage(config, period_seconds=300, lateness_seconds=30, entity_key_column="entity_key"),
    ]
    columns = set()

    for stage in built:
        columns.update(stage.get_needed_columns())

    return columns


def lookup_fields() -> set:
    """Every field the KV Store lookups define, from transforms.conf's own `fields_list`."""
    parsed = _load(TRANSFORMS)
    fields = set()

    for name in parsed.sections():
        if (parsed.has_option(name, "fields_list")):
            fields.update(part.strip() for part in parsed[name]["fields_list"].split(","))

    return {field for field in fields if field}


def created_in(search: str) -> set:
    """Fields the search itself brings into existence, so they need no producer upstream."""
    created = set()

    for clause in re.findall(r"\|\s*eval\s+(.*?)(?=\||$)", search):
        created.update(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", clause))

    # `stats max(x) AS peak` and `values(y) as ys` -- Splunk accepts either case.
    created.update(re.findall(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)", search, flags=re.IGNORECASE))

    # `lookup <table> <match fields> OUTPUT a b` defines a and b for the rest of the pipeline.
    for clause in re.findall(r"\bOUTPUT(?:NEW)?\s+([^|]*)", search, flags=re.IGNORECASE):
        created.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", clause))

    return created


def _identifiers(expression: str) -> set:
    """Bare identifiers in an expression, with string literals removed so their contents are not read as fields."""
    without_strings = re.sub(r"\"[^\"]*\"|'[^']*'", " ", expression)

    # A function call is not a field reference; `floor(` and `now(` name operations, not columns.
    without_calls = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", " ", without_strings)

    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", without_calls))


def referenced_in(search: str) -> set:
    """Fields the search reads: filter terms, table and dedup lists, and where expressions."""
    referenced = set()

    # `field=value`, the shape of every filter term.
    referenced.update(re.findall(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*)\s*=", search))

    for clause in re.findall(r"\|\s*(?:table|dedup|fields)\s+([^|]*)", search):
        referenced.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", clause))

    for clause in re.findall(r"\|\s*where\s+([^|]*)", search):
        referenced.update(_identifiers(clause))

    # An eval's right-hand side reads fields too, and missing it was this linter's own blind spot: a search could
    # compute a value from a field nothing emits and every check here would still pass.
    for clause in re.findall(r"\|\s*eval\s+([^|]*)", search):
        assigned = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", clause))
        referenced.update(_identifiers(re.sub(r"[A-Za-z_][A-Za-z0-9_]*\s*=", " ", clause)) - assigned)

    for clause in re.findall(r"\bby\s+([^|]*)", search):
        referenced.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", clause))

    return referenced


# SPL keywords and operators the regexes above cannot tell from field names.
SPL_WORDS = {
    "search",
    "eval",
    "where",
    "table",
    "dedup",
    "stats",
    "fields",
    "lookup",
    "outputlookup",
    "inputlookup",
    "by",
    "as",
    "OUTPUT",
    "output",
    "OUTPUTNEW",
    "append",
    "AND",
    "OR",
    "NOT",
    "and",
    "or",
    "not",
    "in",
    "if",
    "case",
    "null",
    "true",
    "false",
    "values",
    "count",
    "dc",
    "min",
    "max",
    "sum",
    "avg",
    "latest",
    "earliest",
    "eventstats",
    "streamstats",
    "sort",
    "head",
    "rename",
    "collect",
    "tstats",
    "makeresults",
    "coalesce",
    "mvindex",
    "split",
    "tonumber",
    "tostring",
    "round",
    "now",
    "relative_time",
    "strftime",
    "strptime",
    "like",
    "match",
    "isnotnull",
    "isnull",
    "nullif",
    "spath",
    "mvexpand",
    "bin",
    "span",
    "key_field",
    "kv_store",
    "local",
    "prefix",
    "limit",
}


def golden_columns() -> set:
    """
    Every column the composed pipelines actually emit, read from the checked-in goldens.

    The stages' `needed_columns` say what each stage *adds*; a search also reads what the collector supplied and
    the pipeline carried through, and the goldens are the only artifact in the repository that records both. They
    are regenerated deliberately, so this stays honest as the corpus grows.
    """
    columns = set()

    for name in ("golden_telemetry_expected.csv", "golden_lineage_expected.csv", "golden_session_expected.csv"):
        path = os.path.join(REPO_ROOT, "tests", "morpheus", "determinism", name)

        with open(path, encoding="utf-8") as handle:
            columns.update(handle.readline().strip().split(","))

    return columns


def bucketed_columns() -> set:
    """The keys a bucketed binding row carries, from the producer rather than from a copy of its documentation."""
    from morpheus.utils.binding_table import Binding
    from morpheus.utils.binding_table import BindingTable

    table = BindingTable(name="dhcp_lease",
                         value_columns=["port_key"],
                         bindings=[Binding(key="10.0.0.1", start_ns=0, end_ns=10**11, values=("p", ), uid="u")])

    return set(table.to_bucketed_records()[0])


def notable_fields() -> set:
    """
    Fields one search writes for another to read.

    The detections `| eval` a rule identifier, a risk score and a layer onto every notable; the summary and chain
    searches then read those from the notable index. That is a real contract between two artifacts in this app,
    so it resolves -- but only to something a search in this same app actually creates.
    """
    created = set()

    for search in searches().values():
        created.update(created_in(search))

    return created


def producible() -> set:
    """Everything a field reference can legitimately resolve to."""
    return (stage_columns() | golden_columns() | bucketed_columns() | lookup_fields() | notable_fields()
            | SPLUNK_INTRINSICS | set(KNOWN_UNPRODUCED) | SPL_WORDS)


# --- The checks -----------------------------------------------------------------------------------------------


def test_the_app_is_where_we_think_it_is():
    # Without this every assertion below passes over an empty parse, which is the failure mode a linter must not
    # have: it would report a clean bill of health for a file it never read.
    assert len(searches()) == 13
    assert len(lookup_fields()) > 0
    assert len(stage_columns()) > 40


@pytest.mark.parametrize("name", sorted(searches()))
def test_every_field_a_search_reads_is_a_field_something_writes(name: str):
    search = searches()[name]
    unresolved = referenced_in(search) - created_in(search) - producible()

    assert not unresolved, (
        f"{name} reads {sorted(unresolved)}, which nothing in this repository writes. Either a stage should emit "
        f"it, or it belongs in KNOWN_UNPRODUCED with the reason it does not.")


def test_the_field_every_detection_selects_is_populated():
    # The defect this file was written for. `lineage_id` appeared in all four detections' table clauses while
    # `utils.lineage.lineage_id` had no caller anywhere in the tree, so the column was always blank.
    selecting = {name for (name, search) in searches().items() if "lineage_id" in search}

    assert len(selecting) >= 4, "the detections stopped selecting lineage_id"
    assert "lineage_id" in stage_columns(), "lineage_id is selected by the detections and written by no stage"


def test_the_registry_of_unproduced_fields_is_honest():
    # Every entry must still be referenced by some search. An entry nobody reads is a stale excuse, and leaving it
    # would let a field quietly lose its producer without anything noticing.
    read_anywhere = set()

    for search in searches().values():
        read_anywhere.update(referenced_in(search))

    for (field, reason) in KNOWN_UNPRODUCED.items():
        assert field in read_anywhere, f"{field} is registered as unproduced but no search reads it"
        assert len(reason) > 40, f"{field} needs a reason, not a placeholder"
        assert field not in stage_columns(), f"{field} is registered as unproduced but a stage now writes it"


@pytest.mark.parametrize("stanza", sorted(siem_sourcetypes.stanza_names()))
def test_every_time_prefix_field_is_named_by_a_producer(stanza: str):
    # The other direction of the same contract: props.conf anchors _time on a field, and something has to write
    # it. For an unproduced sourcetype the answer is that nothing does, which siem_sourcetypes already records.
    entry = siem_sourcetypes.describe(stanza)
    anchor = entry.time_column

    if (isinstance(entry, siem_sourcetypes.Unproduced)):
        pytest.skip(f"{stanza} has no producer: {entry.missing}")

    assert anchor in producible() - SPL_WORDS - SPLUNK_INTRINSICS, (
        f"{stanza} anchors _time on {anchor!r} and no stage writes it; every event on this sourcetype would be "
        f"stamped at index time.")


def test_a_bogus_field_would_be_caught():
    # The linter's own negative control. A parser that resolves everything reports a clean bill of health for a
    # search naming a field that does not exist, which is exactly the failure it exists to prevent.
    invented = "a_field_no_stage_will_ever_emit"
    search = f"index=behavior_events sourcetype=morpheus:score:l2 {invented}=true | table _time {invented}"

    assert invented in referenced_in(search) - created_in(search) - producible()
