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

When this file was written it named two: `lineage_id`, selected by all four detections and computed by no stage,
and `binding_table`, the field the L2/L3 refresh search filters on and which `to_bucketed_records` emitted only
when a caller remembered to ask for it. Both are now produced. A third, `join_method`, appeared when the linter
learned to read an aggregate's argument: Chain assembly collected it into `methods`, and it is the name
LineageStampStage gives a parent-child edge's method, which no pipeline stamps. The attribution method every scored
event does carry is BindingResolverStage's `resolution_method`, so the search now reads that and the registry is
empty again.

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


def _parenthesised(text: str) -> list:
    """The contents of each top-level parenthesised group, so `max(eval(if(a, b, c)))` yields its whole argument."""
    (groups, depth, start) = ([], 0, 0)

    for (position, character) in enumerate(text):
        if (character == "("):
            if (depth == 0):
                start = position + 1

            depth += 1
        elif (character == ")" and depth > 0):
            depth -= 1

            if (depth == 0):
                groups.append(text[start:position])

    return groups


def referenced_in(search: str) -> set:
    """Fields the search reads: filter terms, table and dedup lists, and where expressions."""
    referenced = set()

    # `field=value`, the shape of every filter term.
    referenced.update(re.findall(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*)\s*=", search))

    # `field>=20`, `field<0.2`, `field!=x`: a threshold is a filter term too. Missing these was a blind spot, and
    # the worst kind -- a threshold on a field nothing emits compares a null with a number and drops every row.
    referenced.update(re.findall(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*)\s*(?:[<>]=?|!=)", search))

    # An aggregate's argument is read as surely as a filter's: `max(flow_data_len) AS transferred` reads
    # flow_data_len, and naming a column nothing writes there yields a null aggregate rather than an error.
    for clause in re.findall(r"\|\s*(?:stats|eventstats|streamstats|tstats|chart|timechart)\s+([^|\]]*)", search):
        for argument in _parenthesised(clause):
            referenced.update(_identifiers(argument))

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
    # `streamstats current=f` excludes the current row from the running figures, which is how R-P-L3-005 reads
    # each window against the two before it rather than against itself. An argument to the command, not a field.
    "current",
    # `streamstats window=1` carries exactly the previous row forward, which is how R-C-005 reads the sign-in a
    # journey was measured from. An argument to the command, not a field.
    "window",
    # `join type=inner max=0` is how R-C-001 joins its three steps: an inner join keeping every match rather than
    # the first. Arguments to the command, not fields.
    "join",
    "type",
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

    for name in GOLDEN_NAMES:
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


STATS_COMMANDS = {"stats", "tstats", "chart", "timechart"}
"""Commands that replace the rows with their own output, keeping only the aggregates and the group-by fields."""

JOINING_COMMANDS = {"join", "append", "appendcols"}
"""Commands that bring a subsearch's fields into the pipeline."""

OPTION_COMMANDS = {"collect", "outputlookup"}
"""Commands whose `name=value` arguments are options, not fields: `collect index=behavior_summary` reads nothing."""


def _split_top_level(text: str) -> list:
    """Split a pipeline on `|`, leaving pipes inside a quoted string or a `[subsearch]` alone."""
    (parts, depth, quote, current) = ([], 0, None, [])

    for character in text:
        if (quote is not None):
            quote = None if character == quote else quote
        elif (character in "\"'"):
            quote = character
        elif (character == "["):
            depth += 1
        elif (character == "]"):
            depth -= 1
        elif (character == "|" and depth == 0):
            parts.append("".join(current))
            current = []
            continue

        current.append(character)

    parts.append("".join(current))

    return [part.strip() for part in parts]


def _subsearches(command: str) -> tuple:
    """(the command with its top-level subsearches removed, the text of each subsearch)."""
    (outside, inside, depth, current) = ([], [], 0, [])

    for character in command:
        if (character == "["):
            depth += 1

            if (depth == 1):
                continue
        elif (character == "]"):
            depth -= 1

            if (depth == 0):
                inside.append("".join(current).strip())
                current = []
                continue

        (current if depth > 0 else outside).append(character)

    return ("".join(outside), [re.sub(r"^search\s+", "", sub) for sub in inside])


def _stats_outputs(body: str) -> set:
    """The fields a `stats` leaves: its named aggregates, its group-by fields, and `count` if it counts unnamed."""
    head = re.split(r"\bby\b", body, maxsplit=1, flags=re.IGNORECASE)
    outputs = set(re.findall(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)", head[0], flags=re.IGNORECASE))

    if (len(head) > 1):
        outputs.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", head[1]))

    if (re.search(r"\bcount\b(?!\s*\()(?!\s+as\b)", head[0], flags=re.IGNORECASE)):
        outputs.add("count")

    return outputs


def walk_pipeline(search: str) -> tuple:
    """
    Follow which fields exist at each command, and report every field read after a `stats` removed it.

    `stats` replaces the rows with its own output. A field it neither aggregates into a name nor groups by is gone
    for the rest of the pipeline, and reading it afterwards does not fail: every row reads null, a `where` on it
    drops them all, and a `table` shows an empty column. Checking a search's fields against what the stages write
    cannot see this, because the field *is* written -- just not by anything still in the pipeline.

    Returns (fields read after they were dropped, the fields the pipeline ends with). The second is None while no
    `stats` has narrowed the pipeline, meaning every upstream field may still be present.
    """
    (missing, available) = (set(), None)

    for command in _split_top_level(search)[1:]:
        (body, subsearches) = _subsearches(command)
        words = body.split()
        name = words[0].lower() if words else ""
        piece = "| " + body
        sub_outputs = []

        for subsearch in subsearches:
            (sub_missing, sub_available) = walk_pipeline(subsearch)
            missing |= sub_missing
            sub_outputs.append(sub_available)

        if (available is not None and name not in OPTION_COMMANDS):
            missing |= referenced_in(piece) - created_in(piece) - available - SPL_WORDS

        if (name in STATS_COMMANDS):
            available = _stats_outputs(" ".join(words[1:]))
        elif (name in ("table", "fields") and words[1:2] != ["-"]):
            available = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", " ".join(words[1:])))
        elif (name in JOINING_COMMANDS):
            for sub_available in sub_outputs:
                if (available is not None):
                    available = None if sub_available is None else available | sub_available
        elif (available is not None):
            available |= created_in(piece)

    return (missing, available)


# --- The checks -----------------------------------------------------------------------------------------------


def test_the_app_is_where_we_think_it_is():
    # Without this every assertion below passes over an empty parse, which is the failure mode a linter must not
    # have: it would report a clean bill of health for a file it never read.
    assert len(searches()) == 41
    assert len(lookup_fields()) > 0
    assert len(stage_columns()) > 40


@pytest.mark.parametrize("name", sorted(searches()))
def test_every_field_a_search_reads_is_a_field_something_writes(name: str):
    search = searches()[name]
    unresolved = referenced_in(search) - created_in(search) - producible()

    assert not unresolved, (
        f"{name} reads {sorted(unresolved)}, which nothing in this repository writes. Either a stage should emit "
        f"it, or it belongs in KNOWN_UNPRODUCED with the reason it does not.")


GOLDEN_NAMES = ("golden_telemetry_expected.csv",
                "golden_lineage_expected.csv",
                "golden_network_expected.csv",
                "golden_transport_expected.csv",
                "golden_presentation_expected.csv",
                "golden_application_expected.csv",
                "golden_campaign_expected.csv",
                "golden_context_expected.csv",
                "golden_endpoint_expected.csv",
                "golden_saas_expected.csv",
                "golden_session_expected.csv",
                "golden_estate_expected.csv")


def layer_columns() -> dict:
    """Per OSI layer, the columns populated on at least one of that layer's events in the goldens."""
    import collections  # pylint: disable=import-outside-toplevel
    import csv  # pylint: disable=import-outside-toplevel

    found = collections.defaultdict(set)

    for name in GOLDEN_NAMES:
        path = os.path.join(REPO_ROOT, "tests", "morpheus", "determinism", name)

        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                layer = row.get("osi_layer")

                if (layer):
                    found[layer].update(column for (column, value) in row.items() if value != "")

    return found


def chain_steps(search: str) -> list:
    """(layer, text) for each step of a chained search: its base search, and each subsearch it joins."""
    steps = [search.split("| join", 1)[0]] + re.findall(r"\[search\s+([^\]]*)\]", search)
    found = []

    for step in steps:
        layer = re.search(r"sourcetype=morpheus:score:l(\d)", step)

        if (layer is not None):
            found.append((layer.group(1), step))

    return found


def unread_by_layer(search: str, columns: dict) -> dict:
    """Fields each step reads that no event of the step's layer carries, by layer."""
    missing = {}

    for (layer, step) in chain_steps(search):
        fields = referenced_in(step) - created_in(step) - SPLUNK_INTRINSICS - SPL_WORDS
        unresolved = fields - columns.get(layer, set())

        if (len(unresolved) > 0):
            missing[layer] = sorted(unresolved)

    return missing


JOINED_IN_THE_PIPELINE = {
    "R-C-005 - Credential replay across the stack":
        "layer 5 sign-ins resolved to layer 2 ports and sites by BindingResolverStage through the leases and MAC "
        "bindings, so the search reads one layer's events that already carry the other's answer",
}
"""Chained rules whose join is made before the events are written, so their search reads a single layer."""


def test_a_chained_rule_reads_each_step_from_a_layer_that_writes_it():
    """
    The defect this test was written for, which the check above could not see.

    R-C-002 once correlated two detections' notables and grouped them `by src_ip dest_ip`. `dest_ip` is the Splunk
    CIM name and upstream Morpheus's default column; every stage in this fork emits `dst_ip`. Grouping on it put a
    null in one of two group keys on every notable, which does not fail and does not return nothing -- it collapses
    every fingerprint and every beacon in the window into one group and reports any of them against any other.

    `test_every_field_a_search_reads_is_a_field_something_writes` passed throughout, because `dest_ip` *is*
    produced: by the lineage pipeline, which is not the layer whose events R-C-002 reads. "Something, somewhere,
    writes this" is the wrong question for a chained rule. The chained rules now read scored events rather than
    notables, so the right question is whether the layer each step reads puts the field on its events -- which is
    what this asserts, from the goldens, for the base search and every subsearch a chain joins.
    """
    columns = layer_columns()
    chains = {name: search for (name, search) in searches().items() if name.startswith("R-C-")}

    assert len(chains) >= 3, "fewer chained rules than this fork ships; this test has stopped covering them"

    for (name, search) in chains.items():
        steps = 1 if name in JOINED_IN_THE_PIPELINE else 2
        assert len(chain_steps(search)) >= steps, f"{name} has fewer than {steps} steps this test can read"
        assert not unread_by_layer(search, columns), (
            f"{name} reads fields no event of that layer carries: {unread_by_layer(search, columns)}. A chain "
            f"joining on a field its step does not write does not fail -- it joins nulls and correlates the wrong "
            f"pairs.")


def test_a_join_on_a_field_its_layer_does_not_write_would_be_caught():
    # The negative control, with the defect that started this: a layer 6 step joined on `dest_ip`.
    search = ("index=behavior_events sourcetype=morpheus:score:l3 flow_regularity_mature=true "
              "| join type=inner src_ip dest_ip [search index=behavior_events sourcetype=morpheus:score:l6 "
              "ja4_client_first_seen=true | fields src_ip dest_ip]")

    assert unread_by_layer(search, layer_columns()) == {"6": ["dest_ip"]}


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


@pytest.mark.parametrize("name", sorted(searches()))
def test_no_search_reads_a_field_its_own_stats_dropped(name: str):
    (dropped, _) = walk_pipeline(searches()[name])

    assert not dropped, (
        f"{name} reads {sorted(dropped)} after a stats that neither aggregates nor groups by it. Every row reads "
        f"null there; aggregate it, group by it, or read it before the stats.")


def test_a_threshold_on_a_field_nothing_writes_would_be_caught():
    invented = "a_field_no_stage_will_ever_emit"
    # A bare filter term, with no `where` or `table` to catch it another way.
    search = f"index=behavior_events sourcetype=morpheus:score:l3 {invented}>=20 | stats count BY src_ip"

    assert invented in referenced_in(search) - created_in(search) - producible()


def test_an_aggregate_of_a_field_nothing_writes_would_be_caught():
    invented = "a_field_no_stage_will_ever_emit"
    search = (f"index=behavior_events sourcetype=morpheus:score:l3 "
              f"| stats max({invented}) AS peak min(eval(if(dsts_per_src > 3, _time, null()))) AS first BY src_ip")

    assert invented in referenced_in(search) - created_in(search) - producible()


def test_a_field_read_after_the_stats_that_dropped_it_would_be_caught():
    # dst_ip exists before the stats, which keeps only src_ip and the two named aggregates; reading it afterwards
    # reads null. peak and eval'd fields remain readable, and a joined subsearch's fields join the pipeline.
    search = ("index=behavior_events sourcetype=morpheus:score:l3 | stats max(dsts_per_src) AS peak count BY src_ip "
              "| eval doubled = peak * 2 | join type=inner src_ip [search index=behavior_events | stats "
              "min(_time) AS t_login BY src_ip] | where doubled > count AND t_login > 0 AND dst_ip != \"10.0.0.1\"")

    assert walk_pipeline(search) == ({"dst_ip"}, {"src_ip", "peak", "count", "doubled", "t_login"})


def test_a_bogus_field_would_be_caught():
    # The linter's own negative control. A parser that resolves everything reports a clean bill of health for a
    # search naming a field that does not exist, which is exactly the failure it exists to prevent.
    invented = "a_field_no_stage_will_ever_emit"
    search = f"index=behavior_events sourcetype=morpheus:score:l2 {invented}=true | table _time {invented}"

    assert invented in referenced_in(search) - created_in(search) - producible()


# --- What each search reads from its sourcetype, against that sourcetype's wire contract -------------------------
#
# The checks above ask whether *something* writes a field. A deployment's guarantee is narrower: `SiemWireStage`
# refuses a frame missing a column its sourcetype's contract requires, and accepts one missing anything else. A
# search that filters on a column outside the contract therefore works against the corpus, where every stage ran,
# and silently empties against a producer that dropped the column, which the contract said it could. "Binding
# health" read `resolution_method` on `morpheus:score:l3` for weeks, a sourcetype whose producer never writes it,
# and nothing here noticed because *a* producer somewhere did.

SOURCETYPE_FILTER = re.compile(r"sourcetype\s*=\s*\"?([A-Za-z0-9:_*]+)\"?")
NARROWING_COMMANDS = STATS_COMMANDS | JOINING_COMMANDS
PROJECTING_COMMANDS = {"table", "fields"}


def contract_columns(sourcetype: str) -> set:
    """Every column the wire contract guarantees for a sourcetype, or for every sourcetype a wildcard names."""
    from morpheus.utils.siem_sourcetypes import PRODUCED  # pylint: disable=import-outside-toplevel

    pattern = re.compile("^" + re.escape(sourcetype).replace("\\*", ".*") + "$")
    columns = set()

    for (name, contract) in PRODUCED.items():
        if (pattern.match(name)):
            columns.update(contract.required_columns)

            for variant in (contract.variant_columns or ()):
                columns.update(variant)

    return columns


def _read_by_command(name: str, body: str, created: set) -> set:
    """The fields one command reads from the rows it receives, by command."""
    words = body.split()

    if (name == "rename"):
        return set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s+AS\s+", body, flags=re.IGNORECASE)) - created

    if (name == "sort"):
        return {word.lstrip("-+") for word in words[1:] if not word.lstrip("-+").isdigit()} - created

    if (name in JOINING_COMMANDS):
        # `join type=inner a b [subsearch]`: the join fields are read from the rows; the options are not fields.
        return {word for word in words[1:] if "=" not in word and re.fullmatch(r"[A-Za-z_]\w*", word)} - created

    return referenced_in("| " + body) - created_in("| " + body) - created


def reads_by_sourcetype(search: str) -> dict:
    """
    The fields each base search reads from its sourcetype, before anything narrows or widens the rows.

    Attribution stops at the first `stats` (its arguments and group-by fields are read from the source rows and
    counted; what follows reads its output) and at the first `join` (the rows then carry the subsearch's fields
    too). Projections (`table`, `fields`) are not reads: a missing column there shows empty rather than dropping
    rows. Every subsearch is walked the same way, so a chained rule's three base searches are each attributed to
    their own sourcetype.
    """
    reads = {}

    def visit(pipeline: str):
        segments = _split_top_level(pipeline)
        base = re.sub(r"^search\s+", "", segments[0])
        match = SOURCETYPE_FILTER.search(base)
        (created, collected) = (set(), set())
        attributing = match is not None

        if (attributing):
            collected |= referenced_in("| search " + base)

        for segment in segments[1:]:
            (body, subsearches) = _subsearches(segment)

            for subsearch in subsearches:
                visit(subsearch)

            words = body.split()
            name = words[0].lower() if words else ""

            if (not attributing or name in PROJECTING_COMMANDS or name in OPTION_COMMANDS):
                continue

            collected |= _read_by_command(name, body, created)
            created |= created_in("| " + body)

            if (name in NARROWING_COMMANDS):
                attributing = False

        if (match is not None):
            reads.setdefault(match.group(1), set()).update(collected - SPL_WORDS - SPLUNK_INTRINSICS - {"_time"})

    visit(search)

    return reads


KNOWN_HOLLOW_READS = {
    ("Behavior summary - per-layer scores", "morpheus:score:l*"): {
        "risk_score": "written by no stage and collected by no detection; the risk write path is issue #57",
        "rule_id": "written by no stage and collected by no detection; the risk write path is issue #57",
    },
    ("Chain assembly - cross-layer risk", "morpheus:edge"): {
        "lineage_id": "the edge stream carries no lineage_id; the edge producer is issue #63",
        "osi_layer": "the edge stream carries no osi_layer; the edge producer is issue #63",
        "max_abs_z": "a score, read off edges that carry none; the edge producer is issue #63",
        "resolution_method": "carried by the lineage corpus's edges and promised by no contract; issue #63",
        "risk_score": "written by no stage and collected by no detection; the risk write path is issue #57",
        "rule_id": "written by no stage and collected by no detection; the risk write path is issue #57",
    },
}
"""
Reads this linter knows are hollow and the retrospective tracks: a search reading a field its sourcetype's contract
cannot promise because nothing produces it. Each is listed with the issue that closes it, so the check below stays
red on anything new while these stay visible rather than silently excused. An entry whose field the contract has
since gained, or whose search no longer reads it, fails `test_the_registry_of_hollow_reads_is_honest`.
"""


@pytest.mark.parametrize("name", sorted(searches()))
def test_every_field_a_search_reads_from_its_sourcetype_is_in_that_sourcetypes_contract(name: str):
    for (sourcetype, fields) in reads_by_sourcetype(searches()[name]).items():
        contract = contract_columns(sourcetype)

        assert contract, f"{name} reads sourcetype={sourcetype}, which no wire contract describes"

        missing = sorted(fields - contract - set(KNOWN_HOLLOW_READS.get((name, sourcetype), {})))

        assert not missing, (f"{name} reads {missing} from {sourcetype}, which that sourcetype's contract does not "
                             f"guarantee: SiemWireStage would accept a frame without them and the search would "
                             f"silently return nothing. Add them to required_columns or a variant in "
                             f"morpheus.utils.siem_sourcetypes, or stop reading them.")


def test_a_read_outside_the_contract_would_be_caught():
    # The negative control for the check above: a search filtering on a field its sourcetype never guarantees.
    search = ("index=behavior_events sourcetype=morpheus:score:l3 resolution_method=unresolved "
              "| stats count AS total BY src_ip | where total > 0")
    reads = reads_by_sourcetype(search)

    assert "resolution_method" in reads["morpheus:score:l3"]
    assert "resolution_method" not in contract_columns("morpheus:score:l3")
    assert "total" not in reads["morpheus:score:l3"], "a field the search itself creates is not a read"


def test_a_chained_rule_attributes_each_base_search_to_its_own_sourcetype():
    reads = reads_by_sourcetype(searches()["R-C-001 - Lateral movement chain"])

    assert {"dsts_per_src", "src_ip", "window_id"} <= reads["morpheus:score:l3"]
    assert {"auth_result", "target_host_first_seen", "source_ip"} <= reads["morpheus:score:l5"]
    assert {"endpoint_pair_novel", "hostname"} <= reads["morpheus:score:l7"]
    assert "previous_peak" not in reads["morpheus:score:l3"], "a stats output is read from the stats, not the source"


def test_a_wildcard_sourcetype_is_checked_against_every_sourcetype_it_names():
    columns = contract_columns("morpheus:score:l*")

    assert {"dsts_per_src", "travel_status", "ja4_client"} <= columns
    assert "bind_gap_ns" not in columns


def test_the_registry_of_hollow_reads_is_honest():
    # An allowlist that outlives its reason is the same defect it was written to make visible.
    for ((name, sourcetype), fields) in KNOWN_HOLLOW_READS.items():
        reads = reads_by_sourcetype(searches()[name]).get(sourcetype, set())
        contract = contract_columns(sourcetype)

        for (field, reason) in fields.items():
            assert field in reads, f"{name} no longer reads {field} from {sourcetype}; drop the entry ({reason})"
            assert field not in contract, f"{sourcetype} now guarantees {field}; drop the entry ({reason})"
            assert "issue #" in reason, f"{name}/{field}: every hollow read names the issue that closes it"
