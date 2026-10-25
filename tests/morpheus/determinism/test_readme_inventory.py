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
The sentence that counts what this fork has built, in both of the documents that state it.

"N stages and M supporting modules, covered by K distinct tests" is the first concrete claim a reader meets, and
it is made twice: once in the top-level README and once in the guide's summary of what is verified versus
designed. All three numbers were maintained by hand, in both places, and all three went stale. The stage count
was right when it was written and then sat through two commits that added a stage each. The module count sat
through the one that added `dfencoder_scorer`. The test count sat through fourteen. A number a reader can check
and find wrong is worse than no number, because checking it is the cheapest test there is of whether the rest of
the document is maintained.

So the numbers are computed here from the repository and compared against each sentence. What each one counts:

- **Stages**: files named `*_stage.py` under `python/morpheus/morpheus/stages/` that this fork wrote.
- **Supporting modules**: modules under `python/morpheus/morpheus/utils/` that this fork wrote.
- **Distinct tests**: `def test_` definitions across the files the GPU conformance runner names, which is this
  fork's own test suite. Definitions rather than collected items, deliberately, and the word "distinct" in the
  sentence says so. A test parametrized over the execution modes is one test that runs in two places, and the
  old figure counted it twice -- which is most of the difference between the number this replaces and the
  smaller one that replaces it. Definitions are also the count that can be taken without a collection, and a
  document check that had to run pytest in a subprocess to answer would be the slowest test in the suite.

**Authorship is decided by the shape of the copyright line, not by the year in it.** Every file this fork added
carries a single year; every upstream file carries a range ending in the year upstream last touched it. A rule
written against the literal year would need somebody to remember to come back and widen it, which is the habit
this whole file exists because nobody has.
"""

import os
import re

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DOCUMENTS = {
    "README":
        os.path.join(REPO_ROOT, "README.md"),
    "guide":
        os.path.join(REPO_ROOT,
                     "docs",
                     "source",
                     "developer_guide",
                     "guides",
                     "11_predictive_behavioral_analytics_osi.md"),
}
"""Both places the claim is made. They are checked against the repository rather than against each other, so
two documents agreeing on a wrong number is still a failure."""
RUNNER = os.path.join(REPO_ROOT, "ci", "scripts", "gpu_conformance.sh")
STAGES = os.path.join(REPO_ROOT, "python", "morpheus", "morpheus", "stages")
UTILS = os.path.join(REPO_ROOT, "python", "morpheus", "morpheus", "utils")

CLAIM = re.compile(r"([\w-]+)\s+stages\s+and\s+([\w-]+)\s+supporting\s+modules,"
                   r"\s+covered\s+by\s+([\d,]+)\s+distinct\s+tests")
"""The claim, matched across a line break: the guide wraps its prose and the README does not, and a
pattern that only read one of them would leave the other unchecked while looking like it covered both."""

SINGLE_YEAR = re.compile(r"Copyright \(c\) \d{4},")
"""A copyright naming one year rather than a range, which is what every file this fork added carries."""

TEST_DEFINITION = re.compile(r"^\s*def test_", re.MULTILINE)

UNITS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def from_words(text: str) -> int:
    """A written number back to an integer, for the two counts the README spells out.

    Raises rather than returning a sentinel: a word this cannot read is a sentence that has been rewritten, and
    quietly comparing against zero would turn that into a passing test.
    """
    lowered = text.lower()

    if (lowered in UNITS):
        return UNITS.index(lowered)

    (tens, _, unit) = lowered.partition("-")

    if (tens in TENS):
        return TENS[tens] + (UNITS.index(unit) if unit else 0)

    raise ValueError(f"cannot read {text!r} as a number")


def is_forks(path: str) -> bool:
    """Whether this fork wrote the file, read from its copyright line."""
    with open(path, encoding="utf-8") as handle:
        header = "".join(handle.readline() for _ in range(4))

    return SINGLE_YEAR.search(header) is not None


def fork_stages() -> list:
    found = []

    for (directory, _, names) in os.walk(STAGES):
        for name in names:
            path = os.path.join(directory, name)

            if (name.endswith("_stage.py") and is_forks(path)):
                found.append(os.path.relpath(path, REPO_ROOT))

    return sorted(found)


def fork_modules() -> list:
    return sorted(
        os.path.relpath(os.path.join(UTILS, name), REPO_ROOT) for name in os.listdir(UTILS)
        if name.endswith(".py") and name != "__init__.py" and is_forks(os.path.join(UTILS, name)))


def fork_test_files() -> list:
    """The files the GPU conformance runner names, read from the runner rather than restated here.

    That list is itself checked against the fork's test headers by
    `tests/morpheus/utils/test_gpu_conformance_targets.py`, so a test file that exists and is in neither tier
    fails there rather than silently lowering the count here.
    """
    with open(RUNNER, encoding="utf-8") as handle:
        text = handle.read()

    entries = []

    for name in ("TARGETS", "UNMARKED"):
        match = re.search(rf"^{name}=\((.*?)^\)", text, re.MULTILINE | re.DOTALL)

        assert match is not None, f"{name} is not an array in the runner any more"

        entries.extend(line.strip() for line in match.group(1).splitlines()
                       if line.strip() and not line.strip().startswith("#"))

    return sorted(entries)


def count_tests() -> int:
    total = 0

    for relative in fork_test_files():
        with open(os.path.join(REPO_ROOT, relative), encoding="utf-8") as handle:
            total += len(TEST_DEFINITION.findall(handle.read()))

    return total


def claim_in(document: str) -> tuple:
    """The three numbers one document states, as integers."""
    with open(DOCUMENTS[document], encoding="utf-8") as handle:
        match = CLAIM.search(handle.read())

    assert match is not None, (f"the {document} no longer states what this fork has built in the form this "
                               f"checks. Restore the sentence or update the pattern -- do not delete the check, "
                               f"which is the only thing keeping those numbers true.")

    return (from_words(match.group(1)), from_words(match.group(2)), int(match.group(3).replace(",", "")))


def test_the_inputs_are_where_we_think_they_are():
    # Guards the three assertions below against passing over empty walks, which is how a count check becomes a
    # check that two zeroes are equal.
    assert len(fork_stages()) > 20
    assert len(fork_modules()) > 20
    assert len(fork_test_files()) > 60


def test_the_copyright_rule_separates_this_fork_from_upstream():
    # The whole inventory rests on this. Upstream's `type_utils` and this fork's `entity_key` sit in one package,
    # and the only thing between them is the shape of a line near the top.
    assert is_forks(os.path.join(UTILS, "entity_key.py"))
    assert is_forks(os.path.join(STAGES, "telemetry", "tc1_binding_stage.py"))
    assert not is_forks(os.path.join(UTILS, "type_utils.py"))
    assert not is_forks(os.path.join(STAGES, "input", "file_source_stage.py"))


@pytest.mark.parametrize("document", sorted(DOCUMENTS))
def test_the_stated_stage_count_is_the_one_that_is_there(document: str):
    stages = fork_stages()
    stated = claim_in(document)[0]

    assert stated == len(stages), (f"the {document} says {stated} stages; this fork has written {len(stages)}: "
                                   f"{stages}")


@pytest.mark.parametrize("document", sorted(DOCUMENTS))
def test_the_stated_module_count_is_the_one_that_is_there(document: str):
    modules = fork_modules()
    stated = claim_in(document)[1]

    assert stated == len(modules), (f"the {document} says {stated} supporting modules; this fork has written "
                                    f"{len(modules)}: {modules}")


@pytest.mark.parametrize("document", sorted(DOCUMENTS))
def test_the_stated_test_count_is_the_one_that_is_there(document: str):
    written = count_tests()
    stated = claim_in(document)[2]

    assert stated == written, (f"the {document} says {stated} distinct tests; the files the conformance runner "
                               f"names define {written}. Both documents carried the same stale figure for "
                               f"fourteen commits before this check existed.")


STAGES_PAGE = os.path.join(REPO_ROOT, "docs", "source", "stages", "morpheus_stages.md")
SAVED_SEARCHES = os.path.join(REPO_ROOT,
                              "examples",
                              "splunk_lineage_app",
                              "TA-morpheus-lineage",
                              "default",
                              "savedsearches.conf")
REGISTERED_STAGE = re.compile(r"^@register_stage\(.*?\n(?:.*\n)*?class\s+(\w+)\(", re.MULTILINE)
"""A `@register_stage` decorator and the class it decorates, however many decorator lines sit between them."""
RULE_STANZA = re.compile(r"^\[(R-[A-Z]-(?:L\d-)?\d{3}) - ", re.MULTILINE)
ANY_STANZA = re.compile(r"^\[", re.MULTILINE)
RULE_ID = re.compile(r"\bR-[A-Z]-(?:L\d-)?\d{3}\b")


def fork_stage_classes() -> dict:
    """Every registered stage class this fork wrote, keyed by class name, with the module it lives in."""
    found = {}

    for relative in fork_stages():
        with open(os.path.join(REPO_ROOT, relative), encoding="utf-8") as handle:
            text = handle.read()

        module = relative.replace(os.sep, ".").removeprefix("python.morpheus.").removesuffix(".py")

        for match in REGISTERED_STAGE.finditer(text):
            found[match.group(1)] = module

    return found


def test_the_stages_page_names_every_stage_this_fork_registers():
    # The stages page listed sixteen of the thirty-six telemetry stages and six of the nine lineage stages for
    # weeks, because nothing compared it with the tree. A stage a reader cannot find on the reference page is a
    # stage that, to that reader, does not exist.
    classes = fork_stage_classes()

    assert len(classes) > 40

    with open(STAGES_PAGE, encoding="utf-8") as handle:
        page = handle.read()

    missing = sorted(f"{module}.{name}" for (name, module) in classes.items() if f"{module}.{name}" not in page)

    assert missing == [], f"registered in this fork and absent from docs/source/stages/morpheus_stages.md: {missing}"


def shipped_rules() -> tuple:
    """The rule stanzas in the app and the stanza total, read from savedsearches.conf."""
    with open(SAVED_SEARCHES, encoding="utf-8") as handle:
        text = handle.read()

    return (sorted(set(RULE_STANZA.findall(text))), len(ANY_STANZA.findall(text)))


def specified_rules() -> list:
    """Every rule identifier Part 3 of the guide defines in bold, which is how it introduces a rule."""
    with open(DOCUMENTS["guide"], encoding="utf-8") as handle:
        text = handle.read()

    part_3 = text[text.index("## Part 3:"):text.index("## Part 4:")]

    return sorted(set(RULE_ID.findall("\n".join(line for line in part_3.splitlines() if line.startswith("**R-")))))


def test_the_summary_states_how_many_rules_ship_and_how_many_are_specified():
    # The Summary said six detections shipped while thirty-two did. The count is now read from the stanzas and
    # the specification both, so the sentence can only be true.
    (shipped, _) = shipped_rules()
    specified = specified_rules()

    assert len(shipped) > 20 and len(specified) >= len(shipped)

    with open(DOCUMENTS["guide"], encoding="utf-8") as handle:
        summary = handle.read().split("## Part 0:")[0]

    match = re.search(r"([\w-]+) of the ([\w-]+) rules Part 3 specifies ship as saved searches", summary)

    assert match is not None, "the guide's Summary no longer states how many of the specified rules ship"
    assert from_words(match.group(1)) == len(shipped), f"the Summary says {match.group(1)}; {len(shipped)} stanzas ship"
    assert from_words(
        match.group(2)) == len(specified), (f"the Summary says {match.group(2)}; Part 3 defines {len(specified)}")


def test_the_readme_states_the_shipped_detection_count():
    (shipped, _) = shipped_rules()

    with open(DOCUMENTS["README"], encoding="utf-8") as handle:
        readme = handle.read()

    match = re.search(r"\| ([\w-]+) of the app's ([\w-]+) detection searches", readme)

    assert match is not None, "the README's first-detections row no longer states the app's detection count"
    assert from_words(match.group(2)) == len(shipped), f"the README says {match.group(2)}; {len(shipped)} stanzas ship"

    # The row names its own rules; their number is the first figure, and every one of them has to be a rule that
    # ships.
    row = readme[match.start():readme.index("\n", match.start())]
    named = set(RULE_ID.findall(row))

    assert from_words(match.group(1)) == len({rule for rule in named if re.match(r"R-[BDP]-L[125]-", rule)})
    assert named <= set(shipped), sorted(named - set(shipped))


def test_the_summary_names_every_telemetry_class_that_has_a_stage():
    # "The TC-1 and TC-2 feature stages" stood in the Summary for five telemetry classes after their stages landed.
    prefixes = sorted({
        os.path.basename(path)[:3].upper().replace("TC", "TC-")
        for path in fork_stages() if os.path.basename(path).startswith("tc")
    })

    assert prefixes == ["TC-0", "TC-1", "TC-2", "TC-3", "TC-4", "TC-5", "TC-6", "TC-7"], prefixes

    with open(DOCUMENTS["guide"], encoding="utf-8") as handle:
        summary = handle.read().split("## Part 0:")[0]

    assert f"every telemetry class from {prefixes[0]} to {prefixes[-1]}" in summary, (
        "the Summary no longer says feature stages run for every telemetry class; name them or restore the phrase")


CORPORA = os.path.dirname(os.path.abspath(__file__))


def composed_pipelines() -> list:
    """
    The thirteen corpora: every `*_pipeline.py` beside this file that is not a `run_` entry point, nor
    `corpus_pipeline.py`, the helpers the corpora share, which composes nothing of its own.
    """
    return sorted(name for name in os.listdir(CORPORA)
                  if name.endswith("_pipeline.py") and not name.startswith("run_") and name != "corpus_pipeline.py")


def test_every_composed_pipeline_places_the_determinism_stamp():
    # The envelope was a tested stage that no corpus placed for weeks, while two documents said scored events carried
    # it. A corpus that composes its stages without the stamp is a corpus whose golden proves nothing about
    # provenance, so the placement is asserted by name, in the source, for all thirteen.
    names = composed_pipelines()

    assert len(names) == 13, names

    unstamped = []

    for name in names:
        with open(os.path.join(CORPORA, name), encoding="utf-8") as handle:
            text = handle.read()

        if ("DeterminismStampStage(config" not in text or "stamping.envelope_for(" not in text):
            unstamped.append(name)

    assert not unstamped, f"composed without DeterminismStampStage: {unstamped}"


def required_fields() -> dict:
    """Per telemetry class, the fields Part 2 lists as required, for the classes that list them in one place."""
    with open(DOCUMENTS["guide"], encoding="utf-8") as handle:
        text = handle.read()

    part_2 = text[text.index("## Part 2"):text.index("## Part 3:")]
    found = {}

    for match in re.finditer(r"### (TC-\d):[^\n]*\n(.*?)(?=\n### |\Z)", part_2, re.S):
        listed = re.search(r"\*\*Required fields:\*\*(.*?)\n\n", match.group(2), re.S)

        if (listed is not None):
            found[match.group(1)] = re.findall(r"`([a-z0-9_]+)`", listed.group(1))

    return found


def read_fields(fields: list) -> list:
    """The fields a telemetry or lineage stage, or a shipped search, names."""
    sources = []

    for directory in ("telemetry", "lineage"):
        folder = os.path.join(STAGES, directory)
        sources += [os.path.join(folder, name) for name in os.listdir(folder) if name.endswith(".py")]

    text = ""

    for path in sources + [SAVED_SEARCHES]:
        with open(path, encoding="utf-8") as handle:
            text += handle.read()

    return [field for field in fields if re.search(rf"\b{field}\b", text)]


def test_the_readme_says_how_many_required_fields_each_class_reads():
    # Part 2 lists what each class must carry, and for weeks nothing said which of those anything reads: ten of
    # the twenty-one TC-5 fields had no reader while the collection table called the class done. The Read column
    # is recomputed here from the guide and the code, so a field gaining or losing its reader moves the number.
    with open(DOCUMENTS["README"], encoding="utf-8") as handle:
        readme = handle.read()

    stated = dict(re.findall(r"^\| \*\*(TC-\d)\*\* .*\| (\d+ of \d+) \|$", readme, re.MULTILINE))
    required = required_fields()

    assert len(required) >= 6, sorted(required)
    assert set(stated) == set(required), (sorted(stated), sorted(required))

    for (telemetry_class, fields) in required.items():
        assert stated[telemetry_class] == f"{len(read_fields(fields))} of {len(fields)}", telemetry_class


def test_part_2_names_the_tc5_fields_nothing_reads_yet():
    fields = required_fields()["TC-5"]
    unread = [field for field in fields if field not in read_fields(fields)]

    with open(DOCUMENTS["guide"], encoding="utf-8") as handle:
        text = handle.read()

    section = text[text.index("### TC-5: Session"):text.index("### TC-6:")]
    sentence = section[section.index("specified and not yet read by anything"):]
    sentence = sentence[:sentence.index(". ")]

    assert sorted(re.findall(r"`([a-z0-9_]+)`", sentence)) == sorted(unread)


SEARCH_HEAD_RESULTS = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "validate", "search_head_results.json")
EXPECTED_RESULTS = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "validate", "expected_results.json")
VALIDATION = os.path.join(REPO_ROOT, "examples", "splunk_lineage_app", "validate", "VALIDATION.md")
TELEMETRY = os.path.join(STAGES, "telemetry")

HOST_ROW_RULE = "R-P-L3-009"
"""The one layer 3 rule the README's layer 3 row leaves out of its count. It reads the host segment's trajectory
rather than a flow feature, so the README narrates it in the host-segment paragraph under "What this fork is
not" and the layer 3 row says so in as many words. The exception is written here so that a tenth layer 3 rule
cannot hide behind it."""


def readme_text() -> str:
    with open(DOCUMENTS["README"], encoding="utf-8") as handle:
        return handle.read()


def shipped_rules_at(layers: str) -> list:
    """The rule stanzas whose identifier names one of the given layer digits."""
    (shipped, _) = shipped_rules()

    return [rule for rule in shipped if re.match(rf"R-[A-Z]-L[{layers}]-", rule)]


def layer_stages(layers: str) -> list:
    """
    The telemetry stage files this fork wrote for the given layers: `tc<n>_*_stage.py` for each digit, and
    `flow_record_normalize_stage.py`, which feeds layer 3 and is the one stage of theirs not named after a class.
    """
    return sorted(name for name in os.listdir(TELEMETRY)
                  if (re.match(rf"tc[{layers}]_.*_stage\.py$", name) or name == "flow_record_normalize_stage.py")
                  and is_forks(os.path.join(TELEMETRY, name)))


def test_the_boundary_bullet_counts_the_layer_3_4_6_7_stages_and_searches():
    # "Seventeen stages and twenty-four detection searches ship for them" sat through the three predictive rules
    # that took the layers to twenty-seven. Both numbers are read from the tree and the app.
    match = re.search(r"([\w-]+) stages and ([\w-]+) detection searches ship for them", readme_text())

    assert match is not None, "the README's layers 3, 4, 6 and 7 bullet no longer states its stage and search counts"

    stages = layer_stages("3467")
    rules = shipped_rules_at("3467")

    assert len(stages) > 10 and len(rules) > 20

    assert from_words(match.group(1)) == len(stages), (f"the README says {match.group(1)} stages ship for layers 3, "
                                                       f"4, 6 and 7; the tree holds {len(stages)}: {stages}")
    assert from_words(match.group(2)) == len(rules), (f"the README says {match.group(2)} detection searches ship for "
                                                      f"layers 3, 4, 6 and 7; {len(rules)} stanzas do: {rules}")

    # "all twenty-four have run on a search head" repeats the figure in the next sentence.
    repeated = re.search(r"and all\s+([\w-]+) have run on a search head", readme_text())

    assert repeated is not None, "the bullet no longer says all of those searches have run on a search head"
    assert from_words(repeated.group(1)) == len(rules), f"the README says all {repeated.group(1)} have run"


def test_the_siem_row_states_the_latest_search_head_run():
    # The SIEM row ended on the 2026-10-22 run, "all eighty the app now ships, with 224", for a day after the
    # thirteenth run returned eighty-two and wrote 233. The row's closing run is read against the recorded run,
    # the expectations and the stanzas, so the app cannot gain a search without the row noticing. A recorded run
    # is `search_head_results.json` while the expectations it matched are the ones written; once a change to the
    # corpus or the stanzas supersedes it, the run is kept under `search_head_runs/`, the row says what was *then*
    # written and that no run has returned the new expectation yet, and this reads the newest archived run instead.
    import json  # pylint: disable=import-outside-toplevel

    runs = re.findall(r"returned what (is|was then) written for all ([\w-]+)(?: the app now ships)?, with (\d+)",
                      readme_text())

    assert runs, "the README's SIEM row no longer ends on what the latest search-head run returned"

    (tense, spelled, risk) = runs[-1]
    stated_searches = from_words(spelled)
    stated_risk = int(risk)

    with open(EXPECTED_RESULTS, encoding="utf-8") as handle:
        expected = json.load(handle)

    (_, stanzas) = shipped_rules()

    if (os.path.exists(SEARCH_HEAD_RESULTS)):
        assert tense == "is", "a recorded run is committed, so the row's last run is the one the expectations match"
        results_path = SEARCH_HEAD_RESULTS
    else:
        assert tense == "was then", "no recorded run is committed, so the row's last run matched an earlier expectation"
        assert "no run has returned" in readme_text(), "the row must say the current expectation awaits a run"
        results_path = os.path.join(
            os.path.dirname(SEARCH_HEAD_RESULTS),
            "search_head_runs",
            sorted(os.listdir(os.path.join(os.path.dirname(SEARCH_HEAD_RESULTS), "search_head_runs")))[-1])

    with open(results_path, encoding="utf-8") as handle:
        results = json.load(handle)

    assert stated_searches == len(results["searches"]), (f"the README says all {spelled} searches ran; "
                                                         f"{os.path.basename(results_path)} holds "
                                                         f"{len(results['searches'])}")
    assert stated_searches == len(expected["searches"]), (f"the README says all {spelled}; "
                                                          f"expected_results.json expects {len(expected['searches'])}")
    assert stated_searches == stanzas, f"the README says all {spelled}; savedsearches.conf holds {stanzas} stanzas"

    # VALIDATION.md narrates each run as "for all <N> over <events> indexed events, and the detections wrote <R>
    # risk records"; the last such sentence is the latest run. If that phrasing ever goes, this half of the check
    # goes with it and the assertion above against the recorded run's own figure still stands.
    with open(VALIDATION, encoding="utf-8") as handle:
        narrated = re.findall(r"for all ([\w-]+) over [\d,]+ indexed\s+events, and the detections wrote (\d+) risk",
                              handle.read())

    assert narrated, "VALIDATION.md no longer narrates a run in the form this reads; update the pattern"
    assert (stated_searches, stated_risk) == (from_words(narrated[-1][0]), int(narrated[-1][1])), (
        f"the README's SIEM row says {spelled} with {stated_risk}; VALIDATION.md's latest run says "
        f"{narrated[-1][0]} with {narrated[-1][1]}")
    assert stated_risk == results["checks"]["risk_records"]["indexed"], (
        f"the README says {stated_risk} risk records; {os.path.basename(results_path)} indexed "
        f"{results['checks']['risk_records']['indexed']}")


def layer_row(layer: int) -> str:
    text = readme_text()
    start = text.index(f"| **Layer {layer} (TC-{layer})")

    return text[start:text.index("\n", start)]


@pytest.mark.parametrize("layer", [3, 4, 6])
def test_each_layer_row_counts_the_searches_that_ship_for_it(layer: int):
    # The layer 6 row said five detections shipped with six stanzas in the app, because R-P-L6-006 landed in the
    # app and the guide and not here. Each row's "<N> detections ship" is read against its stanzas, with the one
    # exception the layer 3 row states in the same sentence.
    row = layer_row(layer)
    match = re.search(r"([\w-]+)(?: of the ([\w-]+)) detections ship as saved searches", row) or re.search(
        r"([\w-]+)() detections ship as saved searches", row)

    assert match is not None, f"the README's layer {layer} row no longer says how many detections ship"

    rules = shipped_rules_at(str(layer))

    assert rules, layer

    excluded = [rule for rule in rules if layer == 3 and rule == HOST_ROW_RULE]

    if (excluded):
        assert f"the ninth, {HOST_ROW_RULE}," in row, (f"the layer 3 row leaves {HOST_ROW_RULE} out of its count "
                                                       f"and has to say so")

    assert from_words(match.group(1)) == len(rules) - len(excluded), (
        f"the README's layer {layer} row says {match.group(1)} detections ship; {len(rules)} stanzas do: {rules}"
        + (f", of which {excluded} is counted in the host row" if excluded else ""))


def test_the_layer_7_row_accounts_for_every_layer_7_search():
    # The row counts its rules in three sentences -- "<N> detections ship on these", "<M> more ... on them" and
    # "<K> more ... on it" -- and the row said four, two and one with eight stanzas in the app. Their sum is held
    # to the stanzas.
    row = layer_row(7)
    counts = re.findall(r"([\w-]+) (?:detections ship on these|more detections? ships? on (?:them|it))", row)

    assert len(counts) == 3, counts

    rules = shipped_rules_at("7")

    assert sum(from_words(count) for count in counts) == len(rules), (counts, rules)


def test_the_host_paragraph_no_longer_defers_the_tls_and_dns_drift():
    # "The daily drift over the TLS and DNS columns in their own pipelines comes next" outlived the pull request
    # that shipped it by a day. The stanzas say it shipped; the sentence may not say otherwise.
    text = readme_text()

    assert "R-P-L6-006" in shipped_rules()[0] and "R-P-L7-009" in shipped_rules()[0]
    assert re.search(
        r"TLS and DNS columns[^.]*\bcomes next\b",
        text) is None, ("the README still says the daily drift over the TLS and DNS columns comes next; it shipped")
