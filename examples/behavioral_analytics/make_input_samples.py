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
Write each segment's input as the files a collector would send, and the contract those files are held to.

The corpus builders under `tests/morpheus/determinism/` are where every input class's shape has lived: a frame per
class, built in memory and handed to a pipeline. A collector author cannot diff their output against a function.
This writes the same records out as JSON lines, one directory per segment and one file per class, and renders
`collector_contract.md` from `input_contracts.json`, which says per class and per column what the type, the unit and
the collector-side source are.

    python examples/behavioral_analytics/make_input_samples.py

The files are checked in, and `tests/morpheus/determinism/test_input_contracts.py` regenerates both and compares, so a
builder that gains a column without the contract saying what it is fails a test rather than drifting.

Samples are per segment rather than per class because a class's records differ between corpora: the DHCP server's
acknowledgements in the network corpus are not the ones in the application corpus, and each segment's runner has to
reproduce its own golden from its own files.
"""

import json
import math
import os
import pathlib
import sys
import typing

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = HERE.parents[1]
SAMPLES = HERE / "samples"
CONTRACTS = HERE / "input_contracts.json"
CONTRACT_DOCUMENT = HERE / "collector_contract.md"

sys.path.insert(0, str(HERE))

SEGMENTS = {
    "telemetry": "telemetry",
    "network": "network",
    "transport": "transport",
    "session": "session",
    "presentation": "presentation",
    "application": "application",
    "saas": "saas",
    "endpoint": "endpoint",
    "context": "context",
    "host_score": "host_score",
}
"""Each segment a deployment runs, and the corpus module whose builder holds its input. The estate and campaign
corpora are compositions of these segments over one estate, not segments of their own; the host segment is one,
reading three other segments' classes and the leases and inventory they resolve through."""

NOT_INPUT_CLASSES = {
    "tc3_late_delivery": "tc3",
    "tc0_identity_probes": None,
    "tc0_asset_probes": None,
    "tc0_application_probes": None,
}
"""Corpus frames that are not input classes of their own. The late delivery is a flow record that arrived after its
window sealed, held to the flow contract; the probes stand in for the scored rows the context is joined onto, which
in a deployment are another segment's output."""


def contract_class(frame_name: str) -> typing.Optional[str]:
    """The contract a corpus frame is held to, or `None` for a probe."""
    return NOT_INPUT_CLASSES.get(frame_name, frame_name)


def load_contracts() -> dict:
    """`input_contracts.json`, parsed."""
    with open(CONTRACTS, encoding="utf-8") as handle:
        return json.load(handle)


def _plain(value: typing.Any) -> typing.Any:
    # A gap is null however pandas spelled it, and a numpy scalar is the number it holds.
    if (value is None):
        return None

    if (hasattr(value, "item") and not isinstance(value, (str, bytes))):
        value = value.item()

    if (isinstance(value, float) and math.isnan(value)):
        return None

    try:
        import pandas as pd  # pylint: disable=import-outside-toplevel

        if (value is pd.NA or value is pd.NaT):
            return None
    except ImportError:
        pass

    return value


def column_types(class_name: typing.Optional[str]) -> dict[str, str]:
    """Each column a class's contract names, with its type."""
    if (class_name is None):
        return {}

    contracts = load_contracts()
    spec = contracts["classes"][class_name]
    columns = list(spec["columns"]) + (list(contracts["envelope"]) if spec["envelope"] else [])

    return {column["name"]: column["type"] for column in columns}


def _as_contract_type(value: typing.Any, column_type: typing.Optional[str]) -> typing.Any:
    # A builder that holds an integer column with gaps holds it as float, as pandas does; a collector sends the
    # integer, and so does the sample.
    if (column_type == "integer" and isinstance(value, float) and value.is_integer()):
        return int(value)

    return value


def render_frame(frame, class_name: typing.Optional[str] = None) -> str:
    """One JSON object per row, columns in the frame's order, gaps as null, values as their contract types."""
    types = column_types(class_name)
    lines = []

    for row in frame.to_dict(orient="records"):
        record = {name: _as_contract_type(_plain(value), types.get(name)) for (name, value) in row.items()}
        lines.append(json.dumps(record, ensure_ascii=False))

    return "\n".join(lines) + "\n"


def segment_corpus(segment: str) -> dict:
    """The frames a segment's corpus module builds."""
    from corpora import merged  # pylint: disable=import-outside-toplevel

    return merged(SEGMENTS[segment]).build_corpus()


def render_samples() -> dict[pathlib.Path, str]:
    """Every sample file, by path."""
    rendered = {}

    for segment in SEGMENTS:
        for (name, frame) in segment_corpus(segment).items():
            rendered[SAMPLES / segment / f"{name}.jsonlines"] = render_frame(frame, contract_class(name))

    return rendered


def _cell(text: str) -> str:
    return text.replace("|", "\\|") if text else "--"


def _unit(text: str) -> str:
    return f"`{text}`" if text else "--"


CONTRACT_PREAMBLE = """\
# Collector contract

What a collector has to send, per class and per column, and where each value comes from. Written so a collector
author can diff their output against something other than prose.

<!-- Generated from input_contracts.json by make_input_samples.py; edit the JSON and regenerate. -->

Every class has a sample under [`samples/`](./samples), one directory per segment and one JSON lines file per class,
exported from the corpus that segment's golden is produced from; a collector's first test is that its own output can
be swapped in for the sample and the segment still runs. `tests/morpheus/determinism/test_input_contracts.py` holds
every corpus builder to these tables: a required column every builder of the class sends, an optional one only some
do, and nothing a builder sends that the table does not name.

Columns are JSON scalars: an integer is a JSON number with no fraction, a string a JSON string, and a gap is `null`,
never an empty string or a sentinel. Times are integers in nanoseconds since the Unix epoch.

## Envelope

Every record carries these, whatever the class, except the TC-0 inventories, whose clock is the valid and recorded
times each record carries. They are what content-addressed identifiers are computed over and what clock quarantine
is decided on, so a collector that omits one produces records that cannot be traced back to it.

| Column | Type | Unit | Source | Notes |
|---|---|---|---|---|
"""

CLASS_TABLE_HEADER = """\
| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
"""

CONTRACT_POSTSCRIPT = """\
## What a wrong field costs

- A `port_id` that is a bridge port number rather than an interface name produces bindings that join to nothing,
  and every layer 1 to layer 2 correlation silently returns empty.
- A `site_id` that varies between polls forks one entity into several, each with its own baseline.
- An `event_time` that is the poll time rather than the observation time makes every gap measurement report the
  polling cadence instead of the network's behaviour -- which is exactly the measurement R-D-L2-004 separates a
  spoof from a move with.
- A `collector_seq` that restarts at zero makes two different records collide on one identifier.
- A counter sent as a delta rather than a total cannot be told from a reset, and the rate the layer 1 rules read is
  lost on every reboot.

## Sending it

The pipeline reads JSON lines: one object per line, no wrapping array. `mac_table_sample.jsonlines` is the
seventeen-record example `run_mac_spoof.py` reads by default; the per-segment samples are the corpora the runners
reproduce their golden files from.
"""


def render_contract() -> str:
    """`collector_contract.md`, from the JSON contract."""
    contracts = load_contracts()
    parts = [CONTRACT_PREAMBLE]

    for column in contracts["envelope"]:
        parts.append(f"| `{column['name']}` | {column['type']} | {_unit(column['unit'])} | {_cell(column['source'])} | "
                     f"{_cell(column['notes'])} |\n")

    for (name, spec) in contracts["classes"].items():
        envelope = "Carries the envelope." if spec["envelope"] else "Carries no collector envelope."
        parts.append(f"\n## {spec['title']}\n\n`{name}`, layer {spec['layer']}; sample in "
                     f"[`samples/{spec['home']}/{name}.jsonlines`](./samples/{spec['home']}/{name}.jsonlines). "
                     f"{envelope}\n\n{spec['description']}\n\n")
        parts.append(CLASS_TABLE_HEADER)

        for column in spec["columns"]:
            required = "yes" if column["required"] else "no"
            parts.append(f"| `{column['name']}` | {column['type']} | {_unit(column['unit'])} | {required} | "
                         f"{_cell(column['source'])} | {_cell(column['notes'])} |\n")

    parts.append("\n" + CONTRACT_POSTSCRIPT)

    return "".join(parts)


def main() -> int:
    """Write the samples and the contract document."""
    for (path, text) in render_samples().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"{path.relative_to(HERE)!s:56s} {text.count(chr(10)):6d} records")

    CONTRACT_DOCUMENT.write_text(render_contract(), encoding="utf-8")
    print(f"{CONTRACT_DOCUMENT.relative_to(HERE)}")

    return 0


if (__name__ == "__main__"):
    sys.exit(main())
