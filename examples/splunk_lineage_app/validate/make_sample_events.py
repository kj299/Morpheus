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
Generate the sample events a Splunk instance is fed, from the pipeline rather than by hand.

Hand-written sample events prove that a search matches hand-written sample events. These come out of the same
`run_pipeline` the determinism tests call, through the same `SiemWireStage` a deployment would put before its
sink, so what Splunk receives here is what Splunk would receive from the pipeline.

    python examples/splunk_lineage_app/validate/make_sample_events.py

Regenerate whenever the corpus or a stage changes, and review the diff -- these files are checked in so that a
change to what the SIEM would receive is visible in a pull request rather than only on a search head.
"""

import json
import os
import pathlib
import sys

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = HERE.parents[2]
EVENTS = HERE / "sample_events"

sys.path.insert(0, str(REPO_ROOT / "tests" / "morpheus" / "determinism"))

# The class of each telemetry row, and the sourcetype a deployment would send it on.
CLASS_SOURCETYPES = {
    "tc1": "morpheus:score:l1",
    "tc2_mac": "morpheus:score:l2",
    "tc2_arp": "morpheus:score:l2",
    "tc2_auth": "morpheus:score:l2",
    "tc2_binding": "binding:l2",
}


def _render(frame, sourcetype: str) -> list:
    """Put a frame through the wire stage exactly as a pipeline would, and return the lines a sink would write."""
    from morpheus.config import Config
    from morpheus.config import CppConfig
    from morpheus.config import ExecutionMode
    from morpheus.io import serializers
    from morpheus.messages import MessageMeta
    from morpheus.stages.output.siem_wire_stage import SiemWireStage

    CppConfig.set_should_use_cpp(False)
    config = Config()
    config.execution_mode = ExecutionMode.CPU

    stage = SiemWireStage(config, sourcetype=sourcetype, require_columns=False)
    rendered = stage.on_data(MessageMeta(frame.reset_index(drop=True))).copy_dataframe()

    return serializers.df_to_json(rendered, strip_newlines=True)


def main() -> int:
    import lineage_pipeline  # pylint: disable=import-outside-toplevel
    import telemetry_pipeline as tp  # pylint: disable=import-outside-toplevel

    EVENTS.mkdir(parents=True, exist_ok=True)
    telemetry = tp.run_pipeline(tp.build_pipeline_config(), tp.build_corpus())
    written = {}

    by_sourcetype: dict = {}

    for (name, sourcetype) in CLASS_SOURCETYPES.items():
        rows = telemetry[telemetry["telemetry_class"] == name]

        if (len(rows) == 0):
            continue

        by_sourcetype.setdefault(sourcetype, []).extend(_render(rows, sourcetype))

    lineage = lineage_pipeline.run_pipeline(lineage_pipeline.build_pipeline_config(), [lineage_pipeline.build_corpus()])
    by_sourcetype["morpheus:edge"] = _render(lineage, "morpheus:edge")

    bindings = telemetry[telemetry["telemetry_class"] == "tc2_binding"]
    bucketed = tp.build_binding_table(bindings).to_bucketed_frame()
    by_sourcetype["binding:bucketed"] = _render(bucketed, "binding:bucketed")

    for (sourcetype, lines) in sorted(by_sourcetype.items()):
        path = EVENTS / f"{sourcetype.replace(':', '_')}.jsonlines"

        with open(path, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")

        written[sourcetype] = len(lines)
        print(f"{path.name:<32} {len(lines):>5} events")

    with open(EVENTS / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"events_per_sourcetype": written}, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return 0


if (__name__ == "__main__"):
    sys.exit(main())
