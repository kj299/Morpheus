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

# The class of each estate row, and the sourcetype a deployment would send it on.
CLASS_SOURCETYPES = {
    "tc1": "morpheus:score:l1",
    "tc2_mac": "morpheus:score:l2",
    "tc2_arp": "morpheus:score:l2",
    "tc2_auth": "morpheus:score:l2",
    "tc2_binding": "binding:l2",
    "tc1_binding": "binding:l1",
    "tc5_auth": "morpheus:score:l5",
}
"""Layers 1, 2 and 5 come from the estate pipeline, which is the telemetry pipeline plus the authentications the
people at those ports made in the same hour, sealed together.

Rendering layers 1 and 2 from the telemetry pipeline instead would index the same events with chains that stop at
two layers, because a chain is decided by which classes were sealed together. `Chain assembly - cross-layer risk`
reads `dc(osi_layer)` over a `lineage_id`, so the difference between the two runs is the difference between a
search with nothing to assemble and one with three layers to assemble.
"""

SESSION_CLASS_SOURCETYPES = {
    "tc5_auth": "morpheus:score:l5",
    "tc5_session": "morpheus:score:l5",
}
"""The layer 5 classes, which come from their own corpus and their own pipeline.

Both go on one sourcetype. A deployment could split them, and the app's stanza does not care -- but the two
layer 5 rules read only the authentication columns, and sending the session records on the same sourcetype is
what proves those rules stay quiet on rows that carry none of the fields they filter on.
"""


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
    # Imported here rather than at module scope, like everything else this file takes from morpheus: the path
    # setup above has to run before the package is reachable.
    from morpheus.utils.binding_table import DEFAULT_L1_BUCKET_SECONDS  # pylint: disable=import-outside-toplevel

    import estate_pipeline as ep  # pylint: disable=import-outside-toplevel
    import lineage_pipeline  # pylint: disable=import-outside-toplevel
    import network_pipeline  # pylint: disable=import-outside-toplevel
    import presentation_pipeline  # pylint: disable=import-outside-toplevel
    import session_pipeline as sp  # pylint: disable=import-outside-toplevel
    import telemetry_pipeline as tp  # pylint: disable=import-outside-toplevel
    import transport_pipeline  # pylint: disable=import-outside-toplevel

    EVENTS.mkdir(parents=True, exist_ok=True)
    telemetry = ep.run_pipeline(ep.build_pipeline_config(), ep.build_corpus())
    written = {}

    by_sourcetype: dict = {}

    for (name, sourcetype) in CLASS_SOURCETYPES.items():
        rows = telemetry[telemetry["telemetry_class"] == name]

        if (len(rows) == 0):
            continue

        by_sourcetype.setdefault(sourcetype, []).extend(_render(rows, sourcetype))

    sessions = sp.run_pipeline(sp.build_pipeline_config(), sp.build_corpus())

    for (name, sourcetype) in SESSION_CLASS_SOURCETYPES.items():
        rows = sessions[sessions["telemetry_class"] == name]

        if (len(rows) > 0):
            by_sourcetype.setdefault(sourcetype, []).extend(_render(rows, sourcetype))

    lineage = lineage_pipeline.run_pipeline(lineage_pipeline.build_pipeline_config(), [lineage_pipeline.build_corpus()])
    by_sourcetype["morpheus:edge"] = _render(lineage, "morpheus:edge")

    # Layer 3 comes from its own corpus and its own pipeline, as layer 5 does. Its records share no entity with
    # the estate's -- an address is not a port and not a principal -- so they carry their own chains and the
    # cross-layer search sees them as a fourth layer's worth of events rather than as part of the estate's.
    flows = network_pipeline.run_pipeline(network_pipeline.build_pipeline_config(), network_pipeline.build_corpus())
    by_sourcetype["morpheus:score:l3"] = _render(flows, "morpheus:score:l3")

    # Layer 4 likewise. Its entity is a conversation rather than an address, so its records share no entity with
    # layer 3's either, even where the two corpora use the same addresses -- which is the thing Part 4's layer 3
    # to layer 4 hop would have to supply and does not yet.
    packets = transport_pipeline.run_pipeline(transport_pipeline.build_pipeline_config(),
                                              transport_pipeline.build_corpus())
    by_sourcetype["morpheus:score:l4"] = _render(packets, "morpheus:score:l4")

    # Layer 6 likewise, and it is the layer that makes R-C-002 expressible: the chained rule correlates a new
    # client fingerprint here with beaconing at layer 3 on the same host. The two corpora are separate, so the
    # rule still returns nothing -- what changed is that both halves now have producers.
    handshakes = presentation_pipeline.run_pipeline(presentation_pipeline.build_pipeline_config(),
                                                    presentation_pipeline.build_corpus())
    by_sourcetype["morpheus:score:l6"] = _render(handshakes, "morpheus:score:l6")

    bindings = telemetry[telemetry["telemetry_class"] == "tc2_binding"]
    bucketed = tp.build_binding_table(bindings).to_bucketed_frame()

    # The layer 1 history, on the same sourcetype and told apart by `binding_table`. Only the intervals a
    # current-state row has overwritten are expanded: a port whose optic never changed is answered correctly for
    # all time by the unbucketed lookup, so putting it here would cost a row per day to repeat that answer.
    port_bindings = telemetry[telemetry["telemetry_class"] == "tc1_binding"]
    port_history = tp.build_port_binding_table(port_bindings).superseded().to_bucketed_frame(
        bucket_seconds=DEFAULT_L1_BUCKET_SECONDS, key_name="entity_key")

    rendered = _render(bucketed, "binding:bucketed")

    if (len(port_history) > 0):
        rendered.extend(_render(port_history, "binding:bucketed"))

    by_sourcetype["binding:bucketed"] = rendered

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
