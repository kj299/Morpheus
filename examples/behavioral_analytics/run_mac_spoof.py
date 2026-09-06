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
R-D-L2-004 end to end: a MAC address table on disk in, closed binding records a SIEM can parse out.

Every other test in this fork stops one hop short of the wire. They apply a rule's predicate to a DataFrame in
memory, which answers whether the analytics are right and says nothing about whether the bytes a SIEM receives
carry what the rule reads. That last hop is exactly where two of this app's defects were hiding: a timestamp the
SIEM could not parse, and a field the search filtered on that the producer never wrote.

So this runs the whole path. A file, four stages, a file -- and
`tests/morpheus/determinism/test_end_to_end_mac_spoof.py` reads the output back off disk, stamps `_time` by
applying the shipped `props.conf`'s own regex and format to the raw bytes, and applies R-D-L2-004's predicate to
the parsed JSON rather than to a frame.

R-D-L2-004 is the rule this can be done for. It needs one collector and nothing else: no `port_designations`
lookup, no exclusion list, no TC-0 context store. The other three shipped detections each need something an
estate has to supply first, which is why proving one end to end is worth more than gesturing at four.

    python examples/behavioral_analytics/run_mac_spoof.py

Writes `notables.jsonlines` beside the input unless `--output` says otherwise.
"""

import logging
import os
import pathlib
import sys

import click

from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import ExecutionMode
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.output.siem_wire_stage import SiemWireStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.telemetry.tc2_binding_stage import TC2BindingStage
from morpheus.utils.logger import configure_logging

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
SOURCETYPE = "binding:l2"
"""The stanza the output is written for. `props.conf` anchors it on `bind_end`, because the record exists
*because* the binding closed, and a binding opened hours ago and closed by a conflict now would otherwise land
outside every detection window."""


def build_pipeline(config: Config, source: pathlib.Path, output: pathlib.Path) -> LinearPipeline:
    """
    Four stages, in the order the guide argues for.

    `TotalOrderStage` first, because everything after it is stateful in arrival order and each of those stages
    flags out-of-order arrival rather than repairing it -- someone has to impose the order they depend on, once,
    ahead of the first of them. `TC2BindingStage` closes a binding when the same address turns up somewhere else,
    and calls it a conflict rather than a move when the two sightings share an instant, because a device cannot
    be in two places at one time and a poller cannot report it in two places at one time either. `SiemWireStage`
    last, immediately before the sink, so what is serialized is what the SIEM parses.
    """
    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=source, file_type=FileTypes.JSON))
    pipe.add_stage(TotalOrderStage(config))
    # Closed bindings only. The stage can emit a provisional record the moment a binding opens, and those are
    # genuinely useful -- live attribution needs an answer before the binding closes -- but they carry a null
    # `bind_end`, and this sourcetype anchors `_time` on it. Writing them here would put records with no anchor
    # onto a stanza that reads one, which is the silent index-time stamping the app's own configuration warns
    # about. They have their own stanza, `binding:l2:open`, anchored on `bind_start`; a pipeline that wants both
    # writes two sinks rather than conflating them.
    pipe.add_stage(TC2BindingStage(config))
    pipe.add_stage(SiemWireStage(config, sourcetype=SOURCETYPE))
    pipe.add_stage(WriteToFileStage(config, filename=str(output), overwrite=True, include_index_col=False))

    return pipe


@click.command()
@click.option("--source",
              type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
              default=HERE / "mac_table_sample.jsonlines",
              show_default=True,
              help="MAC address table snapshots, one JSON object per line.")
@click.option("--output",
              type=click.Path(dir_okay=False, path_type=pathlib.Path),
              default=HERE / "notables.jsonlines",
              show_default=True,
              help="Where to write the closed binding records.")
def main(source: pathlib.Path, output: pathlib.Path) -> int:
    """Run the pipeline over a MAC address table and write the records R-D-L2-004 reads."""
    configure_logging(log_level=logging.INFO)

    CppConfig.set_should_use_cpp(False)
    config = Config()
    config.execution_mode = ExecutionMode.CPU

    build_pipeline(config, source, output).run()

    logging.getLogger(__name__).info("wrote %s", output)

    return 0


if (__name__ == "__main__"):
    sys.exit(main())  # pylint: disable=no-value-for-parameter
