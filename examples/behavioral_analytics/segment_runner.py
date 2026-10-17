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
Run one segment from collector files to SIEM-ready records, restoring its stages' memory on start.

A segment is the set of input classes one deployment unit carries -- the network segment is the flows and the DHCP
server and EDR inventory they resolve through -- and its composition is the one its corpus module under
`tests/morpheus/determinism/` holds to a golden file. This adds the two ends a deployment needs around it:

    files ──► FileSourceStage ──► (the segment's composition: LineageStamp, TotalOrder, the class stages,
                                  TC0Enrich where it applies, EnvelopeStamp, WindowSeal, DeterminismStamp)
          ──► MinimizationStage ──► SiemWireStage ──► SiemWireSinkStage ──► one file, or one Kafka topic,
                                                                            per sourcetype

The input is a directory holding one JSON lines file per class, named for the class, as `samples/<segment>/` does
and as `collector_contract.md` specifies. Each is read by `FileSourceStage` with JSON's own types kept, and an
integer column with a gap is held as a nullable integer, as the contract says it is.

**Restore on start.** With `--state`, the runner restores every stage that remembers from the latest checkpoint in
that directory before it reads a row, and writes a new checkpoint when it stops, keeping open windows and bindings
rather than flushing them. `--final` marks the last run over the data, which flushes as a replay does. Every scored
row carries the digest of the checkpoint it resumed from as `resumed_from_state`.

**Minimization.** With `--policy`, a YAML policy (see `minimization_policy.example.yaml`) is applied by
`MinimizationStage` immediately before the wire, after every feature has been computed. The key is read from the
environment variable the policy names and never from the file.

The compositions themselves still live beside their corpora, because each is held to its golden there; this runner
imports them rather than holding a second copy that could drift.
"""

import argparse
import logging
import os
import pathlib
import sys
import typing

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = HERE.parents[1]
COMPOSITIONS = REPO_ROOT / "tests" / "morpheus" / "determinism"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(COMPOSITIONS))

L1 = "morpheus:score:l1"
L2 = "morpheus:score:l2"
L7 = "morpheus:score:l7"
LATE_SOURCETYPE = "morpheus:late"
QUARANTINE_SOURCETYPE = "morpheus:quarantine"
QUARANTINE_CLASS = "quarantine"

SEGMENTS: dict[str, dict] = {
    "telemetry": {
        "module": "telemetry_pipeline",
        "sourcetypes": {
            "tc1": L1,
            "tc1_device": L1,
            "tc2_mac": L2,
            "tc2_arp": L2,
            "tc2_auth": L2,
            "tc1_binding": "binding:l1",
            "tc2_binding": "binding:l2",
        },
    },
    "network": {
        "module": "network_pipeline", "sourcetypes": {
            "tc3": "morpheus:score:l3"
        }
    },
    "transport": {
        "module": "transport_pipeline", "sourcetypes": {
            "tc4": "morpheus:score:l4"
        }
    },
    "session": {
        "module": "session_pipeline",
        "sourcetypes": {
            "tc5_auth": "morpheus:score:l5", "tc5_session": "morpheus:score:l5"
        },
    },
    "presentation": {
        "module": "presentation_pipeline", "sourcetypes": {
            "tc6": "morpheus:score:l6"
        }
    },
    "application": {
        "module": "application_pipeline", "sourcetypes": {
            "tc7_dns": L7, "tc7_http": L7, "tc7_service": L7
        }
    },
    "saas": {
        "module": "saas_pipeline", "sourcetypes": {
            "tc7_saas": L7
        }
    },
    "endpoint": {
        "module": "endpoint_pipeline", "sourcetypes": {
            "tc7_endpoint": L7
        }
    },
    "context": {
        "module": "context_pipeline",
        "sourcetypes": {
            "tc0_identity": "context:identity",
            "tc0_asset": "context:asset",
            "tc0_application": "context:application",
        },
    },
}
"""Each segment, the composition that runs it, and the sourcetype each of its classes' records go on. A row whose
window had sealed before it arrived goes on `morpheus:late` and a quarantined record on `morpheus:quarantine`,
whatever its class; a class not listed (the context segment's probes) is not delivered."""

FILE_PARSER_KWARGS = {"convert_dates": False, "dtype": False, "precise_float": True}
"""How `FileSourceStage` parses a sample: JSON's own types, no date guessing on `*_time` columns, and floats read
exactly. pandas' defaults turn an AS number sent as a string into a number and parse `event_time` as a date."""


def _config():
    from morpheus.config import Config  # pylint: disable=import-outside-toplevel
    from morpheus.config import CppConfig  # pylint: disable=import-outside-toplevel
    from morpheus.config import ExecutionMode  # pylint: disable=import-outside-toplevel

    CppConfig.set_should_use_cpp(False)
    config = Config()
    config.execution_mode = ExecutionMode.CPU

    return config


def read_class(path: pathlib.Path):
    """
    One class's records, read by `FileSourceStage` and typed by the class's contract.

    Returns
    -------
    `pandas.DataFrame`
    """
    import make_input_samples  # pylint: disable=import-outside-toplevel
    import pandas as pd  # pylint: disable=import-outside-toplevel

    from morpheus.common import FileTypes  # pylint: disable=import-outside-toplevel
    from morpheus.pipeline import LinearPipeline  # pylint: disable=import-outside-toplevel
    from morpheus.stages.input.file_source_stage import FileSourceStage  # pylint: disable=import-outside-toplevel
    from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage  # pylint: disable=import-outside-toplevel

    config = _config()
    pipe = LinearPipeline(config)
    pipe.set_source(
        FileSourceStage(config,
                        filename=str(path),
                        file_type=FileTypes.JSON,
                        filter_null=False,
                        parser_kwargs=FILE_PARSER_KWARGS))
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    frames = [message.copy_dataframe() for message in sink.get_messages()]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 0 else pd.DataFrame()
    types = make_input_samples.column_types(make_input_samples.contract_class(path.stem))

    for column in frame.columns:
        if (types.get(column) == "integer" and frame[column].isna().any()):
            frame[column] = frame[column].astype("Int64")

    return frame


def read_inputs(input_dir: pathlib.Path) -> dict:
    """Every class file in a segment's input directory, by class name."""
    return {path.stem: read_class(path) for path in sorted(input_dir.glob("*.jsonlines"))}


def load_policy(path: typing.Optional[pathlib.Path]) -> typing.Optional[dict]:
    """
    A minimization policy, with its key read from the environment variable it names.

    Raises
    ------
    KeyError
        If the variable is not set: a policy with no key would pseudonymize with nothing and should not run.
    """
    if (path is None):
        return None

    import yaml  # pylint: disable=import-outside-toplevel

    with open(path, encoding="utf-8") as handle:
        policy = yaml.safe_load(handle)

    variable = policy.get("key_env", "MORPHEUS_MINIMIZATION_KEY")

    if (policy.get("pseudonymize") and variable not in os.environ):
        raise KeyError(f"the policy pseudonymizes and its key variable {variable} is not set")

    return {
        "drop": list(policy.get("drop") or []),
        "pseudonymize": list(policy.get("pseudonymize") or []),
        "key": os.environ.get(variable),
    }


def route(segment: str, result) -> list[tuple[str, str, typing.Any]]:
    """
    The composition's rows, grouped as a sink receives them: `(class, sourcetype, rows)`.

    Late rows go on `morpheus:late` and quarantined records on `morpheus:quarantine`; a class the segment does not
    deliver is left out.
    """
    sourcetypes = SEGMENTS[segment]["sourcetypes"]
    groups = []

    if (len(result) == 0):
        return groups

    for name in sorted(set(result["telemetry_class"].dropna())):
        rows = result[result["telemetry_class"] == name]

        if (name == QUARANTINE_CLASS):
            groups.append((name, QUARANTINE_SOURCETYPE, rows.dropna(axis=1, how="all")))
            continue

        if (name not in sourcetypes):
            continue

        late = (rows["is_late"].astype("boolean").fillna(False).astype(bool)
                if "is_late" in rows.columns else rows.index.to_series().map(lambda _: False))

        if (late.any()):
            groups.append((name, LATE_SOURCETYPE, rows[late]))

        if ((~late).any()):
            groups.append((name, sourcetypes[name], rows[~late]))

    return groups


def deliver(groups: list,
            output_dir: typing.Optional[pathlib.Path],
            policy: typing.Optional[dict] = None,
            kafka_bootstrap_servers: typing.Optional[str] = None,
            kafka_topic_prefix: str = "morpheus.") -> dict[str, int]:
    """
    Put each group through `MinimizationStage`, `SiemWireStage` and `SiemWireSinkStage`.

    Returns
    -------
    dict
        Records written per sourcetype.
    """
    # pylint: disable=import-outside-toplevel
    from morpheus.pipeline import LinearPipeline
    from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
    from morpheus.stages.lineage.minimization_stage import MinimizationStage
    from morpheus.stages.output.siem_wire_sink_stage import SiemWireSinkStage
    from morpheus.stages.output.siem_wire_stage import SiemWireStage

    config = _config()
    written: dict[str, int] = {}

    for (_, sourcetype, rows) in groups:
        pipe = LinearPipeline(config)
        pipe.set_source(InMemorySourceStage(config, dataframes=[rows.reset_index(drop=True)]))

        if (policy is not None):
            pipe.add_stage(
                MinimizationStage(config,
                                  drop=policy["drop"] or None,
                                  pseudonymize=policy["pseudonymize"] or None,
                                  key=policy["key"]))

        pipe.add_stage(SiemWireStage(config, sourcetype=sourcetype, require_columns=False))

        if (kafka_bootstrap_servers is not None):
            sink = SiemWireSinkStage(config,
                                     kafka_topic=kafka_topic_prefix + sourcetype.replace(":", "."),
                                     kafka_bootstrap_servers=kafka_bootstrap_servers)
        else:
            sink = SiemWireSinkStage(config,
                                     filename=str(output_dir / f"{sourcetype.replace(':', '_')}.jsonlines"),
                                     append=True)

        pipe.add_stage(sink)
        pipe.run()
        written[sourcetype] = written.get(sourcetype, 0) + sink.written

    return written


def run(segment: str,
        input_dir: pathlib.Path,
        output_dir: typing.Optional[pathlib.Path] = None,
        state_dir: typing.Optional[pathlib.Path] = None,
        final: bool = True,
        policy_path: typing.Optional[pathlib.Path] = None,
        kafka_bootstrap_servers: typing.Optional[str] = None) -> dict[str, int]:
    """
    Run one segment over the class files in `input_dir` and deliver its records.

    Parameters
    ----------
    segment : str
        One of `SEGMENTS`.
    input_dir : pathlib.Path
        One JSON lines file per class.
    output_dir : pathlib.Path, optional
        Where each sourcetype's records are written. A cold start refuses a directory that already holds records;
        a run resumed from a checkpoint appends to them.
    state_dir : pathlib.Path, optional
        Where the stages' memory is restored from and checkpointed to. Without it the run is a replay.
    final : bool, default = True
        The run reaches the end of the data, so open windows and bindings are flushed. A run that will be followed
        by another passes `False`.
    policy_path : pathlib.Path, optional
        A minimization policy.
    kafka_bootstrap_servers : str, optional
        Produce to Kafka, one topic per sourcetype, instead of writing files.

    Returns
    -------
    dict
        Records written per sourcetype.
    """
    import importlib  # pylint: disable=import-outside-toplevel

    import resumable  # pylint: disable=import-outside-toplevel

    if (segment not in SEGMENTS):
        raise ValueError(f"unknown segment {segment!r}; one of {sorted(SEGMENTS)}")

    if (kafka_bootstrap_servers is None):
        if (output_dir is None):
            raise ValueError("give an output directory or Kafka brokers")

        output_dir.mkdir(parents=True, exist_ok=True)
        resuming = state_dir is not None and any(pathlib.Path(state_dir).glob("checkpoint-*.json"))

        if (not resuming and any(output_dir.glob("*.jsonlines"))):
            raise FileExistsError(f"{output_dir} already holds records and this is a cold start; a resumed run "
                                  f"appends, a new one needs an empty directory")

    module = importlib.import_module(SEGMENTS[segment]["module"])
    policy = load_policy(policy_path)
    corpus = read_inputs(input_dir)
    config = module.build_pipeline_config()

    if (state_dir is None):
        result = module.run_pipeline(config, corpus)
    else:
        with resumable.persisted(str(state_dir), final=final):
            result = module.run_pipeline(config, corpus)

    return deliver(route(segment, result), output_dir, policy, kafka_bootstrap_servers)


def main(argv: typing.Optional[list] = None, segment: typing.Optional[str] = None) -> int:
    """Command line for `run(...)`; a `run_<segment>.py` wrapper fixes the segment."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0].strip())

    if (segment is None):
        parser.add_argument("segment", choices=sorted(SEGMENTS))

    parser.add_argument("--input", type=pathlib.Path, help="One JSON lines file per class. Defaults to the sample.")
    parser.add_argument("--output", type=pathlib.Path, help="Directory for one JSON lines file per sourcetype.")
    parser.add_argument("--state", type=pathlib.Path, help="Checkpoint directory; restored on start.")
    parser.add_argument("--not-final",
                        dest="final",
                        action="store_false",
                        help="Keep open windows and bindings for the next run instead of flushing them.")
    parser.add_argument("--policy", type=pathlib.Path, help="Minimization policy YAML.")
    parser.add_argument("--kafka", dest="kafka_bootstrap_servers", help="Produce to Kafka at these brokers.")
    arguments = parser.parse_args(argv)
    chosen = segment or arguments.segment

    logging.basicConfig(level=logging.INFO)
    written = run(chosen,
                  arguments.input or HERE / "samples" / chosen,
                  output_dir=arguments.output,
                  state_dir=arguments.state,
                  final=arguments.final,
                  policy_path=arguments.policy,
                  kafka_bootstrap_servers=arguments.kafka_bootstrap_servers)

    for (sourcetype, count) in sorted(written.items()):
        print(f"{sourcetype:<28} {count:>6} records")

    return 0


if (__name__ == "__main__"):
    sys.exit(main())
