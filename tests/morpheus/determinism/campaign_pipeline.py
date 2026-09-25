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
One campaign told at three layers, and the composed pipelines that measure it, for the chained rules.

Every other corpus in this fork is one layer's, and they share no entity: an address in the layer 3 corpus is not a
principal in the layer 5 one or a host in the layer 7 one. A chained rule needs the opposite -- the same source, the
same person and the same host seen by three collectors -- so this corpus is written as one estate over nine days and
split by collector. Each layer's records then go through that layer's own stages, exactly as a deployment would run
them, and the chained rule is evaluated over what comes out.

**R-C-001 - Lateral movement chain.** Within 30 minutes, in order: a source's fan-out rises above its own previous
hour; a principal authenticates *from that source* to a host they have never logged into; and that host starts a
process whose ancestry neither it nor its peer group has run in thirty days. No step has to breach a threshold of its
own. The attacker here reaches twenty-five new addresses, half what R-B-L3-001 fires on, and neither a first login to
a server nor one new process is an alert by itself.

The attacker does all three, and six others each do all but one:

- **wrong order**: the new process runs before the login that would have to have started it;
- **too slow**: every step happens, the last one thirty-five minutes after the first;
- **known host**: the login is to a server the principal logs into every day;
- **other source**: the login comes from an address other than the one whose fan-out rose;
- **flat fan-out**: a busy host reaching the same twenty servers every hour, so nothing rose;
- **no new process**: the host only runs what it always runs.

**The steps are joined on values, not on `lineage_id`.** The three layers are sealed on three different entities --
a source address, a principal, a host -- so no lineage chain holds all three, and the rule joins the source address
to the login's source and the login's target to the process's host, carrying each step's lineage identifiers as
evidence. Hosts are compared case-folded, because a login log and an EDR name the same machine differently.

**Each step may precede the one it follows by up to two minutes.** That is the rule's declared join tolerance, twice
a worst-case clock offset of sixty seconds between collectors, as Part 3's governance asks.
"""

import typing

import pandas as pd

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.stages.telemetry.tc3_beacon_stage import TC3BeaconStage
from morpheus.stages.telemetry.tc3_cardinality_stage import TC3CardinalityStage
from morpheus.stages.telemetry.tc3_reach_stage import TC3ReachStage
from morpheus.stages.telemetry.tc3_ttl_stage import TC3TtlStage
from morpheus.stages.telemetry.tc5_novelty_stage import TC5NoveltyStage
from morpheus.stages.telemetry.tc7_endpoint_stage import TC7EndpointStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900
DAY_SECONDS = 86400

ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []

FLOW_CLASS = "tc3"
AUTH_CLASS = "tc5_auth"
PROCESS_CLASS = "tc7_endpoint"
CLASSES = (FLOW_CLASS, AUTH_CLASS, PROCESS_CLASS)

# The rule's own figures, stated once so the corpus is built around them deliberately.
CHAIN_WINDOW_SECONDS = 30 * 60
JOIN_TOLERANCE_SECONDS = 120
SEVERITY = 80
FANOUT_THRESHOLD = 50
"""R-B-L3-001's fan-out threshold, which the attacker stays well under."""

LAST_DAY = 8
"""The day the campaign happens. The eight before it are the history every step is novel against."""
CAMPAIGN_HOUR = 10
SERVER_GROUP = "servers"


class Actor(typing.NamedTuple):
    """One source, the principal who logs in from it, and the server they reach, with what each step does."""

    source: str
    principal: str
    home: str
    """The server the principal logs into every day."""
    target: str
    """The server the campaign's login reaches. The home server for the known-host control."""
    login_source: str
    """Where the login comes from. The fan-out source except in the other-source control."""
    rising: bool
    fanout_minute: float
    login_minute: float
    process_minute: typing.Optional[float]
    """When the novel process runs on the target, or None when it only runs its routine."""


ATTACKER = "dana@example.com"
WRONG_ORDER = "eli@example.com"
TOO_SLOW = "fay@example.com"
KNOWN_HOST = "gus@example.com"
OTHER_SOURCE = "hal@example.com"
FLAT_FANOUT = "ivan@example.com"
NO_NEW_PROCESS = "judy@example.com"

ACTORS = {
    ATTACKER: Actor("10.20.0.5", ATTACKER, "srv-dana", "SRV-DB-02", "10.20.0.5", True, 5, 12, 20),
    WRONG_ORDER: Actor("10.20.0.6", WRONG_ORDER, "srv-eli", "srv-app-03", "10.20.0.6", True, 5, 14, 8),
    TOO_SLOW: Actor("10.20.0.7", TOO_SLOW, "srv-fay", "srv-fs-04", "10.20.0.7", True, 5, 20, 40),
    KNOWN_HOST: Actor("10.20.0.8", KNOWN_HOST, "srv-gus", "srv-gus", "10.20.0.8", True, 5, 12, 20),
    OTHER_SOURCE: Actor("10.20.0.9", OTHER_SOURCE, "srv-hal", "srv-web-05", "10.20.0.99", True, 5, 12, 20),
    FLAT_FANOUT: Actor("10.20.0.10", FLAT_FANOUT, "srv-ivan", "srv-mq-06", "10.20.0.10", False, 5, 12, 20),
    NO_NEW_PROCESS: Actor("10.20.0.11", NO_NEW_PROCESS, "srv-judy", "srv-ci-07", "10.20.0.11", True, 5, 12, None),
}

SERVICES = r"C:\Windows\System32\services.exe"
SVCHOST = r"C:\Windows\System32\svchost.exe"
WMI = r"C:\Windows\System32\wbem\WmiPrvSE.exe"
CMD = r"C:\Windows\System32\cmd.exe"

PREVIOUS_HOUR_DESTINATIONS = 3
RISING_DESTINATIONS = 25
FLAT_DESTINATIONS = 20


def at(day: float, hour: float = 0, minute: float = 0, second: float = 0) -> int:
    """An instant in the corpus, in nanoseconds since the epoch."""
    return int((day * DAY_SECONDS + hour * 3600 + minute * 60 + second) * NS_PER_SECOND)


def novel_image(principal: str) -> str:
    """The process each campaign starts, distinct per actor so one never excuses another in the peer group."""
    return rf"C:\Windows\Temp\{principal.split('@')[0]}-svc.exe"


def servers() -> list:
    """Every server a principal logs into or a campaign reaches, in one peer group."""
    names = set()

    for actor in ACTORS.values():
        names.update((actor.home, actor.target.lower()))

    return sorted(names)


def asset_versions() -> list:
    return [
        make_version("asset",
                     host,
                     0,
                     None,
                     0,
                     values={
                         "owner": "it-ops@example.com",
                         "owning_team": "platform",
                         "criticality": "high",
                         "data_classification": "internal",
                         "peer_group": SERVER_GROUP,
                     }) for host in servers()
    ]


def build_store() -> BitemporalStore:
    """The inventory the process enrichment reads: every server in one peer group."""
    return BitemporalStore("asset", asset_versions())


def _envelope(rows: list, collector: str, schema: str, origin: str, when: int) -> dict:
    return {
        "collector_id": collector,
        "schema_version": schema,
        "origin_hash": f"{origin}@{when}",
        "collector_seq": len(rows),
        "event_time": when,
    }


def _flow(rows: list, source: str, destination: str, when: int):
    record = _envelope(rows, "netflow-01", "tc3.v1", f"{source}->{destination}", when)
    record.update({
        "src_ip": source,
        "dst_ip": destination,
        "dst_port": 445,
        "protocol": "tcp",
        "ip_ttl": 128,
        "bytes_out": 900,
        "bytes_in": 1400,
        "bgp_as_dst": "64512",
    })
    rows.append(record)


def _login(rows: list, principal: str, source: str, target: str, when: int):
    record = _envelope(rows, "dc-01", "tc5_auth.v1", f"{principal}>{target}", when)
    record.update({
        "user_principal": principal,
        "source_ip": source,
        "target_host": target,
        "app": "windows-logon",
        "auth_result": "success",
    })
    rows.append(record)


def _process(rows: list, host: str, parent: str, image: str, when: int, integrity: str = "System"):
    record = _envelope(rows, "edr-01", "tc7_endpoint.v1", f"{host}>{image}", when)
    record.update({
        "hostname": host,
        "parent_image_path": parent,
        "image_path": image,
        "integrity_level": integrity,
        "signature_status": "Valid",
    })
    rows.append(record)


def _flows(rows: list, actor: Actor, index: int):
    """The hour before the campaign and the campaign's hour, from the actor's source."""
    base = f"10.30.{index}."

    if (actor.rising):
        for (count, minute) in enumerate((10, 20, 30)):
            _flow(rows, actor.source, f"{base}{count + 1}", at(LAST_DAY, CAMPAIGN_HOUR - 1, minute))

        for count in range(RISING_DESTINATIONS):
            _flow(rows,
                  actor.source,
                  f"{base}{100 + count}",
                  at(LAST_DAY, CAMPAIGN_HOUR, actor.fanout_minute, count * 10))
    else:
        for hour in (CAMPAIGN_HOUR - 1, CAMPAIGN_HOUR):
            for count in range(FLAT_DESTINATIONS):
                _flow(rows, actor.source, f"{base}{count + 1}", at(LAST_DAY, hour, actor.fanout_minute, count * 10))


def build_corpus() -> dict[str, pd.DataFrame]:
    """The campaign, split by collector: flows, logins and process starts."""
    flows: list = []
    logins: list = []
    processes: list = []

    for day in range(LAST_DAY + 1):
        for (index, actor) in enumerate(ACTORS.values()):
            # Every principal's working day starts with a login to their own server from their own address.
            _login(logins, actor.principal, actor.source, actor.home, at(day, 9, index))

        for host in servers():
            _process(processes, host, SERVICES, SVCHOST, at(day, 9, 30))
            _process(processes, host, SVCHOST, WMI, at(day, 9, 30))

    for (index, actor) in enumerate(ACTORS.values()):
        _flows(flows, actor, index)
        _login(logins,
               actor.principal,
               actor.login_source,
               actor.target,
               at(LAST_DAY, CAMPAIGN_HOUR, actor.login_minute))
        host = actor.target.lower()

        if (actor.process_minute is None):
            _process(processes, host, SVCHOST, WMI, at(LAST_DAY, CAMPAIGN_HOUR, 20))
        else:
            _process(processes,
                     host,
                     SERVICES,
                     novel_image(actor.principal),
                     at(LAST_DAY, CAMPAIGN_HOUR, actor.process_minute))

    def frame(rows: list) -> pd.DataFrame:
        result = pd.DataFrame(rows).sort_values(["event_time", "collector_seq"], kind="mergesort")
        result = result.reset_index(drop=True)
        result["collector_seq"] = range(len(result))

        return result

    return {FLOW_CLASS: frame(flows), AUTH_CLASS: frame(logins), PROCESS_CLASS: frame(processes)}


def build_pipeline_config(execution_mode=None) -> Config:
    """
    A pipeline configuration, defaulting to CPU mode and importable without a GPU.

    Parameters
    ----------
    execution_mode : `morpheus.config.ExecutionMode`, optional
        Mode to build for. Defaults to CPU, which is what the golden file is an artifact of.
    """
    from morpheus.config import CppConfig  # pylint: disable=import-outside-toplevel
    from morpheus.config import ExecutionMode  # pylint: disable=import-outside-toplevel

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.execution_mode = ExecutionMode.CPU if execution_mode is None else execution_mode

    return config


def _stages(config: Config, telemetry_class: str) -> tuple:
    """Each layer's own stages, its OSI layer and the entity it is sealed on."""
    if (telemetry_class == FLOW_CLASS):
        return ([
            TC3CardinalityStage(config, window_seconds=PERIOD_SECONDS),
            TC3ReachStage(config, window_seconds=PERIOD_SECONDS),
            TC3BeaconStage(config, window_seconds=2 * PERIOD_SECONDS),
            TC3TtlStage(config, window_seconds=2 * PERIOD_SECONDS),
        ],
                3, ["src_ip"])

    if (telemetry_class == AUTH_CLASS):
        return ([TC5NoveltyStage(config, target_host_column="target_host")], 5, ["user_principal"])

    return ([TC0EnrichStage(config, store=build_store(), entity_column="hostname"), TC7EndpointStage(config)],
            7, ["hostname"])


def _collect(sink: InMemorySinkStage) -> pd.DataFrame:
    frames = []

    for message in sink.get_messages():
        meta = message.payload() if hasattr(message, "payload") else message
        frame = meta.copy_dataframe()
        frames.append(frame.to_pandas() if hasattr(frame, "to_pandas") else frame)

    if (len(frames) == 0):
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def run_class(config: Config,
              telemetry_class: str,
              dataframes: list[pd.DataFrame],
              impose_order: bool = True) -> pd.DataFrame:
    """Source, stamp, total order, the layer's stages, envelope, seal hourly, sink, for one collector."""
    (stages, osi_layer, entity_columns) = _stages(config, telemetry_class)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    for stage in stages:
        pipe.add_stage(stage)

    pipe.add_stage(EnvelopeStampStage(config, osi_layer=osi_layer, entity_columns=entity_columns))
    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=PERIOD_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        entity_key_column="entity_key"))

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    frame = _collect(sink)
    frame["telemetry_class"] = telemetry_class
    frame["row_key"] = frame["event_uid"]

    return frame


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True) -> pd.DataFrame:
    """
    Each collector through its own layer's pipeline, and the three outputs as one canonical frame.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    corpus : dict
        The frames from `build_corpus`, possibly permuted.
    batches : dict, optional
        How each collector's frame is split across source frames. Defaults to one frame each.
    impose_order : bool, default = True
        Place `TotalOrderStage` ahead of the stateful stages.

    Returns
    -------
    `pandas.DataFrame`
        The three classes' output, tagged with `telemetry_class`, keyed by `row_key`, canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    frames = [run_class(config, name, batches[name], impose_order=impose_order) for name in CLASSES]

    return canonicalize(pd.concat(frames, ignore_index=True), key_columns=KEY_COLUMNS, ignore_columns=IGNORE_COLUMNS)


def render(result: pd.DataFrame) -> str:
    """The canonical CSV rendering the golden file holds."""
    return result.to_csv(index=False, lineterminator="\n")


# --- R-C-001, evaluated the way its search evaluates it ------------------------------------------------------------


def _truthy(value) -> bool:
    return False if pd.isna(value) else bool(value)


def fanout_rises(result: pd.DataFrame) -> pd.DataFrame:
    """Step 1: the first flow per source and hour whose fan-out exceeds the source's peak in the hour before.

    A source with no flows in the hour before has nothing to rise above and is not counted.
    """
    flows = result[(result["telemetry_class"] == FLOW_CLASS) & result["dsts_per_src"].notna()].copy()
    flows["window_id"] = flows["window_id"].astype("int64")
    flows["dsts_per_src"] = flows["dsts_per_src"].astype("int64")
    peaks = flows.groupby(["src_ip", "window_id"])["dsts_per_src"].max().rename("previous_peak").reset_index()
    peaks["window_id"] = peaks["window_id"] + 1
    flows = flows.merge(peaks, on=["src_ip", "window_id"], how="inner")
    flows = flows[flows["dsts_per_src"] > flows["previous_peak"]]

    return (flows.sort_values("event_time").groupby(["src_ip", "window_id"], as_index=False).first()[[
        "src_ip", "event_time", "dsts_per_src", "previous_peak", "lineage_id"
    ]])


def lateral_movement(result: pd.DataFrame,
                     rising: bool = True,
                     new_host: bool = True,
                     novel_process: bool = True,
                     same_source: bool = True,
                     ordered: bool = True,
                     window_seconds: typing.Optional[int] = CHAIN_WINDOW_SECONDS,
                     tolerance_seconds: int = JOIN_TOLERANCE_SECONDS) -> dict:
    """(source, principal, host) chains R-C-001 fires on, each with its three step times.

    Every condition can be switched off, which is how the counterfactual checks ask what a control would have done
    without the one condition that stops it.
    """
    tolerance_ns = tolerance_seconds * NS_PER_SECOND

    if (rising):
        starts = fanout_rises(result)
    else:
        # Without the rise, a source's first flow in every hour starts a chain.
        flows = result[result["telemetry_class"] == FLOW_CLASS]
        starts = (flows.sort_values("event_time").groupby(["src_ip", "window_id"],
                                                          as_index=False).first()[["src_ip", "event_time"]])

    logins = result[(result["telemetry_class"] == AUTH_CLASS) & (result["auth_result"] == "success")]

    if (new_host):
        logins = logins[logins["target_host_first_seen"].map(_truthy)]

    processes = result[result["telemetry_class"] == PROCESS_CLASS]

    if (novel_process):
        processes = processes[processes["endpoint_pair_novel"].map(_truthy)]

    fired: dict = {}

    for (_, start) in starts.iterrows():
        t1 = int(start["event_time"])
        candidates = logins if not same_source else logins[logins["source_ip"] == start["src_ip"]]

        for (_, login) in candidates.iterrows():
            t2 = int(login["event_time"])

            if (ordered and t2 < t1 - tolerance_ns):
                continue

            host = str(login["target_host"]).lower()

            for (_, process) in processes[processes["hostname"].str.lower() == host].iterrows():
                t3 = int(process["event_time"])

                if (ordered and t3 < t2 - tolerance_ns):
                    continue

                if (window_seconds is not None and max(t2, t3) - t1 > window_seconds * NS_PER_SECOND):
                    continue

                key = (start["src_ip"], login["user_principal"], host)
                fired.setdefault(key, (t1, t2, t3))

    return fired
