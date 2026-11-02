# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The layer 7 endpoint sub-class, `tc7_endpoint`, and its composed pipeline, for the determinism harness.

R-B-L7-004 asks whether a process's `(parent_image_path, image_path)` pair is new to its host and to the host's peer
group in thirty days, and the peer group is TC-0's: the pipeline enriches every process with its host's asset record
as known at the process's time, then judges the pair, stamps and seals hourly windows keyed on the host.

The corpus is forty-one days of EDR process starts from seven hosts: three finance workstations and two build
servers in two peer groups, a kiosk the inventory knows but has put in no group, and a lab machine the inventory has
never heard of, which appears on day 36. Every host starts its day with a burst of processes in the same second,
because that is what an EDR log looks like at logon. On day 40, eight processes decide the rule:

- a finance workstation's **word processor starts PowerShell**, at high integrity. Nothing in the finance group has
  done that, and fires -- although a build server runs the same pair every day, because a pairing that is routine
  in another group is no excuse in this one;
- a **remote-access tool** is started at system integrity on a workstation that last ran it thirty-eight days ago, and
  fires, because a pair not seen in thirty days is as unexplained as one never seen;
- the **kiosk starts `mshta.exe`**, and fires on its own history, flagged as judged without a peer group;
- a build server's **`git` starts `curl`** with no integrity level reported, and fires at the default weight, flagged
  as unweighted;
- a **build server compiles** with a pair it has never run and its peer runs every day, and is quiet on the peer
  group alone;
- a workstation starts an **editor installed under a different user's profile** than the peer who runs it daily, and
  is quiet because per-user folders are collapsed;
- the **kiosk starts Notepad**, which it last did on day 20, and is quiet on its own history;
- and the **lab machine**, four days old, starts `nmap`, and is quiet because no comparison is possible yet.

**One host's breadth creeps.** From day 27 the second build server takes up one more each day of the five tool pairs
the first runs every morning, and keeps them: never a pair R-B-L7-004 can call new, since its peer runs them all, and
a host starting more distinct pairs every day than the day before, which is R-P-L7-011's trajectory. The daily
distinct count is measured over a second seal, by day, behind the hourly one. The creep ends nine days before the
decisive day, so the watchlist entries it writes are a month old when a search head's expiry job reads them, as the
other trajectories' entries in their corpora's first weeks are, while R-B-L7-004, which reads the list before the
expiry runs, still raises the build server's curl by the watchlist's boost.

**One host, two spellings.** From day 20 the EDR on the first finance workstation reports it as `FIN-01` where it had
reported `fin-01`, as an agent upgrade does. Keyed as text that is a new host with no history and no inventory record,
judged alone and immature for a week; keyed with the shared host rule it is the same workstation throughout, which is
what the harness asserts, and its Word-starts-PowerShell on day 40 is judged against its peers as before.

**Severity is integrity times criticality.** The inventory calls the build servers high, the finance workstations
medium and the kiosk low, and R-B-L7-004 scales each notable's integrity weight by that, so the four that fire do so
at four different severities, each asserted.
"""

import typing

import pandas as pd

from morpheus.compositions import envelope
from morpheus.compositions import stamping
from morpheus.compositions.corpus_pipeline import _collect
from morpheus.compositions.corpus_pipeline import (  # noqa: F401 pylint: disable=unused-import  # re-exported for callers
    build_pipeline_config, )
from morpheus.compositions.resumable import ResumableLinearPipeline
from morpheus.config import Config
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.collector_sequence_stage import CollectorSequenceStage
from morpheus.stages.lineage.determinism_stamp_stage import DeterminismStampStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.stages.telemetry.tc5_drift_stage import TC5DriftStage
from morpheus.stages.telemetry.tc7_endpoint_stage import HOST_KEY
from morpheus.stages.telemetry.tc7_endpoint_stage import PAIR
from morpheus.stages.telemetry.tc7_endpoint_stage import TC7EndpointStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900
ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []
OSI_LAYER = 7
ENTITY_COLUMNS = ["hostname"]
ENDPOINT_CLASS = "tc7_endpoint"
WINDOW_DAYS = 30
WARMUP_DAYS = 7
DAY_SECONDS = 86400
DRIFT_PREFIX = "day_"
SETTINGS = {
    "period_seconds": PERIOD_SECONDS,
    "lateness_seconds": LATENESS_SECONDS,
    "window_days": WINDOW_DAYS,
    "warmup_days": WARMUP_DAYS,
    "day_seconds": DAY_SECONDS,
}
"""The settings that decide this corpus's output, digested into `config_hash` by `stamping.envelope_for`."""
RULES = ("R-B-L7-004", "R-P-L7-011")
"""The shipped rules that read this corpus's columns; their thresholds are folded into `pipeline_fingerprint`."""
CONTEXT_RECORDED_DAY = 0
FINANCE_GROUP = "finance-workstations"
BUILD_GROUP = "build-servers"
FINANCE_HOSTS = {"fin-01": "alice", "fin-02": "bob", "fin-03": "carol"}
BUILD_HOSTS = ("build-01", "build-02")
KIOSK = "kiosk-01"
DEVICE_ROLES = {"finance": "workstation", "build": "server", "kiosk": "kiosk"}
CRITICALITY = {"finance": "medium", "build": "high", "kiosk": "low"}
OS_VERSIONS = {"finance": "10.0.19045", "build": "10.0.20348", "kiosk": "10.0.19044"}


def at(day: float, hour: float = 0, minute: float = 0, second: float = 0) -> int:
    """An instant in the corpus, in nanoseconds since the epoch."""
    return int((day * DAY_SECONDS + hour * 3600 + minute * 60 + second) * NS_PER_SECOND)


def asset_versions() -> list:
    """The inventory: the finance and build groups, and the kiosk with no group. The lab machine is not in it."""
    versions = []
    owners = dict(FINANCE_HOSTS)

    for host in sorted(set(FINANCE_HOSTS) | set(BUILD_HOSTS) | {KIOSK}):
        group = FINANCE_GROUP if host in FINANCE_HOSTS else (BUILD_GROUP if host in BUILD_HOSTS else None)
        kind = "finance" if host in FINANCE_HOSTS else ("build" if host in BUILD_HOSTS else "kiosk")
        versions.append(
            make_version("asset",
                         host,
                         0,
                         None,
                         at(CONTEXT_RECORDED_DAY),
                         values={
                             "owner": f"{owners[host]}@example.com" if host in owners else "it-ops@example.com",
                             "owning_team": "finance" if host in FINANCE_HOSTS else "platform",
                             "criticality": CRITICALITY[kind],
                             "data_classification": "confidential" if host in FINANCE_HOSTS else "internal",
                             "peer_group": group,
                             "device_role": DEVICE_ROLES[kind],
                             "os_family": "windows",
                             "os_version": OS_VERSIONS[kind],
                         }))

    return versions


def build_store() -> BitemporalStore:
    """The asset store the enrichment reads."""
    return BitemporalStore("asset", asset_versions())


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True) -> pd.DataFrame:
    """
    Source, stamp, total order, enrich with the host's asset record, judge, envelope, seal hourly, sink.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    corpus : dict
        The frames from `build_corpus`, possibly permuted.
    batches : dict, optional
        How the corpus is split across source frames. Defaults to one frame.
    impose_order : bool, default = True
        Place `TotalOrderStage` ahead of the stateful stages.

    Returns
    -------
    `pandas.DataFrame`
        The class's output, tagged with `telemetry_class`, keyed by `row_key`, canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    # Validated once per class, ahead of every stage: ingest_time stamped, the envelope checked, untrusted clocks
    # taken out of the stream and kept.
    (batches, quarantined) = envelope.validate(config, batches, ID_COLUMNS)

    pipe = ResumableLinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=batches[ENDPOINT_CLASS]))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    pipe.add_stage(CollectorSequenceStage(config))

    pipe.add_stage(
        TC0EnrichStage(config, store=build_store(), entity_column="hostname", entity_normalization="hostname"))
    pipe.add_stage(TC7EndpointStage(config, window_days=WINDOW_DAYS, warmup_days=WARMUP_DAYS))
    pipe.add_stage(DeterminismStampStage(config, envelope=stamping.envelope_for(ENDPOINT_CLASS, SETTINGS, rules=RULES)))
    pipe.add_stage(EnvelopeStampStage(config, osi_layer=OSI_LAYER, entity_columns=ENTITY_COLUMNS))
    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=PERIOD_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        entity_key_column="entity_key"))
    # The daily trajectory R-P-L7-011 reads: a second seal over days behind the hourly one, and the drift stage over
    # the distinct process pairs each host started in the day.
    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=DAY_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        column_prefix=DRIFT_PREFIX))
    pipe.add_stage(
        TC5DriftStage(config,
                      entity_column=HOST_KEY,
                      score_column=PAIR,
                      window_column=f"{DRIFT_PREFIX}window_id",
                      aggregate="distinct"))

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    frame = _collect(sink)
    frame["telemetry_class"] = ENDPOINT_CLASS
    frame["row_key"] = frame["event_uid"]

    return canonicalize(envelope.with_quarantine(frame, quarantined),
                        key_columns=KEY_COLUMNS,
                        ignore_columns=IGNORE_COLUMNS)


def render(result: pd.DataFrame) -> str:
    """The canonical CSV rendering the golden file holds."""
    return result.to_csv(index=False, lineterminator="\n")
