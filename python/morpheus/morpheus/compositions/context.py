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
The TC-0 context store, its three producers, and the point-in-time join, for the determinism harness.

TC-0 is not an OSI layer and has no rules of its own. What it has is a requirement the guide calls hard: every record
bitemporal, so that "was this user in Finance on March 3rd" can be answered both as it is understood now and as it was
understood on March 3rd. A corpus that only ever recorded facts on the day they became true would pass every test here
with a single time axis, so this one is built from the cases where the two axes disagree:

- a **move recorded on time**: a principal changes department and the source says so the same day. Both views agree,
  before and after; this is the control that proves the other cases differ for a reason;
- a **retroactive correction**: a department recorded wrongly from the start and put right three weeks later. Asked
  about the first week, the event-time view says what was believed then and the latest view says what was true;
- a **leaver recorded late**: a termination effective on day twelve that reached the source on day eighteen. For the
  six days between, anything the principal did was done by an active employee with their group memberships, as far
  as anyone could have known;
- a **contract that ended silently**: a principal simply absent from a full snapshot, which is how most HR exports
  report a departure. The snapshot diff turns the absence into a retraction, and nothing else in that snapshot -- a
  restatement of everything else, unchanged -- is recorded twice;
- a **reclassification recorded late**: a database declared restricted a week after it became so;
- a **peer group change recorded late**: a workstation that followed its owner into Finance a day before the
  inventory noticed;
- a **decommissioned host**, absent from a snapshot the way the departed contractor is;
- a **server reassigned and recorded late**: a billing service's port handed to a reporting service on day nine and
  the CMDB updated on day ten, so a connection on day nine was to billing as anyone knew then and to reporting as is
  now understood;
- and two records the store must refuse: one with no instant of recording, and one whose interval ends before it
  begins.

The probes are events at chosen instants, enriched twice -- once with what was known at the event's time and once
with everything -- so each case can be asserted as a pair of answers rather than one.
"""

import typing

import pandas as pd

from morpheus.compositions import stamping
from morpheus.compositions.corpus_pipeline import _collect
from morpheus.compositions.corpus_pipeline import (  # noqa: F401 pylint: disable=unused-import  # re-exported for callers
    build_pipeline_config, )
from morpheus.compositions.resumable import ResumableLinearPipeline
from morpheus.config import Config
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.determinism_stamp_stage import DeterminismStampStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc0_application_stage import DEFAULT_APPLICATION_COLUMNS
from morpheus.stages.telemetry.tc0_application_stage import TC0ApplicationStage
from morpheus.stages.telemetry.tc0_asset_stage import DEFAULT_ASSET_COLUMNS
from morpheus.stages.telemetry.tc0_asset_stage import TC0AssetStage
from morpheus.stages.telemetry.tc0_enrich_stage import KNOWLEDGE_MODES
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.stages.telemetry.tc0_identity_stage import DEFAULT_PROFILE_COLUMNS
from morpheus.stages.telemetry.tc0_identity_stage import TC0IdentityStage
from morpheus.utils import bitemporal
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.determinism import canonicalize

KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []
IDENTITY_CLASS = "tc0_identity"
ASSET_CLASS = "tc0_asset"
APPLICATION_CLASS = "tc0_application"
IDENTITY_PROBES = "tc0_identity_probes"
ASSET_PROBES = "tc0_asset_probes"
APPLICATION_PROBES = "tc0_application_probes"
PROBE_CLASSES = (IDENTITY_PROBES, ASSET_PROBES, APPLICATION_PROBES)
SETTINGS = {
    "profile_columns": list(DEFAULT_PROFILE_COLUMNS),
    "asset_columns": list(DEFAULT_ASSET_COLUMNS),
    "application_columns": list(DEFAULT_APPLICATION_COLUMNS),
}
"""The settings that decide the producers' output; a probe class adds the knowledge mode it was enriched under."""
PREFIX_APPLICATION = "ctx_application_"


def _run(config: Config,
         stage,
         dataframes: list[pd.DataFrame],
         telemetry_class: str,
         settings: dict = None) -> pd.DataFrame:
    """Source → the class's one stage → determinism stamp → sink. No sealing: context is a store, not a stream."""
    pipe = ResumableLinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(stage)
    pipe.add_stage(
        DeterminismStampStage(config,
                              envelope=stamping.envelope_for(telemetry_class,
                                                             SETTINGS if settings is None else settings)))
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    return _collect(sink)


def build_store(name: str, produced: pd.DataFrame) -> BitemporalStore:
    """Rebuild a store from a producer's output, the way a consumer of the sourcetype would."""
    return BitemporalStore.from_records(name, produced.to_dict("records"))


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None) -> pd.DataFrame:
    """
    Run the three producers, rebuild each store from what it emitted, and enrich the probes against each twice.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    corpus : dict
        The frames from `build_corpus`, possibly permuted.
    batches : dict, optional
        Per corpus class, how it is split across source frames. Defaults to one frame per class.

    Returns
    -------
    `pandas.DataFrame`
        The producers' output and the enriched probes, tagged with `telemetry_class`, keyed by `row_key`,
        canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    identity = _run(config, TC0IdentityStage(config), batches[IDENTITY_CLASS], IDENTITY_CLASS)
    asset = _run(config, TC0AssetStage(config), batches[ASSET_CLASS], ASSET_CLASS)
    application = _run(config, TC0ApplicationStage(config), batches[APPLICATION_CLASS], APPLICATION_CLASS)

    # Each probe class's store and how its probes name the entity: a principal, a host, or a server's address and
    # port, which the application kind is keyed on.
    stores = {
        IDENTITY_PROBES: (build_store("identity", identity), {
            "entity_column": "user_principal"
        }),
        ASSET_PROBES: (build_store("asset", asset), {
            "entity_column": "hostname"
        }),
        APPLICATION_PROBES: (build_store("application", application), {
            "entity_columns": ["dst_ip", "dst_port"], "prefix": PREFIX_APPLICATION, "set_kinds": {}
        }),
    }

    frames = []

    for (name, produced) in ((IDENTITY_CLASS, identity), (ASSET_CLASS, asset), (APPLICATION_CLASS, application)):
        produced["telemetry_class"] = name
        # A refused row has no identifier of its own, so it is keyed by its position in the source instead.
        produced["row_key"] = [
            uid if isinstance(uid, str) else f"refused-{seq}"
            for (uid, seq) in zip(produced[bitemporal.CONTEXT_UID], produced["source_seq"])
        ]
        frames.append(produced)

    for name in PROBE_CLASSES:
        (store, lookup) = stores[name]

        for knowledge in KNOWLEDGE_MODES:
            settings = {**SETTINGS, "knowledge": knowledge}
            enriched = _run(config,
                            TC0EnrichStage(config, store=store, knowledge=knowledge, **lookup),
                            [frame.copy() for frame in batches[name]],
                            name,
                            settings=settings)
            enriched["telemetry_class"] = name
            enriched["row_key"] = [f"{probe}@{knowledge}" for probe in enriched["probe_id"]]
            frames.append(enriched)

    combined = pd.concat(frames, ignore_index=True)
    # The probes carry no source position, and a column with gaps would otherwise widen to float and render as 4.0.
    combined["source_seq"] = combined["source_seq"].astype("Int64")

    return canonicalize(combined, key_columns=KEY_COLUMNS, ignore_columns=IGNORE_COLUMNS)


def render(result: pd.DataFrame) -> str:
    """The canonical CSV rendering the golden file holds."""
    return result.to_csv(index=False, lineterminator="\n")
