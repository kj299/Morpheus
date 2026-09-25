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
The layer 7 SaaS sub-class, `tc7_saas`, and its composed pipeline, for the determinism harness.

The two SaaS rules are the first in this fork that read the TC-0 context store, so the pipeline enriches every
operation twice before measuring it -- the principal's profile and group memberships, and the target object's data
classification -- both as known at the operation's own time. It then seals hourly windows as the other layer 7
classes do, and weekly windows behind them, Monday to Monday UTC, over which
{py:class}`~morpheus.stages.telemetry.tc5_drift_stage.TC5DriftStage` measures whether a principal's breadth is
rising, exactly as it measures a layer 5 score over days.

**Each condition of both rules has a benign case in this corpus where it is the one condition that keeps the rule
quiet.** R-B-L7-002 is a multiple of the principal's own history for the operation, and the history must exist:

- an **exfiltrator** exports eighty times their own 99th percentile from a restricted object, and fires -- while
  running queries of two to three thousand records all month, which a baseline pooling operations would have let
  excuse the export;
- the **same multiple from a public object** fires too, at a quarter of the severity -- the classification is a
  weight on the notable, not a bar the export has to clear;
- the same multiple from an **object the inventory has never heard of** fires at the default severity, flagged;
- a principal whose **large exports are routine** exports just as much and is well inside their own baseline;
- a principal **exporting just under five times** their baseline stays quiet;
- a **newcomer** with twenty exports behind them has no baseline, because by nearest rank a 99th percentile of
  twenty values is the largest of them;
- and an export that **failed** read nothing and is not measured.

R-P-L7-006 is breadth rising for four consecutive weeks with the role unchanged:

- a **creeper** reaches one more object type each week for four weeks, groups untouched, and is watchlisted;
- a principal with the **same rise and a role change** in the middle of it is quiet -- new duties explain new
  objects;
- a **short rise** of two weeks is quiet;
- a principal who has **always reached everything** has breadth and no rise;
- and a principal whose **role change was recorded late** is watchlisted, because on every week of the rise the
  store said the role was unchanged. That is the event-time knowledge the enrichment defaults to, and the harness
  asserts what the latest knowledge would have said instead.
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
from morpheus.stages.telemetry.tc5_drift_stage import TC5DriftStage
from morpheus.stages.telemetry.tc7_saas_stage import DEFAULT_WEEK_EPOCH
from morpheus.stages.telemetry.tc7_saas_stage import WEEK_SECONDS
from morpheus.stages.telemetry.tc7_saas_stage import TC7SaasStage
from morpheus.utils import bitemporal
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
ENTITY_COLUMNS = ["user_principal"]
SAAS_CLASS = "tc7_saas"

WEEK_PREFIX = "week_"
"""Prefix of the weekly seal's columns, so they sit beside the hourly seal's rather than over them."""

# The rules' own thresholds, stated once so the corpus is built to clear them deliberately.
RECORD_MULTIPLE = 5.0
RISING_WEEKS = 4
SEVERITY = {"restricted": 75, "confidential": 60, "internal": 40, "public": 20}
UNCLASSIFIED_SEVERITY = 40

DAY_SECONDS = 86400
FIRST_MONDAY_DAY = 4
"""1970-01-05, the Monday weeks are counted from. The corpus starts on it so week 0 is a whole week."""

CONTEXT_RECORDED_DAY = 1
"""The context store's first snapshot, recorded before the first operation so every operation can see it."""

# Bulk exporters.
EXFILTRATOR = "nate@example.com"
PUBLIC_EXPORTER = "olga@example.com"
UNKNOWN_EXPORTER = "pat@example.com"
ROUTINE_EXPORTER = "quin@example.com"
NEAR_EXPORTER = "sam@example.com"
NEWCOMER = "rosa@example.com"
FAILED_EXPORTER = "tom@example.com"

# Breadth.
CREEPER = "ivy@example.com"
ROLE_CHANGER = "jack@example.com"
LATE_ROLE_CHANGER = "kate@example.com"
SHORT_RISE = "leo@example.com"
EVERYTHING = "mia@example.com"

# Target objects and their classification in the inventory.
RESTRICTED_OBJECT = "crm-accounts"
PUBLIC_OBJECT = "price-list"
UNKNOWN_OBJECT = "legacy-share"
CONFIDENTIAL_OBJECT = "hr-records"
ROUTINE_OBJECT = "weekly-reports"

OBJECT_CLASSIFICATION = {
    RESTRICTED_OBJECT: "restricted",
    PUBLIC_OBJECT: "public",
    CONFIDENTIAL_OBJECT: "confidential",
    ROUTINE_OBJECT: "internal",
}

EXPORT = "Report.Export"
READ = "Record.Read"
QUERY = "Report.Query"

PRIOR_EXPORTS = 110
"""Exports each baselined exporter has made before the one that matters. Past the hundred a baseline needs."""

NEWCOMER_EXPORTS = 20

BIG_EXPORT_DAY = 33
"""The day the exports that matter happen: the Tuesday of week 4, after four weeks of history that all fall inside
the thirty-day baseline window, and after the first Monday so no export precedes the context store's snapshot."""

OBJECT_TYPES = ["Account", "Contact", "Opportunity", "Case", "Invoice", "Contract", "Lead", "Payroll", "Campaign"]

# Distinct object types each breadth principal reaches in each of weeks 0 to 7.
BREADTH = {
    CREEPER: [2, 2, 2, 2, 3, 4, 5, 6],
    ROLE_CHANGER: [2, 2, 2, 2, 3, 4, 5, 6],
    LATE_ROLE_CHANGER: [2, 2, 2, 2, 3, 4, 5, 6],
    SHORT_RISE: [2, 2, 2, 2, 2, 3, 4, 3],
    EVERYTHING: [8, 8, 8, 8, 8, 8, 8, 8],
}

ROLE_CHANGE_WEEK = 5
"""Jack's and Kate's roles change at the start of week 5, in the middle of their rise."""

LATE_RECORDING_WEEK = 9
"""Kate's change reaches the directory after the corpus ends."""


def at(day: float, hour: float = 0) -> int:
    """An instant `day` days and `hour` hours into the corpus, in nanoseconds since the epoch."""
    return int((day * DAY_SECONDS + hour * 3600) * NS_PER_SECOND)


def week_start(week: int) -> int:
    """Monday 00:00 UTC of week `week`, counted from the corpus's first Monday."""
    return at(FIRST_MONDAY_DAY + 7 * week)


def _groups(principal: str) -> list:
    return [
        make_version(bitemporal.MEMBERSHIP,
                     principal,
                     0,
                     None,
                     at(CONTEXT_RECORDED_DAY),
                     values={"group_name": "sales-users"},
                     key_parts=("sales-users", ))
    ]


def identity_versions() -> list:
    """Every principal in sales-users from the start; Jack and Kate move to sales-managers in week 5."""
    principals = sorted(
        set(BREADTH)
        | {EXFILTRATOR, PUBLIC_EXPORTER, UNKNOWN_EXPORTER, ROUTINE_EXPORTER, NEAR_EXPORTER, NEWCOMER, FAILED_EXPORTER})
    versions = []

    for principal in principals:
        versions.append(
            make_version("profile",
                         principal,
                         0,
                         None,
                         at(CONTEXT_RECORDED_DAY),
                         values={
                             "department": "Sales", "manager": "heidi@example.com", "employment_status": "active"
                         }))
        versions += _groups(principal)

    for (principal, recorded) in ((ROLE_CHANGER, week_start(ROLE_CHANGE_WEEK)), (LATE_ROLE_CHANGER,
                                                                                 week_start(LATE_RECORDING_WEEK))):
        change = week_start(ROLE_CHANGE_WEEK)
        versions.append(
            make_version(bitemporal.MEMBERSHIP,
                         principal,
                         change,
                         None,
                         recorded,
                         bitemporal.RETRACT,
                         values={"group_name": "sales-users"},
                         key_parts=("sales-users", )))
        versions.append(
            make_version(bitemporal.MEMBERSHIP,
                         principal,
                         change,
                         None,
                         recorded,
                         values={"group_name": "sales-managers"},
                         key_parts=("sales-managers", )))

    return versions


def asset_versions() -> list:
    """The inventory's classification of every SaaS object but the one it has never heard of."""
    return [
        make_version("asset",
                     name,
                     0,
                     None,
                     at(CONTEXT_RECORDED_DAY),
                     values={
                         "owner": "frank@example.com",
                         "owning_team": "sales-ops",
                         "criticality": "high" if classification in ("restricted", "confidential") else "low",
                         "data_classification": classification,
                         "peer_group": None,
                     }) for (name, classification) in sorted(OBJECT_CLASSIFICATION.items())
    ]


def build_stores() -> tuple:
    """The identity and asset stores the enrichment reads."""
    return (BitemporalStore("identity", identity_versions()), BitemporalStore("asset", asset_versions()))


def _operation(rows: list,
               principal: str,
               operation: str,
               obj: str,
               object_type: str,
               records: int,
               when: int,
               result: str = "success"):
    rows.append({
        "collector_id": "saas-audit-01",
        "schema_version": "tc7_saas.v1",
        "origin_hash": f"{principal}>{operation}>{obj}@{when}",
        "collector_seq": len(rows),
        "event_time": when,
        "user_principal": principal,
        "operation": operation,
        "target_object": obj,
        "target_object_type": object_type,
        "record_count": records,
        "result": result,
        "client_app": "web",
    })


def _exports(rows: list, principal: str, sizes: list, offset_minutes: int):
    """A history of exports, one every six hours, ending the day before the one that matters."""
    first = at(BIG_EXPORT_DAY - 1) - len(sizes) * 6 * 3600 * NS_PER_SECOND

    for (index, size) in enumerate(sizes):
        _operation(rows,
                   principal,
                   EXPORT,
                   ROUTINE_OBJECT,
                   "Report",
                   size,
                   first + index * 6 * 3600 * NS_PER_SECOND + offset_minutes * 60 * NS_PER_SECOND)


def _build_bulk(rows: list):
    ordinary = [40 + (index * 4) % 21 for index in range(PRIOR_EXPORTS)]  # 40 to 60 records
    heavy = [4000 + (index * 97) % 1001 for index in range(PRIOR_EXPORTS)]  # 4000 to 5000 records
    near = [800 + (index * 13) % 201 for index in range(PRIOR_EXPORTS)]  # 800 to 1000 records

    _exports(rows, EXFILTRATOR, ordinary, 1)
    _exports(rows, PUBLIC_EXPORTER, ordinary, 2)
    _exports(rows, UNKNOWN_EXPORTER, ordinary, 3)
    _exports(rows, ROUTINE_EXPORTER, heavy, 4)
    _exports(rows, NEAR_EXPORTER, near, 5)
    _exports(rows, FAILED_EXPORTER, ordinary, 6)
    _exports(rows, NEWCOMER, ordinary[:NEWCOMER_EXPORTS], 7)

    # The exfiltrator also runs large queries all month. Pooled with the exports, they would set a baseline the
    # export that matters sits comfortably inside; keyed apart, the export is measured against exports.
    first = at(BIG_EXPORT_DAY - 1) - PRIOR_EXPORTS * 6 * 3600 * NS_PER_SECOND

    for index in range(PRIOR_EXPORTS):
        _operation(rows,
                   EXFILTRATOR,
                   QUERY,
                   ROUTINE_OBJECT,
                   "Report",
                   2000 + (index * 9) % 1001,
                   first + index * 6 * 3600 * NS_PER_SECOND + 3 * 3600 * NS_PER_SECOND)

    day = at(BIG_EXPORT_DAY, 10)
    minute = 60 * NS_PER_SECOND

    _operation(rows, EXFILTRATOR, EXPORT, RESTRICTED_OBJECT, "Account", 5000, day + 1 * minute)
    _operation(rows, PUBLIC_EXPORTER, EXPORT, PUBLIC_OBJECT, "PriceBook", 5000, day + 2 * minute)
    _operation(rows, UNKNOWN_EXPORTER, EXPORT, UNKNOWN_OBJECT, "Document", 5000, day + 3 * minute)
    _operation(rows, ROUTINE_EXPORTER, EXPORT, CONFIDENTIAL_OBJECT, "Employee", 5000, day + 4 * minute)
    # Just under five times the largest of 800 to 1000.
    _operation(rows, NEAR_EXPORTER, EXPORT, RESTRICTED_OBJECT, "Account", 4900, day + 5 * minute)
    _operation(rows, FAILED_EXPORTER, EXPORT, RESTRICTED_OBJECT, "Account", 9000, day + 6 * minute, "denied")
    _operation(rows, NEWCOMER, EXPORT, RESTRICTED_OBJECT, "Account", 5000, day + 7 * minute)


def _build_breadth(rows: list):
    for (offset, (principal, weekly)) in enumerate(sorted(BREADTH.items())):
        for (week, types) in enumerate(weekly):
            for (index, object_type) in enumerate(OBJECT_TYPES[:types]):
                # Each type reached twice a week, on Tuesday and Thursday, so a week's count is reached mid-week
                # and restated rather than first seen on the last day. A principal's reads of one day fall in
                # one hour, five minutes apart, which is what gives the permutation check something to reorder.
                for day in (1, 3):
                    when = week_start(week) + at(day, 9 + index / 12) + offset * 60 * NS_PER_SECOND
                    _operation(rows, principal, READ, f"{object_type.lower()}-{week}", object_type, 1 + index, when)


def build_corpus() -> dict[str, pd.DataFrame]:
    """The seeded corpus: one frame of SaaS audit records."""
    rows: list = []
    _build_bulk(rows)
    _build_breadth(rows)

    frame = pd.DataFrame(rows).sort_values("event_time", kind="mergesort").reset_index(drop=True)
    frame["collector_seq"] = range(len(frame))

    return {SAAS_CLASS: frame}


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


def _collect(sink: InMemorySinkStage) -> pd.DataFrame:
    """Everything the sink received, as one host frame."""
    frames = []

    for message in sink.get_messages():
        meta = message.payload() if hasattr(message, "payload") else message
        frame = meta.copy_dataframe()
        frames.append(frame.to_pandas() if hasattr(frame, "to_pandas") else frame)

    if (len(frames) == 0):
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True,
                 knowledge: str = "event") -> pd.DataFrame:
    """
    Source, stamp, total order, enrich twice, measure, envelope, seal hourly, seal weekly, trajectory, sink.

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
    knowledge : str, default = "event"
        What the enrichment knows: the context as recorded by each operation's time, or everything.

    Returns
    -------
    `pandas.DataFrame`
        The class's output, tagged with `telemetry_class`, keyed by `row_key`, canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    (identity, asset) = build_stores()

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=batches[SAAS_CLASS]))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    pipe.add_stage(TC0EnrichStage(config, store=identity, entity_column="user_principal", knowledge=knowledge))
    pipe.add_stage(
        TC0EnrichStage(config, store=asset, entity_column="target_object", knowledge=knowledge, prefix="ctx_object_"))
    pipe.add_stage(TC7SaasStage(config))
    pipe.add_stage(EnvelopeStampStage(config, osi_layer=OSI_LAYER, entity_columns=ENTITY_COLUMNS))
    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=PERIOD_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        entity_key_column="entity_key"))
    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=WEEK_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        epoch=DEFAULT_WEEK_EPOCH,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        column_prefix=WEEK_PREFIX))
    pipe.add_stage(
        TC5DriftStage(config,
                      score_column="saas_object_types_in_week",
                      window_column=f"{WEEK_PREFIX}window_id",
                      aggregate="max"))

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    frame = _collect(sink)
    frame["telemetry_class"] = SAAS_CLASS
    frame["row_key"] = frame["event_uid"]

    return canonicalize(frame, key_columns=KEY_COLUMNS, ignore_columns=IGNORE_COLUMNS)


def render(result: pd.DataFrame) -> str:
    """The canonical CSV rendering the golden file holds."""
    return result.to_csv(index=False, lineterminator="\n")
