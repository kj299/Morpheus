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
The synthetic corpus `morpheus.compositions.saas` is tested over, and the facts about it the harnesses assert.

Moved out of `tests/morpheus/determinism/saas_pipeline.py` under plan section 4.5, which put the composition in the
package and left the data here, where the tests, the sample and fixture generators and the experiments read it. The
composition module's docstring describes what this corpus plants and why.
"""

import pandas as pd

from morpheus.compositions import envelope
from morpheus.compositions.saas import BREADTH
from morpheus.compositions.saas import CONFIDENTIAL_OBJECT
from morpheus.compositions.saas import EXFILTRATOR
from morpheus.compositions.saas import FAILED_EXPORTER
from morpheus.compositions.saas import FIRST_MONDAY_DAY
from morpheus.compositions.saas import NEAR_EXPORTER
from morpheus.compositions.saas import NEWCOMER
from morpheus.compositions.saas import PUBLIC_EXPORTER
from morpheus.compositions.saas import PUBLIC_OBJECT
from morpheus.compositions.saas import RESTRICTED_OBJECT
from morpheus.compositions.saas import ROUTINE_EXPORTER
from morpheus.compositions.saas import ROUTINE_OBJECT
from morpheus.compositions.saas import SAAS_CLASS
from morpheus.compositions.saas import UNKNOWN_EXPORTER
from morpheus.compositions.saas import at
from morpheus.compositions.saas import week_start
from morpheus.utils.binding_table import NS_PER_SECOND

# The rules' own thresholds, stated once so the corpus is built to clear them deliberately.
RECORD_MULTIPLE = 5.0
RISING_WEEKS = 4
SEVERITY = {"restricted": 75, "confidential": 60, "internal": 40, "public": 20}
UNCLASSIFIED_SEVERITY = 40
UNKNOWN_OBJECT = "legacy-share"
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
# The second tenant.
DOUBLING_TENANT = "labs"
SYNC_PRINCIPAL = "sync@labs.example.com"
SYNC_OBJECT = "lab-dataset"
SYNC_RECORDS = 50
SYNC_HOURS = (2, 14)
"""The sync runs at these hours every day, each run in its own hourly window."""
SYNC_EXPORTS = 2
"""Exports a sync run makes: one per shard of the dataset, a minute apart."""
SYNC_DAYS = (FIRST_MONDAY_DAY + 1, 56)
"""The days the sync runs on, first included and last excluded: the whole corpus but its edges."""
DOUBLING_DAY = 54
"""From this day each run exports every shard twice, and the tenant's rows per window double."""
HEALTH_MIN_BUCKETS = 24
"""Windows a history needs before its baseline is published: twelve days of the sync's two runs a day."""


def _operation(rows: list,
               principal: str,
               operation: str,
               obj: str,
               object_type: str,
               records: int,
               when: int,
               result: str = "success",
               tenant_id: str = envelope.TENANT):
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
        "tenant_id": tenant_id,
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


def _build_sync(rows: list):
    """The second tenant's sync: the same exports at the same hours every day, twice as many from `DOUBLING_DAY`."""
    for day in range(*SYNC_DAYS):
        repeats = 2 if day >= DOUBLING_DAY else 1

        for hour in SYNC_HOURS:
            for index in range(SYNC_EXPORTS * repeats):
                _operation(rows,
                           SYNC_PRINCIPAL,
                           EXPORT,
                           SYNC_OBJECT,
                           "Dataset",
                           SYNC_RECORDS,
                           at(day, hour) + (30 + index) * 60 * NS_PER_SECOND,
                           tenant_id=DOUBLING_TENANT)


def build_corpus() -> dict[str, pd.DataFrame]:
    """The seeded corpus: one frame of SaaS audit records."""
    rows: list = []
    _build_bulk(rows)
    _build_breadth(rows)
    _build_sync(rows)

    frame = pd.DataFrame(rows).sort_values("event_time", kind="mergesort").reset_index(drop=True)
    frame["collector_seq"] = range(len(frame))

    return envelope.attach_all({SAAS_CLASS: frame})
