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
"""
The estate's CMDB: which application answers at each address and port, for the layer 4, 6 and 7 corpora.

Moved out of `morpheus.compositions.application_context` under plan section 4.5b, which left the enrichment there and
put the data here.
"""

import pandas as pd

from morpheus.compositions.application_context import produce
from morpheus.config import Config
from morpheus.utils import bitemporal
from morpheus.utils.bitemporal import BitemporalStore

APPLICATIONS = (
    ("10.0.1.20", 445, "file-shares", "it-infrastructure", "svc-files", "medium", "internal"),
    ("10.0.1.20", 443, "file-shares-web", "it-infrastructure", "svc-files", "medium", "internal"),
    ("10.0.1.30", 445, "backup-vault", "it-infrastructure", "svc-backup", "high", "restricted"),
    ("10.0.1.40", 443, "order-api", "commerce", "svc-orders", "high", "confidential"),
    ("10.0.1.50", 443, "badge-controller", "facilities", "svc-badges", "medium", "internal"),
    ("10.0.1.80", 443, "intranet-web", "communications", "svc-intranet", "low", "internal"),
    ("10.0.2.10", 443, "corp-edge-proxy", "platform", "svc-edge", "high", "internal"),
    ("10.0.2.20", 8443, "metrics-collector", "observability", "svc-metrics", "medium", "internal"),
    ("10.0.2.40", 443, "ci-runner", "engineering-productivity", "svc-ci", "medium", "internal"),
)
"""Address, port, application, owning team, service account, criticality and data classification.

Two applications share the file server on different ports, and the reverse proxy fronts two named services on one
port, which is what keys an application on `address:port` rather than on either alone."""


def records() -> pd.DataFrame:
    """The CMDB's records, each holding from the start of the corpora and recorded then."""
    rows = []

    for (address, port, name, team, account, criticality, classification) in APPLICATIONS:
        rows.append({
            "server_address": address,
            "server_port": port,
            "name": name,
            "owning_team": team,
            "service_account": account,
            "criticality": criticality,
            "data_classification": classification,
            bitemporal.VALID_FROM: 0,
            bitemporal.VALID_TO: None,
            bitemporal.RECORDED_AT: 0,
            bitemporal.CHANGE: bitemporal.ASSERT,
        })

    frame = pd.DataFrame(rows)
    frame[bitemporal.VALID_TO] = frame[bitemporal.VALID_TO].astype("Int64")

    return frame


def store(config: Config) -> BitemporalStore:
    """The store a consumer of `context:application` rebuilds from the producer's output."""
    return BitemporalStore.from_records("application", produce(config, records()).to_dict("records"))
