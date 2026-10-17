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
The estate's CMDB application-to-server mapping, and the enrichment that reads it, for the layer 4, 6 and 7 corpora.

Which application answered a connection is a question about the destination's address and port at the connection's
time, and the store that answers it is built the way a deployment would build it: the CMDB's records through
`TC0ApplicationStage`, and the store rebuilt from what the producer wrote, as a consumer of `context:application`
would. The context pipeline is where the store's bitemporal behaviour is asserted -- a server reassigned and
recorded late; here every record is known from the start, because what these corpora need is the join.

The mapping is the estate's, not one corpus's: the same file server sits under the transport corpus's handshakes and
the same reverse proxy under the service corpus's names, so one table serves all three.
"""

import pandas as pd
from resumable import ResumableLinearPipeline

from morpheus.config import Config
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc0_application_stage import DEFAULT_APPLICATION_COLUMNS
from morpheus.stages.telemetry.tc0_application_stage import TC0ApplicationStage
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.utils import bitemporal
from morpheus.utils.bitemporal import BitemporalStore

PREFIX = "ctx_application_"
"""What every column the enrichment writes starts with, so an application's context never collides with a host's."""

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


def produce(config: Config, frame: pd.DataFrame) -> pd.DataFrame:
    """The CMDB records through `TC0ApplicationStage`, as `context:application` carries them."""
    from host_frame import to_host_frame  # pylint: disable=import-outside-toplevel

    pipe = ResumableLinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[frame]))
    pipe.add_stage(TC0ApplicationStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    frames = [to_host_frame(message.copy_dataframe()) for message in sink.get_messages()]

    return pd.concat(frames, ignore_index=True)


def store(config: Config) -> BitemporalStore:
    """The store a consumer of `context:application` rebuilds from the producer's output."""
    return BitemporalStore.from_records("application", produce(config, records()).to_dict("records"))


def enrich_stage(config: Config, application_store: BitemporalStore) -> TC0EnrichStage:
    """The point-in-time join of a connection's `dst_ip:dst_port` to the application that answered it."""
    return TC0EnrichStage(config,
                          store=application_store,
                          entity_columns=["dst_ip", "dst_port"],
                          prefix=PREFIX,
                          set_kinds={})


ATTRIBUTES = tuple(PREFIX + name for name in DEFAULT_APPLICATION_COLUMNS)
"""The attribute columns the enrichment writes."""
