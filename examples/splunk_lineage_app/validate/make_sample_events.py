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
    "tc1_device": "morpheus:score:l1",
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

CAMPAIGN_SOURCETYPES = {
    "tc3": "morpheus:score:l3",
    "tc5_auth": "morpheus:score:l5",
    "tc7_endpoint": "morpheus:score:l7",
    "tc5_session": "morpheus:score:l5",
    "tc4": "morpheus:score:l4",
    "tc6": "morpheus:score:l6",
    "tc7_saas": "morpheus:score:l7",
    "tc1": "morpheus:score:l1",
    "tc2_mac": "morpheus:score:l2",
}
"""The campaign's classes, in the order it runs them, and the sourcetype each layer's records always go on. The port
polls and MAC table walks run last, after the classes they bind to, and carry R-C-003's first two steps."""


def _render(frame, sourcetype: str) -> list:
    """Put a frame through the wire stage exactly as a pipeline would, and return the lines a sink would write."""
    from morpheus.config import Config
    from morpheus.config import CppConfig
    from morpheus.config import ExecutionMode
    from morpheus.messages import MessageMeta
    from morpheus.stages.output.siem_wire_stage import SiemWireStage
    from morpheus.utils.siem_wire import to_wire_lines

    CppConfig.set_should_use_cpp(False)
    config = Config()
    config.execution_mode = ExecutionMode.CPU

    stage = SiemWireStage(config, sourcetype=sourcetype, require_columns=False)
    rendered = stage.on_data(MessageMeta(frame.reset_index(drop=True))).copy_dataframe()

    return to_wire_lines(rendered)


LATE_SOURCETYPE = "morpheus:late"
QUARANTINE_SOURCETYPE = "morpheus:quarantine"
HEALTH_SOURCETYPE = "morpheus:health"


def _late(frame):
    """Which rows WindowSealStage marked late: those go on `morpheus:late`, never on a score sourcetype."""
    if ("is_late" not in frame.columns):
        return frame.index.to_series(index=frame.index).map(lambda _: False)

    return frame["is_late"].astype("boolean").fillna(False).astype(bool)


def _scored(by_sourcetype: dict, frame, sourcetype: str):
    """
    Render a class's rows onto its score sourcetype, and its late rows onto `morpheus:late`.

    A deployment routes on `is_late` before the sink, so a row that arrived after its window sealed never reaches
    the index the detections read. Every detection also filters on `window_complete=true`, which is the second of the
    two guards; this is the first.
    """
    late = _late(frame)

    if (late.any()):
        by_sourcetype.setdefault(LATE_SOURCETYPE, []).extend(_render(frame[late], LATE_SOURCETYPE))

    if ((~late).any()):
        by_sourcetype.setdefault(sourcetype, []).extend(_render(frame[~late], sourcetype))


def _health(by_sourcetype: dict, module, result):
    """The health records a deployment's health branch would write for one corpus's sealed output."""
    import envelope  # pylint: disable=import-outside-toplevel

    from morpheus.stages.lineage.pipeline_health_stage import (  # pylint: disable=import-outside-toplevel
        DEFAULT_MIN_BUCKETS,
    )

    records = envelope.health(module.build_pipeline_config(), result, module.PERIOD_SECONDS, DEFAULT_MIN_BUCKETS)

    if (len(records) > 0):
        by_sourcetype.setdefault(HEALTH_SOURCETYPE, []).extend(_render(records, HEALTH_SOURCETYPE))


def main() -> int:
    # Imported here rather than at module scope, like everything else this file takes from morpheus: the path
    # setup above has to run before the package is reachable.
    import application_pipeline  # pylint: disable=import-outside-toplevel
    import campaign_pipeline  # pylint: disable=import-outside-toplevel
    import context_pipeline  # pylint: disable=import-outside-toplevel
    import endpoint_pipeline  # pylint: disable=import-outside-toplevel
    import estate_pipeline as ep  # pylint: disable=import-outside-toplevel
    import host_score_pipeline  # pylint: disable=import-outside-toplevel
    import network_pipeline  # pylint: disable=import-outside-toplevel
    import pandas as pd  # pylint: disable=import-outside-toplevel
    import presentation_pipeline  # pylint: disable=import-outside-toplevel
    import saas_pipeline  # pylint: disable=import-outside-toplevel
    import session_pipeline as sp  # pylint: disable=import-outside-toplevel
    import telemetry_pipeline as tp  # pylint: disable=import-outside-toplevel
    import transport_pipeline  # pylint: disable=import-outside-toplevel

    from morpheus.utils.binding_table import DEFAULT_L1_BUCKET_SECONDS  # pylint: disable=import-outside-toplevel

    EVENTS.mkdir(parents=True, exist_ok=True)
    telemetry = ep.run_pipeline(ep.build_pipeline_config(), ep.build_corpus())
    written = {}

    by_sourcetype: dict = {}

    for (name, sourcetype) in CLASS_SOURCETYPES.items():
        rows = telemetry[telemetry["telemetry_class"] == name]

        if (len(rows) == 0):
            continue

        _scored(by_sourcetype, rows, sourcetype)

    # The DMZ poller's records, quarantined for a clock four and a half seconds out, on their own sourcetype.
    quarantined = telemetry[telemetry["telemetry_class"] == "quarantine"]
    by_sourcetype[QUARANTINE_SOURCETYPE] = _render(quarantined.dropna(axis=1, how="all"), QUARANTINE_SOURCETYPE)
    _health(by_sourcetype, ep, telemetry)

    sessions = sp.run_pipeline(sp.build_pipeline_config(), sp.build_corpus())

    for (name, sourcetype) in SESSION_CLASS_SOURCETYPES.items():
        rows = sessions[sessions["telemetry_class"] == name]

        if (len(rows) > 0):
            _scored(by_sourcetype, rows, sourcetype)

    _health(by_sourcetype, sp, sessions)

    # The edge stream: one record per hop a record was resolved through and per layer a connection was seen at
    # above the first, written by EdgeEmitStage over the estate's sealed rows here and the campaign's below.
    estate_edges = ep.build_edges(ep.build_pipeline_config(), telemetry)

    # Layer 3 comes from its own corpus and its own pipeline, as layer 5 does. Its records share no entity with
    # the estate's -- an address is not a port and not a principal -- so they carry their own chains and the
    # cross-layer search sees them as a fourth layer's worth of events rather than as part of the estate's.
    flows = network_pipeline.run_pipeline(network_pipeline.build_pipeline_config(), network_pipeline.build_corpus())
    _scored(by_sourcetype, flows, "morpheus:score:l3")
    _health(by_sourcetype, network_pipeline, flows)

    # Layer 4 likewise. Its entity is a conversation rather than an address, so its records share no entity with
    # layer 3's either, even where the two corpora use the same addresses. The hop between the layers is the
    # connection's Community ID, which the connection evidence report joins on; these two corpora are built apart
    # and share none, so the connections it finds come from the campaign corpus below, which carries both layers.
    packets = transport_pipeline.run_pipeline(transport_pipeline.build_pipeline_config(),
                                              transport_pipeline.build_corpus())
    _scored(by_sourcetype, packets, "morpheus:score:l4")
    _health(by_sourcetype, transport_pipeline, packets)

    # Layer 6 likewise. This corpus supplies the layer 6 detections, R-B-L6-001 among them. R-C-002 correlates a
    # new client fingerprint with beaconing at layer 3 from the same host, and this corpus and the network corpus
    # are separate, so its one chain comes from the campaign corpus below, which carries both halves.
    handshakes = presentation_pipeline.run_pipeline(presentation_pipeline.build_pipeline_config(),
                                                    presentation_pipeline.build_corpus())
    _scored(by_sourcetype, handshakes, "morpheus:score:l6")
    _health(by_sourcetype, presentation_pipeline, handshakes)

    # Layer 7's DNS, HTTP, service, SaaS and endpoint classes share one sourcetype, as layer 2's classes do. The
    # SaaS and endpoint classes come from their own pipelines, because they read the TC-0 context store -- and the
    # SaaS one seals weekly windows besides.
    requests = application_pipeline.run_pipeline(application_pipeline.build_pipeline_config(),
                                                 application_pipeline.build_corpus())
    operations = saas_pipeline.run_pipeline(saas_pipeline.build_pipeline_config(), saas_pipeline.build_corpus())
    processes = endpoint_pipeline.run_pipeline(endpoint_pipeline.build_pipeline_config(),
                                               endpoint_pipeline.build_corpus())

    for (module, result) in ((application_pipeline, requests), (saas_pipeline, operations), (endpoint_pipeline,
                                                                                             processes)):
        _scored(by_sourcetype, result, "morpheus:score:l7")
        _health(by_sourcetype, module, result)

    # The host segment: one row per workstation per hour, scored against its peer group's committed model, on a
    # sourcetype of its own. Its hosts are the host corpus's and share nothing with any other corpus's machines, so
    # R-B-L7-004's lookup against the watchlist R-P-L3-009 writes finds none of the endpoint corpus's hosts there.
    hosts = host_score_pipeline.run_pipeline(host_score_pipeline.build_pipeline_config(),
                                             host_score_pipeline.build_corpus())
    _scored(by_sourcetype, hosts, "morpheus:score:host")
    _health(by_sourcetype, host_score_pipeline, hosts)

    # The campaign: one estate seen by all its collectors, which is what R-C-001 and R-C-004 are asserted on. Its
    # records go on the sourcetypes their layers' records always go on, because that is where a chained rule has to
    # find them, and they share no entity with any other corpus's records.
    campaign_config = campaign_pipeline.build_pipeline_config()
    campaign_corpus = campaign_pipeline.build_corpus()
    campaign = campaign_pipeline.run_pipeline(campaign_config, campaign_corpus)

    for (name, sourcetype) in CAMPAIGN_SOURCETYPES.items():
        rows = campaign[campaign["telemetry_class"] == name]
        _scored(by_sourcetype, rows.dropna(axis=1, how="all"), sourcetype)

    _health(by_sourcetype, campaign_pipeline, campaign)
    edges = pd.concat([estate_edges, campaign_pipeline.build_edges(campaign_config, campaign)], ignore_index=True)
    by_sourcetype["morpheus:edge"] = _render(edges, "morpheus:edge")

    # The TC-0 context store: every version the three producers recorded, refused ones included, because a count of
    # refusals is something an estate should be able to see on its search head. The probes are the harness's own
    # and are not sent anywhere.
    context = context_pipeline.run_pipeline(context_pipeline.build_pipeline_config(), context_pipeline.build_corpus())

    context_sourcetypes = [
        (context_pipeline.IDENTITY_CLASS, "context:identity"),
        (context_pipeline.ASSET_CLASS, "context:asset"),
        (context_pipeline.APPLICATION_CLASS, "context:application"),
    ]

    for (name, sourcetype) in context_sourcetypes:
        rows = context[context["telemetry_class"] == name]
        by_sourcetype[sourcetype] = _render(rows.dropna(axis=1, how="all"), sourcetype)

    # The network inventory is context too, but it is the estate's: its records came in with the polls they are
    # joined onto, and they are rendered from the same run.
    network = telemetry[telemetry["telemetry_class"] == tp.NETWORK_CONTEXT_CLASS]
    by_sourcetype["context:network"] = _render(network.dropna(axis=1, how="all"), "context:network")

    bindings = telemetry[telemetry["telemetry_class"] == "tc2_binding"]
    bucketed = tp.build_binding_table(bindings).to_bucketed_frame()

    # The layer 1 history, on the same sourcetype and told apart by `binding_table`. Only the intervals a
    # current-state row has overwritten are expanded: a port whose optic never changed is answered correctly for
    # all time by the unbucketed lookup, so putting it here would cost a row per day to repeat that answer.
    port_bindings = telemetry[telemetry["telemetry_class"] == "tc1_binding"]
    port_history = tp.build_port_binding_table(port_bindings).superseded().to_bucketed_frame(
        bucket_seconds=DEFAULT_L1_BUCKET_SECONDS, key_name="entity_key")

    # The campaign DHCP server's leases, the rows the L2/L3 refresh builds `binding_l2_l3` from: an address, the
    # interface it was handed to, and the switch port the relay saw the request arrive on. Only the desks' leases
    # are rendered. The workstations' run nine days each, which is two and a half thousand five-minute buckets
    # apiece, and they carry no port; a deployment ships them to the same lookup.
    acknowledgements = campaign_corpus[campaign_pipeline.LEASE_CLASS]
    desks = acknowledgements[acknowledgements["port_id"].notna()]
    leases = campaign_pipeline.build_lease_table(campaign_pipeline.close_lease_bindings(
        campaign_config, [desks])).to_bucketed_frame(key_name="ip", table_name="dhcp_lease")

    # The campaign's two sites' MAC bindings, the rows the L2 refresh builds `binding_l2` from: a workstation's MAC
    # and the port it was plugged into. With the leases above they are the walk from a sign-in's address to a port.
    macs = campaign_pipeline.build_mac_table(
        campaign_pipeline.close_mac_bindings(
            campaign_config, [campaign_corpus[campaign_pipeline.MAC_CLASS]])).to_bucketed_frame(table_name="mac_table")

    # The estate's 802.1X supplicant bindings, the rows the L5/L2 refresh builds `binding_l5_l2` from: an identity
    # and the port it was authorized on. The estate's desk sign-ins carry the identity the directory gave them.
    supplicants = ep.build_supplicant_table(
        ep.close_supplicant_bindings(ep.build_pipeline_config(),
                                     [telemetry[telemetry["telemetry_class"] == "tc2_auth"].copy()])).to_bucketed_frame(
                                         key_name="dot1x_identity", table_name="dot1x_supplicant")

    rendered = _render(bucketed, "binding:bucketed")
    rendered.extend(_render(leases, "binding:bucketed"))
    rendered.extend(_render(macs, "binding:bucketed"))
    rendered.extend(_render(supplicants, "binding:bucketed"))

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
