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
One workstation, seen at layers 3, 4, 6 and 7, is one entity and one chain.

Each layer names a host its own way: a flow, a packet, a handshake and a DNS query by source address, a process by
the host name its EDR agent reports. The four corpora that see those layers each resolve the address through the
DHCP server's leases to a network interface and through the EDR inventory to the machine (see `host_identity`), and
this harness composes the four layers' own stages with those hops and seals them together, the way the estate harness
seals its layers, so it can ask the question none of the single-layer corpora can: whether a machine's traffic and its
processes land on one key and one lineage chain.

A record at any one layer is enough to place on a key. What the harness adds is the control: an address leased to one
workstation in the morning and to another after the first released it, inside one hour. Keyed by address, both
machines' flows are one entity and one chain; resolved through the leases, they are two -- which is the reason the
chain is rooted on the machine and not on the address.

The four layers' context is attached on the machine as well, from an inventory keyed by host name, so the owner and
criticality a layer 3 flow carries are the ones its machine's processes carry.

It is a harness over the corpora's own stages rather than a thirteenth corpus with a golden file: what it asserts is
a relation between the layers, which a golden of each layer's rows would record without testing.
"""

import typing

import application_pipeline
import host_identity
import network_pipeline
import pandas as pd
import presentation_pipeline
import pytest
import transport_pipeline

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.chain_anchor_stage import DEFAULT_ANCHOR_COLUMN
from morpheus.stages.lineage.chain_anchor_stage import ChainAnchorStage
from morpheus.stages.lineage.collector_sequence_stage import CollectorSequenceStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.stages.telemetry.tc7_endpoint_stage import TC7EndpointStage
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.entity_key import normalize_hostname

NS_PER_SECOND = 10**9
HOUR = 10
PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900

WORKSTATION = "10.0.0.50"
WORKSTATION_NAME = "WS-0-50"
"""As the EDR agent reports it. The inventory and every layer key it as `ws-0-50`."""

RELEASED = "10.0.0.60"
FIRST_HOLDER = ("aa:10:0a:00:00:3c", "WS-0-60")
SECOND_HOLDER = ("aa:10:0a:00:00:99", "WS-LOANER-07")
RELEASE_MINUTE = 30
"""When the first holder releases the address. The second is granted it a minute later."""

SERVER = "10.0.1.30"
"""A static address no DHCP server has a record of."""

LAYERS = {
    network_pipeline.TELEMETRY_CLASS: 3,
    transport_pipeline.TELEMETRY_CLASS: 4,
    presentation_pipeline.TELEMETRY_CLASS: 6,
    application_pipeline.DNS_CLASS: 7,
    "tc7_endpoint": 7,
}

HOST_CONTEXT = {
    "ws-0-50": ("finance", "high", "ana@example.com"),
    "ws-0-60": ("finance", "standard", "ben@example.com"),
    "ws-loaner-07": ("loaner", "standard", "it-ops@example.com"),
}
"""The asset inventory, keyed by host name: device role, criticality, owner."""


def at(hour: int, minute: int = 0, second: int = 0) -> int:
    return (hour * 3600 + minute * 60 + second) * NS_PER_SECOND


def _row(rows: list, collector: str, schema: str, origin: str, when: int, **fields) -> None:
    rows.append({
        "collector_id": collector,
        "schema_version": schema,
        "origin_hash": f"{origin}@{when}",
        "collector_seq": len(rows),
        "event_time": when,
        **fields,
    })


def _flow(rows: list, source: str, destination: str, when: int) -> None:
    _row(rows,
         "netflow-01",
         "tc3.v1",
         f"{source}->{destination}",
         when,
         exporter_id="rtr-access-1:256",
         src_ip=source,
         src_port=50000 + len(rows),
         dst_ip=destination,
         dst_port=443,
         protocol="tcp",
         ip_ttl=128,
         bytes_out=1200,
         bytes_in=4800,
         bgp_as_dst="64512")


def build_corpus() -> dict[str, pd.DataFrame]:
    """The workstation's hour at four layers, the re-leased address's two holders, and an unleased server."""
    (flows, packets, handshakes, queries, processes) = ([], [], [], [], [])

    _flow(flows, WORKSTATION, SERVER, at(HOUR, 5))
    _row(packets,
         "pcap-01",
         "tc4.v1",
         f"{WORKSTATION}:44000={SERVER}:445",
         at(HOUR, 6),
         src_ip=WORKSTATION,
         src_port=44000,
         dst_ip=SERVER,
         dst_port=445,
         protocol="tcp",
         tcp_flags=transport_pipeline.PSH_ACK,
         data_len=1200)
    _row(handshakes,
         "tls-inspect-01",
         "tc6.v1",
         f"{WORKSTATION}->93.184.216.34",
         at(HOUR, 7),
         src_ip=WORKSTATION,
         src_port=49226,
         dst_ip="93.184.216.34",
         dst_port=443,
         protocol="tcp",
         tls_version="TLSv1.3",
         ja4_client=presentation_pipeline.CHROME,
         certificate_issuer=presentation_pipeline.CORP_CA,
         certificate_fingerprint_sha256="sha256:d9c2e00b50a422cdecfff7dbace937cb",
         cipher_suite=presentation_pipeline.MODERN_SUITE,
         validation_result=presentation_pipeline.VALID,
         certificate_not_before=0,
         certificate_not_after=at(24 * 365),
         content_type_declared=None,
         content_type_detected=None)
    _row(queries,
         "resolver-01",
         "tc7_dns.v1",
         f"{WORKSTATION}?intranet.example.com",
         at(HOUR, 8),
         src_ip=WORKSTATION,
         query_name="intranet.example.com",
         query_type="A")
    _row(processes,
         "edr-01",
         "tc7_endpoint.v1",
         f"{WORKSTATION_NAME}>excel.exe",
         at(HOUR, 9),
         hostname=WORKSTATION_NAME,
         parent_image_path="C:\\Windows\\explorer.exe",
         image_path="C:\\Program Files\\Microsoft Office\\excel.exe",
         integrity_level="Medium",
         signature_status="Valid")

    # The re-leased address, once by each holder, inside the same hour; and a server nobody leased.
    _flow(flows, RELEASED, SERVER, at(HOUR, 10))
    _flow(flows, RELEASED, SERVER, at(HOUR, 50))
    _flow(flows, SERVER, WORKSTATION, at(HOUR, 20))

    hosts = host_identity.build([WORKSTATION], 0, at(HOUR + 1))
    (leases, reports) = (hosts[host_identity.LEASE_CLASS].to_dict("records"),
                         hosts[host_identity.INVENTORY_CLASS].to_dict("records"))

    for (mac, name, message, when) in ((*FIRST_HOLDER, "ack",
                                        at(0, 2)), (*FIRST_HOLDER, "release", at(HOUR, RELEASE_MINUTE)),
                                       (*SECOND_HOLDER, "ack", at(HOUR, RELEASE_MINUTE + 1))):
        _row(leases,
             host_identity.DHCP_COLLECTOR,
             "tc3_dhcp.v1",
             f"{RELEASED}>{mac}:{message}",
             when,
             ip=RELEASED,
             mac=mac,
             dhcp_message=message,
             lease_seconds=host_identity.GRANT_SECONDS if message == "ack" else None)

        if (message == "ack"):
            _row(reports,
                 host_identity.INVENTORY_COLLECTOR,
                 "tc7_host_inventory.v1",
                 mac,
                 at(0, 2),
                 mac=mac,
                 hostname=name,
                 report_valid_seconds=host_identity.GRANT_SECONDS)

    def frame(rows: list) -> pd.DataFrame:
        result = pd.DataFrame(rows).sort_values("event_time", kind="mergesort").reset_index(drop=True)
        result["collector_seq"] = result.groupby("collector_id", sort=False).cumcount()

        return result

    return {
        network_pipeline.TELEMETRY_CLASS: frame(flows),
        transport_pipeline.TELEMETRY_CLASS: frame(packets),
        presentation_pipeline.TELEMETRY_CLASS: frame(handshakes),
        application_pipeline.DNS_CLASS: frame(queries),
        "tc7_endpoint": frame(processes),
        host_identity.LEASE_CLASS: frame(leases),
        host_identity.INVENTORY_CLASS: frame(reports),
    }


def build_host_store() -> BitemporalStore:
    """The asset inventory keyed by host name, the key every layer's machine resolves to."""
    return BitemporalStore(
        "asset",
        [
            make_version(
                "asset", host, 0, None, 0, values={
                    "device_role": role, "criticality": criticality, "owner": owner
                }) for (host, (role, criticality, owner)) in sorted(HOST_CONTEXT.items())
        ])


def _layer_stages(config: Config, name: str, leases, inventory) -> list:
    """The layer's own stages, then the hops to its machine and the chain rooted on it."""
    if (name == "tc7_endpoint"):
        # A process names its machine; the endpoint stage keys it as the inventory does.
        return [TC7EndpointStage(config), ChainAnchorStage(config, candidates=[host_identity.HOST_KEY_COLUMN])]

    if (name == network_pipeline.TELEMETRY_CLASS):
        own = network_pipeline.build_stages(config)
    elif (name == transport_pipeline.TELEMETRY_CLASS):
        own = transport_pipeline.build_stages(config)
    elif (name == presentation_pipeline.TELEMETRY_CLASS):
        # The layer 6 composition resolves its sources itself, because its asset context is looked up on the machine.
        return presentation_pipeline.build_stages(config, hosts=(leases, inventory))
    else:
        own = application_pipeline.build_stages(config, name)

    fallback = "flow_id" if name == transport_pipeline.TELEMETRY_CLASS else "src_ip"

    return own + host_identity.resolver_stages(config, leases, inventory, fallback_anchor=fallback)


def _run_layer(config: Config, name: str, frame: pd.DataFrame, leases, inventory) -> pd.DataFrame:
    from host_frame import to_host_frame  # pylint: disable=import-outside-toplevel

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[frame.copy()]))
    pipe.add_stage(LineageStampStage(config, id_columns=host_identity.ID_COLUMNS))
    pipe.add_stage(TotalOrderStage(config))
    pipe.add_stage(CollectorSequenceStage(config))

    for stage in _layer_stages(config, name, leases, inventory):
        pipe.add_stage(stage)

    # The machine's context, whichever layer saw it.
    pipe.add_stage(
        TC0EnrichStage(config,
                       store=build_host_store(),
                       entity_column=host_identity.HOST_KEY_COLUMN,
                       entity_normalization="hostname",
                       prefix="host_ctx_"))
    pipe.add_stage(EnvelopeStampStage(config, osi_layer=LAYERS[name], entity_columns=[DEFAULT_ANCHOR_COLUMN]))
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    result = pd.concat([to_host_frame(message.copy_dataframe()) for message in sink.get_messages()], ignore_index=True)
    result["telemetry_class"] = name

    return result


def run_pipeline(config: Config, corpus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Each layer through its own stages and the hops to its machine, then every layer sealed together."""
    (leases, inventory) = host_identity.tables(config, {name: [frame] for (name, frame) in corpus.items()})
    layers = [_run_layer(config, name, corpus[name], leases, inventory) for name in LAYERS]

    return _seal(config, pd.concat(layers, ignore_index=True), DEFAULT_ANCHOR_COLUMN)


def _seal(config: Config, union: pd.DataFrame, entity_key_column: str) -> pd.DataFrame:
    """One hourly seal over every layer's rows, so a chain can hold records from all of them."""
    from host_frame import to_host_frame  # pylint: disable=import-outside-toplevel

    union = union.sort_values(list(DEFAULT_ORDER_COLUMNS), kind="stable").reset_index(drop=True)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[union]))
    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=PERIOD_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        entity_key_column=entity_key_column))
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    return pd.concat([to_host_frame(message.copy_dataframe()) for message in sink.get_messages()], ignore_index=True)


def _python_stages() -> None:
    # The stages run as Python whatever the machine, so a test that runs after one the conftest put in C++ mode
    # builds its pipelines the same way as the first.
    from morpheus.config import CppConfig  # pylint: disable=import-outside-toplevel

    CppConfig.set_should_use_cpp(False)


@pytest.fixture(name="pipeline_config", scope="module")
def pipeline_config_fixture() -> Config:
    from morpheus.config import ExecutionMode  # pylint: disable=import-outside-toplevel

    _python_stages()
    config = Config()
    config.execution_mode = ExecutionMode.CPU

    return config


@pytest.fixture(name="result", scope="module")
def result_fixture(pipeline_config: Config) -> pd.DataFrame:
    return run_pipeline(pipeline_config, build_corpus())


def _from(result: pd.DataFrame, source: str, minute: typing.Optional[int] = None) -> pd.DataFrame:
    rows = result[result["src_ip"] == source]

    return rows if minute is None else rows[rows["event_time"] == at(HOUR, minute)]


def _workstation(result: pd.DataFrame) -> pd.DataFrame:
    return result[(result["src_ip"] == WORKSTATION) | (result["telemetry_class"] == "tc7_endpoint")]


def test_the_workstation_is_one_entity_at_four_layers(result: pd.DataFrame):
    rows = _workstation(result)

    assert sorted(rows["osi_layer"].astype(int).unique()) == [3, 4, 6, 7]
    assert set(rows["telemetry_class"]) == set(LAYERS)
    assert set(rows[host_identity.HOST_KEY_COLUMN]) == {normalize_hostname(WORKSTATION_NAME)}
    assert rows["entity_key"].nunique() == 1


def test_the_workstation_is_one_lineage_chain_at_four_layers(result: pd.DataFrame):
    rows = _workstation(result)

    assert rows["lineage_id"].notna().all()
    assert rows["lineage_id"].nunique() == 1
    # And nothing else is in it: the chain is the machine's, not the hour's.
    assert len(result[result["lineage_id"] == rows["lineage_id"].iloc[0]]) == len(rows)


def test_each_hop_names_the_binding_it_resolved_through(result: pd.DataFrame):
    rows = result[result["src_ip"] == WORKSTATION]

    assert set(rows[host_identity.MAC_COLUMN]) == {host_identity.mac_for(WORKSTATION)}
    assert rows[host_identity.LEASE_UID_COLUMN].notna().all()
    assert rows[host_identity.INVENTORY_UID_COLUMN].notna().all()
    assert set(rows["resolution_methods"]) == {"soft:dhcp_lease;soft:host_inventory"}
    assert set(rows["chain_anchor_source"]) == {host_identity.HOST_KEY_COLUMN}


def test_every_layer_carries_the_machines_context(result: pd.DataFrame):
    rows = _workstation(result)
    (role, criticality, owner) = HOST_CONTEXT[normalize_hostname(WORKSTATION_NAME)]

    assert rows["host_ctx_found"].astype(bool).all()
    assert set(rows["host_ctx_device_role"]) == {role}
    assert set(rows["host_ctx_criticality"]) == {criticality}
    assert set(rows["host_ctx_owner"]) == {owner}


def test_a_released_address_is_two_machines_in_one_hour(result: pd.DataFrame):
    before = _from(result, RELEASED, 10)
    after = _from(result, RELEASED, 50)

    assert list(before[host_identity.HOST_KEY_COLUMN]) == [normalize_hostname(FIRST_HOLDER[1])]
    assert list(after[host_identity.HOST_KEY_COLUMN]) == [normalize_hostname(SECOND_HOLDER[1])]
    assert list(before[host_identity.MAC_COLUMN]) == [FIRST_HOLDER[0]]
    assert list(after[host_identity.MAC_COLUMN]) == [SECOND_HOLDER[0]]
    # One window, one address, and still two entities and two chains, each with the holder's own context.
    assert before["window_id"].iloc[0] == after["window_id"].iloc[0]
    assert before["entity_key"].iloc[0] != after["entity_key"].iloc[0]
    assert before["lineage_id"].iloc[0] != after["lineage_id"].iloc[0]
    assert list(after["host_ctx_device_role"]) == [HOST_CONTEXT["ws-loaner-07"][0]]


def test_an_unleased_server_is_rooted_on_its_address(result: pd.DataFrame):
    rows = _from(result, SERVER)

    assert len(rows) == 1
    assert rows[host_identity.HOST_KEY_COLUMN].isna().all()
    assert list(rows["chain_anchor_source"]) == ["src_ip"]
    assert list(rows["resolution_methods"]) == ["unresolved:dhcp_lease;unresolved:host_inventory"]
    assert not rows["host_ctx_found"].astype(bool).any()


def test_keyed_by_address_the_two_holders_would_be_one(pipeline_config: Config):
    # The control's point, measured rather than stated: the same flows sealed on the address put both holders in one
    # chain.
    _python_stages()
    corpus = build_corpus()
    (leases, inventory) = host_identity.tables(pipeline_config, {name: [frame] for (name, frame) in corpus.items()})
    flows = _run_layer(pipeline_config,
                       network_pipeline.TELEMETRY_CLASS,
                       corpus[network_pipeline.TELEMETRY_CLASS],
                       leases,
                       inventory)
    sealed = _seal(pipeline_config, flows, "src_ip")
    rows = sealed[sealed["src_ip"] == RELEASED]

    assert len(rows) == 2
    assert rows["lineage_id"].nunique() == 1
    assert rows[host_identity.HOST_KEY_COLUMN].nunique() == 2
