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
One campaign told at every layer from the switch port up, and the composed pipelines that measure it, for the chained
rules.

The other corpora are each written for one layer's rules or for one join across a few layers (the estate's port
bindings, the host score's host segment); none of them follows one actor through every layer. A chained rule needs
that -- the same source, the same person and the same host seen by several collectors -- so this corpus is written
as one estate over nine days and split by collector. Each layer's records then go through that layer's own stages,
exactly as a deployment would run them, and the chained rule is evaluated over what comes out.

**R-C-001 - Lateral movement chain.** Within 30 minutes, in order: a source's fan-out rises above its own previous
hour; a principal authenticates *from that source* to a host they have never logged into; and that host starts a
process whose ancestry neither it nor its peer group has run in thirty days. No step has to breach a threshold of its
own. The attacker here reaches twenty-five new addresses, half what R-B-L3-001 fires on, and neither a first login to
a server nor one new process is an alert by itself.

The attacker does all three to a server, and seven others each do all but one:

- **wrong order**: the new process runs before the login that would have to have started it;
- **too slow**: every step happens, the last one thirty-five minutes after the first;
- **known host**: the login is to a server the principal logs into every day;
- **other source**: the login comes from an address other than the one whose fan-out rose;
- **flat fan-out**: a busy host reaching the same twenty servers every hour, so nothing rose;
- **no new process**: the host only runs what it always runs;
- **workstation**: every step, in order and in time, onto a finance workstation rather than a server. The rule is
  about a login *to a server*, and the asset inventory is what says which hosts are servers.

**The steps are joined on values, not on `lineage_id`.** The three layers are sealed on three different entities --
a source address, a principal, a host -- so no lineage chain holds all three, and the rule joins the source address
to the login's source and the login's target to the process's host, carrying each step's lineage identifiers as
evidence. A login log and an EDR name the same machine differently -- the directory as an administrator typed it,
the EDR in lower case -- so hosts are joined on the key both stages render with `entity_key.normalize_hostname`:
`target_host_key` on the login, `endpoint_host` on the process.

**Each step may precede the one it follows by up to two minutes.** That is the rule's declared join tolerance, twice
a worst-case clock offset of sixty seconds between collectors, as Part 3's governance asks.

**R-C-004 - Staged exfiltration.** Within two hours, in order: a principal's SaaS export more than five times their
own baseline; a transfer envelope breach from an address the principal holds a session on at that moment; and a
connection from that address to a destination whose certificate issuer nobody in the estate has seen in thirty days.
The same collectors' shapes carry it -- SaaS audit, session start and stop, packets, TLS handshakes -- each through
its own layer's composed pipeline, and the attacker does all three while four others each do all but one:

- **someone else's host**: the breach and the handshake come from an address another principal's session holds;
- **outside the session**: the breach comes from the principal's own address twenty minutes after they logged off;
- **known issuer**: the last connection's certificate is the corporate one every host in the estate sees daily;
- **wrong order**: the breach and the handshake both happen before the export.

**R-C-002 - Command and control establishment.** A settled host presents a TLS client fingerprint it never has, to a
destination, and within the hour starts beaconing to that same destination. The attacker does both; four others do
both with one condition broken:

- **beacon first**: the host was already beaconing to the destination an hour before the new fingerprint;
- **other destination**: the beacon goes somewhere other than where the new fingerprint went;
- **too slow**: the beacon matures sixty-seven minutes after the fingerprint;
- **unsettled host**: five handshakes of history, so the new fingerprint is only new because the host is.

**R-C-005 - Credential replay across the stack.** One principal authenticates successfully from two addresses that
resolve, through the layer 2 bindings, to switch ports at two sites further apart than anyone could have travelled in
the time between. The address goes to a MAC through the estate's DHCP leases, supplied the way the directory is in
the estate harness; the MAC goes to a port through the bindings
{py:class}`~morpheus.stages.telemetry.tc2_binding_stage.TC2BindingStage` closes from the two sites' MAC tables; and
{py:class}`~morpheus.stages.telemetry.tc5_travel_stage.TC5TravelStage` measures the journey between the ports' sites,
from the estate's own record of where its sites are. Nothing here is a geolocation guess: both ends are switch ports.
The attacker signs in at headquarters and, twenty minutes later, in Edinburgh. Five others each fall one step short:

- **same site**: two ports, both at headquarters;
- **time to travel**: Edinburgh three hours later, which a train covers;
- **unresolved address**: the second sign-in comes from an address no lease names, so no port can be claimed;
- **expired lease**: the second address was leased to an Edinburgh desk, but the lease ended an hour earlier;
- **failed sign-in**: the second attempt at Edinburgh is refused, and a failure proves nobody was there.

**R-C-003 - Physical compromise chain.** Within four hours, in order: a layer 1 event on a desk port -- a transceiver
substitution, or the link dropping and coming back, which is a workstation unplugged; a device of a vendor that port
has never carried appearing on it at layer 2; and a layer 3 flow from the address the DHCP server leased that device
to the estate's management network. The headquarters switch's polls are the first step, its table walks the second,
and the flows the third, each through its own layer's stages; the steps are joined on the port, from the poll to the
walk, and on the device's MAC, from the walk to the flow, whose address the DHCP lease named. The attacker unplugs
a desk workstation, plugs a single-board computer into the port and reaches for the switches' own interfaces. Three
others each do two of the three:

- **known vendor**: a desk re-patched, with another of the estate's own workstations on it, whose administrator
  reaches the management network -- the vendor is one the port has always carried;
- **no management**: a printer plugged into a desk port, which only ever talks to the print server;
- **no layer 1 event**: a single-board computer joins a desk port behind the workstation already on it, through a
  desk switch, and reaches the management network, but the port's link never dropped.
"""

import typing

import pandas as pd

from morpheus.compositions import edge_stream
from morpheus.compositions import envelope
from morpheus.compositions import host_identity
from morpheus.compositions import presentation as presentation_pipeline
from morpheus.compositions import saas as saas_pipeline
from morpheus.compositions import stamping
from morpheus.compositions import telemetry as telemetry_pipeline
from morpheus.compositions import transport as transport_pipeline
from morpheus.compositions.corpus_pipeline import _collect
from morpheus.compositions.corpus_pipeline import (  # noqa: F401 pylint: disable=unused-import  # re-exported for callers
    build_pipeline_config, )
from morpheus.compositions.resumable import ResumableLinearPipeline
from morpheus.config import Config
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.binding_resolver_stage import BindingResolverStage
from morpheus.stages.lineage.chain_anchor_stage import DEFAULT_ANCHOR_COLUMN
from morpheus.stages.lineage.collector_sequence_stage import CollectorSequenceStage
from morpheus.stages.lineage.community_id_stage import CommunityIdStage
from morpheus.stages.lineage.determinism_stamp_stage import DeterminismStampStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.stages.telemetry.tc2_binding_stage import TC2BindingStage
from morpheus.stages.telemetry.tc3_baseline_stage import TC3BaselineStage
from morpheus.stages.telemetry.tc3_beacon_stage import TC3BeaconStage
from morpheus.stages.telemetry.tc3_cardinality_stage import TC3CardinalityStage
from morpheus.stages.telemetry.tc3_reach_stage import TC3ReachStage
from morpheus.stages.telemetry.tc3_ttl_stage import TC3TtlStage
from morpheus.stages.telemetry.tc5_novelty_stage import TC5NoveltyStage
from morpheus.stages.telemetry.tc5_session_stage import TC5SessionStage
from morpheus.stages.telemetry.tc5_travel_stage import TC5TravelStage
from morpheus.stages.telemetry.tc7_endpoint_stage import TC7EndpointStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.binding_table import BindingTable
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize
from morpheus.utils.entity_key import normalize_hostname

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900
DAY_SECONDS = 86400
ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []
FLOW_CLASS = "tc3"
AUTH_CLASS = "tc5_auth"
PROCESS_CLASS = "tc7_endpoint"
SESSION_CLASS = "tc5_session"
TRANSFER_CLASS = transport_pipeline.TELEMETRY_CLASS
HANDSHAKE_CLASS = presentation_pipeline.TELEMETRY_CLASS
SAAS_CLASS = saas_pipeline.SAAS_CLASS
CLASSES = (FLOW_CLASS, AUTH_CLASS, PROCESS_CLASS, SESSION_CLASS, TRANSFER_CLASS, HANDSHAKE_CLASS, SAAS_CLASS)
"""The collectors, in the order they are run. The last four are R-C-004's; each of those runs through its layer's own
composed pipeline, which is the one its single-layer rules are asserted over."""
SETTINGS = {"period_seconds": PERIOD_SECONDS, "lateness_seconds": LATENESS_SECONDS}
"""The settings that decide this corpus's output, digested into `config_hash` by `stamping.envelope_for`."""
RULES = ("R-C-001", "R-C-002", "R-C-003", "R-C-004", "R-C-005")
"""The shipped rules that read this corpus's columns; their thresholds are folded into `pipeline_fingerprint`."""
# The rule's own figures, stated once so the corpus is built around them deliberately.
CHAIN_WINDOW_SECONDS = 30 * 60
JOIN_TOLERANCE_SECONDS = 120
LAST_DAY = 8
"""The day the campaign happens. The eight before it are the history every step is novel against."""
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
WORKSTATION_TARGET = "wes@example.com"
ACTORS = {
    ATTACKER: Actor("10.20.0.5", ATTACKER, "srv-dana", "SRV-DB-02", "10.20.0.5", True, 5, 12, 20),
    WRONG_ORDER: Actor("10.20.0.6", WRONG_ORDER, "srv-eli", "srv-app-03", "10.20.0.6", True, 5, 14, 8),
    TOO_SLOW: Actor("10.20.0.7", TOO_SLOW, "srv-fay", "srv-fs-04", "10.20.0.7", True, 5, 20, 40),
    KNOWN_HOST: Actor("10.20.0.8", KNOWN_HOST, "srv-gus", "srv-gus", "10.20.0.8", True, 5, 12, 20),
    OTHER_SOURCE: Actor("10.20.0.9", OTHER_SOURCE, "srv-hal", "srv-web-05", "10.20.0.99", True, 5, 12, 20),
    FLAT_FANOUT: Actor("10.20.0.10", FLAT_FANOUT, "srv-ivan", "srv-mq-06", "10.20.0.10", False, 5, 12, 20),
    NO_NEW_PROCESS: Actor("10.20.0.11", NO_NEW_PROCESS, "srv-judy", "srv-ci-07", "10.20.0.11", True, 5, 12, None),
    WORKSTATION_TARGET: Actor("10.20.0.12", WORKSTATION_TARGET, "srv-wes", "WS-FIN-08", "10.20.0.12", True, 5, 12, 20),
}
WORKSTATIONS = ("ws-fin-08", )
"""The hosts the inventory records as workstations. Every other host here is a server."""


def at(day: float, hour: float = 0, minute: float = 0, second: float = 0) -> int:
    """An instant in the corpus, in nanoseconds since the epoch."""
    return int((day * DAY_SECONDS + hour * 3600 + minute * 60 + second) * NS_PER_SECOND)


EXFIL_WINDOW_SECONDS = 2 * 3600
RECORD_MULTIPLE = 5.0
PORTAL_ISSUER = presentation_pipeline.CORP_CA


class Exfiltrator(typing.NamedTuple):
    """A principal, the address their session holds, and when each of R-C-004's steps happens on the last day."""

    principal: str
    address: str
    """The address the principal's sessions come from."""
    breach_address: str
    """Where the breach and the handshake come from. Another principal's address for the someone-else control."""
    session_end_minute: typing.Optional[float]
    """When the last day's session ends, minutes after the campaign hour starts; None for the evening, as usual."""
    export_minute: float
    breach_minute: float
    handshake_minute: float
    issuer: str


KIM = "kim@example.com"
LEE = "lee@example.com"
MAX = "max@example.com"
NIA = "nia@example.com"
OTO = "oto@example.com"
PIA = "pia@example.com"


def relay_issuer(principal: str) -> str:
    """An authority nobody in the estate has seen, distinct per actor so one's first sighting never covers another's."""
    return f"CN={principal.split('@')[0].title()} Relay CA, O=Unknown"


EXFILTRATORS = {
    KIM: Exfiltrator(KIM, "10.20.1.5", "10.20.1.5", None, 5, 30, 50, relay_issuer(KIM)),
    LEE: Exfiltrator(LEE, "10.20.1.6", "10.20.1.7", None, 5, 30, 50, relay_issuer(LEE)),
    NIA: Exfiltrator(NIA, "10.20.1.8", "10.20.1.8", 10, 5, 30, 50, relay_issuer(NIA)),
    OTO: Exfiltrator(OTO, "10.20.1.10", "10.20.1.10", None, 5, 30, 50, PORTAL_ISSUER),
    PIA: Exfiltrator(PIA, "10.20.1.9", "10.20.1.9", None, 30, 5, 15, relay_issuer(PIA)),
}
SESSION_HOLDERS = {**{actor.principal: actor.address for actor in EXFILTRATORS.values()}, MAX: "10.20.1.7"}
"""Every principal with a daily session and the address it comes from. Max exports nothing; Lee's breach comes from
Max's address."""
C2_WINDOW_SECONDS = 3600
SETTLED_HANDSHAKES = 24
"""Past the twenty R-B-L6-001's floor asks for."""
UNSETTLED_HANDSHAKES = 5


class Beaconer(typing.NamedTuple):
    """A host, how settled its TLS history is, and when and where its new fingerprint and its beacon happen."""

    host: str
    history: int
    fingerprint_destination: str
    fingerprint_minute: float
    beacon_destination: str
    beacon_start_minute: float
    """Minutes after the campaign hour starts; negative for the hour before."""


C2_ATTACKER = "10.20.2.5"
BEACON_FIRST = "10.20.2.6"
OTHER_DESTINATION = "10.20.2.7"
C2_TOO_SLOW = "10.20.2.8"
UNSETTLED = "10.20.2.9"
BEACONERS = {
    C2_ATTACKER: Beaconer(C2_ATTACKER, SETTLED_HANDSHAKES, "203.0.113.100", 0, "203.0.113.100", 2),
    BEACON_FIRST: Beaconer(BEACON_FIRST, SETTLED_HANDSHAKES, "203.0.113.101", 0, "203.0.113.101", -60),
    OTHER_DESTINATION: Beaconer(OTHER_DESTINATION, SETTLED_HANDSHAKES, "203.0.113.102", 0, "203.0.113.112", 2),
    C2_TOO_SLOW: Beaconer(C2_TOO_SLOW, SETTLED_HANDSHAKES, "203.0.113.103", 0, "203.0.113.103", 55),
    UNSETTLED: Beaconer(UNSETTLED, UNSETTLED_HANDSHAKES, "203.0.113.104", 0, "203.0.113.104", 2),
}


def hosts() -> list:
    """Every host a principal logs into or a campaign reaches, as the shared host rule renders it."""
    names = set()

    for actor in ACTORS.values():
        names.update((normalize_hostname(actor.home), normalize_hostname(actor.target)))

    return sorted(names)


def asset_versions() -> list:
    """The inventory: every server in one peer group, and the finance workstation in its own."""
    versions = []

    for host in hosts():
        workstation = host in WORKSTATIONS
        versions.append(
            make_version("asset",
                         host,
                         0,
                         None,
                         0,
                         values={
                             "owner": "finance@example.com" if workstation else "it-ops@example.com",
                             "owning_team": "finance" if workstation else "platform",
                             "criticality": "medium" if workstation else "high",
                             "data_classification": "confidential" if workstation else "internal",
                             "peer_group": "finance-workstations" if workstation else SERVER_GROUP,
                             "device_role": "workstation" if workstation else "server",
                             "os_family": "windows",
                             "os_version": "10.0.19045" if workstation else "10.0.20348",
                         }))

    return versions


def build_address_store() -> BitemporalStore:
    """The inventory the handshake enrichment reads: every address that holds a session or beacons is a managed
    workstation, keyed on the host name its address resolves to, as the layer 6 pipeline looks it up."""
    addresses = sorted(set(SESSION_HOLDERS.values()) | set(BEACONERS))

    return BitemporalStore(
        "asset",
        [
            make_version("asset",
                         normalize_hostname(host_identity.hostname_for(address)),
                         0,
                         None,
                         0,
                         values={
                             "device_role": "workstation", "criticality": "medium", "owner": "it-ops@example.com"
                         }) for address in addresses
        ])


def build_store() -> BitemporalStore:
    """The inventory the process enrichment reads."""
    return BitemporalStore("asset", asset_versions())


MAC_CLASS = "tc2_mac"
"""The two sites' MAC table snapshots. Closed into bindings and resolved through, not emitted as events."""
REPLAY_KMH = 900
"""R-C-005's speed, the guide's and R-D-L5-003's: above any commercial aircraft."""
HQ = "hq"
EDINBURGH = "edinburgh"
SITE_LOCATIONS = {HQ: (51.5074, -0.1278), EDINBURGH: (55.9533, -3.1883)}
"""The estate's own record of where its sites are, which a deployment supplies as it supplies the directory."""
LEASE_CLASS = "tc3_dhcp"
"""The estate's DHCP server: an acknowledgement per desk, which is what says which workstation held an address, and
the daily renewals of the workstations the other campaigns run from."""
INVENTORY_CLASS = host_identity.INVENTORY_CLASS
"""The EDR agents' daily inventory reports: which machine each workstation's network interface belongs to."""


def close_lease_bindings(config: Config, dataframes: list[pd.DataFrame], impose_order: bool = True) -> pd.DataFrame:
    """The DHCP server's acknowledgements, closed into leases: each one held until its term ran out or was released."""
    pipe = ResumableLinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    pipe.add_stage(CollectorSequenceStage(config))
    pipe.add_stage(
        TC2BindingStage(config,
                        key_column="ip",
                        attribute_columns=["mac", "switch_id", "port_id"],
                        action_column="dhcp_message",
                        grant_seconds_column="lease_seconds"))
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    return _collect(sink)


def build_lease_table(bindings: pd.DataFrame) -> BindingTable:
    """Which workstation each desk address was handed to, and for how long, as the DHCP server's leases closed."""
    # A DHCP server every record of which was quarantined granted nothing anyone can see, and the table is empty.
    if (len(bindings) == 0):
        return BindingTable(name="dhcp_lease", value_columns=["mac", "switch_id", "port_id"], bindings=[])

    return BindingTable.from_dataframe(bindings,
                                       name="dhcp_lease",
                                       key_column="ip",
                                       value_columns=["mac", "switch_id", "port_id"],
                                       start_column="bind_start",
                                       end_column="bind_end")


def build_mac_table(bindings: pd.DataFrame) -> BindingTable:
    """Which port, at which site, a workstation was plugged into, as TC2BindingStage closed it."""
    # A MAC table collector every record of which was quarantined closed no bindings, and the table is empty.
    if (len(bindings) == 0):
        return BindingTable(name="mac_table", value_columns=["port_key", "site_id"], bindings=[])

    return BindingTable.from_dataframe(bindings,
                                       name="mac_table",
                                       key_column="mac_address",
                                       value_columns=["port_key", "site_id"],
                                       start_column="bind_start",
                                       end_column="bind_end")


PORT_CLASS = "tc1"
"""The headquarters access switch's SNMP polls of the desk ports R-C-003's actors plug into."""
PHYSICAL_WINDOW_SECONDS = 4 * 3600
"""R-C-003's window, the guide's: the port event, the new vendor and the management flow inside four hours."""
MANAGEMENT_PREFIXES = ("10.99.0.0/24", )
"""The estate's management network, where its switches', routers' and controllers' own interfaces live. A
deployment supplies it as it supplies the directory, and records it against VLAN 99 in `vlan_designations`."""
LEASE_METHOD_COLUMN = "login_lease_resolution"
PORT_METHOD_COLUMN = "login_port_resolution"
LEASE_UID_COLUMN = "login_lease_uid"
"""The DHCP lease a sign-in's source address was resolved to a workstation through."""
PORT_UID_COLUMN = "login_port_uid"
"""The MAC binding the workstation was resolved to a switch port through."""


def _login_ladder(config: Config, mac_table: BindingTable, lease_table: BindingTable) -> list:
    """A sign-in's source address to a workstation through the leases, to a port and site through the MAC bindings,
    and the journey between the sites of a principal's consecutive sign-ins."""
    return [
        BindingResolverStage(config,
                             binding_table=lease_table,
                             key_column="source_ip",
                             output_columns={"mac": "login_mac"},
                             method_column=LEASE_METHOD_COLUMN,
                             uid_column=LEASE_UID_COLUMN),
        BindingResolverStage(config,
                             binding_table=mac_table,
                             key_column="login_mac",
                             output_columns={
                                 "port_key": "login_port_key", "site_id": "login_site_id"
                             },
                             method_column=PORT_METHOD_COLUMN,
                             uid_column=PORT_UID_COLUMN),
        TC5TravelStage(config, column_prefix="site_travel", location_column="login_site_id", locations=SITE_LOCATIONS),
    ]


def _stages(config: Config,
            telemetry_class: str,
            mac_table: typing.Optional[BindingTable] = None,
            lease_table: typing.Optional[BindingTable] = None,
            inventory_table: typing.Optional[BindingTable] = None) -> tuple:
    """Each layer's own stages, its OSI layer and the entity it is sealed on."""
    if (telemetry_class == FLOW_CLASS):
        # The source address to its interface and its machine, and the chain rooted on the machine where there is one,
        # as the network corpus does.
        resolvers = [] if inventory_table is None else host_identity.resolver_stages(
            config, lease_table, inventory_table)

        return (
            [
                CommunityIdStage(config, dst_ip_column="dst_ip", dst_port_column="dst_port"),
                TC3CardinalityStage(config, window_seconds=PERIOD_SECONDS),
                TC3ReachStage(config, window_seconds=PERIOD_SECONDS, management_prefixes=list(MANAGEMENT_PREFIXES)),
                TC3BaselineStage(config, bucket_seconds=PERIOD_SECONDS),
                TC3BeaconStage(config, window_seconds=2 * PERIOD_SECONDS),
                TC3TtlStage(config, window_seconds=2 * PERIOD_SECONDS),
                # The same layer 3 composition as the network corpus, so the flows meet the sourcetype's one contract.
                # There is no fortnight of history here, so every baseline is immature and none of the host-baseline
                # rules can fire on this corpus; the destination's context comes from the address inventory.
                TC0EnrichStage(config, store=build_address_store(), entity_column="dst_ip", prefix="dst_ctx_"),
                *resolvers,
            ],
            3,
            ["src_ip"])

    if (telemetry_class == SESSION_CLASS):
        return ([TC5SessionStage(config)], 5, ["user_principal"])

    # R-C-003's first two steps go through the same layer 1 and layer 2 compositions the telemetry corpus's polls
    # and table walks do, so they meet the same sourcetypes' contracts.
    if (telemetry_class == PORT_CLASS):
        return (telemetry_pipeline.layer_1_stages(config), 1, ["site_id", "device_id", "port_id"])

    if (telemetry_class == MAC_CLASS):
        return (telemetry_pipeline.mac_table_stages(config), 2, ["mac_address"])

    if (telemetry_class == AUTH_CLASS):
        ladder = [] if mac_table is None else _login_ladder(config, mac_table, lease_table)

        return ([TC5NoveltyStage(config, target_host_column="target_host")] + ladder, 5, ["user_principal"])

    return ([
        TC0EnrichStage(config, store=build_store(), entity_column="hostname", entity_normalization="hostname"),
        TC7EndpointStage(config)
    ],
            7, ["hostname"])


def close_mac_bindings(config: Config, dataframes: list[pd.DataFrame], impose_order: bool = True) -> pd.DataFrame:
    """The two sites' MAC tables, closed into bindings by the stage that closes every MAC binding in this fork."""
    pipe = ResumableLinearPipeline(config)
    # Copies, because a source frame is where a stage's needed columns are added, and the same walks are scored as
    # their own class afterwards: given the originals, every walk would arrive there carrying the binding columns.
    pipe.set_source(InMemorySourceStage(config, dataframes=[frame.copy() for frame in dataframes]))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    pipe.add_stage(CollectorSequenceStage(config))

    pipe.add_stage(TC2BindingStage(config))
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    return _collect(sink)


def run_class(config: Config,
              telemetry_class: str,
              dataframes: list[pd.DataFrame],
              impose_order: bool = True,
              mac_table: typing.Optional[BindingTable] = None,
              lease_table: typing.Optional[BindingTable] = None,
              inventory_table: typing.Optional[BindingTable] = None) -> pd.DataFrame:
    """Source, stamp, total order, the layer's stages, determinism stamp, envelope, hourly seal, sink: one collector."""
    (stages, osi_layer, entity_columns) = _stages(config, telemetry_class, mac_table, lease_table, inventory_table)
    # A flow's chain is rooted on the machine it came from where its address resolved to one.
    anchored = telemetry_class == FLOW_CLASS and inventory_table is not None

    pipe = ResumableLinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    pipe.add_stage(CollectorSequenceStage(config))

    for stage in stages:
        pipe.add_stage(stage)

    pipe.add_stage(DeterminismStampStage(config, envelope=stamping.envelope_for(telemetry_class, SETTINGS,
                                                                                rules=RULES)))
    pipe.add_stage(EnvelopeStampStage(config, osi_layer=osi_layer, entity_columns=entity_columns))
    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=PERIOD_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        entity_key_column=DEFAULT_ANCHOR_COLUMN if anchored else "entity_key"))

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    frame = _collect(sink)

    # A class every record of which was quarantined produced nothing to key.
    if (len(frame) == 0):
        return frame

    frame["telemetry_class"] = telemetry_class
    frame["row_key"] = frame["event_uid"]

    return frame


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True) -> pd.DataFrame:
    """
    Each collector through its own layer's pipeline, and every collector's output as one canonical frame.

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
        Every collector's output, plus any records quarantined at validation as the `quarantine` class, tagged
        with `telemetry_class`, keyed by `row_key`, canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    # Validated once per class, ahead of every stage: ingest_time stamped, the envelope checked, untrusted clocks
    # taken out of the stream and kept.
    (batches, quarantined) = envelope.validate(config, batches, ID_COLUMNS)

    # R-C-004's last three collectors go through the composed pipelines their own rules are asserted over, so the
    # chain is measured on exactly what R-B-L4-005, the layer 6 stages and R-B-L7-002 see.
    delegated = {TRANSFER_CLASS: transport_pipeline, HANDSHAKE_CLASS: presentation_pipeline, SAAS_CLASS: saas_pipeline}
    frames = []

    # R-C-005's ladder resolves through bindings the layer 2 collector closes, so those are closed first, the way
    # the estate harness closes its supplicant bindings before resolving through them.
    mac_table = build_mac_table(close_mac_bindings(config, batches[MAC_CLASS], impose_order=impose_order))
    # And the addresses the sign-ins came from are named by the DHCP server's leases, closed the same way.
    lease_table = build_lease_table(close_lease_bindings(config, batches[LEASE_CLASS], impose_order=impose_order))
    # Which machine each workstation's interface belongs to, which the flows, transfers and handshakes resolve through.
    inventory_table = host_identity.inventory_table(
        host_identity.close_inventory(config, batches.get(INVENTORY_CLASS, []), impose_order=impose_order))
    resolved = {"hosts": (lease_table, inventory_table)}
    # Which principal's session held each address. Built from the sessions class's output once it has run; a
    # sessions collector every record of which was quarantined leaves nothing to resolve through.
    sessions = host_identity.session_table(pd.DataFrame(), 0)

    # The handshakes' sources are enriched from this estate's inventory, not from the layer 6 corpus's.
    stores = {HANDSHAKE_CLASS: {"store": build_address_store(), **resolved}, TRANSFER_CLASS: resolved, SAAS_CLASS: {}}

    for name in CLASSES:
        # A collector every record of which was quarantined delivered nothing for its class to run over.
        if (len(batches[name]) == 0):
            continue

        if (name in delegated):
            # The packets, handshakes and exports run after the sessions, and carry the session that held their
            # source address -- the key R-C-004 joins its steps on where it is present.
            frames.append(delegated[name].run_pipeline(config, {name: batches[name][0]},
                                                       batches={name: batches[name]},
                                                       impose_order=impose_order,
                                                       sessions=sessions,
                                                       **stores.get(name, {})))
        else:
            frames.append(
                run_class(config,
                          name,
                          batches[name],
                          impose_order=impose_order,
                          mac_table=mac_table,
                          lease_table=lease_table,
                          inventory_table=inventory_table))

            if (name == SESSION_CLASS and len(frames[-1]) > 0):
                sessions = host_identity.session_table(session_intervals(frames[-1]), at(LAST_DAY + 1))

    # R-C-003's port polls and table walks, emitted as the telemetry corpus emits its own. The walks are also what
    # the MAC bindings above were closed from.
    for name in (PORT_CLASS, MAC_CLASS):
        if (len(batches.get(name, [])) > 0):
            frames.append(run_class(config, name, batches[name], impose_order=impose_order))

    # Every collector whose records the classes emit can be held at once.
    combined = pd.concat(frames, ignore_index=True) if len(frames) > 0 else pd.DataFrame()

    return canonicalize(envelope.with_quarantine(combined, quarantined),
                        key_columns=KEY_COLUMNS,
                        ignore_columns=IGNORE_COLUMNS)


EDGE_HOPS = {
    "hop_method_columns": [
        LEASE_METHOD_COLUMN,
        PORT_METHOD_COLUMN,
        host_identity.LEASE_METHOD_COLUMN,
        host_identity.INVENTORY_METHOD_COLUMN
    ],
    "hop_uid_columns": [
        LEASE_UID_COLUMN, PORT_UID_COLUMN, host_identity.LEASE_UID_COLUMN, host_identity.INVENTORY_UID_COLUMN
    ],
    "hop_layers": [3, 2, 3, 7],
}
"""The campaign's resolution hops: a sign-in's address through the DHCP leases and its workstation through the MAC
bindings, and a flow's, packet's or handshake's address through the leases and its interface through the EDR
inventory. The end of each edge is the layer of the collector the binding was closed from."""


def build_edges(config: Config, result: pd.DataFrame) -> pd.DataFrame:
    """
    The campaign's `morpheus:edge` records: one per hop a record was resolved through, and one per layer a
    connection was seen at above the first, joined on the Community ID every layer stamped.
    """
    return edge_stream.emit(config, result, join_column="community_id", **EDGE_HOPS)


def render(result: pd.DataFrame) -> str:
    """The canonical CSV rendering the golden file holds."""
    return result.to_csv(index=False, lineterminator="\n")


def _truthy(value) -> bool:
    return False if pd.isna(value) else bool(value)


def fanout_rises(result: pd.DataFrame) -> pd.DataFrame:
    """Step 1: the first flow per source and hour whose fan-out exceeds the source's peak in the hour before.

    A source with no flows in the hour before has nothing to rise above and is not counted.
    """
    flows = result.loc[(result["telemetry_class"] == FLOW_CLASS) & result["dsts_per_src"].notna(),
                       ["src_ip", "window_id", "dsts_per_src", "event_time", "lineage_id"]].copy()
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
                     server_only: bool = True,
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

    if (server_only):
        processes = processes[(processes["ctx_device_role"] == "server").fillna(False)]

    fired: dict = {}

    for (_, start) in starts.iterrows():
        t1 = int(start["event_time"])
        candidates = logins if not same_source else logins[logins["source_ip"] == start["src_ip"]]

        for (_, login) in candidates.iterrows():
            t2 = int(login["event_time"])

            if (ordered and t2 < t1 - tolerance_ns):
                continue

            host = login["target_host_key"]

            for (_, process) in processes[processes["endpoint_host"] == host].iterrows():
                t3 = int(process["event_time"])

                if (ordered and t3 < t2 - tolerance_ns):
                    continue

                if (window_seconds is not None and max(t2, t3) - t1 > window_seconds * NS_PER_SECOND):
                    continue

                key = (start["src_ip"], login["user_principal"], host)
                fired.setdefault(key, (t1, t2, t3))

    return fired


def connections_across_layers(result: pd.DataFrame) -> dict:
    """Community IDs every one of the flow, packet and TLS collectors recorded, with the layers' rows for each."""
    layered = result[result["telemetry_class"].isin((FLOW_CLASS, TRANSFER_CLASS, HANDSHAKE_CLASS))
                     & result["community_id"].notna()]
    seen = layered.groupby("community_id")["osi_layer"].nunique()

    return {
        community_id: layered[layered["community_id"] == community_id]
        for (community_id, layers) in seen.items() if layers == 3
    }


def session_intervals(result: pd.DataFrame) -> pd.DataFrame:
    """Each session's principal, address and the interval it was open over. An unended session is open at the end."""
    rows = result[result["telemetry_class"] == SESSION_CLASS].copy()
    rows["event_time"] = rows["event_time"].astype("int64")
    starts = rows[rows["session_lifecycle"] == "start"].groupby(["session_key", "user_principal",
                                                                 "source_ip"])["event_time"].min().rename("opened")
    ends = rows[rows["session_lifecycle"] == "end"].groupby("session_key")["event_time"].max().rename("closed")

    return starts.reset_index().merge(ends.reset_index(), on="session_key", how="left")


HARD_SESSION = "hard:session_key"
VALUE_ADDRESS = "value:source_ip"
HARD_CONNECTION = "hard:community_id"
VALUE_PAIR = "value:src_ip,dst_ip"


def _key(row: pd.Series, column: str) -> typing.Optional[str]:
    value = row.get(column)

    return None if value is None or pd.isna(value) else value


def staged_exfiltration(result: pd.DataFrame,
                        in_session: bool = True,
                        session_bound: bool = True,
                        new_issuer: bool = True,
                        ordered: bool = True,
                        window_seconds: typing.Optional[int] = EXFIL_WINDOW_SECONDS,
                        tolerance_seconds: int = JOIN_TOLERANCE_SECONDS) -> dict:
    """(principal, address) chains R-C-004 fires on, each with its three step times.

    `session_bound` off lets a breach from any address count; `in_session` off keeps the address a session of the
    principal's but drops the requirement that it was open at the breach.
    """
    return {
        chain: tuple(times)
        for (chain, (*times, _)) in exfiltration_chains(
            result, in_session=in_session, session_bound=session_bound, new_issuer=new_issuer, ordered=ordered,
            window_seconds=window_seconds, tolerance_seconds=tolerance_seconds).items()
    }


def exfiltration_chains(result: pd.DataFrame,
                        in_session: bool = True,
                        session_bound: bool = True,
                        new_issuer: bool = True,
                        ordered: bool = True,
                        window_seconds: typing.Optional[int] = EXFIL_WINDOW_SECONDS,
                        tolerance_seconds: int = JOIN_TOLERANCE_SECONDS) -> dict:
    """
    `staged_exfiltration`, with how each chain's steps were joined: the three step times and the join method.

    Where a step carries the `session_key` of the session that held its source address, it is joined to the
    principal's session on that key, `hard:session_key`. Where it carries none -- a collector nothing resolved -- it
    is joined on the address, and a breach only while the session holding that address was open, as the rule
    always did, `value:source_ip`. A chain is hard only if every step was.
    """
    tolerance_ns = tolerance_seconds * NS_PER_SECOND
    window_ns = None if window_seconds is None else window_seconds * NS_PER_SECOND

    exports = result[(result["telemetry_class"] == SAAS_CLASS)
                     & result["saas_baseline_mature"].map(_truthy)
                     & (result["saas_record_ratio"].astype("Float64") > RECORD_MULTIPLE).fillna(False)]
    breaches = result[(result["telemetry_class"] == TRANSFER_CLASS)
                      & (result["flow_data_len_envelope_breached"].map(_truthy)
                         | result["flow_bpp_envelope_breached"].map(_truthy))]
    handshakes = result[result["telemetry_class"] == HANDSHAKE_CLASS]

    if (new_issuer):
        handshakes = handshakes[handshakes["cert_issuer_new_to_estate"].map(_truthy)]

    sessions = session_intervals(result)
    fired: dict = {}

    for (_, export) in exports.iterrows():
        principal = export["user_principal"]
        t1 = int(export["event_time"])
        own = sessions[sessions["user_principal"] == principal]

        # An export placed in a session is joined to that session alone.
        if (_key(export, "session_key") is not None):
            own = own[own["session_key"] == export["session_key"]]

        for (_, breach) in breaches.iterrows():
            t2 = int(breach["event_time"])
            address = breach["src_ip"]
            breach_session = _key(breach, "session_key")

            if (ordered and t2 < t1 - tolerance_ns) or (window_ns is not None and t2 - t1 > window_ns):
                continue

            if (session_bound):
                if (breach_session is not None and in_session):
                    held = own[own["session_key"] == breach_session]
                else:
                    held = own[own["source_ip"] == address]

                    if (in_session):
                        held = held[(held["opened"] <= t2) & (held["closed"].isna() | (held["closed"] >= t2))]

                if (len(held) == 0):
                    continue

            for (_, handshake) in handshakes[handshakes["src_ip"] == address].iterrows():
                t3 = int(handshake["event_time"])
                handshake_session = _key(handshake, "session_key")

                if (ordered and t3 < t2 - tolerance_ns) or (window_ns is not None and t3 - t1 > window_ns):
                    continue

                if (session_bound and in_session and breach_session is not None and handshake_session is not None
                        and handshake_session != breach_session):
                    continue

                hard = all(key is not None for key in (_key(export, "session_key"), breach_session, handshake_session))
                fired.setdefault((principal, address), (t1, t2, t3, HARD_SESSION if hard else VALUE_ADDRESS))

    return fired


def tls_before_beaconing(result: pd.DataFrame,
                         settled: bool = True,
                         same_destination: bool = True,
                         ordered: bool = True,
                         window_seconds: typing.Optional[int] = C2_WINDOW_SECONDS,
                         tolerance_seconds: int = JOIN_TOLERANCE_SECONDS) -> dict:
    """(host, destination) pairs R-C-002 fires on, with the fingerprint's time and the beacon's.

    The beacon is dated when it first matures, which is when R-B-L3-002 would first report it.
    """
    return {
        pair: tuple(times)
        for (pair, (*times,
                    _)) in c2_chains(result, settled=settled, same_destination=same_destination, ordered=ordered,
                                     window_seconds=window_seconds, tolerance_seconds=tolerance_seconds).items()
    }


def c2_chains(result: pd.DataFrame,
              settled: bool = True,
              same_destination: bool = True,
              ordered: bool = True,
              window_seconds: typing.Optional[int] = C2_WINDOW_SECONDS,
              tolerance_seconds: int = JOIN_TOLERANCE_SECONDS) -> dict:
    """
    `tls_before_beaconing`, with how each pair's fingerprint was joined to the flows: the two times and the method.

    A fingerprint whose connection the flow exporter also recorded -- one Community ID at layers 3 and 6 -- takes its
    host and destination from that flow record, `hard:community_id`. One whose connection no flow record carries is
    joined on its own addresses, `value:src_ip,dst_ip`, as the rule always did.
    """
    tolerance_ns = tolerance_seconds * NS_PER_SECOND
    fingerprints = result[(result["telemetry_class"] == HANDSHAKE_CLASS) & result["ja4_client_first_seen"].map(_truthy)]

    if (settled):
        fingerprints = fingerprints[fingerprints["ja4_client_observations"].astype("Int64").fillna(0) >= 20]

    flows = result[(result["telemetry_class"] == FLOW_CLASS) & result["flow_regularity_mature"].map(_truthy)]
    flows = flows[(flows["flow_interval_cv"].astype(float) < 0.15) & (flows["flow_size_cv"].astype(float) < 0.15)]
    beacons = flows.groupby(["src_ip", "dst_ip"])["event_time"].min().astype("int64")
    carried = result[(result["telemetry_class"] == FLOW_CLASS) & result["community_id"].notna()]
    carried = carried.drop_duplicates("community_id").set_index("community_id")[["src_ip", "dst_ip"]]
    fired: dict = {}

    for (_, fingerprint) in fingerprints.iterrows():
        t1 = int(fingerprint["event_time"])
        connection = _key(fingerprint, "community_id")

        if (connection is not None and connection in carried.index):
            (host, reached, method) = (*carried.loc[connection], HARD_CONNECTION)
        else:
            (host, reached, method) = (fingerprint["src_ip"], fingerprint["dst_ip"], VALUE_PAIR)

        for ((source, destination), t2) in beacons.items():
            if (source != host) or (same_destination and destination != reached):
                continue

            if (ordered and t2 < t1 - tolerance_ns):
                continue

            if (window_seconds is not None and t2 - t1 > window_seconds * NS_PER_SECOND):
                continue

            fired.setdefault((source, destination), (t1, int(t2), method))

    return fired


def credential_replays(result: pd.DataFrame, kmh_threshold: float = REPLAY_KMH) -> dict:
    """Principals R-C-005 fires on, each with the port and site it was last seen at and the one it appeared at.

    The search orders each principal's measurable sign-ins and carries the previous one forward, which is the anchor
    the travel stage measured from: a sign-in the stage did not measure or anchor on is neither end of a journey.
    """
    logins = result[(result["telemetry_class"] == AUTH_CLASS)
                    & result["site_travel_status"].isin(["measured", "first_for_principal"])]
    fired: dict = {}

    for (principal, rows) in logins.sort_values("event_time").groupby("user_principal"):
        previous_port = None

        for (index, (_, row)) in enumerate(rows.iterrows()):
            if (index > 0 and row["site_travel_status"] == "measured"
                    and float(row["site_travel_kmh"]) >= kmh_threshold):
                fired.setdefault(principal, (previous_port, row["login_port_key"], float(row["site_travel_kmh"])))

            previous_port = row["login_port_key"]

    return fired


def physical_compromises(result: pd.DataFrame,
                         port_event: bool = True,
                         new_vendor: bool = True,
                         management: bool = True,
                         window_seconds: typing.Optional[int] = PHYSICAL_WINDOW_SECONDS,
                         tolerance_seconds: int = JOIN_TOLERANCE_SECONDS) -> dict:
    """
    (port, MAC) pairs R-C-003 fires on, each with its three step times.

    The port's layer 1 events are joined to the walks on the port, and the walks to the flows on the device's MAC,
    which the flow carries from the DHCP lease that named its address. Each switch drops one condition, so a test can
    show that the condition, and not something else, is what keeps a control quiet.
    """
    tolerance_ns = tolerance_seconds * NS_PER_SECOND
    window_ns = None if window_seconds is None else window_seconds * NS_PER_SECOND

    polls = result[result["telemetry_class"] == PORT_CLASS]

    if (port_event):
        polls = polls[polls["transceiver_serial_changed"].map(_truthy)
                      | (polls["link_flaps"].astype("Int64").fillna(0) > 0)]

    walks = result[(result["telemetry_class"] == MAC_CLASS) & result["port_key"].notna()]

    if (new_vendor):
        walks = walks[walks["oui_first_seen_on_port"].map(_truthy)]

    flows = result[(result["telemetry_class"] == FLOW_CLASS) & result["mac_address"].notna()]

    if (management):
        flows = flows[flows["dst_is_management"].map(_truthy)]

    fired: dict = {}

    for (_, poll) in polls.iterrows():
        t1 = int(poll["event_time"])

        for (_, walk) in walks[walks["port_key"] == poll["entity_key"]].iterrows():
            t2 = int(walk["event_time"])

            if (t2 < t1 - tolerance_ns) or (window_ns is not None and t2 - t1 > window_ns):
                continue

            for (_, flow) in flows[flows["mac_address"] == walk["mac_address"]].iterrows():
                t3 = int(flow["event_time"])

                if (t3 < t2 - tolerance_ns) or (window_ns is not None and t3 - t1 > window_ns):
                    continue

                key = (walk["port_key"], walk["mac_address"])
                (s1, s2, s3) = fired.get(key, (t1, t2, t3))
                fired[key] = (min(s1, t1), min(s2, t2), min(s3, t3))

    return fired
