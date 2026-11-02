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
The snapshot-shaped layer 1 and layer 2 corpus, and the composed telemetry pipeline, for the determinism harness.

The lineage harness proved the lineage substrate deterministic. Nothing had ever run a telemetry stage composed
with another, and the retrospective found that the telemetry stages' own tests were shaped like the tests rather
than like the network. This module is the answer to both. Its corpus is shaped the way collectors actually report:
per-port SNMP polls with uptime and `ifLastChange` in hundredths of a second, a full MAC table snapshot every five
minutes with every address on a port stamped at the snapshot's instant, and ARP at one-second resolution with a
whole burst landing on one tick.

Into that corpus are planted the things the layer 1 and layer 2 features exist to see:

- a **hub**: five MAC addresses behind one access port from the seventh snapshot onward, four above the most the
  port had carried in any of the six before, which is the step R-B-L2-002 fires on;
- a **spoof**: one MAC reported on two ports in the same snapshot;
- a **cross-switch spoof**: one MAC claimed on a second switch two seconds after it was seen on its own, which is
  the shape a sequentially polled estate actually produces and the one the simultaneous case cannot stand in for;
- a **legitimate move**: one MAC that changes port between snapshots, so a whole cadence separates its two
  sightings and the rule that catches the spoof has something it must not fire on;
- a **flood**: twenty gratuitous ARP replies from one host in one second, claiming the gateway;
- a **desk spoof**: five gratuitous replies claiming the gateway from the workstation on the tapped port, six
  minutes after the tap, so one port's chain is accused by a layer 1 and a layer 2 detection in the same
  window -- the man-in-the-middle at a desk that the cross-layer chain search exists to assemble, where the flood
  above accuses a different port's chain and on its own stays under that search's threshold;
- a **reboot**: a device whose uptime and counters restart mid-corpus;
- a **tap**: a step loss of receive power on one port, with transmit power unchanged;
- a **failing optic**: a receive level sliding down a few hundredths of a decibel every poll until the maintenance
  swap below replaces it, which is what R-P-L1-004 fires on, where the tap's step must not;
- a **flap**: a link that went down and up between two polls, visible only through `ifLastChange`;
- an **optic swap**: a transceiver serial that changes on a port while the device records no link transition,
  which is what R-D-L1-001 fires on, and beside it a **maintenance swap** whose serial changes with the link's
  drop recorded in `ifLastChange`, which it must not;
- a **bypass**: an 802.1X success on a port that never started an exchange;
- a **flapper**: a link that goes down and up between every pair of polls for the second half of the hour, which
  R-D-L1-003 fires on and the one flap above must not;
- a **failing cable**: a port whose CRC errors climb every poll for the last quarter of the hour, R-B-L1-005's case;
- a **re-patch** and an **inline insertion**: two ports whose LLDP neighbour changes, the first to another of the
  estate's switches and the second to a device the estate has never seen, which R-D-L1-006 reads, beside a switch
  that comes back from its reboot with no neighbours yet, which it must not;
- a **surge** and a **silence**: a port that starts carrying fifteen times what it ever has, and one that stops
  carrying anything, which R-B-L1-007 reads in both directions;
- a **second VLAN** whose cameras share one vendor until a single-board computer appears on it, which R-B-L2-006
  reads, beside a third camera of the known vendor that it must not;
- a **printer unplugged** and a **guest who left**: a MAC the next table walk no longer lists, closed as absent from
  the snapshot, and one the switch's own notification says was removed, closed as an end somebody observed;
- an 802.1X exchange that **took three attempts** and one that **took no time at all**, which R-B-L2-007 and
  R-B-L2-008 read against the port's own exchanges;
- a **poller whose clock cannot be trusted**: a second SNMP poller, in the DMZ, whose last measured offset is four
  and a half seconds, polling one port three times -- quarantined by `EnvelopeValidateStage` before any stateful
  stage sees it, and returned as the `quarantine` class, which is what reaches `morpheus:quarantine`;
- a **sequence gap**: the ARP sensor numbers three records it never delivers, partway through the hour, which
  `CollectorSequenceStage` reports on the next record it does deliver and the health record counts;
- the **devices themselves**, polled once a minute as `tc1_device` rows keyed `site_id:device_id`: a
  configuration pushed inside the estate's change window and another changed outside it, which R-D-L1-008 fires on
  and the first must not; the rebooting switch's restart seen once, as the device's own event; a **reboot storm**,
  a switch whose restarts come one, then two, then three to a five-minute period, which R-P-L1-009 reads as a
  trajectory where the single restart must not; and a router that loses two thirds of its routes with half its
  BGP peers, which is context on the device row and fires nothing;
- and one thing that must **not** fire: a VRRP pair whose two MACs legitimately share one address, carried on the
  exclusion list, so the ARP rule's exclusion path is exercised rather than assumed.

New plants draw their randomness from a second generator, `EXTRA_SEED`, so that adding one does not move a single
value in the rows that were already here: the ARP and authentication streams, and the counts the documents quote
from them, stay what they were.

The pipeline is one per telemetry class, which is the deployment shape: each class arrives on its own topic and
its stages require its own columns. They compose where the design says they must. Layer 2's binding stage emits
closed bindings; those become a `BindingTable`; layer 2's ARP stream resolves through it; and the port it resolves
to is, byte for byte, the layer 1 `entity_key`. That join is asserted in the harness, not assumed.

Every stateful stage is preceded by `TotalOrderStage`, which is determinism control 8 as a stage. Without it the
permutation check fails, correctly: a counter delta is the difference from the previous sample, and the telemetry
stages flag out-of-order arrival rather than repairing it.
"""

import typing

import pandas as pd

from morpheus.compositions import envelope as universal_envelope
from morpheus.compositions import stamping
from morpheus.compositions.corpus_pipeline import _collect
from morpheus.compositions.corpus_pipeline import (  # noqa: F401 pylint: disable=unused-import  # re-exported for callers
    build_pipeline_config, )
from morpheus.compositions.resumable import ResumableLinearPipeline
from morpheus.config import Config
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.binding_resolver_stage import BindingResolverStage
from morpheus.stages.lineage.chain_anchor_stage import DEFAULT_ANCHOR_COLUMN
from morpheus.stages.lineage.chain_anchor_stage import ChainAnchorStage
from morpheus.stages.lineage.collector_sequence_stage import CollectorSequenceStage
from morpheus.stages.lineage.determinism_stamp_stage import DeterminismStampStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.stages.telemetry.tc0_network_stage import DEVICE as DEVICE_KIND
from morpheus.stages.telemetry.tc0_network_stage import PORT as PORT_KIND
from morpheus.stages.telemetry.tc0_network_stage import VLAN as VLAN_KIND
from morpheus.stages.telemetry.tc0_network_stage import TC0NetworkStage
from morpheus.stages.telemetry.tc1_binding_stage import TC1BindingStage
from morpheus.stages.telemetry.tc1_change_stage import TC1ChangeStage
from morpheus.stages.telemetry.tc1_device_stage import TC1DeviceStage
from morpheus.stages.telemetry.tc1_flap_stage import TC1FlapStage
from morpheus.stages.telemetry.tc1_forecast_stage import TC1ForecastStage
from morpheus.stages.telemetry.tc1_normalize_stage import TC1NormalizeStage
from morpheus.stages.telemetry.tc1_optical_stage import TC1OpticalStage
from morpheus.stages.telemetry.tc1_rate_stage import TC1RateStage
from morpheus.stages.telemetry.tc2_arp_stage import TC2ArpStage
from morpheus.stages.telemetry.tc2_auth_stage import TC2AuthStage
from morpheus.stages.telemetry.tc2_baseline_stage import TC2BaselineStage
from morpheus.stages.telemetry.tc2_binding_stage import TC2BindingStage
from morpheus.stages.telemetry.tc2_cardinality_stage import TC2CardinalityStage
from morpheus.utils import bitemporal
from morpheus.utils.binding_table import BindingTable
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize
from morpheus.utils.lineage import event_uid

PERIOD_SECONDS = 300
LATENESS_SECONDS = 900
CORPUS_SECONDS = 3600
ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []
MAC_A = "aa:bb:cc:00:00:01"
MAC_B = "aa:bb:cc:00:00:02"
MAC_C = "aa:bb:cc:00:00:03"
VRRP_IP = "10.0.0.254"
MAC_ACTION_COLUMN = "mac_action"
"""What a MAC table row says happened: null on a snapshot row, `removed` on the switch's own notification."""
BASELINE_MIN_BUCKETS = 6
"""Snapshot periods a port must have been seen in before R-B-L2-002 has a baseline to measure it against.

Half an hour of five-minute snapshots. A deployment keeps a month of hourly peaks and asks for a day of them, which
is what `TC2BaselineStage` defaults to; this corpus is one hour long, so its periods are the sealing period and its
floor is the six snapshots before the hub arrives. The parameters say so rather than the corpus pretending to be a
month.
"""
OPTIC_TYPE = "10GBASE-LR"
OPTIC_FLOOR_DBM = -14.4
OPTIC_FLOORS = {OPTIC_TYPE: OPTIC_FLOOR_DBM}
"""The one optic type this estate runs, and the level its receiver stops working at, from the datasheet.

Supplied to `TC1ForecastStage` the way the estate's site coordinates are supplied to the travel stage: a fact about
the hardware rather than about any poll, so it is not carried on the event.
"""
AUTH_BASELINE_MIN_SAMPLES = 20
"""Prior exchanges a port needs before its distribution is published. Twenty, inside an hour; the stage's default is
a hundred, and below a hundred the 99th percentile is the slowest exchange the port has had."""
IDENTITIES = {
    MAC_A: "alice-ws",
    MAC_B: "bob-ws",
    MAC_C: "carol-ws",
    "aa:bb:cc:00:00:05": "phone-4021",
    "aa:bb:cc:00:00:06": "desk-4021",
    "de:ad:be:ef:01:01": "unknown-supplicant",
    "de:ad:be:ef:01:02": "unknown-supplicant",
}
"""The identity a supplicant presented, where it presented one. A device doing MAC authentication bypass has no
identity to present, which is what makes `unknown-supplicant` the honest value rather than a blank.

The three workstations present one because the ladder's next rung needs somewhere to stand. An 802.1X identity is
the only thing in this corpus a directory could bind a person to: a MAC is an asset, and asset assignment is a
different fact with a different lifetime. Falling back to the MAC left `dot1x_identity` carrying an address on
most exchanges, which is what the fallback is for and not what an estate with 802.1X deployed actually reports.
Nothing keys on it -- `TC2AuthStage` prefers `mac_address` when both are present, so the exchange key, the port
key and every window are untouched.
"""
DEVICE_CLASS = "tc1_device"
CHANGE_WINDOW = ["00:00", "00:10"]
"""The estate's change window, in UTC. The corpus starts at midnight, so its first ten minutes are inside it."""
NETWORK_CONTEXT_CLASS = "tc0_network"
"""The network inventory's records: what each port, device and VLAN was commissioned as, is for, and belongs to."""
CLASS_ENVELOPE = {
    "tc1": (1, ["site_id", "device_id", "port_id"]),
    "tc1_binding": (1, ["site_id", "device_id", "port_id"]),
    DEVICE_CLASS: (1, ["site_id", "device_id"]),
    "tc2_mac": (2, ["mac_address"]),
    "tc2_binding": (2, ["mac_address"]),
    "tc2_arp": (2, ["arp_sender_mac"]),
    "tc2_auth": (2, ["site_id", "switch_id", "port_id"]),
}
"""The layer each class came from, and the columns Part 2 names as its behavioral subject.

Layer 1 keys on the port and layer 2 on the MAC, which is the guide's own split: at layer 1 the port is the
thing that persists, and at layer 2 the MAC is the thing that moves between ports. The device class keys on the
switch or router itself, because a restart or a configuration push happens to the device and not to any one port.
The 802.1X class is the exception and keys on the port rather than the supplicant, because an exchange is a question
about the port that authorized it -- the same reasoning that gives `TC2AuthStage` its port-keyed timing.
"""
SETTINGS = {
    "period_seconds": PERIOD_SECONDS,
    "lateness_seconds": LATENESS_SECONDS,
    "baseline_min_buckets": BASELINE_MIN_BUCKETS,
    "optic_floors": OPTIC_FLOORS,
    "auth_baseline_min_samples": AUTH_BASELINE_MIN_SAMPLES,
    "change_window": CHANGE_WINDOW,
}
"""The settings that decide this corpus's output, digested into `config_hash` by `stamping.envelope_for`."""
RULES = ("R-D-L1-001",
         "R-D-L1-002",
         "R-D-L1-003",
         "R-P-L1-004",
         "R-B-L1-005",
         "R-D-L1-006",
         "R-B-L1-007",
         "R-D-L1-008",
         "R-P-L1-009",
         "R-D-L1-010",
         "R-B-L2-002",
         "R-D-L2-001",
         "R-D-L2-003",
         "R-D-L2-004",
         "R-D-L2-005",
         "R-B-L2-006",
         "R-B-L2-007",
         "R-B-L2-008")
"""The shipped rules that read this corpus's columns; their thresholds are folded into `pipeline_fingerprint`."""


def device_stages(config: Config, network: typing.Optional[dict] = None) -> list:
    """The device composition: restarts against the device's own periods, configuration changes placed against the
    estate's change window, and routing churn. Given the network context stores, each device's inventory record is
    joined on last, under `ctx_device_`."""
    return [
        TC1DeviceStage(
            config, change_window=list(CHANGE_WINDOW), bucket_seconds=PERIOD_SECONDS, min_buckets=BASELINE_MIN_BUCKETS),
    ] + _context(config, network, DEVICE_KIND, "device_key", DEVICE_PREFIX)


CHAIN_ROOTS = {
    "tc1": ["entity_key"],
    DEVICE_CLASS: ["device_key"],
    "tc2_mac": ["port_key"],
    "tc2_arp": ["resolved_port_key", "arp_sender_ip"],
    "tc2_auth": ["auth_port_key"],
}
"""What each class's correlation chain is rooted on, as candidates in order of preference, decided per row.

A layer 1 sample is about a port, so its chain is rooted on the port's `entity_key`; a device's poll is about the
device, and roots on its `device_key`. A MAC table row and an
802.1X exchange each name the port they were observed on directly, so they root on it too. An ARP observation
names only the address being claimed -- unless the ladder resolved its MAC to a port, in which case the
observation is about that port and joins the port's chain, with `chain_anchor_source` recording that it got
there through the binding table. An unresolved one stays rooted on the address, which is what it was rooted on
before and is still true. The binding classes have no entry because they are not sealed into windows at all.

These five classes are sealed together, in one pass over their union in event-time order, because a chain is a
Merkle root over its members and the members of a port's chain now come from two layers. Sealing each class on
its own would give the same root key two different roots, one per class, and nothing downstream could tell that
they were the same chain."""


def _run_context(config: Config, dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """Source → `TC0NetworkStage` → determinism stamp → sink, as `context_pipeline` runs every TC-0 producer: context
    is a store, not a stream, so nothing here orders, windows or seals it."""
    pipe = ResumableLinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[frame.copy() for frame in dataframes]))
    pipe.add_stage(TC0NetworkStage(config))
    pipe.add_stage(
        DeterminismStampStage(config, envelope=stamping.envelope_for(NETWORK_CONTEXT_CLASS, SETTINGS, rules=RULES)))
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    return _collect(sink)


def _run_class(config: Config,
               dataframes: list[pd.DataFrame],
               stages: list,
               impose_order: bool,
               seal: bool = True,
               anchor: str = None,
               envelope: tuple = None,
               chain: list[str] = None,
               telemetry_class: str = None) -> pd.DataFrame:
    """Source → stamp → (total order) → the class's stages → determinism stamp → (envelope) → (chain anchor) →
    (window seal) → sink.

    The envelope stamp goes after the class's own stages rather than before them, because two of these classes
    are produced by stages that replace the payload wholesale: a binding stage emits one row per interval, not
    one per observation, so anything stamped upstream of it is discarded along with the observations.

    A class given `chain` has its root chosen per row here and is sealed later, in `_seal_chains`, together with
    every other chained class. `anchor` is the older per-class form and seals the class on its own; the two are
    not combined.
    """
    if (chain is not None and anchor is not None):
        raise ValueError("a class is chained per row or anchored per class, not both")

    if (chain is not None and seal):
        raise ValueError("a chained class is sealed with the other chained classes, not on its own")

    pipe = ResumableLinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    pipe.add_stage(CollectorSequenceStage(config))

    for stage in stages:
        pipe.add_stage(stage)

    if (telemetry_class is None):
        raise ValueError("every class is stamped with the determinism envelope, and the envelope names the class")

    pipe.add_stage(DeterminismStampStage(config, envelope=stamping.envelope_for(telemetry_class, SETTINGS,
                                                                                rules=RULES)))

    if (envelope is not None):
        (osi_layer, entity_columns) = envelope
        pipe.add_stage(EnvelopeStampStage(config, osi_layer=osi_layer, entity_columns=entity_columns))

    if (chain is not None):
        pipe.add_stage(ChainAnchorStage(config, candidates=chain))

    if (seal):
        pipe.add_stage(
            WindowSealStage(config,
                            period_seconds=PERIOD_SECONDS,
                            lateness_seconds=LATENESS_SECONDS,
                            order_columns=list(DEFAULT_ORDER_COLUMNS),
                            entity_key_column=anchor))

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    return _collect(sink)


def _seal_chains(config: Config, outputs: dict[str, pd.DataFrame], parts: int = 1) -> dict[str, pd.DataFrame]:
    """
    Seal the chained classes together, so a port's chain holds its events from every layer that reached it.

    The classes are tagged, aligned to one column set, concatenated, and put in event-time order -- the order a
    deployment would see them in, where a layer 1 sample and the layer 2 observations on the same port arrive
    interleaved rather than one whole class after another. Fed class by class instead, the watermark would advance
    through the first class and declare every row of the second late. `parts` splits that ordered union into
    contiguous source frames so the batch-split sweep exercises this pass as well as the per-class ones.

    Each class comes back with its own columns plus the ones sealing adds, so nothing about a class's shape depends
    on which other classes it was sealed beside.
    """
    # A class every record of which was quarantined has nothing to seal, and stays empty.
    empty = {name: frame for (name, frame) in outputs.items() if len(frame) == 0}
    outputs = {name: frame for (name, frame) in outputs.items() if len(frame) > 0}

    if (len(outputs) == 0):
        return empty

    columns = {name: list(frame.columns) for (name, frame) in outputs.items()}
    tagged = []

    for (name, frame) in outputs.items():
        frame = frame.copy()
        frame["telemetry_class"] = name
        tagged.append(frame)

    union = pd.concat(tagged, ignore_index=True)
    union = union.sort_values(list(DEFAULT_ORDER_COLUMNS), kind="stable").reset_index(drop=True)

    size = max(1, len(union) // max(1, parts))
    dataframes = [union.iloc[start:start + size].reset_index(drop=True) for start in range(0, len(union), size)]

    sealer = WindowSealStage(config,
                             period_seconds=PERIOD_SECONDS,
                             lateness_seconds=LATENESS_SECONDS,
                             order_columns=list(DEFAULT_ORDER_COLUMNS),
                             entity_key_column=DEFAULT_ANCHOR_COLUMN)
    added = [name for name in sealer._needed_columns if name not in union.columns]  # pylint: disable=protected-access

    pipe = ResumableLinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(sealer)
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    sealed = _collect(sink)
    result = {}

    for name in outputs:
        rows = sealed[sealed["telemetry_class"] == name]
        result[name] = rows[columns[name] + added].reset_index(drop=True)

    return {**empty, **result}


def build_binding_table(bindings: pd.DataFrame) -> BindingTable:
    """The layer 2 bindings as the resolver consumes them: MAC to the port as layer 1 spells it."""
    # A MAC table collector every record of which was quarantined closed no bindings, and the table is empty.
    if (len(bindings) == 0):
        return BindingTable(name="mac_table", value_columns=["port_key", "vlan_id"], bindings=[])

    return BindingTable.from_dataframe(bindings,
                                       name="mac_table",
                                       key_column="mac_address",
                                       value_columns=["port_key", "vlan_id"],
                                       start_column="bind_start",
                                       end_column="bind_end")


PORT_INVENTORY_TABLE = "port_inventory"
"""What the layer 1 binding source calls itself on the shared bucketed sourcetype.

Several binding sources land on `binding:bucketed` and the refresh searches tell them apart with
`binding_table=...`, so this string is the one the app's L1 history refresh selects on.
"""
PORT_INVENTORY_COLUMNS = ["port_id", "switch_id", "site_id", "transceiver_serial", "lldp_neighbor_chassis_id"]
"""What the `binding_l1` lookups return, in the order the app's `fields_list` names them.

`port_id` and `switch_id` are values here rather than the key because the key is the composed `entity_key` and a
Splunk lookup matches on the two columns separately -- the layer 2 lookup hands them on individually, and nothing
on the search head composes them back into a port key.
"""


def build_port_binding_table(port_bindings: pd.DataFrame) -> BindingTable:
    """The layer 1 bindings as an interval table: the port, to the site, optic and neighbour it held.

    The inverse of the layer 2 table above. There the key is the mobile thing and the port is what it resolves to;
    here the port is the key and the optic is what moves.
    """
    return BindingTable.from_dataframe(port_bindings,
                                       name=PORT_INVENTORY_TABLE,
                                       key_column="entity_key",
                                       value_columns=list(PORT_INVENTORY_COLUMNS),
                                       start_column="bind_start",
                                       end_column="bind_end")


PORT_ANCHOR_COLUMN = "if_index"
"""What layer 1 keys a port on beside its name. The corpus's devices persist `ifIndex`, so a renamed port keeps its
history; an estate that does not persist it must not pass this (see `morpheus.utils.identity_anchor`)."""
PORT_PREFIX = "ctx_port_"
DEVICE_PREFIX = "ctx_device_"
VLAN_PREFIX = "ctx_vlan_"
"""Where each kind's context lands on an event, so a port's owner and its switch's owner never share a column."""


def network_stores(produced: pd.DataFrame) -> dict[str, BitemporalStore]:
    """One store per network kind, rebuilt from what `TC0NetworkStage` emitted, the way a consumer of
    `context:network` would. Separate, so a port's row is joined with port attributes only and never carries an
    empty column for every attribute a device or a VLAN has."""
    records = produced.to_dict("records")

    return {
        kind:
            BitemporalStore.from_records(kind,
                                         [record for record in records if record[bitemporal.CONTEXT_KIND] == kind])
        for kind in (PORT_KIND, DEVICE_KIND, VLAN_KIND)
    }


def _context(config: Config, network: typing.Optional[dict], kind: str, entity_column: str, prefix: str) -> list:
    """The point-in-time join of one network kind onto a class's rows, as the inventory knew it at each row's event
    time, or nothing when the composition was given no inventory."""
    if (network is None):
        return []

    return [TC0EnrichStage(config, store=network[kind], entity_column=entity_column, prefix=prefix, set_kinds={})]


def layer_1_stages(config: Config, network: typing.Optional[dict] = None) -> list:
    """
    The layer 1 composition: counters, the rates they become and the port's history of them, optics, flaps,
    identifier changes. Shared with the campaign corpus, whose port polls meet the same contract.

    The rate history is kept over the sealing period with the corpus's floor, as the layer 2 baseline is; the volume
    history is kept per hour of the day, which an hour-long corpus exercises without contradicting. Given the network
    context stores from `network_stores`, each port's inventory record is joined on last, under `ctx_port_`; the
    campaign corpus has no inventory and passes none.
    """
    return [
        TC1NormalizeStage(config, uptime_column="uptime", uptime_unit="cs", port_anchor_column=PORT_ANCHOR_COLUMN),
        TC1RateStage(config, bucket_seconds=PERIOD_SECONDS, min_buckets=BASELINE_MIN_BUCKETS),
        TC1OpticalStage(config),
        TC1ForecastStage(config, floors=OPTIC_FLOORS),
        TC1FlapStage(config, last_change_column="if_last_change", last_change_unit="cs"),
        TC1ChangeStage(config),
    ] + _context(config, network, PORT_KIND, "entity_key", PORT_PREFIX)


def mac_table_stages(config: Config, network: typing.Optional[dict] = None) -> list:
    """
    The layer 2 composition over MAC table walks: the cardinality features, each port's vendor history, and each
    port's count and each VLAN's vendor count against the peaks of their own earlier periods. Shared with the
    campaign corpus, whose two sites' tables meet the same contract. Given the network context stores, each VLAN's
    inventory record is joined on last, under `ctx_vlan_`.
    """
    return [
        TC2CardinalityStage(config),
        TC2BaselineStage(config, bucket_seconds=PERIOD_SECONDS, min_buckets=BASELINE_MIN_BUCKETS),
        TC2BaselineStage(config,
                         entity_column="vlan_key",
                         value_column="ouis_per_vlan",
                         bucket_seconds=PERIOD_SECONDS,
                         min_buckets=BASELINE_MIN_BUCKETS),
    ] + _context(config, network, VLAN_KIND, "vlan_key", VLAN_PREFIX)


def run_classes(config: Config,
                corpus: dict[str, pd.DataFrame],
                batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                impose_order: bool = True) -> dict[str, pd.DataFrame]:
    """
    Run every telemetry class through its own pipeline, unsealed, and return the outputs per class.

    Split out of `run_pipeline` so that a harness composing more layers than this one can add its classes to the
    union before it is sealed. Sealing is where a chain is decided, and a class sealed after the others can never
    join their chains, so the seam has to be here rather than after.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    corpus : dict
        The frames from `build_corpus`, possibly permuted.
    batches : dict, optional
        Per class, how the corpus is split across source frames. Defaults to one frame per class. The batch-split
        sweep is the caller's to vary.
    impose_order : bool, default = True
        Place `TotalOrderStage` ahead of the stateful stages. The permutation check's negative control turns it off
        to reproduce the removed-sort defect and prove the harness catches it.

    Returns
    -------
    dict
        Class name to its output frame, with `row_key` already set on the classes whose key is not `event_uid`.
        None of them carry window columns yet.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    # Every class's records are validated once, ahead of every pipeline the class feeds: `ingest_time` stamped, the
    # envelope checked, and a record whose clock cannot be trusted taken out before any stateful stage sees it.
    # The network inventory is context, not telemetry: it has no collector envelope to validate and is not sealed
    # into windows. Its records are written first, because every other class is joined against the stores they make.
    network_batches = batches.get(NETWORK_CONTEXT_CLASS)
    batches = {name: frames for (name, frames) in batches.items() if name != NETWORK_CONTEXT_CLASS}
    (batches, quarantined) = universal_envelope.validate(config, batches, ID_COLUMNS)
    outputs = {}
    network = None

    if (network_batches is not None):
        produced = _run_context(config, network_batches)
        network = network_stores(produced)
        produced["row_key"] = [
            uid if isinstance(uid, str) else f"refused-{seq}"
            for (uid, seq) in zip(produced[bitemporal.CONTEXT_UID], produced["source_seq"])
        ]
        outputs[NETWORK_CONTEXT_CLASS] = produced

    if (len(quarantined) > 0):
        quarantined["row_key"] = quarantined["event_uid"]
        outputs[universal_envelope.QUARANTINE_CLASS] = quarantined

    # Layer 1: counters, the rates they become and the port's history of them, optics, flaps, identifier changes.
    outputs["tc1"] = _run_class(config,
                                batches["tc1"],
                                layer_1_stages(config, network),
                                impose_order,
                                seal=False,
                                chain=CHAIN_ROOTS["tc1"],
                                envelope=CLASS_ENVELOPE["tc1"],
                                telemetry_class="tc1")

    # The same layer 1 snapshots, closed into port bindings. This is the ladder's last rung: the table that takes
    # a switch port to the site, optic and neighbour it held at a given moment. The stage emits its own
    # `binding_uid`, built exactly as `binding_table` builds one, so it serves as the row key directly rather than
    # having a second identifier composed beside it the way the layer 2 bindings do.
    port_bindings = _run_class(config,
                               batches["tc1"], [TC1BindingStage(config, port_anchor_column=PORT_ANCHOR_COLUMN)],
                               impose_order,
                               seal=False,
                               envelope=CLASS_ENVELOPE["tc1_binding"],
                               telemetry_class="tc1_binding")
    # A class every record of which was quarantined leaves nothing to key; the class is then simply absent.
    if (len(port_bindings) > 0):
        port_bindings["row_key"] = port_bindings["binding_uid"]
        outputs["tc1_binding"] = port_bindings

    # The devices themselves: restarts, configuration changes and routing churn, keyed on the switch or router.
    outputs[DEVICE_CLASS] = _run_class(config,
                                       batches[DEVICE_CLASS],
                                       device_stages(config, network),
                                       impose_order,
                                       seal=False,
                                       chain=CHAIN_ROOTS[DEVICE_CLASS],
                                       envelope=CLASS_ENVELOPE[DEVICE_CLASS],
                                       telemetry_class=DEVICE_CLASS)

    # Layer 2, from the same snapshots: the cardinality features, each port's count and each VLAN's vendor count
    # against the peaks of their own earlier periods, and the closed bindings.
    outputs["tc2_mac"] = _run_class(config,
                                    batches["tc2_mac"],
                                    mac_table_stages(config, network),
                                    impose_order,
                                    seal=False,
                                    chain=CHAIN_ROOTS["tc2_mac"],
                                    envelope=CLASS_ENVELOPE["tc2_mac"],
                                    telemetry_class="tc2_mac")
    # The table walk is a snapshot of each switch, and the switch's own removal notice is a stop: both ends a
    # source reports, beside the ones the closer infers.
    bindings = _run_class(
        config,
        batches["tc2_mac"],
        [TC2BindingStage(config, action_column=MAC_ACTION_COLUMN, snapshot_scope_columns=["site_id", "switch_id"])],
        impose_order,
        seal=False,
        envelope=CLASS_ENVELOPE["tc2_binding"],
        telemetry_class="tc2_binding")

    if (len(bindings) > 0):
        bindings["row_key"] = [
            event_uid("binding", *values)
            for values in zip(bindings["mac_address"], bindings["port_key"], bindings["bind_start"],
                              bindings["bind_end"], bindings["bind_end_reason"])
        ]
        outputs["tc2_binding"] = bindings

    # Layer 2 ARP, resolved through the bindings the previous pipeline just closed. This is the composition the
    # ladder depends on: a stage's output becomes a table, and another stage resolves through it.
    outputs["tc2_arp"] = _run_class(
        config,
        batches["tc2_arp"],
        [
            TC2ArpStage(config, excluded_sender_ips=[VRRP_IP]),
            BindingResolverStage(config,
                                 binding_table=build_binding_table(bindings),
                                 key_column="arp_sender_mac",
                                 output_columns={
                                     "port_key": "resolved_port_key", "vlan_id": "resolved_vlan_id"
                                 },
                                 uid_column="binding_uid"),
        ],
        impose_order,
        seal=False,
        chain=CHAIN_ROOTS["tc2_arp"],
        envelope=CLASS_ENVELOPE["tc2_arp"],
        telemetry_class="tc2_arp")

    outputs["tc2_auth"] = _run_class(config,
                                     batches["tc2_auth"],
                                     [TC2AuthStage(config, baseline_min_samples=AUTH_BASELINE_MIN_SAMPLES)],
                                     impose_order,
                                     seal=False,
                                     chain=CHAIN_ROOTS["tc2_auth"],
                                     envelope=CLASS_ENVELOPE["tc2_auth"],
                                     telemetry_class="tc2_auth")

    return outputs


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True) -> pd.DataFrame:
    """
    Run every telemetry class, seal the chained ones together, and return one canonicalized frame.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    corpus : dict
        The frames from `build_corpus`, possibly permuted.
    batches : dict, optional
        Per class, how the corpus is split across source frames. Defaults to one frame per class. The batch-split
        sweep is the caller's to vary.
    impose_order : bool, default = True
        Place `TotalOrderStage` ahead of the stateful stages. The permutation check's negative control turns it off
        to reproduce the removed-sort defect and prove the harness catches it.

    Returns
    -------
    `pandas.DataFrame`
        Every class's output, tagged with `telemetry_class`, keyed by `row_key`, canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    outputs = run_classes(config, corpus, batches, impose_order)

    # The chained classes are sealed together, in as many contiguous pieces as the widest per-class split, so
    # control 5 sweeps this pass too.
    parts = max(len(batches[name]) for name in CHAIN_ROOTS)
    outputs.update(_seal_chains(config, {name: outputs[name] for name in CHAIN_ROOTS}, parts=parts))

    frames = []

    for (name, frame) in outputs.items():
        frame = frame.copy()
        frame["telemetry_class"] = name

        if ("row_key" not in frame.columns):
            frame["row_key"] = frame["event_uid"]

        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    return canonicalize(combined, key_columns=KEY_COLUMNS, ignore_columns=IGNORE_COLUMNS)


def render(result: pd.DataFrame) -> str:
    """The byte-exact rendering compared across restarts and against the golden file."""
    return result.to_csv(index=False, lineterminator="\n")
