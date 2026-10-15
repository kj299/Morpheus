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
Which machine an address was, for the corpora that see a host only by its address.

A host is `src_ip` at layers 3 and 6 and in DNS and HTTP, part of a flow identifier at layer 4, and a host name at the
endpoint, and nothing in those corpora said that the address at one layer and the name at another were the same
machine. This module gives each of them the two records an estate actually has for that, and the two hops that
read them:

- the **DHCP server's acknowledgements**, which say which network interface held an address and for how long. Each
  client renews once a day and every renewal grants a day and an hour, so the lease lasts the whole corpus however
  long the client is silent, and it ends where the last grant does;
- the **EDR agent's inventory reports**, which say which machine a network interface belongs to. An agent reports
  once a day and each report stands until the next is due, the same grant arithmetic.

Both are closed into bindings by `TC2BindingStage`, the stage that closes every binding in this fork, and resolved
with `BindingResolverStage` -- the address to the interface through the leases, the interface to the machine through
the inventory -- so a record carries `mac_address` and `host_key`, the machine's normalized name as the endpoint
stages and the asset inventory spell it, with each hop's binding identifier beside it. `ChainAnchorStage` then roots
the record's chain on the machine where it was resolved and on the corpus's own key where it was not.

Only the client pool is leased. Servers are on static addresses and no DHCP server has a record of them, so their
records resolve to no machine, which is what an estate without a static-address inventory gets.
"""

import typing

import pandas as pd

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.binding_resolver_stage import BindingResolverStage
from morpheus.stages.lineage.chain_anchor_stage import ChainAnchorStage
from morpheus.stages.lineage.collector_sequence_stage import CollectorSequenceStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc2_binding_stage import TC2BindingStage
from morpheus.utils.binding_table import BindingTable
from morpheus.utils.entity_key import normalize_hostname

NS_PER_SECOND = 10**9
DAY_NS = 86400 * NS_PER_SECOND

CLIENT_POOL = "10.0.0."
"""The addresses a DHCP server hands out in the single-layer corpora. Every other subnet there is a server's static
address."""

LEASE_CLASS = "tc3_dhcp"
INVENTORY_CLASS = "tc7_host_inventory"
HOST_CLASSES = (LEASE_CLASS, INVENTORY_CLASS)
"""The two classes this module adds to a corpus. They are bindings, not scored events, and no pipeline scores them."""

DHCP_COLLECTOR = "dhcp-01"
INVENTORY_COLLECTOR = "edr-inventory"

GRANT_SECONDS = 86400 + 3600
"""A day and an hour: what each daily renewal grants, and how long each daily inventory report stands."""

HOST_KEY_COLUMN = "host_key"
MAC_COLUMN = "mac_address"
LEASE_UID_COLUMN = "host_lease_uid"
INVENTORY_UID_COLUMN = "host_inventory_uid"
LEASE_METHOD_COLUMN = "host_lease_resolution"
INVENTORY_METHOD_COLUMN = "host_inventory_resolution"

SESSION_KEY_COLUMN = "session_key"
SESSION_METHOD_COLUMN = "session_resolution"
SESSION_UID_COLUMN = "session_binding_uid"

ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]


def is_client(address: typing.Any, pools: tuple = (CLIENT_POOL, )) -> bool:
    """Whether an address is in a pool a DHCP server hands out."""
    return isinstance(address, str) and address.startswith(pools)


def mac_for(address: str) -> str:
    """A client's network interface, numbered from its address so two clients never share one."""
    return "aa:10:" + ":".join(f"{int(part):02x}" for part in address.split("."))


def hostname_for(address: str) -> str:
    """The machine a client's interface belongs to, as its EDR agent names it."""
    return "WS-" + "-".join(address.split(".")[2:]).upper()


def _envelope(rows: list, collector: str, schema: str, origin: str, when: int) -> dict:
    return {
        "collector_id": collector,
        "schema_version": schema,
        "origin_hash": f"{origin}@{when}",
        "collector_seq": len(rows),
        "event_time": when,
    }


def build(addresses: typing.Iterable[str],
          start_ns: int,
          end_ns: int,
          pools: tuple = (CLIENT_POOL, )) -> dict[str, pd.DataFrame]:
    """
    A DHCP server's acknowledgements and an EDR agent's inventory reports for every client address given.

    Parameters
    ----------
    addresses : iterable of str
        Addresses the corpus's records come from. Those outside the client pool are left out: no DHCP server leased
        them.
    start_ns, end_ns : int
        The span the corpus covers. Reports start at the day boundary on or before `start_ns` and run to `end_ns`.
    pools : tuple of str, default = (CLIENT_POOL, )
        Address prefixes the DHCP server hands out. Addresses outside them are servers' static addresses.

    Returns
    -------
    dict
        `LEASE_CLASS` and `INVENTORY_CLASS`, one frame each, in event-time order.
    """
    clients = sorted({address
                      for address in addresses if is_client(address, pools)},
                     key=lambda a: tuple(map(int, a.split("."))))
    leases: list = []
    reports: list = []
    first = start_ns - start_ns % DAY_NS

    for day in range(first, end_ns + 1, DAY_NS):
        for (index, address) in enumerate(clients):
            # Each client renews a little after midnight, one minute apart, so no two acknowledgements share a time.
            when = day + (index + 1) * 60 * NS_PER_SECOND
            lease = _envelope(leases, DHCP_COLLECTOR, "tc3_dhcp.v1", f"{address}>{mac_for(address)}", when)
            lease.update({
                "ip": address, "mac": mac_for(address), "dhcp_message": "ack", "lease_seconds": GRANT_SECONDS
            })
            leases.append(lease)

            report = _envelope(reports, INVENTORY_COLLECTOR, "tc7_host_inventory.v1", f"{mac_for(address)}", when)
            report.update({
                "mac": mac_for(address), "hostname": hostname_for(address), "report_valid_seconds": GRANT_SECONDS
            })
            reports.append(report)

    return {LEASE_CLASS: pd.DataFrame(leases), INVENTORY_CLASS: pd.DataFrame(reports)}


def _close(config: Config, dataframes: list, stage: TC2BindingStage, impose_order: bool) -> pd.DataFrame:
    from host_frame import to_host_frame  # pylint: disable=import-outside-toplevel

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    pipe.add_stage(CollectorSequenceStage(config))
    pipe.add_stage(stage)
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    frames = [to_host_frame(message.copy_dataframe()) for message in sink.get_messages()]

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def close_leases(config: Config, dataframes: list, impose_order: bool = True) -> pd.DataFrame:
    """The DHCP server's acknowledgements, closed into leases that last their whole term."""
    if (len(dataframes) == 0):
        return pd.DataFrame()

    return _close(config,
                  dataframes,
                  TC2BindingStage(config,
                                  key_column="ip",
                                  attribute_columns=["mac"],
                                  action_column="dhcp_message",
                                  grant_seconds_column="lease_seconds"),
                  impose_order)


def close_inventory(config: Config, dataframes: list, impose_order: bool = True) -> pd.DataFrame:
    """The EDR agent's reports, closed into which machine each interface belonged to and when."""
    if (len(dataframes) == 0):
        return pd.DataFrame()

    return _close(config,
                  dataframes,
                  TC2BindingStage(config,
                                  key_column="mac",
                                  attribute_columns=["hostname"],
                                  grant_seconds_column="report_valid_seconds"),
                  impose_order)


def lease_table(bindings: pd.DataFrame) -> BindingTable:
    """The leases, as the address-to-interface hop resolves through them."""
    if (len(bindings) == 0):
        return BindingTable(name="dhcp_lease", value_columns=["mac"], bindings=[])

    return BindingTable.from_dataframe(bindings,
                                       name="dhcp_lease",
                                       key_column="ip",
                                       value_columns=["mac"],
                                       start_column="bind_start",
                                       end_column="bind_end")


def inventory_table(bindings: pd.DataFrame) -> BindingTable:
    """The inventory, as the interface-to-machine hop resolves through it, the name normalized as assets key it."""
    if (len(bindings) == 0):
        return BindingTable(name="host_inventory", value_columns=["hostname"], bindings=[])

    bindings = bindings.copy()
    bindings["hostname"] = [normalize_hostname(name) for name in bindings["hostname"]]

    return BindingTable.from_dataframe(bindings,
                                       name="host_inventory",
                                       key_column="mac",
                                       value_columns=["hostname"],
                                       start_column="bind_start",
                                       end_column="bind_end")


def tables(config: Config, batches: dict, impose_order: bool = True) -> tuple[BindingTable, BindingTable]:
    """The lease and inventory tables a corpus's batches carry, empty where it carries none."""
    leases = close_leases(config, batches.get(LEASE_CLASS, []), impose_order)
    inventory = close_inventory(config, batches.get(INVENTORY_CLASS, []), impose_order)

    return (lease_table(leases), inventory_table(inventory))


def session_table(intervals: pd.DataFrame, end_ns: int) -> BindingTable:
    """
    Which session held each address, and when: a principal's sessions as `TC5SessionStage` keyed them.

    Parameters
    ----------
    intervals : `pandas.DataFrame`
        One row per session: `session_key`, `source_ip`, `opened` and `closed`, the last null for a session that
        never ended.
    end_ns : int
        Where a session that never ended is taken to end: the end of what the corpus covers, not forever.
    """
    if (len(intervals) == 0):
        return BindingTable(name="session", value_columns=[SESSION_KEY_COLUMN], bindings=[])

    rows = intervals.assign(bind_start=intervals["opened"].astype("int64"),
                            bind_end=intervals["closed"].fillna(end_ns).astype("int64"))

    return BindingTable.from_dataframe(rows,
                                       name="session",
                                       key_column="source_ip",
                                       value_columns=[SESSION_KEY_COLUMN],
                                       start_column="bind_start",
                                       end_column="bind_end")


def session_resolver(config: Config, sessions: BindingTable, address_column: str = "src_ip") -> BindingResolverStage:
    """The session that held a record's source address when the record happened, as `session_key`."""
    return BindingResolverStage(config,
                                binding_table=sessions,
                                key_column=address_column,
                                output_columns={SESSION_KEY_COLUMN: SESSION_KEY_COLUMN},
                                method_column=SESSION_METHOD_COLUMN,
                                uid_column=SESSION_UID_COLUMN)


def resolver_stages(config: Config,
                    leases: BindingTable,
                    inventory: BindingTable,
                    address_column: str = "src_ip",
                    fallback_anchor: str = "src_ip") -> list:
    """
    The address to its interface, the interface to its machine, and the chain rooted on the machine.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    leases, inventory : `morpheus.utils.binding_table.BindingTable`
        From `tables`.
    address_column : str, default = "src_ip"
        The column holding the host's address.
    fallback_anchor : str, default = "src_ip"
        What the chain is rooted on where the address resolved to no machine: the corpus's own key.
    """
    return [
        BindingResolverStage(config,
                             binding_table=leases,
                             key_column=address_column,
                             output_columns={"mac": MAC_COLUMN},
                             method_column=LEASE_METHOD_COLUMN,
                             uid_column=LEASE_UID_COLUMN),
        BindingResolverStage(config,
                             binding_table=inventory,
                             key_column=MAC_COLUMN,
                             output_columns={"hostname": HOST_KEY_COLUMN},
                             method_column=INVENTORY_METHOD_COLUMN,
                             uid_column=INVENTORY_UID_COLUMN),
        ChainAnchorStage(config, candidates=[HOST_KEY_COLUMN, fallback_anchor]),
    ]
