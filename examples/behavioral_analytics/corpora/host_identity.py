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
The synthetic corpus `morpheus.compositions.host_identity` is tested over, and the facts about it the harnesses assert.

Moved out of `tests/morpheus/determinism/host_identity.py` under plan section 4.5, which put the composition in the
package and left the data here, where the tests, the sample and fixture generators and the experiments read it. The
composition module's docstring describes what this corpus plants and why.
"""

import typing

import pandas as pd

from morpheus.compositions.host_identity import INVENTORY_CLASS
from morpheus.compositions.host_identity import LEASE_CLASS
from morpheus.compositions.host_identity import hostname_for

NS_PER_SECOND = 10**9
DAY_NS = 86400 * NS_PER_SECOND
CLIENT_POOL = "10.0.0."
"""The addresses a DHCP server hands out in the single-layer corpora. Every other subnet there is a server's static
address."""
HOST_CLASSES = (LEASE_CLASS, INVENTORY_CLASS)
"""The two classes this module adds to a corpus. They are bindings, not scored events, and no pipeline scores them."""
DHCP_COLLECTOR = "dhcp-01"
INVENTORY_COLLECTOR = "edr-inventory"
GRANT_SECONDS = 86400 + 3600
"""A day and an hour: what each daily renewal grants, and how long each daily inventory report stands."""


def is_client(address: typing.Any, pools: tuple = (CLIENT_POOL, )) -> bool:
    """Whether an address is in a pool a DHCP server hands out."""
    return isinstance(address, str) and address.startswith(pools)


def mac_for(address: str) -> str:
    """A client's network interface, numbered from its address so two clients never share one."""
    return "aa:10:" + ":".join(f"{int(part):02x}" for part in address.split("."))


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
          pools: tuple = (CLIENT_POOL, ),
          unmanaged: typing.Iterable[str] = ()) -> dict[str, pd.DataFrame]:
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
    unmanaged : iterable of str, optional
        Clients that are leased an address and run no EDR agent -- a phone on the corporate wireless. Their address
        resolves to an interface and to no machine.

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
    unmanaged = set(unmanaged)

    for day in range(first, end_ns + 1, DAY_NS):
        for address in clients:
            # Every client renews at the day boundary, so a lease covers the corpus from its first second. The
            # acknowledgements share that instant and the server's own sequence orders them.
            when = day
            lease = _envelope(leases, DHCP_COLLECTOR, "tc3_dhcp.v1", f"{address}>{mac_for(address)}", when)
            lease.update({
                "ip": address, "mac": mac_for(address), "dhcp_message": "ack", "lease_seconds": GRANT_SECONDS
            })
            leases.append(lease)

            if (address in unmanaged):
                continue

            report = _envelope(reports, INVENTORY_COLLECTOR, "tc7_host_inventory.v1", f"{mac_for(address)}", when)
            report.update({
                "mac": mac_for(address), "hostname": hostname_for(address), "report_valid_seconds": GRANT_SECONDS
            })
            reports.append(report)

    return {LEASE_CLASS: pd.DataFrame(leases), INVENTORY_CLASS: pd.DataFrame(reports)}
