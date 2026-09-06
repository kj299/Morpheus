# Collector contract

What a collector has to send, per column, and where the value comes from. Written so a collector author can diff
their output against something other than prose.

One class is specified here: the MAC address table, which is what R-D-L2-004 needs and all that
[`run_mac_spoof.py`](./run_mac_spoof.py) consumes. The other telemetry classes in the guide are design; naming
their object identifiers here would suggest a collector exists for them.

## Envelope

Every record carries these, whatever the class. They are what content-addressed identifiers are computed over, so
a collector that omits one produces records that cannot be traced back to it.

| Column | Type | Source | Notes |
|---|---|---|---|
| `event_time` | integer | When the *device* observed it | Nanoseconds since the Unix epoch. Not when the collector ran, and not when the record was indexed. If the device cannot report an observation time, send the poll time and say so in `schema_version`; do not leave it null and do not default it to now. |
| `collector_id` | string | The collector's own stable identity | Must survive a restart. A collector that renames itself on restart renames every record it has ever produced. |
| `collector_seq` | integer | Strictly increasing per `collector_id` | Ties are an envelope violation: `TotalOrderStage` raises rather than falling back on arrival order. |
| `schema_version` | string | The collector's own contract version | For example `TC-2/1.0.0`. |

## MAC address table (TC-2)

One record per `(address, port)` entry in one snapshot. **Send the whole table each poll**, not the delta: a
binding closes because an address was seen elsewhere, and an address that silently drops out of a delta feed
looks the same as one that never moved.

| Column | Type | SNMP | CLI | Notes |
|---|---|---|---|---|
| `mac_address` | string | `dot1dTpFdbAddress` (`1.3.6.1.2.1.17.4.3.1.1`) | `show mac address-table` | Lower case, colon separated. The rendering is normalized downstream, but sending one form consistently lets the raw feed be compared line by line. |
| `port_id` | string | `dot1dBasePortIfIndex` → `ifName` (`1.3.6.1.2.1.31.1.1.1.1`) | the `Ports` column | The interface **name**, not the bridge port number and not the `ifIndex`. `dot1dTpFdbPort` gives a bridge port; it must be mapped through `dot1dBasePortIfIndex` to an `ifIndex` and then to `ifName`, or the identifier will not join to anything layer 1 emits. |
| `switch_id` | string | `sysName` (`1.3.6.1.2.1.1.5`) | `show running-config \| include hostname` | The same value layer 1 sends as `device_id`. One identifier under two names; nothing renames anything. |
| `site_id` | string | inventory, not the device | -- | The device does not know what site it is in. This comes from the collector's own configuration, and it must be stable: it is the first component of every entity key. |
| `vlan_id` | integer | `dot1qVlanFdbId` / the VLAN-indexed community | the `Vlan` column | **Send a number.** Sent as a string it still works, but a real feed sends a number and the rendering has to survive one row with no VLAN widening the column. |

### What each field costs if it is wrong

- A `port_id` that is a bridge port number rather than an interface name produces bindings that join to nothing,
  and every layer 1 to layer 2 correlation silently returns empty.
- A `site_id` that varies between polls forks one entity into several, each with its own baseline.
- An `event_time` that is the poll time rather than the observation time makes every gap measurement report the
  polling cadence instead of the network's behaviour -- which is exactly the measurement R-D-L2-004 separates a
  spoof from a move with.
- A `collector_seq` that restarts at zero makes two different records collide on one identifier.

## Sending it

The pipeline reads JSON lines: one object per line, no wrapping array. `mac_table_sample.jsonlines` is a valid
seventeen-record example; a collector's first test is that its own output can be swapped in for that file and the
pipeline still runs.
