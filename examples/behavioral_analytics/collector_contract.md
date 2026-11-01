# Collector contract

What a collector has to send, per class and per column, and where each value comes from. Written so a collector
author can diff their output against something other than prose.

<!-- Generated from input_contracts.json by make_input_samples.py; edit the JSON and regenerate. -->

Every class has a sample under [`samples/`](./samples), one directory per segment and one JSON lines file per class,
exported from the corpus that segment's golden is produced from; a collector's first test is that its own output can
be swapped in for the sample and the segment still runs. `tests/morpheus/determinism/test_input_contracts.py` holds
every corpus builder to these tables: a required column every builder of the class sends, an optional one only some
do, and nothing a builder sends that the table does not name.

Columns are JSON scalars: an integer is a JSON number with no fraction, a string a JSON string, and a gap is `null`,
never an empty string or a sentinel. Times are integers in nanoseconds since the Unix epoch.

## Envelope

Every record carries these, whatever the class, except the TC-0 inventories, whose clock is the valid and recorded
times each record carries. They are what content-addressed identifiers are computed over and what clock quarantine
is decided on, so a collector that omits one produces records that cannot be traced back to it.

| Column | Type | Unit | Source | Notes |
|---|---|---|---|---|
| `event_time` | integer | `ns since the Unix epoch` | When the device or `service` observed it | Not when the collector ran and not when the record was indexed. `A` device that cannot report its own observation time sends the poll time and says so in `schema_version`. |
| `observed_time` | integer | `ns since the Unix epoch` | When the collector received or read the record | The gap from `event_time` is the collection lag the health searches measure. |
| `collector_id` | string | -- | The collector's own stable identity | Must survive a restart; a collector that renames itself renames every record it has produced. |
| `collector_seq` | integer | `count` | Strictly increasing per `collector_id` | Ties are an envelope violation and `TotalOrderStage` refuses them; a gap is a record that was never delivered, which `CollectorSequenceStage` counts. |
| `schema_version` | string | -- | The collector's own contract version | For example tc3.v1. |
| `origin_hash` | string | -- | What identifies the record at its source | Anything stable the source gives: a flow tuple and its start, a log record's own identifier. Hashed with the three fields above into `lineage_id`. |
| `clock_source` | string | -- | How the device's clock is disciplined | One of `ntp_disciplined`, `ptp`, `gps`, `unsynchronized`. An unsynchronized clock is quarantined. |
| `clock_offset_ms` | float | `ms` | The device's last measured offset from its reference | Integers are accepted. Beyond a second either way the record is quarantined. |
| `sampling_policy` | string | -- | Whether every record is sent | `full`, or the sampling the exporter applies, for example 1:100. `A` count from a sampled feed is not comparable with one from a `full` feed. |
| `tenant_id` | string | -- | The collector's own configuration | The logical scope every health and platform search groups by. |

## Application inventory (TC-0)

`tc0_application`, layer 0; sample in [`samples/context/tc0_application.jsonlines`](./samples/context/tc0_application.jsonlines). Carries no collector envelope.

One record per change to what an application is: its address and port, who owns it and how much it matters. Read by `TC0ApplicationStage` into the bitemporal context store.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `source_seq` | integer | `count` | yes | The inventory's own change sequence | Orders two changes recorded at one instant. |
| `server_address` | string | -- | yes | CMDB or `service` catalogue | The address the application listens on. |
| `server_port` | integer | -- | yes | CMDB or `service` catalogue | -- |
| `name` | string | -- | yes | CMDB or `service` catalogue | The application's name, which `ctx_application_name` carries. |
| `owning_team` | string | -- | yes | CMDB | -- |
| `service_account` | string | -- | yes | CMDB or directory | The account the application runs as. |
| `criticality` | string | -- | yes | CMDB | low, medium, high or critical. |
| `data_classification` | string | -- | yes | CMDB or data catalogue | -- |
| `valid_from` | integer | `ns since the Unix epoch` | yes | When the fact became true | Valid time, which may be before it was recorded. |
| `valid_to` | integer | `ns since the Unix epoch` | yes | When the fact stopped being true | Null while it still holds. |
| `recorded_at` | integer | `ns since the Unix epoch` | yes | When the inventory recorded the change | Known time. `A` correction recorded later does not rewrite what was known before it. |
| `change` | string | -- | yes | The inventory's change type | `assert` or `retract`. |
| `tenant_id` | string | -- | yes | The inventory's own configuration | -- |

## Asset inventory (TC-0)

`tc0_asset`, layer 0; sample in [`samples/context/tc0_asset.jsonlines`](./samples/context/tc0_asset.jsonlines). Carries no collector envelope.

One record per change to what a machine is. Read by `TC0AssetStage` into the bitemporal context store, keyed on the normalized `hostname`.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `source_seq` | integer | `count` | yes | The inventory's own change sequence | -- |
| `hostname` | string | -- | yes | CMDB or EDR inventory | Case-folded and stripped of the domain downstream, as `normalize_hostname` does. |
| `owner` | string | -- | yes | CMDB | `A` principal, in the form identity records use. |
| `owning_team` | string | -- | yes | CMDB | -- |
| `criticality` | string | -- | yes | CMDB | -- |
| `data_classification` | string | -- | yes | CMDB or data catalogue | -- |
| `peer_group` | string | -- | yes | CMDB | The group a host is compared with. |
| `device_role` | string | -- | yes | CMDB or EDR | `workstation`, `server`, `dns_server`, `dhcp_server` among others; R-B-L3-006 reads `workstation`. |
| `os_family` | string | -- | yes | EDR inventory | R-B-L4-004 compares a host's TCP stack with it. |
| `os_version` | string | -- | yes | EDR inventory | -- |
| `valid_from` | integer | `ns since the Unix epoch` | yes | When the fact became true | -- |
| `valid_to` | integer | `ns since the Unix epoch` | yes | When the fact stopped being true | Null while it still holds. |
| `recorded_at` | integer | `ns since the Unix epoch` | yes | When the inventory recorded the change | -- |
| `change` | string | -- | yes | The inventory's change type | `assert` or `retract`. |
| `tenant_id` | string | -- | yes | The inventory's own configuration | -- |

## Identity directory (TC-0)

`tc0_identity`, layer 0; sample in [`samples/context/tc0_identity.jsonlines`](./samples/context/tc0_identity.jsonlines). Carries no collector envelope.

One record per change to a principal: department, manager, employment status, account type, privilege and group membership. Read by `TC0IdentityStage`; the layer 5 searches read what it held at the time of each sign-in.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `source_seq` | integer | `count` | yes | The directory's change sequence (`usnChanged`, or the HR system's own) | -- |
| `user_principal` | string | -- | yes | `userPrincipalName` | Lower case; the same form sign-in records carry. |
| `department` | string | -- | yes | department | Null on a record that changes only another attribute. |
| `manager` | string | -- | yes | manager, resolved to a principal | Null on a record that changes only another attribute. |
| `employment_status` | string | -- | yes | HR system | active, leave or terminated. Null on a record that changes only another attribute. |
| `account_type` | string | -- | yes | Directory or HR system | `human` or `service`; R-B-L5-005 excludes `service` accounts. |
| `privilege_level` | string | -- | yes | PAM or directory group | -- |
| `group_name` | string | -- | yes | `memberOf`, one group per record | `A` membership is a fact of its own with its own valid interval. |
| `valid_from` | integer | `ns since the Unix epoch` | yes | When the fact became true | -- |
| `valid_to` | integer | `ns since the Unix epoch` | yes | When the fact stopped being true | Null while it still holds. |
| `recorded_at` | integer | `ns since the Unix epoch` | yes | When the directory recorded the change | Null only on a record whose known time is its valid time. |
| `change` | string | -- | yes | The directory's change type | `assert` or `retract`. |
| `tenant_id` | string | -- | yes | The directory's own configuration | -- |

## Network inventory (TC-0)

`tc0_network`, layer 0; sample in [`samples/telemetry/tc0_network.jsonlines`](./samples/telemetry/tc0_network.jsonlines). Carries no collector envelope.

One record per change to what a port, a device or a VLAN is: a port's commissioning receive level, each object's designation, owner and criticality, a device's role. Read by `TC0NetworkStage` into the bitemporal context store, keyed as the telemetry keys each object; layer 1, the device class and layer 2 are joined against it at each row's event time, and R-D-L1-010 reads the commissioning level.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `source_seq` | integer | `count` | yes | The inventory's own change sequence | -- |
| `object_kind` | string | -- | yes | The network inventory | `port`, `device` or `vlan`. Any other kind is refused. |
| `site_id` | string | -- | yes | The network inventory | The first part of every key, as the polling collectors send it. |
| `device_id` | string | -- | yes | The network inventory | A port's or a device's switch, as `sysName` reports it. Null on a VLAN. |
| `port_id` | string | -- | yes | The network inventory | A port's name as it was commissioned; a renamed port is found by the name its `ifIndex` was first seen with. Null on a device or a VLAN. |
| `vlan_id` | integer | -- | yes | The network inventory | Null on a port or a device. |
| `commissioning_rx_dbm` | float | `dBm` | yes | The acceptance test the link passed when it was commissioned | A port's receive level then, which R-D-L1-010 compares the polled level with. A new optic is recommissioned, as a new version from the moment it went in. |
| `designation` | string | -- | yes | The network inventory | What a port or a VLAN is for: `access`, `uplink`, `desks`, `cameras`. |
| `device_role` | string | -- | yes | The network inventory | A device's role: `access-switch`, `router`. |
| `owner` | string | -- | yes | The network inventory | Who answers for the object. |
| `owning_team` | string | -- | yes | The network inventory | A device's owning team. |
| `criticality` | string | -- | yes | The network inventory | -- |
| `valid_from` | integer | `ns since the Unix epoch` | yes | When the fact became true | -- |
| `valid_to` | integer | `ns since the Unix epoch` | yes | When the fact stopped being true | Null while it still holds. |
| `recorded_at` | integer | `ns since the Unix epoch` | yes | When the inventory recorded the change | -- |
| `change` | string | -- | yes | The inventory's change type | `assert` or `retract`. |
| `tenant_id` | string | -- | yes | The inventory's own configuration | -- |

## Interface poll (TC-1)

`tc1`, layer 1; sample in [`samples/telemetry/tc1.jsonlines`](./samples/telemetry/tc1.jsonlines). Carries the envelope.

One record per port per poll, with the raw counters as the device reports them. `TC1NormalizeStage` differences the counters itself, so a collector sends totals and `sysUpTime`, never deltas.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `site_id` | string | -- | yes | The collector's inventory | The first component of every entity key; it must not vary between polls. |
| `device_id` | string | -- | yes | `sysName` (`1.3.6.1.2.1.1.5`) | The same value layer 2 sends as `switch_id`. |
| `port_id` | string | -- | yes | `ifName` (`1.3.6.1.2.1.31.1.1.1.1`) | The interface name, not the `ifIndex`. |
| `if_index` | integer | -- | no | `ifIndex` (`1.3.6.1.2.1.2.2.1.1`) | What a renamed port is recognised by. Send it only from a device that persists it across reboots; a reassigned index merges two ports' histories. |
| `uptime` | integer | `centiseconds` | yes | `sysUpTime` (`1.3.6.1.2.1.1.3`) | `TimeTicks`; a fall is a restart, which tells a counter wrap from a reboot. |
| `if_last_change` | integer | `centiseconds` | yes | `ifLastChange` (`1.3.6.1.2.1.2.2.1.9`) | Measured on the same clock as uptime; a flap between polls moves it. |
| `oper_status` | string | -- | yes | `ifOperStatus` (`1.3.6.1.2.1.2.2.1.8`) | `up` or `down`. |
| `link_speed_bps` | integer | `bits/s` | yes | `ifHighSpeed` (`1.3.6.1.2.1.31.1.1.1.15`) times 1,000,000 | Utilization is measured against it. |
| `optical_tx_dbm` | float | `dBm` | yes | ENTITY-SENSOR-MIB or the vendor DOM table | Null on a copper port. |
| `optical_rx_dbm` | float | `dBm` | yes | ENTITY-SENSOR-MIB or the vendor DOM table | `A` one to three `dB` step with transmit steady is the tap R-D-L1-002 reads. |
| `transceiver_serial` | string | -- | yes | `entPhysicalSerialNum` (`1.3.6.1.2.1.47.1.1.1.1.11`) | R-D-L1-001 reads a change. |
| `transceiver_type` | string | -- | yes | `entPhysicalModelName` or the vendor DOM table | Selects the receive floor R-P-L1-004 forecasts against. |
| `lldp_neighbor_chassis_id` | string | -- | yes | `lldpRemChassisId` (`1.0.8802.1.1.2.1.4.1.1.5`) | Null where no neighbour speaks LLDP. |
| `lldp_neighbor_port_id` | string | -- | yes | `lldpRemPortId` (`1.0.8802.1.1.2.1.4.1.1.7`) | Null where no neighbour speaks LLDP. |
| `crc_errors` | integer | `count` | yes | `dot3StatsFCSErrors` (`1.3.6.1.2.1.10.7.2.1.3`) | `A` 32-bit total; the stage handles the wrap. |
| `symbol_errors` | integer | `count` | yes | `dot3StatsSymbolErrors` (`1.3.6.1.2.1.10.7.2.1.18`) | `A` 32-bit total. |
| `input_discards` | integer | `count` | yes | `ifInDiscards` (`1.3.6.1.2.1.2.2.1.13`) | -- |
| `output_discards` | integer | `count` | yes | `ifOutDiscards` (`1.3.6.1.2.1.2.2.1.19`) | -- |
| `if_hc_in_octets` | integer | `octets` | yes | `ifHCInOctets` (`1.3.6.1.2.1.31.1.1.1.6`) | The 64-bit counter, not `ifInOctets`. |
| `if_hc_out_octets` | integer | `octets` | yes | `ifHCOutOctets` (`1.3.6.1.2.1.31.1.1.1.10`) | The 64-bit counter, not `ifOutOctets`. |

## Device poll (TC-1)

`tc1_device`, layer 1; sample in [`samples/telemetry/tc1_device.jsonlines`](./samples/telemetry/tc1_device.jsonlines). Carries the envelope.

One record per switch or router per poll: the device's own behaviour rather than any port's. Read by `TC1DeviceStage`.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `site_id` | string | -- | yes | The collector's inventory | -- |
| `device_id` | string | -- | yes | `sysName` (`1.3.6.1.2.1.1.5`) | -- |
| `sys_uptime` | integer | `centiseconds` | yes | `sysUpTime` (`1.3.6.1.2.1.1.3`) | `A` fall is a restart. |
| `running_config_hash` | string | -- | yes | `A` digest of the running configuration the collector fetches | Any stable digest; only a change matters. R-D-L1-008 reads a change outside the change window. |
| `cpu_util` | float | `fraction 0 to 1` | yes | HOST-RESOURCES-MIB `hrProcessorLoad` or the vendor CPU table, divided by 100 | -- |
| `mem_util` | float | `fraction 0 to 1` | yes | HOST-RESOURCES-MIB `hrStorage` or the vendor memory table | -- |
| `route_count` | integer | `count` | yes | `inetCidrRouteNumber` (`1.3.6.1.2.1.4.24.6`) | Its change between polls is `route_count_delta`. |
| `bgp_peer_count` | integer | `count` | yes | BGP4-MIB `bgpPeerState` counted where established | Zero on a device that runs no BGP. |
| `syslog_rate` | float | `messages per second` | yes | The collector's count of the device's `syslog` over the poll interval | -- |

## ARP observation (TC-2)

`tc2_arp`, layer 2; sample in [`samples/telemetry/tc2_arp.jsonlines`](./samples/telemetry/tc2_arp.jsonlines). Carries the envelope.

One record per ARP packet a sensor saw. Read by `TC2ArpStage`, which counts gratuitous replies and the MACs claiming one address.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `arp_sender_ip` | string | -- | yes | ARP sender protocol address (Zeek arp.log `orig_spa`) | -- |
| `arp_sender_mac` | string | -- | yes | ARP sender hardware address (`orig_sha`) | Lower case, colon separated. |
| `arp_target_ip` | string | -- | yes | ARP target protocol address (`resp_spa`) | Equal to the sender's on a gratuitous `reply`. |
| `arp_operation` | string | -- | yes | ARP opcode | `request` or `reply`. |

## 802.1X exchange (TC-2)

`tc2_auth`, layer 2; sample in [`samples/telemetry/tc2_auth.jsonlines`](./samples/telemetry/tc2_auth.jsonlines). Carries the envelope.

One record per step of an 802.1X exchange on a port. Read by `TC2AuthStage`, which pairs the start with the outcome and measures how long it took.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `site_id` | string | -- | yes | The collector's inventory | -- |
| `switch_id` | string | -- | yes | RADIUS NAS-Identifier, mapped to `sysName` | The same value layer 1 sends as `device_id`. |
| `port_id` | string | -- | yes | RADIUS NAS-Port-Id | The interface name. |
| `mac_address` | string | -- | yes | RADIUS Calling-Station-Id | Lower case, colon separated. |
| `dot1x_identity` | string | -- | yes | RADIUS User-Name | -- |
| `dot1x_result` | string | -- | yes | Access-Request, Access-Accept or Access-Reject | `started`, `success` or `failure`. |

## MAC address table (TC-2)

`tc2_mac`, layer 2; sample in [`samples/telemetry/tc2_mac.jsonlines`](./samples/telemetry/tc2_mac.jsonlines). Carries the envelope.

One record per (address, port) entry in one snapshot. Send the whole table each poll, not the delta: a binding closes because an address was seen elsewhere, and one that silently drops out of a delta feed looks the same as one that never moved.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `mac_address` | string | -- | yes | `dot1dTpFdbAddress` (`1.3.6.1.2.1.17.4.3.1.1`) | Lower case, colon separated. |
| `site_id` | string | -- | yes | The collector's inventory | The device does not know its site. |
| `switch_id` | string | -- | yes | `sysName` (`1.3.6.1.2.1.1.5`) | The same value layer 1 sends as `device_id`. |
| `port_id` | string | -- | yes | `dot1dBasePortIfIndex` mapped to `ifName` | The interface name, not the bridge port number; a bridge port number joins to nothing layer 1 emits. |
| `vlan_id` | integer | -- | yes | `dot1qVlanFdbId` | Send a number. |
| `mac_action` | string | -- | no | `A` MAC-move or MAC-notification trap, where the switch sends one | `removed` on an explicit removal, null in a snapshot. With it, a binding's end is observed rather than inferred. |

## Flow record (TC-3)

`tc3`, layer 3; sample in [`samples/network/tc3.jsonlines`](./samples/network/tc3.jsonlines). Carries the envelope.

One record per flow as an exporter reports it. Read by the TC-3 stages, which measure fan-out, fan-in, reach, beaconing and TTL per host.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `src_ip` | string | -- | yes | IPFIX `sourceIPv4Address` (8) or `sourceIPv6Address` (27) | -- |
| `src_port` | integer | -- | yes | IPFIX `sourceTransportPort` (7) | -- |
| `dst_ip` | string | -- | yes | IPFIX `destinationIPv4Address` (12) or `destinationIPv6Address` (28) | -- |
| `dst_port` | integer | -- | yes | IPFIX `destinationTransportPort` (11) | -- |
| `protocol` | string | -- | yes | IPFIX `protocolIdentifier` (4), named | `tcp`, `udp` or `icmp`. |
| `ip_ttl` | integer | `hops` | yes | IPFIX `minimumTTL` (52) | `TC3TtlStage` compares it with the source's established value. |
| `bytes_out` | integer | `bytes` | yes | IPFIX `octetDeltaCount` (1) in the initiator's direction | An exporter that reports both directions gives both; one that reports a direction per record needs the reverse record paired. |
| `bytes_in` | integer | `bytes` | yes | IPFIX reverse `octetDeltaCount` | -- |
| `bgp_as_dst` | string | -- | yes | IPFIX `bgpDestinationAsNumber` (17) | Null where the exporter has no BGP view. |
| `exporter_id` | string | -- | yes | The exporter's address and IPFIX `observationDomainId` | Two domains on one router are two exporters. |

## DHCP acknowledgement (TC-3)

`tc3_dhcp`, layer 3; sample in [`samples/network/tc3_dhcp.jsonlines`](./samples/network/tc3_dhcp.jsonlines). Carries the envelope.

One record per DHCP message a `server` logged. `TC2BindingStage` closes them into leases, which name the interface behind every address the flows, packets and handshakes come from.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `ip` | string | -- | yes | DHCPACK `yiaddr` | -- |
| `mac` | string | -- | yes | DHCPACK `chaddr` | Lower case, colon separated. |
| `dhcp_message` | string | -- | yes | The message type | `ack` or `release`. |
| `lease_seconds` | integer | `seconds` | yes | Option 51, the lease time | The term the `server` granted, which ends the lease if no renewal follows. |
| `switch_id` | string | -- | no | Option 82 remote-id, mapped to `sysName` | Where the relay inserts it; the layer 2 rung of R-C-005 resolves through it. |
| `port_id` | string | -- | no | Option 82 circuit-id, mapped to `ifName` | -- |

## Packet (TC-4)

`tc4`, layer 4; sample in [`samples/transport/tc4.jsonlines`](./samples/transport/tc4.jsonlines). Carries the envelope.

One record per packet, from a capture point. `TC4FlowStage` rolls them into the thirteen per-flow features the shipped abp-pcap model reads, and `TC4EnvelopeStage` measures each conversation's transfer against its own history.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `src_ip` | string | -- | yes | IP source address | -- |
| `src_port` | integer | -- | yes | TCP or UDP source port | -- |
| `dst_ip` | string | -- | yes | IP destination address | -- |
| `dst_port` | integer | -- | yes | TCP or UDP destination port | -- |
| `protocol` | string | -- | yes | IP protocol, named | -- |
| `tcp_flags` | integer | `bit field` | yes | The TCP flags byte | FIN 1, SYN 2, RST 4, PSH 8, ACK 16. |
| `data_len` | integer | `bytes` | yes | Payload length | -- |
| `tcp_options_order` | string | -- | no | The `SYN` packet's TCP options, in order | Comma separated, for example mss,nop,ws,nop,nop,sok. Present on `SYN` packets only; R-B-L4-004 reads the stack it names. |
| `tcp_mss` | integer | `bytes` | no | The `SYN` packet's MSS option | Null except on a `SYN`. |
| `tcp_window_initial` | integer | `bytes` | no | The `SYN` packet's window | Null except on a `SYN`. |

## Authentication (TC-5)

`tc5_auth`, layer 5; sample in [`samples/session/tc5_auth.jsonlines`](./samples/session/tc5_auth.jsonlines). Carries the envelope.

One record per sign-in attempt at an identity provider or a domain controller. Read by the TC-5 stages; the field names follow the Azure sign-in mapping Part 0's Pass 4 adopted.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `user_principal` | string | -- | yes | `userPrincipalName` | Lower case. |
| `source_ip` | string | -- | yes | `ipAddress` | -- |
| `app` | string | -- | yes | `appDisplayName` | -- |
| `auth_result` | string | -- | yes | `status.errorCode`, named | `success` or `failure`. |
| `target_host` | string | -- | no | `A` domain controller's target computer (4624 `WorkstationName` or the Kerberos `service` host) | Present on an on-premises sign-in; R-C-005 reads it. |
| `source_country` | string | -- | no | `location.countryOrRegion` | ISO 3166 alpha-2, lower case. |
| `source_region` | string | -- | no | location.state | -- |
| `source_city` | string | -- | no | location.city | -- |
| `source_latitude` | float | `degrees` | no | `location.geoCoordinates.latitude` | Implied travel speed is measured from it. |
| `source_longitude` | float | `degrees` | no | `location.geoCoordinates.longitude` | -- |
| `source_asn` | string | -- | no | The address's autonomous system, from an enrichment the collector applies | For example as5089. |
| `device_id` | string | -- | no | `deviceDetail.deviceId` | -- |
| `mfa_used` | boolean | -- | no | `authenticationRequirement` is `multiFactorAuthentication` | -- |
| `mfa_result` | string | -- | no | `mfaDetail` and status, named | `approved` or `denied`; null where no second factor was asked for. |
| `token_type` | string | -- | no | `tokenIssuerType` or the grant | `bearer`, `refresh` among others. |

## Session (TC-5)

`tc5_session`, layer 5; sample in [`samples/session/tc5_session.jsonlines`](./samples/session/tc5_session.jsonlines). Carries the envelope.

One record per session start or end, or one per session where the provider writes a single `summary` record. `TC5SessionStage` pairs them under any of six vocabularies.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `user_principal` | string | -- | yes | The account the session belongs to | -- |
| `session_id` | string | -- | yes | The provider's session identifier (`TargetLogonId` on Windows) | What pairs the start with the end. |
| `session_action` | string | -- | yes | The record's type | start/end, logon/logoff, login/logout, open/close, begin/finish, created/terminated, or `summary`. |
| `source_ip` | string | -- | no | The address the session was opened from | -- |
| `session_start` | integer | `ns since the Unix epoch` | no | `A` `summary` record's own start | Only on a single-record session; null otherwise. |
| `session_end` | integer | `ns since the Unix epoch` | no | `A` `summary` record's own end | Only on a single-record session; null otherwise. |

## TLS handshake (TC-6)

`tc6`, layer 6; sample in [`samples/presentation/tc6.jsonlines`](./samples/presentation/tc6.jsonlines). Carries the envelope.

One record per handshake, with the server's certificate. Read by the TC-6 stages: client fingerprint novelty, certificate issuer novelty, cipher downgrade and declared-against-detected content.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `src_ip` | string | -- | yes | Zeek ssl.log `id.orig_h` | -- |
| `src_port` | integer | -- | yes | `id.orig_p` | -- |
| `dst_ip` | string | -- | yes | `id.resp_h` | -- |
| `dst_port` | integer | -- | yes | `id.resp_p` | -- |
| `sni` | string | -- | yes | ssl.log `server_name` | Null where the client sent none. |
| `protocol` | string | -- | yes | The transport | `tcp`, or `udp` for QUIC. |
| `tls_version` | string | -- | yes | ssl.log version | For example TLSv1.3. |
| `ja4_client` | string | -- | yes | The JA4 fingerprint (the Zeek JA4 package's ja4) | -- |
| `certificate_issuer` | string | -- | yes | x509.log certificate.issuer of the leaf | -- |
| `certificate_fingerprint_sha256` | string | -- | yes | x509.log fingerprint of the leaf | -- |
| `cipher_suite` | string | -- | yes | ssl.log cipher | The IANA name. |
| `validation_result` | string | -- | yes | ssl.log `validation_status` | `ok`, or the `failure`; self signed certificate is what R-D-L6-003 reads. |
| `certificate_not_before` | integer | `ns since the Unix epoch` | yes | x509.log `certificate.not_valid_before` | -- |
| `certificate_not_after` | integer | `ns since the Unix epoch` | yes | x509.log `certificate.not_valid_after` | -- |
| `content_type_declared` | string | -- | yes | The Content-Type an inspecting proxy saw declared | Null without inspection. |
| `content_type_detected` | string | -- | yes | The type the proxy detected from the bytes (Zeek files.log `mime_type`) | Null without inspection; R-D-L6-005 reads a mismatch. |

## DNS query (TC-7)

`tc7_dns`, layer 7; sample in [`samples/application/tc7_dns.jsonlines`](./samples/application/tc7_dns.jsonlines). Carries the envelope.

One record per query a resolver answered. Read by `TC7DnsStage`, which measures subdomain entropy and breadth per registered domain and `NXDOMAIN` ratios per client.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `src_ip` | string | -- | yes | Zeek dns.log `id.orig_h`, or the BIND query log client | -- |
| `query_name` | string | -- | yes | dns.log query | As asked; the registered domain is found downstream from the Public Suffix List. |
| `query_type` | string | -- | yes | dns.log `qtype_name` | `A`, `AAAA`, `TXT` among others. |
| `response_code` | string | -- | yes | dns.log `rcode_name` | `NOERROR`, `NXDOMAIN` among others. |

## Process start (TC-7)

`tc7_endpoint`, layer 7; sample in [`samples/endpoint/tc7_endpoint.jsonlines`](./samples/endpoint/tc7_endpoint.jsonlines). Carries the envelope.

One record per process creation an EDR agent or the Windows System Monitor reported. Read by `TC7EndpointStage`, which keeps each host's parent-to-child pairs.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `hostname` | string | -- | yes | System Monitor event 1 Computer | Case-folded and stripped of the domain downstream. |
| `parent_image_path` | string | -- | yes | System Monitor event 1 `ParentImage` | -- |
| `image_path` | string | -- | yes | System Monitor event 1 Image | -- |
| `integrity_level` | string | -- | yes | System Monitor event 1 `IntegrityLevel` | Null where the agent does not report one. |
| `signature_status` | string | -- | yes | The image's signature check (System Monitor event 7 `SignatureStatus`, or the EDR agent's) | -- |
| `process_guid` | string | -- | no | System Monitor event 1 `ProcessGuid` | -- |
| `parent_process_guid` | string | -- | no | System Monitor event 1 `ParentProcessGuid` | -- |
| `command_line_hash` | string | -- | no | `A` digest of System Monitor event 1 `CommandLine` | The digest, not the line: a command line carries credentials often enough that it should not travel. |

## Host inventory report (TC-7)

`tc7_host_inventory`, layer 7; sample in [`samples/network/tc7_host_inventory.jsonlines`](./samples/network/tc7_host_inventory.jsonlines). Carries the envelope.

One record per interface an EDR agent reports for its machine, renewed on a schedule. `TC2BindingStage` closes them into which machine each interface belonged to and when.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `mac` | string | -- | yes | The agent's interface inventory | Lower case, colon separated. |
| `hostname` | string | -- | yes | The agent's machine name | -- |
| `report_valid_seconds` | integer | `seconds` | yes | How long the agent's report stands | The term a report grants, which ends the binding if no renewal follows. |

## HTTP request (TC-7)

`tc7_http`, layer 7; sample in [`samples/application/tc7_http.jsonlines`](./samples/application/tc7_http.jsonlines). Carries the envelope.

One record per `request` a proxy or sensor saw. Read by `TC7HttpStage`, which measures each client's distinct paths and error ratio against its own history and parses the user agent.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `src_ip` | string | -- | yes | Zeek http.log `id.orig_h`, or the Squid client address | -- |
| `dst_ip` | string | -- | yes | `id.resp_h` | -- |
| `dst_port` | integer | -- | yes | `id.resp_p` | -- |
| `http_method` | string | -- | yes | http.log method | -- |
| `url_path` | string | -- | yes | http.log `uri`, without the query string | -- |
| `status_code` | integer | -- | yes | http.log `status_code` | -- |
| `user_agent` | string | -- | yes | http.log `user_agent` | -- |
| `request_body` | string | -- | yes | A content-inspecting proxy's record of the request body (Zeek `http.log` carries none) | Null on a request with no body, and on every request from a proxy that does not inspect content. Read by `TC7SidScoreStage` for R-B-L7-003; sensitive content by definition, so the risk record never carries it. |

## SaaS audit operation (TC-7)

`tc7_saas`, layer 7; sample in [`samples/saas/tc7_saas.jsonlines`](./samples/saas/tc7_saas.jsonlines). Carries the envelope.

One record per operation a `SaaS` audit log recorded. Read by `TC7SaasStage`, which measures each principal's records per operation against its own history.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `user_principal` | string | -- | yes | M365 `UnifiedAuditLog` `UserId` | Lower case. |
| `operation` | string | -- | yes | Operation | -- |
| `target_object` | string | -- | yes | `ObjectId` | -- |
| `target_object_type` | string | -- | yes | `ItemType`, or the workload's object type | -- |
| `record_count` | integer | `count` | yes | The number of records the operation touched | Where the workload reports it; R-B-L7-002 measures it. |
| `result` | string | -- | yes | `ResultStatus` | `success` or `failure`. |
| `client_app` | string | -- | yes | `ClientInfoString` or the workload's client | -- |
| `source_ip` | string | -- | no | `ClientIP` | -- |

## Service connection (TC-7)

`tc7_service`, layer 7; sample in [`samples/application/tc7_service.jsonlines`](./samples/application/tc7_service.jsonlines). Carries the envelope.

One record per connection a load balancer or sensor saw to a `service`. Read by `TC7ServiceStage`, which keys a `service` on its name and port and measures its clients against its own busiest day.

| Column | Type | Unit | Required | Source | Notes |
|---|---|---|---|---|---|
| `src_ip` | string | -- | yes | The client address | -- |
| `dst_ip` | string | -- | yes | The `service` address | -- |
| `dst_port` | integer | -- | yes | The `service` port | -- |
| `sni` | string | -- | yes | The TLS `server` name | Null where the client sent none; the `service` is then keyed on its address. |
| `bytes_total` | integer | `bytes` | yes | Bytes in both directions | -- |
| `ja4_client` | string | -- | yes | The client's JA4 fingerprint | -- |

## What a wrong field costs

- A `port_id` that is a bridge port number rather than an interface name produces bindings that join to nothing,
  and every layer 1 to layer 2 correlation silently returns empty.
- A `site_id` that varies between polls forks one entity into several, each with its own baseline.
- An `event_time` that is the poll time rather than the observation time makes every gap measurement report the
  polling cadence instead of the network's behaviour -- which is exactly the measurement R-D-L2-004 separates a
  spoof from a move with.
- A `collector_seq` that restarts at zero makes two different records collide on one identifier.
- A counter sent as a delta rather than a total cannot be told from a reset, and the rate the layer 1 rules read is
  lost on every reboot.

## Sending it

The pipeline reads JSON lines: one object per line, no wrapping array. `mac_table_sample.jsonlines` is the
seventeen-record example `run_mac_spoof.py` reads by default; the per-segment samples are the corpora the runners
reproduce their golden files from.
