# R-D-L2-004, end to end

One command turns a MAC address table into the records the shipped Splunk detection reads:

```bash
python examples/behavioral_analytics/run_mac_spoof.py
```

It reads `mac_table_sample.jsonlines` and writes `notables.jsonlines` beside it.

## Why this exists

Every other test in this fork stops one hop short of the wire. They apply a rule's predicate to a DataFrame in
memory, which answers whether the analytics are right and says nothing about whether the bytes a SIEM receives
carry what the rule reads. Two of this app's defects lived in exactly that hop: a timestamp Splunk could not
parse, and a field the refresh search filtered on that the producer never wrote. Neither was visible from a
DataFrame.

This is therefore the narrowest thing here that can honestly be called deployed. `tests/morpheus/determinism/test_end_to_end_mac_spoof.py` runs it, reads the output back off disk,
stamps `_time` by applying the shipped `props.conf`'s own `TIME_PREFIX` and `TIME_FORMAT` to the raw bytes, and
applies R-D-L2-004's predicate to the parsed JSON rather than to a frame.

R-D-L2-004 is the rule this can be done for. It needs one collector and nothing else -- no `port_designations`
lookup, no exclusion list, no TC-0 context store. The other three shipped detections each need something an
estate has to supply first, so proving one rule end to end is worth more than gesturing at four.

## What the sample contains

Four snapshots of one switch's MAC address table, five minutes apart, four hosts. In the third snapshot one
address appears on **two ports in the same snapshot, at the same instant**. A collector walking two switches cannot
produce that; one switch reporting one address twice can, and a device cannot be in two places at once.

The output has six closed binding records, and only one of them is a notable:

| `port_key` | `bind_end_reason` | gap | notable? |
|---|---|---|---|
| `hq:sw1:Gi1/0/2` | `conflict` | 0s | **yes** |
| `hq:sw1:Gi1/0/9` | `displaced` | 300s | no -- a legitimate move, over the rule's 60-second threshold |
| four others | `drained` | -- | no -- the stream ended, which is not an observation about the network |

The second row is the part worth reading twice. The corpus contains a spoof *and* an ordinary device that moved
desks, and the rule separates them on the gap between the two sightings rather than on a suppression list.

## What is not here

The pipeline writes closed bindings only. `TC2BindingStage` can also emit a provisional record the moment a
binding opens, and those matter for live attribution, but they carry a null `bind_end` -- and `binding:l2` anchors
`_time` on it. Writing them here would put records with no anchor onto a stanza that reads one, which is the
silent index-time stamping this app's configuration warns about. They have their own stanza,
`binding:l2:open`, anchored on `bind_start`.

This also does not run Splunk. It produces the bytes and checks them against Splunk's own parsing configuration;
whether a particular Splunk instance is configured the way `props.conf` says is a separate question, answered by
deploying the app.

## Column contract

[`collector_contract.md`](./collector_contract.md) names, per column, the SNMP object identifier or CLI command that produces
it and the unit it arrives in, so a collector author can check their own output against something other
than prose.
