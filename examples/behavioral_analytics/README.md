# Running the segments

One runner per segment takes collector files to the records a SIEM receives:

```bash
python examples/behavioral_analytics/run_network.py --output out/network --state state/network
```

| Runner | Classes it reads | Sourcetypes it writes |
|---|---|---|
| `run_telemetry.py` | `tc1`, `tc1_device`, `tc2_mac`, `tc2_arp`, `tc2_auth` | `morpheus:score:l1`, `morpheus:score:l2`, `binding:l1`, `binding:l2` |
| `run_network.py` | `tc3`, `tc3_dhcp`, `tc7_host_inventory` | `morpheus:score:l3` |
| `run_transport.py` | `tc4`, `tc3_dhcp`, `tc7_host_inventory` | `morpheus:score:l4` |
| `run_session.py` | `tc5_auth`, `tc5_session` | `morpheus:score:l5` |
| `run_presentation.py` | `tc6`, `tc3_dhcp`, `tc7_host_inventory` | `morpheus:score:l6` |
| `run_application.py` | `tc7_dns`, `tc7_http`, `tc7_service`, `tc3_dhcp`, `tc7_host_inventory` | `morpheus:score:l7` |
| `run_saas.py` | `tc7_saas` | `morpheus:score:l7` |
| `run_endpoint.py` | `tc7_endpoint` | `morpheus:score:l7` |
| `run_context.py` | `tc0_identity`, `tc0_asset`, `tc0_application` | `context:identity`, `context:asset`, `context:application` |

Every runner also writes `morpheus:late` for a row that arrived after its window sealed and
`morpheus:quarantine` for a record whose clock could not be trusted. Each reads a directory holding one JSON lines
file per class, `samples/<segment>/` by default, with `FileSourceStage`, runs the segment's composition -- the one
its corpus module under `tests/morpheus/determinism/` holds to a golden file -- and ends in `MinimizationStage`,
`SiemWireStage` and `SiemWireSinkStage`, one file per sourcetype or, with `--kafka`, one topic per sourcetype.
`SiemWireSinkStage` leaves null fields out, which the generic file and Kafka writers do not.

- **`--state DIR`** restores every stage that remembers from the latest checkpoint in `DIR` before the first row,
  and checkpoints when the run stops, keeping open windows and bindings for the next run. `--not-final` says
  another run follows; without it the run flushes as a replay does. Rows carry the digest of the checkpoint they
  resumed from as `resumed_from_state`.
- **`--policy minimization_policy.example.yaml`** pseudonymizes before the wire, with the key from the
  environment variable the policy names. The example policy says how it was arrived at and what it costs.

`tests/morpheus/determinism/test_segment_runners.py` holds every runner to its composition: the files it writes
from the committed samples are, line for line, the in-memory result rendered through the same wire. For the
session, SaaS and endpoint segments it also stops a runner halfway through the data and starts it again from its
checkpoint, and the two runs together write what one does; the other segments rebuild a lookup per run from the
bindings their own run closes, where a deployment keeps those bindings in lookups on the SIEM instead. Under the
example policy no principal from the inputs appears in any segment's output.

## From the `morpheus` command line

Every one of the fifty-eight stages is a `morpheus run pipeline-other` command, and
`tests/morpheus/stages/test_lineage_stage_cli.py` builds each command and holds the compositions below to the
CLI. The runners are the complete form: four stages take Python objects the command line cannot carry --
`determinism-stamp` the envelope, `tc0-enrich` a context store, `binding-resolve` a binding table and `tc5-score`
a model -- and `from-file` reads with the default parser rather than the runners' type-preserving one. What the
CLI composes is therefore each segment's feature stages over its sample, without provenance, context or model
columns, anchored on the record's own key where a binding lookup would have resolved a host; `siem-wire` is told
not to require the columns the omitted stages would have written. An argument the CLI ignores (`floors`,
`key_columns`, a store) keeps its Python default.

Layer 1, from the telemetry segment:

```bash
morpheus run --use_cpu_only pipeline-other \
  from-file --filename samples/telemetry/tc1.jsonlines \
  lineage-stamp --id_columns collector_id --id_columns schema_version --id_columns origin_hash --id_columns collector_seq \
  total-order collector-sequence \
  tc1-normalize --uptime_column uptime --uptime_unit cs \
  tc1-rate --bucket_seconds 300 --min_buckets 6 \
  tc1-optical tc1-forecast \
  tc1-flap --last_change_column if_last_change --last_change_unit cs \
  tc1-change \
  chain-anchor --candidates entity_key \
  envelope-stamp --osi_layer 1 --entity_columns site_id --entity_columns device_id --entity_columns port_id \
  window-seal --period_seconds 300 --lateness_seconds 900 --order_columns event_time --order_columns collector_id --order_columns collector_seq --entity_key_column chain_anchor \
  siem-wire --sourcetype morpheus:score:l1 --require_columns False \
  to-siem-wire --filename out/morpheus_score_l1.jsonlines
```

Layer 3, the network segment:

```bash
morpheus run --use_cpu_only pipeline-other \
  from-file --filename samples/network/tc3.jsonlines \
  lineage-stamp --id_columns collector_id --id_columns schema_version --id_columns origin_hash --id_columns collector_seq \
  total-order collector-sequence \
  community-id --dst_ip_column dst_ip --dst_port_column dst_port \
  tc3-cardinality --window_seconds 3600 \
  tc3-reach --window_seconds 3600 \
  tc3-baseline --bucket_seconds 3600 --min_buckets 24 \
  tc3-beacon --window_seconds 14400 --min_intervals 12 \
  tc3-ttl --window_seconds 14400 --min_samples 5 \
  chain-anchor --candidates src_ip \
  envelope-stamp --osi_layer 3 --entity_columns src_ip \
  window-seal --period_seconds 3600 --lateness_seconds 900 --order_columns event_time --order_columns collector_id --order_columns collector_seq --entity_key_column chain_anchor \
  siem-wire --sourcetype morpheus:score:l3 --require_columns False \
  to-siem-wire --filename out/morpheus_score_l3.jsonlines
```

Layer 4, the transport segment:

```bash
morpheus run --use_cpu_only pipeline-other \
  from-file --filename samples/transport/tc4.jsonlines \
  lineage-stamp --id_columns collector_id --id_columns schema_version --id_columns origin_hash --id_columns collector_seq \
  total-order collector-sequence \
  tc4-flow --bin_seconds 60 \
  tc4-envelope --window_seconds 28800 \
  community-id --dst_ip_column dst_ip --dst_port_column dst_port \
  chain-anchor --candidates flow_id \
  envelope-stamp --osi_layer 4 --entity_columns flow_id \
  window-seal --period_seconds 3600 --lateness_seconds 900 --order_columns event_time --order_columns collector_id --order_columns collector_seq --entity_key_column chain_anchor \
  siem-wire --sourcetype morpheus:score:l4 --require_columns False \
  to-siem-wire --filename out/morpheus_score_l4.jsonlines
```

Layer 5, the authentication class of the session segment. The daily seal and the drift stage behind it read the
model's score, so they go with `tc5-score`:

```bash
morpheus run --use_cpu_only pipeline-other \
  from-file --filename samples/session/tc5_auth.jsonlines \
  lineage-stamp --id_columns collector_id --id_columns schema_version --id_columns origin_hash --id_columns collector_seq \
  total-order collector-sequence \
  tc5-novelty \
  tc5-cadence --min_samples 6 \
  tc5-travel --excluded_source_networks 198.51.100.0/24 \
  tc5-risk --min_denominator 1 \
  envelope-stamp --osi_layer 5 --entity_columns user_principal \
  window-seal --period_seconds 3600 --lateness_seconds 900 --order_columns event_time --order_columns collector_id --order_columns collector_seq --entity_key_column user_principal \
  siem-wire --sourcetype morpheus:score:l5 --require_columns False \
  to-siem-wire --filename out/morpheus_score_l5.jsonlines
```

Layer 6, the presentation segment:

```bash
morpheus run --use_cpu_only pipeline-other \
  from-file --filename samples/presentation/tc6.jsonlines \
  lineage-stamp --id_columns collector_id --id_columns schema_version --id_columns origin_hash --id_columns collector_seq \
  total-order collector-sequence \
  community-id --dst_ip_column dst_ip --dst_port_column dst_port \
  tc6-fingerprint \
  tc6-certificate --min_samples 5 \
  tc6-cipher --min_samples 5 \
  tc6-content \
  chain-anchor --candidates src_ip \
  envelope-stamp --osi_layer 6 --entity_columns src_ip \
  window-seal --period_seconds 3600 --lateness_seconds 900 --order_columns event_time --order_columns collector_id --order_columns collector_seq --entity_key_column chain_anchor \
  siem-wire --sourcetype morpheus:score:l6 --require_columns False \
  to-siem-wire --filename out/morpheus_score_l6.jsonlines
```

Layer 7, the DNS class of the application segment:

```bash
morpheus run --use_cpu_only pipeline-other \
  from-file --filename samples/application/tc7_dns.jsonlines \
  lineage-stamp --id_columns collector_id --id_columns schema_version --id_columns origin_hash --id_columns collector_seq \
  total-order collector-sequence \
  tc7-dns \
  chain-anchor --candidates src_ip \
  envelope-stamp --osi_layer 7 --entity_columns src_ip \
  window-seal --period_seconds 3600 --lateness_seconds 900 --order_columns event_time --order_columns collector_id --order_columns collector_seq --entity_key_column chain_anchor \
  siem-wire --sourcetype morpheus:score:l7 --require_columns False \
  to-siem-wire --filename out/morpheus_score_l7.jsonlines
```

Layer 7, the SaaS segment, with its weekly seal and the breadth trajectory over it:

```bash
morpheus run --use_cpu_only pipeline-other \
  from-file --filename samples/saas/tc7_saas.jsonlines \
  lineage-stamp --id_columns collector_id --id_columns schema_version --id_columns origin_hash --id_columns collector_seq \
  total-order collector-sequence \
  tc7-saas \
  envelope-stamp --osi_layer 7 --entity_columns user_principal \
  window-seal --period_seconds 3600 --lateness_seconds 900 --order_columns event_time --order_columns collector_id --order_columns collector_seq --entity_key_column entity_key \
  window-seal --period_seconds 604800 --lateness_seconds 900 --epoch 1970-01-05 --order_columns event_time --order_columns collector_id --order_columns collector_seq --column_prefix week_ \
  tc5-drift --score_column saas_object_types_in_week --window_column week_window_id --aggregate max \
  siem-wire --sourcetype morpheus:score:l7 --require_columns False \
  to-siem-wire --filename out/morpheus_score_l7.jsonlines
```

Layer 7, the endpoint segment:

```bash
morpheus run --use_cpu_only pipeline-other \
  from-file --filename samples/endpoint/tc7_endpoint.jsonlines \
  lineage-stamp --id_columns collector_id --id_columns schema_version --id_columns origin_hash --id_columns collector_seq \
  total-order collector-sequence \
  tc7-endpoint --window_days 30 --warmup_days 7 \
  envelope-stamp --osi_layer 7 --entity_columns hostname \
  window-seal --period_seconds 3600 --lateness_seconds 900 --order_columns event_time --order_columns collector_id --order_columns collector_seq --entity_key_column entity_key \
  siem-wire --sourcetype morpheus:score:l7 --require_columns False \
  to-siem-wire --filename out/morpheus_score_l7.jsonlines
```

The identity class of the context segment, which is a store rather than a stream and is not sealed:

```bash
morpheus run --use_cpu_only pipeline-other \
  from-file --filename samples/context/tc0_identity.jsonlines \
  tc0-identity \
  siem-wire --sourcetype context:identity --require_columns False \
  to-siem-wire --filename out/context_identity.jsonlines
```

The test parses every block above into a pipeline, so a renamed option fails it, and runs the last one, so the
command line is shown to produce records and not only to parse.

## From real feeds

`fixtures/` holds six of the samples as the feeds that produce them write them, and `fixtures/make_fixtures.py`
renders and reads them back:

| Class | Fixture | Read with |
|---|---|---|
| `tc3` | `network/tc3.pcap`, IPFIX from three routers, one message per flow | `morpheus.utils.ipfix.read_pcap`, then `FlowRecordNormalizeStage` |
| `tc6` | `presentation/ssl.log` and `x509.log` (Zeek, with the JA4 package's `ja4`) | `collector_schemas.zeek_ssl` |
| `tc7_http` | `application/http.log` (Zeek) | `collector_schemas.zeek_http` |
| `tc7_dns` | `application/dns.log` (Zeek) | `collector_schemas.zeek_dns` |
| `tc7_saas` | `saas/audit.jsonlines`, unified audit log records | `collector_schemas.m365_audit` |
| `tc7_endpoint` | `endpoint/sysmon.xml`, events 1 and 7 as `wevtutil qe /f:xml` writes them | `collector_schemas.sysmon_process` |

A fixture carries only what its format carries. The collector's own facts -- its identity, contract version,
tenant, clock discipline and sampling -- are a `CollectorProfile`, the exporters' names are an inventory the stage
is given, and `collector_seq` counts records in the order read. `tests/morpheus/utils/test_collector_schemas.py`
holds each normalized fixture to its sample, column for column, and runs the network, application, SaaS and
endpoint segments over them to the same files the samples give, byte for byte. Zeek does not see inside a TLS
session, so the two content types an inspecting proxy reports are null from the Zeek schema, and the
presentation segment's content rule needs a feed that supplies them.

# R-D-L2-004, end to end

One command turns a MAC address table into the records the shipped Splunk detection reads:

```bash
python examples/behavioral_analytics/run_mac_spoof.py
```

It reads `mac_table_sample.jsonlines` and writes `notables.jsonlines` beside it. With
`--policy minimization_policy.example.yaml` it pseudonymizes the addresses and ports before the wire, and the one
notable still fires.

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
lookup, no exclusion list, no TC-0 context store. Several of the fifty-six shipped detections need something an
estate has to supply first, a designation list, an exclusion list or the TC-0 context store, and the rest have not
been run this way yet, so proving one rule end to end is worth more than gesturing at fifty-six.

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

[`collector_contract.md`](./collector_contract.md) names, per class and per column, the type, the unit, the collector-side
source (an SNMP object identifier, an IPFIX element, a Zeek or Windows System Monitor field) and whether it is required, for
all twenty input classes, so a collector author can check their own output against something other than prose. It is
rendered from [`input_contracts.json`](./input_contracts.json), and [`samples/`](./samples) holds every segment's
corpus as the JSON lines a collector would send, one file per class. Both are regenerated by
`make_input_samples.py` and checked by `tests/morpheus/determinism/test_input_contracts.py`.
