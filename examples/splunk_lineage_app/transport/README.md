<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Getting the records into Splunk

The segment runners under `examples/behavioral_analytics/` end in `SiemWireSinkStage`, which writes each
sourcetype's records to a Kafka topic or to a file. This directory holds the configuration that takes them the
rest of the way, rendered by `make_transport.py` from the sourcetype inventory in
`morpheus.utils.siem_sourcetypes`, so the names below are the names the pipeline uses. Edit the script, not the
files; `tests/morpheus/utils/test_splunk_transport.py` fails when the two differ.

| Sourcetype | Kafka topic | Runner file | Index |
|---|---|---|---|
| `binding:bucketed` | `binding.bucketed` | `binding_bucketed.jsonlines` | `behavior_bindings` |
| `binding:l1` | `binding.l1` | `binding_l1.jsonlines` | `behavior_bindings` |
| `binding:l2` | `binding.l2` | `binding_l2.jsonlines` | `behavior_bindings` |
| `binding:l2:open` | `binding.l2.open` | `binding_l2_open.jsonlines` | `behavior_bindings` |
| `context:application` | `context.application` | `context_application.jsonlines` | `behavior_context` |
| `context:asset` | `context.asset` | `context_asset.jsonlines` | `behavior_context` |
| `context:identity` | `context.identity` | `context_identity.jsonlines` | `behavior_context` |
| `morpheus:edge` | `morpheus.edge` | `morpheus_edge.jsonlines` | `behavior_lineage` |
| `morpheus:health` | `morpheus.health` | `morpheus_health.jsonlines` | `behavior_health` |
| `morpheus:late` | `morpheus.late` | `morpheus_late.jsonlines` | `behavior_late` |
| `morpheus:quarantine` | `morpheus.quarantine` | `morpheus_quarantine.jsonlines` | `behavior_quarantine` |
| `morpheus:score:host` | `morpheus.score.host` | `morpheus_score_host.jsonlines` | `behavior_events` |
| `morpheus:score:l1` | `morpheus.score.l1` | `morpheus_score_l1.jsonlines` | `behavior_events` |
| `morpheus:score:l2` | `morpheus.score.l2` | `morpheus_score_l2.jsonlines` | `behavior_events` |
| `morpheus:score:l3` | `morpheus.score.l3` | `morpheus_score_l3.jsonlines` | `behavior_events` |
| `morpheus:score:l4` | `morpheus.score.l4` | `morpheus_score_l4.jsonlines` | `behavior_events` |
| `morpheus:score:l5` | `morpheus.score.l5` | `morpheus_score_l5.jsonlines` | `behavior_events` |
| `morpheus:score:l6` | `morpheus.score.l6` | `morpheus_score_l6.jsonlines` | `behavior_events` |
| `morpheus:score:l7` | `morpheus.score.l7` | `morpheus_score_l7.jsonlines` | `behavior_events` |

## Kafka, with Splunk Connect for Kafka

`kafka-connect-splunk.json` is the sink connector's configuration, applied with
`curl -X PUT -H 'Content-Type: application/json' --data @kafka-connect-splunk.json
http://connect:8083/connectors/morpheus-lineage-splunk/config` (drop the outer `name` for the `PUT` form, or `POST` the
file as it is to `/connectors`). It is the app's recommended transport: Kafka holds the records when Splunk is
slow or away, and a consumer offset can be reset to replay them.

- **`topics`, `splunk.sourcetypes` and `splunk.indexes` are positional.** The three lists are in the same order,
  so the record on the first topic is indexed with the first sourcetype into the first index. A topic is its
  sourcetype with the colons as dots, which is what the runners produce to (`--kafka`).
- **`splunk.hec.raw` is `true`.** The raw endpoint hands each record to the indexer's parsing pipeline, where the
  sourcetype's `props.conf` stanza anchors `_time` on `event_time` -- the one thing the app's own validation
  exercised. The event endpoint would instead stamp the Kafka record's timestamp on the event
  (`splunk.hec.use.record.timestamp`, true by default), which is when the record was produced, not when the
  behavior happened, and every windowed detection would read the pipeline's schedule rather than the estate's.
- **`splunk.hec.raw.line.breaker` is a newline.** The connector posts records in batches and puts this between
  them; the stanzas' default line breaking splits on it. Leave it: with no breaker the batch is one event.
- **`splunk.hec.ack.enabled` is `true`.** The connector commits a Kafka offset only after the indexer
  acknowledges the record, so a record is delivered once or is still on the topic. The token has `useACK = 1`
  to match.
- **The token is read from a secrets file.** `${file:...}` is a Kafka Connect configuration provider; the
  worker needs `config.providers=file` and `config.providers.file.class=org.apache.kafka.common.config.provider.FileConfigProvider`.
  Put the token nowhere else.
- **`splunk.hec.uri`** is a placeholder: the indexers that run the `[http]` input, or a load balancer in front
  of them.

## Without a broker

`inputs.conf` holds the other two options from the guide's Part 4.

- **The HEC token** (`[http]` and `[http://morpheus-lineage]`) is what the connector above posts to, and what
  `HttpClientSinkStage` would post to if a stream went to Splunk directly. It allows exactly the indexes the
  sourcetypes use, so a wrong topic mapping is refused rather than indexed where no search reads. Generate the
  token on the instance; the value here is a placeholder.
- **The monitors** are a universal forwarder reading the files a runner writes with `--output`, one directory
  per segment under `/var/spool/morpheus/`. Each monitor names the sourcetype and index for one file name, and matches it
  in every segment's directory. The runners append across restarts, which a monitor follows. `crcSalt = <SOURCE>`
  stops two segments' files for one sourcetype -- which begin with the same envelope fields -- from being taken
  for one file already read.

## What is and is not tested

The test holds these files to the inventory: every produced sourcetype has a topic, a monitor and an index; the
index is the one the app's own searches read that sourcetype from; every index exists in `indexes.conf` and
every sourcetype in `props.conf`; the topic and file names are the ones `SiemWireSinkStage` is given by the
runners; and the three positional lists in the connector are the same length and in the same order. What it does
not do is run a connector or a forwarder: the behavior of the raw endpoint and the line breaker described above
is the connector's documented behavior, and the search-head run under `../validate/` is where to confirm it
against a live instance.
