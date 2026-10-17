#!/usr/bin/env python
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
Render the transport configuration from the sourcetype inventory, so the topics, files, sourcetypes and indexes
it names are the ones the pipeline produces and the app parses.

Three files, all written beside this script:

- `kafka-connect-splunk.json`, a Splunk Connect for Kafka sink: one topic per produced sourcetype, each mapped to
  its sourcetype and index, posted to the HTTP Event Collector's raw endpoint.
- `inputs.conf`, the alternative without a broker: the HEC token stanza the connector posts to, and a universal
  forwarder's monitor stanzas over the files the segment runners write.
- `README.md`, which says when to use which and what each setting is for.

`tests/morpheus/utils/test_splunk_transport.py` asserts the committed files are what this renders, and holds the
rendering to `props.conf`, `indexes.conf`, the saved searches and the runners' own naming.
"""

import json
import os
import sys

from morpheus.utils.siem_sourcetypes import PRODUCED
from morpheus.utils.siem_sourcetypes import index_for
from morpheus.utils.siem_wire import filename_for
from morpheus.utils.siem_wire import topic_for

HERE = os.path.dirname(os.path.abspath(__file__))

CONNECTOR_NAME = "morpheus-lineage-splunk"
HEC_URI = "https://hec.example.internal:8088"
"""Where the collector listens. One or several indexers, or a load balancer in front of them, comma-separated."""

HEC_TOKEN_REFERENCE = "${file:/etc/kafka-connect/secrets.properties:morpheus.hec.token}"
"""The token, read through a Kafka Connect configuration provider rather than written into this file."""

HEC_TOKEN_NAME = "morpheus-lineage"
SPOOL = "/var/spool/morpheus"
"""Where the segment runners write, one directory per segment, when the files are what a forwarder reads."""

LICENSE = """# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#"""


def sourcetypes() -> list:
    """The produced sourcetypes, in the one order every rendering uses."""
    return sorted(PRODUCED)


def indexes() -> list:
    """Every index the produced sourcetypes land in, sorted."""
    return sorted({index_for(name) for name in sourcetypes()})


def render_connector() -> str:
    """The Splunk Connect for Kafka sink configuration, as `PUT /connectors/<name>/config` takes it."""
    names = sourcetypes()
    config = {
        "connector.class": "com.splunk.kafka.connect.SplunkSinkConnector",
        "tasks.max": "4",
        "topics": ",".join(topic_for(name) for name in names),
        "splunk.hec.uri": HEC_URI,
        "splunk.hec.token": HEC_TOKEN_REFERENCE,
        "splunk.hec.raw": "true",
        "splunk.hec.raw.line.breaker": "\n",
        "splunk.indexes": ",".join(index_for(name) for name in names),
        "splunk.sourcetypes": ",".join(names),
        "splunk.sources": "morpheus",
        "splunk.hec.ack.enabled": "true",
        "splunk.hec.ssl.validate.certs": "true",
        "key.converter": "org.apache.kafka.connect.storage.StringConverter",
        "value.converter": "org.apache.kafka.connect.storage.StringConverter",
    }

    return json.dumps({"name": CONNECTOR_NAME, "config": config}, indent=2) + "\n"


def render_inputs() -> str:
    """The HEC token the connector posts to, and a forwarder's monitors over the runners' files."""
    lines = [
        LICENSE,
        "# Deploy the [http] stanzas to the indexers (or heavy forwarders) that receive from the Kafka connector,",
        "# and the [monitor] stanzas to the universal forwarder on the host where the segment runners write.",
        "# Neither is used without the other half of the choice: a deployment on Kafka needs only the token, and",
        "# one on files needs only the monitors. Both name every sourcetype and index from",
        "# morpheus.utils.siem_sourcetypes, and a test holds them to it; edit make_transport.py, not this file.",
        "",
        "[http]",
        "disabled = 0",
        "port = 8088",
        "enableSSL = 1",
        "",
        f"[http://{HEC_TOKEN_NAME}]",
        "disabled = 0",
        "# Generate the token on the instance and keep it with the connector's secrets, never in this file.",
        "token = REPLACE-WITH-THE-GENERATED-TOKEN",
        "# The connector names an index on every post, so the default is a backstop, and the allowed list is what",
        "# stops a misconfigured topic from landing anywhere else.",
        "index = behavior_events",
        f"indexes = {','.join(indexes())}",
        "# Indexer acknowledgement, so the connector commits a Kafka offset only once the indexer has the record.",
        "useACK = 1",
        "",
        "# One monitor per sourcetype over every segment's directory. The runners append to these files across",
        "# restarts, which a monitor follows. crcSalt is load-bearing: two segments' files for one sourcetype begin",
        "# with the same envelope fields, and without it Splunk takes the second file for the first, already read.",
    ]

    for name in sourcetypes():
        lines += [
            "",
            f"[monitor://{SPOOL}/*/{filename_for(name)}]",
            "disabled = 0",
            f"sourcetype = {name}",
            f"index = {index_for(name)}",
            "crcSalt = <SOURCE>",
        ]

    return "\n".join(lines) + "\n"


def render_readme() -> str:
    """The directory's README: which transport to use, what each setting does, and what is and is not tested."""
    rows = "\n".join(f"| `{name}` | `{topic_for(name)}` | `{filename_for(name)}` | `{index_for(name)}` |"
                     for name in sourcetypes())

    return f"""<!--
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
{rows}

## Kafka, with Splunk Connect for Kafka

`kafka-connect-splunk.json` is the sink connector's configuration, applied with
`curl -X PUT -H 'Content-Type: application/json' --data @kafka-connect-splunk.json
http://connect:8083/connectors/{CONNECTOR_NAME}/config` (drop the outer `name` for the `PUT` form, or `POST` the
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
- **The token is read from a secrets file.** `${{file:...}}` is a Kafka Connect configuration provider; the
  worker needs `config.providers=file` and `config.providers.file.class=org.apache.kafka.common.config.provider.FileConfigProvider`.
  Put the token nowhere else.
- **`splunk.hec.uri`** is a placeholder: the indexers that run the `[http]` input, or a load balancer in front
  of them.

## Without a broker

`inputs.conf` holds the other two options from the guide's Part 4.

- **The HEC token** (`[http]` and `[http://{HEC_TOKEN_NAME}]`) is what the connector above posts to, and what
  `HttpClientSinkStage` would post to if a stream went to Splunk directly. It allows exactly the indexes the
  sourcetypes use, so a wrong topic mapping is refused rather than indexed where no search reads. Generate the
  token on the instance; the value here is a placeholder.
- **The monitors** are a universal forwarder reading the files a runner writes with `--output`, one directory
  per segment under `{SPOOL}/`. Each monitor names the sourcetype and index for one file name, and matches it
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
"""


def main(argv: list = None) -> int:
    """Write the three files, or with `--check` say whether they are current."""
    check = "--check" in (argv or sys.argv[1:])
    rendered = {
        "kafka-connect-splunk.json": render_connector(),
        "inputs.conf": render_inputs(),
        "README.md": render_readme(),
    }
    stale = []

    for (name, text) in rendered.items():
        path = os.path.join(HERE, name)

        if (check):
            with open(path, encoding="utf-8") as handle:
                if (handle.read() != text):
                    stale.append(name)
        else:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)

    if (stale):
        print("stale: " + ", ".join(stale))
        return 1

    return 0


if (__name__ == "__main__"):
    sys.exit(main())
