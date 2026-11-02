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
Writes records to a file or a Kafka topic the way a SIEM must receive them: one JSON object per record, with every
null field left out.

`SiemWireStage` renders the timestamps a stanza parses and says, in its own documentation, that dropping nulls is
the sink's job and that the sink has to serialize with `morpheus.utils.siem_wire.to_wire_lines`. Neither sink this
repository ships does: `WriteToFileStage` and `WriteToKafkaStage` both send a null as `"field":null`, which the
first search-head run showed a SIEM reading as a value rather than as absent. This is the sink that keeps the
promise, so a runner can end in a stage rather than in code after the pipeline.

Kafka is the transport the app recommends: a Kafka Connect for Splunk connector reads the topic and forwards each
record to HEC unchanged, so the record must already be in its wire form when it is produced.

**Every record reaches its topic or the stage says so.** Each message is keyed on the record's `entity_key`, so one
entity's records stay on one partition and in order for whatever consumes the topic next. A produce the broker does
not acknowledge is retried; one that still fails goes to the quarantine topic with the reason and the line, when the
stage has one, and otherwise stops the pipeline with an error. The upstream sink logged a failed produce and moved on
(guide 13, plan item 4), which is a record lost with nothing downstream able to tell.
"""

import json
import logging
import os
import typing

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.column_assign import to_host_frame
from morpheus.utils.siem_wire import to_wire_lines

logger = logging.getLogger(__name__)


def _kafka_producer(bootstrap_servers: str):
    # Deferred, so a file sink never needs the Kafka client installed.
    import confluent_kafka  # pylint: disable=import-outside-toplevel

    return confluent_kafka.Producer({"bootstrap.servers": bootstrap_servers})


@register_stage("to-siem-wire")
class SiemWireSinkStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Serialize each record with `to_wire_lines` and write it to a file or produce it to a Kafka topic.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    filename : str, optional
        File to write, one wire line per record. Exactly one of `filename` and `kafka_topic` is given.
    overwrite : bool, default = False
        Replace an existing file rather than refusing to. A runner that resumes from a checkpoint appends
        instead, with `append`.
    append : bool, default = False
        Add to an existing file rather than replacing it, so a process resumed from a checkpoint continues the
        file its predecessor wrote.
    kafka_topic : str, optional
        Topic to produce each record to.
    kafka_bootstrap_servers : str, default = "localhost:9092"
        The brokers, when `kafka_topic` is given.
    key_column : str, default = "entity_key"
        The record field each message is keyed on. A record without it is produced with no key.
    max_retries : int, default = 3
        How many more times a message the broker did not acknowledge is produced before it is given up on.
    quarantine_topic : str, optional
        Where a message given up on is produced instead, as a record naming the topic it was meant for, the error
        and the line. Without one, a message given up on raises.
    flush_timeout_seconds : float, default = 30
        How long each flush waits for the broker to acknowledge. A message still unacknowledged then counts as
        failed.
    """

    _producer_factory: typing.ClassVar[typing.Callable[[str], typing.Any]] = staticmethod(_kafka_producer)
    """How the Kafka producer is built. A class attribute so a test can stand a recorder in for a broker."""

    def __init__(self,
                 c: Config,
                 filename: str = None,
                 overwrite: bool = False,
                 append: bool = False,
                 kafka_topic: str = None,
                 kafka_bootstrap_servers: str = "localhost:9092",
                 key_column: str = "entity_key",
                 max_retries: int = 3,
                 quarantine_topic: str = None,
                 flush_timeout_seconds: float = 30.0):
        super().__init__(c)

        if ((filename is None) == (kafka_topic is None)):
            raise ValueError("SiemWireSinkStage writes to a file or to a Kafka topic; give exactly one of filename "
                             "and kafka_topic")

        if (max_retries < 0):
            raise ValueError(f"max_retries must not be negative, received {max_retries}")

        if (flush_timeout_seconds <= 0):
            raise ValueError(f"flush_timeout_seconds must be positive, received {flush_timeout_seconds}")

        if (overwrite and append):
            raise ValueError("overwrite replaces the file and append continues it; give at most one")

        if (filename is not None and os.path.exists(filename) and not (overwrite or append)):
            raise FileExistsError(f"{filename} exists; pass overwrite to replace it or append to continue it")

        self._filename = filename
        self._kafka_topic = kafka_topic
        self._bootstrap_servers = kafka_bootstrap_servers
        self._key_column = key_column
        self._max_retries = max_retries
        self._quarantine_topic = quarantine_topic
        self._flush_timeout = flush_timeout_seconds
        self._producer = None
        self._written = 0
        self._quarantined = 0

        if (filename is not None and not append):
            # Truncated here rather than at the first batch, so a run that emits nothing leaves an empty file
            # rather than its predecessor's.
            with open(filename, "w", encoding="utf-8"):
                pass

    @property
    def name(self) -> str:
        """Stage name."""
        return "to-siem-wire"

    @property
    def written(self) -> int:
        """Records written so far."""
        return self._written

    @property
    def quarantined(self) -> int:
        """Records the broker would not take, produced to the quarantine topic instead."""
        return self._quarantined

    def _key(self, line: str) -> typing.Optional[bytes]:
        value = json.loads(line).get(self._key_column)

        return None if value is None else str(value).encode("utf-8")

    def _deliver(self, topic: str, messages: list[tuple[typing.Optional[bytes], bytes]]) -> list:
        """
        Produce every message and wait for the broker; return the ones it did not acknowledge, with the error.

        The delivery report is the only place a failed produce shows: `produce` succeeds as soon as the message is
        queued locally.
        """
        failed: list = []
        outcome: dict = {}

        def report(position: int):

            def on_delivery(error, _message):
                outcome[position] = error

            return on_delivery

        for (position, (key, value)) in enumerate(messages):
            while True:
                try:
                    self._producer.produce(topic, value, key=key, on_delivery=report(position))
                    break
                except BufferError:
                    # The local queue is full; serving delivery reports drains it.
                    self._producer.poll(1.0)

        self._producer.flush(self._flush_timeout)

        for (position, message) in enumerate(messages):
            error = outcome.get(position, "no delivery report before the flush timed out")

            if (error is not None):
                failed.append((message, error))

        return failed

    def _produce(self, lines: list[str]):
        messages = [(self._key(line), line.encode("utf-8")) for line in lines]
        failed = self._deliver(self._kafka_topic, messages)
        attempt = 0

        while (failed and attempt < self._max_retries):
            attempt += 1
            logger.warning("SiemWireSinkStage: the broker did not acknowledge %d messages to %s; retry %d of %d.",
                           len(failed),
                           self._kafka_topic,
                           attempt,
                           self._max_retries)
            failed = self._deliver(self._kafka_topic, [message for (message, _) in failed])

        if (not failed):
            return

        if (self._quarantine_topic is None):
            raise RuntimeError(f"SiemWireSinkStage could not produce {len(failed)} records to {self._kafka_topic} "
                               f"after {self._max_retries} retries (first error: {failed[0][1]}), and has no "
                               f"quarantine topic to put them on. Stopping rather than losing them.")

        records = [(key,
                    json.dumps({
                        "quarantine_reason": "produce_failed",
                        "intended_topic": self._kafka_topic,
                        "error": str(error),
                        "payload": value.decode("utf-8"),
                    }).encode("utf-8")) for ((key, value), error) in failed]
        unplaced = self._deliver(self._quarantine_topic, records)

        if (unplaced):
            raise RuntimeError(f"SiemWireSinkStage could not produce {len(unplaced)} records to {self._kafka_topic} "
                               f"or to the quarantine topic {self._quarantine_topic} (first error: {unplaced[0][1]}).")

        self._quarantined += len(failed)
        logger.error("SiemWireSinkStage put %d records the broker would not take for %s on %s.",
                     len(failed),
                     self._kafka_topic,
                     self._quarantine_topic)

    def accepted_types(self) -> tuple:
        """
        Accepted input types for this stage.

        Returns
        -------
        tuple
            Accepted input types.
        """
        return (ControlMessage, MessageMeta)

    def supports_cpp_node(self) -> bool:
        """Whether this stage supports a C++ node."""
        return False

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write every record of the message in its wire form.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Records already rendered by `SiemWireStage`.

        Returns
        -------
        The input message, unchanged.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        lines = to_wire_lines(to_host_frame(meta.copy_dataframe()))

        if (self._filename is not None):
            with open(self._filename, "a", encoding="utf-8") as out:
                for line in lines:
                    out.write(line + "\n")
        else:
            if (self._producer is None):
                self._producer = type(self)._producer_factory(self._bootstrap_servers)

            self._produce(lines)

        self._written += len(lines)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
