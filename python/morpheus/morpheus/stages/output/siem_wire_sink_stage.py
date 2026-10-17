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
"""

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
    """

    _producer_factory: typing.ClassVar[typing.Callable[[str], typing.Any]] = staticmethod(_kafka_producer)
    """How the Kafka producer is built. A class attribute so a test can stand a recorder in for a broker."""

    def __init__(self,
                 c: Config,
                 filename: str = None,
                 overwrite: bool = False,
                 append: bool = False,
                 kafka_topic: str = None,
                 kafka_bootstrap_servers: str = "localhost:9092"):
        super().__init__(c)

        if ((filename is None) == (kafka_topic is None)):
            raise ValueError("SiemWireSinkStage writes to a file or to a Kafka topic; give exactly one of filename "
                             "and kafka_topic")

        if (overwrite and append):
            raise ValueError("overwrite replaces the file and append continues it; give at most one")

        if (filename is not None and os.path.exists(filename) and not (overwrite or append)):
            raise FileExistsError(f"{filename} exists; pass overwrite to replace it or append to continue it")

        self._filename = filename
        self._kafka_topic = kafka_topic
        self._bootstrap_servers = kafka_bootstrap_servers
        self._producer = None
        self._written = 0

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

            for line in lines:
                self._producer.produce(self._kafka_topic, line.encode("utf-8"))

            self._producer.flush()

        self._written += len(lines)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
