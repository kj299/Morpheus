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
"""Renders the timestamps a SIEM parses, immediately before the sink that serializes them."""

import logging
import typing

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.siem_sourcetypes import sourcetype as lookup_sourcetype
from morpheus.utils.siem_wire import render_event_time_series

logger = logging.getLogger(__name__)


@register_stage("siem-wire")
class SiemWireStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Render a batch's timestamps into the wire format the SIEM's parsing configuration expects.

    Inside the pipeline an event time is an integer count of nanoseconds. On the wire it has to be a quoted string
    the SIEM can parse, and the failure when it is not is the quietest one in this whole design: Splunk's
    `TIME_PREFIX` requires a quote after the field name, an integer serializes without one, no rule matches, no
    error is raised, and `_time` falls back to the moment the record was indexed. Every windowed detection then
    silently becomes a rule about when the pipeline was busy. The app's `props.conf` devotes a paragraph to this
    and `morpheus.utils.siem_wire` devotes a docstring to it; this stage is the thing that actually prevents it.

    Placed last, immediately before the sink that serializes. It renders in place, so what the sink receives is
    what the SIEM receives.

    Which columns get rendered is not a parameter. It comes from `morpheus.utils.siem_sourcetypes`, which records
    for each stanza in the shipped app what its `TIME_PREFIX` anchors on, and a contract test asserts that record
    against the configuration file itself. Naming the sourcetype is therefore the whole configuration: a caller
    cannot render the wrong column for a stanza without the two halves of that contract disagreeing first.

    A record usually carries more than one timestamp, and all of them are rendered rather than only the anchor. A
    consumer reading `bind_start` off a closed binding should not have to know that one field on the record is a
    string and its sibling is a nineteen-digit integer.

    Columns whose names end in `_ns` are deliberately untouched. They are the exact values a consumer computes
    with, nothing takes `_time` from them, and rounding them to microseconds to fit a timestamp format would
    quietly change arithmetic that depends on them.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    sourcetype : str
        A stanza in the shipped app's `props.conf`, for example `morpheus:score:l2`. Naming one nothing in this
        fork produces raises, and the message says what would have to be built.
    require_columns : bool, default = True
        Raise when the batch is missing a column the sourcetype requires. Turning this off renders whatever is
        present and warns instead, which is for exploratory runs rather than for a pipeline feeding a SIEM.
    time_unit : str, default = "ns"
        Unit of the numeric timestamps on the way in.
    """

    def __init__(self, c: Config, sourcetype: str, require_columns: bool = True, time_unit: str = "ns"):
        super().__init__(c)

        self._sourcetype = lookup_sourcetype(sourcetype)
        self._require_columns = require_columns
        self._time_unit = time_unit

        for column in self._sourcetype.time_columns:
            self._needed_columns[column] = TypeId.STRING

    @property
    def name(self) -> str:
        """Stage name."""
        return "siem-wire"

    @property
    def sourcetype(self) -> str:
        """The `props.conf` stanza this stage renders for."""
        return self._sourcetype.name

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

    def _check_columns(self, present: typing.Iterable):
        present = set(present)
        expected = (self._sourcetype.time_column, ) + tuple(self._sourcetype.required_columns)
        missing = [column for column in expected if column not in present]

        if (len(missing) == 0):
            return

        message = f"Records for sourcetype {self._sourcetype.name!r} are missing {', '.join(missing)}."

        if (self._sourcetype.time_column in missing):
            # Worth saying out loud only when it is the anchor that is absent, because that failure is silent:
            # the SIEM raises nothing, it just stamps the record at index time.
            message += (f" {self._sourcetype.time_column!r} is the column this sourcetype anchors on; a record "
                        f"without it is stamped at index time by the SIEM rather than by its own event time.")

        if (self._require_columns):
            raise KeyError(message)

        logger.warning("%s", message)

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Render this sourcetype's timestamps in place and return the message.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming batch.

        Returns
        -------
        The same message, with every one of the sourcetype's time columns that the batch carries replaced by its
        wire rendering. Nulls stay null: a binding that has not closed has no end, and inventing one would be a
        worse answer than the SIEM seeing an absent field.

        Raises
        ------
        KeyError
            If `require_columns` is set and the batch lacks a column the sourcetype requires.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            self._check_columns(df.columns)

            for column in self._sourcetype.time_columns:
                if (column not in df.columns):
                    continue

                rendered = render_event_time_series(to_host_list(df, column), time_unit=self._time_unit)
                assign_str_column(df, column, rendered)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
