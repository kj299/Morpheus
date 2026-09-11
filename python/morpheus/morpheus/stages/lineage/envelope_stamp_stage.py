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
Stamps the two envelope fields that identify what a record is about and which layer it came from.

Part 2 requires `entity_key` and a telemetry class on every record from every layer, and two shipped searches
group by `osi_layer` and `entity_key`. Neither was fully supplied: nothing emitted `osi_layer` at all, and
`entity_key` reached only the layer 1 stages that happen to compose it for their own use. Splunk's `stats by`
drops a row whose grouping field is absent, so the behavior summary and the chain assembly returned nothing --
not because the estate was quiet but because the rows could not be grouped.

That stayed invisible for as long as it did because those searches were empty for a second reason too: nothing
produced `max_abs_z` either, so an empty result had an explanation nobody had to look past. Giving the scoring
path a producer removed the first explanation and left the second one visible, which is a good argument for
fixing the more obvious gap first.

**The entity key is per telemetry class and this stage does not guess it.** Part 2 names a different subject for
each: a port at layer 1, a MAC at layer 2, a principal at layer 5. The caller says which columns compose it, and
a row missing any of them carries no key rather than a fabricated one -- `compose_key` refuses a partial
identity for the same reason it always has, since `None:sw1:Gi1/0/1` would pool every siteless row under one
invented site.

**An existing key is left alone.** The TC-1 stages compose `entity_key` themselves because they need it to key
their own state, and a stage that overwrote it would be able to disagree with the state those stages keep. When
the column is already there this stage stamps only the layer, unless it is explicitly asked to overwrite.
"""

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
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.entity_key import compose_key
from morpheus.utils.entity_key import normalize_text

logger = logging.getLogger(__name__)

ENTITY_KEY_COLUMN = "entity_key"
OSI_LAYER_COLUMN = "osi_layer"

MIN_LAYER = 0
MAX_LAYER = 7
"""Zero is TC-0, the identity and asset context, which is not an OSI layer but is stamped like one."""


@register_stage("envelope-stamp", modes=[])
class EnvelopeStampStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Stamp `osi_layer` and, where it is missing, `entity_key`.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    osi_layer : int
        The layer these records came from, 0 through 7. Constant for the class, because a class is a layer.
    entity_columns : list of str
        The columns composing this class's behavioral subject, in order. Layer 1 composes a port from
        `site_id`, `device_id` and `port_id`; layer 5 names a single column, `user_principal`.
    overwrite : bool, default = False
        Replace an `entity_key` that is already present. Off, because the TC-1 stages compose their own and key
        their state on it; overwriting would let this stage disagree with state those stages keep.

    Raises
    ------
    ValueError
        If the layer is outside 0 through 7, or no entity columns are named.
    """

    def __init__(self, c: Config, osi_layer: int, entity_columns: list[str], overwrite: bool = False):
        super().__init__(c)

        if (not isinstance(osi_layer, int) or isinstance(osi_layer, bool)):
            raise ValueError(f"osi_layer must be an integer, received {osi_layer!r}")

        if (not MIN_LAYER <= osi_layer <= MAX_LAYER):
            raise ValueError(f"osi_layer must be between {MIN_LAYER} and {MAX_LAYER}, received {osi_layer}")

        if (not entity_columns):
            raise ValueError("entity_columns must name at least one column; a record with no subject cannot be "
                             "grouped, which is the defect this stage exists to fix")

        self._osi_layer = osi_layer
        self._entity_columns = list(entity_columns)
        self._overwrite = overwrite

        self._needed_columns[OSI_LAYER_COLUMN] = TypeId.INT64
        self._needed_columns[ENTITY_KEY_COLUMN] = TypeId.STRING

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "envelope-stamp"

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
        Stamp the layer on every row, and the entity key on every row that does not already carry one.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming records.

        Returns
        -------
        The same message, with the envelope fields attached.

        Raises
        ------
        KeyError
            If an entity column is absent and no `entity_key` is already present to fall back on.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            row_count = len(df)

            # Per row, and on the value rather than on the column's existence. A stage declares the columns it
            # writes through `_needed_columns`, and Morpheus creates them before the stage runs -- so this
            # column is always present by the time control arrives here, and a presence check would declare
            # every row already keyed and compose nothing. That is exactly what it did: layer 1 looked correct
            # because its own stages fill the column, and every other class shipped an empty string.
            existing = to_host_list(df, ENTITY_KEY_COLUMN) if ENTITY_KEY_COLUMN in df.columns else None
            missing = [name for name in self._entity_columns if name not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"EnvelopeStampStage requires columns {missing} to compose {ENTITY_KEY_COLUMN}. "
                               f"Available columns: {sorted(df.columns)}")

            parts = {name: to_host_list(df, name) for name in self._entity_columns}
            keys = []

            for position in range(row_count):
                held = normalize_text(existing[position]) if existing is not None else None

                if (held is not None and not self._overwrite):
                    keys.append(held)
                    continue

                keys.append(compose_key([parts[name][position] for name in self._entity_columns]))

            keyless = sum(1 for key in keys if key is None)

            assign_str_column(df, ENTITY_KEY_COLUMN, keys)
            assign_nullable_int_column(df, OSI_LAYER_COLUMN, [self._osi_layer] * row_count)

            if (keyless > 0):
                logger.warning(
                    "EnvelopeStampStage composed no %s for %d of %d rows: a part of the key was missing. They "
                    "carry no key rather than a partial one, and a search grouping by it will not return them.",
                    ENTITY_KEY_COLUMN,
                    keyless,
                    row_count)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
