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
Reduce each sealed window to one row per entity: the feature frame a per-entity model scores.

A host is described by several layers and scored by none of them. Its fan-out step and reach are on its flows, its
TLS client fingerprints on its handshakes, its novel process pairs on its EDR records -- each a column on a row of
its own class, keyed by the `host_key` each class resolved. A model of the host needs those as one vector per
host per window, and no row of any class is that vector. This stage builds it.

It sits behind a `WindowSealStage` that merges the classes' rows and seals them by the hour, and it reads each
sealed window as one message, which is what the sealer emits. Every row of the window with an entity is grouped
by that entity, and each group becomes one row whose features are named reductions of the group's columns:

- `max`, `min`, `mean`, `sum` over the values present;
- `last`, the value on the group's last row that carries one, in the window's order;
- `count_true`, how many rows say true -- a count of novel process pairs, from rows that each say whether theirs
  was.

**A feature no row of the window carries is null, not zero.** A host with flows and no EDR records in an hour has
no novel process count for that hour; it does not have a count of zero, and a model told it had zero would score
the absence of an agent as a quiet hour. `TC5ScoreStage` leaves a row with a gap unscored, which is the claim the
data supports.

**Only complete rows are reduced.** A row the sealer marks late, or not part of a complete window, is counted and
left out: the window it belongs to has already been published, and a frame rebuilt from it would be a second
answer for the same hour. A late row is the sealer's to route.

Each output row carries the window's own columns (`window_id`, its bounds, `revision`, `sealed_by`, `is_late`,
`window_complete`), `event_time` at the window's start, the window's chain identifier where the sealer stamped one
on the entity, how many rows were reduced into it, and the columns `carry_columns` names, taken from the group's
first row that has a value -- the peer group a shared model is resolved from, the tenant. It also carries an
envelope of its own: this stage is the collector of the frame, so `collector_id` and `schema_version` are its
parameters, `origin_hash` is the entity and the window, and `collector_seq` numbers the rows it has written, carried
across a restart by its checkpoint. `event_uid` is the digest of those, so a frame row has an identity downstream
stages can stamp, seal and order.
"""

import hashlib
import logging
import math
import typing

import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.checkpoint_mixin import CheckpointMixin
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.column_assign import to_host_frame
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.type_utils import is_cudf_type

logger = logging.getLogger(__name__)

REDUCTIONS = ("max", "min", "mean", "sum", "last", "count_true")
"""What a feature may be: a named reduction of one column over an entity's rows in the window."""

WINDOW_COLUMNS = ("window_id", "window_start_ns", "window_end_ns", "revision", "sealed_by")
"""The sealer's columns a frame row keeps, from its window."""

ROWS_COLUMN = "frame_rows"
"""How many of the window's rows were reduced into the frame row."""


def _present(value: typing.Any) -> bool:
    if (value is None or value is pd.NA):
        return False

    try:
        return not (isinstance(value, float) and math.isnan(value))
    except TypeError:
        return True


def _truthy(value: typing.Any) -> bool:
    return _present(value) and (value is True or str(value).strip().lower() in ("true", "1"))


def reduce_values(reduction: str, values: list) -> typing.Any:
    """
    One feature's value from the values of its column on an entity's rows, in window order.

    Parameters
    ----------
    reduction : str
        One of `REDUCTIONS`.
    values : list
        The column's value on each of the entity's rows. Nulls are not values.

    Returns
    -------
    The reduced value, or `None` where no row carries one.
    """
    if (reduction == "count_true"):
        present = [value for value in values if _present(value)]

        return None if len(present) == 0 else sum(1 for value in present if _truthy(value))

    present = [float(value) for value in values if _present(value) and not isinstance(value, str)]

    return None if len(present) == 0 else _REDUCERS[reduction](present)


_REDUCERS = {
    "max": max,
    "min": min,
    "sum": sum,
    "last": lambda present: present[-1],
    "mean": lambda present: sum(present) / len(present),
}


@register_stage("entity-frame")
class EntityFrameStage(CheckpointMixin, GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Reduce each sealed window's rows to one feature row per entity.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    features : dict
        Output column to `[source column, reduction]`, the reduction one of `max`, `min`, `mean`, `sum`, `last` or
        `count_true`.
    entity_column : str, default = "host_key"
        What a frame row is about. Rows with no entity are counted and left out.
    carry_columns : list of str, default = None
        Columns copied onto the frame row from the entity's first row that has a value.
    collector_id : str, default = "entity-frame"
        The frame's collector, as its envelope names it.
    schema_version : str, default = "entity_frame.v1"
        The frame's schema, as its envelope names it.
    first_seq : int, default = 0
        The `collector_seq` of the first frame row a cold start writes.
    """

    _checkpoint_attributes = ("_next_seq", )

    def __init__(self,
                 c: Config,
                 features: dict,
                 entity_column: str = "host_key",
                 carry_columns: list[str] = None,
                 collector_id: str = "entity-frame",
                 schema_version: str = "entity_frame.v1",
                 first_seq: int = 0):
        super().__init__(c)

        if (not features):
            raise ValueError("features must name at least one output column")

        self._features = {}

        for (name, spec) in features.items():
            if (not isinstance(spec, (list, tuple)) or len(spec) != 2 or spec[1] not in REDUCTIONS):
                raise ValueError(f"feature {name!r} is {spec!r}; each feature is [source column, reduction] with "
                                 f"the reduction one of {list(REDUCTIONS)}")

            self._features[name] = (spec[0], spec[1])

        if (first_seq < 0):
            raise ValueError(f"first_seq must not be negative, received {first_seq}")

        self._entity_column = entity_column
        self._carry_columns = list(carry_columns or [])
        self._collector_id = collector_id
        self._schema_version = schema_version
        self._next_seq = first_seq

    @property
    def name(self) -> str:
        """Stage name."""
        return "entity-frame"

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

    @property
    def output_columns(self) -> list[str]:
        """The columns every frame row carries, in order."""
        return ([self._entity_column] + [column for column in self._carry_columns if column != self._entity_column] +
                list(self._features) + [
                    ROWS_COLUMN,
                    "event_time",
                    *WINDOW_COLUMNS,
                    "is_late",
                    "window_complete",
                    "lineage_id",
                    "collector_id",
                    "schema_version",
                    "origin_hash",
                    "collector_seq",
                    "event_uid",
                ])

    def reduce(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        The frame rows for one sealed window, numbered after the last window's.

        Parameters
        ----------
        frame : `pandas.DataFrame`
            One window's rows, as the sealer emits them.

        Returns
        -------
        `pandas.DataFrame`
            One row per entity, entities in sorted order, with exactly `output_columns`.

        Raises
        ------
        KeyError
            If the entity column, the window columns, or a feature's source column is absent. A merged stream
            carries every class's columns, null on the rows of the classes that do not write them.
        """
        required = [self._entity_column, *WINDOW_COLUMNS, *sorted({source for (source, _) in self._features.values()})]
        missing = [column for column in required if column not in frame.columns]

        if (len(missing) > 0):
            raise KeyError(f"EntityFrameStage requires columns {missing}; place it directly behind a "
                           f"WindowSealStage. Available columns: {sorted(frame.columns)}")

        groups: dict = {}
        skipped = 0

        for record in frame.to_dict("records"):
            complete = record.get("window_complete")
            entity = normalize_text(record.get(self._entity_column))

            if (_truthy(record.get("is_late")) or (_present(complete) and not _truthy(complete)) or entity is None):
                skipped += 1
                continue

            groups.setdefault(entity, []).append(record)

        if (skipped > 0):
            logger.debug("EntityFrameStage left %d late, incomplete or entity-less rows out of a window", skipped)

        rows = []

        for entity in sorted(groups):
            members = groups[entity]
            first = members[0]
            row = {self._entity_column: entity}

            for column in self._carry_columns:
                if (column != self._entity_column):
                    row[column] = next((member.get(column) for member in members if _present(member.get(column))), None)

            for (name, (source, reduction)) in self._features.items():
                row[name] = reduce_values(reduction, [member.get(source) for member in members])

            row[ROWS_COLUMN] = len(members)
            row["event_time"] = int(first["window_start_ns"])

            for column in WINDOW_COLUMNS:
                row[column] = first[column]

            row["is_late"] = False
            row["window_complete"] = True
            lineages = sorted({str(member["lineage_id"]) for member in members if _present(member.get("lineage_id"))})
            row["lineage_id"] = lineages[0] if len(lineages) == 1 else None
            row["collector_id"] = self._collector_id
            row["schema_version"] = self._schema_version
            row["origin_hash"] = f"{entity}@{int(first['window_id'])}:{int(first['revision'])}"
            row["collector_seq"] = self._next_seq
            row["event_uid"] = hashlib.sha256(
                f"{self._collector_id}|{self._schema_version}|{row['origin_hash']}".encode()).hexdigest()
            rows.append(row)
            self._next_seq += 1

        out = pd.DataFrame(rows, columns=self.output_columns)

        for column in ("event_time",
                       "window_id",
                       "window_start_ns",
                       "window_end_ns",
                       "revision",
                       "collector_seq",
                       ROWS_COLUMN):
            out[column] = out[column].astype("int64")

        for (name, (_, reduction)) in self._features.items():
            out[name] = out[name].astype("Int64" if reduction == "count_true" else "float64")

        return out

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Replace a sealed window's rows with its frame rows.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            One sealed window.

        Returns
        -------
        A message of the same type holding the frame rows, or `None` where the window gave none.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return None

        frame = meta.copy_dataframe()
        reduced = self.reduce(to_host_frame(frame))

        if (len(reduced) == 0):
            return None

        if (is_cudf_type(frame)):
            import cudf  # pylint: disable=import-outside-toplevel

            reduced = cudf.from_pandas(reduced)

        out = MessageMeta(reduced)

        if (isinstance(message, ControlMessage)):
            message.payload(out)
            return message

        return out

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data), ops.filter(lambda out: out is not None))
        builder.make_edge(input_node, node)

        return node
