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
Control 4 as a composition: one scoring branch per shard, every entity on the same branch on every run.

`RouterStage` routes whole messages, and a source hands over frames holding many entities, so sharding needs two
things the guide's sketch leaves implicit: a stage that cuts each frame into one message per shard, and a key
function that reads a message's shard off it. `ShardSplitStage` is the first, and `shard_key_fn` the second. Both
decide with `stable_shard`, so the assignment is the same on every machine and every restart, which is what
makes a sharded run comparable to an unsharded one at all.

A row whose key is null has no per-entity state to keep together and `stable_shard` gives it no shard; it goes
to the first branch, deliberately, so it is scored once rather than dropped.
"""

import typing

import mrc
import numpy as np
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.sharding import shard_names
from morpheus.utils.sharding import stable_shard

NULL_KEY_SHARD = 0
"""Where a row with no entity key is sent."""


def _shard_of(value: typing.Any, shards: int) -> int:
    if (value is None or (isinstance(value, float) and np.isnan(value))):
        return NULL_KEY_SHARD

    index = stable_shard(str(value), shards)

    return NULL_KEY_SHARD if index is None else index


def _keys_of(meta: MessageMeta, column: str) -> list:
    frame = meta.copy_dataframe()

    if (hasattr(frame, "to_pandas")):
        frame = frame.to_pandas()

    return frame[column].tolist()


class ShardSplitStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Cut each frame into one message per shard present in it, rows in their original order.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    key_column : str
        The entity key the shards are decided on.
    shards : int
        How many branches follow.
    """

    def __init__(self, c: Config, key_column: str, shards: int):
        super().__init__(c)

        if (shards <= 0):
            raise ValueError(f"shards must be positive, received {shards}")

        self._key_column = key_column
        self._shards = shards

    @property
    def name(self) -> str:
        return "shard-split"

    def accepted_types(self) -> tuple:
        return (ControlMessage, MessageMeta)

    def supports_cpp_node(self) -> bool:
        return False

    def split(self, message: typing.Union[ControlMessage, MessageMeta]) -> list:
        """One message per shard with rows in it, in shard order; a message of the input's type each."""
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return []

        if (self._key_column not in meta.get_column_names()):
            raise KeyError(f"shard key column {self._key_column!r} is not in the frame")

        assigned = np.array([_shard_of(value, self._shards) for value in _keys_of(meta, self._key_column)])
        frame = meta.copy_dataframe()
        out = []

        for shard in sorted(set(assigned.tolist())):
            part = MessageMeta(frame.iloc[np.flatnonzero(assigned == shard)].reset_index(drop=True))

            if (isinstance(message, ControlMessage)):
                piece = ControlMessage(message)
                piece.payload(part)
                out.append(piece)
            else:
                out.append(part)

        return out

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.split), ops.flatten())
        builder.make_edge(input_node, node)

        return node


def shard_key_fn(key_column: str, shards: int) -> typing.Callable[[object], str]:
    """
    The `RouterStage` key function for messages `ShardSplitStage` cut: the shard of the first row names them all.

    Parameters
    ----------
    key_column : str
        The entity key the shards were decided on.
    shards : int
        How many branches follow.

    Returns
    -------
    callable
        Message to port name.
    """
    names = shard_names(shards)

    def key_fn(message) -> str:
        meta = message.payload() if isinstance(message, ControlMessage) else message

        return names[_shard_of(_keys_of(meta, key_column)[0], shards)]

    return key_fn
