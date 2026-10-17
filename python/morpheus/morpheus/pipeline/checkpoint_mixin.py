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
Mixin for stages that remember: what they hold leaves as a snapshot and comes back into a fresh instance.

A stage lists the attributes that are its memory in `_checkpoint_attributes` -- the trackers, the clock, the open
windows -- and nothing else. Configuration is not state: a fresh instance is built from the same arguments and is
handed only what the stream taught the old one. `tests/morpheus/stages/test_stage_checkpoint.py` drives every stage
and fails if an attribute that changed is not listed, so the list cannot quietly fall behind the code.

A stop for a checkpoint is not the end of the stream. A window still open, or a binding that has not closed, is
published by `on_completed` when a replay ends; a stage stopped so that its successor can resume must keep them
instead, or the successor would republish a window that was flushed early and miss a binding's real end.
`suspend_end_of_stream` is that difference.
"""

import typing

from morpheus.utils.tracker_state import STATE_FORMAT
from morpheus.utils.tracker_state import StateError
from morpheus.utils.tracker_state import decode
from morpheus.utils.tracker_state import encode


def _stage_name(stage: typing.Any) -> str:
    cls = type(stage)

    return f"{cls.__module__}:{cls.__qualname__}"


class CheckpointMixin:
    """A stage whose memory can be snapshotted and restored."""

    _checkpoint_attributes: typing.ClassVar[tuple[str, ...]] = ()
    """The attributes that are this stage's memory. Everything else is rebuilt from the constructor's arguments."""

    @property
    def end_of_stream_suspended(self) -> bool:
        """Whether the stream's end is a stop for a checkpoint, so open windows and bindings are kept, not flushed."""
        return getattr(self, "_end_of_stream_suspended", False)

    def suspend_end_of_stream(self, suspended: bool = True) -> None:
        """
        Treat the coming end of the stream as a stop for a checkpoint rather than the end of the data.

        Parameters
        ----------
        suspended : bool, default = True
            `True` keeps whatever `on_completed` would have flushed; `False` restores the default.
        """
        self._end_of_stream_suspended = suspended

    def checkpoint_state(self) -> dict:
        """
        This stage's memory, as plain Python types.

        Returns
        -------
        dict
            `{"format", "stage", "attrs"}`, JSON-safe. Two instances that have seen the same rows give the same
            canonical bytes.
        """
        return {
            "format": STATE_FORMAT,
            "stage": _stage_name(self),
            "attrs": [[name, encode(getattr(self, name))] for name in self._checkpoint_attributes],
        }

    def restore_state(self, state: dict) -> None:
        """
        Continue from a snapshot taken by `checkpoint_state` on an instance of the same class.

        Parameters
        ----------
        state : dict
            What `checkpoint_state` returned, directly or after a JSON round trip.

        Raises
        ------
        StateError
            If the snapshot is of another format, another stage class, or a different set of attributes.
        """
        if (state.get("format") != STATE_FORMAT):
            raise StateError(f"Snapshot format {state.get('format')!r} is not {STATE_FORMAT}")

        if (state.get("stage") != _stage_name(self)):
            raise StateError(f"Snapshot is of {state.get('stage')!r}, not {_stage_name(self)!r}")

        names = [name for (name, _) in state["attrs"]]

        if (names != list(self._checkpoint_attributes)):
            raise StateError(f"Snapshot holds {names}, but {type(self).__name__} remembers "
                             f"{list(self._checkpoint_attributes)}; it was written by another version of the stage")

        for (name, value) in state["attrs"]:
            setattr(self, name, decode(value))
