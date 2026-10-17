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
A whole pipeline's memory as one snapshot, written when it stops and read when its successor starts.

**A snapshot is taken when every stage has stopped, not while one is still working.** The issue that asked for this
(#68) named the window seal as the moment. Inside a running segment that moment is not a consistent one: the stages
are separate nodes with queues between them, and when the seal stage publishes a window the stages upstream of it
may already be holding the next batch. A snapshot then would record trackers one batch ahead of the windows beside
them. So a runner stops the pipeline -- the source is exhausted, `suspend_end_of_stream` keeps every open window and
binding instead of flushing it -- and snapshots the quiescent stages. The snapshot is still named by the window seal:
`sealed_through_window_id` is the latest window published complete, and every window after it is in the snapshot.

**Stages are matched by position and class.** A snapshot written by one composition and read into another would hand
one stage's trackers to a different stage of the same class, so restoring refuses unless the stateful stages appear
in the same order with the same classes, and each stage refuses attributes it does not recognise.

**The digest is what a resumed row carries.** `snapshot_digest` is the SHA-256 of the snapshot's canonical bytes; a
runner passes it to the determinism envelope as `resumed_from`, so every row scored afterwards names the history it
was scored against.
"""

import os
import re
import tempfile
import typing

from morpheus.pipeline.checkpoint_mixin import CheckpointMixin
from morpheus.utils.tracker_state import StateError
from morpheus.utils.tracker_state import from_bytes
from morpheus.utils.tracker_state import state_digest
from morpheus.utils.tracker_state import to_bytes

SNAPSHOT_FORMAT = 1
"""Version of the pipeline snapshot layout."""

CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(?P<sequence>\d{8})\.json$")
"""File names a `CheckpointDirectory` writes and reads."""


def stateful_stages(stages: typing.Iterable) -> list[tuple[int, CheckpointMixin]]:
    """The stages that remember, with their positions in the composition."""
    return [(position, stage) for (position, stage) in enumerate(stages) if isinstance(stage, CheckpointMixin)]


def suspend_end_of_stream(stages: typing.Iterable, suspended: bool = True) -> None:
    """Make the coming end of the stream a stop for a checkpoint on every stage that would otherwise flush."""
    for (_, stage) in stateful_stages(stages):
        stage.suspend_end_of_stream(suspended)


def snapshot_stages(stages: typing.Sequence) -> dict:
    """
    Every stateful stage's memory, in composition order.

    Parameters
    ----------
    stages : sequence
        The composition's stages, source and sink included or not; only their order matters.

    Returns
    -------
    dict
        `{"format", "sealed_through_window_id", "stages"}`, JSON-safe.
    """
    sealed = [
        stage.sealed_through_window_id for stage in stages
        if getattr(stage, "sealed_through_window_id", None) is not None
    ]

    entries = [{
        "position": position, "state": stage.checkpoint_state()
    } for (position, stage) in stateful_stages(stages)]
    sealed_through = max(sealed) if len(sealed) > 0 else None

    return {"format": SNAPSHOT_FORMAT, "sealed_through_window_id": sealed_through, "stages": entries}


def restore_stages(stages: typing.Sequence, snapshot: dict) -> None:
    """
    Hand each stateful stage of a freshly built composition the memory its predecessor stopped with.

    Parameters
    ----------
    stages : sequence
        The same composition `snapshot_stages` was called on, built again from the same arguments.
    snapshot : dict
        What `snapshot_stages` returned.

    Raises
    ------
    StateError
        If the snapshot is of another format or another composition.
    """
    if (snapshot.get("format") != SNAPSHOT_FORMAT):
        raise StateError(f"Pipeline snapshot format {snapshot.get('format')!r} is not {SNAPSHOT_FORMAT}")

    present = stateful_stages(stages)
    expected = [(entry["position"], entry["state"]["stage"]) for entry in snapshot["stages"]]
    actual = [(position, f"{type(stage).__module__}:{type(stage).__qualname__}") for (position, stage) in present]

    if (expected != actual):
        raise StateError(f"The snapshot was taken of a different composition: it holds {expected}, and this one is "
                         f"{actual}")

    for ((_, stage), entry) in zip(present, snapshot["stages"]):
        stage.restore_state(entry["state"])


def snapshot_digest(snapshot: dict) -> str:
    """SHA-256 of a pipeline snapshot's canonical bytes, as `sha256:<hex>`."""
    return state_digest(snapshot)


class CheckpointDirectory:
    """
    Snapshots of one composition, one file per stop, numbered in the order they were written.

    Each file is written to a temporary name and renamed into place, so a process killed mid-write leaves the
    previous checkpoint as the latest rather than a truncated one.

    Parameters
    ----------
    path : str
        The directory. Created if absent.
    """

    def __init__(self, path: str):
        self._path = path
        os.makedirs(path, exist_ok=True)

    @property
    def path(self) -> str:
        """The directory."""
        return self._path

    def _sequences(self) -> list[int]:
        found = []

        for name in os.listdir(self._path):
            match = CHECKPOINT_PATTERN.match(name)

            if (match is not None):
                found.append(int(match.group("sequence")))

        return sorted(found)

    def write(self, snapshot: dict) -> str:
        """
        Write a snapshot as the next checkpoint.

        Returns
        -------
        str
            The checkpoint's path.
        """
        sequences = self._sequences()
        sequence = (sequences[-1] + 1) if len(sequences) > 0 else 0
        target = os.path.join(self._path, f"checkpoint-{sequence:08d}.json")

        (handle, temporary) = tempfile.mkstemp(dir=self._path, prefix=".checkpoint-", suffix=".tmp")

        with os.fdopen(handle, "wb") as out:
            out.write(to_bytes(snapshot))

        os.replace(temporary, target)

        return target

    def latest(self) -> typing.Optional[dict]:
        """The most recently written snapshot, or `None` if the directory holds none."""
        sequences = self._sequences()

        if (len(sequences) == 0):
            return None

        with open(os.path.join(self._path, f"checkpoint-{sequences[-1]:08d}.json"), "rb") as source:
            return from_bytes(source.read())
