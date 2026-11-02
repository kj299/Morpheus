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
Control 13's seventh check: stop every pipeline halfway through its input, restore it, finish, and get the golden.

Every composition in this package builds its segments with `ResumableLinearPipeline` rather than Morpheus's
`LinearPipeline`. Outside `stop_and_resume()` the two are the same. Inside it, `run()` does what a deployment that
restarts does:

1. Every stage is copied as it is added, before the pipeline has built or run anything, which is what a fresh
   process constructs from the same arguments.
2. The source's rows are cut at the halfway row, and the first half runs with every stateful stage told that the end
   of the stream is a stop for a checkpoint, so open windows and bindings are kept rather than flushed.
3. The stopped stages are snapshotted to bytes, and only the bytes cross to the second half, so nothing a stage
   holds can reach the resumed one except through `to_state`.
4. The copies are restored from the bytes, a second pipeline runs them over the second half, and its output is
   appended to the first sink's, so the caller reads one stream as it would from one run.

The check is the uninterrupted golden. A stage whose snapshot leaves anything out resumes with less history than it
stopped with, and the second half differs. The cut falls at an arbitrary row rather than a frame boundary, which is
safe because check 5 already shows a pipeline's output does not depend on where its batches are divided.
"""

import contextlib
import copy
import typing

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.determinism_envelope import COLD_START
from morpheus.utils.stage_checkpoint import CheckpointDirectory
from morpheus.utils.stage_checkpoint import restore_stages
from morpheus.utils.stage_checkpoint import snapshot_digest
from morpheus.utils.stage_checkpoint import snapshot_stages
from morpheus.utils.stage_checkpoint import suspend_end_of_stream
from morpheus.utils.tracker_state import from_bytes
from morpheus.utils.tracker_state import to_bytes

_ACTIVE: list = []
"""Non-empty while `stop_and_resume()` is in effect. Holds the per-run record the context returns."""

_PERSISTED: list = []
"""Non-empty while `persisted()` is in effect: what the previous process left, and what this one will leave."""

PERSISTED_FORMAT = 1
"""Version of the checkpoint a composition writes: one stage snapshot per linear segment, in the order run."""


class ResumeRecord:
    """What a stop-and-resume run did, so a test can show the check was not vacuous."""

    def __init__(self):
        self.segments = 0
        """Linear segments that were stopped and resumed."""

        self.stateful_segments = 0
        """Of those, the segments holding at least one stage with memory."""

        self.snapshot_bytes = 0
        """Total size of the snapshots that crossed from a first half to a second."""


@contextlib.contextmanager
def stop_and_resume() -> typing.Iterator[ResumeRecord]:
    """Within this block, every `ResumableLinearPipeline.run()` stops halfway, restores and finishes."""
    record = ResumeRecord()
    _ACTIVE.append(record)

    try:
        yield record
    finally:
        _ACTIVE.pop()


class PersistRecord:
    """A composition's checkpoint across processes: the one restored from, and the segments this run snapshots."""

    def __init__(self, previous: typing.Optional[dict], final: bool = False):
        self.previous = previous
        """The checkpoint the last process wrote, or `None` on a cold start."""

        self.final = final
        """Whether this run reaches the end of the data, so open windows and bindings are flushed, not kept."""

        self.resumed_from = COLD_START if previous is None else snapshot_digest(previous)
        """What every row scored in this process carries as `resumed_from_state`."""

        self.segments: list = []
        """One stage snapshot per linear segment run, in order."""

    def restore(self, stages: list) -> None:
        """Hand the next segment its predecessor's memory, if the previous process ran that far."""
        position = len(self.segments)

        if (self.previous is not None and position < len(self.previous["segments"])):
            restore_stages(stages, self.previous["segments"][position])


@contextlib.contextmanager
def persisted(directory: str, final: bool = False) -> typing.Iterator[PersistRecord]:
    """
    Within this block, every `ResumableLinearPipeline.run()` restores its stages from the last checkpoint in
    `directory`, stops without flushing, and contributes its snapshot to the checkpoint written when the block ends.

    The segments of a composition are matched by the order they run in, and each segment's stages by position and
    class, so a checkpoint written by one composition refuses to restore into another. A block that raises writes
    nothing, so the previous checkpoint stays the latest. With `final`, the run is the last over its data: open
    windows are sealed and open bindings drained as a replay would, and the checkpoint is not written.
    """
    store = CheckpointDirectory(directory)
    record = PersistRecord(store.latest(), final=final)
    _PERSISTED.append(record)

    try:
        yield record
    finally:
        _PERSISTED.pop()

    if (not final):
        store.write({"format": PERSISTED_FORMAT, "resumed_from": record.resumed_from, "segments": record.segments})


def resumed_from() -> str:
    """The digest of the checkpoint this process resumed from, or `COLD_START`; what the envelope stamps."""
    return _PERSISTED[-1].resumed_from if len(_PERSISTED) > 0 else COLD_START


def _split(frames: list, at_row: int) -> tuple[list, list]:
    (first, second) = ([], [])
    seen = 0

    for frame in frames:
        rows = len(frame)

        if (seen + rows <= at_row):
            first.append(frame)
        elif (seen >= at_row):
            second.append(frame)
        else:
            cut = at_row - seen
            first.append(frame.iloc[:cut].reset_index(drop=True))
            second.append(frame.iloc[cut:].reset_index(drop=True))

        seen += rows

    return (first, second)


class ResumableLinearPipeline(LinearPipeline):
    """
    `LinearPipeline`, which under `stop_and_resume()` runs as two processes would: one that stops halfway and one
    that resumes from its snapshot.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._resume_config = c
        self._pristine: list = []

    def _run_persisted(self, record: PersistRecord):
        middle = self._linear_stages[1:-1]
        record.restore(middle)
        suspend_end_of_stream(middle, suspended=not record.final)
        super().run()
        record.segments.append(snapshot_stages(middle))

    def add_stage(self, stage):  # pylint: disable=arguments-differ
        if (len(_ACTIVE) > 0 and not isinstance(stage, InMemorySinkStage)):
            # Copied before it is added, so the copy has never been wired, built or shown a row.
            self._pristine.append(copy.deepcopy(stage))

        return super().add_stage(stage)

    def run(self):
        if (len(_ACTIVE) == 0 and len(_PERSISTED) > 0):
            return self._run_persisted(_PERSISTED[-1])

        if (len(_ACTIVE) == 0):
            return super().run()

        record = _ACTIVE[-1]
        source = self._linear_stages[0]
        sink = self._linear_stages[-1]
        middle = self._linear_stages[1:-1]

        if (not isinstance(source, InMemorySourceStage) or not isinstance(sink, InMemorySinkStage)):
            raise TypeError("stop_and_resume() handles a pipeline from an in-memory source to an in-memory sink")

        frames = list(source._dataframes)  # pylint: disable=protected-access
        total = sum(len(frame) for frame in frames)
        (first, second) = _split(frames, total // 2)

        source._dataframes = first  # pylint: disable=protected-access
        suspend_end_of_stream(middle)
        super().run()

        snapshot = to_bytes(snapshot_stages(middle))
        record.segments += 1
        record.stateful_segments += 1 if len(from_bytes(snapshot)["stages"]) > 0 else 0
        record.snapshot_bytes += len(snapshot)

        resumed = self._pristine
        restore_stages(resumed, from_bytes(snapshot))

        pipe = LinearPipeline(self._resume_config)
        pipe.set_source(InMemorySourceStage(self._resume_config, dataframes=second))

        for stage in resumed:
            pipe.add_stage(stage)

        second_sink = pipe.add_stage(InMemorySinkStage(self._resume_config))
        pipe.run()

        for message in second_sink.get_messages():
            sink._append_message(message)  # pylint: disable=protected-access

        return None
