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
The two helpers every composed corpus module needs and none of them should own: the configuration a corpus is run
under, and the rule that turns what a sink received into one frame.

Each corpus module used to carry its own copy of both. The copies mostly agreed in substance but not in text, and
a copy that drifts is a corpus whose golden was produced under a different configuration or a different
concatenation than its neighbors, which no test would notice until a golden moved. The two places where copies
differed in substance are kept as explicit parameters rather than unified away: `use_cpp` (only the lineage corpus
exposed it) and `drop_empty` (only the host score corpus dropped batches with no rows before concatenating, and its
golden was generated that way).
"""

import pandas as pd

from morpheus.compositions.host_frame import to_host_frame
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage


def build_pipeline_config(execution_mode=None, use_cpp: bool = False) -> Config:
    """
    A pipeline configuration, defaulting to CPU mode and importable without a GPU.

    Parameters
    ----------
    execution_mode : `morpheus.config.ExecutionMode`, optional
        Mode to build for. Defaults to CPU, which is what every golden file is an artifact of. The parameter exists
        so the same corpus and the same golden can be driven in GPU mode and the two compared, since the per-stage
        `gpu_mode` variants are unit tests and none of them composes a pipeline. Resolved inside the function
        rather than as a default argument so that importing a corpus module still does not require a GPU.
    use_cpp : bool, default = False
        Whether stages may use their C++ implementations. Every corpus runs the Python implementations; only the
        lineage corpus has ever exposed the switch, and no caller sets it.
    """
    from morpheus.config import CppConfig  # pylint: disable=import-outside-toplevel
    from morpheus.config import ExecutionMode  # pylint: disable=import-outside-toplevel

    CppConfig.set_should_use_cpp(use_cpp)

    config = Config()
    config.execution_mode = ExecutionMode.CPU if execution_mode is None else execution_mode

    return config


def _collect(sink: InMemorySinkStage, drop_empty: bool = False) -> pd.DataFrame:
    """
    Everything the sink received, as one host frame.

    Parameters
    ----------
    sink : `morpheus.stages.output.in_memory_sink_stage.InMemorySinkStage`
        The sink whose messages to collect. A `ControlMessage` is unwrapped to its payload; anything else is taken
        to be a `MessageMeta` already.
    drop_empty : bool, default = False
        Drop batches with no rows before concatenating. An empty batch still has columns, and pandas lets their
        dtypes take part in the concatenation's dtype inference, so the switch can change the dtypes of the result.
        Only the host score corpus sets it, kept from its own copy of this helper; on its current corpus the
        result is the same either way. Every other corpus concatenates every batch.
    """
    frames = []

    for message in sink.get_messages():
        meta = message.payload() if isinstance(message, ControlMessage) else message
        # Integer columns are carried as nullable integers in both modes before anything is joined, so the
        # concatenation cannot widen them to float in one mode and not the other; see `host_frame`.
        frames.append(to_host_frame(meta.copy_dataframe()))

    if drop_empty:
        frames = [frame for frame in frames if len(frame) > 0]

    if (len(frames) == 0):
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
