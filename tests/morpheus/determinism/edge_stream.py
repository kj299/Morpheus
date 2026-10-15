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
A corpus's `morpheus:edge` records: its sealed rows through `EdgeEmitStage`, as a branch of the sealed stream.

`EdgeEmitStage` replaces each batch with its edges, so a deployment places it on its own branch beside the sink the
scored rows go to. The composed corpora run their layers as separate pipelines and keep the scored rows as their
golden output, so the branch is run here, over every class's sealed rows at once, in the total order the stages
would have seen them in -- which is what lets a connection seen at layers 3, 4 and 6 by three pipelines be joined
the way one stream through one stage would join it.

Edges are not part of any golden file. They are a function of the golden rows, and the harnesses assert them.
"""

import pandas as pd

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.edge_emit_stage import EDGE_COLUMNS
from morpheus.stages.lineage.edge_emit_stage import EdgeEmitStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS


def emit(config: Config, rows: pd.DataFrame, **stage_kwargs) -> pd.DataFrame:
    """
    The edges `EdgeEmitStage` writes for these sealed rows.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    rows : `pandas.DataFrame`
        Sealed rows, of every class the edges join. Columns a class does not carry are null on its rows.
    **stage_kwargs
        Forwarded to `EdgeEmitStage`.

    Returns
    -------
    `pandas.DataFrame`
        One row per edge, in `EDGE_COLUMNS` order.
    """
    from host_frame import to_host_frame  # pylint: disable=import-outside-toplevel

    rows = rows[rows["event_uid"].notna()]

    if (len(rows) == 0):
        return pd.DataFrame(columns=list(EDGE_COLUMNS))

    ordered = rows.sort_values(list(DEFAULT_ORDER_COLUMNS), kind="stable").reset_index(drop=True)

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[ordered]))
    pipe.add_stage(EdgeEmitStage(config, **stage_kwargs))
    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    frames = [to_host_frame(message.copy_dataframe()) for message in sink.get_messages()]
    edges = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(EDGE_COLUMNS))

    return edges[list(EDGE_COLUMNS)]
