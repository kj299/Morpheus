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
The universal envelope every corpus's collectors attach, and the two passes every composed pipeline runs over it.

Part 2's envelope is fourteen fields, and the corpora carried four of the collector's ten: `collector_id`,
`collector_seq`, `schema_version` and `origin_hash`. `attach` gives every record the rest -- when the sensor saw it,
the collector's clock source and last measured offset, its sampling policy, and the tenant it belongs to -- from one
table, so the twelve corpora agree on what the fields mean and a collector's clock is the same in every corpus it
appears in.

Each collector's offset is a few milliseconds either side of true, fixed by its name, because a disciplined clock is
never exactly right and a corpus whose every offset is zero would never exercise the comparison. The few collectors
whose clocks a corpus plants as wrong say so in their own corpus.

`validate` is the pass a deployment puts at the head of each class's segment: `EnvelopeValidateStage`, which stamps
`ingest_time` and takes a record whose clock cannot be trusted out of the stream. A corpus replays a delivery, so
`ingest_time` is the record's own `observed_time` plus a fixed delay rather than the wall clock -- a golden that
changed on every run would prove nothing -- and the quarantined records come back as their own class, which is what a
deployment sends to `morpheus:quarantine`. It runs once per class, ahead of every pipeline the class feeds, because
two of the telemetry corpus's classes each feed two pipelines and a record quarantined twice would be two records.

`health` is the pass behind the sealer: `PipelineHealthStage` over a corpus's sealed output, one record per window,
collector, class and tenant. It reads the finished result rather than running inside the pipeline because the
result is what a deployment's health branch would see, and it is kept out of the golden files because every harness
counts a class's rows and a health record is not a row of any class.
"""

import typing
import zlib

import pandas as pd
from resumable import ResumableLinearPipeline

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.envelope_validate_stage import EnvelopeValidateStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.pipeline_health_stage import PipelineHealthStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage

NS_PER_SECOND = 10**9

TENANT = "corp"
"""The tenant every corpus record belongs to unless its corpus names another."""

SAMPLING_POLICY = "full"
"""Every corpus keeps every record; a sampled feed would say `1:N`."""

INGEST_DELAY_SECONDS = 2
"""How long after the sensor saw a record the replayed delivery says it reached Morpheus."""

DEFAULT_CLOCK_SOURCE = "ntp_disciplined"

CLOCK_SOURCES = {
    "pcap-01": "ptp",
    "tls-inspect-01": "ptp",
}
"""The packet sensors and the inspection point are timestamped in hardware; everything else disciplines to NTP."""

OFFSET_SPREAD_MS = 20
"""A disciplined clock's last measured offset lies within this many milliseconds of true, either side."""

QUARANTINE_CLASS = "quarantine"
"""The class quarantined records are returned under, and what a deployment sends to `morpheus:quarantine`."""

HEALTH_CLASS = "health"
"""The class health records are returned under, and what a deployment sends to `morpheus:health`."""


def clock_offset_ms(collector: str) -> int:
    """A collector's last measured offset: a few milliseconds either side of true, fixed by its name."""
    return zlib.crc32(collector.encode("utf-8")) % (2 * OFFSET_SPREAD_MS + 1) - OFFSET_SPREAD_MS


def attach(frame: pd.DataFrame,
           tenant_id: typing.Optional[str] = TENANT,
           sampling_policy: str = SAMPLING_POLICY,
           clock_sources: typing.Optional[dict] = None,
           offsets: typing.Optional[dict] = None) -> pd.DataFrame:
    """
    Give every record the envelope fields its collector owes and the corpus had not yet written.

    Parameters
    ----------
    frame : `pandas.DataFrame`
        A class's records, carrying `collector_id` and `event_time`.
    tenant_id : str, default = "corp"
        The tenant, where the records do not already name one.
    sampling_policy : str, default = "full"
        The collector's sampling policy.
    clock_sources : dict, optional
        Clock source by collector, over the corpus-wide table.
    offsets : dict, optional
        Clock offset in milliseconds by collector, over the name-derived one.

    Returns
    -------
    `pandas.DataFrame`
        A copy with `observed_time`, `clock_source`, `clock_offset_ms`, `sampling_policy` and `tenant_id`, each kept
        where the corpus had already written it.
    """
    frame = frame.copy()
    collectors = [str(value) for value in frame["collector_id"]]
    sources = {**CLOCK_SOURCES, **(clock_sources or {})}
    measured = offsets or {}

    def keep(column: str, values: list) -> list:
        if (column not in frame.columns):
            return values

        return [value if not pd.isna(value) else default for (value, default) in zip(frame[column], values)]

    # The sensor saw the record when it happened unless the corpus says it saw it later.
    frame["observed_time"] = keep("observed_time", list(frame["event_time"]))
    frame["clock_source"] = keep("clock_source", [sources.get(name, DEFAULT_CLOCK_SOURCE) for name in collectors])
    frame["clock_offset_ms"] = keep("clock_offset_ms",
                                    [measured.get(name, clock_offset_ms(name)) for name in collectors])
    frame["sampling_policy"] = keep("sampling_policy", [sampling_policy] * len(frame))
    frame["tenant_id"] = keep("tenant_id", [tenant_id] * len(frame))

    return frame


def replay_ingest_clock(observed: typing.Sequence[typing.Optional[int]]) -> list:
    """A replay's ingest clock: each record reaches Morpheus `INGEST_DELAY_SECONDS` after the sensor saw it."""
    return [None if value is None else int(value) + INGEST_DELAY_SECONDS * NS_PER_SECOND for value in observed]


def _messages(sink: InMemorySinkStage) -> list:
    # Sibling module; imported here because this file's own directory is put on the path by whoever imports it.
    from host_frame import to_host_frame

    frames = []

    for message in sink.get_messages():
        meta = message.payload() if isinstance(message, ControlMessage) else message
        frames.append(to_host_frame(meta.copy_dataframe()))

    return frames


def validate(config: Config, batches: dict, id_columns: list) -> tuple[dict, pd.DataFrame]:
    """
    Stamp `ingest_time`, check the envelope, and take quarantined records out of every class's frames.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    batches : dict
        Per class, the source frames, in the split the caller chose. The split is kept.
    id_columns : list of str
        The columns `LineageStampStage` builds `event_uid` from, so a quarantined record keeps its identity.

    Returns
    -------
    tuple of (dict, `pandas.DataFrame`)
        The accepted frames per class, and every quarantined record tagged with the class it came from in
        `quarantined_from`. An empty frame when nothing was quarantined.
    """
    accepted: dict = {}
    held: list = []

    for (name, frames) in batches.items():
        quarantined: list = []
        pipe = ResumableLinearPipeline(config)
        pipe.set_source(InMemorySourceStage(config, dataframes=[frame.copy() for frame in frames]))
        pipe.add_stage(LineageStampStage(config, id_columns=id_columns))
        pipe.add_stage(EnvelopeValidateStage(config, ingest_clock=replay_ingest_clock,
                                             on_quarantine=quarantined.append))
        sink = pipe.add_stage(InMemorySinkStage(config))
        pipe.run()

        # A source frame every record of which was quarantined leaves an empty one behind; it is not a batch.
        accepted[name] = [frame for frame in _messages(sink) if len(frame) > 0]

        for frame in quarantined:
            frame = frame.copy()
            frame["quarantined_from"] = name
            held.append(frame)

    if (len(held) == 0):
        return (accepted, pd.DataFrame())

    return (accepted, pd.concat(held, ignore_index=True))


def health(config: Config,
           result: pd.DataFrame,
           period_seconds: int,
           min_buckets: int,
           classes: typing.Optional[typing.Sequence[str]] = None) -> pd.DataFrame:
    """
    One health record per window, collector, class and tenant, over a corpus's sealed output.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    result : `pandas.DataFrame`
        A corpus's `run_pipeline` output, its classes tagged in `telemetry_class`.
    period_seconds : int
        The corpus's sealing period, which is the health history's bucket.
    min_buckets : int
        Windows a history needs before its baseline is published.
    classes : list of str, optional
        The classes measured. Defaults to every sealed class; the quarantine class is never one, since its records
        reached no window.

    Returns
    -------
    `pandas.DataFrame`
        The health records, in window order.
    """
    sealed = result[result["window_id"].notna()] if "window_id" in result.columns else result.iloc[0:0]
    sealed = sealed[sealed["telemetry_class"] != QUARANTINE_CLASS]

    if (classes is not None):
        sealed = sealed[sealed["telemetry_class"].isin(list(classes))]

    if (len(sealed) == 0):
        return pd.DataFrame()

    frames = []

    # One class at a time, as each class's own segment would have its own health branch. The union carries every
    # class's columns, and a column another class writes would otherwise count as zero here rather than as nothing
    # this class can answer.
    for name in sorted(set(sealed["telemetry_class"])):
        rows = sealed[sealed["telemetry_class"] == name]
        # The order a sealer emits them in: window by window, so each history sees its windows in turn.
        rows = rows.sort_values(["window_id", "row_key"], kind="stable").reset_index(drop=True)
        rows = rows.dropna(axis=1, how="all")

        pipe = ResumableLinearPipeline(config)
        pipe.set_source(InMemorySourceStage(config, dataframes=[rows]))
        pipe.add_stage(PipelineHealthStage(config, period_seconds=period_seconds, min_buckets=min_buckets))
        sink = pipe.add_stage(InMemorySinkStage(config))
        pipe.run()

        frames.extend(_messages(sink))

    if (len(frames) == 0):
        return pd.DataFrame()

    records = pd.concat(frames, ignore_index=True)

    return records.sort_values(["window_id", "collector_id", "telemetry_class", "tenant_id"],
                               kind="stable").reset_index(drop=True)


STEP_MULTIPLE = 2
"""How many times its own peak a feed's row count must reach to be a platform step, as 'Platform baseline step'
states it."""

HEALTH_KEY = ["collector_id", "telemetry_class", "tenant_id"]


def platform_steps(records: pd.DataFrame, multiple: int = STEP_MULTIPLE) -> pd.DataFrame:
    """
    What 'Platform baseline step' returns over a set of health records: the record ending each held step.

    A step is two consecutive records of one collector, class and tenant, each at least `multiple` times the peak the
    first of them was measured against, with that peak mature. One window over its history is a burst; a feed whose
    volume changed holds there into the next window it reports. The order is the search's: by key, then by time.

    Parameters
    ----------
    records : `pandas.DataFrame`
        Health records, from `health`.
    multiple : int, default = 2
        The multiple of the peak.

    Returns
    -------
    `pandas.DataFrame`
        The second record of each held step, with the first's `window_id`, rows and peak beside it.
    """
    if (len(records) == 0):
        return records

    ordered = records.sort_values(HEALTH_KEY + ["event_time", "window_id"], kind="stable").reset_index(drop=True)
    grouped = ordered.groupby(HEALTH_KEY, sort=False)
    ordered["previous_window_id"] = grouped["window_id"].shift()
    ordered["previous_rows"] = grouped["rows"].shift()
    ordered["previous_baseline"] = grouped["rows_baseline_max"].shift()
    previous_mature = grouped["rows_baseline_mature"].shift().astype("boolean").fillna(False).astype(bool)
    threshold = multiple * ordered["previous_baseline"].astype("Float64")
    held = previous_mature & (ordered["previous_rows"] >= threshold).fillna(False) & (ordered["rows"]
                                                                                      >= threshold).fillna(False)

    return ordered[held.astype(bool)].reset_index(drop=True)


def attach_all(corpus: dict, **kwargs) -> dict:
    """`attach` over every class of a corpus."""
    return {name: attach(frame, **kwargs) for (name, frame) in corpus.items()}


def with_quarantine(result: pd.DataFrame, quarantined: pd.DataFrame) -> pd.DataFrame:
    """A corpus's tagged output with its quarantined records beside it, as the `quarantine` class."""
    if (len(quarantined) == 0):
        return result

    quarantined = quarantined.copy()
    quarantined["telemetry_class"] = QUARANTINE_CLASS
    quarantined["row_key"] = quarantined["event_uid"]

    return pd.concat([result, quarantined], ignore_index=True)
