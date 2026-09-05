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
"""Scores optical power against a per-port rolling baseline."""

import logging
import math
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
from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.optical_baseline import DEFAULT_MIN_SAMPLES
from morpheus.utils.optical_baseline import NS_PER_SECOND
from morpheus.utils.optical_baseline import OpticalBaselineTracker

logger = logging.getLogger(__name__)

DEFAULT_CHANNEL_COLUMNS = ["optical_tx_dbm", "optical_rx_dbm"]
"""Optical channels the TC-1 telemetry class carries."""

DEFAULT_WINDOW_SECONDS = 6 * 3600
"""Trailing window the baseline is taken over, in seconds of event time."""

BASELINE_SUFFIX = "_baseline"
DEVIATION_SUFFIX = "_deviation"
SAMPLES_SUFFIX = "_baseline_samples"


@register_stage("tc1-optical")
class TC1OpticalStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Score each port's optical power against what that port has been reporting.

    An absolute optical level is not interpretable on its own: -7 dBm is healthy on one link and alarming on
    another, because the correct value depends on the optic, the fibre length and the patch path. The diagnostic
    quantity is the deviation from the port's own recent history, which is what this stage writes.

    Two things move it. Degradation drifts the receive level down over days as a connector fouls or a bend
    tightens. An inline tap steps it down at once, because a passive splitter has to divert light to see it and the
    diverted light is missing at the far end: one to three dB for the common ones. That step is the security signal,
    and no absolute threshold loose enough to avoid alarming on healthy links would catch it.

    Transmit and receive are scored separately because they fail differently. A falling transmit level is the local
    laser ageing; a falling receive level is something that happened to the path.

    The stage is stateful across messages, holding a trailing window of readings per port, and must run
    single-engine. For parallelism, shard by device upstream and give each shard its own instance, which is
    determinism control 4. Place it after
    `morpheus.stages.telemetry.tc1_normalize_stage.TC1NormalizeStage`, which supplies `entity_key`. Ports with no
    optic in the cage report no level and are carried through with null deviations rather than being baselined
    against nothing, so the stage is safe to run over a mixed copper and fibre estate.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    entity_key_column : str, default = "entity_key"
        Column holding the port identity the baseline is kept per. Anything coarser than the port is meaningless
        here, since two ports on one switch legitimately sit tens of dB apart.
    time_column : str, default = "event_time"
        Column holding the sample's event time. Event time, never ingest time: a window measured in arrival order
        describes the collector's scheduling rather than the link.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    channel_columns : list of str, optional
        Optical power columns, in dBm. Defaults to `["optical_tx_dbm", "optical_rx_dbm"]`.
    window_seconds : int, default = 21600
        Trailing window the baseline is taken over. See the note on what the length costs.
    min_samples : int, default = 5
        Prior readings required before a baseline is published. Below this a port carries null deviations.

    Notes
    -----
    The baseline follows the link, so a step is a transient signal rather than a standing state: once the window
    is half rolled past the last pre-step reading, the median describes the new level and the deviation returns to
    zero. It has to be caught within half of `window_seconds` of the step, since a median turns
    once half the retained samples are on the new level. Lengthening the window holds the evidence longer
    at the cost of a reference that lags a legitimate re-patch for just as long. A degradation slower than the
    window is invisible for the same reason, since the baseline drifts down with it; catching that needs a
    comparison against a commissioning value, which is asset context and belongs in TC-0.
    """

    def __init__(self,
                 c: Config,
                 entity_key_column: str = "entity_key",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 channel_columns: list[str] = None,
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 min_samples: int = DEFAULT_MIN_SAMPLES):
        super().__init__(c)

        channel_columns = list(DEFAULT_CHANNEL_COLUMNS) if channel_columns is None else list(channel_columns)

        if (len(channel_columns) == 0):
            raise ValueError("channel_columns must name at least one optical channel")

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        self._entity_key_column = entity_key_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._channel_columns = channel_columns

        self._tracker = OpticalBaselineTracker(channel_names=channel_columns,
                                               window_ns=window_seconds * NS_PER_SECOND,
                                               min_samples=min_samples)

        for name in channel_columns:
            self._needed_columns[f"{name}{BASELINE_SUFFIX}"] = TypeId.FLOAT64
            self._needed_columns[f"{name}{DEVIATION_SUFFIX}"] = TypeId.FLOAT64
            self._needed_columns[f"{name}{SAMPLES_SUFFIX}"] = TypeId.INT64

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc1-optical"

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

    @staticmethod
    def _reading(value: typing.Any) -> typing.Optional[float]:
        """Return a host value as a float, or `None` where the port reported no level."""
        if (value is None):
            return None

        try:
            reading = float(value)
        except (TypeError, ValueError):
            return None

        # A null in a float column arrives as NaN, which is not a level and must not enter a baseline.
        return None if math.isnan(reading) else reading

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the baseline, the deviation, and the supporting sample count for every optical channel.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming message.

        Returns
        -------
        The input message, with the baseline columns populated.

        Raises
        ------
        KeyError
            If the entity key, the time column, or a declared channel column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._entity_key_column, self._time_column] + self._channel_columns
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC1OpticalStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            entity_keys = to_host_list(df, self._entity_key_column)
            raw_times = to_host_list(df, self._time_column)
            channels = {name: to_host_list(df, name) for name in self._channel_columns}

            baselines: dict[str, list] = {name: [] for name in self._channel_columns}
            deviations: dict[str, list] = {name: [] for name in self._channel_columns}
            counts: dict[str, list] = {name: [] for name in self._channel_columns}
            unordered = 0
            keyless = 0

            for (position, entity_key) in enumerate(entity_keys):
                # A row whose key is null has no identity to hold state against. Passing `str(None)` would pool every
                # such row in the estate under one tracker entity named "None", where one port's transceiver becomes
                # another's substitution and one port's optical reading becomes another's baseline. The contract in
                # `morpheus.utils.entity_key` is that a null key gets no per-entity features and the stage says how
                # many rows that happened to.
                key = normalize_text(entity_key)

                if (key is None):
                    for name in self._channel_columns:
                        baselines[name].append(None)
                        deviations[name].append(None)
                        counts[name].append(0)

                    keyless += 1
                    continue

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                if (event_time_ns is None):
                    for name in self._channel_columns:
                        baselines[name].append(None)
                        deviations[name].append(None)
                        counts[name].append(0)

                    unordered += 1
                    continue

                result = self._tracker.observe(
                    key,
                    event_time_ns, {name: self._reading(channels[name][position])
                                    for name in self._channel_columns})

                for name in self._channel_columns:
                    baselines[name].append(result.baselines[name])
                    deviations[name].append(result.deviations[name])
                    counts[name].append(result.sample_counts[name])

                unordered += int(result.out_of_order)

            for name in self._channel_columns:
                assign_nullable_float_column(df, f"{name}{BASELINE_SUFFIX}", baselines[name])
                assign_nullable_float_column(df, f"{name}{DEVIATION_SUFFIX}", deviations[name])
                assign_nullable_int_column(df, f"{name}{SAMPLES_SUFFIX}", counts[name])

        if (keyless > 0):
            logger.warning(
                "TC1OpticalStage saw %d of %d rows with a null entity key; they carry no baseline or "
                "deviation. A null key means a missing site, device, or port upstream, not a port named "
                "\"None\".",
                keyless,
                len(entity_keys))

        if (unordered > 0):
            logger.warning(
                "TC1OpticalStage saw %d of %d samples out of order or without a usable event time; they carry no "
                "deviation and did not enter any baseline. Shard by device and preserve per-port ordering "
                "upstream.",
                unordered,
                len(entity_keys))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
