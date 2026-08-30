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
The event-time rendering a SIEM parses off the wire.

Inside a pipeline an event time is an integer count of nanoseconds: exact, cheap to compare, and what
`window_id_from_timestamp` and the binding tables consume. On the wire it has to be something the SIEM can parse
into its own time field, and a nineteen-digit integer is not that. Splunk's timestamp extraction anchors on a
prefix and applies a `strptime` format, so an unrendered nanosecond value matches nothing, and `_time` falls back
to index time -- which turns every windowed rule into a rule about when the pipeline happened to be busy.

`render_event_time` produces the one rendering the shipped Splunk app is configured to parse. Emit it as the
`event_time` field on every record leaving the pipeline. The rendering is microsecond precision, matching what
Splunk's `%6N` reads; where the exact nanosecond value must survive the hop, carry it alongside in a separate
numeric field rather than widening this one.
"""

import datetime
import typing

from morpheus.utils.binding_table import to_epoch_ns

EVENT_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
"""`strftime` format for the rendered portion, before the fixed UTC suffix."""

EVENT_TIME_SUFFIX = "UTC"
"""Zone suffix, read by Splunk's `%Z`. Fixed rather than local, so the rendering never depends on host settings."""

SPLUNK_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%6N%Z"
"""The `TIME_FORMAT` the shipped Splunk app uses for this rendering. `%6N` is Splunk's six-digit subsecond."""


def render_event_time(value: typing.Any, time_unit: str = "ns") -> typing.Optional[str]:
    """
    Render one event time in the wire format the SIEM parses.

    Parameters
    ----------
    value : any
        Nanoseconds since the Unix epoch, or anything else `morpheus.utils.binding_table.to_epoch_ns` accepts:
        a `datetime`, a `pandas.Timestamp`, or a parsable date string.
    time_unit : str, default = "ns"
        Unit for numeric input.

    Returns
    -------
    str or None
        For example `2026-08-30T18:25:00.123456UTC`, or `None` when the value is null. Precision is microseconds;
        sub-microsecond digits are truncated rather than rounded, so the rendering never moves an event across a
        window boundary it did not cross.
    """
    epoch_ns = to_epoch_ns(value, time_unit=time_unit)

    if (epoch_ns is None):
        return None

    # Truncating integer division rather than float seconds: dividing by 1e9 loses precision above 2**53 ns, which
    # arrives in 2255 but also whenever a test uses a large synthetic timestamp.
    (whole_seconds, remainder_ns) = divmod(epoch_ns, 10**9)
    moment = datetime.datetime.fromtimestamp(whole_seconds, datetime.timezone.utc)

    return f"{moment.strftime('%Y-%m-%dT%H:%M:%S')}.{remainder_ns // 1000:06d}{EVENT_TIME_SUFFIX}"


def render_event_time_series(values: typing.Sequence, time_unit: str = "ns") -> list[typing.Optional[str]]:
    """
    Render a column of event times, for assignment back onto a DataFrame before a SIEM sink.

    Parameters
    ----------
    values : sequence
        Per-row event times.
    time_unit : str, default = "ns"
        Unit for numeric input.

    Returns
    -------
    list
        One rendering per row, `None` where the input was null.
    """
    return [render_event_time(value, time_unit=time_unit) for value in values]
