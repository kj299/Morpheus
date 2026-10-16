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
What R-D-L7-005 reads: how often a client is told no, and how many different things it asked for.

Enumeration is a client walking a server's namespace looking for what exists. It shows as two things together: a
high ratio of 4xx responses to 2xx, because most of what it asks for is not there, and a large number of distinct
paths, because it is asking for everything. The rule wants both -- a 4xx:2xx ratio above 0.7 with more than 200
distinct `url_path` values from one client in ten minutes -- and each alone is ordinary. A crawler requests
thousands of distinct paths and gets 200 for nearly all of them. A misconfigured client hammering one missing
resource gets nothing but 404 on a single path.

**The ratio is written exactly as the rule names it, and the rule is read as a multiplication.** 4xx divided by
2xx is undefined for a client that has received no 2xx at all -- which is the purest enumerator there is, one
that has found nothing. A ratio column would carry a null for it, and a search thresholding that column would
silently exclude the client the rule most exists to catch. The stage therefore emits both counts beside the
ratio, and the shipped search tests `4xx > 0.7 * 2xx`, which is the same inequality with the division removed
and is true for a client with 4xx responses and none else.

**Only 2xx and 4xx responses enter the ratio.** The rule names those two classes and nothing else; a 301 is
neither a success nor a refusal of the path asked for, and a 5xx says something about the server rather than
about whether the path exists. Every request, whatever its status, counts toward the distinct paths.

**The client is keyed by source address by default**, because the guide gives HTTP no entity key of its own and
the source is what behaves. An estate with a reliable identity at its proxy should key on the principal.

**The distinct path count is also measured against the client's own history.** Two hundred paths in ten minutes is
an enumerator from a workstation and an ordinary morning for the vulnerability scanner that walks the web estate
every day. The stage keeps each client's busiest ten minutes of every day, through
{py:mod}`~morpheus.utils.bucket_peak`, and writes the highest of them as `http_distinct_paths_baseline` and this
row's count as a multiple of it as `http_distinct_paths_ratio`. R-D-L7-005 keeps two hundred as a floor and
requires twice the client's busiest day above it; a client with less than a week of days has no baseline, and is
held to the floor alone, which is the rule as the guide wrote it.

**What client software sent the request is a feature of its own.** `http_ua_family` is the `User-Agent` reduced to
its family by {py:mod}`~morpheus.utils.user_agent`, `http_ua_os` the operating system it claims, and
`http_ua_family_first_seen` whether this client has sent this family before -- `None` on its first request, which is
not novel to anything. The operating system is what R-B-L4-004 sets against the stack fingerprint the client's
connections carry.
"""

import logging
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
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.binding_table import to_epoch_ns
from morpheus.utils.bucket_peak import BucketPeakTracker
from morpheus.utils.bucket_peak import measure_rows
from morpheus.utils.bucket_peak import ratios_to_reference
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.distinct_window import DistinctWindowTracker
from morpheus.utils.entity_key import compose_key
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.ratio_window import RatioWindowTracker
from morpheus.utils.user_agent import user_agent_family
from morpheus.utils.user_agent import user_agent_os
from morpheus.utils.value_novelty import ValueNoveltyTracker

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 600
"""Trailing window both measurements are taken over. Ten minutes, which is what R-D-L7-005 names."""

DEFAULT_BASELINE_BUCKET_SECONDS = 86400
"""The period a client's busiest window is kept per. A day."""

DEFAULT_BASELINE_WINDOW_SECONDS = 30 * 86400
"""How far back a client's days reach."""

DEFAULT_BASELINE_MIN_DAYS = 7
"""Days of history a client needs before it has a baseline. A week, so a weekly rhythm is in it."""

CLIENT_KEY = "http_client_key"
STATUS_CLASS = "http_status_class"
CLIENT_ERRORS = "http_4xx_in_window"
SUCCESSES = "http_2xx_in_window"
ERROR_RATIO = "http_4xx_to_2xx_ratio"
DISTINCT_PATHS = "http_distinct_paths"
WINDOW_SATURATED = "http_window_saturated"
DISTINCT_PATHS_BASELINE = "http_distinct_paths_baseline"
DISTINCT_PATHS_BASELINE_DAYS = "http_distinct_paths_baseline_days"
DISTINCT_PATHS_BASELINE_MATURE = "http_distinct_paths_baseline_mature"
DISTINCT_PATHS_RATIO = "http_distinct_paths_ratio"
UA_FAMILY = "http_ua_family"
UA_OS = "http_ua_os"
UA_FAMILY_FIRST_SEEN = "http_ua_family_first_seen"


@register_stage("tc7-http", ignore_args=["key_columns"])
class TC7HttpStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write each client's windowed 4xx and 2xx counts, their ratio, and its distinct path count.

    The stage is stateful across messages and must run single-engine, or sharded by the same key the windows are
    kept on.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    key_columns : list of str, optional
        Columns composing the client. Defaults to `["src_ip"]`.
    status_column : str, default = "status_code"
        Column holding the HTTP response status.
    path_column : str, default = "url_path"
        Column holding the requested path, without its query string.
    user_agent_column : str, default = "user_agent"
        Column holding the request's `User-Agent` header. The column may be absent, in which case the user-agent
        columns are null.
    time_column : str, default = "event_time"
        Column holding the request's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    window_seconds : int, default = 600
        Trailing window both measurements are taken over.
    max_samples : int, default = 4096
        Requests retained per client regardless of the window.
    max_entities : int, default = 500000
        Clients tracked before the least recently seen is forgotten.
    baseline_bucket_seconds : int, default = 86400
        The period a client's busiest window of distinct paths is kept per.
    baseline_window_seconds : int, default = 2592000
        How far back those periods reach.
    baseline_min_days : int, default = 7
        Periods of history a client needs before `http_distinct_paths_baseline` is published.
    """

    def __init__(self,
                 c: Config,
                 key_columns: list[str] = None,
                 status_column: str = "status_code",
                 path_column: str = "url_path",
                 user_agent_column: str = "user_agent",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 max_samples: int = 4096,
                 max_entities: int = 500_000,
                 baseline_bucket_seconds: int = DEFAULT_BASELINE_BUCKET_SECONDS,
                 baseline_window_seconds: int = DEFAULT_BASELINE_WINDOW_SECONDS,
                 baseline_min_days: int = DEFAULT_BASELINE_MIN_DAYS):
        super().__init__(c)

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        key_columns = ["src_ip"] if key_columns is None else list(key_columns)

        if (len(key_columns) == 0):
            raise ValueError("key_columns must name at least one column")

        self._key_columns = key_columns
        self._status_column = status_column
        self._path_column = path_column
        self._user_agent_column = user_agent_column
        self._time_column = time_column
        self._time_unit = time_unit

        window_ns = window_seconds * NS_PER_SECOND

        # The ratio's own figure is computed from the counts, so the tracker's minimum denominator is set to one and
        # its ratio ignored; its job here is to keep the two counts over the right window.
        self._responses = RatioWindowTracker(window_ns=window_ns,
                                             min_denominator=1,
                                             max_samples=max_samples,
                                             max_entities=max_entities)
        self._paths = DistinctWindowTracker(window_ns=window_ns, max_samples=max_samples, max_entities=max_entities)
        self._baseline = BucketPeakTracker(bucket_ns=baseline_bucket_seconds * NS_PER_SECOND,
                                           window_ns=baseline_window_seconds * NS_PER_SECOND,
                                           min_buckets=baseline_min_days,
                                           max_entities=max_entities)
        # A client opens a dozen connections in one second, so requests at one instant are measured, not refused.
        self._agents = ValueNoveltyTracker([UA_FAMILY], max_entities=max_entities, simultaneous=True)

        self._needed_columns[CLIENT_KEY] = TypeId.STRING
        self._needed_columns[STATUS_CLASS] = TypeId.STRING
        self._needed_columns[CLIENT_ERRORS] = TypeId.INT64
        self._needed_columns[SUCCESSES] = TypeId.INT64
        self._needed_columns[ERROR_RATIO] = TypeId.FLOAT64
        self._needed_columns[DISTINCT_PATHS] = TypeId.INT64
        self._needed_columns[WINDOW_SATURATED] = TypeId.BOOL8
        self._needed_columns[DISTINCT_PATHS_BASELINE] = TypeId.INT64
        self._needed_columns[DISTINCT_PATHS_BASELINE_DAYS] = TypeId.INT64
        self._needed_columns[DISTINCT_PATHS_BASELINE_MATURE] = TypeId.BOOL8
        self._needed_columns[DISTINCT_PATHS_RATIO] = TypeId.FLOAT64
        self._needed_columns[UA_FAMILY] = TypeId.STRING
        self._needed_columns[UA_OS] = TypeId.STRING
        self._needed_columns[UA_FAMILY_FIRST_SEEN] = TypeId.BOOL8

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc7-http"

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
    def _status_class(value: typing.Any) -> typing.Optional[str]:
        """`2xx`, `4xx` and so on, or `None` for anything that is not a three-digit status."""
        try:
            if (value is None or value != value):  # pylint: disable=comparison-with-itself
                return None

            number = float(value)
        except (TypeError, ValueError):
            return None

        if (number != int(number) or not 100 <= number <= 599):
            return None

        return f"{int(number) // 100}xx"

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the client key, the status class, the windowed counts and ratio, and the distinct path count.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming HTTP request records.

        Returns
        -------
        The input message, with the HTTP columns populated.

        Raises
        ------
        KeyError
            If a key column, the status column, the path column, or the time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = list(self._key_columns) + [self._status_column, self._path_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC7HttpStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            key_parts = {name: to_host_list(df, name) for name in self._key_columns}
            statuses = to_host_list(df, self._status_column)
            paths = to_host_list(df, self._path_column)
            raw_times = to_host_list(df, self._time_column)
            agents = (to_host_list(df, self._user_agent_column) if self._user_agent_column in df.columns else [None] *
                      len(statuses))

            keys: list = []
            classes: list = []
            errors: list = []
            successes: list = []
            ratios: list = []
            distinct: list = []
            saturated: list = []
            times: list = []
            families: list = []
            systems: list = []
            first_seen: list = []
            unusable = 0
            unordered = 0

            for position, status in enumerate(statuses):
                key = compose_key([normalize_text(key_parts[name][position]) for name in self._key_columns])
                status_class = self._status_class(status)
                path = normalize_text(paths[position])

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                keys.append(key)
                classes.append(status_class)
                times.append(event_time_ns)

                family = user_agent_family(agents[position])
                families.append(family)
                systems.append(user_agent_os(agents[position]))

                # A request with no header says nothing about the client's software, so it is not a sample of it.
                if (key is None or event_time_ns is None or family is None):
                    first_seen.append(None)
                else:
                    novelty = self._agents.observe(key, event_time_ns, {UA_FAMILY: family})
                    first_seen.append(novelty.first_seen[UA_FAMILY])

                if (key is None or event_time_ns is None):
                    unusable += 1
                    errors.append(None)
                    successes.append(None)
                    ratios.append(None)
                    distinct.append(None)
                    saturated.append(False)
                    continue

                window_saturated = False

                # Every request counts toward the paths, whatever it was answered with.
                if (path is not None):
                    path_result = self._paths.observe(key, event_time_ns, path)
                    distinct.append(path_result.distinct)
                    window_saturated = path_result.saturated
                    unordered += int(path_result.out_of_order)
                else:
                    distinct.append(None)

                # Only the two classes the rule names enter the ratio.
                if (status_class in ("2xx", "4xx")):
                    response = self._responses.observe(key, event_time_ns, status_class == "4xx")
                    client_errors = response.numerator
                    ok = response.denominator - response.numerator

                    errors.append(client_errors)
                    successes.append(ok)
                    # Undefined, not zero, for a client that has had no success -- see the module docstring for why
                    # the search reads the counts rather than this column.
                    ratios.append(client_errors / ok if ok > 0 else None)
                    window_saturated = window_saturated or response.saturated
                    unordered += int(response.out_of_order)
                else:
                    errors.append(None)
                    successes.append(None)
                    ratios.append(None)

                saturated.append(window_saturated)

            assign_str_column(df, CLIENT_KEY, keys)
            assign_str_column(df, STATUS_CLASS, classes)
            assign_nullable_int_column(df, CLIENT_ERRORS, errors)
            assign_nullable_int_column(df, SUCCESSES, successes)
            assign_nullable_float_column(df, ERROR_RATIO, ratios)
            assign_nullable_int_column(df, DISTINCT_PATHS, distinct)
            assign_nullable_bool_column(df, WINDOW_SATURATED, saturated)

            # The day's busiest window per client, after the window counts so each row is measured as it was counted.
            measured = measure_rows(self._baseline, keys, distinct, times)

            assign_nullable_int_column(df, DISTINCT_PATHS_BASELINE, measured.references)
            assign_nullable_int_column(df, DISTINCT_PATHS_BASELINE_DAYS, measured.buckets)
            assign_nullable_bool_column(df, DISTINCT_PATHS_BASELINE_MATURE, measured.mature)
            assign_nullable_float_column(df, DISTINCT_PATHS_RATIO, ratios_to_reference(measured, distinct))
            assign_str_column(df, UA_FAMILY, families)
            assign_str_column(df, UA_OS, systems)
            assign_nullable_bool_column(df, UA_FAMILY_FIRST_SEEN, first_seen)

            if (unusable > 0):
                logger.warning(
                    "TC7HttpStage left %d of %d requests out of their client's windows for want of a key "
                    "or a usable event time.",
                    unusable,
                    len(statuses))

            if (unordered > 0):
                logger.warning(
                    "TC7HttpStage saw %d requests arrive earlier than their client's previous one; they were not "
                    "counted. Equal timestamps are accepted, since a proxy logging at one-second resolution puts an "
                    "enumerator's whole burst on one tick.",
                    unordered)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
