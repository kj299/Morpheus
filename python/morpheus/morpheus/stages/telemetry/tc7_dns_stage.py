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
The three measurements R-B-L7-001 needs from a DNS query, each of which is useless alone.

The rule is query name entropy above 4.0 bits per character, mean label length above 30, and more than 100 distinct
subdomains under one registered domain in an hour -- all three, because each has an ordinary explanation on its own.
Entropy alone flags every content delivery network, whose asset hostnames are random by design. Long labels alone
flag tenant hostnames that happen to be long hashes. A hundred distinct subdomains alone flags any large SaaS
provider. A tunnel is the traffic that is all three at once: random-looking payload, as much of it per label as the
protocol allows, and a new name for every chunk of data.

{py:mod}`~morpheus.utils.query_entropy` supplies the parts of the name, by the Public Suffix List's own algorithm, and
the entropy of the part below the registered domain. {py:mod}`~morpheus.utils.distinct_window` counts distinct
subdomains per registered domain over the trailing window.

**The distinct count is kept per registered domain, not per client**, which is the rule as written: "more than 100
distinct subdomains under one registered domain". It also catches a tunnel spread across several hosts, each of
which stays under the count on its own. The client is still the sealing entity -- it is what behaves, and it is
what a chain at this layer roots on -- and the search lists which clients contributed.

**A query for a registered domain itself has no subdomain and contributes nothing to the count.** It is not a
subdomain, and counting the apex would let a domain reach the threshold with one fewer real name than the rule
requires.

**Each client is also measured against its own days.** A hundred distinct subdomains in an hour is a tunnel from a
workstation and every hour of the day for the endpoint agent that looks up each file hash it sees by DNS. The stage
keeps, per client, the most subdomains-per-domain its queries have carried on each day, through
{py:mod}`~morpheus.utils.bucket_peak`, and writes the highest of a month of days as
`dns_subdomains_per_domain_baseline` and this query's figure as a multiple of it as `dns_subdomains_per_domain_ratio`.
R-B-L7-001 keeps a hundred as a floor and requires twice the client's busiest day above it. A client with less than a
week of days has no baseline and is held to the floor alone.

**A client's own window carries the domain generation signal.** Malware that finds its controller by generating
names asks for many registered domains it has never asked for, and is told most of them do not exist. The stage
counts the distinct registered domains each client asked for in the window, `dns_domains_per_client`, with the same
daily baseline beside it, and the share of its answered queries the resolver answered `NXDOMAIN`,
`dns_nxdomain_ratio`, which is null below `min_queries` -- a client told no twice in three queries has a typo, not a
generator. Neither alone is the signal: a browser on a busy morning reaches many domains that exist, and a laptop
with a stale search suffix is told no constantly about one.
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
from morpheus.pipeline.checkpoint_mixin import CheckpointMixin
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.binding_table import NS_PER_SECOND
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
from morpheus.utils.event_clock import CLOCK_REFUSED_COLUMN
from morpheus.utils.event_clock import DEFAULT_MAX_SKEW_SECONDS
from morpheus.utils.event_clock import clock_for_skew_seconds
from morpheus.utils.event_clock import event_times_ns
from morpheus.utils.event_clock import guard_times
from morpheus.utils.event_clock import refuse_rows
from morpheus.utils.query_entropy import mean_label_length
from morpheus.utils.query_entropy import registered_domain
from morpheus.utils.query_entropy import subdomain
from morpheus.utils.query_entropy import subdomain_entropy
from morpheus.utils.ratio_window import RatioWindowTracker

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 3600
"""Trailing window the distinct subdomains are counted over. An hour, which is what R-B-L7-001 names."""

DEFAULT_MIN_QUERIES = 20
"""Answered queries a client needs in the window before its NXDOMAIN share is published."""

DEFAULT_BASELINE_BUCKET_SECONDS = 86400
"""The period a client's busiest hour is kept per. A day."""

DEFAULT_BASELINE_WINDOW_SECONDS = 30 * 86400
"""How far back a client's days reach."""

DEFAULT_BASELINE_MIN_DAYS = 7
"""Days of history a client needs before it has a baseline. A week, so a weekly rhythm is in it."""

NXDOMAIN = "NXDOMAIN"

REGISTERED_DOMAIN = "dns_registered_domain"
SUBDOMAIN = "dns_subdomain"
SUBDOMAIN_ENTROPY = "dns_subdomain_entropy"
MEAN_LABEL_LENGTH = "dns_mean_label_length"
SUBDOMAINS_PER_DOMAIN = "dns_subdomains_per_domain"
SUBDOMAINS_SATURATED = "dns_subdomains_saturated"
CLIENT_KEY = "dns_client_key"
DOMAINS_PER_CLIENT = "dns_domains_per_client"
DOMAINS_SATURATED = "dns_domains_saturated"
NXDOMAIN_IN_WINDOW = "dns_nxdomain_in_window"
ANSWERED_IN_WINDOW = "dns_answered_in_window"
NXDOMAIN_RATIO = "dns_nxdomain_ratio"
BASELINE_SUFFIX = "_baseline"
BASELINE_DAYS_SUFFIX = "_baseline_days"
BASELINE_MATURE_SUFFIX = "_baseline_mature"
RATIO_SUFFIX = "_ratio"
BASELINED = (SUBDOMAINS_PER_DOMAIN, DOMAINS_PER_CLIENT)
"""The two counts each client is measured against its own days on."""


@register_stage("tc7-dns", ignore_args=["key_columns"])
class TC7DnsStage(CheckpointMixin, GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Split each query name into its parts and write the three measurements the tunnelling rule reads.

    The stage is stateful across messages and must run single-engine, or sharded by registered domain.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    key_columns : list of str, optional
        Columns composing the client. Defaults to `["src_ip"]`.
    query_column : str, default = "query_name"
        Column holding the queried name.
    response_column : str, default = "response_code"
        Column holding the resolver's answer, `NOERROR`, `NXDOMAIN` and so on. The column may be absent, in which
        case the NXDOMAIN columns are null.
    time_column : str, default = "event_time"
        Column holding the query's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    window_seconds : int, default = 3600
        Trailing window the distinct subdomains are counted over.
    max_samples : int, default = 4096
        Queries retained per registered domain regardless of the window.
    max_entities : int, default = 500000
        Registered domains, and clients, tracked before the least recently seen is forgotten.
    min_queries : int, default = 20
        Answered queries a client needs in the window before `dns_nxdomain_ratio` is published.
    baseline_bucket_seconds : int, default = 86400
        The period a client's busiest window is kept per.
    baseline_window_seconds : int, default = 2592000
        How far back those periods reach.
    baseline_min_days : int, default = 7
        Periods of history a client needs before its baselines are published.
    max_clock_skew_seconds : int, default = 604800
        How far ahead of the stream's own progress a row's event time may be before it is refused. Every window and
        every day here expires on event time, so one query from a resolver whose clock is wrong by years would empty
        each domain's subdomains and each client's domains, answers and days at once. A refused row carries no
        windowed measurement from this stage, sets `clock_refused`, and moves no horizon; its name is still split
        and its entropy read, since those need no history. See `morpheus.utils.event_clock`.
    """

    _checkpoint_attributes = ("_tracker", "_domains", "_answers", "_baselines", "_clock")

    def __init__(self,
                 c: Config,
                 key_columns: list[str] = None,
                 query_column: str = "query_name",
                 response_column: str = "response_code",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 max_samples: int = 4096,
                 max_entities: int = 500_000,
                 min_queries: int = DEFAULT_MIN_QUERIES,
                 baseline_bucket_seconds: int = DEFAULT_BASELINE_BUCKET_SECONDS,
                 baseline_window_seconds: int = DEFAULT_BASELINE_WINDOW_SECONDS,
                 baseline_min_days: int = DEFAULT_BASELINE_MIN_DAYS,
                 max_clock_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS):
        super().__init__(c)

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        key_columns = ["src_ip"] if key_columns is None else list(key_columns)

        if (len(key_columns) == 0):
            raise ValueError("key_columns must name at least one column")

        self._key_columns = key_columns
        self._query_column = query_column
        self._response_column = response_column
        self._time_column = time_column
        self._time_unit = time_unit

        window_ns = window_seconds * NS_PER_SECOND

        self._tracker = DistinctWindowTracker(window_ns=window_ns, max_samples=max_samples, max_entities=max_entities)
        self._domains = DistinctWindowTracker(window_ns=window_ns, max_samples=max_samples, max_entities=max_entities)
        if (min_queries < 1):
            raise ValueError(f"min_queries must be at least 1, received {min_queries}")

        # The answers a share is taken over are kept up to at least the number it needs, or a tight sample cap for the
        # subdomain window would leave the share unpublishable.
        self._answers = RatioWindowTracker(window_ns=window_ns,
                                           min_denominator=min_queries,
                                           max_samples=max(max_samples, min_queries),
                                           max_entities=max_entities)
        self._baselines = {
            column:
                BucketPeakTracker(bucket_ns=baseline_bucket_seconds * NS_PER_SECOND,
                                  window_ns=baseline_window_seconds * NS_PER_SECOND,
                                  min_buckets=baseline_min_days,
                                  max_entities=max_entities)
            for column in BASELINED
        }

        self._needed_columns[REGISTERED_DOMAIN] = TypeId.STRING
        self._needed_columns[SUBDOMAIN] = TypeId.STRING
        self._needed_columns[SUBDOMAIN_ENTROPY] = TypeId.FLOAT64
        self._needed_columns[MEAN_LABEL_LENGTH] = TypeId.FLOAT64
        self._needed_columns[SUBDOMAINS_PER_DOMAIN] = TypeId.INT64
        self._needed_columns[SUBDOMAINS_SATURATED] = TypeId.BOOL8
        self._needed_columns[CLIENT_KEY] = TypeId.STRING
        self._needed_columns[DOMAINS_PER_CLIENT] = TypeId.INT64
        self._needed_columns[DOMAINS_SATURATED] = TypeId.BOOL8
        self._needed_columns[NXDOMAIN_IN_WINDOW] = TypeId.INT64
        self._needed_columns[ANSWERED_IN_WINDOW] = TypeId.INT64
        self._needed_columns[NXDOMAIN_RATIO] = TypeId.FLOAT64

        for column in BASELINED:
            self._needed_columns[column + BASELINE_SUFFIX] = TypeId.INT64
            self._needed_columns[column + BASELINE_DAYS_SUFFIX] = TypeId.INT64
            self._needed_columns[column + BASELINE_MATURE_SUFFIX] = TypeId.BOOL8
            self._needed_columns[column + RATIO_SUFFIX] = TypeId.FLOAT64

        self._clock = clock_for_skew_seconds(max_clock_skew_seconds)
        self._needed_columns[CLOCK_REFUSED_COLUMN] = TypeId.BOOL8

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc7-dns"

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

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the registered domain, the subdomain, its entropy and label length, and the distinct count.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming DNS query records.

        Returns
        -------
        The input message, with the DNS columns populated.

        Raises
        ------
        KeyError
            If the query column or the time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = list(self._key_columns) + [self._query_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC7DnsStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            names = to_host_list(df, self._query_column)
            key_parts = {name: to_host_list(df, name) for name in self._key_columns}
            responses = (to_host_list(df, self._response_column) if self._response_column in df.columns else [None] *
                         len(names))
            # Guarded once, before any window or day is shown a time: a refused time becomes `None`, which the rows
            # below already treat as a query with nothing to count.
            (times, clock_refused) = guard_times(self._clock, event_times_ns(df, self._time_column, self._time_unit))

            domains: list = []
            below: list = []
            entropies: list = []
            label_means: list = []
            per_domain: list = []
            saturated: list = []
            keys: list = []
            per_client: list = []
            client_saturated: list = []
            refused: list = []
            answered: list = []
            refused_share: list = []
            unordered = 0

            for (position, name) in enumerate(names):
                domain = registered_domain(name)
                part = subdomain(name)

                domains.append(domain)
                below.append(part)
                entropies.append(subdomain_entropy(name))
                label_means.append(mean_label_length(part))
                event_time_ns = times[position]

                key = compose_key([normalize_text(key_parts[column][position]) for column in self._key_columns])
                keys.append(key)

                self._observe_client(key,
                                     event_time_ns,
                                     domain,
                                     normalize_text(responses[position]),
                                     (per_client, client_saturated, refused, answered, refused_share))

                # An apex query is not a subdomain, so it neither counts toward the figure nor reads one.
                if (domain is None or part is None or event_time_ns is None):
                    per_domain.append(None)
                    saturated.append(False)
                    continue

                result = self._tracker.observe(domain, event_time_ns, part)

                per_domain.append(result.distinct)
                saturated.append(result.saturated)
                unordered += int(result.out_of_order)

            assign_str_column(df, REGISTERED_DOMAIN, domains)
            assign_str_column(df, SUBDOMAIN, below)
            assign_nullable_float_column(df, SUBDOMAIN_ENTROPY, entropies)
            assign_nullable_float_column(df, MEAN_LABEL_LENGTH, label_means)
            assign_nullable_int_column(df, SUBDOMAINS_PER_DOMAIN, per_domain)
            assign_nullable_bool_column(df, SUBDOMAINS_SATURATED, saturated)
            assign_str_column(df, CLIENT_KEY, keys)
            assign_nullable_int_column(df, DOMAINS_PER_CLIENT, per_client)
            assign_nullable_bool_column(df, DOMAINS_SATURATED, client_saturated)
            assign_nullable_int_column(df, NXDOMAIN_IN_WINDOW, refused)
            assign_nullable_int_column(df, ANSWERED_IN_WINDOW, answered)
            assign_nullable_float_column(df, NXDOMAIN_RATIO, refused_share)

            # Each client's days, after the window counts, so each query is measured as it was counted.
            for (column, values) in ((SUBDOMAINS_PER_DOMAIN, per_domain), (DOMAINS_PER_CLIENT, per_client)):
                measured = measure_rows(self._baselines[column], keys, values, times)

                assign_nullable_int_column(df, column + BASELINE_SUFFIX, measured.references)
                assign_nullable_int_column(df, column + BASELINE_DAYS_SUFFIX, measured.buckets)
                assign_nullable_bool_column(df, column + BASELINE_MATURE_SUFFIX, measured.mature)
                assign_nullable_float_column(df, column + RATIO_SUFFIX, ratios_to_reference(measured, values))

            refuse_rows(df,
                        clock_refused,
                        logger=logger,
                        stage="TC7DnsStage",
                        what="queries",
                        outcome="they entered no window, set clock_refused and expired nothing.")

            if (unordered > 0):
                logger.warning(
                    "TC7DnsStage saw %d of %d queries arrive earlier than their registered domain's previous one; "
                    "they were not counted. Equal timestamps are accepted -- a resolver logging at one-second "
                    "resolution puts a whole tunnel burst on one tick.",
                    unordered,
                    len(names))

        return message

    def _observe_client(self, key, event_time_ns, domain, response, outputs: tuple) -> None:
        """The client's distinct registered domains and its NXDOMAIN share, appended to `outputs`."""
        (per_client, client_saturated, refused, answered, refused_share) = outputs

        if (key is None or event_time_ns is None):
            for column in (per_client, refused, answered, refused_share):
                column.append(None)

            client_saturated.append(False)
            return

        if (domain is not None):
            result = self._domains.observe(key, event_time_ns, domain)
            per_client.append(result.distinct)
            client_saturated.append(result.saturated)
        else:
            per_client.append(None)
            client_saturated.append(False)

        # A query the record has no answer for is not a refusal and not an answer.
        if (response is None):
            for column in (refused, answered, refused_share):
                column.append(None)

            return

        share = self._answers.observe(key, event_time_ns, response.upper() == NXDOMAIN)
        refused.append(share.numerator)
        answered.append(share.denominator)
        refused_share.append(share.ratio)

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
