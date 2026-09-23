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
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.distinct_window import DistinctWindowTracker
from morpheus.utils.query_entropy import mean_label_length
from morpheus.utils.query_entropy import registered_domain
from morpheus.utils.query_entropy import subdomain
from morpheus.utils.query_entropy import subdomain_entropy

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 3600
"""Trailing window the distinct subdomains are counted over. An hour, which is what R-B-L7-001 names."""

REGISTERED_DOMAIN = "dns_registered_domain"
SUBDOMAIN = "dns_subdomain"
SUBDOMAIN_ENTROPY = "dns_subdomain_entropy"
MEAN_LABEL_LENGTH = "dns_mean_label_length"
SUBDOMAINS_PER_DOMAIN = "dns_subdomains_per_domain"
SUBDOMAINS_SATURATED = "dns_subdomains_saturated"


@register_stage("tc7-dns")
class TC7DnsStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Split each query name into its parts and write the three measurements the tunnelling rule reads.

    The stage is stateful across messages and must run single-engine, or sharded by registered domain.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    query_column : str, default = "query_name"
        Column holding the queried name.
    time_column : str, default = "event_time"
        Column holding the query's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    window_seconds : int, default = 3600
        Trailing window the distinct subdomains are counted over.
    max_samples : int, default = 4096
        Queries retained per registered domain regardless of the window.
    max_entities : int, default = 500000
        Registered domains tracked before the least recently seen is forgotten.
    """

    def __init__(self,
                 c: Config,
                 query_column: str = "query_name",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 max_samples: int = 4096,
                 max_entities: int = 500_000):
        super().__init__(c)

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        self._query_column = query_column
        self._time_column = time_column
        self._time_unit = time_unit

        self._tracker = DistinctWindowTracker(window_ns=window_seconds * NS_PER_SECOND,
                                              max_samples=max_samples,
                                              max_entities=max_entities)

        self._needed_columns[REGISTERED_DOMAIN] = TypeId.STRING
        self._needed_columns[SUBDOMAIN] = TypeId.STRING
        self._needed_columns[SUBDOMAIN_ENTROPY] = TypeId.FLOAT64
        self._needed_columns[MEAN_LABEL_LENGTH] = TypeId.FLOAT64
        self._needed_columns[SUBDOMAINS_PER_DOMAIN] = TypeId.INT64
        self._needed_columns[SUBDOMAINS_SATURATED] = TypeId.BOOL8

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
            missing = [column for column in (self._query_column, self._time_column) if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC7DnsStage requires columns {missing} which are not present in the DataFrame. "
                               f"Available columns: {sorted(df.columns)}")

            names = to_host_list(df, self._query_column)
            raw_times = to_host_list(df, self._time_column)

            domains: list = []
            below: list = []
            entropies: list = []
            label_means: list = []
            per_domain: list = []
            saturated: list = []
            unordered = 0

            for (position, name) in enumerate(names):
                domain = registered_domain(name)
                part = subdomain(name)

                domains.append(domain)
                below.append(part)
                entropies.append(subdomain_entropy(name))
                label_means.append(mean_label_length(part))

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

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

            if (unordered > 0):
                logger.warning(
                    "TC7DnsStage saw %d of %d queries arrive earlier than their registered domain's previous one; "
                    "they were not counted. Equal timestamps are accepted -- a resolver logging at one-second "
                    "resolution puts a whole tunnel burst on one tick.",
                    unordered,
                    len(names))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
