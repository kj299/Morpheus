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
"""Scores ARP for the two shapes that poisoning takes."""

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
from morpheus.utils.column_assign import assign_nullable_float_column
from morpheus.utils.column_assign import assign_nullable_int_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.distinct_window import DistinctWindowTracker
from morpheus.utils.ratio_window import NS_PER_SECOND
from morpheus.utils.ratio_window import RatioWindowTracker

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 300
"""Trailing window both measures cover, matching the reconciliation cadence the telemetry class specifies."""

REPLY_VALUES = frozenset({"2", "reply", "arp_reply", "response", "is-at"})
"""Renderings of the ARP reply opcode seen from collectors in the wild."""

GRATUITOUS_COLUMN = "arp_is_gratuitous"
RATIO_COLUMN = "gratuitous_arp_ratio"
NUMERATOR_COLUMN = "gratuitous_arp_count"
DENOMINATOR_COLUMN = "arp_count_in_window"
CLAIMANTS_COLUMN = "macs_claiming_sender_ip"
EXCLUDED_COLUMN = "arp_sender_ip_excluded"

RATIO_SATURATED_COLUMN = "gratuitous_arp_ratio_saturated"
"""Whether the sender's sample cap bound, which makes the ratio a reading over the retained tail rather than the
whole window."""

CLAIMANTS_SATURATED_COLUMN = "macs_claiming_sender_ip_saturated"
"""Whether the address's sample cap bound, which makes the claimant count a lower bound rather than an exact one."""


@register_stage("tc2-arp", ignore_args=["excluded_sender_ips"])
class TC2ArpStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Score ARP for the two shapes cache poisoning takes.

    **The proportion of a host's ARP that is gratuitous.** A gratuitous ARP claims an address unprompted, and
    legitimate ones are rare and bursty: a failover, an interface coming up, a duplicate-address probe. Poisoning
    is a sustained stream of them, because the attacker has to keep overwriting the victim's cache before the real
    owner corrects it. A raw count would mostly measure how chatty a host is, so the feature is the share of that
    host's ARP which is gratuitous, published only once the window holds enough events for a share to mean
    anything.

    **The number of MAC addresses claiming one IP.** This is the condition R-D-L2-003 names: an address that
    resolves to more than one MAC inside a window is either a poisoning attempt or a first-hop redundancy protocol
    doing its job. The two are indistinguishable from the packets alone, so `excluded_sender_ips` carries the HSRP
    and VRRP virtual addresses. The guide is explicit that the rule is unusable without that list, and this stage
    does not pretend otherwise: it computes the count either way and marks each row with whether its sender address
    was excluded, leaving the rule to decide.

    A gratuitous ARP is one whose sender and target protocol addresses are equal, so `target_ip_column` is
    required. Note that the TC-2 required-field list in the guide omits it, which would make this feature
    uncomputable from the fields as specified; the list has been corrected alongside this stage.

    By default only gratuitous *replies* count toward the numerator, matching the guide. Gratuitous requests exist
    too, since an RFC 5227 announcement is a request, and `include_gratuitous_requests` widens the numerator to
    cover them where a collector reports both.

    The stage is stateful across messages and must run single-engine. Both measures key on values rather than on
    location, so shard by sender if at all.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    sender_ip_column : str, default = "arp_sender_ip"
        Column holding the sender protocol address.
    sender_mac_column : str, default = "arp_sender_mac"
        Column holding the sender hardware address. The gratuitous proportion is taken per sender MAC, since that
        identifies the device doing the claiming.
    target_ip_column : str, default = "arp_target_ip"
        Column holding the target protocol address. Required: gratuitous means sender equals target.
    operation_column : str, default = "arp_operation"
        Column holding the opcode. Compared case-insensitively against several renderings, so `2`, `reply` and
        `is-at` are all understood.
    time_column : str, default = "event_time"
        Column holding the packet's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in `time_column`. Ignored for datetime columns.
    window_seconds : int, default = 300
        Trailing window both measures cover.
    min_denominator : int, default = 10
        ARP events required in the window before a ratio is published. Below this the ratio is null, because a
        proportion over two packets is noise.
    include_gratuitous_requests : bool, default = False
        Count gratuitous requests toward the numerator as well as replies.
    excluded_sender_ips : list of str, optional
        Addresses whose multi-MAC claims are legitimate, typically HSRP and VRRP virtuals. Rows whose sender
        address is listed are marked, not dropped, so the exclusion is visible rather than silent.
    """

    def __init__(self,
                 c: Config,
                 sender_ip_column: str = "arp_sender_ip",
                 sender_mac_column: str = "arp_sender_mac",
                 target_ip_column: str = "arp_target_ip",
                 operation_column: str = "arp_operation",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 min_denominator: int = 10,
                 include_gratuitous_requests: bool = False,
                 excluded_sender_ips: list[str] = None):
        super().__init__(c)

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        self._sender_ip_column = sender_ip_column
        self._sender_mac_column = sender_mac_column
        self._target_ip_column = target_ip_column
        self._operation_column = operation_column
        self._time_column = time_column
        self._time_unit = time_unit
        self._include_requests = include_gratuitous_requests
        self._excluded = {str(address).strip().lower() for address in (excluded_sender_ips or [])}

        window_ns = window_seconds * NS_PER_SECOND
        self._ratio = RatioWindowTracker(window_ns=window_ns, min_denominator=min_denominator)
        self._claimants = DistinctWindowTracker(window_ns=window_ns)

        self._needed_columns[GRATUITOUS_COLUMN] = TypeId.BOOL8
        self._needed_columns[RATIO_COLUMN] = TypeId.FLOAT64
        self._needed_columns[NUMERATOR_COLUMN] = TypeId.INT64
        self._needed_columns[DENOMINATOR_COLUMN] = TypeId.INT64
        self._needed_columns[CLAIMANTS_COLUMN] = TypeId.INT64
        self._needed_columns[EXCLUDED_COLUMN] = TypeId.BOOL8
        self._needed_columns[RATIO_SATURATED_COLUMN] = TypeId.BOOL8
        self._needed_columns[CLAIMANTS_SATURATED_COLUMN] = TypeId.BOOL8

        # Mark this stage to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc2-arp"

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
    def _text(value: typing.Any) -> typing.Optional[str]:
        """Normalize a host value, collapsing every flavor of missing to `None`."""
        if (value is None):
            return None

        if (isinstance(value, float) and math.isnan(value)):
            return None

        return str(value).strip().lower()

    @classmethod
    def _is_reply(cls, operation: typing.Any) -> bool:
        """Whether the opcode names a reply, across the renderings collectors use."""
        value = cls._text(operation)

        return value in REPLY_VALUES if value is not None else False

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the gratuitous classification, the proportion, and the claimant count.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming ARP observations.

        Returns
        -------
        The input message, with the ARP columns populated.

        Raises
        ------
        KeyError
            If a sender, target, or time column is absent.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = [self._sender_ip_column, self._sender_mac_column, self._target_ip_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC2ArpStage requires columns {missing} which are not present in the DataFrame. "
                               f"A gratuitous ARP is one whose sender and target addresses are equal, so the "
                               f"target address is not optional. Available columns: {sorted(df.columns)}")

            sender_ips = to_host_list(df, self._sender_ip_column)
            sender_macs = to_host_list(df, self._sender_mac_column)
            target_ips = to_host_list(df, self._target_ip_column)
            raw_times = to_host_list(df, self._time_column)

            has_operation = self._operation_column in df.columns
            operations = to_host_list(df, self._operation_column) if has_operation else [None] * len(sender_ips)

            gratuitous: list = []
            ratios: list = []
            numerators: list = []
            denominators: list = []
            claimants: list = []
            claimants_saturated: list = []
            ratio_saturated: list = []
            excluded: list = []
            unordered = 0
            keyless = 0

            for (position, raw_sender_ip) in enumerate(sender_ips):
                sender_ip = self._text(raw_sender_ip)
                sender_mac = self._text(sender_macs[position])
                target_ip = self._text(target_ips[position])

                # Gratuitous means the sender is announcing its own address rather than asking about someone
                # else's. A null on either side is not a claim of equality.
                is_gratuitous = sender_ip is not None and sender_ip == target_ip
                gratuitous.append(is_gratuitous)
                excluded.append(sender_ip in self._excluded if sender_ip is not None else False)

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                if (event_time_ns is None):
                    ratios.append(None)
                    numerators.append(None)
                    denominators.append(None)
                    claimants.append(None)
                    ratio_saturated.append(False)
                    claimants_saturated.append(False)
                    unordered += 1
                    continue

                counts = is_gratuitous and (self._include_requests or not has_operation
                                            or self._is_reply(operations[position]))

                # Each measure keys on its own value, and a missing key means no measure rather than every keyless
                # packet in the estate being pooled under the string "None".
                if (sender_mac is None):
                    ratios.append(None)
                    numerators.append(None)
                    denominators.append(None)
                    ratio_saturated.append(False)
                    keyless += 1
                else:
                    ratio = self._ratio.observe(sender_mac, event_time_ns, counts)
                    ratios.append(ratio.ratio)
                    numerators.append(ratio.numerator)
                    denominators.append(ratio.denominator)
                    # The cap binding means the denominator is the retained tail, not the window. A ratio read as
                    # exact when it is a reading over a truncated population is the failure the trackers compute
                    # this flag to prevent, and discarding it here undoes that.
                    ratio_saturated.append(ratio.saturated)
                    unordered += int(ratio.out_of_order)

                # A claim needs both an address and a MAC. A packet with no MAC claims nothing, and counting it would
                # make the address look contested by a phantom.
                if (sender_ip is None or sender_mac is None):
                    claimants.append(None)
                    claimants_saturated.append(False)
                    keyless += int(sender_ip is None and sender_mac is not None)
                else:
                    claim = self._claimants.observe(sender_ip, event_time_ns, sender_mac)
                    claimants.append(claim.distinct)
                    # Under a flood the cap evicts the earliest samples, and the legitimate owner's announcement can
                    # be among them. The count is then a lower bound, and R-D-L2-003 reading it as exact would miss
                    # a contested address rather than over-report one.
                    claimants_saturated.append(claim.saturated)
                    unordered += int(claim.out_of_order)

            df[GRATUITOUS_COLUMN] = gratuitous
            df[EXCLUDED_COLUMN] = excluded
            df[RATIO_SATURATED_COLUMN] = ratio_saturated
            df[CLAIMANTS_SATURATED_COLUMN] = claimants_saturated
            assign_nullable_float_column(df, RATIO_COLUMN, ratios)
            assign_nullable_int_column(df, NUMERATOR_COLUMN, numerators)
            assign_nullable_int_column(df, DENOMINATOR_COLUMN, denominators)
            assign_nullable_int_column(df, CLAIMANTS_COLUMN, claimants)

        if (keyless > 0):
            logger.warning(
                "TC2ArpStage saw %d of %d packets with a null sender MAC or sender IP; the measure keyed on the "
                "missing value is null for those rows.",
                keyless,
                len(sender_ips))

        if (unordered > 0):
            logger.warning(
                "TC2ArpStage saw %d of %d packets out of order or without a usable event time; they did not enter "
                "any window. Preserve per-sender ordering upstream.",
                unordered,
                len(sender_ips))

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
