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
Whether a destination's certificate is the one it usually presents, and whether it vouches for itself.

Two rules read this stage, and they are opposite in character. R-D-L6-002 is comparative: an issuer that differs
from the issuer this destination has settled on catches interception, including the well-intentioned kind that
broke a policy, and it needs a per-destination reference because an estate with a dozen certificate authorities
has no single correct answer. R-D-L6-003 is absolute: a self-signed certificate on a connection leaving the
estate is almost always either a misconfiguration or attacker infrastructure, and needs no history at all --
which is why it fires on the first observation while the issuer rule waits for a reference.

**The reference is a mode over the destination's own prior handshakes**, taken through
{py:mod}`~morpheus.utils.established_value`. A mode rather than a set, because a destination behind a load
balancer with one certificate rotated early presents two issuers legitimately and the commoner one is the
expectation; `cert_issuer_distinct` rides along so a search can weigh a difference from a destination that has
only ever presented one issuer against the same difference from one that presents four.

**"External" is `parsers/ip.py`'s judgement, not a copy of the range list.** `TC3ReachStage` classifies
destinations from the same module, and a second copy here would be a second thing to keep correct -- the layer 3
corpus already found what happens when a classification and its documentation drift apart.

**The validity window is carried rather than judged.** Attacker-generated infrastructure tends to short
certificate lifetimes and the guide names the distribution as a feature, but where the cutoff sits is an estate's
question: ninety days is Let's Encrypt's normal and a red flag in an estate that issues for a year. The stage
emits the span in days and the searches put a threshold on it, which is the same division of labour every other
windowed figure in this fork uses.

**An issuer new to the whole estate is a third, separate question**, which R-C-004 asks: has anyone here seen this
authority in the last thirty days? The per-destination reference cannot answer it, because a destination seen for
the first time has no reference and every issuer it presents is equally unremarkable to it. The estate's issuers are
kept through {py:mod}`~morpheus.utils.pair_history` with the estate as the one entity, and the answer waits for the
estate's own warm-up, because on its first day every issuer is new to it.
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
from morpheus.utils.entity_key import compose_key
from morpheus.utils.entity_key import normalize_text
from morpheus.utils.established_value import DEFAULT_MIN_SAMPLES
from morpheus.utils.established_value import ValueHistoryTracker
from morpheus.utils.established_value import mode_of
from morpheus.utils.pair_history import PairHistoryTracker

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 30 * 24 * 3600
"""Trailing window the issuer reference is taken over. Thirty days, which is what the layer 6 rules name."""

DEFAULT_ESTATE_WARMUP_SECONDS = 7 * 24 * 3600
"""History the estate needs before an issuer can be called new to it."""

ESTATE = "estate"
"""The one entity the estate-wide issuer history is kept under."""

SELF_SIGNED_RESULTS = frozenset({
    "self-signed",
    "self_signed",
    "selfsigned",
    "self signed",
    "unable to get local issuer certificate",
    "depth zero self signed cert",
    "self signed certificate",
    "self signed certificate in certificate chain",
})
"""Validation outcomes that mean the certificate vouches for itself.

The spellings OpenSSL and Zeek actually emit, rather than one canonical form, because a rule that recognized
only the tidy spelling would silently never fire against the feed most estates have.
"""

DESTINATION_KEY = "tls_destination_key"
ISSUER_ESTABLISHED = "cert_issuer_established"
ISSUER_DIFFERS = "cert_issuer_differs"
ISSUER_DISTINCT = "cert_issuer_distinct"
ISSUER_MATURE = "cert_issuer_mature"
ISSUER_SATURATED = "cert_issuer_saturated"
SELF_SIGNED = "cert_self_signed"
DESTINATION_GLOBAL = "cert_destination_is_global"
SELF_SIGNED_EXTERNAL = "cert_self_signed_external"
VALIDITY_DAYS = "cert_validity_days"
ISSUER_NEW_TO_ESTATE = "cert_issuer_new_to_estate"


@register_stage("tc6-certificate", ignore_args=["key_columns"])
class TC6CertificateStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Compare each handshake's certificate against its destination's established issuer, and flag self-signing.

    The stage is stateful across messages and must run single-engine, or sharded by the same key the reference
    is kept on.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    key_columns : list of str, optional
        Columns composing the destination the issuer reference is kept for. Defaults to `["dst_ip"]`. An estate
        terminating many names on one address should add `sni`, so a reference is per name rather than per
        address and a shared host stops looking like an estate with a dozen issuers.
    issuer_column : str, default = "certificate_issuer"
        Column holding the issuing authority's distinguished name.
    validation_column : str, default = "validation_result"
        Column holding the chain validation outcome.
    destination_column : str, default = "dst_ip"
        Column holding the destination address, classified to decide whether the connection leaves the estate.
    not_before_column : str, default = "certificate_not_before"
        Column holding the start of the certificate's validity.
    not_after_column : str, default = "certificate_not_after"
        Column holding the end of it.
    time_column : str, default = "event_time"
        Column holding the handshake's event time. Event time, never ingest time.
    time_unit : str, default = "ns"
        Unit for numeric timestamps in the time columns. Ignored for datetime columns.
    window_seconds : int, default = 2592000
        Trailing window the issuer reference is taken over.
    min_samples : int, default = 5
        Prior handshakes required before a reference is published.
    max_samples : int, default = 512
        Handshakes retained per destination regardless of the window.
    estate_warmup_seconds : int, default = 604800
        History the estate needs before `cert_issuer_new_to_estate` answers. The estate-wide history is kept over
        `window_seconds`, like the per-destination one.
    """

    def __init__(self,
                 c: Config,
                 key_columns: list[str] = None,
                 issuer_column: str = "certificate_issuer",
                 validation_column: str = "validation_result",
                 destination_column: str = "dst_ip",
                 not_before_column: str = "certificate_not_before",
                 not_after_column: str = "certificate_not_after",
                 time_column: str = "event_time",
                 time_unit: str = "ns",
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 max_samples: int = 512,
                 estate_warmup_seconds: int = DEFAULT_ESTATE_WARMUP_SECONDS):
        super().__init__(c)

        if (window_seconds <= 0):
            raise ValueError(f"window_seconds must be positive, received {window_seconds}")

        key_columns = ["dst_ip"] if key_columns is None else list(key_columns)

        if (len(key_columns) == 0):
            raise ValueError("key_columns must name at least one column")

        self._key_columns = key_columns
        self._issuer_column = issuer_column
        self._validation_column = validation_column
        self._destination_column = destination_column
        self._not_before_column = not_before_column
        self._not_after_column = not_after_column
        self._time_column = time_column
        self._time_unit = time_unit

        self._tracker = ValueHistoryTracker(reduction=mode_of,
                                            window_ns=window_seconds * NS_PER_SECOND,
                                            min_samples=min_samples,
                                            max_samples=max_samples)
        self._estate = PairHistoryTracker(window_ns=window_seconds * NS_PER_SECOND,
                                          warmup_ns=estate_warmup_seconds * NS_PER_SECOND,
                                          max_entities=1)

        self._needed_columns[DESTINATION_KEY] = TypeId.STRING
        self._needed_columns[ISSUER_ESTABLISHED] = TypeId.STRING
        self._needed_columns[ISSUER_DISTINCT] = TypeId.INT64
        self._needed_columns[VALIDITY_DAYS] = TypeId.FLOAT64

        for column in (ISSUER_DIFFERS,
                       ISSUER_MATURE,
                       ISSUER_SATURATED,
                       SELF_SIGNED,
                       DESTINATION_GLOBAL,
                       SELF_SIGNED_EXTERNAL,
                       ISSUER_NEW_TO_ESTATE):
            self._needed_columns[column] = TypeId.BOOL8

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc6-certificate"

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
    def _is_self_signed(value: typing.Any) -> typing.Optional[bool]:
        """Whether a validation outcome says the certificate vouches for itself, or `None` where none was given."""
        normalized = normalize_text(value)

        return None if normalized is None else normalized.strip().lower() in SELF_SIGNED_RESULTS

    def _validity_days(self, not_before: typing.Any, not_after: typing.Any) -> typing.Optional[float]:
        """The certificate's lifetime in days, or `None` where either bound is missing or unreadable."""
        try:
            start = to_epoch_ns(not_before, time_unit=self._time_unit)
            end = to_epoch_ns(not_after, time_unit=self._time_unit)
        except ValueError:
            return None

        if (start is None or end is None):
            return None

        # A negative span is a parsing fault or a certificate whose bounds are the wrong way round. Either is
        # worth seeing rather than clamping to zero, which would read as a certificate valid for no time at all.
        return (end - start) / (NS_PER_SECOND * 86400)

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]):
        """
        Write the destination's established issuer, this certificate's relation to it, and the self-signing flags.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming handshake records.

        Returns
        -------
        The input message, with the certificate columns populated.

        Raises
        ------
        KeyError
            If a key column, the issuer column, or the time column is absent.
        """
        from morpheus.parsers import ip  # pylint: disable=import-outside-toplevel

        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            required = list(self._key_columns) + [self._issuer_column, self._time_column]
            missing = [column for column in required if column not in df.columns]

            if (len(missing) > 0):
                raise KeyError(f"TC6CertificateStage requires columns {missing} which are not present in the "
                               f"DataFrame. Available columns: {sorted(df.columns)}")

            key_parts = {name: to_host_list(df, name) for name in self._key_columns}
            issuers = to_host_list(df, self._issuer_column)
            raw_times = to_host_list(df, self._time_column)
            rows = len(issuers)

            validations = (to_host_list(df, self._validation_column)
                           if self._validation_column in df.columns else [None] * rows)
            not_befores = (to_host_list(df, self._not_before_column)
                           if self._not_before_column in df.columns else [None] * rows)
            not_afters = (to_host_list(df, self._not_after_column) if self._not_after_column in df.columns else [None] *
                          rows)

            globals_by_row = self._classify_destinations(df, ip, rows)

            keys: list = []
            established: list = []
            differs: list = []
            distinct: list = []
            mature: list = []
            saturated: list = []
            self_signed: list = []
            external_self_signed: list = []
            validity: list = []
            new_to_estate: list = []
            unusable = 0
            unordered = 0

            for position in range(rows):
                key = compose_key([normalize_text(key_parts[name][position]) for name in self._key_columns])
                issuer = normalize_text(issuers[position])

                try:
                    event_time_ns = to_epoch_ns(raw_times[position], time_unit=self._time_unit)
                except ValueError:
                    event_time_ns = None

                signed = self._is_self_signed(validations[position])
                is_global = globals_by_row[position]

                keys.append(key)
                self_signed.append(signed)
                validity.append(self._validity_days(not_befores[position], not_afters[position]))

                # Both halves must be known. A self-signed certificate whose destination could not be classified
                # is not a self-signed certificate to an internal host, and saying so would be an answer the
                # record does not support.
                external_self_signed.append(None if (
                    signed is None or is_global is None) else bool(signed and is_global))

                if (issuer is None or event_time_ns is None):
                    new_to_estate.append(None)
                else:
                    sighting = self._estate.observe(ESTATE, event_time_ns, issuer)
                    new_to_estate.append(None if (sighting.out_of_order or not sighting.mature) else not sighting.seen)

                if (key is None or issuer is None or event_time_ns is None):
                    unusable += 1
                    established.append(None)
                    differs.append(None)
                    distinct.append(None)
                    mature.append(False)
                    saturated.append(False)
                    continue

                result = self._tracker.observe(key, event_time_ns, issuer)

                established.append(result.reference)
                differs.append(None if result.reference is None else issuer != result.reference)
                distinct.append(result.distinct)
                mature.append(result.mature)
                saturated.append(result.saturated)
                unordered += int(result.out_of_order)

            assign_str_column(df, DESTINATION_KEY, keys)
            assign_str_column(df, ISSUER_ESTABLISHED, established)
            assign_nullable_bool_column(df, ISSUER_DIFFERS, differs)
            assign_nullable_int_column(df, ISSUER_DISTINCT, distinct)
            assign_nullable_bool_column(df, ISSUER_MATURE, mature)
            assign_nullable_bool_column(df, ISSUER_SATURATED, saturated)
            assign_nullable_bool_column(df, SELF_SIGNED, self_signed)
            assign_nullable_bool_column(df, DESTINATION_GLOBAL, globals_by_row)
            assign_nullable_bool_column(df, SELF_SIGNED_EXTERNAL, external_self_signed)
            assign_nullable_float_column(df, VALIDITY_DAYS, validity)
            assign_nullable_bool_column(df, ISSUER_NEW_TO_ESTATE, new_to_estate)

            if (unusable > 0):
                logger.warning(
                    "TC6CertificateStage left %d of %d handshakes out of their destination's issuer history for "
                    "want of a key, an issuer, or a usable event time.",
                    unusable,
                    rows)

            if (unordered > 0):
                logger.warning(
                    "TC6CertificateStage saw %d of %d handshakes arrive earlier than their destination's "
                    "previous one; they did not enter the reference.",
                    unordered,
                    rows)

        return message

    def _classify_destinations(self, df, ip_module, rows: int) -> list:
        """Whether each row's destination is globally routable, or `None` where it cannot be decided.

        Classified through `parsers/ip.py` rather than against a copy of the reserved ranges, so this stage and
        `TC3ReachStage` cannot come to disagree about what counts as leaving the estate.
        """
        if (self._destination_column not in df.columns):
            return [None] * rows

        addresses = [normalize_text(value) for value in to_host_list(df, self._destination_column)]
        usable = [value for value in addresses if value is not None]

        if (len(usable) == 0):
            return [None] * rows

        import pandas as pd  # pylint: disable=import-outside-toplevel

        verdicts = dict(zip(usable, list(ip_module.is_global(pd.Series(usable)))))

        return [None if value is None else bool(verdicts.get(value)) for value in addresses]

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
