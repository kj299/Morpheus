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
Applies an estate's minimization policy at the boundary, so that "minimize" can be a setting rather than a wish.

The guide leaves retention, lawful basis and minimization to the deploying organization, and that is the right
place for them. But an organization that decided to minimize had nothing to decide *with*: every column this
fork produced went to the SIEM, and the only way to send less was to stop running a stage. This is the mechanism
that was missing. What to apply is still the estate's decision, and `morpheus.utils.personal_data` is the
inventory the decision is made against.

**Placement is not a preference.** This belongs at the end of a segment, beside
{py:class}`~morpheus.stages.output.siem_wire_stage.SiemWireStage`, after every feature has been computed. Almost
every stage in this design keys per-entity state on an identifier, so minimizing earlier does one of two things:
dropping the identifier leaves the stages with nothing to key on, and pseudonymizing it makes every stage key on
a digest, which works and buys nothing -- the state, the window and the score are all still that person's, and
the only thing that changed is that the pipeline's own logs are harder to read.

**Pseudonymizing one name for a person and not the others minimizes nothing**, and that is the easiest mistake
to make here. At layer 5 the principal appears as `user_principal`, again as `entity_key`, and again as
`chain_anchor`; digesting the first and shipping the other two is a policy that looks applied and is not. The
stage checks for it: after the policy runs, a kept column holding a value that a pseudonymized column also held
raises rather than ships.

That check compares values, and its limit is worth knowing before relying on it. It finds the principal's own
string wherever it was copied. It cannot find a *different* identifier for the same person -- the 802.1X
identity the directory resolved is another name for whoever `user_principal` names, and no comparison of values
will ever say so. The inventory in `morpheus.utils.personal_data` is what a policy should be written from; the
check is the thing that catches what writing it from the inventory still missed.

**What this does not do.** It does not make the output anonymous, and nothing downstream should be described
that way. The pseudonyms are stable, so the mapping exists; the binding lookups beside them exist precisely to
turn an address back into a person and will keep doing so; and the estate holds the key. It reduces what a
casual reader of one index can see. It is not a reason to treat what remains as no longer personal data, and it
does nothing at all about retention, which is a property of the index rather than of the record.
"""

import logging
import typing

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.personal_data import AMBIGUOUS
from morpheus.utils.personal_data import BOUNDED_DOMAIN
from morpheus.utils.personal_data import CATEGORIES
from morpheus.utils.personal_data import COLUMNS
from morpheus.utils.personal_data import DEFAULT_DIGEST_LENGTH
from morpheus.utils.personal_data import PERSONAL_CATEGORIES
from morpheus.utils.personal_data import PER_ROW
from morpheus.utils.personal_data import classify
from morpheus.utils.personal_data import pseudonymize

logger = logging.getLogger(__name__)

CLASS_COLUMN = "telemetry_class"
"""Where the stage reads the class from when it is not told one, so the ambiguous names resolve themselves."""


@register_stage("minimize", modes=[], ignore_args=["key"])
class MinimizationStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Drop or pseudonymize columns according to a policy the estate chooses.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    drop : list of str, optional
        Categories from `morpheus.utils.personal_data` or explicit column names to remove entirely. A column
        that is not in the frame is not an error: a policy is written once for an estate and applied to segments
        carrying different classes.
    pseudonymize : list of str, optional
        Categories or column names to replace with keyed digests. Naming a column in
        `morpheus.utils.personal_data.BOUNDED_DOMAIN` raises: its domain is small enough that the mapping falls
        out of the digest frequencies whatever the key is, so the honest choices there are to keep or to drop.
    key : str or bytes, optional
        HMAC key, required when `pseudonymize` names anything. Supply it from the estate's secret store; it is
        never written to the output, and rotating it breaks every join against digests made under the old one.
    telemetry_class : str, optional
        Which class this segment carries. It resolves the three column names that mean different things in
        different classes, and with it a policy written by category is expanded once, at construction. Without
        it the stage reads `telemetry_class` off each frame instead, so one policy can be written for an estate
        and mean the right thing in every segment it runs in. A frame carrying more than one class raises rather
        than picking, and a frame carrying none raises for any category whose membership depends on the class.
    digest_length : int, default = 32
        Hex characters kept from each pseudonym.
    allow_unmasked_copies : bool, default = False
        Ship a batch in which a kept column holds a value one of the pseudonymized columns also held. The check
        exists because that is the normal way this is got wrong, and turning it off should be a decision with a
        reason.
    """

    def __init__(
            self,
            c: Config,
            drop: list[str] = None,
            pseudonymize: list[str] = None,  # pylint: disable=redefined-outer-name
            key: typing.Union[str, bytes] = None,
            telemetry_class: str = None,
            digest_length: int = DEFAULT_DIGEST_LENGTH,
            allow_unmasked_copies: bool = False):
        super().__init__(c)

        drop = [] if drop is None else list(drop)
        masked = [] if pseudonymize is None else list(pseudonymize)

        if (len(drop) == 0 and len(masked) == 0):
            raise ValueError("a minimization policy that names nothing removes nothing; drop the stage instead "
                             "of configuring it to do nothing, so the pipeline says what it does")

        (self._drop_categories, self._drop_columns) = self._split(drop, "drop")
        (self._masked_categories, self._masked_columns) = self._split(masked, "pseudonymize")

        # Validated against what the policy resolves to without knowing the class. The only columns that answer
        # differs by are the three in `AMBIGUOUS`, none of which is a bounded domain, so nothing below can be
        # true of the deferred part and false here.
        self._drop = self._expand(self._drop_categories, self._drop_columns, telemetry_class, strict=False)
        self._masked = self._expand(self._masked_categories, self._masked_columns, telemetry_class, strict=False)

        both = sorted(set(self._drop) & set(self._masked))

        if (len(both) > 0):
            raise ValueError(f"{both} are named for both dropping and pseudonymizing. One of the two is what "
                             f"was meant and the stage will not decide which.")

        refused = sorted(set(self._masked) & BOUNDED_DOMAIN)

        if (len(refused) > 0):
            raise ValueError(f"{refused} have domains fixed by their own definition, so a keyed digest of them "
                             f"is read straight off the frequencies and pseudonymizing them minimizes nothing. "
                             f"Keep them or drop them.")

        if (len(self._masked) > 0 and key is None):
            raise ValueError("pseudonymize needs a key. An unkeyed digest of a principal is recovered from a "
                             "directory, which every estate running this has.")

        if (digest_length <= 0):
            raise ValueError(f"digest_length must be positive, received {digest_length}")

        self._key = key
        self._telemetry_class = telemetry_class
        self._digest_length = digest_length
        self._allow_unmasked_copies = allow_unmasked_copies

    @staticmethod
    def _split(named: list[str], parameter: str) -> tuple:
        """Separate the category names from the column names, refusing anything that is neither."""
        categories = []
        columns = []

        for entry in named:
            if (entry in CATEGORIES):
                categories.append(entry)
            elif (entry in COLUMNS or entry in AMBIGUOUS or entry in PER_ROW):
                columns.append(entry)
            else:
                raise ValueError(f"{parameter} names {entry!r}, which is neither a category nor a column "
                                 f"morpheus.utils.personal_data classifies. Classify it first, so that what it "
                                 f"says about a person is written down before a policy acts on it.")

        return (sorted(set(categories)), sorted(set(columns)))

    @staticmethod
    def _expand(categories: list[str], columns: list[str], telemetry_class: typing.Optional[str],
                strict: bool) -> list[str]:
        """What a policy resolves to for one telemetry class.

        A category means different columns in different classes, so the expansion happens where the class is
        known. With one configured the stage resolves at construction; without one it resolves per batch, off
        the frame, which is what lets a single policy be written for an estate and applied to every segment.

        `strict` is the difference between the two moments. At construction the class may legitimately be
        unknown and the ambiguous names are left out, which is safe because they are checked again per batch.
        Per batch, not knowing is an error: silently omitting exactly the columns whose meaning depends on the
        class would produce a policy that looks applied and is not.
        """
        resolved = set(columns)

        for category in categories:
            resolved.update(column for (column, assigned) in COLUMNS.items() if assigned == category)

            for (column, meanings) in AMBIGUOUS.items():
                if (telemetry_class is not None):
                    if (meanings.get(telemetry_class) == category):
                        resolved.add(column)
                elif (strict and category in meanings.values()):
                    raise ValueError(f"the policy names the category {category!r}, which {column!r} belongs to "
                                     f"in some telemetry classes and not others, and this batch does not say "
                                     f"which class it is. Give the stage a telemetry_class, carry a "
                                     f"{CLASS_COLUMN} column, or name the columns individually.")

        return sorted(resolved)

    @property
    def name(self) -> str:
        """Stage name."""
        return "minimize"

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

    def _class_of(self, df) -> typing.Optional[str]:
        """The class this frame carries, from the configured value or from the frame itself."""
        if (self._telemetry_class is not None):
            return self._telemetry_class

        if (CLASS_COLUMN not in df.columns):
            return None

        present = sorted({value for value in to_host_list(df, CLASS_COLUMN) if value is not None})

        if (len(present) > 1):
            raise ValueError(f"the batch carries telemetry classes {present}. `device_id` and `entity_key` "
                             f"mean different things in each, so a policy cannot be applied to a mixed frame; "
                             f"minimize per segment, where a class is one thing.")

        return present[0] if len(present) == 1 else None

    def _unmasked_copies(self, df, originals: dict, dropping: list[str],
                         telemetry_class: typing.Optional[str]) -> list[tuple]:
        """Kept personal columns still holding a value one of the pseudonymized columns held.

        The values compared against are the ones captured before the digests were written, which is the whole
        point: reading them back off the column afterwards would compare the kept columns against the digests
        and find nothing, every time.

        Compared against the personal columns only. An operational column repeating a principal's name would be
        a stranger problem than this check is for, and scanning every operational column on every batch would
        cost more than it found.
        """
        masked = list(originals)
        hidden: set = set()

        for values in originals.values():
            hidden.update(value for value in values if value is not None)

        if (len(hidden) == 0):
            return []

        found = []

        for column in df.columns:
            if (column in masked or column in dropping):
                continue

            if (column not in PER_ROW):
                try:
                    category = classify(column, telemetry_class)
                except (KeyError, ValueError):
                    continue

                if (category not in PERSONAL_CATEGORIES):
                    continue

            shared = sorted({value for value in to_host_list(df, column) if value in hidden})

            if (len(shared) > 0):
                found.append((column, len(shared)))

        return found

    def on_data(self, message: typing.Union[ControlMessage, MessageMeta]) -> typing.Union[ControlMessage, MessageMeta]:
        """
        Apply the policy to one batch, in place.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming records.

        Returns
        -------
        The same message, with the policy applied.

        Raises
        ------
        ValueError
            If the batch mixes telemetry classes, or if a kept column still holds a value that a pseudonymized
            column held and `allow_unmasked_copies` is off.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            telemetry_class = self._class_of(df)

            # Resolved against this batch's class rather than the one guessed at construction, so a policy
            # written by category means the right thing in each segment it runs in.
            to_drop = self._expand(self._drop_categories, self._drop_columns, telemetry_class, strict=True)
            to_mask = self._expand(self._masked_categories, self._masked_columns, telemetry_class, strict=True)

            both = sorted(set(to_drop) & set(to_mask))

            if (len(both) > 0):
                raise ValueError(f"for telemetry class {telemetry_class!r} the policy resolves {both} into both "
                                 f"dropping and pseudonymizing. One of the two is what was meant.")

            masked = [column for column in to_mask if column in df.columns]

            # Captured before anything is overwritten. The check below compares what the kept columns hold
            # against what the pseudonymized ones held, and after the write those values are gone.
            originals = {column: to_host_list(df, column) for column in masked}

            for column in masked:
                assign_str_column(df, column, pseudonymize(originals[column], self._key, self._digest_length))

            if (len(masked) > 0):
                copies = self._unmasked_copies(df, originals, to_drop, telemetry_class)

                if (len(copies) > 0 and not self._allow_unmasked_copies):
                    named = ", ".join(f"{column} ({count} values)" for (column, count) in copies)
                    raise ValueError(f"{named} still hold values that were pseudonymized elsewhere in the same "
                                     f"row, so the policy has been applied to one name for a person and not to "
                                     f"the others. Add them to the policy, or set allow_unmasked_copies if the "
                                     f"overlap is a coincidence of values rather than the same identity.")

                if (len(copies) > 0):
                    logger.warning(
                        "MinimizationStage shipped a batch where %d kept column(s) repeat a pseudonymized "
                        "value: %s. allow_unmasked_copies is on, so this is a warning rather than a refusal.",
                        len(copies), [column for (column, _) in copies])

            dropped = [column for column in to_drop if column in df.columns]

            if (len(dropped) > 0):
                df.drop(columns=dropped, inplace=True)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
