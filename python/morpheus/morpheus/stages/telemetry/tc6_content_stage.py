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
Whether what the sender said the content was and what the inspection point found it to be are the same kind of
thing.

R-D-L6-005 is a strong tunneling indicator and a weak mismatch detector, and the difference between those two is
the word "category" in the rule. A declared `image/png` detected as `image/jpeg` is a re-encoding, a
misconfigured thumbnailer, or a client that guessed from the extension, and an estate has thousands of them. A
declared `image/png` detected as a zip archive is someone moving a file past a content filter. Reporting both
would bury the second under the first, which is how a detection stops being read.

{py:mod}`~morpheus.utils.media_type` holds the map from type to category, and the map is coarse on purpose: its
only job is to separate "the same kind of thing, encoded differently" from "not the same kind of thing at all".

**The stage is stateless**, and it is the only layer 6 stage that is. The question is about one record, needs no
history, and has no maturity: the first handshake an estate ever sees can answer it. That is also why this rule
and the self-signed one fire immediately while the issuer and cipher rules wait for a reference -- a distinction
worth keeping visible, because "the rule is quiet" means something different in each case.

**An unrecognized type on either side yields no verdict rather than a negative one.** The alternative is the map
claiming a coverage it does not have, and the count on the row is how an estate sees where the map ends.
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
from morpheus.utils.column_assign import assign_nullable_bool_column
from morpheus.utils.column_assign import assign_str_column
from morpheus.utils.column_assign import to_host_list
from morpheus.utils.media_type import category
from morpheus.utils.media_type import crosses_category

logger = logging.getLogger(__name__)

DECLARED_CATEGORY = "content_category_declared"
DETECTED_CATEGORY = "content_category_detected"
CATEGORY_CROSSED = "content_category_crossed"
CATEGORY_UNCLASSIFIED = "content_category_unclassified"


@register_stage("tc6-content")
class TC6ContentStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write the category of each record's declared and detected content types, and whether they disagree.

    Stateless, so it may run at any concurrency.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    declared_column : str, default = "content_type_declared"
        Column holding the type the sender declared.
    detected_column : str, default = "content_type_detected"
        Column holding the type the inspection point found.
    """

    def __init__(self,
                 c: Config,
                 declared_column: str = "content_type_declared",
                 detected_column: str = "content_type_detected"):
        super().__init__(c)

        self._declared_column = declared_column
        self._detected_column = detected_column

        self._needed_columns[DECLARED_CATEGORY] = TypeId.STRING
        self._needed_columns[DETECTED_CATEGORY] = TypeId.STRING
        self._needed_columns[CATEGORY_CROSSED] = TypeId.BOOL8
        self._needed_columns[CATEGORY_UNCLASSIFIED] = TypeId.BOOL8

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        """Stage name."""
        return "tc6-content"

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
        Write the two categories and the verdict between them.

        Parameters
        ----------
        message : `morpheus.messages.ControlMessage` or `morpheus.messages.MessageMeta`
            Incoming handshake or transfer records.

        Returns
        -------
        The input message, with the content columns populated.

        Raises
        ------
        KeyError
            If neither content type column is present. A frame carrying one of the two is answered with no
            verdict on every row, which is correct; a frame carrying neither is a feed that cannot support this
            rule at all, and is worth failing on rather than filling with nulls.
        """
        meta = message.payload() if isinstance(message, ControlMessage) else message

        if (meta is None or meta.count == 0):
            return message

        with meta.mutable_dataframe() as df:
            if (self._declared_column not in df.columns and self._detected_column not in df.columns):
                raise KeyError(f"TC6ContentStage requires at least one of {self._declared_column!r} and "
                               f"{self._detected_column!r}, and the DataFrame carries neither. Available "
                               f"columns: {sorted(df.columns)}")

            rows = meta.count
            declared_values = (to_host_list(df, self._declared_column)
                               if self._declared_column in df.columns else [None] * rows)
            detected_values = (to_host_list(df, self._detected_column)
                               if self._detected_column in df.columns else [None] * rows)

            declared: list = []
            detected: list = []
            crossed: list = []
            unclassified: list = []

            for position in range(rows):
                left = category(declared_values[position])
                right = category(detected_values[position])
                verdict = crosses_category(declared_values[position], detected_values[position])

                declared.append(left)
                detected.append(right)
                crossed.append(verdict)

                # A row where something was declared or detected and the map could not place it. A row carrying
                # neither type is not a gap in the map, it is a record this rule does not apply to.
                present = declared_values[position] is not None or detected_values[position] is not None
                unclassified.append(present and verdict is None)

            assign_str_column(df, DECLARED_CATEGORY, declared)
            assign_str_column(df, DETECTED_CATEGORY, detected)
            assign_nullable_bool_column(df, CATEGORY_CROSSED, crossed)
            assign_nullable_bool_column(df, CATEGORY_UNCLASSIFIED, unclassified)

            unplaced = sum(1 for value in unclassified if value)

            if (unplaced > 0):
                logger.warning(
                    "TC6ContentStage could not place %d of %d records' content types in a category, so "
                    "R-D-L6-005 cannot fire on them. Add the types to morpheus.utils.media_type rather than "
                    "reading the silence as an absence of mismatches.",
                    unplaced,
                    rows)

        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
