#!/usr/bin/env python
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

import pytest

from morpheus.utils.media_type import ARCHIVE
from morpheus.utils.media_type import CATEGORIES
from morpheus.utils.media_type import DOCUMENT
from morpheus.utils.media_type import EXECUTABLE
from morpheus.utils.media_type import IMAGE
from morpheus.utils.media_type import TEXT
from morpheus.utils.media_type import category
from morpheus.utils.media_type import crosses_category
from morpheus.utils.media_type import normalize


@pytest.mark.parametrize(
    ("value", "expected"),
    [("image/png", IMAGE), ("image/jpeg", IMAGE), ("video/mp4", "video"), ("audio/mpeg", "audio"), ("text/html", TEXT),
     ("application/json", TEXT), ("application/zip", ARCHIVE), ("application/gzip", ARCHIVE),
     ("application/pdf", DOCUMENT), ("application/x-dosexec", EXECUTABLE), ("font/woff2", "font"),
     ("application/x-x509-ca-cert", "certificate")])
def test_a_type_lands_in_the_category_it_belongs_to(value: str, expected: str):
    assert category(value) == expected


def test_the_subtype_decides_before_the_top_level():
    # Why the map consults the subtype first. `application/*` is where everything interesting lives, and the
    # top level puts an archive, a document and an executable in one bucket -- which is exactly the traffic
    # R-D-L6-005 is for.
    assert len({category("application/zip"), category("application/pdf"), category("application/x-dosexec")}) == 3


def test_a_re_encoding_does_not_cross_a_boundary():
    # The negative half of the rule, and the reason the comparison is between categories rather than types. An
    # estate has thousands of these: a thumbnailer, a client guessing from the extension, a proxy transcoding.
    assert crosses_category("image/png", "image/jpeg") is False


def test_a_file_hidden_behind_a_declared_image_does_cross_one():
    assert crosses_category("image/png", "application/zip") is True
    assert crosses_category("image/png", "application/x-dosexec") is True


def test_an_archive_and_an_executable_are_not_the_same_finding():
    # Both are containers of code and both are separate categories, because an archive crossing a boundary is
    # smuggling and an executable arriving where an image was declared is worse, and they want different
    # responses.
    assert category("application/zip") != category("application/x-dosexec")
    assert crosses_category("application/zip", "application/x-dosexec") is True


def test_a_charset_parameter_is_not_a_content_change():
    assert normalize("text/html; charset=utf-8") == "text/html"
    assert crosses_category("text/html; charset=utf-8", "text/html") is False


def test_an_unrecognized_type_has_no_category_and_no_verdict():
    # `None` rather than `False`, because a comparison against a category the map could not supply is no answer
    # rather than a negative one, and returning `False` would claim a coverage this map does not have.
    assert category("application/x-invented-here") is None
    assert crosses_category("image/png", "application/x-invented-here") is None
    assert crosses_category("application/x-invented-here", "image/png") is None


def test_something_that_is_not_a_media_type_has_no_category():
    assert category("not-a-media-type") is None
    assert category("") is None
    assert category(None) is None


def test_every_category_the_map_returns_is_one_it_declares():
    from morpheus.utils.media_type import _SUBTYPES  # pylint: disable=import-outside-toplevel
    from morpheus.utils.media_type import _TOP_LEVELS  # pylint: disable=import-outside-toplevel

    assert set(_SUBTYPES.values()) <= set(CATEGORIES)
    assert set(_TOP_LEVELS.values()) <= set(CATEGORIES)


def test_the_map_is_keyed_in_lower_case():
    from morpheus.utils.media_type import _SUBTYPES  # pylint: disable=import-outside-toplevel

    assert all(key == key.lower() for key in _SUBTYPES)


def test_case_is_folded():
    assert category("IMAGE/PNG") == IMAGE
