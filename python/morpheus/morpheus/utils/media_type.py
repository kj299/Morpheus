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
What kind of thing a media type is, so that a declared type and a detected one can be said to disagree.

R-D-L6-005 fires when `content_type_declared` and `content_type_detected` differ "in a way that crosses a
category boundary, for example declared `image/png` and detected as an archive". The qualifier is the rule. A
declared `image/png` detected as `image/jpeg` is a re-encoding, a misconfigured thumbnailer, or a client that
guessed from the extension, and an estate has thousands of them. A declared `image/png` detected as a zip
archive is someone moving a file past a content filter. Both are mismatches; only one is a detection, and a rule
that reported every mismatch would bury the second under the first.

So the comparison is between categories rather than between types, and this module is the map. The categories
are coarse on purpose -- their job is to separate "the same kind of thing, encoded differently" from "not the
same kind of thing at all", and a finer split would start reporting the first again.

**`ARCHIVE` and `EXECUTABLE` are separate categories even though both are containers of code.** An archive
crossing a boundary is exfiltration or smuggling; an executable arriving where an image was declared is the more
serious finding, and the two want different responses.

**An unrecognized type has no category.** As with the cipher table, a default would either manufacture findings
for every type this map has not heard of or hide real ones. `category` returns `None`, the rule declines, and
the stage counts what it could not classify so that the gap is visible rather than silent.

**The subtype is consulted before the top-level type**, because `application/*` is where the interesting types
live and the top level says almost nothing: `application/zip`, `application/pdf` and `application/x-dosexec`
are an archive, a document and an executable, and treating them as one category would make the rule useless on
exactly the traffic it is for.
"""

import typing

IMAGE = "image"
VIDEO = "video"
AUDIO = "audio"
TEXT = "text"
DOCUMENT = "document"
ARCHIVE = "archive"
EXECUTABLE = "executable"
FONT = "font"
CERTIFICATE = "certificate"

CATEGORIES = (IMAGE, VIDEO, AUDIO, TEXT, DOCUMENT, ARCHIVE, EXECUTABLE, FONT, CERTIFICATE)

_SUBTYPES: dict = {
    # Archives and compression.
    "zip": ARCHIVE,
    "x-zip-compressed": ARCHIVE,
    "gzip": ARCHIVE,
    "x-gzip": ARCHIVE,
    "x-tar": ARCHIVE,
    "x-bzip2": ARCHIVE,
    "x-xz": ARCHIVE,
    "x-7z-compressed": ARCHIVE,
    "x-rar-compressed": ARCHIVE,
    "vnd.rar": ARCHIVE,
    "java-archive": ARCHIVE,

  # Executables and their loaders.
    "x-dosexec": EXECUTABLE,
    "x-msdownload": EXECUTABLE,
    "vnd.microsoft.portable-executable": EXECUTABLE,
    "x-elf": EXECUTABLE,
    "x-executable": EXECUTABLE,
    "x-sharedlib": EXECUTABLE,
    "x-mach-binary": EXECUTABLE,
    "x-msi": EXECUTABLE,

  # Documents, which are containers but not of code.
    "pdf": DOCUMENT,
    "msword": DOCUMENT,
    "rtf": DOCUMENT,
    "vnd.ms-excel": DOCUMENT,
    "vnd.ms-powerpoint": DOCUMENT,
    "vnd.openxmlformats-officedocument.wordprocessingml.document": DOCUMENT,
    "vnd.openxmlformats-officedocument.spreadsheetml.sheet": DOCUMENT,
    "vnd.openxmlformats-officedocument.presentationml.presentation": DOCUMENT,
    "vnd.oasis.opendocument.text": DOCUMENT,

  # Structured text that arrives under `application/`.
    "json": TEXT,
    "xml": TEXT,
    "javascript": TEXT,
    "x-javascript": TEXT,
    "x-www-form-urlencoded": TEXT,
    "sql": TEXT,
    "csv": TEXT,

  # Certificates and keys, which an inspection point sees often enough to name.
    "x-x509-ca-cert": CERTIFICATE,
    "x-pem-file": CERTIFICATE,
    "pkix-cert": CERTIFICATE,
    "pkcs7-mime": CERTIFICATE,
    "x-pkcs12": CERTIFICATE,
}
"""Subtype to category, consulted first because `application/*` carries the types that matter."""

_TOP_LEVELS: dict = {
    "image": IMAGE,
    "video": VIDEO,
    "audio": AUDIO,
    "text": TEXT,
    "font": FONT,
}
"""Top-level type to category, for the levels that are a category on their own."""


def normalize(value: typing.Any) -> typing.Optional[str]:
    """
    A media type reduced to `type/subtype`, lower-cased, or `None` where there is none.

    Parameters
    ----------
    value : any
        The type as the collector reported it, possibly carrying parameters.

    Returns
    -------
    str or None

    Notes
    -----
    Parameters are dropped: `text/html; charset=utf-8` is the same type as `text/html`, and a rule that treated
    a charset change as a content change would fire on every encoding migration in the estate.
    """
    if (value is None):
        return None

    text = str(value).split(";", maxsplit=1)[0].strip().lower()

    return text or None


def category(value: typing.Any) -> typing.Optional[str]:
    """
    The category a media type belongs to, or `None` where this map does not recognize it.

    Parameters
    ----------
    value : any
        The type as the collector reported it.

    Returns
    -------
    str or None
        One of `CATEGORIES`, or `None`.
    """
    normalized = normalize(value)

    if (normalized is None or "/" not in normalized):
        return None

    (top_level, _, subtype) = normalized.partition("/")
    found = _SUBTYPES.get(subtype)

    if (found is not None):
        return found

    return _TOP_LEVELS.get(top_level)


def crosses_category(declared: typing.Any, detected: typing.Any) -> typing.Optional[bool]:
    """
    Whether the declared and detected types are different kinds of thing.

    Parameters
    ----------
    declared : any
        What the sender said the content was.
    detected : any
        What the inspection point found it to be.

    Returns
    -------
    bool or None
        `None` where either side is absent or unrecognized, because a comparison against a category this map
        could not supply is not a negative answer -- it is no answer, and reporting it as `False` would be the
        map claiming a coverage it does not have.
    """
    (left, right) = (category(declared), category(detected))

    if (left is None or right is None):
        return None

    return left != right
