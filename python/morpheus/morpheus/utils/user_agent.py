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
What a `User-Agent` header claims: which client software sent the request, and which operating system it ran on.

Two features read this. User-agent novelty per client is the guide's layer 7 feature: a workstation whose requests
have always come from one browser and now come from a scripting library is running something it did not run before.
R-B-L4-004 compares the operating system the header claims with the one the connection's TCP stack betrays, which
is the comparison a proxy, a tunnel or a forged header fails.

**A family, not a version.** Browsers update every few weeks, and a novelty feature keyed on the full string would
call every update a new client. The family -- `chrome`, `firefox`, `curl`, `python-requests` -- is what changes when
the software does, and the version is what changes when it is patched.

**The order of the browser tokens is the order the browsers themselves impose.** Every Chromium browser carries a
`Chrome/` token and a `Safari/` token, and Edge carries `Edg/` beside both, so a browser is identified by its most
specific token first: Edge, then Chrome, then Firefox, then Safari. Anything that is not a browser is named by its
own product token, which for a library or a command-line tool is the whole of what it says about itself.

**A tool claims no operating system, and is not given one.** `curl/8.5.0` runs everywhere; reading Linux into it
would make every Windows administrator's curl a stack mismatch. The operating system is `None` unless the header
names one, and R-B-L4-004 compares only where both sides have spoken.
"""

import re
import typing

from morpheus.utils.entity_key import is_missing

BROWSER_TOKENS = (("Edg/", "edge"), ("EdgA/", "edge"), ("Chrome/", "chrome"), ("CriOS/", "chrome"),
                  ("Firefox/", "firefox"), ("FxiOS/", "firefox"), ("Safari/", "safari"))
"""Product tokens a browser is recognized by, most specific first."""

OS_TOKENS = (("iPhone", "ios"), ("iPad", "ios"), ("Android", "android"), ("CrOS", "chromeos"),
             ("Windows NT", "windows"), ("Macintosh", "macos"), ("Mac OS X", "macos"), ("Linux", "linux"), ("X11",
                                                                                                            "linux"))
"""Platform tokens an operating system is recognized by, most specific first: an Android or iOS browser's header also
says Linux or Mac OS X."""

STACK_FAMILY = {
    "windows": "windows",
    "macos": "darwin",
    "ios": "darwin",
    "linux": "linux",
    "android": "linux",
    "chromeos": "linux",
}
"""The TCP stack each operating system runs. iOS shares macOS's, and Android and ChromeOS share Linux's, so a stack
fingerprint can tell a Windows machine from a Linux one but not an Android phone from a Linux server."""

_PRODUCT = re.compile(r"^\s*([A-Za-z][A-Za-z0-9._+-]*)")


def _text(value: typing.Any) -> str:
    """The header as text, stripped; empty for a missing or blank header."""
    if (is_missing(value)):
        return ""

    return str(value).strip()


def user_agent_family(value: typing.Any) -> typing.Optional[str]:
    """
    The client software a `User-Agent` header names.

    Parameters
    ----------
    value : str
        The header, as the proxy logged it.

    Returns
    -------
    str or None
        A browser's family for a `Mozilla/` header that carries a browser token, `mozilla` for one that carries
        none, and otherwise the header's own product token, lowercased. `None` for an empty header.
    """
    text = _text(value)

    if (len(text) == 0):
        return None

    if (text.startswith("Mozilla/")):
        for (token, family) in BROWSER_TOKENS:
            if (token in text):
                return family

        return "mozilla"

    match = _PRODUCT.match(text)

    return match.group(1).lower() if match is not None else None


def user_agent_os(value: typing.Any) -> typing.Optional[str]:
    """
    The operating system a `User-Agent` header claims, or `None` where it claims none.

    Parameters
    ----------
    value : str
        The header, as the proxy logged it.

    Returns
    -------
    str or None
        One of `windows`, `macos`, `ios`, `android`, `chromeos` and `linux`, or `None`.
    """
    text = _text(value)

    if (len(text) == 0):
        return None

    for (token, system) in OS_TOKENS:
        if (token in text):
            return system

    return None


def stack_family_for(system: typing.Any) -> typing.Optional[str]:
    """
    The TCP stack an operating system runs, as `morpheus.utils.stack_fingerprint` names stacks.

    Parameters
    ----------
    system : str
        An operating system as `user_agent_os` names it.

    Returns
    -------
    str or None
        `windows`, `darwin` or `linux`, or `None` for an unrecognized or absent system.
    """
    return STACK_FAMILY.get(system) if isinstance(system, str) else None
