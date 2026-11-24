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
"""
What text an agent reads or writes is searched for: instructions aimed at a model, characters a person cannot see,
channels that carry data out, and credentials.

Every function here is handed text that must be treated as hostile, and none of them acts on it. `defang` makes a
quoted piece of it safe to show to a person or to hand to another model: hidden characters become visible escapes,
links stop being links, and the quote is cut short. Matched credentials are never quoted, only their kind and length.
"""

import base64
import binascii
import dataclasses
import re

DIRECTIVES = (
    ("instruction_override",
     3,
     re.compile(
         r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?\b(?:previous|prior|above|earlier|all|any|"
         r"your|the)\b[^.\n]{0,20}?\b(?:instructions?|prompts?|rules|directions|guidelines|guardrails)\b",
         re.I)),
    ("role_reassignment",
     2,
     re.compile(
         r"\byou are (?:now|no longer)\b|\bfrom now on,? you\b|\bact as (?:an? )?(?:unrestricted|jailbroken|"
         r"dan\b|developer mode)",
         re.I)),
    ("concealment",
     3,
     re.compile(
         r"\b(?:do not|don't|never|without)\s+(?:tell|telling|inform|informing|mention|mentioning|reveal|revealing|"
         r"show|showing|alert|alerting|notify|notifying)\b[^.\n]{0,30}?\b(?:the\s+)?(?:user|human|operator|developer)",
         re.I)),
    ("role_marker",
     2,
     re.compile(
         r"<\|(?:system|im_start|im_end)\|>|\[/?(?:SYSTEM|INST)\]|<\s*/?\s*(?:important|system|instructions?|"
         r"admin)\s*>|^\s*(?:system|assistant)\s*:",
         re.I | re.M)),
    ("tool_directive",
     2,
     re.compile(
         r"\bbefore (?:using|calling|running|invoking) (?:this|any|the|other) tools?\b|\b(?:call|invoke|use|run|"
         r"execute)\s+(?:the\s+)?[`'\"]?[a-z][a-z0-9_.-]{2,40}[`'\"]?\s+tool\b",
         re.I)),
    ("sensitive_target",
     2,
     re.compile(
         r"~/\.ssh\b|\bid_(?:rsa|ed25519|ecdsa)\b|\.aws/credentials|(?<![\w.])\.env\b|\.npmrc\b|\.pypirc\b|"
         r"\.git-credentials\b|\.kube/config\b|\.docker/config\.json|/etc/shadow\b|\bLogin Data\b|\bkeychain\b",
         re.I)),
    ("exfiltration_instruction",
     3,
     re.compile(
         r"\b(?:send|post|upload|forward|exfiltrate|transmit|append|include|encode|copy|mirror|back up)\b"
         r"[^.\n]{0,60}?\b(?:to|at|into|in)\b[^.\n]{0,40}?"
         r"(?:https?://|webhook|pastebin|ngrok|requestbin|\bthe url\b|query (?:string|parameter))",
         re.I)),
)
"""Phrasings that address a model rather than a reader, each with the weight it adds to a score."""

EXFIL_DOMAINS = re.compile(
    r"\b(?:webhook\.site|requestbin\.(?:com|net)|[a-z0-9-]{1,63}\.pipedream\.net|"
    r"[a-z0-9-]{1,63}\.ngrok(?:-free)?\.(?:app|io|dev)|[a-z0-9-]{1,63}\.trycloudflare\.com|pastebin\.com|"
    r"transfer\.sh|discord(?:app)?\.com/api/webhooks|hooks\.slack\.com|"
    r"[a-z0-9-]{1,63}\.oast\.(?:fun|live|site|me|pro|online)|[a-z0-9-]{1,63}\.interact\.sh|"
    r"[a-z0-9-]{1,63}\.burpcollaborator\.net)",
    re.I)
"""Hosts that exist to receive data from somewhere they should not: request catchers, tunnels, pastes, webhooks."""

MARKDOWN_IMAGE_WITH_DATA = re.compile(r"!\[[^\]]{0,200}\]\(\s{0,10}(https?://[^)\s?]{1,512}\?[^)\s]{0,2048})\s{0,10}\)",
                                      re.I)
"""An image whose address carries a query string: rendering it sends whatever the string holds.

Every repetition is bounded and no two adjacent ones can match the same character, so hostile text cannot make the
match backtrack; whether the query carries a value (`=`) is checked after the match."""

HTML_IMAGE_WITH_DATA = re.compile(
    r"<img\b[^>]{0,200}?\bsrc\s{0,5}=\s{0,5}[\"']?(https?://[^\"'>\s?]{1,512}\?[^\"'>\s]{0,2048})", re.I)

BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")

SECRETS = (
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----")),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai_style_api_key", re.compile(r"\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
)
"""Credential shapes, matched by their published prefixes and lengths."""

ZERO_WIDTH = frozenset([*range(0x200B, 0x2010), *range(0x2060, 0x2065), 0xFEFF, 0x180E, 0x00AD])
BIDI_CONTROLS = frozenset([*range(0x202A, 0x202F), *range(0x2066, 0x206A)])
TAG_CHARACTERS = range(0xE0000, 0xE0080)
"""Unicode tag characters: invisible, and each one maps to an ASCII character, so a sentence can hide in them."""
VARIATION_SELECTORS_SUPPLEMENT = range(0xE0100, 0xE01F0)


@dataclasses.dataclass(frozen=True)
class Indicator:
    """One thing found in a piece of text, with the weight it adds to the text's score."""

    kind: str
    name: str
    weight: int
    offset: int
    evidence: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _hidden_kind(codepoint: int) -> str | None:
    if (codepoint in ZERO_WIDTH):
        return "zero_width"
    if (codepoint in BIDI_CONTROLS):
        return "bidi_control"
    if (codepoint in TAG_CHARACTERS):
        return "tag_character"
    if (codepoint in VARIATION_SELECTORS_SUPPLEMENT):
        return "variation_selector"
    return None


def defang(text: str, limit: int = 160) -> str:
    """
    A quotable form of hostile text: hidden characters shown as `<U+XXXX>`, links broken, whitespace collapsed, and
    cut to `limit` characters.
    """
    shown = []

    for character in text:
        if (_hidden_kind(ord(character)) is not None):
            shown.append(f"<U+{ord(character):04X}>")
        else:
            shown.append(character)

    quoted = re.sub(r"\s+", " ", "".join(shown)).strip()
    quoted = re.sub(r"(?i)\bhttp(s?)://", r"hxxp\1://", quoted)
    quoted = re.sub(r"(?i)(hxxps?://)([^/\s?#]+)", lambda m: m.group(1) + m.group(2).replace(".", "[.]"), quoted)

    return quoted if len(quoted) <= limit else quoted[:limit - 1] + "…"


def _context(text: str, start: int, end: int, margin: int = 40) -> str:
    return defang(text[max(0, start - margin):end + margin])


def decode_tag_smuggling(text: str) -> str:
    """The ASCII a run of Unicode tag characters spells, which a reader of the text does not see."""
    return "".join(chr(ord(c) - 0xE0000) for c in text if ord(c) in TAG_CHARACTERS and 0x20 <= ord(c) - 0xE0000 < 0x7F)


def hidden_characters(text: str) -> list[Indicator]:
    """Characters that render as nothing or reorder what renders, grouped into one indicator per run."""
    found = []
    index = 0

    while index < len(text):
        kind = _hidden_kind(ord(text[index]))

        # A byte-order mark at the very start of a document is ordinary.
        if (kind is None or (index == 0 and text[index] == "\ufeff")):
            index += 1
            continue

        end = index

        while end < len(text) and _hidden_kind(ord(text[end])) == kind:
            end += 1

        run = end - index
        weight = 3 if kind == "tag_character" else (2 if run >= 3 or kind == "bidi_control" else 1)
        found.append(Indicator("hidden_character", kind, weight, index, f"{run} x {kind} at offset {index}"))
        index = end

    return found


def directives(text: str) -> list[Indicator]:
    found = []

    for (name, weight, pattern) in DIRECTIVES:
        for match in pattern.finditer(text):
            found.append(Indicator("directive", name, weight, match.start(), _context(text, match.start(),
                                                                                      match.end())))

    return found


def exfiltration_channels(text: str) -> list[Indicator]:
    found = []

    for pattern in (MARKDOWN_IMAGE_WITH_DATA, HTML_IMAGE_WITH_DATA):
        for match in pattern.finditer(text):
            if ("=" not in match.group(1).split("?", 1)[1]):
                continue

            found.append(Indicator("exfil_channel", "image_with_query_data", 3, match.start(), defang(match.group(1))))

    for match in EXFIL_DOMAINS.finditer(text):
        found.append(Indicator("exfil_channel", "collection_host", 2, match.start(), defang(match.group(0))))

    return found


def find_secrets(text: str) -> list[Indicator]:
    """Credentials in the text. The evidence names the kind and length and never quotes the value."""
    found = []

    for (name, pattern) in SECRETS:
        for match in pattern.finditer(text):
            found.append(
                Indicator("secret", name, 3, match.start(), f"{name}, {len(match.group(0))} characters, redacted"))

    return found


def encoded_directives(text: str) -> list[Indicator]:
    """Base64 runs that decode to text carrying a directive: instructions hidden from a reader but not a model."""
    found = []

    for match in BASE64_RUN.finditer(text):
        blob = match.group(0)

        try:
            decoded = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=False).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            continue

        if (directives(decoded)):
            found.append(Indicator("encoded_directive", "base64", 3, match.start(), defang(decoded)))

    return found


def scan_text(text: str) -> list[Indicator]:
    """Every indicator in `text`, including directives spelled in tag characters, ordered by where each starts."""
    if (not text):
        return []

    found = [*hidden_characters(text), *directives(text), *exfiltration_channels(text), *encoded_directives(text)]
    smuggled = decode_tag_smuggling(text)

    for indicator in directives(smuggled):
        found.append(dataclasses.replace(indicator, kind="smuggled_directive", weight=indicator.weight + 1))

    return sorted(found, key=lambda indicator: (indicator.offset, indicator.kind, indicator.name))


def score(indicators: list[Indicator]) -> int:
    """The weight of a text's indicators, counting each kind and name once so that repetition does not inflate it."""
    distinct = {}

    for indicator in indicators:
        key = (indicator.kind, indicator.name)
        distinct[key] = max(distinct.get(key, 0), indicator.weight)

    return sum(distinct.values())
