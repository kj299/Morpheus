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

Every function here is handed text that must be treated as hostile, and none of them acts on it. Three promises are
kept here, and `tests/morpheus/determinism/test_agentic_hunt.py` holds each:

- **Nothing quoted carries a credential.** `redact` replaces every credential shape with its kind and length, and
  `defang` and `clean` both redact first, so no caller can quote text without it.
- **Nothing quoted is live.** `defang` shows invisible and control characters as `<U+XXXX>`, breaks links, and cuts
  the quote short. `clean` does the same for names and identifiers, which a hunter copies into a search, without
  rewriting their dots.
- **No text can stall a scan or flood its reader.** No pattern here can be made to backtrack more than linearly,
  matches are capped per pattern, and a very long text is scanned at its head and its tail.

A pattern is evidence, not a verdict, so `assess` grades what it finds: text is only called a likely injection when
indicators of two different categories agree (say, an instruction to the model and a place to send data), or when
one of them was hidden, which ordinary prose never is.
"""

import base64
import binascii
import codecs
import collections
import dataclasses
import hashlib
import html
import itertools
import re
import unicodedata
import urllib.parse

MAX_MATCHES = 50
"""The most matches counted per pattern in one text; a text made of matches costs no more than this."""

MAX_SHOWN = 3
"""The most indicators returned per kind and name; the rest are counted, not listed."""

MAX_SCAN = 1_000_000
"""The most characters scanned in one text: this much of its head and, past that, the same again of its tail."""

# --- Credentials ----------------------------------------------------------------------------------------------

EXAMPLE_VALUES = frozenset(["AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"])
"""Values vendors publish for documentation. They are redacted like any other, and reported as examples."""

SECRETS = (
    ("private_key",
     re.compile(r"-----BEGIN (?:[A-Z0-9]{1,12} ){0,3}PRIVATE KEY(?: BLOCK)?-----"
                r"(?:[^-]{0,8192}-----END (?:[A-Z0-9]{1,12} ){0,3}PRIVATE KEY(?: BLOCK)?-----)?")),
    ("aws_access_key_id", re.compile(r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])")),
    ("aws_secret_access_key",
     re.compile(r"(?i:aws_secret_access_key|secret_?access_?key)[\"']?\s{0,3}[=:]\s{0,3}[\"']?([A-Za-z0-9/+=]{40})"
                r"(?![A-Za-z0-9/+=])")),
    ("github_token", re.compile(r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{40,255})")),
    ("gitlab_token", re.compile(r"(?<![A-Za-z0-9_])glpat-[A-Za-z0-9_-]{20,64}")),
    ("slack_token", re.compile(r"(?<![A-Za-z0-9])xox[abposr]-[A-Za-z0-9-]{10,255}")),
    ("slack_webhook", re.compile(r"hooks\.slack\.com/services/(T[A-Z0-9]{6,16}/B[A-Z0-9]{6,16}/[A-Za-z0-9]{16,64})")),
    ("google_api_key", re.compile(r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9_-])")),
    ("google_oauth_token", re.compile(r"(?<![A-Za-z0-9_])ya29\.[A-Za-z0-9_-]{30,512}")),
    ("anthropic_api_key", re.compile(r"(?<![A-Za-z0-9_-])sk-ant-[A-Za-z0-9_-]{20,255}")),
    ("openai_style_api_key",
     re.compile(r"(?<![A-Za-z0-9_-])sk-(?!ant-)(?:(?:proj|svcacct|admin)-)?[A-Za-z0-9_]{32,255}(?![A-Za-z0-9_-])")),
    ("stripe_key", re.compile(r"(?<![A-Za-z0-9_])[sr]k_live_[A-Za-z0-9]{20,255}")),
    ("npm_token", re.compile(r"(?<![A-Za-z0-9_])npm_[A-Za-z0-9]{36}(?![A-Za-z0-9])")),
    ("pypi_token", re.compile(r"(?<![A-Za-z0-9_])pypi-AgE[A-Za-z0-9_-]{50,512}")),
    ("huggingface_token", re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{30,64}(?![A-Za-z0-9])")),
    ("sendgrid_key", re.compile(r"(?<![A-Za-z0-9_])SG\.[A-Za-z0-9_-]{16,64}\.[A-Za-z0-9_-]{16,64}")),
    ("jwt",
     re.compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{10,2048}\.eyJ[A-Za-z0-9_-]{10,4096}\.[A-Za-z0-9_-]{10,1024}")),
    ("bearer_token", re.compile(r"(?i:\bbearer)\s{1,3}([A-Za-z0-9._~+/=-]{20,2048})")),
    ("url_password", re.compile(r"://[^/\s:@\"']{1,64}:([^@\s/\"']{3,128})@")),
)
"""Credential shapes, matched by their published prefixes and lengths. Where a pattern has a group, the group is the
credential and the rest is its label."""

GENERIC_SECRET = re.compile(
    r"(?i:\b(?:api[_-]?key|secret|token|passw(?:or)?d|auth)[a-z0-9_-]{0,20})[\"']?\s{0,3}[=:]\s{0,3}[\"']?"
    r"([A-Za-z0-9/+_=.-]{20,512})")
"""A long value assigned to a name that says it is a secret. Redacted, since over-redaction costs nothing, but too
loose to report as a leak on its own."""

# --- Characters a reader does not see ---------------------------------------------------------------------------

ZERO_WIDTH = frozenset([*range(0x200B, 0x2010), *range(0x2060, 0x2065), 0xFEFF, 0x180E])
SOFT_HYPHEN = 0x00AD
BIDI_CONTROLS = frozenset([*range(0x202A, 0x202F), *range(0x2066, 0x206A)])
TAG_CHARACTERS = range(0xE0000, 0xE0080)
"""Unicode tag characters: invisible, and each one maps to an ASCII character, so a sentence can hide in them."""
VARIATION_SELECTORS = range(0xFE00, 0xFE10)
VARIATION_SELECTORS_SUPPLEMENT = range(0xE0100, 0xE01F0)
"""Variation selectors: 256 invisible characters in all, so each can stand for one byte of a hidden message."""
OTHER_INVISIBLE = frozenset(
    [0x034F, 0x115F, 0x1160, 0x17B4, 0x17B5, 0x180B, 0x180C, 0x180D, 0x180E, 0x180F, 0x3164, 0xFFA0, 0x2800, 0x061C])
"""Characters that render as nothing or as blank and are in none of the ranges above."""

CONFUSABLES = dict((
    (0x0430, "a"),
    (0x0441, "c"),
    (0x0435, "e"),
    (0x043E, "o"),
    (0x0440, "p"),
    (0x0445, "x"),
    (0x0443, "y"),
    (0x0456, "i"),
    (0x0458, "j"),
    (0x0455, "s"),
    (0x04BB, "h"),
    (0x043A, "k"),
    (0x043C, "m"),
    (0x0442, "t"),
    (0x0432, "b"),
    (0x043D, "h"),
    (0x0410, "A"),
    (0x0412, "B"),
    (0x0421, "C"),
    (0x0415, "E"),
    (0x041D, "H"),
    (0x041A, "K"),
    (0x041C, "M"),
    (0x041E, "O"),
    (0x0420, "P"),
    (0x0422, "T"),
    (0x0425, "X"),
    (0x0406, "I"),
    (0x03BF, "o"),
    (0x03B1, "a"),
    (0x03B5, "e"),
    (0x03B9, "i"),
    (0x03BA, "k"),
    (0x03BD, "v"),
    (0x03C1, "p"),
    (0x03C4, "t"),
    (0x03C5, "u"),
    (0x03C7, "x"),
    (0x0391, "A"),
    (0x0392, "B"),
    (0x0395, "E"),
    (0x0399, "I"),
    (0x039A, "K"),
    (0x039C, "M"),
    (0x039D, "N"),
    (0x039F, "O"),
    (0x03A1, "P"),
    (0x03A4, "T"),
    (0x03A7, "X"),
))
"""Cyrillic and Greek letters drawn like Latin ones, folded to Latin in the copy the directives are matched on."""

# --- Phrasing ---------------------------------------------------------------------------------------------------

ADDRESS = "address"
CONCEAL = "conceal"
TARGET = "target"
EXFIL = "exfiltrate"
HIDDEN = "hidden"
CATEGORIES = (ADDRESS, CONCEAL, TARGET, EXFIL, HIDDEN)
"""What an indicator is evidence of: text that addresses or redirects the model, that asks for secrecy, that names a
credential, that names a way out, or that was hidden from a reader."""

DIRECTIVES = (
    ("instruction_override",
     ADDRESS,
     3,
     re.compile(
         r"\b(?:ignore|disregard|forget|discard)\b[^.\n]{0,40}?\b(?:previous|prior|above|earlier|"
         r"preceding|original|initial|system|all|any|your|the|these|those)\b[^.\n]{0,20}?\b(?:instructions?|prompts?|"
         r"rules|directions|guidelines|guardrails|constraints)\b|\b(?:override|bypass)\b[^.\n]{0,40}?\b(?:previous|"
         r"prior|above|earlier|original|initial|system|all|your)\b[^.\n]{0,20}?\b(?:instructions?|prompts?|guardrails|"
         r"safety\s{1,3}(?:checks?|rules|filters?))\b",
         re.I)),
    ("new_instructions",
     ADDRESS,
     2,
     re.compile(
         r"\b(?:new|updated|revised|additional|real|actual|true)\s{1,3}(?:system\s{1,3})?instructions?\s{0,3}[:\-]|"
         r"\b(?:new|next|updated|real)\s{1,3}(?:task|job|goal|objective|mission)\s{1,3}(?:for\s{1,3}(?:the|you|an?)\b|"
         r"is\b|:)|\byour\s{1,3}(?:new|updated|real|actual|true)\s{1,3}(?:task|job|goal|objective|role|persona|"
         r"instructions?|purpose)\b|\b(?:guidance|instructions?|rules|directions)\s{1,3}(?:you\s{1,3}(?:were|have"
         r"\s{1,3}been)\s{1,3}given|above|so\s{1,3}far)\b[^.\n]{0,30}?\b(?:out\s{1,3}of\s{1,3}date|obsolete|void|"
         r"superseded|no\s{1,3}longer|cancell?ed|invalid)\b|\b(?:previous|prior|earlier|above)\s{1,3}instructions\s{1,3}"
         r"are\s{1,3}(?:void|obsolete|cancell?ed|superseded|invalid|out\s{1,3}of\s{1,3}date)\b|\b(?:set|put)\s{1,3}"
         r"aside\s{1,3}(?:everything|anything|all)\b[^.\n]{0,40}?\b(?:asked|told|instructed|given)\b|\bhigher\s{1,3}"
         r"priority\s{1,3}than\s{1,3}(?:anything|everything|any|all)\b|\bthe\s{1,3}task\s{1,3}above\s{1,3}is\s{1,3}"
         r"(?:complete|done|finished)\b|\bpolicy\s{1,3}update\s{0,3}:",
         re.I)),
    ("role_reassignment",
     ADDRESS,
     2,
     re.compile(
         r"\byou are (?:now|no longer)\b|\bfrom now on,? you\b|\bact as (?:an? )?(?:unrestricted|jailbroken|"
         r"dan\b|developer mode)|\bpretend\s{1,3}(?:you\s{1,3}are|to\s{1,3}be|that\s{1,3}you)\b|\byour\s{1,3}new\s{1,3}"
         r"persona\b|\bact\s{1,3}as\s{1,3}(?:the|an?|my)\b[^.\n]{0,40}?\b(?:admin|root|shell|owner|system|terminal|"
         r"superuser)\b|\b(?:with|has|have)\s{1,3}no\s{1,3}(?:restrictions|rules|limits|guardrails|filters)\b|"
         r"\bin\s{1,3}(?:maintenance|developer|debug|god|admin|unrestricted)\s{0,3}mode\b[^.\n]{0,30}?\byou\b",
         re.I)),
    ("addressed_to_model",
     ADDRESS,
     2,
     re.compile(
         r"\b(?:note|message|attention|instructions?|reminder|notice|task|request)\s{1,3}(?:for|to)\s{1,3}(?:the\s{1,3}|"
         r"any\s{1,3}|all\s{1,3})?(?:coding\s{1,3}|ai\s{1,3})?(?:ai|llm|language model|assistant|agent|chatbot|copilot|"
         r"code\s{1,3}reviewer)s?\b(?!\s{1,3}(?:in|on|at|is|are|was|were|of|that|which|configuration|settings?)\b)|\bif you are an? (?:ai|llm|language model|assistant|agent)\b|"
         r"\bas an ai (?:assistant|agent|model),? you\b|\b(?:attention|important|notice)\s{0,3}[,:]?\s{1,3}(?:ai|llm|"
         r"model|assistant|agent)s?\b|\b(?:ai|llm)\s{1,3}(?:agents?|assistants?|models?|code\s{1,3}reviewers?|systems?|"
         r"tools?)(?:\s{0,3}:\s|\b[^.:\n]{0,40}?\b(?:must|shall|are\s{1,3}required)\b)|\bwhen\s{1,3}an?\s{1,3}(?:ai|llm|"
         r"assistant|agent|language model)\b[^.\n]{0,30}?\b(?:reads?|sees?|process(?:es)?|summari[sz]es?|encounters?)\b|"
         r"(?:^|[.>\n])\s{0,4}(?:model|assistant|agent|ai)\s{0,2}(?:note\s{0,2})?[,:]\s{1,3}(?:your|you|please|"
         r"the\s{1,3}next|collect|after|when|before)\b",
         re.I | re.M)),
    ("role_marker",
     ADDRESS,
     2,
     re.compile(
         r"<\|(?:system|im_start|im_end|start_header_id|end_header_id|eot_id|begin_of_text)\|>|\[/?(?:SYSTEM|INST)\]|"
         r"<[ \t]{0,8}/?[ \t]{0,8}(?:important(?:_note)?|system|instructions?|admin|system-reminder|tool_result|"
         r"function_results)[ \t]{0,8}>|^[ \t]{0,8}(?:system|assistant)[ \t]{0,8}:|"
         r"^[ \t]{0,8}#{1,6}[ \t]{0,3}(?-i:SYSTEM)\b(?![ \t]{1,3}(?-i:REQUIREMENTS|OVERVIEW|DESIGN|ARCHITECTURE))|"
         r"\"role\"\s{0,3}:\s{0,3}\"system\"|(?-i:\b(?:Human|User)):\s[^\n]{0,200}?(?-i:\b(?:Assistant|AI)):\s",
         re.I | re.M)),
    ("tool_mention",
     ADDRESS,
     1,
     re.compile(
         r"\b(?:call|invoke|use|run|execute)\s{1,3}(?:the\s{1,3})?[`'\"]?(?!(?:this|that|any|each|another|other|same|"
         r"following|a|an|the)\b)[a-z][a-z0-9_.-]{2,40}[`'\"]?\s{1,3}tool\b",
         re.I)),
    ("tool_directive",
     ADDRESS,
     2,
     re.compile(
         r"\bbefore (?:using|calling|running|invoking) (?:this|any|the|other) tools?\b|"
         r"\b(?:you\s{1,3})?must\s{1,3}first\s{1,3}(?:call|invoke|use|run|"
         r"read)\b|\bprior\s{1,3}to\s{1,3}(?:any|each|every)\s{1,3}use\s{1,3}of\s{1,3}(?:the|this)\s{1,3}tool\b|"
         r"\balways\s{1,3}(?:set|pass|include|put|fill)\s{1,3}(?:the\s{1,3})?[`'\"]?[a-z_]{1,40}[`'\"]?\s{1,3}(?:field|"
         r"parameter|argument|header)\b|\bpass\s{1,3}(?:its|the|their|this|that)\s{1,3}(?:contents?|result|output|"
         r"value|data)\b[^.\n]{0,30}?\b(?:as|in|into)\s{1,3}(?:the\s{1,3})?[`'\"]?[a-z_]{1,40}[`'\"]?\s{1,3}(?:field|"
         r"parameter|argument)\b",
         re.I)),
    ("concealment",
     CONCEAL,
     3,
     re.compile(
         r"\b(?:do not|don't|never|without)\s{1,3}(?:tell|telling|inform|informing|alert|alerting|notify|notifying|"
         r"warn|warning)\s{1,3}(?:the\s{1,3})?(?:user|human|operator|developer)s?\b(?!\s{1,3}(?:name|menu|id|list|"
         r"interface|profile|table)\b)|\b(?:do not|don't|never|without)\s{1,3}(?:mention|mentioning|reveal|revealing|"
         r"show|showing|disclose|disclosing|say|saying|report|reporting)\b[^\n]{0,40}?\bto\s{1,3}(?:the\s{1,3})?"
         r"(?:user|human|operator|developer)s?\b|\bwithout\s{1,3}(?:the\s{1,3})?(?:user|human|operator)(?:'s)?\s{1,3}"
         r"(?:knowledge|knowing|noticing|consent|approval)\b|\bkeep\s{1,3}(?:this|it)\s{1,3}(?:secret|hidden|"
         r"confidential|between us)\b",
         re.I)),
    ("quiet_action",
     CONCEAL,
     2,
     re.compile(
         r"\b(?:silently|quietly|secretly|covertly|discreetly)\s{1,3}(?:add|send|open|run|include|upload|copy|read|"
         r"post|execute|forward|push|fetch|collect|attach|append|email)\b|\b(?:no|not\s{1,3}any)\s{1,3}need\s{1,3}to"
         r"\s{1,3}(?:mention|tell|inform|report|surface|disclose)\b|\bneed\s{1,3}not\s{1,3}be\s{1,3}(?:surfaced|"
         r"mentioned|reported|disclosed|shown)\b|\bkeep\s{1,3}(?:it|this|that|the\s{1,3}[a-z_]{1,30})\s{1,3}out"
         r"\s{1,3}of\s{1,3}your\s{1,3}(?:explanation|reply|response|summary|answer|output)\b|\b(?:do\s{1,3}not|don't|"
         r"never)\s{1,3}(?:mention|reveal|disclose|surface)\s{1,3}(?:this|it|that|these|any\s{1,3}of\s{1,3}this)\b",
         re.I)),
    ("sensitive_target",
     TARGET,
     2,
     re.compile(
         r"~/\.ssh\b|\bid_(?:rsa|ed25519|ecdsa)\b|\.aws/credentials|(?<![\w.])\.env\b(?!\.(?:example|sample|template|"
         r"dist))|\.npmrc\b|\.pypirc\b|\.git-credentials\b|\.kube/config\b|\.docker/config\.json|/etc/shadow\b|"
         r"\b(?:deploy|ssh|private|cloud|signing)\s{1,3}keys?\b|\b(?:session|auth\w{0,10})\s{1,3}cookies?\b|"
         r"\brepository\s{1,3}secrets\b|\b(?:hidden|system)\s{1,3}prompt\b(?=[^.\n]{0,60}\b(?:reveal|print|show|"
         r"repeat|output|leak|send)\b)|\b(?:reveal|print|show|repeat|output|leak|send)\b[^.\n]{0,60}?\b(?:hidden|"
         r"system)\s{1,3}prompt\b",
         re.I)),
    ("sensitive_topic",
     TARGET,
     1,
     re.compile(
         r"\bapi keys?\b|\baccess tokens?\b|\bsystem prompt\b|\bconversation history\b|\bcredentials\b|"
         r"\benv(?:ironment)?\s{1,3}var(?:iable)?s\b|\b(?:session|auth\w{0,10})\s{1,3}tokens?\b",
         re.I)),
    ("exfiltration_instruction",
     EXFIL,
     3,
     re.compile(
         r"\b(?:send|post|upload|forward|exfiltrate|transmit|leak|email|submit|deliver|copy|mirror|back up)\b"
         r"[^\n\[\]()|]{0,80}?\b(?:to|at|into)\b[^\n\[\]()|]{0,40}?(?:(?<!\]\()https?://|\bwebhook\b|\bpastebin\b|"
         r"\bngrok\b|\brequestbin\b|\b(?:the|this|that|following)\s{1,3}(?:url|endpoint|address|server|link)\b)"
         r"|\b(?:append|include|encode|embed|put|add|attach)\b[^\n]{0,80}?\b(?:in|into|to|as)\b[^\n]{0,40}?"
         r"(?:\bquery (?:string|parameter)s?\b|\b(?:image|link) (?:url|link|address)\b)",
         re.I)),
    ("send_to_address",
     EXFIL,
     2,
     re.compile(
         r"\b(?:send|post|upload|forward|email|push|transmit|submit|bcc)\b[^\n]{0,80}?\b(?:to|at)\s{1,3}"
         r"(?:[\w.+-]{1,40}@)?(?:[a-z0-9-]{1,40}\.){1,4}[a-z]{2,12}\b(?!\.[a-z0-9])|\bset\s{1,3}the\s{1,3}(?:bcc|cc|to)"
         r"\s{1,3}field\s{1,3}to\b|\bcurl\b[^\n]{0,80}?\s(?:-d|--data[a-z-]{0,10}|-F|-T)\s{0,3}@[^\n]{0,80}?https?://|"
         r"\b(?:nslookup|dig|host)\s[^\n]{0,40}?\$\(",
         re.I)),
    ("url_with_data_slot",
     EXFIL,
     2,
     re.compile(
         r"https?://[^\s\"'<>)\]]{1,200}?(?:\{[^{}\n]{0,40}?(?:content|data|conversation|history|secret|token|key|"
         r"password|credential|env|file|output|result|summary|context|prompt|base64|cookie|session|value)[^{}\n]{0,40}\}"
         r"|[?&/#=](?-i:[A-Z]{3,20}(?:_[A-Z]{2,20}){1,4})(?![A-Za-z]))|"
         r"<[^<>\n]{3,80}>\.(?:[a-z0-9-]{1,40}\.){1,4}[a-z]{2,12}\b",
         re.I)),
)
"""Phrasings that address a model rather than a reader: name, category, weight, pattern.

Every gap is bounded and none of two adjacent repetitions can match the same character, so no text makes a pattern
backtrack more than linearly; `test_no_pattern_can_be_stalled` times every one of them on hostile shapes."""

FOREIGN_DIRECTIVES = (
    ("instruction_override",
     ADDRESS,
     3,
     re.compile(
         r"ignora(?:r)?\s{1,3}(?:todas\s{1,3})?las\s{1,3}instrucciones\s{1,3}(?:anteriores|previas)|"
         r"ignore[zr]?\s{1,3}(?:toutes\s{1,3})?les\s{1,3}instructions\s{1,3}pr\u00e9c\u00e9dentes|"
         r"ignorieren?\s{1,3}(?:sie\s{1,3})?alle\s{1,3}(?:vorherigen|bisherigen|vorigen)\s{1,3}anweisungen|"
         r"ignore\s{1,3}todas\s{1,3}as\s{1,3}instru\u00e7\u00f5es\s{1,3}anteriores|"
         r"ignora\s{1,3}tutte\s{1,3}le\s{1,3}istruzioni\s{1,3}precedenti|"
         r"\u0438\u0433\u043d\u043e\u0440\u0438\u0440\u0443\u0439(?:\u0442\u0435)?\s{1,3}\u0432\u0441\u0435\s{1,3}"
         r"\u043f\u0440\u0435\u0434\u044b\u0434\u0443\u0449\u0438\u0435\s{1,3}"
         r"\u0438\u043d\u0441\u0442\u0440\u0443\u043a\u0446\u0438\u0438|"
         r"\u5ffd\u7565(?:\u4e4b\u524d|\u4ee5\u4e0a|\u5148\u524d|\u4e0a\u8ff0)\u7684?(?:\u6240\u6709)?\u7684?"
         r"(?:\u6307\u4ee4|\u6307\u793a|\u8bf4\u660e)|"
         r"\u3053\u308c\u307e\u3067\u306e\u6307\u793a\u3092(?:\u3059\u3079\u3066)?\u7121\u8996|"
         r"\u0627\u0644\u062a\u0639\u0644\u064a\u0645\u0627\u062a\s{1,3}\u0627\u0644\u0633\u0627\u0628\u0642\u0629|"
         r"\uc774\uc804\uc758?\s{0,3}\ubaa8\ub4e0\s{0,3}\uc9c0\uc2dc(?:\uc0ac\ud56d)?\ub97c\s{0,3}\ubb34\uc2dc",
         re.I)),
    ("concealment",
     CONCEAL,
     3,
     re.compile(
         r"no\s{1,3}(?:le\s{1,3})?digas\s{1,3}(?:nada\s{1,3})?al\s{1,3}usuario|"
         r"ne\s{1,3}(?:le\s{1,3})?di(?:s|tes)\s{1,3}(?:rien|pas)\s{1,3}\u00e0\s{1,3}l'utilisateur|"
         r"sagen?\s{1,3}(?:sie\s{1,3})?dem\s{1,3}benutzer\s{1,3}nichts|"
         r"n\u00e3o\s{1,3}(?:conte|diga)\s{1,3}(?:nada\s{1,3})?ao\s{1,3}usu\u00e1rio|"
         r"non\s{1,3}dir(?:e|lo)\s{1,3}(?:nulla\s{1,3})?all'utente|"
         r"\u043d\u0435\s{1,3}\u0441\u043e\u043e\u0431\u0449\u0430\u0439(?:\u0442\u0435)?\s{1,3}"
         r"\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044e|"
         r"\u4e0d\u8981\u544a\u8bc9\u7528\u6237|"
         r"\u30e6\u30fc\u30b6\u30fc\u306b\u306f?\u77e5\u3089\u305b\u306a\u3044|"
         r"\u0644\u0627\s{1,3}\u062a\u062e\u0628\u0631\s{1,3}\u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645|"
         r"\uc0ac\uc6a9\uc790\uc5d0\uac8c\s{0,3}(?:\uc54c\ub9ac\uc9c0|\ub9d0\ud558\uc9c0)\s{0,3}\ub9c8",
         re.I)),
)
"""The stock override and concealment phrasings in Spanish, French, German, Portuguese, Italian, Russian, Chinese,
Japanese, Arabic and Korean. As in English, they catch the phrase an attacker copies and not a paraphrase of it; a
text is read for these before look-alike letters are folded, which would turn Cyrillic into Latin."""

SQUEEZED_DIRECTIVES = (
    ("instruction_override",
     re.compile(r"(?:ignore|disregard|forget)(?:all|any)?(?:of)?(?:the|your|my)?(?:previous|prior|above|earlier)"
                r"(?:system)?(?:instructions|prompts|rules)")),
    ("concealment", re.compile(r"(?:donot|dont|never)(?:tell|inform|notify|alert)(?:the)?(?:user|human|operator)")),
)
"""The two stock phrasings with everything but letters taken out, which is how they read when spaced out letter by
letter, split over list items, table cells or comment lines, or broken by punctuation."""

LEET = str.maketrans("01345@$7", "oieasast")

COLLECTION_HOSTS = re.compile(
    r"(?<![A-Za-z0-9.-])(?:webhook\.site|requestbin\.(?:com|net)|[a-z0-9-]{1,63}\.pipedream\.net|"
    r"[a-z0-9-]{1,63}\.ngrok(?:-free)?\.(?:app|io|dev)|[a-z0-9-]{1,63}\.trycloudflare\.com|pastebin\.com|"
    r"transfer\.sh|discord(?:app)?\.com/api/webhooks|hooks\.slack\.com|"
    r"[a-z0-9-]{1,63}\.oast\.(?:fun|live|site|me|pro|online)|[a-z0-9-]{1,63}\.interact\.sh|"
    r"[a-z0-9-]{1,63}\.burpcollaborator\.net)",
    re.I)
"""Hosts that exist to receive data from somewhere they should not: request catchers, tunnels, pastes, webhooks."""

_ALT = r"(?:[^\]\[\n]|\[[^\]\[\n]{0,100}\]){0,1000}"
MARKDOWN_IMAGE = re.compile(r"!\[" + _ALT + r"\]\(\s{0,10}<?((?:https?:)?//[^)\n>]{1,2048})>?\s{0,10}\)", re.I)
HTML_IMAGE = re.compile(
    r"<(?:img|iframe|video|audio|source|embed|object|link|script|input)\b[^>]{0,300}?\b(?:src|srcset|poster|data|"
    r"href)\s{0,5}=\s{0,5}[\"']?((?:https?:)?//[^\"'>\s]{1,2048})|\burl\(\s{0,5}[\"']?((?:https?:)?//[^\"')\s]{1,2048})",
    re.I)
"""What a renderer loads without being asked: an image, a frame, a media poster, a prefetch, a stylesheet's `url()`."""
MARKDOWN_LINK = re.compile(r"(?<!!)\[" + _ALT + r"\]\(\s{0,10}<?(https?://[^)\n>]{1,2048})>?\s{0,10}\)", re.I)
REFERENCE_DEFINITION = re.compile(r"^[ \t]{0,3}\[[^\]\n]{1,100}\]:[ \t]{1,4}<?(https?://[^\s>]{1,2048})", re.M | re.I)
PLACEHOLDER = re.compile(r"<[^<>\n]{1,60}>|\{\{?[^{}\n]{1,60}\}\}?|\$\{?[A-Za-z_][A-Za-z0-9_]{1,40}\}?|%7B|"
                         r"\[[A-Z_ ]{3,40}\]|(?<![A-Za-z])[A-Z]{3,20}(?:_[A-Z]{2,20}){1,4}(?![A-Za-z])")
"""A slot in an address for the reader to fill: `<contents>`, `{data}`, `$SECRET`, `SESSION_TOKEN`. A model fills it
with what it has."""

MAX_COMMENT = 20_000
HIDDEN_STYLE = re.compile(
    r"<[a-z][a-z0-9]{0,10}\b[^>]{0,300}?style\s{0,3}=\s{0,3}[\"'][^\"'>]{0,200}?(?:display\s{0,3}:\s{0,3}none|"
    r"font-size\s{0,3}:\s{0,3}0|opacity\s{0,3}:\s{0,3}0|visibility\s{0,3}:\s{0,3}hidden)[^>]{0,300}>([^<]{1,2000})",
    re.I)
HIDDEN_ATTRIBUTE = re.compile(r"\b(?:alt|title|aria-label|data-[a-z-]{1,30})\s{0,3}=\s{0,3}\"([^\"\n]{20,2000})\"",
                              re.I)

BASE64_RUN = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{32,4096}={0,2}")
ENCODED_VALUE = re.compile(r"[A-Za-z0-9+/_=-]{64,4096}")
BASE32_RUN = re.compile(r"(?<![A-Z2-7])[A-Z2-7]{40,4096}={0,6}")
UNICODE_ESCAPES = re.compile(r"(?:\\u[0-9a-fA-F]{4}){8,2048}")
MAX_WHOLE_TEXT_DECODE = 100_000
SHORT_BASE64_RUN = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{24,4096}={0,2}")
"""Long enough for the shortest credential worth carrying: a twenty-character key id is twenty-eight in base64."""
HEX_RUN = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}){20,2048}(?![0-9A-Fa-f])")

STRONG_KINDS = frozenset(
    ["smuggled_directive", "encoded_directive", "obfuscated_directive", "hidden_text", "hidden_markup"])
"""Indicators that mean an instruction was hidden from a reader. One of these is enough: prose is never hidden."""

LIKELY_SCORE = 3
HIGH_SCORE = 6


@dataclasses.dataclass(frozen=True)
class Indicator:
    """One thing found in a piece of text, with the weight it adds to the text's score."""

    kind: str
    name: str
    weight: int
    offset: int
    evidence: str
    category: str = ADDRESS
    count: int = 1
    """How many times this kind and name were found; only the first few are listed."""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Assessment:
    """What a text's indicators add up to."""

    score: int
    level: str
    """`none`, `weak` (below the score), `suspicious` (the score from one category) or `likely` (two categories agree,
    or something was hidden)."""
    categories: tuple
    indicators: tuple
    hidden_text: str
    """Text decoded from invisible characters, defanged; empty when there is none."""
    partial: bool
    """Whether the text was longer than `MAX_SCAN` and only its head and tail were scanned."""

    @property
    def likely(self) -> bool:
        return self.level == "likely"

    @property
    def strong(self) -> bool:
        return self.likely and self.score >= HIGH_SCORE


# --- Quoting safely ---------------------------------------------------------------------------------------------


def _is_example(value: str) -> bool:
    return value in EXAMPLE_VALUES


def _credential_spans(text: str, most: int | None = MAX_MATCHES) -> list[tuple]:
    """
    `(start, end, name, value)` for every credential in `text`, longest first where two overlap.

    `most` bounds the matches of each kind, for reporting. Redaction passes `None`: the fifty-first credential in a
    text is as much a credential as the first.
    """
    spans = []

    for (name, pattern) in (*SECRETS, ("generic_secret", GENERIC_SECRET)):
        for match in itertools.islice(pattern.finditer(text), most):
            group = 1 if pattern.groups else 0
            value = match.group(group)

            if (name == "openai_style_api_key"
                    and not (any(c.isdigit() for c in value) and any(c.isupper() for c in value))):
                # A real key mixes cases and digits; `sk-estimator-doc-link` and its like do not.
                continue

            if (name == "generic_secret" and (not any(c.isdigit() for c in value) or value.isdigit())):
                continue

            spans.append((match.start(group), match.end(group), name, value))

    spans.sort(key=lambda span: (span[0], -(span[1] - span[0])))
    kept = []

    for span in spans:
        if (not kept or span[0] >= kept[-1][1]):
            kept.append(span)

    return kept


def _marked(text: str, low: int, high: int, spans: list) -> str:
    """`text[low:high]` with each credential span replaced by its marker; a span cut by either end goes whole."""
    parts = []
    position = low

    for (start, end, name, _) in spans:
        if (end <= position or start >= high):
            continue

        parts.append(text[position:max(position, start)])
        parts.append(f"<redacted:{name}:{end - start}>")
        position = end

    parts.append(text[position:high])

    return "".join(parts)


_ENCODED_RUN = re.compile(r"(?:%[0-9A-Fa-f]{2})+")
MAX_DECODED_REDACTION = 200_000


def _encoded_spans(text: str) -> list[tuple]:
    """
    Credentials that only show once percent-encoding is undone, as spans of `text` itself.

    `d=aws_access_key_id%3DAKIA...` hides the key from a pattern that wants a word boundary before it, and
    `%41KIA...` hides it from any pattern. Each decoded character is traced back to the characters it came from, a
    whole encoded run at a time, so the span covers at least the credential.
    """
    pieces = []
    origin = []
    position = 0

    for run in _ENCODED_RUN.finditer(text):
        pieces.append(text[position:run.start()])
        origin.extend((index, index + 1) for index in range(position, run.start()))
        decoded = urllib.parse.unquote(run.group(0))
        pieces.append(decoded)
        origin.extend([(run.start(), run.end())] * len(decoded))
        position = run.end()

    pieces.append(text[position:])
    origin.extend((index, index + 1) for index in range(position, len(text)))

    return [(origin[start][0], origin[end - 1][1], name, value)
            for (start, end, name, value) in _credential_spans("".join(pieces), None) if end > start]


def redact(text: str) -> str:
    """
    `text` with every credential replaced by `<redacted:kind:length>`, a marker with no space to break a quote on.

    Credentials are looked for in the text as given and, where it is percent-encoded, in what it decodes to.
    """
    redacted = _marked(text, 0, len(text), _credential_spans(text, None))

    if ("%" in redacted and len(redacted) <= MAX_DECODED_REDACTION and _ENCODED_RUN.search(redacted)):
        redacted = _marked(redacted, 0, len(redacted), _encoded_spans(redacted))

    return redacted


def _invisible(codepoint: int) -> bool:
    return (codepoint in ZERO_WIDTH or codepoint == SOFT_HYPHEN or codepoint in BIDI_CONTROLS
            or codepoint in TAG_CHARACTERS or codepoint in VARIATION_SELECTORS
            or codepoint in VARIATION_SELECTORS_SUPPLEMENT or codepoint in OTHER_INVISIBLE)


def _visible(text: str) -> str:
    """`text` with every invisible or control character shown as `<U+XXXX>` and every odd space made a space."""
    shown = []

    for character in text:
        codepoint = ord(character)

        if (character in "\t\n\r"):
            shown.append(" ")
        elif (codepoint < 0x7F and codepoint >= 0x20):
            shown.append(character)
        elif (_invisible(codepoint) or unicodedata.category(character) in ("Cc", "Cf", "Co", "Cs", "Cn")):
            shown.append(f"<U+{codepoint:04X}>")
        elif (unicodedata.category(character) in ("Zs", "Zl", "Zp")):
            shown.append(" ")
        else:
            shown.append(character)

    return "".join(shown)


_SCHEME = re.compile(r"(?i)(?<![a-z0-9])([a-z][a-z0-9+.-]{1,15})://([^/\s?#]{0,255})")
_BARE_LINK = re.compile(
    r"(?i)(?<![a-z0-9@/.])(www)\.(?=[a-z0-9-]{1,63}\.)|(?<![a-z0-9])(mailto|javascript|data|vbscript):")

_BARE_DOT = re.compile(r"(?<!\[)\.(?!\])")


def _break_scheme(match: re.Match) -> str:
    """Break one link. Breaking twice changes nothing, so a quote can pass through `defang` and then `clean`."""
    scheme = match.group(1)
    lowered = scheme.lower()
    host = _BARE_DOT.sub("[.]", match.group(2))

    if (lowered in ("http", "https")):
        return "hxxp" + scheme[4:] + "://" + host

    if (lowered in ("hxxp", "hxxps")):
        return scheme + "://" + host

    return f"{scheme}[:]//{host}"


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


def defang(text: str, limit: int = 160) -> str:
    """
    A quotable form of hostile text: credentials redacted, invisible and control characters shown as `<U+XXXX>`,
    links of every scheme broken, whitespace collapsed, and cut to `limit` characters.
    """
    quoted = re.sub(r" {2,}", " ", _visible(redact(str(text)))).strip()
    quoted = _SCHEME.sub(_break_scheme, quoted)
    quoted = _BARE_LINK.sub(lambda m: "www[.]" if m.group(1) else f"{m.group(2)}[:]", quoted)

    return _cut(quoted, limit)


def clean(text, limit: int = 200) -> str:
    """
    A name or identifier made safe to show and still usable to search with: credentials redacted, invisible and
    control characters shown, links broken, and cut to `limit`. Other dots and slashes are left alone, so a path
    stays a path.
    """
    cleaned = re.sub(r" {2,}", " ", _visible(redact(str(text)))).strip()
    cleaned = _SCHEME.sub(_break_scheme, cleaned)

    return _cut(_BARE_LINK.sub(lambda m: "www[.]" if m.group(1) else f"{m.group(2)}[:]", cleaned), limit)


CREDENTIAL_REACH = 9000
"""Further than any credential pattern reaches: a private key block is the longest, and it is bounded below this."""


def _context(text: str, start: int, end: int, margin: int = 40) -> str:
    """
    A quotable window of `text` around `[start, end)`.

    A credential the window's edge would cut through is redacted whole first: a slice taken before redaction would
    leave half a key, which no pattern recognises and the quote would then carry.
    """
    low = max(0, start - margin)
    high = min(len(text), end + margin)
    base = max(0, low - CREDENTIAL_REACH)
    spans = [(first + base, last + base, name, value)
             for (first, last, name, value) in _credential_spans(text[base:high + CREDENTIAL_REACH], None)]

    return defang(_marked(text, low, high, spans))


# --- Hidden text ------------------------------------------------------------------------------------------------


def decode_tag_smuggling(text: str) -> str:
    """The ASCII a run of Unicode tag characters spells, which a reader of the text does not see."""
    return "".join(chr(ord(c) - 0xE0000) for c in text if ord(c) in TAG_CHARACTERS and 0x20 <= ord(c) - 0xE0000 < 0x7F)


def _printable_ascii(data: bytes) -> str:
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError:
        return ""

    return decoded if len(decoded) >= 6 and sum(c.isprintable() or c in "\n\t" for c in decoded) >= len(decoded) * 0.9 \
        else ""


def decode_variation_selectors(text: str) -> str:
    """The text spelled by variation selectors used as bytes (FE00-FE0F are 0-15, E0100-E01EF are 16-255)."""
    data = bytearray()

    for character in text:
        codepoint = ord(character)

        if (codepoint in VARIATION_SELECTORS_SUPPLEMENT):
            data.append(codepoint - 0xE0100 + 16)
        elif (codepoint in VARIATION_SELECTORS and len(data) > 0):
            data.append(codepoint - 0xFE00)

    return _printable_ascii(bytes(data)) if len(data) >= 6 else ""


def decode_zero_width_binary(text: str) -> str:
    """The text spelled by two zero-width characters used as bits, eight to a character, in either polarity."""
    bits = [character for character in text if character in "\u200b\u200c"]

    if (len(bits) < 48):
        return ""

    for one in ("\u200c", "\u200b"):
        data = bytearray()

        for start in range(0, len(bits) - 7, 8):
            byte = 0

            for bit in bits[start:start + 8]:
                byte = (byte << 1) | (1 if bit == one else 0)

            data.append(byte)

        decoded = _printable_ascii(bytes(data))

        if (decoded):
            return decoded

    return ""


def _benign_invisible(text: str, index: int) -> bool:
    """Whether the invisible character at `index` is doing its ordinary job: joining emoji or shaping a script."""
    codepoint = ord(text[index])
    previous = ord(text[index - 1]) if index > 0 else 0

    if (codepoint == 0xFEFF and index == 0):
        return True

    if (codepoint in VARIATION_SELECTORS):
        return previous >= 0x203C  # after a symbol or an emoji, where it selects the presentation

    if (codepoint == 0x200D):
        return previous >= 0x2190 or previous in VARIATION_SELECTORS or 0x0600 <= previous <= 0x0DFF

    if (codepoint == 0x200C):
        return 0x0600 <= previous <= 0x0DFF

    return False


def _flag_tags(text: str) -> set:
    """Indexes of tag characters that are part of an emoji flag: a black flag, tags, and the cancel tag."""
    indexes = set()
    position = text.find("\U0001F3F4")

    while position != -1:
        end = position + 1

        while end < len(text) and ord(text[end]) in TAG_CHARACTERS and end - position <= 8:
            end += 1

        if (end > position + 1 and ord(text[end - 1]) == 0xE007F):
            indexes.update(range(position + 1, end))

        position = text.find("\U0001F3F4", end)

    return indexes


def hidden_characters(text: str) -> list[Indicator]:
    """Characters that render as nothing or reorder what renders: one indicator per kind, counting its characters."""
    flags = _flag_tags(text) if "\U0001F3F4" in text else set()
    first = {}
    counts = collections.Counter()

    for (index, character) in enumerate(text):
        codepoint = ord(character)

        if (codepoint < 0xAD):
            continue

        if (codepoint in ZERO_WIDTH):
            kind = "zero_width"
        elif (codepoint in BIDI_CONTROLS):
            kind = "bidi_control"
        elif (codepoint in TAG_CHARACTERS):
            kind = "tag_character"
        elif (codepoint in VARIATION_SELECTORS_SUPPLEMENT or codepoint in VARIATION_SELECTORS):
            kind = "variation_selector"
        else:
            continue

        if (index in flags or _benign_invisible(text, index)):
            continue

        counts[kind] += 1
        first.setdefault(kind, index)

    found = []

    for (kind, count) in sorted(counts.items()):
        if (kind == "tag_character"):
            weight = 3
        elif (kind == "bidi_control"):
            weight = 2
        elif (kind == "variation_selector"):
            weight = 2 if count >= 4 else 0
        else:
            weight = 2 if count >= 3 else 1

        if (weight > 0):
            found.append(
                Indicator("hidden_character",
                          kind,
                          weight,
                          first[kind],
                          f"{count} x {kind}, the first at offset {first[kind]}",
                          HIDDEN,
                          count))

    return found


def hidden_text(text: str) -> list[Indicator]:
    """Whole messages spelled in invisible characters, decoded; and the directives found in them."""
    found = []
    channels = (("tag_characters", decode_tag_smuggling), ("variation_selectors", decode_variation_selectors),
                ("zero_width_binary", decode_zero_width_binary))

    for (name, decode) in channels:
        decoded = decode(text)

        if (len(decoded) < 6):
            continue

        found.append(Indicator("hidden_text", name, 3, 0, defang(decoded), HIDDEN))

        for indicator in directives(decoded):
            found.append(dataclasses.replace(indicator, kind="smuggled_directive", weight=indicator.weight + 1))

    return found


# --- Directives -------------------------------------------------------------------------------------------------


def matching_view(text: str) -> str:
    """
    The text as a model would read it rather than as it is stored: entities decoded, compatibility forms and
    look-alike letters folded to plain Latin, and invisible characters removed.
    """
    view = html.unescape(text) if "&" in text else text
    view = unicodedata.normalize("NFKC", view)

    if (not view.isascii()):
        # A dot under a letter, or any other mark laid over it, does not change what a model reads.
        view = "".join(character for character in unicodedata.normalize("NFKD", view)
                       if not _invisible(ord(character)) and unicodedata.category(character) != "Mn")
        view = view.translate(CONFUSABLES)

    if (view.count("%") >= 3):
        view = urllib.parse.unquote(view)

    return view


def directives(text: str) -> list[Indicator]:
    found = []

    for (name, category, weight, pattern) in DIRECTIVES:
        matches = list(itertools.islice(pattern.finditer(text), MAX_MATCHES))

        for match in matches[:MAX_SHOWN]:
            found.append(
                Indicator("directive",
                          name,
                          weight,
                          match.start(),
                          _context(text, match.start(), match.end()),
                          category,
                          len(matches)))

    return found


def foreign_directives(text: str) -> list[Indicator]:
    """The stock phrasings in other languages, looked for before look-alike letters are folded."""
    found = []
    view = text if text.isascii() else unicodedata.normalize("NFKC", text)

    for (name, category, weight, pattern) in FOREIGN_DIRECTIVES:
        match = pattern.search(view)

        if (match):
            found.append(
                Indicator("directive",
                          name,
                          weight,
                          match.start(),
                          _context(view, match.start(), match.end()),
                          category))

    return found


def squeezed(text: str, leet: bool = False) -> str:
    """The letters of a text and nothing else, in lower case; with `leet`, digits read as the letters they stand for."""
    view = text.lower()

    if (leet):
        view = view.translate(LEET)

    return "".join(character for character in view if "a" <= character <= "z")


def obfuscated_directives(text: str, plain: list[Indicator]) -> list[Indicator]:
    """Directives that appear only once the text is read as a model reads it: the obfuscation is the evidence."""
    view = matching_view(text)
    seen = {indicator.name for indicator in plain}
    found = []

    if (view != text):
        found = [
            dataclasses.replace(indicator, kind="obfuscated_directive", weight=indicator.weight + 1)
            for indicator in directives(view) if indicator.name not in seen
        ]
        seen |= {indicator.name for indicator in found}

    # Spelled out letter by letter, split across list items or table cells, or written with digits for letters.
    for leet in (False, True):
        letters = squeezed(view[:MAX_WHOLE_TEXT_DECODE], leet)

        for (name, pattern) in SQUEEZED_DIRECTIVES:
            if (name not in seen and pattern.search(letters)):
                seen.add(name)
                found.append(
                    Indicator("obfuscated_directive",
                              name,
                              4,
                              0,
                              "the phrase is spelled across punctuation, lines or digits",
                              CONCEAL if name == "concealment" else ADDRESS))

    return found


def html_comments(text: str):
    """
    `(offset, inner text)` of each HTML comment, an unclosed last one included: a renderer hides everything after
    an opener that is never closed. Found by searching, not by pattern, so a page of openers costs one pass.
    """
    position = 0

    while (True):
        start = text.find("<!--", position)

        if (start == -1):
            return

        end = text.find("-->", start + 4)

        if (end == -1):
            yield (start, text[start + 4:start + 4 + MAX_COMMENT])

            return

        if (end - start <= MAX_COMMENT):
            yield (start, text[start + 4:end])

        position = end + 3


def hidden_markup(text: str) -> list[Indicator]:
    """Directives inside an HTML comment, an element styled not to show, or an attribute a reader never sees."""
    found = []

    hidden = [("html_comment", offset, inner) for (offset, inner) in itertools.islice(html_comments(text), MAX_MATCHES)]

    for (name, pattern) in (("hidden_element", HIDDEN_STYLE), ("attribute", HIDDEN_ATTRIBUTE)):
        hidden.extend(
            (name, match.start(), match.group(1)) for match in itertools.islice(pattern.finditer(text), MAX_MATCHES))

    for (name, offset, inner) in hidden:
        if (not any(indicator.name == name for indicator in found) and directives(matching_view(inner))):
            found.append(Indicator("hidden_markup", name, 2, offset, defang(inner), HIDDEN))

    return found


def _carries_data(url: str) -> bool:
    """Whether an address has room for a model to put data in: a slot to fill, or a long or encoded value."""
    if (PLACEHOLDER.search(url) or " " in url.strip() or _credential_spans(url)):
        return True

    try:
        query = urllib.parse.urlsplit(url if "://" in url else "https:" + url).query
    except ValueError:
        return False

    # A long value that is prose or one encoded run is data. A long value that is another address is a badge
    # pointing at its own definition.
    return any(
        len(value) >= 64 and "://" not in value and (" " in value or ENCODED_VALUE.fullmatch(value))
        for values in urllib.parse.parse_qs(query).values() for value in values)


def _address(match: re.Match) -> str:
    return next(group for group in match.groups() if group)


def exfiltration_channels(text: str) -> list[Indicator]:
    """
    Ways out that need no tool call: something a renderer loads by itself, or a link, whose address a model can fill
    with data, and hosts that exist to collect it. A resource with an ordinary query string, which every status
    badge has, is counted and weighs 1.
    """
    found = []
    images = [
        match for pattern in (MARKDOWN_IMAGE, HTML_IMAGE)
        for match in itertools.islice(pattern.finditer(text), MAX_MATCHES)
    ]
    loaded = [match for match in images if _carries_data(_address(match)) or COLLECTION_HOSTS.search(_address(match))]
    plain = [match for match in images if match not in loaded and "?" in _address(match) and "=" in _address(match)]

    for (name, weight, matches) in (("image_with_data_slot", 3, loaded), ("image_with_query_data", 1, plain)):
        for match in matches[:MAX_SHOWN]:
            found.append(
                Indicator("exfil_channel", name, weight, match.start(), defang(_address(match)), EXFIL, len(matches)))

    links = [
        match for pattern in (MARKDOWN_LINK, REFERENCE_DEFINITION)
        for match in itertools.islice(pattern.finditer(text), MAX_MATCHES) if PLACEHOLDER.search(match.group(1))
    ]

    for match in links[:MAX_SHOWN]:
        found.append(
            Indicator("exfil_channel",
                      "link_with_data_slot",
                      2,
                      match.start(),
                      defang(match.group(1)),
                      EXFIL,
                      len(links)))

    hosts = list(itertools.islice(COLLECTION_HOSTS.finditer(text), MAX_MATCHES))

    for match in hosts[:MAX_SHOWN]:
        found.append(
            Indicator("exfil_channel", "collection_host", 2, match.start(), defang(match.group(0)), EXFIL, len(hosts)))

    return found


def find_secrets(text: str) -> list[Indicator]:
    """
    Credentials in the text. The evidence names the kind and length and never quotes the value. A value a vendor
    publishes as an example is named as one; a long value merely assigned to a secret-sounding name is not reported.
    """
    found = []

    for (start, end, name, value) in _credential_spans(text):
        if (name == "generic_secret"):
            continue

        example = _is_example(value)
        found.append(
            Indicator(
                "secret",
                "published_example_key" if example else name,
                0 if example else 3,
                start,
                f"{name}, {end - start} characters, redacted" + (" (a published example value)" if example else ""),
                TARGET))

    return found


def _readable(data: bytes) -> str:
    """Bytes as the text they most plausibly are: UTF-8, UTF-16 where every other byte is zero, or nothing."""
    if (len(data) >= 8 and data[1::2].count(0) >= len(data) // 2 - 1):
        try:
            return data.decode("utf-16-le")
        except UnicodeDecodeError:
            return ""

    decoded = data.decode("utf-8", errors="ignore")

    # A few stray bytes ahead of the text do not make it unreadable; mostly stray bytes do.
    return decoded if len(decoded) >= len(data) * 0.8 else ""


def _decoded(blob: str) -> str:
    padded = blob.replace("-", "+").replace("_", "/")

    try:
        decoded = _readable(base64.b64decode(padded + "=" * (-len(padded) % 4), validate=False))
    except (binascii.Error, ValueError):
        return ""

    # Encoded twice is still encoded.
    if (BASE64_RUN.fullmatch(decoded.strip())):
        try:
            inner = decoded.strip().replace("-", "+").replace("_", "/")
            decoded = _readable(base64.b64decode(inner + "=" * (-len(inner) % 4), validate=False)) or decoded
        except (binascii.Error, ValueError):
            pass

    return decoded


STOCK_DIRECTIVES = DIRECTIVES[0][3], next(pattern for (name, _, _, pattern) in DIRECTIVES if name == "concealment")


def encoded_directives(text: str) -> list[Indicator]:
    """
    Text that carries a directive once decoded: base64 (twice over, URL-safe, or of UTF-16), base32, hex, `\\u`
    escapes, and the whole text in ROT13 or written backwards. Instructions hidden from a reader, not from a model.
    """
    found = []
    joined = re.sub(r"(?<=[A-Za-z0-9+/_-])\n(?=[A-Za-z0-9+/_=-])", "", text) if "\n" in text else text
    candidates = [("base64", match.start(), _decoded(match.group(0)))
                  for match in itertools.islice(BASE64_RUN.finditer(joined), MAX_MATCHES)]

    for match in itertools.islice(HEX_RUN.finditer(text), MAX_MATCHES):
        try:
            candidates.append(("hex", match.start(), bytes.fromhex(match.group(0)).decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            continue

    for match in itertools.islice(BASE32_RUN.finditer(text), MAX_MATCHES):
        blob = match.group(0).rstrip("=")

        try:
            candidates.append(("base32", match.start(), _readable(base64.b32decode(blob + "=" * (-len(blob) % 8)))))
        except (binascii.Error, ValueError):
            continue

    for match in itertools.islice(UNICODE_ESCAPES.finditer(text), MAX_MATCHES):
        candidates.append(
            ("unicode_escapes", match.start(), "".join(chr(int(code, 16)) for code in match.group(0).split("\\u")[1:])))

    for (name, offset, decoded) in candidates:
        if (decoded and directives(matching_view(decoded)) and not any(i.name == name for i in found)):
            found.append(Indicator("encoded_directive", name, 3, offset, defang(decoded), HIDDEN))

    # The whole text turned over: only the stock phrasings are looked for, since any text read backwards or in
    # ROT13 is noise in which a loose pattern would find something.
    if (len(text) <= MAX_WHOLE_TEXT_DECODE and not any(pattern.search(text) for pattern in STOCK_DIRECTIVES)):
        for (name, turned) in (("rot13", codecs.decode(text, "rot13")), ("reversed", text[::-1])):
            match = next((match for match in (pattern.search(turned) for pattern in STOCK_DIRECTIVES) if match), None)

            if (match):
                found.append(
                    Indicator("encoded_directive", name, 3, 0, _context(turned, match.start(), match.end()), HIDDEN))

    return found


def encoded_secrets(text: str) -> list[Indicator]:
    """Credentials inside base64 runs: how a key is carried in an address once a filter looks for its plain form."""
    found = []

    for match in itertools.islice(SHORT_BASE64_RUN.finditer(text), MAX_MATCHES):
        for indicator in find_secrets(_decoded(match.group(0))):
            found.append(
                dataclasses.replace(indicator,
                                    offset=match.start(),
                                    evidence=indicator.evidence + ", carried as base64"))

    return found


# --- Scoring ----------------------------------------------------------------------------------------------------


def _scannable(text: str) -> tuple:
    if (len(text) <= 2 * MAX_SCAN):
        return (text, False)

    return (text[:MAX_SCAN] + "\n" + text[-MAX_SCAN:], True)


def scan_text(text: str) -> list[Indicator]:
    """Every indicator in `text`, at most `MAX_SHOWN` of each kind and name, ordered by weight and then position."""
    if (not text or not isinstance(text, str)):
        return []

    (text, _) = _scannable(text)
    plain = [*directives(text), *foreign_directives(text)]
    found = [
        *hidden_characters(text),
        *hidden_text(text),
        *plain,
        *obfuscated_directives(text, plain),
        *hidden_markup(text),
        *exfiltration_channels(text),
        *encoded_directives(text)
    ]

    return sorted(found, key=lambda indicator: (-indicator.weight, indicator.offset, indicator.kind, indicator.name))


def score(indicators: list[Indicator]) -> int:
    """The weight of a text's indicators, counting each kind and name once so that repetition does not inflate it."""
    distinct = {}

    for indicator in indicators:
        key = (indicator.kind, indicator.name)
        distinct[key] = max(distinct.get(key, 0), indicator.weight)

    return sum(distinct.values())


_ASSESSED: collections.OrderedDict = collections.OrderedDict()
_ASSESSED_LIMIT = 4096


def assess(text: str) -> Assessment:
    """
    Grade a text. The result is remembered by the text's digest, so a hunt that meets the same tool result in three
    places, or a stream that hunts a session again with every batch, scans it once.
    """
    if (not text or not isinstance(text, str)):
        return Assessment(0, "none", (), (), "", False)

    key = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).digest()
    cached = _ASSESSED.get(key)

    if (cached is not None):
        _ASSESSED.move_to_end(key)

        return cached

    (_, partial) = _scannable(text)
    indicators = scan_text(text)
    total = score(indicators)
    categories = tuple(sorted({indicator.category for indicator in indicators if indicator.weight > 0}))
    weighty = {indicator.category for indicator in indicators if indicator.weight >= 2}
    strong = any(indicator.kind in STRONG_KINDS for indicator in indicators)
    # Two kinds of evidence agree, and one of them is about the model or about hiding: a page that names a
    # credential and gives an address with a slot in it is API documentation.
    # A status badge or a passing mention of "API keys" weighs 1 and agrees with nothing.
    agreed = len(weighty) >= 2 and (bool({ADDRESS, CONCEAL, HIDDEN} & weighty)
                                    or any(indicator.category == EXFIL and indicator.weight >= 3
                                           for indicator in indicators))

    if (not indicators):
        level = "none"
    elif (total < LIKELY_SCORE):
        level = "weak"
    elif (strong or agreed):
        level = "likely"
    else:
        level = "suspicious"

    hidden = next((indicator.evidence for indicator in indicators if indicator.kind == "hidden_text"), "")
    result = Assessment(total, level, categories, tuple(indicators), hidden, partial)
    _ASSESSED[key] = result

    if (len(_ASSESSED) > _ASSESSED_LIMIT):
        _ASSESSED.popitem(last=False)

    return result
