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
"""
The CPU tokenizer against the reference.

The expected ids are what Hugging Face's `tokenizers` 0.23.3 `BertWordPieceTokenizer` produced from the inventory's
vocabulary with `lowercase=True`, `clean_text=True`, `handle_chinese_chars=True` and no special tokens. Each string
reaches one step of the algorithm the module docstring lists, so a regression names the step it broke.
"""

import os

import pytest

from morpheus.utils.wordpiece import MAX_WORD_CHARACTERS
from morpheus.utils.wordpiece import WordPieceTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VOCABULARY = os.path.join(REPO_ROOT, "models", "inventory", "sid-minibert", "bert-base-uncased-vocab.txt")

REFERENCE = [
    ('accents', 'Héllo WÖRLD naïve café', [7592, 2088, 15743, 7668]),
    ('cjk', '日本語 mixed 中文', [1864, 1876, 1950, 3816, 1746, 1861]),
    ('whitespace', 'tab\tnew\nline\rcr', [21628, 2047, 2240, 13675]),
    ('format characters', '\u200bzero\u200dwidth\ufeff', [5717, 9148, 11927, 2232]),
    ('control characters', '\x00null\x01ctrl\x7f', [19701, 6593, 12190]),
    ('punctuation',
     "email: Alice.O'Neil+tag@example.co.uk; pwd=P@ss/w0rd",
     [
         10373,
         1024,
         5650,
         1012,
         1051,
         1005,
         6606,
         1009,
         6415,
         1030,
         2742,
         1012,
         2522,
         1012,
         2866,
         1025,
         1052,
         21724,
         1027,
         1052,
         1030,
         7020,
         1013,
         1059,
         2692,
         4103
     ]),
    ('compatibility characters', 'Ⅻ ﬁ ½ ² full-width ＡＢＣ', [100, 1984, 1092, 1082, 2440, 1011, 9381, 100]),
    ('other scripts',
     'Ελληνικά Кириллица',
     [
         1159,
         29727,
         29727,
         24824,
         16177,
         18199,
         29726,
         14608,
         1189,
         10325,
         16856,
         10325,
         29436,
         29436,
         10325,
         29751,
         10260
     ]),
    ('emoji', 'emoji 😀 ok', [7861, 29147, 2072, 100, 7929]),
    ('key',
     'aws_secret_access_key: wJalrXUtnFEMI/K7MDENG',
     [
         22091,
         2015,
         1035,
         3595,
         1035,
         3229,
         1035,
         3145,
         1024,
         1059,
         25787,
         2099,
         2595,
         4904,
         2078,
         7959,
         4328,
         1013,
         1047,
         2581,
         26876,
         13159
     ]),
    ('empty', '', []),
    ('blank', '   ', []),
]


@pytest.fixture(name="tokenizer", scope="module")
def tokenizer_fixture() -> WordPieceTokenizer:
    return WordPieceTokenizer(VOCABULARY)


@pytest.mark.parametrize(("text", "expected"), [(text, ids) for (_, text, ids) in REFERENCE],
                         ids=[name for (name, _, _) in REFERENCE])
def test_the_ids_are_the_reference_tokenizer_s(tokenizer: WordPieceTokenizer, text: str, expected: list):
    assert tokenizer.encode(text) == expected


def test_a_word_longer_than_the_limit_is_one_unknown(tokenizer: WordPieceTokenizer):
    # The reference cuts at 100 characters: 100 still splits into pieces, 101 is a single [UNK].
    unknown = tokenizer.word_ids("[UNK]")

    assert len(tokenizer.encode("a" * MAX_WORD_CHARACTERS)) > 1
    assert tokenizer.encode("a" * (MAX_WORD_CHARACTERS + 1)) == unknown


def test_a_word_no_pieces_cover_is_one_unknown_not_a_partial_cover(tokenizer: WordPieceTokenizer):
    # A word that splits into known pieces up to a character the vocabulary lacks is [UNK] whole; returning the
    # pieces it did find would hand the model a word that was not there.
    assert tokenizer.encode("ab☃cd") == tokenizer.word_ids("[UNK]")


def test_none_is_the_empty_string(tokenizer: WordPieceTokenizer):
    assert tokenizer.encode(None) == []


def test_a_vocabulary_without_an_unknown_token_is_refused(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_text("[PAD]\nhello\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\[UNK\]"):
        WordPieceTokenizer(str(path))
