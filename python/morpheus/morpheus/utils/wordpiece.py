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
BERT's uncased WordPiece tokenization, on the CPU, from the vocabulary file alone.

Upstream tokenized for the SID model with cuDF's GPU subword tokenizer, reading a hash table compiled from the
`bert-base-uncased` vocabulary. That needs a GPU, so the CPU path to R-B-L7-003 needs its own tokenizer, and a
tokenizer that differs from the one a model was trained against is a silent source of wrong scores: the model still
answers, about different tokens. This is the reference algorithm, BERT's basic tokenizer followed by greedy longest-
match WordPiece, written out so the steps can be read against the reference:

1. Clean: drop NUL, U+FFFD and every character in Unicode category C except tab, newline and carriage return, and
   turn every whitespace character into a space.
2. Put spaces around each CJK ideograph, so each becomes a word of its own.
3. Lower-case, decompose to NFD and drop the combining marks (category Mn), which strips accents.
4. Split on whitespace, then split each word around every punctuation character: ASCII symbols as well as Unicode
   category P.
5. Split each word into the longest vocabulary entries that cover it, left to right, every piece after the first
   prefixed with `##`. A word with no such cover, or longer than 100 characters, is the single token `[UNK]`.

`tests/morpheus/utils/test_wordpiece.py` holds it to the token ids Hugging Face's `tokenizers` 0.23.3 produced, with
the same vocabulary and settings, for strings built to reach every step above. Over the SID validation set's 2,000
payloads the two agree on every id.
"""

import unicodedata

UNKNOWN = "[UNK]"
MAX_WORD_CHARACTERS = 100


def _is_cjk(codepoint: int) -> bool:
    return ((0x4E00 <= codepoint <= 0x9FFF) or (0x3400 <= codepoint <= 0x4DBF) or (0x20000 <= codepoint <= 0x2A6DF)
            or (0x2A700 <= codepoint <= 0x2B73F) or (0x2B740 <= codepoint <= 0x2B81F)
            or (0x2B820 <= codepoint <= 0x2CEAF) or (0xF900 <= codepoint <= 0xFAFF)
            or (0x2F800 <= codepoint <= 0x2FA1F))


def _is_punctuation(character: str) -> bool:
    codepoint = ord(character)

    if ((33 <= codepoint <= 47) or (58 <= codepoint <= 64) or (91 <= codepoint <= 96) or (123 <= codepoint <= 126)):
        return True

    return unicodedata.category(character).startswith("P")


def _is_whitespace(character: str) -> bool:
    return character in " \t\n\r" or unicodedata.category(character) == "Zs"


def _is_control(character: str) -> bool:
    return character not in "\t\n\r" and unicodedata.category(character).startswith("C")


class WordPieceTokenizer:
    """
    Tokenize text into the ids of a BERT uncased vocabulary.

    Parameters
    ----------
    vocabulary_path : str
        The vocabulary, one token per line; a token's id is its line number from zero.
    """

    def __init__(self, vocabulary_path: str):
        with open(vocabulary_path, encoding="utf-8") as handle:
            self._vocabulary = {line.rstrip("\n"): index for (index, line) in enumerate(handle)}

        if (UNKNOWN not in self._vocabulary):
            raise ValueError(f"{vocabulary_path} has no {UNKNOWN} token")

        self._unknown = self._vocabulary[UNKNOWN]

    def words(self, text: str) -> list:
        """The words and punctuation marks of `text`, after steps 1 to 4."""
        characters = []

        for character in text:
            codepoint = ord(character)

            if (codepoint in (0, 0xFFFD) or _is_control(character)):
                continue

            if (_is_cjk(codepoint)):
                characters.append(f" {character} ")
            elif (_is_whitespace(character)):
                characters.append(" ")
            else:
                characters.append(character)

        cleaned = unicodedata.normalize("NFD", "".join(characters).lower())
        cleaned = "".join(character for character in cleaned if unicodedata.category(character) != "Mn")

        words = []

        for word in cleaned.split():
            current = ""

            for character in word:
                if (_is_punctuation(character)):
                    if (current):
                        words.append(current)
                        current = ""

                    words.append(character)
                else:
                    current += character

            if (current):
                words.append(current)

        return words

    def word_ids(self, word: str) -> list:
        """The vocabulary ids of one word, step 5."""
        if (len(word) > MAX_WORD_CHARACTERS):
            return [self._unknown]

        ids = []
        start = 0

        while (start < len(word)):
            end = len(word)
            found = None

            while (start < end):
                piece = word[start:end] if start == 0 else f"##{word[start:end]}"

                if (piece in self._vocabulary):
                    found = self._vocabulary[piece]
                    break

                end -= 1

            if (found is None):
                return [self._unknown]

            ids.append(found)
            start = end

        return ids

    def encode(self, text: str) -> list:
        """
        The vocabulary ids of `text`, with no special tokens added.

        Parameters
        ----------
        text : str
            Any text; `None` is the empty string.
        """
        return [token for word in self.words(text or "") for token in self.word_ids(word)]
