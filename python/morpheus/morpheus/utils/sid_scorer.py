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
The shipped SID model, scored on the CPU from the inventory rather than behind a Triton endpoint.

R-B-L7-003 is the upstream sensitive-information model applied to request and response bodies. Upstream served it
from Triton with a GPU tokenizer in front; this tokenizes with {py:class}`~morpheus.utils.wordpiece.WordPieceTokenizer`
and runs the same ONNX file with ONNX Runtime on the CPU. The model reads 256 token ids and an attention mask and
returns one logit per label; a label's probability is the logistic of its logit.

**A body longer than the model's 256 tokens is read in overlapping windows, not cut.** The model's recorded
settings are a maximum of 256 tokens, a stride of 64, and no truncation: a window starts every 64 tokens until one
reaches the end, and each label's probability is the highest any window gave it. Cutting at 256 tokens instead
reads the start of a payload and is blind to a key that arrives after it, which is where a key tends to sit in a
long configuration dump. Over the 2,000 validation payloads, 38 of which run past 256 tokens, the windows decide
99.91% of labels correctly and the cut 99.87%.

**A score does not depend on what it was batched with.** Triton's dynamic batching is one of the reasons D2 gives
for removing the upstream path. ONNX Runtime runs here on one thread with sequential execution, and the windows of
a batch are independent rows; `tests/morpheus/utils/test_sid_scorer.py` checks that a window scored alone and
scored inside batches of other sizes gives the same bits.

**An empty body is not scored.** A window of no tokens is a mask of zeros, which asks the model a question it was
never trained on, so a row whose text tokenizes to nothing has no score rather than whatever that answer would be.
"""

import numpy as np

from morpheus.utils.model_inventory import InventoryEntry
from morpheus.utils.wordpiece import WordPieceTokenizer

BATCH_ROWS = 32


def windows(ids: list, max_length: int, stride: int, truncation: bool) -> list:
    """
    The windows a token sequence is read in.

    One window when the sequence fits or `truncation` is set, holding at most `max_length` tokens. Otherwise a window
    of `max_length` tokens starting every `stride` tokens, the last being the first to reach the end.
    """
    if (len(ids) <= max_length or truncation):
        return [ids[:max_length]]

    result = []
    start = 0

    while (True):
        result.append(ids[start:start + max_length])

        if (start + max_length >= len(ids)):
            return result

        start += stride


class SidScorer:
    """
    Score text with the inventory's SID model.

    Parameters
    ----------
    entry : InventoryEntry
        The inventory entry, already verified by `load_entry`.

    Raises
    ------
    ValueError
        If the entry is not an ONNX model, or its settings ask for something this does not do.
    """

    def __init__(self, entry: InventoryEntry):
        import onnxruntime  # pylint: disable=import-outside-toplevel

        settings = entry.settings
        tokenizer = settings["tokenizer"]

        if (settings.get("format") != "onnx"):
            raise ValueError(f"{entry.name} is {settings.get('format')!r}, and this scores 'onnx'")

        if (not tokenizer["lower_case"] or tokenizer["special_tokens"]):
            raise ValueError(f"{entry.name} asks for a tokenizer this does not implement: {tokenizer}")

        with open(entry.files["labels"], encoding="utf-8") as handle:
            self._labels = [line.strip() for line in handle if line.strip()]

        self._tokenizer = WordPieceTokenizer(entry.files["vocabulary"])
        self._max_length = int(tokenizer["max_length"])
        self._stride = int(tokenizer["stride"])
        self._truncation = bool(tokenizer["truncation"])
        self._threshold = float(settings["threshold"])
        self._inputs = list(settings["inputs"])
        self._version = entry.version

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        self._session = onnxruntime.InferenceSession(entry.model_path, options, providers=["CPUExecutionProvider"])

        outputs = self._session.get_outputs()

        if (len(outputs) != 1 or outputs[0].shape[-1] != len(self._labels)):
            raise ValueError(f"{entry.name} returns {[output.shape for output in outputs]}, and the inventory names "
                             f"{len(self._labels)} labels")

    @property
    def version(self) -> str:
        """The scored model, as `name:<digest>`."""
        return self._version

    @property
    def labels(self) -> list:
        """The labels, in the order of the model's outputs."""
        return list(self._labels)

    @property
    def threshold(self) -> float:
        """The probability above which a label is present."""
        return self._threshold

    def token_ids(self, text: str) -> list:
        """The vocabulary ids of `text`, as the scorer tokenizes it."""
        return self._tokenizer.encode(text)

    def windows(self, ids: list) -> list:
        """The windows a token sequence is read in, under the inventory's settings; see `windows`."""
        return windows(ids, self._max_length, self._stride, self._truncation)

    def logits(self, token_windows: list) -> np.ndarray:
        """
        The model's output for each window, float32, in batches of `BATCH_ROWS`.

        Parameters
        ----------
        token_windows : list
            Token id lists, none longer than `max_length` and none empty.
        """
        ids = np.zeros((len(token_windows), self._max_length), dtype=np.int32)
        mask = np.zeros_like(ids)

        for (row, window) in enumerate(token_windows):
            ids[row, :len(window)] = window
            mask[row, :len(window)] = 1

        batches = [
            self._session.run(
                None, {
                    self._inputs[0]: ids[start:start + BATCH_ROWS], self._inputs[1]: mask[start:start + BATCH_ROWS]
                })[0] for start in range(0, len(token_windows), BATCH_ROWS)
        ]

        return np.concatenate(batches) if batches else np.zeros((0, len(self._labels)), dtype=np.float32)

    def probabilities(self, texts: list) -> list:
        """
        Each text's probability per label, the highest over its windows.

        Parameters
        ----------
        texts : list
            Strings; `None` is empty.

        Returns
        -------
        list
            One entry per text: a float64 array in `labels` order, or `None` for a text with no tokens.
        """
        token_windows = []
        owners = []

        for (position, text) in enumerate(texts):
            ids = self.token_ids(text)

            if (not ids):
                continue

            for window in self.windows(ids):
                token_windows.append(window)
                owners.append(position)

        probabilities = 1.0 / (1.0 + np.exp(-self.logits(token_windows).astype(np.float64)))
        result = [None] * len(texts)

        for (row, position) in enumerate(owners):
            result[position] = (probabilities[row] if result[position] is None else np.maximum(
                result[position], probabilities[row]))

        return result
