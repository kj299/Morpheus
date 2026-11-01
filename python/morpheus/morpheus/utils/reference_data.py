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
The reference datasets a scored row depends on, read once per process and pinned by digest in the envelope.

A model is not the only file outside the configuration that decides what a feature means. The Public Suffix List
decides which part of a query name is the registered domain, so it decides the distinct-subdomain count and the
entropy R-B-L7-001 reads; the list is revised every few days upstream, and a re-run over the same events with a
newer copy can move a row across the rule's threshold with nothing in the configuration changed. Until plan section
4.2 the envelope could not say which copy produced a row (guide 13, plan item 3). It now can: every dataset this
module registers is digested from the bytes the process actually parsed, and the digest reaches every scored row
as `reference_data` and is folded into `pipeline_fingerprint`, as the models' digests reach theirs as
`model_version`.

**The bytes a stage parses and the bytes the envelope digests are the same bytes.** `read_dataset` reads a file
once per process and caches the text with its digest; `morpheus.utils.query_entropy` builds its rules from that
text and `reference_digests` reports that digest. A list replaced on disk under a running process therefore changes
neither the process's rules nor its envelope until it restarts, and after a restart both change together. A digest
taken separately from the file would be able to name a list the stage never read.

**Every registered dataset is in every envelope**, including the envelope of a pipeline with no DNS stage. Naming
only the datasets a pipeline reads would need each composition to declare them, and a composition that forgot one
would claim comparability across a list change it was exposed to. The cost of the conservative choice is that a
list update makes the fingerprints of pipelines that never read it differ too, which says "not known to be
comparable" where the truth is "comparable" -- the safe direction for a field whose purpose is refusing to guess.

**The time zone database is not a reference dataset of these pipelines, by decision.** The plan item names it
beside the suffix list, and it would be one if any stage resolved a named zone. None does: TC-5's cadence features
take a fixed offset precisely so that a `tzdata` update cannot move them
(`morpheus.stages.telemetry.tc5_cadence_stage`), and every other stage works in UTC nanoseconds.
`tests/morpheus/utils/test_reference_data.py` holds the package to that, failing on the first import of a zone
database; the stage that introduces one must register the database here, so its digest reaches the envelope with
it.
"""

import hashlib
import os
import threading
import typing

from morpheus.utils.determinism_envelope import DIGEST_CHARACTERS

PUBLIC_SUFFIX_LIST = "public_suffix_list"

REFERENCE_DATASETS = {PUBLIC_SUFFIX_LIST: "public_suffix_list.dat"}
"""Every reference dataset a composed stage reads: its name in the envelope, and its file in the data directory."""

_CACHE: dict = {}
_LOCK = threading.Lock()


def _data_dir() -> str:
    import morpheus  # pylint: disable=import-outside-toplevel

    return morpheus.DATA_DIR


def dataset_path(name: str, data_dir: typing.Optional[str] = None) -> str:
    """
    Where a registered dataset is read from.

    Raises
    ------
    KeyError
        If the dataset is not registered. A stage reading an unregistered file would be reading something the
        envelope cannot name, so the lookup refuses rather than building a path.
    """
    if (name not in REFERENCE_DATASETS):
        raise KeyError(f"{name!r} is not a registered reference dataset; register it in "
                       f"morpheus.utils.reference_data.REFERENCE_DATASETS so its digest reaches the envelope")

    return os.path.join(data_dir or _data_dir(), REFERENCE_DATASETS[name])


def read_dataset(name: str, data_dir: typing.Optional[str] = None) -> tuple[str, str]:
    """
    A registered dataset's text and the digest of its bytes, read once per process and path.

    Parameters
    ----------
    name : str
        A key of `REFERENCE_DATASETS`.
    data_dir : str, optional
        The directory to read from; the package's data directory when omitted.

    Returns
    -------
    tuple of str
        `(text, digest)`, the digest being the first `DIGEST_CHARACTERS` hex characters of the bytes' sha256.
    """
    path = dataset_path(name, data_dir)

    with _LOCK:
        if (path not in _CACHE):
            with open(path, "rb") as handle:
                content = handle.read()

            _CACHE[path] = (content.decode("utf-8"), hashlib.sha256(content).hexdigest()[:DIGEST_CHARACTERS])

        return _CACHE[path]


def reference_digests(data_dir: typing.Optional[str] = None) -> dict:
    """Every registered dataset's name and the digest of the copy this process reads."""
    return {name: read_dataset(name, data_dir)[1] for name in sorted(REFERENCE_DATASETS)}


def render_reference_data(digests: dict) -> str:
    """
    The digests as the one text column a scored row carries: `name:digest` pairs in name order, joined by `;`.

    One column rather than one per dataset, so registering a dataset adds a pair to a field every consumer already
    reads rather than a column every contract would have to learn.
    """
    if (not digests):
        raise ValueError("no reference data digests; an envelope that names none cannot say which copy it read")

    return ";".join(f"{name}:{digests[name]}" for name in sorted(digests))


def clear_cache():
    """Forget every dataset read so far. For tests: a deployment pins the copy it started with."""
    with _LOCK:
        _CACHE.clear()
