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
The fork's model inventory: which file is which model, checked against the digest it was committed with.

Two of the design's rules need a model that is not a per-entity baseline. R-B-L4-001 reads the shipped
`abp-pcap-xgb` booster over the thirteen layer 4 flow features, and R-B-L7-003 reads the shipped SID model over
request and response bodies. Upstream served both from a Triton model repository, which resolves a model by name
and nothing else: whatever file sits under that name scores the row, and nothing in the output says which file it
was. `models/inventory/inventory.json` is the answer this fork keeps instead. Each entry names its files with the
sha256 each was committed with, the settings the model was trained under, and where it came from.

**A file whose digest does not match is refused, not loaded.** A model that changed underneath its name changes
every score it produces while every row still names the same model. `load_entry` hashes each file and raises on a
mismatch, so the version a scorer writes beside its score names the bytes that produced it.

**A Git LFS pointer is named as one.** The SID model and its validation set are stored in Git LFS, and a checkout
that never fetched them holds a 130-byte text file in their place. Hashing that and reporting a digest mismatch
would send someone looking for a corrupted model; `ModelFileUnavailable` says what the file is and how to fetch it.

**The version is the model file's own digest**, as `name:<first sixteen hex digits>`, the same shape the layer 5
manifest pins, and the entry records it so a test can tell when a file and its entry drift apart.
"""

import dataclasses
import hashlib
import json
import os

REPOSITORY_INVENTORY = os.path.join("models", "inventory", "inventory.json")
"""Where the inventory sits in a checkout of this repository, relative to its root."""

DIGEST_LENGTH = 16

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class ModelFileUnavailable(FileNotFoundError):
    """A model file is a Git LFS pointer, or is missing, rather than the model."""


@dataclasses.dataclass(frozen=True)
class InventoryEntry:
    """
    One model of the inventory, its files verified.

    Attributes
    ----------
    name : str
        The entry's key, the model's name.
    version : str
        `name:<digest>`, the digest being the first sixteen hex digits of the model file's sha256.
    model_path : str
        The model file.
    files : dict
        Every file of the entry, by its role, as a path.
    settings : dict
        What the model was trained under and how it is read: features, tokenizer, labels, threshold.
    """

    name: str
    version: str
    model_path: str
    files: dict
    settings: dict


def file_sha256(path: str) -> str:
    """
    The sha256 of a file, refusing a Git LFS pointer or a missing file.

    Raises
    ------
    ModelFileUnavailable
        If the file does not exist, or is an LFS pointer rather than its content.
    """
    if (not os.path.isfile(path)):
        raise ModelFileUnavailable(f"{path} does not exist")

    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        head = handle.read(len(LFS_POINTER_PREFIX))

        if (head == LFS_POINTER_PREFIX):
            raise ModelFileUnavailable(f"{path} is a Git LFS pointer, not the file. Fetch it with "
                                       f"`git lfs install && git lfs pull --include=\"{path}\"` from the "
                                       f"repository root")

        digest.update(head)

        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def load_entry(inventory_path: str, name: str) -> InventoryEntry:
    """
    Read one entry of the inventory and verify every file it names.

    Parameters
    ----------
    inventory_path : str
        The inventory document; its files are named relative to its directory.
    name : str
        The entry to read.

    Returns
    -------
    InventoryEntry
        The entry, with each file's path resolved and its digest checked.

    Raises
    ------
    KeyError
        If the inventory has no such entry.
    ModelFileUnavailable
        If a file is missing or is an unfetched Git LFS pointer.
    ValueError
        If a file's digest is not the one recorded, or the recorded version is not the model file's digest.
    """
    with open(inventory_path, encoding="utf-8") as handle:
        inventory = json.load(handle)

    if (name not in inventory["models"]):
        raise KeyError(f"the inventory at {inventory_path} has no model {name!r}; it has "
                       f"{sorted(inventory['models'])}")

    entry = inventory["models"][name]
    root = os.path.dirname(os.path.abspath(inventory_path))
    files = {}

    for (role, record) in entry["files"].items():
        path = os.path.join(root, record["path"])
        actual = file_sha256(path)

        if (actual != record["sha256"]):
            raise ValueError(f"{path} has sha256 {actual}, and the inventory records {record['sha256']} for "
                             f"{name}'s {role}. A model file changed without its entry; restore the file or "
                             f"record the new one deliberately")

        files[role] = path

    version = f"{name}:{entry['files']['model']['sha256'][:DIGEST_LENGTH]}"

    if (entry["version"] != version):
        raise ValueError(f"the inventory records {name} as {entry['version']!r}, and its model file makes it "
                         f"{version!r}")

    return InventoryEntry(name=name,
                          version=version,
                          model_path=files["model"],
                          files=files,
                          settings=dict(entry["settings"]))
