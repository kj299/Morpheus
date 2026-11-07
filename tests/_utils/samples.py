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
Test data more than one of the fork's tests builds, written once (guide 13 plan item 6).

The stage parameter liveness registry drives each stage with a scenario, and several of its scenarios are the stage
test's own frame or store; each kept its copy until the copies became the second definition this module removes.
"""

from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.model_manifest import ModelManifest

ALICE = "alice@example.com"
BOB = "bob@example.com"
WINDOW = 484512
"""The window a manifest is pinned for."""


def host_asset_store() -> BitemporalStore:
    """One host the inventory knows, `fin-01`, at medium criticality, as the asset stage keys it."""
    return BitemporalStore("asset", [make_version("asset", "fin-01", 0, None, 0, values={"criticality": "medium"})])


def arp_observations() -> dict:
    """
    Three ARP observations: two the ladder placed on a port and one it could not, each with its own address.

    The candidates disagree on every row where both have a value, so preferring one over the other is visible
    in the output -- which is what makes the order a parameter rather than a set.
    """
    return {
        "resolved_port_key": ["hq:sw1:Gi1/0/1", None, "hq:sw1:Gi1/0/2"],
        "arp_sender_ip": ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
    }


def model_manifest(**kwargs) -> ModelManifest:
    """A manifest pinned for `WINDOW`, naming a model for each of Alice and Bob, with any field overridden."""
    defaults = {"window_id": WINDOW, "models": {ALICE: "dfp-alice:14", BOB: "dfp-bob:3"}}
    defaults.update(kwargs)

    return ModelManifest(**defaults)
