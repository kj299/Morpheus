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
The synthetic corpus `morpheus.compositions.host_score` is tested over, and the facts about it the harnesses assert.

Moved out of `tests/morpheus/determinism/host_score_pipeline.py` under plan section 4.5, which put the composition in
the package and left the data here, where the tests, the sample and fixture generators and the experiments read it. The
composition module's docstring describes what this corpus plants and why.
"""

import random

from corpora import merged  # noqa: F401
from corpora.records import envelope as _envelope

from morpheus.compositions import envelope

host_identity = merged('host_identity')
import pandas as pd

from morpheus.compositions.estate_context import EstateContext
from morpheus.compositions.host_score import DAY_S
from morpheus.compositions.host_score import EPOCH_S
from morpheus.compositions.host_score import FLOW_CLASS
from morpheus.compositions.host_score import HANDSHAKE_CLASS
from morpheus.compositions.host_score import HOUR_S
from morpheus.compositions.host_score import MANIFEST_WINDOW
from morpheus.compositions.host_score import POPULATION
from morpheus.compositions.host_score import PROCESS_CLASS
from morpheus.compositions.host_score import TRAINING_DAYS
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.model_manifest import ModelManifest

CORPUS_SEED = 20261021
WORK_HOURS = tuple(range(9, 17))
"""Every host's working hours, every day: eight hourly windows a day per host."""
CLIENT_POOLS = ("10.0.1.", "10.0.2.", "10.0.3.")
DRIFTING = "10.0.2.24"
"""The engineering workstation whose behaviour moves a little further each day of the scored week."""
DRIFT_FROM_DAY = 1
"""The scored day the drift starts on; day 0 of the scored week is still the fortnight's habits."""
STEPPED = "10.0.1.14"
"""The finance workstation that changes once and stays changed."""
STEP_DAY = 2
JOINER = "10.0.1.15"
"""The finance laptop issued on the scored week's first day, with no fortnight of its own."""
KIOSK = "10.0.3.31"
"""The lobby kiosk: inventoried, and in no peer group."""
SERVERS = tuple(f"10.0.0.{index}" for index in range(1, 25))
"""The estate's servers. Static addresses, outside every DHCP pool."""
PUBLIC = tuple(f"93.184.216.{index}" for index in range(1, 61))
"""Addresses on the public internet."""
BROWSER = "t13d1516h2_8daaf6152771_b186095e22b6"
BROWSER_UPDATED = "t13d1516h2_8daaf6152771_02713d6af862"
UPDATER = "t13d0312h2_55b375c5d22e_cd85d2d88918"
EXPLORER = r"C:\Windows\explorer.exe"
SERVICES = r"C:\Windows\System32\services.exe"
SVCHOST = r"C:\Windows\System32\svchost.exe"
USUAL_PAIRS = ((EXPLORER, r"C:\Program Files\Browser\browser.exe"), (SERVICES, SVCHOST),
               (EXPLORER, r"C:\Program Files\Office\outlook.exe"))


def _flow(rows: list, source: str, destination: str, when: int, bytes_out: int, bytes_in: int):
    record = _envelope(rows, "netflow-01", "tc3.v1", f"{source}->{destination}", when)
    record.update({
        "src_ip": source,
        "src_port": 49152 + len(rows) % 16384,
        "dst_ip": destination,
        "dst_port": 443,
        "protocol": "tcp",
        "ip_ttl": 128,
        "bytes_out": bytes_out,
        "bytes_in": bytes_in,
        "bgp_as_dst": None if destination in SERVERS else "64500",
        "exporter_id": f"rtr-campus-{source.split('.')[2]}:256",
    })
    rows.append(record)


def _handshake(rows: list, source: str, destination: str, when: int, fingerprint: str):
    record = _envelope(rows, "tls-inspect-01", "tc6.v1", f"{source}->{destination}", when)
    record.update({
        "src_ip": source,
        "src_port": 49152 + len(rows) % 16384,
        "dst_ip": destination,
        "dst_port": 443,
        "sni": None,
        "protocol": "tcp",
        "tls_version": "TLSv1.3",
        "ja4_client": fingerprint,
        "certificate_issuer": "CN=Public Web CA, O=Example Trust",
        "certificate_fingerprint_sha256": f"{destination}|public",
        "cipher_suite": "TLS_AES_256_GCM_SHA384",
        "validation_result": "ok",
        "certificate_not_before": when - 10 * DAY_S * NS_PER_SECOND,
        "certificate_not_after": when + 355 * DAY_S * NS_PER_SECOND,
        "content_type_declared": None,
        "content_type_detected": None,
    })
    rows.append(record)


def _process(rows: list, address: str, parent: str, image: str, when: int):
    host = host_identity.hostname_for(address)
    record = _envelope(rows, "edr-01", "tc7_endpoint.v1", f"{host}>{image}", when)
    record.update({
        "hostname": host,
        "parent_image_path": parent,
        "image_path": image,
        "integrity_level": "Medium",
        "signature_status": "Valid",
    })
    rows.append(record)


def intensity(address: str, day: int) -> int:
    """How far a host's behaviour has moved from its fortnight on a corpus day: 0 for every ordinary hour."""
    scored = day - TRAINING_DAYS

    if (address == DRIFTING and scored >= DRIFT_FROM_DAY):
        return scored - DRIFT_FROM_DAY + 1

    if (address == STEPPED and scored >= STEP_DAY):
        return 4

    return 0


def active(address: str, day: int) -> bool:
    """Whether a host is in use on a corpus day: every host every day, except the joiner before it is issued."""
    return address != JOINER or day >= TRAINING_DAYS


def browser_update_day(index: int) -> int:
    """
    The corpus day a host's browser updates to its next release, which offers a client stack it never has.

    Staggered over the fortnight's second week, as an estate's updates are, so the training rows hold hosts that
    have offered two client stacks and hosts that have offered three. Were every host's count the same on every
    training row, the model would learn that the count never moves, and the first new stack anywhere would score in
    the billions -- which is what the first fit of these models did.
    """
    return TRAINING_DAYS // 2 + 1 + index % 5


SCORED_DAYS = 7

CORPUS_DAYS = TRAINING_DAYS + SCORED_DAYS

FINANCE = "finance-workstations"

ENGINEERING = "engineering-workstations"

HOSTS = {
    "10.0.1.11": FINANCE,
    "10.0.1.12": FINANCE,
    "10.0.1.13": FINANCE,
    "10.0.1.14": FINANCE,
    "10.0.2.21": ENGINEERING,
    "10.0.2.22": ENGINEERING,
    "10.0.2.23": ENGINEERING,
    "10.0.2.24": ENGINEERING,
    "10.0.1.15": FINANCE,
    "10.0.3.31": None,
}
"""Each workstation's address, which its DHCP lease fixes for the corpus, and its peer group in the inventory."""


def at(day: int, hour: int, minute: int = 0, second: int = 0) -> int:
    """An instant in the corpus, in epoch nanoseconds, `day` counted from the first day of the fortnight."""
    return (EPOCH_S + day * DAY_S + hour * HOUR_S + minute * 60 + second) * NS_PER_SECOND


def host_key(address: str) -> str:
    """The machine an address is leased to, as `host_key` names it."""
    return host_identity.hostname_for(address).lower()


def _hour(rng: random.Random,
          flows: list,
          handshakes: list,
          processes: list,
          address: str,
          day: int,
          hour: int,
          index: int):
    """One working hour of one host: its flows, its handshakes and its processes."""
    level = intensity(address, day)
    servers = rng.sample(SERVERS, rng.randint(3, 6) + 2 * level)
    public = rng.sample(PUBLIC, (1 if rng.random() < 0.4 else 0) + 2 * level)

    for (position, destination) in enumerate(servers + public):
        bytes_in = rng.randint(1200, 3200)
        bytes_out = rng.randint(500, 1500) * (1 + level)
        _flow(flows, address, destination, at(day, hour, 2 + position, index), bytes_out, bytes_in)

    browser = BROWSER_UPDATED if day >= browser_update_day(index) else BROWSER
    fingerprints = [browser] + ([UPDATER] if rng.random() < 0.15 else [])

    if (level > 0 and hour == WORK_HOURS[0]):
        # The drifting machine offers a client stack it has never offered, each day the change goes on; the stepped
        # one took up one new stack the day it changed, and keeps using it.
        stamp = day if address == DRIFTING else TRAINING_DAYS + STEP_DAY
        fingerprints.append(f"t13d{level:02d}11h2_{address.replace('.', '')[-6:]}a0b1c2_{stamp:012d}")

    for (position, fingerprint) in enumerate(fingerprints):
        _handshake(handshakes, address, rng.choice(PUBLIC), at(day, hour, 40 + position, index), fingerprint)

    for (position, (parent, image)) in enumerate(USUAL_PAIRS):
        _process(processes, address, parent, image, at(day, hour, 50 + position, index))

    novel = (1 if rng.random() < 0.05 else 0) + (level if hour in (WORK_HOURS[0], WORK_HOURS[4]) else 0)

    for count in range(novel):
        # A program no machine in its group has run: a software update, or something that is not one.
        image = rf"C:\Users\Public\{host_key(address)}-{day}-{hour}-{count}.exe"
        _process(processes, address, EXPLORER, image, at(day, hour, 56, index + 10 * count))


def build_corpus() -> dict[str, pd.DataFrame]:
    """Three weeks of ten workstations, split by collector, with the leases and inventory reports that name them."""
    rng = random.Random(CORPUS_SEED)
    flows: list = []
    handshakes: list = []
    processes: list = []

    for day in range(CORPUS_DAYS):
        for hour in WORK_HOURS:
            for (index, address) in enumerate(sorted(HOSTS)):
                if (active(address, day)):
                    _hour(rng, flows, handshakes, processes, address, day, hour, index)

    def frame(rows: list) -> pd.DataFrame:
        result = pd.DataFrame(rows).sort_values(["event_time", "collector_seq"], kind="mergesort")
        result = result.reset_index(drop=True)
        result["collector_seq"] = result.groupby("collector_id", sort=False).cumcount()

        return result

    workstations = host_identity.build(sorted(HOSTS), at(0, 0), at(CORPUS_DAYS, 0) - 1, pools=CLIENT_POOLS)

    return envelope.attach_all({
        FLOW_CLASS: frame(flows),
        HANDSHAKE_CLASS: frame(handshakes),
        PROCESS_CLASS: frame(processes),
        host_identity.LEASE_CLASS: workstations[host_identity.LEASE_CLASS],
        host_identity.INVENTORY_CLASS: workstations[host_identity.INVENTORY_CLASS],
    })


def asset_versions() -> list:
    """The inventory: every workstation, with its peer group where it has one, recorded before the corpus starts."""
    return [
        make_version("asset",
                     host_key(address),
                     0,
                     None,
                     at(0, 0),
                     values={
                         "owner": "it-ops@example.com",
                         "peer_group": group,
                         "device_role": "kiosk" if group is None else "workstation",
                         "os_family": "windows",
                         "criticality": "low" if group is None else "medium",
                     }) for (address, group) in sorted(HOSTS.items())
    ]


def build_store() -> BitemporalStore:
    """The asset store every class is enriched from."""
    return BitemporalStore("asset", asset_versions())


class _Unscored:
    """The scorer of a run whose manifest scores nothing: asked anything, it is a wiring mistake."""

    def score(self, model_version: str, features: list) -> list:
        raise AssertionError(f"asked to score {len(features)} rows against {model_version}; nothing is scored here")


def unscored() -> tuple[_Unscored, ModelManifest]:
    """
    A scorer and manifest that score nothing, for the run the training frames are taken from.

    The features are what the stages emit and do not depend on any score, so the frames exist before the models do.
    """
    return (_Unscored(),
            ModelManifest(window_id=MANIFEST_WINDOW,
                          models={},
                          fallback=f"{POPULATION}:untrained",
                          scores_from_ns=at(CORPUS_DAYS + 1, 0)))


def context() -> EstateContext:
    """The estate's context this corpus is run under: the asset inventory of every workstation, with its peer group."""
    return EstateContext(asset=build_store())
