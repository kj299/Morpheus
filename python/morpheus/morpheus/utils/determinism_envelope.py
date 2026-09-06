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
The block of fields that makes a scored event reproducible, and the hashes inside it.

This is determinism control 12, and controls 2 and 3 in the parts they contribute to it. The guide calls this
what turns determinism from a claim into a verifiable property, and the reason is narrow and practical: six
months after an alert, the question is not "was the pipeline deterministic" but "what exactly produced this
number". An event that cannot answer that has no defensible provenance however deterministic the pipeline was.

**Two events with the same `pipeline_fingerprint` were produced by the same system.** Two with different ones
are not directly comparable, and a drift analysis that ignores the difference attributes a configuration change
to a behavioral change -- which at layer 5 means telling somebody an employee's behavior shifted when what
shifted was a threshold.

**Nothing may be omitted quietly.** `Config.to_string` covers the pipeline's own settings and not the schema
definitions, the thresholds, the code commit or the image digest, and a fingerprint that silently skipped what
it did not know would claim comparability it cannot support. Every component is therefore required, and a
deployment that genuinely does not know one passes `UNKNOWN` -- which is itself hashed, so events produced
without a known commit are visibly not comparable with events produced with one. Refusing to guess is the whole
value of the field.

**Integers are rendered through the same rule as every other identifier in this fork.** A threshold arriving as
`3` in one run and `3.0` in another -- which is what reading a column that holds a gap does to it -- would
otherwise fork one configuration into two fingerprints and make every event before the change incomparable with
every event after it, for no reason at all.
"""

import dataclasses
import hashlib
import json
import typing

from morpheus.utils.entity_key import render_integral

UNKNOWN = "unknown"
"""What a deployment passes for a component it genuinely cannot determine.

Hashed like any other value rather than skipped, so a fingerprint computed without a code commit differs from
one computed with it. That is the honest outcome: the two runs are not known to be comparable.
"""

TIERS = ("D0", "D1", "D2", "D3")
"""The four determinism tiers, as Part 5 defines them.

D0 bit-exact, D1 decision-stable, D2 lineage-stable, D3 explanation-stable. The tier is declared per pipeline
and recorded on every event, because a consumer cannot tell from a score which guarantee produced it.
"""

DIGEST_CHARACTERS = 16
"""Hex characters kept from each digest. Sixty-four bits, which is what the guide's own examples carry."""

TIER_COLUMN = "determinism_tier"
FINGERPRINT_COLUMN = "pipeline_fingerprint"
CONFIG_HASH_COLUMN = "config_hash"


def config_hash(config, decimals: int = DIGEST_CHARACTERS) -> str:
    """
    A digest of the pipeline configuration, from the configuration's own serialization.

    Parameters
    ----------
    config : `morpheus.config.Config`
        The pipeline configuration. `Config.save` and `Config.to_string` serialize with `sort_keys=True`, so the
        digest does not depend on dictionary iteration order.
    decimals : int, default = 16
        Hex characters to keep.

    Returns
    -------
    str
        The digest.
    """
    return hashlib.sha256(config.to_string().encode("utf-8")).hexdigest()[:decimals]


def _canonical(value: typing.Any) -> typing.Any:
    """Render a component so that two runs meaning the same thing hash the same."""
    if (isinstance(value, bool)):
        # Before the numeric branch: a bool is an int, and rendering True as 1 would make it collide with a
        # threshold of one.
        return "true" if value else "false"

    if (isinstance(value, (int, float))):
        return render_integral(value)

    if (isinstance(value, dict)):
        return {str(key): _canonical(item) for (key, item) in sorted(value.items(), key=lambda pair: str(pair[0]))}

    if (isinstance(value, (list, tuple))):
        return [_canonical(item) for item in value]

    return value


def pipeline_fingerprint(*,
                         configuration: str,
                         schema_versions: dict,
                         thresholds: dict,
                         code_commit: str,
                         image_digest: str,
                         decimals: int = DIGEST_CHARACTERS) -> str:
    """
    A digest over everything that decides what a score means.

    Every component is required. A deployment that cannot determine one passes `UNKNOWN`, which is hashed like
    any other value, so the resulting fingerprint differs from one computed with the component known.

    Parameters
    ----------
    configuration : str
        The `config_hash`.
    schema_versions : dict
        Telemetry class to feature schema version, for example `{"TC-5": "2.1.0"}`.
    thresholds : dict
        Rule identifier to threshold value, for every threshold in the scoring path.
    code_commit : str
        The commit the pipeline was built from.
    image_digest : str
        The container image digest.
    decimals : int, default = 16
        Hex characters to keep.

    Returns
    -------
    str
        The digest.

    Raises
    ------
    ValueError
        If any component is `None` or an empty string. Omitting one silently is what the fingerprint exists to
        prevent; pass `UNKNOWN` deliberately instead.
    """
    components = {
        "config_hash": configuration,
        "schema_versions": schema_versions,
        "thresholds": thresholds,
        "code_commit": code_commit,
        "image_digest": image_digest,
    }

    for (name, value) in components.items():
        if (value is None or value == "" or value == {}):
            raise ValueError(f"pipeline_fingerprint component {name!r} is {value!r}. Every component is "
                             f"required; pass determinism_envelope.UNKNOWN if it genuinely cannot be "
                             f"determined, so that the gap is hashed rather than hidden.")

    rendered = json.dumps(_canonical(components), sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:decimals]


@dataclasses.dataclass(frozen=True)
class DeterminismEnvelope:
    """
    The fields every scored event carries so that the number on it can be reproduced.

    Parameters
    ----------
    tier : str
        One of `TIERS`.
    fingerprint : str
        From `pipeline_fingerprint`.
    configuration : str
        From `config_hash`.
    code_commit : str
        The commit the pipeline was built from.
    image_digest : str
        The container image digest.
    feature_schema_version : str
        The version of the feature schema the scored columns were produced under, for example `TC-5/2.1.0`.
    rng_seed : int
        The seed handed to `morpheus.utils.seed.manual_seed`.
    """

    tier: str
    fingerprint: str
    configuration: str
    code_commit: str
    image_digest: str
    feature_schema_version: str
    rng_seed: int

    def __post_init__(self):
        if (self.tier not in TIERS):
            raise ValueError(f"determinism tier {self.tier!r} is not one of {list(TIERS)}. The tier is a claim "
                             f"a consumer relies on; an unrecognized one cannot be interpreted.")

    def to_columns(self) -> dict:
        """
        The envelope as flat columns, which is how it reaches a SIEM.

        Flat rather than nested, because the guide's own JSON example nests it under `determinism` and every
        sink in Part 4 reads flat fields. The nesting is a document's convenience; a search reads columns.

        Returns
        -------
        dict
            Column name to value.
        """
        return {
            TIER_COLUMN: self.tier,
            FINGERPRINT_COLUMN: self.fingerprint,
            CONFIG_HASH_COLUMN: self.configuration,
            "code_commit": self.code_commit,
            "image_digest": self.image_digest,
            "feature_schema_version": self.feature_schema_version,
            "rng_seed": self.rng_seed,
        }
