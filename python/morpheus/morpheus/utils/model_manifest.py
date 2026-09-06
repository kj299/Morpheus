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
Which model version scored which entity, decided once at the start of a window and held for all of it.

This is determinism control 1. `DFPInferenceStage` resolves a model by calling
`ModelManager.load_user_model(client, user_id, fallback_user_ids)`, which returns whatever version is current
and caches it for ten minutes across at most ten entities. Two consequences the guide names: a retraining event
part-way through a run changes scores, and a cache eviction can change them back. Neither is visible in the
output, so a score that moved because the model moved is indistinguishable from one that moved because the
entity did -- which is the whole question a behavioral pipeline exists to answer.

**A manifest is resolved once and refuses to answer for any other window.** Holding it "for the whole window" is
the control, and a data structure that will happily answer a question from the next window makes that an
instruction rather than a guarantee. `resolve` therefore takes the window it is being asked about and raises if
it is not the one the manifest was built for. A caller that wants the next window builds the next manifest,
which is the moment a new model version is allowed in.

**A missing model is refused rather than substituted, unless a fallback is declared.** `DFPInferenceStage`
silently falls back to `generic_user`. An event scored against a population model is a different claim from one
scored against the entity's own, and the difference has to be visible: where a fallback is declared, the
resolution says so and `model_fallback_used` reaches the SIEM; where none is, resolving an unknown entity
raises, because scoring it against nothing in particular and reporting a number is worse than not scoring it.

This module resolves nothing itself. It holds what a resolver decided and enforces how that decision is used --
no MLflow client, no registry, no network. What produces the mapping is a deployment's business; what this
guarantees is that the mapping does not change underneath a window.
"""

import dataclasses
import typing

MODEL_VERSION_COLUMN = "model_version"
MODEL_FALLBACK_COLUMN = "model_fallback_used"


@dataclasses.dataclass(frozen=True)
class Resolution:
    """
    Which model an entity is scored against, and whether it is the entity's own.

    Attributes
    ----------
    model_version : str
        The pinned model, as `name:version`. Never a bare name: a name alone is the "latest" resolution this
        control exists to forbid.
    fallback_used : bool
        The entity had no model of its own and was scored against the declared fallback. An event carrying
        `True` is a claim about a population, not about this entity's own history.
    """

    model_version: str
    fallback_used: bool


@dataclasses.dataclass(frozen=True)
class ModelManifest:
    """
    The models pinned for one window.

    Parameters
    ----------
    window_id : int
        The window this manifest was resolved for. `resolve` refuses any other, which is what makes "held for
        the whole window" a property rather than a convention.
    models : dict
        Entity key to `name:version`.
    fallback : str, optional
        The `name:version` to use for an entity with no model of its own. `None` means an unknown entity is
        refused rather than scored against a population model.
    """

    window_id: int
    models: dict
    fallback: typing.Optional[str] = None

    def __post_init__(self):
        for (entity, version) in self.models.items():
            _require_pinned(version, f"model for {entity!r}")

        if (self.fallback is not None):
            _require_pinned(self.fallback, "fallback model")

    @property
    def entities(self) -> int:
        """How many entities carry a model of their own."""
        return len(self.models)

    def resolve(self, entity: typing.Optional[str], window_id: int) -> Resolution:
        """
        The model this entity is scored against in this window.

        Parameters
        ----------
        entity : str or None
            The entity key.
        window_id : int
            The window being scored. Must be the one this manifest was resolved for.

        Returns
        -------
        `Resolution`

        Raises
        ------
        ValueError
            If the window is not this manifest's, or the entity has no model and no fallback is declared.
        """
        if (window_id != self.window_id):
            raise ValueError(f"manifest was resolved for window {self.window_id} and was asked about "
                             f"{window_id}. A window scored against models pinned for a different one is the "
                             f"defect this control exists to prevent; build a manifest per window.")

        version = self.models.get(entity) if entity is not None else None

        if (version is not None):
            return Resolution(model_version=version, fallback_used=False)

        if (self.fallback is None):
            raise ValueError(f"no model is pinned for {entity!r} and no fallback is declared. Scoring it "
                             f"against nothing in particular and reporting a number is worse than not scoring "
                             f"it; declare a fallback if a population model is acceptable here.")

        return Resolution(model_version=self.fallback, fallback_used=True)


def _require_pinned(version: str, what: str) -> None:
    """A version must name a version. A bare name resolves to whatever is current, which is the defect."""
    if (not isinstance(version, str) or ":" not in version or version.rsplit(":", 1)[1] == ""):
        raise ValueError(f"{what} is {version!r}, which does not pin a version. Use 'name:version'; a bare "
                         f"name resolves to whatever is current, which is what control 1 forbids.")
