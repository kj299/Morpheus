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
What the estate says about its own people, machines and applications, handed to a composition as an input.

The TC-0 context a composition enriches with -- the identity directory, the asset inventory, the CMDB's application
map -- is a property of the estate the pipeline runs in, not of the pipeline. Until plan section 4.5b most
compositions built it themselves, from a synthetic estate written into their own modules, so a segment runner fed a
real estate's traffic enriched it with that fixture: a real workstation's peer group came from the test data, or was
absent because the test data did not name it. Each composition now takes an `EstateContext` and refuses to run
without the stores it reads, and nothing in the package holds an estate.

**The context arrives the way a deployment receives it.** The context segment turns the identity provider's, the
asset inventory's and the CMDB's records into `context:identity`, `context:asset` and `context:application` rows;
`from_directory` reads those rows back, one JSON lines file per kind, and rebuilds each store with
`BitemporalStore.from_records`, the same rule a consumer of those sourcetypes uses. `write_directory` is the inverse,
so a corpus's stores can be written as sample context and read back to identical versions.
"""

import dataclasses
import json
import os
import typing

from morpheus.utils.binding_table import BindingTable
from morpheus.utils.bitemporal import BitemporalStore

KINDS = ("identity", "asset", "application")
"""The three context stores, by the name each is built under and the sourcetype it arrives on."""

EXTRAS_FILE = "extras.json"
"""Where `from_directory` reads the extras a directory carries: one JSON object, name to value."""


def file_name(kind: str) -> str:
    """The file a kind's records are read from: `context_<kind>.jsonlines`, as the sample events name them."""
    return f"context_{kind}.jsonlines"


class MissingContext(ValueError):
    """A composition was asked to run without context it enriches with."""


@dataclasses.dataclass(frozen=True)
class EstateContext:
    """
    The estate's context stores. Each is optional here; a composition says which it needs.

    Parameters
    ----------
    identity : `BitemporalStore`, optional
        Who each principal is: department, role, groups, status.
    asset : `BitemporalStore`, optional
        What each host or address is: role, criticality, owner, peer group, operating system.
    application : `BitemporalStore`, optional
        Which application answers at each address and port.
    directory : `BindingTable`, optional
        The identity provider's record of which 802.1X identity each principal presents, and for how long. Read by
        the estate composition, which joins a sign-in to the desk port its principal authenticated on; not one of
        the three `context:*` stores, so `from_directory` does not read it.
    extras : mapping, optional
        Inputs one composition alone reads, by name: the campaign's site coordinates and management prefixes, for
        example, which the estate supplies as it supplies the directory. Read with `require_extra`.
    """

    identity: typing.Optional[BitemporalStore] = None
    asset: typing.Optional[BitemporalStore] = None
    application: typing.Optional[BitemporalStore] = None
    directory: typing.Optional[BindingTable] = None
    extras: typing.Mapping[str, typing.Any] = dataclasses.field(default_factory=dict)

    def require(self, composition: str, *kinds: str) -> tuple:
        """
        The stores a composition reads, in the order named.

        Raises
        ------
        MissingContext
            If any is absent. Enriching without it would write null context onto every row, which a rule reads as
            "unknown host" rather than as "context not supplied".
        """
        missing = [kind for kind in kinds if getattr(self, kind) is None]

        if (missing):
            raise MissingContext(f"{composition} enriches with the estate's {', '.join(missing)} context and none was "
                                 f"given. Pass an EstateContext holding it, read with EstateContext.from_directory "
                                 f"from the context segment's {', '.join(file_name(kind) for kind in missing)}.")

        return tuple(getattr(self, kind) for kind in kinds)

    def require_extra(self, composition: str, *names: str) -> tuple:
        """
        The named inputs from `extras`, in the order named.

        Raises
        ------
        MissingContext
            If any is absent.
        """
        missing = [name for name in names if name not in self.extras]

        if (missing):
            raise MissingContext(f"{composition} reads the estate's {', '.join(missing)}, and none was given in the "
                                 f"context's extras.")

        return tuple(self.extras[name] for name in names)

    @classmethod
    def from_directory(cls, directory: str) -> "EstateContext":
        """
        Read whichever of the three kinds' record files the directory holds, and its extras; what is absent is absent.

        Extras are read from `extras.json`, and only those JSON can carry: a list, a mapping, a number, a string.
        """
        stores = {}
        extras_path = os.path.join(directory, EXTRAS_FILE)

        if (os.path.exists(extras_path)):
            with open(extras_path, encoding="utf-8") as handle:
                stores["extras"] = json.load(handle)

        for kind in KINDS:
            path = os.path.join(directory, file_name(kind))

            if (os.path.exists(path)):
                with open(path, encoding="utf-8") as handle:
                    records = [json.loads(line) for line in handle if line.strip()]

                stores[kind] = BitemporalStore.from_records(kind, records)

        return cls(**stores)

    def render(self) -> dict[str, str]:
        """
        Each file `write_directory` writes, by name, with its text: a kind's records one JSON object per line, and the
        extras JSON can carry. An extra it cannot (a store, a binding table) is left out, and a directory read back
        without it fails `require_extra` rather than running without it.
        """
        rendered = {}

        for kind in KINDS:
            store = getattr(self, kind)

            if (store is not None):
                rendered[file_name(kind)] = "".join(
                    json.dumps(record, sort_keys=True) + "\n" for record in store.to_records())

        serializable = {name: value for (name, value) in self.extras.items() if _is_json(value)}

        if (serializable):
            rendered[EXTRAS_FILE] = json.dumps(serializable, indent=2, sort_keys=True) + "\n"

        return rendered

    def write_directory(self, directory: str) -> list[str]:
        """Write the files `render` names into `directory`; return their paths."""
        os.makedirs(directory, exist_ok=True)
        written = []

        for (name, text) in self.render().items():
            path = os.path.join(directory, name)

            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)

            written.append(path)

        return written


def _is_json(value: typing.Any) -> bool:
    try:
        json.dumps(value)
    except TypeError:
        return False

    return True
