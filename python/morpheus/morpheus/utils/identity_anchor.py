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
The name an entity was first reported under, held against an identifier that does not change when the name does.

A poller reports a port as the name the device gives it, and a device as its hostname. Both are configuration: an
operator renames `Gi1/0/5` to `uplink-core-2`, a switch is re-hostnamed when it moves rack. A key composed from those
names starts a new entity at each rename, so every per-port baseline relearns from nothing and every "first seen"
fires again on hardware that has been there for years. Gap G39 in the retrospective.

**The anchor is what the hardware reports about itself.** For a device, its chassis serial. For a port, its
`ifIndex`, which is stable across renames and, on a device configured to persist it, across reboots. The first name
an anchor is seen with becomes the entity's name for as long as the anchor is remembered; later names are reported
beside it, not substituted for it.

**An anchor that is not persistent is worse than none.** A device that renumbers `ifIndex` at boot hands one port's
index to another, and anchoring on it merges two ports' histories. Pass a port anchor only for an estate that
persists it (`snmp ifindex persist`, or the vendor's equivalent); otherwise the key stays the reported name, which is
the old behaviour and fails the other way, by splitting.

**No anchor, no change.** A row whose anchor is missing keeps its reported name, so the stage behaves exactly as it
did before anchors existed for every collector that does not send one.
"""

import collections
import typing

from morpheus.utils.entity_key import normalize_text
from morpheus.utils.tracker_state import StateMixin

DEFAULT_MAX_ANCHORS = 100_000
"""Anchors remembered before the least recently seen is forgotten. A forgotten anchor's next name becomes its name."""


class AnchoredNames(StateMixin):
    """
    The canonical name for each anchor within a scope, the first name the anchor was reported with.

    Parameters
    ----------
    max_anchors : int, default = 100000
        Anchors remembered, least recently seen dropped first.

    Raises
    ------
    ValueError
        If `max_anchors` is not positive.
    """

    def __init__(self, max_anchors: int = DEFAULT_MAX_ANCHORS):
        if (max_anchors <= 0):
            raise ValueError(f"max_anchors must be positive, received {max_anchors}")

        self._max_anchors = max_anchors
        self._names: collections.OrderedDict[tuple, str] = collections.OrderedDict()

    def resolve(self,
                scope: typing.Sequence[typing.Any],
                anchor: typing.Any,
                reported: typing.Any,
                learn: bool = True) -> typing.Any:
        """
        The name to key an entity on.

        Parameters
        ----------
        scope : sequence
            What the anchor is unique within: the site for a chassis serial, the site and the canonical device for an
            `ifIndex`.
        anchor : any
            The stable identifier, or a missing value.
        reported : any
            The name the collector reported.
        learn : bool, default = True
            Whether this row may teach the map an anchor it has not seen, or refresh how recently one was seen. A row
            the stage refuses passes `False`, so it is named consistently but moves nothing.

        Returns
        -------
        any
            The first name this anchor was reported with in this scope, or `reported` itself when the anchor, the
            scope or the name is missing. A missing name is never remembered, so a later row with a name supplies it.
        """
        anchor_text = normalize_text(anchor)
        scope_text = tuple(normalize_text(part) for part in scope)

        if (anchor_text is None or normalize_text(reported) is None or any(part is None for part in scope_text)):
            return reported

        key = scope_text + (anchor_text, )
        name = self._names.get(key)

        if (not learn):
            return reported if name is None else name

        if (name is None):
            name = normalize_text(reported)
            self._names[key] = name

        self._names.move_to_end(key)

        while (len(self._names) > self._max_anchors):
            self._names.popitem(last=False)

        return name

    def __len__(self) -> int:
        return len(self._names)


class PortIdentity(StateMixin):
    """
    A port's canonical names: its device by chassis serial within the site, the port by `ifIndex` within that device.

    The one rule both layer 1 stages key a port by, so the counters and the bindings agree on which port a renamed
    one is.

    Parameters
    ----------
    max_anchors : int, default = 100000
        Anchors remembered for devices, and separately for ports.
    """

    def __init__(self, max_anchors: int = DEFAULT_MAX_ANCHORS):
        self._devices = AnchoredNames(max_anchors)
        self._ports = AnchoredNames(max_anchors)

    def names(self,
              site: typing.Any,
              device: typing.Any,
              port: typing.Any,
              device_anchor: typing.Any,
              port_anchor: typing.Any,
              learn: bool = True) -> tuple:
        """
        The site, device and port to key the row on, as `(site, device, port)`.

        The port's anchor is scoped by the canonical device, so a re-hostnamed switch keeps its ports as well.
        `learn` is as `AnchoredNames.resolve` takes it.
        """
        device = self._devices.resolve((site, ), device_anchor, device, learn=learn)
        port = self._ports.resolve((site, device), port_anchor, port, learn=learn)

        return (site, device, port)
