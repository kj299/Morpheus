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
How fast a principal would have had to travel between two consecutive authentications to make both of them.

This is the measurement R-D-L5-003 reads. The rule is one of the few at layer 5 that needs no model: two
successful authentications, a great-circle distance, an elapsed time, and a speed no aircraft reaches.

**The tracker measures; it does not decide what to measure.** Which authentications are eligible -- successful
ones, not from a VPN egress range, not token refreshes carrying the original location -- is the caller's
judgement, and `morpheus.stages.telemetry.tc5_travel_stage` makes it. An excluded record must not become the
anchor the next one is measured against, so the caller simply does not observe it, and the previous eligible
location stays where it was.

**Elapsed time has a floor, and that floor is what keeps the worst case from reading as the best one.** Two
successful authentications from opposite sides of the world bearing the same timestamp are the most impossible
travel there is, and dividing by a zero elapsed time yields either an exception or a null -- so a rule written as
`implied_kmh >= 900` would miss precisely the case it exists to catch. Most identity providers stamp to the
second, so two records an instant apart are really within one, and the floor says so: the speed is computed
against `max(elapsed, min_elapsed)` and comes out enormous rather than undefined. The true elapsed time is
reported beside it, unfloored, and a flag says when the floor bound.

**On the arithmetic.** The distance is the haversine formula on a sphere of the IUGG mean radius, which is
accurate to about half a percent against the ellipsoid -- far inside the margin of an IP geolocation database,
which is what supplies these coordinates and is routinely wrong by a city. Longitude needs no unwrapping: the
formula takes the sine of half the difference, and `sin` is symmetric about a right angle, so a hop across the
antimeridian measures as the short way round without special handling.

Both outputs are quantized under determinism control 9. That is the one place here where a platform could in
principle disagree: `sin`, `cos` and `asin` are library functions whose last bit is not guaranteed identical
across implementations, and a value sitting exactly on a rounding boundary could quantize two ways. A rule
thresholding at 900 km/h is nowhere near such a boundary, and one written to be sensitive at the fourth decimal
of a figure derived from a city-accurate coordinate would be measuring the wrong thing anyway.
"""

import collections
import dataclasses
import math
import typing

from morpheus.utils.determinism import DEFAULT_FLOAT_DECIMALS
from morpheus.utils.determinism import quantize_value

NS_PER_SECOND = 10**9

EARTH_RADIUS_KM = 6371.0088
"""IUGG mean radius. Named rather than inlined so the choice is reviewable."""

DEFAULT_MIN_ELAPSED_NS = NS_PER_SECOND
"""Floor on elapsed time, one second.

Identity providers stamp to the second, so two records at the same instant are really within one. The floor is
what makes an impossible speed enormous rather than undefined.
"""

DEFAULT_MAX_ENTITIES = 100_000
"""Principals holding a previous location before the least recently seen is forgotten."""


@dataclasses.dataclass(frozen=True)
class GeoVelocityResult:
    """
    The outcome of observing one eligible authentication.

    Attributes
    ----------
    distance_km : float or None
        Great-circle distance from the principal's previous eligible authentication. Quantized. `None` where
        there is no previous one to measure from.
    elapsed_ns : int or None
        True elapsed time since that authentication, unfloored, so a consumer can recompute the speed at its own
        precision. `None` where there is no previous one.
    implied_kmh : float or None
        `distance_km` divided by the elapsed time, with the floor applied. Quantized. `None` where there is no
        previous authentication.
    elapsed_floored : bool
        The elapsed time was below the floor, so `implied_kmh` is computed against the floor rather than against
        `elapsed_ns`. A rule reading a very large speed should look at this before believing the figure to four
        decimal places.
    first_for_entity : bool
        This principal had no previous eligible authentication. The location is now recorded and the next one
        will be measured against it.
    out_of_order : bool
        The authentication's event time was not after the previous one's. Nothing is measured and the recorded
        location is left alone, because measuring backwards would report a speed for a journey in reverse.
    """

    distance_km: typing.Optional[float]
    elapsed_ns: typing.Optional[int]
    implied_kmh: typing.Optional[float]
    elapsed_floored: bool
    first_for_entity: bool
    out_of_order: bool


@dataclasses.dataclass
class _Location:
    latitude: float
    longitude: float
    event_time_ns: int


def great_circle_km(latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float) -> float:
    """
    Great-circle distance between two coordinates, in kilometres, unquantized.

    Public because it is the arithmetic a reader will want to check independently, and because a caller with two
    coordinates and no stream has no reason to build a tracker to get one number.

    Parameters
    ----------
    latitude_a, longitude_a : float
        The first coordinate, in degrees.
    latitude_b, longitude_b : float
        The second coordinate, in degrees.

    Returns
    -------
    float
        Distance in kilometres.
    """
    (phi_a, phi_b) = (math.radians(latitude_a), math.radians(latitude_b))
    delta_phi = math.radians(latitude_b - latitude_a)
    delta_lambda = math.radians(longitude_b - longitude_a)

    # The sine of half the longitude difference is symmetric about a right angle, so a hop across the antimeridian
    # measures the short way round with no wrapping.
    haversine = (math.sin(delta_phi / 2)**2 + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2)**2)

    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(haversine)))


def validate_coordinate(latitude: typing.Any, longitude: typing.Any) -> bool:
    """
    Whether a pair of values is a usable coordinate.

    Offered so a caller can screen a collector's output without catching an exception per row. A coordinate
    outside the globe is data rather than a programming error, and the right thing to do with it is to decline to
    measure -- a nonsense distance would read as impossible travel, which is an alert about a geolocation
    database rather than about a principal.

    Parameters
    ----------
    latitude, longitude : any
        Candidate values.

    Returns
    -------
    bool
        True where both are real numbers inside the globe's bounds.
    """
    if (isinstance(latitude, bool) or isinstance(longitude, bool)):
        # `bool` is an `int`, and a column of flags reaching this by accident would place every row at the equator.
        return False

    if (not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float))):
        return False

    if (latitude != latitude or longitude != longitude):  # pylint: disable=comparison-with-itself
        return False

    return -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0


class GeoVelocityTracker:
    """
    Per-entity implied travel speed between consecutive observations.

    The tracker holds one previous location per entity, so its memory is bounded by the entity count rather than
    by the stream. Results depend only on the sequence of observations it has been shown, so replaying a stream
    reproduces them.

    Parameters
    ----------
    min_elapsed_ns : int, default = 1 second
        Floor on the elapsed time used for the speed. See the module docstring: without it, the most extreme case
        divides by zero.
    max_entities : int, default = 100000
        Entities retained before the least recently seen is dropped. A dropped entity's next observation reads as
        its first, which reports no speed rather than one measured from a location that has been forgotten.
    decimals : int, default = 4
        Decimal places the distance and the speed are rounded to, under determinism control 9.
    """

    def __init__(self,
                 min_elapsed_ns: int = DEFAULT_MIN_ELAPSED_NS,
                 max_entities: int = DEFAULT_MAX_ENTITIES,
                 decimals: int = DEFAULT_FLOAT_DECIMALS):
        if (min_elapsed_ns <= 0):
            raise ValueError(f"min_elapsed_ns must be positive, received {min_elapsed_ns}")

        if (max_entities <= 0):
            raise ValueError(f"max_entities must be positive, received {max_entities}")

        self._min_elapsed_ns = min_elapsed_ns
        self._max_entities = max_entities
        self._decimals = decimals

        self._locations: collections.OrderedDict[str, _Location] = collections.OrderedDict()

    @property
    def tracked_entities(self) -> int:
        """Entities currently holding a previous location."""
        return len(self._locations)

    def observe(self, entity_key: str, event_time_ns: int, latitude: float, longitude: float) -> GeoVelocityResult:
        """
        Record one eligible authentication and return the speed it implies.

        Parameters
        ----------
        entity_key : str
            The principal this authentication belongs to.
        event_time_ns : int
            Event time, in nanoseconds since the epoch.
        latitude : float
            Where the authentication came from, in degrees of latitude.
        longitude : float
            Where the authentication came from, in degrees of longitude.

        Returns
        -------
        `GeoVelocityResult`

        Raises
        ------
        ValueError
            If the coordinate is outside the globe. Screen with `validate_coordinate` rather than catching this:
            a bad coordinate is a fact about a geolocation database and belongs in a counted warning, not in an
            exception per row.
        """
        if (not validate_coordinate(latitude, longitude)):
            raise ValueError(f"coordinate ({latitude}, {longitude}) is not inside the globe")

        previous = self._locations.get(entity_key)

        if (previous is None):
            self._locations[entity_key] = _Location(float(latitude), float(longitude), event_time_ns)
            self._evict()

            return GeoVelocityResult(distance_km=None,
                                     elapsed_ns=None,
                                     implied_kmh=None,
                                     elapsed_floored=False,
                                     first_for_entity=True,
                                     out_of_order=False)

        self._locations.move_to_end(entity_key)

        if (event_time_ns < previous.event_time_ns):
            # Measuring backwards would report a speed for a journey in reverse, and admitting the record would
            # then measure the next one against a location that arrived late. Equal timestamps are not out of
            # order: two authentications at one instant from two places is the signal, not a duplicate.
            return GeoVelocityResult(distance_km=None,
                                     elapsed_ns=None,
                                     implied_kmh=None,
                                     elapsed_floored=False,
                                     first_for_entity=False,
                                     out_of_order=True)

        distance_km = great_circle_km(previous.latitude, previous.longitude, latitude, longitude)
        elapsed_ns = event_time_ns - previous.event_time_ns
        floored = elapsed_ns < self._min_elapsed_ns
        divisor_ns = self._min_elapsed_ns if floored else elapsed_ns
        hours = divisor_ns / (3600 * NS_PER_SECOND)

        previous.latitude = float(latitude)
        previous.longitude = float(longitude)
        previous.event_time_ns = event_time_ns

        return GeoVelocityResult(distance_km=quantize_value(distance_km, decimals=self._decimals),
                                 elapsed_ns=elapsed_ns,
                                 implied_kmh=quantize_value(distance_km / hours, decimals=self._decimals),
                                 elapsed_floored=floored,
                                 first_for_entity=False,
                                 out_of_order=False)

    def _evict(self) -> None:
        """Forget the least recently seen entities until the cap holds."""
        while (len(self._locations) > self._max_entities):
            self._locations.popitem(last=False)
