#!/usr/bin/env python
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

import pytest

from morpheus.utils.geo_velocity import NS_PER_SECOND
from morpheus.utils.geo_velocity import GeoVelocityTracker
from morpheus.utils.geo_velocity import great_circle_km
from morpheus.utils.geo_velocity import validate_coordinate

HOUR_NS = 3600 * NS_PER_SECOND
ALICE = "alice@example.com"

LONDON = (51.5074, -0.1278)
NEW_YORK = (40.7128, -74.0060)
PARIS = (48.8566, 2.3522)


def tracker(**kwargs) -> GeoVelocityTracker:
    return GeoVelocityTracker(**kwargs)


def test_a_known_distance_is_measured_correctly():
    # London to New York is about 5570 km by great circle. Checked against a published figure rather than against
    # this implementation's own output, which would only prove it is consistent with itself.
    assert great_circle_km(*LONDON, *NEW_YORK) == pytest.approx(5570, abs=15)


def test_london_to_paris_is_the_short_hop_it_should_be():
    assert great_circle_km(*LONDON, *PARIS) == pytest.approx(344, abs=5)


def test_the_same_point_is_no_distance_at_all():
    assert great_circle_km(*LONDON, *LONDON) == pytest.approx(0.0, abs=1e-9)


def test_the_antipode_is_half_the_circumference():
    assert great_circle_km(0.0, 0.0, 0.0, 180.0) == pytest.approx(20015, abs=5)


def test_a_hop_across_the_antimeridian_takes_the_short_way():
    # 179.9E to 179.9W is a fifth of a degree apart, not 359.8 degrees. The formula handles it with no unwrapping,
    # which is worth asserting because the obvious hand-rolled version does not.
    near = great_circle_km(0.0, 179.9, 0.0, -179.9)

    assert near == pytest.approx(22.2, abs=1.0)
    assert near < great_circle_km(0.0, 0.0, 0.0, 1.0) * 1.1


def test_the_first_observation_reports_no_speed():
    result = tracker().observe(ALICE, 0, *LONDON)

    assert result.first_for_entity is True
    assert result.implied_kmh is None
    assert result.distance_km is None


def test_an_ordinary_flight_is_an_ordinary_speed():
    subject = tracker()
    subject.observe(ALICE, 0, *LONDON)

    # Seven hours to New York is well inside what an aircraft does.
    result = subject.observe(ALICE, 7 * HOUR_NS, *NEW_YORK)

    assert result.implied_kmh == pytest.approx(796, abs=5)
    assert result.elapsed_ns == 7 * HOUR_NS
    assert result.elapsed_floored is False


def test_the_same_journey_in_one_hour_is_not():
    subject = tracker()
    subject.observe(ALICE, 0, *LONDON)

    result = subject.observe(ALICE, HOUR_NS, *NEW_YORK)

    assert result.implied_kmh == pytest.approx(5570, abs=15)
    assert result.implied_kmh > 900


def test_two_authentications_at_one_instant_read_as_enormous_rather_than_undefined():
    # The case the floor exists for. A rule written as `implied_kmh >= 900` would otherwise miss the most
    # impossible travel there is, because dividing by a zero elapsed time yields no number to compare.
    subject = tracker()
    subject.observe(ALICE, 0, *LONDON)

    result = subject.observe(ALICE, 0, *NEW_YORK)

    assert result.out_of_order is False
    assert result.elapsed_ns == 0
    assert result.elapsed_floored is True
    assert result.implied_kmh > 900
    # One second of floor over 5570 km.
    assert result.implied_kmh == pytest.approx(5570 * 3600, rel=0.01)


def test_the_true_elapsed_time_is_reported_unfloored():
    # So a consumer can recompute the speed at its own precision rather than inheriting this module's floor.
    subject = tracker(min_elapsed_ns=60 * NS_PER_SECOND)
    subject.observe(ALICE, 0, *LONDON)

    result = subject.observe(ALICE, NS_PER_SECOND, *PARIS)

    assert result.elapsed_ns == NS_PER_SECOND
    assert result.elapsed_floored is True

    # And the speed is computed against the floor, not against the true elapsed time. Asserting the flag alone
    # leaves room for a floor that is reported and not applied, which is a distinction a sabotage found.
    assert result.implied_kmh == pytest.approx(result.distance_km * 60, rel=1e-6)
    assert result.implied_kmh < result.distance_km * 3600


def test_each_observation_becomes_the_anchor_for_the_next():
    subject = tracker()
    subject.observe(ALICE, 0, *LONDON)
    subject.observe(ALICE, HOUR_NS, *PARIS)

    # Measured from Paris, not from London.
    result = subject.observe(ALICE, 2 * HOUR_NS, *PARIS)

    assert result.distance_km == pytest.approx(0.0, abs=1e-3)


def test_principals_do_not_share_a_location():
    subject = tracker()
    subject.observe(ALICE, 0, *LONDON)

    result = subject.observe("bob@example.com", HOUR_NS, *NEW_YORK)

    assert result.first_for_entity is True
    assert result.implied_kmh is None


def test_an_out_of_order_observation_measures_nothing_and_moves_nothing():
    subject = tracker()
    subject.observe(ALICE, 10 * HOUR_NS, *LONDON)

    late = subject.observe(ALICE, 0, *NEW_YORK)

    assert late.out_of_order is True
    assert late.implied_kmh is None

    # The anchor is still London, not the record that arrived late.
    after = subject.observe(ALICE, 11 * HOUR_NS, *LONDON)

    assert after.distance_km == pytest.approx(0.0, abs=1e-3)


def test_a_coordinate_outside_the_globe_is_refused():
    # A nonsense distance would read as impossible travel, which is an alert about a geolocation database rather
    # than about a principal.
    subject = tracker()

    with pytest.raises(ValueError, match="not inside the globe"):
        subject.observe(ALICE, 0, 91.0, 0.0)

    with pytest.raises(ValueError, match="not inside the globe"):
        subject.observe(ALICE, 0, 0.0, 181.0)


@pytest.mark.parametrize(("latitude", "longitude"), [(0.0, 0.0), (90.0, 180.0), (-90.0, -180.0), (51, -1)],
                         ids=["origin", "north_east_corner", "south_west_corner", "integers"])
def test_a_usable_coordinate_validates(latitude, longitude):
    assert validate_coordinate(latitude, longitude) is True


@pytest.mark.parametrize(("latitude", "longitude"),
                         [(None, 0.0), (0.0, None), ("51.5", -0.1), (float("nan"), 0.0), (91.0, 0.0), (0.0, 181.0),
                          (True, False)],
                         ids=["null_lat", "null_lon", "string", "nan", "off_globe", "off_meridian", "booleans"])
def test_an_unusable_coordinate_does_not(latitude, longitude):
    # Booleans are the interesting one: `bool` is an `int`, so a column of flags reaching this by accident would
    # otherwise place every row on the equator and measure journeys to it.
    assert validate_coordinate(latitude, longitude) is False


def test_the_least_recently_seen_entity_is_forgotten():
    subject = tracker(max_entities=2)

    subject.observe("a", 0, *LONDON)
    subject.observe("b", HOUR_NS, *LONDON)
    subject.observe("c", 2 * HOUR_NS, *LONDON)

    assert subject.tracked_entities == 2

    # "a" was evicted, so its next observation reports no speed rather than one measured from a forgotten place.
    revived = subject.observe("a", 3 * HOUR_NS, *NEW_YORK)

    assert revived.first_for_entity is True


def test_replaying_a_stream_reproduces_it():
    journey = [(0, LONDON), (HOUR_NS, PARIS), (2 * HOUR_NS, NEW_YORK), (2 * HOUR_NS, LONDON)]

    def run():
        subject = tracker()

        return [subject.observe(ALICE, time, *place).implied_kmh for (time, place) in journey]

    assert run() == run()


def test_the_constructor_refuses_values_it_cannot_use():
    with pytest.raises(ValueError, match="min_elapsed_ns must be positive"):
        GeoVelocityTracker(min_elapsed_ns=0)

    with pytest.raises(ValueError, match="max_entities must be positive"):
        GeoVelocityTracker(max_entities=0)
