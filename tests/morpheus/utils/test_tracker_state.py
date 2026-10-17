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
"""
Every tracker resumes from its snapshot as if it had never stopped.

Each case drives a tracker through a seeded sequence of observations chosen to reach its eviction bound, stops it
halfway, writes its state to bytes, rebuilds a tracker from those bytes alone and finishes the sequence on it. The
rebuilt tracker must give the same answer to every remaining observation as one that ran straight through, and end
in byte-identical state.
"""

import ast
import collections
import fractions
import importlib
import json
import os
import random
import subprocess
import sys
import typing

import numpy as np
import pandas as pd
import pytest

import morpheus.utils
from morpheus.stages.telemetry.tc1_device_stage import RebootCounter
from morpheus.utils.arrival_regularity import ArrivalRegularityTracker
from morpheus.utils.binding_closer import BindingCloser
from morpheus.utils.bucket_peak import BucketPeakTracker
from morpheus.utils.counter_delta import CounterTracker
from morpheus.utils.cyclic_histogram import CyclicHistogramTracker
from morpheus.utils.distinct_window import DistinctWindowTracker
from morpheus.utils.drift_trajectory import DriftTrajectoryTracker
from morpheus.utils.established_value import ValueHistoryTracker
from morpheus.utils.event_clock import EventClock
from morpheus.utils.geo_velocity import GeoVelocityTracker
from morpheus.utils.link_flap import LinkFlapTracker
from morpheus.utils.optical_baseline import OpticalBaselineTracker
from morpheus.utils.optical_forecast import OpticalForecastTracker
from morpheus.utils.outcome_run import OutcomeRunTracker
from morpheus.utils.pair_history import PairHistoryTracker
from morpheus.utils.ratio_window import RatioWindowTracker
from morpheus.utils.session_timer import SessionTimer
from morpheus.utils.tracker_state import StateError
from morpheus.utils.tracker_state import decode
from morpheus.utils.tracker_state import encode
from morpheus.utils.tracker_state import from_bytes
from morpheus.utils.tracker_state import state_digest
from morpheus.utils.tracker_state import to_bytes
from morpheus.utils.transfer_envelope import TransferEnvelopeTracker
from morpheus.utils.ttl_profile import TtlProfileTracker
from morpheus.utils.value_novelty import ValueNoveltyTracker
from morpheus.utils.window_seal import WindowSealer

NS = 10**9
MINUTE_NS = 60 * NS
HOUR_NS = 60 * MINUTE_NS
STEPS = 240
"""Observations per case: enough for every tracker to mature, evict and, where it can, expire."""

KEYS = [f"entity-{index}" for index in range(5)]
"""More entities than each tracker is allowed to hold, so the eviction order is part of what must survive."""

Op = typing.Callable[[typing.Any], typing.Any]


def _times(rng: random.Random) -> typing.Iterator[int]:
    """Event times that mostly advance and sometimes regress, so the out-of-order paths are in the snapshot too."""
    now = 0

    while True:
        now += rng.choice([MINUTE_NS, 5 * MINUTE_NS, 20 * MINUTE_NS, HOUR_NS])
        yield now - (2 * MINUTE_NS if rng.random() < 0.1 else 0)


def _keyed(rng: random.Random, observe: typing.Callable[[typing.Any, str, int], typing.Any]) -> list[Op]:
    times = _times(rng)
    ops = []

    for _ in range(STEPS):
        (key, now, draw) = (rng.choice(KEYS), next(times), rng.random())
        ops.append(lambda tracker, key=key, now=now, draw=draw: observe(tracker, key, now, draw))

    return ops


def _binding_closer_ops(rng: random.Random) -> list[Op]:
    macs = ["00:00:5e:00:53:01", "00:00:5e:00:53:02", "00:00:5e:00:53:03"]

    def step(tracker: BindingCloser, key: str, now: int, draw: float):
        if (draw < 0.1):
            return tracker.close(key, now)

        if (draw < 0.15):
            return tracker.expire(now)

        return tracker.observe(key, now, {"mac": macs[int(draw * 100) % 3]})

    return _keyed(rng, step)


def _session_timer_ops(rng: random.Random) -> list[Op]:

    def step(tracker: SessionTimer, key: str, now: int, draw: float):
        if (draw < 0.5):
            return tracker.begin(key, now)

        if (draw < 0.9):
            return tracker.complete(key, now, outcome="success" if draw < 0.8 else "failure")

        return tracker.expire(now)

    return _keyed(rng, step)


def _window_sealer_ops(rng: random.Random) -> list[Op]:
    times = _times(rng)
    ops = []

    for _ in range(STEPS // 4):
        batch = [next(times) for _ in range(rng.randint(1, 6))]
        ops.append(lambda tracker, batch=batch: tracker.observe(batch))

    return ops + [lambda tracker: tracker.flush()]


def _event_clock_ops(rng: random.Random) -> list[Op]:
    times = _times(rng)
    ops = []

    for _ in range(STEPS):
        now = next(times)
        # An occasional row a year ahead, which the clock must refuse whether or not it has restarted.
        now += 365 * 24 * HOUR_NS if rng.random() < 0.05 else 0
        ops.append(lambda tracker, now=now: tracker.accept(now))

    return ops


def _keyed_by(observe: typing.Callable) -> typing.Callable[[random.Random], list[Op]]:
    return lambda rng: _keyed(rng, observe)


def _arrival(tracker, key, now, draw):
    return tracker.observe(key, now, size=round(draw * 1500, 3))


def _peak(tracker, key, now, draw):
    return tracker.observe(key, now, int(draw * 10))


def _counters(tracker, key, now, draw):
    counters = {"in_octets": int(draw * 2**32), "in_errors": int(draw * 50)}

    # An uptime that falls behind the clock now and then is a reboot, so the reset path is in the snapshot too.
    return tracker.observe(key, now, counters, uptime_ns=now if draw > 0.1 else now // 7)


def _hour(tracker, key, now, draw):
    return tracker.observe(key, now, int(draw * 24))


def _distinct(tracker, key, now, draw):
    return tracker.observe(key, now, f"value-{int(draw * 8)}")


def _drift(tracker, key, now, draw):
    return tracker.observe(key, now // HOUR_NS, draw * 3)


def _established(tracker, key, now, draw):
    return tracker.observe(key, now, f"TLS1.{int(draw * 3) + 1}")


def _travel(tracker, key, now, draw):
    return tracker.observe(key, now, draw * 80 - 40, draw * 300 - 150)


def _flap(tracker, key, now, draw):
    return tracker.observe(key, now, "up" if draw < 0.7 else "down", last_change_ns=now - int(draw * HOUR_NS))


def _optics(tracker, key, now, draw):
    return tracker.observe(key, now, {"rx_dbm": -3.0 - draw, "tx_dbm": None if draw < 0.1 else -2.0 + draw / 10})


def _forecast(tracker, key, now, draw):
    reading = -5.0 - draw - now / (1000 * HOUR_NS)

    return tracker.observe(key, now, reading, floor_dbm=-14.0, optic=f"SN{int(draw * 2)}")


def _outcome(tracker, key, now, draw):
    return tracker.observe(key, now, draw < 0.2)


def _pair(tracker, key, now, draw):
    return tracker.observe(key, now, ("10.0.0.1", f"10.0.1.{int(draw * 9)}"))


def _ratio(tracker, key, now, draw):
    return tracker.observe(key, now, draw < 0.3)


def _reboots(tracker, key, now, draw):
    return tracker.count(tracker.state(key), now, draw < 0.3)


def _transfer(tracker, key, now, draw):
    return tracker.observe(key, now, draw * 10**6)


def _ttl(tracker, key, now, draw):
    return tracker.observe(key, now, [64, 63, 128, 255][int(draw * 4)])


def _identifiers(tracker, key, now, draw):
    return tracker.observe(key, now, {"serial": f"SN{int(draw * 6)}", "chassis": f"ch-{int(draw * 2)}"})


CASES = {
    "arrival_regularity": (lambda: ArrivalRegularityTracker(min_intervals=3, max_entities=3), _keyed_by(_arrival)),
    "binding_closer": (lambda: BindingCloser(["mac"], idle_timeout_ns=3 * HOUR_NS, max_open=3), _binding_closer_ops),
    "bucket_peak": (lambda: BucketPeakTracker(bucket_ns=HOUR_NS, window_ns=24 * HOUR_NS, min_buckets=3, max_entities=3),
                    _keyed_by(_peak)),
    "counter_delta": (lambda: CounterTracker(["in_octets", "in_errors"], counter_bits=32, max_entities=3),
                      _keyed_by(_counters)),
    "cyclic_histogram": (lambda: CyclicHistogramTracker(24, min_samples=3, max_entities=3), _keyed_by(_hour)),
    "distinct_window": (lambda: DistinctWindowTracker(window_ns=6 * HOUR_NS, max_entities=3), _keyed_by(_distinct)),
    "drift_trajectory": (lambda: DriftTrajectoryTracker(min_windows=2, max_entities=3), _keyed_by(_drift)),
    "established_value": (lambda: ValueHistoryTracker(min_samples=2, max_entities=3), _keyed_by(_established)),
    "event_clock": (lambda: EventClock(max_skew_ns=24 * HOUR_NS), _event_clock_ops),
    "geo_velocity": (lambda: GeoVelocityTracker(max_entities=3), _keyed_by(_travel)),
    "link_flap": (lambda: LinkFlapTracker(max_entities=3), _keyed_by(_flap)),
    "optical_baseline": (lambda: OpticalBaselineTracker(["rx_dbm", "tx_dbm"], min_samples=3, max_entities=3),
                         _keyed_by(_optics)),
    "optical_forecast": (lambda: OpticalForecastTracker(min_samples=3, max_entities=3), _keyed_by(_forecast)),
    "outcome_run": (lambda: OutcomeRunTracker(max_entities=3), _keyed_by(_outcome)),
    "pair_history": (lambda: PairHistoryTracker(max_entities=3), _keyed_by(_pair)),
    "ratio_window": (lambda: RatioWindowTracker(min_denominator=2, max_entities=3), _keyed_by(_ratio)),
    "reboot_counter": (lambda: RebootCounter(bucket_ns=6 * HOUR_NS, max_devices=3), _keyed_by(_reboots)),
    "session_timer": (lambda: SessionTimer(timeout_ns=2 * HOUR_NS, max_pending=3), _session_timer_ops),
    "transfer_envelope": (lambda: TransferEnvelopeTracker(min_samples=3, max_entities=3), _keyed_by(_transfer)),
    "ttl_profile": (lambda: TtlProfileTracker(min_samples=3, max_entities=3), _keyed_by(_ttl)),
    "value_novelty": (lambda: ValueNoveltyTracker(["serial", "chassis"], max_values=3, max_entities=3),
                      _keyed_by(_identifiers)),
    "window_sealer": (lambda: WindowSealer(period_ns=HOUR_NS, lateness_ns=10 * MINUTE_NS), _window_sealer_ops),
}


def _run(tracker: typing.Any, ops: list[Op]) -> list:
    # Results are compared encoded, so a NaN equals a NaN and a numpy integer is not mistaken for an int.
    return [encode(op(tracker)) for op in ops]


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_restored_tracker_continues_as_if_it_had_never_stopped(name: str):
    (factory, make_ops) = CASES[name]
    ops = make_ops(random.Random(name))
    half = len(ops) // 2

    straight = factory()
    expected = _run(straight, ops)

    stopped = factory()
    _run(stopped, ops[:half])
    snapshot = to_bytes(stopped.to_state())

    resumed = type(stopped).from_state(from_bytes(snapshot))

    assert _run(resumed, ops[half:]) == expected[half:]
    assert to_bytes(resumed.to_state()) == to_bytes(straight.to_state())


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_same_history_gives_the_same_bytes(name: str):
    (factory, make_ops) = CASES[name]
    ops = make_ops(random.Random(name))

    (first, second) = (factory(), factory())
    _run(first, ops)
    _run(second, ops)

    assert to_bytes(first.to_state()) == to_bytes(first.to_state())
    assert to_bytes(first.to_state()) == to_bytes(second.to_state())
    assert state_digest(first.to_state()) == state_digest(second.to_state())
    # And the bytes are JSON, so a snapshot can be read by something other than this module.
    assert json.loads(to_bytes(first.to_state()))["class"].startswith("morpheus.")


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_snapshot_is_not_empty(name: str):
    # A tracker whose state encodes to nothing would pass the round trip trivially.
    (factory, make_ops) = CASES[name]
    tracker = factory()
    before = to_bytes(tracker.to_state())
    _run(tracker, make_ops(random.Random(name)))

    assert to_bytes(tracker.to_state()) != before


def _utils_classes() -> list[tuple[str, str, list[str]]]:
    """(module, class, base names) for every class defined under morpheus.utils, read from source.

    Read rather than imported, because some of those modules need a GPU to import and a tracker must not escape
    this check by living beside one.
    """
    found = []
    utils_dir = os.path.dirname(morpheus.utils.__file__)

    for file_name in sorted(os.listdir(utils_dir)):
        if (not file_name.endswith(".py")):
            continue

        with open(os.path.join(utils_dir, file_name), encoding="utf-8") as source:
            tree = ast.parse(source.read())

        for node in tree.body:
            if (isinstance(node, ast.ClassDef)):
                bases = [base.id for base in node.bases if isinstance(base, ast.Name)]
                found.append((f"morpheus.utils.{file_name[:-3]}", node.name, bases))

    return found


def _resolve(module: str, name: str) -> type:
    return getattr(importlib.import_module(module), name)


def test_every_tracker_has_a_case():
    covered = {type(factory()) for (factory, _) in CASES.values()}
    stateful = {_resolve(module, name) for (module, name, bases) in _utils_classes() if "StateMixin" in bases}

    assert stateful | {RebootCounter} == covered


def test_every_class_named_a_tracker_carries_state():
    trackers = [(module, name, bases) for (module, name, bases) in _utils_classes() if name.endswith("Tracker")]

    assert len(trackers) >= 17
    assert all("StateMixin" in bases
               for (_, _, bases) in trackers), [name for (_, name, bases) in trackers if "StateMixin" not in bases]


_DIGEST_SCRIPT = """
import random, sys
sys.path.insert(0, {test_dir!r})
import test_tracker_state as t
(factory, make_ops) = t.CASES["value_novelty"]
tracker = factory()
t._run(tracker, make_ops(random.Random("value_novelty")))
print(t.state_digest(tracker.to_state()))
"""


def test_the_bytes_do_not_depend_on_the_hash_seed():
    # Sets of strings iterate in an order that changes with PYTHONHASHSEED; the snapshot must not.
    script = _DIGEST_SCRIPT.format(test_dir=os.path.dirname(os.path.abspath(__file__)))
    digests = set()

    for seed in ("0", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run([sys.executable, "-c", script], env=env, check=True, capture_output=True, text=True)
        digests.add(result.stdout.strip())

    assert len(digests) == 1


@pytest.mark.parametrize("value",
                         [
                             None,
                             True,
                             3,
                             2**70,
                             -0.5,
                             float("inf"),
                             "text",
                             b"\x00\xff",
                             (1, "a"),
                             [1, (2, 3)],
                             {
                                 "b": 1, "a": 2
                             },
                             {
                                 3: "x", (1, 2): "y"
                             },
                             collections.OrderedDict([("z", 1), ("a", 2)]),
                             collections.deque([1, 2, 3], maxlen=5),
                             collections.Counter({"a": 2}),
                             {"x", "y"},
                             frozenset({1, 2}),
                             np.int64(7),
                             np.float32(1.5),
                             np.bool_(True),
                             pd.Timestamp("2026-10-17T01:02:03.000000004Z"),
                             pd.Timestamp(5, unit="ns"),
                             pd.Timedelta(9, unit="ns"),
                             pd.NaT,
                             np.array([[1, 2], [3, 4]], dtype=np.int32),
                         ],
                         ids=repr)
def test_values_keep_their_type(value: typing.Any):
    restored = decode(from_bytes(to_bytes(encode(value))))

    assert type(restored) is type(value)

    if (isinstance(value, np.ndarray)):
        assert restored.dtype == value.dtype and (restored == value).all()
    elif (value is not pd.NaT):
        assert restored == value

    if (isinstance(value, (dict, collections.deque))):
        assert list(restored) == list(value)

    if (isinstance(value, collections.deque)):
        assert restored.maxlen == value.maxlen


def test_nan_survives():
    restored = decode(from_bytes(to_bytes(encode([float("nan"), np.float64("nan")]))))

    assert np.isnan(restored[0]) and isinstance(restored[1], np.float64) and np.isnan(restored[1])


def test_a_frame_keeps_its_columns_dtypes_and_index():
    frame = pd.DataFrame({
        "event_time": pd.Series([3, 1], dtype="int64"),
        "src_ip": ["10.0.0.1", None],
        "bytes": [1.5, float("nan")],
        "seen": pd.Series([True, False], dtype="bool"),
        "count": pd.Series([1, None], dtype="Int64"),
        "at": pd.to_datetime([1, 2], unit="s"),
    }).iloc[[1, 0]]

    restored = decode(from_bytes(to_bytes(encode(frame))))

    pd.testing.assert_frame_equal(restored, frame)


def test_a_range_index_stays_a_range_index():
    frame = pd.DataFrame({"a": [1, 2, 3]})
    restored = decode(from_bytes(to_bytes(encode(frame))))

    assert isinstance(restored.index, pd.RangeIndex)
    pd.testing.assert_frame_equal(restored, frame)


def test_classes_outside_morpheus_are_refused_both_ways():
    with pytest.raises(StateError, match="only Morpheus classes"):
        encode(fractions.Fraction(1, 3))

    with pytest.raises(StateError, match="only module-level Morpheus functions"):
        encode(lambda: None)

    with pytest.raises(StateError, match="Refusing to rebuild"):
        decode({"$t": "obj", "cls": "os:PathLike", "attrs": []})

    with pytest.raises(StateError, match="Refusing to rebuild"):
        decode({"$t": "func", "name": "os:system"})


def test_a_named_morpheus_function_is_kept_by_name():
    tracker = ValueHistoryTracker(min_samples=1)
    restored = ValueHistoryTracker.from_state(from_bytes(to_bytes(tracker.to_state())))

    assert restored._reduction is tracker._reduction  # pylint: disable=protected-access


def test_a_snapshot_of_another_class_or_format_is_refused():
    state = BucketPeakTracker().to_state()

    with pytest.raises(StateError, match="not 'morpheus.utils.distinct_window"):
        DistinctWindowTracker.from_state(state)

    with pytest.raises(StateError, match="format"):
        BucketPeakTracker.from_state(dict(state, format=99))
