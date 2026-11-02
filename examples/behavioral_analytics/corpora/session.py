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
The synthetic corpus `morpheus.compositions.session` is tested over, and the facts about it the harnesses assert.

Moved out of `tests/morpheus/determinism/session_pipeline.py` under plan section 4.5, which put the composition in the
package and left the data here, where the tests, the sample and fixture generators and the experiments read it. The
composition module's docstring describes what this corpus plants and why.
"""

import random
import typing

import pandas as pd

from morpheus.compositions import envelope as universal_envelope
from morpheus.compositions.session import ALICE
from morpheus.compositions.session import BATCH
from morpheus.compositions.session import BOB
from morpheus.compositions.session import CAROL
from morpheus.compositions.session import DAVE
from morpheus.compositions.session import ERIN
from morpheus.compositions.session import FLIGHT_DAY
from morpheus.compositions.session import GRACE
from morpheus.compositions.session import JOINER_DAY
from morpheus.compositions.session import TRAINING_DAYS
from morpheus.compositions.session import at
from morpheus.utils.binding_table import NS_PER_SECOND

CORPUS_SEED = 20260906
CORPUS_DAYS = 7
TELEMETRY_CLASSES = ("tc5_auth", "tc5_session")
LONDON = (51.5074, -0.1278)
NEW_YORK = (40.7128, -74.0060)
FRANKFURT = (50.1109, 8.6821)
DUBLIN = (53.3498, -6.2603)
AMSTERDAM = (52.3676, 4.9041)
LONDON_PLACE = ("gb", "england", "london")
NEW_YORK_PLACE = ("us", "ny", "new-york")
FRANKFURT_PLACE = ("de", "hesse", "frankfurt")
DUBLIN_PLACE = ("ie", "leinster", "dublin")
AMSTERDAM_PLACE = ("nl", "north-holland", "amsterdam")
VPN_EGRESS_IP = "198.51.100.7"
"""The concentrator's egress range, standing in for the list an estate supplies. Without it R-D-L5-003 fires on
every VPN user every day, which is the same shape as R-D-L2-003's exclusion list and the same reason: this
repository cannot know an estate's ranges."""
OFFICE_HOURS = (9, 11, 14, 16)
"""The hours the office workers authenticate at. Four a day, five days a week, which is what gives the histogram
enough of a shape for an outlier to be an outlier."""
WEEKDAYS = (0, 1, 2, 3, 4)
BATCH_HOUR = 3
"""The service account's nightly hour. The same hour as the off-hours anomaly, on purpose."""
OFF_HOURS_DAY = 3
OFF_HOURS_HOUR = 3
"""When the office worker authenticates at an hour she has never used. The service account authenticates at the
same hour on the same night, and the two rows must not read alike."""
IMPOSSIBLE_DAY = 4
IMPOSSIBLE_HOUR = 14
IMPOSSIBLE_GAP_S = 1800
"""London to New York in half an hour, measured from the principal's own ordinary office login at that hour. The
hour is one she already uses, so the journey is the only thing anomalous about the row -- a fresh hour would put
a cadence signal on it too and neither assertion would isolate anything."""
REFRESH_OFFSET_S = 900
"""A token refresh issued from London between the two authentications. It carries the original location, so it is
evidence of nothing about where the principal is, and an anchor that moved to it would erase the journey."""
FLIGHT_DEPART_HOUR = 8
FLIGHT_HOURS = 8
"""The legitimate crossing. Eight hours is what the aircraft takes, and R-D-L5-003 must stay quiet about it."""
FATIGUE_DAY = 5
FATIGUE_HOUR = 2
FATIGUE_DENIALS = 5
FATIGUE_INTERVAL_S = 90
"""Five denials at ninety-second intervals and then an approval: eight minutes, inside the rule's ten."""
TAKEOVER_DAY = 2
TAKEOVER_HOURS = (14, 15, 16, 17)
TAKEOVER_APPS = ("payroll", "admin-console", "file-share")
TAKEOVER_DEVICE = "unmanaged-7f3a"
TAKEOVER_IP = "192.0.2.41"
"""Erin's afternoon on the Wednesday of the scored week: three sign-ins an hour from Amsterdam, on a device and
to applications her account has never used, without the second factor, and two of them failing. She signed in
from London that morning, and four hours is enough to get to Amsterdam, which is why the deterministic travel
rule must stay quiet and the per-principal models are the only thing that can say this afternoon is unlike her
fortnight. Two failures rather than three, so the failure-run rule stays quiet too and the case is the models'
alone."""
FUMBLE_DAY = 2
FUMBLE_HOUR = 14
"""Two failed passwords and then a success. Failure-then-success with none of the denial volume R-D-L5-004
requires, which is what most of an estate does on a Monday morning."""
DEVICES = {
    ALICE: "laptop-alice",
    BOB: "laptop-bob",
    CAROL: "laptop-carol",
    DAVE: "laptop-dave",
    BATCH: "runner-1",
    ERIN: "laptop-erin",
    GRACE: "laptop-grace",
}
ASNS = {
    LONDON_PLACE: "as5089",
    NEW_YORK_PLACE: "as7018",
    FRANKFURT_PLACE: "as3320",
    DUBLIN_PLACE: "as2856",
    AMSTERDAM_PLACE: "as1136",
}
HOME_IPS = {
    ALICE: "203.0.113.11",
    BOB: "203.0.113.12",
    CAROL: "203.0.113.13",
    DAVE: "203.0.113.14",
    BATCH: "203.0.113.15",
    ERIN: "203.0.113.16",
    GRACE: "203.0.113.17",
}
ABANDONED_SESSION = "sess-abandoned"
ABANDONED_START_DAY = 1
ABANDONED_STOP_DAY = 6
"""A session whose stop arrives five days after its start. Long past the timeout, so the start was abandoned and
the stop reads unpaired -- which is the point: a session that never closes is a lost stop record rather than a
five-day session, and reporting one would put an absurd duration on an ordinary working day."""
UNPAIRED_SESSION = "sess-unpaired"
DUPLICATED_SESSION = "sess-duplicated"
LONG_SESSION_DAY = 4
LONG_SESSION_END_HOUR = 20
"""Carol's last working day runs from nine to eight: eleven hours, against four days of eight."""
BATCH_SESSION_MINUTES = (10, 240)
BATCH_LAST_SESSION_MINUTES = 300
"""The service account's nightly sessions alternate between a short check and a long run, and the last night's is
the longest it has had. Bimodal, which is why R-B-L5-005 excludes service accounts rather than hoping a percentile
of two shapes means anything."""
SINGLE_RECORD_SESSION = "sess-dave-sso"
SINGLE_RECORD_DAY = 4
SINGLE_RECORD_HOURS = (13, 16)
INVERTED_SESSION = "sess-dave-sso-skew"
INVERTED_DAY = 5
INVERTED_HOURS = (17, 16)
"""Two sessions from an identity provider that emits one record per session with both ends in it. The first is
three hours; the second ends an hour before it starts, which is a defect in the record and is reported as one."""


def _envelope(rng: random.Random, collector: str, schema: str, seq: int) -> dict:
    return {
        "collector_id": collector,
        "schema_version": schema,
        "origin_hash": f"{rng.getrandbits(64):016x}",
        "collector_seq": seq,
    }


def build_corpus() -> dict[str, pd.DataFrame]:
    """
    Build the fixed corpus, one frame per telemetry class.

    Every value derives from `CORPUS_SEED`, so the corpus is as fixed as a checked-in file while remaining
    reviewable as code. Rows are generated in event order with a monotonic `collector_seq`, which is the
    envelope's own requirement; the harness's permutation check is what scrambles them.
    """
    rng = random.Random(CORPUS_SEED)

    return universal_envelope.attach_all({"tc5_auth": _build_auth(rng), "tc5_session": _build_sessions(rng)})


def _auth_row(principal: str,
              time_s: int,
              place: tuple,
              coordinate: tuple,
              app: str = "wiki",
              result: str = "success",
              mfa_used: bool = True,
              mfa_result: typing.Optional[str] = "approved",
              token_type: str = "bearer",
              source_ip: typing.Optional[str] = None,
              device_id: typing.Optional[str] = None) -> dict:
    (country, region, city) = place

    return {
        "event_time": time_s * NS_PER_SECOND,
        "user_principal": principal,
        "source_country": country,
        "source_region": region,
        "source_city": city,
        "source_latitude": coordinate[0],
        "source_longitude": coordinate[1],
        "source_asn": ASNS[place],
        "source_ip": HOME_IPS[principal] if source_ip is None else source_ip,
        "app": app,
        "device_id": DEVICES[principal] if device_id is None else device_id,
        "auth_result": result,
        "mfa_used": mfa_used,
        "mfa_result": mfa_result,
        "token_type": token_type,
    }


def _ordinary_day(day: int) -> list[dict]:
    """What everybody with a model does on a day nothing happens: the habits the fortnight trains on."""
    events = []

    if (day % 7 in WEEKDAYS):
        for hour in OFFICE_HOURS:
            events.append(_auth_row(ALICE, at(day, hour), LONDON_PLACE, LONDON))
            events.append(_auth_row(CAROL, at(day, hour, 10), LONDON_PLACE, LONDON))
            events.append(_auth_row(BOB, at(day, hour, 20), LONDON_PLACE, LONDON))
            events.append(_auth_row(ERIN, at(day, hour, 30), LONDON_PLACE, LONDON))

    events.append(
        _auth_row(BATCH, at(day, BATCH_HOUR), DUBLIN_PLACE, DUBLIN, app="batch", mfa_used=False, mfa_result=None))

    for hour in (10, 15):
        events.append(_auth_row(DAVE, at(day, hour), LONDON_PLACE, LONDON))
        events.append(_auth_row(DAVE, at(day, hour, 600), FRANKFURT_PLACE, FRANKFURT, source_ip=VPN_EGRESS_IP))

    return events


def _build_auth(rng: random.Random) -> pd.DataFrame:
    """Identity provider sign-in events for the training fortnight and the scored week, with every planted case."""
    events: list[dict] = []

    # The fortnight the models are trained on: ordinary habits and nothing else.
    for day in range(-TRAINING_DAYS, 0):
        events.extend(_ordinary_day(day))

    # The office workers' ordinary week. Carol and Bob keep the same hours as Alice, which is what makes each
    # principal's own histogram the thing an outlier is measured against rather than a population's.
    for day in range(CORPUS_DAYS):
        if (day not in WEEKDAYS):
            continue

        for hour in OFFICE_HOURS:
            events.append(_auth_row(ALICE, at(day, hour), LONDON_PLACE, LONDON))
            events.append(_auth_row(CAROL, at(day, hour, 10), LONDON_PLACE, LONDON))

            # Bob crosses the Atlantic mid-week and works from New York afterwards.
            if (day < FLIGHT_DAY):
                events.append(_auth_row(BOB, at(day, hour, 20), LONDON_PLACE, LONDON))
            elif (day > FLIGHT_DAY):
                events.append(_auth_row(BOB, at(day, hour, 20), NEW_YORK_PLACE, NEW_YORK))

            # Erin keeps her hours, except for the afternoon somebody else has her account.
            if (day != TAKEOVER_DAY or hour < TAKEOVER_HOURS[0]):
                events.append(_auth_row(ERIN, at(day, hour, 30), LONDON_PLACE, LONDON))

            # Grace, from her first day, keeps the office's hours too.
            if (day >= JOINER_DAY):
                events.append(_auth_row(GRACE, at(day, hour, 40), LONDON_PLACE, LONDON))

    # The afternoon Erin's account is used by somebody else.
    for hour in TAKEOVER_HOURS:
        for (index, app) in enumerate(TAKEOVER_APPS):
            failed = hour == TAKEOVER_HOURS[0] and index < 2
            events.append(
                _auth_row(ERIN,
                          at(TAKEOVER_DAY, hour, 300 + index * 900),
                          AMSTERDAM_PLACE,
                          AMSTERDAM,
                          app=app,
                          result="failure" if failed else "success",
                          mfa_used=False,
                          mfa_result=None,
                          source_ip=TAKEOVER_IP,
                          device_id=TAKEOVER_DEVICE))

    # The service account, every night at the same hour, from the datacentre.
    for day in range(CORPUS_DAYS):
        events.append(
            _auth_row(BATCH, at(day, BATCH_HOUR), DUBLIN_PLACE, DUBLIN, app="batch", mfa_used=False, mfa_result=None))

    # The VPN user, alternating between home and the concentrator several times a day, every day. Apparent
    # journeys of six hundred kilometres in minutes, which the exclusion list is what stops from firing.
    for day in range(CORPUS_DAYS):
        for hour in (10, 15):
            events.append(_auth_row(DAVE, at(day, hour), LONDON_PLACE, LONDON))
            events.append(_auth_row(DAVE, at(day, hour, 600), FRANKFURT_PLACE, FRANKFURT, source_ip=VPN_EGRESS_IP))

    # The legitimate crossing: eight hours, which is what the aircraft takes.
    events.append(_auth_row(BOB, at(FLIGHT_DAY, FLIGHT_DEPART_HOUR), LONDON_PLACE, LONDON))
    events.append(_auth_row(BOB, at(FLIGHT_DAY, FLIGHT_DEPART_HOUR + FLIGHT_HOURS), NEW_YORK_PLACE, NEW_YORK))

    # The impossible one. Its origin is the principal's own ordinary office login at this hour rather than a
    # planted second row at the same instant: two authentications on one timestamp are a duplicate, not a
    # journey, and the cumulative counters would refuse the later of them as out of order.
    events.append(
        _auth_row(ALICE,
                  at(IMPOSSIBLE_DAY, IMPOSSIBLE_HOUR, REFRESH_OFFSET_S),
                  LONDON_PLACE,
                  LONDON,
                  token_type="refresh"))
    events.append(_auth_row(ALICE, at(IMPOSSIBLE_DAY, IMPOSSIBLE_HOUR, IMPOSSIBLE_GAP_S), NEW_YORK_PLACE, NEW_YORK))

    # The off-hours authentication, and nothing else about it out of the ordinary.
    events.append(_auth_row(ALICE, at(OFF_HOURS_DAY, OFF_HOURS_HOUR), LONDON_PLACE, LONDON))

    # The fumbled password: two failures and a success, with the factor never challenged.
    for attempt in range(2):
        events.append(
            _auth_row(ALICE,
                      at(FUMBLE_DAY, FUMBLE_HOUR, 30 + attempt * 30),
                      LONDON_PLACE,
                      LONDON,
                      result="failure",
                      mfa_used=False,
                      mfa_result=None))

    events.append(
        _auth_row(ALICE, at(FUMBLE_DAY, FUMBLE_HOUR, 90), LONDON_PLACE, LONDON, mfa_used=False, mfa_result=None))

    # The fatigue burst: five denials and then the approval that makes it actionable.
    for challenge in range(FATIGUE_DENIALS):
        events.append(
            _auth_row(CAROL,
                      at(FATIGUE_DAY, FATIGUE_HOUR, challenge * FATIGUE_INTERVAL_S),
                      LONDON_PLACE,
                      LONDON,
                      result="failure",
                      mfa_result="denied"))

    events.append(
        _auth_row(CAROL, at(FATIGUE_DAY, FATIGUE_HOUR, FATIGUE_DENIALS * FATIGUE_INTERVAL_S), LONDON_PLACE, LONDON))

    events.sort(key=lambda event: (event["event_time"], event["user_principal"]))
    rows = []

    for (seq, event) in enumerate(events, start=1):
        rows.append({**event, **_envelope(rng, "idp", "TC-5/1.0.0", seq)})

    return pd.DataFrame(rows)


def _build_sessions(rng: random.Random) -> pd.DataFrame:
    """Session lifecycle records: starts and their separate stops, plus the three shapes that go wrong."""
    events: list[tuple] = []

    # An ordinary working session per office worker per weekday: one start in the morning, one stop in the
    # evening, emitted as two records with nothing in either saying how long it ran. Carol's last one runs late.
    for day in WEEKDAYS:
        for (index, principal) in enumerate((ALICE, BOB, CAROL)):
            session_id = f"sess-{principal.split('@')[0]}-{day}"
            long_day = principal == CAROL and day == LONG_SESSION_DAY
            events.append((at(day, 9, index), session_id, principal, "start"))
            events.append((at(day, LONG_SESSION_END_HOUR if long_day else 17, index), session_id, principal, "end"))

    # The service account's nightly sessions, short and long by turns, and the longest on the last night.
    for day in range(CORPUS_DAYS):
        minutes = BATCH_LAST_SESSION_MINUTES if day == CORPUS_DAYS - 1 else BATCH_SESSION_MINUTES[day % 2]
        session_id = f"sess-batch-{day}"
        events.append((at(day, BATCH_HOUR), session_id, BATCH, "start"))
        events.append((at(day, BATCH_HOUR, minutes * 60), session_id, BATCH, "end"))
    # A session abandoned past the timeout, whose stop then arrives days later. It must read as unpaired rather
    # than as a five-day session.
    events.append((at(ABANDONED_START_DAY, 10), ABANDONED_SESSION, DAVE, "start"))
    events.append((at(ABANDONED_STOP_DAY, 10), ABANDONED_SESSION, DAVE, "end"))

    # A stop with no start, which is what the beginning of any stream looks like and what a collector restart
    # produces in the middle of one.
    events.append((at(2, 12), UNPAIRED_SESSION, DAVE, "end"))

    # One identifier collecting two starts. Not a retry: a session identifier is meant to be unique to a session,
    # so this is a duplicating collector or an identifier being reused. The stop that follows is timed from the
    # second start, and `session_starts` says there were two.
    events.append((at(3, 9), DUPLICATED_SESSION, DAVE, "start"))
    events.append((at(3, 9, 30), DUPLICATED_SESSION, DAVE, "start"))
    events.append((at(3, 11), DUPLICATED_SESSION, DAVE, "end"))

    events.sort(key=lambda event: (event[0], event[1]))

    rows = []

    for (seq, (time_s, session_id, principal, action)) in enumerate(events, start=1):
        rows.append({
            "event_time": time_s * NS_PER_SECOND,
            "session_id": session_id,
            "user_principal": principal,
            "session_action": action,
            "session_start": None,
            "session_end": None,
            **_envelope(rng, "radius-accounting", "TC-5/1.0.0", seq),
        })

    # The identity provider's own session records, one per session with both ends in it, from a collector of its
    # own. Each is emitted when the session ends, which is when such a record exists.
    single = [(SINGLE_RECORD_SESSION, SINGLE_RECORD_DAY, SINGLE_RECORD_HOURS),
              (INVERTED_SESSION, INVERTED_DAY, INVERTED_HOURS)]

    for (seq, (session_id, day, (start_hour, end_hour))) in enumerate(single, start=1):
        rows.append({
            "event_time": at(day, max(start_hour, end_hour)) * NS_PER_SECOND,
            "session_id": session_id,
            "user_principal": DAVE,
            "session_action": "session",
            "session_start": at(day, start_hour) * NS_PER_SECOND,
            "session_end": at(day, end_hour) * NS_PER_SECOND,
            **_envelope(rng, "idp-sessions", "TC-5/1.0.0", seq),
        })

    frame = pd.DataFrame(rows).sort_values(["event_time", "collector_id", "collector_seq"], kind="mergesort")
    frame["session_start"] = frame["session_start"].astype("Int64")
    frame["session_end"] = frame["session_end"].astype("Int64")

    return frame.reset_index(drop=True)
