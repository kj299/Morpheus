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
The week-long layer 5 corpus, and the composed session pipeline, for the determinism harness.

The layer 1 and 2 harness proved that fourteen stages in a row reach the answer a golden file holds. The five
TC-5 stages had never been run composed with each other, and the two deterministic layer 5 rules had no corpus to
be asserted against -- which is why neither shipped as a saved search when its features landed. This module is
both.

**A week, not an hour.** The layer 1 and 2 corpus covers an hour because a poller's cadence is a minute. Layer 5
is event-driven and its features are about habit: an hour-of-day histogram needs days to say anything, an
impossible journey needs hours to be impossible, and a cumulative location count needs somewhere to have been
before. The corpus therefore runs seven days from a Monday midnight, and the window period is an hour rather than
five minutes.

Into it are planted the things the layer 5 features exist to see, each with the thing they must **not** fire on
beside it:

- an **impossible journey**: one principal in London and, half an hour later, in New York;
- a **legitimate flight**: another principal making the same crossing in eight hours, which is what an aircraft
  does and what R-D-L5-003 must stay quiet about;
- a **token refresh** issued from the origin between those two authentications, which carries the original
  location and would erase the journey if it were allowed to become the anchor;
- a **VPN user** whose apparent location alternates between home and a concentrator in another country several
  times a day, carried on the egress exclusion list, so the rule's exclusion path is exercised rather than
  assumed;
- a **multi-factor fatigue burst**: five denials inside eight minutes and then an approval;
- a **fumbled password**: two failures and a success, which is failure-then-success without any of the denial
  volume R-D-L5-004 requires, and is what most of an estate does on a Monday;
- an **off-hours authentication** at 03:00 for a principal who has only ever worked office hours;
- a **service account** that authenticates at 03:00 every night, for which the same hour is unremarkable -- the
  negative control that makes the cadence feature a statement about a principal rather than about a clock;
- a **session that never ends**, whose stop record was lost, and a **stop with no start**, which is what the
  beginning of any stream looks like;
- and a **session identifier used twice**, which is a duplicating collector rather than a retry.

The pipeline is one per telemetry class, which is the deployment shape: authentication events arrive from the
identity provider, session lifecycle records from the concentrators and the RADIUS accounting stream, and the two
have different required columns. Every stateful stage is preceded by `TotalOrderStage`, which is determinism
control 8 as a stage.
"""

import random
import typing

import pandas as pd

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc5_cadence_stage import TC5CadenceStage
from morpheus.stages.telemetry.tc5_drift_stage import TC5DriftStage
from morpheus.stages.telemetry.tc5_novelty_stage import TC5NoveltyStage
from morpheus.stages.telemetry.tc5_risk_stage import TC5RiskStage
from morpheus.stages.telemetry.tc5_score_stage import Scorer
from morpheus.stages.telemetry.tc5_score_stage import TC5ScoreStage
from morpheus.utils.model_manifest import ModelManifest
from morpheus.stages.telemetry.tc5_session_stage import TC5SessionStage
from morpheus.stages.telemetry.tc5_travel_stage import TC5TravelStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize

CORPUS_SEED = 20260906

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900
CORPUS_DAYS = 7

HOUR_S = 3600
DAY_S = 24 * HOUR_S

CORPUS_EPOCH_S = 4 * DAY_S
"""The corpus starts on a Monday midnight UTC.

1970-01-01 was a Thursday, so four days on is the first Monday. Anchoring there rather than at the epoch itself
means day zero of the corpus is a weekday and the weekend falls where a reader expects it, without the corpus
depending on any real date -- and therefore without a golden file that ages.
"""

ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []

TELEMETRY_CLASSES = ("tc5_auth", "tc5_session")

CADENCE_MIN_SAMPLES = 6
"""Prior observations before a principal's histogram is called mature, for this corpus.

Below the stage's own default of 32, and deliberately: a week of one estate's logins is not a production history,
and a harness whose maturity flag is false on every row would assert nothing about it. Six rather than eight so
that the service account, which authenticates once a night, is mature by the last night of the corpus -- without
that it could never serve as the negative control it exists to be. The number is here rather than in the stage
because it is a property of the corpus.
"""

# --- Places -------------------------------------------------------------------------------------------------

LONDON = (51.5074, -0.1278)
NEW_YORK = (40.7128, -74.0060)
FRANKFURT = (50.1109, 8.6821)
DUBLIN = (53.3498, -6.2603)

LONDON_PLACE = ("gb", "england", "london")
NEW_YORK_PLACE = ("us", "ny", "new-york")
FRANKFURT_PLACE = ("de", "hesse", "frankfurt")
DUBLIN_PLACE = ("ie", "leinster", "dublin")

VPN_EGRESS_NETWORK = "198.51.100.0/24"
VPN_EGRESS_IP = "198.51.100.7"
"""The concentrator's egress range, standing in for the list an estate supplies. Without it R-D-L5-003 fires on
every VPN user every day, which is the same shape as R-D-L2-003's exclusion list and the same reason: this
repository cannot know an estate's ranges."""

# --- Principals ---------------------------------------------------------------------------------------------

ALICE = "alice@example.com"
BOB = "bob@example.com"
CAROL = "carol@example.com"
DAVE = "dave@example.com"
BATCH = "svc-batch@example.com"

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

FLIGHT_DAY = 3
FLIGHT_DEPART_HOUR = 8
FLIGHT_HOURS = 8
"""The legitimate crossing. Eight hours is what the aircraft takes, and R-D-L5-003 must stay quiet about it."""

FATIGUE_DAY = 5
FATIGUE_HOUR = 2
FATIGUE_DENIALS = 5
FATIGUE_INTERVAL_S = 90
"""Five denials at ninety-second intervals and then an approval: eight minutes, inside the rule's ten."""

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
}

ASNS = {
    LONDON_PLACE: "as5089",
    NEW_YORK_PLACE: "as7018",
    FRANKFURT_PLACE: "as3320",
    DUBLIN_PLACE: "as2856",
}

HOME_IPS = {
    ALICE: "203.0.113.11",
    BOB: "203.0.113.12",
    CAROL: "203.0.113.13",
    DAVE: "203.0.113.14",
    BATCH: "203.0.113.15",
}

# --- Sessions -----------------------------------------------------------------------------------------------

ABANDONED_SESSION = "sess-abandoned"
ABANDONED_START_DAY = 1
ABANDONED_STOP_DAY = 6
"""A session whose stop arrives five days after its start. Long past the timeout, so the start was abandoned and
the stop reads unpaired -- which is the point: a session that never closes is a lost stop record rather than a
five-day session, and reporting one would put an absurd duration on an ordinary working day."""
UNPAIRED_SESSION = "sess-unpaired"
DUPLICATED_SESSION = "sess-duplicated"
SESSION_TIMEOUT_SECONDS = 12 * HOUR_S
"""Shorter than the stage's own default, because this corpus is a week and a session abandoned after a day and a
half would never be abandoned inside it."""


def _envelope(rng: random.Random, collector: str, schema: str, seq: int) -> dict:
    return {
        "collector_id": collector,
        "schema_version": schema,
        "origin_hash": f"{rng.getrandbits(64):016x}",
        "collector_seq": seq,
    }


def at(day: int, hour: int, second: int = 0) -> int:
    """Epoch seconds for an hour on a corpus day, counted from the corpus's Monday midnight."""
    return CORPUS_EPOCH_S + day * DAY_S + hour * HOUR_S + second


def build_corpus() -> dict[str, pd.DataFrame]:
    """
    Build the fixed corpus, one frame per telemetry class.

    Every value derives from `CORPUS_SEED`, so the corpus is as fixed as a checked-in file while remaining
    reviewable as code. Rows are generated in event order with a monotonic `collector_seq`, which is the
    envelope's own requirement; the harness's permutation check is what scrambles them.
    """
    rng = random.Random(CORPUS_SEED)

    return {"tc5_auth": _build_auth(rng), "tc5_session": _build_sessions(rng)}


def _auth_row(principal: str,
              time_s: int,
              place: tuple,
              coordinate: tuple,
              app: str = "wiki",
              result: str = "success",
              mfa_used: bool = True,
              mfa_result: typing.Optional[str] = "approved",
              token_type: str = "bearer",
              source_ip: typing.Optional[str] = None) -> dict:
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
        "device_id": DEVICES[principal],
        "auth_result": result,
        "mfa_used": mfa_used,
        "mfa_result": mfa_result,
        "token_type": token_type,
    }


def _build_auth(rng: random.Random) -> pd.DataFrame:
    """Identity provider sign-in events for a week, with every planted case in place."""
    events: list[dict] = []

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
    # evening, emitted as two records with nothing in either saying how long it ran.
    for day in WEEKDAYS:
        for (index, principal) in enumerate((ALICE, BOB, CAROL)):
            session_id = f"sess-{principal.split('@')[0]}-{day}"
            events.append((at(day, 9, index), session_id, principal, "start"))
            events.append((at(day, 17, index), session_id, principal, "end"))

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
            **_envelope(rng, "radius-accounting", "TC-5/1.0.0", seq),
        })

    return pd.DataFrame(rows)


def build_pipeline_config(execution_mode=None) -> Config:
    """
    A pipeline configuration, defaulting to CPU mode and importable without a GPU.

    Parameters
    ----------
    execution_mode : `morpheus.config.ExecutionMode`, optional
        Mode to build for. Defaults to CPU, which is what the golden file is an artifact of. The parameter exists
        so the same corpus and the same golden can be driven in GPU mode and the two compared. Resolved inside the
        function rather than as a default argument so that importing this module still does not require a GPU.
    """
    from morpheus.config import CppConfig
    from morpheus.config import ExecutionMode

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.execution_mode = ExecutionMode.CPU if execution_mode is None else execution_mode

    return config


def _collect(sink: InMemorySinkStage) -> pd.DataFrame:
    # Sibling module; imported here because this file's own directory is put on the path by whoever imports it.
    from host_frame import to_host_frame

    frames = []

    for message in sink.get_messages():
        meta = message.payload() if isinstance(message, ControlMessage) else message
        frames.append(to_host_frame(meta.copy_dataframe()))

    if (len(frames) == 0):
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


CHAIN_ANCHORS = {
    "tc5_auth": "user_principal",
    "tc5_session": "session_key",
}
"""What each class's correlation chain is rooted on.

An authentication is about the principal making it. A session record is about the session, whose key is the
principal and the identifier composed together -- so a chain of session records is one principal's one session,
which is a narrower and more useful thing to follow than every session that principal opened in the window.
"""

CLASS_ENVELOPE = {
    "tc5_auth": (5, ["user_principal"]),
    "tc5_session": (5, ["user_principal"]),
}
"""The layer and the behavioral subject Part 2 names for each layer 5 class.

The principal rather than the session: a session is an episode belonging to a principal, and grouping a summary
by the episode would give one row per logon rather than one per person, which is the opposite of what a
behavioral rollup is for. `session_id` stays on the record as its own column for the rules that join on it.
"""

DRIFT_PREFIX = "day_"
"""Prefix on the daily sealer's columns, so the hourly windows keep theirs."""


def _run_class(config: Config,
               dataframes: list[pd.DataFrame],
               stages: list,
               impose_order: bool,
               anchor: str,
               envelope: tuple = None,
               daily: list = None) -> pd.DataFrame:
    """Source → stamp → (total order) → the class's stages → (envelope) → window seal → (daily seal → daily
    stages) → sink, as one frame.

    The hourly seal is the one the chains and the shipped detections are built on. A class given `daily` is
    sealed a second time, into days, behind it -- with the daily columns prefixed so the hourly ones keep their
    identity -- and the daily stages run on those complete days. That is where a trajectory across days is
    measured: R-P-L5-006 asks about four consecutive daily windows, and a day is not something an hourly window
    knows about.
    """
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
    pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

    if (impose_order):
        pipe.add_stage(TotalOrderStage(config))

    for stage in stages:
        pipe.add_stage(stage)

    if (envelope is not None):
        (osi_layer, entity_columns) = envelope
        pipe.add_stage(EnvelopeStampStage(config, osi_layer=osi_layer, entity_columns=entity_columns))

    pipe.add_stage(
        WindowSealStage(config,
                        period_seconds=PERIOD_SECONDS,
                        lateness_seconds=LATENESS_SECONDS,
                        order_columns=list(DEFAULT_ORDER_COLUMNS),
                        entity_key_column=anchor))

    if (daily is not None):
        pipe.add_stage(
            WindowSealStage(config,
                            period_seconds=DAY_S,
                            lateness_seconds=LATENESS_SECONDS,
                            order_columns=list(DEFAULT_ORDER_COLUMNS),
                            column_prefix=DRIFT_PREFIX))

        for stage in daily:
            pipe.add_stage(stage)

    sink = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    return _collect(sink)


SCORED_FEATURES = [
    "logcount",
    "locincrement",
    "appincrement",
    "deviceincrement",
    "asns_in_window",
    "hour_surprise_bits",
    "weekday_surprise_bits",
    "mfa_ratio",
    "auth_attempts_in_window",
    "auth_failures_in_window",
]
"""The ten TC-5 derived features, the same set `examples/layer5_model/run_model.py` trains on."""

SCORING_WINDOW = 0
SCORING_MANIFEST = ModelManifest(window_id=SCORING_WINDOW, models={}, fallback="reference-arithmetic:0")
"""Every principal resolves to the same placeholder, and `model_fallback_used` is true on every scored row.

That is the honest resolution for a corpus with no trained models in it. An event carrying a fallback is a claim
about a population rather than about the entity's own history, which is exactly what these scores are.
"""

REFERENCE_PARAMETERS = {
    "logcount": (4.038095, 1.72888),
    "locincrement": (1.32381, 0.467928),
    "appincrement": (1.0, 0.0),
    "deviceincrement": (1.0, 0.0),
    "asns_in_window": (1.285714, 0.451754),
    "hour_surprise_bits": (3.349456, 0.878393),
    "weekday_surprise_bits": (2.90969, 0.871514),
    "mfa_ratio": (0.905805, 0.260939),
    "auth_attempts_in_window": (1.2, 0.785584),
    "auth_failures_in_window": (0.238095, 0.889342),
}
"""Per-feature mean and standard deviation, computed once over the whole corpus and frozen here.

Frozen rather than computed, and the reason is the first thing control 5 caught. The first version of the
scorer below derived these from the rows it was handed, which made every score a function of how the stream
happened to be batched: the same authentication scored differently in a batch of ten and a batch of a hundred,
and `test_batch_split_sweep` failed on the first run. A trained model does not have this problem, because its
parameters are learned once and fixed before any scoring happens. Freezing them here is the stand-in for that,
and it is also what keeps the stub from fitting on the data it is scoring -- a leak no determinism control would
catch, because every run would leak identically.

A deviation of zero means the feature never varies in this corpus; those score zero rather than dividing by it.
"""


class ReferenceScorer:
    """**Not a model.** Arithmetic that stands in for one so the scoring path can be tested.

    `TC5ScoreStage` takes a scorer rather than training one, which is what lets the path be exercised where
    there is no Torch and no card. Something has to occupy that slot in the composed pipeline, and the choice is
    between a stub and leaving control 13's six checks unable to reach the stage at all.

    It returns each feature's distance from a frozen mean in units of a frozen deviation -- a z-score in the
    literal sense and nothing more. Deterministic, independent of batching, independent of the order rows arrive
    in, with no learned parameters, no history and no notion of normal beyond the constants above.

    **No detection claim attaches to any number it produces.** The scores in the golden file are arithmetic, not
    evidence. A threshold tuned against them would be tuned against this docstring. What the golden proves is
    that the path from features to scores is deterministic, batch-invariant and permutation-stable, which is a
    statement about plumbing and a precondition for a real model rather than a substitute for one.
    """

    def score(self, model_version: str, features: list) -> list:
        del model_version
        scores = []

        for row in features:
            scored = {}

            for (name, value) in row.items():
                (mean, deviation) = REFERENCE_PARAMETERS[name]
                scored[name] = 0.0 if deviation == 0 else (float(value) - mean) / deviation

            scores.append(scored)

        return scores


def run_pipeline(config: Config,
                 corpus: dict[str, pd.DataFrame],
                 batches: typing.Optional[dict[str, list[pd.DataFrame]]] = None,
                 impose_order: bool = True,
                 scorer: typing.Optional[Scorer] = None,
                 manifest: typing.Optional[ModelManifest] = None) -> pd.DataFrame:
    """
    Run every layer 5 telemetry class through its pipeline and return one canonicalized frame.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration.
    corpus : dict
        The frames from `build_corpus`, possibly permuted.
    batches : dict, optional
        Per class, how the corpus is split across source frames. Defaults to one frame per class. The batch-split
        sweep is the caller's to vary.
    impose_order : bool, default = True
        Place `TotalOrderStage` ahead of the stateful stages. The permutation check's negative control turns it
        off, and every stage here is cumulative, so the difference is visible.
    scorer : `morpheus.stages.telemetry.tc5_score_stage.Scorer`, optional
        What answers for the scores. Defaults to `ReferenceScorer`, the frozen arithmetic the golden file is
        built on. `examples/layer5_model/run_model.py` passes a `DfencoderScorer` holding the models it trained,
        which is how the composed pipeline is run with the autoencoder the guide names -- on the machine that
        can run one.
    manifest : `morpheus.utils.model_manifest.ModelManifest`, optional
        What pins each principal to a version. Defaults to `SCORING_MANIFEST`, which resolves everyone to the
        reference placeholder. Passed together with `scorer`, or the versions one resolves are ones the other
        does not hold.

    Returns
    -------
    `pandas.DataFrame`
        Both classes' output, tagged with `telemetry_class`, keyed by `row_key`, canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    if ((scorer is None) != (manifest is None)):
        raise ValueError("scorer and manifest are passed together or not at all; a manifest resolving versions "
                         "the scorer does not hold is the mismatch this pairing exists to prevent")

    scorer = ReferenceScorer() if scorer is None else scorer
    manifest = SCORING_MANIFEST if manifest is None else manifest

    outputs = {}

    outputs["tc5_auth"] = _run_class(
        config,
        batches["tc5_auth"],
        [
            TC5NoveltyStage(config),
            TC5CadenceStage(config, min_samples=CADENCE_MIN_SAMPLES),
            TC5TravelStage(config, excluded_source_networks=(VPN_EGRESS_NETWORK, )),
            TC5RiskStage(config, min_denominator=1),
            TC5ScoreStage(config, scorer=scorer, manifest=manifest, feature_columns=SCORED_FEATURES),
        ],
        impose_order,
        anchor=CHAIN_ANCHORS["tc5_auth"],
        envelope=CLASS_ENVELOPE["tc5_auth"],
        daily=[TC5DriftStage(config, window_column=f"{DRIFT_PREFIX}window_id", aggregate="mean")])

    outputs["tc5_session"] = _run_class(config,
                                        batches["tc5_session"],
                                        [TC5SessionStage(config, timeout_seconds=SESSION_TIMEOUT_SECONDS)],
                                        impose_order,
                                        anchor=CHAIN_ANCHORS["tc5_session"],
                                        envelope=CLASS_ENVELOPE["tc5_session"])

    frames = []

    for (name, frame) in outputs.items():
        frame = frame.copy()
        frame["telemetry_class"] = name
        frame["row_key"] = frame["event_uid"]
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    return canonicalize(combined, key_columns=KEY_COLUMNS, ignore_columns=IGNORE_COLUMNS)


def render(result: pd.DataFrame) -> str:
    """The byte-exact rendering compared across restarts and against the golden file."""
    return result.to_csv(index=False, lineterminator="\n")
