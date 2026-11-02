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
The week-long layer 5 corpus, and the composed session pipeline, for the determinism harness.

The layer 1 and 2 harness proved that fourteen stages in a row reach the answer a golden file holds. This module
does the same for layer 5: the seven TC-5 stages are composed here, and every layer 5 saved search in `RULES` is
asserted against this corpus.

**A week, not an hour.** The layer 1 and 2 corpus covers an hour because a poller's cadence is a minute. Layer 5
is event-driven and its features are about habit: an hour-of-day histogram needs days to say anything, an
impossible journey needs hours to be impossible, and a cumulative location count needs somewhere to have been
before. The scored week therefore runs seven days from a Monday midnight, and the window period is an hour rather
than five minutes.

**And a fortnight before it that is never scored.** Each principal's model is trained on the two weeks before the
scored week, in which everybody does what they always do, and scores only the week after. That is the line
between a model and a memory: a model asked about the rows it was fitted on reports how well it memorized them,
and every run would leak identically, so no determinism control would notice. The fortnight runs through the
same stages as the week, because the features are cumulative and a scored row's history is part of what it
measures, and its rows reach the golden file with null scores and no model version. The trained models are
committed under `examples/layer5_model/models/`, pinned in the scoring manifest by a digest of their numbers, and
scored without Torch.

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
- a **session identifier used twice**, which is a duplicating collector rather than a retry;
- a **long interactive session**: one office worker's last working day runs eleven hours against four of eight,
  which R-B-L5-005 reports, beside a **service account** whose nightly sessions alternate between ten minutes and
  four hours and end on a five-hour one -- the bimodal shape the rule excludes by account type rather than by luck;
- **single-record sessions** from an identity provider that states both ends in one record, one of them with its
  end before its start;
- a **deviating principal**: an office worker whose account, one afternoon of the scored week, signs in from a
  city it has never used, on a device it has never used, to applications it has never opened, with the second
  factor skipped and two failures among the successes. Nothing about it is impossible, so the deterministic travel
  rule stays quiet; it is the case the per-principal models exist for, and R-B-L5-001 and R-B-L5-002 fire on it;
- an **unseen principal**: a joiner who first signs in on the Monday of the scored week, has no model of her own,
  and is scored against the declared population fallback, which the two model rules refuse to read;
- and an **identity store**: each principal's profile and groups, a leaver whose termination is recorded the day
  after it takes effect, a group change on the day of a relocation, and the service account marked as one. Both
  layer 5 classes are enriched from it as known at each event's time.
The pipeline is one per telemetry class, which is the deployment shape: authentication events arrive from the
identity provider, session lifecycle records from the concentrators and the RADIUS accounting stream, and the two
have different required columns. Every stateful stage is preceded by `TotalOrderStage`, which is determinism
control 8 as a stage.
"""

import functools
import typing

import pandas as pd

from morpheus.compositions import envelope as universal_envelope
from morpheus.compositions import stamping
from morpheus.compositions.corpus_pipeline import _collect
from morpheus.compositions.corpus_pipeline import (  # noqa: F401 pylint: disable=unused-import  # re-exported for callers
    build_pipeline_config, )
from morpheus.compositions.repository import repository_path
from morpheus.compositions.resumable import ResumableLinearPipeline
from morpheus.config import Config
from morpheus.pipeline import Pipeline
from morpheus.stages.general.router_stage import RouterStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.lineage.collector_sequence_stage import CollectorSequenceStage
from morpheus.stages.lineage.determinism_stamp_stage import DeterminismStampStage
from morpheus.stages.lineage.envelope_stamp_stage import EnvelopeStampStage
from morpheus.stages.lineage.lineage_stamp_stage import LineageStampStage
from morpheus.stages.lineage.shard_split_stage import ShardSplitStage
from morpheus.stages.lineage.shard_split_stage import shard_key_fn
from morpheus.stages.lineage.total_order_stage import TotalOrderStage
from morpheus.stages.lineage.window_seal_stage import WindowSealStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.telemetry.tc0_enrich_stage import TC0EnrichStage
from morpheus.stages.telemetry.tc0_identity_stage import GROUP_ATTRIBUTE
from morpheus.stages.telemetry.tc0_identity_stage import PROFILE
from morpheus.stages.telemetry.tc5_cadence_stage import TC5CadenceStage
from morpheus.stages.telemetry.tc5_drift_stage import TC5DriftStage
from morpheus.stages.telemetry.tc5_novelty_stage import TC5NoveltyStage
from morpheus.stages.telemetry.tc5_risk_stage import TC5RiskStage
from morpheus.stages.telemetry.tc5_score_stage import Scorer
from morpheus.stages.telemetry.tc5_score_stage import TC5ScoreStage
from morpheus.stages.telemetry.tc5_session_stage import TC5SessionStage
from morpheus.stages.telemetry.tc5_travel_stage import TC5TravelStage
from morpheus.utils.binding_table import NS_PER_SECOND
from morpheus.utils.bitemporal import MEMBERSHIP
from morpheus.utils.bitemporal import BitemporalStore
from morpheus.utils.bitemporal import make_version
from morpheus.utils.determinism import DEFAULT_ORDER_COLUMNS
from morpheus.utils.determinism import canonicalize
from morpheus.utils.dfencoder_scorer import DfencoderScorer
from morpheus.utils.dfencoder_scorer import load_models
from morpheus.utils.model_manifest import ModelManifest
from morpheus.utils.sharding import shard_names

PERIOD_SECONDS = 3600
LATENESS_SECONDS = 900
TRAINING_DAYS = 14
"""The fortnight before the scored week that the models are trained on. Day indices in it are negative."""
HOUR_S = 3600
DAY_S = 24 * HOUR_S
CORPUS_EPOCH_S = (4 + TRAINING_DAYS) * DAY_S
"""The scored week starts on a Monday midnight UTC, and the training fortnight on the Monday two weeks before.

1970-01-01 was a Thursday, so four days on is the first Monday, where the fortnight starts. Anchoring there rather
than at the epoch itself means day zero of the scored week is a weekday and the weekend falls where a reader
expects it, without the corpus depending on any real date -- and therefore without a golden file that ages.
"""
SCORES_FROM_NS = CORPUS_EPOCH_S * 1_000_000_000
"""Where the training data ends and scoring begins: the scored week's Monday midnight, in epoch nanoseconds."""
ID_COLUMNS = ["collector_id", "schema_version", "origin_hash", "collector_seq"]
KEY_COLUMNS = ["telemetry_class", "row_key"]
IGNORE_COLUMNS: list[str] = []
CADENCE_MIN_SAMPLES = 6
"""Prior observations before a principal's histogram is called mature, for this corpus.

Below the stage's own default of 32, and deliberately: a week of one estate's logins is not a production history,
and a harness whose maturity flag is false on every row would assert nothing about it. Six rather than eight so
that the service account, which authenticates once a night, is mature by the last night of the corpus -- without
that it could never serve as the negative control it exists to be. The number is here rather than in the stage
because it is a property of the corpus.
"""
VPN_EGRESS_NETWORK = "198.51.100.0/24"
ALICE = "alice@example.com"
BOB = "bob@example.com"
CAROL = "carol@example.com"
DAVE = "dave@example.com"
BATCH = "svc-batch@example.com"
ERIN = "erin@example.com"
GRACE = "grace@example.com"
FLIGHT_DAY = 3
JOINER_DAY = 0
"""Grace's first day, the Monday of the scored week. She is in no training window and has no model."""
SESSION_BASELINE_MIN_SAMPLES = 4
"""Prior sessions before a principal's duration baseline is published, for this corpus.

Below the stage's own floor of a hundred, deliberately, for the reason `CADENCE_MIN_SAMPLES` gives: a week holds five
working days, and a floor nobody reaches would make the baseline assert nothing. By nearest rank a 99th percentile
of four sessions is the longest of them, which is what this corpus's long session is measured against.
"""
LEAVER_DAY = 5
LEAVER_RECORDED_DAY = 6
"""Dave leaves on the Saturday; the directory hears on the Sunday. His Saturday sign-ins carry `active` as known at
their time, which is what a detection could have known, and his Sunday ones carry `terminated`."""
GROUP_CHANGE_GROUPS = ("london-office", "newyork-office")
"""Bob's office group, changed on the day he flies, recorded the same day."""
SESSION_TIMEOUT_SECONDS = 12 * HOUR_S
"""Shorter than the stage's own default, because this corpus is a week and a session abandoned after a day and a
half would never be abandoned inside it."""


def _profile(principal: str,
             department: str,
             status: str,
             valid_from_s: int,
             recorded_s: int,
             account_type: str = "human",
             privilege: str = "standard"):
    return make_version(PROFILE,
                        principal,
                        valid_from_s * NS_PER_SECOND,
                        None,
                        recorded_s * NS_PER_SECOND,
                        values={
                            "department": department,
                            "employment_status": status,
                            "account_type": account_type,
                            "privilege_level": privilege,
                        })


def _member(principal: str, group: str, valid_from_s: int, recorded_s: int, change: str = "assert"):
    return make_version(MEMBERSHIP,
                        principal,
                        valid_from_s * NS_PER_SECOND,
                        None,
                        recorded_s * NS_PER_SECOND,
                        change, {GROUP_ATTRIBUTE: group}, (group, ))


def identity_versions() -> list:
    """The identity store the layer 5 classes are enriched from: what the directory recorded, and when."""
    start = at(-TRAINING_DAYS, 0) - DAY_S
    (london, newyork) = GROUP_CHANGE_GROUPS

    return [
        _profile(ALICE, "Finance", "active", start, start, privilege="elevated"),
        _profile(BOB, "Engineering", "active", start, start),
        _profile(CAROL, "Sales", "active", start, start),
        _profile(DAVE, "Finance", "active", start, start),
        _profile(BATCH, "Engineering", "active", start, start, account_type="service", privilege="elevated"),
        _member(ALICE, "finance-users", start, start),
        _member(BOB, london, start, start),
        _member(CAROL, "sales-users", start, start),
        _member(DAVE, "finance-users", start, start),
        _member(BATCH, "batch-jobs", start, start),
        _profile(ERIN, "Finance", "active", start, start),
        _member(ERIN, "finance-users", start, start),
        _profile(GRACE, "Sales", "active", at(JOINER_DAY, 0), at(JOINER_DAY, 0)),
        _member(GRACE, "sales-users", at(JOINER_DAY, 0), at(JOINER_DAY, 0)),
        _member(BOB, london, at(FLIGHT_DAY, 0), at(FLIGHT_DAY, 0), "retract"),
        _member(BOB, newyork, at(FLIGHT_DAY, 0), at(FLIGHT_DAY, 0)),
        _profile(DAVE, "Finance", "terminated", at(LEAVER_DAY, 0), at(LEAVER_RECORDED_DAY, 0)),
    ]


def build_identity_store() -> BitemporalStore:
    """The store, rebuilt per run so no state leaks between the runs control 13 compares."""
    return BitemporalStore("identity", identity_versions())


def at(day: int, hour: int, second: int = 0) -> int:
    """Epoch seconds for an hour on a corpus day, counted from the corpus's Monday midnight."""
    return CORPUS_EPOCH_S + day * DAY_S + hour * HOUR_S + second


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
SETTINGS = {
    "period_seconds": PERIOD_SECONDS,
    "lateness_seconds": LATENESS_SECONDS,
    "day_seconds": DAY_S,
    "cadence_min_samples": CADENCE_MIN_SAMPLES,
    "session_timeout_seconds": SESSION_TIMEOUT_SECONDS,
    "session_baseline_min_samples": SESSION_BASELINE_MIN_SAMPLES,
    "excluded_source_networks": [VPN_EGRESS_NETWORK],
    "training_days": TRAINING_DAYS,
    "scores_from_ns": SCORES_FROM_NS,
}
"""The settings that decide this corpus's output, digested into `config_hash` by `stamping.envelope_for`."""
RULES = ("R-D-L5-003",
         "R-D-L5-004",
         "R-D-L5-007",
         "R-D-L5-008",
         "R-D-L5-009",
         "R-B-L5-001",
         "R-B-L5-002",
         "R-B-L5-005",
         "R-P-L5-006")
"""The shipped rules that read this corpus's columns; their thresholds are folded into `pipeline_fingerprint`."""


def _run_class(config: Config,
               dataframes: list[pd.DataFrame],
               stages: typing.Callable[[], list],
               impose_order: bool,
               anchor: str,
               envelope: tuple = None,
               daily: typing.Callable[[], list] = None,
               telemetry_class: str = None,
               manifest: ModelManifest = None,
               shards: int = None,
               shard_column: str = "user_principal") -> pd.DataFrame:
    """Source → stamp → (total order) → the class's stages → determinism stamp → (envelope) → window seal →
    (daily seal → daily stages) → sink, as one frame.

    `stages` and `daily` are factories rather than lists because a sharded run needs one instance of every
    stateful stage per branch. With `shards`, the stream is cut on `shard_column` after the collector sequence
    and everything from the class's stages to the sink runs once per shard (control 4); the frames the sinks
    collect are concatenated, and canonicalization puts them in the one order.

    The determinism stamp carries `manifest` on the scored class, so every scored row names the model version it
    was scored against and whether that was a fallback; on the other class the model columns are null.

    The hourly seal is the one the chains and the shipped detections are built on. A class given `daily` is
    sealed a second time, into days, behind it -- with the daily columns prefixed so the hourly ones keep their
    identity -- and the daily stages run on those complete days. That is where a trajectory across days is
    measured: R-P-L5-006 asks about four consecutive daily windows, and a day is not something an hourly window
    knows about.
    """
    if (telemetry_class is None):
        raise ValueError("every class is stamped with the determinism envelope, and the envelope names the class")

    def branch() -> list:
        """The class's stages through the seals, as a deployment composes them behind the shared prefix."""
        built = list(stages())
        built.append(
            DeterminismStampStage(config,
                                  envelope=stamping.envelope_for(telemetry_class, SETTINGS, rules=RULES),
                                  manifest=manifest,
                                  entity_column="user_principal"))

        if (envelope is not None):
            (osi_layer, entity_columns) = envelope
            built.append(EnvelopeStampStage(config, osi_layer=osi_layer, entity_columns=entity_columns))

        built.append(
            WindowSealStage(config,
                            period_seconds=PERIOD_SECONDS,
                            lateness_seconds=LATENESS_SECONDS,
                            order_columns=list(DEFAULT_ORDER_COLUMNS),
                            entity_key_column=anchor))

        if (daily is not None):
            built.append(
                WindowSealStage(config,
                                period_seconds=DAY_S,
                                lateness_seconds=LATENESS_SECONDS,
                                order_columns=list(DEFAULT_ORDER_COLUMNS),
                                column_prefix=DRIFT_PREFIX))
            built.extend(daily())

        return built

    if (shards is None):
        pipe = ResumableLinearPipeline(config)
        pipe.set_source(InMemorySourceStage(config, dataframes=dataframes))
        pipe.add_stage(LineageStampStage(config, id_columns=ID_COLUMNS))

        if (impose_order):
            pipe.add_stage(TotalOrderStage(config))

        pipe.add_stage(CollectorSequenceStage(config))

        for stage in branch():
            pipe.add_stage(stage)

        sink = pipe.add_stage(InMemorySinkStage(config))
        pipe.run()

        return _collect(sink)

    # Control 4: the shared prefix, then a router over the stable hash of the entity key, then one branch per
    # shard holding its own instances of every stateful stage, and one sink per branch.
    pipe = Pipeline(config)
    previous = pipe.add_stage(InMemorySourceStage(config, dataframes=dataframes))
    prefix = [LineageStampStage(config, id_columns=ID_COLUMNS)]

    if (impose_order):
        prefix.append(TotalOrderStage(config))

    prefix += [CollectorSequenceStage(config), ShardSplitStage(config, key_column=shard_column, shards=shards)]

    for stage in prefix:
        pipe.add_stage(stage)
        pipe.add_edge(previous, stage)
        previous = stage

    router = pipe.add_stage(RouterStage(config, keys=shard_names(shards), key_fn=shard_key_fn(shard_column, shards)))
    pipe.add_edge(previous, router)
    sinks = []

    for (index, port) in enumerate(router.output_ports):
        previous = port

        for stage in branch():
            pipe.add_stage(stage)
            pipe.add_edge(previous, stage)
            previous = stage

        sinks.append(pipe.add_stage(InMemorySinkStage(config)))
        pipe.add_edge(previous, sinks[index])

    pipe.run()
    frames = [_collect(sink) for sink in sinks]

    return pd.concat([frame for frame in frames if len(frame) > 0], ignore_index=True)


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
FALLBACK_VERSION = "reference-arithmetic:0"
"""The population fallback the manifest declares for a principal with no model of its own.

`DeterminismStampStage`, placed behind the scorer, writes `model_fallback_used` as true beside it, and R-B-L5-001
and R-B-L5-002 refuse to read such a row: a score against a population is a claim about the population rather than
about the principal's own history.
"""
MODELS_PATH = repository_path("examples", "layer5_model", "models", "session_models.json")
"""The committed per-principal models, trained on the fortnight by `examples/layer5_model/train_models.py`."""


class PinnedScorer:
    """
    Answers for every version the manifest can resolve: each principal's own model, and the population fallback.

    The pinned versions go to `DfencoderScorer`, which holds the committed models as `NumpyAutoEncoder`s; the
    fallback goes to `ReferenceScorer`. A version neither holds is refused by `DfencoderScorer`, which is the
    wiring mistake that refusal exists for.
    """

    def __init__(self, models: dict):
        self._models = DfencoderScorer(models, SCORED_FEATURES)
        self._fallback = ReferenceScorer()

    @property
    def versions(self) -> list[str]:
        """Every version this scorer answers for, sorted."""
        return sorted(self._models.versions + [FALLBACK_VERSION])

    def score(self, model_version: str, features: list) -> list:
        if (model_version == FALLBACK_VERSION):
            return self._fallback.score(model_version, features)

        return self._models.score(model_version, features)


def build_scoring(models: dict) -> tuple[PinnedScorer, ModelManifest]:
    """
    The scorer and the manifest for a set of trained models, built together so neither pins what the other lacks.

    Parameters
    ----------
    models : dict
        Principal to `(pinned version, fitted model)`.
    """
    scorer = PinnedScorer(dict(models.values()))
    manifest = ModelManifest(window_id=SCORING_WINDOW,
                             models={principal: version
                                     for (principal, (version, _)) in sorted(models.items())},
                             fallback=FALLBACK_VERSION,
                             scores_from_ns=SCORES_FROM_NS)

    return (scorer, manifest)


@functools.lru_cache(maxsize=1)
def committed_scoring() -> tuple[PinnedScorer, ModelManifest]:
    """The committed models, read and checked against their recorded versions once per process."""
    return build_scoring(load_models(MODELS_PATH))


def scoring_manifest() -> ModelManifest:
    """
    The scoring manifest: every principal with a model pinned to the digest of its committed numbers, the
    population fallback for anybody else, and scoring from the end of the fortnight the models were trained on.
    """
    return committed_scoring()[1]


def training_frames(result: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    What each principal's model is trained on: its authentication rows from the fortnight, as the ten features.

    Taken from a pipeline run rather than computed separately, because the features are what the stages emit and
    a training set derived any other way would describe a different pipeline. Rows with a gap in any feature are
    dropped rather than imputed: these columns are counts and surprise scores, and a null means the row carried no
    such measurement rather than a measurement of zero.
    """
    auth = result[result["telemetry_class"] == "tc5_auth"]
    training = auth[auth["event_time"].astype("int64") < SCORES_FROM_NS]
    frames = {}

    for principal in sorted(set(training["user_principal"].dropna())):
        rows = training[training["user_principal"] == principal]
        frames[principal] = rows[SCORED_FEATURES].dropna().astype("float64").reset_index(drop=True)

    return frames


REFERENCE_PARAMETERS = {
    "logcount": (3.582609, 0.962009),
    "locincrement": (1.23913, 0.427483),
    "appincrement": (1.0, 0.0),
    "deviceincrement": (1.0, 0.0),
    "asns_in_window": (1.23913, 0.427483),
    "hour_surprise_bits": (2.887763, 0.828301),
    "weekday_surprise_bits": (2.762205, 0.672246),
    "mfa_ratio": (0.93913, 0.239612),
    "auth_attempts_in_window": (1.0, 0.0),
    "auth_failures_in_window": (0.0, 0.0),
}
"""Per-feature mean and sample standard deviation over every principal's rows in the training fortnight, frozen here.

This is the population the manifest's fallback scores a principal with no model against, so it is fitted on the
fortnight alone, as the models are, and never on the week it scores. `test_session_harness` recomputes it from
the fortnight and fails if these constants drift from it.

Frozen rather than computed, and the reason is the first thing control 5 caught. The first version of the
scorer below derived these from the rows it was handed, which made every score a function of how the stream
happened to be batched: the same authentication scored differently in a batch of ten and a batch of a hundred,
and `test_batch_split_sweep` failed on the first run. A trained model does not have this problem, because its
parameters are learned once and fixed before any scoring happens. Freezing them here is the stand-in for that,
and it is also what keeps the stub from fitting on the data it is scoring -- a leak no determinism control would
catch, because every run would leak identically.

A deviation of zero means the feature never varied in the fortnight; those score zero rather than dividing by it.
"""


class ReferenceScorer:
    """**Not a model.** Arithmetic that stands in for one so the scoring path can be tested.

    It is the manifest's declared fallback: the population a principal with no model of their own is scored
    against, which in this corpus is the joiner. Until each principal had a committed model it answered for
    everybody; it now answers for nobody who has a fortnight of history.

    It returns each feature's distance from a frozen mean in units of a frozen deviation -- a z-score in the
    literal sense and nothing more. Deterministic, independent of batching, independent of the order rows arrive
    in, with no learned parameters, no history and no notion of normal beyond the constants above.

    **No detection claim attaches to any number it produces.** Its scores in the golden file are arithmetic, not
    evidence, and every row carrying one says `model_fallback_used` is true, which is why R-B-L5-001 and R-B-L5-002
    refuse to read them. A threshold tuned against them would be tuned against this docstring.
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
                 manifest: typing.Optional[ModelManifest] = None,
                 shards: typing.Optional[int] = None,
                 shard_column: str = "user_principal") -> pd.DataFrame:
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
        What answers for the scores. Defaults to the committed models behind `PinnedScorer`, which is what the
        golden file is built on. `examples/layer5_model/run_model.py` passes the models it has just trained on
        a card, which is how the same path is run with Torch's own numbers.
    manifest : `morpheus.utils.model_manifest.ModelManifest`, optional
        What pins each principal to a version. Defaults to the committed manifest from `scoring_manifest`.
        Passed together with `scorer`, or the versions one resolves are ones the other does not hold.
    shards : int, optional
        Run each class's stages on this many branches, routed by the stable hash of `shard_column` (control 4),
        rather than on one. The output is the same frame; the harness asserts it.
    shard_column : str, default = "user_principal"
        The entity key the shards are decided on. Both classes are keyed on the principal, so a principal's
        authentications and sessions share a branch. The harness's negative control shards on a column that
        splits a principal and shows the output change.

    Returns
    -------
    `pandas.DataFrame`
        Both classes' output, tagged with `telemetry_class`, keyed by `row_key`, canonicalized.
    """
    if (batches is None):
        batches = {name: [frame.copy()] for (name, frame) in corpus.items()}

    # Validated once per class, ahead of every stage: ingest_time stamped, the envelope checked, untrusted clocks
    # taken out of the stream and kept.
    (batches, quarantined) = universal_envelope.validate(config, batches, ID_COLUMNS)

    if ((scorer is None) != (manifest is None)):
        raise ValueError("scorer and manifest are passed together or not at all; a manifest resolving versions "
                         "the scorer does not hold is the mismatch this pairing exists to prevent")

    if (scorer is None):
        (scorer, manifest) = committed_scoring()

    outputs = {}

    outputs["tc5_auth"] = _run_class(
        config,
        batches["tc5_auth"], lambda: [
            TC0EnrichStage(config, store=build_identity_store(), entity_column="user_principal"),
            TC5NoveltyStage(config),
            TC5CadenceStage(config, min_samples=CADENCE_MIN_SAMPLES),
            TC5TravelStage(config, excluded_source_networks=(VPN_EGRESS_NETWORK, )),
            TC5RiskStage(config, min_denominator=1),
            TC5ScoreStage(config, scorer=scorer, manifest=manifest, feature_columns=SCORED_FEATURES), ],
        impose_order,
        anchor=CHAIN_ANCHORS["tc5_auth"],
        envelope=CLASS_ENVELOPE["tc5_auth"],
        daily=lambda: [TC5DriftStage(config, window_column=f"{DRIFT_PREFIX}window_id", aggregate="mean")],
        telemetry_class="tc5_auth",
        manifest=manifest,
        shards=shards,
        shard_column=shard_column)

    outputs["tc5_session"] = _run_class(
        config,
        batches["tc5_session"], lambda: [
            TC0EnrichStage(config, store=build_identity_store(), entity_column="user_principal"),
            TC5SessionStage(config, timeout_seconds=SESSION_TIMEOUT_SECONDS, baseline_min_samples=
                            SESSION_BASELINE_MIN_SAMPLES), ],
        impose_order,
        anchor=CHAIN_ANCHORS["tc5_session"],
        envelope=CLASS_ENVELOPE["tc5_session"],
        telemetry_class="tc5_session",
        shards=shards,
        shard_column=shard_column)

    frames = []

    for (name, frame) in outputs.items():
        # A class every record of which was quarantined produced nothing, and is absent rather than an empty frame.
        if (len(frame) == 0):
            continue

        frame = frame.copy()
        frame["telemetry_class"] = name
        frame["row_key"] = frame["event_uid"]
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    return canonicalize(universal_envelope.with_quarantine(combined, quarantined),
                        key_columns=KEY_COLUMNS,
                        ignore_columns=IGNORE_COLUMNS)


def render(result: pd.DataFrame) -> str:
    """The byte-exact rendering compared across restarts and against the golden file."""
    return result.to_csv(index=False, lineterminator="\n")
