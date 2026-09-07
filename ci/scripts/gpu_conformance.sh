#!/bin/bash
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
#
# One command for the verdict this repository cannot render itself.
#
# Everything else here runs in CI. This cannot: there is no GPU, and the fork's claim to support one rests on
# variants that only execute on a machine with a card in it. So the run has to be one command, it has to say what
# it actually ran, and it has to fail in a way nobody can mistake for a pass.
#
# Three ways a GPU run has silently reported success in this project's history, each guarded here:
#
#   1. No CUDA device, so every gpu_mode test deselects and pytest exits 0 on "no tests ran".
#   2. A marker filter quietly drops a file -- `-m gpu_mode` deselects the digest equivalence gate, which carries
#      no mode marker -- so the run is green and the thing you wanted checked never executed.
#   3. `--run_slow` omitted, so the cross-restart checks report SKIPPED among a wall of PASSED.
#
# Writes a JSON artifact naming the card, the driver, the date and the counts, so the result can be pasted into a
# document as evidence rather than summarized from memory.
#
#   ./ci/scripts/gpu_conformance.sh [output.json]

set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/../.." &> /dev/null && pwd )"
ARTIFACT="${1:-${REPO_ROOT}/gpu_conformance.json}"

# Numba's default driver bindings read back an invalid CUDA context through the WSL shim: cuCtxGetDevice returns a
# garbage device number and the run dies inside libcuda. This is an environment defect rather than a code one, and
# it costs a day to rediscover, so it is set here rather than documented.
export NUMBA_CUDA_USE_NVIDIA_BINDING="${NUMBA_CUDA_USE_NVIDIA_BINDING:-1}"

# What this fork added, which is the verdict this script exists to render. Every one of these builds its corpus
# in code rather than reading a checked-in fixture -- a deliberate choice the guide argues for, and the reason
# this tier needs no Git LFS objects and cannot be blocked by a checkout that lacks them.
#
# The two lists are maintained by hand and kept total by a test. `tests/morpheus/utils/test_gpu_conformance_targets.py`
# identifies this fork's own test files by their copyright header and fails if one is in neither list, because a
# selection that quietly drops a file is the exact failure this script was written to prevent -- and it happened
# twice. First, five TC-5 stage files sat outside both tiers through two merges, so the layer 5 work went
# unmeasured while the runner reported a clean verdict on everything else. Second, this tier named the whole
# `tests/morpheus/determinism` directory, which reads as complete and was not: `-m gpu_mode` selects only the
# 29 tests in the five files that carry a mode marker, and the other six files -- the liveness registry, the
# first-detection corpus, the representation invariance suite, the Splunk package -- were deselected on every
# GPU run this has ever rendered a verdict from. Files are named individually for that reason: a directory entry
# cannot be checked against the markers inside it, and something that cannot be checked is what goes stale.
TARGETS=(
    tests/morpheus/determinism/test_determinism_harness.py
    tests/morpheus/determinism/test_gpu_parity.py
    tests/morpheus/determinism/test_session_harness.py
    tests/morpheus/determinism/test_siem_wire_contract.py
    tests/morpheus/determinism/test_telemetry_harness.py
    tests/morpheus/stages/test_binding_resolver_stage.py
    tests/morpheus/stages/test_determinism_stamp_stage.py
    tests/morpheus/stages/test_community_id_stage.py
    tests/morpheus/stages/test_lineage_stamp_stage.py
    tests/morpheus/stages/test_siem_wire_stage.py
    tests/morpheus/stages/test_tc1_binding_stage.py
    tests/morpheus/stages/test_tc1_change_stage.py
    tests/morpheus/stages/test_tc1_feature_stage.py
    tests/morpheus/stages/test_tc1_flap_stage.py
    tests/morpheus/stages/test_tc1_normalize_stage.py
    tests/morpheus/stages/test_tc1_optical_stage.py
    tests/morpheus/stages/test_tc2_arp_stage.py
    tests/morpheus/stages/test_tc2_auth_stage.py
    tests/morpheus/stages/test_tc2_binding_stage.py
    tests/morpheus/stages/test_tc2_cardinality_stage.py
    tests/morpheus/stages/test_tc5_cadence_stage.py
    tests/morpheus/stages/test_tc5_drift_stage.py
    tests/morpheus/stages/test_tc5_novelty_stage.py
    tests/morpheus/stages/test_tc5_risk_stage.py
    tests/morpheus/stages/test_tc5_score_stage.py
    tests/morpheus/stages/test_tc5_session_stage.py
    tests/morpheus/stages/test_tc5_travel_stage.py
    tests/morpheus/stages/test_total_order_stage.py
    tests/morpheus/stages/test_window_seal_stage.py
    tests/morpheus/utils/test_column_assign.py
)
# Files with no execution-mode marker. On a machine with a GPU the default mode is GPU, so these run on the device
# without being selected by the marker -- which is the only way they run at all, and why they are a tier rather
# than an omission. `test_lineage_cudf.py` is the one that matters most: its digest equivalence gate asserts the
# GPU and CPU hashing paths agree, and a marker filter dropped it once already.
UNMARKED=(
    tests/morpheus/determinism/test_end_to_end_mac_spoof.py
    tests/morpheus/determinism/test_first_detections.py
    tests/morpheus/determinism/test_layer5_model_runner.py
    tests/morpheus/determinism/test_representation_invariance.py
    tests/morpheus/determinism/test_splunk_validation_package.py
    tests/morpheus/determinism/test_stage_parameter_liveness.py
    tests/morpheus/stages/test_lineage_stage_cli.py
    tests/morpheus/utils/test_binding_closer.py
    tests/morpheus/utils/test_binding_table.py
    tests/morpheus/utils/test_community_id.py
    tests/morpheus/utils/test_counter_delta.py
    tests/morpheus/utils/test_cyclic_histogram.py
    tests/morpheus/utils/test_determinism.py
    tests/morpheus/utils/test_determinism_envelope.py
    tests/morpheus/utils/test_distinct_window.py
    tests/morpheus/utils/test_drift_trajectory.py
    tests/morpheus/utils/test_entity_key.py
    tests/morpheus/utils/test_event_clock.py
    tests/morpheus/utils/test_geo_velocity.py
    tests/morpheus/utils/test_gpu_conformance_report.py
    tests/morpheus/utils/test_gpu_conformance_targets.py
    tests/morpheus/utils/test_lineage.py
    tests/morpheus/utils/test_model_manifest.py
    tests/morpheus/utils/test_lineage_cudf.py
    tests/morpheus/utils/test_link_flap.py
    tests/morpheus/utils/test_optical_baseline.py
    tests/morpheus/utils/test_outcome_run.py
    tests/morpheus/utils/test_ratio_window.py
    tests/morpheus/utils/test_session_timer.py
    tests/morpheus/utils/test_sharding.py
    tests/morpheus/utils/test_siem_sourcetypes.py
    tests/morpheus/utils/test_siem_wire.py
    tests/morpheus/utils/test_splunk_app_contracts.py
    tests/morpheus/utils/test_splunk_field_contracts.py
    tests/morpheus/utils/test_value_novelty.py
    tests/morpheus/utils/test_window_seal.py
)
# The wider suite, which is context rather than this fork's verdict: upstream stages, upstream fixtures. It reads
# files stored in Git LFS, so it is skipped rather than failed on a checkout without them.
WIDER=(
    tests/morpheus/stages
    tests/morpheus/utils
)
# The floor the collected count has to clear, computed from the tier rather than pinned. A constant here goes
# stale the moment the tier grows -- which is how a tier that had lost five files still cleared a floor written
# when it had fourteen.
#
# Three per entry, which is what the smallest contributor in the tier actually collects: the stage files run
# twenty or more mode variants each, but `test_gpu_parity.py` runs three, and a floor above what an honest tier
# collects fails good runs. This is a gross-loss backstop and nothing finer -- it catches a filter or a missing
# dependency that took most of the suite, and it would not notice one file going missing. That is the totality
# test's job, and the reason this script no longer trusts a floor to do it.
MINIMUM_SELECTED=$(( ${#TARGETS[@]} * 3 ))


# Written with printf rather than with python, because the failure path is exactly where the environment may be
# the thing that is broken -- and it was: run outside the container, `python` is not on PATH, so the artifact this
# function exists to leave behind was never written and the run looked like one that had not happened. Every
# reason string below is quote-free and single-line for the same reason; keep them that way.
fail() {
    echo ""
    echo "FAILED: $1"
    echo ""
    printf '{\n  "verdict": "failed",\n  "reason": "%s",\n  "at": "%s"\n}\n' \
        "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ARTIFACT" || true
    exit 1
}

cd "${REPO_ROOT}" || fail "cannot enter ${REPO_ROOT}"

# pytest puts a test's outcome on the line after its identifier when the identifier does not fit the terminal,
# and every identifier in the marked tier carries a `[gpu_mode]` suffix. On a narrow terminal that wrapped all of
# them, so a tier of 379 passing tests parsed as zero counted and 379 unaccounted. The report module reads a
# wrapped outcome now too; this stops the wrapping happening in the first place, and neither alone is trusted.
export COLUMNS=200

echo "=== device ==="
if ! command -v nvidia-smi > /dev/null 2>&1; then
    fail "no CUDA device: nvidia-smi is not on PATH. This verdict needs a machine with a GPU; it cannot be rendered in CI or in a CPU-only container."
fi

DEVICE=$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null | head -1)
if [[ -z "${DEVICE}" ]]; then
    fail "no CUDA device: nvidia-smi reported no GPU. A silent skip must never read as a pass, so this is an error rather than a deselection."
fi
echo "${DEVICE}"

echo ""
echo "=== interpreter ==="
# Checked separately from the device, because the two live in different places. `nvidia-smi` is visible from a
# WSL host that has no Morpheus environment at all, so the device check passes there and everything after it
# fails for a reason that has nothing to do with a GPU: the first run outside the container collected zero tests
# and the floor reported a marker filter dropping the suite, which was a confident and wrong diagnosis. Name the
# real cause here instead.
if ! command -v python > /dev/null 2>&1; then
    fail "python is not on PATH. This runs inside the Morpheus environment, not on the host beside it -- nvidia-smi is visible from a WSL host that has no environment at all, which is why the device check above passed."
fi

if ! python -c "import morpheus" > /dev/null 2>&1; then
    fail "python is on PATH but cannot import morpheus, so every collection below would come back empty and be reported as a suite that went missing. Activate the environment first."
fi

echo "$(python -c 'import sys; print(sys.executable)')"
python -c "import morpheus; print(f'morpheus {morpheus.__version__}')" 2>/dev/null || echo "morpheus imports"

echo ""
echo "=== test data ==="
# Every fixture under tests/tests_data is stored in Git LFS. A checkout without the objects has 128-byte pointer
# files in their place, and a test reading one gets "Parquet magic bytes not found" or a CSV of one line of YAML.
# That is what 37 failures across test_file_source_stage_pipe turned out to be the first time this ran, and it
# cost an investigation to learn. A pre-flight that names it costs a second.
POINTERS=$(git ls-files tests/tests_data 2>/dev/null | while read -r f; do
    head -c 40 "$f" 2>/dev/null | grep -q "git-lfs" && echo "$f"
done | wc -l)

if [[ "${POINTERS}" -gt 0 ]]; then
    # Not fatal. This fork's own tests build their corpora in code and read none of these, so the verdict below
    # is unaffected; what is lost is the wider upstream suite, which reads them and would otherwise report a wall
    # of failures whose common cause is a checkout rather than a GPU.
    echo "${POINTERS} fixtures under tests/tests_data are unfetched Git LFS pointers."
    echo "The wider upstream suite will be skipped. It reads those files; this fork's own tests do not."
    echo "To include it: git lfs install && git lfs pull -- both, in that order, because on a clone where LFS was"
    echo "never set up 'git lfs pull' alone prints 'Skipping object checkout' and exits zero having done nothing."
    RUN_WIDER=0
else
    echo "test fixtures are real files; the wider upstream suite will run too"
    RUN_WIDER=1
fi

echo ""
echo "=== gpu_mode variants ==="
# Kept rather than counted and discarded. A count says a run does not add up; the names say which test it does
# not add up by, and the first run where the tiers were total came back one short with nothing failed and no
# name to look for. The report module reconciles against these.
python -m pytest -m gpu_mode --run_slow --collect-only -q "${TARGETS[@]}" \
    > /tmp/gpu_conformance_marked_collected.txt 2>/dev/null
SELECTED=$(grep -c "::" /tmp/gpu_conformance_marked_collected.txt) || SELECTED=0
echo "collected ${SELECTED} gpu_mode tests"

if [[ "${SELECTED}" -lt "${MINIMUM_SELECTED}" ]]; then
    fail "only ${SELECTED} gpu_mode tests collected, expected at least ${MINIMUM_SELECTED}. A marker filter or a missing dependency has dropped part of the suite, and a run that skips what you meant to check is worth less than one that fails."
fi

# Verbose rather than quiet, deliberately. With -q pytest names failures only in its end-of-run summary, so a
# run that aborts -- a segfault in a device stage is exactly the kind of thing this is looking for -- takes the
# names down with it and the artifact reports a failure it cannot describe. Streamed per-test lines survive the
# process dying, and the last one names where it died.
python -m pytest -m gpu_mode --run_slow -v --tb=short "${TARGETS[@]}" 2>&1 | tee /tmp/gpu_conformance_marked.log
MARKED_STATUS=${PIPESTATUS[0]}

echo ""
echo "=== coverage that carries no mode marker ==="
# The digest equivalence gate asserts the device and host hashing paths agree. It has no gpu_mode marker, so
# `-m gpu_mode` deselects it -- which is exactly how it went unrun after the host digest changed.
#
# Collected first, for the same reason the marked tier is: the artifact reconciles what it counted against what
# pytest said it would run, and without a number to reconcile against, a parser that drops tests reports a pass.
python -m pytest --run_slow --collect-only -q "${UNMARKED[@]}" \
    > /tmp/gpu_conformance_unmarked_collected.txt 2>/dev/null
UNMARKED_SELECTED=$(grep -c "::" /tmp/gpu_conformance_unmarked_collected.txt) || UNMARKED_SELECTED=0
echo "collected ${UNMARKED_SELECTED} unmarked tests"

python -m pytest --run_slow -v --tb=short "${UNMARKED[@]}" 2>&1 | tee /tmp/gpu_conformance_unmarked.log
UNMARKED_STATUS=${PIPESTATUS[0]}

echo ""
echo "=== wider upstream suite ==="
WIDER_STATUS=0
if [[ "${RUN_WIDER}" -eq 1 ]]; then
    python -m pytest -m gpu_mode --run_slow -v --tb=short "${WIDER[@]}" 2>&1 | tee /tmp/gpu_conformance_wider.log
    WIDER_STATUS=${PIPESTATUS[0]}
else
    echo "skipped: see the test data note above"
    : > /tmp/gpu_conformance_wider.log
fi

echo ""
echo "=== artifact ==="
# The reporting lives in a module beside this script rather than in a heredoc, because it has been wrong twice
# and a parser nobody can test is a parser that will be wrong a third time. `tests/morpheus/utils/
# test_gpu_conformance_report.py` exercises it against the log shapes that broke it.
python "${SCRIPT_DIR}/gpu_conformance_report.py" \
    "$ARTIFACT" "$DEVICE" "$SELECTED" "$MARKED_STATUS" "$UNMARKED_SELECTED" "$UNMARKED_STATUS" \
    "$RUN_WIDER" "$WIDER_STATUS"
REPORT_STATUS=$?

# The report's own verdict decides, not just the exit codes. A tier can exit zero having lost tests to a parser
# that could not read their names, and that is not a pass -- the reconciliation in the artifact is what catches
# it, so this has to defer to the artifact rather than to the status alone.
if [[ "${MARKED_STATUS}" -ne 0 || "${UNMARKED_STATUS}" -ne 0 || "${REPORT_STATUS}" -ne 0 ]]; then
    echo ""
    echo "FAILED: see ${ARTIFACT} for what failed."
    exit 1
fi

echo ""
echo "PASSED. Artifact written to ${ARTIFACT}."
