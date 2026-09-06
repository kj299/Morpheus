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

# Every gpu_mode variant this fork adds, plus the files whose GPU coverage carries no mode marker and would
# otherwise be deselected without anyone noticing.
TARGETS=(
    tests/morpheus/determinism
    tests/morpheus/stages
    tests/morpheus/utils
)
UNMARKED=(
    tests/morpheus/utils/test_lineage_cudf.py
)
MINIMUM_SELECTED=200

fail() {
    echo ""
    echo "FAILED: $1"
    echo ""
    python - "$ARTIFACT" "$1" <<'PY' || true
import json, sys, datetime
json.dump({"verdict": "failed", "reason": sys.argv[2],
           "at": datetime.datetime.now(datetime.timezone.utc).isoformat()},
          open(sys.argv[1], "w"), indent=2)
PY
    exit 1
}

cd "${REPO_ROOT}" || fail "cannot enter ${REPO_ROOT}"

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
echo "=== gpu_mode variants ==="
SELECTED=$(python -m pytest -m gpu_mode --run_slow --collect-only -q "${TARGETS[@]}" 2>/dev/null | grep -c "::") || SELECTED=0
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
python -m pytest --run_slow -v --tb=short "${UNMARKED[@]}" 2>&1 | tee /tmp/gpu_conformance_unmarked.log
UNMARKED_STATUS=${PIPESTATUS[0]}

echo ""
echo "=== artifact ==="
python - "$ARTIFACT" "$DEVICE" "$SELECTED" "$MARKED_STATUS" "$UNMARKED_STATUS" <<'PY'
import datetime
import json
import re
import sys

(path, device, selected, marked_status, unmarked_status) = sys.argv[1:6]


def summarize(log: str) -> dict:
    """
    Read the streamed per-test lines rather than the summary, so an aborted run still says what happened.

    A summary is written once, at the end. A run that dies partway through has no end, and the first version of
    this reported `"failures": []` for a run with dozens of them -- which is the same uselessness it was built to
    prevent, wearing a different hat.
    """
    try:
        with open(log, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return {"note": f"no log at {log}"}

    outcomes = re.findall(r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)", text, flags=re.MULTILINE)
    counts: dict = {}

    for (_, outcome) in outcomes:
        counts[outcome.lower()] = counts.get(outcome.lower(), 0) + 1

    started = re.findall(r"^(\S+::\S+)", text, flags=re.MULTILINE)
    reported = {name for (name, _) in outcomes}
    unfinished = [name for name in started if name not in reported]

    return {
        "counts": counts,
        "failures": sorted({name for (name, outcome) in outcomes if outcome in ("FAILED", "ERROR")}),
        # A test that started and never reported its outcome is where the process died.
        "died_in": unfinished[-1] if unfinished else None,
        "log": log,
    }


marked = summarize("/tmp/gpu_conformance_marked.log")
unmarked = summarize("/tmp/gpu_conformance_unmarked.log")
ok = marked_status == "0" and unmarked_status == "0"

def describe(status: str) -> str:
    """A shell exit code, in words. 134 is SIGABRT, which is what a device stage crashing looks like from here."""
    code = int(status)

    if (code == 0):
        return "exited cleanly"

    if (code > 128):
        return f"killed by signal {code - 128}" + (" (SIGABRT: a crash, not a test failure)" if code == 134 else "")

    return f"exited {code}"


report = {
    "verdict": "passed" if ok else "failed",
    "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "device": device,
    "gpu_mode": {"collected": int(selected), "outcome": describe(marked_status), **marked},
    "unmarked_gpu_coverage": {"outcome": describe(unmarked_status), **unmarked},
}

with open(path, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
    handle.write("\n")

print(json.dumps(report, indent=2))
PY

if [[ "${MARKED_STATUS}" -ne 0 || "${UNMARKED_STATUS}" -ne 0 ]]; then
    echo ""
    echo "FAILED: see ${ARTIFACT} for what failed."
    exit 1
fi

echo ""
echo "PASSED. Artifact written to ${ARTIFACT}."
