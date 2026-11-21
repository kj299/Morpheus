# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# The package's smoke test, run by conda-build in a fresh environment that holds only the package and what it
# requires at run time. It needs no GPU: an image build has none, and the fork's stages all run in CPU mode.
# It is run from the directory conda-build copied the recipe's `source_files` into.

set -euo pipefail

SAMPLE="examples/behavioral_analytics/samples/network/tc3.jsonlines"
OUTPUT="$(mktemp -d)/smoke.jsonlines"

# The command line starts and lists the fork's stages.
# Read into a variable and searched there: `grep -q` on a pipe stops reading at its first match, and under pipefail
# the command it stopped listening to would fail the script.
morpheus --help > /dev/null
STAGES="$(morpheus run pipeline-other --help)"
grep -q "tc3-baseline" <<< "${STAGES}" || { echo "the command line does not list the fork's stages"; exit 1; }

# A CPU pipeline runs end to end through one of the fork's stages and writes every record it read.
morpheus --log_level=WARNING run --use_cpu_only pipeline-other \
    from-file --filename "${SAMPLE}" \
    deserialize \
    lineage-stamp --id_columns src_ip --id_columns dst_ip --id_columns event_time \
    serialize \
    to-file --filename "${OUTPUT}" --overwrite

EXPECTED=$(wc -l < "${SAMPLE}")
WRITTEN=$(wc -l < "${OUTPUT}")

if [[ "${WRITTEN}" != "${EXPECTED}" ]]; then
    echo "the smoke pipeline read ${EXPECTED} records and wrote ${WRITTEN}"
    exit 1
fi

grep -q '"event_uid"' "${OUTPUT}" || { echo "the smoke pipeline's output carries no event_uid"; exit 1; }

echo "morpheus-core smoke test passed: ${WRITTEN} records"
