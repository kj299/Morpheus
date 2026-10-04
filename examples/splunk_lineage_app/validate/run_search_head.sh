#!/usr/bin/env bash
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

# The host side of the search-head run: start the container, wait for it, run run_search_head.py inside it under
# Splunk's own interpreter, and copy the result out to search_head_results.json beside this script. Needs Docker
# and nothing else; no Python on the host, no license file (the image starts under Splunk's built-in trial).
#
#   SPLUNK_PASSWORD='choose-one' examples/splunk_lineage_app/validate/run_search_head.sh
#
# The container is left running so a search can be rerun by hand at http://localhost:8000; `docker compose down -v`
# in this directory removes it and its data.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER=morpheus-lineage-validate

if [[ -z "${SPLUNK_PASSWORD:-}" ]]; then
    echo "Set SPLUNK_PASSWORD to the admin password the container should start with (eight characters or more)."
    exit 2
fi

cd "${HERE}"
SPLUNK_PASSWORD="${SPLUNK_PASSWORD}" docker compose up -d

echo "waiting for the search head to report healthy"
for _ in $(seq 1 120); do
    STATE="$(docker inspect -f '{{.State.Health.Status}}' "${CONTAINER}" 2>/dev/null || echo starting)"
    if [[ "${STATE}" == "healthy" ]]; then
        break
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null)" != "true" ]]; then
        echo "The container stopped. Its log says why:"
        docker logs --tail 40 "${CONTAINER}"
        exit 1
    fi
    sleep 10
done

if [[ "${STATE}" != "healthy" ]]; then
    echo "The search head did not become healthy in twenty minutes; docker logs ${CONTAINER} says why."
    exit 1
fi

set +e
docker exec -u splunk -e SPLUNK_PASSWORD="${SPLUNK_PASSWORD}" "${CONTAINER}" \
    /opt/splunk/bin/splunk cmd python3 /splunk_lineage_app/validate/run_search_head.py
STATUS=$?
set -e

docker cp "${CONTAINER}:/tmp/search_head_results.json" "${HERE}/search_head_results.json"
echo "Copied to ${HERE}/search_head_results.json"
exit "${STATUS}"
