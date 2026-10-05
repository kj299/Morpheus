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

# The host side of the search-head run: start the container, wait for it, install the app and the validation
# settings, restart, run run_search_head.py inside it under Splunk's own interpreter, and copy the result out to
# search_head_results.json beside this script. Needs Docker and nothing else; no Python on the host, no license
# file (the image starts under Splunk's built-in trial).
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

# Ready means the image's own health check passes, which it does once provisioning has finished. Nothing here waits
# on the splunk command-line client, because a client call that never returns is how the first runs hung: one
# waited an hour on a call with no time limit. The wait reports once a minute so it never looks hung either.
wait_until_healthy() {
    local what="$1"
    local state=""
    echo "waiting for the search head (${what}); a first start provisions for five to ten minutes"
    for attempt in $(seq 1 180); do
        if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null)" != "true" ]]; then
            echo "The container stopped. Its log says why:"
            docker logs --tail 60 "${CONTAINER}"
            exit 1
        fi
        state="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' \
            "${CONTAINER}" 2>/dev/null || echo unknown)"
        if [[ "${state}" == "healthy" ]]; then
            echo "  healthy after about $(( (attempt - 1) * 10 ))s"
            return 0
        fi
        if (( attempt % 6 == 0 )); then
            echo "  still ${state} after $(( attempt * 10 ))s"
            echo "      container log: $(docker logs --tail 1 "${CONTAINER}" 2>&1 | cut -c1-100)"
        fi
        sleep 10
    done
    echo "The search head was not healthy after thirty minutes (last state: ${state}). docker logs ${CONTAINER} says:"
    docker logs --tail 40 "${CONTAINER}"
    exit 1
}

wait_until_healthy "first start"

# Install both apps by copying them in, rather than mounting them: the image changes the owner of everything under
# /opt/splunk/etc as it provisions, and a read-only mount there stops it. A copy is rerun every time, so an edited
# saved search is what gets tested.
docker exec -u root "${CONTAINER}" bash -c '
    set -e
    rm -rf /opt/splunk/etc/apps/TA-morpheus-lineage /opt/splunk/etc/apps/morpheus_validation
    cp -r /splunk_lineage_app/TA-morpheus-lineage /opt/splunk/etc/apps/TA-morpheus-lineage
    cp -r /splunk_lineage_app/validate/validation_app /opt/splunk/etc/apps/morpheus_validation
    chown -R splunk:splunk /opt/splunk/etc/apps/TA-morpheus-lineage /opt/splunk/etc/apps/morpheus_validation'

# Restart the container rather than calling `splunk restart` through docker exec: the restarted splunkd inherits the
# exec session's output and the call need never return. The copied apps live in the container's filesystem and
# survive it; the image provisions again and reports healthy when done.
echo "apps installed; restarting the container so Splunk reads them"
docker restart "${CONTAINER}" >/dev/null
wait_until_healthy "after installing the apps"

# One authenticated call, time-limited, before the run depends on hundreds of them. The admin password is fixed at
# the container's first start, so a different SPLUNK_PASSWORD on a later run cannot log in.
if ! CHECK="$(timeout 120 docker exec -u splunk "${CONTAINER}" /opt/splunk/bin/splunk search "| makeresults" \
        -auth "admin:${SPLUNK_PASSWORD}" </dev/null 2>&1)"; then
    echo "An authenticated search did not succeed. It said:"
    echo "${CHECK:-nothing, and did not return within two minutes}"
    if grep -q "Login failed" <<<"${CHECK}"; then
        echo "The container keeps the password it was first started with. Remove it with 'docker compose down -v'"
        echo "in ${HERE} and run this again with the password you want."
    fi
    exit 1
fi

set +e
timeout 7200 docker exec -u splunk -e SPLUNK_PASSWORD="${SPLUNK_PASSWORD}" "${CONTAINER}" \
    /opt/splunk/bin/splunk cmd python3 /splunk_lineage_app/validate/run_search_head.py </dev/null
STATUS=$?
set -e

docker cp "${CONTAINER}:/tmp/search_head_results.json" "${HERE}/search_head_results.json"
echo "Copied to ${HERE}/search_head_results.json"
exit "${STATUS}"
