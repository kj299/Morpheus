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

# The fork's documentation checks, as its CI runs them (guide 13, plan item 14): the prose of every tracked Markdown and
# reStructuredText file through Vale, then the Sphinx build of docs/source. Either failing fails the script.
#
# Run it from the repository's root in the environment ci/scripts/cpu_env/linux-64-docs.lock describes, after
# overlaying the sources (ci/scripts/cpu_env/overlay_fork_sources.py), since Sphinx imports the package to document it.
#
# Upstream built the documentation through CMake, as a target that depends on the compiled package. The fork compiles
# nothing, so this calls sphinx-build directly with the arguments docs/CMakeLists.txt gives it. The build fails on an
# error, and on any warning of the kinds in FATAL_WARNINGS below: a page that links to a page or a heading that does
# not exist, or a page no table of contents reaches. The upstream pages raise other warnings, mostly from generated
# API pages, which are counted in the output and do not fail the build.

set -o pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." &> /dev/null && pwd )"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build/docs}"
FATAL_WARNINGS='toc\.not_included|toc\.not_readable|myst\.xref_missing|ref\.doc|ref\.myst|unknown document|toctree contains reference to nonexisting document|document isn.t included in any toctree|cross-reference target not found'

cd "${REPO_ROOT}" || exit 2

echo "== Vale: the prose of every tracked document"
vale sync > /dev/null || exit 1
# CHANGELOG.md is upstream's and immutable, as ci/scripts/documentation_checks.sh also holds.
git ls-files "*.md" "*.rst" | grep -v -E '^CHANGELOG\.md$' | xargs vale || exit 1

echo "== Sphinx: the documentation build"
mkdir -p "${BUILD_DIR}"
LOG="${BUILD_DIR}/sphinx.log"
(cd docs/source && BUILD_DIR="${BUILD_DIR}" sphinx-build -b html -j auto -T . "${BUILD_DIR}/html") 2>&1 | tee "${LOG}" | tail -n 5
STATUS=${PIPESTATUS[0]}

if [[ "${STATUS}" != "0" ]]; then
    echo ">>>> FAILED: sphinx-build exited ${STATUS}; see ${LOG}"
    exit 1
fi

echo "$(grep -c 'WARNING' "${LOG}") warnings in all"

if grep -E "WARNING.*(${FATAL_WARNINGS})" "${LOG}"; then
    echo ">>>> FAILED: the build has the broken links or unreachable pages above"
    exit 1
fi

echo ">>>> PASSED: documentation built to ${BUILD_DIR}/html"
