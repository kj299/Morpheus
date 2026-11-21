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

# Prints the project's version, with its "v" prefix, for the scripts that name a package or an image after it.
#
# Upstream's releases are git tags (v25.10.00), and where the clone has a tag the nearest one is the version, as
# `git describe --tags --abbrev=0` gives it. This fork carries no tags, so a clone of it has nothing to describe:
# the conda build then had no version to give its package and the release image no tag. Without a tag the version
# is the one CMakeLists.txt gives the project, which is what a build of this tree calls itself.
#
#   ./ci/scripts/project_version.sh        prints, for example, v25.10.00

set -eo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/../.." &> /dev/null && pwd )"

TAG="$(git -C "${REPO_ROOT}" describe --tags --abbrev=0 2> /dev/null || true)"

if [[ -n "${TAG}" ]]; then
    echo "${TAG}"
    exit 0
fi

VERSION="$(sed -n '/^project(morpheus/,/)/s/^[[:space:]]*VERSION[[:space:]]\{1,\}\([0-9][0-9.]*\).*/\1/p' "${REPO_ROOT}/CMakeLists.txt")"

if [[ -z "${VERSION}" ]]; then
    echo "project_version.sh: no git tag, and CMakeLists.txt states no project VERSION" >&2
    exit 1
fi

echo "v${VERSION}"
