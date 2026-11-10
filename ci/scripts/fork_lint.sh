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

# The fork's Python lint, as its CI runs it (guide 13, plan item 14), over the Python files a change adds or modifies:
# yapf and isort must leave each file as it is, and pylint must report no error. Pylint's other messages are printed
# and do not fail the run, because the pylint the CPU environment resolves is newer than the one the repository's
# configuration was written for and reports conventions that configuration never weighed.
#
# Usage: ci/scripts/fork_lint.sh [BASE]
# BASE is what the change is compared with: a branch, a tag or a commit. It defaults to origin/branch-25.10. Run it
# from the repository's root in the environment ci/scripts/cpu_env/linux-64-cpu.lock describes.

BASE="${1:-origin/branch-25.10}"
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." &> /dev/null && pwd )"

cd "${REPO_ROOT}" || exit 2

MERGE_BASE="$(git merge-base "${BASE}" HEAD)" || { echo "cannot find where HEAD left ${BASE}"; exit 2; }
mapfile -t FILES < <(git diff --name-only --diff-filter=AM "${MERGE_BASE}" HEAD -- '*.py')

if [[ "${#FILES[@]}" == "0" ]]; then
    echo "No Python file added or modified since ${BASE}"
    exit 0
fi

echo "Checking ${#FILES[@]} Python files added or modified since ${BASE}"
FAILED=0

python -m yapf --diff "${FILES[@]}" || { echo ">>>> FAILED: yapf would reformat the files above; run python -m yapf -i on them"; FAILED=1; }

python -m isort --settings-path pyproject.toml --check-only --diff "${FILES[@]}" \
    || { echo ">>>> FAILED: isort would reorder the imports above"; FAILED=1; }

python -m pylint "${FILES[@]}"
python -m pylint --errors-only "${FILES[@]}" || { echo ">>>> FAILED: pylint reports the errors above"; FAILED=1; }

[[ "${FAILED}" == "0" ]] && echo ">>>> PASSED: yapf, isort and pylint"
exit ${FAILED}
