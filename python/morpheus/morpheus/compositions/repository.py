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
Where the compositions find the repository files they read: the Splunk app's saved searches, the model inventory
and the committed layer 5 models.

Under `tests/` the compositions found these relative to their own file. In the package they may be installed
anywhere, so the repository is located instead: `MORPHEUS_REPOSITORY` when it is set, otherwise the nearest
directory above the working directory, then above this module, that holds the Splunk app. A deployment that ships
the app and the models elsewhere sets the variable. Locating never fails, so importing a
composition does not fail for want of a repository; opening a path that is not there does.
"""

import os

MARKER = os.path.join("examples", "splunk_lineage_app", "TA-morpheus-lineage")
"""A directory only this repository has."""


def _search(start: str):
    current = os.path.abspath(start)

    while True:
        if (os.path.isdir(os.path.join(current, MARKER))):
            return current

        parent = os.path.dirname(current)

        if (parent == current):
            return None

        current = parent


def repository_root() -> str:
    """The repository the compositions read their files from; the working directory when none is found."""
    configured = os.environ.get("MORPHEUS_REPOSITORY")

    if (configured):
        return configured

    return _search(os.getcwd()) or _search(os.path.dirname(os.path.abspath(__file__))) or os.getcwd()


def repository_path(*parts: str) -> str:
    """A path inside the repository."""
    return os.path.join(repository_root(), *parts)
