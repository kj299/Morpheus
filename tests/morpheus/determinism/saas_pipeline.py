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
"""
The composition `morpheus.compositions.saas` and its corpus `examples/behavioral_analytics/corpora/saas.py` as one
namespace, under the name the tests have always imported.

Plan section 4.5 moved the composition into the package and the corpus out to the examples. This is read-only: a
test that needs to change what the composition reads patches `morpheus.compositions.saas`.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "examples",
                                "behavioral_analytics"))

# pylint: disable=wrong-import-position
from corpora import merged  # noqa: E402

sys.modules[__name__] = merged("saas")
