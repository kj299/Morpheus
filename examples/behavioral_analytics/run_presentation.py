#!/usr/bin/env python
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
The presentation segment, from collector files to SIEM-ready records. See `segment_runner.py` for the options.

    python examples/behavioral_analytics/run_presentation.py --output out/presentation --state state/presentation
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import segment_runner  # noqa: E402  # pylint: disable=wrong-import-position

if (__name__ == "__main__"):
    sys.exit(segment_runner.main(segment="presentation"))
