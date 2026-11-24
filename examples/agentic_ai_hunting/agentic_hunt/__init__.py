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
Hunting for malicious behavior by and through AI agents: indirect prompt injection, poisoned and shadowed MCP tools,
credential theft, risky commands, runaway agents and agents disabling their own guardrails.

The package needs only the Python standard library, so a hunter can run it anywhere the telemetry is. `hunt.py` is
its command line and `mcp_server.py` serves the same hunts to an MCP client.
"""
