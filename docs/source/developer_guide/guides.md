<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Developer Guides

## Morpheus Stages

Morpheus includes a number of pre-defined stage implementations to choose from when building a custom
pipeline, each of which can be included and configured to suit your application.

- [List of available Morpheus stages](../stages/morpheus_stages.md)

There are likely going to be situations that require writing a custom stage. The following guide outlines how to
create your own stage in Python, including its constructor arguments and the columns it asks Morpheus to
pre-allocate.

- [Simple Python Stage](./guides/1_simple_python_stage.md)

> **Note**: The code for the above guide can be found in the `examples/developer_guide/1_simple_python_stage` directory of the Morpheus repository.

## Morpheus Control messages

- [Control Messages Overview](./guides/9_control_messages.md)

## Solution Design

Where the guides above cover extending Morpheus, this one covers deploying it against a specific
enterprise problem end to end: behavioral analytics spanning all seven OSI layers, feeding a SIEM, and
producing output a detection engineer can reproduce and defend. It analyzes the codebase, specifies the
telemetry each layer must produce and the detection rules worth writing, works the Splunk side down to
the configuration stanzas, and gives thirteen controls for keeping the output deterministic. Its Part 6
is the ledger of which parts are implemented and which are still design.

- [Predictive Behavioral Analytics Across OSI Layers 1-7](./guides/11_predictive_behavioral_analytics_osi.md)
- [Behavioral Analytics Retrospective](./guides/12_behavioral_analytics_retrospective.md) - What the fork can say today about the behavior of users, systems, network objects, nodes, applications and platforms, scored per entity class, with the verified gaps and the plan that closes them
- [Repository Audit](./guides/13_repository_audit.md) - Every tracked file's purpose, why it is here and its verdict against the objective, the removals that follow and the plan of new approaches for what the objective still needs
