..
   SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


.. This role is needed at the index to set the default backtick role
.. role:: py(code)
   :language: python
   :class: highlight

Welcome to Morpheus Documentation
=================================

.. image:: ./img/morpheus-banner.png
  :alt: NVIDIA Morpheus

NVIDIA Morpheus is an open AI application framework that provides cybersecurity developers with a highly optimized AI framework and pre-trained AI capabilities that allow them to instantaneously inspect all IP traffic across their data center fabric. The Morpheus developer framework allows teams to build their own optimized pipelines that address cybersecurity and information security use cases. Bringing a new level of security to data centers, Morpheus provides development capabilities around dynamic protection, real-time telemetry, adaptive policies, and cyber defenses for detecting and remediating cybersecurity threats.

What This Fork Is For
---------------------

This fork extends Morpheus toward one goal: **behavioral analytics that spans all seven OSI layers,
feeds a SIEM, and produces output a detection engineer can reproduce and defend six months later.**

Upstream Morpheus supplies the streaming runtime, the per-entity autoencoder, and the feature DSL. What
it does not supply is the substrate that turns seven layers of telemetry into one story about one
entity: identifiers stable across a replay, attribution from an IP at layer 3 back to a physical port at
layer 1, windows that close on event time rather than on arrival, and the per-layer features the
detection rules need. That substrate is what this fork adds: the lineage stages, the telemetry stages
for TC-0 to TC-7, the determinism controls, and an installable Splunk app.

The design was written down in full before any of it was built. See
:doc:`developer_guide/guides/11_predictive_behavioral_analytics_osi` for the analysis, the per-layer
telemetry requirements, the detection rules, the Splunk configuration, and the determinism controls.
Its Part 6 is the running ledger of what is implemented and what is still design; the short version is
that every layer has running feature stages and thirty-two of the thirty-nine specified rules ship as
saved searches, the per-user model's slot is filled by frozen arithmetic, the collectors are out of
scope, and GPU execution mode has been measured on one machine. :doc:`The retrospective
<developer_guide/guides/12_behavioral_analytics_retrospective>` scores what that adds up to per class of
entity.

Features
--------

 * Built on RAPIDS
    * Built on the RAPIDS™ libraries, Morpheus runs the analysis of logs and telemetry on the GPU, and this fork's
      stages also run on the CPU.
 * Behavioral analytics across OSI layers 1 to 7
    * The fork's telemetry stages turn each layer's records into per-entity features, the detection rules in the
      Splunk app read them, and the lineage stages tie each detection back to the records it came from.
 * Deterministic replay
    * Identifiers, windows and scores depend only on the input, so a replay of the same records produces the same
      detections. The golden tests check this on every change.

Getting Started
---------------

Using Morpheus
^^^^^^^^^^^^^^
 * :doc:`Getting Started with This Fork on a CPU <fork_getting_started>` - From a clone to a running pipeline and passing tests with no GPU and nothing to build
 * :doc:`getting_started` - Using pre-built Docker containers, building Docker containers from source, and fetching models and datasets
 * :doc:`Morpheus Conda Packages <conda_packages>`- Using Morpheus Libraries via the pre-built Conda Packages
 * :doc:`basics/overview` - Brief overview of the command line interface
 * :doc:`basics/building_a_pipeline` - Introduction to building a pipeline using the command line interface
 * :doc:`basics/cpu_only_mode` - Running Morpheus and designing stages for CPU-only execution mode
 * :doc:`Morpheus Examples <examples>` - Example pipelines using both the Python API and command line interface
 * :doc:`Models <models_and_datasets>` - The models the fork's stages score with, and how each score is pinned to its model
 * :doc:`Developer Guides <developer_guide/guides>` - Covers extending Morpheus with custom stages
 * :doc:`Predictive Behavioral Analytics Across OSI Layers 1-7 <developer_guide/guides/11_predictive_behavioral_analytics_osi>` - The design guide this fork implements, and the ledger of what is built
 * :doc:`Behavioral Analytics Retrospective <developer_guide/guides/12_behavioral_analytics_retrospective>` - What the fork can say about each class of entity on a network, the verified gaps, and the plan

Modifying Morpheus
^^^^^^^^^^^^^^^^^^
 * :doc:`Contributing to This Fork <developer_guide/fork_contributing>` - Making a change to the fork and getting it through its CI
 * :doc:`developer_guide/contributing` - Covers building from source, making changes and contributing to Morpheus
 * :doc:`Testing the Kafka Stages <developer_guide/kafka_testing>` - Checking the Kafka source and sink stages by hand against a local broker


.. toctree::
   :caption: Using Morpheus
   :maxdepth: 20
   :hidden:

   fork_getting_started
   getting_started
   conda_packages
   basics/overview
   basics/building_a_pipeline
   basics/cpu_only_mode
   models_and_datasets
   examples/index
   developer_guide/guides/index

.. toctree::
   :caption: Modifying Morpheus
   :maxdepth: 20
   :hidden:

   developer_guide/architecture
   developer_guide/fork_contributing
   developer_guide/contributing
   developer_guide/kafka_testing

.. toctree::
   :caption: API
   :maxdepth: 20
   :hidden:

   py_api
   _lib/index

.. toctree::
   :caption: Morpheus Stages
   :maxdepth: 20
   :hidden:

   stages/morpheus_stages

.. toctree::
   :maxdepth: 20
   :caption: Extra Information
   :hidden:

   extra_info/glossary
   extra_info/troubleshooting
   extra_info/known_issues
   Code of Conduct <https://docs.rapids.ai/resources/conduct/>
   License <https://github.com/nv-morpheus/Morpheus/blob/main/LICENSE>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
