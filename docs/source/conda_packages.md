<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Morpheus Conda Packages

> **Note for this fork**: The packages on the `nvidia` channel come from the upstream project. They hold its stages and
> none of this fork's telemetry, lineage or delivery stages. To run this fork, install the locked environment and
> overlay the fork's sources onto it, as [Getting Started with This Fork on a CPU](./fork_getting_started.md)
> describes. The rest of this page describes the upstream packages and is kept for reference.

The Morpheus stages are the building blocks for creating pipelines. Upstream organizes them into libraries by use case, `morpheus-core` and `morpheus-dfp`. This fork keeps the source of `morpheus-core` only: the digital fingerprinting package was removed under ruling D4 of the [repository audit](./developer_guide/guides/13_repository_audit.md), and its source is in the upstream repository at commit [`c6a3b56d`](https://github.com/nv-morpheus/Morpheus/tree/c6a3b56db1c77c2f375ae0833a65a7b6d2272269/python/morpheus_dfp).

The libraries are hosted as Conda packages on the [`nvidia`](https://anaconda.org/nvidia/) channel.

## Morpheus Core
The `morpheus-core` library contains the core stages that are common across all use cases. The Morpheus core library is built from the source code in the `python/morpheus` directory of the Morpheus repository. The core library is installed as a dependency when you install any of the other Morpheus libraries.
To set up a Conda environment with the [`morpheus-core`](https://anaconda.org/nvidia/morpheus-core) library, you can run the following commands:

### Create a Conda Environment
```bash
export CONDA_ENV_NAME=morpheus
conda create -n ${CONDA_ENV_NAME} python=3.12
conda activate ${CONDA_ENV_NAME}
```

### Add Conda Channels
These channels are required for installing the runtime dependencies:
```bash
conda config --env --add channels conda-forge &&\
  conda config --env --add channels nvidia &&\
  conda config --env --add channels rapidsai &&\
  conda config --env --add channels pytorch
```

### Install the `morpheus-core` Library
```bash
conda install -c nvidia morpheus-core=25.10
```
The `morpheus-core` Conda package installs the `morpheus` Python package. It also pulls down all the necessary Conda runtime dependencies for the core stages including [`mrc`](https://anaconda.org/nvidia/mrc) and [`libmrc`](https://anaconda.org/nvidia/libmrc).

### Install Additional PyPI Dependencies
Some of the stages in the core library require additional dependencies that are hosted on PyPI. These dependencies are included as a requirements file in the `morpheus` Python package. The requirements files can be located and installed by running the following command:
```bash
MORPHEUS_CORE_PKG_DIR=$(dirname $(python -c "import morpheus; print(morpheus.__file__)"))
pip install -r ${MORPHEUS_CORE_PKG_DIR}/requirements_morpheus_core_arch-x86_64.txt
```

## Miscellaneous
### Morpheus Examples
The Morpheus examples are not included in the Morpheus Conda packages. To use them, you need to clone the Morpheus repository and run the examples from source. For details, refer to the [Morpheus Examples](./examples.md).
