# syntax=docker/dockerfile:1.3

# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# ============== LEGEND =================================================== #
#       [conda_bld_morpheus_core]
#               /
# [base] -> [conda_env] -> [base_extended] -> [runtime_conda_create] -> [runtime]
#                               \
#                         [conda_env_dev] -> [development]
#                               \
#                         [build_docs]
#                               \
#                         [git_clone]
#
#  base: Contains all system packages common across all environments
#  conda_env: Create morpheus conda environment and set it as the new base
#  base_extended: Optional environmental elements, libraries, packages, etc.
#  conda_env_dev: Full development environment with all necessary conda
#    packages.
#  conda_bld_morpheus_core: Utility stage used to build and cache the
#    morpheus-core conda package for runtime environments. (should not be
#    extended)
#  runtime: Full runtime environment with Morpheus installed as a conda
#    package
#  development: Full development environment able to build against a Morpheus
#    source directory.
#  build_docs: Build the Morpheus documentation
#  git_clone: Clone the Morpheus repository to ensure we do not unintentionally
#    copy local build artifacts
# ========================================================================= #

# ============== BUILDING THIS FORK ======================================= #
#  docker/build_container_dev.sh builds [development]: the environment to
#    compile and test a mounted checkout in. scripts/compile.sh, run inside
#    it, compiles for the card it finds.
#  docker/build_container_release.sh builds [runtime]: morpheus-core
#    installed as a conda package, with the examples, the Splunk app and the
#    model inventory the fork's stages score with.
#
#  No step of an image build has a GPU. The conda package is therefore
#  tested with a smoke test that needs none
#  (ci/conda/recipes/morpheus-core/morpheus_core_test.sh); the test tiers run
#  from a checkout, in CI on a CPU and with ci/scripts/gpu_conformance.sh on
#  a card.
#
#  Build arguments beyond the versions below:
#    CMAKE_CUDA_ARCHITECTURES  what the conda package is compiled for. RAPIDS
#      (the default) is every architecture RAPIDS supports; a package for one
#      card names its architecture, 89 for compute capability 8.9.
#    CONDA_IMAGE  the image conda is copied from. Pinned, so a rebuild starts
#      from the same conda.
# ========================================================================= #

# Args used in FROM commands must come first
ARG FROM_IMAGE="nvidia/cuda"
ARG CUDA_MAJOR_VER=12
ARG CUDA_MINOR_VER=8
ARG CUDA_REV_VER=0
ARG LINUX_DISTRO=ubuntu
ARG LINUX_VER=22.04

# =====Global Options=====
ARG PYTHON_VER=3.12
# Allows running the docker build from a different directory than MORPHEUS_ROOT. Needed for using Morpheus in a
# submodule
ARG MORPHEUS_ROOT_HOST=.
# Supply a channel alias to use for conda. This is needed if the conda channels go down
ARG CONDA_CHANNEL_ALIAS="https://conda.anaconda.org"
# This should reflect the output of the `arch` command.
ARG REAL_ARCH=x86_64
# The image conda is copied from. `condaforge/mambaforge:latest` has resolved to this tag since 2024-11-27, when the
# image stopped being published, so naming the tag changes nothing today and keeps a rebuild from moving if `latest`
# is ever repointed. It carries mamba 1.x, which the `boa` installed below for `conda mambabuild` requires.
ARG CONDA_IMAGE="condaforge/mambaforge:24.9.2-0"

# ============ Stage: conda_source ============
# Where /opt/conda is copied from. A stage of its own because COPY --from takes a stage or a literal image name and
# does not expand a build argument.
FROM --platform=$TARGETPLATFORM ${CONDA_IMAGE} AS conda_source

# ============ Stage: base ============
# Configure the base conda environment
FROM --platform=$TARGETPLATFORM ${FROM_IMAGE}:${CUDA_MAJOR_VER}.${CUDA_MINOR_VER}.${CUDA_REV_VER}-base-${LINUX_DISTRO}${LINUX_VER} AS base

ARG CUDA_MAJOR_VER
ARG CUDA_MINOR_VER
ARG PYTHON_VER
ARG REAL_ARCH

ENV DEBIAN_FRONTEND=noninteractive
ENV REAL_ARCH=${REAL_ARCH}

# Remove the file which breaks apt caching since we will handle it ourselves. See
# https://vsupalov.com/buildkit-cache-mount-dockerfile/
RUN rm -f /etc/apt/apt.conf.d/docker-clean

# Install dependencies to build
RUN --mount=type=cache,id=apt,target=/var/cache/apt \
    apt-get update &&\
    apt-get upgrade -y &&\
    apt-get install --no-install-recommends -y \
        bc \
        build-essential \
        cuda-compiler-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-cudart-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-cupti-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-gdb-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        curl \
        jq \
        libcublas-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcufft-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcurand-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcusolver-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libnuma1 \
        openssh-client \
        pkg-config \
        tar \
        unzip \
        wget \
        zip \
    && rm -rf /var/lib/apt/lists/*

# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

ENV CUDA_PATH=/usr/local/cuda

# All code will be under /workspace
ENV MORPHEUS_ROOT=/workspace
WORKDIR ${MORPHEUS_ROOT}

COPY --from=conda_source /opt/conda /opt/conda

ENV PATH="${PATH}:/opt/conda/bin"

# Install mamba to speed the solve up
# Certificates are verified: every package the image is built from arrives over this configuration.
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    conda config --add pkgs_dirs /opt/conda/pkgs &&\
    # Install mamba, boa and git here. Conda build breaks with other git installs
    /opt/conda/bin/mamba install -y -n base -c conda-forge "boa" "git >=2.35.3" "python=${PYTHON_VER}" "tini=0.19"

# ============ Stage: conda_env ============
# Create the conda environment and install all dependencies
FROM --platform=$TARGETPLATFORM base AS conda_env

ARG MORPHEUS_ROOT_HOST

# Create a base environment
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    # Create the environment and install as little dependencies as possible
    CONDA_ALWAYS_YES=true /opt/conda/bin/mamba create -n morpheus -c conda-forge  \
      python=${PYTHON_VER} && \
    # Activate the environment automatically from .bashrc
    conda init bash && \
    echo "conda activate morpheus" >> ~/.bashrc

# Set the permanent conda channels to use for morpheus
RUN source activate morpheus &&\
    conda config --env --add channels conda-forge &&\
    conda config --env --add channels nvidia &&\
    conda config --env --add channels rapidsai

# Set the entrypoint to use the entrypoint.sh script which sets the conda env
COPY ${MORPHEUS_ROOT_HOST}/docker/entrypoint.sh ./docker/
ENTRYPOINT [ "/opt/conda/bin/tini", "--", "/workspace/docker/entrypoint.sh" ]

# Reset the shell back to normal
SHELL ["/bin/bash", "-c"]

# ============ Stage: conda_bld_morpheus_core ============
# Now build the morpheus conda package
FROM --platform=$TARGETPLATFORM conda_env AS conda_bld_morpheus_core

ARG MORPHEUS_ROOT_HOST
ARG CUDA_MAJOR_VER
ARG CUDA_MINOR_VER

# to upload the image use -
# --build-arg CONDA_TOKEN=<anaconda-token>
# --build-arg CONDA_UPLOAD=upload
ARG CONDA_TOKEN
ARG CONDA_UPLOAD
# What the package is compiled for; see the header.
ARG CMAKE_CUDA_ARCHITECTURES=RAPIDS

ENV GIT_CLONE_PROTECTION_ACTIVE=false

RUN --mount=type=cache,id=apt,target=/var/cache/apt \
    apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt install --no-install-recommends -y \
        cuda-compiler-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-cudart-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-cupti-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-nvml-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-nvtx-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcublas-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcufft-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcurand-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcusolver-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} && \
    rm -rf /var/lib/apt/lists/*

# Copy the source
COPY . ./

RUN --mount=type=cache,id=workspace_cache,target=/workspace/.cache,sharing=locked \
    --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=pip_cache,target=/root/.cache/pip,sharing=locked \
    # Install git-lfs before running the build to avoid errors during conda build
    /opt/conda/bin/mamba install -y -n base -c conda-forge "git-lfs" &&\
    source activate base &&\
    git lfs install &&\
    # Need to get around recent versions of git locking paths until they are deemed safe
    git config --global --add safe.directory "*" &&\
    # Change to the morpheus directory and build the conda package
    cd ${MORPHEUS_ROOT_HOST} &&\
    # The package's version is the nearest tag, or without one the version CMakeLists.txt gives the project
    # (ci/scripts/project_version.sh). Stubs are off because generating them needs a driver.
    MORPHEUS_PYTHON_BUILD_STUBS=OFF CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} CONDA_TOKEN=${CONDA_TOKEN} \
        CONDA_BLD_PATH=/opt/conda/conda-bld ./ci/conda/recipes/run_conda_build.sh morpheus-core ${CONDA_UPLOAD}

# ============ Stage: base_extended ============
# Extension point for optional environmental elements layered over conda_env.
# Empty today; later stages build from it so additions land in one place.
FROM --platform=$TARGETPLATFORM conda_env AS base_extended

# ============ Stage: conda_env_dev ============
# Create the development conda environment
FROM --platform=$TARGETPLATFORM base_extended AS conda_env_dev

ARG MORPHEUS_ROOT_HOST
ARG CUDA_MAJOR_VER
ARG CUDA_MINOR_VER
ARG CONDA_CHANNEL_ALIAS
ARG REAL_ARCH

# Copy the development dependencies file
COPY ${MORPHEUS_ROOT_HOST}/conda/environments/dev_cuda-${CUDA_MAJOR_VER}${CUDA_MINOR_VER}_arch-${REAL_ARCH}.yaml ./conda/environments/dev.yaml

# Update the morpheus environment
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=pip_cache,target=/root/.cache/pip,sharing=locked \
    # Temp add channel_alias to get around conda 404 errors
    conda config --env --set channel_alias ${CONDA_CHANNEL_ALIAS} &&\
    /opt/conda/bin/conda env update --solver=libmamba -n morpheus --file conda/environments/dev.yaml &&\
    # Remove channel_alias to use the normal channel in the container
    conda config --env --remove-key channel_alias

# ============ Stage: build_docs ============
# Now build the morpheus docs
FROM --platform=$TARGETPLATFORM conda_env_dev AS build_docs

ARG MORPHEUS_ROOT_HOST

# Copy the source
COPY . ./

RUN --mount=type=cache,id=workspace_cache,target=/workspace/.cache,sharing=locked \
    --mount=type=bind,from=conda_bld_morpheus_core,source=/opt/conda/conda-bld,target=/opt/conda/conda-bld \
    --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=pip_cache,target=/root/.cache/pip,sharing=locked \
    source activate morpheus &&\
    # The channels the package was built against in run_conda_build.sh, and no others
    CONDA_ALWAYS_YES=true /opt/conda/bin/mamba install -n morpheus \
        -c local \
        -c conda-forge \
        -c rapidsai \
        -c nvidia \
        -c nvidia/label/dev \
        morpheus-core && \
    # Change to the morpheus directory and build the docs, as CI builds them: Sphinx imports the package installed
    # above, so nothing is compiled a second time. The pages land in build/docs/html, where the runtime stage copies
    # them from. Vale is CI's check and is not in this environment.
    cd ${MORPHEUS_ROOT_HOST} &&\
    SKIP_PROSE_CHECK=1 ./ci/scripts/fork_docs.sh

# ============ Stage: runtime_conda_create ============
# Setup conda for the runtime environment
FROM --platform=$TARGETPLATFORM base_extended AS runtime_conda_create
ARG MORPHEUS_ROOT_HOST
ARG CUDA_MAJOR_VER
ARG CUDA_MINOR_VER
ARG REAL_ARCH

COPY "${MORPHEUS_ROOT_HOST}/conda/environments/runtime_cuda-${CUDA_MAJOR_VER}${CUDA_MINOR_VER}_arch-${REAL_ARCH}.yaml" "./conda/environments/"

# Mount the morpheus-core conda package built in `conda_bld_morpheus_core`
RUN --mount=type=bind,from=conda_bld_morpheus_core,source=/opt/conda/conda-bld,target=/opt/conda/conda-bld \
    --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=pip_cache,target=/root/.cache/pip,sharing=locked \
    # CVE-2018-20225 for the base pip, not the env one
    # conda will ignore the request to remove pip
    python -m pip uninstall -y pip && \
    source activate morpheus &&\
    # Install morpheus
    CONDA_ALWAYS_YES=true /opt/conda/bin/mamba install -n morpheus \
        -c local \
        -c conda-forge \
        -c rapidsai \
        -c nvidia \
        -c nvidia/label/dev \
        morpheus-core && \
    /opt/conda/bin/conda env update --solver=libmamba -n morpheus --file \
        conda/environments/runtime_cuda-${CUDA_MAJOR_VER}${CUDA_MINOR_VER}_arch-${REAL_ARCH}.yaml

# ============ Stage: git_clone ============
# Perform a clone of the git repository this ensures that when we copy files from the source repository, we aren't
# unintentionally including build artifacts or other files that shouldn't be in the final image
FROM --platform=$TARGETPLATFORM conda_env_dev AS git_clone

ARG MORPHEUS_ROOT_HOST

# Source the morpheus env to pick up the git-lfs package
RUN --mount=type=bind,source=.,target=/opt/host_repo \
    source activate morpheus &&\
    # Use a local clone to allow unpushed commits to be included
    git clone file:///opt/host_repo/${MORPHEUS_ROOT_HOST} /tmp/morpheus_repo &&\
    cd /tmp/morpheus_repo &&\
    git lfs install &&\
    # The LFS files the runtime image ships: the examples' data, and the model inventory's SID model
    ./scripts/fetch_data.py fetch examples inventory

# ============ Stage: runtime ============
# Setup container for runtime environment
FROM --platform=$TARGETPLATFORM runtime_conda_create AS runtime

ARG MORPHEUS_ROOT_HOST

# Only copy specific files/folders over that are necessary for runtime
COPY --from=git_clone "/tmp/morpheus_repo/conda/environments/*.yaml" "./conda/environments/"
COPY --from=git_clone "/tmp/morpheus_repo/docker" "./docker"
COPY --from=build_docs "/workspace/${MORPHEUS_ROOT_HOST}/build/docs/html" "./docs"
COPY --from=git_clone "/tmp/morpheus_repo/examples" "./examples"
# The two shipped models TC4AbpScoreStage and TC7SidScoreStage score with, and the inventory that pins their digests
COPY --from=git_clone "/tmp/morpheus_repo/models/inventory" "./models/inventory"
COPY --from=git_clone "/tmp/morpheus_repo/scripts" "./scripts"
COPY --from=git_clone "/tmp/morpheus_repo/*.md" "./"
COPY --from=git_clone "/tmp/morpheus_repo/LICENSE" "./"
COPY --from=git_clone "/tmp/morpheus_repo/thirdparty" "./thirdparty"

RUN /opt/conda/bin/conda clean -afy && \
    # Ensure the conda-bld directory is indexed even if empty
    mkdir -p /opt/conda/conda-bld && \
    /opt/conda/bin/conda index /opt/conda/conda-bld

# The compositions read the Splunk app's saved searches, the model inventory and the committed layer 5 models from
# the repository. The package is installed outside it here, so say where those files are.
ENV MORPHEUS_REPOSITORY=${MORPHEUS_ROOT}

# Use morpheus by default
CMD [ "morpheus" ]

# ============ Stage: development ============
# Install and configure development only packages
FROM --platform=$TARGETPLATFORM conda_env_dev AS development

ARG CUDA_MAJOR_VER
ARG CUDA_MINOR_VER

RUN --mount=type=cache,id=apt,target=/var/cache/apt \
    apt update && \
    TZ=Etc/UTC \
    apt install --no-install-recommends -y \
        cuda-compiler-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-cudart-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-cupti-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-nvtx-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcublas-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcufft-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcurand-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcusolver-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        openjdk-11-jre-headless && \
    rm -rf /var/lib/apt/lists/*

# Setup git to allow other users to access /workspace. Requires git 2.35.3 or
# greater. See https://marc.info/?l=git&m=164989570902912&w=2. Only enable for
# development
RUN git config --global --add safe.directory "*"
