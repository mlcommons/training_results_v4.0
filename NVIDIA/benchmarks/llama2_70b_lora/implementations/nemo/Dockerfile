# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:24.04-py3
FROM ${FROM_IMAGE_NAME}
WORKDIR /workspace/ft-llm

RUN git config --global user.name "a" && \
    git config --global user.email "a"

## NeMo
ARG NEMO_REVISION=v2.0.0.rc0.beta

RUN git clone https://github.com/NVIDIA/NeMo.git && \
    cd NeMo && \
    echo NEMO_REVISION=${NEMO_REVISION} && \
    git checkout ${NEMO_REVISION} && \
    echo NEMO_COMMIT_HASH=$(git rev-parse HEAD) && \
    pip install --no-build-isolation -e ".[nlp]"

## Megatron-core
ARG MEGATRON_REVISION=core_v0.7.0.beta

RUN pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA/Megatron-LM.git && \
    cd Megatron-LM && \
    echo MEGATRON_REVISION=${MEGATRON_REVISION} && \
    git checkout ${MEGATRON_REVISION} && \
    echo MEGATRON_COMMIT_HASH=$(git rev-parse HEAD) && \
    pip install . && \
    cd megatron/core/datasets && \
    make
ENV PYTHONPATH "${PYTHONPATH}:/workspace/ft-llm/megatron-lm"

ENV NVTE_FRAMEWORK=pytorch

## TransformerEngine
ARG TE_REVISION=v1.6rc2
ENV CUSTOM_TE_REVISION ${TE_REVISION}

RUN if [ "${TE_REVISION}" != SKIP ]; then \
      git clone https://github.com/NVIDIA/TransformerEngine.git && \
      cd TransformerEngine && \
      git submodule init && git submodule update && \
      echo TE_REVISION=${TE_REVISION} && \
      git checkout ${CUSTOM_TE_REVISION} && \
      echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
      NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install --force-reinstall --no-deps . \
    ; fi


ADD . /workspace/ft-llm
RUN pip install -r requirements.txt
