# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6

ENV NEMO_REVISION=v2.0.0.rc0.beta

RUN git clone https://github.com/NVIDIA/NeMo.git && \
    cd NeMo && \
    git config user.email "email@email.com" && \
    git config user.name "name name" && \
    git checkout ${NEMO_REVISION} && \
    echo NEMO_COMMIT_HASH=$(git rev-parse HEAD) && \
    bash reinstall.sh

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
ENV PYTHONPATH "${PYTHONPATH}:/workspace/megatron-lm"

ARG TE_REVISION=v1.6rc1
RUN if [ "${TE_REVISION}" != SKIP ]; then \
      NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install --force-reinstall --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@${TE_REVISION} \
    ; fi

ENV MAX_JOBS=4
ARG APEX_REVISION=24.04.01
RUN git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    git checkout ${APEX_REVISION} && \
    CFLAGS="-g0" NVCC_APPEND_FLAGS="--threads 8" pip install -v --no-build-isolation --no-cache-dir --disable-pip-version-check --config-settings "--build-option=--cpp_ext --cuda_ext --bnp --xentropy --deprecated_fused_adam --deprecated_fused_lamb --fast_multihead_attn --distributed_lamb --fast_layer_norm --transducer --distributed_adam --fmha --fast_bottleneck --nccl_p2p --peer_memory --permutation_search --focal_loss --fused_conv_bias_relu --index_mul_2d --cudnn_gbn --group_norm --gpu_direct_storage" . && \
    rm -rf build

# Set working directory
WORKDIR /workspace/sd

# Copy code
COPY . .

# install LDM
RUN pip install -r requirements.txt
