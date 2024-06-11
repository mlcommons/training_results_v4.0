# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


ARG FROM_IMAGE_NAME=nvcr.io/nvdlfwea/pytorch:24.04-py3
FROM ${FROM_IMAGE_NAME}

# Document build setup
ARG FROM_IMAGE_NAME
ENV CUSTOM_FROM_IMAGE_NAME ${FROM_IMAGE_NAME}

# Custom libraries version
WORKDIR /workspace/

## 1. Apex
# This commit https://github.com/NVIDIA/apex/commit/52e18c894223800cb611682dce27d88050edf1de 
# Introduces changes to apex which make it it incompatible with MegatronDistributedFusedAdam implemented in nemo
# This solves incompatibility issue https://github.com/NVIDIA/NeMo/commit/6a1b27142e44eca6174ee35267a38096fabcca3f
# There is another issue: nemo uses its own implementation of LayerNorm which invokes private apex API which was changed here: https://github.com/NVIDIA/apex/commit/6ff45486f432f91eb86937a0def5eb5f2cf792ae
ARG APEX_REVISION=SKIP
ENV CUSTOM_APEX_REVISION ${APEX_REVISION}
ARG APEX_MAX_JOBS=4

RUN if [ "${APEX_REVISION}" != SKIP ]; then \
      git clone https://github.com/NVIDIA/apex && \
      cd apex && \
      echo APEX_REVISION=${APEX_REVISION} && \
      git checkout ${APEX_REVISION} && \
      echo APEX_COMMIT_HASH=$(git rev-parse HEAD) && \
      MAX_JOBS=${APEX_MAX_JOBS} NVCC_APPEND_FLAGS="--threads 8" pip install -v --no-build-isolation --no-cache-dir --disable-pip-version-check --config-settings "--build-option=--cpp_ext --cuda_ext --bnp --xentropy --deprecated_fused_adam --deprecated_fused_lamb --fast_multihead_attn --distributed_lamb --fast_layer_norm --transducer --distributed_adam --fmha --fast_bottleneck --nccl_p2p --peer_memory --permutation_search --focal_loss --fused_conv_bias_relu --index_mul_2d --cudnn_gbn --group_norm" . \
    ; fi

## 2. Transformer Engine
# the following two variables are needed when building TE
ARG TE_REVISION=release_v1.6
ENV CUSTOM_TE_REVISION ${TE_REVISION}

RUN if [ "${TE_REVISION}" != SKIP ]; then \
      NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install --force-reinstall --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@${TE_REVISION} \
    ; fi

## 3. NeMo
ARG NEMO_REVISION=r2.0.0.rc0.beta
ENV CUSTOM_NEMO_REVISION ${NEMO_REVISION}
ARG NEMO_BASE_VERSION=r2.0.0
ENV CUSTOM_NEMO_BASE_VERSION ${NEMO_BASE_VERSION}

### Base version
RUN if [ "${NEMO_REVISION}" == SKIP ]; then \
      if [ -d /opt/bignlp/NeMo ]; then \
        ln -s /opt/bignlp/NeMo \
      ; else \
        echo "Error: NEMO_REVISION=SKIP but there is no BigNLP NeMo installation in base image." && \
        exit 1 \
      ; fi \
    ; else \
      git clone https://github.com/NVIDIA/NeMo.git && \
      cd NeMo && \
      git config user.email "email@email.com" && \
      git config user.name "name name" && \  
      echo NEMO_REVISION=${NEMO_REVISION} && \
      git checkout ${NEMO_REVISION} && \
      echo NEMO_COMMIT_HASH=$(git rev-parse HEAD) && \
      pip uninstall -y nemo-toolkit && \
      pip install "cython<3.0.0" && \
      pip install --no-build-isolation -e ".[nlp]" \
    ; fi

### Make (has to be called after all changes to repo)
RUN cd NeMo && \
      cd nemo/collections/nlp/data/language_modeling/megatron && \
      make

# 4. Megatron-core
ARG GITLAB_CLONE_ACCESS_TOKEN
ENV GITLAB_CLONE_ACCESS_TOKEN=$GITLAB_CLONE_ACCESS_TOKEN
ARG MEGATRON_REVISION=core_r0.7.0.beta
ENV CUSTOM_MEGATRON_REVISION ${MEGATRON_REVISION}

RUN if [ "${MEGATRON_REVISION}" != SKIP ]; then \
      pip uninstall -y megatron-core && \
      git clone https://github.com/NVIDIA/Megatron-LM.git && \
      cd Megatron-LM && \
      git config user.email "docker@dummy.com" && \
      git config user.name "Docker Build" && \
      git checkout ${CUSTOM_MEGATRON_REVISION} && \
      echo MEGATRON_COMMIT_HASH=$(git rev-parse HEAD) && \
      # layernorm1p fix
      git remote add jbaczek https://github.com/jbaczek/Megatron-LM.git && \                                           
      git fetch jbaczek && \
      git cherry-pick e24f993f3b683a5257adc0271eb08be31342ab46 bba7ba5afb244b2208055e5f3c29c3f53b2b1f4c 0954a2824931b36ca17a48f4e60a6f322a31babf && \ 
      pip install . && \
      cd megatron/core/datasets && \
      make \
    ; fi

ENV PYTHONPATH "${PYTHONPATH}:/workspace/Megatron-LM"

# Pin PL version
# RUN pip install --force-reinstall --no-deps pytorch-lightning==2.0.7

## 5. Benchmark dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

## 6. Use nccl-rdma-sharp-plugins from master to pick a fix after HPCX2.18 release [For 1-SM DP feature]
## TODO: Remove after the next HPCX release.
RUN rm -rf /opt/hpcx/nccl_rdma_sharp_plugin && \
    ln -s /usr/lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/x86_64-linux-gnu/libibverbs.so && \
    git clone https://github.com/Mellanox/nccl-rdma-sharp-plugins && \
    cd nccl-rdma-sharp-plugins/ && \
    ./autogen.sh && \
    ./configure --prefix=/opt/hpcx/nccl_rdma_sharp_plugin --with-cuda=/usr/local/cuda --with-sharp=/opt/hpcx/sharp/ && \
    make -j install && \
    cd ../ && \
    rm -rf nccl-rdma-sharp-plugins/

# Benchmark code
WORKDIR /workspace/llm

COPY . .
ENV PYTHONPATH "/workspace/llm:/workspace/NeMo:${PYTHONPATH}"

