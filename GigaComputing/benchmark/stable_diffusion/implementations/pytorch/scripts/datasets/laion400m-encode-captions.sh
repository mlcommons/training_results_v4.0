#!/usr/bin/env bash
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

# Encode the captions in the dataset with moments requires 2.7 TB space in
# total. Please make sure `out_dir` has enough space first.

DGXNNODES=1
DGXNGPU=8
BATCHSIZE=8

torchrun --nnodes=1 --nproc_per_node=${DGXNGPU} \
    encode_captions.py \
    +cluster_type=BCP \
    "trainer.num_nodes=${DGXNNODES}" \
    "trainer.devices=${DGXNGPU}" \
    "model.micro_batch_size=${BATCHSIZE}" \
    "model.global_batch_size=$((DGXNGPU * DGXNNODES * BATCHSIZE))" \
    +out_dir="/datasets/laion-400m/webdataset-moments-filtered-encoded"
