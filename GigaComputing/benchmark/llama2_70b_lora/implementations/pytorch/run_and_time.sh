#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

: "${LOGGER:=""}"

readonly node_rank="${SLURM_NODEID:-0}"
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"

if [ "$LOGGER" = "apiLog.sh" ];
then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
  if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ];
  then
    LOGGER=$LOGGER
    echo "Using LOGGER=${LOGGER}"
  else
    LOGGER=""
  fi
fi

NSYS_OUT="${NSYS_PREFIX:="lora"}_n${node_rank}_p${local_rank}"
NSYSCMD=""
if [ "${NSYS_FLAG:-0}" -eq 1 ]
then
    NSYSCMD="nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --sample=none --cpuctxsw=none --trace=cuda,nvtx -f true --stats true -o ${NSYS_OUT}"
fi

declare -a CMD

IB_BIND=''
if [[ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 && "${ENABLE_IB_BINDING:-}" == "1" ]]; then
    IB_BIND='--ib=single'
fi

CPU_EXCLUSIVE=''
if [[ "${ENABLE_CPU_EXCLUSIVE:-1}" == "1" ]]; then
    CPU_EXCLUSIVE='--cpu=exclusive'
fi

if [[ -n "${SLURM_LOCALID-}" ]] && [[ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars
    CMD=( 'bindpcie' ${CPU_EXCLUSIVE} ${IB_BIND} '--' ${NSYSCMD} 'python' '-u')
else
    # interactive run on single node, no need to bind
    CMD=( ${NSYSCMD} 'torchrun' '--nproc_per_node=8' )
fi

if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ]
then
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"
fi

${LOGGER:-} ${CMD[@]} train.py; ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ]
then
    # end timing
    end=$(date +%s)
    end_fmt=$(date +%Y-%m-%d\ %r)
    echo "ENDING TIMING RUN AT $end_fmt"
    # report result
    result=$(( $end - $start ))
    result_name="LLM_FINETUNING"
    echo "RESULT,$result_name,,$result,nvidia,$start_fmt"
fi
