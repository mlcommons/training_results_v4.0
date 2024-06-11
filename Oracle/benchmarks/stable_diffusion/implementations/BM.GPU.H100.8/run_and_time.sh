#!/bin/bash
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

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HOME=/hf_home

SYNTH_DATA=${SYNTH_DATA:-0}
CONFIG_DATA=""
if [ "${SYNTH_DATA}" -gt 0 ]; then
    CONFIG_DATA+="+model.data.synthetic_data=true"
    CONFIG_DATA+=" +model.data.synthetic_data_length=10000"
    CONFIG_DATA+=" model.data.train.dataset_path=null"
    CONFIG_DATA+=" model.data.webdataset.local_root_path=null"
fi

CAPTURE_CUDAGRAPH_ITERS=${CAPTURE_CUDAGRAPH_ITERS:-"15"}

USE_DIST_OPTIMIZER=${USE_DIST_OPTIMIZER:-"False"}
if [ "${USE_DIST_OPTIMIZER}" = "True" ]; then
    OPTIMIZER_CONF="optim@model.optim=distributed_fused_adam"
else
    OPTIMIZER_CONF="optim@model.optim=megatron_fused_adam"
fi

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# CLEAR YOUR CACHE HERE
python -c "
from mlperf_logging.mllog import constants
from mlperf_logging_utils import mllogger
mllogger.event(key=constants.CACHE_CLEAR, value=True)"

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
    CMD=( 'bindpcie' ${CPU_EXCLUSIVE} ${IB_BIND} '--' 'python' '-u')
else
    # docker or single gpu, no need to bind
    CMD=( 'python' '-u' )
fi

if [ "$LOGGER" = "apiLog.sh" ];
then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
  readonly node_rank="${SLURM_NODEID:-0}"
  readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
  if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ];
  then
    LOGGER=$LOGGER
    echo "Using LOGGER=${LOGGER}"
    echo "###########################################################################################!!!"
  else
    LOGGER=""
  fi
fi

# Assert $RANDOM is usable
if [ -z "$RANDOM" ]; then
    echo "RANDOM is not set!" >&2
    exit 1
fi

echo "RANDOM_SEED=${RANDOM_SEED}"
mkdir -p "/tmp/nemologs"


${LOGGER:-} ${CMD[@]} main.py \
    "trainer.num_nodes=${DGXNNODES}" \
    "trainer.devices=${DGXNGPU}" \
    "trainer.max_steps=${CONFIG_MAX_STEPS}" \
    ${OPTIMIZER_CONF} \
    "model.optim.lr=${LEARNING_RATE}" \
    "model.optim.sched.warmup_steps=${WARMUP_STEPS}" \
    "model.micro_batch_size=${BATCHSIZE}" \
    "model.global_batch_size=$((DGXNGPU * DGXNNODES * BATCHSIZE))" \
    "model.unet_config.use_flash_attention=${FLASH_ATTENTION}" \
    "model.capture_cudagraph_iters=${CAPTURE_CUDAGRAPH_ITERS}" \
    "exp_manager.exp_dir=/tmp/nemologs" \
    "exp_manager.checkpoint_callback_params.every_n_train_steps=${CHECKPOINT_STEPS}" \
    "name=${EXP_NAME}" \
    "model.seed=${RANDOM_SEED}" \
    ${CONFIG_DATA} \
    --config-path "${CONFIG_PATH}" \
    --config-name "${CONFIG_NAME}" || exit 1

readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"

if [ "${SYNTH_DATA}" -eq 0 ]; then
    # Move checkpoints to /nemologs but only on rank 0
    if [ "$SLURM_NODEID" -eq 0 ] && [ "$local_rank" -eq 0 ]; then
        echo "Moving checkpoints to nemologs"
        ls -l /tmp/nemologs
        cp -r /tmp/nemologs/* /nemologs
    fi

    # wait until all checkpoints are copied
    python sync_workers.py
    
    # disable SHARP
    export NCCL_SHARP_DISABLE=1
    export NCCL_COLLNET_ENABLE=0


    CKPT_PATH="/nemologs/${EXP_NAME}/checkpoints/"
    echo "CKPT_PATH=${CKPT_PATH}"

    python infer_and_eval.py \
    "trainer.num_nodes=${DGXNNODES}" \
    "trainer.devices=${DGXNGPU}" \
    "custom.sd_checkpoint_dir=${CKPT_PATH}" \
    "custom.num_prompts=${INFER_NUM_IMAGES}" \
    "custom.infer_start_step=${INFER_START_STEP}" \
    "infer.batch_size=${INFER_BATCH_SIZE}"
fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# runtime
runtime=$(( $end - $start ))
result_name="stable_diffusion"

echo "RESULT,$result_name,$runtime,$USER,$start_fmt"
