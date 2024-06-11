#!/bin/bash

# Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
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

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

set -x # enables the printed command

# if SLURM_LOCAL_ID is set and non-zero, then we disables printed commands
[ "${SLURM_LOCALID-0}" -ne 0 ] && set +x

set -e # enables exit when error

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# COMMAND LINE VARIABLES
# debug related
TIMETAG=${TIMETAG:-"0"} # output format, does it include time tags?
DEBUG=${DEBUG:-"1"} # Do we include additional debugging outputs? 
CONTINUE_TRAINING=${CONTINUE_TRAINING:-"0"} # should we resume training after we reach target accuracy?

# training configs
EPOCHS=${EPOCHS:-"3"} # Maximum tolerable epochs
LOG_EVERY=${LOG_EVERY:-"1"}

EVAL_FREQUENCY=${EVAL_FREQUENCY:-"0.2"} # performs evaluation once <EVAL_FREQUENCY> x <train_dataset_size> amount of samples are trained
TARGET_ACCURACY=${TARGET_ACCURACY:-"0.72"} # should not touch
VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE:-"4096"} # local batch size for evaluation, should already be tuned

BATCH_SIZE=${BATCH_SIZE:-"512"} # local batch size
MAX_STEPS=${MAX_STEPS:-"-1"}

# Reproduction study
SEED=${SEED:-$RANDOM}

# Dataset related
DATASET_PATH=${DATASET_PATH:-"/data"} # Mapped WholeGraph dataset path
GRAPH_PATH=${GRAPH_PATH:-"/graph"} # mapped graph structure path, for better perf
DATASET_SIZE=${DATASET_SIZE:-"full"}
# num_classes: fixed to "2983"
USE_CONCAT_EMBEDDING=${USE_CONCAT_EMBEDDING:-1}
CONCAT_EMBEDDING_MODE=${CONCAT_EMBEDDING_MODE:-"offline"}
FP8_EMBEDDING=${FP8_EMBEDDING:-"1"}

# WholeGraph configs
WG_SHARDING_LOCATION=${WG_SHARDING_LOCATION:-"cpu"} # --wg_sharding_location, ["cpu", "gpu"]
WG_SHARDING_PARTITION=${WG_SHARDING_PARTITION:-"global"} # --wg_sharding_partition, ["local", "node", "global"]
WG_SHARDING_TYPE=${WG_SHARDING_TYPE:-"continuous"} # --wg_sharding_type, ["continuous", "chunk", "distributed"]
WG_GATHER_SM=${WG_GATHER_SM:-"-1"} # --wg_gather_sm, int
SAMPLING_DEVICE=${SAMPLING_DEVICE:-"cpu"} # --sampling_device, ['cpu', 'cuda']
GRAPH_DEVICE=${GRAPH_DEVICE:-"cpu"} # --graph_device, ['cpu', 'cuda']
GRAPH_SHARDING_PARTITION=${GRAPH_SHARDING_PARTITION:-"node"} # --graph_sharding_partition, ['node', 'global']
NUM_SAMPLING_THREADS=${NUM_SAMPLING_THREADS:-"1"} # --num_sampling_threads, int
NUM_WORKERS=${NUM_WORKERS:-"0"} # --num_workers, int

# model configs
FAN_OUT=${FAN_OUT:-"10,15"} # this also controls the number of layers for the RGAT model
HIDDEN_DIM=${HIDDEN_DIM:-"128"}
NUM_HEADS=${NUM_HEADS:-"4"}
DROPOUT=${DROPOUT:-"0.2"}

# optimizer & training configs
LEARNING_RATE=${LEARNING_RATE:-"0.001"}
DECAY=0 # not finalized in the reference branch

# scheduler configs
SCHED_STEPSIZE=${SCHED_STEPSIZE:-25} # not active
SCHED_GAMMA=${SCHED_GAMMA:-"0.25"} # not active

# perf configs
GATCONV_BACKEND=${GATCONV_BACKEND:-"cugraph"} # --gatconv_backend, ['native', 'cugraph']
CUGRAPH_SWITCHES=${CUGRAPH_SWITCHES:-"0010"} # --cugraph_switches
PAD_NODE_COUNT_TO=${PAD_NODE_COUNT_TO:-"-1"} # --pad_node_count_to, int

# debug configs
REPEAT_INPUT_AFTER=${REPEAT_INPUT_AFTER:-"-1"} # --repeat_input_after, int

CMD_SUFFIX=()
if [ "${TIMETAG:-0}" = "1" ]; then
    CMD_SUFFIX=(${CMD_SUFFIX[@]} --internal_results)
fi

if [ "${DEBUG:-0}" != "1" ]; then
    CMD_SUFFIX=(${CMD_SUFFIX[@]} --no_debug)
fi

if [ "${AMP:-0}" = "1" ]; then
    CMD_SUFFIX=(${CMD_SUFFIX[@]} --amp)
fi

if [ "${DIST_ADAM:-1}" = "1" ]; then
    CMD_SUFFIX=(${CMD_SUFFIX[@]} --dist_adam)
fi

if [ "${CONTINUE_TRAINING:-0}" = "1" ]; then
    CMD_SUFFIX=(${CMD_SUFFIX[@]} --continue_training)
fi

if [ "${DGL_NATIVE_SAMPLER:-0}" = "1" ]; then
    CMD_SUFFIX=(${CMD_SUFFIX[@]} --dgl_native_sampler)
fi

if [ "${TRAIN_OVERLAP:-0}" = "1" ]; then
    CMD_SUFFIX=(${CMD_SUFFIX[@]} --train_overlap)
fi

if [ "${EVAL_OVERLAP:-0}" = "1" ]; then
    CMD_SUFFIX=(${CMD_SUFFIX[@]} --eval_overlap)
fi

if [ "${HIGH_PRIORITY_EMBED_STREAM:-0}" = "1" ]; then
    CMD_SUFFIX=(${CMD_SUFFIX[@]} --high_priority_embed_stream)
fi

if [ "${MAX_STEPS}" -ge 0 ]; then 
    CMD_SUFFIX=(${CMD_SUFFIX[@]} --limit_train_batches $MAX_STEPS --limit_eval_batches $MAX_STEPS)
fi

if [ "${USE_CONCAT_EMBEDDING}" -gt 0 ]; then 
    CMD_SUFFIX=(${CMD_SUFFIX[@]} --concat_embedding_mode $CONCAT_EMBEDDING_MODE)
fi

if [ "${FP8_EMBEDDING}" -gt 0 ]; then
    CMD_SUFFIX=(${CMD_SUFFIX[@]} --fp8_embedding)
fi

echo "RANK ${RANK-}: LOCAL_RANK ${LOCAL_RANK-}, MASTER_ADDR ${MASTER_ADDR-}, MASTER_PORT ${MASTER_PORT-}, WORLD_SIZE ${WORLD_SIZE-}, MLPERF_SLURM_FIRSTNODE ${MLPERF_SLURM_FIRSTNODE-}, SLURM_JOB_ID ${SLURM_JOB_ID-}, SLURM_NTASKS ${SLURM_NTASKS-}, SLURM_PROCID ${SLURM_PROCID-}, SLURM_LOCALID ${SLURM_LOCALID-}, OMP_NUM_THREADS ${OMP_NUM_THREADS-}"

# run benchmark
echo "running benchmark"
DATESTAMP=${DATESTAMP:-$(date +'%y%m%d%H%M%S%N')}

# not set for now. Might use in the future. 
if [[ ${NVTX_FLAG:-0} -gt 0 || ${NSYS_FLAG:-0} -gt 0 ]]; then
  NSYS_OUT="/results/graph_neural_network_pytorch_${DGXNNODES}x${DGXNGPU}x${BATCH_SIZE}_${SLURM_JOBID}_${SLURM_PROCID}_${DATESTAMP}.nsys-rep"
  PRE_CMD=(
    'nsys' 'profile' 
    '--capture-range' 'cudaProfilerApi' 
    '--capture-range-end' 'stop' 
    '--sample=none' 
    '--cpuctxsw=none' 
    '--trace=cuda,nvtx' 
    '--stats' 'true' 
    '--cuda-graph-trace=node' 
    '-f' 'true' 
    '-o' ${NSYS_OUT})
else
  PRE_CMD=()
fi

COMMAND=(
    "train.py"
    --path "${DATASET_PATH}" --graph_load_path $GRAPH_PATH --dataset_size ${DATASET_SIZE} --num_classes 2983
    --wg_sharding_type ${WG_SHARDING_TYPE} --wg_sharding_location ${WG_SHARDING_LOCATION}
    --wg_sharding_partition ${WG_SHARDING_PARTITION}
    --sampling_device ${SAMPLING_DEVICE} --graph_device ${GRAPH_DEVICE} --graph_sharding_partition ${GRAPH_SHARDING_PARTITION}
    --num_sampling_threads ${NUM_SAMPLING_THREADS} --num_workers ${NUM_WORKERS}
    --fan_out ${FAN_OUT} --hidden_channels ${HIDDEN_DIM} --num_heads ${NUM_HEADS} --dropout ${DROPOUT}
    --learning_rate ${LEARNING_RATE} --decay ${DECAY}
    --sched_stepsize ${SCHED_STEPSIZE} --sched_gamma ${SCHED_GAMMA}
    --epochs ${EPOCHS} --log_every ${LOG_EVERY} 
    --eval_frequency ${EVAL_FREQUENCY} --target_accuracy ${TARGET_ACCURACY} 
    --validation_batch_size ${VALIDATION_BATCH_SIZE} --batch_size ${BATCH_SIZE} --seed ${SEED}
    --gatconv_backend ${GATCONV_BACKEND} --cugraph_switches ${CUGRAPH_SWITCHES} --pad_node_count_to ${PAD_NODE_COUNT_TO}
    --wg_gather_sm ${WG_GATHER_SM} --repeat_input_after ${REPEAT_INPUT_AFTER}
    ${CMD_SUFFIX[@]}
)

if [ ! -n "${SLURM_LOCALID-}" ]; then
    # Single-node Docker, we've been launched with `torch_run`
    COMMAND=(torchrun --nnodes=1 --nproc-per-node=${DGXNGPU:-1} ${COMMAND[@]}) # nproc_per_node should be made configurable
else
    COMMAND=(python3 ${COMMAND[@]})
fi

# APILogging only on rank 0
if [ "$LOGGER" = "apiLog.sh" ];
then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
  # Collect logs only on a single rank as a workaround for a NCCL logging bug
  # LOCAL_RANK is set with an enroot hook for Pytorch containers
  # SLURM_LOCALID is set by Slurm
  # OMPI_COMM_WORLD_LOCAL_RANK is set by mpirun
  readonly node_rank="${SLURM_NODEID:-0}"
  readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
  if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ];
  then
    LOGGER=$LOGGER
  else
    LOGGER=""
  fi
fi

${LOGGER:-} ${COMMAND[@]}; ret_code=$?

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="GRAPH_NEURAL_NETWORK"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"
