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

# Those variables must be set specifically for each config
: ${DGXNNODES:?DGXNNODES must be set}
: ${DGXNGPU:?DGXNGPU must be set}
: ${BATCHSIZE:?BATCHSIZE must be set}

# Enable offline CLIP text processing
CLIP_TEXT_ENCODER=${CLIP_TEXT_ENCODER:-offline}
if [ "${CLIP_TEXT_ENCODER}" == "offline" ]; then

    # if OFFLINE CLIP is used we force the use of the appropriate config
    if [ -z "${CONFIG_NAME:-}" ]; then
    export CONFIG_NAME="sd2_mlperf_train_moments_encoded"
    else
    echo "WARNING: CONFIG_NAME is explicitly set and CLIP_TEXT_ENCODER=offline. CLIP_TEXT_ENCODER will be ignored."
    fi
fi

# Training knobs
export EXP_NAME=${EXP_NAME:-stable-diffusion2-train-$(date +%y%m%d%H%M%S%N)}
export RANDOM_SEED=${RANDOM_SEED:-$RANDOM}
export CONFIG_PATH=${CONFIG_PATH:-conf}
export CONFIG_NAME=${CONFIG_NAME:-sd2_mlperf_train_moments}
export CONFIG_MAX_STEPS=${CONFIG_MAX_STEPS:-1000}
export INFER_NUM_IMAGES=${INFER_NUM_IMAGES:-30000}
export INFER_START_STEP=${INFER_START_STEP:-0}
export INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-32}
export BASE_LR=${BASE_LR:-"0.0000001"}
export WARMUP_STEPS=${WARMUP_STEPS:-1000}

# Print the variables
echo "EXP_NAME=${EXP_NAME}"
echo "RANDOM_SEED=${RANDOM_SEED}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "CONFIG_MAX_STEPS=${CONFIG_MAX_STEPS}"
echo "INFER_NUM_IMAGES=${INFER_NUM_IMAGES}"
echo "INFER_START_STEP=${INFER_START_STEP}"
echo "INFER_BATCH_SIZE=${INFER_BATCH_SIZE}"
echo "BASE_LR=${BASE_LR}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"

GLOBAL_BATCH_SIZE=$(expr $DGXNNODES \* $DGXNGPU \* $BATCHSIZE)
export LEARNING_RATE=$(echo "$BASE_LR * $GLOBAL_BATCH_SIZE" | bc -l)

# By default we create a checkpoint every 512000 samples (benchmark requirements)
export CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-$(( 512000 / GLOBAL_BATCH_SIZE ))}
echo "CHECKPOINT_STEPS=${CHECKPOINT_STEPS}"

# Performance knobs
export SBATCH_NETWORK=${SBATCH_NETWORK:-sharp}
export FLASH_ATTENTION=${FLASH_ATTENTION:-True}
export USE_DIST_OPTIMIZER=${USE_DIST_OPTIMIZER:-True}

# Runner knobs
export CHECK_COMPLIANCE=${CHECK_COMPLIANCE:-1}
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=$(( ${NEXP:-1} * ${WALLTIME}))

echo "DGXSYSTEM=${DGXSYSTEM}"
