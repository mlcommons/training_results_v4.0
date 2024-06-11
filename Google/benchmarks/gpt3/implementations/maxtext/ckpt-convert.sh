#!/bin/bash
set -euox pipefail

BRANCH=${BRANCH:-f586e43f7ee92c701515fe0a2db17dc50f18dc81}

if [[ ! -d "maxtext" ]]; then
  git clone https://github.com/google/maxtext.git
fi

# switch branch
cd maxtext
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"

bash preflight.sh PLATFORM=gke
sleep 60

# flags set as default

# hlo dump
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump_file"

# debug
export TPU_STDERR_LOG_LEVEL=0
export TF_CPP_MIN_LOG_LEVEL=0
export TPU_MIN_LOG_LEVEL=0

BASE_OUTPUT_DIRECTORY=${BASE_OUTPUT_DIRECTORY:-"gs://some-bucket"}

python MaxText/convert_gpt3_ckpt_from_paxml.py \
  --paxml-ckpt-path=gs://mlperf-llm-public2/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000 \
  --maxtext-model-name=gpt3-175b \
  --run-name=convergence_test \
  --base-output-directory="${BASE_OUTPUT_DIRECTORY}"