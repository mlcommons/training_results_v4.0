#!/bin/bash

# Prerequiste: pip install xpk
# Example:
# WORKLOAD_NAME=${USER}-v5p-1024 SCRIPT=v5p-1024.sh bash xpk_run.sh

SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd )"

set -euox pipefail
PROJECT=${PROJECT:-some-cloud-project-id}
ZONE=${ZONE:-us-east5-c}
CLUSTER_NAME=${CLUSTER_NAME:-mlperf-v5p-1024-us-east5-c}
DEVICE_TYPE=${DEVICE_TYPE:-v5p-1024}
NUM_SLICES=${NUM_SLICES:-1}

# in maxtext repo
# bash docker_build_dependency_image.sh MODE=nightly
BASE_IMAGE=${BASEIMAGE:-gcr.io/cloud-tpu-v2-images/maxtext_base_image-2024_02_28_large}

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
WORKLOAD_NAME=${WORKLOAD_NAME:-${USER}-run}

BASE_OUTPUT_DIRECTORY=${BASE_OUTPUT_DIRECTORY:-"gs://some-bucket"}

# One of v5p-1024.sh, v5p-2048.sh, v5p-3072.sh, v5p-12288.sh or checkpoint-convert.sh
SCRIPT=${SCRIPT:-v5p-1024.sh}

# xpk will pack current directory into a docker image, go to the script directory first
pushd "${SCRIPTS_DIR}"
python ~/xpk/xpk.py workload create --cluster="${CLUSTER_NAME}" --workload="${WORKLOAD_NAME}-${TIMESTAMP}" --device-type="${DEVICE_TYPE}"\
  --num-slices="${NUM_SLICES}" --zone="${ZONE}" --project="${PROJECT}" --base-docker-image "${BASE_IMAGE}"\
  --command "BASE_OUTPUT_DIRECTORY=${BASE_OUTPUT_DIRECTORY} WORKLOAD_NAME=${WORKLOAD_NAME} TIMESTAMP=${TIMESTAMP} USER=${USER} bash ${SCRIPT}"
popd
