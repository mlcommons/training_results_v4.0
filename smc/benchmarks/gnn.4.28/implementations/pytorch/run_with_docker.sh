#!/bin/bash
#SBATCH --job-name graph_neural_networks

# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

set -euxo pipefail

: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATA_DIR:?DATA_DIR not set}"
: "${GRAPH_DIR:?GRAPH_DIR not set}"

: "${MLPERF_RULESET:=4.0.0}"
: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${WORK_DIR:=/workspace/gnn}"
: "${CONTAINER_DATA_DIR:=/data}"
: "${CONTAINER_GRAPH_DIR:=/graph}"

: "${LOGDIR:=${PWD}/results}"
: "${SCRATCH_SPACE:="/raid/scratch"}"

: "${TIME_TAGS:=0}"
: "${DROPCACHE_CMD:="sudo /sbin/sysctl vm.drop_caches=3"}"
CLEAR_CACHES=${CLEAR_CACHES:-1}

# Other vars
readonly _config_file="./config_${DGXSYSTEM}.sh"

: "${MASTER_PORT:=29500}"
export MASTER_PORT

export MASTER_ADDR=$(ip -4 -o addr | egrep -v 'enp|127.0.0.1|docker' | awk '{print $4}' | awk -F / '{print $1}' | tail -n1)
echo "using MASTER_ADDR \"${MASTER_ADDR}\""

export MODEL_NAME="graph_neural_network"
export MODEL_FRAMEWORK="pytorch"
LOGBASE="${DATESTAMP}"
SPREFIX="${MODEL_NAME}_${MODEL_FRAMEWORK}_${DGXNNODES}x${DGXNGPU}x${BATCH_SIZE}_${DATESTAMP}"

if [ ${TIME_TAGS} -gt 0 ]; then
    LOGBASE="${SPREFIX}_mllog"
fi

readonly _logfile_base="${LOGDIR}/${LOGBASE}"
readonly _cont_name="${MODEL_NAME}"
_cont_mounts=(
    "--volume=${DATA_DIR}:${CONTAINER_DATA_DIR}" 
    "--volume=${GRAPH_DIR}:${CONTAINER_GRAPH_DIR}"
    "--volume=${LOGDIR}:/results"
)

# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DATA_DIR)
_config_env+=(GRAPH_DIR)
_config_env+=(DGXSYSTEM)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Cleanup container
cleanup_docker() {
    docker container rm -f "${_cont_name}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

# Setup container
nvidia-docker run --rm --init --detach \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --name="${_cont_name}" "${_cont_mounts[@]}" \
    "${CONT}" sleep infinity
# Make sure container has time to finish initialization
sleep 30
docker exec -it "${_cont_name}" true

# Run experiments
for _experiment_index in $(seq -w 1 "${NEXP}"); do
(

    echo "Beginning trial ${_experiment_index} of ${NEXP}"

    # Clear caches
    if [ "${CLEAR_CACHES}" -eq 1 ]; then
            bash -c "echo -n 'Clearing cache on ' && hostname && sync && ${DROPCACHE_CMD}"
            docker exec -it ${_cont_name} python3 -c "
from utility.logger import mllogger
mllogger.event(key=mllogger.constants.CACHE_CLEAR, value=True)"
        fi

        docker exec -it ${_config_env[@]} ${_cont_name} bash ./run_and_time.sh
    ) |& tee "${_logfile_base}_${_experiment_index}.log"

    if [ "${CHECK_COMPLIANCE:-1}" -eq 1 ]; then
        docker exec -it "${_config_env[@]}" "${_cont_name}"  \
           python3 -m mlperf_logging.compliance_checker --usage training \
           --ruleset "${MLPERF_RULESET}"                                 \
           --log_output "/results/compliance_${DATESTAMP}.out"           \
           "/results/${DATESTAMP}_${_experiment_index}.log" \
        || true
    fi
done
