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

set -euxo pipefail

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${MODEL:?MODEL not set}"

# Vars with defaults
: "${NEXP:=1}"
: "${SEED:=${SEED-$RANDOM}}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=1}"
: "${MLPERF_RULESET:=4.0.0}"
: "${LOGDIR:=./results}"


# Other vars
readonly _config_file="./configs/config_${DGXSYSTEM}.sh"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=llama2_lora
_cont_mounts=("--volume=${DATADIR}:/data" "--volume=${MODEL}:/ckpt" "--volume=${LOGDIR}:/results")


# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DATADIR)
_config_env+=(MODEL)
_config_env+=(DGXSYSTEM)
_config_env+=(SEED)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

cleanup_docker() {
    if docker ps --format '{{.Names}}' | grep -q "^${_cont_name}$"; then
        docker container rm -f "${_cont_name}" || true
    fi
}

cleanup_docker
trap 'set -eux; cleanup_docker' EXIT


# Setup container
docker run --rm --gpus all --init --detach \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --name="${_cont_name}" "${_cont_mounts[@]}" \
    "${CONT}" sleep infinity
# Make sure container has time to finish initialization
sleep 5
docker exec -it "${_cont_name}" true


# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
  (
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
    if [[ $CLEAR_CACHES == 1 ]]; then
      bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
    fi

    docker exec -it ${_config_env[@]} ${_cont_name} ./run_and_time.sh
  ) |& tee "${_logfile_base}_${_experiment_index}.log"

    if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
      docker exec -it "${_config_env[@]}" "${_cont_name}"  \
           python3 -m mlperf_logging.compliance_checker --usage training \
           --ruleset "${MLPERF_RULESET}"                                 \
           --log_output "/results/compliance_${DATESTAMP}.out"           \
           "/results/${DATESTAMP}_${_experiment_index}.log" \
    || true
    fi
done
