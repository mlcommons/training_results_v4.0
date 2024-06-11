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


export DGXNNODES=1
export DGXNGPU=8
export BATCHSIZE=64
export CONFIG_MAX_STEPS=5000
export INFER_START_STEP=2000
#export CONFIG_MAX_STEPS=1100
#export INFER_START_STEP=300

export WALLTIME=480

# Set clocks and walltime for maxQ and minEDP runs
if [[ "${SET_MAXQ_CLK:-0}" == "1" ]]; then
  export MAXQ_CLK=1050
  WALLTIME_MINUTES=$(expr ${WALLTIME_MINUTES} + ${WALLTIME_MINUTES} / 2) # 50% longer walltime
elif [[ "${SET_MINEDP_CLK:-0}" == "1" ]]; then
  export MINEDP_CLK=1290
  WALLTIME_MINUTES=$(expr ${WALLTIME_MINUTES} + ${WALLTIME_MINUTES} / 3) # 33% longer walltime
fi

# Load default settings
source $(dirname ${BASH_SOURCE[0]})/config_D74H-7U_common.sh

#export NCCL_DEBUG=INFO
#export NCCL_P2P_LEVEL=NVL
#export TORCH_DISTRIBUTED_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=


#For troubleshooting
#export INFER_NUM_IMAGES=3752
#export HYDRA_FULL_ERROR=1
