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

export DGXNNODES=64
export DGXNGPU=8
export BATCHSIZE=4
export CONFIG_MAX_STEPS=2250
export INFER_START_STEP=1500

export BASE_LR="0.00000008"

export WALLTIME=30

# Set clocks and walltime for maxQ and minEDP runs
if [[ "${SET_MAXQ_CLK:-0}" == "1" ]]; then
  export MAXQ_CLK=1110
  WALLTIME_MINUTES=$(expr ${WALLTIME_MINUTES} + ${WALLTIME_MINUTES} / 2) # 50% longer walltime
elif [[ "${SET_MINEDP_CLK:-0}" == "1" ]]; then
  export MINEDP_CLK=1290
  WALLTIME_MINUTES=$(expr ${WALLTIME_MINUTES} + ${WALLTIME_MINUTES} / 3) # 33% longer walltime
fi

# Load default settings
source $(dirname ${BASH_SOURCE[0]})/config_common.sh
