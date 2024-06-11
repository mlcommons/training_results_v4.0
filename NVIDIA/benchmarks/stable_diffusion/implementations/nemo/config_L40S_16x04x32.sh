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

export BATCHSIZE=16
export CONFIG_MAX_STEPS=3500
export INFER_START_STEP=2000

export DGXNGPU=4
export DGXNNODES=16

export WALLTIME=240

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

