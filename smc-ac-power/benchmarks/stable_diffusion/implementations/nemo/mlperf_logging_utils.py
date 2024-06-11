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

from mlperf_common.frameworks.pyt import PyTCommunicationHandler
from mlperf_common.logging import MLLoggerWrapper, constants

mllogger = MLLoggerWrapper(PyTCommunicationHandler(), value=None)


def extract_step_from_ckpt_name(ckpt_name):
    ckpt_name = ckpt_name[ckpt_name.find("-step=") + len("-step=") :]
    ckpt_name = ckpt_name[: ckpt_name.find("-")]
    return int(ckpt_name)


def extract_consumed_samples_from_ckpt_name(ckpt_name):
    ckpt_name = ckpt_name[
        ckpt_name.find("-consumed_samples=") + len("-consumed_samples=") :
    ]
    ckpt_name = ckpt_name[: ckpt_name.find(".")]
    return int(ckpt_name)


def extract_timestamp_from_ckpt_name(ckpt_name):
    ckpt_name = ckpt_name[ckpt_name.find("-timestamp=") + len("-timestamp=") :]
    ckpt_name = ckpt_name[: ckpt_name.find("-")]
    return int(float(ckpt_name))
