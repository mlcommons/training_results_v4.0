# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import subprocess

from huggingface_hub import snapshot_download
from scripts.hash import hash_directory

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/data", type=str, help="Path to the dataset location")
args = parser.parse_args()

snapshot_download(
    "regisss/scrolls_gov_report_preprocessed_mlperf_2",
    local_dir=args.data_dir,
    local_dir_use_symlinks=False,
    repo_type="dataset",
)
subprocess.run(
    f"mv {args.data_dir}/data/* {args.data_dir}/ && rm -rf {args.data_dir}/data {args.data_dir}/README.md",
    shell=True,
    executable="/bin/bash",
)

directory_hash = hash_directory(args.data_dir)
assert directory_hash == "5dc7cdefbd5700e8f2e9d17c2dd29032"
print(f"Succesfully downloaded and verified dataset with hash {directory_hash}")
