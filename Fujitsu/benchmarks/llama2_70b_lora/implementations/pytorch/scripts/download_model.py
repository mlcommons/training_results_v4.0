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

from huggingface_hub import snapshot_download
from scripts.hash import hash_directory

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="/model", type=str, help="Path to the model location")
args = parser.parse_args()

snapshot_download(
    "regisss/llama2-70b-fused-qkv-mlperf", local_dir=args.model_dir, local_dir_use_symlinks=False, max_workers=16
)
directory_hash = hash_directory(args.model_dir)
assert directory_hash == "742093293d1c0c227cfe458365d32ab4"
print(f"Succesfully downloaded and verified model with hash {directory_hash}")
