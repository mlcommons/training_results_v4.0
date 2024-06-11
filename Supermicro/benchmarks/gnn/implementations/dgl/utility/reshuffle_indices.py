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

import argparse
import torch
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_indices", type=str, default="train_idx.pt")
    parser.add_argument("--path_to_shuffle_map", type=str, default="paper_shuffle_map.pt")
    parser.add_argument("--path_to_output_indices", type=str, default="train_idx_permuted.pt")

    args = parser.parse_args()

    shuffle_map=torch.load(args.path_to_shuffle_map)
    idx=torch.load(args.path_to_indices)
    idx_permuted = shuffle_map[idx]


    torch.save(idx_permuted, args.path_to_output_indices)
