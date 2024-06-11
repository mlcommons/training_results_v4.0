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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="/data/igbh")
    parser.add_argument("--dataset_size", type=str, default="full")

    args = parser.parse_args()
    size = args.dataset_size
    path = args.data_dir

    # check train_idx and val_idx size correctness

    expected_size = {'tiny':100000, 'small':1000000, 'medium':10000000, 'large':100000000, 'full':157675969}[size]
    expected_train_size = int(expected_size * 0.6)
    expected_val_size = int(expected_size * 0.005)

    train_idx = torch.load(f"{path}/{size}/processed/train_idx.pt")
    assert train_idx.shape[0] == expected_train_size, f"Expecting {expected_train_size} train indices, found {train_idx.shape[0]}"

    val_idx = torch.load(f"{path}/{size}/processed/val_idx.pt")
    assert val_idx.shape[0] == expected_val_size, f"Expecting {expected_val_size} val indices, found {val_idx.shape[0]}"
