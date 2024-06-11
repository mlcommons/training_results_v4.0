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
import numpy as np
import yaml
import sys
sys.path.append('/workspace/gnn')

from common.helper import FP8Helper

GPU_MEM_LIM = 40 * (1024**3) # 40 GB

def convert(orig_feature, helper):
    bucket_size = GPU_MEM_LIM // orig_feature.itemsize

    assert orig_feature.itemsize * bucket_size == GPU_MEM_LIM

    converted = np.zeros(orig_feature.shape, dtype=np.uint8)

    start = 0
    while start < orig_feature.shape[0]:
        end = min(start + bucket_size, orig_feature.shape[0])
        converted[start:end] = helper.cast_np_to_fp8(orig_feature[start:end])
        start = end

    return converted

def main(folder, fp8_format, scale):
    with open(folder+'/config.yml', "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda:0')
    helper = FP8Helper(device=device, fp8_format=fp8_format, scale=scale)

    for node in config['nodes']:
        print(f"Start operating on {node}")
        feat = np.fromfile(
            folder + "/" + config['nodes'][node]['feat_filename'], 
            dtype=np.dtype(config['nodes'][node]['feat_dtype'])
        )

        print(f"    Read to memory completed")

        converted = convert(feat, helper)

        print(f"    FP8 conversion completed")

        with open(folder + "/" + config['nodes'][node]['feat_filename'], "wb") as f:
            converted.tofile(f)

        config['nodes'][node]['feat_dtype'] = "int8"

        del converted
        del feat

        print(f"    Converted data written to disk")

    print(f"Start operating on concatenated features")
    feat = np.fromfile(
        folder + "/" + config['concatenated_features']['path'],
        dtype=np.dtype(config['concatenated_features']['precision'])
    )

    print(f"    Read to memory completed")
    converted = convert(feat, helper)
    print(f"    FP8 conversion completed")
    with open(folder + "/" + config['concatenated_features']['path'], "wb") as f:
        converted.tofile(f)

    config['concatenated_features']['precision'] = 'int8'

    print(f"    Converted data written to disk")

    config['fp8'] = {"scale": scale, "format": fp8_format}

    with open(folder+"/"+"config.yml", "w") as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default="/converted")
    parser.add_argument('--fp8_format', type=str, default="e4m3", choices=['e4m3', 'e5m2'])
    parser.add_argument('--scale', type=float, default=1.0)

    args = parser.parse_args()

    main(folder=args.data_dir, fp8_format=args.fp8_format, scale=args.scale)