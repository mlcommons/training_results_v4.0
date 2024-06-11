# Copyright (c) 2018-2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
import argparse
import hashlib
import json
from tqdm import tqdm

def verify_dataset(results_dir):
    with open('checksum.json') as f:
        source = json.load(f)

    assert len(source) == len(os.listdir(results_dir))
    for volume in tqdm(os.listdir(results_dir)):
        with open(os.path.join(results_dir, volume), 'rb') as f:
            data = f.read()
            md5_hash = hashlib.md5(data).hexdigest()
            assert md5_hash == source[volume], f"Invalid hash for {volume}."
    print("Verification completed. All files' checksums are correct.")


if __name__ == '__main__':

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--results_dir', dest='results_dir', required=True)

    args = PARSER.parse_args()
    verify_dataset(args.results_dir)


