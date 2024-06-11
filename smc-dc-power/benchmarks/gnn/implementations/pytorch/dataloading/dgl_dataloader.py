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

import torch

class TorchLoader_DGL:
    def __init__(self, graph, index, sampler, **kwargs):
        self.sampler = sampler
        self.dataloader = torch.utils.data.DataLoader(index, **kwargs)
        self.graph = graph

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self
    
    def __next__(self):
        batch = self.sampler.sample(self.graph, {"paper": next(self.iterator)})
        return batch