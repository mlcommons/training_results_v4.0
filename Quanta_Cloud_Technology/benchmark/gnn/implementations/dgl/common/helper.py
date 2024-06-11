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

import math
import torch
import transformer_engine.pytorch.cpp_extensions as texcpp
import transformer_engine_extensions as tex
from transformer_engine.pytorch.constants import TE_DType

class BlockWiseRoundRobinSharder():

    def __init__(self, block_size, num_bucket, num_embedding):
        self.block_size = block_size
        self.num_bucket = num_bucket
        # padding
        self.num_embedding = math.ceil(num_embedding / (block_size*num_bucket)) \
            * block_size*num_bucket
        self.block_per_bucket = self.num_embedding // block_size // num_bucket
    
    def get_num_embedding_w_padding(self):
        return self.num_embedding
    
    def map(self, x):
        block_id = x // self.block_size
        bucket_id = block_id % self.num_bucket
        block_offset = block_id // self.num_bucket
        y = (bucket_id*self.block_per_bucket + block_offset) * self.block_size + x % self.block_size
        return y
    

class FP8Helper:
    def __init__(self, device, fp8_format='e4m3', scale=1.0):
        self.device = device
        self.fp8_format = fp8_format
        self.meta = tex.FP8TensorMeta()
        self.meta.scale = torch.ones(1,dtype=torch.float32, device=device) * scale
        self.meta.scale_inv = torch.ones(1, dtype=torch.float32, device=device) / scale
        self.meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device=device)

    def cast_to_fp8_precision(self, x):
        fp8_type = tex.DType.kFloat8E4M3 if self.fp8_format == 'e4m3' else tex.DType.kFloat8E5M2
        input_type = TE_DType[x.dtype]
        x_fp8_precision = texcpp.cast_to_fp8(x, self.meta, tex.FP8FwdTensors.GEMM1_INPUT, fp8_type)
        x_fp8_precision = texcpp.cast_from_fp8(x_fp8_precision, self.meta, 
                                               tex.FP8FwdTensors.GEMM1_INPUT, fp8_type, input_type)
        return x_fp8_precision
    
    def fp8_to_fp16(self, x):
        fp8_type = tex.DType.kFloat8E4M3 if self.fp8_format == 'e4m3' else tex.DType.kFloat8E5M2
        x = x.view(dtype=torch.uint8)

        if x.shape[0] == 0:
            return torch.tensor([], dtype=torch.float16).view(x.shape).to(x.device)
        
        out_type = TE_DType[torch.float16]
        x_fp16 = texcpp.cast_from_fp8(x, self.meta, tex.FP8FwdTensors.GEMM1_INPUT, fp8_type, out_type)
        return x_fp16
    
    def cast_np_to_fp8(self, np_in):
        torch_in = torch.from_numpy(np_in).to(self.device)
        fp8_type = tex.DType.kFloat8E4M3 if self.fp8_format == 'e4m3' else tex.DType.kFloat8E5M2
        torch_out = texcpp.cast_to_fp8(torch_in, self.meta, tex.FP8FwdTensors.GEMM1_INPUT, fp8_type)
        np_out = torch_out.cpu().numpy()
        del torch_out
        del torch_in
        return np_out