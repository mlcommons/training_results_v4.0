# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import Tensor, HalfTensor, BoolTensor
from typing import Tuple
from model.frozen_bn import FrozenBatchNorm2d
import nvfuser
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
import functools

def partially_contig_tensor(
    fd: "nvfuser.FusionDefinition",
    x: torch.Tensor,
) -> "nvfuser.Tensor":
    return fd.define_tensor(
        sizes=x.size(),
        strides=x.stride(),
        #shape=[-1] * x.ndim,
        #contiguity=nvfuser.compute_contiguity(x.size(), x.stride()),
        dtype=torch_dtype_to_nvfuser_dtype(x.dtype),
    )

permute_in = functools.partial(torch.permute, dims=(0, 2, 3, 1))
permute_out = functools.partial(torch.permute, dims=(0, 3, 1, 2))

# BN-ReLU fusion
class bn_relu_wrapper(FrozenBatchNorm2d):
    def __init__(self, num_features, eps=1e-5, n=None):
        super(bn_relu_wrapper, self).__init__(num_features, eps, n)

    def forward(self, x):
        return bn_relu_jit.apply(x, self.scale, self.bias_term)

class bn_relu_jit(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, input, scale, bias):
        out = fwd_bn_relu_jit(permute_in(input), permute_in(scale), permute_in(bias))
        bn_relu_out = permute_out(out[0])
        relu_mask = permute_out(out[1])

        ctx.save_for_backward(scale, relu_mask)
        return bn_relu_out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        scale, relu_mask = ctx.saved_tensors

        out = bwd_bn_relu_jit(permute_in(grad_output), permute_in(scale), permute_in(relu_mask))
        grad_input = permute_out(out)
        return grad_input, None, None

def fwd_bn_relu_jit(input: HalfTensor, scale: HalfTensor, bias: HalfTensor) -> Tuple[HalfTensor, BoolTensor]:
    tensors = [input, scale, bias]

    with nvfuser.FusionDefinition() as fd:
        x = partially_contig_tensor(fd, tensors[0])
        s = partially_contig_tensor(fd, tensors[1])
        b = partially_contig_tensor(fd, tensors[2])
        z = fd.define_scalar(0)

        T0 = fd.ops.mul(x, s)
        T1 = fd.ops.add(T0, b)
        T2 = fd.ops.relu(T1)
        T3 = fd.ops.cast(T2, dtype=nvfuser.DataType.Half)
        T4 = fd.ops.gt(T1, z)

        fd.add_output(T3)
        fd.add_output(T4)

    bn_relu, relu_mask = fd.execute(tensors)

    return bn_relu, relu_mask

def bwd_bn_relu_jit(grad_output: HalfTensor, scale: HalfTensor, relu_mask: BoolTensor) -> HalfTensor:
    with nvfuser.FusionDefinition() as fd:
        x = partially_contig_tensor(fd, grad_output)
        s = partially_contig_tensor(fd, scale)
        m = partially_contig_tensor(fd, relu_mask)

        T0 = fd.ops.mul(x, s)
        T1 = fd.ops.mul(T0, m)
        T2 = fd.ops.cast(T1, dtype=nvfuser.DataType.Half)

        fd.add_output(T2)

    grad_input = fd.execute([grad_output, scale, relu_mask])[0]

    return grad_input

# BN-Add-ReLU fusion
class bn_add_relu_jit(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, input1, scale1, bias1, input2):
        out = fwd_bn_add_relu_jit(permute_in(input1), permute_in(scale1), permute_in(bias1), permute_in(input2))
        bn_relu_out = permute_out(out[0])
        relu_mask = permute_out(out[1])

        ctx.save_for_backward(scale1, relu_mask)
        return bn_relu_out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        scale, relu_mask = ctx.saved_tensors

        out = bwd_bn_add_relu_jit(permute_in(grad_output), permute_in(scale), permute_in(relu_mask))
        grad_input1 = permute_out(out[0])
        grad_input2 = permute_out(out[1])
        return grad_input1, None, None, grad_input2

def fwd_bn_add_relu_jit(input1: HalfTensor, scale1: HalfTensor, bias1: HalfTensor,
                        input2: HalfTensor) -> Tuple[HalfTensor, BoolTensor]:
    with nvfuser.FusionDefinition() as fd:
        x1 = partially_contig_tensor(fd, input1)
        s1 = partially_contig_tensor(fd, scale1)
        b1 = partially_contig_tensor(fd, bias1)
        x2 = partially_contig_tensor(fd, input2)
        z = fd.define_scalar(0)

        T0 = fd.ops.mul(x1, s1)
        T1 = fd.ops.add(T0, b1)
        T2 = fd.ops.add(T1, x2)
        T3 = fd.ops.relu(T2)
        T4 = fd.ops.cast(T3, dtype=nvfuser.DataType.Half)
        T5 = fd.ops.gt(T2, z)

        fd.add_output(T4)
        fd.add_output(T5)

    bn_add_relu, relu_mask = fd.execute([input1, scale1, bias1, input2])

    return bn_add_relu, relu_mask

def bwd_bn_add_relu_jit(grad_output: HalfTensor, scale: HalfTensor,
                        relu_mask: BoolTensor) -> Tuple[HalfTensor, HalfTensor]:
    with nvfuser.FusionDefinition() as fd:
        x = partially_contig_tensor(fd, grad_output)
        s = partially_contig_tensor(fd, scale)
        m = partially_contig_tensor(fd, relu_mask)

        T0 = fd.ops.mul(x, m)
        T1 = fd.ops.cast(T0, dtype=nvfuser.DataType.Half)
        T2 = fd.ops.mul(T1, s)
        T3 = fd.ops.cast(T2, dtype=nvfuser.DataType.Half)

        fd.add_output(T3)
        fd.add_output(T1)

    grad_input1, grad_input2 = fd.execute([grad_output, scale, relu_mask])

    return grad_input1, grad_input2

# BN-BN-Add-ReLU fusion
class bn_bn_add_relu_jit(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, input1, scale1, bias1, input2, scale2, bias2):
        out = fwd_bn_bn_add_relu_jit(permute_in(input1), permute_in(scale1), permute_in(bias1),
                                     permute_in(input2), permute_in(scale2), permute_in(bias2))
        bn_relu_out = permute_out(out[0])
        relu_mask = permute_out(out[1])

        ctx.save_for_backward(scale1, scale2, relu_mask)
        return bn_relu_out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        scale1, scale2, relu_mask = ctx.saved_tensors

        out = bwd_bn_bn_add_relu_jit(permute_in(grad_output), permute_in(scale1), permute_in(scale2), permute_in(relu_mask))
        grad_input1 = permute_out(out[0])
        grad_input2 = permute_out(out[1])
        return grad_input1, None, None, grad_input2, None, None

def fwd_bn_bn_add_relu_jit(input1: HalfTensor, scale1: HalfTensor, bias1: HalfTensor,
                           input2: HalfTensor, scale2: HalfTensor, bias2: HalfTensor) -> Tuple[HalfTensor, BoolTensor]:
    with nvfuser.FusionDefinition() as fd:
        x1 = partially_contig_tensor(fd, input1)
        s1 = partially_contig_tensor(fd, scale1)
        b1 = partially_contig_tensor(fd, bias1)
        x2 = partially_contig_tensor(fd, input2)
        s2 = partially_contig_tensor(fd, scale2)
        b2 = partially_contig_tensor(fd, bias2)
        z = fd.define_scalar(0)

        T0 = fd.ops.mul(x1, s1)
        T1 = fd.ops.add(T0, b1)
        T2 = fd.ops.mul(x2, s2)
        T3 = fd.ops.add(T2, b2)
        T4 = fd.ops.add(T1, T3)
        T5 = fd.ops.relu(T4)
        T6 = fd.ops.cast(T5, dtype=nvfuser.DataType.Half)
        T7 = fd.ops.gt(T4, z)

        fd.add_output(T6)
        fd.add_output(T7)

    bn_add_relu, relu_mask = fd.execute([input1, scale1, bias1, input2, scale2, bias2])

    return bn_add_relu, relu_mask

def bwd_bn_bn_add_relu_jit(grad_output: HalfTensor, scale1: HalfTensor, scale2: HalfTensor,
                           relu_mask: BoolTensor) -> Tuple[HalfTensor, HalfTensor]:
    with nvfuser.FusionDefinition() as fd:
        x = partially_contig_tensor(fd, grad_output)
        s1 = partially_contig_tensor(fd, scale1)
        s2 = partially_contig_tensor(fd, scale2)
        m = partially_contig_tensor(fd, relu_mask)

        T0 = fd.ops.mul(x, m)
        T1 = fd.ops.mul(T0, s1)
        T2 = fd.ops.cast(T1, dtype=nvfuser.DataType.Half)
        T3 = fd.ops.mul(T0, s2)
        T4 = fd.ops.cast(T3, dtype=nvfuser.DataType.Half)

        fd.add_output(T2)
        fd.add_output(T4)

    grad_input1, grad_input2 = fd.execute([grad_output, scale1, scale2, relu_mask])

    return grad_input1, grad_input2
