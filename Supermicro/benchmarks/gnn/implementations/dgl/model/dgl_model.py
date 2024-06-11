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
import torch.nn.functional as F
import cugraph_dgl
import math
from dgl.nn.pytorch import HeteroGraphConv
from typing import Optional, Union

from cugraph_dgl.nn.conv.base import BaseConv, SparseGraph
import pylibcugraphops.pytorch as ops_torch


try:
    import dgl
except ModuleNotFoundError:
    DGL_AVAILABLE = False
    dgl = None


def glorot(value):
    if isinstance(value, torch.Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


class DGLGATConvCustom(dgl.nn.pytorch.GATConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if hasattr(self, 'fc'):
            glorot(self.fc.weight)
        else:
            glorot(self.fc_src.weight)
            glorot(self.fc_dst.weight)
        glorot(self.attn_l)
        glorot(self.attn_r)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, torch.nn.Linear):
            glorot(self.res_fc.weight)


class cuGraphGATConvCustom(cugraph_dgl.nn.conv.GATConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def reset_parameters(self):
        r"""Reinitialize learnable parameters."""
        if hasattr(self, "lin"):
            glorot(self.lin.weight)
        else:
            glorot(self.lin_src.weight)
            glorot(self.lin_dst.weight)

        glorot(
            self.attn_weights.view(-1, self.num_heads, self.out_feats)
        )
        if self.lin_edge is not None:
            glorot(self.lin_edge.weight)

        if self.lin_res is not None:
            glorot(self.lin_edge.weight)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(
        self,
        g: Union[SparseGraph, dgl.DGLHeteroGraph],
        nfeat: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        efeat: Optional[torch.Tensor] = None,
        max_in_degree: Optional[int] = None,
        deterministic_dgrad: bool = False,
        deterministic_wgrad: bool = False,
        high_precision_dgrad: bool = False,
        high_precision_wgrad: bool = False,
        pad_node_count_to: int = -1,
    ) -> torch.Tensor:
        r"""Forward computation.

        Parameters
        ----------
        graph : DGLGraph or SparseGraph
            The graph.
        nfeat : torch.Tensor or (torch.Tensor, torch.Tensor)
            Node features. If given as a tuple, the two elements correspond to
            the source and destination node features, respectively, in a
            bipartite graph.
        efeat: torch.Tensor, optional
            Optional edge features.
        max_in_degree : int
            Maximum in-degree of destination nodes. When :attr:`g` is generated
            from a neighbor sampler, the value should be set to the corresponding
            :attr:`fanout`. This option is used to invoke the MFG-variant of
            cugraph-ops kernel.
        deterministic_dgrad : bool, default=False
            Optional flag indicating whether the feature gradients
            are computed deterministically using a dedicated workspace buffer.
        deterministic_wgrad: bool, default=False
            Optional flag indicating whether the weight gradients
            are computed deterministically using a dedicated workspace buffer.
        high_precision_dgrad: bool, default=False
            Optional flag indicating whether gradients for inputs in half precision
            are kept in single precision as long as possible and only casted to
            the corresponding input type at the very end.
        high_precision_wgrad: bool, default=False
            Optional flag indicating whether gradients for weights in half precision
            are kept in single precision as long as possible and only casted to
            the corresponding input type at the very end.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where
            :math:`H` is the number of heads, and :math:`D_{out}` is size of
            output feature.
        """
        if isinstance(g, dgl.DGLHeteroGraph):
            if not self.allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    raise dgl.base.DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

        bipartite = isinstance(nfeat, (list, tuple))

        _graph = self.get_cugraph_ops_CSC(
            g, is_bipartite=bipartite, max_in_degree=max_in_degree
        )
        if deterministic_dgrad:
            _graph.add_reverse_graph()

        if bipartite:
            nfeat = (self.feat_drop(nfeat[0]), self.feat_drop(nfeat[1]))
            nfeat_dst_orig = nfeat[1]
        else:
            nfeat = self.feat_drop(nfeat)
            nfeat_dst_orig = nfeat[: g.num_dst_nodes()]

        if efeat is not None:
            if self.lin_edge is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}.edge_feats must be set to "
                    f"accept edge features."
                )
            efeat = self.lin_edge(efeat)

        if bipartite:
            input_src = nfeat[0]
            input_dst = nfeat[1]
            if pad_node_count_to > 0:
                num_src = nfeat[0].shape[0]
                num_dst = nfeat[1].shape[0]
                if num_src < pad_node_count_to:
                    input_src = torch.nn.functional.pad(nfeat[0], (0, 0, 0, pad_node_count_to-num_src))
                if num_dst < pad_node_count_to:
                    input_dst = torch.nn.functional.pad(nfeat[1], (0, 0, 0, pad_node_count_to-num_dst))

            if not hasattr(self, "lin_src"):
                nfeat_src = self.lin(input_src)
                nfeat_dst = self.lin(input_dst)
            else:
                nfeat_src = self.lin_src(input_src)
                nfeat_dst = self.lin_dst(input_dst)
            
            if pad_node_count_to > 0:
                if num_src < pad_node_count_to:
                    nfeat_src = nfeat_src[:num_src]
                if num_dst < pad_node_count_to:
                    nfeat_dst = nfeat_dst[:num_dst]
        else:
            if not hasattr(self, "lin"):
                raise RuntimeError(
                    f"{self.__class__.__name__}.in_feats is expected to be an "
                    f"integer when the graph is not bipartite, "
                    f"but got {self.in_feats}."
                )
            
            if pad_node_count_to > 0:
                num_nodes = nfeat.shape[0]
                if num_nodes < pad_node_count_to:
                    nfeat = torch.nn.functional.pad(nfeat, (0, 0, 0, pad_node_count_to-num_src))
            nfeat = self.lin(nfeat)
            if pad_node_count_to > 0:
                if num_nodes < pad_node_count_to:
                    nfeat = nfeat[:num_nodes]

        out = ops_torch.operators.mha_gat_n2n(
            (nfeat_src, nfeat_dst) if bipartite else nfeat,
            self.attn_weights,
            _graph,
            num_heads=self.num_heads,
            activation="LeakyReLU",
            negative_slope=self.negative_slope,
            concat_heads=self.concat,
            edge_feat=efeat,
            deterministic_dgrad=deterministic_dgrad,
            deterministic_wgrad=deterministic_wgrad,
            high_precision_dgrad=high_precision_dgrad,
            high_precision_wgrad=high_precision_wgrad,
        )[: g.num_dst_nodes()]

        if self.concat:
            out = out.view(-1, self.num_heads, self.out_feats)

        if self.residual:
            res = self.lin_res(nfeat_dst_orig).view(-1, self.num_heads, self.out_feats)
            if not self.concat:
                res = res.mean(dim=1)
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        return out


class RGAT_DGL(torch.nn.Module):
    def __init__(
            self, 
            etypes, 
            in_feats, h_feats, num_classes, 
            num_layers=2, n_heads=4, dropout=0.2,
            gatconv_backend='native', switches=[], pad_node_count_to=-1, with_trim=None):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        if gatconv_backend == 'native':
            GATConv = DGLGATConvCustom
            self.kw_dict = None
        elif gatconv_backend == 'cugraph':
            GATConv = cuGraphGATConvCustom
            self.kw_dict = {etype: {
                'high_precision_wgrad': switches[0] == '1',
                'high_precision_dgrad': switches[1] == '1',
                'deterministic_wgrad': switches[2] == '1',
                'deterministic_dgrad': switches[3] == '1',
                'pad_node_count_to': pad_node_count_to,
            } for etype in etypes}
        else:
            raise NotImplementedError

        
        self.layers.append(HeteroGraphConv({
            etype: GATConv(in_feats, h_feats // n_heads, n_heads)
            for etype in etypes}))
        
        for _ in range(num_layers-2):
            self.layers.append(HeteroGraphConv({
                etype: GATConv(h_feats, h_feats // n_heads, n_heads)
                for etype in etypes}))

        self.layers.append(HeteroGraphConv({
            etype: GATConv(h_feats, h_feats // n_heads, n_heads)
            for etype in etypes}))
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(h_feats, num_classes)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h, mod_kwargs=self.kw_dict)
            h = dgl.apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
            if l != len(self.layers) - 1:
                h = dgl.apply_each(h, F.leaky_relu)
                h = dgl.apply_each(h, self.dropout)
        return self.linear(h['paper'])
    

class FeatureExtractor_DGL:
    def __init__(self, formats=None):
        self.formats = formats

    def extract_graph_structure(self, batch, device):
        if self.formats != None:
            return [block.to(device).formats(self.formats) for block in batch[-1]]
        else:
            return [block.to(device) for block in batch[-1]]

    
    def extract_inputs_and_outputs(self, sampled_subgraph, device, features):
        # input to the batch argument would be a list of blocks
        # the sampled sbgraph is already moved to device in extract_graph_structure
        if features is None or features.feature == {}:
            batch_inputs = sampled_subgraph[0].srcdata['feat']
        else:
            batch_inputs = features.get_input_features(
                sampled_subgraph[0].srcdata[dgl.NID], 
                device
            )
        batch_labels = sampled_subgraph[-1].dstdata['label']['paper']
        return batch_inputs, batch_labels