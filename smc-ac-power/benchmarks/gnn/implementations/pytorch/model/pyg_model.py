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

try:
    import torch_geometric as pyg
except ModuleNotFoundError:
    PYG_AVAILABLE = False
    GLT_AVAILABLE = False # GLT is using PyG models
    pyg = None



class RGAT_PyG(torch.nn.Module):
    r""" [Relational GNN model](https://arxiv.org/abs/1703.06103).

    Args:
        etypes: edge types.
        in_dim: input size.
        h_dim: Dimension of hidden layer.
        out_dim: Output dimension.
        num_layers: Number of conv layers.
        dropout: Dropout probability for hidden layers.
        model: "rsage" or "rgat".
        heads: Number of multi-head-attentions for GAT.
        node_type: The predict node type for node classification.

    """
    def __init__(
            self, 
            etypes, 
            in_feats, h_feats, num_classes, 
            num_layers=2, n_heads=4, dropout=0.2, 
            with_trim=False):
        super().__init__()
        self.lin = torch.nn.Linear(h_feats, num_classes)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_feats if i == 0 else h_feats
            h_dim = h_feats
            self.convs.append(
                pyg.nn.HeteroConv({
                    etype: pyg.nn.GATConv(in_dim, h_dim // n_heads, heads=n_heads, add_self_loops=False)
                    for etype in etypes
                })
            )
        self.dropout = torch.nn.Dropout(dropout)
        self.with_trim = with_trim
        self.layers = self.convs

    def forward(
            self, 
            batch, 
            x_dict, 
        ):
        edge_index_dict = batch.edge_index_dict
        for i, conv in enumerate(self.convs):
            if self.with_trim:
                x_dict, edge_index_dict, _ = pyg.utils.trim_to_layer(
                    layer=i,
                    num_sampled_nodes_per_hop=batch.num_sampled_nodes,
                    num_sampled_edges_per_hop=batch.num_sampled_edges,
                    x=x_dict,
                    edge_index=edge_index_dict
                )
            for key in list(edge_index_dict.keys()):
                if key[0] not in x_dict or key[-1] not in x_dict:
                    del edge_index_dict[key]
                    
            x_dict = conv(x_dict, edge_index_dict)
            if i != len(self.convs) - 1:
                x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
                x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        return self.lin(x_dict["paper"])[:batch['paper'].batch_size]


class FeatureExtractor_PyG:
    def __init__(self):
        pass

    def extract_graph_structure(self, batch, device):
        return batch.to(device)
    
    def extract_inputs_and_outputs(self, sampled_subgraph, device, features):
        batch_size = sampled_subgraph['paper'].batch_size
        if features is None or features.feature == {}:
            batch_inputs = {
                key: value.to(torch.float32)
                for key, value in sampled_subgraph.x_dict.items()
            }
            return sampled_subgraph.x_dict, sampled_subgraph['paper'].y[:batch_size]
        else:
            batch_inputs = features.get_input_features(
                sampled_subgraph.n_id_dict, device
            )
        batch_labels = sampled_subgraph['paper'].y[:batch_size]
        return batch_inputs, batch_labels