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
from dataloading.dgl_dataloader import TorchLoader_DGL
from dataloading.sampler import PyGSampler
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from .overlap_dataloader import PrefetchInterleaver

DGL_AVAILABLE = True
GLT_AVAILABLE = True
PYG_AVAILABLE = True

try:
    import dgl
except ModuleNotFoundError:
    DGL_AVAILABLE = False
    dgl = None

try:
    import graphlearn_torch as glt
except ModuleNotFoundError:
    GLT_AVAILABLE = False
    glt = None

try:
    import torch_geometric as pyg
except ModuleNotFoundError:
    PYG_AVAILABLE = False
    GLT_AVAILABLE = False # GLT is using PyG models
    pyg = None


def check_dgl_available():
    assert DGL_AVAILABLE, "DGL Not available in the container"


def check_glt_available():
    assert GLT_AVAILABLE, "GLT not available in the container"
    assert False, "GLT backend currently not ready"


def check_pyg_available():
    assert PYG_AVAILABLE, "PyG not available in the container"


def build_graph(graph_structure, backend, features):
    
    if backend.lower() == "dgl":
        check_dgl_available()
        
        graph = dgl.heterograph(graph_structure.edge_dict, graph_structure.num_nodes)
        graph.predict = "paper"

        assert features is not None, "Features must not be none!"
        
        for node, d in features.config['nodes'].items():
            if graph.num_nodes(ntype=node) < d['node_count']:
                graph.add_nodes(d['node_count'] - graph.num_nodes(ntype=node), ntype=node)
            else: 
                assert graph.num_nodes(ntype=node) == d['node_count'], f"\
                Graph has more {node} nodes ({graph.num_nodes(ntype=node)}) \
                    than feature shape ({d['node_count']})"

        graph.nodes['paper'].data['label'] = graph_structure.label.to(graph.device)
    
        return graph
    else:
        raise ValueError(f"{backend} backend not supported")


def get_loader(
    graph, index, fanouts, backend,
    use_torch=True, num_sampling_threads=1,
    pyg_style_sampler=True, enable_overlap=False,
    feature_extractor=None, features=None, repeat_input_after=-1, high_priority_embed_stream=True,
    **kwargs):
    if backend.lower() == "dgl":
        check_dgl_available()
        fanouts = [int(fanout) for fanout in fanouts.split(",")]
        if pyg_style_sampler:
            sampler = PyGSampler(fanouts=fanouts, num_threads=num_sampling_threads)
        else:
            sampler = dgl.dataloading.NeighborSampler(fanouts)
        if use_torch:
            return TorchLoader_DGL(graph, index, fanouts, sampler, **kwargs)
        else:
            dgl_dataloader = dgl.dataloading.DataLoader(
                graph, {"paper": index}, 
                sampler,
                **kwargs
            )
            if enable_overlap == False:
                return dgl_dataloader
            else:
                sample_stream = torch.cuda.Stream()
                embed_stream = torch.cuda.Stream(priority = -1 if high_priority_embed_stream else 0)
                device = kwargs.get('device', None)
                overlap_dataloader = PrefetchInterleaver(
                    dgl_dataloader,
                    sample_stream, embed_stream,
                    feature_extractor, device, features, repeat_input_after)
                return overlap_dataloader

    elif backend.lower() == "glt":
        check_glt_available()

    elif backend.lower() == "pyg":
        check_pyg_available()
        fanouts = list(reversed([int(fanout) for fanout in fanouts.split(",")]))
        if use_torch:
            assert False, "Not supported for PyG + separate sampler from dataloader"
        else:
            return pyg.loader.NeighborLoader(
                graph, 
                num_neighbors=fanouts,
                input_nodes=('paper', index),
                **kwargs
            )
    else:
        assert False, "Unrecognized backend " + backend