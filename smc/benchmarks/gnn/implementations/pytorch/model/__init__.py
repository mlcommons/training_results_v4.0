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

from model.pyg_model import RGAT_PyG, FeatureExtractor_PyG
from model.dgl_model import RGAT_DGL, FeatureExtractor_DGL

def get_model(backend, gatconv_backend, switches, pad_node_count_to, graph, **model_kwargs):
    if backend.lower() == "dgl":
        check_dgl_available()
        etypes = graph.etypes
        return RGAT_DGL(etypes=etypes, **model_kwargs, gatconv_backend=gatconv_backend, 
                        switches=switches, pad_node_count_to=pad_node_count_to)
    elif backend.lower() in ['pyg', 'glt']:
        check_pyg_available()
        etypes = graph.edge_index_dict.keys()
        return RGAT_PyG(etypes=etypes, **model_kwargs)
    else:
        raise NotImplementedError(f"Backend {backend} not supported")


def get_feature_extractor(backend, formats=None):
    if backend.lower() == "dgl":
        check_dgl_available()
        return FeatureExtractor_DGL(formats=formats)
    elif backend.lower() in ['pyg', "glt"]:
        check_pyg_available()
        return FeatureExtractor_PyG()
    else:
        raise NotImplementedError(f"Backend {backend} not supported")