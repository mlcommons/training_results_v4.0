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

# ----------------
# General Libraries
import torch
import os.path as osp
import numpy as np
from common.helper import BlockWiseRoundRobinSharder, FP8Helper

# ----------------
# WholeGraph Libraries and Utils
import yaml
import pylibwholegraph.torch as wgth

def load_config(wholegraph_path):        
    with open(f"{wholegraph_path}/config.yml", "r") as f:
        return yaml.safe_load(f)

def create_wholememory_tensor_from_file(full_filepath, shape, dtype, comm, wmtype, location, sampling_device):
    assert len(shape) == 1 or len(shape) == 2
    last_dim_size = shape[-1] if len(shape) == 2 else 0
    torch_dtype = torch.from_numpy(np.array([], dtype=dtype)).dtype
    wg_tensor = wgth.create_wholememory_tensor_from_filelist(
        comm,
        wmtype,
        location,
        full_filepath,
        torch_dtype,
        last_dim_size,
    )
    assert wg_tensor.shape == shape
    return wg_tensor.get_global_tensor(host_view = sampling_device=='cpu')


class IGBHeteroGraphStructure:
    """
    Synchronously (optionally parallelly) loads the edge relations for IGBH. 
    Current IGBH edge relations are not yet converted to torch tensor. 
    """
    def __init__(
            self,
            config, 
            path, dataset_size, num_classes,

            # WholeGraph related
            # wholegraph is used by default
            wholegraph_comms=None,
            graph_device='cpu',
            sampling_device='cuda',
            graph_sharding_partition='node',
        ):
        
        self.dir = path
        self.dataset_size = dataset_size
        self.label_file = f'node_label_{"19" if num_classes != 2983 else "2K"}.npy'
        self.full_num_trainable_nodes = (227130858 if num_classes != 2983 else 157675969)

        self.config = config

        self.graph_device = graph_device
        self.sampling_device = sampling_device
        self.graph_sharding_partition = graph_sharding_partition

        # This class only stores the edge data, labels, and the train/val/test indices
        assert wholegraph_comms is not None
        self.setup_wholegraph(
            wholegraph_path=path, 
            # WholeGraph path now contains all information needed for loading the dataset.  
            # so we don't need additional path arguments
            wholegraph_comms=wholegraph_comms
        )

        self.edge_dict, self.num_nodes = self.load_edge_dict(path=self.dir)
        self.label = self.load_labels()
        self.train_indices, self.val_indices = self.get_train_val_indices()


    # WholeGraph functions for loading the graph relations 
    def setup_wholegraph(self, wholegraph_path, wholegraph_comms):
        self.node_comm = wholegraph_comms['node']
        self.global_comm = wholegraph_comms['global']

    def load_edge_dict(self, path):

        # graph augmentation (add reverse edges, etc) are done in preprocessing step
        graph_data_dict = {}
        config_nodes = self.config['nodes']
        config_edges = self.config['edges']
        num_nodes_dict = {}
        for node_name, node_config in config_nodes.items():
            num_nodes_dict[node_name] = node_config['node_count']
        for edge_name, edge_config in config_edges.items():

            src_name, rel_name, dst_name = edge_name.split("__")
            graph_format = edge_config['format']
            graph_filenames = edge_config['filenames']
            graph_array_len = edge_config['array_len']
            graph_array_dtype = edge_config['array_dtype']
            graph_data_array = []
            for i in range(len(graph_filenames)):
                file_full_path = osp.join(path, graph_filenames[i])
                if i < 2:
                    # The first tensor is the column index pointer. The second one is the row 
                    # indices
                    graph_data_array.append(
                        create_wholememory_tensor_from_file(
                            file_full_path,
                            (graph_array_len[i],),
                            np.dtype(graph_array_dtype[i]),
                            self.node_comm if self.graph_sharding_partition=='node' \
                                else self.global_comm,
                            'continuous',
                            self.graph_device,
                            self.sampling_device,
                        )
                    )
                else:
                    # The third one is the edge IDs. edge_id is not needed, hence set as empty
                    graph_data_array.append(torch.tensor([], dtype=torch.int64))
            graph_data_dict[(src_name, rel_name, dst_name)] = (graph_format, tuple(graph_data_array))

        return graph_data_dict, num_nodes_dict 
    
    def load_labels(self):
        return torch.from_numpy(np.load(f"{self.dir}/{self.label_file}")).to(torch.long)
        
    def get_train_val_indices(self):
        assert (
            osp.exists(osp.join(self.dir, "train_idx.pt"))
            and
            osp.exists(osp.join(self.dir, "val_idx.pt"))
        ), "Train and val indices not found. Please first run preprocessing.py to ensure the integrity of the dataset."

        return (
            torch.load(osp.join(self.dir, "train_idx.pt")),
            torch.load(osp.join(self.dir, "val_idx.pt"))
        )
        

class Features:
    """
    Lazily initializes the features for IGBH. 

    Features will be initialized only when *build_features* is called. 
    """
    def __init__(
            self, 
            # by default uses WholeGraph
            path, dataset_size, 

            # WholeGraph related
            embedding_tensor_dict,
            wholegraph_comms,
            concat_embedding_mode=None,
            wg_gather_sm=-1,

            fp8_embedding=False,
        ):
        self.path = path
        self.concat_embedding_file_path = None
        self.dataset_size = dataset_size
        self.feature = {}

        self.embedding_tensor_dict = embedding_tensor_dict
        self.wholegraph_comms = wholegraph_comms
        self.config = load_config(self.path)
        self.concat_embedding_mode = concat_embedding_mode

        config_nodes = self.config['nodes']

        if fp8_embedding:
            self.fp8_helper = FP8Helper(device="cuda", scale=self.config['fp8']['scale'], fp8_format=self.config['fp8']['format'])
        else:
            self.fp8_helper = None

        if self.concat_embedding_mode is not None:
            ### Start eligbility check
            list_embedding_tensor_dict = list(self.embedding_tensor_dict.values())
            wg_option = list_embedding_tensor_dict[0]
            for i in range(1, len(list_embedding_tensor_dict)):
                if list_embedding_tensor_dict[i] != wg_option:
                    raise Exception("concat embedding requires all embedding tables to have the \
                                    same WG sharding option!")
            list_config_nodes = list(config_nodes.values())
            feat_dtype = list_config_nodes[0]['feat_dtype']
            feat_dim = list_config_nodes[0]['feat_dim']
            for i in range(1, len(list_config_nodes)):
                if list_config_nodes[i]['feat_dtype'] != feat_dtype:
                    raise Exception("concat embedding requires all embedding tables to have the \
                                    same dtype!")
                if list_config_nodes[i]['feat_dim'] != feat_dim:
                    raise Exception("concat embedding requires all embedding tables to have the \
                                    same embedding width!")
            ### End eligbility check

            feat_comm = self.wholegraph_comms[wg_option['partition']]
            torch_dtype = torch.from_numpy(np.array([], dtype=feat_dtype)).dtype
            self.list_node = []
            list_node_count = []
            self.list_node_file = []
            fixed_node_traversal_order = self.config['concatenated_features']['node_orders']
            for node_name in fixed_node_traversal_order:
                self.list_node.append(node_name)
                list_node_count.append(config_nodes[node_name]['node_count'])
                self.list_node_file.append(config_nodes[node_name]['feat_filename'])
            
            node_counts = torch.tensor(list_node_count).to(torch.int64).to('cuda')
            self.node_offsets = torch.zeros(node_counts.numel() + 1, dtype=torch.int64, device="cuda")
            self.node_offsets[1:] = torch.cumsum(node_counts, dim=0)
            if concat_embedding_mode == 'offline':
                concatenated_config = self.config['concatenated_features']

                block_size = concatenated_config["block_size"]
                num_bucket = concatenated_config['num_buckets']
                self.concat_embedding_file_path = osp.join(self.path, concatenated_config['path'])

                self.sharder = BlockWiseRoundRobinSharder(block_size, num_bucket, 
                                                          self.node_offsets[-1])
                
            num_total_node = concatenated_config['total_number_of_nodes']
            
            node_storage = wgth.create_embedding(
                comm=feat_comm,
                memory_type=wg_option['type'],
                memory_location=wg_option['location'],
                dtype=torch_dtype,
                sizes=(num_total_node, feat_dim),
                gather_sms = wg_gather_sm,
            )
            self.feature = node_storage
        else:    
            for node_name, node_config in config_nodes.items():
                node_option = self.embedding_tensor_dict[node_name] 
                node_count = config_nodes[node_name]['node_count']
                feat_dtype = np.dtype(node_config['feat_dtype'])
                feat_dim = node_config['feat_dim']
                feat_comm = self.wholegraph_comms[node_option['partition']]
                torch_dtype = torch.from_numpy(np.array([], dtype=feat_dtype)).dtype
                node_storage = wgth.create_embedding(
                    comm=feat_comm,
                    memory_type=node_option['type'],
                    memory_location=node_option['location'],
                    dtype=torch_dtype,
                    sizes=(node_count, feat_dim),
                    gather_sms = wg_gather_sm,
                )
                self.feature[node_name] = node_storage

    def warm_up(self):
        if self.concat_embedding_mode is not None:
            indices = torch.zeros((1,), dtype=torch.int64, device='cuda')
            self.feature.gather(indices)
        else:
            for node_name, node_config in self.config['nodes'].items():
                indices = torch.zeros((1,), dtype=torch.int64, device='cuda')
                self.feature[node_name].gather(indices)

    def build_features(self):
        config_nodes = self.config['nodes']
        if self.concat_embedding_mode is not None:
            if self.concat_embedding_mode == 'offline': 
                self.feature.get_embedding_tensor().from_filelist(self.concat_embedding_file_path)
            else:
                list_feat_file_fullpath = [osp.join(self.path, file) for file in self.list_node_file]
                self.feature.get_embedding_tensor().from_filelist(list_feat_file_fullpath)
        else:
            for node_name, node_config in config_nodes.items():
                feat_filename = node_config['feat_filename']
                feat_file_fullpath = osp.join(self.path, feat_filename)
                self.feature[node_name].get_embedding_tensor().from_filelist(feat_file_fullpath)

    def get_input_features(self, input_dict, device):
        list_idx = []
        if self.concat_embedding_mode is not None:
            list_idx = [
                input_dict[ntype] + self.node_offsets[i] for i, ntype in enumerate(self.list_node)
            ]
            concat_idx = torch.concat(list_idx)
            if self.concat_embedding_mode == 'offline':
                concat_idx = self.sharder.map(concat_idx)

            concat_out = self.feature.gather(concat_idx).to(device)
            num_requested_nodes = [idx.size(0) for idx in list_idx]
            tensor_num_requested_nodes = torch.tensor(num_requested_nodes).to(torch.int64)
            idx_offsets = torch.zeros(len(list_idx) + 1, dtype=torch.int64)
            idx_offsets[1:] = torch.cumsum(tensor_num_requested_nodes, dim=0)

            results = {
                key: concat_out[idx_offsets[i]:idx_offsets[i+1]].detach()
                for i, key in enumerate(self.list_node)
            }

            if self.fp8_helper is not None:
                return {
                    node: self.fp8_helper.fp8_to_fp16(embedding)
                    for node, embedding in results.items()
                }
            else:
                return results
        else: 
            results = {
                key: self.feature[key].gather(value).to(device)
                for key, value in input_dict.items()
            }

            if self.fp8_helper is not None:
                return {
                    node: self.fp8_helper.fp8_to_fp16(embedding)
                    for node, embedding in results.items()
                }
            else:
                return results