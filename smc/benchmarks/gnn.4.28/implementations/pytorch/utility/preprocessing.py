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
import os
import numpy as np
from scipy.sparse import coo_matrix
import yaml
import torch
import math
import time
import sys
sys.path.append('/workspace/gnn')
from common.helper import BlockWiseRoundRobinSharder


NODE_COUNTS = {
    "tiny": {'paper': 100000, 'author': 357041, 'institute': 8738, 'fos': 84220, 'journal': 8101, 'conference': 398},
    "small": {'paper': 1000000, 'author': 1926066, 'institute': 14751, 'fos': 190449, 'journal': 15277, 'conference': 1215},
    "medium": {'paper': 10000000, 'author': 15544654, 'institute': 23256, 'fos': 415054, 'journal': 37565, 'conference': 4189},
    "large": {'paper': 100000000, 'author': 116959896, 'institute': 26524, 'fos': 649707, 'journal': 48820, 'conference': 4490},
    "full": {'paper': 269346174, 'author': 277220883, 'institute': 26918, 'fos': 712960, 'journal': 49052, 'conference': 4547},
}

def merge_full_config(default_config, fix_config):
    merged_config = {}
    for key, value in fix_config.items():
        if key not in default_config:
            raise KeyError('key {} in fix_config {}: {} but not in default_config'.format(key, key, value))
    for key, value in default_config.items():
        merged_config[key] = fix_config[key] if key in fix_config else default_config[key]
    return merged_config


def expand_csc(csc_indptr, csc_indices, csc_data):
    current_col_pointer = 0
    expanded_csc_indptr = [current_col_pointer]
    expanded_csc_indices = []

    for i in range(csc_indptr.shape[0]-1):
        ptr_slice = slice(csc_indptr[i], csc_indptr[i+1])
        row_indices = csc_indices[ptr_slice]
        values = csc_data[ptr_slice]

        assert row_indices.shape == values.shape
        for row_index, value in zip(row_indices, values):
            # dense_matrix[row_index, i] = value
            expanded_csc_indices.extend([row_index] * value)
            current_col_pointer += value

        expanded_csc_indptr.append(current_col_pointer)
        
    return (np.array(expanded_csc_indptr), np.array(expanded_csc_indices), np.ones(len(expanded_csc_indices)))


def build_graph_edge(src, dst, nsrc, ndst, graph_format, add_self_loop, add_reverse, need_unique: bool = False):
    if graph_format not in ['coo', 'csr', 'csc']:
        raise ValueError('graph_format should be coo, csr or csc')
    if len(src) != len(dst):
        raise ValueError(
            'convert_coo_to_csr src and dst should have same length, but {} v.s. {}'.format(len(src), len(dst)))
    if add_reverse or add_self_loop:
        assert nsrc == ndst, 'only add reverse or add self loop for same node type'
        assert src.dtype == dst.dtype

    edge_data = {"src": src, "dst": dst}

    if add_self_loop:
        # first remove self loops: 
        mask = edge_data['src'] != edge_data['dst']
        edge_data['src'] = edge_data['src'][mask]
        edge_data['dst'] = edge_data['dst'][mask]

        # and then adds self loops: 
        self_loop_data = np.arange(nsrc, dtype=src.dtype)
        edge_data['src'] = np.concatenate((edge_data['src'], self_loop_data))
        edge_data['dst'] = np.concatenate((edge_data['dst'], self_loop_data))

    if add_reverse:
        edge_data['src'], edge_data['dst'] = np.concatenate((edge_data['src'], edge_data['dst'])), np.concatenate((edge_data['dst'], edge_data['src']))
    
    graph_src = edge_data['src']
    graph_dst = edge_data['dst']

    if graph_format == 'coo':
        return graph_format, (graph_src, graph_dst)
    values = np.ones(len(graph_src), dtype=src.dtype) # each edge will appear once, multiple entries in COO matrix makes duplicate edges.
    coo_graph = coo_matrix((values, (graph_src, graph_dst)), shape=(nsrc, ndst))
    if graph_format == 'csr':
        csr_graph = coo_graph.tocsr()
        return graph_format, (csr_graph.indptr.astype(dtype='int64'), csr_graph.indices.astype(dtype='int64'), csr_graph.data)
    if graph_format == 'csc':
        csc_graph = coo_graph.tocsc()
        csc_ptr, csc_indices, csc_data = expand_csc(csc_graph.indptr, csc_graph.indices, csc_graph.data)
        print(f"WG-preprocessing: edge ID shape {csc_data.shape}; data shape {csc_graph.shape}")
        return graph_format, (csc_ptr.astype(dtype='int64'), csc_indices.astype(dtype='int64'), csc_data.astype(dtype="int64"))
    raise ValueError('graph_format={}'.format(graph_format))


def save_graph_edge(save_path, graph_edge_name, graph_data):
    graph_format, array_data = graph_data
    if graph_format not in ['coo', 'csr', 'csc']:
        raise ValueError('graph_format should be coo, csr or csc')

    def save_array(graph_array, array_suffix):
        array_file_name = '{}____{}'.format(graph_edge_name, array_suffix)
        array_full_path = os.path.join(save_path, array_file_name)
        with open(array_full_path, 'wb') as f:
            graph_array.tofile(f)
        return array_file_name

    filenames = []
    if graph_format == 'coo':
        filename_0 = save_array(array_data[0], 'coo_src_indices')
        filename_1 = save_array(array_data[1], 'coo_dst_indices')
        filenames = [filename_0, filename_1]
    elif graph_format == 'csr':
        filename_0 = save_array(array_data[0], 'csr_indptr')
        filename_1 = save_array(array_data[1], 'csr_indices')
        filename_2 = save_array(array_data[2], 'csr_edgeid')
        filenames = [filename_0, filename_1, filename_2]
    elif graph_format == 'csc':
        filename_0 = save_array(array_data[0], 'csc_indptr')
        filename_1 = save_array(array_data[1], 'csc_indices')
        filename_2 = save_array(array_data[2], 'csc_edgeid')
        filenames = [filename_0, filename_1, filename_2]
    else:
        raise ValueError('graph_format should be coo, csr or csc')
    graph_edge_dict = {
        'format': graph_format,
        'filenames': filenames,
        'array_len': [len(graph_array) for graph_array in array_data],
        'array_dtype': [graph_array.dtype.name for graph_array in array_data],
    }
    return graph_edge_dict


def convert_igbh_dataset(
        root_dir, 
        convert_dir,
        dataset_size="full",
        precision="float16", 
        shuffle_nodes=["author"],
        load_fp16_features=False,
        graph_storage_copy="",
        # concat feature options
        concat_features=True, 
        concat_features_block_size=1,
        concat_features_num_buckets=4096
):
    # make sure that the train/val indices are there
    assert (
        os.path.exists(os.path.join(root_dir, "train_idx.pt")) 
        and 
        os.path.exists(os.path.join(root_dir, "val_idx.pt"))
    ), (
        "Train and validation indices not found. Please run GLT's split_seeds.py first."
    )

    if not os.path.exists(convert_dir):
        print('creating convert dir {}'.format(convert_dir, ))
        os.makedirs(convert_dir)
    
    do_save_copy = (
        graph_storage_copy is not None 
        and
        graph_storage_copy != ""
        and
        graph_storage_copy != convert_dir
    )

    if do_save_copy and not os.path.exists(graph_storage_copy):
        print("creating graph storage copy dir {}".format(graph_storage_copy, ))
        os.makedirs(graph_storage_copy)


    # start converting IGBH dataset
    print(
        'converting IGBH dataset with GLT reverse-edge-policy from {} to {}'.format(root_dir, convert_dir) + 
        ("" if not do_save_copy else f", storing a copy of the graph structure at {graph_storage_copy}")
    )

    # Additionally, the graph is augmented in the following way: 
    #     - for <paper cites paper>, we add reverse edges on top of its original edges, 
    #       which is the same as dgl.add_reverse_edges(graph)
    #     - for all other edges of form <src edge dst>, we introduce reverse edges as a new relation
    #       of form <dst, rev_edge, src>. 
    
    # The following code constructs how the graph should be preprocessed. 
    # For node features: 
    #     - node indices are stored in int64. 
    #     - node features are stored as numpy binary file, in the specified <precision>.
    # For edges: 
    #     each edge is augmented according to the above rule.
    #     the COO -> CSR/CSC conversion optionally occurs at this step. 
    #     and then we save the CSR/CSC format. 

    default_node_config = {
        'feat_dtype': precision,
        'node_id_dtype': 'int64',
        'shuffle': False,
        "node_index_mapping": None
    }
    default_edge_config = {
        'build_forward': True,
        'build_reverse': False,
        'add_self_loop': False,
        'add_reverse_edges': False,
        'forward_sparse_format': 'csc',  # 'coo', 'csr'
        'reverse_sparse_format': 'csc',  # 'coo', 'csr'
    }
    nodes_configs = {
        'paper': {},
        'author': {},
        'institute': {},
        'fos': {},
        'journal': {},
        'conference': {}
    }
    edges_configs = {
        ('paper', 'cites', 'paper'): {},
        ('paper', 'topic', 'fos'): {},
        ('paper', 'written_by', 'author'): {},
        ('author', 'affiliated_to', 'institute'): {},
        ('paper', 'published', 'journal'): {},
        ('paper', 'venue', 'conference'): {},
    }

    edges_configs[('paper', 'cites', 'paper')]['add_self_loop'] = True
    # for all heterogeneous edges, add their reverse edges to the graph as new edges
    # for paper cites paper, add its reverse edge to the same paper cites paper relation
    for (src, edge, dst), config in edges_configs.items():
        if src != dst:
            edges_configs[(src, edge, dst)]['build_reverse'] = True
        else:
            edges_configs[(src, edge, dst)]['add_reverse_edges'] = True

    print("Node configurations: ")
    for key, value in nodes_configs.items():
        value['shuffle'] = (key in shuffle_nodes)
        nodes_configs[key] = merge_full_config(default_node_config, value)
        print('    node_configs[{}]=\n    {}'.format(key, nodes_configs[key]))

    print("Edge configurations: ")
    for key, value in edges_configs.items():
        edges_configs[key] = merge_full_config(default_edge_config, value)
        print('    edges_configs[{}]=\n    {}'.format(key, edges_configs[key]))


    feat_dim = 1024

    concat_feature_path = "concat_features.bin"
    concat_embedding_indices = None
    embed_block_size = concat_features_block_size
    num_embed_bucket = concat_features_num_buckets
    
    number_of_nodes = NODE_COUNTS[dataset_size]
    total_number_of_nodes = sum(number_of_nodes.values())

    if concat_features:
        # preallocate memory to avoid allocating more memory at numpy.concat
        sharder = BlockWiseRoundRobinSharder(embed_block_size, num_embed_bucket, total_number_of_nodes)
        total_number_of_nodes = sharder.get_num_embedding_w_padding()
        concat_embeddings = np.zeros([total_number_of_nodes, feat_dim], dtype=np.dtype(precision))
        concat_embedding_indices = torch.arange(total_number_of_nodes, dtype=torch.int64)
        offset = 0
        for node in nodes_configs:
            nnodes = number_of_nodes[node]
            start = offset
            end = offset + nnodes
            offset = end
            node_indices = None
            if end == total_number_of_nodes:
                node_indices = concat_embedding_indices[start:]
            elif end < total_number_of_nodes:
                node_indices = concat_embedding_indices[start:end]
            else:
                assert False, f"Node start & end should align with total number of nodes, but got {end} > {total_number_of_nodes}"

            nodes_configs[node]['concat_node_index_mapping'] = sharder.map(node_indices)

    # Before we operate on every node type's feature, we save the paper labels. 
    # Previous preprocessing code only stores the node labels together with the WholeGraph dataset 
    # for IGBH-Large & IGBH-Full. 

    loader = lambda num_classes, src_directory, dataset_size: np.load(
        f"{src_directory}/paper/node_label_{num_classes}.npy"
    )
    if dataset_size in ['large', 'full']:
        loader = lambda num_classes, src_directory, dataset_size: np.memmap(
            f"{src_directory}/paper/node_label_{num_classes}.npy",
            dtype="float32", mode="r", shape=(269346174 if dataset_size == "full" else 100000000)
        )

    for n_classes in ['19', '2K']:
        print(f"    Start to load node label {n_classes} for {dataset_size}")
        paper_node_labels = loader(n_classes, root_dir, dataset_size)
        if  nodes_configs['paper']['node_index_mapping'] is None:
            if nodes_configs['paper']['shuffle']:
                nodes_configs['paper']['node_index_mapping'] = torch.randperm(paper_node_labels.shape[0])
            else:
                nodes_configs['paper']['node_index_mapping'] = torch.arange(paper_node_labels.shape[0])

        assert nodes_configs['paper']['node_index_mapping'] is not None

        reverse_indices = nodes_configs['paper']['node_index_mapping'].sort(dim=0).indices
        mapped_node_labels = paper_node_labels[reverse_indices]
        
        np.save(f"{convert_dir}/node_label_{n_classes}.npy", mapped_node_labels)
        print(f"    Save complete for node label {n_classes} for {dataset_size}")

        if do_save_copy:
            np.save(f"{graph_storage_copy}/node_label_{n_classes}.npy", mapped_node_labels)
            print(f"    Saved node label {n_classes} copy for {dataset_size}")

    print(f"    Start to save the train / val indices")
    # convert original index to new index 
    torch.save(
        nodes_configs['paper']['node_index_mapping'][
            torch.load(os.path.join(root_dir, "train_idx.pt"))
        ], os.path.join(convert_dir, "train_idx.pt"))
    torch.save(
        nodes_configs['paper']['node_index_mapping'][
            torch.load(os.path.join(root_dir, "val_idx.pt"))
        ], os.path.join(convert_dir, "val_idx.pt"))
    print(f"    train / val indices save complete")

    if do_save_copy:
        # convert original index to new index 
        torch.save(
            nodes_configs['paper']['node_index_mapping'][
                torch.load(os.path.join(root_dir, "train_idx.pt"))
            ], os.path.join(graph_storage_copy, "train_idx.pt"))
        torch.save(
            nodes_configs['paper']['node_index_mapping'][
                torch.load(os.path.join(root_dir, "val_idx.pt"))
            ], os.path.join(graph_storage_copy, "val_idx.pt"))
        print(f"    Saved a copy of the indices")

    # operate on every node type's feature
    nodes_feat_dict = {}
    node_orders = []
    for node_name, node_config in nodes_configs.items():
        # feature loading
        # special case for loading author/paper feature for IGBH-Large & IGBH-Full
        print(f"Operating on node feature {node_name}")
        print(f"    Start loading the numpy node feature {node_name}")
        load_start = time.time()
        if not load_fp16_features:
            if dataset_size in ['large', 'full'] and node_name in ['paper', 'author']:
                dict_num_rows = {'large': {'paper': 100000000, 'author': 116959896},
                                'full': {'paper': 269346174, 'author': 277220883}}
                node_feat = np.memmap(
                    os.path.join(
                        root_dir, 
                        node_name, 
                        'node_feat.npy'
                    ), 
                    dtype='float32', mode='r',  
                    shape=(dict_num_rows[dataset_size][node_name], feat_dim)
                )

            else:
                node_feat = np.load(os.path.join(root_dir, node_name, 'node_feat.npy'))
        else:
            node_feat = torch.load(os.path.join(root_dir, node_name, "node_feat_fp16.pt")).numpy()

        print(f"    Loading feature done for {node_name}, time used {time.time() - load_start}")

        # conversion and export to npy
        target_dtype = np.dtype(node_config['feat_dtype'])
        assert len(node_feat.shape) == 2
        if node_feat.dtype != target_dtype:
            print('    Converting {} to {} type'.format(node_name, target_dtype))
            output_node_feat = node_feat.astype(target_dtype)
        else:
            output_node_feat = node_feat

        # Either assigns a random shuffled index mapping to the node
        # or assigns an ordered index mapping to the node
        # regardless of whether we concatenate the feature
        nnodes = node_feat.shape[0]
        if node_config['node_index_mapping'] is not None:
            assert nnodes == node_config['node_index_mapping'].shape[0], f"Node {node_name} having inconsistent node index mapping shape {node_config['node_index_mapping'].shape}, expecting {nnodes}."
        else:
            if node_config['shuffle']: 
                node_config['node_index_mapping'] = torch.randperm(nnodes)
            else:
                node_config['node_index_mapping'] = torch.arange(nnodes)

        
        reverse_indices = node_config['node_index_mapping'].sort(dim=0).indices
        output_node_feat = output_node_feat[reverse_indices]

        output_filename = '{}_node_feat.bin'.format(node_name)
        output_full_path = os.path.join(convert_dir, output_filename)
        feat_map_filename = f"{node_name}_shuffle_map.pt"

        node_orders.append(node_name)
        assert node_config['concat_node_index_mapping'] is not None
        concat_embeddings[node_config['concat_node_index_mapping']] = output_node_feat

        nodes_feat_dict[node_name] = {
            'node_count': output_node_feat.shape[0],
            'feat_dtype': node_config['feat_dtype'],
            'feat_filename': output_filename,
            'feat_dim': output_node_feat.shape[1],
            'feat_map_filename': feat_map_filename
        }

        print('    Writing node_feat for {}, dict={}'.format(node_name, nodes_feat_dict[node_name]))
        with open(output_full_path, 'wb') as f:
            output_node_feat.tofile(f)

        torch.save(node_config['node_index_mapping'], os.path.join(convert_dir, feat_map_filename))
        del node_feat
        del output_node_feat
        print('Writing / concatenating node_feat for {} done'.format(node_name, ))

    if concat_features:
        output_full_path = os.path.join(convert_dir, concat_feature_path)
        print('    Writing concatenated features to disk')
        with open(output_full_path, "wb") as f:
            concat_embeddings.tofile(f)
        print("Writing concat node feature done.")

    # operate on every edge index
    edge_graph_dict = {}
    for (edge_src_name, edge_type_name, edge_dst_name), edge_config in edges_configs.items():
        edge_name = '{}__{}__{}'.format(edge_src_name, edge_type_name, edge_dst_name)
        reversed_edge_name = '{}__reverse_{}__{}'.format(edge_dst_name, edge_type_name, edge_src_name)
        print(f"Start operating on edge {edge_name}")
        edge_dir_name = edge_name
        edge_index = np.load(os.path.join(root_dir, edge_dir_name, 'edge_index.npy'))
        assert len(edge_index.shape) == 2
        assert edge_index.shape[1] == 2

        assert nodes_configs[edge_src_name]['node_index_mapping'] is not None, f"Missing node index mapping for {edge_src_name}. "
        assert nodes_configs[edge_dst_name]['node_index_mapping'] is not None, f"Missing node index mapping for {edge_dst_name}. "
        src_index_mapping = nodes_configs[edge_src_name]['node_index_mapping']
        dst_index_mapping = nodes_configs[edge_dst_name]['node_index_mapping']

        # original edge index is in unordered COO format
        coo_src_ids = edge_index[:, 0]
        coo_dst_ids = edge_index[:, 1]
        coo_src_ids = coo_src_ids.astype(np.dtype(nodes_configs[edge_src_name]['node_id_dtype']))
        coo_dst_ids = coo_dst_ids.astype(np.dtype(nodes_configs[edge_dst_name]['node_id_dtype']))
        coo_src_ids = np.apply_along_axis(lambda node_id: src_index_mapping[node_id], 0, coo_src_ids)
        coo_dst_ids = np.apply_along_axis(lambda node_id: dst_index_mapping[node_id], 0, coo_dst_ids)

        # there might be some nodes that have no connections of specific edge type
        src_node_count = nodes_feat_dict[edge_src_name]['node_count']
        dst_node_count = nodes_feat_dict[edge_dst_name]['node_count']

        assert src_node_count == src_index_mapping.shape[0], f"Edge {edge_name}: src node count ({src_node_count}) does not match index mapping shape ({src_index_mapping.shape[0]})"
        assert dst_node_count == dst_index_mapping.shape[0], f"Edge {edge_name}: dst node count ({dst_node_count}) does not match index mapping shape ({dst_index_mapping.shape[0]})"

        if edge_config['build_forward']:
            # build forward edge
            # note here that COO->CSR/CSC conversion happens at this untimed step. 
            graph_edge_data = build_graph_edge(coo_src_ids, coo_dst_ids, src_node_count, dst_node_count,
                                               edge_config['forward_sparse_format'], edge_config['add_self_loop'],
                                               edge_config['add_reverse_edges'])
            edge_graph_dict_entry = save_graph_edge(convert_dir,
                                                    edge_name,
                                                    graph_edge_data)
            
            if do_save_copy:
                _ = save_graph_edge(
                    graph_storage_copy, 
                    edge_name,
                    graph_edge_data
                )
            edge_graph_dict["__".join([edge_src_name, edge_type_name, edge_dst_name])] = edge_graph_dict_entry
            print('    written forward edge {}, dict={}'.format(edge_name, edge_graph_dict_entry))
        
        if edge_config['build_reverse']:
            # build reverse edge
            # note here that COO->CSR/CSC conversion happens at this untimed step. 
            graph_edge_data = build_graph_edge(coo_dst_ids, coo_src_ids, dst_node_count, src_node_count,
                                               edge_config['reverse_sparse_format'], edge_config['add_self_loop'],
                                               edge_config['add_reverse_edges'])
            edge_graph_dict_entry = save_graph_edge(convert_dir,
                                                    reversed_edge_name,
                                                    graph_edge_data)
            if do_save_copy:
                _ = save_graph_edge(
                    graph_storage_copy, 
                    reversed_edge_name,
                    graph_edge_data
                )
            edge_graph_dict["__".join([edge_dst_name, 'reverse_{}'.format(edge_type_name), edge_src_name])] = edge_graph_dict_entry
            print('    written reverse edge {}, dict={}'.format(reversed_edge_name, edge_graph_dict_entry))

    # config dict to manage all graph details
    config_dict = {
        'nodes': nodes_feat_dict,
        'edges': edge_graph_dict,
        'concatenated_features': {
            "concatenated": concat_features,
            "path": concat_feature_path,
            "num_buckets": num_embed_bucket,
            "block_size": embed_block_size,
            "precision": precision,
            "node_orders": node_orders,
            "total_number_of_nodes": total_number_of_nodes,
            "feat_dim": feat_dim,
        }
    }
    config_file_fullpath = os.path.join(convert_dir, 'config.yml')
    print('Saving config to {}'.format(config_file_fullpath, ))
    with open(config_file_fullpath, 'w') as f:
        yaml.dump(config_dict, f)
    print('Convert IGBH dataset done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data',
        help='root dir containing the datasets, consistent with the original IGB dataset organization')
    
    # This option will help us save time - otherwise it takes 3 hours to np.memmap
    parser.add_argument("--load_fp16_features", action="store_true", 
                        help="Should we directly load FP16 features exported from GLT? Default is false")
    parser.add_argument('--convert_dir', type=str, default='/converted',
        help='output dir containing converted dataset')
    parser.add_argument('--precision', type=str, default='float16',
                        choices=['float16', 'float32'],
                        help='target precision for node embedding')
    parser.add_argument('--size', type=str, default='full',
                        choices=['tiny', "small", 'medium', 'large', 'full'])
    
    parser.add_argument(
        '--shuffle', type=str, nargs="*", 
        choices=['paper', 'author', 'conference', 'journal', 'fos', 'institute'], 
        help="Whether we shuffle this node type")
    
    parser.add_argument(
        "--graph_storage_copy", type=str, 
        default="",
        help="if specified, saves another copy of the graph structure to this location"
    )
    
    parser.add_argument(
        "--seed", type=int, default=0
    )

    parser.add_argument(
        "--concat_features", action="store_true"
    )

    parser.add_argument(
        "--concat_feature_block_size", type=int, default=1,
        help="Number of features inside each hashing bucket"
    )

    parser.add_argument(
        '--concat_feature_num_buckets', type=int, default=4096,
        help="Number of hashing buckets"
    )
    
    system_args = parser.parse_args()

    if system_args.shuffle is None:
        system_args.shuffle = []

    torch.manual_seed(system_args.seed)
    convert_igbh_dataset(
        root_dir=f"{system_args.data_dir}/{system_args.size}/processed",
        convert_dir=f"{system_args.convert_dir}/{system_args.size}",
        dataset_size=system_args.size,
        precision=system_args.precision,
        shuffle_nodes=system_args.shuffle,
        load_fp16_features=system_args.load_fp16_features,
        concat_features=system_args.concat_features,
        concat_features_block_size=system_args.concat_feature_block_size,
        concat_features_num_buckets=system_args.concat_feature_num_buckets,
        graph_storage_copy=f"{system_args.graph_storage_copy}/{system_args.size}"
    )
