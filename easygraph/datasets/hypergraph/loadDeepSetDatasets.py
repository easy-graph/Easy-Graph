import os.path as osp

import numpy as np
import scipy.sparse as sp
import torch

from torch_geometric.data import Data
from torch_sparse import coalesce


__all__ = ["load_line_expansion_dataset"]


def load_line_expansion_dataset(
    path=None, dataset="cocitation-cora", train_percent=0.5
):
    # load edges, features, and labels.
    print("Loading {} dataset...".format(dataset))

    file_name = f"{dataset}.content"
    p2idx_features_labels = osp.join(path, dataset, file_name)
    idx_features_labels = np.genfromtxt(p2idx_features_labels, dtype=np.dtype(str))
    # features = np.array(idx_features_labels[:, 1:-1])
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    #     labels = encode_onehot(idx_features_labels[:, -1])
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))

    print("load features")

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    file_name = f"{dataset}.edges"
    p2edges_unordered = osp.join(path, dataset, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered, dtype=np.int32)

    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(edges_unordered.shape)

    print("load edges")

    # From adjacency matrix to edge_list
    edge_index = edges.T
    #     ipdb.set_trace()
    assert edge_index[0].max() == edge_index[1].min() - 1

    # check if values in edge_index is consecutive. i.e. no missing value for node_id/he_id.
    assert len(np.unique(edge_index)) == edge_index.max() + 1

    num_nodes = edge_index[0].max() + 1
    num_he = edge_index[1].max() - num_nodes + 1
    edge_index = np.hstack((edge_index, edge_index[::-1, :]))

    # build torch data class
    data = Data(
        x=torch.FloatTensor(np.array(features[:num_nodes].todense())),
        edge_index=torch.LongTensor(edge_index),
        y=labels[:num_nodes],
    )

    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = len(np.unique(edge_index))
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index, None, total_num_node_id_he_id, total_num_node_id_he_id
    )
    n_x = num_nodes
    #     n_x = n_expanded
    num_class = len(np.unique(labels[:num_nodes].numpy()))
    data.n_x = n_x
    # add parameters to attribute

    data.train_percent = train_percent
    data.num_hyperedges = num_he

    return data
