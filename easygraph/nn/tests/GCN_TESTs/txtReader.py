import os
import pandas as pd
import numpy as np
from node2vec import Node2Vec
import networkx as nx
import torch

class TxtDataset:
    def __init__(self, X, Y, edge_index=None):
        self.x = torch.tensor(X)
        self.y = torch.tensor(Y)
        self.num_nodes = Y.shape[0]
        self.edge_index = edge_index

def TxtGraphReader(root, name):
    path = os.path.join(root, name, name + '.txt')
    edges = np.loadtxt(path, comments='#', dtype=int)
    unique_edges = set()
    for u, v in edges:
        if u == v:
            continue
        if (u,v) not in unique_edges and (v,u) not in unique_edges:
            unique_edges.add((u, v))

    edges = edges.T

    num_nodes = max(edges.flatten()) + 1
    feature_dim = 128
    class_num = 3

    if os.path.exists(os.path.join(root, name, 'X.npy')) and os.path.exists(os.path.join(root, name, 'Y.npy')):
        X = np.load(os.path.join(root, name, 'X.npy'))
        Y = np.load(os.path.join(root, name, 'Y.npy'))
    else:
        X = np.random.uniform(-10, 10, size=(num_nodes, feature_dim))
        Y = np.random.randint(0, class_num, size=(num_nodes, ))
        np.save(os.path.join(root, name, 'X.npy'), X)
        np.save(os.path.join(root, name, 'Y.npy'), Y)

    dataset = TxtDataset(X=X, Y=Y, edge_index=edges)

    return [dataset]
