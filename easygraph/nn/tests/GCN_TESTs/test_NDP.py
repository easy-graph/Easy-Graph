import torch
from torch_sparse import SparseTensor
import time
import os
import random
import numpy as np
import scipy.sparse as sp
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.datasets import Coauthor, Planetoid
import easygraph as eg
transform = T.NormalizeFeatures()

# =========== 配置 ===========
DEVICE = torch.device('cpu')
# NUM_THREADS = 32   # 根据你机器调整
# os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
# os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
# torch.set_num_threads(NUM_THREADS)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def spmm_dim_partition(adj_t: SparseTensor, x: torch.Tensor, weight: torch.Tensor,
                       dim_block: int = 64):
    N, Fin = x.shape
    Fout = weight.shape[1]
    out = torch.zeros(N, Fout, dtype=x.dtype)

    # 分块计算
    for d0 in range(0, Fout, dim_block):
        d1 = min(Fout, d0 + dim_block)
        xw_block = x @ weight[:, d0:d1]
        out[:, d0:d1] = adj_t.matmul(xw_block)
    return out

DATASETS = [
    # ('Coauthor', 'CS'),
    # ('Coauthor', 'Physics'),
    # ('Planetoid', 'Cora'),
    # ('Planetoid', 'Citeseer'),
    # ('Planetoid', 'PubMed'),
    ('ogb', 'ogbn-arxiv'),

]

_load = torch.load
def load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _load(*args, **kwargs)
torch.load = load

if __name__ == "__main__":
    
    _N = 20

    for backend_type, dataset_name in DATASETS:
        # -------------------- 加载数据集 --------------------
        if backend_type == 'Coauthor':
            dataset = Coauthor(root=f'/tmp/Coauthor', name=dataset_name)
        elif backend_type == 'Planetoid':
            dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name, transform=transform)
        elif backend_type == 'ogb':
            dataset = PygNodePropPredDataset(name=dataset_name, root='/tmp/OGB')
        else:
            raise ValueError(f"Unknown dataset type {backend_type}")

        data = dataset[0]
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        X = data.x.float().to(DEVICE)                    
        in_channels = X.shape[1]
        if dataset_name == 'ogbn-arxiv':
            out_channels = 256
        else:
            out_channels = 16

# =========== 构建归一化稀疏矩阵 A = D^{-1/2} A D^{-1/2} ===========
        row = edge_index[0].cpu().numpy()
        col = edge_index[1].cpu().numpy()
        # compute degree and normalization values (like GCN)
        deg = np.zeros(num_nodes, dtype=np.float32)
        for r in row:
            deg[r] += 1
        deg_inv_sqrt = np.zeros_like(deg)
        nz = deg > 0
        deg_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
        val = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        val_torch = torch.tensor(val, dtype=torch.float32)

        adj_t = SparseTensor(
            row=torch.tensor(row, dtype=torch.long),
            col=torch.tensor(col, dtype=torch.long),
            value=val_torch,
            sparse_sizes=(num_nodes, num_nodes)
        )

        W = torch.randn(in_channels, out_channels, dtype=torch.float32)

        T_B = []
        T_P = []
        for _ in range(_N):
            # Baseline: 全量计算
            t1 = time.perf_counter()
            y_baseline = adj_t.matmul(X @ W)
            t2 = time.perf_counter()

            # 分特征维度计算
            t3 = time.perf_counter()
            y_part = spmm_dim_partition(adj_t, X, W, dim_block=32)
            t4 = time.perf_counter()

            T_B.append(t2-t1)
            T_P.append(t4-t3)

        
        print(f'dataset:{dataset_name}')
        # print(f'TB:{T_B}')
        # print(f'TB:{T_P}')
        print(f"Baseline time: {np.mean(T_B) :.4f}s")
        print(f"Dim-partitioned time: {np.mean(T_P):.4f}s")
        print(f"Speedup: { np.mean(T_B) / np.mean(T_P):.2f}x")
        print("Result match:", torch.allclose(y_baseline, y_part, atol=1e-5))
