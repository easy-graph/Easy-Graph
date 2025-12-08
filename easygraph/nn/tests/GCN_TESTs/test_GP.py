# test_chunked_spmm.py
import os
import time
import random
import numpy as np
import torch
import scipy.sparse as sp
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.datasets import Coauthor, Planetoid
import easygraph as eg
import metis
transform = T.NormalizeFeatures()
# =========== 配置 ===========
DEVICE = torch.device('cpu')
NUM_THREADS = 32   # 根据你机器调整
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
torch.set_num_threads(NUM_THREADS)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DATASETS = [
    ('Coauthor', 'CS'),
    ('Coauthor', 'Physics'),
    ('Planetoid', 'Cora'),
    ('Planetoid', 'Citeseer'),
    ('Planetoid', 'PubMed'),
    ('ogb', 'ogbn-arxiv'),
]

_load = torch.load
def load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _load(*args, **kwargs)
torch.load = load

# =========== 加载示例数据集（可替换） ===========
# 这里默认用 ogbn-arxiv（如果不可用请改成你本地数据）
for backend_type, dataset_name in DATASETS:
    # -------------------- 加载数据集 --------------------
    if backend_type == 'Coauthor':
        dataset = Coauthor(root=f'/root/Easy-Graph/easygraph/nn/tests/data/Coauthor', name=dataset_name)
    elif backend_type == 'Planetoid':
        dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name, transform=transform)
    elif backend_type == 'ogb':
        dataset = PygNodePropPredDataset(name=dataset_name, root='/root/Easy-Graph/easygraph/nn/tests/data/OGB')
    else:
        raise ValueError(f"Unknown dataset type {backend_type}")

    data = dataset[0]
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    X = data.x.float().to(DEVICE)                      # (N, in_channels)
    in_channels = X.shape[1]
    # 设一个 out_dim 做比较
    out_channels = 256
    nparts = 32

    print(f"Dataset nodes: {num_nodes}, feat dim: {in_channels}, out dim: {out_channels}")
    print(f"Threads: {torch.get_num_threads()}")

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

    adj_gp = SparseTensor(
        row=torch.tensor(row, dtype=torch.long),
        col=torch.tensor(col, dtype=torch.long),
        value=val_torch,
        sparse_sizes=(num_nodes, num_nodes)
    )

    # =========== 基础实现： out = A.matmul(X @ W) ===========
    def baseline_spmm(A, X, W, repeat=200, warmup=2):
        # X: (N, in_ch), W: (in_ch, out_ch)
        # for _ in range(warmup):
        #     _ = A.matmul(X @ W)
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            _ = A.matmul(X @ W)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        return float(np.mean(times)), float(np.std(times))

    # =========== Chunked 实现（feature-dim chunking） ===========
    def chunked_spmm(A, X, W, chunk_size=64, repeat=200, warmup=2):
        # split output channels into chunks of size chunk_size
        out_ch = W.shape[1]
        nchunks = (out_ch + chunk_size - 1) // chunk_size
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            outs = []
            for i in range(nchunks):
                s = i * chunk_size
                e = min((i + 1) * chunk_size, out_ch)
                w_chunk = W[:, s:e]
                xw = X @ w_chunk
                out_chunk = A.matmul(xw)
                outs.append(out_chunk)
            out = torch.cat(outs, dim=1)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        return float(np.mean(times)), float(np.std(times))

    # =========== 主测试流程 ===========
    def test_chunked_variants(X, A, in_ch, out_ch, chunk_sizes=[16,32,64,128,256,512], repeat=200):
        # random weight
        W = torch.randn(in_ch, out_ch, dtype=torch.float32)

        # print("\nRunning baseline ...")
        t_base_mean, t_base_std = baseline_spmm(A, X, W, repeat=repeat)
        # print(f"Baseline avg: {t_base_mean*1000:.3f} ms (std {t_base_std*1000:.3f} ms)")

        results = []
        for cs in chunk_sizes:
            # print(f"\nRunning chunked (chunk_size={cs}) ...")
            t_mean, t_std = chunked_spmm(A, X, W, chunk_size=cs, repeat=repeat)
            speedup = t_base_mean / t_mean if t_mean>0 else float('nan')
            # print(f"Chunk {cs} avg: {t_mean*1000:.3f} ms (std {t_std*1000:.3f} ms) | speedup: {speedup:.3f}x")
            results.append((cs, t_mean, t_std, speedup))
        return (t_base_mean, t_base_std, results)

    base_mean, base_std, details = test_chunked_variants(X, adj_gp, in_channels, out_channels,
                                                        chunk_sizes=[16,32,64,128,256,512,1024], repeat=20)
    print(f"network:{dataset_name}")
    print(f"Baseline {base_mean*1000:.3f} ms")
    for cs, t_mean, t_std, speedup in details:
        print(f"chunk={cs}: {t_mean*1000:.3f} ms | speedup {speedup:.3f}x")
