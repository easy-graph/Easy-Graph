# test_int8_cpu.py
import os
import time
import numpy as np
import torch
from torch_sparse import SparseTensor
from torch_geometric.datasets import Coauthor, Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T

transform = T.NormalizeFeatures()
DEVICE = torch.device('cpu')
torch.set_num_threads(32)

_load = torch.load
def load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _load(*args, **kwargs)
torch.load = load

# 加载数据集
# dataset = Coauthor(root=f'/root/Easy-Graph/easygraph/nn/tests/data/Coauthor', name='CS')
dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/root/Easy-Graph/easygraph/nn/tests/data/OGB')
data = dataset[0]

X = data.x.float().to(DEVICE)
edge_index = data.edge_index
num_nodes = data.num_nodes
in_channels = X.shape[1]
out_channels = 256

# 构造归一化邻接矩阵
row = edge_index[0].numpy()
col = edge_index[1].numpy()
deg = np.zeros(num_nodes, dtype=np.float32)
for r in row:
    deg[r] += 1
deg_inv_sqrt = np.zeros_like(deg)
nz = deg > 0
deg_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
val = deg_inv_sqrt[row] * deg_inv_sqrt[col]
val_torch = torch.tensor(val, dtype=torch.float32)

A = SparseTensor(
    row=torch.tensor(row, dtype=torch.long),
    col=torch.tensor(col, dtype=torch.long),
    value=val_torch,
    sparse_sizes=(num_nodes, num_nodes)
).to(DEVICE)

# 测试函数
def measure_time(A, X, W, repeat=100):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = A.matmul(X @ W)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.mean(times), np.std(times)

# ===== FP32 baseline =====
W32 = torch.randn(in_channels, out_channels, dtype=torch.float32, device=DEVICE)
mean32, std32 = measure_time(A, X, W32)

# ===== INT8 动态量化 =====
# PyTorch 量化方式: 先用 nn.Linear 包装，再动态量化
linear_fp32 = torch.nn.Linear(in_channels, out_channels, bias=False)
linear_fp32.weight.data = W32.t().contiguous()  # nn.Linear weight shape: (out_features, in_features)

# ===== INT8 动态量化 =====
linear_int8 = torch.quantization.quantize_dynamic(
    linear_fp32, {torch.nn.Linear}, dtype=torch.qint8
)

# 测试 INT8
def measure_time_int8(A, X, linear_int8, repeat=100):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        # 直接访问 weight 属性，不要加括号
        W_int8 = linear_int8.weight.dequantize().t()
        _ = A.matmul(X @ W_int8)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.mean(times), np.std(times)


mean_int8, std_int8 = measure_time_int8(A, X, linear_int8)

speedup = mean32 / mean_int8 if mean_int8 > 0 else float('nan')

print(f"CPU-only INT8 test on {DEVICE}")
print(f"FP32   : {mean32*1000:.3f} ms ± {std32*1000:.3f}")
print(f"INT8   : {mean_int8*1000:.3f} ms ± {std_int8*1000:.3f}")
print(f"Speedup: {speedup:.3f}x")
