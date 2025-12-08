import torch
from torch_sparse import SparseTensor
import time
import numpy as np

def spmm_dim_partition(adj_t: SparseTensor, x: torch.Tensor, weight: torch.Tensor,
                       dim_block: int = 64):
    """
    只分特征维度的高效 SpMM: out = adj_t.matmul(x @ weight)
    
    参数：
        adj_t: SparseTensor, shape (N, N)
        x: torch.Tensor, shape (N, Fin)
        weight: torch.Tensor, shape (Fin, Fout)
        dim_block: int, 每个特征块的大小
    
    返回：
        out: torch.Tensor, shape (N, Fout)
    """
    N, Fin = x.shape
    Fout = weight.shape[1]
    out = torch.zeros(N, Fout, dtype=x.dtype)

    # 分块计算
    for d0 in range(0, Fout, dim_block):
        d1 = min(Fout, d0 + dim_block)
        # 密集矩阵乘法: 特征块
        xw_block = x @ weight[:, d0:d1]
        # 稀疏乘法
        out[:, d0:d1] = adj_t.matmul(xw_block)
    return out

# ===================== 测试 =====================

if __name__ == "__main__":
    # 模拟一个大稀疏图
    N = 200000
    Fin = 256
    Fout = 4096
    density = 0.001

    # 输入特征和权重
    x = torch.randn(N, Fin)
    W = torch.randn(Fin, Fout)

    # 构造稀疏图 (torch_sparse.SparseTensor)
    nnz = int(N * N * density)
    row = torch.randint(0, N, (nnz,))
    col = torch.randint(0, N, (nnz,))
    value = torch.rand(nnz)
    adj_t = SparseTensor(row=row, col=col, value=value, sparse_sizes=(N, N))

    TB = []
    TP = []
    for _ in range(20):
        # Baseline: 全量计算
        t1 = time.time()
        y_baseline = adj_t.matmul(x @ W)
        t2 = time.time()
        TB.append(t2-t1)
        # 分特征维度计算
        t3 = time.time()
        y_part = spmm_dim_partition(adj_t, x, W, dim_block=1024)
        t4 = time.time()
        TP.append(t4-t3)

    print(f"Baseline time: {np.mean(TB):.4f}s")
    print(f"Dim-partitioned time: {np.mean(TP):.4f}s")
    print(f"Speedup: {np.mean(TB) / np.mean(TP):.2f}x")
    print("Result match:", torch.allclose(y_baseline, y_part, atol=1e-5))
