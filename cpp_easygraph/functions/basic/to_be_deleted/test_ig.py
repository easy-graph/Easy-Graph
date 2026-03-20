"""基准测试 – RoadNetCA 数据集 | igraph

运行示例:
  python test_ig.py                        # 默认 (5 进程 × 3 次采样，含 1 次热身)
  python test_ig.py --fast                 # 快速模式
  python test_ig.py -o results_ig.json     # 保存结果
  pyperf compare_to results_eg.json results_ig.json results_gt.json

说明:
  - 图构建在计时之外（模块级），每个 worker 进程独立初始化，保证进程隔离。
  - pyperf 自动处理热身、多次重复与统计（中位数 ± 标准差）。
  - 若需固定线程数以公平对比多库，建议: OMP_NUM_THREADS=1 python test_ig.py
"""
import sys
import pyperf
import igraph

from easygraph.datasets import RoadNetCADataset
# import graph_tool.all as gt  # 见 test_gt.py

# ── 数据集 & 图构建（在计时外，每个 worker 进程独立初始化）────────────────────
data_eg = RoadNetCADataset()
edges = [(u, v) for u, v, *_ in data_eg[0].edges]

G_ig = igraph.Graph(edges=edges, directed=False)


# ── 基准函数 ────────────────────────────────────────────────────────────────
def bench_local_clustering():
    # per-vertex local transitivity；mode="zero" 对度<2 的节点返回 0
    return G_ig.transitivity_local_undirected(mode="zero")


# ── 线程信息（仅主进程打印）──────────────────────────────────────────────────
def _print_thread_info():
    import os
    import multiprocessing
    import ctypes
    print("\n[Thread Info]")
    print(f"  OMP_NUM_THREADS  = {os.environ.get('OMP_NUM_THREADS', '(unset)')}")
    print(f"  CPU cores        = {multiprocessing.cpu_count()}")
    for libname in ("libgomp.so.1", "libgomp.so", "libiomp5.so", "libiomp5.so.5"):
        try:
            lib = ctypes.CDLL(libname)
            if hasattr(lib, 'omp_get_max_threads'):
                lib.omp_get_max_threads.restype = ctypes.c_int
                print(f"  {libname} omp_get_max_threads() = {lib.omp_get_max_threads()}")
                break
        except Exception:
            continue
    print()


if __name__ == "__main__":
    if '--worker' not in sys.argv:
        print(f"RoadNetCA: {G_ig.vcount()} nodes, {G_ig.ecount()} edges")
        _print_thread_info()

    runner = pyperf.Runner(processes=5)
    runner.bench_func('igraph: transitivity_local_undirected', bench_local_clustering)
